using SparseArrays
using LinearAlgebra
using NonlinearSolve
using Plots
using Krylov
using Trapz
using IterativeSolvers
using LaTeXStrings
using Random
using Plots.PlotMeasures
using Statistics
using DifferentialEquations
include("../pchip.jl") 


Random.seed!(1234)

"""
This script is used to plot exemplar 1d binary mixture simulations

Case 1: chi = 6, x_{1} = x_{2} = 1. 
"""


"""
Functions for Solvers and Compute
"""

@recipe function f(::Type{Val{:samplemarkers}}, x, y, z; step = 100)
    n = length(y)
    sx, sy = x[1:step:n], y[1:step:n]
    # add an empty series with the correct type for legend markers
    @series begin
        seriestype := :path
        markershape --> :auto
        x := []
        y := []
    end
    # add a series for the line
    @series begin
        primary := false # no legend entry
        markershape := :none # ensure no markers
        seriestype := :path
        seriescolor := get(plotattributes, :seriescolor, :auto)
        x := x
        y := y
    end
    # return  a series for the sampled markers
    primary := false
    seriestype := :scatter
    markershape --> :auto
    x := sx
    y := sy
end

function fh_deriv(phi,chi,N1,N2)
    df = (1/N1).*log.(phi) .+ (1/N1) .- (1/N2).*log.(1 .- phi) .- (1/N2) .- 2*chi.*phi .+ chi
    return df
end

function flory_huggins(phi,chi, N1,N2)
    return (1/N1) * (phi .* log.(phi)) + (1/N2) * (1 .- phi) .* log.(1 .- phi) + chi .* phi .* (1 .- phi)
end

function compute_energy(phi_vals, dx, xvals, kappa, chi, N1,N2)
    nx = length(phi_vals)
    #Compute Gradient
    dc_dx = zeros(nx)
    #Left boundary
    dc_dx[1] = (phi_vals[2]-phi_vals[1])/dx
    #Interior
    for i in 2:nx - 1
        dc_dx[i] = (phi_vals[i + 1] - phi_vals[i - 1])/(2*dx)
    end
    dc_dx[nx] = (phi_vals[nx] - phi_vals[nx-1])/dx

    energy_density = flory_huggins(phi_vals,chi,N1,N2) + (kappa/2)*dc_dx.^2

    return trapz(xvals,energy_density)
end


function spline_generator(chi, N1, N2, knots)


    function tanh_sinh_spacing(n, β)
        points = 0.5 * (1 .+ tanh.(β .* (2 * collect(0:n-1) / (n-1) .- 1)))
        return points
    end
    
    phi_vals_ = collect(tanh_sinh_spacing(knots-4,14))

    pushfirst!(phi_vals_,1e-16)
    push!(phi_vals_,1-1e-16)

    f_vals_ = fh_deriv(phi_vals_,chi,N1,N2)

    phi_vals = pushfirst!(phi_vals_,0)
    push!(phi_vals,1)

    #Compute value at eps
    eps_val = BigFloat("1e-40")
    one_big = BigFloat(1)

    f_eps = fh_deriv(eps_val,BigFloat(chi),BigFloat(N1), BigFloat(N2))
    f_eps1 = fh_deriv(one_big-eps_val, BigFloat(chi),BigFloat(N1), BigFloat(N2))

    f_eps_float = Float64(f_eps)
    f_eps1_float = Float64(f_eps1)

    f_vals = pushfirst!(f_vals_,f_eps_float)
    push!(f_vals, f_eps1_float)

    # Build and return the spline function using pchip
    spline = pchip(phi_vals, f_vals)
    return spline
end

function CH(ϕ, dx, params)
    χ, κ, N₁, N₂, energy_method = params
    spline = spline_generator(χ,N₁,N₂,100)

    dfdphi = ϕ -> begin 
        if energy_method == "analytical"
            -2 .* χ .* ϕ .+ χ - (1/N₂).*log.(1-ϕ) .+ (1/N₁).*log.(ϕ)
        else
            spline.(ϕ)
        end
    end

    mobility(ϕ) = ϕ .* (1 .- ϕ)

    function M_func_half(ϕ₁,ϕ₂,option=2)
        if option == 1
            M_func = 0.5 .*(mobility.(ϕ₁) .+ mobility.(ϕ₂))
        elseif option == 2
            M_func = mobility.(0.5 .* (ϕ₁ .+ ϕ₂))
        elseif option == 3
            M_func = (2 .* mobility.(ϕ₁) .* mobility.(ϕ₂)) ./ (mobility.(ϕ₁) .+ mobility.(ϕ₂))
        end
        return M_func
    end
    # Define chemical potential
    μ = similar(ϕ)
    μ[1] = dfdphi(ϕ[1]) - (2 * κ / (dx^2)) * (ϕ[2] - ϕ[1])
    μ[end] = dfdphi(ϕ[end]) - (2 * κ / (dx^2)) * (ϕ[end-1] - ϕ[end])
    μ[2:end-1] = dfdphi.(ϕ[2:end-1]) - (κ / (dx^2)) .* (ϕ[3:end] - 2 .* ϕ[2:end-1] .+ ϕ[1:end-2])

    # Define LHS (time derivative of ϕ)
    f = similar(ϕ)
    f[1] = (2 / (dx^2)) * (M_func_half(ϕ[1], ϕ[2]) * (μ[2] - μ[1]))
    f[end] = (2 / (dx^2)) * (M_func_half(ϕ[end], ϕ[end-1]) * (μ[end-1] - μ[end]))
    f[2:end-1] = (1 / (dx^2)) .* (M_func_half.(ϕ[2:end-1], ϕ[3:end]) .* (μ[3:end] .- μ[2:end-1]) .-
                                   M_func_half.(ϕ[2:end-1], ϕ[1:end-2]) .* (μ[2:end-1] .- μ[1:end-2]))

    return f
end

function mol_solver(chi, N1, N2, dx, L, tend, energy_method)
    #Simulation Parameters
    L = L
    tf = tend
    nx = Int(L / dx) + 1
    xvals = range(0, L, length = nx)
    if (N1/N2) < 10
        kappa = (2 / 3) * chi
    else
        kappa = (1/3)*chi
    end

    # Initial condition: small random perturbation around c0
    c0_ = 0.5
    c0 = c0_ .+ 0.02 * (rand(nx) .- 0.5)

    #Set up MOL bits
    params = (chi, kappa, N1, N2,energy_method)

    function ode_system!(du, u, p, t)
        du .= CH(u, dx, params)
        println(t)
    end

    # Set up the problem
    prob = ODEProblem(ode_system!, c0, (0.0, tf))
    sol = solve(prob, TRBDF2(),reltol=1e-6, abstol=1e-8,maxiters=1e7)

    #Compute energy and mass conservation
    t_evals = range(0,tf, 1000)
    c_avg = zeros(length(t_evals))
    energy = zeros(length(t_evals))


    for(i,t) in enumerate(t_evals)
        sol_ = sol(t)
        c_avg[i] = mean(sol_)
        energy[i] = compute_energy(sol_,dx,xvals,kappa,chi,N1,N2)
    end

    return sol, t_evals, c_avg, energy
end

function impliciteuler(chi, N1, N2, dx, dt, L, tend, energy_method, times_to_plot)
    # Simulation parameters
    L = L
    tf = tend
    nt = Int(round(tf / dt))
    tvals = range(0,tf,nt)
    nx = Int(L / dx) + 1
    x = range(0, L, length = nx)
    if (N1/N2) < 10
        kappa = (2 / 3) * chi
    else
        kappa = (1/3)*chi
    end

    # Initial condition: small random perturbation around c₀
    c0_ = 0.5
    c = c0_ .+ 0.02 * (rand(nx) .- 0.5)

    results = Dict{Float64, Vector{Float64}}()

    # Store initial conditions if necessary
    if any(isapprox(0.0, t; atol=dt/10000) for t in times_to_plot)
        results[0.0] = copy(c)
    end

    # Initialize arrays to store results
    c_avg = zeros(nt)
    energy = zeros(nt)

    #Store Initial conditions
    c_avg[1] = mean(c)
    energy[1] = compute_energy(c,dx,x,kappa,chi,N1,N2)


    spline = spline_generator(chi,N1,N2,100)
    dfdphi = phi -> begin 
        if energy_method == "analytical"
            -2 .* chi .* phi .+ chi - (1/N2).*log.(1-phi) .+ (1/N1).*log.(phi)
        elseif energy_method =="spline"
            spline.(phi)
        else
            error("Energy method is wrong")
        end
    end

    function M_func(phi)
        return phi .* (1 .- phi)
    end

    function M_func_half(phi1, phi2)
        return M_func(0.5 .* (phi1 .+ phi2))
    end
    
    
    function ie_residual!(F, c_new, p)
        c_old = p.c_old
        dt = p.dt
        dx = p.dx
        kappa = p.kappa
        nx = length(c_new)

        c_work = c_new

        # Compute mu_new
        mu_new = similar(c_new)

        # Left boundary (Neumann BC)
        mu_new[1] = dfdphi(c_work[1]) - (2.0 * kappa / dx^2) * (c_work[2] - c_work[1])

        # Interior points
        for i in 2:nx - 1
            mu_new[i] = dfdphi(c_work[i]) - (kappa / dx^2) * (c_work[i + 1] - 2.0 * c_work[i] + c_work[i - 1])
        end

        # Right boundary (Neumann BC)
        mu_new[nx] = dfdphi(c_work[nx]) - (2.0 * kappa / dx^2) * (c_work[nx - 1] - c_work[nx])

        # Compute residuals F
        # Left boundary (Neumann BC)
        M_iphalf = M_func_half(c_work[1], c_work[2])
        F[1] = (c_new[1] - c_old[1]) / dt - (2.0 / dx^2) * M_iphalf * (mu_new[2] - mu_new[1])
        # Interior points
        for i in 2:nx - 1
            M_iphalf = M_func_half(c_work[i], c_work[i + 1])
            M_imhalf = M_func_half(c_work[i], c_work[i - 1])
            F[i] = (c_new[i] - c_old[i]) / dt - (1.0 / dx^2) * (
                M_iphalf * (mu_new[i + 1] - mu_new[i]) - M_imhalf * (mu_new[i] - mu_new[i - 1])
            )
        end
        # Right boundary (Neumann BC)
        M_imhalf = M_func_half(c_work[nx], c_work[nx - 1])
        F[nx] = (c_new[nx] - c_old[nx]) / dt - (2.0 / dx^2) * M_imhalf * (mu_new[nx] - mu_new[nx - 1])
    end

     # Ensure the times to plot are sorted and set up an index pointer.
     times_to_plot = sort(times_to_plot)
     next_index = 2
     tol = dt / 1000

     for n in 1:nt
        t = n * dt
        println(t)
        c_old = copy(c)
        p = (c_old = c_old, dt = dt, dx = dx, kappa = kappa)
        c_guess = copy(c_old)
    
        # Set up and solve the nonlinear problem for the current time step.
        term_cond = AbsNormSafeTerminationMode(
            NonlinearSolve.L2_NORM; protective_threshold = nothing,
            patience_steps = 100, patience_objective_multiplier = 3,
            min_max_factor = 1.3,
            )
        
        problem = NonlinearProblem(ie_residual!, c_guess, p)
        solver = solve(problem, NewtonRaphson(linsolve = KrylovJL_GMRES()),
                        show_trace=Val(false),abstol=1e-8,
                        termination_condition=term_cond)
        c = solver.u

        #Check solver residual
        abstol = 1e-7
        if norm(solver.resid,Inf) > abstol
            error("Sovler did not converge")
        end

        while next_index <= length(times_to_plot) && t >= times_to_plot[next_index] - tol
            results[times_to_plot[next_index]] = copy(c)
            next_index += 1
        end

        #Compute and store energy and mean c
        if n < nt - 1e-8
            c_avg[n+1] = mean(c)
            energy[n+1] = compute_energy(c,dx,x,kappa,chi,N1,N2)
        end


    end

    stored_times = sort(collect(keys(results)))
    solutions = [results[t] for t in stored_times]
    return stored_times, solutions, x, tvals, c_avg, energy

end

"""
Plot Case 1: Chi = 6, x1 = x2 = 1
"""
# t_vals_case1 = [0.0,4.0,5.0,6.0,7.0,8.0,10.0,25.0,50.0]

# #### Backwards Euler FULL

# tevals_be_ana1, sol_be_ana_1, xvals, tvals_be_ana1, cvals_be_ana1, energy_be_ana1 = impliciteuler(6,1,1,0.1,0.1,4,50,"analytical",t_vals_case1)
# tevals_be_ana2, sol_be_ana_2, xvals2, tvals_be_ana2, cvals_be_ana2, energy_be_ana2 = impliciteuler(6,1,1,0.05,0.05,4,50,"analytical",t_vals_case1)
# tevals_be_ana3, sol_be_ana_3, xvals3, tvals_be_ana3, cvals_be_ana3, energy_be_ana3 = impliciteuler(6,1,1,0.025,0.025,4,50,"analytical",t_vals_case1)

# p13 = plot(xlabel=L"x", ylabel=L"\phi_{1}",
#         grid=false,tickfont=Plots.font("Computer Modern", 10),
#         titlefont=Plots.font("Computer Modern",11),
#         legendfont=Plots.font("Computer Modern",7),
#         title=L"\textrm{Backwards \ Euler, Full,} \Delta x = \Delta t = 0.1"
# )

# for (t,sol) in zip(tevals_be_ana1,sol_be_ana_1)
#     plot!(p13,xvals,sol, label="t=$(t)",linewidth=2)
# end

# p23 = plot(xlabel=L"x", ylabel=L"\phi_{1}",
#         grid=false,tickfont=Plots.font("Computer Modern", 10),
#         titlefont=Plots.font("Computer Modern",11),
#         legendfont=Plots.font("Computer Modern",7),
#         title=L"\textrm{Backwards \ Euler, Full,} \Delta x = \Delta t = 0.05"
# )

# for (t,sol) in zip(tevals_be_ana2,sol_be_ana_2)
#     plot!(p23,xvals2,sol, label="t=$(t)",linewidth=2)
# end

# p33 = plot(xlabel=L"x", ylabel=L"\phi_{1}",
#         grid=false,tickfont=Plots.font("Computer Modern", 10),
#         titlefont=Plots.font("Computer Modern",11),
#         legendfont=Plots.font("Computer Modern",7),
#         title=L"\textrm{Backwards \ Euler, Full,} \Delta x = \Delta t = 0.025"
# )

# for (t,sol) in zip(tevals_be_ana3,sol_be_ana_3)
#     plot!(p33,xvals3,sol, label="t=$(t)",linewidth=2)
# end

# p43 = plot(
#   xlabel = L"t",
#   ylabel = L"\bar{\phi}_{1}",
#   grid  = false,
#   y_guidefontcolor   = :blue,
#   y_foreground_color_axis   = :blue,
#   y_foreground_color_text   = :blue,
#   y_foreground_color_border = :blue,
#   tickfont   = Plots.font("Computer Modern", 10),
#   titlefont  = Plots.font("Computer Modern", 11),
#   legendfont = Plots.font("Computer Modern", 8),
#   ylims      = (0.45,0.55),
# )

# plot!(p43, tvals_be_ana1, cvals_be_ana1; color = :blue, seriestype=:samplemarkers,step=50, marker=:circle,markercolor=:blue, markersize=2, markerstrokewidth=0, markerstrokecolor=:blue, label = L"\Delta x = \Delta t = 0.1")
# plot!(p43, tvals_be_ana2, cvals_be_ana2; color=:blue,linestyle=:solid, label=L"\Delta x = \Delta t = 0.05")
# plot!(p43, tvals_be_ana3, cvals_be_ana3; color=:blue,linestyle=:dot, label=L"\Delta x = \Delta t = 0.025")

# p43_axis2 = twinx(p43)

# plot!(
#   p43_axis2,
#   tvals_be_ana1,
#   energy_be_ana1;
#   color         = :red,
#   ylabel        = L"\mathrm{Energy}",
#   label         = "",
#   y_guidefontcolor   = :red,
#   y_foreground_color_axis   = :red,
#   y_foreground_color_text   = :red,
#   y_foreground_color_border = :red,
#   tickfont   = Plots.font("Computer Modern", 10),
#   seriestype=:samplemarkers,step=50, marker=:circle,markercolor=:red, markersize=2, markerstrokewidth=0, markerstrokecolor=:red,
# )
# plot!(p43_axis2,tvals_be_ana2,energy_be_ana2;color=:red,linestyle=:solid,label="")
# plot!(p43_axis2,tvals_be_ana3,energy_be_ana3;color=:red,linestyle=:dot,label="")

# ### BACKWARDS EULER SPLINE

# tevals_be_spline1, sol_be_spline_1, xvals, tvals_be_spline1, cvals_be_spline1, energy_be_spline1 = impliciteuler(6,1,1,0.1,0.1,4,50,"spline",t_vals_case1)
# tevals_be_spline2, sol_be_spline_2, xvals2, tvals_be_spline2, cvals_be_spline2, energy_be_spline2 = impliciteuler(6,1,1,0.05,0.05,4,50,"spline",t_vals_case1)
# tevals_be_spline3, sol_be_spline_3, xvals3, tvals_be_spline3, cvals_be_spline3, energy_be_spline3 = impliciteuler(6,1,1,0.025,0.025,4,50,"spline",t_vals_case1)

# p14 = plot(xlabel=L"x", ylabel=L"\phi_{1}",
#         grid=false,tickfont=Plots.font("Computer Modern", 10),
#         titlefont=Plots.font("Computer Modern",11),
#         legendfont=Plots.font("Computer Modern",7),
#         title=L"\textrm{Backwards \ Euler, Spline,} \Delta x = \Delta t = 0.1"
# )

# for (t,sol) in zip(tevals_be_spline1,sol_be_spline_1)
#     plot!(p14,xvals,sol, label="t=$(t)",linewidth=2)
# end

# p24 = plot(xlabel=L"x", ylabel=L"\phi_{1}",
#         grid=false,tickfont=Plots.font("Computer Modern", 10),
#         titlefont=Plots.font("Computer Modern",11),
#         legendfont=Plots.font("Computer Modern",7),
#         title=L"\textrm{Backwards \ Euler, Spline,} \Delta x = \Delta t = 0.05"
# )

# for (t,sol) in zip(tevals_be_spline2,sol_be_spline_2)
#     plot!(p24,xvals2,sol, label="t=$(t)",linewidth=2)
# end

# p34 = plot(xlabel=L"x", ylabel=L"\phi_{1}",
#         grid=false,tickfont=Plots.font("Computer Modern", 10),
#         titlefont=Plots.font("Computer Modern",11),
#         legendfont=Plots.font("Computer Modern",7),
#         title=L"\textrm{Backwards \ Euler, Spline,} \Delta x = \Delta t = 0.025"
# )

# for (t,sol) in zip(tevals_be_spline3,sol_be_spline_3)
#     plot!(p34,xvals3,sol, label="t=$(t)",linewidth=2)
# end

# p44 = plot(
#   xlabel = L"t",
#   ylabel = L"\bar{\phi}_{1}",
#   grid  = false,
#   y_guidefontcolor   = :blue,
#   y_foreground_color_axis   = :blue,
#   y_foreground_color_text   = :blue,
#   y_foreground_color_border = :blue,
#   tickfont   = Plots.font("Computer Modern", 10),
#   titlefont  = Plots.font("Computer Modern", 11),
#   legendfont = Plots.font("Computer Modern", 8),
#   ylims      = (0.45,0.55),
# )

# plot!(p44, tvals_be_spline1, cvals_be_spline1; color = :blue, seriestype=:samplemarkers,step=50, marker=:circle,markercolor=:blue, markersize=2, markerstrokewidth=0, markerstrokecolor=:blue, label = L"\Delta x = \Delta t = 0.1")
# plot!(p44, tvals_be_spline2, cvals_be_spline2; color=:blue,linestyle=:solid, label=L"\Delta x = \Delta t = 0.05")
# plot!(p44, tvals_be_spline3, cvals_be_spline3; color=:blue,linestyle=:dot, label=L"\Delta x = \Delta t = 0.025")

# p44_axis2 = twinx(p44)

# plot!(
#   p44_axis2,
#   tvals_be_spline1,
#   energy_be_spline1;
#   color         = :red,
#   ylabel        = L"\mathrm{Energy}",
#   label         = "",
#   y_guidefontcolor   = :red,
#   y_foreground_color_axis   = :red,
#   y_foreground_color_text   = :red,
#   y_foreground_color_border = :red,
#   tickfont   = Plots.font("Computer Modern", 10),
#   seriestype=:samplemarkers,step=50, marker=:circle,markercolor=:red, markersize=2, markerstrokewidth=0, markerstrokecolor=:red
# )
# plot!(p44_axis2,tvals_be_spline2,energy_be_spline2;color=:red,linestyle=:solid,label="")
# plot!(p44_axis2,tvals_be_spline3,energy_be_spline3;color=:red,linestyle=:dot,label="")




# ###TRBDF2 FULL

# sol_mol_ana1, tvalsana1, cvalsana1, energyvalsana1 = mol_solver(6,1,1,0.1,4.0,50,"analytical")
# sol_mol_ana2, tvalsana2, cvalsana2, energyvalsana2 = mol_solver(6,1,1,0.05,4.0,50,"analytical")
# sol_mol_ana3, tvalsana3, cvalsana3, energyvalsana3 = mol_solver(6,1,1,0.025,4.0,50,"analytical")

# p12 = plot(xlabel=L"x", ylabel=L"\phi_{1}",
#         grid=false,tickfont=Plots.font("Computer Modern", 10),
#         titlefont=Plots.font("Computer Modern",11),
#         legendfont=Plots.font("Computer Modern",7),
#         title=L"\textrm{TRBDF2, Full,} \Delta x = 0.1"
# )

# for t in t_vals_case1
#     plot!(p12,range(0.0,4.0,length(sol_mol_ana1(0.0))),sol_mol_ana1(t),label="t=$(t)",linewidth=2)
# end

# p22 = plot(xlabel=L"x", ylabel=L"\phi_{1}",
#         grid=false,tickfont=Plots.font("Computer Modern", 10),
#         titlefont=Plots.font("Computer Modern",11),
#         legendfont=Plots.font("Computer Modern",7),
#         title=L"\textrm{TRBDF2, Full,} \Delta x = 0.05"
# )

# for t in t_vals_case1
#     plot!(p22,range(0.0,4.0,length(sol_mol_ana2(0.0))),sol_mol_ana2(t),label="t=$(t)",linewidth=2)
# end

# p32 = plot(xlabel=L"x", ylabel=L"\phi_{1}",
#         grid=false,tickfont=Plots.font("Computer Modern", 10),
#         titlefont=Plots.font("Computer Modern",11),
#         legendfont=Plots.font("Computer Modern",7),
#         title=L"\textrm{TRBDF2, Full,} \Delta x = 0.025"
# )

# for t in t_vals_case1
#     plot!(p32,range(0.0,4.0,length(sol_mol_ana3(0.0))),sol_mol_ana3(t),label="t=$(t)",linewidth=2)
# end


# p42 = plot(
#   xlabel = L"t",
#   ylabel = L"\bar{\phi}_{1}",
#   grid  = false,
#   y_guidefontcolor   = :blue,
#   y_foreground_color_axis   = :blue,
#   y_foreground_color_text   = :blue,
#   y_foreground_color_border = :blue,
#   tickfont   = Plots.font("Computer Modern", 10),
#   titlefont  = Plots.font("Computer Modern", 11),
#   legendfont = Plots.font("Computer Modern", 8),
#   ylims      = (0.45,0.55),
# )

# plot!(p42, tvalsana1, cvalsana1; color = :blue, seriestype=:samplemarkers,step=50, marker=:circle,markercolor=:blue, markersize=2, markerstrokewidth=0, markerstrokecolor=:blue, label = L"\Delta x = 0.1")
# plot!(p42,tvalsana2,cvalsana2; color=:blue,linestyle=:solid, label=L"\Delta x = 0.05")
# plot!(p42,tvalsana3,cvalsana3; color=:blue,linestyle=:dot, label=L"\Delta x = 0.025")

# p42_axis2 = twinx(p42)

# plot!(
#   p42_axis2,
#   tvalsana1,
#   energyvalsana1;
#   color         = :red,
#   ylabel        = L"\mathrm{Energy}",
#   label         = "",
#   y_guidefontcolor   = :red,
#   y_foreground_color_axis   = :red,
#   y_foreground_color_text   = :red,
#   y_foreground_color_border = :red,
#   tickfont   = Plots.font("Computer Modern", 10),
#   seriestype=:samplemarkers,step=50, marker=:circle,markercolor=:red, markersize=2, markerstrokewidth=0, markerstrokecolor=:red,
# )
# plot!(p42_axis2,tvalsana2,energyvalsana2;color=:red,linestyle=:solid,label="")
# plot!(p42_axis2,tvalsana3,energyvalsana3;color=:red,linestyle=:dot,label="")


# ### TRBDF2 SPLINE

# sol_mol_spline1, tvals1, cvals1, energyvals1 = mol_solver(6,1,1,0.1,4.0,50,"spline")
# sol_mol_spline2, tvals2, cvals2, energyvals2 = mol_solver(6,1,1,0.05,4.0,50,"spline")
# sol_mol_spline3, tvals3, cvals3, energyvals3 = mol_solver(6,1,1,0.025,4.0,50,"spline")



# p1 = plot(xlabel=L"x", ylabel=L"\phi_{1}",
#         grid=false,tickfont=Plots.font("Computer Modern", 10),
#         titlefont=Plots.font("Computer Modern",11),
#         legendfont=Plots.font("Computer Modern",7),
#         title=L"\textrm{TRBDF2, Spline,} \Delta x = 0.1"
# )

# for t in t_vals_case1
#     plot!(p1,range(0.0,4.0,length(sol_mol_spline1(0.0))),sol_mol_spline1(t),label="t=$(t)",linewidth=2)
# end

# p2 = plot(xlabel=L"x", ylabel=L"\phi_{1}",
#         grid=false,tickfont=Plots.font("Computer Modern", 10),
#         titlefont=Plots.font("Computer Modern",11),
#         legendfont=Plots.font("Computer Modern",7),
#         title=L"\textrm{TRBDF2, Spline,} \Delta x = 0.05"
# )

# for t in t_vals_case1
#     plot!(p2,range(0.0,4.0,length(sol_mol_spline2(0.0))),sol_mol_spline2(t),label="t=$(t)",linewidth=2)
# end

# p3 = plot(xlabel=L"x", ylabel=L"\phi_{1}",
#         grid=false,tickfont=Plots.font("Computer Modern", 10),
#         titlefont=Plots.font("Computer Modern",11),
#         legendfont=Plots.font("Computer Modern",7),
#         title=L"\textrm{TRBDF2, Spline,} \Delta x = 0.025"
# )

# for t in t_vals_case1
#     plot!(p3,range(0.0,4.0,length(sol_mol_spline3(0.0))),sol_mol_spline3(t),label="t=$(t)",linewidth=2)
# end


# p4 = plot(
#   xlabel = L"t",
#   ylabel = L"\bar{\phi}_{1}",
#   grid  = false,
#   y_guidefontcolor   = :blue,
#   y_foreground_color_axis   = :blue,
#   y_foreground_color_text   = :blue,
#   y_foreground_color_border = :blue,
#   tickfont   = Plots.font("Computer Modern", 10),
#   titlefont  = Plots.font("Computer Modern", 11),
#   legendfont = Plots.font("Computer Modern", 8),
#   ylims      = (0.45,0.55),
# )

# plot!(p4, tvals1, cvals1; color = :blue, seriestype=:samplemarkers,step=50, marker=:circle,markercolor=:blue, markersize=2, markerstrokewidth=0, markerstrokecolor=:blue, label = L"\Delta x = 0.1")
# plot!(p4,tvals2,cvals2;color=:blue,linestyle=:solid, label=L"\Delta x = 0.05")
# plot!(p4,tvals3,cvals3;color=:blue,linestyle=:dot, label=L"\Delta x = 0.025")

# p4_axis2 = twinx(p4)

# plot!(
#   p4_axis2,
#   tvals1,
#   energyvals1;
#   color         = :red,
#   ylabel        = L"\mathrm{Energy}",
#   label         = "",
#   y_guidefontcolor   = :red,
#   y_foreground_color_axis   = :red,
#   y_foreground_color_text   = :red,
#   y_foreground_color_border = :red,
#   tickfont   = Plots.font("Computer Modern", 10),
#   seriestype=:samplemarkers,step=50, marker=:circle,markercolor=:red, markersize=2, markerstrokewidth=0, markerstrokecolor=:red,
# )
# plot!(p4_axis2,tvals2,energyvals2;color=:red,linestyle=:solid,label="")
# plot!(p4_axis2,tvals3,energyvals3;color=:red,linestyle=:dot,label="")

# p1_all = plot(p13,p23,p33,p43, p14,p24,p34,p44, p12,p22,p32,p42, p1,p2,p3,p4, layout=(4,4),size=(1600,1600),dpi=300,
#                 bottom_margin = 8Plots.mm, left_margin = 6Plots.mm, right_margin=8Plots.mm)
              

# # savefig("./1d_demixing_benchmark1.png")
# display(p1_all)

"""
Case 2 chi = 30, x1 = 1, x2=1
"""

t_vals_case2 = [0.0,0.1,0.2,0.5,0.8,1.0,2.0,3.0,4.0]

sol_mol_spline6, tvals6, cvals6, energyvals6 = mol_solver(30,1,1,0.025,4.0,4,"spline")
sol_mol_spline4, tvals4, cvals4, energyvals4 = mol_solver(30,1,1,0.1,4.0,4,"spline")
sol_mol_spline5, tvals5, cvals5, energyvals5 = mol_solver(30,1,1,0.05,4.0,4,"spline")





p1 = plot(xlabel=L"x", ylabel=L"\phi_{1}",
        grid=false,tickfont=Plots.font("Computer Modern", 10),
        titlefont=Plots.font("Computer Modern",11),
        legendfont=Plots.font("Computer Modern",7),
        title=L"\textrm{TRBDF2, Spline,} \Delta x = 0.1"
)

for t in t_vals_case2
    plot!(p1,range(0.0,4.0,length(sol_mol_spline4(0.0))),sol_mol_spline4(t),label="t=$(t)",linewidth=2)
end

p2 = plot(xlabel=L"x", ylabel=L"\phi_{1}",
        grid=false,tickfont=Plots.font("Computer Modern", 10),
        titlefont=Plots.font("Computer Modern",11),
        legendfont=Plots.font("Computer Modern",7),
        title=L"\textrm{TRBDF2, Spline,} \Delta x = 0.05"
)

for t in t_vals_case2
    plot!(p2,range(0.0,4.0,length(sol_mol_spline5(0.0))),sol_mol_spline5(t),label="t=$(t)",linewidth=2)
end

p3 = plot(xlabel=L"x", ylabel=L"\phi_{1}",
        grid=false,tickfont=Plots.font("Computer Modern", 10),
        titlefont=Plots.font("Computer Modern",11),
        legendfont=Plots.font("Computer Modern",7),
        title=L"\textrm{TRBDF2, Spline,} \Delta x = 0.025"
)

for t in t_vals_case2
    plot!(p3,range(0.0,4.0,length(sol_mol_spline6(0.0))),sol_mol_spline6(t),label="t=$(t)",linewidth=2)
end


p4 = plot(
  xlabel = L"t",
  ylabel = L"\bar{\phi}_{1}",
  grid  = false,
  y_guidefontcolor   = :blue,
  y_foreground_color_axis   = :blue,
  y_foreground_color_text   = :blue,
  y_foreground_color_border = :blue,
  tickfont   = Plots.font("Computer Modern", 10),
  titlefont  = Plots.font("Computer Modern", 11),
  legendfont = Plots.font("Computer Modern", 8),
  ylims      = (0.45,0.55),
)

plot!(p4, tvals4, cvals4; color = :blue, seriestype=:samplemarkers,step=50, marker=:circle,markercolor=:blue, markersize=2, markerstrokewidth=0, markerstrokecolor=:blue, label = L"\Delta x = 0.1")
plot!(p4,tvals5,cvals5;color=:blue,linestyle=:solid, label=L"\Delta x = 0.05")
plot!(p4,tvals6,cvals6;color=:blue,linestyle=:dot, label=L"\Delta x = 0.025")

p4_axis2 = twinx(p4)

plot!(
  p4_axis2,
  tvals4,
  energyvals4;
  color         = :red,
  ylabel        = L"\mathrm{Energy}",
  label         = "",
  y_guidefontcolor   = :red,
  y_foreground_color_axis   = :red,
  y_foreground_color_text   = :red,
  y_foreground_color_border = :red,
  tickfont   = Plots.font("Computer Modern", 10),
  seriestype=:samplemarkers,step=50, marker=:circle,markercolor=:red, markersize=2, markerstrokewidth=0, markerstrokecolor=:red,
)
plot!(p4_axis2,tvals5,energyvals5;color=:red,linestyle=:solid,label="")
plot!(p4_axis2,tvals6,energyvals6;color=:red,linestyle=:dot,label="")

p2_all = plot(p1,p2,p3,p4, layout=(1,4),size=(1600,350),dpi=300,
                bottom_margin = 8Plots.mm, left_margin = 6Plots.mm, right_margin=8Plots.mm)
              

savefig("./1d_demixing_benchmark2.png")
display(p2_all)


