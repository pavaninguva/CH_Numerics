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

Random.seed!(1234)

"""
This script is used to run the 1d method
"""


"""
Functions for Solvers
"""

function spline_generator(χ,N1,N2,knots=100)
    #Def log terms 
    log_terms(ϕ) =  (ϕ./N1).*log.(ϕ) .+ ((1 .-ϕ)./N2).*log.(1 .-ϕ)

    function tanh_sinh_spacing(n, β)
        points = 0.5 * (1 .+ tanh.(β .* (2 * collect(0:n-1) / (n-1) .- 1)))
        return points
    end
    
    phi_vals_ = collect(tanh_sinh_spacing(knots-2,14))
    f_vals_ = log_terms(phi_vals_)

    #Append boundary values
    phi_vals = pushfirst!(phi_vals_,0)
    f_vals = pushfirst!(f_vals_,0)
    push!(phi_vals,1)
    push!(f_vals,0)

    spline = BSplineKit.interpolate(phi_vals, f_vals,BSplineOrder(4))
    d_spline = Derivative(1)*spline

    df_spline(phi) = d_spline.(phi) .+ χ.*(1 .- 2*phi)

    return df_spline
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

function mol_solver(chi, N1, N2, dx, energy_method)
    #Simulation Parameters
    L = 4.0
    tf = 100.0
    nx = Int(L / dx) + 1
    x = range(0, L, length = nx)
    kappa = (2 / 3) * chi

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
    sol = solve(prob, TRBDF2(),reltol=1e-8, abstol=1e-8)

    return sol
end

function impliciteuler(chi, N1, N2, dx, dt, energy_method, times_to_plot)
    # Simulation parameters
    L = 4.0
    tf = 100.0
    nt = Int(round(tf / dt))
    nx = Int(L / dx) + 1
    x = range(0, L, length = nx)
    kappa = (2 / 3) * chi

    # Initial condition: small random perturbation around c₀
    c0_ = 0.5
    c = c0_ .+ 0.02 * (rand(nx) .- 0.5)

    results = Dict{Float64, Vector{Float64}}()

    # Store initial conditions if necessary
    if any(isapprox(0.0, t; atol=dt/10000) for t in times_to_plot)
        results[0.0] = copy(c)
    end


    spline = spline_generator(chi,N1,N2,100)
    dfdphi = phi -> begin 
        if energy_method == "analytical"
            -2 .* chi .* phi .+ chi - (1/N2).*log.(1-phi) .+ (1/N1).*log.(phi)
        else
            spline.(phi)
        end
    end

    function M_func(phi)
        return phi .* (1 .- phi)
    end

    function M_func_half(phi1, phi2)
        return M_func(0.5 .* (phi1 .+ phi2))
    end

    max_allowed_iterations = 2
    prev_residual_norm=Ref(Inf)
    function check_residual_callback(state)
        if state.iteration == 1
            prev_residual_norm[] = state.f_norm
        else
            # Compare current residual norm with the previous one.
            if state.f_norm > prev_residual_norm[]
                error("Residual norm did not decrease (iteration $(state.iteration)): previous norm = $(prev_residual_norm[]), current norm = $(state.f_norm)")
            end
            # Update for the next iteration.
            prev_residual_norm[] = state.f_norm
        end
        return true  # Return true to continue iterations.
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
        

        problem = NonlinearProblem(ie_residual!, c_guess, p)
        solver = solve(problem, NewtonRaphson(linsolve = KrylovJL_GMRES()),
                        show_trace=Val(false),abstol=1e-8)
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
    end

    stored_times = sort(collect(keys(results)))
    solutions = [results[t] for t in stored_times]
    return stored_times, solutions, x

end


"""
Run solvers for Plots
"""
# t_vals = [0.0,4.0,5.0,6.0,7.0,8.0,10.0,25.0,100.0]


# sol_mol_ana_1 = mol_solver(6,1,1,0.02,"analytical")
# sol_mol_spline_1 = mol_solver(6,1,1,0.02,"spline")


# tvals, sol_be_ana_1, xvals = impliciteuler(6,1,1,0.02,0.1,"analytical",t_vals)

# tvals2, sol_be_spline_1, xvals2 = impliciteuler(6,1,1,0.02,0.1,"spline",t_vals)


tvals_2 = [0.0,0.5,1.0,1.5,2.0,5.0,10.0,100.0]
sol_mol_spline_2 = mol_solver(20,1,1,0.02,"spline")

tvals3, sol_be_spline_2, xvals3 = impliciteuler(20,1,1,0.02,0.005,"spline",tvals_2)


"""
Plot
"""

#Backward Euler Analytical
# p1 = plot(xlabel=L"x", ylabel=L"\phi_{1}",
#         grid=false,tickfont=Plots.font("Computer Modern", 10),
#         titlefont=Plots.font("Computer Modern",12),
#         legendfont=Plots.font("Computer Modern",10),
#         title="Backward Euler, Full"
# )

# for (t,sol) in zip(tvals,sol_be_ana_1)
#     plot!(p1,xvals,sol, label="t=$(t)",linewidth=2)
# end


# #Backward Euler Spline
# p2 = plot(xlabel=L"x", ylabel=L"\phi_{1}",
#         grid=false,tickfont=Plots.font("Computer Modern", 10),
#         titlefont=Plots.font("Computer Modern",12),
#         legendfont=Plots.font("Computer Modern",10),
#         title="Backward Euler, Spline"
# )

# for (t,sol) in zip(tvals2,sol_be_spline_1)
#     plot!(p2,xvals,sol, label="t=$(t)",linewidth=2)
# end


# #TRBDF2 Analytical
# p3 = plot(xlabel=L"x", ylabel=L"\phi_{1}",
#         grid=false,tickfont=Plots.font("Computer Modern", 10),
#         titlefont=Plots.font("Computer Modern",12),
#         legendfont=Plots.font("Computer Modern",10),
#         title="TRBDF2, Full"
# )

# for t in t_vals
#     plot!(p3,range(0.0,4.0,length(sol_mol_ana_1(0.0))),sol_mol_ana_1(t),label="t=$(t)",linewidth=2)
# end

# #TRBDF2 Spline

# p4 = plot(xlabel=L"x", ylabel=L"\phi_{1}",
#         grid=false,tickfont=Plots.font("Computer Modern", 10),
#         titlefont=Plots.font("Computer Modern",12),
#         legendfont=Plots.font("Computer Modern",10),
#         title="TRBDF2, Spline"
# )

# for t in t_vals
#     plot!(p4,range(0.0,4.0,length(sol_mol_spline_1(0.0))),sol_mol_spline_1(t),label="t=$(t)",linewidth=2)
# end

# p_all_1 = plot(p1,p2,p3,p4,layout=(2,2),size=(1000,1000),dpi=300, leftmargin=3mm,bottommargin=3mm,rightmargin=3mm)

# savefig(p_all_1,"1d_benchmark_plots_chi6.png")



#Backward Euler Spline
p5 = plot(xlabel=L"x", ylabel=L"\phi_{1}",
        grid=false,tickfont=Plots.font("Computer Modern", 10),
        titlefont=Plots.font("Computer Modern",12),
        legendfont=Plots.font("Computer Modern",10),
        title="Backward Euler, Spline"
)

for (t,sol) in zip(tvals3,sol_be_spline_2)
    plot!(p5,xvals3,sol, label="t=$(t)",linewidth=2)
end

p6 = plot(xlabel=L"x", ylabel=L"\phi_{1}",
        grid=false,tickfont=Plots.font("Computer Modern", 10),
        titlefont=Plots.font("Computer Modern",12),
        legendfont=Plots.font("Computer Modern",10),
        title="TRBDF2, Spline"
)

for t in tvals_2
    plot!(p6,range(0.0,4.0,length(sol_mol_spline_2(0.0))),sol_mol_spline_2(t),label="t=$(t)",linewidth=2)
end

p_all_2 = plot(p5,p6,layout=(1,2),size=(1000,500),dpi=300, leftmargin=3mm,bottommargin=3mm,rightmargin=3mm)