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
    if any(isapprox(0.0, t; atol=dt/10) for t in times_to_plot)
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
     next_index = 1
     tol = dt / 10

     for n in 1:nt
        t = n * dt
        println(t)
        c_old = copy(c)
        p = (c_old = c_old, dt = dt, dx = dx, kappa = kappa)
        c_guess = copy(c_old)
    
        # Set up and solve the nonlinear problem for the current time step.
        problem = NonlinearProblem(ie_residual!, c_guess, p)
        solver = solve(problem, NewtonRaphson(linsolve = KrylovJL_GMRES()))
        c = solver.u

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
t_vals = [0.0,5.0,10.0,15.0,20.0,25.0,100.0]

sol_mol_ana_1 = mol_solver(6,1,1,0.05,"analytical")
sol_mol_spline_1 = mol_solver(6,1,1,0.05,"spline")


tvals, sol_be_ana_1, xvals = impliciteuler(6,1,1,0.02,0.1,"analytical",t_vals)





"""
Plot
"""

#Backward Euler Analytical
p1 = plot(xlabel=L"x", ylabel=L"\phi_{1}",
        grid=false,tickfont=Plots.font("Computer Modern", 10),
        titlefont=Plots.font("Computer Modern",12),
        legendfont=Plots.font("Computer Modern",10),
        title="Backward Euler, Full"
)

for (t,sol) in zip(tvals,sol_be_ana_1)
    plot!(p1,xvals,sol, label="t=$(t)")
end


#TRBDF2 Analytical
p3 = plot(xlabel=L"x", ylabel=L"\phi_{1}",
        grid=false,tickfont=Plots.font("Computer Modern", 10),
        titlefont=Plots.font("Computer Modern",12),
        legendfont=Plots.font("Computer Modern",10),
        title="TRBDF2, Full"
)

for t in t_vals
    plot!(p3,range(0.0,4.0,length(sol_mol_ana_1(0.0))),sol_mol_ana_1(t),label="t=$(t)",linewidth=2)
end

#TRBDF2 Spline

p4 = plot(xlabel=L"x", ylabel=L"\phi_{1}",
        grid=false,tickfont=Plots.font("Computer Modern", 10),
        titlefont=Plots.font("Computer Modern",12),
        legendfont=Plots.font("Computer Modern",10),
        title="TRBDF2, Spline"
)

for t in t_vals
    plot!(p4,range(0.0,4.0,length(sol_mol_spline_1(0.0))),sol_mol_spline_1(t),label="t=$(t)",linewidth=2)
end

plot(p1,p1,p3,p4,layout=(2,2),size=(1000,1000),dpi=300, leftmargin=3mm,bottommargin=3mm,rightmargin=3mm)



