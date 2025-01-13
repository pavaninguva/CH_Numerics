using BSplineKit
using SparseArrays
using LinearAlgebra
using NonlinearSolve
using Plots
using Krylov
using IterativeSolvers
using Random
using NaNMath
using DataFrames
using LaTeXStrings
using DifferentialEquations
using CSV
using DataFrames
using Plots.PlotMeasures

"""
This script tests the ability of the solvers to converge 
for different values of chi and dx by varying dt

"""


"""
Functions
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

function impliciteuler(chi, N1, N2, dx, dt, energy_method)
    #Simulation Parameters
    L = 4.0
    tf = 100.0
    nt = Int(tf / dt)
    nx = Int(L / dx) + 1
    x = range(0, L, length = nx)
    kappa = (2 / 3) * chi

    # Initial condition: small random perturbation around c0
    c0_ = 0.5
    c = c0_ .+ 0.02 * (rand(nx) .- 0.5)

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

    for n = 1:nt
        c_old = copy(c)
        p = (c_old = c_old, dt=dt, dx = dx, kappa = kappa)
        c_guess = copy(c_old)
        # Create the NonlinearProblem
        problem = NonlinearProblem(ie_residual!, c_guess, p)
        # Solve the nonlinear system
        solver = solve(problem, NewtonRaphson(linsolve = KrylovJL_GMRES()))
        # Update c for the next time step
        c = solver.u
    end
end

# Function to read existing data or initialize a DataFrame
function load_or_initialize_csv(file_name::String)
    if isfile(file_name)
        return CSV.read(file_name, DataFrame)
    else
        return DataFrame(chi=Float64[], dx=Float64[], max_dt=Float64[])
    end
end

# Function to save results to a CSV file
function save_results_to_csv(file_name::String, results::DataFrame)
    CSV.write(file_name, results)
end


"""
Testing MOL Solver
"""

function param_sweep_min_dt(chi_values, dx_values; N1=1.0, N2=1.0, energy_method="analytical",results_file)
    # Load existing results if the file exists
    results = Dict()
    if isfile(results_file)
        existing_data = CSV.read(results_file, DataFrame)
        for row in eachrow(existing_data)
            key = (row.chi, row.dx)
            results[key] = row.min_dt
        end
    end
    new_results = DataFrame(chi=Float64[], dx=Float64[], min_dt=Float64[])
    # Initialize the min_dt_matrix for results
    min_dt_matrix = fill(NaN, length(chi_values), length(dx_values))
    for (i, chi) in enumerate(chi_values)
        for (j, dx) in enumerate(dx_values)
            key = (chi,dx)
            if haskey(results, key)
                println("Reusing result for chi=$chi, dx=$dx, energy_method=$energy_method, timestepping=BDF")
                min_dt_matrix[i, j] = results[key]
                continue  # Skip to the next parameter combination
            end

            println("Running simulation for chi=$chi, dx=$dx, energy_method=$energy_method, timestepping=BDF")
            sol = nothing
            try
                sol = mol_solver(chi, N1, N2, dx, energy_method)
            catch e
                @warn "Solver failed for chi=$chi, dx=$dx with error $e"
                min_dt_matrix[i, j] = NaN
                continue
            end

            # Check solver return code for divergence
            if sol.retcode == :Diverged || sol.retcode == :Failure
                @warn "Solver diverged or failed for chi=$chi, dx=$dx"
                min_dt_matrix[i,j] = NaN
                continue
            end

            t_values = sol.t
            if length(t_values) > 1
                dt_values = diff(t_values)
                min_dt = minimum(dt_values)
                min_dt_matrix[i, j] = min_dt
            else
                # Solver didn't advance
                @warn "Solver did not advance for chi=$chi, dx=$dx"
                min_dt_matrix[i, j] = NaN
            end

            # Save the result to the new DataFrame
            push!(new_results, (chi, dx, min_dt))
            results[key] = min_dt
        end
    end
    # Append new results to the CSV file
    if !isempty(new_results)
        if isfile(results_file)
            # Combine existing data with new results and save
            combined_data = vcat(CSV.read(results_file, DataFrame), new_results)
            CSV.write(results_file, combined_data)
        else
            # Save new results if file doesn't exist
            CSV.write(results_file, new_results)
        end
    end

    return min_dt_matrix
end


chi_values = 6:1:20  
dx_values = [0.02,0.025,0.04,0.05,0.08,0.1,0.2] 

min_dt_matrix_spline = param_sweep_min_dt(chi_values, dx_values; N1=1.0, N2=1.0, energy_method="spline",results_file="./1d_dt_bdf_spline.csv")
min_dt_matrix_analytical = param_sweep_min_dt(chi_values,dx_values,N1=1.0,N2=1.0,energy_method="analytical",results_file="./1d_dt_bdf_ana.csv")

log_min_dt_spline = log10.(min_dt_matrix_spline)
finite_values_spline = log_min_dt_spline[.!isnan.(log_min_dt_spline)]
cmin_spline = minimum(finite_values_spline)
cmax_spline = maximum(finite_values_spline)

log_min_dt_ana = log10.(min_dt_matrix_analytical)
finite_values_ana = log_min_dt_ana[.!isnan.(log_min_dt_ana)]
cmin_ana = minimum(finite_values_ana)
cmax_ana = maximum(finite_values_ana)

p1= heatmap(dx_values, chi_values, log_min_dt_ana,
    xlabel=L"\Delta x", ylabel=L"\chi_{12}",
    color=:viridis, nan_color=:grey,
    clims=(cmin_ana, cmax_ana),
    colorbar_title = L"\log_{10}(\min(\Delta t))",
    xscale = :log10, grid=false,tickfont=Plots.font("Computer Modern", 10),
    title="TRBDF2, Full",
    titlefont=Plots.font("Computer Modern",12),size=(500,500))


p2= heatmap(dx_values, chi_values, log_min_dt_spline,
    xlabel=L"\Delta x", ylabel=L"\chi_{12}",
    color=:viridis, nan_color=:grey,
    clims=(cmin_spline, cmax_spline),
    colorbar_title = L"\log_{10}(\min(\Delta t))",
    xscale = :log10, grid=false,tickfont=Plots.font("Computer Modern", 10),
    title="TRBDF2, Spline",
    titlefont=Plots.font("Computer Modern",12),size=(500,500))

"""
Running Backward Euler
"""


function run_dt_sweep(chi_values, dx_values; N1=1.0, N2=1.0, energy_method="analytical", dt_start=1.0, dt_min=1e-4,results_file)

    # Load existing results if the file exists
    results = Dict()
    if isfile(results_file)
        existing_data = CSV.read(results_file, DataFrame)
        for row in eachrow(existing_data)
            key = (row.chi, row.dx)
            results[key] = row.largest_stable_dt
        end
    end

    # Function to attempt a simulation given parameters and dt
    function try_simulation(chi, N1, N2, dx, dt, energy_method)
        try
            impliciteuler(chi, N1, N2, dx, dt, energy_method)
            # If we get here, it means the simulation ran without throwing an error.
            return true
        catch e
            # If there's an error, consider the simulation failed
            return false
        end
    end

    # Function to find the largest stable dt for given chi and dx
    function find_largest_stable_dt(chi, N1, N2, dx; energy_method="analytical", dt_start=1.0, dt_min=1e-4)
        dt = dt_start
        while dt >= dt_min
            println("Testing chi=$chi, dx=$dx with dt=$dt,energy_method=$energy_method, Timestepping = Backward Euler")
            success = try_simulation(chi, N1, N2, dx, dt, energy_method)
            if success
                # If successful, we found a stable dt
                return dt
            else
                dt /= 2
            end
        end
        # If we reach here, no stable dt was found above dt_min
        return NaN
    end

    # Initialize the matrix to hold the largest stable dt
    largest_stable_dt = fill(NaN, length(chi_values), length(dx_values))

    # DataFrame to store new results
    new_results = DataFrame(chi=Float64[], dx=Float64[], largest_stable_dt=Float64[])

    # Sweep through chi_values and dx_values
    for (i, chi) in enumerate(chi_values)
        for (j, dx) in enumerate(dx_values)
            key = (chi, dx)
            # Reuse result if available
            if haskey(results, key)
                println("Reusing result for chi=$chi, dx=$dx, energy_method=$energy_method, Timestepping = Backward Euler")
                largest_stable_dt[i, j] = results[key]
                continue
            end
            dt_found = find_largest_stable_dt(chi, N1, N2, dx; energy_method=energy_method, dt_start=dt_start, dt_min=dt_min)
            largest_stable_dt[i,j] = dt_found
            # Save the result to the new DataFrame
            push!(new_results, (chi, dx, dt_found))
            results[key] = dt_found
        end
    end

    # Append new results to the CSV file
    if !isempty(new_results)
        if isfile(results_file)
            # Combine existing data with new results and save
            combined_data = vcat(CSV.read(results_file, DataFrame), new_results)
            CSV.write(results_file, combined_data)
        else
            # Save new results if file doesn't exist
            CSV.write(results_file, new_results)
        end
    end

    return largest_stable_dt
end

dt_vals_backwards_euler_ana = run_dt_sweep(chi_values, dx_values; N1=1.0, N2=1.0, energy_method="analytical", dt_start=0.25, dt_min=1e-4,results_file="./1d_dt_ie_ana.csv")
log_dt_be_ana = log10.(dt_vals_backwards_euler_ana)
finite_values_be_ana = log_dt_be_ana[.!isnan.(log_dt_be_ana)]
cmin_be_ana = minimum(finite_values_be_ana)
cmax_be_ana = maximum(finite_values_be_ana)

p3= heatmap(dx_values, chi_values, log_dt_be_ana,
    xlabel=L"\Delta x", ylabel=L"\chi_{12}",
    color=:viridis, nan_color=:grey,
    clims=(cmin_be_ana, cmax_be_ana),
    colorbar_title = L"\log_{10}(\max(\Delta t))",
    xscale = :log10, grid=false,tickfont=Plots.font("Computer Modern", 10),
    title="Backward Euler, Full",
    titlefont=Plots.font("Computer Modern",12),size=(500,500))


dt_vals_backwards_euler_spline = run_dt_sweep(chi_values, dx_values; N1=1.0, N2=1.0, energy_method="spline", dt_start=0.25, dt_min=1e-4,results_file="./1d_dt_ie_spline.csv")
log_dt_be_spline = log10.(dt_vals_backwards_euler_spline)
finite_values_be_spline = log_dt_be_spline[.!isnan.(log_dt_be_spline)]
cmin_be_spline = minimum(finite_values_be_spline)
cmax_be_spline = maximum(finite_values_be_spline)

p4= heatmap(dx_values, chi_values, log_dt_be_spline,
    xlabel=L"\Delta x", ylabel=L"\chi_{12}",
    color=:viridis, nan_color=:grey,
    clims=(cmin_be_ana, cmax_be_ana),
    colorbar_title = L"\log_{10}(\max(\Delta t))",
    xscale = :log10, grid=false,tickfont=Plots.font("Computer Modern", 10),
    title="Backward Euler, Full",
    titlefont=Plots.font("Computer Modern",12),size=(500,500))


p_all = plot(p1,p2,p3,p4, layout=4, size=(1400,1400), dpi=300, leftmargin=3mm)
savefig(p_all,"1d_benchmark.png")