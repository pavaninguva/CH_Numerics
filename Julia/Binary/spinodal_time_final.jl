using SparseArrays
using LinearAlgebra
using NonlinearSolve
using Plots
using Krylov
using Trapz
using IterativeSolvers
using Random
using NaNMath
using DataFrames
using LaTeXStrings
using CSV
# using BSplineKit
using GLM
using DifferentialEquations
include("../pchip.jl") 

"""
This code is used to generate the plot:
spinodal_times.png

It runs both the implicit euler and adaptive TRBDF2 schemes to 
solve the CH equation for a range of simulation and material Parameters

It tests if the datasets are available in csv form and if not, the simulation
are run

"""

Random.seed!(1234)  # For reproducibility

"""
Supporting functions
"""

# Function to compute phiA and phiB for given chi, N1, N2
function compute_binodal(chi, N1, N2)
    
    # Parameters
    params = (chi, N1, N2)

    # Define the function to solve
    function binodal_eqs!(F, phi, params)
        chi, N1, N2 = params

        function fh_deriv(phi, chi, N1, N2)
            df = (1/N1)*NaNMath.log(phi) + (1/N1) - (1/N2)*NaNMath.log(1-phi) - (1/N2) - 2*chi*phi + chi
            return df
        end

        function osmotic(phi, chi, N1, N2)
            osmo = phi*((1/N1)-(1/N2)) - (1/N2)*NaNMath.log(1-phi) - chi*phi^2
            return osmo
        end

            phiA = phi[1]
            phiB = phi[2]
            dF = fh_deriv(phiA, chi, N1, N2) - fh_deriv(phiB, chi, N1, N2)
            dO = osmotic(phiA, chi, N1, N2) - osmotic(phiB, chi, N1, N2)
            F[1] = dF
            F[2] = dO
        end

    # Create the NonlinearProblem using NonlinearSolve.jl

    # Initial guess for phiA and phiB
    phi_guess = [1e-3, 1 - 1e-3]
    problem = NonlinearProblem(binodal_eqs!, phi_guess, params)

    # Solve the problem
    solution = solve(problem, RobustMultiNewton(),show_trace=Val(false))

    phiA = solution.u[1]
    phiB = solution.u[2]

    # Ensure phiA < phiB
    if phiA > phiB
        phiA, phiB = phiB, phiA
    end

    return phiA, phiB
    
end

# function spline_generator(χ,N1,N2,knots=100)

#     #Def log terms 
#     log_terms(ϕ) =  (ϕ./N1).*log.(ϕ) .+ ((1 .-ϕ)./N2).*log.(1 .-ϕ)

#     function tanh_sinh_spacing(n, β)
#         points = 0.5 * (1 .+ tanh.(β .* (2 * collect(0:n-1) / (n-1) .- 1)))
#         return points
#     end
    
#     phi_vals_ = collect(tanh_sinh_spacing(knots-2,14))
#     f_vals_ = log_terms(phi_vals_)

#     #Append boundary values
#     phi_vals = pushfirst!(phi_vals_,0)
#     f_vals = pushfirst!(f_vals_,0)
#     push!(phi_vals,1)
#     push!(f_vals,0)

#     spline = BSplineKit.interpolate(phi_vals, f_vals,BSplineOrder(4))
#     d_spline = Derivative(1)*spline

#     df_spline(phi) = d_spline.(phi) .+ χ.*(1 .- 2*phi)

#     return df_spline
# end

function fh_deriv(phi,chi,N1,N2)
    df = (1/N1).*log.(phi) .+ (1/N1) .- (1/N2).*log.(1 .- phi) .- (1/N2) .- 2*chi.*phi .+ chi
    return df
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

"""
MOL Solver functions
"""

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

function mol_time(chi, N1, N2, dx, phiA, phiB, time_method, energy_method)
    #Simulation Parameters
    L = 4.0
    tf = 1000.0
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

    #Callback for checking if lower or upper
    function condition_lower(u, t, integrator)
        minimum(u) - 0.9*phiA  # Hits zero when any state equals the lower threshold
    end

    # Condition function for the ContinuousCallback (upper threshold)
    function condition_upper(u, t, integrator)
        maximum(u) - 0.9*phiB  # Hits zero when any state equals the upper threshold
    end

    # Affect function for the ContinuousCallback
    function affect!(integrator)
        println("Terminating at t = $(integrator.t)")
        terminate!(integrator)
    end

    # Create the ContinuousCallbacks
    cb_lower = ContinuousCallback(condition_lower, affect!)
    cb_upper = ContinuousCallback(condition_upper, affect!)

    # Combine callbacks
    cb = CallbackSet(cb_lower, cb_upper)


    # Set up the problem
    prob = ODEProblem(ode_system!, c0, (0.0, tf))
    if time_method == "BDF"
        sol = solve(prob, TRBDF2(),callback=cb,reltol=1e-8, abstol=1e-8)
    elseif time_method == "Rosenbrock"
        sol = solve(prob, Rosenbrock23(),callback=cb,reltol=1e-8, abstol=1e-8)
    end
    println("Solution terminated at t = $(sol.t[end])")
    return sol.t[end]

end

"""
Implicit Euler solver functions

"""
# Modified impliciteuler function that returns the time tau
function impliciteuler(chi, N1, N2, dx, dt, phiA, phiB,energy_method)
    L = 4.0
    tf = 1000.0
    nx = Int(L / dx) + 1
    x = range(0, L, length = nx)
    nt = Int(tf / dt)
    kappa = (2 / 3) * chi

    # Initial condition: small random perturbation around c0
    c0 = 0.5
    c = c0 .+ 0.02 * (rand(nx) .- 0.5)

    # Time when c_max and c_min are within 1% of phiB and phiA
    tau = NaN

    function M_func(phi)
        return phi .* (1 .- phi)
    end
    function M_func_half(phi1, phi2)
        return M_func(0.5 .* (phi1 .+ phi2))
    end
    
    spline = spline_generator(chi,N1,N2,100)

    dfdphi = (phi) -> begin 
        if energy_method == "analytical"
            fh_deriv(phi,chi,N1,N2)
        else
            spline.(phi)
        end
    end

    # Residual function with Neumann boundary conditions
    function residual!(F, c_new, p)
        c_old = p.c_old
        dt = p.dt
        dx = p.dx
        kappa = p.kappa
        chi = p.chi
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


    n= 0
    while isnan(tau) && n * dt < tf
        n += 1
        # Save the old concentration profile
        c_old = copy(c)

        # Parameters to pass to the residual function
        p = (c_old = c_old, dt = dt, dx = dx, kappa = kappa, chi = chi)

        # Initial guess for c_new
        c_guess = copy(c_old)

        # Create the NonlinearProblem
        problem = NonlinearProblem(residual!, c_guess, p)

        # Solve the nonlinear system
        solver = solve(problem, NewtonRaphson(linsolve = KrylovJL_GMRES()), show_trace = Val(false))

        # Update c for the next time step
        c = solver.u

        # Max and min Values
        max_c = maximum(c)
        min_c = minimum(c)
        
        # Check if c_max or c_min are within 10% of phiB or phiA
        if abs(max_c - phiB) / abs(phiB) <= 0.1 || abs(min_c - phiA) / abs(phiA) <= 0.1
            tau = n * dt
            break  # Exit the time-stepping loop
        end
    end

    # Return the time tau when c reaches equilibrium within 1%
    return tau
end

"""
Parameter sweep for IE
"""
function tau0(phi0, kappa, chi, N1, N2)
    #Mobility function
    mobility(phi) = phi.*(1-phi)

    chi_s(N1,N2,chi,phi) = 0.5*(1/(N1*phi0) + 1/(N2*(1-phi0)))

    tau = (kappa)/(mobility(phi0)*(chi-chi_s(N1,N2,chi,phi0))^2)

    return tau

end
# Parameters
N1 = 1.0
N2 = 1.0
dx_values = [0.02, 0.05, 0.1]
dt_values = [0.01,0.05,0.1]

# Range of chi values
chi_values = range(3,12,15)

tau_values_full_ie = []
tau_values_spline_ie = []


# Filenames for the CSV files
ie_full_csv = "../Julia/Binary/SpinodalTimes_Data/spinodal_times_ie_full.csv"
ie_spline_csv = "../Julia/Binary/SpinodalTimes_Data/spinodal_times_ie_spline.csv"

if isfile(ie_full_csv) && isfile(ie_spline_csv)
    println("CSV files found. Loading DataFrames from CSV.")
    df_ie_full = CSV.read(ie_full_csv, DataFrame)
    df_ie_spline = CSV.read(ie_spline_csv, DataFrame)
else
    println("Files not found for IE, running now")
    for dt in dt_values
        for dx in dx_values
            for chi in chi_values
                println("Running for dx=$dx, dt=$dt, chi=$chi")
                #Compute binodal
                phiA, phiB = compute_binodal(chi, N1, N2)
                #Store taus for 10 runs
                taus_full = []
                taus_spline = []
                for i in 1:10
                    println("Run $i")
                    tau_spline = impliciteuler(chi,N1,N2,dx,dt,phiA,phiB,"spline")
                    tau_full = impliciteuler(chi,N1,N2,dx,dt,phiA,phiB,"analytical")
                    println("Tau_Full = $tau_full")
                    println("Tau_Spline = $tau_spline")
                    push!(taus_full, tau_full)
                    push!(taus_spline, tau_spline)
                end
                # Store tau values with chi
                for tau in taus_full
                    push!(tau_values_full_ie, (chi, dx, dt, tau))
                end
                for tau in taus_spline
                    push!(tau_values_spline_ie, (chi, dx, dt, tau))
                end
            end
        end
    end

    df_ie_full = DataFrame(
            tau0 = [tau0(0.5, (2/3)*x[1], x[1], N1, N2) for x in tau_values_full_ie],
            tau = [x[4] for x in tau_values_full_ie],
            dx = [x[2] for x in tau_values_full_ie],
            dt = [x[3] for x in tau_values_full_ie]
        )

    df_ie_spline = DataFrame(
        tau0 = [tau0(0.5, (2/3)*x[1], x[1], N1, N2) for x in tau_values_spline_ie],
        tau = [x[4] for x in tau_values_spline_ie],
        dx = [x[2] for x in tau_values_spline_ie],
        dt = [x[3] for x in tau_values_spline_ie]
    )
    CSV.write(ie_full_csv, df_ie_full)
    CSV.write(ie_spline_csv, df_ie_spline)
end



"""
Parameter sweep for BDF
"""

# Filenames for the CSV files
bdf_full_csv = "../Julia/Binary/SpinodalTimes_Data/spinodal_times_bdf_full.csv"
bdf_spline_csv = "../Julia/Binary/SpinodalTimes_Data/spinodal_times_bdf_spline.csv"

if isfile(bdf_full_csv) && isfile(bdf_spline_csv)
    println("CSV files found. Loading DataFrames from CSV.")
    df_bdf_full = CSV.read(bdf_full_csv, DataFrame)
    df_bdf_spline = CSV.read(bdf_spline_csv, DataFrame)
else
    # Initialize array to store tau values
    tau_values_full_bdf = []
    tau_values_spline_bdf = []

    for dx in dx_values
        println("Running for dx = $dx")
        for chi in chi_values
            println("Chi = $chi")
            # Compute phiA and phiB
            phiA, phiB = compute_binodal(chi, N1, N2)
            println("phiA = $phiA, phiB = $phiB")
            #Run 10 times
            taus_full = []
            taus_spline = []
            for i in 1:10
                println("Run $i")
                tau_full = mol_time(chi, N1, N2, dx, phiA, phiB,"BDF","analytical")
                tau_spline = mol_time(chi, N1, N2, dx, phiA, phiB,"BDF","spline")
                println("Tau_Full = $tau_full")
                println("Tau_Spline = $tau_spline")
                push!(taus_full, tau_full)
                push!(taus_spline, tau_spline)
            end
            # Store tau values with chi
            for tau in taus_full
                push!(tau_values_full_bdf, (chi, dx, tau))
            end
            for tau in taus_spline
                push!(tau_values_spline_bdf, (chi, dx, tau))
            end
        end
    end

    df_bdf_full = DataFrame(
        tau0 = [tau0(0.5, (2/3)*x[1], x[1], N1, N2) for x in tau_values_full_bdf],
        tau = [x[3] for x in tau_values_full_bdf],
        dx = [x[2] for x in tau_values_full_bdf]
    )

    df_bdf_spline = DataFrame(
        tau0 = [tau0(0.5, (2/3)*x[1], x[1], N1, N2) for x in tau_values_spline_bdf],
        tau = [x[3] for x in tau_values_spline_bdf],
        dx = [x[2] for x in tau_values_spline_bdf]
    )

    CSV.write(bdf_full_csv, df_bdf_full)
    CSV.write(bdf_spline_csv, df_bdf_spline)
end



"""
Make the big plot
"""

colors = [palette(:auto)[1], palette(:auto)[1], palette(:auto)[1], palette(:auto)[2], palette(:auto)[2], palette(:auto)[2], palette(:auto)[3], palette(:auto)[3], palette(:auto)[3]]
markers = [:circle, :square, :+, :circle, :square, :+,:circle, :square, :+ ]


#### IE Full
labels_ie = [
    L"\Delta x=0.02, \Delta t=0.02",
    L"\Delta x=0.05, \Delta t=0.02",
    L"\Delta x=0.1, \Delta t=0.02",
    L"\Delta x=0.02, \Delta t=0.05",
    L"\Delta x=0.05, \Delta t=0.05",
    L"\Delta x=0.1, \Delta t=0.05",
    L"\Delta x=0.02, \Delta t=0.1",
    L"\Delta x=0.05, \Delta t=0.1",
    L"\Delta x=0.1, \Delta t=0.1"
]

p1 = scatter(
    xlabel = L"\tau_0",
    ylabel = L"\tau",
    legend = :topleft,size=(400,400),tickfont=Plots.font("Computer Modern", 12), grid=false,
    legendfont=Plots.font("Computer Modern",8),dpi=300,xaxis=:log, yaxis=:log,
    title="Backwards Euler, Full",
    titlefont=Plots.font("Computer Modern",12)
)

grouped_df = groupby(df_ie_full, [:dx, :dt])

for (i,subdf) in enumerate(grouped_df)
    # Extract the unique dx and dt values from the group
    dx_val = unique(subdf.dx)[1]
    dt_val = unique(subdf.dt)[1]
    println((dx_val,dt_val))
    scatter!(p1, subdf.tau0, subdf.tau, label=labels_ie[i], marker=markers[i],color=colors[i],alpha=0.6)
end

#Regression
IE_Full_DF = DataFrame(x=log10.(df_ie_full.tau0),y=log10.(df_ie_full.tau))
IE_Full_DF[!,:adjusted_y] = IE_Full_DF.y .- IE_Full_DF.x
#Compute first OLS
model_ie_full_ols = lm(@formula(adjusted_y~ 1), IE_Full_DF)
println(10 .^ coef(model_ie_full_ols))
println(10 .^ confint(model_ie_full_ols))

m_ie_full = 10 .^ coef(model_ie_full_ols)[1]
ie_full_tau0_vals = range(minimum(df_ie_full.tau0), maximum(df_ie_full.tau0),100)
ie_full_tau_vals = m_ie_full.*ie_full_tau0_vals
plot!(p1,ie_full_tau0_vals,ie_full_tau_vals,
    label = "q = $(round(m_ie_full, digits=5))",
    color = :black,
    linewidth = 2,
    linestyle = :dash,
    alpha=0.5)


#### IE Spline
p2 = scatter(
    xlabel = L"\tau_0",
    ylabel = L"\tau",
    legend = :topleft,size=(400,400),tickfont=Plots.font("Computer Modern", 12), grid=false,
    legendfont=Plots.font("Computer Modern",8),dpi=300,xaxis=:log, yaxis=:log,
    title="Backwards Euler, Spline",
    titlefont=Plots.font("Computer Modern",12)
)

grouped_df = groupby(df_ie_spline, [:dx, :dt])

for (i,subdf) in enumerate(grouped_df)
    # Extract the unique dx and dt values from the group
    dx_val = unique(subdf.dx)[1]
    dt_val = unique(subdf.dt)[1]
    println((dx_val,dt_val))
    scatter!(p2, subdf.tau0, subdf.tau, label=labels_ie[i], marker=markers[i],color=colors[i],alpha=0.6)
end

#Regression
IE_Spline_DF = DataFrame(x=log10.(df_ie_spline.tau0),y=log10.(df_ie_spline.tau))
IE_Spline_DF[!,:adjusted_y] = IE_Spline_DF.y .- IE_Spline_DF.x
#Compute first OLS
model_ie_spline_ols = lm(@formula(adjusted_y~ 1), IE_Spline_DF)
println(10 .^ coef(model_ie_spline_ols))
println(10 .^ confint(model_ie_spline_ols))

m_ie_spline = 10 .^ coef(model_ie_spline_ols)[1]
ie_spline_tau0_vals = range(minimum(df_ie_spline.tau0), maximum(df_ie_spline.tau0),100)
ie_spline_tau_vals = m_ie_spline.*ie_spline_tau0_vals
plot!(p2,ie_spline_tau0_vals,ie_spline_tau_vals,
    label = "q = $(round(m_ie_spline, digits=5))",
    color = :black,
    linewidth = 2,
    linestyle = :dash,
    alpha=0.5)


### TRBDF2 Full
bdf_labels = [L"\Delta x = 0.1", L"\Delta x = 0.05", L"\Delta x = 0.02"]
p3 = scatter(
    xlabel = L"\tau_0",
    ylabel = L"\tau",
    legend = :topleft,size=(400,400),tickfont=Plots.font("Computer Modern", 12), grid=false,
    legendfont=Plots.font("Computer Modern",8),dpi=300,xaxis=:log, yaxis=:log,
    title="TRBDF2, Full",
    titlefont=Plots.font("Computer Modern",12)
)

for (i,dx) in enumerate(dx_values)
    subset_df = filter(row -> row[:dx] == dx, df_bdf_full)
    scatter!(p3, subset_df.tau0, subset_df.tau,
        label=bdf_labels[i],
        marker = markers[i]
    )
end

## BDF - Full
BDF_Full_DF = DataFrame(x=log10.(df_bdf_full.tau0),y=log10.(df_bdf_full.tau))
BDF_Full_DF[!,:adjusted_y] = BDF_Full_DF.y .- BDF_Full_DF.x
#Compute first OLS
model_bdf_full_ols = lm(@formula(adjusted_y~ 1), BDF_Full_DF)
println(10 .^ coef(model_bdf_full_ols))
println(10 .^ confint(model_bdf_full_ols))

m_BDF_full = 10 .^ coef(model_bdf_full_ols)[1]
bdf_full_tau0_vals = range(minimum(df_bdf_full.tau0), maximum(df_bdf_full.tau0),100)
bdf_full_tau_vals = m_BDF_full.*bdf_full_tau0_vals
plot!(p3,bdf_full_tau0_vals,bdf_full_tau_vals,
    label = "q = $(round(m_BDF_full, digits=5))",
    color = :black,
    linewidth = 2,
    linestyle = :dash,
    alpha=0.5)

### TRBDF2 Spline
p4 = scatter(
    xlabel = L"\tau_0",
    ylabel = L"\tau",
    legend = :topleft,size=(400,400),tickfont=Plots.font("Computer Modern", 12), grid=false,
    legendfont=Plots.font("Computer Modern",8),dpi=300,xaxis=:log, yaxis=:log,
    title="TRBDF2, Spline",
    titlefont=Plots.font("Computer Modern",12)
)

for (i,dx) in enumerate(dx_values)
    subset_df = filter(row -> row[:dx] == dx, df_bdf_spline)
    scatter!(p4, subset_df.tau0, subset_df.tau,
        label=bdf_labels[i],
        marker = markers[i]
    )
end

## BDF - Spline
BDF_spline_DF = DataFrame(x=log10.(df_bdf_spline.tau0),y=log10.(df_bdf_spline.tau))
BDF_spline_DF[!,:adjusted_y] = BDF_spline_DF.y .- BDF_spline_DF.x
#Compute first OLS
model_bdf_spline_ols = lm(@formula(adjusted_y~ 1), BDF_spline_DF)
println(10 .^ coef(model_bdf_spline_ols))
println(10 .^ confint(model_bdf_spline_ols))

m_BDF_spline = 10 .^ coef(model_bdf_spline_ols)[1]
bdf_spline_tau0_vals = range(minimum(df_bdf_spline.tau0), maximum(df_bdf_spline.tau0),100)
bdf_spline_tau_vals = m_BDF_spline.*bdf_spline_tau0_vals
plot!(p4,bdf_spline_tau0_vals,bdf_spline_tau_vals,
    label = "q = $(round(m_BDF_spline, digits=5))",
    color = :black,
    linewidth = 2,
    linestyle = :dash,
    alpha=0.5)


"""
Make Big Plot
"""

# Combine all plots into one figure
p_all = plot(p1, p2, p3, p4, layout=4, size=(1000, 1000))
savefig(p_all,"spinodal_times.png")


