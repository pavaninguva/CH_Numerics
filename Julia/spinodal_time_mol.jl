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
using BSplineKit
using GLM

"""
This code is used to generate the plot:
spinodal_times.png

It also reads in data generated from spinodal_time.jl 
for the Backwards Euler Case

"""

Random.seed!(1234)  # For reproducibility

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

# Parameters
N1 = 1.0
N2 = 1.0
dx_values = [0.1, 0.05, 0.02]

# Range of chi values
chi_values = range(3,12,15)

# Initialize array to store tau values
tau_values_full = []
tau_values_spline = []

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
            push!(tau_values_full, (chi, dx, tau))
        end
        for tau in taus_spline
            push!(tau_values_spline, (chi, dx, tau))
        end
    end
end


function tau0(phi0, kappa, chi, N1, N2)
    #Mobility function
    mobility(phi) = phi.*(1-phi)

    chi_s(N1,N2,chi,phi) = 0.5*(1/(N1*phi0) + 1/(N2*(1-phi0)))

    tau = (kappa)/(mobility(phi0)*(chi-chi_s(N1,N2,chi,phi0))^2)

    return tau

end

df_bdf_full = DataFrame(
    tau0 = [tau0(0.5, (2/3)*x[1], x[1], N1, N2) for x in tau_values_full],
    tau = [x[3] for x in tau_values_full],
    dx = [x[2] for x in tau_values_full]
)

df_bdf_spline = DataFrame(
    tau0 = [tau0(0.5, (2/3)*x[1], x[1], N1, N2) for x in tau_values_spline],
    tau = [x[3] for x in tau_values_spline],
    dx = [x[2] for x in tau_values_spline]
)



"""
Make the big plot
"""

#IE Analytical
df1_IE_Ana = CSV.read("./Spinodal_Times_Data/Backwards_Euler_Full/spinodal_times_dt0.1_dx0.1.csv", DataFrame)
df2_IE_Ana = CSV.read("./Spinodal_Times_Data/Backwards_Euler_Full/spinodal_times_dt0.1_dx0.05.csv",DataFrame)
df3_IE_Ana = CSV.read("./Spinodal_Times_Data/Backwards_Euler_Full/spinodal_times_dt0.1_dx0.02.csv", DataFrame)
df4_IE_Ana = CSV.read("./Spinodal_Times_Data/Backwards_Euler_Full/spinodal_times_dt0.05_dx0.1.csv",DataFrame)
df5_IE_Ana = CSV.read("./Spinodal_Times_Data/Backwards_Euler_Full/spinodal_times_dt0.05_dx0.05.csv", DataFrame)
df6_IE_Ana = CSV.read("./Spinodal_Times_Data/Backwards_Euler_Full/spinodal_times_dt0.05_dx0.02.csv",DataFrame)
df7_IE_Ana = CSV.read("./Spinodal_Times_Data/Backwards_Euler_Full/spinodal_times_dt0.01_dx0.1.csv", DataFrame)
df8_IE_Ana = CSV.read("./Spinodal_Times_Data/Backwards_Euler_Full/spinodal_times_dt0.01_dx0.05.csv",DataFrame)
df9_IE_Ana = CSV.read("./Spinodal_Times_Data/Backwards_Euler_Full/spinodal_times_dt0.01_dx0.02.csv",DataFrame)

IE_Ana_dfs = [df1_IE_Ana, df2_IE_Ana, df3_IE_Ana, df4_IE_Ana, df5_IE_Ana, df6_IE_Ana, df7_IE_Ana, df8_IE_Ana, df9_IE_Ana]
labels = [
    L"\Delta t=0.1, \Delta x=0.1",
    L"\Delta t=0.1, \Delta x=0.05",
    L"\Delta t=0.1, \Delta x=0.02",
    L"\Delta t=0.05, \Delta x=0.1",
    L"\Delta t=0.05, \Delta x=0.05",
    L"\Delta t=0.05, \Delta x=0.02",
    L"\Delta t=0.01, \Delta x=0.1",
    L"\Delta t=0.01, \Delta x=0.05",
    L"\Delta t=0.01, \Delta x=0.02"
]

p1 = scatter(
    xlabel = L"\tau_0",
    ylabel = L"\tau",
    legend = :topleft,size=(400,400),tickfont=Plots.font("Computer Modern", 12), grid=false,
    legendfont=Plots.font("Computer Modern",8),dpi=300,xaxis=:log, yaxis=:log,
    title="Backwards Euler, Full",
    titlefont=Plots.font("Computer Modern",12)
)

colors = [palette(:auto)[1], palette(:auto)[1], palette(:auto)[1], palette(:auto)[2], palette(:auto)[2], palette(:auto)[2], palette(:auto)[3], palette(:auto)[3], palette(:auto)[3]]
markers = [:circle, :square, :+, :circle, :square, :+,:circle, :square, :+ ]

for (i, df) in enumerate(IE_Ana_dfs)
    scatter!(p1,
        df.tau0,
        df.tau,
        label = labels[i],
        color = colors[i],
        marker = markers[i],
        ms = 5,  # marker size
        alpha = 0.6
    )
end

IE_full_df_all = vcat(df1_IE_Ana,df2_IE_Ana,df3_IE_Ana,df4_IE_Ana,df5_IE_Ana,df6_IE_Ana,df7_IE_Ana,df8_IE_Ana,df9_IE_Ana)

#IE - Full
IE_Full_DF = DataFrame(x=log10.(IE_full_df_all.tau0), y=log10.(IE_full_df_all.tau))
IE_Full_DF[!,:adjusted_y] = IE_Full_DF.y .- IE_Full_DF.x
#Compute OLS
model_ie_full_ols = lm(@formula(adjusted_y~ 1), IE_Full_DF)
println(10 .^ coef(model_ie_full_ols))
println(10 .^ confint(model_ie_full_ols))

m_IE_full = 10 .^ coef(model_ie_full_ols)[1]
IE_full_tau0_vals = range(minimum(IE_full_df_all.tau0), maximum(IE_full_df_all.tau0),100)
IE_full_tau_vals = m_IE_full.*IE_full_tau0_vals
plot!(p1,IE_full_tau0_vals,IE_full_tau_vals,
    label = "q = $(round(m_IE_full, digits=5))",
    color = :black,
    linewidth = 2,
    linestyle = :dash,
    alpha=0.5)


## IE spline
df1_IE_spline = CSV.read("./Spinodal_Times_Data/Backwards_Euler_Spline/spinodal_times_IE_Spline_dt0.1_dx0.1.csv", DataFrame)
df2_IE_spline = CSV.read("./Spinodal_Times_Data/Backwards_Euler_Spline/spinodal_times_IE_Spline_dt0.1_dx0.05.csv", DataFrame)
df3_IE_spline = CSV.read("./Spinodal_Times_Data/Backwards_Euler_Spline/spinodal_times_IE_Spline_dt0.1_dx0.02.csv", DataFrame)
df4_IE_spline = CSV.read("./Spinodal_Times_Data/Backwards_Euler_Spline/spinodal_times_IE_Spline_dt0.05_dx0.1.csv", DataFrame)
df5_IE_spline = CSV.read("./Spinodal_Times_Data/Backwards_Euler_Spline/spinodal_times_IE_Spline_dt0.05_dx0.05.csv", DataFrame)
df6_IE_spline = CSV.read("./Spinodal_Times_Data/Backwards_Euler_Spline/spinodal_times_IE_Spline_dt0.05_dx0.02.csv", DataFrame)
df7_IE_spline = CSV.read("./Spinodal_Times_Data/Backwards_Euler_Spline/spinodal_times_IE_Spline_dt0.01_dx0.1.csv", DataFrame)
df8_IE_spline = CSV.read("./Spinodal_Times_Data/Backwards_Euler_Spline/spinodal_times_IE_Spline_dt0.01_dx0.05.csv", DataFrame)
df9_IE_spline = CSV.read("./Spinodal_Times_Data/Backwards_Euler_Spline/spinodal_times_IE_Spline_dt0.01_dx0.02.csv", DataFrame)


IE_Spline_dfs = [df1_IE_spline, df2_IE_spline, df3_IE_spline, df4_IE_spline, df5_IE_spline, df6_IE_spline, df7_IE_spline, df8_IE_spline, df9_IE_spline]

p2 = scatter(
    xlabel = L"\tau_0",
    ylabel = L"\tau",
    legend = :topleft,size=(400,400),tickfont=Plots.font("Computer Modern", 12), grid=false,
    legendfont=Plots.font("Computer Modern",8),dpi=300,xaxis=:log, yaxis=:log,
    title="Backwards Euler, Spline",
    titlefont=Plots.font("Computer Modern",12)
)

for (i, df) in enumerate(IE_Spline_dfs)
    scatter!(p2,
        df.tau0,
        df.tau,
        label = labels[i],
        color = colors[i],
        marker = markers[i],
        ms = 5,  # marker size
        alpha = 0.6
    )
end

IE_spline_df_all = vcat(df1_IE_spline,df2_IE_spline,df3_IE_spline,df4_IE_spline,df5_IE_spline,df6_IE_spline,df7_IE_spline,df8_IE_spline,df9_IE_spline)

#IE_Spline
IE_spline_DF = DataFrame(x=log10.(IE_spline_df_all.tau0), y=log10.(IE_spline_df_all.tau))
IE_spline_DF[!,:adjusted_y] = IE_spline_DF.y .- IE_spline_DF.x
#Compute first OLS
model_ie_spline_ols = lm(@formula(adjusted_y~ 1), IE_spline_DF)
println(10 .^ coef(model_ie_spline_ols))
println(10 .^ confint(model_ie_spline_ols))

m_IE_spline = 10 .^ coef(model_ie_spline_ols)[1]
IE_spline_tau0_vals = range(minimum(IE_spline_df_all.tau0), maximum(IE_spline_df_all.tau0),100)
IE_spline_tau_vals = m_IE_spline.*IE_full_tau0_vals
plot!(p2,IE_spline_tau0_vals,IE_spline_tau_vals,
    label = "q = $(round(m_IE_spline, digits=5))",
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
p_all = plot(p1, p2, p3, p4, layout=4, size=(1000, 500))
savefig(p_all,"spinodal_times.png")


