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

"""
This code is used to generate the csv files containing the 
data in:
spinodal_times.png
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

# Modified impliciteuler function that returns the time tau
function impliciteuler(chi, N1, N2, dx, dt, phiA, phiB)
    L = 4.0
    tf = 10000.0
    nx = Int(L / dx) + 1
    x = range(0, L, length = nx)
    nt = Int(tf / dt)
    kappa = (2 / 3) * chi

    # Initial condition: small random perturbation around c0
    c0 = 0.5
    c = c0 .+ 0.02 * (rand(nx) .- 0.5)

    # Initialize arrays to store results
    c_max = zeros(nt)
    c_min = zeros(nt)
    c_avg = zeros(nt)
    energy = zeros(nt)
    time_array = []

    # Time when c_max and c_min are within 1% of phiB and phiA
    tau = NaN

    # Functions
    function flory_huggins(phi, chi, N1, N2)
        # Use NaNMath.log to avoid domain errors
        return (1 / N1) * (phi .* NaNMath.log.(phi)) + (1 / N2) * (1 .- phi) .* NaNMath.log.(1 .- phi) + chi .* phi .* (1 .- phi)
    end

    function dfdphi_fh(phi, chi)
        # Use NaNMath.log to avoid domain errors
        return (1 / N1) * NaNMath.log.(phi) - (1 / N2) * NaNMath.log.(1 .- phi) + chi * (1 .- 2 .* phi)
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
    
        
        # spline = Spline1D(phi_vals,f_vals)
        # d_spline(phi) = Dierckx.derivative(spline,phi)
        spline = BSplineKit.interpolate(phi_vals, f_vals,BSplineOrder(4))
        d_spline = Derivative(1)*spline
    
        df_spline(phi,chi) = d_spline.(phi) .+ chi.*(1 .- 2*phi)
    
        return df_spline
    end

    function M_func(phi)
        return phi .* (1 .- phi)
    end

    function M_func_half(phi1, phi2)
        return M_func(0.5 .* (phi1 .+ phi2))
    end
    
    spline = spline_generator(chi,N1,N2)

    energy_method="spline"

    dfdphi = (phi, chi) -> begin 
        if energy_method == "analytical"
            # println("Using Analytical FH")
            dfdphi_fh(phi,chi)
        else
            
            spline.(phi,chi)
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
        mu_new[1] = dfdphi(c_work[1], chi) - (2.0 * kappa / dx^2) * (c_work[2] - c_work[1])

        # Interior points
        for i in 2:nx - 1
            mu_new[i] = dfdphi(c_work[i], chi) - (kappa / dx^2) * (c_work[i + 1] - 2.0 * c_work[i] + c_work[i - 1])
        end

        # Right boundary (Neumann BC)
        mu_new[nx] = dfdphi(c_work[nx], chi) - (2.0 * kappa / dx^2) * (c_work[nx - 1] - c_work[nx])

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

    # Time-stepping loop
    if energy_method == "analytical"
        println("Using Analytical FH")
    else
        println("Using Spline")
    end
    n= 0
    while isnan(tau) && n * dt < tf
        println(n*dt)
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

        # Mass conservation
        avg_c = (1 / L) * trapz(x, c)
        push!(c_avg, avg_c)

        # Max and min Values
        max_c = maximum(c)
        min_c = minimum(c)
        println(max_c,min_c)
        push!(c_max, max_c)
        push!(c_min, min_c)
        push!(time_array, n * dt)

        # Check if c_max or c_min are within 10% of phiB or phiA
        if abs(max_c - phiB) / abs(phiB) <= 0.1 || abs(min_c - phiA) / abs(phiA) <= 0.1
            tau = n * dt
            break  # Exit the time-stepping loop
        end
    end

    # Return the time tau when c reaches equilibrium within 1%
    return tau
end

# Parameters
N1 = 1.0
N2 = 1.0
dx = 0.02
dt = 0.05

# Range of chi values
chi_values = range(3,12,15)

# Initialize array to store tau values
tau_values = []

# For each chi, compute phiA and phiB, run simulations 10 times, collect tau values
for chi in chi_values
    println("Chi = $chi")

    # Compute phiA and phiB
    phiA, phiB = compute_binodal(chi, N1, N2)
    println("phiA = $phiA, phiB = $phiB")

  
    # Run simulations 10 times
    taus = []
    for i in 1:10
        println("  Run $i")
        tau = impliciteuler(chi, N1, N2, dx, dt, phiA, phiB)
        println("Tau = $tau")
        push!(taus, tau)
    end

    # Store tau values with chi
    for tau in taus
        push!(tau_values, (chi, tau))
    end
end

# Convert tau_values to DataFrame

function tau0(phi0, kappa, chi, N1, N2)
    #Mobility function
    mobility(phi) = phi.*(1-phi)

    chi_s(N1,N2,chi,phi) = 0.5*(1/(N1*phi0) + 1/(N2*(1-phi0)))

    tau = (kappa)/(mobility(phi0)*(chi-chi_s(N1,N2,chi,phi0))^2)

    return tau

end

df = DataFrame(tau0 = [tau0(0.5,(2/3)*x[1], x[1],1,1) for x in tau_values], tau = [x[2] for x in tau_values])
CSV.write("spinodal_times_IE_Spline_dt0.05_dx0.02.csv",df)

#Fit straight line y = mx
m1 = sum(df.tau0 .* df.tau) / sum(df.tau0.^2)
tau0_vals = range(minimum(df.tau0), maximum(df.tau0),100)
println(m1)


# Plot
p1 = scatter(df.tau0, df.tau, xlabel = L"\tau_0", ylabel = L"\tau", label =L"\\Delta x = 0.1, \Delta t = 0.01")
plot!(p1,tau0_vals,m1.*tau0_vals,label="",linestyle=:dash)
plot!(p1,size=(500,500),tickfont=Plots.font("Computer Modern", 12), grid=false,
legendfont=Plots.font("Computer Modern",8),dpi=300,xaxis=:log, yaxis=:log)