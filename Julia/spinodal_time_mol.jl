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

    
    # spline = Spline1D(phi_vals,f_vals)
    # d_spline(phi) = Dierckx.derivative(spline,phi)
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

    function M_func_half(ϕ₁,ϕ₂,option=1)
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

function mol_time(chi, N1, N2, dx, phiA, phiB)
    #Simulation Parameters
    L = 4.0
    tf = 1000.0
    nx = Int(L / dx) + 1
    x = range(0, L, length = nx)
    kappa = (2 / 3) * chi

    # Initial condition: small random perturbation around c0
    c0 = 0.5
    c = c0 .+ 0.02 * (rand(nx) .- 0.5)

    #Set up MOL bits
end