using Plots
using NonlinearSolve
using NaNMath
using LaTeXStrings
using BSplineKit
using DifferentialEquations
include("./pchip.jl") 
"""
This script runs the analytical and numerical methods to compute the 
interfacial tension for a binary mixture. 

The main output is interfacial_tension.png
"""

"""
Functions for analytical
"""

function trapezoidal_integration(x, y)
    n = length(x)
    integral = 0.0
    for i in 1:(n-1)
        # integral += 0.5 * (x[i+1] - x[i]) * (y[i+1] + y[i])
        # Replace NaN values in y with 0
        y_i = isnan(y[i]) ? 0.0 : y[i]
        y_ip1 = isnan(y[i+1]) ? 0.0 : y[i+1]

        # Trapezoidal rule integration step
        integral += 0.5 * (x[i+1] - x[i]) * (y_ip1 + y_i)
    end
    return integral
end

#Compute the binodal
function solve_binodal(chi, N1, N2)
    # Define the binodal function
    function binodal!(F, phi, params)
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

    # Set up the problem and initial guess
    initial_guess = [1e-6, 1-1e-6]
    params = (chi, N1, N2)
    problem = NonlinearProblem(binodal!, initial_guess, params)

    # Solve using NonlinearSolve
    solution = solve(problem, RobustMultiNewton())
    eq_vals = sort!(solution.u)

    # Return the values of phi
    return eq_vals
end

function fh(chi,N1,N2, u)
    f = ((1/N1).*u.*log.(u) + (1/N2).*(1 .- u).*log.(1 .- u) .+ chi.*u.*(1 .- u))
    return f
end

function interfacial_tension(chi,N1,N2,energy_method)
    #Solve equilibrium composition
    eq_vals = solve_binodal(chi, N1, N2)

    #Define kappa
    asym_factor = N2/N1
    if asym_factor < 0.1
        kappa = 1/3*chi
    else 
        kappa = 2/3*chi
    end

    # spline = spline_generator(chi,N1,N2,100)

    # f = u -> begin
    #     if energy_method == "analytical"
    #         fh(chi,N1,N2,u)
    #     else
    #         spline.(u)
    #     end
    # end
    f(u) = fh(chi,N1,N2,u)
    df(u) = f(u) -(u.*(f(eq_vals[2]) - f(eq_vals[1])) .+ (eq_vals[2]*f(eq_vals[1]) - eq_vals[1]*f(eq_vals[2])))./(eq_vals[2]-eq_vals[1])
    int_fun(u) = NaNMath.sqrt.(2 .* kappa .*df.(u))

    #Compute integral
    phi_vals = range(eq_vals[1],eq_vals[2],1000)
    int_vals = int_fun(phi_vals)
    val = trapezoidal_integration(phi_vals,int_vals)

    return val
end


"""
Functions for Simulation

"""
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

# function spline_generator(χ,N1,N2,knots)

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

# Function to compute interfacial tension
function compute_interfacial_tension(ϕ, xvals, κ)
    # Compute the derivative dϕ/dx using central differences
    dϕdx = similar(ϕ)
    dϕdx[1] = (ϕ[2] - ϕ[1]) / (xvals[2] - xvals[1])
    dϕdx[end] = (ϕ[end] - ϕ[end-1]) / (xvals[end] - xvals[end-1])
    # dϕdx[2:end-1] = (ϕ[3:end] - ϕ[1:end-2]) / (2 * dx)
    for i in 2:(length(xvals)-1)
        h1 = xvals[i] - xvals[i-1]
        h2 = xvals[i+1] - xvals[i]
        dϕdx[i] = (ϕ[i+1] - ϕ[i-1]) / (h1 + h2)
    end

    # Compute the integrand κ * (dϕ/dx)^2
    integrand = κ .* (dϕdx.^2)

    # Perform trapezoidal integration
    σ = trapezoidal_integration(xvals, integrand)

    return σ
end


function CH(ϕ, dx, params)
    χ, κ, N₁, N₂, energy_method = params
    spline = spline_generator(χ,N₁,N₂,200)

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

function solve_CH(chi,N1,N2,t_end,energy_method)
    #Set up grid
    L = 5.0
    nx = 201
    dx = L/(nx-1)
    xvals = LinRange(0,L,nx)

    function sigmoid(x, a, b)
        return 1 ./ (1 .+ exp.(-a .* (x - b)))
    end

    #Define initial conditions
    # Parameters for the sigmoid
    a = 5        # Controls the slope of the transition
    b = L / 2    # Midpoint of the transition

    # Define the initial condition across the domain
    ϕ₀ = 0.2 .+ 0.6 .* sigmoid.(xvals, a, b)

    #Pack Parameters
    kappa = (2/3)*chi
    params = (chi,kappa,N1,N2,energy_method)

    function ode_system!(du,u,p,t)
        du .= CH(u,dx,params)
    end

    # Set up the problem
    prob = ODEProblem(ode_system!, ϕ₀, (0.0, t_end))
    sol = solve(prob, TRBDF2(), reltol=1e-6, abstol=1e-8)
    ϕ_end=sol(t_end)

    σ_end = compute_interfacial_tension(ϕ_end, xvals, kappa)

    return σ_end
end
"""
Compute IFT using analytical formula
"""

chi_vals1 = range(2.5,50,length=100)
chi_vals2 = range(0.02,0.3,length=100)
ift_vals_analytical1 = zeros(length(chi_vals1))
ift_vals_analytical2 = zeros(length(chi_vals2))


for i in 1:length(chi_vals1)
    ift_vals_analytical1[i]= interfacial_tension(chi_vals1[i],1,1,"analytical")
    ift_vals_analytical2[i]= interfacial_tension(chi_vals2[i],100,50,"analytical")
end


"""
Compute IFT using Simulations
"""
chi_vals3 = range(3,50,length=10)
ift_vals_spline1 = zeros(length(chi_vals3))

for i in 1:length(chi_vals3)
    if chi_vals3[i] < 20
        t_end = 100
    elseif chi_vals3[i] < 30
        t_end = 10
    elseif chi_vals3[i] < 40
        t_end = 8
    else
        t_end = 5
    end
    println(t_end)
    ift_vals_spline1[i] = solve_CH(chi_vals3[i],1,1,t_end,"spline")
    println("beep")
end




"""
Plot
"""

chicrit(N1,N2) = sqrt((N1^2) + (N2^2))/(2*N1*N2)

p = plot(size=(600, 400),dpi=300,grid=false)
plot!(p,legend=:topleft,
    xlabel=L"\chi_{12}/\chi_{C}", ylabel=L"\textrm{Scaled \ Interfacial \ Tension} \ \tilde{\sigma}",
    tickfont=Plots.font("Computer Modern", 10),
    legendfont=Plots.font("Computer Modern",8)
    )

plot!(p,chi_vals1./2,ift_vals_analytical1,label=L"\textrm{Analytical (Full)}, x_{1} = x_{2} = 1",color="black")
plot!(p,chi_vals3./2,ift_vals_spline1,label=L"\textrm{Simulation (Spline)}, x_{1} = x_{2} = 1",seriestype=:scatter,markershape=:+,color="black",markersize=7)


plot!(p,chi_vals2./(chicrit(100,50)),ift_vals_analytical2,label=L"\textrm{Analytical (Full)}, x_{1} = 100, x_{2} = 50",color="blue")