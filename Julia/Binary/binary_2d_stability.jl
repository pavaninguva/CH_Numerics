using SparseArrays
using LinearAlgebra
using NonlinearSolve
using Plots
using Krylov
using Trapz
using IterativeSolvers
using Random
using Statistics
using Sundials
include("../pchip.jl")
using NaNMath
using DataFrames
using LaTeXStrings
using DifferentialEquations
using CSV
using Plots.PlotMeasures
using LinearSolve
using SparseConnectivityTracer
using ADTypes

const datadir = joinpath(@__DIR__, "2d_stability_data")


"""
Functions
"""

function dfdphi_ana(phi, chi, N1, N2)
    return (1/N1)*log.(phi) - (1/N2)*log.(1 .- phi) + chi * (1 .- 2 .* phi)
end

function compute_energy(x,y,dx,c,chi,N1,N2,kappa)
    nx = length(x)
    ny = length(y)

    grad_cx = zeros(nx, ny)
    grad_cy = zeros(nx, ny)

    # Interior points
    for i in 2:nx-1
        for j in 2:ny-1
            grad_cx[i,j] = (c[i+1,j] - c[i-1,j]) / (2 * dx)
            grad_cy[i,j] = (c[i,j+1] - c[i,j-1]) / (2 * dx)
        end
    end

    # Boundaries (forward/backward differences)
    for j in 1:ny
        grad_cx[1,j] = (c[2,j] - c[1,j]) / dx  # Forward difference
        grad_cx[nx,j] = (c[nx,j] - c[nx-1,j]) / dx  # Backward difference
    end

    for i in 1:nx
        grad_cy[i,1] = (c[i,2] - c[i,1]) / dx  # Forward difference
        grad_cy[i,ny] = (c[i,ny] - c[i,ny-1]) / dx  # Backward difference
    end

    energy_density = flory_huggins(c, chi, N1, N2) .+ (kappa / 2) * (grad_cx.^2 .+ grad_cy.^2)

    return trapz((x,y),energy_density)
end

function spline_generator(chi, N1, N2, knots)

    function tanh_sinh_spacing(n, β)
        points = 0.5 * (1 .+ tanh.(β .* (2 * collect(0:n-1) / (n-1) .- 1)))
        return points
    end
    
    phi_vals_ = collect(tanh_sinh_spacing(knots-4,14))

    pushfirst!(phi_vals_,1e-16)
    push!(phi_vals_,1-1e-16)

    f_vals_ = dfdphi_ana(phi_vals_,chi,N1,N2)

    phi_vals = pushfirst!(phi_vals_,0)
    push!(phi_vals,1)

    #Compute value at eps
    eps_val = BigFloat("1e-40")
    one_big = BigFloat(1)

    f_eps = dfdphi_ana(eps_val,BigFloat(chi),BigFloat(N1), BigFloat(N2))
    f_eps1 = dfdphi_ana(one_big-eps_val, BigFloat(chi),BigFloat(N1), BigFloat(N2))

    f_eps_float = Float64(f_eps)
    f_eps1_float = Float64(f_eps1)

    f_vals = pushfirst!(f_vals_,f_eps_float)
    push!(f_vals, f_eps1_float)

    # Build and return the spline function using pchip
    spline = pchip(phi_vals, f_vals)
    return spline
end

function tend_generator(chi)
    if chi < 5
        tend = 100
    elseif 5 <= chi < 10
        tend = 30
    elseif 10 <= chi < 15
        tend = 15
    elseif 15 <= chi < 20
        tend = 10
    elseif 20 <= chi < 25
        tend = 8
    elseif 25 <= chi < 30
        tend = 5
    elseif chi >= 30
        tend = 4
    end
    return tend
end

"""
Backwards Euler Function
"""


# Residual function with Neumann boundary conditions
function residual!(F, c_new, p)
    c_old = p.c_old
    dt = p.dt
    dx = p.dx
    dy = dx  # Assuming square grid
    kappa = p.kappa
    chi = p.chi
    N1 = p.N1
    N2 = p.N2
    nx, ny = p.nx, p.ny
    energy_method= p.energy_method

    spline = spline_generator(chi, N1, N2,100)
    if energy_method == "analytical"
        dfdphi = phi -> dfdphi_ana(phi,chi,N1, N2)
    else
        dfdphi = phi -> spline.(phi)
    end

    # dfdphi = phi -> dfdphi_ana(phi,chi,N1,N2)

    # Compute mu_new
    mu_new = similar(c_new)

    function M_func(phi)
        return phi .* (1 .- phi)
    end
    
    function M_func_half(phi1, phi2)
        return M_func(0.5 .* (phi1 .+ phi2))
    end

    # Compute mu_new for all nodes
    for i in 1:nx
        for j in 1:ny
                if i == 1 && j > 1 && j < ny
                    laplacian_c = ((2.0 * (c_new[2,j] - c_new[1,j])) / dx^2) + (c_new[1,j+1] - 2.0 * c_new[1,j] + c_new[1,j-1]) / dy^2
                elseif i == nx && j > 1 && j < ny
                    laplacian_c = ((2.0 * (c_new[nx-1,j] - c_new[nx,j])) / dx^2) + (c_new[nx,j+1] - 2.0 * c_new[nx,j] + c_new[nx,j-1]) / dy^2
                elseif j == 1 && i > 1 && i < nx
                    laplacian_c = ((c_new[i+1,1] - 2.0 * c_new[i,1] + c_new[i-1,1]) / dx^2) + (2.0 * (c_new[i,2] - c_new[i,1])) / dy^2
                elseif j == ny && i > 1 && i < nx
                    laplacian_c = ((c_new[i+1,ny] - 2.0 * c_new[i,ny] + c_new[i-1,ny]) / dx^2) + (2.0 * (c_new[i,ny-1] - c_new[i,ny])) / dy^2
                elseif i == 1 && j == 1
                    laplacian_c = ((2.0 * (c_new[2,1] - c_new[1,1])) / dx^2) + (2.0 * (c_new[1,2] - c_new[1,1])) / dy^2
                elseif i == nx && j == 1
                    laplacian_c = ((2.0 * (c_new[nx-1,1] - c_new[nx,1])) / dx^2) + (2.0 * (c_new[nx,2] - c_new[nx,1])) / dy^2
                elseif i == 1 && j == ny
                    laplacian_c = ((2.0 * (c_new[2,ny] - c_new[1,ny])) / dx^2) + (2.0 * (c_new[1,ny-1] - c_new[1,ny])) / dy^2
                elseif i == nx && j == ny
                    laplacian_c = ((2.0 * (c_new[nx-1,ny] - c_new[nx,ny])) / dx^2) + (2.0 * (c_new[nx,ny-1] - c_new[nx,ny])) / dy^2
                else
                    # Interior nodes
                    laplacian_c = (c_new[i+1,j] - 2.0 * c_new[i,j] + c_new[i-1,j]) / dx^2 + (c_new[i,j+1] - 2.0 * c_new[i,j] + c_new[i,j-1]) / dy^2
                end
                mu_new[i,j] = dfdphi(c_new[i,j]) - kappa*laplacian_c
            end
        end

    # Compute residuals F
    for i in 1:nx
        for j in 1:ny
            if i == 1 && j > 1 && j < ny
                M_iphalf = M_func_half(c_new[1,j], c_new[2,j])
                M_jphalf = M_func_half(c_new[1,j], c_new[1,j+1])
                M_jmhalf = M_func_half(c_new[1,j], c_new[1,j-1])

                Jx_iphalf = 2.0 * M_iphalf * (mu_new[2,j] - mu_new[1,j]) / dx^2
                Jy_jphalf = M_jphalf * (mu_new[1,j+1] - mu_new[1,j]) / dy^2
                Jy_jmhalf = M_jmhalf * (mu_new[1,j] - mu_new[1,j-1]) / dy^2

                div_J = Jx_iphalf + (Jy_jphalf - Jy_jmhalf)

                F[1,j] = (c_new[1,j] - c_old[1,j]) / dt - div_J
            elseif i == nx && j > 1 && j < ny
                M_imhalf = M_func_half(c_new[nx,j], c_new[nx-1,j])
                M_jphalf = M_func_half(c_new[nx,j], c_new[nx,j+1])
                M_jmhalf = M_func_half(c_new[nx,j], c_new[nx,j-1])

                Jx_imhalf = 2.0 * M_imhalf * (mu_new[nx-1,j] - mu_new[nx,j]) / dx^2
                Jy_jphalf = M_jphalf * (mu_new[nx,j+1] - mu_new[nx,j]) / dy^2
                Jy_jmhalf = M_jmhalf * (mu_new[nx,j] - mu_new[nx,j-1]) / dy^2

                div_J = Jx_imhalf + (Jy_jphalf - Jy_jmhalf)

                F[nx,j] = (c_new[nx,j] - c_old[nx,j]) / dt - div_J
            elseif j == 1 && i > 1 && i < nx
                M_iphalf = M_func_half(c_new[i,1], c_new[i+1,1])
                M_imhalf = M_func_half(c_new[i,1], c_new[i-1,1])
                M_jphalf = M_func_half(c_new[i,1], c_new[i,2])

                Jx_iphalf = M_iphalf * (mu_new[i+1,1] - mu_new[i,1]) / dx^2
                Jx_imhalf = M_imhalf * (mu_new[i,1] - mu_new[i-1,1]) / dx^2
                Jy_jphalf = 2.0 * M_jphalf * (mu_new[i,2] - mu_new[i,1]) / dy^2

                div_J = (Jx_iphalf - Jx_imhalf) + Jy_jphalf

                F[i,1] = (c_new[i,1] - c_old[i,1]) / dt - div_J
            elseif j == ny && i > 1 && i < nx
                M_iphalf = M_func_half(c_new[i,ny], c_new[i+1,ny])
                M_imhalf = M_func_half(c_new[i,ny], c_new[i-1,ny])
                M_jmhalf = M_func_half(c_new[i,ny], c_new[i,ny-1])

                Jx_iphalf = M_iphalf * (mu_new[i+1,ny] - mu_new[i,ny]) / dx^2
                Jx_imhalf = M_imhalf * (mu_new[i,ny] - mu_new[i-1,ny]) / dx^2
                Jy_jmhalf = 2.0 * M_jmhalf * (mu_new[i,ny-1] - mu_new[i,ny]) / dy^2

                div_J = (Jx_iphalf - Jx_imhalf) + Jy_jmhalf

                F[i,ny] = (c_new[i,ny] - c_old[i,ny]) / dt - div_J
            elseif i == 1 && j == 1
                M_iphalf = M_func_half(c_new[1,1], c_new[2,1])
                M_jphalf = M_func_half(c_new[1,1], c_new[1,2])

                Jx_iphalf = 2.0 * M_iphalf * (mu_new[2,1] - mu_new[1,1]) / dx^2
                Jy_jphalf = 2.0 * M_jphalf * (mu_new[1,2] - mu_new[1,1]) / dy^2

                div_J = Jx_iphalf + Jy_jphalf

                F[1,1] = (c_new[1,1] - c_old[1,1]) / dt - div_J
            elseif i == nx && j == 1
                M_imhalf = M_func_half(c_new[nx,1], c_new[nx-1,1])
                M_jphalf = M_func_half(c_new[nx,1], c_new[nx,2])

                Jx_imhalf = 2.0 * M_imhalf * (mu_new[nx-1,1] - mu_new[nx,1]) / dx^2
                Jy_jphalf = 2.0 * M_jphalf * (mu_new[nx,2] - mu_new[nx,1]) / dy^2

                div_J = Jx_imhalf + Jy_jphalf

                F[nx,1] = (c_new[nx,1] - c_old[nx,1]) / dt - div_J
            elseif i == 1 && j == ny
                M_iphalf = M_func_half(c_new[1,ny], c_new[2,ny])
                M_jmhalf = M_func_half(c_new[1,ny], c_new[1,ny-1])

                Jx_iphalf = 2.0 * M_iphalf * (mu_new[2,ny] - mu_new[1,ny]) / dx^2
                Jy_jmhalf = 2.0 * M_jmhalf * (mu_new[1,ny-1] - mu_new[1,ny]) / dy^2

                div_J = Jx_iphalf + Jy_jmhalf

                F[1,ny] = (c_new[1,ny] - c_old[1,ny]) / dt - div_J
            elseif i == nx && j == ny
                M_imhalf = M_func_half(c_new[nx,ny], c_new[nx-1,ny])
                M_jmhalf = M_func_half(c_new[nx,ny], c_new[nx,ny-1])

                Jx_imhalf = 2.0 * M_imhalf * (mu_new[nx-1,ny] - mu_new[nx,ny]) / dx^2
                Jy_jmhalf = 2.0 * M_jmhalf * (mu_new[nx,ny-1] - mu_new[nx,ny]) / dy^2

                div_J = Jx_imhalf + Jy_jmhalf

                F[nx,ny] = (c_new[nx,ny] - c_old[nx,ny]) / dt - div_J

            else
                # Interior nodes
                M_iphalf = M_func_half(c_new[i,j], c_new[i+1,j])
                M_imhalf = M_func_half(c_new[i,j], c_new[i-1,j])
                M_jphalf = M_func_half(c_new[i,j], c_new[i,j+1])
                M_jmhalf = M_func_half(c_new[i,j], c_new[i,j-1])

                Jx_iphalf = M_iphalf * (mu_new[i+1,j] - mu_new[i,j]) / dx^2
                Jx_imhalf = M_imhalf * (mu_new[i,j] - mu_new[i-1,j]) / dx^2
                Jy_jphalf = M_jphalf * (mu_new[i,j+1] - mu_new[i,j]) / dy^2
                Jy_jmhalf = M_jmhalf * (mu_new[i,j] - mu_new[i,j-1]) / dy^2

                div_J = (Jx_iphalf - Jx_imhalf) + (Jy_jphalf - Jy_jmhalf)

                F[i,j] = (c_new[i,j] - c_old[i,j]) / dt - div_J
            end
        end
    end
end

function impliciteuler_2d(chi, N1, N2, dx, L, dt, tf, energy_method)
    L = L
    tf = tf
    nx = Int(L / dx) + 1
    ny = nx  # Assuming square domain
    x = range(0, L, length = nx)
    y = range(0, L, length = ny)
    nt = Int(tf / dt)
    if N1/N2 < 10
        kappa = (1 / 3) * chi
    else
        kappa = (2/3)*chi
    end

    # Initial condition: small random perturbation around c0
    c0 = 0.5
    c = c0 .+ 0.02 * (rand(nx, ny) .- 0.5)

    for n = 1:nt
        println("Time step: $n, Time: $(n*dt)")

        # Save the old concentration profile
        c_old = copy(c)

        # Parameters to pass to the residual function
        p = (c_old = c_old, dt = dt, dx = dx, kappa = kappa, chi = chi, nx = nx, ny = ny, N1 = N1, N2 = N2, energy_method=energy_method)

        # # Initial guess for c_new
        # c_guess = copy(c_old)

        # Create the NonlinearProblem
        problem = NonlinearProblem(residual!, c_old, p)

        term_cond = AbsNormSafeTerminationMode(
            NonlinearSolve.L2_NORM; protective_threshold = nothing,
            patience_steps = 100, patience_objective_multiplier = 3,
            min_max_factor = 1.3,
            )

        # Solve the nonlinear system
        solver = solve(problem, NewtonRaphson(linsolve = KrylovJL_GMRES()), show_trace = Val(false),termination_condition=term_cond,abstol=1e-8)
        
        # Update c for the next time step
        c_new_vec = solver.u
        c = reshape(c_new_vec, nx, ny)

        abstol = 1e-7
        if norm(solver.resid,Inf) > abstol || any(x -> x>1 || x < 0, c)
            error("Sovler did not converge")
        end

        
    end
end

"""
TRBDF2 Function
"""

function CH_mol2d(phi, params)
    chi, kappa, N1, N2, dx, dy, nx, ny, energy_method = params

    spline = spline_generator(chi, N1, N2,100)
    if energy_method == "analytical"
        dfdphi = phi -> dfdphi_ana(phi,chi,N1, N2)
    else
        dfdphi = phi -> spline.(phi)
    end

    #Define mobility
    function M_func(phi)
        return phi .* (1 .- phi)
    end
    
    function M_func_half(phi1, phi2)
        return M_func(0.5 .* (phi1 .+ phi2))
    end

    #Define chemical potential
    # Compute mu_new
    mu_new = similar(phi)

    # Compute mu_new for all nodes
    for i in 1:nx
        for j in 1:ny
                if i == 1 && j > 1 && j < ny
                    laplacian_c = ((2.0 * (phi[2,j] - phi[1,j])) / dx^2) + (phi[1,j+1] - 2.0 * phi[1,j] + phi[1,j-1]) / dy^2
                elseif i == nx && j > 1 && j < ny
                    laplacian_c = ((2.0 * (phi[nx-1,j] - phi[nx,j])) / dx^2) + (phi[nx,j+1] - 2.0 * phi[nx,j] + phi[nx,j-1]) / dy^2
                elseif j == 1 && i > 1 && i < nx
                    laplacian_c = ((phi[i+1,1] - 2.0 * phi[i,1] + phi[i-1,1]) / dx^2) + (2.0 * (phi[i,2] - phi[i,1])) / dy^2
                elseif j == ny && i > 1 && i < nx
                    laplacian_c = ((phi[i+1,ny] - 2.0 * phi[i,ny] + phi[i-1,ny]) / dx^2) + (2.0 * (phi[i,ny-1] - phi[i,ny])) / dy^2
                elseif i == 1 && j == 1
                    laplacian_c = ((2.0 * (phi[2,1] - phi[1,1])) / dx^2) + (2.0 * (phi[1,2] - phi[1,1])) / dy^2
                elseif i == nx && j == 1
                    laplacian_c = ((2.0 * (phi[nx-1,1] - phi[nx,1])) / dx^2) + (2.0 * (phi[nx,2] - phi[nx,1])) / dy^2
                elseif i == 1 && j == ny
                    laplacian_c = ((2.0 * (phi[2,ny] - phi[1,ny])) / dx^2) + (2.0 * (phi[1,ny-1] - phi[1,ny])) / dy^2
                elseif i == nx && j == ny
                    laplacian_c = ((2.0 * (phi[nx-1,ny] - phi[nx,ny])) / dx^2) + (2.0 * (phi[nx,ny-1] - phi[nx,ny])) / dy^2
                else
                    # Interior nodes
                    laplacian_c = (phi[i+1,j] - 2.0 * phi[i,j] + phi[i-1,j]) / dx^2 + (phi[i,j+1] - 2.0 * phi[i,j] + phi[i,j-1]) / dy^2
                end
                mu_new[i,j] = dfdphi(phi[i,j]) - kappa*laplacian_c
        end
    end


    F = similar(phi)
    #Define LHS
    for i in 1:nx
        for j in 1:ny
            if i == 1 && j > 1 && j < ny
                M_iphalf = M_func_half(phi[1,j], phi[2,j])
                M_jphalf = M_func_half(phi[1,j], phi[1,j+1])
                M_jmhalf = M_func_half(phi[1,j], phi[1,j-1])

                Jx_iphalf = 2.0 * M_iphalf * (mu_new[2,j] - mu_new[1,j]) / dx^2
                Jy_jphalf = M_jphalf * (mu_new[1,j+1] - mu_new[1,j]) / dy^2
                Jy_jmhalf = M_jmhalf * (mu_new[1,j] - mu_new[1,j-1]) / dy^2

                div_J = Jx_iphalf + (Jy_jphalf - Jy_jmhalf)

                F[1,j] = div_J
            elseif i == nx && j > 1 && j < ny
                M_imhalf = M_func_half(phi[nx,j], phi[nx-1,j])
                M_jphalf = M_func_half(phi[nx,j], phi[nx,j+1])
                M_jmhalf = M_func_half(phi[nx,j], phi[nx,j-1])

                Jx_imhalf = 2.0 * M_imhalf * (mu_new[nx-1,j] - mu_new[nx,j]) / dx^2
                Jy_jphalf = M_jphalf * (mu_new[nx,j+1] - mu_new[nx,j]) / dy^2
                Jy_jmhalf = M_jmhalf * (mu_new[nx,j] - mu_new[nx,j-1]) / dy^2

                div_J = Jx_imhalf + (Jy_jphalf - Jy_jmhalf)

                F[nx,j] =  div_J
            elseif j == 1 && i > 1 && i < nx
                M_iphalf = M_func_half(phi[i,1], phi[i+1,1])
                M_imhalf = M_func_half(phi[i,1], phi[i-1,1])
                M_jphalf = M_func_half(phi[i,1], phi[i,2])

                Jx_iphalf = M_iphalf * (mu_new[i+1,1] - mu_new[i,1]) / dx^2
                Jx_imhalf = M_imhalf * (mu_new[i,1] - mu_new[i-1,1]) / dx^2
                Jy_jphalf = 2.0 * M_jphalf * (mu_new[i,2] - mu_new[i,1]) / dy^2

                div_J = (Jx_iphalf - Jx_imhalf) + Jy_jphalf

                F[i,1] =  div_J
            elseif j == ny && i > 1 && i < nx
                M_iphalf = M_func_half(phi[i,ny], phi[i+1,ny])
                M_imhalf = M_func_half(phi[i,ny], phi[i-1,ny])
                M_jmhalf = M_func_half(phi[i,ny], phi[i,ny-1])

                Jx_iphalf = M_iphalf * (mu_new[i+1,ny] - mu_new[i,ny]) / dx^2
                Jx_imhalf = M_imhalf * (mu_new[i,ny] - mu_new[i-1,ny]) / dx^2
                Jy_jmhalf = 2.0 * M_jmhalf * (mu_new[i,ny-1] - mu_new[i,ny]) / dy^2

                div_J = (Jx_iphalf - Jx_imhalf) + Jy_jmhalf

                F[i,ny] = div_J
            elseif i == 1 && j == 1
                M_iphalf = M_func_half(phi[1,1], phi[2,1])
                M_jphalf = M_func_half(phi[1,1], phi[1,2])

                Jx_iphalf = 2.0 * M_iphalf * (mu_new[2,1] - mu_new[1,1]) / dx^2
                Jy_jphalf = 2.0 * M_jphalf * (mu_new[1,2] - mu_new[1,1]) / dy^2

                div_J = Jx_iphalf + Jy_jphalf

                F[1,1] = div_J
            elseif i == nx && j == 1
                M_imhalf = M_func_half(phi[nx,1], phi[nx-1,1])
                M_jphalf = M_func_half(phi[nx,1], phi[nx,2])

                Jx_imhalf = 2.0 * M_imhalf * (mu_new[nx-1,1] - mu_new[nx,1]) / dx^2
                Jy_jphalf = 2.0 * M_jphalf * (mu_new[nx,2] - mu_new[nx,1]) / dy^2

                div_J = Jx_imhalf + Jy_jphalf

                F[nx,1] =  div_J
            elseif i == 1 && j == ny
                M_iphalf = M_func_half(phi[1,ny], phi[2,ny])
                M_jmhalf = M_func_half(phi[1,ny], phi[1,ny-1])

                Jx_iphalf = 2.0 * M_iphalf * (mu_new[2,ny] - mu_new[1,ny]) / dx^2
                Jy_jmhalf = 2.0 * M_jmhalf * (mu_new[1,ny-1] - mu_new[1,ny]) / dy^2

                div_J = Jx_iphalf + Jy_jmhalf

                F[1,ny] = div_J
            elseif i == nx && j == ny
                M_imhalf = M_func_half(phi[nx,ny], phi[nx-1,ny])
                M_jmhalf = M_func_half(phi[nx,ny], phi[nx,ny-1])

                Jx_imhalf = 2.0 * M_imhalf * (mu_new[nx-1,ny] - mu_new[nx,ny]) / dx^2
                Jy_jmhalf = 2.0 * M_jmhalf * (mu_new[nx,ny-1] - mu_new[nx,ny]) / dy^2

                div_J = Jx_imhalf + Jy_jmhalf

                F[nx,ny] =  div_J

            else
                # Interior nodes
                M_iphalf = M_func_half(phi[i,j], phi[i+1,j])
                M_imhalf = M_func_half(phi[i,j], phi[i-1,j])
                M_jphalf = M_func_half(phi[i,j], phi[i,j+1])
                M_jmhalf = M_func_half(phi[i,j], phi[i,j-1])

                Jx_iphalf = M_iphalf * (mu_new[i+1,j] - mu_new[i,j]) / dx^2
                Jx_imhalf = M_imhalf * (mu_new[i,j] - mu_new[i-1,j]) / dx^2
                Jy_jphalf = M_jphalf * (mu_new[i,j+1] - mu_new[i,j]) / dy^2
                Jy_jmhalf = M_jmhalf * (mu_new[i,j] - mu_new[i,j-1]) / dy^2

                div_J = (Jx_iphalf - Jx_imhalf) + (Jy_jphalf - Jy_jmhalf)

                F[i,j] =  div_J
            end
        end
    end
    return F
end

function mol_solver(chi, N1, N2, dx, L, energy_method)
     #Simulation Parameters
     L = L
     tf = tend_generator(chi)
     nx = Int(L / dx) + 1
     ny = nx
     xvals = range(0, L, length = nx)
     yvals = xvals
     dy = dx
    if (N1/N2) < 10
        kappa = (2 / 3) * chi
    else
        kappa = (1/3)*chi
    end

    # Initial condition: small random perturbation around c0
    c0_ = 0.5
    c0 = c0_ .+ 0.02 * (rand(nx,ny) .- 0.5)

    #Set up MOL bits
    params = (chi, kappa, N1, N2, dx, dy, nx, ny, energy_method)

    function ode_system!(du, u, p, t)
        du .= CH_mol2d(u,params)
        println(t)
    end

    params_jac = (chi,kappa,N1,N2,dx,dy,nx,ny,"analytical")
    function ode_system_jac!(du,u,p,t)
        du .= CH_mol2d(u,params_jac)
    end

    #Set up sparse bits
    detector = TracerSparsityDetector()
    du0 = copy(c0)
    jac_sparsity = ADTypes.jacobian_sparsity((du,u) -> ode_system_jac!(du,u,params,0.0), du0,c0,detector)
    
    f = ODEFunction(ode_system!; jac_prototype=float.(jac_sparsity))

    #Set up stagnation bits
    DT_THRESH = 1e-8
    MAX_STEPS = 1000
    
    # stagnation_condition(u, t, integrator) = true

    # function stagnation_affect!(integrator)
    #     if integrator.dt < DT_THRESH
    #         integrator.p[:small_dt_counter] += 1      # increment
    #     else
    #         integrator.p[:small_dt_counter] = 0       # reset
    #     end

    #     if integrator.p[:small_dt_counter] ≥ MAX_STEPS
    #         error("Integrator stagnated: dt < $DT_THRESH for $MAX_STEPS consecutive steps")
    #     end
    # end

    # cb = DiscreteCallback(stagnation_condition, stagnation_affect!)

    # function stagnation_condition(u, t, integrator)
    #     dt_small   = integrator.dt < DT_THRESH
    #     new_time   = t > integrator.p[:last_event_t]   # prevents zero-length loop
    #     return dt_small && new_time
    # end
    # function stagnation_affect!(integrator)
    #     if integrator.dt < DT_THRESH
    #         integrator.p[:small_dt_counter] += 1
    #     else
    #         integrator.p[:small_dt_counter] = 0     
    #     end
    #     if integrator.p[:small_dt_counter] ≥ MAX_STEPS
    #         error("Integrator stagnated: dt < $DT_THRESH for $MAX_STEPS consecutive steps")
    #     end
    # end
    # cb = DiscreteCallback(stagnation_condition, stagnation_affect!; save_positions=(false,false))

    # p0   = Dict(:small_dt_counter => 0,
    #         :last_event_t      => -Inf)
    prob = ODEProblem(f,c0,(0.0,tf))
    sol = solve(prob, TRBDF2(),reltol=1e-8, abstol=1e-8,maxiters=1e7)

end


"""
Testing functions
"""
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

function param_sweep_min_dt(chi_values, dx_values; N1=1.0, N2=1.0, L = 20.0, energy_method="analytical",results_file)
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
            else
                println("Running simulation for chi=$chi, dx=$dx, energy_method=$energy_method, timestepping=BDF")
                sol = nothing
                try
                    sol = mol_solver(chi, N1, N2, dx, L,energy_method)
                    # Check solver return code for divergence
                    if sol.retcode == :Diverged || sol.retcode == :Failure
                        @warn "Solver diverged or failed for chi=$chi, dx=$dx"
                        min_dt_matrix[i,j] = NaN
                    else
                        t_values = sol.t
                        if length(t_values) > 1
                            min_dt_matrix[i, j] = minimum(diff(t_values))
                        else
                            # Solver didn't advance
                            @warn "Solver did not advance for chi=$chi, dx=$dx"
                            min_dt_matrix[i, j] = NaN
                        end
                    end
                catch e
                    @warn "Solver failed for chi=$chi, dx=$dx with error $e"
                    min_dt_matrix[i, j] = NaN
                end

                # Save the result to the new DataFrame
                push!(new_results, (chi, dx, min_dt_matrix[i,j]))
                # results[key] = min_dt_matrix[i,j]
            end
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

chi_values = 4:1:16
dx_values = [0.1,0.16,0.2,0.25,0.4,0.5]



min_dt_matrix_spline = param_sweep_min_dt(chi_values, dx_values; N1=1.0, N2=1.0, energy_method="spline",results_file= joinpath(datadir,"2d_dt_bdf_spline.csv"))
min_dt_matrix_analytical = param_sweep_min_dt(chi_values,dx_values,N1=1.0,N2=1.0,energy_method="analytical",results_file=joinpath(datadir,"2d_dt_bdf_ana.csv"))

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


p_all = plot(p1,p2, layout=2, size=(1400,700), dpi=300, leftmargin=3mm, righhmargin=3mm,bottommargin=3mm)