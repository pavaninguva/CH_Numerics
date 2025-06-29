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
using WriteVTK
using Printf
using CSV
using DataFrames
using LaTeXStrings
using DifferentialEquations
using LinearSolve
using SparseConnectivityTracer
using ADTypes
Random.seed!(1234)

const datadir = joinpath(@__DIR__, "Binary_Case2_CSV")

"""
This script is used the generate the results for
case study 2 - Demixing of a symmetric binary mixture

The outputs are the VTK files for chi = 6 and the csv files to plot 
the energy and mass conservation of the system 
"""

"""
Functions
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


function flory_huggins(phi,chi, N1,N2)
    return (1/N1) * (phi .* log.(phi)) + (1/N2) * (1 .- phi) .* log.(1 .- phi) + chi .* phi .* (1 .- phi)
end

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

function impliciteuler_2d(chi, N1, N2, dx, L, dt, tf, energy_method,save_vtk=false)
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

    #Store ICs in VTK
    if save_vtk
        vtk_grid(@sprintf("snapshot_%04d", 0), x, y) do vtk
            vtk["u"]    = c
            vtk["time"] = fill(0.0, size(c))
        end
    end

    # Initialize arrays to store results
    c_avg = zeros(nt+1)
    energy = zeros(nt+1)

    c_avg[1] = mean(c)
    energy[1] = compute_energy(x,y,dx,c,chi,N1,N2,kappa)

    stride = 10

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

        # Solve the nonlinear system
        solver = solve(problem, NewtonRaphson(linsolve = KrylovJL_GMRES()), show_trace = Val(false))
        
        # Update c for the next time step
        c_new_vec = solver.u
        c = reshape(c_new_vec, nx, ny)

        # Compute statistics for plotting
        c_avg[n+1] = mean(c)
        energy[n+1] = compute_energy(x,y,dx,c,chi,N1,N2,kappa)

        if save_vtk && (n % stride == 0)
            vtk_grid(@sprintf("snapshot_%04d", n), x, y) do vtk
                vtk["u"]    = c
                vtk["time"] = fill(n*dt, size(c))
            end
        end

    end
    # Save the animation as a GIF
    time_vals = range(0,tf,nt+1)

    # Return the final concentration profile and computed data
    return c_avg, energy, time_vals
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

function mol_solver(chi, N1, N2, dx, L, tend, energy_method, save_vtk=false)
     #Simulation Parameters
     L = L
     tf = tend
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
    prob = ODEProblem(f,c0,(0.0,tf))
    sol = solve(prob, TRBDF2(),reltol=1e-6, abstol=1e-8,maxiters=1e7)

     # Set up the problem
    #  prob = ODEProblem(ode_system!, c0, (0.0, tf))
    #  sol = solve(prob, TRBDF2(linsolve=KrylovJL_GMRES()),reltol=1e-6, abstol=1e-8,maxiters=1e7)

     #Compute energy and mass conservation
    t_evals = range(0,tf, 1000)
    c_avg = zeros(length(t_evals))
    energy = zeros(length(t_evals))

    for(i,t) in enumerate(t_evals)
        sol_ = sol(t)
        c_avg[i] = mean(sol_)
        energy[i] = compute_energy(xvals,yvals,dx,sol_,chi,N1,N2,kappa)
    end

    #
    if save_vtk
        tvtk_vals = range(0,tf,201)
        for (i,t) in enumerate(tvtk_vals)
            c = sol(t)
            vtk_grid(@sprintf("snapshot_%04d", i), xvals, yvals) do vtk
                vtk["u"]    = c
                vtk["time"] = fill(t, size(c))
            end
        end
    end


    return c_avg, energy, t_evals
end



"""
Backwards Euler, Full
"""
# c_avg_be_full1, energy_be_full1, time_vals_be_full1 = impliciteuler_2d(6.0,1,1,0.4,20,0.1,50,"analytical",false)
# c_avg_be_full2, energy_be_full2, time_vals_be_full2 = impliciteuler_2d(6.0,1,1,0.2,20,0.05,50,"analytical",false)
# c_avg_be_full3, energy_be_full3, time_vals_be_full3 = impliciteuler_2d(6.0,1,1,0.1,20,0.025,50,"analytical",false)

# #Save c_avg and energy data as csv
# for (suffix, c_avg, energy, tvals) in (
#     ("dx_04_dt_01", c_avg_be_full1, energy_be_full1, time_vals_be_full1),
#     ("dx_02_dt_005", c_avg_be_full2, energy_be_full2, time_vals_be_full2),
#     ("dx_01_dt_0025", c_avg_be_full3, energy_be_full3, time_vals_be_full3),
# )
#     df = DataFrame(
#     time = tvals,
#     c_avg = c_avg,
#     energy = energy,
#     )
#     fname = @sprintf("backwardseuler2d_full_%s.csv", suffix)
#     CSV.write(fname, df)
#     println("Wrote $fname")
# end

const suffix_map = [
  ("dx_04_dt_01", "full1"),
  ("dx_02_dt_005", "full2"),
  ("dx_01_dt_0025", "full3"),
]

for (file_sfx, var_sfx) in suffix_map
    fname = joinpath(datadir, "backwardseuler2d_full_$(file_sfx).csv")
    println("Reading ", fname)
    df = CSV.read(fname, DataFrame)

    tvals = df.time
    cavg  = df.c_avg
    energ = df.energy

    t_sym = Symbol("time_vals_be_$(var_sfx)")
    c_sym = Symbol("c_avg_be_$(var_sfx)")
    e_sym = Symbol("energy_be_$(var_sfx)")

    @eval Main begin
      $(t_sym) = $tvals
      $(c_sym) = $cavg
      $(e_sym) = $energ
    end
end

p1 = plot(
    xlabel = L"t",
    ylabel = L"\bar{\phi}_{1}",
    title = "Backwards Euler, Full",
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

plot!(p1, time_vals_be_full1, c_avg_be_full1; color = :blue, seriestype=:samplemarkers,step=50, marker=:circle,markercolor=:blue, markersize=2, markerstrokewidth=0, markerstrokecolor=:blue, label = L"\Delta x = 0.4, \Delta t = 0.1")
plot!(p1, time_vals_be_full2, c_avg_be_full2; color = :blue, linestyle=:solid, label = L"\Delta x = 0.2, \Delta t = 0.05")
plot!(p1, time_vals_be_full3, c_avg_be_full3; color = :blue, linestyle=:dot, label = L"\Delta x = 0.1, \Delta t = 0.025")
p1_axis2 = twinx(p1)

plot!(
  p1_axis2,
  time_vals_be_full1,
  energy_be_full1;
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
plot!(p1_axis2,time_vals_be_full2,energy_be_full2;color=:red,linestyle=:solid,label="")
plot!(p1_axis2,time_vals_be_full3,energy_be_full3;color=:red,linestyle=:dot,label="")

"""
Backwards Euler Spline
"""
# c_avg_be_spline1, energy_be_spline1, time_vals_be_spline1 = impliciteuler_2d(6.0,1,1,0.4,20,0.1,50,"spline",false)
# c_avg_be_spline2, energy_be_spline2, time_vals_be_spline2 = impliciteuler_2d(6.0,1,1,0.2,20,0.05,50,"spline",false)
# c_avg_be_spline3, energy_be_spline3, time_vals_be_spline3 = impliciteuler_2d(6.0,1,1,0.1,20,0.025,50,"spline",true)
# #Save c_avg and energy data as csv
# for (suffix, c_avg, energy, tvals) in (
#     ("dx_04_dt_01", c_avg_be_spline1, energy_be_spline1, time_vals_be_spline1),
#     ("dx_02_dt_005", c_avg_be_spline2, energy_be_spline2, time_vals_be_spline2),
#     ("dx_01_dt_0025", c_avg_be_spline3, energy_be_spline3, time_vals_be_spline3)
# )
#     df = DataFrame(
#     time = tvals,
#     c_avg = c_avg,
#     energy = energy,
#     )
#     fname = @sprintf("backwardseuler2d_spline_%s.csv", suffix)
#     CSV.write(fname, df)
#     println("Wrote $fname")
# end

const suffix_map_spline = [
  ("dx_04_dt_01", "spline1"),
  ("dx_02_dt_005", "spline2"),
  ("dx_01_dt_0025", "spline3"),
]

for (file_sfx, var_sfx) in suffix_map_spline
    fname = joinpath(datadir, "backwardseuler2d_spline_$(file_sfx).csv")
    println("Reading ", fname)
    df = CSV.read(fname, DataFrame)

    tvals = df.time
    cavg  = df.c_avg
    energ = df.energy

    t_sym = Symbol("time_vals_be_$(var_sfx)")
    c_sym = Symbol("c_avg_be_$(var_sfx)")
    e_sym = Symbol("energy_be_$(var_sfx)")

    @eval Main begin
      $(t_sym) = $tvals
      $(c_sym) = $cavg
      $(e_sym) = $energ
    end
end

p2 = plot(
    xlabel = L"t",
    ylabel = L"\bar{\phi}_{1}",
    title = "Backwards Euler, Spline",
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

plot!(p2, time_vals_be_spline1, c_avg_be_spline1; color = :blue, seriestype=:samplemarkers,step=50, marker=:circle,markercolor=:blue, markersize=2, markerstrokewidth=0, markerstrokecolor=:blue, label = L"\Delta x = 0.4, \Delta t = 0.1")
plot!(p2, time_vals_be_spline2, c_avg_be_spline2; color = :blue, linestyle=:solid, label = L"\Delta x = 0.2, \Delta t = 0.05")
plot!(p2, time_vals_be_spline3, c_avg_be_spline3; color = :blue, linestyle=:dot, label = L"\Delta x = 0.1, \Delta t = 0.025")
p2_axis2 = twinx(p2)

plot!(
  p2_axis2,
  time_vals_be_spline1,
  energy_be_spline1;
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
plot!(p2_axis2,time_vals_be_spline2,energy_be_spline2;color=:red,linestyle=:solid,label="")
plot!(p2_axis2,time_vals_be_spline3,energy_be_spline3;color=:red,linestyle=:dot,label="")

# """
# TRBDF2 Analytical
# """

# c_avg_bdf_full1, energy_bdf_full1, time_vals_bdf_full1 = mol_solver(6,1,1,0.4,20,50,"analytical")
# c_avg_bdf_full2, energy_bdf_full2, time_vals_bdf_full2 = mol_solver(6,1,1,0.2,20,50,"analytical")
# c_avg_bdf_full3, energy_bdf_full3, time_vals_bdf_full3 = mol_solver(6,1,1,0.1,20,50,"analytical")

# for (suffix, c_avg, energy, tvals) in (
#     ("dx_04", c_avg_bdf_full1, energy_bdf_full1, time_vals_bdf_full1),
#     ("dx_02", c_avg_bdf_full2, energy_bdf_full2, time_vals_bdf_full2),
#     ("dx_01", c_avg_bdf_full3, energy_bdf_full3, time_vals_bdf_full3)
# )
#     df = DataFrame(
#     time = tvals,
#     c_avg = c_avg,
#     energy = energy,
#     )
#     fname = @sprintf("bdf2d_analytical_%s.csv", suffix)
#     CSV.write(fname, df)
#     println("Wrote $fname")
# end

const suffix_map_bdf_full = [
  ("dx_04", "full1"),
  ("dx_02", "full2"),
  ("dx_01", "full3"),
]

for (file_sfx, var_sfx) in suffix_map_bdf_full
    # fname = @sprintf("./Binary_Case2_CSV/bdf2d_analytical_%s.csv", file_sfx)
    fname = joinpath(datadir, "bdf2d_analytical_$(file_sfx).csv")
    println("Reading ", fname)
    df = CSV.read(fname, DataFrame)

    tvals = df.time
    cavg  = df.c_avg
    energ = df.energy

    t_sym = Symbol("time_vals_bdf_$(var_sfx)")
    c_sym = Symbol("c_avg_bdf_$(var_sfx)")
    e_sym = Symbol("energy_bdf_$(var_sfx)")

    @eval Main begin
      $(t_sym) = $tvals
      $(c_sym) = $cavg
      $(e_sym) = $energ
    end
end

p3 = plot(
    xlabel = L"t",
    ylabel = L"\bar{\phi}_{1}",
    title = "TRBDF2, Full",
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

plot!(p3, time_vals_bdf_full1, c_avg_bdf_full1; color = :blue, seriestype=:samplemarkers,step=50, marker=:circle,markercolor=:blue, markersize=2, markerstrokewidth=0, markerstrokecolor=:blue, label = L"\Delta x = 0.4")
plot!(p3, time_vals_bdf_full2, c_avg_bdf_full2; color = :blue, linestyle=:solid, label = L"\Delta x = 0.2")
plot!(p3, time_vals_bdf_full3, c_avg_bdf_full3; color = :blue, linestyle=:dot, label = L"\Delta x = 0.1")
p3_axis2 = twinx(p3)

plot!(
  p3_axis2,
  time_vals_bdf_full1,
  energy_bdf_full1;
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
plot!(p3_axis2,time_vals_bdf_full2,energy_bdf_full2;color=:red,linestyle=:solid,label="")
plot!(p3_axis2,time_vals_bdf_full3,energy_bdf_full3;color=:red,linestyle=:dot,label="")


# """
# TRBDF2 Spline
# """

# c_avg_bdf_spline1, energy_bdf_spline1, time_vals_bdf_spline1 = mol_solver(6,1,1,0.4,20,50,"spline")
# c_avg_bdf_spline2, energy_bdf_spline2, time_vals_bdf_spline2 = mol_solver(6,1,1,0.2,20,50,"spline")
# c_avg_bdf_spline3, energy_bdf_spline3, time_vals_bdf_spline3 = mol_solver(6,1,1,0.1,20,50,"spline",true)


# for (suffix, c_avg, energy, tvals) in (
#     ("dx_04", c_avg_bdf_spline1, energy_bdf_spline1, time_vals_bdf_spline1),
#     ("dx_02", c_avg_bdf_spline2, energy_bdf_spline2, time_vals_bdf_spline2),
#     ("dx_01", c_avg_bdf_spline3, energy_bdf_spline3, time_vals_bdf_spline3)
# )
#     df = DataFrame(
#     time = tvals,
#     c_avg = c_avg,
#     energy = energy,
#     )
#     fname = @sprintf("bdf2d_spline_%s.csv", suffix)
#     CSV.write(fname, df)
#     println("Wrote $fname")
# end



const suffix_map_bdf_spline = [
  ("dx_04", "spline1"),
  ("dx_02", "spline2"),
  ("dx_01", "spline3"),
]

for (file_sfx, var_sfx) in suffix_map_bdf_spline
    fname = joinpath(datadir, "bdf2d_spline_$(file_sfx).csv")
    println("Reading ", fname)
    df = CSV.read(fname, DataFrame)

    tvals = df.time
    cavg  = df.c_avg
    energ = df.energy

    t_sym = Symbol("time_vals_bdf_$(var_sfx)")
    c_sym = Symbol("c_avg_bdf_$(var_sfx)")
    e_sym = Symbol("energy_bdf_$(var_sfx)")

    @eval Main begin
      $(t_sym) = $tvals
      $(c_sym) = $cavg
      $(e_sym) = $energ
    end
end


p4 = plot(
    xlabel = L"t",
    ylabel = L"\bar{\phi}_{1}",
    title = "TRBDF2, Spline",
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

plot!(p4, time_vals_bdf_spline1, c_avg_bdf_spline1; color = :blue, seriestype=:samplemarkers,step=50, marker=:circle,markercolor=:blue, markersize=2, markerstrokewidth=0, markerstrokecolor=:blue, label = L"\Delta x = 0.4")
plot!(p4, time_vals_bdf_spline2, c_avg_bdf_spline2; color = :blue, linestyle=:solid, label = L"\Delta x = 0.2")
plot!(p4, time_vals_bdf_spline3, c_avg_bdf_spline3; color = :blue, linestyle=:dot, label = L"\Delta x = 0.1")
p4_axis2 = twinx(p4)

plot!(
  p4_axis2,
  time_vals_bdf_spline1,
  energy_bdf_spline1;
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
plot!(p4_axis2,time_vals_bdf_spline2,energy_bdf_spline2;color=:red,linestyle=:solid,label="")
plot!(p4_axis2,time_vals_bdf_spline3,energy_bdf_spline3;color=:red,linestyle=:dot,label="")

"""
Combined Plot
"""

p_all = plot(p1,p2,p3,p4, layout=(2,2),size=(800,700),dpi=300,
                bottom_margin = 3Plots.mm, left_margin = 3Plots.mm, right_margin=3Plots.mm)
savefig("2d_benchmarking_case2.png")

display(p_all)