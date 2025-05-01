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
include("./pchip.jl")
using WriteVTK
using Printf

Random.seed!(1234)

"""
Functions
"""
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
Backwards Euler, Full
"""
c_avg_be_full1, energy_be_full1, time_vals_be_full1 = impliciteuler_2d(6.0,1,1,0.4,20,0.1,50,"analytical",false)
c_avg_be_full2, energy_be_full2, time_vals_be_full2 = impliciteuler_2d(6.0,1,1,0.2,20,0.05,50,"analytical",false)
c_avg_be_full3, energy_be_full3, time_vals_be_full3 = impliciteuler_2d(6.0,1,1,0.1,20,0.025,50,"analytical",true)



p1 = plot(
    xlabel = L"t",
    ylabel = L"\bar{\phi}_{1}",
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

display(p1)