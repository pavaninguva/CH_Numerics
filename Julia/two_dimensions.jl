using SparseArrays
using LinearAlgebra
using NonlinearSolve
using Plots
using Krylov
using Trapz
using IterativeSolvers
using Random

Random.seed!(1234)

function impliciteuler_2d(chi, N1, N2, dx, dt)
    L = 4.0
    tf = 5.0
    nx = Int(L / dx) + 1
    ny = nx  # Assuming square domain
    x = range(0, L, length = nx)
    y = range(0, L, length = ny)
    nt = Int(tf / dt)
    kappa = (2 / 3) * chi  # Gradient energy term

    # Initial condition: small random perturbation around c0
    c0 = 0.5
    c = c0 .+ 0.02 * (rand(nx, ny) .- 0.5)

    # Initialize arrays to store results
    c_max = zeros(nt)
    c_min = zeros(nt)
    c_avg = zeros(nt)
    energy = zeros(nt)

    # Functions
    function flory_huggins(phi,chi, N1,N2)
    return (1/N1) * (phi .* log.(phi)) + (1/N2) * (1 .- phi) .* log.(1 .- phi) + chi .* phi .* (1 .- phi)
    end

    function dfdphi(phi, chi, N1, N2)
        return (1/N1)*log.(phi) - (1/N2)*log.(1 .- phi) + chi * (1 .- 2 .* phi)
    end

    function M_func(phi)
        return phi .* (1 .- phi)
    end

    function M_func_half(phi1, phi2)
        return M_func(0.5 .* (phi1 .+ phi2))
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

        # Ensure c_new stays within (0,1)
        c_work = c_new

        # Compute mu_new
        mu_new = similar(c_new)

        for i in 2:nx-1
            for j in 2:ny-1
                laplacian_c = (c_work[i+1,j] - 2.0 * c_work[i,j] + c_work[i-1,j]) / dx^2 +
                              (c_work[i,j+1] - 2.0 * c_work[i,j] + c_work[i,j-1]) / dy^2
                mu_new[i,j] = dfdphi(c_work[i,j], chi, N1, N2) - kappa * laplacian_c
            end
        end

        # Neumann Boundary Conditions
        # Left boundary (i=1)
        for j in 2:ny-1
            laplacian_c = (2.0 * (c_work[2,j] - c_work[1,j])) / dx^2 +
                          (c_work[1,j+1] - 2.0 * c_work[1,j] + c_work[1,j-1]) / dy^2
            mu_new[1,j] = dfdphi(c_work[1,j], chi, N1, N2) - kappa * laplacian_c
        end

        # Right boundary (i=nx)
        for j in 2:ny-1
            laplacian_c = (2.0 * (c_work[nx-1,j] - c_work[nx,j])) / dx^2 +
                          (c_work[nx,j+1] - 2.0 * c_work[nx,j] + c_work[nx,j-1]) / dy^2
            mu_new[nx,j] = dfdphi(c_work[nx,j], chi, N1, N2) - kappa * laplacian_c
        end

        # Bottom boundary (j=1)
        for i in 2:nx-1
            laplacian_c = (c_work[i+1,1] - 2.0 * c_work[i,1] + c_work[i-1,1]) / dx^2 +
                          (2.0 * (c_work[i,2] - c_work[i,1])) / dy^2
            mu_new[i,1] = dfdphi(c_work[i,1], chi, N1, N2) - kappa * laplacian_c
        end

        # Top boundary (j=ny)
        for i in 2:nx-1
            laplacian_c = (c_work[i+1,ny] - 2.0 * c_work[i,ny] + c_work[i-1,ny]) / dx^2 +
                          (2.0 * (c_work[i,ny-1] - c_work[i,ny])) / dy^2
            mu_new[i,ny] = dfdphi(c_work[i,ny], chi, N1, N2) - kappa * laplacian_c
        end

        # Corners
        # Bottom-left corner (i=1, j=1)
        laplacian_c = (2.0 * (c_work[2,1] - c_work[1,1])) / dx^2 +
                      (2.0 * (c_work[1,2] - c_work[1,1])) / dy^2
        mu_new[1,1] = dfdphi(c_work[1,1], chi, N1, N2) - kappa * laplacian_c

        # Bottom-right corner (i=nx, j=1)
        laplacian_c = (2.0 * (c_work[nx-1,1] - c_work[nx,1])) / dx^2 +
                      (2.0 * (c_work[nx,2] - c_work[nx,1])) / dy^2
        mu_new[nx,1] = dfdphi(c_work[nx,1], chi, N1, N2) - kappa * laplacian_c

        # Top-left corner (i=1, j=ny)
        laplacian_c = (2.0 * (c_work[2,ny] - c_work[1,ny])) / dx^2 +
                      (2.0 * (c_work[1,ny-1] - c_work[1,ny])) / dy^2
        mu_new[1,ny] = dfdphi(c_work[1,ny], chi, N1, N2) - kappa * laplacian_c

        # Top-right corner (i=nx, j=ny)
        laplacian_c = (2.0 * (c_work[nx-1,ny] - c_work[nx,ny])) / dx^2 +
                      (2.0 * (c_work[nx,ny-1] - c_work[nx,ny])) / dy^2
        mu_new[nx,ny] = dfdphi(c_work[nx,ny], chi, N1, N2) - kappa * laplacian_c

        # Compute residuals F

        # Interior points
        for i in 2:nx-1
            for j in 2:ny-1
                # Mobility at interfaces
                M_iphalf = M_func_half(c_work[i,j], c_work[i+1,j])
                M_imhalf = M_func_half(c_work[i,j], c_work[i-1,j])
                M_jphalf = M_func_half(c_work[i,j], c_work[i,j+1])
                M_jmhalf = M_func_half(c_work[i,j], c_work[i,j-1])

                # Fluxes
                Jx_iphalf = -M_iphalf * (mu_new[i+1,j] - mu_new[i,j]) / dx
                Jx_imhalf = -M_imhalf * (mu_new[i,j] - mu_new[i-1,j]) / dx
                Jy_jphalf = -M_jphalf * (mu_new[i,j+1] - mu_new[i,j]) / dy
                Jy_jmhalf = -M_jmhalf * (mu_new[i,j] - mu_new[i,j-1]) / dy

                # Divergence of fluxes
                div_J = (Jx_iphalf - Jx_imhalf) / dx + (Jy_jphalf - Jy_jmhalf) / dy

                # Residual
                F[i,j] = (c_new[i,j] - c_old[i,j]) / dt + div_J
            end
        end

        # Left boundary (i=1)
    for j in 2:ny-1
        M_iphalf = M_func_half(c_work[1,j], c_work[2,j])
        M_imhalf = M_func_half(c_work[1,j], c_work[1,j])  # Neumann BC

        M_jphalf = M_func_half(c_work[1,j], c_work[1,j+1])
        M_jmhalf = M_func_half(c_work[1,j], c_work[1,j-1])

        # Fluxes
        Jx_iphalf = -M_iphalf * (mu_new[2,j] - mu_new[1,j]) / dx
        Jx_imhalf = 0.0  # Neumann BC implies zero flux at boundary

        Jy_jphalf = -M_jphalf * (mu_new[1,j+1] - mu_new[1,j]) / dy
        Jy_jmhalf = -M_jmhalf * (mu_new[1,j] - mu_new[1,j-1]) / dy

        # Divergence of fluxes
        div_J = (Jx_iphalf - Jx_imhalf) / dx + (Jy_jphalf - Jy_jmhalf) / dy

        F[1,j] = (c_new[1,j] - c_old[1,j]) / dt + div_J
    end

    # Right boundary (i=nx)
    for j in 2:ny-1
        M_iphalf = M_func_half(c_work[nx,j], c_work[nx,j])  # Neumann BC
        M_imhalf = M_func_half(c_work[nx,j], c_work[nx-1,j])

        M_jphalf = M_func_half(c_work[nx,j], c_work[nx,j+1])
        M_jmhalf = M_func_half(c_work[nx,j], c_work[nx,j-1])

        # Fluxes
        Jx_iphalf = 0.0  # Neumann BC implies zero flux at boundary
        Jx_imhalf = -M_imhalf * (mu_new[nx,j] - mu_new[nx-1,j]) / dx

        Jy_jphalf = -M_jphalf * (mu_new[nx,j+1] - mu_new[nx,j]) / dy
        Jy_jmhalf = -M_jmhalf * (mu_new[nx,j] - mu_new[nx,j-1]) / dy

        # Divergence of fluxes
        div_J = (Jx_iphalf - Jx_imhalf) / dx + (Jy_jphalf - Jy_jmhalf) / dy

        F[nx,j] = (c_new[nx,j] - c_old[nx,j]) / dt + div_J
    end

    # Bottom boundary (j=1)
    for i in 2:nx-1
        M_iphalf = M_func_half(c_work[i,1], c_work[i+1,1])
        M_imhalf = M_func_half(c_work[i,1], c_work[i-1,1])

        M_jphalf = M_func_half(c_work[i,1], c_work[i,2])
        M_jmhalf = M_func_half(c_work[i,1], c_work[i,1])  # Neumann BC

        # Fluxes
        Jx_iphalf = -M_iphalf * (mu_new[i+1,1] - mu_new[i,1]) / dx
        Jx_imhalf = -M_imhalf * (mu_new[i,1] - mu_new[i-1,1]) / dx

        Jy_jphalf = -M_jphalf * (mu_new[i,2] - mu_new[i,1]) / dy
        Jy_jmhalf = 0.0  # Neumann BC implies zero flux at boundary

        # Divergence of fluxes
        div_J = (Jx_iphalf - Jx_imhalf) / dx + (Jy_jphalf - Jy_jmhalf) / dy

        F[i,1] = (c_new[i,1] - c_old[i,1]) / dt + div_J
    end

    # Top boundary (j=ny)
    for i in 2:nx-1
        M_iphalf = M_func_half(c_work[i,ny], c_work[i+1,ny])
        M_imhalf = M_func_half(c_work[i,ny], c_work[i-1,ny])

        M_jphalf = M_func_half(c_work[i,ny], c_work[i,ny])  # Neumann BC
        M_jmhalf = M_func_half(c_work[i,ny], c_work[i,ny-1])

        # Fluxes
        Jx_iphalf = -M_iphalf * (mu_new[i+1,ny] - mu_new[i,ny]) / dx
        Jx_imhalf = -M_imhalf * (mu_new[i,ny] - mu_new[i-1,ny]) / dx

        Jy_jphalf = 0.0  # Neumann BC implies zero flux at boundary
        Jy_jmhalf = -M_jmhalf * (mu_new[i,ny] - mu_new[i,ny-1]) / dy

        # Divergence of fluxes
        div_J = (Jx_iphalf - Jx_imhalf) / dx + (Jy_jphalf - Jy_jmhalf) / dy

        F[i,ny] = (c_new[i,ny] - c_old[i,ny]) / dt + div_J
    end

    # Corners
    # Bottom-left corner (i=1, j=1)
    M_iphalf = M_func_half(c_work[1,1], c_work[2,1])
    M_imhalf = M_func_half(c_work[1,1], c_work[1,1])  # Neumann BC

    M_jphalf = M_func_half(c_work[1,1], c_work[1,2])
    M_jmhalf = M_func_half(c_work[1,1], c_work[1,1])  # Neumann BC

    # Fluxes
    Jx_iphalf = -M_iphalf * (mu_new[2,1] - mu_new[1,1]) / dx
    Jx_imhalf = 0.0  # Neumann BC

    Jy_jphalf = -M_jphalf * (mu_new[1,2] - mu_new[1,1]) / dy
    Jy_jmhalf = 0.0  # Neumann BC

    # Divergence of fluxes
    div_J = (Jx_iphalf - Jx_imhalf) / dx + (Jy_jphalf - Jy_jmhalf) / dy

    F[1,1] = (c_new[1,1] - c_old[1,1]) / dt + div_J

    # Bottom-right corner (i=nx, j=1)
    M_iphalf = M_func_half(c_work[nx,1], c_work[nx,1])  # Neumann BC
    M_imhalf = M_func_half(c_work[nx,1], c_work[nx-1,1])

    M_jphalf = M_func_half(c_work[nx,1], c_work[nx,2])
    M_jmhalf = M_func_half(c_work[nx,1], c_work[nx,1])  # Neumann BC

    # Fluxes
    Jx_iphalf = 0.0  # Neumann BC
    Jx_imhalf = -M_imhalf * (mu_new[nx,1] - mu_new[nx-1,1]) / dx

    Jy_jphalf = -M_jphalf * (mu_new[nx,2] - mu_new[nx,1]) / dy
    Jy_jmhalf = 0.0  # Neumann BC

    # Divergence of fluxes
    div_J = (Jx_iphalf - Jx_imhalf) / dx + (Jy_jphalf - Jy_jmhalf) / dy

    F[nx,1] = (c_new[nx,1] - c_old[nx,1]) / dt + div_J

    # Top-left corner (i=1, j=ny)
    M_iphalf = M_func_half(c_work[1,ny], c_work[2,ny])
    M_imhalf = M_func_half(c_work[1,ny], c_work[1,ny])  # Neumann BC

    M_jphalf = M_func_half(c_work[1,ny], c_work[1,ny])  # Neumann BC
    M_jmhalf = M_func_half(c_work[1,ny], c_work[1,ny-1])

    # Fluxes
    Jx_iphalf = -M_iphalf * (mu_new[2,ny] - mu_new[1,ny]) / dx
    Jx_imhalf = 0.0  # Neumann BC

    Jy_jphalf = 0.0  # Neumann BC
    Jy_jmhalf = -M_jmhalf * (mu_new[1,ny] - mu_new[1,ny-1]) / dy

    # Divergence of fluxes
    div_J = (Jx_iphalf - Jx_imhalf) / dx + (Jy_jphalf - Jy_jmhalf) / dy

    F[1,ny] = (c_new[1,ny] - c_old[1,ny]) / dt + div_J

    # Top-right corner (i=nx, j=ny)
    M_iphalf = M_func_half(c_work[nx,ny], c_work[nx,ny])  # Neumann BC
    M_imhalf = M_func_half(c_work[nx,ny], c_work[nx-1,ny])

    M_jphalf = M_func_half(c_work[nx,ny], c_work[nx,ny])  # Neumann BC
    M_jmhalf = M_func_half(c_work[nx,ny], c_work[nx,ny-1])

    # Fluxes
    Jx_iphalf = 0.0  # Neumann BC
    Jx_imhalf = -M_imhalf * (mu_new[nx,ny] - mu_new[nx-1,ny]) / dx

    Jy_jphalf = 0.0  # Neumann BC
    Jy_jmhalf = -M_jmhalf * (mu_new[nx,ny] - mu_new[nx,ny-1]) / dy

    # Divergence of fluxes
    div_J = (Jx_iphalf - Jx_imhalf) / dx + (Jy_jphalf - Jy_jmhalf) / dy

    F[nx,ny] = (c_new[nx,ny] - c_old[nx,ny]) / dt + div_J
        
    end

    # Time-stepping loop
    anim = Animation()
    for n = 1:nt
        println("Time step: $n")

        # Save the old concentration profile
        c_old = copy(c)

        # Parameters to pass to the residual function
        p = (c_old = c_old, dt = dt, dx = dx, kappa = kappa, chi = chi, nx = nx, ny = ny, N1 = N1, N2 = N2)

        # Initial guess for c_new
        c_guess = copy(c_old)


        # Create the NonlinearProblem
        problem = NonlinearProblem(residual!, c_guess, p)

        # Solve the nonlinear system
        solver = solve(problem, NewtonRaphson(linsolve = KrylovJL_GMRES()), show_trace = Val(true))

        # Update c for the next time step
        c_new_vec = solver.u
        c = reshape(c_new_vec, nx, ny)

        # Compute statistics for plotting
        c_avg[n] = (1/L^2)*trapz((x,y),c)
        c_max[n] = maximum(c)
        c_min[n] = minimum(c)

        # Energy computation (optional)
        # Compute gradients
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
        energy[n] = trapz((x,y),energy_density)

        # Plotting (for animation)
        if mod(n, 10) == 0
            contourf(x, y, c', levels = 20, colorbar = true,
                     title = "Time step: $n", xlabel = "x", ylabel = "y")
            frame(anim)
        end
    end

    # Save the animation
    # gif(anim, "cahn_hilliard_2d_animation.gif", fps = 10)

    time_vals = (1:nt) * dt

    # Return the final concentration profile and computed data
    return c, c_max, c_min, c_avg, energy, time_vals
end

# Run the main function
c_final, c_max, c_min, c_avg, energy, time_vals = impliciteuler_2d(10.0, 1.0, 1.0, 0.1, 0.1)

# Plot max, min, and average concentrations over time
plt = plot(time_vals, c_max, label = "Max(ϕ)", xlabel = "Time", ylabel = "Concentration",
           title = "Concentration Extremes and Average over Time")
plot!(time_vals, c_min, label = "Min(ϕ)")
plot!(time_vals, c_avg, label = "Average(ϕ)")
display(plt)