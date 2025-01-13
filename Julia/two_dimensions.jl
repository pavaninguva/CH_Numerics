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
using ForwardDiff

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
    energy_method = p.energy_method

    # spline = spline_generator(chi, N1, N2,100)
    #Logic for swtiching between spline and analytical
    # dfdphi = ϕ -> begin 
    #     if energy_method == "analytical"
    #         dfdphi_ana(ϕ,N1,N2,chi)
    #     else
    #         spline.(ϕ)
    #     end
    # end
    dfdphi = phi -> dfdphi_ana(phi,chi,N1, N2)

    # Compute mu_new
    mu_new = similar(c_new)

    # Helper function to calculate chemical potential
    function compute_mu(i, j)
        laplacian_c = (c_new[i+1,j] - 2.0 * c_new[i,j] + c_new[i-1,j]) / dx^2 +
                      (c_new[i,j+1] - 2.0 * c_new[i,j] + c_new[i,j-1]) / dy^2
        return dfdphi(c_new[i,j]) - kappa * laplacian_c
    end

    function M_func(phi)
        return phi .* (1 .- phi)
    end
    
    function M_func_half(phi1, phi2)
        return M_func(0.5 .* (phi1 .+ phi2))
    end

    # Compute mu_new for all nodes
    for i in 1:nx
        for j in 1:ny
            if i == 1 || i == nx || j == 1 || j == ny
                # Boundary nodes
                if i == 1 && j > 1 && j < ny
                    laplacian_c = (2.0 * (c_new[2,j] - c_new[1,j])) / dx^2 +
                                  (c_new[1,j+1] - 2.0 * c_new[1,j] + c_new[1,j-1]) / dy^2
                elseif i == nx && j > 1 && j < ny
                    laplacian_c = (2.0 * (c_new[nx-1,j] - c_new[nx,j])) / dx^2 +
                                  (c_new[nx,j+1] - 2.0 * c_new[nx,j] + c_new[nx,j-1]) / dy^2
                elseif j == 1 && i > 1 && i < nx
                    laplacian_c = (c_new[i+1,1] - 2.0 * c_new[i,1] + c_new[i-1,1]) / dx^2 +
                                  (2.0 * (c_new[i,2] - c_new[i,1])) / dy^2
                elseif j == ny && i > 1 && i < nx
                    laplacian_c = (c_new[i+1,ny] - 2.0 * c_new[i,ny] + c_new[i-1,ny]) / dx^2 +
                                  (2.0 * (c_new[i,ny-1] - c_new[i,ny])) / dy^2
                elseif i == 1 && j == 1
                    laplacian_c = (2.0 * (c_new[2,1] - c_new[1,1])) / dx^2 +
                                  (2.0 * (c_new[1,2] - c_new[1,1])) / dy^2
                elseif i == nx && j == 1
                    laplacian_c = (2.0 * (c_new[nx-1,1] - c_new[nx,1])) / dx^2 +
                                  (2.0 * (c_new[nx,2] - c_new[nx,1])) / dy^2
                elseif i == 1 && j == ny
                    laplacian_c = (2.0 * (c_new[2,ny] - c_new[1,ny])) / dx^2 +
                                  (2.0 * (c_new[1,ny-1] - c_new[1,ny])) / dy^2
                elseif i == nx && j == ny
                    laplacian_c = (2.0 * (c_new[nx-1,ny] - c_new[nx,ny])) / dx^2 +
                                  (2.0 * (c_new[nx,ny-1] - c_new[nx,ny])) / dy^2
                end

                mu_new[i,j] = dfdphi(c_new[i,j]) - kappa * laplacian_c
            else
                # Interior nodes
                mu_new[i,j] = compute_mu(i, j)
            end
        end
    end

    # Compute residuals F
    for i in 1:nx
        for j in 1:ny
            if i == 1 || i == nx || j == 1 || j == ny
                # Boundary nodes: Update based on ghost node approach
                if i == 1 && j > 1 && j < ny
                    M_iphalf = M_func_half(c_new[1,j], c_new[2,j])
                    M_jphalf = M_func_half(c_new[1,j], c_new[1,j+1])
                    M_jmhalf = M_func_half(c_new[1,j], c_new[1,j-1])

                    Jx_iphalf = 2.0 * M_iphalf * (mu_new[2,j] - mu_new[1,j]) / dx^2
                    Jy_jphalf = M_jphalf * (mu_new[1,j+1] - mu_new[1,j]) / dy^2
                    Jy_jmhalf = M_jmhalf * (mu_new[1,j-1] - mu_new[1,j]) / dy^2

                    div_J = Jx_iphalf + (Jy_jphalf - Jy_jmhalf)

                    F[1,j] = (c_new[1,j] - c_old[1,j]) / dt - div_J
                elseif i == nx && j > 1 && j < ny
                    M_imhalf = M_func_half(c_new[nx,j], c_new[nx-1,j])
                    M_jphalf = M_func_half(c_new[nx,j], c_new[nx,j+1])
                    M_jmhalf = M_func_half(c_new[nx,j], c_new[nx,j-1])

                    Jx_imhalf = 2.0 * M_imhalf * (mu_new[nx-1,j] - mu_new[nx,j]) / dx^2
                    Jy_jphalf = M_jphalf * (mu_new[nx,j+1] - mu_new[nx,j]) / dy^2
                    Jy_jmhalf = M_jmhalf * (mu_new[nx,j-1] - mu_new[nx,j]) / dy^2

                    div_J = Jx_imhalf + (Jy_jphalf - Jy_jmhalf)

                    F[nx,j] = (c_new[nx,j] - c_old[nx,j]) / dt - div_J
                elseif j == 1 && i > 1 && i < nx
                    M_iphalf = M_func_half(c_new[i,1], c_new[i+1,1])
                    M_imhalf = M_func_half(c_new[i,1], c_new[i-1,1])
                    M_jphalf = M_func_half(c_new[i,1], c_new[i,2])

                    Jx_iphalf = M_iphalf * (mu_new[i+1,1] - mu_new[i,1]) / dx^2
                    Jx_imhalf = M_imhalf * (mu_new[i-1,1] - mu_new[i,1]) / dx^2
                    Jy_jphalf = 2.0 * M_jphalf * (mu_new[i,2] - mu_new[i,1]) / dy^2

                    div_J = (Jx_iphalf - Jx_imhalf) + Jy_jphalf

                    F[i,1] = (c_new[i,1] - c_old[i,1]) / dt - div_J
                elseif j == ny && i > 1 && i < nx
                    M_iphalf = M_func_half(c_new[i,ny], c_new[i+1,ny])
                    M_imhalf = M_func_half(c_new[i,ny], c_new[i-1,ny])
                    M_jmhalf = M_func_half(c_new[i,ny], c_new[i,ny-1])

                    Jx_iphalf = M_iphalf * (mu_new[i+1,ny] - mu_new[i,ny]) / dx^2
                    Jx_imhalf = M_imhalf * (mu_new[i-1,ny] - mu_new[i,ny]) / dx^2
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
                end
            else
                # Interior nodes
                M_iphalf = M_func_half(c_new[i,j], c_new[i+1,j])
                M_imhalf = M_func_half(c_new[i,j], c_new[i-1,j])
                M_jphalf = M_func_half(c_new[i,j], c_new[i,j+1])
                M_jmhalf = M_func_half(c_new[i,j], c_new[i,j-1])

                Jx_iphalf = -M_iphalf * (mu_new[i+1,j] - mu_new[i,j]) / dx^2
                Jx_imhalf = -M_imhalf * (mu_new[i,j] - mu_new[i-1,j]) / dx^2
                Jy_jphalf = -M_jphalf * (mu_new[i,j+1] - mu_new[i,j]) / dy^2
                Jy_jmhalf = -M_jmhalf * (mu_new[i,j] - mu_new[i,j-1]) / dy^2

                div_J = (Jx_iphalf - Jx_imhalf) + (Jy_jphalf - Jy_jmhalf)

                F[i,j] = (c_new[i,j] - c_old[i,j]) / dt - div_J
            end
        end
    end
end

function impliciteuler_2d(chi, N1, N2, dx, dt, energy_method::String)
    L = 10.0
    tf = 5.0
    nx = Int(L / dx) + 1
    ny = nx  # Assuming square domain
    x = range(0, L, length = nx)
    y = range(0, L, length = ny)
    nt = Int(tf / dt)
    kappa = (2 / 3) * chi  # Gradient energy term
    println(energy_method)

    # Initial condition: small random perturbation around c0
    c0 = 0.5
    c = c0 .+ 0.02 * (rand(nx, ny) .- 0.5)

    # Initialize arrays to store results
    c_max = zeros(nt)
    c_min = zeros(nt)
    c_avg = zeros(nt)
    energy = zeros(nt)

    for n = 1:nt
        println("Time step: $n")

        # Save the old concentration profile
        c_old = copy(c)

        # Parameters to pass to the residual function
        p = (c_old = c_old, dt = dt, dx = dx, kappa = kappa, chi = chi, nx = nx, ny = ny, N1 = N1, N2 = N2, energy_method=energy_method)

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
        # c_avg[n] = (1/L^2)*trapz((x,y),c)
        c_avg[n] = mean(c)
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

    end

    time_vals = (1:nt) * dt

    # Return the final concentration profile and computed data
    return c, c_max, c_min, c_avg, energy, time_vals
end

# Run the main function
c_final, c_max, c_min, c_avg, energy, time_vals = impliciteuler_2d(6.0, 1.0, 1.0, 0.1, 0.1,"analytical")

# Plot max, min, and average concentrations over time
plt = plot(time_vals, c_max, label = "Max(ϕ)", xlabel = "Time", ylabel = "Concentration",
           title = "Concentration Extremes and Average over Time")
plot!(time_vals, c_min, label = "Min(ϕ)")
plot!(time_vals, c_avg, label = "Average(ϕ)")
