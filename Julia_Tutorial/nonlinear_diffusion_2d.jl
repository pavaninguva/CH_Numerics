# Import necessary libraries
using LinearAlgebra
using Plots
using SparseArrays
using NonlinearSolve

# Define parameters
Lx = 1.0                   # Length of the domain in x direction
Ly = 1.0                   # Length of the domain in y direction
Nx = 50                    # Number of grid points in x direction
Ny = 50                    # Number of grid points in y direction
dx = Lx / (Nx - 1)         # Spatial step size in x
dy = Ly / (Ny - 1)         # Spatial step size in y
dt = 1e-3                  # Time step size
t_final = 0.01             # Final time
Nt = Int(t_final / dt)     # Number of time steps
D0 = 1.0                   # Constant diffusion coefficient

# Define spatial grid
x = LinRange(0, Lx, Nx)
y = LinRange(0, Ly, Ny)

# Define the initial condition
function initial_condition(x, y)
    return exp(-100 * ((x - 0.5)^2 + (y - 0.5)^2))
end

u0 = [initial_condition(xi, yi) for yi in y, xi in x]

# Define the nonlinear diffusion coefficient
function D(u)
    return D0 * (1 .* u) .* u
end

# Define the residual function for the nonlinear system
function residual!(du, u, p)
    dx, dy, dt, u_old = p
    Nx, Ny = size(u)

    # Apply boundary conditions (u = u_old at the boundaries)
    for j in 1:Ny
        du[j, 1] = u[j, 1] - u_old[j, 1]     # Left boundary (x = 0)
        du[j, Nx] = u[j, Nx] - u_old[j, Nx]  # Right boundary (x = Lx)
    end

    for i in 1:Nx
        du[1, i] = u[1, i] - u_old[1, i]     # Bottom boundary (y = 0)
        du[Ny, i] = u[Ny, i] - u_old[Ny, i]  # Top boundary (y = Ly)
    end

    # Compute the residual for interior points
    for j in 2:Ny-1, i in 2:Nx-1
        D_x_plus = D(0.5 * (u[j, i] + u[j, i+1]))
        D_x_minus = D(0.5 * (u[j, i] + u[j, i-1]))
        D_y_plus = D(0.5 * (u[j, i] + u[j+1, i]))
        D_y_minus = D(0.5 * (u[j, i] + u[j-1, i]))

        du[j, i] = u[j, i] - u_old[j, i] - dt * (
            (D_x_plus * (u[j, i+1] - u[j, i]) - D_x_minus * (u[j, i] - u[j, i-1])) / dx^2 +
            (D_y_plus * (u[j+1, i] - u[j, i]) - D_y_minus * (u[j, i] - u[j-1, i])) / dy^2
        )
    end
end

# Nonlinear solver for 2D diffusion
function solve_nonlinear_diffusion(u_old, dt, dx, dy)
    p = (dx, dy, dt, u_old)
    
    # Define nonlinear solver problem
    problem = NonlinearProblem(residual!, u_old, p)
    sol = solve(problem, NewtonRaphson())
    return sol.u
end

# Initialize the solution matrix for storing time evolution
u_matrix = zeros(Nt, Ny, Nx)
u_matrix[1, :, :] = u0

# Time-stepping loop
for n in 2:Nt
    u_old = u_matrix[n-1, :, :]
    u_new = solve_nonlinear_diffusion(u_old, dt, dx, dy)
    u_matrix[n, :, :] = u_new
    println(n)
end

# Plot and animate the results
anim = @animate for n in 1:Nt
    p = plot(heatmap(x, y, u_matrix[n, :, :], c=:viridis, color=:blue, title="Nonlinear Diffusion in 2D", xlabel="x", ylabel="y"))
end

# Save the animation as MP4
mp4(anim, "nonlinear_diffusion_2d.mp4", fps=30)
