# Import necessary libraries
using LinearAlgebra
using Plots
using SparseArrays
using NonlinearSolve

# Define parameters
L = 1.0                    # Length of the domain
Nx = 100                   # Number of grid points
dx = L / (Nx - 1)          # Spatial step size
dt = 1e-4                  # Time step size
t_final = 0.1             # Final time
Nt = Int(t_final / dt)      # Number of time steps
D0 = 1.0                   # Constant diffusion coefficient


# Define spatial grid
x = LinRange(0, L, Nx)

# Define the initial condition
function initial_condition(x)
    return exp(-100 * (x - 0.5)^2)
end

u0 = initial_condition.(x)

# Define the nonlinear diffusion coefficient
function D(u)
    return D0 * (1 .* u).*u
end

function residual!(du, u, p)
    dx, dt, u_old = p
    Nx = length(u)

    #Boundary conditions
    du[1] = u[1] - u_old[1]  # u(0) = u_old(0)
    du[Nx] = u[Nx] - u_old[Nx]  # u(L) = u_old(L)

    # Interior points
    D_plus = D.(0.5 * (u[2:Nx-1] + u[3:Nx]))
    D_minus = D.(0.5 * (u[2:Nx-1] + u[1:Nx-2]))
    du[2:Nx-1] = u[2:Nx-1] .- u_old[2:Nx-1] .- (dt / dx^2) .* (D_plus .* (u[3:Nx] .- u[2:Nx-1]) .- D_minus .* (u[2:Nx-1] .- u[1:Nx-2]))
end

function solve_nonlinear_diffusion(u_old, dt, dx)
    p = (dx, dt, u_old)
    
    # Define nonlinear solver problem
    problem = NonlinearProblem(residual!,u_old,p)
    sol = solve(problem, NewtonRaphson())
    return sol.u
end

# Initialize the solution matrix
u_matrix = zeros(Nt, Nx)
u_matrix[1, :] = u0

# Time-stepping loop
for n in 2:Nt
    u_old = u_matrix[n-1, :]
    u_new = solve_nonlinear_diffusion(u_old, dt, dx)
    u_matrix[n, :] = u_new
end

# Plot and animate the results
anim = @animate for n in 1:Nt
    plot(x, u_matrix[n, :], title="Nonlinear Diffusion", xlabel="x", 
        ylabel="u(x,t)", label="t = $(round(n*dt, digits=4))",grid=false,ylims=(0,1))
end

# Save the animation
mp4(anim, "nonlinear_diffusion.mp4", fps=30)


