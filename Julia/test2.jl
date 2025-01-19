using NonlinearSolve, Plots, LinearAlgebra

# Parameters
Nx, Ny = 100, 100         # Grid points
Lx, Ly = 1.0, 1.0         # Domain size
dx, dy = Lx/Nx, Ly/Ny     # Grid spacing
dt = 1e-3                 # Time step
tf = 1.0                  # Final time
κ = 1.0                   # Gradient energy coefficient
χ = 2.0                   # Flory–Huggins parameter

# Derived functions
function free_energy_derivative(ϕ)
    return log(ϕ) - log(1 - ϕ) + χ * (1 - 2ϕ)
end

function mobility(ϕ)
    return ϕ * (1 - ϕ)
end

# Apply Neumann boundary conditions via ghost nodes
function apply_boundary_conditions!(ϕ)
    # Left and right edges
    for j in 2:Ny-1
        ϕ[1, j] = ϕ[2, j]  # Left boundary
        ϕ[Nx, j] = ϕ[Nx-1, j]  # Right boundary
    end
    # Top and bottom edges
    for i in 2:Nx-1
        ϕ[i, 1] = ϕ[i, 2]  # Bottom boundary
        ϕ[i, Ny] = ϕ[i, Ny-1]  # Top boundary
    end
    # Corners
    ϕ[1, 1] = ϕ[2, 2]         # Bottom-left
    ϕ[Nx, 1] = ϕ[Nx-1, 2]     # Bottom-right
    ϕ[1, Ny] = ϕ[2, Ny-1]     # Top-left
    ϕ[Nx, Ny] = ϕ[Nx-1, Ny-1] # Top-right
end

# Laplacian operator with ghost nodes
function laplacian(ϕ, Nx, Ny, dx, dy)
    apply_boundary_conditions!(ϕ)
    lapϕ = zeros(size(ϕ))
    for i in 2:Nx-1, j in 2:Ny-1
        lapϕ[i, j] = (ϕ[i+1, j] - 2ϕ[i, j] + ϕ[i-1, j]) / dx^2 +
                     (ϕ[i, j+1] - 2ϕ[i, j] + ϕ[i, j-1]) / dy^2
    end
    return lapϕ
end

# Residual function for NonlinearSolve
function residual!(res, ϕ, p, t)
    apply_boundary_conditions!(ϕ)
    μ = free_energy_derivative.(ϕ) - κ * laplacian(ϕ, Nx, Ny, dx, dy)
    mobility_term = mobility.(ϕ) .* laplacian(μ, Nx, Ny, dx, dy)
    res .= (ϕ - p) / dt - mobility_term
end

# Initial condition
ϕ0 = 0.5 .+ 0.1 * rand(Nx, Ny)

# Nonlinear solver setup
prob = NonlinearProblem(residual!, ϕ0, ϕ0)
solver = solve(prob, NewtonRaphson())

# Time-stepping loop
ϕ = copy(ϕ0)
t = 0.0
frames = []
while t < tf
    t += dt
    solver.u = copy(ϕ)
    solve!(solver)
    ϕ .= solver.u
    push!(frames, heatmap(ϕ, color=:viridis, title="t = $t"))
end

# Create animation
animate(frames, fps=20)