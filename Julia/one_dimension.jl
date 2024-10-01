# using SparseArrays
# using LinearAlgebra
# using NonlinearSolve
# using Plots


# function main()
#     # Parameters
#     N = 200           # Number of grid points
#     L = 5.0           # Length of the domain
#     dx = L / N        # Spatial grid spacing
#     x = dx .* (0:N-1) # Grid points
#     dt = 1e-2         # Time step size
#     nt = 1000        # Number of time steps
#     chi = 9.0         # Interaction parameter
#     kappa = (2/3) * chi  # Gradient energy term

#     # Initial condition: small random perturbation around c0
#     c0 = 0.5
#     c = c0 .+ 0.05 * (rand(N) .- 0.5)
#     c = clamp.(c, 1e-6, 1 - 1e-6)  # Ensure c stays within (0,1)

#     # Construct first derivative operator D1 with periodic boundary conditions
#     e = ones(N)
#     off_diag = ones(N - 1)

#     # D1 construction
#     D1 = spdiagm(-1 => -0.5 * off_diag, 1 => 0.5 * off_diag) / dx

#     # Apply periodic boundary conditions to D1
#     D1[1, N] = -0.5 / dx
#     D1[N, 1] = 0.5 / dx

#     # Construct Laplacian operator D2 with periodic boundary conditions
#     D2 = spdiagm(-1 => off_diag, 0 => -2 * e, 1 => off_diag) / dx^2

#     # Apply periodic boundary conditions to D2
#     D2[1, N] = 1 / dx^2
#     D2[N, 1] = 1 / dx^2
#     anim = Animation()

#     # For animation
#     plt = plot(x, c, ylim = (0, 1), title = "Time step: 0", xlabel = "x", ylabel = "Concentration c")
#     anim = @animate for n = 1:nt
#         println(n)
#         # Save the old concentration profile
#         c_old = copy(c)

#         # Parameters to pass to the residual function
#         p = (c_old = c_old, D1 = D1, D2 = D2, dt = dt, chi = chi, kappa = kappa)

#         # Initial guess for c_new
#         c_guess = copy(c_old)

#         # Define the residual function
#         function residual!(F, c_new, p)
#             c_old = p.c_old
#             D1 = p.D1
#             D2 = p.D2
#             dt = p.dt
#             chi = p.chi
#             kappa = p.kappa

#             # Create a working copy of c_new to avoid modifying the input argument
#             c_work = clamp.(c_new, 1e-6, 1 - 1e-6)

#             # Compute mobility M(c_new)
#             M = c_work .* (1 .- c_work)

#             # Compute f'(c_new)
#             fp = log.(c_work) .- log.(1 .- c_work) .+ chi .* (1 .- 2 .* c_work)

#             # Compute mu_new
#             mu_new = fp .- kappa .* (D2 * c_work)

#             # Compute grad_mu
#             grad_mu = D1 * mu_new

#             # Compute flux = M .* grad_mu
#             flux = M .* grad_mu

#             # Compute divergence of flux
#             div_flux = D1 * flux

#             # Compute residual
#             F .= c_new .- dt .* div_flux .- c_old
#         end

#         # Create the NonlinearProblem
#         problem = NonlinearProblem(residual!, c_guess, p)

#         # Solve the nonlinear system
#         solver = solve(problem, NewtonRaphson(), show_trace=Val(true))

#         # Update c for the next time step
#         c = solver.u

#         # Ensure c stays within (0,1)
#         c = clamp.(c, 1e-6, 1 - 1e-6)

        

#         # Plotting (for animation)
#         if mod(n, 10) == 0
#             plt = plot(x, c, ylim = (0, 1), title = "Time step: $n", xlabel = "x", ylabel = "Concentration c", legend = false)
#             # Capture the frame
#             frame(anim)
#         end
#     end

#     # Save the animation
#     gif(anim, "cahn_hilliard_animation.gif", fps = 30)

#     # Return the final concentration profile
#     return c
# end



# # Run the main function
# c_final = main()

using SparseArrays
using LinearAlgebra
using NonlinearSolve
using Plots
using Krylov


function main()
    # Parameters
    N = 200           # Number of grid points
    L = 5.0           # Length of the domain
    dx = L / N        # Spatial grid spacing
    x = dx .* (0:N-1) # Grid points
    dt = 1e-4         # Time step size
    nt = 20000         # Number of time steps
    chi = 18.0         # Interaction parameter
    kappa = (2/3) * chi  # Gradient energy term

    # Initial condition: small random perturbation around c0
    c0 = 0.5
    c = c0 .+ 0.05 * (rand(N) .- 0.5)
    # c = clamp.(c, 1e-6, 1 - 1e-6)  # Ensure c stays within (0,1)

    # Functions
    function dfdphi(phi, chi)
        return log.(phi) - log.(1 .- phi) + chi * (1 .- 2 .* phi)
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
        kappa = p.kappa
        chi = p.chi
        N = length(c_new)

        # Ensure c_new stays within (0,1)
        # c_work = clamp.(c_new, 1e-6, 1 .- 1e-6)
        c_work = c_new

        # Compute mu_new
        mu_new = similar(c_new)

        # Left boundary (Neumann BC)
        mu_new[1] = dfdphi(c_work[1], chi) - (2.0 * kappa / dx^2) * (c_work[2] - c_work[1])

        # Interior points
        for i in 2:N-1
            mu_new[i] = dfdphi(c_work[i], chi) - (kappa / dx^2) * (c_work[i+1] - 2.0 * c_work[i] + c_work[i-1])
        end

        # Right boundary (Neumann BC)
        mu_new[N] = dfdphi(c_work[N], chi) - (2.0 * kappa / dx^2) * (c_work[N-1] - c_work[N])

        # Compute residuals F

        # Left boundary (Neumann BC)
        M_iphalf = M_func_half(c_work[1], c_work[2])
        F[1] = (c_new[1] - c_old[1]) / dt - (2.0 / dx^2) * M_iphalf * (mu_new[2] - mu_new[1])

        # Interior points
        for i in 2:N-1
            M_iphalf = M_func_half(c_work[i], c_work[i+1])
            M_imhalf = M_func_half(c_work[i], c_work[i-1])
            F[i] = (c_new[i] - c_old[i]) / dt - (1.0 / dx^2) * (
                M_iphalf * (mu_new[i+1] - mu_new[i]) - M_imhalf * (mu_new[i] - mu_new[i-1])
            )
        end

        # Right boundary (Neumann BC)
        M_imhalf = M_func_half(c_work[N], c_work[N-1])
        F[N] = (c_new[N] - c_old[N]) / dt - (2.0 / dx^2) * M_imhalf * (mu_new[N] - mu_new[N-1])
    end

    # Animation setup
    anim = Animation()
    anim = @animate for n = 1:nt
        println("Time step: $n")

        # Save the old concentration profile
        c_old = copy(c)

        # Parameters to pass to the residual function
        p = (c_old = c_old, dt = dt, dx = dx, kappa = kappa, chi = chi)

        # Initial guess for c_new
        c_guess = copy(c_old)

        # Create the NonlinearProblem
        problem = NonlinearProblem(residual!, c_guess, p)

        # Solve the nonlinear system
        solver = solve(problem, NewtonRaphson(linsolve = KrylovJL_GMRES()),show_trace=Val(true))

        # Update c for the next time step
        c = solver.u

        # Ensure c stays within (0,1)
        c = clamp.(c, 1e-6, 1 .- 1e-6)

        # Plotting (for animation)
        if mod(n, 10) == 0
            plot(x, c, ylim = (0, 1), title = "Time step: $n", xlabel = "x", ylabel = "Concentration c", legend = false)
            frame(anim)
        end
    end

    # Save the animation
    gif(anim, "cahn_hilliard_animation.gif", fps = 30)

    # Return the final concentration profile
    return c
end

# Run the main function
c_final = main()