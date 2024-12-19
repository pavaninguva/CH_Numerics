using SparseArrays
using LinearAlgebra
using NonlinearSolve
using Plots
using Krylov
using Trapz
using IterativeSolvers
using LaTeXStrings
using Random
using Statistics

Random.seed!(1234)


function impliciteuler(chi,N1,N2,dx,dt)
    L = 4.0
    tf = 20.0
    nx = Int(L/dx) + 1
    x = range(0,L,nx)
    dt = dt         # Time step size
    nt = Int(tf/dt)        # Number of time steps
    chi = chi         # Interaction parameter
    kappa = (2/3) * chi  # Gradient energy term

    # Initial condition: small random perturbation around c0
    c0 = 0.5
    c = c0 .+ 0.02 * (rand(nx) .- 0.5)

    # Initialize arrays to store results
    c_max = zeros(nt)
    c_min = zeros(nt)
    c_avg = zeros(nt)
    energy = zeros(nt)


    # Functions
    function flory_huggins(phi,chi, N1,N2)
        return (1/N1) * (phi .* log.(phi)) + (1/N2) * (1 .- phi) .* log.(1 .- phi) + chi .* phi .* (1 .- phi)
    end

    function dfdphi(phi, chi)
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
        kappa = p.kappa
        chi = p.chi
        nx = length(c_new)
 
        # Ensure c_new stays within (0,1)
        c_work = c_new

        # Compute mu_new
        mu_new = similar(c_new)

        # Left boundary (Neumann BC)
        mu_new[1] = dfdphi(c_work[1], chi) - (2.0 * kappa / dx^2) * (c_work[2] - c_work[1])

        # Interior points
        for i in 2:nx-1
            mu_new[i] = dfdphi(c_work[i], chi) - (kappa / dx^2) * (c_work[i+1] - 2.0 * c_work[i] + c_work[i-1])
        end

        # Right boundary (Neumann BC)
        mu_new[nx] = dfdphi(c_work[nx], chi) - (2.0 * kappa / dx^2) * (c_work[nx-1] - c_work[nx])

        # Compute residuals F

        # Left boundary (Neumann BC)
        M_iphalf = M_func_half(c_work[1], c_work[2])
        F[1] = (c_new[1] - c_old[1]) / dt - (2.0 / dx^2) * M_iphalf * (mu_new[2] - mu_new[1])

        # Interior points
        for i in 2:nx-1
            M_iphalf = M_func_half(c_work[i], c_work[i+1])
            M_imhalf = M_func_half(c_work[i], c_work[i-1])
            F[i] = (c_new[i] - c_old[i]) / dt - (1.0 / dx^2) * (
                M_iphalf * (mu_new[i+1] - mu_new[i]) - M_imhalf * (mu_new[i] - mu_new[i-1])
            )
        end

        # Right boundary (Neumann BC)
        M_imhalf = M_func_half(c_work[nx], c_work[nx-1])
        F[nx] = (c_new[nx] - c_old[nx]) / dt - (2.0 / dx^2) * M_imhalf * (mu_new[nx] - mu_new[nx-1])
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

        #Compute stuff for Plotting
        
        #Mass conservation
        c_avg[n] = mean(c)

        #Max and min Values
        c_max[n] = maximum(c)
        c_min[n] = minimum(c)

        #energy
        dc_dx = zeros(nx)
        # Left boundary
        dc_dx[1] = (c[2] - c[1]) / dx  # Forward difference
        # Interior points
        for i in 2:nx - 1
            dc_dx[i] = (c[i + 1] - c[i - 1]) / (2 * dx)
        end
        # Right boundary
        dc_dx[nx] = (c[nx] - c[nx - 1]) / dx  # Backward difference

        energy_density = flory_huggins(c,chi,N1,N2) + (kappa/2)*dc_dx.^2

        energy[n] = trapz(x,energy_density)

        # Plotting (for animation)
        if mod(n, 10) == 0
            @assert length(x) == length(c) "Lengths of x and c do not match!"
            plot(x, c, ylim = (0, 1), title = "Time step: $n", xlabel = "x", ylabel = "Concentration c", legend = false)
            frame(anim)
        end
    end

    time = (1:nt)*dt

    # Return the final concentration profile
    return c, c_max, c_min, c_avg, energy, time

end

# Run the main function
c_final1, c_max1, c_min1, c_avg1, energy1, time1 = impliciteuler(6.0, 1.0, 1.0, 0.1, 0.05)
c_final2, c_max2, c_min2, c_avg2, energy2, time2 = impliciteuler(6.0, 1.0, 1.0, 0.05, 0.05)
c_final3, c_max3, c_min3, c_avg3, energy3, time3 = impliciteuler(6.0, 1.0, 1.0, 0.02, 0.05)
c_final4, c_max4, c_min4, c_avg4, energy4, time4 = impliciteuler(6.0, 1.0, 1.0, 0.01, 0.05)

# Plot max, min, and average concentrations over time
p1 = plot(time1, c_max1, ylabel = L"\max(\phi)", xlabel = L"\textrm{Time}", label=L"\Delta t = 0.1, \Delta x = 0.05",color=:blue,y_guidefontcolor=:blue,y_foreground_color_axis=:blue,y_foreground_color_text=:blue,y_foreground_color_border=:blue)
plot!(p1,time2,c_max2, label=L"\Delta t = 0.1, \Delta x = 0.05",color=:blue, linestyle=:dash)
plot!(p1,time3,c_max3, label=L"\Delta t = 0.1, \Delta x = 0.01",color=:blue, linestyle=:dot)
plot!(p1,time4,c_max4, label=L"\Delta t = 0.1, \Delta x = 0.005",color=:blue, linestyle=:dashdot)
ylims!(p1,0,1)
axis1 = twinx(p1)
ylims!(axis1,0,1)
plot!(axis1,time1, c_min1, ylabel = L"\min(\phi)",color = :red,label="",y_guidefontcolor=:red,y_foreground_color_axis=:red,y_foreground_color_text=:red,y_foreground_color_border=:red)
plot!(axis1,time2,c_min2,color = :red,linestyle=:dash,label="")
plot!(axis1,time3,c_min3,color = :red,linestyle=:dot,label="")
plot!(axis1,time4,c_min4,color = :red,linestyle=:dashdot,label="")

p2 = plot(time1, c_avg1, ylabel = L"\bar{\phi}", xlabel = L"\textrm{Time}", label="",color=:blue,y_guidefontcolor=:blue,y_foreground_color_axis=:blue,y_foreground_color_text=:blue,y_foreground_color_border=:blue)
plot!(p2,time2,c_avg2,color=:blue, linestyle=:dash,label="")
plot!(p2,time3,c_avg3,color=:blue, linestyle=:dot,label="")
plot!(p2,time4,c_avg4,color=:blue, linestyle=:dashdot,label="")
ylims!(p2,0,1)
axis2 = twinx(p2)
plot!(axis2,time1, energy1, ylabel = L"\textrm{Energy}",color = :red, label="",y_guidefontcolor=:red,y_foreground_color_axis=:red,y_foreground_color_text=:red,y_foreground_color_border=:red)
plot!(axis2,time2,energy2,color=:red, linestyle=:dash,label="")
plot!(axis2,time3,energy3,color=:red, linestyle=:dot,label="")
plot!(axis2,time4,energy4,color=:red, linestyle=:dashdot,label="")

plot!(p1,p2,layout=(1,2), size=(800, 400), 
    tickfont=Plots.font("Computer Modern", 12), grid=false,
    legendfont=Plots.font("Computer Modern",8),dpi=300, 
    bottom_margin = 3Plots.mm, left_margin = 3Plots.mm, right_margin=3Plots.mm, legend=(0.55,0.5))
