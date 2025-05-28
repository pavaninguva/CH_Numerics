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
Random.seed!(0000)
const datadir = joinpath(@__DIR__, "Binary_Case2_CSV")

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
    sol = solve(prob, TRBDF2(),reltol=1e-8, abstol=1e-9,maxiters=1e7)

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
Run case 
"""

c_avg_bdf_spline1, energy_bdf_spline1, time_vals_bdf_spline1 = mol_solver(30,1,1,0.1,20,2,"spline")
c_avg_bdf_spline2, energy_bdf_spline2, time_vals_bdf_spline2 = mol_solver(30,1,1,0.05,20,2,"spline")
c_avg_bdf_spline3, energy_bdf_spline3, time_vals_bdf_spline3 = mol_solver(30,1,1,0.04,20,2,"spline")


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



# const suffix_map_bdf_spline = [
#   ("dx_04", "spline1"),
#   ("dx_02", "spline2"),
#   ("dx_01", "spline3"),
# ]

# for (file_sfx, var_sfx) in suffix_map_bdf_spline
#     fname = joinpath(datadir, "bdf2d_spline_$(file_sfx).csv")
#     println("Reading ", fname)
#     df = CSV.read(fname, DataFrame)

#     tvals = df.time
#     cavg  = df.c_avg
#     energ = df.energy

#     t_sym = Symbol("time_vals_bdf_$(var_sfx)")
#     c_sym = Symbol("c_avg_bdf_$(var_sfx)")
#     e_sym = Symbol("energy_bdf_$(var_sfx)")

#     @eval Main begin
#       $(t_sym) = $tvals
#       $(c_sym) = $cavg
#       $(e_sym) = $energ
#     end
# end


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
    size=(500,400),dpi=300
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

display(p4)
# """
# Combined Plot
# """

# p_all = plot(p1,p2,p3,p4, layout=(2,2),size=(800,700),dpi=300,
#                 bottom_margin = 3Plots.mm, left_margin = 3Plots.mm, right_margin=3Plots.mm)
# savefig("2d_benchmarking_case2.png")

# display(p_all)