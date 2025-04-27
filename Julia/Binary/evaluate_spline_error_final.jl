using Plots
using LaTeXStrings
using Statistics
include("./pchip.jl")

"""
This script is used to generate the following figures: 
1. spline_error_pchip_nonuniform.png 
2. spline_error_pchip_uniform.png
"""

"""
Define functions
"""


function fh_deriv(phi,chi,N1,N2)
    df = (1/N1).*log.(phi) .+ (1/N1) .- (1/N2).*log.(1 .- phi) .- (1/N2) .- 2*chi.*phi .+ chi
    return df
end

function generate_spline(chi, N1, N2, knots)

    function tanh_sinh_spacing(n, β)
        points = 0.5 * (1 .+ tanh.(β .* (2 * collect(0:n-1) / (n-1) .- 1)))
        return points
    end
    
    phi_vals_ = collect(tanh_sinh_spacing(knots-4,14))

    pushfirst!(phi_vals_,1e-16)
    push!(phi_vals_,1-1e-16)

    f_vals_ = fh_deriv(phi_vals_,chi,N1,N2)

    phi_vals = pushfirst!(phi_vals_,0)
    push!(phi_vals,1)

    #Compute value at eps
    eps_val = BigFloat("1e-40")
    one_big = BigFloat(1)

    f_eps = fh_deriv(eps_val,BigFloat(chi),BigFloat(N1), BigFloat(N2))
    f_eps1 = fh_deriv(one_big-eps_val, BigFloat(chi),BigFloat(N1), BigFloat(N2))

    f_eps_float = Float64(f_eps)
    f_eps1_float = Float64(f_eps1)

    f_vals = pushfirst!(f_vals_,f_eps_float)
    push!(f_vals, f_eps1_float)

    # Build and return the spline function using pchip
    spline = pchip(phi_vals, f_vals)
    return spline
end

# function generate_spline(chi, N1, N2, knots)

#     eps_big  = BigFloat("1e-16")
#     one_big  = BigFloat(1)

#     phi_big  = collect(range(eps_big, stop = one_big - eps_big, length = knots))

#     f_vals = Float64.(fh_deriv(phi_big,BigFloat(chi),BigFloat(N1), BigFloat(N2)))

#     phi_vals = Float64.(phi_big)
#     phi_vals[1] = 0.0
#     phi_vals[end] = 1.0

#     # Build and return the spline function using pchip
#     spline = pchip(phi_vals, f_vals)
#     return spline
# end

function compute_errors(f_true, f_approx, phi_vals)
    y_true = f_true.(phi_vals)
    y_approx = f_approx.(phi_vals)
    errors = y_true .- y_approx
    rmse = sqrt(mean(errors.^2))
    mae = mean(abs.(errors))
    return rmse, mae
end

# Define phi test values (for error computation and plotting)
phi_test = collect(range(1e-16, 1-1e-16, length=1000))

# Define the range of knot counts to test
knots_range = 20:20:1000

# Two cases:
# Case 1: chi = 50, N1 = 1, N2 = 1
chi1, N1_1, N2_1 = 50, 1, 1
# Case 2: chi = 10, N1 = 100, N2 = 1
chi2, N1_2, N2_2 = 10, 100, 1

# Arrays to store error metrics for each case vs. number of knots
rmse_case1 = Float64[]
mae_case1  = Float64[]
rmse_case2 = Float64[]
mae_case2  = Float64[]
knots_values = Int[]

for knots in knots_range
    push!(knots_values, knots)
    # Build spline approximations for both cases
    spline1 = generate_spline(chi1, N1_1, N2_1, knots)
    spline2 = generate_spline(chi2, N1_2, N2_2, knots)
    # Compute errors for case 1
    rmse1, mae1 = compute_errors(phi -> fh_deriv(phi, chi1, N1_1, N2_1), spline1, phi_test)
    push!(rmse_case1, rmse1)
    push!(mae_case1, mae1)
    # Compute errors for case 2
    rmse2, mae2 = compute_errors(phi -> fh_deriv(phi, chi2, N1_2, N2_2), spline2, phi_test)
    push!(rmse_case2, rmse2)
    push!(mae_case2, mae2)
end

fixed_knots = 100
spline1_fixed = generate_spline(chi1, N1_1, N2_1, fixed_knots)
spline2_fixed = generate_spline(chi2, N1_2, N2_2, fixed_knots)

# Evaluate true function and spline over phi_test for both cases
f_true_case1   = [fh_deriv(phi, chi1, N1_1, N2_1) for phi in phi_test]
f_spline_case1 = [spline1_fixed(phi) for phi in phi_test]

f_true_case2   = [fh_deriv(phi, chi2, N1_2, N2_2) for phi in phi_test]
f_spline_case2 = [spline2_fixed(phi) for phi in phi_test]

# --- First subplot: function vs. spline approximation ---
p1 = plot(grid=false, xlabel=L"\phi_{1}", ylabel=L"\frac{\partial f}{\partial \phi_{1}}")
plot!(p1,phi_test, f_true_case1, label="Analytical, "*L"\chi_{12}=50, x_{1}=x_{2} = 1", lw=2,extra_kwargs=Dict(:subplot=>Dict(:legend_hfactor=>1.2)))
plot!(p1, phi_test, f_spline_case1, label="Spline, "*L"\chi_{12}=50, x_{1}=x_{2} = 1", lw=2, ls=:dash)
plot!(p1, phi_test, f_true_case2, label="Analytical, "*L"\chi_{12}=10, x_{1}= 100, x_{2} = 1", lw=2)
plot!(p1, phi_test, f_spline_case2, label="Spline, "*L"\chi_{12}=10, x_{1}=100, x_{2} = 1", lw=2, ls=:dash)


# --- Second subplot: Error metrics vs. number of knots ---
p2 = scatter(size=(500,500),grid=false,xlabel=L"m", ylabel=L"\textrm{Error}",legend=:bottomleft)
scatter!(p2,knots_values, rmse_case1, label="RMSE, "*L"\chi_{12}=50, x_{1}=x_{2} = 1", marker=:circle,xaxis=:log, yaxis=:log)
scatter!(p2, knots_values, mae_case1, label="MAE, "*L"\chi_{12}=50, x_{1}=x_{2} = 1", marker=:diamond)
scatter!(p2, knots_values, rmse_case2, label="RMSE, "*L"\chi_{12}=10, x_{1}= 100, x_{2} = 1", marker=:star5)
scatter!(p2, knots_values, mae_case2, label="MAE, "*L"\chi_{12}=10, x_{1}= 100, x_{2} = 1", marker=:hexagon)


# --- Combine the subplots with the specified formatting ---
plot(p1, p2, layout=(1,2), size=(800, 400),
    tickfont=Plots.font("Computer Modern", 10),
    legendfont=Plots.font("Computer Modern", 7),
    dpi=300,bottom_margin = 3Plots.mm, left_margin = 5Plots.mm, right_margin=3Plots.mm)

savefig("./spline_error_pchip_uniform.png")