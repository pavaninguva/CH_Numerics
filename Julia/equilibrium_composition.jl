using LinearAlgebra
using Plots
using NonlinearSolve
using NaNMath
using LaTeXStrings
using CSV                # Added for CSV handling
using DataFrames         # Added for DataFrame creation


"""
This code is used to generate binodal_spinodal.png in the paper
"""

# Define the function for the binodal calculations
function binodal!(F, phi, params)
    chi, N1, N2 = params

    function fh_deriv(phi, chi, N1, N2)
        df = (1/N1)*NaNMath.log(phi) + (1/N1) - (1/N2)*NaNMath.log(1-phi) - (1/N2) - 2*chi*phi + chi
        return df
    end

    function osmotic(phi, chi, N1, N2)
        osmo = phi*((1/N1)-(1/N2)) - (1/N2)*NaNMath.log(1-phi) - chi*phi^2
        return osmo
    end

    phiA = phi[1]
    phiB = phi[2]
    dF = fh_deriv(phiA, chi, N1, N2) - fh_deriv(phiB, chi, N1, N2)
    dO = osmotic(phiA, chi, N1, N2) - osmotic(phiB, chi, N1, N2)
    F[1] = dF
    F[2] = dO
end

# Function to solve for the binodal points
function binodal_solver(N1, N2, chi_scale_vals)
    phic = sqrt(N2)/(sqrt(N1) + sqrt(N2))
    chic = (sqrt(N1) + sqrt(N2))^2 / (2*N1*N2)

    phi1bn = zeros(length(chi_scale_vals))
    phi2bn = zeros(length(chi_scale_vals))

    x0 = [1e-3, 1 - 1e-3]

    for i in 1:length(chi_scale_vals)
        chi_scaled = chi_scale_vals[i] * chic
        params = (chi_scaled, N1, N2)
        prob = NonlinearProblem(binodal!, x0, params)
        sol = solve(prob, RobustMultiNewton())
        phi1bn[i] = sol.u[1]
        phi2bn[i] = sol.u[2]
        println("beep")
        x0 = copy(sol.u)
    end

    return [phic, chic], phi1bn, phi2bn
end

# Function to compute spinodal curve
function spinodal_phi_chi(N1, N2)
    phi_vals = range(1e-6, 1 - 1e-6, length=1000)
    chi_spinodal = zeros(length(phi_vals))
    for i in 1:length(phi_vals)
        phi = phi_vals[i]
        chi_spinodal[i] = 0.5 * ((1 / N1) / phi + (1 / N2) / (1 - phi))
    end
    return phi_vals, chi_spinodal
end

# Test for N1 = N2 = 1
chi_vals = range(50.0, 1.0, length=5000)
critvals, phi1_, phi2_ = binodal_solver(1, 1, chi_vals)
phi_spinodal, chi_spinodal = spinodal_phi_chi(1, 1)
chi_c = critvals[2]
chi_c_over_chi_spinodal = chi_c ./ chi_spinodal

#N1 = 10, N2= 1
chi_vals2 = range(10.0,1.0,length=5000)
critvals2, phi1_2, phi2_2 = binodal_solver(10,1,chi_vals2)
phi_spinodal2, chi_spinodal2 = spinodal_phi_chi(10,1)
chi_c2 = critvals2[2]
chi_c_over_chi_spinodal2 = chi_c2./ chi_spinodal2

# Export data to CSV files
# Prepare binodal data
# binodal_df = DataFrame(
#     chi_inv = 1.0 ./ chi_vals,
#     phi1 = phi1_,
#     phi2 = phi2_
# )
# CSV.write("binodal_data.csv", binodal_df)

# # Prepare spinodal data
# spinodal_df = DataFrame(
#     phi = phi_spinodal,
#     chi_c_over_chi = chi_c_over_chi_spinodal
# )
# CSV.write("spinodal_data.csv", spinodal_df)

# # Prepare critical point data
# crit_point_df = DataFrame(
#     phi = [critvals[1]],
#     chi_inv = [1.0]
# )
# CSV.write("critical_point.csv", crit_point_df)

# Plot the results
p = plot(phi1_, chi_vals .^ -1, color="blue", label=L"\textrm{Binodal}, x_{1} = x_{2} = 1", lw=1)
plot!(p, phi2_, chi_vals .^ -1, color="blue", lw=1, label="")
plot!(p, phi_spinodal, chi_c_over_chi_spinodal, color="red", label=L"\textrm{Spinodal}, x_{1} = x_{2} = 1", lw=1)
scatter!(p, [critvals[1]], [1.0], color="black", ms=3, label=L"\textrm{Critical \ Point}, x_{1} = x_{2} = 1")

plot!(p, phi1_2, chi_vals2 .^ -1,linestyle=:dash, color="blue",label=L"\textrm{Binodal}, x_{1} = 10,  x_{2} = 1", lw=1)
plot!(p, phi2_2, chi_vals2 .^ -1, color="blue", lw=1, label="",linestyle=:dash)
plot!(p, phi_spinodal2, chi_c_over_chi_spinodal2, color="red", linestyle=:dash, label=L"\textrm{Spinodal}, x_{1} = 10, x_{2} = 1", lw=1)
scatter!(p, [critvals2[1]], [1.0], color="black", ms=3, label=L"\textrm{Critical \ Point}, x_{1} = 10, x_{2} = 1")

xlims!(p, 0, 1)
ylims!(p, 0.0, 1.1)
xlabel!(p, L"\phi_{1}")
ylabel!(p, L"\frac{\chi_c}{\chi_{12}}")
plot!(p, legend=:bottom, size=(600, 600), 
    tickfont=Plots.font("Computer Modern", 12), grid=false,
    legendfont=Plots.font("Computer Modern",8),dpi=300)

# savefig(p,"binodal_spinodal.png")


# Display the plot
# display(p)
