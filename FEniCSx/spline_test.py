import numpy as np
from scipy.interpolate import make_interp_spline, CubicSpline, PchipInterpolator
import matplotlib.pyplot as plt
from math import sqrt
from mpmath import mp
import mpmath as mpmath

# Set high precision
mp.dps = 50 
MpfType = type(mp.mpf("1e-40"))


#formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')

def fh_deriv(phi, chi, N1, N2):
    return (1/N1)*np.log(phi) + (1/N1) \
           - (1/N2)*np.log(1 - phi) - (1/N2) \
           + chi - 2*chi*phi


# def spline_generator2(chi, N1, N2, knots):
#     #Define small eps
#     eps = 1e-40
    
#     def tanh_sinh_spacing(n, beta):
#         # Return n points between 0 and 1 based on a tanh-sinh distribution
#         # Indices go from 0 to n-1
#         i = np.arange(n, dtype=float)
#         return 0.5 * (1.0 + np.tanh(beta*(2.0*i/(n-1) - 1.0)))
    
#     phi_vals_ = tanh_sinh_spacing(knots - 4, 14.0)
#     #Insert eps
#     phi_vals_ = np.insert(phi_vals_,0,1e-16)
#     phi_vals_ = np.insert(phi_vals_,0,eps)

#     phi_vals_ = np.append(phi_vals_,1.0-1e-16)

#     #Compute dfdphi vals
#     dfdphi = fh_deriv(phi_vals_,chi,N1,N2)

#     #compute eps_right
#     eps_right = mp.mpf('1') - mp.mpf(f'{eps}')

#     def df(phi):
#         return (1/N1) * mpmath.log(phi) + (1/N1) - (1/N2) * mpmath.log(1 - phi) - (1/N2) + chi - 2 * chi * phi
    
#     dfdphi = np.append(dfdphi, float(df(eps_right)))

#     #Update phi_vals
#     phi_vals = np.append(phi_vals_,1.0)
#     phi_vals[0] = 0.0

#     print(dfdphi)

#     spline = CubicSpline(phi_vals,dfdphi)
#     def df_spline(phi):
#         return spline(phi)
    
#     return df_spline



# def fh(phi, chi, N1, N2):
#     return (phi / N1)*np.log(phi) + ((1 - phi)/N2)*np.log(1 - phi) + chi*phi*(1 - phi)



# """
# Compute RMSE and MAE
# """

# # knots_values = [10, 20, 25, 40, 50, 75, 100, 150, 200,500]

# # # Arrays to store errors
# # rmse_f = np.zeros(len(knots_values))
# # mae_f = np.zeros(len(knots_values))

# # rmse_df = np.zeros(len(knots_values))
# # mae_df = np.zeros(len(knots_values))

# # rmse_df_int = np.zeros(len(knots_values))
# # mae_df_int = np.zeros(len(knots_values))


# # for j, knots in enumerate(knots_values):
# #     # Generate the spline for chosen # of knots
# #     df_test = spline_generator2(chi=50, N1=1, N2=1, knots=knots)

# #     # Define test points for comparison
# #     phi_test = np.linspace(1e-16, 1 - 1e-16, 100)
# #     phi_test_int = np.linspace(1e-2,   1 - 1e-2,   100)  # "interior"

# #     # Evaluate the spline
# #     # f_spline_vals = f_test(phi_test)
# #     df_spline_vals = df_test(phi_test)

# #     # Evaluate the analytical solution
# #     # f_ana_vals = fh(phi_test, 50, 1, 1)
# #     df_ana_vals = fh_deriv(phi_test, 50, 1, 1)

# #     # Evaluate derivative on interior subset
# #     df_spline_int_vals = df_test(phi_test_int)
# #     df_ana_int_vals = fh_deriv(phi_test_int, 50, 1, 1)

# #     # ---- Compute errors (RMSE, MAE) for f
# #     # f
# #     # f_diff = f_spline_vals - f_ana_vals
# #     # rmse_f[j] = np.sqrt(np.mean(f_diff**2))
# #     # mae_f[j] = np.max(np.abs(f_diff))

# #     # df/dphi
# #     df_diff = df_spline_vals - df_ana_vals
# #     rmse_df[j] = np.sqrt(np.mean(df_diff**2))
# #     mae_df[j] = np.max(np.abs(df_diff))

# #     # df/dphi (interior)
# #     df_diff_int = df_spline_int_vals - df_ana_int_vals
# #     rmse_df_int[j] = np.sqrt(np.mean(df_diff_int**2))
# #     mae_df_int[j] = np.max(np.abs(df_diff_int))

# # # Create a figure and a single axes using subplots
# # fig, ax = plt.subplots(figsize=(6, 5))

# # # Plot data
# # ax.plot(knots_values, rmse_f,  label=r"RMSE $f$", color="black")
# # ax.plot(knots_values, mae_f,   label=r"MAE $f$",  color="black", linestyle="--")

# # ax.plot(knots_values, rmse_df, label=r"RMSE $\frac{\partial f}{\partial \phi}$", color="red")
# # ax.plot(knots_values, mae_df,  label=r"MAE $\frac{\partial f}{\partial \phi}$",  color="red", linestyle="--")

# # ax.plot(knots_values, rmse_df_int, label=r"RMSE (Interior) $\frac{\partial f}{\partial \phi}$", color="blue")
# # ax.plot(knots_values, mae_df_int,  label=r"MAE (Interior) $\frac{\partial f}{\partial \phi}$",  color="blue", linestyle="--")

# # # Set log scale on both axes
# # ax.set_xscale("log")
# # ax.set_yscale("log")
# # ax.set_xlabel("Number of knots (m)")
# # ax.set_ylabel("Error")
# # ax.legend(loc="lower left")
# # fig.tight_layout()
# # fig.savefig("scipy_spline_error.png",dpi=300)


# """
# Plot splines with m=100
# """

# df1 = spline_generator2(chi=50, N1=1,  N2=1,   knots=100)
# df2 = spline_generator2(chi=1,  N1=100, N2=50, knots=100)

# # Choose a set of phi values for plotting:
# phi_vals = np.linspace(1e-16, 1 - 1e-16, 1000)

# fig2, ax2 = plt.subplots(1,1, figsize=(10, 5))

# # Analytical curves
# ax2.plot(phi_vals, fh_deriv(phi_vals, 50, 1, 1),
#          label=r"Analytical, $\chi_{12}=50, x_{1} = x_{2} = 1$", lw=2, color="black",
#          marker="o", markerfacecolor="white", markevery=50)
# ax2.plot(phi_vals, fh_deriv(phi_vals, 1, 100, 50),
#          label=r"Analytical, $\chi_{12}=1, x_{1}=100, x_{2}=50$", lw=2, color="black",
#          marker="s", markerfacecolor="white", markevery=50)

# # Spline curves
# ax2.plot(phi_vals, df1(phi_vals),
#          label=r"Spline, $m=100$", color="red", linestyle=":", linewidth=2, alpha=0.7)
# ax2.plot(phi_vals, df2(phi_vals),
#          label="", color="red", linestyle=":", linewidth=2, alpha=0.7)
# ax2.set_xlabel(r"$\phi_{1}$")
# ax2.set_ylabel(r"$\frac{\partial f}{\partial \phi_{1}}$")
# ax2.legend(loc="lower left", fontsize=10)

# fig2.tight_layout()
# # fig2.savefig("scipy_spline_plot.png",dpi=300)

# plt.show()


def generate_spline(chi, N1, N2, knots):
    def dfdphi(c):
        return -2*chi*c + chi - (1/N2)*np.log(1-c) + (1/N1)*np.log(c) + (1/N1) - (1/N2)
    
    spline_pot = PchipInterpolator(np.linspace(0,1,knots),dfdphi(np.linspace(1e-16,1-1e-16,knots)))

    df_spline = spline_pot

    return df_spline


############################################
# Define a function to compute error metrics
############################################
def compute_errors(f_true, f_approx, phi_vals):
    """
    f_true: function computing the "true" value
    f_approx: spline function approximation
    phi_vals: array of test phi values
    Returns RMSE and MAE.
    """
    y_true = np.array([f_true(phi) for phi in phi_vals])
    y_approx = np.array([f_approx(phi) for phi in phi_vals])
    errors = y_true - y_approx
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))
    return rmse, mae

############################################
# Main Benchmarking and Plotting Code
############################################

# Define phi test values (for error computation and plotting)
phi_test = np.linspace(1e-16, 1 - 1e-16, 1000)

# Define the range of knot counts to test (from 20 to 1000 in steps of 20)
knots_range = np.arange(20,1000,30)

# Define two cases:
# Case 1: chi = 50, N1 = 1, N2 = 1
chi1, N1_1, N2_1 = 50, 1, 1
# Case 2: chi = 30, N1 = 1000, N2 = 1
chi2, N1_2, N2_2 = 10, 100, 1

# Initialize lists to store error metrics for each case vs. number of knots
rmse_case1 = []
mae_case1  = []
rmse_case2 = []
mae_case2  = []
knots_values = []

# Loop over the different numbers of knots
for knots in knots_range:
    knots_values.append(knots)
    # Build spline approximations for both cases
    spline1 = generate_spline(chi1, N1_1, N2_1, knots)
    spline2 = generate_spline(chi2, N1_2, N2_2, knots)
    # Compute errors for case 1
    rmse1, mae1 = compute_errors(lambda phi: fh_deriv(phi, chi1, N1_1, N2_1), spline1, phi_test)
    rmse_case1.append(rmse1)
    mae_case1.append(mae1)
    # Compute errors for case 2
    rmse2, mae2 = compute_errors(lambda phi: fh_deriv(phi, chi2, N1_2, N2_2), spline2, phi_test)
    rmse_case2.append(rmse2)
    mae_case2.append(mae2)

# Build fixed splines with a fixed number of knots for plotting function comparisons
fixed_knots = 100
spline1_fixed = generate_spline(chi1, N1_1, N2_1, fixed_knots)
spline2_fixed = generate_spline(chi2, N1_2, N2_2, fixed_knots)

# Evaluate the true function and the spline approximation over phi_test for both cases.
f_true_case1   = np.array([fh_deriv(phi, chi1, N1_1, N2_1) for phi in phi_test])
f_spline_case1 = np.array([spline1_fixed(phi) for phi in phi_test])
f_true_case2   = np.array([fh_deriv(phi, chi2, N1_2, N2_2) for phi in phi_test])
f_spline_case2 = np.array([spline2_fixed(phi) for phi in phi_test])

############################################
# Plotting
############################################

# Create two subplots side by side.
fig, (ax1, ax2) = plt.subplots(1, 2, dpi=300,figsize=(8,4))

# --- First subplot: function vs. spline approximation ---
ax1.plot(phi_test, f_true_case1, label=r"Analytical, $\chi_{12}=50, x_{1}=x_{2}=1$", lw=2)
ax1.plot(phi_test, f_spline_case1, label=r"Spline, $\chi_{12}=50, x_{1}=x_{2}=1$", lw=2, linestyle='--')
ax1.plot(phi_test, f_true_case2, label=r"Analytical, $\chi_{12}=10, x_{1}=100, x_{2}=1$", lw=2)
ax1.plot(phi_test, f_spline_case2, label=r"Spline, $\chi_{12}=10, x_{1}=100, x_{2}=1$", lw=2, linestyle='--')
ax1.set_xlabel(r"$\phi_{1}$")
ax1.set_ylabel(r"$\frac{\partial f}{\partial \phi_{1}}$")
ax1.legend(fontsize=8,loc="lower left")

# --- Second subplot: Error metrics vs. number of knots ---
ax2.plot(knots_values, rmse_case1, marker='o', label=r"RMSE, $\chi_{12}=50, , x_{1}=x_{2}=1$", lw=2)
ax2.plot(knots_values, mae_case1, marker='D', label=r"MAE, , $\chi_{12}=50, , x_{1}=x_{2}=1$", lw=2)
ax2.plot(knots_values, rmse_case2, marker='*', label=r"RMSE, $\chi_{12}=10, x_{1}=100, x_{2}=1$", lw=2)
ax2.plot(knots_values, mae_case2, marker='h', label=r"MAE, $\chi_{12}=10, x_{1}=100, x_{2}=1$", lw=2)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel(r"m")
ax2.set_ylabel("Error")
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig("./spline_error_pchip_python.png",dpi=300)