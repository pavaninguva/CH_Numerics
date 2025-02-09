import numpy as np
from scipy.interpolate import make_interp_spline, CubicSpline
import matplotlib.pyplot as plt
from math import sqrt
from mpmath import mp
import mpmath as mpmath

# Set high precision
mp.dps = 50 

#formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')

def fh_deriv(phi, chi, N1, N2):
    return (1/N1)*np.log(phi) + (1/N1) \
           - (1/N2)*np.log(1 - phi) - (1/N2) \
           + chi - 2*chi*phi


def spline_generator2(chi, N1, N2, knots):
    #Define small eps
    eps = 1e-40
    
    def tanh_sinh_spacing(n, beta):
        # Return n points between 0 and 1 based on a tanh-sinh distribution
        # Indices go from 0 to n-1
        i = np.arange(n, dtype=float)
        return 0.5 * (1.0 + np.tanh(beta*(2.0*i/(n-1) - 1.0)))
    
    phi_vals_ = tanh_sinh_spacing(knots - 4, 14.0)
    #Insert eps
    phi_vals_ = np.insert(phi_vals_,0,1e-16)
    phi_vals_ = np.insert(phi_vals_,0,eps)

    phi_vals_ = np.append(phi_vals_,1.0-1e-16)

    #Compute dfdphi vals
    dfdphi = fh_deriv(phi_vals_,chi,N1,N2)

    #compute eps_right
    eps_right = mp.mpf('1') - mp.mpf(f'{eps}')

    def df(phi):
        return (1/N1) * mpmath.log(phi) + (1/N1) - (1/N2) * mpmath.log(1 - phi) - (1/N2) + chi - 2 * chi * phi
    
    dfdphi = np.append(dfdphi, float(df(eps_right)))

    #Update phi_vals
    phi_vals = np.append(phi_vals_,1.0)
    phi_vals[0] = 0.0

    print(dfdphi)

    spline = CubicSpline(phi_vals,dfdphi)
    def df_spline(phi):
        return spline(phi)
    
    return df_spline



def fh(phi, chi, N1, N2):
    return (phi / N1)*np.log(phi) + ((1 - phi)/N2)*np.log(1 - phi) + chi*phi*(1 - phi)



"""
Compute RMSE and MAE
"""

# knots_values = [10, 20, 25, 40, 50, 75, 100, 150, 200,500]

# # Arrays to store errors
# rmse_f = np.zeros(len(knots_values))
# mae_f = np.zeros(len(knots_values))

# rmse_df = np.zeros(len(knots_values))
# mae_df = np.zeros(len(knots_values))

# rmse_df_int = np.zeros(len(knots_values))
# mae_df_int = np.zeros(len(knots_values))


# for j, knots in enumerate(knots_values):
#     # Generate the spline for chosen # of knots
#     df_test = spline_generator2(chi=50, N1=1, N2=1, knots=knots)

#     # Define test points for comparison
#     phi_test = np.linspace(1e-16, 1 - 1e-16, 100)
#     phi_test_int = np.linspace(1e-2,   1 - 1e-2,   100)  # "interior"

#     # Evaluate the spline
#     # f_spline_vals = f_test(phi_test)
#     df_spline_vals = df_test(phi_test)

#     # Evaluate the analytical solution
#     # f_ana_vals = fh(phi_test, 50, 1, 1)
#     df_ana_vals = fh_deriv(phi_test, 50, 1, 1)

#     # Evaluate derivative on interior subset
#     df_spline_int_vals = df_test(phi_test_int)
#     df_ana_int_vals = fh_deriv(phi_test_int, 50, 1, 1)

#     # ---- Compute errors (RMSE, MAE) for f
#     # f
#     # f_diff = f_spline_vals - f_ana_vals
#     # rmse_f[j] = np.sqrt(np.mean(f_diff**2))
#     # mae_f[j] = np.max(np.abs(f_diff))

#     # df/dphi
#     df_diff = df_spline_vals - df_ana_vals
#     rmse_df[j] = np.sqrt(np.mean(df_diff**2))
#     mae_df[j] = np.max(np.abs(df_diff))

#     # df/dphi (interior)
#     df_diff_int = df_spline_int_vals - df_ana_int_vals
#     rmse_df_int[j] = np.sqrt(np.mean(df_diff_int**2))
#     mae_df_int[j] = np.max(np.abs(df_diff_int))

# # Create a figure and a single axes using subplots
# fig, ax = plt.subplots(figsize=(6, 5))

# # Plot data
# ax.plot(knots_values, rmse_f,  label=r"RMSE $f$", color="black")
# ax.plot(knots_values, mae_f,   label=r"MAE $f$",  color="black", linestyle="--")

# ax.plot(knots_values, rmse_df, label=r"RMSE $\frac{\partial f}{\partial \phi}$", color="red")
# ax.plot(knots_values, mae_df,  label=r"MAE $\frac{\partial f}{\partial \phi}$",  color="red", linestyle="--")

# ax.plot(knots_values, rmse_df_int, label=r"RMSE (Interior) $\frac{\partial f}{\partial \phi}$", color="blue")
# ax.plot(knots_values, mae_df_int,  label=r"MAE (Interior) $\frac{\partial f}{\partial \phi}$",  color="blue", linestyle="--")

# # Set log scale on both axes
# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_xlabel("Number of knots (m)")
# ax.set_ylabel("Error")
# ax.legend(loc="lower left")
# fig.tight_layout()
# fig.savefig("scipy_spline_error.png",dpi=300)


"""
Plot splines with m=100
"""

df1 = spline_generator2(chi=50, N1=1,  N2=1,   knots=100)
df2 = spline_generator2(chi=1,  N1=100, N2=50, knots=100)

# Choose a set of phi values for plotting:
phi_vals = np.linspace(1e-16, 1 - 1e-16, 1000)

fig2, ax2 = plt.subplots(1,1, figsize=(10, 5))

# Analytical curves
ax2.plot(phi_vals, fh_deriv(phi_vals, 50, 1, 1),
         label=r"Analytical, $\chi_{12}=50, x_{1} = x_{2} = 1$", lw=2, color="black",
         marker="o", markerfacecolor="white", markevery=50)
ax2.plot(phi_vals, fh_deriv(phi_vals, 1, 100, 50),
         label=r"Analytical, $\chi_{12}=1, x_{1}=100, x_{2}=50$", lw=2, color="black",
         marker="s", markerfacecolor="white", markevery=50)

# Spline curves
ax2.plot(phi_vals, df1(phi_vals),
         label=r"Spline, $m=100$", color="red", linestyle=":", linewidth=2, alpha=0.7)
ax2.plot(phi_vals, df2(phi_vals),
         label="", color="red", linestyle=":", linewidth=2, alpha=0.7)
ax2.set_xlabel(r"$\phi_{1}$")
ax2.set_ylabel(r"$\frac{\partial f}{\partial \phi_{1}}$")
ax2.legend(loc="lower left", fontsize=10)

fig2.tight_layout()
# fig2.savefig("scipy_spline_plot.png",dpi=300)

plt.show()