import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
from math import sqrt

#formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')

def spline_generator(chi, N1, N2, knots):
    def log_terms(phi):
        # Vectorized log terms
        return (phi/N1)*np.log(phi) + ((1 - phi)/N2)*np.log(1 - phi)
    
    def tanh_sinh_spacing(n, beta):
        # Return n points between 0 and 1 based on a tanh-sinh distribution
        # Indices go from 0 to n-1
        i = np.arange(n, dtype=float)
        return 0.5 * (1.0 + np.tanh(beta*(2.0*i/(n-1) - 1.0)))
    
    phi_vals_ = tanh_sinh_spacing(knots - 2, 14.0)

    # Evaluate the log-terms for those interior points
    f_vals_ = log_terms(phi_vals_)

    phi_vals = np.insert(phi_vals_, 0, 0.0)
    phi_vals = np.append(phi_vals, 1.0)
    f_vals = np.insert(f_vals_, 0, 0.0)
    f_vals = np.append(f_vals, 0.0)

    spline = make_interp_spline(phi_vals, f_vals, k=3)
    spline_derivative = spline.derivative()

    def f_spline(phi):
        return spline(phi) + chi*phi*(1.0 - phi)

    def df_spline(phi):
        return spline_derivative(phi) + chi*(1.0 - 2.0*phi)

    return f_spline, df_spline

def fh(phi, chi, N1, N2):
    return (phi / N1)*np.log(phi) + ((1 - phi)/N2)*np.log(1 - phi) + chi*phi*(1 - phi)

def fh_deriv(phi, chi, N1, N2):
    return (1/N1)*np.log(phi) + (1/N1) \
           - (1/N2)*np.log(1 - phi) - (1/N2) \
           + chi - 2*chi*phi

"""
Compute RMSE and MAE
"""

knots_values = [5, 10, 20, 25, 40, 50, 75, 100, 150, 200]

# Arrays to store errors
rmse_f = np.zeros(len(knots_values))
mae_f = np.zeros(len(knots_values))

rmse_df = np.zeros(len(knots_values))
mae_df = np.zeros(len(knots_values))

rmse_df_int = np.zeros(len(knots_values))
mae_df_int = np.zeros(len(knots_values))


for j, knots in enumerate(knots_values):
    # Generate the spline for chosen # of knots
    f_test, df_test = spline_generator(chi=50, N1=1, N2=1, knots=knots)

    # Define test points for comparison
    phi_test = np.linspace(1e-16, 1 - 1e-16, 100)
    phi_test_int = np.linspace(1e-2,   1 - 1e-2,   100)  # "interior"

    # Evaluate the spline
    f_spline_vals = f_test(phi_test)
    df_spline_vals = df_test(phi_test)

    # Evaluate the analytical solution
    f_ana_vals = fh(phi_test, 50, 1, 1)
    df_ana_vals = fh_deriv(phi_test, 50, 1, 1)

    # Evaluate derivative on interior subset
    df_spline_int_vals = df_test(phi_test_int)
    df_ana_int_vals = fh_deriv(phi_test_int, 50, 1, 1)

    # ---- Compute errors (RMSE, MAE) for f
    # f
    f_diff = f_spline_vals - f_ana_vals
    rmse_f[j] = np.sqrt(np.mean(f_diff**2))
    mae_f[j] = np.max(np.abs(f_diff))

    # df/dphi
    df_diff = df_spline_vals - df_ana_vals
    rmse_df[j] = np.sqrt(np.mean(df_diff**2))
    mae_df[j] = np.max(np.abs(df_diff))

    # df/dphi (interior)
    df_diff_int = df_spline_int_vals - df_ana_int_vals
    rmse_df_int[j] = np.sqrt(np.mean(df_diff_int**2))
    mae_df_int[j] = np.max(np.abs(df_diff_int))

# Create a figure and a single axes using subplots
fig, ax = plt.subplots(figsize=(6, 5))

# Plot data
ax.plot(knots_values, rmse_f,  label=r"RMSE $f$", color="black")
ax.plot(knots_values, mae_f,   label=r"MAE $f$",  color="black", linestyle="--")

ax.plot(knots_values, rmse_df, label=r"RMSE $\frac{\partial f}{\partial \phi}$", color="red")
ax.plot(knots_values, mae_df,  label=r"MAE $\frac{\partial f}{\partial \phi}$",  color="red", linestyle="--")

ax.plot(knots_values, rmse_df_int, label=r"RMSE (Interior) $\frac{\partial f}{\partial \phi}$", color="blue")
ax.plot(knots_values, mae_df_int,  label=r"MAE (Interior) $\frac{\partial f}{\partial \phi}$",  color="blue", linestyle="--")

# Set log scale on both axes
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Number of knots (m)")
ax.set_ylabel("Error")
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig("scipy_spline_error.png",dpi=300)


"""
Plot splines with m=100
"""

f1, df1 = spline_generator(chi=50, N1=1,  N2=1,   knots=100)
f2, df2 = spline_generator(chi=1,  N1=100, N2=50, knots=100)

# Choose a set of phi values for plotting:
phi_vals = np.linspace(1e-16, 1 - 1e-16, 1000)

fig2, ax2 = plt.subplots(1, 2, figsize=(10, 5))

ax2[0].plot(phi_vals, fh(phi_vals, 50, 1, 1),
         label="", lw=2, color="black",
         marker="o", markerfacecolor="white", markevery=50)
ax2[0].plot(phi_vals, fh(phi_vals, 1, 100, 50),
         label="", lw=2, color="black",
         marker="s", markerfacecolor="white", markevery=50)

# Spline curves
ax2[0].plot(phi_vals, f1(phi_vals),
         label="", color="red", linestyle=":", linewidth=2, alpha=0.7)
ax2[0].plot(phi_vals, f2(phi_vals),
         label="", color="red", linestyle=":", linewidth=2, alpha=0.7)

ax2[0].set_xlabel(r"$\phi_{1}$")
ax2[0].set_ylabel(r"$f$")

# Analytical curves
ax2[1].plot(phi_vals, fh_deriv(phi_vals, 50, 1, 1),
         label=r"Analytical, $\chi_{12}=50, x_{1} = x_{2} = 1$", lw=2, color="black",
         marker="o", markerfacecolor="white", markevery=50)
ax2[1].plot(phi_vals, fh_deriv(phi_vals, 1, 100, 50),
         label=r"Analytical, $\chi_{12}=1, x_{1}=100, x_{2}=50$", lw=2, color="black",
         marker="s", markerfacecolor="white", markevery=50)

# Spline curves
ax2[1].plot(phi_vals, df1(phi_vals),
         label=r"Spline, $m=100$", color="red", linestyle=":", linewidth=2, alpha=0.7)
ax2[1].plot(phi_vals, df2(phi_vals),
         label="", color="red", linestyle=":", linewidth=2, alpha=0.7)
ax2[1].set_xlabel(r"$\phi_{1}$")
ax2[1].set_ylabel(r"$\frac{\partial f}{\partial \phi_{1}}$")
ax2[1].legend(loc="lower left", fontsize=10)

fig2.tight_layout()
fig2.savefig("scipy_spline_plot.png",dpi=300)

plt.show()