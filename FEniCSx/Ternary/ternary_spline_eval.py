import numpy as np
from spline_funcs import *
import matplotlib.pyplot as plt


#formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')

    

"""
Define Functions
"""

def dfdphi1(phi1,phi2,chi12,chi13,N1):
    return (1/N1)*np.log(phi1) + (1/N1) + chi12*phi2 + chi13*(1-phi1-phi2)

def generate_spline(chi12, chi13, N1, knots):
    """
    chi12, chi13: interaction parameters
    N1          : parameter in derivative
    knots       : number of uniform knots per dimension
    returns     : callable spline phi1,phi2 -> dfdphi1(phi1,phi2)
    """
    domain_min, domain_max = 1e-16, 1 - 1e-16

    # the "true" 2D derivative function
    def dfdphi1(phi1, phi2):
        return (1/N1)*np.log(phi1) + (1/N1) \
               + chi12*phi2 \
               + chi13*(1 - phi1 - phi2)

    # build uniform grid of knots
    phi1_knots = np.linspace(domain_min, domain_max, knots)
    phi2_knots = np.linspace(domain_min, domain_max, knots)
    P1, P2 = np.meshgrid(phi1_knots, phi2_knots, indexing='ij')
    Z = dfdphi1(P1, P2)

    # build and return the tensor‐product PCHIP
    spline2d = Pchip2D(phi1_knots, phi2_knots, Z)
    return spline2d

def compute_errors_2d(f_true, spline, eval_pts1, eval_pts2):
    """Compute RMSE and MAE between f_true and spline on a grid."""
    P1, P2 = np.meshgrid(eval_pts1, eval_pts2, indexing='ij')
    approx_vals = np.empty_like(P1)
    true_vals   = f_true(P1, P2)
    approx_vals
    for i in range(P1.shape[0]):
        for j in range(P1.shape[1]):
            approx_vals[i,j] = spline(P1[i,j],P2[i,j])
    err = true_vals - approx_vals
    rmse = np.sqrt(np.mean(err**2))
    mae  = np.max(np.abs(err))
    return rmse, mae



#Case 1: chi12 = chi13 = chi23 = 30, N1 = N2 = N3 = 1
chi = 30
N1 = N2 = N3 = 1

rmse_case1 = []
mae_case1 = []
knot_vals = []

phi_test = np.linspace(1e-16, 1-1e-16, 1000)
knots_range = np.arange(30,1000,30)

for knot in knots_range:
    print(knot)
    knot_vals.append(knot)

    #Generate spline
    spline2d = generate_spline(chi,chi,N1,knot)
    print("spline generated")
    rmse, mae = compute_errors_2d(
        lambda u,v: (1/N1)*np.log(u) + (1/N1) + chi*v + chi*(1-u-v),
        spline2d,
        phi_test, phi_test
    )

    rmse_case1.append(rmse)
    mae_case1.append(mae)


fig, ax1 = plt.subplots(1,1,dpi=300,figsize=(5,4))
ax1.plot(knot_vals, rmse_case1, marker="o", label=r"RMSE, $\chi_{12}=\chi_{13}=\chi_{23}= 30, , x_{1}=x_{2}=x_{3}=1$", lw=2)
ax1.plot(knot_vals, mae_case1, marker="D", label=r"MAE, $\chi_{12}=\chi_{13}=\chi_{23}= 30, , x_{1}=x_{2}=x_{3}=1$", lw=2)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel(r"m")
ax1.set_ylabel("Error")
ax1.legend(fontsize=8)

fig.tight_layout()
fig.savefig("ternary_spline_error.png")

plt.show()


