import numpy as np
from spline_funcs import *
import matplotlib.pyplot as plt
from spline_funcs import generate_spline_dfdphi1, generate_spline_dfdphi2, generate_spline_dfdphi3


#formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')

    

"""
Define Functions
"""

def dfdphi1(phi1,phi2,chi12,chi13,N1):
    return (1/N1)*np.log(phi1) + (1/N1) + chi12*phi2 + chi13*(1-phi1-phi2)

def dfdphi2(phi1,phi2, chi12, chi23, N2):
    return (1/N2)*np.log(phi2) + (1/N2) + chi12*phi1 + chi23*(1-phi1-phi2)

def compute_errors_2d(f_true, spline, eval_pts1, eval_pts2):
    """Compute RMSE and MAE between f_true and spline on a grid."""
    P1, P2 = np.meshgrid(eval_pts1, eval_pts2, indexing='ij')
    approx_vals = spline(P1,P2,grid=False)
    true_vals   = f_true(P1, P2)
    err = true_vals - approx_vals
    rmse = np.sqrt(np.mean(err**2))
    mae  = np.max(np.abs(err))
    return rmse, mae

def compute_errors_2d_dfdc(knots,chi13,chi23,N3):

    #generate test domain
    phi1_test = np.linspace(1e-16, 1-1e-16, 100)
    phi2_test = np.linspace(1e-16, 1-1e-16, 100)
    P1, P2 = np.meshgrid(phi1_test, phi2_test, indexing='ij')

    mask = (P1 + P2 < 1)
    phi1_int = P1[mask]
    phi2_int = P2[mask]

    # phi1_b = np.linspace(1e-15, 1-1e-15,100)
    # phi2_b = 1 -phi1_b

    # 3) Combine interior + boundary domains
    phi1_all = phi1_int
    phi2_all = phi2_int
    # phi1_all = np.concatenate([phi1_int, phi1_b])
    # phi2_all = np.concatenate([phi2_int, phi2_b])
    phi3_all = np.clip((1-phi1_all-phi2_all),1e-16,None)

    dfdc_ref = ((1/N3) * np.log(phi3_all)
            + (1/N3)
            + chi13 * phi1_all
            + chi23 * phi2_all
            )
    
    
    #Generate spline
    dfdc_spline, spline_knotvals = generate_spline_dfdphi3(chi13,chi23,N3,knots,True)

    spline_vals = dfdc_spline(phi1_all,phi2_all)

    #Compute error and outputs
    err = dfdc_ref - spline_vals
    print(np.isnan(spline_vals).any())
    rmse = np.sqrt(np.mean(err**2))
    mae  = np.max(np.abs(err))
    print(rmse,mae)

    return rmse, mae, spline_knotvals




#Case 1: chi12 = chi13 = chi23 = 30, N1 = N2 = N3 = 1
chi = 30
N1 = N2 = N3 = 1

rmse_case1_dfdphi1 = []
mae_case1_dfdph1 = []
rmse_case1_dfdphi2 = []
mae_case1_dfdph2 = []
rmse_case1_dfdphi3 = []
mae_case1_dfdphi3 = []

knot_vals = []
dfdphi3_knot_vals = []

phi_test = np.linspace(1e-16, 1-1e-16, 100)
knots_range = np.arange(50,1200,100)

for knot in knots_range:
    print(knot)
    knot_vals.append(knot)

    #Generate spline 1
    spline2d_dfdphi1 = generate_spline_dfdphi1(chi,chi,N1,knot)
    print("spline dfdphi1 generated")
    rmse, mae = compute_errors_2d(
        lambda u,v: (1/N1)*np.log(u) + (1/N1) + chi*v + chi*(1-u-v),
        spline2d_dfdphi1,
        phi_test, phi_test
    )

    rmse_case1_dfdphi1.append(rmse)
    mae_case1_dfdph1.append(mae)

    #Generate spline 2
    spline2d_dfdphi2 = generate_spline_dfdphi2(chi,chi,N2,knot)
    print("spline dfdphi2 generated")
    rmse2, mae2 = compute_errors_2d(
        lambda u,v: (1/N2)*np.log(v) + (1/N2) + chi*u + chi*(1-u-v),
        spline2d_dfdphi2,
        phi_test, phi_test
    )

    rmse_case1_dfdphi2.append(rmse2)
    mae_case1_dfdph2.append(mae2)

    #Spline 3
    print("spline dfdphi3 generated")
    rmse3, mae3, s3_knotvals = compute_errors_2d_dfdc(knot,chi,chi,N3)
    rmse_case1_dfdphi3.append(rmse3)
    mae_case1_dfdphi3.append(mae3)
    dfdphi3_knot_vals.append(s3_knotvals)



fig, ax1 = plt.subplots(1,3,dpi=300,figsize=(10,4))
ax1[0].plot([k**2 for k in knot_vals], rmse_case1_dfdphi1, marker="o", label=r"RMSE", lw=2)
ax1[0].plot([k**2 for k in knot_vals], mae_case1_dfdph1, marker="D", label=r"MAE", lw=2)
ax1[0].set_xscale('log')
ax1[0].set_yscale('log')
ax1[0].set_xlabel(r"m")
ax1[0].set_ylabel("Error")
ax1[0].legend(fontsize=8)
ax1[0].set_title(r"$\frac{\partial f}{\partial \phi_{1}}$")

ax1[1].plot([k**2 for k in knot_vals], rmse_case1_dfdphi2, marker="o", lw=2)
ax1[1].plot([k**2 for k in knot_vals], mae_case1_dfdph2, marker="D", lw=2)
ax1[1].set_xscale('log')
ax1[1].set_yscale('log')
ax1[1].set_xlabel(r"m")
ax1[1].set_ylabel("Error")
ax1[1].set_title(r"$\frac{\partial f}{\partial \phi_{2}}$")

ax1[2].plot(dfdphi3_knot_vals, rmse_case1_dfdphi3, marker="o", lw=2)
ax1[2].plot(dfdphi3_knot_vals, mae_case1_dfdphi3, marker="D", lw=2)
ax1[2].set_xscale('log')
ax1[2].set_yscale('log')
ax1[2].set_xlabel(r"m")
ax1[2].set_ylabel("Error")
ax1[2].set_title(r"$\frac{\partial f}{\partial \phi_{3}}$")



fig.tight_layout()
fig.savefig("ternary_spline_error.png")

plt.show()


