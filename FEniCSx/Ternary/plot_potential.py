import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from spline_funcs import generate_spline_dfdphi1, generate_spline_dfdphi2, generate_spline_dfdphi3

#formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')

# Parameters (adjust as needed)
chi12, chi13, chi23 = 30.0, 30.0, 30.0
N1, N2, N3 = 1.0, 1.0, 1.0

# Domain setup (exclude exact 0 or 1 to avoid log singularities)
domain_min, domain_max = 1e-16, 1 - 1e-16
n = 500
phi1 = np.linspace(domain_min, domain_max, n)
phi2 = np.linspace(domain_min, domain_max, n)
P1, P2 = np.meshgrid(phi1, phi2, indexing='ij')

# Compute partial derivatives
dfda = (1/N1) * np.log(P1) + (1/N1) + chi12 * P2 + chi13 * (1 - P1 - P2)
dfdb = (1/N2) * np.log(P2) + (1/N2) + chi12 * P1 + chi23 * (1 - P1 - P2)

dfda_spline = generate_spline_dfdphi1(chi12,chi13,N1,400)
dfda_spline_vals = dfda_spline(P1,P2)

dfdb_spline = generate_spline_dfdphi2(chi12,chi23,N2,400)
dfdb_spline_vals = dfdb_spline(P1,P2)


#Sort out dfdc
mask = (P1 + P2 < 1)
phi1_int = P1[mask]
phi2_int = P2[mask]

dfdc_int = (
    (1/N3) * np.log(1 - phi1_int - phi2_int)
  + (1/N3)
  + chi13 * phi1_int
  + chi23 * phi2_int
)
eps = 1e-16
n_boundary=400
phi1_b = np.linspace(eps, 1 - eps, n_boundary)
phi2_b = (1 - eps) - phi1_b   # ensures phi1+phi2 = 1-eps

# now phi3_b = eps exactly, so log never diverges
phi3_b = 1 - phi1_b - phi2_b
phi3_b = np.clip(phi3_b, eps, None)

dfdc_b = (
    (1/N3) * np.log(phi3_b)
  + (1/N3)
  + chi13 * phi1_b
  + chi23 * phi2_b
)

# 3) Combine interior + boundary domains
phi1_all = np.concatenate([phi1_int, phi1_b])
phi2_all = np.concatenate([phi2_int, phi2_b])
dfdc_all = np.concatenate([dfdc_int, dfdc_b])

#Compute spline for dfdphi3
dfdphi3_spline = generate_spline_dfdphi3(chi13,chi23,N3,400)
dfdc_spline_vals = dfdphi3_spline(phi1_all,phi2_all)

# 4) Triangulate and plot
triang = tri.Triangulation(phi1_all, phi2_all)

# Plotting
fig, axs = plt.subplots(2, 3, figsize=(12, 7), dpi=300)

# df/da heatmap
pc1 = axs[0,0].pcolormesh(P1, P2, dfda, shading='auto')
# axs[0].set_title(r'$\partial f/\partial \phi_{1}$')
axs[0,0].set_xlabel(r'$\phi_{1}$')
axs[0,0].set_ylabel(r'$\phi_{2}$')
axs[0,0].set_ylim(-0.05,1.05)
axs[0,0].set_xlim(-0.05,1.05)
fig.colorbar(pc1, ax=axs[0,0], label=r'$\frac{\partial f}{\partial \phi_{1}}$')

# df/db heatmap
pc2 = axs[0,1].pcolormesh(P1, P2, dfdb, shading='auto')
axs[0,1].set_title(r'$\textrm{Analytical}$')
axs[0,1].set_xlabel(r'$\phi_{1}$')
axs[0,1].set_ylabel(r'$\phi_{2}$')
axs[0,1].set_ylim(-0.05,1.05)
axs[0,1].set_xlim(-0.05,1.05)
fig.colorbar(pc2, ax=axs[0,1], label=r'$\frac{\partial f}{\partial \phi_{2}}$')

# df/dc heatmap on simplex
pc3 = axs[0,2].tripcolor(triang, dfdc_all, shading='gouraud',cmap="viridis")
# axs[2].set_title(r'$\partial f/\partial \phi_{3}$')
axs[0,2].set_xlabel(r'$\phi_{1}$')
axs[0,2].set_ylabel(r'$\phi_{2}$')
fig.colorbar(pc3, ax=axs[0,2], label=r'$\frac{\partial f}{\partial \phi_{3}}$')


#Plot splines
pc4 = axs[1,0].pcolormesh(P1, P2, dfda_spline_vals, shading='auto')
axs[1,0].set_xlabel(r'$\phi_{1}$')
axs[1,0].set_ylabel(r'$\phi_{2}$')
axs[1,0].set_ylim(-0.05,1.05)
axs[1,0].set_xlim(-0.05,1.05)
fig.colorbar(pc4, ax=axs[1,0], label=r'$\frac{\partial f}{\partial \phi_{1}}$')

pc5 = axs[1,1].pcolormesh(P1, P2, dfdb_spline_vals, shading='auto')
axs[1,1].set_title(r'$\textrm{Spline}$')
axs[1,1].set_xlabel(r'$\phi_{1}$')
axs[1,1].set_ylabel(r'$\phi_{2}$')
axs[1,1].set_ylim(-0.05,1.05)
axs[1,1].set_xlim(-0.05,1.05)
fig.colorbar(pc5, ax=axs[1,1], label=r'$\frac{\partial f}{\partial \phi_{2}}$')

# df/dc heatmap on simplex
pc6 = axs[1,2].tripcolor(triang, dfdc_spline_vals, shading='gouraud',cmap="viridis")
axs[1,2].set_xlabel(r'$\phi_{1}$')
axs[1,2].set_ylabel(r'$\phi_{2}$')
fig.colorbar(pc6, ax=axs[1,2], label=r'$\frac{\partial f}{\partial \phi_{3}}$')
plt.tight_layout()

fig.savefig("exemplar_ternary_pot_plot.png")


plt.show()