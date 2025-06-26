import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

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

print(phi3_b)
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

# 4) Triangulate and plot
triang = tri.Triangulation(phi1_all, phi2_all)

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

# df/da heatmap
pc1 = axs[0].pcolormesh(P1, P2, dfda, shading='auto')
axs[0].set_title(r'$\partial f/\partial \phi_{1}$')
axs[0].set_xlabel(r'$\phi_{1}$')
axs[0].set_ylabel(r'$\phi_{2}$')
axs[0].set_ylim(-0.05,1.05)
axs[0].set_xlim(-0.05,1.05)
fig.colorbar(pc1, ax=axs[0], label=r'$\partial f/\partial \phi_{1}$')

# df/db heatmap
pc2 = axs[1].pcolormesh(P1, P2, dfdb, shading='auto')
axs[1].set_title(r'$\partial f/\partial \phi_{2}$')
axs[1].set_xlabel(r'$\phi_{1}$')
axs[1].set_ylabel(r'$\phi_{2}$')
axs[1].set_ylim(-0.05,1.05)
axs[1].set_xlim(-0.05,1.05)
fig.colorbar(pc2, ax=axs[1], label=r'$\partial f/\partial \phi_{2}$')

# df/dc heatmap on simplex
pc3 = axs[2].tripcolor(triang, dfdc_all, shading='gouraud',cmap="viridis")
axs[2].set_title(r'$\partial f/\partial \phi_{3}$')
axs[2].set_xlabel(r'$\phi_{1}$')
axs[2].set_ylabel(r'$\phi_{2}$')
fig.colorbar(pc3, ax=axs[2], label=r'$\partial f/\partial \phi_{3}$')

plt.tight_layout()
plt.show()