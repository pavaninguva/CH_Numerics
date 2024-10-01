import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
import copy

plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')
plt.rc("font", size=14)

def binodal(phi, chi, N1, N2):
    def fh_deriv(phi, chi, N1, N2):
        df = (1/N1)*np.log(phi) + (1/N1) - (1/N2)*np.log(1-phi) - (1/N2) -2*chi*phi + chi
        return df

    def osmotic(phi, chi, N1, N2):
        osmo = phi*((1/N1)-(1/N2)) - (1/N2)*np.log(1-phi) - chi*phi**2
        return osmo

    phiA = phi[0]
    phiB = phi[1]
    dF = fh_deriv(phiA, chi, N1, N2) - fh_deriv(phiB, chi, N1, N2)
    dO = osmotic(phiA, chi, N1, N2) - osmotic(phiB, chi, N1, N2)
    return np.array([dF, dO])


def binodal_solver(N1, N2, chi_scale_vals):
    phic = np.sqrt(N2)/(np.sqrt(N1)+np.sqrt(N2))
    chic = (np.sqrt(N1)+np.sqrt(N2))**2/(2*N1*N2)

    phi1bn = np.zeros(len(chi_scale_vals))
    phi2bn = np.zeros(len(chi_scale_vals))

    x0 = [1e-3, 1-1e-3]

    for i in range(len(chi_scale_vals)):
        vals = root(lambda x: binodal(x, chi_scale_vals[i]*chic, N1, N2), x0)
        phi1bn[i] = vals.x[0]
        phi2bn[i] = vals.x[1]
        print("beep")
        x0 = copy.copy(vals.x)

    return [phic, chic], phi1bn, phi2bn



chi_vals = np.linspace(5.0, 1.0, 10000)
critvals, phi1_, phi2_ = binodal_solver(2, 1, chi_vals)

fig1, ax1 = plt.subplots(1, 1, figsize=(6, 6))

ax1.scatter(phi1_, chi_vals**-1, color="blue", s=1, label='Binodal')
ax1.scatter(phi2_, chi_vals**-1, color="blue", s=1)
ax1.scatter(critvals[0], 1.0, color="black", s=50, label='Critical Point')
ax1.set_xlim(0, 1)
ax1.set_ylim(0.0, 1.1)
ax1.set_xlabel(r"$\phi$")
ax1.set_ylabel(r"$\frac{\chi_c}{\chi}$")
fig1.tight_layout()

plt.show()