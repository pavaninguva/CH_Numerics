import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import root


"""
This code implements a naive BDF-2 scheme for timestepping. 

Running it shows that it is numerically unstable
"""

# Parameters
N1 = 1
N2 = 1
chi = 5
kappa = (2/3)*chi

def dfdphi(phi):
    return -2*chi*phi + chi - (1/N2)*np.log(1-phi) + (1/N1)*np.log(phi)

def mobility(phi, option=1):
    if option == 1:
        return phi * (1 - phi)
    elif option == 2:
        return 1
    else:
        raise ValueError("Invalid mobility option")

def M_func_half(phi, phi_, option="2"):
    if option == "1":
        return 0.5 * (mobility(phi) + mobility(phi_))
    elif option == "2":
        return mobility(0.5 * (phi + phi_))
    elif option == "3":
        return (2 * mobility(phi) * mobility(phi_)) / (mobility(phi) + mobility(phi_))
    else:
        raise ValueError("Specified option for mobility interpolation not available")

def backward_euler_fd_res(phi_new, phi_old):
    # Define chemical potential mu_new
    mu_new = np.zeros_like(phi_old)
    
    mu_new[0] = dfdphi(phi_new[0]) - (2 * kappa / (dx**2)) * (phi_new[1] - phi_new[0])
    mu_new[-1] = dfdphi(phi_new[-1]) - (2 * kappa / (dx**2)) * (phi_new[-2] - phi_new[-1])
    mu_new[1:-1] = dfdphi(phi_new[1:-1]) - (kappa / (dx**2)) * (phi_new[2:] - 2 * phi_new[1:-1] + phi_new[:-2])

    # Define residual for BE step
    res = np.zeros_like(phi_old)
    res[0] = (phi_new[0] - phi_old[0]) / dt - (2 / (dx**2)) * (M_func_half(phi_new[0], phi_new[1])) * (mu_new[1] - mu_new[0])
    res[-1] = (phi_new[-1] - phi_old[-1]) / dt - (2 / (dx**2)) * (M_func_half(phi_new[-1], phi_new[-2])) * (mu_new[-2] - mu_new[-1])
    res[1:-1] = (phi_new[1:-1] - phi_old[1:-1]) / dt - (1 / (dx**2)) * (
        M_func_half(phi_new[1:-1], phi_new[2:]) * (mu_new[2:] - mu_new[1:-1]) -
        M_func_half(phi_new[1:-1], phi_new[:-2]) * (mu_new[1:-1] - mu_new[:-2])
    )
    return res

def bdf2_fd_res(phi_new, phi_old, phi_older):
    # Define chemical potential mu_new
    mu_new = np.zeros_like(phi_old)
    
    mu_new[0] = dfdphi(phi_new[0]) - (2 * kappa / (dx**2)) * (phi_new[1] - phi_new[0])
    mu_new[-1] = dfdphi(phi_new[-1]) - (2 * kappa / (dx**2)) * (phi_new[-2] - phi_new[-1])
    mu_new[1:-1] = dfdphi(phi_new[1:-1]) - (kappa / (dx**2)) * (phi_new[2:] - 2 * phi_new[1:-1] + phi_new[:-2])

    # Define residual for BDF-2 step
    res = np.zeros_like(phi_old)
    res[0] = (3 * phi_new[0] - 4 * phi_old[0] + phi_older[0]) / (2 * dt) - (2 / (dx**2)) * (M_func_half(phi_new[0], phi_new[1])) * (mu_new[1] - mu_new[0])
    res[-1] = (3 * phi_new[-1] - 4 * phi_old[-1] + phi_older[-1]) / (2 * dt) - (2 / (dx**2)) * (M_func_half(phi_new[-1], phi_new[-2])) * (mu_new[-2] - mu_new[-1])
    res[1:-1] = (3 * phi_new[1:-1] - 4 * phi_old[1:-1] + phi_older[1:-1]) / (2 * dt) - (1 / (dx**2)) * (
        M_func_half(phi_new[1:-1], phi_new[2:]) * (mu_new[2:] - mu_new[1:-1]) -
        M_func_half(phi_new[1:-1], phi_new[:-2]) * (mu_new[1:-1] - mu_new[:-2])
    )
    return res

# Simulation params
L = 10.0
nx = 101
dx = L / (nx - 1)
xvals = np.linspace(0, L, nx)

# Time-stepping parameters
tf = 5.0
nsteps = 2000
dt = tf / nsteps

# Initial conditions
phi0 = 0.5 + 0.05 * np.random.rand(np.size(xvals))
phi = phi0.copy()
phi_old = None
phi_older = None

# Animation setup
fig, ax = plt.subplots()
line, = ax.plot(xvals, phi, label=r"$\phi$")
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ax.set_xlim(0.0, L)
ax.set_ylim(-0.2, 1.2)
ax.set_xlabel("x")
ax.set_ylabel(r"$\phi$")
ax.hlines(y=0.0, xmin=0, xmax=L, color="r")
ax.hlines(y=1.0, xmin=0, xmax=L, color="r")
ax.legend()

# Main time-stepping loop
def update(n):
    global phi, phi_old, phi_older

    if n == 0:
        # First time step: Backward Euler
        phi_older = phi.copy()  # For BDF-2, store initial condition
        phi_old = phi.copy()
        def wrapped_residual(phi_new):
            return backward_euler_fd_res(phi_new, phi_old)
    else:
        # Subsequent time steps: BDF-2
        def wrapped_residual(phi_new):
            return bdf2_fd_res(phi_new, phi_old, phi_older)
    
    sol = root(wrapped_residual, phi_old, method="hybr", tol=1e-10)
    phi_new = sol.x
    
    # Update time step storage
    phi_older = phi_old.copy()
    phi_old = phi.copy()
    phi = phi_new.copy()

    line.set_ydata(phi)
    time_text.set_text(f'Time = {n * dt:.2f}')
    return line, time_text

ani = animation.FuncAnimation(fig, update, frames=nsteps, blit=True, repeat=False)
plt.show()