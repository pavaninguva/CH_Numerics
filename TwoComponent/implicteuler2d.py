import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import root

# Parameters
N1 = 1
N2 = 1
chi = 5
kappa = (2/3)*chi

def dfdphi(phi):
    return -2 * chi * phi + chi - (1 / N2) * np.log(1 - phi) + (1 / N1) * np.log(phi)

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

def apply_boundary_conditions(phi):
    """Apply ghost node boundary conditions (zero flux)"""
    # Left and right boundaries (Neumann BCs)
    phi[:, 0] = phi[:, 1]  # Left boundary
    phi[:, -1] = phi[:, -2]  # Right boundary
    
    # Top and bottom boundaries (Neumann BCs)
    phi[0, :] = phi[1, :]  # Top boundary
    phi[-1, :] = phi[-2, :]  # Bottom boundary

def backward_euler_fd_res(phi_new, phi_old):
    # Apply boundary conditions to phi_new
    apply_boundary_conditions(phi_new)
    
    # Define chemical potential mu_new (with ghost nodes)
    mu_new = np.zeros_like(phi_old)
    
    # Laplacian in 2D with central differences
    mu_new[1:-1, 1:-1] = (
        dfdphi(phi_new[1:-1, 1:-1]) - 
        (kappa / (dx**2)) * (phi_new[2:, 1:-1] - 2 * phi_new[1:-1, 1:-1] + phi_new[:-2, 1:-1]) -
        (kappa / (dy**2)) * (phi_new[1:-1, 2:] - 2 * phi_new[1:-1, 1:-1] + phi_new[1:-1, :-2])
    )
    
    # Apply boundary conditions to mu_new
    apply_boundary_conditions(mu_new)
    
    # Define residual for BE step in 2D
    res = np.zeros_like(phi_old)
    
    # Update interior points
    res[1:-1, 1:-1] = (
        (phi_new[1:-1, 1:-1] - phi_old[1:-1, 1:-1]) / dt -
        (1 / (dx**2)) * (
            M_func_half(phi_new[1:-1, 1:-1], phi_new[2:, 1:-1]) * (mu_new[2:, 1:-1] - mu_new[1:-1, 1:-1]) -
            M_func_half(phi_new[1:-1, 1:-1], phi_new[:-2, 1:-1]) * (mu_new[1:-1, 1:-1] - mu_new[:-2, 1:-1])
        ) -
        (1 / (dy**2)) * (
            M_func_half(phi_new[1:-1, 1:-1], phi_new[1:-1, 2:]) * (mu_new[1:-1, 2:] - mu_new[1:-1, 1:-1]) -
            M_func_half(phi_new[1:-1, 1:-1], phi_new[1:-1, :-2]) * (mu_new[1:-1, 1:-1] - mu_new[1:-1, :-2])
        )
    )
    
    return res

# Simulation params
Lx, Ly = 20.0, 20.0
nx, ny = 101, 101
dx, dy = Lx / (nx - 1), Ly / (ny - 1)
xvals = np.linspace(0, Lx, nx)
yvals = np.linspace(0, Ly, ny)

# Time-stepping parameters
tf = 20.0
nsteps = 200
dt = tf / nsteps

# Initial conditions: small random perturbation around 0.5
phi0 = 0.5 + 0.05 * np.random.rand(ny, nx)
phi = phi0.copy()

# Animation setup
fig, ax = plt.subplots()
cax = ax.imshow(phi, vmin=0, vmax=1, extent=[0, Lx, 0, Ly], origin="lower", cmap="viridis")
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color="white")
ax.set_xlabel("x")
ax.set_ylabel("y")

# Main time-stepping loop
def update(n):
    global phi
    phi_old = phi.copy()

    def wrapped_residual(phi_new):
        return backward_euler_fd_res(phi_new.reshape(ny, nx), phi_old).ravel()
    
    sol = root(wrapped_residual, phi_old.ravel(), method="krylov", tol=1e-10)
    phi_new = sol.x.reshape(ny, nx)
    phi = phi_new.copy()
    print("beep")
    
    # Update the plot
    cax.set_data(phi)
    time_text.set_text(f'Time = {n * dt:.2f}')
    return cax, time_text

ani = animation.FuncAnimation(fig, update, frames=nsteps, blit=True, repeat=False)
plt.colorbar(cax)
plt.show()
