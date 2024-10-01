import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import root
from scipy.optimize import newton_krylov

#Parameters
N1 = 1
N2 = 1
chi = 5
kappa = (2/3)*chi

def fh(phi):
    return (1/N1)*phi*np.log(phi) + (1/N2)*(1-phi)*np.log(1-phi) + chi*phi*(1-phi)


def dfdphi(phi):
    return -2*chi*phi + chi - (1/N2)*np.log(1-phi) + (1/N1)*np.log(phi)

def mobility(phi,option=1):
    if option == 1:
        mobility = phi*(1-phi)
    elif option ==2:
        mobility = 1
    return mobility

def M_func_half(phi, phi_,option="2"):
    if option == "1":
        M_func = 0.5*(mobility(phi)+mobility(phi_))
    elif option == "2":
        M_func = mobility(0.5*(phi+phi_))
    elif option == "3":
        M_func = (2*mobility(phi)*mobility(phi_))/(mobility(phi) + mobility(phi_))
    else:
        raise ValueError("Specified option for mobility interpolation not available")
    return M_func


def backward_euler_fd_res(phi_new, phi_old, dx, dt):
    #Define chem_pot
    mu_new = np.zeros_like(phi_old)

    mu_new[0] = dfdphi(phi_new[0]) -(2*kappa/(dx**2))*(phi_new[1] - phi_new[0])
    mu_new[-1] = dfdphi(phi_new[-1]) - (2*kappa/(dx**2))*(phi_new[-2]-phi_new[-1])
    mu_new[1:-1] = dfdphi(phi_new[1:-1]) - (kappa/(dx**2))*(phi_new[2:] -2*phi_new[1:-1] + phi_new[:-2])
    
    # Define discretized equations as F(phi_new) = 0
    res = np.zeros_like(phi_old)

    res[0] = (phi_new[0] - phi_old[0])/dt - (2/(dx**2))*(M_func_half(phi_new[0],phi_new[1]))*(mu_new[1] - mu_new[0])
    res[-1] = (phi_new[-1] - phi_old[-1])/dt - (2/(dx**2))*(M_func_half(phi_new[-1],phi_new[-2]))*(mu_new[-2]-mu_new[-1])
    res[1:-1] = (phi_new[1:-1] - phi_old[1:-1])/dt - (1/(dx**2))*(M_func_half(phi_new[1:-1], phi_new[2:])*(mu_new[2:]-mu_new[1:-1]) - M_func_half(phi_new[1:-1],phi_new[:-2])*(mu_new[1:-1]-mu_new[:-2]))

    return res

#Simulation params
#Domain length
L = 10.0
nx = 201
dx = L/(nx-1)
xvals = np.linspace(0,L,nx)

#Timestepping parameters
tf = 20.0
nsteps = 2000
dt = tf/nsteps

#Initial conditions
# phi0 = 0.5 + 0.05*np.random.rand(np.size(xvals))
# phi0 = 0.5 + 0.1*np.sin((6*np.pi*xvals)/L)

# def sigmoid(x, a, b):
#     return 1 / (1 + np.exp(-a * (x - b)))

# # Parameters for the sigmoid function
# a = 5  # Controls the steepness of the sigmoid
# b = L / 2  # Center of the sigmoid function
# phi0 = 0.2 + 0.6* sigmoid(xvals, a, b)
# phi = phi0.copy()

# # Animation setup
# fig, ax = plt.subplots()
# line, = ax.plot(xvals, phi, label=r"$\phi$")
# time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
# ax.set_xlim(0.0, L)
# ax.set_ylim(-0.2, 1.2)
# ax.set_xlabel("x")
# ax.set_ylabel(r"$\phi$")
# ax.hlines(y=0.0, xmin=0, xmax=L,color="r")
# ax.hlines(y=1.0, xmin=0, xmax=L,color="r")
# ax.legend()

# Main time-stepping loop
# def update(n):
#     global phi
#     phi_old = phi.copy()
#     # phi_old = np.clip(phi_old_,0+1e-8,1-1e-8)

#     def wrapped_residual(phi_new):
#         return backward_euler_fd_res(phi_new, phi_old)
    
#     sol = root(wrapped_residual, phi_old,method="hybr",tol=1e-10)
#     # sol = newton_krylov(wrapped_residual,phi_old)
#     phi_new = sol.x
#     phi = phi_new.copy()
#     # print(np.log(phi))
    
#     line.set_ydata(phi)
#     time_text.set_text(f'Time = {n * dt:.2f}')
#     return line, time_text

# ani = animation.FuncAnimation(fig, update, frames=nsteps, blit=True, repeat=False)

# plt.show()


def compute_energy(phi, dx):
    grad_phi = np.gradient(phi, dx)
    energy = np.sum((kappa/2) * grad_phi**2) * dx
    return energy

def compute_mass(phi, dx):
    total_mass = np.sum(phi) * dx
    return total_mass

def update(n, phi, phi_old, dx, dt, energies, masses, max_vals, min_vals):
    phi_older = phi_old.copy()
    phi_old[:] = phi.copy()

    def wrapped_residual(phi_new):
        return backward_euler_fd_res(phi_new, phi_old, dx, dt)

    sol = root(wrapped_residual, phi_old, method="krylov", tol=1e-10)
    phi_new = sol.x
    phi[:] = phi_new.copy()

    # Update energy, mass, max, min values
    energies.append(compute_energy(phi, dx))
    masses.append(compute_mass(phi, dx))
    max_vals.append(np.max(phi))
    min_vals.append(np.min(phi))

    return phi

L = 10.0
dx_values = [L/101, L/201, L/301]  # Test different dx values
dt_values = [0.01, 0.005, 0.0025]  # Test different dt values
tf = 10.0
nsteps = 1000

# Initialize plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

for dx, dt in zip(dx_values, dt_values):
    xvals = np.linspace(0, L, int(L/dx))
    phi0 = 0.5 + 0.05 * np.random.rand(np.size(xvals))  # Initial condition
    phi = phi0.copy()
    phi_old = phi0.copy()

    # Arrays to store energy, mass, and max/min values
    energies = []
    masses = []
    max_vals = []
    min_vals = []

    # Time-stepping loop
    for n in range(nsteps):
        phi = update(n, phi, phi_old, dx, dt, energies, masses, max_vals, min_vals)

    # Plot total energy
    time = np.arange(nsteps) * dt
    ax1.plot(time, energies, label=f'dx={dx:.4f}, dt={dt:.4f}')
    
    # Plot total mass, max, and min values on second plot with two y-axes
    ax2.plot(time, masses, label=f'Mass dx={dx:.4f}, dt={dt:.4f}')
    ax2.plot(time, max_vals, label=f'Max $\phi$ dx={dx:.4f}, dt={dt:.4f}', linestyle='--')
    ax2.plot(time, min_vals, label=f'Min $\phi$ dx={dx:.4f}, dt={dt:.4f}', linestyle=':')

# Customize plots
ax1.set_xlabel('Time')
ax1.set_ylabel('Total Energy')
ax1.legend()
ax1.grid(True)

ax2.set_xlabel('Time')
ax2.set_ylabel('Mass/Max/Min $\phi$')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()