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


def theta_fd_res(phi_new, phi_old, theta=1):
    #The theta scheme introduces a parameter theta which can be varied from 0 - 1 
    #Theta = 1: backward euler
    #Theta = 0: Forward euler
    #Theta = 0.5: Crank-Nicolson

    #Define chem_pot
    mu_new = np.zeros_like(phi_old)
    mu_old = np.zeros_like(phi_old)

    mu_new[0] = dfdphi(phi_new[0]) -(2*kappa/(dx**2))*(phi_new[1] - phi_new[0])
    mu_new[-1] = dfdphi(phi_new[-1]) - (2*kappa/(dx**2))*(phi_new[-2]-phi_new[-1])
    mu_new[1:-1] = dfdphi(phi_new[1:-1]) - (kappa/(dx**2))*(phi_new[2:] -2*phi_new[1:-1] + phi_new[:-2])

    mu_old[0] = dfdphi(phi_old[0]) -(2*kappa/(dx**2))*(phi_old[1] - phi_old[0])
    mu_old[-1] = dfdphi(phi_old[-1]) - (2*kappa/(dx**2))*(phi_old[-2]-phi_old[-1])
    mu_old[1:-1] = dfdphi(phi_old[1:-1]) - (kappa/(dx**2))*(phi_old[2:] -2*phi_old[1:-1] + phi_old[:-2])

    res = np.zeros_like(phi_old)

    res[0] =  (phi_new[0] - phi_old[0])/dt - (theta*((2/(dx**2))*(M_func_half(phi_new[0],phi_new[1]))*(mu_new[1] - mu_new[0])) +
                                              (1-theta)*(2/(dx**2))*(M_func_half(phi_old[0],phi_old[1]))*(mu_old[1] - mu_old[0])
                                              )
    res[-1] = (phi_new[-1] - phi_old[-1])/dt -(theta*((2/(dx**2))*(M_func_half(phi_new[-1],phi_new[-2]))*(mu_new[-2]-mu_new[-1])) +
                                               (1-theta)*((2/(dx**2))*(M_func_half(phi_old[-1],phi_old[-2]))*(mu_old[-2]-mu_old[-1]))
                                                )
    res[1:-1] = (phi_new[1:-1] - phi_old[1:-1])/dt - (theta*((1/(dx**2))*(M_func_half(phi_new[1:-1], phi_new[2:])*(mu_new[2:]-mu_new[1:-1]) - M_func_half(phi_new[1:-1],phi_new[:-2])*(mu_new[1:-1]-mu_new[:-2]))) +
                                                      (1-theta)*((1/(dx**2))*(M_func_half(phi_old[1:-1], phi_old[2:])*(mu_old[2:]-mu_old[1:-1]) - M_func_half(phi_old[1:-1],phi_old[:-2])*(mu_old[1:-1]-mu_old[:-2])))
                                                        )
    
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
phi0 = 0.5 + 0.05*np.random.rand(np.size(xvals))
# phi0 = 0.5 + 0.1*np.sin((6*np.pi*xvals)/L)

# def sigmoid(x, a, b):
#     return 1 / (1 + np.exp(-a * (x - b)))

# # Parameters for the sigmoid function
# a = 5  # Controls the steepness of the sigmoid
# b = L / 2  # Center of the sigmoid function
# phi0 = 0.2 + 0.6* sigmoid(xvals, a, b)
phi = phi0.copy()

# Animation setup
fig, ax = plt.subplots()
line, = ax.plot(xvals, phi, label=r"$\phi$")
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ax.set_xlim(0.0, L)
ax.set_ylim(-0.2, 1.2)
ax.set_xlabel("x")
ax.set_ylabel(r"$\phi$")
ax.hlines(y=0.0, xmin=0, xmax=L,color="r")
ax.hlines(y=1.0, xmin=0, xmax=L,color="r")
ax.legend()

# Main time-stepping loop
def update(n):
    global phi
    phi_old = phi.copy()
    # phi_old = np.clip(phi_old_,0+1e-8,1-1e-8)

    def wrapped_residual(phi_new):
        return theta_fd_res(phi_new, phi_old)
    
    sol = root(wrapped_residual, phi_old,method="hybr",tol=1e-10)
    # sol = newton_krylov(wrapped_residual,phi_old)
    phi_new = sol.x
    phi = phi_new.copy()
    # print(np.log(phi))
    
    line.set_ydata(phi)
    time_text.set_text(f'Time = {n * dt:.2f}')
    return line, time_text

ani = animation.FuncAnimation(fig, update, frames=nsteps, blit=True, repeat=False)

plt.show()