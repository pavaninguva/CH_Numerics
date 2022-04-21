from fipy import *
from fipy.tools import numerix 
import numpy as np
import matplotlib.pyplot as plt
import warnings

"""
Format
"""
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')

warnings.filterwarnings("error")

"""
Simulation Parameters
"""
#Length of simulation domain
Lx = 5
#Number of Cells
nx = 50
#Compute dx
dx = Lx/nx

#Initial Composition
phi0 = 0.5
#Noise Magnitude
noise_mag = 0.01
#FH Parameters
Nchi = 5.0
kappa = (2.0/3.0)*Nchi

#Simulation time
t_end = 40.0
dt = 0.1

"""
Construct Problem
"""
#Create mesh with periodic BCs
# mesh = PeriodicGrid1D(nx=nx,dx=dx)
#Mesh with No-flux BCs
mesh = Grid1D(nx=nx,dx=dx)

#Extract mesh cell centers
x = mesh.cellCenters[0]

#Define variables
phi = CellVariable(name=r"$\phi$", mesh=mesh, hasOld=1)
mu = CellVariable(name=r"$\mu$", mesh=mesh, hasOld=1)

#Set Initial Conditions
#Set Uniform Noise
noise = UniformNoiseVariable(mesh=mesh, minimum=(phi0-noise_mag), maximum=(phi0+noise_mag))
phi[:] = noise
#Set Step Function
# phi.setValue(0.75)
# phi.setValue(0.25, where=x< Lx/2)
#Set Tanh
# phi.setValue(0.4*(1-numerix.tanh(x-Lx/2)))

#Compute Initial Total Mass
phi0_np = np.asarray(phi)
x_np = np.asarray(x)
# phi0_tot = np.trapz(phi_0_np,x=x_np)
phi0_tot = np.sum(phi0_np)

#Compute initial IFT
grad_phi0 = np.gradient(phi0_np,x_np,edge_order=2)
int_gradphi0 = np.trapz(kappa*grad_phi0**2,x=x_np)
IFT0 = int_gradphi0

#Define Problem
dgda = (1-2*phi)*Nchi + numerix.log(phi) - numerix.log(1-phi)

eq1 = TransientTerm(var=phi) == DiffusionTerm(coeff=(phi*(1-phi)),var=mu)
eq2 = ImplicitSourceTerm(coeff=1, var=mu) == dgda - DiffusionTerm(coeff=kappa,var=phi)

#Couple equations
eq = eq1 & eq2

print ("Problem Specified")

"""
Perform Solution
"""
solver = LinearGMRESSolver(tolerance=1e-10, precon="redundant")

t = 0.0
t_list = [0.0]
mass_list = [0.0]
IFT_list = [IFT0]
counter = 0
breaker = False
clipping = False

while t < t_end -1e-5:
    #Update time and counter
    t = t +dt
    #Update
    phi.updateOld()
    mu.updateOld()
    res = 1e4
    while res > 1e-10:
        try:
            if clipping is True:
                phi.setValue(1e-10, where=phi<0.)
                phi.setValue(1.0-1e-10, where=phi>1.0)
            else:
                pass
            res = eq.sweep(dt=dt,solver=solver)
        except RuntimeWarning:
            print("Simulation Diverged at t=%s"%t)
            breaker = True
            print("Terminating Simulation at t=%s"%t)
            break
    if breaker is True:
        break
    #Compute mass conservation
    phi_vals_np = np.asarray(phi)
    # mass_list.append(phi0_tot - np.trapz(phi_vals_np,x=x_np))
    mass_list.append(abs(phi0_tot - np.sum(phi_vals_np.copy())))
    t_list.append(t)
    #Compute interfacial tension
    grad_phi = np.gradient(phi_vals_np,x_np,edge_order=2)
    IFT_list.append(np.trapz(kappa*grad_phi**2,x=x_np))


    
    print("Current Simulation Time is %s"%t)

"""
Plotting
"""
#Plot mass deviation
fig1 = plt.figure(num=1)
plt.semilogy(t_list,mass_list)
plt.xlabel(r"$\tilde{t}$")
plt.ylabel("Mass Deviation")
plt.tight_layout()

#Plot dphi/dx at end
fig2 = plt.figure(num=2)
plt.plot(x_np,grad_phi)
plt.xlabel(r"$\tilde{x}$")
plt.ylabel(r"$\frac{\partial \phi}{\partial \tilde{x}}$")
# plt.ylim((0.0,1.0))
plt.tight_layout()

#Plot IFT
fig3 = plt.figure(num=3)
plt.plot(t_list,IFT_list)
plt.xlabel(r"$\tilde{t}$")
plt.ylabel(r"$\frac{\gamma}{R_{G} \rho_{m}RT}$")
plt.tight_layout()

#Plot Solution at final time
fig4 = plt.figure(num=4)
plt.plot(x_np,phi_vals_np)
plt.xlabel(r"$\tilde{x}$")
plt.ylabel(r"$\phi$")
# plt.ylim((0.0,1.0))
plt.tight_layout()




plt.show()




