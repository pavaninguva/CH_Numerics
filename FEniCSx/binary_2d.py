import numpy as np
from mpi4py import MPI
import basix
import ufl
from basix.ufl import element, mixed_element
from petsc4py import PETSc
from dolfinx import fem, mesh, io
from ufl import grad, inner, ln, dx
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

import matplotlib.pyplot as plt
import os
# os.environ["QT_QPA_PLATFORM"] = "xcb"

#formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')

"""
This script implements a simple Backwards Euler method for solving
the CH equation in mixed form
"""

def cahn_hilliard(ic_fun, chi, N1, N2, stride, tend, deltax, dt):
    #Simulation parameters
    Lx = Ly = 50.0
    nx = ny = int(Lx/deltax)
    kappa = (2/3)*chi

    t = 0.0
    tend = tend
    nsteps = int(tend/dt)
    tvals = np.linspace(0.0,tend,nsteps+1)

    #Create lists for storing energy, phi_min, phi_max
    energy_list = []
    phi_min_list = []
    phi_max_list = []
    phi_avg_list = []

    #Set up mesh
    domain = mesh.create_rectangle(MPI.COMM_WORLD, [[0.0,0.0],[Lx, Ly]], [nx,ny])
    P1 = element("Lagrange",domain.basix_cell(),1)
    ME = fem.functionspace(domain,mixed_element([P1,P1]))
    q,v = ufl.TestFunctions(ME)

    #Define solution variables and split 
    u = fem.Function(ME)
    u0 = fem.Function(ME)

    #Solution variables
    c,mu = ufl.split(u)
    c0,mu0 = ufl.split(u0)

    #Define chemical potential
    c = ufl.variable(c)
    f = c*ln(c) + (1-c)*ln(1-c) + chi*c*(1-c)
    dfdc = ufl.diff(f,c)

    F0 = inner(c,q)*dx - inner(c0,q)*dx + (c*(1-c))*dt*inner(grad(mu),grad(q))*dx
    F1 = inner(mu,v)*dx - inner(dfdc,v)*dx - kappa*inner(grad(c),grad(v))*dx
    F = F0 + F1

    #Apply Initial conditions
    u.x.array[:] = 0.0
    u.sub(0).interpolate(ic_fun)
    u.x.scatter_forward()
    c = u.sub(0)
    u0.x.array[:] = u.x.array

    #Write to VTK and write ICs
    writer = io.VTXWriter(domain.comm, "sim.bp",[c],"BP4")
    writer.write(0.0)

    #Compute energy at t =0
    energy_density = fem.form(((1/N1)*c*ln(c) + (1/N2)*(1-c)*ln(1-c) + chi*c*(1-c) + (kappa/2)*inner(grad(c),grad(c)))*dx)
    total_energy = fem.assemble_scalar(energy_density)
    energy_list.append(total_energy)

    #Extract c_min and c_max
    c_min = np.min(c.collapse().x.array)
    c_max = np.max(c.collapse().x.array)
    c_avg = np.average(c.collapse().x.array)
    phi_min_list.append(c_min)
    phi_max_list.append(c_max)
    phi_avg_list.append(c_avg)


    #Setup solver
    problem = NonlinearProblem(F, u)
    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.convergence_criterion = "residual"
    solver.atol = 1e-8
    solver.report = True

    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "preonly"
    opts[f"{option_prefix}pc_type"] = "lu"
    ksp.setFromOptions()

    #Introduce skipping for output to output only every nth step
    stride = stride
    counter = 0

    while t < tend -1e-8: 
        #Update t
        t += dt
        #Solve
        res = solver.solve(u)
        print(f"Step {int(t/dt)}: num iterations: {res[0]}")

        #Update u0
        u0.x.array[:] = u.x.array

        #Evaluate total energy and append
        total_energy = fem.assemble_scalar(energy_density)
        energy_list.append(total_energy)

        #Phi_min and Phi_max
        c_min = np.min(c.collapse().x.array)
        c_max = np.max(c.collapse().x.array)
        c_avg = np.average(c.collapse().x.array)
        phi_min_list.append(c_min)
        phi_max_list.append(c_max)
        phi_avg_list.append(c_avg)


        #Update counter and write to VTK
        counter = counter +1
        if counter % stride == 0:
            writer.write(t)

    #Close VTK file
    writer.close()
    return tvals, phi_max_list, phi_min_list, phi_avg_list ,energy_list

"""
Test case
"""
def initial_condition(x):
    values = 0.5 + 0.02*(0.5-np.random.rand(x.shape[1]))
    return values

tvals, phi_max, phi_min, phi_avg, energy_vals = cahn_hilliard(initial_condition,chi=6,N1=1,N2=1,stride=1,tend=5,deltax=1.0,dt=0.01)
        

    
#Plot
fig, ax1 = plt.subplots()

ax1.plot(tvals,phi_max, label=r"$\phi_{1,\max}$",linestyle="--",color="blue")
ax1.plot(tvals,phi_min,label=r"$\phi_{1,\min}$",linestyle="-.",color="blue")
ax1.plot(tvals,phi_avg,label=r"$\bar{1,\phi}}$",linestyle="-",color="blue")
ax1.set_xlabel(r"Time ($\tilde{t}$)")
ax1.set_ylabel(r"$\phi_{1}$")
ax1.tick_params(axis='y', labelcolor='blue')         
ax1.yaxis.label.set_color('blue')

ax2 = ax1.twinx()
ax2.plot(tvals, energy_vals,linestyle="-", color="red")
ax2.set_ylabel("Total Energy")
ax2.tick_params(axis='y', labelcolor='red')
ax2.yaxis.label.set_color('red')

fig.tight_layout()

plt.show()




