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


#formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')

"""
This script implements a simple Backwards Euler method for solving
the ternary CH equation in mixed form
"""

def ternary(ic_fun, chi12, chi13, chi23, x1,x2,x3,tend, deltax, dt, stride, return_data=False, order=1):
    #Simulation parameters
    Lx = Ly = 50.0
    nx = ny = int(Lx/deltax)

    t = 0.0
    tend = tend
    nsteps = int(tend/dt)
    tvals = np.linspace(0.0,tend,nsteps+1)

    kappa_1 = (2/3)*chi13
    kappa_2 = (2/3)*chi23
    kappa_12 = (1/3)*(chi13 + chi23 - chi12)

    #Create lists for storing energy, phi_min, phi_max
    energy_list = []
    phi1_min_list = []
    phi1_max_list = []
    phi1_avg_list = []
    phi2_min_list = []
    phi2_max_list = []
    phi2_avg_list = []

    #Set up mesh
    domain = mesh.create_rectangle(MPI.COMM_WORLD, [[0.0,0.0],[Lx, Ly]], [nx,ny])
    P1 = element("Lagrange",domain.basix_cell(),order)
    ME = fem.functionspace(domain,mixed_element([P1,P1,P1,P1,P1]))
    q1,q2,v1,v2,v3 = ufl.TestFunctions(ME)

    #Define solution variables and split 
    u = fem.Function(ME)
    u0 = fem.Function(ME)

    a, b, mu_12, mu_13, mu_23 = ufl.split(u)
    a0, b0, mu_120, mu_130, mu_230 = ufl.split(u0)

    

