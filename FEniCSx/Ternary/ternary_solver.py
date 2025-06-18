import numpy as np
from mpi4py import MPI
import basix
import ufl
from basix.ufl import element, mixed_element,quadrature_element
from petsc4py import PETSc
from dolfinx import fem, mesh, io, log
from ufl import grad, inner, ln, dx
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

from ufl import grad, inner, ln, Measure, derivative
from scipy.interpolate import PchipInterpolator
import multiprocessing
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os

def cahn_hilliard_analytical(ic_fun, chi12, chi13, chi23, N1,N2,N3,stride,tend,deltax, dt, return_data = False, return_vtk=False)
    #Simulation parameters
    Lx = Ly = 20.0
    nx = ny = int(Lx/deltax)

    asym_n1n2 = (N2/N1)
    asym_n1n3 = (N3/N1)
    
    if asym_n1n3 > 0.1:
        kappa1 = (2/3)*chi13
    else:
        kappa1 = (1/3)*chi13

    if asym_n1n3 > 0.1 and asym_n1n2 > 0.1:
        kappa2 = (2/3)*chi23
    elif asym_n1n2 > 0.1 or asym_n1n3 > 0.1 :
        kappa2 = (1/3)*chi23
    else:
        kappa2 = 0

    if asym_n1n2 > 0.1 and asym_n1n3 > 0.1:
        kappa12 = (1/3)*chi13 + (1/3)*chi23 - (1/3)*chi12
    elif asym_n1n2 > 0.1 and asym_n1n3 <= 0.1: 
        kappa12 = (1/6)*chi13 + (1/6)*chi23 -(1/3)*chi12
    elif asym_n1n2 <= 0.1 and asym_n1n3 > 0.1:
        kappa12 = (1/3)*chi13 + (1/6)*chi23 - (1/6)*chi12
    else: 
        kappa12 = (1/6)*chi13 - (1/6)*chi12
    
    #Timestepping parameters
    t = 0.0
    tend = tend
    nsteps = int(tend/dt)
    tvals = np.linspace(0.0,tend,nsteps+1)

    if return_data:
        energy_list = []
        phi1_min_list = []
        phi1_max_list = []
        phi1_avg_list = []
        phi2_min_list = []
        phi2_max_list = []
        phi2_avg_list = []

    #Set up mesh
    domain = mesh.create_rectangle(MPI.COMM_WORLD, [[0.0,0.0],[Lx, Ly]], [nx,ny])

    #Define test functions
    P1 = element("Lagrange",domain.basix_cell(),2)
    ME = fem.functionspace(domain,mixed_element([P1,P1,P1,P1,P1]))
    q1, q2, v1, v2, v3 = ufl.TestFunctions(ME)

    #Define solution variables and split 
    u = fem.Function(ME)
    u0 = fem.Function(ME)

    a, b, mu_AB, mu_AC, mu_BC = ufl.split(u)
    a0, b0, mu_AB0, mu_AC0, mu_BC0 = ufl.split(u0)

    #Define model equations
    


