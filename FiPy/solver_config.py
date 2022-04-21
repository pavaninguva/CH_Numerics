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
Solver Parameters
"""
precon_list = [
    "jacobi","bjacobi","sor","eisenstat","icc","ilu","asm",
    "gasm","gamg","bddc","ksp","composite","lu","cholesky",
    "redundant"
]

solver_list = [
    "BICG","CGS","GMRES","LU","PCG"
]

"""
Define Solver Function
"""

def solver_function(precon, solver,nx,dt,Nchi):
    #Simulation parameters
    Lx = 5
    nx = nx
    dx = Lx/nx
    phi0 = 0.5
    noise_mag = 0.01
    Nchi = Nchi
    kappa = (2.0/3.0)*Nchi

    #Simulation time
    t_end = 40.0
    dt = dt

    #Construct Problem
    mesh = Grid1D(nx=nx,dx=dx)
    x = mesh.cellCenters[0]

    #Define variables and set IC
    phi = CellVariable(name=r"$\phi$", mesh=mesh, hasOld=1)
    mu = CellVariable(name=r"$\mu$", mesh=mesh, hasOld=1)
    noise = UniformNoiseVariable(mesh=mesh, minimum=(phi0-noise_mag), maximum=(phi0+noise_mag))
    phi[:] = noise

    #Define equation
    dgda = (1-2*phi)*Nchi + numerix.log(phi) - numerix.log(1-phi)
    eq1 = TransientTerm(var=phi) == DiffusionTerm(coeff=(phi*(1-phi)),var=mu)
    eq2 = ImplicitSourceTerm(coeff=1, var=mu) == dgda - DiffusionTerm(coeff=kappa,var=phi)
    eq = eq1 & eq2

    if solver == "BICG":
        solver = LinearBicgSolver(tolerance=1e-10,precon=precon)
    elif solver == "CG":
        solver = LinearCGSSolver(tolerance=1e-10,precon=precon)
    elif solver == "GMRES":





