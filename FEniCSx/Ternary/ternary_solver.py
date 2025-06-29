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

"""
Analytical form
"""

def cahn_hilliard_analytical(ic_fun_a, ic_fun_b, chi12, chi13, chi23, N1,N2,N3,stride,tend,deltax, dt, return_data = False, return_vtk=False):
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
        a_min_list = []
        a_max_list = []
        a_avg_list = []
        b_min_list = []
        b_max_list = []
        b_avg_list = []

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
    a = ufl.variable(a)
    b = ufl.variable(b)

    f = (1/N1)*a*ln(a) + (1/N2)*b*ln(b) + (1/N3)*(1-a-b)*ln(1-a-b) + a*b*chi12 + a*(1-a-b)*chi13 + b*(1-a-b)*chi23

    dfda = (1/N1)*ln(a) + (1/N1) + chi12*b + chi13*(1-a-b)
    dfdb = (1/N2)*ln(b) + (1/N2) + chi12*a + chi23*(1-a-b)
    dfdc =  (1/N3)*ln(1-a-b) + (1/N3) + chi13*a + chi23*b

    F_a = (
        inner(a,q1)*dx - inner(a0,q1)*dx
        + dt*a*b*inner(grad(mu_AB), grad(q1))*dx
        + dt*a*(1-a-b)*inner(grad(mu_AC),grad(q1))*dx
    )

    F_b = (
        inner(b,q2)*dx - inner(b0,q2)*dx
        -dt*a*b*inner(grad(mu_AB),grad(q2))*dx
        +dt*b*(1-a-b)*inner(grad(mu_BC),grad(q2))*dx
    )

    F_mu_AB = (
        inner(mu_AB,v1)*dx 
        -inner(dfda,v1)*dx
        +inner(dfdb,v1)*dx
        -(kappa1-kappa12)*inner(grad(a),grad(v1))*dx
        +(kappa2 - kappa12)*inner(grad(b),grad(v1))*dx
    )

    F_mu_AC = (
        inner(mu_AC,v2)*dx
        -inner(dfda,v2)*dx
        +inner(dfdc,v2)*dx
        -kappa1*inner(grad(a),grad(v2))*dx
        -kappa12*inner(grad(b),grad(v2))*dx
    )

    F_mu_BC = (
        inner(mu_BC,v3)*dx
        -inner(dfdb,v3)*dx
        +inner(dfdc,v3)*dx
        -kappa12*inner(grad(a),grad(v3))*dx
        -kappa2*inner(grad(b),grad(v3))*dx
    )

    #Couple everything
    F = F_a + F_b + F_mu_AB + F_mu_AC + F_mu_BC

    #Apply initial conditions
    u.x.array[:] = 0.0
    u.sub(0).interpolate(ic_fun_a)
    u.sub(1).interpolate(ic_fun_b)
    u.x.scatter_forward()
    u0.x.array[:] = u.x.array
    a = u.sub(0)
    b = u.sub(1)

    if return_vtk:
        writer_a = io.VTXWriter(domain.comm,"a.bp",[a],"BP4")
        writer_a.write(0.0)
        writer_b = io.VTXWriter(domain.comm,"b.bp",[b],"BP4")
        writer_b.write(0.0)

    if return_data:
        energy_density = fem.form((
            f 
            + (kappa1/2)*inner(grad(a),grad(a))
            + (kappa2/2)*inner(grad(b),grad(b))
            + kappa12*inner(grad(a),grad(b))
        )*dx)

        total_energy = fem.assemble_scalar(energy_density)
        energy_list.append(total_energy)

        #Extract phi_min/max/avg
        a_min_list.append(np.min(a.collapse().x.array))
        a_max_list.append(np.max(a.collapse().x.array))
        a_avg_list.append(np.average(a.collapse().x.array))
        b_min_list.append(np.min(b.collapse().x.array))
        b_max_list.append(np.max(b.collapse().x.array))
        b_avg_list.append(np.average(b.collapse().x.array))

    #Setup solver
    problem = NonlinearProblem(F,u)
    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.convergence_criterion = "residual"
    solver.atol = 1e-8
    solver.report = False
    solver.error_on_nonconvergence=True

    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "preonly"
    opts[f"{option_prefix}pc_type"] = "lu"
    ksp.setFromOptions()

    #Set up time stepping
    stride = stride
    counter = 0

    while t < tend -1e-8:
        #Update t
        t += dt
        res = solver.solve(u)
        print(f"For (Chi {chi12,chi13,chi13}, dx {deltax}, dt {dt}): Time {t}: %Complete {(t/tend)*100}, num iterations: {res[0]}")

        #update u0
        u0.x.array[:] = u.x.array

        #Output files
        if return_data:
            total_energy = fem.assemble_scalar(energy_density)
            energy_list.append(total_energy)

            #Extract phi_min/max/avg
            a_min_list.append(np.min(a.collapse().x.array))
            a_max_list.append(np.max(a.collapse().x.array))
            a_avg_list.append(np.average(a.collapse().x.array))
            b_min_list.append(np.min(b.collapse().x.array))
            b_max_list.append(np.max(b.collapse().x.array))
            b_avg_list.append(np.average(b.collapse().x.array))

        counter = counter +1
        if return_vtk:
            if counter % stride == 0:
                writer_a.write(t)
                writer_b.write(t)
            
    if return_vtk:
        writer_a.close()
        writer_b.close()
    
    if return_data:
        return tvals, a_avg_list, b_avg_list, energy_list
    else:
        return
    

"""
Spline form
"""




def cahn_hilliard_spline(ic_fun_a, ic_fun_b, chi12, chi13, chi23, N1,N2,N3,stride,tend,deltax, dt, return_data = False, return_vtk=False):
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
        a_min_list = []
        a_max_list = []
        a_avg_list = []
        b_min_list = []
        b_max_list = []
        b_avg_list = []

        










