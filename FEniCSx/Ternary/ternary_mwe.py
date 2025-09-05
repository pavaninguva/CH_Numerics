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

from dolfinx_external_operator import (
    FEMExternalOperator,
    evaluate_external_operators,
    evaluate_operands,
    replace_external_operators,
)

from linear_solvers import LinearProblem

def cahn_hilliard_spline(ic_fun_a, ic_fun_b, chi12, chi13, chi23, N1,N2,N3, stride,tend,deltax, dt):
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

    #Solver parameters
    max_iter = 100
    rel_tol = 1e-6
    abs_tol = 1e-8


    #Set up mesh
    domain = mesh.create_rectangle(MPI.COMM_WORLD, [[0.0,0.0],[Lx, Ly]], [nx,ny])

    #Define test functions
    P1 = element("Lagrange",domain.basix_cell(),2)
    ME = fem.functionspace(domain,mixed_element([P1,P1,P1,P1,P1]))
    q1, q2, v1, v2, v3 = ufl.TestFunctions(ME)

    #Define solution variables and split 
    u = fem.Function(ME)
    u0 = fem.Function(ME)
    du = fem.Function(ME)
    me = ufl.TrialFunction(ME)

    a, b, mu_AB, mu_AC, mu_BC = ufl.split(u)
    a0, b0, mu_AB0, mu_AC0, mu_BC0 = ufl.split(u0)



    quadrature_degree = 2
    Qe=quadrature_element(domain.topology.cell_name(), degree=quadrature_degree, value_shape=())
    Q = fem.functionspace(domain, Qe)
    dx = Measure(
    "dx",
    metadata={"quadrature_scheme": "default", "quadrature_degree": 1},)

    dfda = FEMExternalOperator(a,b,function_space=Q)
    dfdb = (1/N2)*ln(b) + (1/N2) + chi12*a + chi23*(1-a-b)
    dfdc =  (1/N3)*ln(1-a-b) + (1/N3) + chi13*a + chi23*b

    def dfda_impl(a,b):
        output = (1/N1)*np.log(a) + (1/N1) + chi12*b + chi13*(1-a-b)
        return output.reshape(-1)
    def d2fda2_impl(a,b):
        output = (1/N1)*(1/a) - chi13
        return output.reshape(-1)
    def d2fdab_impl(a,b):
        output = np.ones_like(a) * (chi12 - chi13)
        return output.reshape(-1)
    
    def dfda_external(derivatives):
        if derivatives == (0,0):
            return dfda_impl
        elif derivatives == (1,0):
            return d2fda2_impl
        elif derivatives == (0,1):
            return d2fdab_impl
        else:
            raise NotImplementedError(f"No external function is defined for the requested derivative {derivatives}.")
    
    dfda.external_function = dfda_external


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


    J = derivative(F, u, me)
    J_expanded = ufl.algorithms.expand_derivatives(J)
    F_replaced, F_external_operators = replace_external_operators(F)
    J_replaced, J_external_operators = replace_external_operators(J_expanded)


    evaluated_operands = evaluate_operands(F_external_operators)
    _ = evaluate_external_operators(F_external_operators, evaluated_operands)
    _ = evaluate_external_operators(J_external_operators, evaluated_operands)

    #Set up solver
    problem = LinearProblem(J_replaced, -F_replaced, du)

    while t < tend - 1e-8:
        #Update t
        t += dt
        print(t)

        #Solve
        problem.assemble_vector()
        residual_0 = problem.b.norm()
        residual = residual_0
        print(residual)
        for iteration in range(max_iter):
            if iteration > 0:
                if residual < abs_tol and rel_error < rel_tol:
                    break
            problem.assemble_matrix()
            problem.solve(du)
            du.x.scatter_forward()
            u.x.petsc_vec.axpy(1.0, du.x.petsc_vec)
            evaluated_operands = evaluate_operands(F_external_operators)
            _ = evaluate_external_operators(F_external_operators, evaluated_operands)
            _ = evaluate_external_operators(J_external_operators, evaluated_operands)
            
            problem.assemble_vector()
            #Update residual
            residual = problem.b.norm()
            rel_error = np.abs(residual / residual_0)
            iteration += 1
            print(f"Iteration:{iteration}, Residual:{residual}")
        if iteration == max_iter:
            raise RuntimeError("Maximum number of iterations, exiting") 
        if residual > abs_tol or rel_error > rel_tol:
            raise RuntimeError("Simulation did not converge")
        print(f"For (Chi {chi12,chi13,chi13}, dx {deltax}, dt {dt}): Time {t}: %Complete {(t/tend)*100}, num iterations: {iteration}")

        #Update u0
        u0.x.array[:] = u.x.array[:]

        
    return