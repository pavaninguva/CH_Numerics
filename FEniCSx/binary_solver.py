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
import ufl
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

from solvers import LinearProblem

"""
This script contains the solvers for the binary Cahn-Hilliard equation. 

Both the analytical and spline implementations are included. 
"""

"""
Analytical
"""

def cahn_hilliard_analytical(ic_fun, chi, N1, N2, stride, tend, deltax, dt, return_data=False, return_vtk=False):
    #Simulation parameters
    Lx = Ly = 20.0
    nx = ny = int(Lx/deltax)

    asym_factor = (N1/N2)
    if asym_factor < 0.1:
        kappa = (1/3)*chi
    else:
        kappa = (2/3)*chi

    t = 0.0
    tend = tend
    nsteps = int(tend/dt)
    tvals = np.linspace(0.0,tend,nsteps+1)

    #Create lists for storing energy, phi_min, phi_max
    if return_data:
        energy_list = []
        phi_min_list = []
        phi_max_list = []
        phi_avg_list = []

    #Set up mesh
    domain = mesh.create_rectangle(MPI.COMM_WORLD, [[0.0,0.0],[Lx, Ly]], [nx,ny])
    P1 = element("Lagrange",domain.basix_cell(),2)
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
    if return_vtk:
        writer = io.VTXWriter(domain.comm, "PS.bp",[c],"BP4")
        writer.write(0.0)

    #Compute energy at t =0
    if return_data:
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
    solver.report = False
    solver.error_on_nonconvergence=True

    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "preonly"
    opts[f"{option_prefix}pc_type"] = "lu"
    ksp.setFromOptions()

    #Introduce skipping for output to output only every nth step
    stride = stride
    counter = 0

    # log.set_log_level(log.LogLevel.INFO)
    while t < tend -1e-8: 
        #Update t
        t += dt
        #Solve
        res = solver.solve(u)
        # print(res)
        # print(f"Step {int(t/dt)}: num iterations: {res[0]}")
        print(f"For (Chi {chi}, dx {deltax}, dt {dt}): Time {t}: %Complete {(t/tend)*100}, num iterations: {res[0]}")

        #Update u0
        u0.x.array[:] = u.x.array

        if return_data:
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
        if return_vtk:
            if counter % stride == 0:
                writer.write(t)

    #Close VTK file
    if return_vtk:
        writer.close()

    if return_data:
        return tvals, phi_max_list, phi_min_list, phi_avg_list ,energy_list
    else:
        return
    
"""
Spline
"""

def spline_generator(chi, N1, N2, knots):
    def dfdphi(c):
        return -2*chi*c + chi - (1/N2)*np.log(1-c) + (1/N1)*np.log(c) + (1/N1) - (1/N2)
    
    spline_pot = PchipInterpolator(np.linspace(0,1,knots),dfdphi(np.linspace(1e-16,1-1e-16,knots)))

    df_spline = spline_pot
    d2f_spline = spline_pot.derivative(1)

    return df_spline, d2f_spline


def cahn_hilliard_spline(ic_fun, chi, N1, N2, stride, tend, deltax, dt, return_data=False, return_vtk = False):
    #Simulation parameters
    Lx = Ly = 20.0
    nx = ny = int(Lx/deltax)
    
    asym_factor = (N1/N2)
    if asym_factor < 0.1:
        kappa = (1/3)*chi
    else:
        kappa = (2/3)*chi

    max_iter = 100
    rel_tol = 1e-6
    abs_tol = 1e-8

    t = 0.0
    tend = tend
    nsteps = int(tend/dt)
    tvals = np.linspace(0.0,tend,nsteps+1)

    if return_data:
        #Create lists for storing energy, phi_min, phi_max
        energy_list = []
        phi_min_list = []
        phi_max_list = []
        phi_avg_list = []

    #Set up mesh
    domain = mesh.create_rectangle(MPI.COMM_WORLD, [[0.0,0.0],[Lx, Ly]], [nx,ny])
    P1 = element("Lagrange",domain.basix_cell(),2)
    ME = fem.functionspace(domain,mixed_element([P1,P1]))
    q,v = ufl.TestFunctions(ME)

    #Define solution variables and split 
    u = fem.Function(ME)
    u0 = fem.Function(ME)
    du = fem.Function(ME) # change in solution
    me = ufl.TrialFunction(ME)

    #Solution variables
    c,mu = ufl.split(u)
    c0,mu0 = ufl.split(u0)

    #Spline bits
    df_spline, d2f_spline = spline_generator(chi,N1,N2,800)

    quadrature_degree = 2
    Qe=quadrature_element(domain.topology.cell_name(), degree=quadrature_degree, value_shape=())
    Q = fem.functionspace(domain, Qe)
    dx = Measure(
    "dx",
    metadata={"quadrature_scheme": "default", "quadrature_degree": 1},)

    dfdc = FEMExternalOperator(c,function_space=Q)

    def dfdc_impl(c):
        output = df_spline(c)
        return output.reshape(-1)
    
    def d2fd2c_impl(c):
        output = d2f_spline(c)
        return output.reshape(-1)
    
    def dfdc_external(derivatives):
        if derivatives == (0,):
            return dfdc_impl
        elif derivatives == (1,):  
            return(d2fd2c_impl)
        else:
            return NotImplementedError
        
    dfdc.external_function = dfdc_external

    #Define weak form of CH equation
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
    if return_vtk:
        writer = io.VTXWriter(domain.comm, "sim_spline.bp",[c],"BP4")
        writer.write(0.0)

    if return_data:

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

    #Set up bits for custom newton solver
    J = derivative(F, u, me)
    J_expanded = ufl.algorithms.expand_derivatives(J)
    F_replaced, F_external_operators = replace_external_operators(F)
    J_replaced, J_external_operators = replace_external_operators(J_expanded)
    evaluated_operands = evaluate_operands(F_external_operators)
    _ = evaluate_external_operators(F_external_operators, evaluated_operands)
    _ = evaluate_external_operators(J_external_operators, evaluated_operands)

    #Set up solver
    problem = LinearProblem(J_replaced, -F_replaced, du)

    #Introduce skipping for output to output only every nth step
    stride = stride
    counter = 0

    #Timestepping
    while t < tend - 1e-8:
        #Update t
        t += dt

        #Solve
        problem.assemble_vector()
        residual_0 = problem.b.norm()
        residual = residual_0
        # if MPI.COMM_WORLD.rank == 0:
            # print(f"Step {int(t/dt)}")
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
        if iteration == max_iter:
            raise RuntimeError("Maximum number of iterations, exiting") 
        if residual > abs_tol or rel_error > rel_tol:
            raise RuntimeError("Simulation did not converge")
            # if MPI.COMM_WORLD.rank == 0:
                # print(f"    it# {iteration}: residual: {residual}, relative error: {rel_error}")
        print(f"For (Chi {chi}, dx {deltax}, dt {dt}): Time {t}: %Complete {round((t/tend)*100,3)}, num iterations: {iteration}")

        #Update u0
        u0.x.array[:] = u.x.array[:]

        #Evaluate total energy and append
        if return_data:
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
        if return_vtk:
            if counter % stride == 0:
                writer.write(t)

    #Close VTK file
    if return_vtk:
        writer.close()

    if return_data:
        return tvals, phi_max_list, phi_min_list, phi_avg_list ,energy_list
    else:
        return
