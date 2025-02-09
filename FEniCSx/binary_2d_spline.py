import numpy as np
from mpi4py import MPI
import basix
from basix.ufl import element, mixed_element,quadrature_element
from petsc4py import PETSc
from dolfinx import fem, mesh, io, plot
from dolfinx.fem import form
# from dolfinx.fem.petsc import NonlinearProblem
# from dolfinx.nls.petsc import NewtonSolver
from ufl import grad, inner, ln, Measure, derivative
from scipy.interpolate import make_interp_spline, CubicSpline
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

from mpmath import mp
import mpmath as mpmath

# Set high precision
mp.dps = 50 

#formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')

"""
Set up spline
"""

def spline_generator(chi, N1, N2, knots):
    # def log_terms(phi):
    #     # Vectorized log terms
    #     return (phi/N1)*np.log(phi) + ((1 - phi)/N2)*np.log(1 - phi)
    
    # def tanh_sinh_spacing(n, beta):
    #     # Return n points between 0 and 1 based on a tanh-sinh distribution
    #     # Indices go from 0 to n-1
    #     i = np.arange(n, dtype=float)
    #     return 0.5 * (1.0 + np.tanh(beta*(2.0*i/(n-1) - 1.0)))
    
    # phi_vals_ = tanh_sinh_spacing(knots - 2, 14.0)

    # # Evaluate the log-terms for those interior points
    # f_vals_ = log_terms(phi_vals_)

    # phi_vals = np.insert(phi_vals_, 0, 0.0)
    # phi_vals = np.append(phi_vals, 1.0)
    # f_vals = np.insert(f_vals_, 0, 0.0)
    # f_vals = np.append(f_vals, 0.0)

    # spline = CubicSpline(phi_vals, f_vals)
    # spline_derivative = spline.derivative()
    # spline_derivative2 = spline.derivative(nu=2)


    # def f_spline(phi):
    #     return spline(phi) + chi*phi*(1.0 - phi)

    # def df_spline(phi):
    #     return spline_derivative(phi) + chi*(1.0 - 2.0*phi)
    
    # def d2f_spline(phi):
    #     return spline_derivative2(phi) - 2.0*chi

    def dfdphi(c):
        return -2*chi*c + c - np.log(1-c) + np.log(c)
    
    spline_pot = CubicSpline(np.linspace(0,1,200),dfdphi(np.linspace(1e-16,1-1e-16,200)))

    f_spline = None
    df_spline = spline_pot
    d2f_spline = spline_pot.derivative(1)

    return df_spline, d2f_spline

# def fh_deriv(phi, chi, N1, N2):
#     return (1/N1)*np.log(phi) + (1/N1) \
#            - (1/N2)*np.log(1 - phi) - (1/N2) \
#            + chi - 2*chi*phi


# def spline_generator(chi, N1, N2, knots):
#     #Define small eps
#     eps = 1e-40
    
#     def tanh_sinh_spacing(n, beta):
#         # Return n points between 0 and 1 based on a tanh-sinh distribution
#         # Indices go from 0 to n-1
#         i = np.arange(n, dtype=float)
#         return 0.5 * (1.0 + np.tanh(beta*(2.0*i/(n-1) - 1.0)))
    
#     phi_vals_ = tanh_sinh_spacing(knots - 4, 14.0)
#     #Insert eps
#     phi_vals_ = np.insert(phi_vals_,0,1e-16)
#     phi_vals_ = np.insert(phi_vals_,0,eps)

#     phi_vals_ = np.append(phi_vals_,1.0-1e-16)

#     #Compute dfdphi vals
#     dfdphi = fh_deriv(phi_vals_,chi,N1,N2)

#     #compute eps_right
#     eps_right = mp.mpf('1') - mp.mpf(f'{eps}')

#     def df(phi):
#         return (1/N1) * mpmath.log(phi) + (1/N1) - (1/N2) * mpmath.log(1 - phi) - (1/N2) + chi - 2 * chi * phi
    
#     dfdphi = np.append(dfdphi, float(df(eps_right)))

#     #Update phi_vals
#     phi_vals = np.append(phi_vals_,1.0)
#     phi_vals[0] = 0.0

#     print(dfdphi)

#     spline = CubicSpline(phi_vals,dfdphi)
#     def df_spline(phi):
#         return spline(phi)
    
#     d2f_ = spline.derivative(1)
#     def d2f_spline(phi):
#         return d2f_(phi)
    
#     return df_spline, d2f_spline


def cahn_hilliard(ic_fun, chi, N1, N2, stride, tend, deltax, dt, return_data=False):
    #Simulation parameters
    Lx = Ly = 20.0
    nx = ny = int(Lx/deltax)
    kappa = (2/3)*chi

    max_iter = 1000
    rel_tol = 1e-6
    abs_tol = 1e-8

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
    df_spline, d2f_spline = spline_generator(chi,N1,N2,100)

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
    if return_data:
        writer = io.VTXWriter(domain.comm, "sim_spline.bp",[c],"BP4")
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
        if MPI.COMM_WORLD.rank == 0:
            print(f"Step {int(t/dt)}")
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
            if MPI.COMM_WORLD.rank == 0:
                print(f"    it# {iteration}: residual: {residual}, relative error: {rel_error}")

        #Update u0
        u0.x.array[:] = u.x.array[:]

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
        if return_data:
            if counter % stride == 0:
                writer.write(t)

    #Close VTK file
    if return_data:
        writer.close()

    if return_data:
        return tvals, phi_max_list, phi_min_list, phi_avg_list ,energy_list
    else:
        return
    

"""
Test Case
"""

def initial_condition(x):
    values = 0.5 + 0.02*(0.5-np.random.rand(x.shape[1]))
    return values

tvals, phi_max, phi_min, phi_avg, energy_vals = cahn_hilliard(initial_condition,chi=20,N1=1,N2=1,stride=10,tend=5,deltax=0.25,dt=0.005,return_data=True)
        
#Plot
fig, ax1 = plt.subplots()
ax1.plot(tvals,phi_max, label=r"$\phi_{1,\max}$",linestyle="--",color="blue")
ax1.plot(tvals,phi_min,label=r"$\phi_{1,\min}$",linestyle="-.",color="blue")
ax1.plot(tvals,phi_avg,label=r"$\bar{1,\phi}}$",linestyle="-",color="blue")
ax1.set_xlabel(r"Time ($\tilde{t}$)")
ax1.set_ylabel(r"$\phi_{1}$")
ax1.tick_params(axis='y', labelcolor='blue')         
ax1.yaxis.label.set_color('blue')
ax1.axhline(1.0,color="blue")
ax1.axhline(0.0,color="blue")
ax2 = ax1.twinx()
ax2.plot(tvals, energy_vals,linestyle="-", color="red")
ax2.set_ylabel("Total Energy")
ax2.tick_params(axis='y', labelcolor='red')
ax2.yaxis.label.set_color('red')
fig.tight_layout()

plt.show()