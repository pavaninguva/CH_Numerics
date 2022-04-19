import random
from tkinter import Variable
from dolfin import *
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
nx = 100

#Initial Composition
phi_0 = 0.5
#Noise Magnitude
noise_mag = 0.03
#FH Parameters
Nchi = 20.0

#Simulation time
t_end = 1.0
dt = 0.001
#Theta is for timestepping: 1 for Backward Euler, 0.5 for Crank-Nicolson
theta = 1.0

parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True

"""
Specify Problem
"""
#Specify Mesh
mesh = IntervalMesh(nx, 0.0,Lx)

P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
CH = FunctionSpace(mesh, P1*P1)

dch = TrialFunction(CH)
h_1, j_1 = TestFunctions(CH)

ch = Function(CH)
ch0 = Function(CH)

# Split mixed functions
da, dmu = split(dch)
phi, mu = split (ch)
phi0, mu0 = split(ch0)

class CahnHilliardEquation(NonlinearProblem):
    def __init__(self, a, L):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
        # self.reset_sparsity = True
    def F(self, b, x):
        assemble(self.L, tensor=b)
    def J(self, A, x):
        assemble(self.a, tensor=A)

class InitialConditions(UserExpression):
    def __init__(self, **kwargs):
        random.seed(2 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)
    def eval(self, values, x):
        # [0] corresponds to the concentration field for species 1
        # [1] coresponds to the mu field
        #Uniform with noise
        # values[0] = phi_0 + 2.0*noise_mag*(0.5 - random.random())
        #Step function
        if between(x[0],(0.0,Lx/2)):
            values[0] = 0.2
        else:
            values[0] = 0.8
        values[1] = 0.0
    def value_shape(self):
        return (2,)

ch_init = InitialConditions(degree=1)
ch.interpolate(ch_init)
ch0.interpolate(ch_init)

phi = variable(phi)

#Declare Equation
g = phi*ln(phi) + (1.0-phi)*ln(1.0-phi) + Nchi*phi*(1.0-phi)
dgdphi = diff(g,phi)
kappa = (2.0/3.0)*Nchi

mu_mid = (1.0 - theta)*mu0 + theta*mu

F_phi = (
    phi*h_1*dx 
    - phi0*h_1*dx 
    + dt*phi*(1.0-phi)*dot(grad(mu_mid),grad(h_1))*dx 
)

F_mu = (
    mu*j_1*dx 
    - dgdphi*j_1*dx 
    - kappa*dot(grad(phi),grad(j_1))*dx 
)

F = F_phi + F_mu

#Compute Jacobian
a = derivative(F,ch,dch)

"""
Configure Solver
"""
class CustomSolver(NewtonSolver):

        def __init__(self):
            NewtonSolver.__init__(self, mesh.mpi_comm(),
                                PETScKrylovSolver(), PETScFactory.instance())

        def solver_setup(self, A, P, problem, iteration):
            self.linear_solver().set_operator(A)

            PETScOptions.set("ksp_type", "gmres")
            PETScOptions.set("ksp_monitor")
            PETScOptions.set("pc_type", "hypre")
            PETScOptions.set("pc_hypre_type", "euclid")
            PETScOptions.set("ksp_rtol", "1.0e-8")
            PETScOptions.set("ksp_atol", "1.0e-16")
            PETScOptions.set('ksp_max_it', '1000')

            self.linear_solver().set_from_options()

problem = CahnHilliardEquation(a,F)
solver = CustomSolver()

"""
Perform Solution
"""
# file = File("output.pvd", "compressed")
t = 0.0
while t < t_end -1e-8:
    #Update time
    t = t+dt
    print("Current Simulation Time is %s"%t)
    ch0.vector()[:] = ch.vector()
    solver.solve(problem, ch.vector())
    # file << (ch.split()[0], t)


"""
Plotting
"""

phi_vals = ch.vector().get_local()[::2]
x_vals = mesh.coordinates()[:,0]

#Plot solution at final time
fig1 = plt.figure(num=1)
plt.plot(x_vals,phi_vals)
plt.xlabel(r"$\tilde{x}$")
plt.ylabel(r"$\phi$")
# plt.ylim((0.0,1.0))
plt.tight_layout()

plt.show()




