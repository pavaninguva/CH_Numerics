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
from scipy import interpolate
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

#formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')

# Read the binodal data
binodal_df = pd.read_csv('binodal_data.csv')

# Read the spinodal data
spinodal_df = pd.read_csv('spinodal_data.csv')

# Read the critical point data
critical_point_df = pd.read_csv('critical_point.csv')

def cahn_hilliard(a0,chi=6,stride=20,output="none", save_image=True, scheme="spline"):
    #Define time parameters
    t = 0.0
    tend = 100.0
    nsteps = 10000 
    dt = (tend-t)/nsteps
    max_iter = 1000
    rel_tol = 1e-6
    abs_tol = 1e-8

    #Model parameters
    kappa = (2/3)*chi

    #Define mesh
    nx = ny = 50
    domain = mesh.create_rectangle(MPI.COMM_WORLD, [[0.0,0.0],[20.0, 20.0]], [nx,ny])

    #Define model
    P1 = element("Lagrange",domain.basix_cell(),1)
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

    #Use external operator to input spline approximation
    #Compute spline_pot
    def dfdphi(c):
        return -2*chi*c + c - np.log(1-c) + np.log(c)
    
    spline_pot = interpolate.CubicSpline(np.linspace(0,1,200),dfdphi(np.linspace(1e-16,1-1e-16,200)))
    
    quadrature_degree = 1
    Qe=quadrature_element(domain.topology.cell_name(), degree=quadrature_degree, value_shape=())
    Q = fem.functionspace(domain, Qe)
    dx = Measure(
    "dx",
    metadata={"quadrature_scheme": "default", "quadrature_degree": 1},
)
    dfdc = FEMExternalOperator(c,function_space=Q)

    def dfdc_impl(c):
        output = spline_pot(c)
        return output.reshape(-1)
    
    def d2fd2c_impl(c):
        output = spline_pot.derivative(1)(c)
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

    def initial_condition(x):
        values = a0 + 0.02*(0.5-np.random.rand(x.shape[1]))
        return values
    
    J = derivative(F, u, me)
    J_expanded = ufl.algorithms.expand_derivatives(J)
    F_replaced, F_external_operators = replace_external_operators(F)
    J_replaced, J_external_operators = replace_external_operators(J_expanded)
    evaluated_operands = evaluate_operands(F_external_operators)
    _ = evaluate_external_operators(F_external_operators, evaluated_operands)
    _ = evaluate_external_operators(J_external_operators, evaluated_operands)
    
    #Apply IC
    u.x.array[:] = 0.0
    u.sub(0).interpolate(initial_condition)
    u.x.scatter_forward()
    # c = u.sub(0)
    u0.x.array[:] = u.x.array

    #Define solver file
    if output == "bp":
        filename = str(a0)+ ".bp"
        writer = io.VTXWriter(domain.comm, filename,[c.collapse()],"BP4")
        writer.write(0.0)
    elif output == "np":
        # #Get coordinates of DOFs
        # coordinates = ME.sub(0).collapse()[0].tabulate_dof_coordinates()
        # #Set up concentration
        # output_file = np.zeros((np.size(u.sub(0).collapse().x.array),int(nsteps/stride)+1))
        # #Set up time vals
        # tvals = np.zeros(int(nsteps/stride)+1)
        # #Insert ICs to array
        # tvals[0] = 0.0
        # output_file[:,0] = c.collapse().x.array
        # Create directory to save data if it doesn't exist
        data_dir = "simulation_data"
        os.makedirs(data_dir, exist_ok=True)

        # Filenames for data files
        filename_prefix = f"a0_{a0}_chi_{chi}"
        tvals_filename = os.path.join(data_dir, f"{filename_prefix}_tvals.csv")
        coords_filename = os.path.join(data_dir, f"{filename_prefix}_coordinates.csv")
        output_filename = os.path.join(data_dir, f"{filename_prefix}_output.csv")

        # Check if data files already exist
        if os.path.isfile(tvals_filename) and os.path.isfile(coords_filename) and os.path.isfile(output_filename):
            print(f"Data files for a0={a0}, chi={chi} already exist. Skipping simulation.")
            return

        # Get coordinates of DOFs
        coordinates = ME.sub(0).collapse()[0].tabulate_dof_coordinates()
        # Set up concentration array
        num_time_steps = int(nsteps / stride) + 1
        output_file = np.zeros((coordinates.shape[0], num_time_steps))
        # Set up time values array
        tvals = np.zeros(num_time_steps)
        # Insert initial conditions to arrays
        tvals[0] = t
        output_file[:, 0] = u.sub(0).collapse().x.array
    elif output == "none":
        pass

    #Setup solver
    problem = LinearProblem(J_replaced, -F_replaced, du)

    #Introduce skipping for output to output only every nth step
    stride = stride
    counter = 0

    #Timestepping
    while t < tend:
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

        u0.x.array[:] = u.x.array[:]
        counter = counter +1
        if counter % stride == 0:
            print("beep")
            if output == "bp":
                writer.write(t)
            elif output == "np":
                output_file[:,int(counter/stride)] = u.sub(0).collapse().x.array
                print(counter/stride)
                tvals[int(counter/stride)] = t
            elif output == "none":
                pass

    # Save data when output == "np"
    if output == "np":
        # Create filenames including a0 and chi values
        filename_prefix = f"a0_{a0:.4f}_chi_{chi:.4f}"
        # Save tvals
        tvals_filename = os.path.join(data_dir, f"{filename_prefix}_tvals.csv")
        np.savetxt(tvals_filename, tvals, delimiter=",")
        # Save coordinates
        coords_filename = os.path.join(data_dir, f"{filename_prefix}_coordinates.csv")
        np.savetxt(coords_filename, coordinates, delimiter=",")
        # Save output_file (concentration data)
        output_filename = os.path.join(data_dir, f"{filename_prefix}_output.csv")
        np.savetxt(output_filename, output_file, delimiter=",")
        print(f"Data saved for a0={a0}, chi={chi}")
    
    # Save image at final time step if requested
    if save_image:
        # Extract data for plotting
        coords = ME.sub(0).collapse()[0].tabulate_dof_coordinates()
        xvals = coords[:, 0]
        yvals = coords[:, 1]
        c_vals = u.sub(0).collapse().x.array

        # Set color range between 0 and 1
        cmap = plt.cm.viridis

        # Plot with tripcolor
        fig1, ax1 = plt.subplots(figsize=(3, 3))
        mesh_plot = ax1.tripcolor(xvals, yvals, c_vals, shading='gouraud', cmap=cmap, vmin=0, vmax=1)
        ax1.set_axis_off()
        fig1.tight_layout(pad=0)

        # Create directory to save images if it doesn't exist
        image_dir = "simulation_images"
        os.makedirs(image_dir, exist_ok=True)

        # Save the figure with a filename encoding the parameters
        filename = f"chi_{chi}_a0_{a0}.png"
        filepath = os.path.join(image_dir, filename)
        fig1.savefig(filepath, dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close(fig1)

    if output == "bp":
        writer.close()
    elif output == "np":
        return tvals, coordinates, output_file
    elif output == "none":
        return None
    
"""
Binodal- Spinodal Plot with Simulations at t=100 overlayed
"""
    
# Create the binodal-spinodal plot
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(binodal_df['phi1'], binodal_df['chi_inv'], color='blue', label='Binodal')
ax.plot(binodal_df['phi2'], binodal_df['chi_inv'], color='blue')
ax.plot(spinodal_df['phi'], spinodal_df['chi_c_over_chi'], color='red', label='Spinodal')
ax.scatter(critical_point_df['phi'], critical_point_df['chi_inv'], color='black', label='Critical Point')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.set_xlabel('$\\phi$',fontsize=18)
ax.set_ylabel('$\\frac{\\chi_c}{\\chi}$',fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.tick_params(axis='both', which='minor', labelsize=14)
ax.legend(fontsize=12)

# Critical chi value (from your Julia code)
chi_c = 2  # For N1 = N2 = 1

# Define the chi and a0 values you want to include
chi_values = [14,8, 4, 3]
a0_values = [0.1, 0.35, 0.5, 0.75, 0.9]

# Directory where images are saved
image_dir = "simulation_images"

# Ensure the image directory exists
os.makedirs(image_dir, exist_ok=True)

# Loop over the chi and a0 values
for chi in chi_values:
    for a0 in a0_values:
        # Build the filename
        filename = f"chi_{chi}_a0_{a0}.png"
        filepath = os.path.join(image_dir, filename)

        # Check if the image file exists
        if os.path.isfile(filepath):
            print(f"Image file {filepath} found. Loading image.")
        else:
            print(f"Image file {filepath} not found. Running simulation.")
            # Run the simulation to generate the image
            cahn_hilliard(a0, chi=chi)
            # Verify that the image was created
            if not os.path.isfile(filepath):
                print(f"Error: Image file {filepath} was not created.")
                continue

        # Read the image
        image = plt.imread(filepath)

        # Calculate chi_c / chi for placement on the plot
        chi_c_over_chi = chi_c / chi

        # Create an OffsetImage
        imagebox = OffsetImage(image, zoom=0.15)

        # Create an AnnotationBbox at the correct location
        ab = AnnotationBbox(imagebox, (a0, chi_c_over_chi), frameon=False)

        # Add the AnnotationBbox to the plot
        ax.add_artist(ab)

fig.savefig("data_vis.png",dpi=300)

"""
Run for generating dataset
"""
chi_val = [6]
a0_vals = np.linspace(0.1,0.9,100).tolist()

# Directory where data files are saved
data_dir = "simulation_data"

# Ensure the data directory exists
os.makedirs(data_dir, exist_ok=True)

for chi in chi_val:
    for a0 in a0_vals:
        # Format a0 and chi to fixed decimal places to avoid floating-point issues
        a0_formatted = f"{a0:.4f}"
        chi_formatted = f"{chi:.4f}"
        # Create filename prefix for this a0 and chi
        filename_prefix = f"a0_{a0_formatted}_chi_{chi_formatted}"
        # Check if data files exist
        tvals_filename = os.path.join(data_dir, f"{filename_prefix}_tvals.csv")
        coords_filename = os.path.join(data_dir, f"{filename_prefix}_coordinates.csv")
        output_filename = os.path.join(data_dir, f"{filename_prefix}_output.csv")
        
        if (os.path.isfile(tvals_filename) and os.path.isfile(coords_filename) and os.path.isfile(output_filename)):
            print(f"Data files for a0={a0}, chi={chi} already exist. Skipping simulation.")
        else:
            print(f"Running simulation for a0={a0}, chi={chi}")
            cahn_hilliard(a0, chi=chi, output="np", save_image=False)



plt.show()