import numpy as np
from mpi4py import MPI
import basix
import ufl
from basix.ufl import element, mixed_element
from petsc4py import PETSc
from dolfinx import fem, mesh, io, log
from ufl import grad, inner, ln, dx
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

import matplotlib.pyplot as plt

from multiprocessing import Pool, cpu_count
from functools import partial
import csv
import os


#formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')

"""
This script implements a simple Backwards Euler method for solving
the binary CH equation in mixed form
"""

def cahn_hilliard(ic_fun, chi, N1, N2, stride, tend, deltax, dt, return_data=False):
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
    if return_data:
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
Test case
"""
# run_test_case = True
# def initial_condition(x):
#     values = 0.5 + 0.02*(0.5-np.random.rand(x.shape[1]))
#     return values

# if run_test_case:

#     tvals, phi_max, phi_min, phi_avg, energy_vals = cahn_hilliard(initial_condition,chi=6,N1=1,N2=1,stride=1,tend=100,deltax=1.0,dt=0.05,return_data=True)
        
#     #Plot
#     fig, ax1 = plt.subplots()
#     ax1.plot(tvals,phi_max, label=r"$\phi_{1,\max}$",linestyle="--",color="blue")
#     ax1.plot(tvals,phi_min,label=r"$\phi_{1,\min}$",linestyle="-.",color="blue")
#     ax1.plot(tvals,phi_avg,label=r"$\bar{1,\phi}}$",linestyle="-",color="blue")
#     ax1.set_xlabel(r"Time ($\tilde{t}$)")
#     ax1.set_ylabel(r"$\phi_{1}$")
#     ax1.tick_params(axis='y', labelcolor='blue')         
#     ax1.yaxis.label.set_color('blue')
#     ax1.axhline(1.0,color="blue")
#     ax1.axhline(0.0,color="blue")
#     ax2 = ax1.twinx()
#     ax2.plot(tvals, energy_vals,linestyle="-", color="red")
#     ax2.set_ylabel("Total Energy")
#     ax2.tick_params(axis='y', labelcolor='red')
#     ax2.yaxis.label.set_color('red')
#     fig.tight_layout()


"""
Running parameter sweep to see stability region
"""

def get_tend_for_chi(chi):
    if 3 <= chi <= 5:
        return 50.0
    elif 5 < chi <= 8:
        return 20.0
    elif 8 < chi <= 10:
        return 15.0
    else:
        return 10.0


def find_critical_dt(ic_fun, chi, N1, N2, stride, tend, deltax, dt_start=0.5, dt_min=1e-4):
    """
    For a given set of parameters (chi and deltax), try running the simulation
    starting at dt_start. If it fails (raises an exception), halve dt and try again,
    until dt falls below dt_min. Return the dt value at which the simulation first
    succeeds. If none succeed, return np.nan.
    """
    dt = dt_start
    while dt >= dt_min:
        try:
            # Run the simulation with current dt.
            # Set return_data=False to keep things fast.
            cahn_hilliard(ic_fun, chi, N1, N2, stride, tend, deltax, dt, return_data=False)
        except Exception as e:
            # Simulation failed; optionally print a message and try a smaller dt.
            print(f"chi={chi}, deltax={deltax}, dt={dt} failed with error: {e}")
            dt = dt / 2.0
            continue
        # Success!
        print(f"chi={chi}, deltax={deltax} succeeded with dt = {dt}")
        return dt
    return np.nan

def worker(params):
    """
    Worker function for multiprocessing.
    Expects a tuple:
      (chi, deltax, dt_start, dt_min, ic_fun, N1, N2, stride, tend)
    Returns (chi, deltax, critical_dt).
    """
    chi, deltax, dt_start, dt_min, ic_fun, N1, N2, stride = params
    tend = get_tend_for_chi(chi)
    critical_dt = find_critical_dt(ic_fun, chi, N1, N2, stride, tend, deltax, dt_start, dt_min)
    return (chi, deltax, critical_dt)

def read_existing_results(csv_filename):
    """
    If the CSV file exists, read it and return a dictionary with keys (chi, deltax)
    and values critical_dt. Otherwise, return an empty dictionary.
    """
    existing = {}
    if os.path.exists(csv_filename):
        with open(csv_filename, mode="r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Convert keys to float.
                chi_val = float(row["chi"])
                deltax_val = float(row["deltax"])
                dt_val = float(row["critical_dt"])
                existing[(chi_val, deltax_val)] = dt_val
    return existing

def append_results_to_csv(csv_filename, results):
    """
    Append a list of tuples (chi, deltax, critical_dt) to the CSV file.
    If the file does not exist, write a header first.
    """
    file_exists = os.path.exists(csv_filename)
    with open(csv_filename, mode="a", newline="") as csvfile:
        fieldnames = ["chi", "deltax", "critical_dt"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for chi, deltax, dt_val in results:
            writer.writerow({"chi": chi, "deltax": deltax, "critical_dt": dt_val})

def plot_heatmap_from_results(chi_values, deltax_values, results):
    """
    Given arrays chi_values and deltax_values and a dictionary of results
    {(chi, deltax): critical_dt, ...}, create and display a heatmap.
    """
    heatmap = np.empty((len(chi_values), len(deltax_values)))
    heatmap[:] = np.nan  # initialize with NaNs

    # Build a lookup dictionary for easier access
    res_dict = {(float(chi), float(deltax)): dt for chi, deltax, dt in results}

    for i, chi in enumerate(chi_values):
        for j, deltax in enumerate(deltax_values):
            key = (chi, deltax)
            if key in res_dict:
                heatmap[i, j] = res_dict[key]
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap, extent=[deltax_values[0], deltax_values[-1],
                                chi_values[0], chi_values[-1]],
               origin='lower', aspect='auto', cmap='viridis')
    plt.xlabel('deltax')
    plt.ylabel('chi')
    plt.title('Critical dt for Successful Simulation')
    cbar = plt.colorbar()
    cbar.set_label('Critical dt')
    plt.show()


def run_parameter_study_mp(chi_values, deltax_values, dt_start, dt_min,
                           ic_fun, N1, N2, stride, num_workers=4, csv_filename="./2d_full_min_dt_fenics.csv"):
    """
    For each (chi, deltax) combination, determine the minimum dt (starting from dt_start)
    that allows the simulation to run successfully. The results are saved/appended to a CSV file.
    Then, a heatmap is generated where the x-axis is deltax, y-axis is chi, and the color corresponds
    to the critical dt.
    """
    # Read existing results from CSV if available.
    existing_results = read_existing_results(csv_filename)
    
    # Build a list of parameter tuples for every combination that hasn't been run yet.
    param_list = []
    for chi in chi_values:
        for deltax in deltax_values:
            key = (float(chi), float(deltax))
            if key not in existing_results:
                param_list.append((chi, deltax, dt_start, dt_min, ic_fun, N1, N2, stride))
    
    print(f"Total new combinations to test: {len(param_list)}")
    
    new_results = []
    if param_list:
        with Pool(processes=num_workers) as pool:
            new_results = pool.map(worker, param_list)
        # Append the new results to CSV.
        append_results_to_csv(csv_filename, new_results)
    else:
        print("No new parameter combinations to run; all already in CSV.")

    # Combine new results with existing results for plotting.
    combined_results = []
    # Add existing results.
    for (chi, deltax), dt_val in existing_results.items():
        combined_results.append((chi, deltax, dt_val))
    # Add new results.
    combined_results.extend(new_results)
    
    # Plot heatmap.
    plot_heatmap_from_results(chi_values, deltax_values, combined_results)


if __name__ == '__main__':
    # Define simulation-specific parameters locally.
    N1 = 1
    N2 = 1
    stride = 10

    def ic_fun(x):
        values = 0.5 + 0.02*(0.5-np.random.rand(x.shape[1]))
        return values
    
    # Define the parameter ranges.
    chi_values = np.array([3.0,4.0,5.0,6.0,7.0,8.0])       # Example: 10, 20, 30, 40, 50
    deltax_values = np.array([0.16,0.2,0.25,0.4,0.5])    

    # Run the parameter study.
    run_parameter_study_mp(chi_values, deltax_values,
                           dt_start=0.5, dt_min=1e-4,
                           ic_fun=ic_fun, N1=N1, N2=N2, stride=stride,
                           num_workers=4, csv_filename="./2d_full_min_dt_fenics.csv")
