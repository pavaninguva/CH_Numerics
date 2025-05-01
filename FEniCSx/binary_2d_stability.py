import numpy as np
from binary_solver import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from multiprocessing import Pool, cpu_count
from functools import partial
import csv
import os

#formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')


"""
Running parameter sweep to see stability region
"""


def plot_dual_heatmaps_pcolormesh(csv_filename1, csv_filename2):
    """
    Reads results from two CSV files (each with columns: chi, deltax, critical_dt)
    and creates two subplots (one for each CSV file) using pcolormesh with gouraud shading.
    The subplots share a common colorbar.
    
    This function independently extracts the unique chi and deltax values from each CSV file.
    It then computes a common vmin/vmax from both datasets.
    """
    # Read results from each CSV file using the helper function.
    results1_dict = read_existing_results(csv_filename1)
    results2_dict = read_existing_results(csv_filename2)
    
    # For CSV file 1, extract unique chi and deltax values.
    chi_vals1 = sorted({chi for (chi, _) in results1_dict.keys()})
    deltax_vals1 = sorted({deltax for (_, deltax) in results1_dict.keys()})
    res_list1 = [(chi, deltax, dt) for (chi, deltax), dt in results1_dict.items()]
    heatmap1 = build_heatmap(chi_vals1, deltax_vals1, res_list1)
    
    # For CSV file 2, extract unique chi and deltax values.
    chi_vals2 = sorted({chi for (chi, _) in results2_dict.keys()})
    deltax_vals2 = sorted({deltax for (_, deltax) in results2_dict.keys()})
    res_list2 = [(chi, deltax, dt) for (chi, deltax), dt in results2_dict.items()]
    heatmap2 = build_heatmap(chi_vals2, deltax_vals2, res_list2)
    
    # Determine common color scale from both heatmaps.
    vmin = min(np.nanmin(heatmap1), np.nanmin(heatmap2))
    vmax = max(np.nanmax(heatmap1), np.nanmax(heatmap2))

    # Determine x and y lims.
    chimin = min(np.nanmin(chi_vals1), np.nanmin(chi_vals2))
    chimax = max(np.nanmax(chi_vals1), np.nanmax(chi_vals2))

    dxmin = min(np.nanmin(deltax_vals1), np.nanmin(deltax_vals2))
    dxmax = max(np.nanmax(deltax_vals1), np.nanmax(deltax_vals2))
    
    # Create a meshgrid for each dataset.
    X1, Y1 = np.meshgrid(deltax_vals1, chi_vals1)
    X2, Y2 = np.meshgrid(deltax_vals2, chi_vals2)


    
    # Create subplots.
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot for CSV file 1.
    im1 = axs[0].pcolormesh(X1, Y1, heatmap1, shading='auto', cmap="viridis", vmin=vmin, vmax=vmax,norm="log")
    axs[0].set_title('Full')
    axs[0].set_xlabel(r'$\Delta x$')
    axs[0].set_ylabel(r'$\chi_{12}$')
    axs[0].set_xlim((dxmin,dxmax))
    axs[0].set_ylim((chimin,chimax))
    
    # Plot for CSV file 2.
    im2 = axs[1].pcolormesh(X2, Y2, heatmap2, shading='auto', cmap="viridis", vmin=vmin, vmax=vmax,norm="log")
    axs[1].set_title('Spline')
    axs[1].set_xlabel(r'$\Delta x$')
    axs[1].set_ylabel(r'$\chi_{12}$')
    axs[1].set_xlim((dxmin,dxmax))
    axs[1].set_ylim((chimin,chimax))
    
    # Create one common colorbar for both subplots.
    cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(im2, cax=cbar_ax, orientation='vertical', label=r' $\min(\Delta t)$')
    
    plt.tight_layout(rect=[0, 0, 0.93, 1])
    plt.show()

def get_tend_for_chi(chi):
    if 3 <= chi <= 5:
        return 50.0
    elif 5 < chi <= 8:
        return 20.0
    elif 8 < chi <= 10:
        return 15.0
    elif 10 < chi <= 15:
        return 10.0
    elif 15 < chi <=18:
        return 7.0
    else:
        return 5.0


def find_critical_dt(sim_func,ic_fun, chi, N1, N2, stride, tend, deltax, dt_start=0.5, dt_min=1e-4):
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
            sim_func(ic_fun, chi, N1, N2, stride, tend, deltax, dt, return_data=False)
        except Exception as e:
            # Simulation failed; optionally print a message and try a smaller dt.
            print(f"chi={chi}, deltax={deltax}, dt={dt} failed with error: {e}")
            dt = dt / 2.0
            continue
        # Success!
        print(f"chi={chi}, deltax={deltax} succeeded with dt = {dt}")
        return dt
    return np.nan

def worker(params, sim_func):
    """
    Worker function for multiprocessing.
    Expects a tuple:
      (chi, deltax, dt_start, dt_min, ic_fun, N1, N2, stride, tend)
    Returns (chi, deltax, critical_dt).
    """
    chi, deltax, dt_start, dt_min, ic_fun, N1, N2, stride = params
    tend = get_tend_for_chi(chi)
    critical_dt = find_critical_dt(sim_func,ic_fun, chi, N1, N2, stride, tend, deltax, dt_start, dt_min)
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

def build_heatmap(chi_values, deltax_values, results_list):
    """
    Given arrays chi_values and deltax_values and a list of tuples
    [(chi, deltax, critical_dt), ...], build and return a heatmap array.
    """
    heatmap = np.empty((len(chi_values), len(deltax_values)))
    heatmap[:] = np.nan  # initialize with NaNs

    # Build a lookup dictionary for easier access.
    res_dict = {(float(chi), float(deltax)): dt for chi, deltax, dt in results_list}

    for i, chi in enumerate(chi_values):
        for j, deltax in enumerate(deltax_values):
            key = (chi, deltax)
            if key in res_dict:
                heatmap[i, j] = res_dict[key]
    return heatmap


def run_parameter_study_mp(chi_values, deltax_values, dt_start, dt_min,
                           ic_fun, N1, N2, stride, num_workers, csv_filename, sim_func):
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
        worker_func = partial(worker, sim_func=sim_func)
        with Pool(processes=num_workers) as pool:
            new_results = pool.map(worker_func, param_list)
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
    return combined_results
    


if __name__ == '__main__':
    # Define simulation-specific parameters locally.
    N1 = 1
    N2 = 1
    stride = 10

    def ic_fun(x):
        values = 0.5 + 0.02*(0.5-np.random.rand(x.shape[1]))
        return values
    
    # Define the parameter ranges.
    chi_values = np.array([3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0])  
    deltax_values = np.array([0.1,0.16,0.2,0.25,0.4,0.5,0.8])    

    # Run the parameter study.
    spline_csv = "./2d_spline_min_dt_fenics.csv"

    analytical_csv = "./2d_full_min_dt_fenics.csv"

    run_parameter_study_mp(chi_values, deltax_values,
                           dt_start=0.5, dt_min=1e-4,
                           ic_fun=ic_fun, N1=N1, N2=N2, stride=stride,
                           num_workers=5, csv_filename=spline_csv,sim_func=cahn_hilliard_spline)
    
    # run_parameter_study_mp(chi_values, deltax_values,
    #                        dt_start=(1.0/8192), dt_min=1e-4,
    #                        ic_fun=ic_fun, N1=N1, N2=N2, stride=stride,
    #                        num_workers=4, csv_filename=analytical_csv,sim_func=cahn_hilliard_analytical)
    
    plot_dual_heatmaps_pcolormesh(analytical_csv,spline_csv)
