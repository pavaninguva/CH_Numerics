import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pandas as pd

from binary_solver import *

rng = np.random.RandomState(0)


#formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')


"""
Test Case
"""

def initial_condition(x):
    values = 0.5 + 0.02*(0.5-rng.rand(x.shape[1]))
    return values

#Run Simulations

tvals1, phi_max1, phi_min1, phi_avg1, energy_vals1 = cahn_hilliard_analytical(initial_condition,chi=6,N1=1,N2=1,stride=10,tend=30,deltax=0.4,dt=0.1,return_data=True, return_vtk=False)
tvals2, phi_max2, phi_min2, phi_avg2, energy_vals2 = cahn_hilliard_analytical(initial_condition,chi=6,N1=1,N2=1,stride=10,tend=30,deltax=0.2,dt=0.05,return_data=True, return_vtk=False)
tvals3, phi_max3, phi_min3, phi_avg3, energy_vals3 = cahn_hilliard_analytical(initial_condition,chi=6,N1=1,N2=1,stride=10,tend=30,deltax=0.1,dt=0.025,return_data=True, return_vtk=True)

# Export as csv

df_analytical1 = pd.DataFrame({
    "tvals": tvals1,
    "phi_max": phi_max1,
    "phi_min":phi_min1,
    "phi_avg": phi_avg1,
    "energy_vals":energy_vals1
})

df_analytical2 = pd.DataFrame({
    "tvals": tvals2,
    "phi_max": phi_max2,
    "phi_min":phi_min2,
    "phi_avg": phi_avg2,
    "energy_vals":energy_vals2
})

df_analytical3 = pd.DataFrame({
    "tvals": tvals3,
    "phi_max": phi_max3,
    "phi_min":phi_min3,
    "phi_avg": phi_avg3,
    "energy_vals":energy_vals3
})

df_analytical1.to_csv("./binary_analytical_2d_dx_04_dt_01.csv")
df_analytical2.to_csv("./binary_analytical_2d_dx_02_dt_005.csv")
df_analytical3.to_csv("./binary_analytical_2d_dx_01_dt_0025.csv")

#Read back 

# df_analytical1 = pd.read_csv("./binary_analytical_2d_dx_04_dt_01.csv")
# tvals1       = df_analytical1["tvals"].to_numpy()
# phi_max1     = df_analytical1["phi_max"].to_numpy()
# phi_min1     = df_analytical1["phi_min"].to_numpy()
# phi_avg1     = df_analytical1["phi_avg"].to_numpy()
# energy_vals1 = df_analytical1["energy_vals"].to_numpy()

# df_analytical2 = pd.read_csv("./binary_analytical_2d_dx_02_dt_005.csv")
# tvals2       = df_analytical2["tvals"].to_numpy()
# phi_max2     = df_analytical2["phi_max"].to_numpy()
# phi_min2     = df_analytical2["phi_min"].to_numpy()
# phi_avg2     = df_analytical2["phi_avg"].to_numpy()
# energy_vals2 = df_analytical2["energy_vals"].to_numpy()

# df_analytical3 = pd.read_csv("./binary_analytical_2d_dx_01_dt_0025.csv")
# tvals3       = df_analytical3["tvals"].to_numpy()
# phi_max3     = df_analytical3["phi_max"].to_numpy()
# phi_min3     = df_analytical3["phi_min"].to_numpy()
# phi_avg3     = df_analytical3["phi_avg"].to_numpy()
# energy_vals3 = df_analytical3["energy_vals"].to_numpy()


tvals1_spline, phi_max1_spline, phi_min1_spline, phi_avg1_spline, energy_vals1_spline = cahn_hilliard_spline(initial_condition,chi=6,N1=1,N2=1,stride=10,tend=30,deltax=0.4,dt=0.1,return_data=True, return_vtk=False)
tvals2_spline, phi_max2_spline, phi_min2_spline, phi_avg2_spline, energy_vals2_spline = cahn_hilliard_spline(initial_condition,chi=6,N1=1,N2=1,stride=10,tend=30,deltax=0.2,dt=0.05,return_data=True, return_vtk=False)
tvals3_spline, phi_max3_spline, phi_min3_spline, phi_avg3_spline, energy_vals3_spline = cahn_hilliard_spline(initial_condition,chi=6,N1=1,N2=1,stride=10,tend=30,deltax=0.1,dt=0.025,return_data=True, return_vtk=True)

# cahn_hilliard_spline(initial_condition,chi=6,N1=1,N2=1,stride=10,tend=50,deltax=0.1,dt=0.025,return_data=False, return_vtk=True)

df_spline1 = pd.DataFrame({
    "tvals": tvals1_spline,
    "phi_max": phi_max1_spline,
    "phi_min":phi_min1_spline,
    "phi_avg": phi_avg1_spline,
    "energy_vals":energy_vals1_spline
})

df_spline2 = pd.DataFrame({
    "tvals": tvals2_spline,
    "phi_max": phi_max2_spline,
    "phi_min":phi_min2_spline,
    "phi_avg": phi_avg2_spline,
    "energy_vals":energy_vals2_spline
})

df_spline3 = pd.DataFrame({
    "tvals": tvals3_spline,
    "phi_max": phi_max3_spline,
    "phi_min":phi_min3_spline,
    "phi_avg": phi_avg3_spline,
    "energy_vals":energy_vals3_spline
})

df_spline1.to_csv("./binary_spline_2d_dx_04_dt_01.csv")
df_spline2.to_csv("./binary_spline_2d_dx_02_dt_005.csv")
df_spline3.to_csv("./binary_spline_2d_dx_01_dt_0025.csv")

# df_spline1 = pd.read_csv("./binary_spline_2d_dx_04_dt_01.csv")
# tvals1_spline       = df_spline1["tvals"].to_numpy()
# phi_max1_spline     = df_spline1["phi_max"].to_numpy()
# phi_min1_spline     = df_spline1["phi_min"].to_numpy()
# phi_avg1_spline     = df_spline1["phi_avg"].to_numpy()
# energy_vals1_spline = df_spline1["energy_vals"].to_numpy()

# df_spline2 = pd.read_csv("./binary_spline_2d_dx_02_dt_005.csv")
# tvals2_spline       = df_spline2["tvals"].to_numpy()
# phi_max2_spline     = df_spline2["phi_max"].to_numpy()
# phi_min2_spline     = df_spline2["phi_min"].to_numpy()
# phi_avg2_spline     = df_spline2["phi_avg"].to_numpy()
# energy_vals2_spline = df_spline2["energy_vals"].to_numpy()

# df_spline3 = pd.read_csv("./binary_spline_2d_dx_01_dt_0025.csv")
# tvals3_spline       = df_spline3["tvals"].to_numpy()
# phi_max3_spline     = df_spline3["phi_max"].to_numpy()
# phi_min3_spline     = df_spline3["phi_min"].to_numpy()
# phi_avg3_spline     = df_spline3["phi_avg"].to_numpy()
# energy_vals3_spline = df_spline3["energy_vals"].to_numpy()


#Plot
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(8,4))
ax1.plot(tvals1,phi_avg1,label=r"$\Delta x = 0.4, \Delta t = 0.1$",linestyle="-",color="blue")
ax1.plot(tvals2,phi_avg2,label=r"$\Delta x = 0.2, \Delta t = 0.05$",linestyle="-.",color="blue")
ax1.plot(tvals3,phi_avg3,label=r"$\Delta x = 0.1, \Delta t = 0.025$",linestyle=":",color="blue")
ax1.set_xlabel(r"Time ($\tilde{t}$)")
ax1.set_ylabel(r"$\phi_{1}$")
ax1.tick_params(axis='y', labelcolor='blue')         
ax1.yaxis.label.set_color('blue')
ax1.set_ylim(0.45,0.55)
ax1.set_title("Full")
ax1.legend()
ax12 = ax1.twinx()
ax12.plot(tvals1, energy_vals1,linestyle="-", color="red")
ax12.plot(tvals2, energy_vals2,linestyle="-.", color="red")
ax12.plot(tvals3, energy_vals3,linestyle=":", color="red")
ax12.set_ylabel("Energy")
ax12.tick_params(axis='y', labelcolor='red')
ax12.yaxis.label.set_color('red')

ax2.plot(tvals1_spline,phi_avg1_spline,label=r"$\Delta x = 0.4, \Delta t = 0.1$",linestyle="-",color="blue")
ax2.plot(tvals2_spline,phi_avg2_spline,label=r"$\Delta x = 0.2, \Delta t = 0.05$",linestyle="-.",color="blue")
ax2.plot(tvals3_spline,phi_avg3_spline,label=r"$\Delta x = 0.1, \Delta t = 0.025$",linestyle=":",color="blue")
ax2.set_xlabel(r"Time ($\tilde{t}$)")
ax2.set_ylabel(r"$\phi_{1}$")
ax2.tick_params(axis='y', labelcolor='blue')         
ax2.yaxis.label.set_color('blue')
ax2.set_ylim(0.45,0.55)
ax2.set_title("Spline")
ax2.legend()
ax22 = ax2.twinx()
ax22.plot(tvals1_spline, energy_vals1_spline,linestyle="-", color="red")
ax22.plot(tvals2_spline, energy_vals2_spline,linestyle="-.", color="red")
ax22.plot(tvals3_spline, energy_vals3_spline,linestyle=":", color="red")
# ax12.plot(tvals2, energy_vals2,linestyle="-.", color="red")
ax22.set_ylabel("Energy")
ax22.tick_params(axis='y', labelcolor='red')
ax22.yaxis.label.set_color('red')
fig.tight_layout()

plt.savefig("case2_benchmarking_fenics_performance.png",dpi=300)

plt.show()