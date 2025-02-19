import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from binary_solver import cahn_hilliard_spline


#formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')


    

"""
Test Case
"""

def initial_condition(x):
    values = 0.5 + 0.02*(0.5-np.random.rand(x.shape[1]))
    return values

tvals, phi_max, phi_min, phi_avg, energy_vals = cahn_hilliard_spline(initial_condition,chi=8,N1=1,N2=1,stride=10,tend=20,deltax=0.4,dt=0.1,return_data=True)
        
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