import numpy as np
from ternary_solver import *

#Run test
def ic_fun_a(x):
    values = 0.2 + 0.02*(0.5-np.random.rand(x.shape[1]))
    return values

def ic_fun_b(x):
    values = 0.2 + 0.02*(0.5-np.random.rand(x.shape[1]))
    return values

cahn_hilliard_spline(ic_fun_a,ic_fun_b,6,6,6,1,1,1,20,100,0.5,0.1,False,True)