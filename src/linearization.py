from .thermal_model import linearize
import numpy as np

def representative_siso(A,B,C,zone=0): return (np.array([B[zone,zone]]),np.array([1.,-A[zone,zone]]))
