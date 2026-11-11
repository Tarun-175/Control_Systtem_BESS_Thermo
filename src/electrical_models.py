import numpy as np
from scipy import signal

def rlc(R=10.,L=.1,C=.001):
    # series RLC, capacitor voltage / source voltage
    return signal.TransferFunction([1],[L*C,R*C,1])
