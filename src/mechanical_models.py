from scipy import signal

def mass_spring_damper(m=1.,c=2.,k=10.):
    return signal.TransferFunction([1],[m,c,k])
