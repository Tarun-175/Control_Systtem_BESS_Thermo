import numpy as np

def controller(kind,kp=1.,ki=1.,kd=.1,tf=.02):
    if kind=='P': return np.array([kp]),np.array([1.])
    if kind=='PI': return np.array([kp,ki]),np.array([1.,0.])
    if kind=='PID filtered': return np.array([kp*tf+kd,kp+ki*tf,ki]),np.array([tf,1.,0.])
    raise ValueError(kind)
def lead(k=1.,T=1.,alpha=.2): return np.array([k*T,k]),np.array([alpha*T,1.])
def lag(k=1.,T=1.,beta=5.): return np.array([k*T,k]),np.array([beta*T,1.])
def lead_lag(k=1.,T1=1.,alpha=.2,T2=2.,beta=5.):
    return np.polymul([T1,1],[T2,1])*k,np.polymul([alpha*T1,1],[beta*T2,1])
