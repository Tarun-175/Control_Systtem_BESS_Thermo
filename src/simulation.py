import numpy as np
from scipy.integrate import solve_ivp
from .thermal_model import rhs,scenario_profile
from .controllers import Controller
from .performance_metrics import metrics

def simulate(kind='ThermoGuard',scenario='Normal operation',duration=1800,dt=2,ambient=298.15,current=180,setpoint=303.15,budget=2.,heat_fault=18000,fault_on=300,fault_duration=600,health_fault=.45,sensor_bias=0.,sensor_noise=0.,initial=None,kp=.12,ki=.0008,kd=0.,fixed=.35):
    if duration<=0 or dt<=0 or budget<0: raise ValueError('duration and dt must be positive; budget must be nonnegative')
    t=np.arange(0,duration+dt/2,dt); q,amb= scenario_profile(scenario,t,duration,current,heat_fault,fault_on,fault_duration,ambient)
    health=np.ones((len(t),4));
    if scenario in ('Cooling actuator degradation','High load + cooling degradation'):
        health[t>=fault_on,0]=health_fault
    T=np.tile(np.asarray(initial if initial is not None else [ambient+5]*4),(len(t),1)).astype(float)
    measured=T.copy(); U=np.zeros_like(T); ctrl=Controller(kind,setpoint,kp,ki,kd,budget,fixed)
    for k in range(len(t)-1):
        measured[k]=T[k].copy();
        if scenario=='Temperature sensor bias': measured[k,0]+=sensor_bias
        if sensor_noise: measured[k]+=np.random.default_rng(k).normal(0,sensor_noise,4)
        rate=np.zeros(4) if k==0 else (measured[k]-measured[k-1])/dt
        U[k]=ctrl.command(measured[k],rate,q[k],health[k],dt)
        # RK4 fixed-step integration of the coupled nonlinear thermal model.
        f=lambda x:rhs(x,U[k],amb[k],q[k],health[k])
        h=dt; x=T[k]; a=f(x); b=f(x+h*a/2); c=f(x+h*b/2); d=f(x+h*c)
        T[k+1]=x+h*(a+2*b+2*c+d)/6
    U[-1]=U[-2]; measured[-1]=T[-1];
    if scenario=='Temperature sensor bias': measured[:,0]+=sensor_bias
    if not np.isfinite(T).all(): raise FloatingPointError('non-finite temperature')
    return {'t':t,'T':T,'measured':measured,'U':U,'heat':q,'ambient':amb,'health':health,'metrics':metrics(t,T,U,setpoint)}

def compare(**kwargs): return {k:simulate(kind=k,**kwargs) for k in ['Open-loop','PI/PID','ThermoGuard']}
