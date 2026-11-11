import numpy as np
from scipy import signal

def response(num,den,t,kind='step'):
    sys=signal.TransferFunction(num,den)
    if kind=='step': return signal.step(sys,T=t)
    if kind=='impulse': return signal.impulse(sys,T=t)
    if kind=='ramp': return t,signal.lsim(sys,U=t,T=t)[1]
    raise ValueError(kind)

def specs(t,y,final=None):
    y=np.asarray(y); final=float(y[-1] if final is None else final); peak=int(np.argmax(y)); lo=.1*final; hi=.9*final
    i1=np.where(y>=lo)[0]; i9=np.where(y>=hi)[0]
    return {'rise_10_90_s':float(t[i9[0]]-t[i1[0]]) if len(i1) and len(i9) else np.nan,'peak_time_s':float(t[peak]),'overshoot_percent':float(max(0,(y[peak]-final)/max(abs(final),1e-12)*100)),'settling_2pct_s':float(t[np.where(np.abs(y-final)<=.02*max(abs(final),1e-12))[0][0]]) if np.any(np.abs(y-final)<=.02*max(abs(final),1e-12)) else np.nan,'steady_state_error':float(1-final)}
