import numpy as np
from scipy import signal

def bode(num,den,w=None):
    w=np.logspace(-3,3,600) if w is None else w
    _,h=signal.freqs(num,den,w); mag=20*np.log10(np.maximum(abs(h),1e-15)); phase=np.unwrap(np.angle(h))*180/np.pi
    # Linear interpolation at 0 dB and -180 degree crossings.
    def crossing(x,y,target):
        if len(x)==0: return np.nan
        if abs(y[0]-target)<1e-4: return float(x[0])
        diff=y-target
        ix=np.where(diff[:-1]*diff[1:]<=0)[0]
        if len(ix):
            i=ix[0]
            y0,y1=diff[i],diff[i+1]
            if y1==y0: return float(x[i])
            frac=(0-y0)/(y1-y0)
            log_x=np.log(x[i])+frac*(np.log(x[i+1])-np.log(x[i]))
            return float(np.exp(log_x))
        if abs(y[-1]-target)<1e-4: return float(x[-1])
        return np.nan
    wg=crossing(w,mag,0); wp=crossing(w,phase,-180)
    pm=float(np.interp(np.log(wg),np.log(w),phase)+180) if np.isfinite(wg) else np.nan
    gm=float(-np.interp(np.log(wp),np.log(w),mag)) if np.isfinite(wp) else np.inf
    return w,mag,phase,{'gain_crossover_rad_s':wg,'phase_crossover_rad_s':wp,'phase_margin_deg':pm,'gain_margin_db':gm}

