import numpy as np
from scipy.integrate import trapezoid

def metrics(t,T,U,setpoint):
    avg=T.mean(axis=1); peak=float(T.max()); err=avg-setpoint
    # 10–90% rise time referenced to initial mean and final mean where possible.
    y0=avg[0]; yf=avg[-1]; span=yf-y0
    rise=np.nan
    if abs(span)>1e-9:
        ix=np.where(avg>=y0+.9*span)[0]
        if len(ix): rise=float(t[ix[0]]-t[0])
    band=max(.2,.02*abs(setpoint)); settled=np.where(np.abs(avg-setpoint)<=band)[0]
    settling=float(t[settled[0]]) if len(settled) and np.all(np.abs(avg[settled[0]:]-setpoint)<=band) else np.nan
    return {'max_temperature_C':peak-273.15,'zone_max_C':(T.max(axis=0)-273.15).tolist(),'max_spread_K':float(np.max(T.max(1)-T.min(1))),'rise_time_s':rise,'settling_time_s':settling,'peak_time_s':float(t[np.argmax(avg)]),'overshoot_K':float(max(0,avg.max()-setpoint)),'steady_state_error_K':float(setpoint-avg[-1]),'IAE_Ks':float(trapezoid(np.abs(err),t)),'time_above_threshold_s':float(trapezoid((T.max(1)>setpoint+5).astype(float),t)),'cooling_energy_proxy_s':float(trapezoid((U**3).sum(1),t)),'peak_command':float(U.max()),'max_budget_utilization':float(U.sum(1).max())}
