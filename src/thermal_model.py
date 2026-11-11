import numpy as np
from config.default_parameters import CAPACITANCE, NATURAL_H, COOLING_BETA, COUPLING, GAMMA

def rhs(T, u, ambient, heat, health=None):
    health = np.ones(4) if health is None else np.asarray(health)
    T=np.asarray(T); u=np.clip(np.asarray(u),0,1)
    exchange=COUPLING @ T - COUPLING.sum(axis=1)*T
    return (heat-(NATURAL_H+COOLING_BETA*health*u)*(T-ambient)+exchange)/CAPACITANCE

def linearize(ambient=298.15, operating_temp=303.15, operating_u=.35, health=None):
    health=np.ones(4) if health is None else np.asarray(health)
    A=np.diag(-(NATURAL_H+COOLING_BETA*health*operating_u+COUPLING.sum(1))/CAPACITANCE)+COUPLING/CAPACITANCE[:,None]
    B=np.diag(-COOLING_BETA*health*(operating_temp-ambient)/CAPACITANCE)
    E=np.diag(1/CAPACITANCE)
    return A,B,np.eye(4),np.zeros((4,4)),E

def scenario_profile(name,t,duration,current,heat_fault=18000.,fault_on=300.,fault_duration=600.,ambient=298.15):
    I=np.full(4,float(current)); q=GAMMA*I**2; amb=np.full(len(t),ambient)
    if name in ('Charging-load increase','Discharging-load increase','High load + cooling degradation'):
        mask=t>=fault_on; I[:,] if False else None
        I=np.tile(I,(len(t),1)); I[mask]*=1.35; q=(I**2)*GAMMA
    else: q=np.tile(q,(len(t),1))
    if name in ('Localized hotspot Zone 1','High ambient + hotspot','High load + cooling degradation'):
        mask=(t>=fault_on)&(t<=fault_on+fault_duration); q[mask,0]+=heat_fault
    if name=='Localized hotspot Zone 2':
        mask=(t>=fault_on)&(t<=fault_on+fault_duration); q[mask,1]+=heat_fault
    if name=='High ambient temperature': amb[:]=ambient+12
    if name=='High ambient + hotspot': amb[:]=ambient+10
    return q,amb
