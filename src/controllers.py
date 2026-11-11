import numpy as np

class Controller:
    def __init__(self,kind,setpoint=303.15,kp=.12,ki=.0008,kd=0.,budget=2.,fixed=.35):
        self.kind=kind; self.sp=setpoint; self.kp=kp; self.ki=ki; self.kd=kd; self.budget=budget; self.fixed=fixed; self.integral=0.; self.prev=None
    def command(self,measured,rate,heat,health,dt):
        e=self.sp-np.asarray(measured)
        if self.kind=='Open-loop': return np.full(4,self.fixed)
        if self.kind=='PI/PID':
            self.integral+=float(np.mean(e))*dt
            derivative=0 if self.prev is None else (float(np.mean(e))-self.prev)/dt
            self.prev=float(np.mean(e)); raw=self.kp*float(np.mean(e))+self.ki*self.integral+self.kd*derivative
            u=np.full(4,np.clip(raw,0,1)); return u
        # Hotspot priority combines positive temperature deviation, rising rate and heat feedforward.
        error=np.maximum(np.asarray(measured)-self.sp,0)
        priority=.15+error+20*np.maximum(rate,0)+np.maximum(heat,0)/5000
        desired=np.clip(.18+.16*np.maximum(error,0)+heat/90000,0,1)/np.maximum(health,.15)
        weight=priority/priority.sum()
        u=np.minimum(desired, self.budget*weight)
        # Redistribute remaining budget iteratively to zones with unused actuator headroom.
        for _ in range(5):
            rem=max(0.,self.budget-u.sum()); room=np.maximum(0,1-u); w=priority*room
            if rem<1e-10 or w.sum()<1e-10: break
            u+=np.minimum(room,rem*w/w.sum())
        return np.clip(u,0,1)
