import numpy as np
from scipy import signal

def routh(coeff):
    c=np.asarray(coeff,dtype=float); n=len(c)-1; cols=(len(c)+1)//2; tab=np.zeros((n+1,cols)); tab[0,:len(c[::2])]=c[::2]; tab[1,:len(c[1::2])]=c[1::2]
    eps=1e-9
    for i in range(2,n+1):
        if abs(tab[i-1,0])<eps: tab[i-1,0]=eps
        for j in range(cols-1): tab[i,j]=(tab[i-1,0]*tab[i-2,j+1]-tab[i-2,0]*tab[i-1,j+1])/tab[i-1,0]
        if np.all(np.abs(tab[i])<eps):
            power=n-i+1
            for j in range(cols):
                p=power-2*j
                if p>=0: tab[i,j]=tab[i-1,j]*p
    return tab

def classify(coeff):
    tab=routh(coeff); changes=np.sum(np.diff(np.sign(tab[:,0]))!=0)
    return {'table':tab,'right_half_plane_poles':int(changes),'stable':bool(changes==0)}

def root_locus(num,den,gains=None):
    gains=np.logspace(-3,3,240) if gains is None else np.asarray(gains); return gains,np.array([np.roots(np.polyadd(den,g*np.pad(num,(len(den)-len(num),0)))) for g in gains])
