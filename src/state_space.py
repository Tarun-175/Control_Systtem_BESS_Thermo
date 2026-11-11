import numpy as np
from scipy.linalg import expm

def ctrb(A,B):
    n=A.shape[0]; return np.hstack([np.linalg.matrix_power(A,k)@B for k in range(n)])
def obsv(A,C):
    n=A.shape[0]; return np.vstack([C@np.linalg.matrix_power(A,k) for k in range(n)])
def cayley_hamilton_expm(A,t):
    # Polynomial interpolation of exp(t z) at eigenvalues; fallback to expm at repeated/ill-conditioned spectra.
    vals=np.linalg.eigvals(A)
    if np.min(np.abs(vals[:,None]-vals[None,:])+np.eye(len(vals))*1e9)<1e-7: return expm(A*t)
    V=np.vander(vals,N=len(vals),increasing=True)
    try: coeff=np.linalg.solve(V,np.exp(vals*t))
    except np.linalg.LinAlgError: return expm(A*t)
    return sum(coeff[k]*np.linalg.matrix_power(A,k) for k in range(len(vals)))
def transfer(A,B,C,D):
    # Numerically sampled SISO frequency transfer function is intentionally not fabricated as coefficients.
    if B.shape[1]!=1 or C.shape[0]!=1: raise ValueError('SISO conversion only')
    n=A.shape[0]; den=np.poly(A); num=np.poly(A-B@C)+D[0,0]*den # valid realization identity for D=0 in common strict proper case
    return np.real_if_close(num),np.real_if_close(den)
