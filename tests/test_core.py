import numpy as np
from src.simulation import simulate,compare
from src.stability import classify,root_locus
from src.frequency_domain import bode
from src.state_space import ctrb,obsv,cayley_hamilton_expm
from src.block_diagrams import series,parallel,feedback
from src.signal_flow import mason
from scipy.linalg import expm

def test_routh():
 assert classify([1,2,1])['stable']; assert not classify([1,-1,-2])['stable']
def test_root_locus():
 k,p=root_locus([1],[1,2,1], [0,1]); assert p.shape==(2,2)
def test_bode():
 *_,m=bode([1],[1,1]); assert np.isfinite(m['phase_margin_deg'])
def test_state():
 A=np.array([[-1.,0],[0,-2.]]);B=np.eye(2);C=np.eye(2)
 assert np.allclose(cayley_hamilton_expm(A,1),expm(A))
 assert np.linalg.matrix_rank(ctrb(A,B))==2 and np.linalg.matrix_rank(obsv(A,C))==2
def test_block_mason():
 assert series(2,3)==6 and parallel(2,3)==5 and feedback(2,1)==2/3
 assert np.isclose(mason([(2,{1,2})],[(-.5,{2,3})])['transfer'],4/3)
def test_simulation_limits_and_fault():
 x=simulate(duration=60,dt=2,scenario='Normal operation',budget=1.2)
 assert np.isfinite(x['T']).all() and np.max(x['U'])<=1 and np.max(x['U'].sum(1))<=1.200001
 y=simulate(duration=60,dt=2,scenario='Cooling actuator degradation',fault_on=20,health_fault=.3)
 assert y['health'][15,0]==.3

def test_comparison_common():
 r=compare(duration=60,dt=2); assert set(r)=={'Open-loop','PI/PID','ThermoGuard'}
