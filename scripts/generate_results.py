import sys,pathlib,json
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
from src.simulation import compare
r=compare(scenario='Localized hotspot Zone 1',duration=1200,dt=2,fault_on=240,budget=2.)
out=pathlib.Path(__file__).resolve().parents[1]/'results';
with open(out/'metrics'/'hotspot_comparison.json','w') as f: json.dump({k:v['metrics'] for k,v in r.items()},f,indent=2)
fig,ax=plt.subplots(figsize=(10,5))
for name,x in r.items(): ax.plot(x['t'],x['T'].max(1)-273.15,label=name)
ax.set(xlabel='Time (s)',ylabel='Maximum zone temperature (°C)',title='Zone 1 hotspot controller comparison');ax.grid(True);ax.legend();fig.tight_layout();fig.savefig(out/'plots'/'hotspot_comparison.png',dpi=160)
