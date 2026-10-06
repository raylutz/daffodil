import sys
sys.breakpointhook = lambda *a, **k: print("BREAKPOINT")
from daffodil.daf import Daf
import daffodil.daf as dm
def mk(): return Daf(cols=['k','v'], lol=[['a',1],['b',2],['c',3]], keyfield='k')
try: mk().apply_in_place(lambda r: None)
except Exception as e: print('aip', type(e), e)
d = mk()
def inc(kl): kl['v'] = kl['v'] * 10
d.apply_in_place(inc, by='row_klist', rowkeys=['b']); print(d.lol)
try: mk().apply_in_place(inc, by='bogus')
except Exception as e: print('aip2', type(e), e)
# manifest
man = Daf(cols=['name'], lol=[['c1'],['c2']])
store = {'c1': Daf(cols=['x','y'], lol=[[1,2],[3,4]]), 'c2': Daf(cols=['x','y'], lol=[[10,20]])}
saved = {}
def f(daf, cols): return ({'name': 'out_'+daf.lol[0][0].__str__(), 'rows': len(daf)}, daf)
try:
  r = man.manifest_apply(f, load_func=lambda cs: store[cs['name']], save_func=lambda cs, dd: saved.__setitem__(cs['name'], dd), by='table')
  print(r.lol, r.columns(), saved.keys())
except Exception as e: print('ma', type(e), e)
try: man.manifest_reduce(Daf.sum_da)
except Exception as e: print('mr', type(e), e)
try: print(man.manifest_reduce(Daf.sum_da, load_func=lambda cs: store[cs['name']]))
except Exception as e: print('mr2', type(e), e)
r = man.manifest_process(lambda cs, mult=1: {'name': cs['name'], 'n': len(store[cs['name']])*mult}, mult=2); print(r.lol, r.columns())
g = Daf(cols=['g','v'], lol=[['x',1],['y',2],['x',3]])
print({k: v.lol for k, v in g.groupby(colnames=['g']).items()})
