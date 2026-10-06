import time
from daffodil.daf import Daf
big=[[i,i,i] for i in range(200_000)]
def run(label, fn, n=5):
    ts=[]
    for _ in range(n):
        d=Daf(lol=[list(r) for r in big], cols=['g','x','y'])
        t=time.perf_counter(); fn(d); ts.append(time.perf_counter()-t)
    print(f'{label:34} best {min(ts):.3f}s  median {sorted(ts)[n//2]:.3f}s')
run("by='row' (dict, written by name)", lambda d: d.apply_in_place(lambda r: {**r,'y':r['y']+1}))
run("by='row' (returns the same dict)", lambda d: d.apply_in_place(lambda r: r.__setitem__('y', r['y']+1) or r))
run("by='row_klist'", lambda d: d.apply_in_place(lambda kl: kl.__setitem__('y', kl['y']+1), by='row_klist'))
