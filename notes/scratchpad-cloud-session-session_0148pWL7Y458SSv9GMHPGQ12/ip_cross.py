import time
from daffodil.daf import Daf
def make(nrows, ncols):
    cols=[f'c{i}' for i in range(ncols)]
    return lambda: Daf(lol=[[r*ncols+i for i in range(ncols)] for r in range(nrows)], cols=cols)
def best(mk, fn, n=5):
    ts=[]
    for _ in range(n):
        d=mk(); t=time.perf_counter(); fn(d); ts.append(time.perf_counter()-t)
    return min(ts)
for ncols in [2,3,4,5,6,8,10,15,20]:
    nrows=600_000//ncols
    mk=make(nrows,ncols)
    r=best(mk, lambda d: d.apply_in_place(lambda r: {**r,'c1':r['c1']+1}))
    k=best(mk, lambda d: d.apply_in_place(lambda kl: kl.__setitem__('c1', kl['c1']+1), by='row_klist'))
    print(f'{ncols:>3} cols ({nrows:>6} rows): row {r:.3f}s  row_klist {k:.3f}s  ratio row/klist {r/k:.2f}')
