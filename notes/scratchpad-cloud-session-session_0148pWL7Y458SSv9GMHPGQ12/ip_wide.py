import time
from daffodil.daf import Daf
def make(nrows, ncols):
    cols=[f'c{i}' for i in range(ncols)]
    return lambda: Daf(lol=[[r*ncols+i for i in range(ncols)] for r in range(nrows)], cols=cols)
def best(mk, fn, n=3):
    ts=[]
    for _ in range(n):
        d=mk(); t=time.perf_counter(); fn(d); ts.append(time.perf_counter()-t)
    return min(ts)
for nrows, ncols in [(200_000,3),(20_000,30),(2_000,300),(200,1000),(2_000,1000)]:
    mk=make(nrows,ncols)
    r1=best(mk, lambda d: d.apply_in_place(lambda r: {**r,'c1':r['c1']+1}))
    r2=best(mk, lambda d: d.apply_in_place(lambda r: r.__setitem__('c1', r['c1']+1) or r))
    k =best(mk, lambda d: d.apply_in_place(lambda kl: kl.__setitem__('c1', kl['c1']+1), by='row_klist'))
    print(f'{nrows:>8} rows x {ncols:>4} cols | row returns {{**r}} {r1:.3f}s | row returns same dict {r2:.3f}s | row_klist {k:.3f}s')
