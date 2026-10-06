import time
import numpy as np
from daffodil.daf import Daf
from daffodil.lib import daf_utils

_M = object()
def D2_sentinel(lod):
    cols = list(lod[0]); n = len(cols); lol = []
    for r in lod:
        if r and isinstance(r, dict):
            vals = [r.get(k, _M) for k in cols]
            if _M in vals:
                hit = n - vals.count(_M); vals = ['' if v is _M else v for v in vals]
            else: hit = n
            if len(r) > hit: raise ValueError('extra')
            lol.append(vals)
    return lol

def D3_keys(lod):
    cols = list(lod[0]); colset = set(cols); lol = []
    for r in lod:
        if r and isinstance(r, dict):
            if not r.keys() <= colset: raise ValueError('extra')
            lol.append([r.get(k, '') for k in cols])
    return lol

def D4_len_then_keys(lod):
    """exact, and skips the key comparison when it can: only a record with the full width needs its keys looked at once."""
    cols = list(lod[0]); colset = set(cols); n = len(cols); lol = []
    for r in lod:
        if r and isinstance(r, dict):
            row = [r.get(k, '') for k in cols]
            if not r.keys() <= colset: raise ValueError('extra')
            lol.append(row)
    return lol

def orig(lod):
    cols = list(lod[0]); return [list(daf_utils.set_cols_da(r, cols).values()) for r in lod if r and isinstance(r, dict)]

def best(f, lod, n=5):
    ts=[]
    for _ in range(n):
        t=time.perf_counter(); f(lod); ts.append(time.perf_counter()-t)
    return min(ts)
print('== time to build the rows, best of 5 (no problem in the data)')
for nrows, ncols in [(200_000,10),(20_000,100),(2_000,1000)]:
    lod=[{f'k{i}': r*ncols+i for i in range(ncols)} for r in range(nrows)]
    res={n: best(f, lod) for n,f in [('orig',orig),('sentinel',D2_sentinel),('keys<=set',D3_keys)]}
    print(f'{nrows:>7} x {ncols:>4}: ' + '  '.join(f'{k} {v:.3f}s' for k,v in res.items()))
print('== records with 10% of the keys missing (gaps), 2,000 x 1000')
import random; random.seed(2)
lod=[{f'k{i}': i for i in range(1000) if random.random()>0.1} for r in range(2000)]
lod[0]={f'k{i}': i for i in range(1000)}
res={n: best(f, lod) for n,f in [('orig',orig),('sentinel',D2_sentinel),('keys<=set',D3_keys)]}
print('   ' + '  '.join(f'{k} {v:.3f}s' for k,v in res.items()))
print('== a cell that is a NumPy array')
lod=[{'a':1,'b':2},{'a':np.array([1,2]),'b':3}]
for name,f in [('orig',orig),('sentinel',D2_sentinel),('keys<=set',D3_keys)]:
    try: f(lod); print(f'   {name:10} ok')
    except Exception as e: print(f'   {name:10} {type(e).__name__}: {str(e)[:70]}')
