import time
import numpy as np
from daffodil.daf import Daf
NULL = ''
_M = object()
def mk(nrows, ncols, blank_every):
    cols = [f'c{i}' for i in range(ncols)]
    lol = [[('' if (blank_every and (r + i) % blank_every == 0) else r * ncols + i) for i in range(ncols)] for r in range(nrows)]
    return Daf(lol=lol, cols=cols)
def B_percol(d, colnames=None, default=_M):                 # per column, as to_donpa does now, plus the replacement
    colnames = d.columns() if colnames is None else colnames
    out = {}
    for c in colnames:
        vals = d.col(c)
        if default is not _M:
            vals = [default if (v is NULL or v is None or v != v) else v for v in vals]
        out[c] = np.array(vals)
    return out
def B_zip(d, colnames=None, default=_M):                    # one transpose with zip, then per column
    colnames = d.columns() if colnames is None else colnames
    if colnames == d.columns():
        columns = zip(*d.lol)
    else:
        idxs = [d.hd[c] for c in colnames]
        columns = (tuple(row[i] for row in d.lol) for i in idxs)
    out = {}
    for c, vals in zip(colnames, columns):
        if default is not _M:
            vals = [default if (v is NULL or v is None or v != v) else v for v in vals]
        out[c] = np.array(vals)
    return out
def best(f, n=3):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return min(ts)
print('seconds, best of 3')
print(f'{"table":24} {"blanks":>8} | {"to_donpa() now":>14} {"default=0 (B)":>14} {"B with zip":>11}')
for nrows, ncols, blank in [(200_000, 10, 0), (200_000, 10, 20), (20_000, 100, 20), (2_000, 1000, 20)]:
    d = mk(nrows, ncols, blank)
    now = best(lambda: d.to_donpa())
    b = best(lambda: B_percol(d, default=0))
    z = best(lambda: B_zip(d, default=0))
    print(f'{nrows:>8} x {ncols:<5} cols {("1/"+str(blank)) if blank else "none":>9} | {now:>14.3f} {b:>14.3f} {z:>11.3f}')
d = mk(20_000, 100, 20)
a, b, z = d.to_donpa(), B_percol(d, default=0), B_zip(d, default=0)
print('\nsame arrays from the two versions of B:', all((b[k] == z[k]).all() for k in b), '| dtype of c1: now', a['c1'].dtype, ' B', b['c1'].dtype)
