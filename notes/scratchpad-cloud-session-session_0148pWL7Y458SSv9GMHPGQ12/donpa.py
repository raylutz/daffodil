import time
import numpy as np
from daffodil.daf import Daf
NULL = ''
_MISSING = object()

def to_donpa_B(d, colnames=None, default=_MISSING):
    if colnames is None: colnames = d.columns()
    out = {}
    for col in colnames:
        vals = d.col(col)
        if default is not _MISSING:
            vals = [default if (v is NULL or v is None or v != v) else v for v in vals]
        out[col] = np.array(vals)
    return out

d = Daf(lol=[[1, '', 'a'], [2, 3, ''], [None, 5, 'c']], cols=['n', 'm', 't'])
print('Original, default=0:')
for k, v in d.to_donpa(default=0).items(): print(f'   {k}: {v!r}')
print('Option B, default=0 (the Daf is not changed):')
b = to_donpa_B(d, default=0)
for k, v in b.items(): print(f'   {k}: {v!r}')
print('   Daf after:', d.lol)
print('Option B, default=np.nan for the numeric columns:')
bn = to_donpa_B(d, ['n', 'm'], default=np.nan)
for k, v in bn.items(): print(f'   {k}: {v!r}')
print('Option B, default=None omitted: same as the original')
print('   ', to_donpa_B(d)['m'], d.to_donpa()['m'])
print('\nthe same through to_pandas_df(use_donpa=True, default=0), today:')
d2 = Daf(lol=[[1, ''], [2, 3]], cols=['n', 'm'])
print('   ', d2.to_pandas_df(use_donpa=True, default=0).values.tolist(), '| Daf after:', d2.lol)
print('and to_pandas_df(default=0) without donpa, today (this one works, and changes the Daf):')
d3 = Daf(lol=[[1, ''], [2, 3]], cols=['n', 'm'])
print('   ', d3.to_pandas_df(default=0).values.tolist(), '| Daf after:', d3.lol)
big = Daf(lol=[[i, '' if i % 10 == 0 else i, i] for i in range(200_000)], cols=['a', 'b', 'c'])
def best(f, n=3):
    ts=[]
    for _ in range(n): t=time.perf_counter(); f(); ts.append(time.perf_counter()-t)
    return min(ts)
print(f'\n200,000 rows x 3 columns: to_donpa() {best(lambda: big.to_donpa()):.3f}s   to_donpa with default (B) {best(lambda: to_donpa_B(big, default=0)):.3f}s')
