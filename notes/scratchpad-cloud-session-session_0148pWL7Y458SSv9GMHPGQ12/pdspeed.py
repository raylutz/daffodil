import time, warnings
import numpy as np, pandas as pd
from daffodil.daf import Daf
warnings.filterwarnings('ignore')
NULL = ''
def mk(nrows, ncols, blank_every):
    cols = [f'c{i}' for i in range(ncols)]
    lol = [[('' if (blank_every and (r + i) % blank_every == 0) else r * ncols + i) for i in range(ncols)] for r in range(nrows)]
    return cols, lol
def best(f, n=3):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); r = f(); ts.append(time.perf_counter() - t)
    return min(ts), r
def donpa_B(d, cols, default):
    out = {}
    for c in cols:
        vals = d.col(c)
        vals = [default if (v is NULL or v is None or v != v) else v for v in vals]
        out[c] = np.array(vals)
    return pd.DataFrame(out)
def zip_dict(d, cols, default):
    dd = {c: col for c, col in zip(cols, zip(*d.lol))}
    df = pd.DataFrame(dd)
    return df.replace('', default) if default is not None else df
def direct(d, cols, default):
    df = pd.DataFrame(d.lol, columns=cols)
    return df.replace('', default) if default is not None else df
print('seconds, best of 3. The blank cells are replaced with 0 where a default is used.')
print(f'{"table":22} {"blanks":>7} | {"to_pandas_df()":>14} {"use_csv":>8} {"use_donpa":>10} {"default=0":>10} | {"donpa+default (B)":>18} {"DataFrame(lol)":>15} {"zip to dict":>12}')
for nrows, ncols, blank in [(200_000, 10, 0), (200_000, 10, 20), (20_000, 100, 20), (2_000, 1000, 20)]:
    cols, lol = mk(nrows, ncols, blank)
    d = Daf(lol=lol, cols=cols)
    res = []
    for f in (lambda: d.to_pandas_df(),
              lambda: d.to_pandas_df(use_csv=True),
              lambda: d.to_pandas_df(use_donpa=True)):
        res.append(best(f)[0])
    # default=0 mutates the Daf, so give each run a fresh copy
    def with_default():
        dd = Daf(lol=[list(r) for r in lol], cols=cols)
        t = time.perf_counter(); dd.to_pandas_df(default=0); return time.perf_counter() - t
    res.append(min(with_default() for _ in range(3)))
    res.append(best(lambda: donpa_B(d, cols, 0))[0])
    res.append(best(lambda: direct(d, cols, 0))[0])
    res.append(best(lambda: zip_dict(d, cols, 0))[0])
    print(f'{nrows:>7} x {ncols:<4} cols {("1/"+str(blank)) if blank else "none":>10} | ' + ' '.join(f'{x:>{w}.3f}' for x, w in zip(res, (14, 8, 10, 10, 18, 15, 12))))
cols, lol = mk(20_000, 100, 20); d = Daf(lol=lol, cols=cols)
print('\ndtypes of column c1 (has blanks): to_pandas_df()', d.to_pandas_df()['c1'].dtype, '| donpa B', donpa_B(d, cols, 0)['c1'].dtype, '| zip to dict + replace', zip_dict(d, cols, 0)['c1'].dtype)
