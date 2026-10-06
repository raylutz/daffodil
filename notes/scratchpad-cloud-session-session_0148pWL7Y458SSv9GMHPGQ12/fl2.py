import time
from daffodil.daf import Daf
from daffodil.lib import daf_utils

def build_orig(lod, cols):
    return [list(daf_utils.set_cols_da(r, cols).values()) for r in lod if r and isinstance(r, dict)]

def D_len(lod):          # detect only: a record with more keys than columns
    cols = list(lod[0])
    n = len(cols)
    lol = []
    for r in lod:
        if r and isinstance(r, dict):
            if len(r) > n:
                extra = [k for k in r if k not in set(cols)]
                raise ValueError(f"from_lod: record has keys {extra[:5]} that are not columns of the first record. Pass cols=.")
            lol.append(list(daf_utils.set_cols_da(r, cols).values()))
    return Daf(cols=cols, lol=lol)

_M = object()
def D_exact(lod):        # detect exactly, at C speed: count misses with a sentinel
    cols = list(lod[0])
    n = len(cols)
    lol = []
    for r in lod:
        if r and isinstance(r, dict):
            vals = [r.get(k, _M) for k in cols]
            if _M in vals:
                hit = n - vals.count(_M)
                vals = ['' if v is _M else v for v in vals]
            else:
                hit = n
            if len(r) > hit:
                extra = [k for k in r if k not in set(cols)]
                raise ValueError(f"from_lod: record has keys {extra[:5]} that are not columns of the first record. Pass cols=.")
            lol.append(vals)
    return Daf(cols=cols, lol=lol)

def C_set(lod):
    cols = list(lod[0]); colset = set(cols)
    lol = []
    for r in lod:
        if r and isinstance(r, dict):
            if not r.keys() <= colset: raise ValueError('extra')
            lol.append(list(daf_utils.set_cols_da(r, cols).values()))
    return Daf(cols=cols, lol=lol)

def B_two(lod):
    names = {}
    for r in lod:
        if r and isinstance(r, dict):
            for k in r: names[k] = None
    cols = list(names)
    return Daf(cols=cols, lol=[[r.get(k, '') for k in cols] for r in lod if r and isinstance(r, dict)])

def orig(lod): return Daf.from_lod(lod)

def best(f, lod, n=3):
    ts=[]
    for _ in range(n):
        t=time.perf_counter(); f(lod); ts.append(time.perf_counter()-t)
    return min(ts)

print('== cost, same-key records (nothing wrong), all values present')
for nrows, ncols in [(200_000,10),(20_000,100),(2_000,1000)]:
    lod=[{f'k{i}': r*ncols+i for i in range(ncols)} for r in range(nrows)]
    res = {n: best(f, lod) for n, f in [('orig',orig),('D len check',D_len),('D exact',D_exact),('C set check',C_set),('B two pass',B_two)]}
    print(f'{nrows:>7} x {ncols:>4}: ' + '  '.join(f'{k} {v:.3f}s' for k,v in res.items()))
print('== what each detects (first record a,b,c)')
cases = {
 'later record has a 4th key':           [{'a':1,'b':2,'c':3},{'a':1,'b':2,'c':3,'d':4}],
 'later record: 2 of 3 keys + new key':  [{'a':1,'b':2,'c':3},{'a':1,'b':2,'d':9}],
 'later record: swaps c for d (same size)': [{'a':1,'b':2,'c':3},{'a':1,'b':2,'d':9,}],
 'later record: fewer keys, no new key': [{'a':1,'b':2,'c':3},{'a':1}],
}
cases['later record: swaps c for d (same size)'] = [{'a':1,'b':2,'c':3},{'a':1,'b':2,'d':9}]
cases['later record: 3 keys incl. new, misses b'] = [{'a':1,'b':2,'c':3},{'a':1,'c':3,'d':9}]
del cases['later record: 2 of 3 keys + new key']
cases['later record: 2 keys incl. new (len < width)'] = [{'a':1,'b':2,'c':3},{'a':1,'d':9}]
cases.pop('later record: swaps c for d (same size)')
for name, lod in cases.items():
    out=[]
    for fname, f in [('len check',D_len),('exact',D_exact)]:
        try: f([dict(x) for x in lod]); out.append(f'{fname}: NOT detected')
        except ValueError as e: out.append(f'{fname}: detected')
    print(f'   {name:44} ' + ' | '.join(out))
