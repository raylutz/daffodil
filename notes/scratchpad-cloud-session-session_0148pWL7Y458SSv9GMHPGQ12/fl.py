import time
from daffodil.daf import Daf

def orig(lod): return Daf.from_lod(lod)

def optB(lod):
    """columns are all keys found, in the order first seen. Missing keys are NULL."""
    names = {}
    for rec in lod:
        if rec and isinstance(rec, dict):
            for k in rec: names[k] = None
    cols = list(names)
    lol = [[rec.get(k, '') for k in cols] for rec in lod if rec and isinstance(rec, dict)]
    return Daf(cols=cols, lol=lol)

def optC(lod):
    """columns from the first dict, as before. A later key that is not a column raises."""
    d = Daf.from_lod(lod)
    colset = set(d.columns())
    for rec in lod:
        if rec and isinstance(rec, dict) and not rec.keys() <= colset:
            raise ValueError(f"from_lod: keys {sorted(set(rec) - colset)} are not in the columns of the first record.")
    return d

cases = {
 'same keys':                [{'a':1,'b':2},{'a':3,'b':4}],
 'later dict misses a key':  [{'a':1,'b':2},{'a':3}],
 'later dict has a new key': [{'a':1},{'a':3,'b':9}],
 'first dict is short':      [{'a':1},{'a':2,'c':7},{'a':3,'b':5}],
 'empty dict between':       [{'a':1},{},{'a':2}],
}
for name, lod in cases.items():
    print('--', name)
    for oname, f in [('orig', orig), ('B', optB), ('C', optC)]:
        try: r = f([dict(x) for x in lod]); print(f'   {oname:5} {r.lol} {r.columns()}')
        except Exception as e: print(f'   {oname:5} {type(e).__name__}: {e}')
print('== speed, 200,000 records of 10 keys')
lod = [{f'k{i}': r*10+i for i in range(10)} for r in range(200_000)]
for oname, f in [('orig', orig), ('B', optB), ('C', optC)]:
    ts=[]
    for _ in range(3):
        t=time.perf_counter(); f(lod); ts.append(time.perf_counter()-t)
    print(f'   {oname:5} {min(ts):.3f}s')
print('== from_cols_dol')
for name, dol in [('equal', {'A':[1,2],'B':[3,4]}), ('second shorter', {'A':[1,2],'B':[3]}), ('second longer', {'A':[1],'B':[3,4]})]:
    try: r = Daf.from_cols_dol(dol); print(f'   {name:15}', r.lol, r.columns())
    except Exception as e: print(f'   {name:15}', type(e).__name__, e)
