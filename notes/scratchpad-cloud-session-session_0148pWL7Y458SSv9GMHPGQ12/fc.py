import time
from daffodil.daf import Daf
def optB(dol):
    cols = list(dol); n = len(dol[cols[0]])
    for c in cols:
        if len(dol[c]) != n:
            raise ValueError(f"from_cols_dol: column '{c}' has {len(dol[c])} values, but '{cols[0]}' has {n}.")
    return Daf(cols=cols, lol=[list(r) for r in zip(*dol.values())])
def optC(dol):
    cols = list(dol); n = max(len(v) for v in dol.values())
    return Daf(cols=cols, lol=[[dol[c][i] if i < len(dol[c]) else '' for c in cols] for i in range(n)])
for name, dol in [('equal', {'A':[1,2],'B':[3,4]}), ('second shorter', {'A':[1,2],'B':[3]}), ('second longer', {'A':[1],'B':[3,4]})]:
    for oname, f in [('orig', Daf.from_cols_dol), ('B', optB), ('C', optC)]:
        try: r = f(dol); print(f'{name:15} {oname:5}', r.lol)
        except Exception as e: print(f'{name:15} {oname:5}', type(e).__name__, e)
big = {f'c{i}': list(range(200_000)) for i in range(10)}
for oname, f in [('orig', Daf.from_cols_dol), ('B', optB), ('C', optC)]:
    ts=[]
    for _ in range(3):
        t=time.perf_counter(); f(big); ts.append(time.perf_counter()-t)
    print(f'200,000 x 10 {oname:5} {min(ts):.3f}s')
