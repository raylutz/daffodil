import time
import daffodil.lib.daf_utils as du
NULL=''
cols=[f"c{j}" for j in range(10)]
rec={c:1 for c in cols}
N=300000
def t(f):
    best=9
    for _ in range(5):
        s=time.perf_counter()
        for _ in range(N): f()
        best=min(best,time.perf_counter()-s)
    return best/N*1e9
print("today:  list(set_cols_da(rec, cols).values())   %.0f ns per record" % t(lambda: list(du.set_cols_da(rec,cols).values())))
print("direct: [rec.get(c, NULL) for c in cols]        %.0f ns per record" % t(lambda: [rec.get(c,NULL) for c in cols]))
print("only the set_cols_da call (makes the dict):     %.0f ns per record" % t(lambda: du.set_cols_da(rec,cols)))
