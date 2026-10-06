import time
from daffodil.daf import Daf
N=200000
rows=[[str(i), str(i*0.5), f"s{i%100}", str(i%7), '1'] for i in range(N)]
cols=['a','b','c','n','m']
def run(label, kw):
    best=9
    for _ in range(3):
        d=Daf(lol=[list(r) for r in rows], cols=cols)
        t=time.perf_counter(); d.apply_dtypes(silent_error=True, **kw); best=min(best,time.perf_counter()-t)
    print(f"{label:44} {best:.3f} s")
run("one int column (dtypes={'a': int})", dict(dtypes={'a': int}))
run("two columns (a int, b float)", dict(dtypes={'a': int, 'b': float}))
run("four columns (a, b, n int; m int)", dict(dtypes={'a': int, 'b': float, 'n': int, 'm': int}))
