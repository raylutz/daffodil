import time
from daffodil.daf import Daf
cols=[f"c{i}" for i in range(50)]; dt={c:int for c in cols}; lol=[[0]*50]
def t(f,n=20000):
    s=time.perf_counter()
    for _ in range(n): f()
    return (time.perf_counter()-s)/n*1e6
print("Daf(lol, cols, dtypes) 50 cols: %.1f us" % t(lambda: Daf(lol=lol, cols=cols, dtypes=dt)))
print("Daf(lol, cols)         50 cols: %.1f us" % t(lambda: Daf(lol=lol, cols=cols)))
