import time
from daffodil.daf import Daf
N=200000
lod=[{f"c{j}":i for j in range(10)} for i in range(N)]
cols=[f"c{j}" for j in range(10)]
def t(f):
    best=9
    for _ in range(3):
        s=time.perf_counter(); f(); best=min(best,time.perf_counter()-s)
    return round(best,3)
print("no cols                 :", t(lambda: Daf.from_lod(lod)))
print("cols = all 10 (checked) :", t(lambda: Daf.from_lod(lod, cols=cols)))
print("cols = 5, ignore extras :", t(lambda: Daf.from_lod(lod, cols=cols[:5], ignore_extra_keys=True)))
