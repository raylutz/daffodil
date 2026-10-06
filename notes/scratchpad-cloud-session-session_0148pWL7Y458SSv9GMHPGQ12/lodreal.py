import time
from daffodil.daf import Daf
N=200000
uniform=[{f"c{j}":i for j in range(10)} for i in range(N)]
late=[dict(r) for r in uniform]; late[N-1]['extra']=1
def t(x):
    best=9
    for _ in range(3):
        s=time.perf_counter(); Daf.from_lod(x); best=min(best,time.perf_counter()-s)
    return best
print("real from_lod, uniform:", round(t(uniform),3), "s | one new key in the last dict:", round(t(late),3), "s")
