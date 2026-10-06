import time, tracemalloc, copy
from daffodil.daf import Daf
from daffodil.lib.daf_types import CopyBits
big=Daf(lol=[[i]*50 for i in range(200000)],cols=[f"c{i}" for i in range(50)],keyfield='c0')
big.keys()
def run(f):
    best=9
    for _ in range(3):
        t=time.perf_counter(); r=f(); best=min(best,time.perf_counter()-t); 
    return best
def mem(f):
    tracemalloc.start(); r=f(); cur,peak=tracemalloc.get_traced_memory(); tracemalloc.stop(); return cur/1e6
for lv in ['shallow','sortable','editable','deep']:
    f=lambda: big.copy(lv)
    print(f"{lv:9} {run(f):.4f} s  {mem(f):.1f} MB")
t=time.perf_counter()
for _ in range(1000): big.copy('shallow')
print("shallow x1000 avg", (time.perf_counter()-t)/1000)
