from daffodil.daf import Daf
import sys
sys.breakpointhook = lambda *a, **k: print("    (breakpoint reached)")
def target(): return Daf(lol=[[1,2,3],[4,5,6],[7,8,9]], cols=['a','b','c'])
cases = [
    ("column, short list ", lambda d: d.set_irows_icols([0,1,2], 1, [100])),
    ("column, long list  ", lambda d: d.set_irows_icols([0,1,2], 1, [100,200,300,400])),
    ("block, short list  ", lambda d: d.set_irows_icols([0,1], [0,1], [100])),
    ("block, long list   ", lambda d: d.set_irows_icols([0,1], [0,1], [100,200,300])),
]
for label, fn in cases:
    d = target()
    try:
        fn(d); print(f"{label}: {d.lol}")
    except Exception as e:
        print(f"{label}: {type(e).__name__}: {e}; target = {d.lol}")
