import sys
from daffodil.daf import Daf
sys.breakpointhook = lambda *a, **k: print("    (breakpoint reached, hook returned)")
for label, fn in [("sort_by_colname('zz')       ", lambda d: d.sort_by_colname('zz')),
                  ("sort_by_colnames(['a','zz'])", lambda d: d.sort_by_colnames(['a', 'zz']))]:
    d = Daf(cols=['a', 'b'], lol=[[2, 1], [1, 2]])
    try:
        fn(d); print(f"{label}: no error, lol={d.lol}")
    except Exception as e:
        print(f"{label}: {type(e).__name__}: {e}")
