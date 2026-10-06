import sys
sys.breakpointhook = lambda *a, **k: print("    (breakpoint reached, hook returned)")
from daffodil.daf import Daf
def run(label, fn):
    try: print(f"{label}: returns {fn()!r}")
    except Exception as e: print(f"{label}: {type(e).__name__}: {e!r}")
def bad(row, acc, cols=None, **k): raise RuntimeError('boom')
run("reduce, func always fails      ", lambda: Daf(cols=['a'], lol=[[1], [2]]).reduce(bad))
run("sum_da, text value             ", lambda: Daf.sum_da({'a': 'x', 'b': 2}, {'a': 0, 'b': 0}))
run("sum_da, astype, acc missing col", lambda: Daf.sum_da({'a': 1}, {}, cols=['a'], astype=int))
run("reduce sum_da, normal          ", lambda: Daf(cols=['a', 'b'], lol=[[1, 'x'], [2, 3]]).reduce(Daf.sum_da))
