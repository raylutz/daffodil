import sys, traceback
sys.breakpointhook = lambda *a, **k: print("  (hook ran and returned)")
def lookup(d, k):
    try:
        return d[k]
    except KeyError as exc_info:
        breakpoint()
        raise
def outside_except(x):
    if x < 0:
        breakpoint()
        raise ValueError(f"x must not be negative, got {x}")
    return x
for fn, arg in ((lambda: lookup({'a': 1}, 'zz')), None), ((lambda: outside_except(-1)), None):
    try:
        fn()
    except Exception as e:
        print(f"  caught {type(e).__name__}: {e}")
        for fr in traceback.extract_tb(e.__traceback__)[-2:]:
            print(f"    line {fr.lineno}: {fr.line}")
