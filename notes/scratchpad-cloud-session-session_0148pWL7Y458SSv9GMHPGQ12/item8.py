import sys
from daffodil.daf import Daf
sys.breakpointhook = lambda *a, **k: print("    (breakpoint reached, hook returned)")
d = Daf(cols=['x'], lol=[['a']])
for label, settings in [("setting missing", {}), ("setting empty  ", {'spec': []})]:
    try:
        r = d.alter_daf_per_setting(settings, 'spec', {})
        print(f"{label}: no error, returned {r.lol}")
    except Exception as e:
        print(f"{label}: {type(e).__name__}: {e}")
