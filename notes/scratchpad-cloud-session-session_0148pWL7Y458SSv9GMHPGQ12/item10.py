import sys, json
sys.breakpointhook = lambda *a, **k: print("    (breakpoint reached)")
from daffodil.lib import daf_utils as u
import numpy as np
for label, val in [("float nan", [1.5, float('nan')]), ("numpy inf", {'x': np.float64('inf')}), ("normal   ", {'x': 1, 'y': [2, 'a']})]:
    try:
        out = u.json_encode(val)
        print(f"{label}: returns {out!r}")
        try: json.loads(out, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c))); print("           strict JSON parse: ok")
        except Exception as e: print(f"           strict JSON parse fails: {e}")
    except Exception as e:
        print(f"{label}: {type(e).__name__}: {e}")
