import sys, random, math
sys.path.insert(0, '/tmp/claude-0/-home-user-daffodil/95e36c09-64f8-59ba-9bd6-c962257ad9e3/scratchpad')
import numpy as np
from old_convert import convert_type_value as old
from daffodil.lib.daf_utils import convert_type_value as new, get_converter
vals = ['', None, float('nan'), '0','1','0.0','1.0','False','FALSE','True','TRUE','false','true','yes','no','abc','12','-3',' 7 ','1.9','-1.9','1e3','1E3','1_000','1,000','12abc','nan','NaN','inf','-inf','Infinity','1e999',
        '9007199254740993','12345678901234567890','12345678901234567890.0','+5','٣',' ','\t','0x10','00012',
        0, 1, 5, -2, 3.7, True, False, 10**30, float('inf'), np.int64(4), np.float64(2.5), np.int32(7),
        '[1, 2]','{"a": 1}',"{'a': 1}",'[1, 2','{bad',[1,2],{'a':1},(1,2),b'5', 'x'*3]
types = [int, float, bool, str, list, dict, tuple, set, np.int64]
def run(f, v, t):
    try: r = f(v, t)
    except Exception as e: return ('EXC', type(e).__name__)
    return ('OK', repr(r) if not (isinstance(r, float) and r != r) else 'nan', type(r).__name__)
diffs = {}
same = 0
for t in types:
    for v in vals:
        a = run(old, v, t); b = run(new, v, t)
        c = run(lambda v, t: get_converter(t)(v), v, t)
        assert b == c, (v, t, b, c)             # the looked up converter equals the function
        if a == b: same += 1
        else: diffs.setdefault(t.__name__, []).append((v, a, b))
print('same results:', same)
for t, lst in diffs.items():
    print(f'-- to {t}: {len(lst)} differ')
    for v, a, b in lst: print(f'     {v!r:28} old {a}   new {b}')
