import time
from daffodil.lib.daf_utils import _convert_int as cur, _convert_float as curf, _INT_WORDS
def fast_int(val):
    if val.__class__ is str:
        if val.isdigit() and val.isascii():          # plain digits: the common case
            return int(val)
        if val == '':
            return ''
        known = _INT_WORDS.get(val)
        if known is not None:
            return known
        if '.' in val or 'e' in val or 'E' in val:
            try: return int(float(val))
            except (ValueError, OverflowError): return val
        try: return int(val)
        except ValueError: return val
    return cur(val)
def fast_float(val):
    if val.__class__ is str:
        try: return float(val) if val != '' else ''
        except ValueError: return val
    return curf(val)
def best(f, data, n=5):
    ts=[]
    for _ in range(n):
        t=time.perf_counter(); [f(v) for v in data]; ts.append(time.perf_counter()-t)
    return min(ts)
ints=[str(i) for i in range(1_000_000)]
mixed=[(str(i) if i%20 else '') for i in range(1_000_000)]
dec=[f'{i}.5' for i in range(1_000_000)]
print('1,000,000 values, ns per value')
for name, data in [('int text', ints), ('int text, 5% empty', mixed), ('decimal text', dec)]:
    a=best(cur,data); b=best(fast_int,data)
    print(f'  to int   {name:20} current {a*1000:5.0f}   fast path {b*1000:5.0f}')
for name, data in [('decimal text', dec), ('int text', ints)]:
    a=best(curf,data); b=best(fast_float,data)
    print(f'  to float {name:20} current {a*1000:5.0f}   fast path {b*1000:5.0f}')
import sys; sys.path.insert(0,'/tmp/claude-0/-home-user-daffodil/95e36c09-64f8-59ba-9bd6-c962257ad9e3/scratchpad')
from old_convert import convert_type_value as old
a=best(lambda v: old(v,int), ints); print(f'  old function to int, int text: {a*1000:5.0f}')
a=best(lambda v: old(v,float), dec); print(f'  old function to float, decimal text: {a*1000:5.0f}')
