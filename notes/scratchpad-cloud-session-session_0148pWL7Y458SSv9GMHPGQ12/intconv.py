import time
from decimal import Decimal, InvalidOperation
print("Python itself:")
for s in ['12', '1.0', '1.9', '1e3', ' 7 ', '1_000', '12345678901234567890', '12345678901234567890.0']:
    try: r = repr(int(s))
    except ValueError as e: r = 'ValueError'
    print(f'   int({s!r:26}) -> {r}')

def current(v):
    try: return int(float(v))
    except ValueError: return ''
def try_first(v):                      # plain digits first, a decimal point or exponent second
    try: return int(v)
    except ValueError:
        try: return int(float(v))
        except ValueError: return ''
def check_first(v):                    # look for a point or exponent before choosing
    if '.' in v or 'e' in v or 'E' in v:
        try: return int(float(v))
        except ValueError: return ''
    try: return int(v)
    except ValueError: return ''
def check_first_exact(v):              # same, but a decimal is cut exactly with Decimal, not through a float
    if '.' in v or 'e' in v or 'E' in v:
        try: return int(Decimal(v))
        except (InvalidOperation, ValueError, OverflowError): return ''
    try: return int(v)
    except ValueError: return ''

cases = ['12','-3',' 7 ','1.9','1.0','1e3','12345678901234567890','12345678901234567890.0','abc','12abc','','1,000','nan','inf']
print(f'\n{"text":26} {"current":>22} {"try first":>22} {"check first":>22} {"check, exact":>22}')
for s in cases:
    row=[]
    for f in (current,try_first,check_first,check_first_exact):
        try: row.append(repr(f(s)))
        except Exception as e: row.append(type(e).__name__)
    print(f'{s!r:26} ' + ' '.join(f'{r:>22}' for r in row))

def best(f, data, n=5):
    ts=[]
    for _ in range(n):
        t=time.perf_counter(); [f(v) for v in data]; ts.append(time.perf_counter()-t)
    return min(ts)
plain=[str(i) for i in range(500_000)]
dec=[f'{i}.0' for i in range(500_000)]
mixed=[(str(i) if i%10 else f'{i}.0') for i in range(500_000)]
print('\n500,000 values, best of 5')
print(f'{"":22} {"current":>9} {"try first":>10} {"check first":>12} {"check, exact":>13}')
for name,data in [('plain ints "123"',plain),('all decimals "123.0"',dec),('1 in 10 decimal',mixed)]:
    print(f'{name:22} ' + ' '.join(f'{best(f,data):>{w}.3f}' for f,w in zip((current,try_first,check_first,check_first_exact),(9,10,12,13))))
