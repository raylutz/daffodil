import time, tracemalloc
from daffodil.daf import Daf
nrows, ncols = 2_000, 1_000
def lines():
    yield ','.join(f'c{i}' for i in range(ncols)) + '\n'
    for r in range(nrows):
        yield ','.join(str(r*ncols+i) for i in range(ncols)) + '\n'
want = ['c5','c900','c1']
def peak_and_time(f):
    tracemalloc.start(); t=time.perf_counter(); r=f(); el=time.perf_counter()-t; cur, pk = tracemalloc.get_traced_memory(); tracemalloc.stop(); return r, pk/1e6, el
a, pa, ta = peak_and_time(lambda: Daf.from_csv_buff(lines())[:, want])
b, pb, tb = peak_and_time(lambda: Daf.from_csv_buff(lines(), include_cols=want))
print(f'real method, 2,000 x 1,000 stream, 3 columns wanted (time includes the line generator, and tracing slows both):')
print(f'  read all, then select   peak {pa:6.1f} MB   {ta:.2f}s')
print(f'  include_cols            peak {pb:6.1f} MB   {tb:.2f}s    same rows: {a.lol == b.lol}   columns {b.columns()}')
text = ''.join(lines())
def best(f, n=3):
    ts=[]
    for _ in range(n):
        t=time.perf_counter(); f(); ts.append(time.perf_counter()-t)
    return min(ts)
print(f'  no tracing, from a string: read all then select {best(lambda: Daf.from_csv_buff(text)[:, want]):.3f}s   include_cols {best(lambda: Daf.from_csv_buff(text, include_cols=want)):.3f}s   no include_cols (unchanged path) {best(lambda: Daf.from_csv_buff(text)):.3f}s')
