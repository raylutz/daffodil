import time
from daffodil.daf import Daf
def best(f, n=5):
    ts=[]
    for _ in range(n):
        t=time.perf_counter(); f(); ts.append(time.perf_counter()-t)
    return min(ts)
out=[]
for nrows, ncols in [(200_000,10),(20_000,100),(2_000,1000)]:
    lod=[{f'k{i}': r*ncols+i for i in range(ncols)} for r in range(nrows)]
    out.append(f'{nrows}x{ncols}: {best(lambda: Daf.from_lod(lod)):.3f}s')
print('  '.join(out))
