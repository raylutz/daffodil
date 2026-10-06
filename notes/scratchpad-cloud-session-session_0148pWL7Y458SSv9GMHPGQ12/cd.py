import time
from daffodil.daf import Daf
big = {f'c{i}': list(range(200_000)) for i in range(10)}
ts=[]
for _ in range(3):
    t=time.perf_counter(); Daf.from_cols_dol(big); ts.append(time.perf_counter()-t)
print(f'200,000 x 10: {min(ts):.3f}s')
