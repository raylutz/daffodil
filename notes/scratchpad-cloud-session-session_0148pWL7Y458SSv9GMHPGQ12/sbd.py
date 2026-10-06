import time
from daffodil.daf import Daf
big = Daf(lol=[[i % 100]*50 for i in range(200000)], cols=[f'c{i}' for i in range(50)])
t=time.time(); r = big.select_by_dict({'c0': 7}); print(f'select_by_dict on 200,000 rows x 50: {time.time()-t:.3f}s, {len(r)} rows, shared={r.lol[0] is big.lol[7]}')
