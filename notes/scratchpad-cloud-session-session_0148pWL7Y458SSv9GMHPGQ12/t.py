import time
from daffodil.daf import Daf
big = Daf(lol=[[i,i,i] for i in range(200000)], cols=['a','b','c'])
best = min((lambda t: (sum(kl['a'] for kl in big.iter_klist()), time.perf_counter()-t)[1])(time.perf_counter()) for _ in range(7))
print(f'best of 7, 200,000 rows read-only: {best:.3f}s')
