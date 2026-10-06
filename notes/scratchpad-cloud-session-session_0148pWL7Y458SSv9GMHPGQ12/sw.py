import time
from daffodil.daf import Daf
big = Daf(lol=[[i % 100]*50 for i in range(200000)], cols=[f'c{i}' for i in range(50)])
def best(f, n=3):
    r=[]
    for _ in range(n):
        t=time.perf_counter(); out=f(); r.append(time.perf_counter()-t)
    return min(r), out
t1,a = best(lambda: big.select_by_dict({'c0': 7}))
t2,b = best(lambda: big.select_where(lambda row: row['c0'] == 7))
t3,c = best(lambda: big.select_where_idxs(lambda row: row['c0'] == 7))
ci = big.hd['c0']
t4,d = best(lambda: [r for r in big.lol if r[ci] == 7])
print(f'select_by_dict            {t1:.3f}s  rows={len(a)}')
print(f'select_where (KeyedList)  {t2:.3f}s  rows={len(b)}  same rows as by_dict: {a.lol == b.lol}  shared: {b.lol[0] is big.lol[7]}')
print(f'select_where_idxs (dict)  {t3:.3f}s  rows={len(c)}')
print(f'plain list comprehension  {t4:.3f}s  rows={len(d)}')
