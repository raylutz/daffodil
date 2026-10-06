import time
from daffodil.daf import Daf
big = Daf(lol=[[i % 100]*50 for i in range(200000)], cols=[f'c{i}' for i in range(50)])
def best(f, k=5):
    r=[]
    for _ in range(k):
        t=time.perf_counter(); out=f(); r.append(time.perf_counter()-t)
    return min(r), out
sl = big[1:]
t1,a = best(lambda: sl.select_where_idxs(lambda row: row['c0'] == 7))
def by_dict_idxs(daf, sel):
    pairs=[(daf.hd[c],v) for c,v in sel.items()]; i,v=pairs[0]; rest=pairs[1:]
    return [n for n,row in enumerate(daf.lol) if row[i]==v and all(row[j]==w for j,w in rest)]
t2,b = best(lambda: by_dict_idxs(sl, {'c0': 7}))
t3,c = best(lambda: big[1:])
print(f'select_where_idxs now (after big[1:]) {t1:.3f}s  n={len(a)}')
print(f'idxs by position, one field           {t2:.4f}s  same: {a==b}')
print(f'the big[1:] slice alone               {t3:.4f}s')
