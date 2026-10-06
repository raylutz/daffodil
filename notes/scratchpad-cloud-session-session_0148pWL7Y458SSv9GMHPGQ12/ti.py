import time, weakref
from daffodil.daf import Daf
big = Daf(lol=[[i % 100]*50 for i in range(200000)], cols=[f'c{i}' for i in range(50)])
def best(f, k=5):
    r=[]
    for _ in range(k):
        t=time.perf_counter(); out=f(); r.append(time.perf_counter()-t)
    return min(r), out
t1,a = best(lambda: big.select_by_dict({'c0': 7}))
def with_positions():
    idxs = [n for n,row in enumerate(big.lol) if row[0] == 7]
    return big.clone_empty(lol=[big.lol[n] for n in idxs]), idxs
t2,(b,idxs) = best(with_positions)
print(f'select_by_dict now                 {t1:.4f}s')
print(f'select + remember positions        {t2:.4f}s  ({len(idxs)} positions)')
# the by-identity version: result keeps a link to the source, positions found when asked
t3,pos = best(lambda: (lambda m: [m[id(r)] for r in a.lol])({id(r): n for n,r in enumerate(big.lol)}))
print(f'to_idxs by identity at call time   {t3:.4f}s  same as scan: {pos == idxs}')
# stale positions after the source changes
src = Daf(lol=[[1,'a'],[2,'b'],[3,'a']], cols=['id','v'])
r = src.select_by_dict({'v': 'a'}); remembered = [0, 2]
src.insert_irow(0, [0, 'z'])
print('after insert at 0, remembered positions', remembered, '-> rows', [src.lol[i] for i in remembered], '(wrong)')
print('by identity still finds', [n for n, row in enumerate(src.lol) if any(row is x for x in r.lol)])
