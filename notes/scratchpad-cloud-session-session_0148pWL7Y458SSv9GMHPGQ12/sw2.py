import time, cProfile, pstats, io
from daffodil.daf import Daf
from daffodil.keyedlist import KeyedList, KeyedIndex
big = Daf(lol=[[i % 100]*50 for i in range(200000)], cols=[f'c{i}' for i in range(50)])
def best(f, k=5):
    r=[]
    for _ in range(k):
        t=time.perf_counter(); out=f(); r.append(time.perf_counter()-t)
    return min(r), out
pred = lambda row: row['c0'] == 7

t,_ = best(lambda: big.select_where(pred));                     print(f'select_where now                   {t:.3f}s')
t,_ = best(lambda: [None for _ in big.iter_klist()]);           print(f'  just iterate iter_klist()         {t:.3f}s')
t,_ = best(lambda: [kl['c0'] for kl in big.iter_klist()]);      print(f'  iterate + one kl["c0"]            {t:.3f}s')

kidx = KeyedIndex(big.hd)
t,_ = best(lambda: [KeyedList(kidx, r) for r in big.lol]);      print(f'  build 200,000 KeyedLists, no loop {t:.3f}s')
kl = KeyedList(kidx, big.lol[0])
t,_ = best(lambda: [kl['c0'] for _ in range(200000)]);          print(f'  200,000 x kl["c0"]                {t:.3f}s')

# reuse one KeyedList object, swap its values
def reuse():
    k = KeyedList(kidx, big.lol[0]); out=[]
    for r in big.lol:
        k._values = r
        if pred(k): out.append(r)
    return out
t,a = best(reuse); print(f'  reuse ONE KeyedList, swap values  {t:.3f}s  ({len(a)} rows)')

pr = cProfile.Profile(); pr.enable(); big.select_where(pred); pr.disable()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('tottime').print_stats(6); print('\n'.join(s.getvalue().splitlines()[4:16]))
