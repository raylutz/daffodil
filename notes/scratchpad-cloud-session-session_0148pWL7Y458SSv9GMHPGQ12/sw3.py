import time
from daffodil.daf import Daf
from daffodil.keyedlist import KeyedList, KeyedIndex
big = Daf(lol=[[i % 100]*50 for i in range(200000)], cols=[f'c{i}' for i in range(50)])
def best(f, k=5):
    r=[]
    for _ in range(k):
        t=time.perf_counter(); out=f(); r.append(time.perf_counter()-t)
    return min(r), out
pred = lambda row: row['c0'] == 7
t0,a0 = best(lambda: big.select_where(pred)); print(f'now                              {t0:.3f}s')

orig_init, orig_get = KeyedList.__init__, KeyedList.__getitem__
def fast_init(self, arg1=None, arg2=None, default=None):
    if type(arg1) is KeyedIndex and type(arg2) is list:        # the case DafIterator uses, first
        if len(arg1) != len(arg2):
            raise ValueError("hd and values must have the same length")
        self.hd = arg1; self._values = arg2; self._hd_shared = True
        return
    orig_init(self, arg1, arg2, default)
KeyedList.__init__ = fast_init
t1,a1 = best(lambda: big.select_where(pred)); print(f'fast __init__                    {t1:.3f}s  same rows: {a1.lol==a0.lol}')

def fast_get(self, key):
    try:
        return self._values[self.hd[key]]       # the usual case: one hashable key
    except TypeError:
        return orig_get(self, key)             # a list of keys, or an unhashable key
KeyedList.__getitem__ = fast_get
t2,a2 = best(lambda: big.select_where(pred)); print(f'fast __init__ + fast __getitem__  {t2:.3f}s  same rows: {a2.lol==a0.lol}')
kl = KeyedList(KeyedIndex(big.hd), big.lol[0])
print('list key still works:', kl[['c0','c1']], '| missing key:', end=' ')
try: kl['zz']
except Exception as e: print(type(e).__name__)
