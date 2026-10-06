import time, random
from daffodil.daf import Daf

def fast_select_by_dict(self, selector_da, expectmax=-1, inverse=False, keyfield=''):
    hd = self.hd
    if any(k not in hd for k in selector_da):
        # a selector column that is not a column: nothing matches, so inverse keeps every row.
        result_lol = list(self.lol) if inverse else []
    else:
        pairs = [(hd[k], v) for k, v in selector_da.items()]
        if len(pairs) == 1:
            (i, v), = pairs
            result_lol = [row for row in self.lol if (row[i] == v) is not inverse]
        else:
            result_lol = [row for row in self.lol if all(row[i] == v for i, v in pairs) is not inverse]
    if expectmax != -1 and len(result_lol) > expectmax:
        raise LookupError("expectmax")
    return Daf(cols=self.columns(), lol=result_lol, keyfield=keyfield or self.keyfield, dtypes=self.dtypes)

# --- same answers as the current method?
random.seed(1)
vals = [0, 1, 2, 'a', 'b', '', None, 1.0, True, (1,2), [1,2]]
bad = 0; n = 0
for trial in range(300):
    rows = [[random.choice(vals) for _ in range(4)] for _ in range(random.randint(0, 12))]
    d = Daf(lol=rows, cols=['a','b','c','d'], keyfield='a' if trial % 2 else '')
    for sel in ({}, {'a': 1}, {'b': 'a'}, {'a': 1, 'c': 'b'}, {'zz': 1}, {'a': [1,2]}, {'d': None}, {'c': ''}):
        for inv in (False, True):
            n += 1
            try: want = Daf.select_by_dict(d, sel, inverse=inv).lol
            except Exception as e: want = type(e).__name__
            try: got = fast_select_by_dict(d, sel, inverse=inv).lol
            except Exception as e: got = type(e).__name__
            if want != got:
                bad += 1
                if bad <= 5: print('DIFF', sel, inv, rows[:3], want, got)
print(f'{n} comparisons, {bad} differences')

# --- timing, end to end, including building the Daf
big = Daf(lol=[[i % 100]*50 for i in range(200000)], cols=[f'c{i}' for i in range(50)])
def best(f, k=5):
    r=[]
    for _ in range(k):
        t=time.perf_counter(); out=f(); r.append(time.perf_counter()-t)
    return min(r), out
t1,a = best(lambda: big.select_by_dict({'c0': 7}))
t2,b = best(lambda: big.select_where(lambda row: row['c0'] == 7))
t3,c = best(lambda: fast_select_by_dict(big, {'c0': 7}))
t4,d = best(lambda: fast_select_by_dict(big, {'c0': 7, 'c1': 7}))
t5,e = best(lambda: fast_select_by_dict(big, {'c0': 7}, inverse=True))
print(f'select_by_dict now       {t1:.3f}s')
print(f'select_where             {t2:.3f}s')
print(f'fast select_by_dict      {t3:.4f}s  same rows: {a.lol == c.lol}  shared: {c.lol[0] is big.lol[7]}  type: {type(c).__name__}  keyfield: {c.keyfield!r}')
print(f'fast, two fields         {t4:.4f}s  rows={len(d)}')
print(f'fast, inverse (199,000)  {t5:.4f}s  rows={len(e)}')
