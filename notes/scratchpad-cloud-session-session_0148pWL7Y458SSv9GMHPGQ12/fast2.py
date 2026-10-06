import time, random
from daffodil.daf import Daf, KeysDisabledError

def fast2(self, selector_da, expectmax=-1, inverse=False, keyfield=''):
    if self.lol and not self.hd:
        raise KeysDisabledError("select_by_dict(): this Daf has no column names. Call set_cols() to name them.")

    hd = self.hd
    if any(col not in hd for col in selector_da):
        result_lol = list(self.lol) if inverse else []      # an unknown column matches nothing.
    else:
        pairs = [(hd[col], val) for col, val in selector_da.items()]
        if not pairs:
            result_lol = [] if inverse else list(self.lol)  # an empty selector matches every row.
        elif len(pairs) == 1:
            icol, val = pairs[0]
            result_lol = [row_la for row_la in self.lol if (row_la[icol] == val) is not inverse]
        else:
            icol, val = pairs[0]
            rest = pairs[1:]
            result_lol = [row_la for row_la in self.lol
                          if (row_la[icol] == val and all(row_la[i] == v for i, v in rest)) is not inverse]

    if expectmax != -1 and len(result_lol) > expectmax:
        raise LookupError(f"select_by_dict(): {len(result_lol)} rows match, more than expectmax={expectmax}.")

    return Daf(cols=self.columns(), lol=result_lol, keyfield=keyfield or self.keyfield, dtypes=self.dtypes)

random.seed(2); vals=[0,1,2,'a','b','',None,1.0,True,(1,2),[1,2]]; bad=n=0
for trial in range(300):
    rows=[[random.choice(vals) for _ in range(4)] for _ in range(random.randint(0,12))]
    d=Daf(lol=rows, cols=['a','b','c','d'], keyfield='a' if trial%2 else '')
    for sel in ({}, {'a':1}, {'b':'a'}, {'a':1,'c':'b'}, {'zz':1}, {'a':[1,2]}, {'d':None}, {'c':''}, {'a':0,'b':0,'c':0}):
        for inv in (False, True):
            for em in (-1, 1):
                n+=1
                r=[]
                for f in (Daf.select_by_dict, fast2):
                    try: r.append(f(d, sel, expectmax=em, inverse=inv).lol)
                    except Exception as e: r.append(type(e).__name__)
                if r[0]!=r[1]: bad+=1; print('DIFF',sel,inv,em,r)
print(n,'comparisons,',bad,'differences')
h = Daf(lol=[[1,2]])            # rows, no column names
for f in (Daf.select_by_dict, fast2):
    try: f(h, {'a':1}); print('no error')
    except Exception as e: print(f.__name__, '->', type(e).__name__)
big = Daf(lol=[[i % 100]*50 for i in range(200000)], cols=[f'c{i}' for i in range(50)])
for label, f in (('now', Daf.select_by_dict), ('new', fast2)):
    for sel in ({'c0':7}, {'c0':7,'c1':7}):
        t=min((lambda t0:(f(big,sel), time.perf_counter()-t0)[1])(time.perf_counter()) for _ in range(5))
        print(f'{label} {len(sel)} field(s): {t:.4f}s')
