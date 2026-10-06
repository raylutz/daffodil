import time
from daffodil.daf import Daf
N=200000
lol=[[i, i%50, f"s{i%1000}", i*2, 'x'] for i in range(N)]
d=Daf(lol=lol, cols=['id','grp','name','val','tag'], keyfield='id')
names=set(f"s{i}" for i in range(0,1000,10))      # 100 of 1000 names: 10% of rows match
def t(f, n=3):
    best=9
    for _ in range(n):
        s=time.perf_counter(); r=f(); best=min(best,time.perf_counter()-s)
    return round(best,3), r
def native(daf, col, values, inverse=False):       # what a built-in loop would do: compare cells by position
    icol=daf.hd[col]
    rows=[r for r in daf.lol if (r[icol] in values) is not inverse]
    return Daf(lol=rows, cols=daf.columns(), keyfield=daf.keyfield)
a=t(lambda: d.select_where(lambda row: row['name'] in names))
b=t(lambda: d.select_irows([i for i,v in enumerate(d.col('name')) if v in names]))
c=t(lambda: native(d,'name',names))
print("select_where lambda      ", a[0], "s  rows", a[1].num_rows())
print("col + comprehension      ", b[0], "s  rows", b[1].num_rows())
print("a native positional loop ", c[0], "s  rows", c[1].num_rows(), " same rows:", c[1].lol==a[1].lol)
e=t(lambda: native(d,'name',names,inverse=True)); print("native, inverse          ", e[0], "s  rows", e[1].num_rows())
print("select_krows on the keyfield, 20,000 keys:", t(lambda: d.select_krows(list(range(0,N,10))))[0], "s")
print()
r=d.select_by_dict({'name': names}); print("select_by_dict({'name': a set}) today -> rows:", r.num_rows(), "  (equality with a set never matches)")
r=d.select_by_dict({'name': list(names)}); print("select_by_dict({'name': a list}) today -> rows:", r.num_rows())
