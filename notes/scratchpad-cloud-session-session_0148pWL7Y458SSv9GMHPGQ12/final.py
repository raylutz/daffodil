import time
from daffodil.daf import Daf
N=200000
rows=[[i, i%50, f"s{i%1000}", i*2, 'x'] for i in range(N)]
d=Daf(lol=rows, cols=['id','grp','name','val','tag'], keyfield='id')
def t(f, n=3):
    best=9
    for _ in range(n):
        s=time.perf_counter(); r=f(); best=min(best,time.perf_counter()-s)
    return round(best,3), r
names=[f"s{i}" for i in range(0,1000,10)]
loda=[{'name': n} for n in names]
pairs=[{'grp': g, 'tag': 'x', 'name': f"s{g}"} for g in range(50)]
nameset=set(names)
print("200,000 rows x 5 columns, best of three")
a=t(lambda: d.select_where(lambda row: row['name'] in nameset)); print("select_where lambda, name in set        ", a[0], "s", a[1].num_rows(), "rows")
b=t(lambda: d.select_by_dict(loda));  print("select_by_dict, 100 dicts, 1 column     ", b[0], "s", b[1].num_rows(), "rows  same rows:", b[1].lol==a[1].lol)
c=t(lambda: d.select_by_dict(pairs)); print("select_by_dict, 50 dicts of 3 columns   ", c[0], "s", c[1].num_rows(), "rows")
e=t(lambda: d.select_by_dict(loda, inverse=True)); print("select_by_dict, same, inverse           ", e[0], "s", e[1].num_rows(), "rows")
print("select_by_dict, single dict, as before  ", t(lambda: d.select_by_dict({'grp': 7}))[0], "s")
rows2=[list(r) for r in rows]; rows2[5][2]=['unhashable']      # one cell that cannot be hashed
d2=Daf(lol=rows2, cols=['id','grp','name','val','tag'], keyfield='id')
f=t(lambda: d2.select_by_dict(loda)); print("same, with one unhashable cell (slow path)", f[0], "s", f[1].num_rows(), "rows")
