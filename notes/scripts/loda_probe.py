import time
from daffodil.daf import Daf
N=200000
lol=[[i, i%50, f"s{i%1000}", i*2, 'x'] for i in range(N)]
d=Daf(lol=lol, cols=['id','grp','name','val','tag'], keyfield='id')
names=[f"s{i}" for i in range(0,1000,10)]              # 100 names
loda=[{'name': n} for n in names]
pairs=[{'grp': g, 'tag': 'x', 'name': f"s{g}"} for g in range(50)]   # composite: grp, tag, name together
def t(f, n=3):
    best=9
    for _ in range(n):
        s=time.perf_counter(); r=f(); best=min(best,time.perf_counter()-s)
    return round(best,3), r
def naive(daf, loda):                    # any() over every dict, for every row
    hd=daf.hd
    chk=[[(hd[c],v) for c,v in da.items()] for da in loda]
    return Daf(lol=[r for r in daf.lol if any(all(r[i]==v for i,v in ps) for ps in chk)], cols=daf.columns())
def grouped(daf, loda):                  # group the dicts by their keys, one set of value tuples per group
    hd=daf.hd; groups={}
    for da in loda:
        groups.setdefault(tuple(da), set()).add(tuple(da.values()))
    plan=[(tuple(hd[c] for c in keys), vals) for keys, vals in groups.items()]
    if len(plan)==1 and len(plan[0][0])==1:
        (i,), vals = plan[0]; vals={v[0] for v in vals}
        rows=[r for r in daf.lol if r[i] in vals]
    else:
        rows=[r for r in daf.lol if any(tuple([r[i] for i in idx]) in vals for idx, vals in plan)]
    return Daf(lol=rows, cols=daf.columns())
a=t(lambda: naive(d,loda));    print("100 dicts, one column, naive any() per row:  ", a[0], "s rows", a[1].num_rows())
b=t(lambda: grouped(d,loda));  print("100 dicts, one column, grouped sets:         ", b[0], "s rows", b[1].num_rows(), "same:", a[1].lol==b[1].lol)
c=t(lambda: naive(d,pairs));   print("50 dicts of 3 columns, naive:                ", c[0], "s rows", c[1].num_rows())
e=t(lambda: grouped(d,pairs)); print("50 dicts of 3 columns, grouped sets:         ", e[0], "s rows", e[1].num_rows(), "same:", c[1].lol==e[1].lol)
f=t(lambda: d.select_where(lambda r: r['name'] in set(names))); print("select_where lambda for comparison:          ", f[0], "s")
