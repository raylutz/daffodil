import time
from daffodil.daf import Daf
NULL=''
def current(lod):
    return Daf.from_lod(lod)
def grow(lod):                       # option B: add columns as they appear, pad once at the end
    cols=list(lod[0].keys()); colset=set(cols); lol=[]; grew=False
    for rec in lod:
        if rec and isinstance(rec,dict):
            if not rec.keys() <= colset:
                for k in rec:
                    if k not in colset: cols.append(k); colset.add(k)
                grew=True
            lol.append([rec.get(c,NULL) for c in cols])
    if grew:
        n=len(cols)
        for row in lol:
            if len(row)<n: row.extend([NULL]*(n-len(row)))
    return Daf(cols=cols,lol=lol)
def prescan(lod):                    # option C: one pass for the union of keys, then build
    cols=list(dict.fromkeys(k for rec in lod if rec and isinstance(rec,dict) for k in rec))
    lol=[[rec.get(c,NULL) for c in cols] for rec in lod if rec and isinstance(rec,dict)]
    return Daf(cols=cols,lol=lol)
N=200000
uniform=[{f"c{j}":i for j in range(10)} for i in range(N)]
late=[dict(r) for r in uniform]; late[N-1]['extra']=1
def t(f,x):
    best=9
    for _ in range(3):
        s=time.perf_counter(); f(x); best=min(best,time.perf_counter()-s)
    return best
print("uniform keys, 200,000 records x 10")
print("  current :", round(t(current,uniform),3),"s")
print("  grow    :", round(t(grow,uniform),3),"s")
print("  prescan :", round(t(prescan,uniform),3),"s")
print("one new key in the last record")
print("  grow    :", round(t(grow,late),3),"s")
print("  prescan :", round(t(prescan,late),3),"s")
g=grow(late); p=prescan(late)
print("same result:", g.columns()==p.columns() and g.lol==p.lol, "| cols", g.columns()[-2:], "| last row", g.lol[-1][-2:], "| first row", g.lol[0][-2:])
mid=[{'a':1},{'a':2,'b':5},{'c':9}]
print("order of columns, [{a},{a,b},{c}]:", grow(mid).columns(), grow(mid).lol)
