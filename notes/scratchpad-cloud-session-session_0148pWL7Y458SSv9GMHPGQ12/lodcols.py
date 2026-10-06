import time
from daffodil.daf import Daf
import daffodil.lib.daf_utils as du
NULL=''
N=200000
lod=[{f"c{j}":i for j in range(10)} for i in range(N)]
allcols=[f"c{j}" for j in range(10)]; subset=allcols[:5]
def cur(cols):   return [list(du.set_cols_da(r,cols).values()) for r in lod if r and isinstance(r,dict)]
def get(cols):   return [[r.get(c,NULL) for c in cols] for r in lod if r and isinstance(r,dict)]
def chk(cols):
    cs=set(cols); out=[]
    for r in lod:
        if r and isinstance(r,dict):
            if not r.keys() <= cs: raise ValueError
            out.append([r.get(c,NULL) for c in cols])
    return out
def t(f,*a):
    best=9
    for _ in range(3):
        s=time.perf_counter(); f(*a); best=min(best,time.perf_counter()-s)
    return round(best,3)
print("cols = all 10 keys:  current", t(cur,allcols), "| get, no check", t(get,allcols), "| get + check", t(chk,allcols))
print("cols = 5 of 10 keys: current", t(cur,subset),  "| get, no check", t(get,subset))
