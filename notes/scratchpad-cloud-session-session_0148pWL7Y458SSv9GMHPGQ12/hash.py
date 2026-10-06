import time
from daffodil.daf import Daf
def ref(daf, loda):                       # the meaning: a row matches if ANY dict matches ALL its fields, by ==
    hd=daf.hd
    return [r for r in daf.lol if any(all(r[hd[c]]==v for c,v in da.items()) for da in loda)]
def grouped_v1(daf, loda):                # what I prototyped: all values go into a set of tuples
    hd=daf.hd; groups={}
    for da in loda: groups.setdefault(tuple(da),set()).add(tuple(da.values()))
    plan=[(tuple(hd[c] for c in k), v) for k,v in groups.items()]
    return [r for r in daf.lol if any(tuple([r[i] for i in idx]) in vals for idx,vals in plan)]
def grouped_v2(daf, loda):                # hashable values in a set, unhashable ones in a list, and a try for unhashable cells
    hd=daf.hd; groups={}
    for da in loda:
        key=tuple(da); vals=tuple(da.values())
        g=groups.setdefault(key,(set(),[]))
        try: g[0].add(vals)
        except TypeError: g[1].append(vals)
    plan=[(tuple(hd[c] for c in k), hs, us) for k,(hs,us) in groups.items()]
    out=[]
    for r in daf.lol:
        for idx,hs,us in plan:
            probe=tuple([r[i] for i in idx])
            try: hit = probe in hs
            except TypeError: hit = False          # an unhashable cell equals no hashable value
            if not hit and us: hit = any(probe==u for u in us)
            if hit: out.append(r); break
    return out
cells=Daf(lol=[[1,{'a','b'}],[2,['x']],[3,'s'],[4,{'c'}],[5,True],[6,1.0],[7,['x']]], cols=['id','v'])
cases=[("hashable selectors, some cells are sets and lists", [{'v':'s'},{'v':1}]),
       ("a set as a selector value",                         [{'v':{'a','b'}}]),
       ("a list as a selector value",                        [{'v':['x']}]),
       ("mixed: string, set, list",                          [{'v':'s'},{'v':{'c'}},{'v':['x']}]),
       ("1 matches True and 1.0 as == does",                 [{'v':1}])]
for label,loda in cases:
    want=[r[0] for r in ref(cells,loda)]
    try: g1=[r[0] for r in grouped_v1(cells,loda)]
    except Exception as e: g1=type(e).__name__
    g2=[r[0] for r in grouped_v2(cells,loda)]
    print(f"{label:50} ref {want}  v1 {g1}  v2 {g2}  v2 correct: {g2==want}")
N=200000
d=Daf(lol=[[i, i%50, f"s{i%1000}", i*2, 'x'] for i in range(N)], cols=['id','grp','name','val','tag'])
loda=[{'name': f"s{i}"} for i in range(0,1000,10)]
for nm,f in (("v1",grouped_v1),("v2",grouped_v2)):
    best=9
    for _ in range(3):
        s=time.perf_counter(); f(d,loda); best=min(best,time.perf_counter()-s)
    print("200,000 rows, 100 dicts,", nm, round(best,3), "s")
