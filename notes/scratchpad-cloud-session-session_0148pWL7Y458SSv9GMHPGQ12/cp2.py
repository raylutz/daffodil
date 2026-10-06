import time
from daffodil.daf import Daf
d=Daf(lol=[[2,'b'],[1,'a']],cols=['id','v'],keyfield='id')
c=d.copy(); c.append([3,'c'])
print("shallow: after c.append  d rows:", d.num_rows(), " d._kd stale:", len(d._kd) if d._kd else None)
d=Daf(lol=[[2,'b'],[1,'a']],cols=['id','v'])
c=d.copy(); c.drop_cols(['v'])
print("shallow: after c.drop_cols  d:", d.columns(), d.lol)
d=Daf(lol=[[2,'b'],[1,'a']],cols=['id','v'])
c=d.copy('sortable'); c.drop_cols(['v'])
print("sortable: after c.drop_cols  d:", d.columns(), d.lol)
d=Daf(lol=[[2,'b'],[1,'a']],cols=['id','v'])
c=d.copy(); c.reverse() if hasattr(c,'reverse') else None
big=Daf(lol=[[i,i,i] for i in range(200000)],cols=['a','b','c'])
for lv in ['shallow','sortable','editable']:
    t=time.perf_counter(); big.copy(lv); print(lv, round(time.perf_counter()-t,5),'s')
