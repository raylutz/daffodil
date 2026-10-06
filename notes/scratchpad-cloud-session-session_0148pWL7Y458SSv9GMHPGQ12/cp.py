import time
from daffodil.daf import Daf
d=Daf(lol=[[2,'b'],[1,'a']],cols=['id','v'])
c=d.copy()
print("shallow shares outer:", c.lol is d.lol, " hd shared:", c.hd is d.hd, " dtypes shared:", c.dtypes is d.dtypes)
c.append([3,'c'])
print("after c.append: d rows =", d.num_rows(), " c rows =", c.num_rows())
c=d.copy(); d2=Daf(lol=[[2,'b'],[1,'a']],cols=['id','v']); c=d2.copy(); c.insert_icol(1,'w',[0,0]) if False else None
c=d2.copy(); c['x']=[7,8]
print("after c['x']=..: d cols", d2.columns(), " c cols", c.columns(), " d rows", d2.lol)
big=Daf(lol=[[i,i,i] for i in range(200000)],cols=['a','b','c'])
for lv in ['shallow','sortable','editable']:
    t=time.perf_counter(); big.copy(lv); print(lv, round(time.perf_counter()-t,4),'s')
