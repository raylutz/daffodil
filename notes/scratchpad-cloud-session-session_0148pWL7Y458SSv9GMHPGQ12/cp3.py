import time
from daffodil.daf import Daf
for lv in ['shallow','sortable']:
    d=Daf(lol=[[2,'b'],[1,'a']],cols=['id','v'])
    c=d.copy(lv); c.insert_icol(1,[0,0],'w')
    print(lv,"insert_icol: d cols",d.columns(),"d rows",d.lol,"| c cols",c.columns())
big=Daf(lol=[[i,i,i] for i in range(200000)],cols=['a','b','c'])
t=time.perf_counter()
for _ in range(20): list(big.lol)
print("outer list only:", round((time.perf_counter()-t)/20,5),'s')
