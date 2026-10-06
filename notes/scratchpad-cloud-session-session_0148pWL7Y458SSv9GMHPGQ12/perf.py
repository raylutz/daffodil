import time, copy
from daffodil.daf import Daf
N=200000
lol=[[i, i%50, f"s{i%1000}", i*2, 'x'] for i in range(N)]
d=Daf(lol=lol, cols=['id','grp','name','val','tag'], keyfield='id')
def t(f, n=3):
    best=9
    for _ in range(n):
        s=time.perf_counter(); f(); best=min(best,time.perf_counter()-s)
    return round(best,3)
print("200,000 rows x 5 cols")
print("select_where(lambda row: row['grp']==7)        ", t(lambda: d.select_where(lambda row: row['grp']==7)), "s")
print("select_by_dict({'grp': 7})                      ", t(lambda: d.select_by_dict({'grp':7})), "s")
print("select_where(two equalities, lambda)            ", t(lambda: d.select_where(lambda row: row['grp']==7 and row['tag']=='x')), "s")
print("select_by_dict({'grp': 7, 'tag': 'x'})          ", t(lambda: d.select_by_dict({'grp':7,'tag':'x'})), "s")
names=set(f"s{i}" for i in range(0,1000,10))
print("select_where(lambda row: row['name'] in set)    ", t(lambda: d.select_where(lambda row: row['name'] in names)), "s")
col=d.col('name')
print("col() + comprehension + select_irows            ", t(lambda: d.select_irows([i for i,v in enumerate(d.col('name')) if v in names])), "s")
w=Daf(lol=[[i]*50 for i in range(N)], cols=[f"c{j}" for j in range(50)])
print("copy of 200,000 x 50: deep / editable / sortable:", t(lambda: w.copy('deep'),1), "/", t(lambda: w.copy('editable')), "/", t(lambda: w.copy('sortable')), "s")
print("copy.deepcopy(daf) 200,000 x 50                 ", t(lambda: copy.deepcopy(w),1), "s")
