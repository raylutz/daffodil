from daffodil.daf import Daf
d=Daf(lol=[[1,'x',5],[1,'y',6],[2,'z',7]],cols=['g','v','n'],dtypes={'g':int,'v':str,'n':int},keyfield='n')
for label,res in [("groupby",d.groupby('g')),("groupby_cols",d.groupby_cols(['g']) if hasattr(d,'groupby_cols') else None)]:
    if res is None: continue
    first=next(iter(res.values())) if isinstance(res,dict) else res
    print(label,"dtypes:",first.dtypes,"keyfield:",repr(first.keyfield),"cols:",first.columns())
