from daffodil.daf import Daf
d=Daf(lol=[[1,'a'],[2,'b'],[3,'c']],cols=['id','v'],keyfield=['id','v'],dtypes={'id':int,'v':str},name='n',disp_cols=['id'])
it=iter(d); next(it); next(it)
print("_iter_index on original:", d._iter_index)
c=d.copy()
print("_iter_index on copy:", c._iter_index)
for k,v in vars(d).items():
    mut = isinstance(v,(list,dict,set))
    print(f"{k:12} {type(v).__name__:6} mutable container: {str(mut):5} shared by copy(): {getattr(c,k) is v}")
