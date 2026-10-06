from daffodil.daf import Daf
d=Daf(lol=[[2,'b'],[1,'a']],cols=['id','v'],keyfield='id',dtypes={'id':int,'v':str},name='n')
d.select_krows([2])
for k,v in vars(d).items():
    print(f"{k:12} {type(v).__name__:10} shared by shallow copy: {getattr(d.copy(),k) is v}")
