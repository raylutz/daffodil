from daffodil.daf import Daf
class Sub(Daf): pass
d=Sub(lol=[[1,'a']],cols=['id','v'],keyfield='id',dtypes={'id':int,'v':str},name='orig')
d.attrs['k']=[1]; d.md_max_rows=3; d.itermode='dict' if False else d.itermode
d.select_krows([1])
c=d.clone_empty()
k=d.copy()
for attr in ['name','keyfield','dtypes','attrs','hd','lol','_kd','md_max_rows','disp_cols']:
    cv=getattr(c,attr); kv=getattr(k,attr); dv=getattr(d,attr)
    print(f"{attr:12} clone_empty: {'same obj' if cv is dv else 'new'} {cv!r:.28}  | copy: {'same obj' if kv is dv else 'new'}")
print("class: clone_empty", type(c).__name__, "| copy", type(k).__name__)
