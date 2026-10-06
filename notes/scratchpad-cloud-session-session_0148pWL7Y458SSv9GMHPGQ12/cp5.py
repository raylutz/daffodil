from daffodil.daf import Daf
for mode in ['shallow now','outer only','outer + invalidate kd']:
    d=Daf(lol=[[2,'b'],[1,'a']],cols=['id','v'],keyfield='id'); d.select_krows([2])
    c=d.copy()
    if mode!='shallow now': c.lol=list(d.lol)
    if mode.endswith('kd'): c._invalidate_kd()
    print(mode,"kd shared:", c._kd is d._kd)
    c.append([3,'c'])
    print("  d rows",d.num_rows()," d kd keys",list(d._kd) if d._kd else d._kd," c rows",c.num_rows())
