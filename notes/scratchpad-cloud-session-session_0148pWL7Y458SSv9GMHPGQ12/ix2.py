from daffodil.daf import Daf
def mk(): return Daf(lol=[[1,'a'],[2,'b'],[3,'c']], cols=['id','v'], keyfield='id')
for name, f in [('insert_col', lambda s: s.insert_col('w',['x','y'],0)),
                ('insert_idx_col', lambda s: s.insert_idx_col())]:
    d = mk(); s = d.select_irows([0,1]); f(s)
    print(name, 'original rows:', d.lol, 'cols:', list(d.hd))
