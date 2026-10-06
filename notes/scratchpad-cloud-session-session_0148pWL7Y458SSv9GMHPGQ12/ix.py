from daffodil.daf import Daf
d = Daf(lol=[[1,'a'],[2,'b'],[3,'c']], cols=['id','v'], keyfield='id')
s = d.select_irows([0,1])
r = s.insert_idx_col()
print('returns self:', r is s)
print('s.lol', s.lol, 'cols', list(s.hd))
print('d.lol', d.lol, 'cols', list(d.hd))
print('row shared', s.lol[0] is d.lol[0])
