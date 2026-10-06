import sys
from daffodil.daf import Daf
d = Daf(lol=[[1,'a'],[2,'b'],[3,'c']], cols=['id','v'], keyfield='id')
s = d.select_irows([0,1])
for kl in s.iter_klist():
    print('klist row is the live row:', kl._la is s.lol[0] if hasattr(kl,'_la') else type(kl))
    break
for kl in s.iter_klist():
    kl['v'] = kl['v'].upper()
print('d.lol =', d.lol, '| s.lol =', s.lol)
