import sys, time
from daffodil.daf import Daf
d = Daf(lol=[[1,'a'],[2,'b'],[3,'c']], cols=['id','v'], keyfield='id')
s = d.select_irows([0,1])
print('refcounts d rows:', [sys.getrefcount(r) for r in d.lol])
print('refcounts s rows:', [sys.getrefcount(r) for r in s.lol])
del s
print('after del s     :', [sys.getrefcount(r) for r in d.lol])
big = Daf(lol=[[i]*50 for i in range(200000)], cols=[f'c{i}' for i in range(50)])
t=time.time(); shared = any(sys.getrefcount(r) > 2 for r in big.lol); print('scan 200k rows: %.3fs'%(time.time()-t))
t=time.time(); big.lol = [list(r) for r in big.lol]; print('copy 200k rows: %.3fs'%(time.time()-t))
