import sys, time
from daffodil.daf import Daf
big = Daf(lol=[[i]*50 for i in range(200000)], cols=[f'c{i}' for i in range(50)])
def shared(lol): return any(sys.getrefcount(r) > 3 for r in lol)
t=time.time(); print('unshared ->', shared(big.lol), '%.3fs'%(time.time()-t))
s = big.select_irows(list(range(100000)))
t=time.time(); print('big, shared with s ->', shared(big.lol), '%.3fs'%(time.time()-t))
t=time.time(); print('s, shared with big ->', shared(s.lol), '%.3fs'%(time.time()-t))
alias = [r for r in big.lol[-1:]]   # a user keeps a row elsewhere
t=time.time(); print('only last row aliased ->', shared(big.lol), '%.3fs'%(time.time()-t))
