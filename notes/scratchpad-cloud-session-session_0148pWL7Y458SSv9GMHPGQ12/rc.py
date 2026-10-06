import sys, timeit
from daffodil.daf import Daf
def mk(): return Daf(lol=[[1,'a'],[2,'b'],[3,'c']], cols=['id','v'], keyfield='id')
rc = sys.getrefcount
d = mk(); print('own: outer', rc(d.lol), ' row', rc(d.lol[1]))
s = d.select_irows([0,1]); print('selection s: s.outer', rc(s.lol), ' d.outer', rc(d.lol), ' row0', rc(d.lol[0]), ' row2 (not selected)', rc(d.lol[2]))
a = d.select_irows([0,1,2]); print('alias: a.lol is d.lol', a.lol is d.lol, ' outer', rc(d.lol), ' row', rc(d.lol[1]))
c = d.copy(); print('copy(): c.lol is d.lol', c.lol is d.lol, ' outer', rc(d.lol))
mine=[[1,'a']]; e = Daf(lol=mine, cols=['id','v']); print('Daf(lol=mine): e.lol is mine', e.lol is mine, ' outer', rc(e.lol))
# a cell write on the aliased selection today
d = mk(); a = d.select_irows([0,1,2]); a[0,'v']='X'; print('write on alias: d.lol[0] =', d.lol[0])
# cost of the two checks per cell write
d = mk()
print('two refcount reads: %.0f ns' % (timeit.timeit(lambda: (rc(d.lol), rc(d.lol[1])), number=200000)/200000*1e9))
print('d[1,"v"]=x today:   %.0f ns' % (timeit.timeit(lambda: d.__setitem__((1,'v'),'x'), number=200000)/200000*1e9))
