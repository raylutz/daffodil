import sys
from daffodil.daf import Daf
rc = sys.getrefcount
def mk(): return Daf(lol=[[1,'a'],[2,'b'],[3,'c']], cols=['id','v'], keyfield='id')

d = mk(); a = d.select_irows([0,1,2])           # alias, nothing else alive
print('alias only: same list', a.lol is d.lol, '| outer', rc(d.lol), '| rows', [rc(r) for r in d.lol])
d = mk(); c = d.copy()                          # default shallow copy
print('copy():     same list', c.lol is d.lol, '| outer', rc(d.lol), '| rows', [rc(r) for r in d.lol])
d = mk(); b = d                                 # two names, one Daf object
print('b = d:      same Daf', b is d, '| outer', rc(d.lol), '| rows', [rc(r) for r in d.lol])

# prototype of the write: outer first, then row
OWN_OUTER, OWN_ROW = 2, 2
def write(daf, irow, icol, val):
    if rc(daf.lol) > OWN_OUTER:
        daf.lol = list(daf.lol)
    if rc(daf.lol[irow]) > OWN_ROW + 0:
        daf.lol[irow] = list(daf.lol[irow])
    daf.lol[irow][icol] = val

for name, mkpair in [('alias', lambda d: d.select_irows([0,1,2])),
                     ('copy()', lambda d: d.copy()),
                     ('selection', lambda d: d.select_irows([0,1]))]:
    d = mk(); s = mkpair(d)
    write(s, 0, 1, 'X')
    print(f'write on {name:9}: d={d.lol}  s={s.lol}')
d = mk(); b = d; write(b, 0, 1, 'X'); print('write via b = d:      d =', d.lol, '(same object, both names see it)')
