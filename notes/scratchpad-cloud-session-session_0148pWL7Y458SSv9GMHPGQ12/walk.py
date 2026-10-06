import time
from daffodil.daf import Daf
def mk(): return Daf(lol=[[1,'a'],[2,'b'],[3,'c'],[4,'d']], cols=['id','v'], keyfield='id')

print('1. selection of rows 0 and 2, then a read-only klist loop')
d = mk(); s = d.select_irows([0,2]); ids = [id(r) for r in s.lol]
seen = [kl['v'] for kl in s.iter_klist()]
print('   read', seen, '| rows still the same objects:', [id(r) for r in s.lol]==ids, '| shared with d:', s.lol[0] is d.lol[0])

print('2. same selection, klist loop that writes every row')
d = mk(); s = d.select_irows([0,2])
for kl in s.iter_klist(): kl['v'] = kl['v'].upper()
print('   s =', s.lol); print('   d =', d.lol)

print('3. loop that writes only the second row of the selection')
d = mk(); s = d.select_irows([0,2])
for i, kl in enumerate(s.iter_klist()):
    if i == 1: kl['v'] = 'Z'
print('   s =', s.lol, '| d =', d.lol)
print('   row 0 still shared:', s.lol[0] is d.lol[0], '| row 1 shared:', s.lol[1] is d.lol[2])

print('4. loop on the original while a selection exists')
d = mk(); s = d.select_irows([0,2])
for kl in d.iter_klist(): kl['v'] = '*'
print('   d =', d.lol); print('   s =', s.lol)

print('5. Daf with its own rows: writes go in place, no copy')
d = mk(); ids = [id(r) for r in d.lol]
for kl in d.iter_klist(): kl['v'] = '*'
print('   d =', d.lol, '| same row objects:', [id(r) for r in d.lol]==ids)

print('6. alias a = d.select_irows(all in order), then a loop that writes')
d = mk(); a = d.select_irows([0,1,2,3])
for kl in a.iter_klist(): kl['v'] = '#'
print('   a =', a.lol); print('   d =', d.lol)

print('7. klists held in a list, then written later')
d = mk(); s = d.select_irows([0,2])
held = list(s.iter_klist())
for kl in held: kl['v'] = 'H'
print('   s =', s.lol, '| d =', d.lol)

print('8. a held klist, rows sorted before the write')
d = mk(); s = d.select_irows([0,2]); it = s.iter_klist(); kl = next(it)
s.lol.reverse(); kl['v'] = 'W'
print('   s =', s.lol, '| d =', d.lol)

print('9. read-only loop speed, 200,000 rows x 3 cols')
big = Daf(lol=[[i,i,i] for i in range(200000)], cols=['a','b','c'])
sel = big.select_irows(list(range(0,200000,2)))
for name, x in (('own rows ', big), ('shared   ', sel)):
    t=time.time(); n=sum(kl['a'] for kl in x.iter_klist()); print(f'   {name}: {time.time()-t:.3f}s ({len(x)} rows)')
