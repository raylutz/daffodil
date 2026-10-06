import time
from daffodil.daf import Daf
def mk(): return Daf(lol=[[1,'a'],[2,'b'],[3,'c']], cols=['id','v'], keyfield='id')
d = mk(); n = len(d)
paths = {
 'select_irows(list all in order)': lambda d: d.select_irows(list(range(n))),
 'select_irows(range all)':         lambda d: d.select_irows(range(n)),
 'select_irows(slice all)':         lambda d: d.select_irows(slice(None)),
 'select_irows([], inverse=True)':  lambda d: d.select_irows([], inverse=True),
 'd[list all in order]':            lambda d: d[list(range(n))],
 'd[:]':                            lambda d: d[:],
 'select_krows(all keys)':          lambda d: d.select_krows([1,2,3]),
 'select_krows([], inverse=True)':  lambda d: d.select_krows([], inverse=True),
 'select_records_daf([], inverse)': lambda d: d.select_records_daf([], inverse=True),
 'select_where(all true)':          lambda d: d.select_where(lambda r: True),
 'copy()':                          lambda d: d.copy(),
 'copy("sortable")':                lambda d: d.copy('sortable'),
}
print('same row list object as the original?')
for k,f in paths.items():
    try: print(f'  {k:34}', f(mk()).lol is None or f(d).lol is d.lol)
    except Exception as e: print(f'  {k:34} raised {type(e).__name__}: {e}')

print('\nscenarios on s = select_irows(list all in order)')
def sc(name, act):
    d = mk(); s = d.select_irows(list(range(len(d))))
    try: act(d, s); err=''
    except Exception as e: err=f' raised {type(e).__name__}'
    print(f'  {name:32} d.lol={d.lol}  s.lol={s.lol}{err}')
sc('s.append([4,"d"])',        lambda d,s: s.append([4,'d']))
sc('d.append([4,"d"])',        lambda d,s: d.append([4,'d']))
sc('s.remove_key(2)',          lambda d,s: s.remove_key(2))
sc('s.sort_by_colname("v",rev)',lambda d,s: s.sort_by_colname('v', reverse=True))
sc('s.lol.sort(reverse)',      lambda d,s: s.lol.sort(key=lambda r:-r[0]))
sc('s.insert_irow(0,[0,"z"])', lambda d,s: s.insert_irow(0,[0,'z']))
sc('s.lol.pop()',              lambda d,s: s.lol.pop())

print('\nkey index after d.append when s was built first')
d = mk(); s = d.select_irows(list(range(len(d)))); s.keys(); d.append([4,'d'])
try: print('  s has key 4:', 4 in s.keys(), ' len(s):', len(s), ' len(d):', len(d))
except Exception as e: print('  raised', type(e).__name__, e)

print('\ncost, 200,000 rows x 50 cols')
big = Daf(lol=[[i]*50 for i in range(200000)], cols=[f'c{i}' for i in range(50)])
t=time.time(); big.select_irows(list(range(200000))); print('  select_irows(all list): %.4fs' % (time.time()-t))
