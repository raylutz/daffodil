import time
from daffodil.daf import Daf

def orig(d, f):
    for idx, row in enumerate(d):
        d.lol[idx] = list(f(row).values())

def klist_mut(d, f):            # existing by='row_klist': func mutates the KeyedList
    d.apply_in_place(f, by='row_klist')

def byname_update(d, f):        # by='row' through a KeyedList: write the returned dict back by name
    kidx = d._get_kidx()
    from daffodil.keyedlist import KeyedList
    for idx, row in enumerate(d.lol):
        kl = KeyedList(kidx, row)
        rec = f(dict(zip(d.hd, row)))
        for k, v in rec.items():
            if k in d.hd:
                kl[k] = v

def mk(kf=''):
    return Daf(lol=[['a',1,10],['b',2,20]], cols=['g','x','y'], keyfield=kf)

print('=== existing by=row_klist: what a function can do to the row')
def f_ok(kl):      kl['y'] = kl['y'] + 1
def f_reorder(kl): kl['y'] = 1; kl['g'] = 'z'; kl['x'] = 9
def f_new(kl):     kl['new'] = 5
def f_del(kl):     del kl['x']
def f_key(kl):     kl['g'] = kl['g'].upper()
for name, f in [('set existing', f_ok), ('set 3 in any order', f_reorder), ('new key', f_new), ('delete key', f_del), ('change key col', f_key)]:
    d = mk('g')
    try:
        d.apply_in_place(f, by='row_klist'); print(f'{name:20}', d.lol, d.columns(), d.keys())
    except Exception as e: print(f'{name:20}', 'EXC', type(e).__name__, str(e)[:60])

print('=== by=row through a KeyedList, dict written back by name')
cases = {
 'normal':        lambda r: {**r, 'y': r['y']+1},
 'keys reordered':lambda r: {'y': r['y'], 'g': r['g'], 'x': r['x']},
 'only y':        lambda r: {'y': r['y']},
 'extra key':     lambda r: {**r, 'new': 5},
 'change key':    lambda r: {**r, 'g': r['g'].upper()},
}
for name, f in cases.items():
    d = mk('g'); byname_update(d, f); d._invalidate_kd()
    print(f'{name:16}', d.lol, d.columns(), d.keys())

print('=== speed, 200,000 rows')
big = [[i, i, i] for i in range(200_000)]
def timeit(label, fn):
    d = Daf(lol=[list(r) for r in big], cols=['g','x','y'])
    t = time.time(); fn(d); print(f'   {label:34} {time.time()-t:.3f}s')
timeit('orig by=row (dict, positional)', lambda d: orig(d, lambda r: {**r, 'y': r['y']+1}))
timeit('existing by=row_klist', lambda d: d.apply_in_place(lambda kl: kl.__setitem__('y', kl['y']+1), by='row_klist'))
timeit('by=row via klist, written by name', lambda d: byname_update(d, lambda r: {**r, 'y': r['y']+1}))

def byname_direct(d, f):        # C2: by='row', returned dict written back by name, no KeyedList object per row
    hd = d.hd
    for row_da, row in zip(d, d.lol):
        for k, v in f(row_da).items():
            i = hd.get(k)
            if i is not None:
                row[i] = v
print('=== C2 direct write-back by name')
for name, f in cases.items():
    d = mk('g'); byname_direct(d, f); d._invalidate_kd()
    print(f'{name:16}', d.lol)
timeit('C2 by=row, written back by name', lambda d: byname_direct(d, lambda r: {**r, 'y': r['y']+1}))
