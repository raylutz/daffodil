import time
from daffodil.daf import Daf

def orig(d, f):
    for idx, row in enumerate(d):
        d.lol[idx] = list(f(row).values())

def optA(d, f):            # the suggestion, literally: append(respect_kd=True) inside the loop
    for n, row in enumerate(d):
        if n > 50: raise RuntimeError('did not stop: grew past 50 rows')
        d.append(f(row), respect_kd=True)

def optA2(d, f):           # same idea, safe: collect into a fresh Daf by name, then swap the rows in
    new = d.clone_empty()
    new.keyfield = ''
    for row in d:
        new.record_append(f(row), respect_kd=False)
    d.lol = new.lol

def optB(d, f):            # place by name at the same position, as assign_record_irow() does
    hd = d.hd
    for idx, row in enumerate(d):
        rec = f(row)
        d.lol[idx] = [rec.get(c, '') for c in hd]

def mk(kf=''):
    return Daf(lol=[['a',1,10],['b',2,20]], cols=['g','x','y'], keyfield=kf)

cases = {
 'normal (same key order)':      lambda r: {**r, 'y': r['y']+1},
 'keys reordered':               lambda r: {'y': r['y'], 'g': r['g'], 'x': r['x']},
 'key missing (only y)':         lambda r: {'y': r['y']},
 'extra key new':                lambda r: {**r, 'new': 5},
}
for kf in ['', 'g']:
    print('==== keyfield =', repr(kf))
    for cname, f in cases.items():
        print('--', cname)
        for oname, o in [('orig', orig), ('A append(respect_kd=True)', optA), ('A2 fresh Daf', optA2), ('B by name', optB)]:
            d = mk(kf)
            try: o(d, f); print(f'   {oname:28}', d.lol)
            except Exception as e: print(f'   {oname:28}', 'EXC', type(e).__name__, str(e)[:50])
print('==== func changes the key value (keyfield g)')
f = lambda r: {**r, 'g': r['g'].upper()}
for oname, o in [('orig', orig), ('A append(respect_kd=True)', optA), ('A2 fresh Daf', optA2), ('B by name', optB)]:
    d = mk('g')
    try: o(d, f); print(f'   {oname:28}', d.lol)
    except Exception as e: print(f'   {oname:28}', 'EXC', type(e).__name__, str(e)[:50])
print('==== speed, 200,000 rows, same-order dict')
big = [[i, i, i] for i in range(200_000)]
f = lambda r: {**r, 'y': r['y']+1}
for oname, o in [('orig', orig), ('A2 fresh Daf', optA2), ('B by name', optB)]:
    d = Daf(lol=[list(r) for r in big], cols=['g','x','y'])
    t=time.time(); o(d, f); print(f'   {oname:28} {time.time()-t:.3f}s')
