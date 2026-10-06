import time
from daffodil.daf import Daf

def optB(d, id_cols, varname_col='variable', value_col='value', wide_cols=None):
    """group by id with a dict, any order. Columns are the names in first-seen order, or wide_cols."""
    rows = {}
    names = {}
    for row in d:
        idt = tuple(row[c] for c in id_cols)
        rec = rows.get(idt)
        if rec is None:
            rec = rows[idt] = {}
        rec[row[varname_col]] = row[value_col]
        names[row[varname_col]] = None
    cols = list(id_cols) + (list(wide_cols) if wide_cols else list(names))
    lol = [list(idt) + [rec.get(n, '') for n in cols[len(id_cols):]] for idt, rec in rows.items()]
    return Daf(lol=lol, cols=cols)

def optC(d, id_cols, varname_col='variable', value_col='value'):
    """keep the streaming scan, but raise if an id comes back after another id."""
    seen = set(); last = None
    for row in d:
        idt = tuple(row[c] for c in id_cols)
        if idt != last:
            if idt in seen: raise ValueError(f"narrow_to_wide: rows for id {idt} are not together")
            seen.add(idt); last = idt
    return d.narrow_to_wide(id_cols, varname_col, value_col)

cols=['id','variable','value']
tests = {
 'sorted':          [['x','a',1],['x','b',2],['y','a',3],['y','b',4]],
 'unsorted':        [['x','a',1],['y','a',3],['x','b',2],['y','b',4]],
 'missing combo':   [['x','a',1],['x','b',2],['y','a',3]],
 'var only in 2nd id': [['x','a',1],['y','a',3],['y','c',9]],
 'repeat (x,a)':    [['x','a',1],['x','a',5]],
}
for name, lol in tests.items():
    d = Daf(lol=[list(r) for r in lol], cols=cols)
    r0 = d.narrow_to_wide(['id'])
    rb = optB(d, ['id'])
    try: rc = optC(d, ['id']); rc = (rc.lol, rc.columns())
    except Exception as e: rc = f'{type(e).__name__}: {e}'
    print(f'-- {name}\n   orig {r0.lol} {r0.columns()}\n   B    {rb.lol} {rb.columns()}\n   C    {rc}')
d = Daf(lol=[['x','a',1],['y','a',3],['x','b',2]], cols=cols)
print('B with wide_cols=[b,a]:', optB(d,['id'],wide_cols=['b','a']).lol)
import random
big=[[i%50000, f'v{i//50000}', i] for i in range(200_000)]
random.seed(1); srt=sorted(big)
for label, lol in [('sorted by id', srt)]:
    d=Daf(lol=[list(r) for r in lol], cols=cols)
    t=time.time(); d.narrow_to_wide(['id']); t1=time.time()-t
    t=time.time(); optB(d,['id']); t2=time.time()-t
    t=time.time(); optC(d,['id']); t3=time.time()-t
    print(f'200,000 rows sorted: orig {t1:.3f}s  B {t2:.3f}s  C {t3:.3f}s')
