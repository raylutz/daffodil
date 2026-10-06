import copy as cp, time
from daffodil.daf import Daf

def mk(): return Daf(lol=[[1,' a ',10],[2,' b ',20],[3,' c ',30]],cols=['id','v','n'],keyfield='id',dtypes={'id':int,'v':str,'n':int})

def structural(d):
    c = cp.copy(d)
    c.lol = list(d.lol)
    c.hd = dict(d.hd)
    c.dtypes = dict(d.dtypes) if d.dtypes else d.dtypes
    c._invalidate_kd()
    c.attrs = cp.deepcopy(d.attrs)
    return c
def rows(d):
    c = structural(d); c.lol = [list(r) for r in d.lol]; return c

LEVELS = [('shallow', lambda d: d.copy()),
          ('for_sorting', lambda d: d.copy(for_sorting=True)),
          ('structural', structural),
          ('rows', rows),
          ('deep', lambda d: d.copy(deep=True))]

def snap(d): return (cp.deepcopy(d.lol), dict(d.hd), dict(d.dtypes or {}), d.keyfield)
def consistent(d):
    w = len(d.hd)
    if any(len(r) != w for r in d.lol): return False
    if d.keyfield and d._kd:
        try:
            ki = d.hd[d.keyfield]
            return all(d.lol[i][ki] == k for k, i in d._kd.items())
        except Exception: return False
    return True

def A(f): return f
ACTIONS = [
 ('append row',            'rows list',  lambda c: c.append([4,'d',40])),
 ('extend rows',           'rows list',  lambda c: c.extend([{'id':4,'v':'d','n':40}])),
 ('insert_irow',           'rows list',  lambda c: c.insert_irow(0,[0,'z',0])),
 ('remove_key',            'rows list',  lambda c: c.remove_key(2)),
 ('sort_by_colname',       'rows list',  lambda c: c.sort_by_colname('v',reverse=True)),
 ('set_lol',               'rows list',  lambda c: c.set_lol([[9,'q',9]])),
 ('set_keyfield',          'header',     lambda c: c.set_keyfield('v')),
 ('set_cols (rename)',     'header',     lambda c: c.set_cols(['x','y','z'])),
 ('rename_cols',           'header',     lambda c: c.rename_cols({'v':'vv'})),
 ('set_dtypes',            'header',     lambda c: c.set_dtypes({'id':float})),
 ('assign_col (add col)',  'columns',    lambda c: c.assign_col('new',[1,2,3])),
 ('insert_col',            'columns',    lambda c: c.insert_col('new2',1,[1,2,3]) if False else c.insert_col('new2',[1,2,3],1)),
 ('drop_cols',             'columns',    lambda c: c.drop_cols(['v'])),
 ('set_icol (cells)',      'cells',      lambda c: c.set_icol(1,'X')),
 ('set_col_irows',         'cells',      lambda c: c.set_col_irows('v',[0],'X')),
 ('c[0,"v"]= (setitem)',   'cells',      lambda c: c.__setitem__((0,'v'),'X')),
 ('replace_in_columns',    'cells',      lambda c: c.replace_in_columns(['v'],' a ','Q') if True else None),
 ('apply_in_place row',    'cells',      lambda c: c.apply_in_place(lambda r: {**r,'v':'Z'}, by='row')),
 ('apply_in_place klist','cells',lambda c: c.apply_in_place(lambda r: r.__setitem__('v','Z') or r, by='row_klist')),
 ('strip',                 'cells',      lambda c: c.strip()),
 ('lol[0][1]= direct',     'cells',      lambda c: c.lol[0].__setitem__(1,'X')),
]

def run():
    res = {}
    for aname, grp, act in ACTIONS:
        for lname, mkc in LEVELS:
            d = mk(); d.keys()
            s0 = snap(d); c = mkc(d); c.keys() if c.keyfield else None
            try:
                act(c); err = None
            except Exception as e:
                err = type(e).__name__
            changed = snap(d) != s0
            ok_o = consistent(d)
            ok_c = consistent(c) and err is None
            if err: v = 'err:'+err
            elif not ok_o: v = 'ORIG BROKEN'
            elif changed: v = 'orig changed'
            elif not ok_c: v = 'copy broken'
            else: v = 'safe'
            res[(aname,lname)] = v
    return res
res = run()
names=[l for l,_ in LEVELS]
print(f"{'action':24s}{'group':10s}"+''.join(f"{n:>14s}" for n in names))
for aname, grp, _ in ACTIONS:
    print(f"{aname:24s}{grp:10s}"+''.join(f"{res[(aname,n)]:>14s}" for n in names))

R,C=200_000,50
big=Daf(lol=[[i]*C for i in range(R)],cols=[f'c{i}' for i in range(C)],keyfield='c0'); big.keys()
import tracemalloc
print()
for lname, mkc in LEVELS:
    tracemalloc.start(); t=time.perf_counter(); c=mkc(big); dt=time.perf_counter()-t
    cur,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
    print(f"{lname:12s}{dt:9.4f} s  extra memory {cur/1e6:8.1f} MB"); del c
