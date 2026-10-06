import sys, json
from daffodil.daf import Daf
def show(d):
    if isinstance(d, Daf): return {'lol': d.lol, 'cols': d.columns(), 'key': repr(d.keyfield), 'dtypes': {k: getattr(v,'__name__',str(v)) for k,v in (d.dtypes or {}).items()}}
    if isinstance(d, dict): return {repr(k): show(v) for k, v in d.items()}
    return repr(d)
def mk(kf=''):
    return Daf(lol=[['a',1,10,'p'],['b',2,20,'q'],['a',3,30,'r'],['c','',40,'s'],['b',5,50,'t']], cols=['g','x','y','z'], keyfield=kf, dtypes={'g':str,'x':int,'y':int,'z':str})
out = {}
def run(name, f):
    try: out[name] = show(f())
    except Exception as e: out[name] = f'EXC {type(e).__name__}'
for kf in ['', 'y']:
    d = mk(kf); k = f'kf={kf!r} '
    run(k+'groupby g', lambda: d.groupby('g')); run(k+'groupby omit', lambda: d.groupby('x', omit_nulls=True))
    run(k+'groupby colnames', lambda: d.groupby(colnames=['g','z'])); run(k+'groupby list', lambda: d.groupby(colname=['g','z']))
    run(k+'groupby one in colnames', lambda: d.groupby(colnames=['g']))
    run(k+'groupby_cols', lambda: d.groupby_cols(['g'])); run(k+'groupby_cols2', lambda: d.groupby_cols(['g','z']))
    run(k+'multi_groupby', lambda: d.multi_groupby(['g','z'])); run(k+'multi_groupby str', lambda: d.multi_groupby('g'))
    run(k+'multi_groupby omit', lambda: d.multi_groupby(['x'], omit_nulls=True))
    run(k+'groupby_reduce', lambda: d.groupby_reduce('g', Daf.sum_da, reduce_cols=['x','y']))
    run(k+'groupby_reduce none', lambda: d.groupby_reduce('g', Daf.sum_da))
    run(k+'groupby_reduce y only', lambda: d.groupby_reduce('g', Daf.sum_da, reduce_cols=['y']))
    run(k+'groupby_reduce unknown col', lambda: d.groupby_reduce('g', Daf.sum_da, reduce_cols=['y','nope']))
    run(k+'groupby_reduce all unknown', lambda: d.groupby_reduce('g', Daf.sum_da, reduce_cols=['nope']))
    run(k+'groupsum_daf', lambda: d.groupsum_daf('g', reduce_cols=['y']))
    run(k+'groupby_cols_reduce', lambda: d.groupby_cols_reduce(['g','z'], Daf.sum_da, reduce_cols=['x','y']))
    run(k+'groupby_cols_reduce none', lambda: d.groupby_cols_reduce(['g'], Daf.sum_da))
    run(k+'multi_groupby_reduce', lambda: d.multi_groupby_reduce(['g','z'], Daf.sum_da, reduce_cols=['y']))
    run(k+'multi_groupsum', lambda: d.multi_groupsum(['g'], reduce_cols=['x','y']))
    run(k+'groupby_reduce count', lambda: d.groupby_reduce('g', Daf.count_values_da, reduce_cols=['z']))
    run(k+'groupby_reduce table', lambda: d.groupby_reduce('g', lambda daf, cols: {'n': len(daf)}, by='table'))
    run(k+'groupby_reduce col', lambda: d.groupby_reduce('g', lambda col_la, red, **kw: red+[len(col_la)], by='col'))
sp = Daf(lol=[['a',{'u':1}],['b',{'u':2,'w':3}],['a',{'w':4}]], cols=['g','extra'])
run('sparse groupby_reduce', lambda: sp.groupby_reduce('g', Daf.sum_da, by='sparse_row', reduce_cols=['u','w'], indirect_col='extra'))
json.dump(out, open(sys.argv[1], 'w'), indent=1, sort_keys=True, default=str)
print('scenarios', len(out))
