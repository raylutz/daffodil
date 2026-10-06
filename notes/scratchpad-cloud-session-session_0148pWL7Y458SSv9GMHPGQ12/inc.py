import csv, io, time, tracemalloc, operator
from daffodil.daf import Daf

nrows, ncols = 2_000, 1_000
header = ','.join(f'c{i}' for i in range(ncols))
body = '\n'.join(','.join(str(r*ncols+i) for i in range(ncols)) for r in range(nrows))
text = header + '\n' + body + '\n'
want = ['c5', 'c900', 'c1']

def B(text, include_cols):
    reader = csv.reader(io.StringIO(text))
    cols = next(reader)
    pos = {c: i for i, c in enumerate(cols)}
    missing = [c for c in include_cols if c not in pos]
    if missing: raise KeyError(f"include_cols: not in the file: {missing}")
    idxs = [pos[c] for c in include_cols]
    pick = operator.itemgetter(*idxs)
    lol = [list(pick(row)) if len(idxs) > 1 else [pick(row)] for row in reader]
    return Daf(cols=list(include_cols), lol=lol)

def full_then_select(text, include_cols):          # what a caller does today: read all, then select
    d = Daf.from_csv_buff(text)
    return d[:, include_cols]

def best(f, n=3):
    ts=[]
    for _ in range(n):
        t=time.perf_counter(); r=f(); ts.append(time.perf_counter()-t)
    return min(ts)
def peak(f):
    tracemalloc.start(); r=f(); cur, pk = tracemalloc.get_traced_memory(); tracemalloc.stop(); return cur/1e6, pk/1e6

print(f'file: {nrows} rows x {ncols} columns, text {len(text)/1e6:.1f} MB; wanted columns {want}')
a = full_then_select(text, want); b = B(text, want)
print('same rows:', a.lol == b.lol, '| columns', b.columns(), '| first row', b.lol[0])
print(f'time   read all then select   {best(lambda: full_then_select(text, want)):.3f}s     read only the columns {best(lambda: B(text, want)):.3f}s')
ca, pa = peak(lambda: full_then_select(text, want)); cb, pb = peak(lambda: B(text, want))
print(f'memory kept after the call    {ca:.1f} MB (the selected Daf)      {cb:.1f} MB')
print(f'peak while reading            {pa:.1f} MB                        {pb:.1f} MB')
print('what the library does today with include_cols:')
d = Daf.from_csv_buff(text[:200] + '\n', include_cols=['c1']) if False else None
small = 'a,b,c\n1,2,3\n'
r = Daf.from_csv_buff(small, include_cols=['b']); print('   columns', r.columns(), r.lol)
try: B(small, ['zz'])
except KeyError as e: print('B, unknown name:', e)
