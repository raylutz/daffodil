import time
from daffodil.daf import Daf

def mk(nrows, ncols):
    cols=[f'c{i}' for i in range(ncols)]
    return Daf(lol=[[str(r*ncols+i) for i in range(ncols)] for r in range(nrows)], cols=cols)
def best(nrows, ncols, fn, n=3):
    ts=[]
    for _ in range(n):
        d=mk(nrows,ncols); t=time.perf_counter(); fn(d); ts.append(time.perf_counter()-t)
    return min(ts)
to_int = lambda v: int(v) if v != '' else ''

def per_column(d, cols):                       # apply_to_col once per column
    for c in cols: d.apply_to_col(c, to_int)
def per_row_dict(d, cols):                     # apply_in_place by row, returning a dict of the changed columns
    d.apply_in_place(lambda r: {c: to_int(r[c]) for c in cols})
def per_row_klist(d, cols):                    # apply_in_place by row_klist
    def f(kl):
        for c in cols: kl[c] = to_int(kl[c])
    d.apply_in_place(f, by='row_klist')
def in_loop(d, cols):                          # a converter inside the loop that apply_dtypes already has
    idxs=[d.hd[c] for c in cols]
    for row in d.lol:
        for i in idxs: row[i] = to_int(row[i])
def builtin(d, cols):
    d.apply_dtypes(dtypes={c:int for c in cols})

print(f'{"rows x cols":>14} {"convert":>8} | {"apply_to_col":>12} {"row dict":>9} {"row_klist":>10} {"in-loop":>8} {"built-in int":>12}')
for nrows, ncols, nconv in [(200_000,3,1),(200_000,3,3),(2_000,1000,1),(2_000,1000,20),(2_000,1000,1000)]:
    cols=[f'c{i}' for i in range(nconv)]
    r=[best(nrows,ncols,lambda d,f=f: f(d,cols)) for f in (per_column, per_row_dict, per_row_klist, in_loop)]
    # the built-in apply_dtypes needs dtypes for every column or silent_error
    def b(d, cols=cols): d.apply_dtypes(dtypes={c:int for c in cols}, silent_error=True)
    bi=best(nrows,ncols,b)
    print(f'{nrows:>7}x{ncols:<6} {nconv:>8} | ' + ' '.join(f'{x:>{w}.3f}' for x,w in zip(r+[bi],(12,9,10,8,12))))
