import time
from daffodil.daf import Daf
def best(mk, dtypes, n=3):
    ts=[]
    for _ in range(n):
        d=mk(); t=time.perf_counter(); d.apply_dtypes(dtypes=dtypes, silent_error=True); ts.append(time.perf_counter()-t)
    return min(ts)
out=[]
for nrows, ncols, label in [(200_000,3,'int'),(200_000,3,'float'),(2_000,1000,'int'),(2_000,1000,'float'),(2_000,1000,'str'),(2_000,1000,'mixed')]:
    cols=[f'c{i}' for i in range(ncols)]
    if label=='float': mk=lambda: Daf(lol=[[str(r*ncols+i+0.5) for i in range(ncols)] for r in range(nrows)], cols=cols); dt={c:float for c in cols}
    elif label=='str': mk=lambda: Daf(lol=[[str(r*ncols+i) for i in range(ncols)] for r in range(nrows)], cols=cols); dt={c:str for c in cols}
    elif label=='mixed':
        mk=lambda: Daf(lol=[[str(r*ncols+i) if i%3 else f'{r}.5' for i in range(ncols)] for r in range(nrows)], cols=cols)
        dt={c:(int if i%3==1 else float if i%3==0 else int) for i,c in enumerate(cols)}
    else: mk=lambda: Daf(lol=[[str(r*ncols+i) for i in range(ncols)] for r in range(nrows)], cols=cols); dt={c:int for c in cols}
    out.append(f'{nrows}x{ncols} {label}: {best(mk, dt):.3f}s')
print('\n'.join(out))
