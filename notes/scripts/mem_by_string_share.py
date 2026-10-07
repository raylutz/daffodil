"""Kept memory of a 1000 x 1000 table: Daffodil vs pandas, by share of string columns and by route into pandas."""
import sys, gc, tracemalloc, subprocess
R, C = 1000, 1000

def make(nstr, big):
    m = 1_000_003 if big else 100
    return [[f's{i}_{j}' if j < nstr else (i*7919 + j*104729) % m for j in range(C)] for i in range(R)]

def one(kind, nstr, big):
    import pandas as pd
    from daffodil.daf import Daf
    cols = [f'c{j}' for j in range(C)]
    gc.collect(); tracemalloc.start()
    lol = make(nstr, big)
    base = tracemalloc.get_traced_memory()[0]          # the cells themselves
    if kind == 'daf':
        obj = Daf(lol=lol, cols=cols)
    elif kind == 'pandas':
        obj = pd.DataFrame(lol, columns=cols)
    elif kind == 'pandas via to_pandas_df':
        obj = Daf(lol=lol, cols=cols).to_pandas_df()
    if kind != 'daf':
        del lol
    gc.collect()
    kept = tracemalloc.get_traced_memory()[0]
    extra = ''
    if kind != 'daf':
        extra = f" dtypes={dict(obj.dtypes.astype(str).value_counts())} deep={obj.memory_usage(deep=True).sum()/1e6:.0f}MB"
    print(f'{kept/1e6:.0f}{extra}')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        one(sys.argv[1], int(sys.argv[2]), sys.argv[3] == 'big'); sys.exit()
    for big in ('small', 'big'):
        print(f'\nints {"0..1,000,002" if big=="big" else "0..99"}; kept MB (tracemalloc)')
        for nstr in (1, 100, 500, 1000):
            row = []
            for kind in ('daf', 'pandas', 'pandas via to_pandas_df'):
                out = subprocess.run([sys.executable, __file__, kind, str(nstr), big], capture_output=True, text=True)
                row.append(f'{kind}: {out.stdout.strip() or out.stderr[-300:]}')
            print(f'{nstr:>4} str cols | ' + ' | '.join(row))
