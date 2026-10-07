"""Build a table row by row, the pandas way and the Daffodil way. 1 str + 9 int columns by default.
Prints time and memory. Each case runs in its own process. Run: uv run python notes/scripts/bench_build_rows.py
"""
import sys, time, tracemalloc, gc
import os
N, NINT = int(os.environ.get('BENCH_ROWS', 200_000)), int(os.environ.get('BENCH_INTS', 9))   # set BENCH_ROWS and BENCH_INTS for another shape
cols = ['id'] + [f'c{i}' for i in range(NINT)]

def gen_rows(big):                  # incoming records, made one at a time with fresh objects
    m = 1_000_003 if big else 100
    for i in range(N):
        yield [f'rec{i:06d}'] + [(i * 7919 + j * 104729) % m for j in range(NINT)]

def run(name, big):
    import pandas as pd
    from daffodil.daf import Daf
    t0 = time.perf_counter()
    if name == 'pandas-lod':               # append a dict per row, then convert
        lod = []
        for r in gen_rows(big): lod.append(dict(zip(cols, r)))
        t1 = time.perf_counter(); obj = pd.DataFrame(lod); del lod
    elif name == 'pandas-lol':             # append a list per row, then convert
        lol = []
        for r in gen_rows(big): lol.append(r)
        t1 = time.perf_counter(); obj = pd.DataFrame(lol, columns=cols); del lol
    elif name == 'pandas-dol':             # append each value to its column list, then convert
        dol = {c: [] for c in cols}
        col_lists = list(dol.values())
        for r in gen_rows(big):
            for col_list, v in zip(col_lists, r): col_list.append(v)
        t1 = time.perf_counter(); obj = pd.DataFrame(dol); del dol, col_lists
    elif name == 'daf-append':             # append each row to the Daf
        obj = Daf(cols=cols)
        for r in gen_rows(big): obj.append(r)
        t1 = time.perf_counter()
    elif name == 'daf-lol':                # append a list per row, then wrap
        lol = []
        for r in gen_rows(big): lol.append(r)
        t1 = time.perf_counter(); obj = Daf(lol=lol, cols=cols); del lol
    t2 = time.perf_counter()
    return obj, t1 - t0, t2 - t1

if __name__ == '__main__':
    if len(sys.argv) > 1:
        name, big, mode = sys.argv[1], sys.argv[2] == 'big', sys.argv[3]
        import pandas, daffodil.daf   # imports outside the measurement
        gc.collect()
        if mode == 'mem':
            tracemalloc.start(); obj, *_ = run(name, big); gc.collect()
            cur, peak = tracemalloc.get_traced_memory(); print(f'{peak/1e6:.0f} {cur/1e6:.0f}')
        else:
            best = None
            for _ in range(3):
                obj, b, c = run(name, big); del obj; gc.collect()
                if best is None or b + c < sum(best): best = (b, c)
            print(f'{best[0]*1000:.0f} {best[1]*1000:.0f}')
        sys.exit()
    import subprocess, pandas, platform
    print(f'Python {platform.python_version()}, pandas {pandas.__version__}, {N:,} rows, 1 str + {NINT} int columns, best of 3')
    for big in ('small', 'big'):
        print(f'\nints: {"0 to 1,000,002 (distinct objects)" if big=="big" else "0 to 99 (shared small ints)"}')
        print(f'{"case":12} {"build ms":>9} {"convert ms":>10} {"total ms":>9} {"peak MB":>8} {"kept MB":>8}')
        for name in ('pandas-lod', 'pandas-lol', 'pandas-dol', 'daf-append', 'daf-lol'):
            t = subprocess.run([sys.executable, __file__, name, big, 'time'], capture_output=True, text=True).stdout.split()
            m = subprocess.run([sys.executable, __file__, name, big, 'mem'], capture_output=True, text=True).stdout.split()
            b, c = int(t[0]), int(t[1])
            print(f'{name:12} {b:>9} {c:>10} {b+c:>9} {m[0]:>8} {m[1]:>8}')
