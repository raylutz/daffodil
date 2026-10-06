import time
from daffodil.daf import Daf
def mk(nrows, ncols, blank_every):
    cols = [f'c{i}' for i in range(ncols)]
    return Daf(lol=[[('' if (blank_every and (r + i) % blank_every == 0) else r * ncols + i) for i in range(ncols)] for r in range(nrows)], cols=cols)
def best(f, n=3):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return min(ts)
for nrows, ncols, blank in [(200_000, 10, 0), (200_000, 10, 20), (20_000, 100, 20), (2_000, 1000, 20)]:
    d = mk(nrows, ncols, blank)
    print(f'{nrows:>7} x {ncols:<5} blanks {blank or "none":>5}:  to_donpa() {best(lambda: d.to_donpa()):.3f}s   to_donpa(default=0) {best(lambda: d.to_donpa(default=0)):.3f}s')
