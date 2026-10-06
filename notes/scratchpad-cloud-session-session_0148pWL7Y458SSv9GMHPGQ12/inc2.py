import csv, tracemalloc, operator
from daffodil.daf import Daf
nrows, ncols = 2_000, 1_000
def lines():                                   # a source that streams one line at a time, like S3's iter_lines()
    yield ','.join(f'c{i}' for i in range(ncols))
    for r in range(nrows):
        yield ','.join(str(r*ncols+i) for i in range(ncols))
def B(line_iter, include_cols):
    reader = csv.reader(line_iter)
    cols = next(reader); pos = {c: i for i, c in enumerate(cols)}
    idxs = [pos[c] for c in include_cols]; pick = operator.itemgetter(*idxs)
    return Daf(cols=list(include_cols), lol=[list(pick(row)) for row in reader])
tracemalloc.start(); d = Daf.from_csv_buff(lines()); d = d[:, ['c5','c900','c1']]; cur, pk = tracemalloc.get_traced_memory(); tracemalloc.stop()
print(f'streaming source, read all then select: peak {pk/1e6:6.1f} MB')
tracemalloc.start(); d2 = B(lines(), ['c5','c900','c1']); cur, pk = tracemalloc.get_traced_memory(); tracemalloc.stop()
print(f'streaming source, keep only 3 columns:  peak {pk/1e6:6.1f} MB   same rows: {d.lol == d2.lol}')
