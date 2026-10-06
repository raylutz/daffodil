import time
from daffodil.daf import Daf
def mk(nrows, ncols, ngroups):
    cols = [f'c{i}' for i in range(ncols)]
    lol = [[r % ngroups] + [r * ncols + i for i in range(1, ncols)] for r in range(nrows)]
    return Daf(lol=lol, cols=cols)
def best(f, n=3):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); r = f(); ts.append(time.perf_counter() - t)
    return min(ts), r
def multi_groupby_B(d, groupby_colnames, colnames=None):
    """groups hold only `colnames` (all columns if None). The rows are made by picking indexes, with no dict per row."""
    gcols = [groupby_colnames] if isinstance(groupby_colnames, str) else list(groupby_colnames)
    keep = list(colnames) if colnames else d.columns()
    idxs = [d.hd[c] for c in keep]
    out = {}
    for g in gcols:
        gi = d.hd[g]; groups = {}
        for row in d.lol:
            groups.setdefault(row[gi], []).append([row[i] for i in idxs])
        out[g] = {k: Daf(lol=v, cols=keep) for k, v in groups.items()}
    return out
print('seconds, best of 3. Group by one column, 20 groups.')
for nrows, ncols in [(20_000, 10), (2_000, 100), (2_000, 1000)]:
    d = mk(nrows, ncols, 20)
    t_now, a = best(lambda: d.multi_groupby(['c0']))
    t_all, b = best(lambda: multi_groupby_B(d, ['c0']))
    t_few, c = best(lambda: multi_groupby_B(d, ['c0'], colnames=['c1', 'c2', 'c3']))
    same = all(a['c0'][k].lol == b['c0'][k].lol for k in a['c0'])
    print(f'{nrows:>6} x {ncols:<5} multi_groupby now {t_now:.3f}   same, new code, all columns {t_all:.3f} (same rows: {same})   colnames=3 columns {t_few:.4f}')
print()
d = mk(2_000, 1000, 20)
t_now, r1 = best(lambda: d.groupby_reduce('c0', Daf.sum_da, reduce_cols=['c1', 'c2', 'c3']))
print(f'groupby_reduce on 2,000 x 1,000 with 3 reduce columns, now: {t_now:.3f}s')

def groupby_B(d, colname, cols=None):
    keep = list(cols) if cols else d.columns()
    idxs = [d.hd[c] for c in keep]; gi = d.hd[colname]; groups = {}
    for row in d.lol:
        groups.setdefault(row[gi], []).append([row[i] for i in idxs])
    return {k: Daf(lol=v, cols=keep) for k, v in groups.items()}
def groupby_reduce_C(d, colname, func, reduce_cols):
    grouped = groupby_B(d, colname, cols=reduce_cols)          # groups hold only the reduce columns
    return Daf.reduce_dodaf_to_daf(colname=colname, func=func, grouped_dodaf=grouped, reduce_cols=list(reduce_cols))
for nrows, ncols in [(20_000, 10), (2_000, 100), (2_000, 1000)]:
    d = mk(nrows, ncols, 20)
    rc = ['c1', 'c2', 'c3']
    t_now, a = best(lambda: d.groupby_reduce('c0', Daf.sum_da, reduce_cols=rc))
    t_c, b = best(lambda: groupby_reduce_C(d, 'c0', Daf.sum_da, rc))
    same = {r[0]: r for r in a.lol} and all([r[a.hd['c1']] for r in a.lol] == [r[b.hd['c1']] for r in b.lol] for _ in [0])
    print(f'groupby_reduce {nrows:>6} x {ncols:<5} 3 reduce columns: now {t_now:.4f}s   groups hold only those columns {t_c:.4f}s   same sums: {[r[a.hd["c1"]] for r in a.lol]==[r[b.hd["c1"]] for r in b.lol]}')
