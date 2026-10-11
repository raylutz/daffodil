import time, tracemalloc, gc, gb_data, gb_classes as C, gb_daffodil as D
rows = gb_data.make_grades(10_000)          # 200,000 grades
texts = {'classes': C.dump(C.build(rows)), 'daffodil': D.dump(D.build(rows))}
def best(f, n=3):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); r = f(); ts.append(time.perf_counter() - t)
    return min(ts), r
print(f'{len(rows):,} grades. File sizes: JSON {len(texts["classes"])/1e6:.1f} MB, CSV {len(texts["daffodil"])/1e6:.1f} MB')
print(f'{"":10} {"load":>7} {"averages":>9} {"math avg":>9} {"dump":>7}   memory')
for name, mod, subj in [('classes', C, lambda b: b.subject_average('math')), ('daffodil', D, lambda b: D.subject_average(b, 'math'))]:
    tl, book = best(lambda: mod.load(texts[name]))
    ta, _ = best(lambda: mod.student_averages(book))
    ts, _ = best(lambda: subj(book))
    td, _ = best(lambda: mod.dump(book))
    gc.collect(); tracemalloc.start(); b2 = mod.load(texts[name]); gc.collect()
    mem = tracemalloc.get_traced_memory()[0]; tracemalloc.stop(); del b2
    print(f'{name:10} {tl:7.3f} {ta:9.3f} {ts:9.3f} {td:7.3f}   {mem/1e6:5.1f} MB')
