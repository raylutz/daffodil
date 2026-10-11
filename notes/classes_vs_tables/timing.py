import json, time, tracemalloc, fmt, make_data, impl_marshmallow as M, impl_pydantic as P, impl_plain as X, impl_daffodil as D
recs = make_data.make_records(100_000)
def best(f, n=3):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); r = f(); ts.append(time.perf_counter() - t)
    return min(ts), r
print(f'{"":12} {"load":>7} {"totals":>7} {"flag":>7} {"dump":>7} {"total":>7}   memory after load')
for name, mod in [('marshmallow', M), ('pydantic', P), ('plain', X), ('daffodil', D)]:
    text = fmt.encode(name, recs)
    tl, obj = best(lambda: mod.load(text))
    tt, _ = best(lambda: mod.totals(obj))
    tf, _ = best(lambda: mod.flag(obj), 1)
    td, _ = best(lambda: mod.dump(obj))
    tracemalloc.start(); o2 = mod.load(text); mem = tracemalloc.get_traced_memory()[0]; tracemalloc.stop(); del o2
    print(f'{name:12} {tl:7.3f} {tt:7.3f} {tf:7.3f} {td:7.3f} {tl+tt+tf+td:7.3f}   {mem/1e6:5.1f} MB')
