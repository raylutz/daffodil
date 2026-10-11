"""The size of the file each version saves, as it saves it, and gzipped."""
import gzip, sys
sys.path.insert(0, 'gradebook')
import make_data, fmt, impl_marshmallow as M, impl_pydantic as P, impl_plain as X, impl_daffodil as D
import gb_data, gb_classes as GC, gb_classes_plain as GX, gb_daffodil as GD

def sizes(text: str) -> tuple:
    raw = text.encode()
    return len(raw), len(gzip.compress(raw, compresslevel=6))

def show(title: str, rows: list) -> None:
    print(f'\n{title}')
    print(f'  {"":16} {"format":7} {"bytes":>11} {"gzipped":>10}  per record')
    base = rows[-1][2]
    for label, form, text, n in rows:
        raw, gz = sizes(text)
        print(f'  {label:16} {form:7} {raw:>11,} {gz:>10,}  {raw / n:6.1f} B   {raw / sizes(base)[0]:.1f}x')

recs = make_data.make_records(100_000)
rows = []
for label, mod in [('marshmallow', M), ('pydantic', P), ('plain classes', X), ('daffodil', D)]:
    obj = mod.load(fmt.encode('daffodil' if mod is D else label, recs))
    mod.flag(obj)
    rows.append((label, 'CSV' if mod is D else 'JSON', mod.dump(obj), len(recs)))
show('Ballot batches, 100,000 records, after flag', rows)

grades = gb_data.make_grades(10_000)
rows = [(label, form, mod.dump(mod.build(grades)), len(grades))
        for label, form, mod in [('pydantic', 'JSON', GC), ('plain classes', 'JSON', GX), ('daffodil table', 'CSV', GD)]]
show('Gradebook, 200,000 grades', rows)
