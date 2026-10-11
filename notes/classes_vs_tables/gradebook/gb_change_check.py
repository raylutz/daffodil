"""After adding a date to each grade: do both agree, and do old files still load?"""
import random, gb_data, gb_classes as C1, gb_daffodil as D1, gb_classes_v2 as C2, gb_daffodil_v2 as D2
rows = gb_data.make_grades(50)
rng = random.Random(3)
dated = [{**r, 'date': f'2026-{rng.randrange(1, 13):02d}-01'} for r in rows]
c, d = C2.build(dated), D2.build(dated)
ca, da = C2.student_averages(c, since='2026-07-01'), D2.student_averages(d, since='2026-07-01')
print('since July, same averages:', ca.keys() == da.keys() and all(abs(ca[k] - da[k]) < 1e-9 for k in ca), len(ca))
for name, old_text, mod in [('classes', C1.dump(C1.build(rows)), C2), ('daffodil', D1.dump(D1.build(rows)), D2)]:
    try:
        book = mod.load(old_text)
        avg = mod.student_averages(book)['s00000']
        print(f'{name:9} old file loads; average {avg:.6f}')
    except Exception as e:
        print(f'{name:9} old file fails: {type(e).__name__}: {str(e)[:150]}')
print('v1 average', f"{C1.student_averages(C1.build(rows))['s00000']:.6f}")
