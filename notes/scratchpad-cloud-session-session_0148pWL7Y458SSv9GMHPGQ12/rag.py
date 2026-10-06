from daffodil.daf import Daf
csv_text = "id,name,age\n1,Ann,30\n2,Bob\n3,Cy,40,extra\n"
d = Daf.from_csv_buff(csv_text)
print('lol          :', d.lol)
print('columns      :', d.columns())
print('is_rectangular:', d.is_rectangular() if hasattr(d,'is_rectangular') else 'no method')
for label, f in (('row 2 as dict', lambda: d[1].to_dict() if hasattr(d[1],'to_dict') else d[1]),
                 ('col age      ', lambda: d.col('age')),
                 ('to_md        ', lambda: d.to_md()),
                 ('select_by_dict age=40', lambda: d.select_by_dict({'age': '40'}).lol),
                 ('iter_dict    ', lambda: list(d.iter_dict()))):
    try: print(label, '->', f())
    except Exception as e: print(label, '->', type(e).__name__ + ':', str(e)[:90])
