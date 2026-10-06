import csv, io, itertools, timeit
from daffodil.lib import daf_utils as u

def opt2(str_csv, num_rows=3):
    buff = io.StringIO(str_csv); reader = csv.reader(buff)
    sample = list(itertools.islice(reader, num_rows))
    max_col = max((len(r) for r in sample), default=0)
    buff.seek(0); buff_out = io.StringIO()
    writer = csv.writer(buff_out, quoting=csv.QUOTE_MINIMAL, dialect='unix')
    for row in csv.reader(buff):
        writer.writerow(row + [''] * (max_col - len(row)))
    return buff_out.getvalue()

def opt3(str_csv, num_rows=3):
    rows = list(csv.reader(io.StringIO(str_csv)))
    max_col = max((len(r) for r in rows), default=0)
    buff_out = io.StringIO()
    writer = csv.writer(buff_out, quoting=csv.QUOTE_MINIMAL, dialect='unix')
    for row in rows:
        writer.writerow(row + [''] * (max_col - len(row)))
    return buff_out.getvalue()

cases = [("1 short row    ", 'a,b,c\n1,2\n'), ("header only    ", 'a,b,c\n'), ("empty          ", ''),
         ("wide 4th row   ", 'a,b\n1,2\n3,4\n5,6,7\n')]
for name, fn in (("original", u.add_trailing_columns_csv), ("option 2", opt2), ("option 3", opt3)):
    print(name)
    for label, txt in cases:
        try: print(f"  {label}: {fn(txt)!r}")
        except Exception as e: print(f"  {label}: {type(e).__name__}")
big = "\n".join(",".join(str(i * j) for j in range(10)) for i in range(200000)) + "\n"
for name, fn in (("original", u.add_trailing_columns_csv), ("option 2", opt2), ("option 3", opt3)):
    print(f"{name} 200,000 rows x 10 cols: {min(timeit.repeat(lambda: fn(big), number=1, repeat=3)):.2f} s")
