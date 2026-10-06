import io, timeit
import xlsxwriter
from daffodil.daf import Daf
from daffodil.lib import daf_utils as u
def xlsx(rows):
    buf = io.BytesIO(); wb = xlsxwriter.Workbook(buf, {'in_memory': True}); ws = wb.add_worksheet()
    for i, r in enumerate(rows): ws.write_row(i, 0, r)
    wb.close(); return buf.getvalue()
def run(label, fn):
    try: print(f"{label}: {fn()!r}")
    except Exception as e: print(f"{label}: {type(e).__name__}: {e}")
run("2 rows, from_excel_buff     ", lambda: Daf.from_excel_buff(xlsx([['a', 'b', 'c'], [1, 2]])).lol)
run("1 row (header only)         ", lambda: Daf.from_excel_buff(xlsx([['a', 'b', 'c']])).lol)
run("3 rows                      ", lambda: Daf.from_excel_buff(xlsx([['a', 'b', 'c'], [1, 2], [3, 4, 5]])).lol)
run("later row wider (4th row)   ", lambda: u.xlsx_to_csv(xlsx([['a', 'b'], [1, 2], [3, 4], [5, 6, 7]])))
run("empty csv text              ", lambda: u.add_trailing_columns_csv(''))
big = "\n".join(",".join(str(i * j) for j in range(10)) for i in range(200000)) + "\n"
t = min(timeit.repeat(lambda: u.add_trailing_columns_csv(big), number=1, repeat=3))
print(f"200,000 rows x 10 cols      : {t:.2f} s")
