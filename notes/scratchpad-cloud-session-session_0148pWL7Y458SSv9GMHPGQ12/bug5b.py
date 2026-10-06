import io, re, zipfile
import xlsxwriter
from daffodil.lib import daf_utils as u
def xlsx(rows):
    buf = io.BytesIO(); wb = xlsxwriter.Workbook(buf, {'in_memory': True}); ws = wb.add_worksheet()
    for i, r in enumerate(rows): ws.write_row(i, 0, r)
    wb.close(); return buf.getvalue()
def strip_dimension(data):
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as zin, zipfile.ZipFile(out, 'w') as zout:
        for item in zin.infolist():
            b = zin.read(item.filename)
            if item.filename.startswith('xl/worksheets/sheet'):
                b = re.sub(rb'<dimension[^>]*/>', b'', b)
            zout.writestr(item, b)
    return out.getvalue()
rows = [['a', 'b', 'c'], [1, 2], [3, 4, 5], [6]]
d = xlsx(rows)
print("normal file, padding off   :", u.xlsx_to_csv(d, add_trailing_blank_cols=False))
d2 = strip_dimension(d)
print("no <dimension>, padding off:", u.xlsx_to_csv(d2, add_trailing_blank_cols=False))
print("no <dimension>, padding on :", u.xlsx_to_csv(d2))
print("text with wide 4th row     :", repr(u.add_trailing_columns_csv('a,b\n1,2\n3,4\n5,6,7\n')))
