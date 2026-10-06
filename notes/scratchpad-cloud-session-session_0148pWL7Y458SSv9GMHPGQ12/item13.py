import sys, os, tempfile, sqlite3
from types import MappingProxyType
sys.breakpointhook = lambda *a, **k: print("      (breakpoint reached)")
from daffodil.daf import Daf
from daffodil.lib import daf_utils as u, daf_sql
u.error_beep = lambda: None
daf_sql.logs.error_beep = lambda: None
def run(label, fn):
    try: print(f"{label}: returns {fn()!r}")
    except Exception as e: print(f"{label}: {type(e).__name__}: {e}")
run("test_strbool(object())      ", lambda: u.test_strbool(object()))
run("convert_type_value('x', set)", lambda: u.convert_type_value('x', set))
fp = os.path.join(tempfile.mkdtemp(), 'nodir', 'x.bin')
run("write_buff_to_fp bad dir    ", lambda: (u.write_buff_to_fp(b'1', fp, rtype='binary'), os.path.exists(fp)))
run("len_slice(slice('a','b'),5) ", lambda: u.len_slice(slice('a', 'b'), 5))
run("slice_to_range(slice('a',5))", lambda: u.slice_to_range(slice('a', 5), 10))
con = sqlite3.connect(':memory:'); con.execute('create table t (a)')
run("create_index bad column     ", lambda: daf_sql.create_index_at_cursor(con.cursor(), 'nope', 't'))
run("select_icols slice, ragged  ", lambda: Daf(lol=[[1, 2, 3], [4]], cols=['a', 'b', 'c']).select_icols(slice(0, 3)).lol)
run("record_append mappingproxy  ", lambda: Daf(lol=[[1, 2]], cols=['a', 'b']).record_append(MappingProxyType({'a': 10, 'b': 20})).lol)
run("compare_lists tuple ref     ", lambda: u.compare_lists(('a', 'b'), ['b', 'c']))
run("insert col, ragged rows     ", lambda: u.insert_col_in_lol_at_icol(1, ['X', 'Y'], [[1, 2, 3], [4, 5]]))
