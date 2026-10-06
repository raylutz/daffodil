import time
from daffodil.daf import Daf
from daffodil.lib import daf_utils

def new_name(d):
    """a spreadsheet style name for the next column, made unique."""
    base = daf_utils._calculate_single_column_name(len(d.hd))
    name = base; n = 1
    while name in d.hd:
        name = f'{base}_{n}'; n += 1
    return name

def assign_icol_B(d, icol=-1, col_la=None, default=''):
    orig_assign(d, icol, col_la, default)
    if d.hd and (icol < 0 or icol >= len(d.hd)):             # a column was added at the right
        d._cols_to_hd(list(d.hd) + [new_name(d)])

def insert_icol_B(d, icol=-1, col_la=None, colname='', default=''):
    if d.hd and not colname:
        colname = new_name(d)
    return orig_insert(d, icol, col_la, colname, default)

def annotate_B(d, other, mapping):
    for my_field in mapping:
        if my_field not in d.hd:
            d.assign_col(my_field)                           # add the column first, filled with NULL
    return orig_annotate(d, other, mapping)

def regex_select_B(d, col1, col2='', regex=''):
    if col2 and col2 not in d.hd:
        d.insert_col(col2)
    return orig_regex_select(d, col1, col2, regex)

def replace_regex_B(d, col, col2='', replace_regex=''):
    if col2 and col2 not in d.hd:
        d.insert_col(col2)
    return orig_replace_regex(d, col, col2, replace_regex)

orig_assign, orig_insert, orig_annotate = Daf.assign_icol, Daf.insert_icol, Daf.annotate_daf
orig_regex_select, orig_replace_regex = Daf.set_col2_from_col1_using_regex_select, Daf.apply_replace_regex

def mk(): return Daf(lol=[[1,'ab12'],[2,'cd34']], cols=['id','s'], keyfield='id')
def state(d, label):
    ok = all(len(r)==len(d.hd) for r in d.lol)
    print(f'-- {label}\n   columns {d.columns()}   row lengths {[len(r) for r in d.lol]}   names match data: {ok}\n   lol {d.lol}')
o = Daf(lol=[[1,'P'],[2,'Q']], cols=['id','w'], keyfield='id')
d = mk(); assign_icol_B(d, -1, ['x','y']); state(d, 'assign_icol(-1, [x, y])')
d = mk(); insert_icol_B(d, 1, ['x','y']); state(d, 'insert_icol(1, [x, y]) with no colname')
d = mk(); insert_icol_B(d, 1, ['x','y'], colname='mine'); state(d, 'insert_icol(1, [x, y], colname="mine")   as before')
d = Daf(lol=[[1,'a']]); insert_icol_B(d, 1, ['x']); state(d, 'insert_icol on a Daf with no column names   as before')
d = Daf(lol=[[1,2,3]], cols=['A','B','C']); assign_icol_B(d, -1, [9]); state(d, "a Daf that already has a column named D... names 'A','B','C' then added")
d = Daf(lol=[[1,2]], cols=['id','C']); assign_icol_B(d, -1, [9]); state(d, "a generated name that is taken: columns id, C, adding at index 2 gives C?")
d = mk(); annotate_B(d, o, {'newcol':'w'}); state(d, "annotate_daf(o, {'newcol': 'w'})")
d = mk(); regex_select_B(d, 's', 'n', r'(\d+)'); state(d, "set_col2_from_col1_using_regex_select('s', 'n')")
d = mk(); replace_regex_B(d, 's', 't', '/ab//'); state(d, "apply_replace_regex('s', 't')")
d = mk(); regex_select_B(d, 's', regex=r'(\d+)'); state(d, "regex select, no col2   as before")
print('== cost: the added code runs only when a column is added, so the normal paths are the same.')
