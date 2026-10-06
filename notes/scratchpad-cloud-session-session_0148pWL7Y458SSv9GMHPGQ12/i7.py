from daffodil.daf import Daf
def mk(): return Daf(lol=[[1,'ab12'],[2,'cd34']], cols=['id','s'], keyfield='id')
def state(d, label):
    print(f'-- {label}')
    print(f'   columns {d.columns()}  row lengths {[len(r) for r in d.lol]}  num_cols() {d.num_cols()}  is_rectangular-with-names? {all(len(r)==len(d.hd) for r in d.lol)}')
    print(f'   lol {d.lol}')
    return d
state(mk().assign_icol(-1, ['x','y']) or mk(), 'placeholder')
d = mk(); d.assign_icol(-1, ['x','y']); state(d, 'assign_icol(-1, [x, y])')
d = mk(); d.insert_icol(1, ['x','y']); state(d, 'insert_icol(1, [x, y])   no colname')
d = mk(); d.insert_icol(-1, ['x','y']); state(d, 'insert_icol(-1, [x, y])  no colname')
o = Daf(lol=[[1,'P'],[2,'Q']], cols=['id','w'], keyfield='id')
d = mk(); d.annotate_daf(o, {'newcol':'w'}); state(d, "annotate_daf(o, {'newcol': 'w'})")
d = mk(); d.set_col2_from_col1_using_regex_select('s','n',regex=r'(\d+)'); state(d, "set_col2_from_col1_using_regex_select('s','n')")
d = mk(); d.apply_replace_regex('s','t',replace_regex='/ab//'); state(d, "apply_replace_regex('s','t')")
d = mk(); d.apply_in_place(lambda kl: kl.__setitem__('new', 5), by='row_klist'); state(d, "apply_in_place(row_klist) adds a key")
print('\n== what the mismatch does next, using the assign_icol(-1) case')
d = mk(); d.assign_icol(-1, ['x','y'])
for label, f in [('shape()', lambda: d.shape()),
                 ('to_csv_buff()', lambda: d.to_csv_buff(line_terminator='\n')),
                 ('str(d)', lambda: str(d).splitlines()[:5]),
                 ('d[:, 2]', lambda: d[:, 2].lol),
                 ('d.to_lod()', lambda: d.to_lod()),
                 ('d.col_to_la via name x', lambda: d.col('x', silent_error=True)),
                 ('d.is_rectangular()', lambda: d.is_rectangular()),
                 ('to_pandas_df()', lambda: d.to_pandas_df().shape),
                 ('to_json -> from_json round trip', lambda: Daf.from_json(d.to_json()).lol),
                 ]:
    try: print(f'   {label:34}', f())
    except Exception as e: print(f'   {label:34} EXC {type(e).__name__}: {str(e)[:70]}')
