import sys; sys.path.insert(0, '/tmp/claude-0/-home-user-daffodil/95e36c09-64f8-59ba-9bd6-c962257ad9e3/scratchpad')
from addex import add_examples
T = ">>> d = Daf(lol=[['r1', 'a', 10], ['r2', 'b', ''], ['r3', 'a', 30]], cols=['id', 'v', 'n'], keyfield='id')\n"
add_examples('src/daffodil/daf.py', 'Daf', {
'krows_to_irows': T + """
>>> d.krows_to_irows('r2')
[1]
>>> d.krows_to_irows(['r3', 'r1'])
[2, 0]
>>> d.krows_to_irows(('r1', 'r2'))
slice(0, 2, 1)
>>> d.krows_to_irows(['r1'], inverse=True)
[1, 2]
>>> d.krows_to_irows(['zz'], silent_error=True)
[]
>>> d.krows_to_irows(['zz'])
Traceback (most recent call last):
    ...
KeyError: 'zz'
""",
'kcols_to_icols': T + """
>>> d.kcols_to_icols('v')
[1]
>>> d.kcols_to_icols(['n', 'id'])
[2, 0]
>>> d.kcols_to_icols(('id', 'v'))
slice(0, 2, 1)
>>> d.kcols_to_icols('v', inverse=True)
[0, 2]
>>> d.kcols_to_icols('zz')
Traceback (most recent call last):
    ...
KeyError: 'zz'
""",
'select_records_daf': T + """
>>> d.select_records_daf(['r3', 'r1']).lol
[['r3', 'a', 30], ['r1', 'a', 10]]
>>> d.select_records_daf([]).lol
[]
>>> kept = d.select_records_daf([], inverse=True)
>>> kept.lol == d.lol, kept.lol is d.lol, kept.lol[0] is d.lol[0]
(True, False, True)
""",
'irow_la': T + """
>>> d.irow_la(1)
['r2', 'b', '']
>>> row = d.irow_la(0)
>>> row[1] = 'Z'
>>> d.lol[0]
['r1', 'Z', 10]
>>> d.irow_la(9)
Traceback (most recent call last):
    ...
IndexError: list index out of range
""",
'col_to_la': T + """
>>> d.col_to_la('v')
['a', 'b', 'a']
>>> d.col_to_la('v', unique=True)
['a', 'b']
>>> d.col_to_la('n', omit_nulls=True)
[10, 30]
>>> d.col_to_la('n', astype=str)
['10', '', '30']
>>> d.col_to_la('zz', silent_error=True)
[]
""",
'icol_to_la': T + """
>>> d.icol_to_la(1)
['a', 'b', 'a']
>>> d.icol_to_la(-1)
[10, '', 30]
>>> d.icol_to_la(1, unique=True)
['a', 'b']
>>> d.icol_to_la(5)
Traceback (most recent call last):
    ...
IndexError: icol: column position 5 is out of range for 3 columns.
""",
})
p='src/daffodil/daf.py'; s=open(p).read()
old="""        The rows are shared with this Daf, as in `select_krows()`. With no keys and
        `inverse` True, the new Daf even uses the row list of this Daf itself, so adding
        a row to one adds it to the other."""
assert s.count(old)==1
s=s.replace(old,"""        The rows are shared with this Daf, as in `select_krows()`. The new Daf has its own
        row list, also with no keys and `inverse` True, so adding a row to one does not
        add it to the other.""")
open(p,'w').write(s)
