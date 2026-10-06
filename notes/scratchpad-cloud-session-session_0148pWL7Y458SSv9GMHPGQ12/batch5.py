import sys; sys.path.insert(0, '/tmp/claude-0/-home-user-daffodil/95e36c09-64f8-59ba-9bd6-c962257ad9e3/scratchpad')
from editdoc import edit_doc
P = 'src/daffodil/daf.py'
edit_doc(P, 'Daf', 'isin',
  prose="Do not use the list of bools as a column selector, as in `my_daf[:, mask]`. A list of bools is read as a list of positions, where False is 0 and True is 1, so the wrong columns are chosen, and some more than once. To keep or leave out columns by name, make a list of the names first, as in the example.",
  examples="""
>>> d = Daf(lol=[[1, 2, 3]], cols=['a', 'b', 'c'])
>>> omit = Daf.isin(d.columns(), ['b'])
>>> d[:, [name for name, drop in zip(d.columns(), omit) if not drop]].columns()
['a', 'c']
""")
edit_doc(P, 'Daf', 'set_keyfield',
  prose="A Daf that has column names and no rows can have a keyfield. It applies to the rows that are added later.",
  examples="""
>>> e = Daf(cols=['id', 'v']).set_keyfield('id')
>>> e.keyfield
'id'
>>> e.append({'id': 7, 'v': 'x'}).keys()
[7]
""")
edit_doc(P, 'Daf', 'to_csv_buff',
  prose="A dict whose keys are not text, such as `{1: 'a'}`, is written with its keys as they are.",
  examples="""
>>> Daf(lol=[[{1: 'a'}]], cols=['x']).to_csv_buff(line_terminator='\\n')
"x\\n{1: 'a'}\\n"
""")
edit_doc(P, 'Daf', 'krows_to_irows',
  prose="The first lookup builds the key index, by reading the keyfield column. In one test with 200,000 rows that took 0.05 s, and a later lookup took about 2 microseconds. Adding or removing rows clears the index, and the next lookup builds it again.")
edit_doc(P, 'Daf', 'select_icols',
  prose="With `flip=True` the columns are turned into rows as they are selected. That costs less than selecting them and then calling `transpose()`. In one test with 5 of 50 columns and 20,000 rows it took 0.003 s, against 0.034 s.")
edit_doc(P, 'Daf', 'append',
  prose="None, an empty dict, an empty list and an empty Daf add nothing.",
  examples="""
>>> e = Daf(lol=[[1, 'a']], cols=['id', 'v'])
>>> e.append(None).append({}).append([]).lol
[[1, 'a']]
""")
