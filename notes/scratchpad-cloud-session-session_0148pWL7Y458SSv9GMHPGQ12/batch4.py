import sys; sys.path.insert(0, '/tmp/claude-0/-home-user-daffodil/95e36c09-64f8-59ba-9bd6-c962257ad9e3/scratchpad')
from addex import add_examples
add_examples('src/daffodil/daf.py', 'Daf', {
'manifest_apply': """
>>> chunks = {'a': Daf(cols=['n'], lol=[[1], [2]]), 'b': Daf(cols=['n'], lol=[[10]])}
>>> saved = []
>>> def load(spec):
...     return chunks[spec['chunk']]
>>> def double(daf, cols=None):
...     new = Daf(cols=['n'], lol=[[row[0] * 2] for row in daf.lol])
...     return {'rows': len(new)}, new
>>> def save(spec, daf):
...     saved.append(daf.lol)
...     return 'saved'
>>> manifest = Daf(cols=['chunk'], lol=[['a'], ['b']])
>>> result = manifest.manifest_apply(double, load, save, by='table')
>>> result.columns(), result.lol
(['rows'], [[2], [1]])
>>> saved
[[[2], [4]], [[20]]]
""",
'reduce_dodaf_to_daf': """
>>> groups = {'a': Daf(lol=[[1], [2]], cols=['n']), 'b': Daf(lol=[[3]], cols=['n'])}
>>> result = Daf.reduce_dodaf_to_daf('g', Daf.sum_da, groups)
>>> result.columns(), result.lol, result.keyfield
(['n', 'g'], [[3, 'a'], [3, 'b']], 'g')
""",
'multi_groupsum': """
>>> d = Daf(lol=[['a', 'x', 1], ['a', 'y', 2], ['b', 'x', 3]], cols=['g', 'k', 'n'])
>>> sums = d.multi_groupsum(colnames=['g', 'k'], reduce_cols=['n'])
>>> sums['g'].lol
[['a', '', 3], ['b', '', 3]]
>>> sums['k'].lol
[['', 'x', 4], ['', 'y', 2]]
>>> d.multi_groupsum()
Traceback (most recent call last):
    ...
ValueError: multi_groupsum: colnames is required
""",
'valuecounts_for_colnames_ls_selectedby_colname': """
>>> d = Daf(lol=[['a', 'x', 1], ['a', 'y', 2], ['b', 'x', 3]], cols=['g', 'k', 'n'])
>>> d.valuecounts_for_colnames_ls_selectedby_colname(['k'], 'g', 'a')
{'k': {'x': 1, 'y': 1}}
>>> d.valuecounts_for_colnames_ls_selectedby_colname(['g', 'k'], 'g', 'a')
{'g': {'a': 2}, 'k': {'x': 1, 'y': 1}}
""",
'derive_join_translator_daf': """
>>> tr = Daf.derive_join_translator_daf('id', 'id', ['id', 'v'], ['id', 'w'], 'L', 'R')
>>> tr.columns()
['resolved_colname', 'source_name', 'source_colname', 'is_keyfield']
>>> tr.lol
[['id', 'L', 'id', True], ['v', 'L', 'v', False], ['w', 'R', 'w', False]]
>>> tagged = Daf.derive_join_translator_daf('id', 'id', ['id', 'v'], ['id', 'w'], 'L', 'R', tag_other=True)
>>> tagged.col('resolved_colname')
['id', 'v', 'w_R']
>>> omitted = Daf.derive_join_translator_daf('id', 'id', ['id', 'v'], ['id', 'w', 'x'], 'L', 'R', omit_other_cols=['x'])
>>> omitted.col('resolved_colname')
['id', 'v', 'w']
""",
'join_records': """
>>> tr = Daf.derive_join_translator_daf('id', 'id', ['id', 'v'], ['id', 'w'], 'L', 'R')
>>> Daf.join_records([{'id': 1, 'v': 'a'}, {'id': 1, 'w': 'b'}], tr)
{'id': 1, 'v': 'a', 'w': 'b'}
>>> Daf.join_records([{'id': 1, 'v': 'a'}, None], tr)
{'id': 1, 'v': 'a', 'w': ''}
>>> Daf.join_records([{'id': 1, 'v': 'a'}, None], tr, fill=0)
{'id': 1, 'v': 'a', 'w': 0}
""",
'md_daf_table_snippet': """
>>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['x', 'y'])
>>> print(d.md_daf_table_snippet(), end='')
| x | y |
| -: | -: |
| 1 | a |
| 2 | b |
<BLANKLINE>
%% daf rows=2; cols=2; keyfield=''; name=''
""",
'to_md_cols': """
>>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['x', 'y'])
>>> print(d.to_md_cols(), end='')
| x | 1 | 2 |
| y | a | b |
""",
'daf_to_lol_summary': """
>>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['x', 'y'])
>>> d.daf_to_lol_summary()
[['x', 'y'], [1, 'a'], [2, 'b']]
>>> big = Daf(lol=[[i, i] for i in range(20)], cols=['x', 'y'])
>>> big.daf_to_lol_summary(max_rows=4)
[['x', 'y'], [0, 0], [1, 1], ['...', '...'], [18, 18], [19, 19]]
""",
})
p='src/daffodil/daf.py'; s=open(p).read()
def rep(old,new):
    global s
    assert s.count(old)==1,(s.count(old),old[:70]); s=s.replace(old,new)
rep("            by: Must be `table`.\n            cols: Passed to `func` as the keyword `cols`.",
    "            by: Must be `table`. The default is `row`, which applies `func` to each row, so always pass `by='table'`.\n            cols: Passed to `func` as the keyword `cols`.")
rep("        There is no header. The first column holds the column names of the Daf. Use it\n        for a Daf with few rows and many columns.",
    "        There is no header row and no separator row. The first column holds the column names of the Daf.\n        Use it for a Daf with few rows and many columns. Some Markdown renderers, such as Python-Markdown,\n        show this text as plain text, because they need a header and a separator row to make a table.")
open(p,'w').write(s)
