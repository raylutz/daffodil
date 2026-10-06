import sys; sys.path.insert(0, '/tmp/claude-0/-home-user-daffodil/95e36c09-64f8-59ba-9bd6-c962257ad9e3/scratchpad')
from editdoc import edit_doc
P = 'src/daffodil/daf.py'
edit_doc(P, 'Daf', 'select_where',
  prose="To test a value against a list, a set or another table, write the test in the function. There is no need to build a list of bools first, as the deprecated `isin()` did. Build a set of the values before the call, so that each lookup is fast and the set is built once. The function can use `and`, `or`, `not` and any other Python. It is called once for each row, so for a test on one column of a large Daf it is not the fastest way. A comprehension over `col()`, followed by `select_irows()`, is faster. In one test with 200,000 rows and 1,000 values, `select_where()` took 0.11 s and the comprehension took 0.02 s.",
  examples="""
>>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30], [4, 'a', 40]], cols=['id', 'v', 'n'])
>>> keep = {'a', 'c'}
>>> d.select_where(lambda row: row['v'] in keep).lol
[[1, 'a', 10], [3, 'c', 30], [4, 'a', 40]]
>>> d.select_where(lambda row: row['v'] not in keep).lol
[[2, 'b', 20]]
>>> d.select_where(lambda row: row['v'] in keep and row['n'] > 15).lol
[[3, 'c', 30], [4, 'a', 40]]
>>> d.select_where(lambda row: row['id'] % 2 == 0 or row['v'] == 'c').lol
[[2, 'b', 20], [3, 'c', 30], [4, 'a', 40]]

The values can come from another Daf. Make the set once, outside the function:

>>> other = Daf(lol=[['a'], ['c']], cols=['v'], keyfield='v')
>>> other_keys = set(other.keys())
>>> d.select_where(lambda row: row['v'] in other_keys).lol
[[1, 'a', 10], [3, 'c', 30], [4, 'a', 40]]

For a large Daf, the faster form picks the positions from the column:

>>> d.select_irows([irow for irow, v in enumerate(d.col('v')) if v in keep]).lol
[[1, 'a', 10], [3, 'c', 30], [4, 'a', 40]]
""")
p='src/daffodil/daf.py'; s=open(p).read()
old="        the `isin()` of pandas. Daffodil does not use it, and it will be removed."
assert s.count(old)==1
s=s.replace(old,"        the `isin()` of pandas. Daffodil does not use it, and it will be removed. `select_where()`\n        shows how to test values against a set in one pass.")
open(p,'w').write(s)
p='README.md'; r=open(p).read()
row="|`df[df[colname] > 5]`                              |`daf.select_where(lambda row: row[colname] > 5)`           |"
i=r.index(row); j=r.index('\n', i)
new="\n|`df[df[colname].isin(values)]`                    |`daf.select_where(lambda row: row[colname] in values_set)` |select rows whose value is in a set, in one pass. No list of bools is built. Make the set once, before the call. |"
r=r[:j]+new+r[j:]
open(p,'w').write(r)
