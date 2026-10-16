# Tables larger than memory

A [SqlDaf][daffodil.sqldaf.SqlDaf] is a Daf whose rows live in a SQLite table, in a file on
disk. Its size is limited only by disk space, so it suits tables too large to hold in memory.
SQL does the work. When the rows you need are few, pull them into an ordinary Daf with
`to_daf()`, and work on them there.

A Daf in memory is faster. On 1,000,000 rows, a sort took 2.4 s on a SqlDaf, against 1.0 s on
a Daf. Whole-table operations ran 2 to 6 times slower in our timings, and a lookup by key took
about 39 µs, against 3.4 µs. So keep a table in a Daf unless its size calls for SQLite.

## Making and opening a table

Every SqlDaf opens the database file named by `SqlDaf.db_path`. Set it once, at the start.
Then make a table as a Daf is made, or load one from a Daf with `from_daf()`.

```pycon
>>> import os, tempfile
>>> from daffodil.daf import Daf
>>> from daffodil.sqldaf import SqlDaf
>>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
>>> votes_daf = Daf(cols=['ballot_id', 'precinct', 'votes'], keyfield='ballot_id',
...                 lol=[['b1', 'p1', "{'c1': 1}"], ['b2', 'p1', "{'c1': 1, 'c2': 1}"],
...                      ['b3', 'p2', "{'c2': 1}"]])
>>> votes_sdaf = SqlDaf.from_daf(votes_daf, name='votes')
>>> len(votes_sdaf)
3
```

A table in the file opens again by its name alone. Its keyfield and dtypes come back with it,
from a record that SqlDaf keeps in the same file.

```pycon
>>> SqlDaf('votes').keyfield
'ballot_id'
```

## Loading a CSV file

`from_csv()` reads a file one row at a time, so memory use does not grow with the file. Every
value is stored as the text of the file, as `Daf.from_csv()` reads it without dtypes.

Text is never less than a number in SQLite. So a condition on a number finds nothing until the
column is converted. `apply_dtypes()` converts it, in the table itself. As with a Daf, name the
columns to convert with `cols=`, or give a dtype for every column.

```pycon
>>> csv_path = os.path.join(os.path.dirname(SqlDaf.db_path), 'marks.csv')
>>> _ = open(csv_path, 'w').write('ballot_id,n\nb1,3\nb2,12\nb3,7\n')
>>> marks_sdaf = SqlDaf.from_csv(csv_path, keyfield='ballot_id')
>>> marks_sdaf.select_where_sql('n < ?', [5]).to_daf().lol
[]
>>> _ = marks_sdaf.apply_dtypes(dtypes={'n': int}, cols='n')
>>> marks_sdaf.select_where_sql('n < ?', [5]).to_daf().lol
[['b1', 3]]
```

The table is named after the file, here marks. The dtypes are kept with it, so they come back
when the table is opened again.

## Selections

Indexing and the select methods give a selection, a child of the table. It refers to rows of
its parent and runs no query until it is read. Read it into memory with `to_daf()`.

```pycon
>>> p1_sdaf = votes_sdaf.select_by_dict({'precinct': 'p1'})
>>> p1_sdaf.is_child
True
>>> p1_sdaf.to_daf().col('ballot_id')
['b1', 'b2']
```

`select_where_sql()` is the fastest way to select, since SQLite does all the work.
`select_where()` takes a Python function of a row, as on a Daf, and calls it for every row.

To read one row, use `select_record()` by key, or `iloc()` by position. Each is one query,
returns a dict and makes no selection.

```pycon
>>> votes_sdaf.select_record('b3')
{'ballot_id': 'b3', 'precinct': 'p2', 'votes': "{'c2': 1}"}
```

## Changing rows and values

Rows are added and values set as on a Daf. Each change is written to the file at once.

```pycon
>>> _ = marks_sdaf.append({'ballot_id': 'b4', 'n': 1})
>>> marks_sdaf['b2', 'n'] = 13
>>> marks_sdaf.select_record('b2')
{'ballot_id': 'b2', 'n': 13}
>>> len(marks_sdaf)
4
```

Each write is one statement. For many small writes, put them in one `transaction()`. Then the
file is locked once for the block, and the writes are committed together, or undone together
if the block raises.

```pycon
>>> with marks_sdaf.transaction():
...     for ballot_id in ('b1', 'b3'):
...         marks_sdaf[ballot_id, 'n'] = 0
>>> marks_sdaf.col('n')
[0, 13, 0, 1]
```

### Deleting rows

Don't delete rows from a table that has selections. A selection made before the delete can
then show a row that was added later, which it never selected. SQLite reuses the row id of a
deleted last row. Leave rows out with a selection instead, as with a Daf. A table that has had
rows deleted also cannot be read by position, so `iloc()` and `sdaf[n]` raise ValueError on it.

## Dict cells and reductions

A dict cell stays as the text that a CSV file holds, as written by Python. Nothing converts
it when it is stored. A reduction by `sparse_row` reads the text of each cell as it goes.

```pycon
>>> votes_sdaf.reduce(Daf.sum_da, by='sparse_row', indirect_col='votes')
{'c1': 2, 'c2': 2}
```

## Groups

`groupby()` gives a selection for each value of a column. It makes an index on that column,
so each group is found without reading the whole table. The index stays with the table.

```pycon
>>> groups_klodaf = votes_sdaf.groupby('precinct')
>>> list(groups_klodaf.keys())
['p1', 'p2']
>>> groups_klodaf['p2'].to_daf().col('ballot_id')
['b3']
```

## Joins

`join()` joins two tables of the same file on their keyfields, in SQL. The columns and the
rows come as `Daf.join()` gives them. A cell with no match is Python NULL.

```pycon
>>> names_sdaf = SqlDaf.from_daf(Daf(cols=['ballot_id', 'precinct'], keyfield='ballot_id',
...                                  lol=[['b1', 'p1'], ['b2', 'p2']]), name='names')
>>> marks_sdaf.join(names_sdaf, how='left').to_daf().lol
[['b1', 0, 'p1'], ['b2', 13, 'p2'], ['b3', 0, ''], ['b4', 1, '']]
```

## Missing values

There are two kinds of NULL. Python NULL is daffodil's missing value, the empty string.
SQL NULL is the database's. SqlDaf writes Python NULL as SQL NULL, and reads it back as
Python NULL. The methods that group, count or select by value treat SQL NULL and the text ''
as one value, as a Daf does. In your own SQL, test for both, as `(col IS NULL OR col = '')`.

## What a SqlDaf does not do

A SqlDaf supports the most common Daf methods, under the same names. Every other Daf method
raises NotImplementedError, which names it. So nothing quietly loads a large table into memory.

```pycon
>>> marks_sdaf.transpose()
Traceback (most recent call last):
...
NotImplementedError: transpose() is not supported on a SqlDaf yet. Use sql(), or to_daf() on a few rows.
```

For anything not covered, `sql()` runs a statement of your own. It gives the result as a Daf.

```pycon
>>> votes_sdaf.sql("SELECT precinct, COUNT(*) AS n FROM votes GROUP BY precinct").lol
[['p1', 2], ['p2', 1]]
```

## Tables that last

A table made by `copy()`, or by a sort, is a TEMP table. It goes when the file is closed.
`save()` makes it a permanent table, under a name, to open again later. `SqlDaf.close()`
closes the file for every SqlDaf on it.

The [SqlDaf API][daffodil.sqldaf.SqlDaf] lists every method, grouped by what it does.
