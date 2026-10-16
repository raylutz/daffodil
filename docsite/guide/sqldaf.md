# Tables in SQLite

A [SqlDaf][daffodil.sqldaf.SqlDaf] is a Daf whose rows live in a SQLite table. Use it for a
table too large to hold in memory. The rows stay in a database file, and SQL does the work.
When the rows you need are few, pull them into an ordinary Daf with `to_daf()`.

A SqlDaf is slower than a Daf for a table that fits in memory. Each query passes through
sqlite3. On 1,000,000 rows, a sort took 2.4 s, against 1.0 s for a Daf. So keep a table in a
Daf unless its size calls for SQLite.

## Making and opening a table

Every SqlDaf opens the file named by `SqlDaf.db_path`. Set it once. Then make a table as a Daf
is made, or load one from a Daf or a CSV file with `from_daf()` or `from_csv()`.

```pycon
>>> import tempfile
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

A table in the file opens again by its name alone. Its keyfield and dtypes come back with it.

```pycon
>>> SqlDaf('votes').keyfield
'ballot_id'
```

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

To read one row, use `select_record()` by key, or `iloc()` by position. Each is one query,
returns a dict and makes no selection.

```pycon
>>> votes_sdaf.select_record('b3')
{'ballot_id': 'b3', 'precinct': 'p2', 'votes': "{'c2': 1}"}
```

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

## Missing values

There are two kinds of NULL. Python NULL is daffodil's missing value, the empty string.
SQL NULL is the database's. SqlDaf writes Python NULL as SQL NULL, and reads it back as
Python NULL. The methods that group, count or select by value treat SQL NULL and the text ''
as one value, as a Daf does. In your own SQL, test for both, as `(col IS NULL OR col = '')`.

## What a SqlDaf does not do

A SqlDaf has only the methods written for SQL. Every other Daf method raises
NotImplementedError, which names it. So nothing quietly loads a large table into memory.
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
