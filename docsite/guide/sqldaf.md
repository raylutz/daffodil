# Tables larger than memory

A [SqlDaf][daffodil.sqldaf.SqlDaf] stores a table in a SQLite database file, so its rows stay
out of Python memory. It supports tables larger than the memory available, and many
operations run directly in SQLite. It has many of the Daf methods, under the same names. Some
of them behave differently, because the data is in a database. [Differences from
Daf](#differences-from-daf) lists them.

Prefer an ordinary Daf when the table, and any copies you make of it, fit comfortably in
memory. A Daf is faster. Use SqlDaf when memory becomes the limit.
[Performance](#performance) gives timings.

## SQLite version

SqlDaf needs SQLite 3.48 or later. It uses Python's own `sqlite3` module, and the SQLite in it
depends on how that Python was built. For example, the system Python 3.11 of Ubuntu 22.04 has
SQLite 3.37.2, and the Python 3.12 of Ubuntu 24.04 has 3.45. Check yours with
`python -c "import sqlite3; print(sqlite3.sqlite_version)"`.

An older SQLite raises RuntimeError when SqlDaf first opens a file. Then use a Python with a newer
one. Every Python that uv installs has SQLite 3.53, from 3.10 to 3.13, as in `uv python install 3.12`.

SQLite must not be loaded twice in one process with the same file open in both, or file locks
can be lost. So code that opens SqlDaf's files should use Python's own `sqlite3` too, not a
second build such as pysqlite3. `daffodil.lib.daf_sql` gives the module SqlDaf uses:

```python
from daffodil.lib.daf_sql import sqlite3
```

## Making and opening a table

Every SqlDaf opens the database file named by `SqlDaf.db_path`. Set it once, at the start.
Here it is a new file in a temporary folder, so the examples can run anywhere.

```pycon
>>> import os, tempfile
>>> from daffodil.daf import Daf
>>> from daffodil.sqldaf import SqlDaf
>>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
```

Make a table as a Daf is made: give its name, its columns and its rows. The name must not be
in use in the file yet, or the constructor raises ValueError.

```pycon
>>> items_sdaf = SqlDaf('items', cols=['id', 'grp', 'qty'], keyfield='id',
...                     lol=[['a1', 'x', 3], ['a2', 'x', 12], ['a3', 'y', 7]])
>>> items_sdaf
<BLANKLINE>
| id | grp | qty |
| -: | --: | --: |
| a1 |   x |   3 |
| a2 |   x |  12 |
| a3 |   y |   7 |
<BLANKLINE>
%% daf rows=3; cols=3; keyfield='id'; name='items'
<BLANKLINE>
>>> len(items_sdaf)
3
```

A table in the file opens again by its name alone. Its keyfield and dtypes come back with it,
from a record that SqlDaf keeps in the same file. `from_daf()` makes a table from a Daf.

```pycon
>>> SqlDaf('items').keyfield
'id'
```

## Three kinds of result

Methods give one of three kinds of result. Knowing which one you have tells you what a change
to it does.

| Result | Where its rows are | Replacing an existing cell |
| :----- | :----------------- | :------------------------- |
| A selection | in its parent's table | changes the parent's row |
| An independent SqlDaf | in a table of its own | changes only it |
| A Daf | in Python memory | changes only it |

Other changes to a selection, such as adding rows, are in the table under
[Changing rows and values](#changing-rows-and-values).

- Indexing by rows alone, as in `sdaf['a1']` or `sdaf[0:2]`, and the select methods give a
  selection.
- Indexing with a column part, as in `sdaf[:, ['id', 'qty']]` or `sdaf['a1', 'qty']`, gives an
  independent SqlDaf of new rows, as does `select_cols()`.
- `copy()`, a sort and a join give an independent SqlDaf.
- `to_daf()`, `sql()`, `groupsum_daf()` and `value_counts_daf()` give a Daf.
  `sum()` and the `valuecounts_for_*` methods give a dict.

Whether a table lasts is a separate matter. A table made by the constructor, a loader or
`save()` is permanent: it stays in the file. A table made by `copy()`, a sort or a join is a
TEMP table: it is removed when the file is closed. See [Keeping tables](#keeping-tables).

## Loading a CSV file

`from_csv()` reads a file one row at a time, so the file is never all in memory. The table is
named after the file. CSV values are loaded as text, as `Daf.from_csv()` reads them without
dtypes.

Convert the number columns before filtering, sorting or adding them up. Until then, '12' is
text, not the number 12, and a condition such as `n < 5` finds nothing. `apply_dtypes()`
converts a column in the table itself. As with a Daf, name the columns to convert with
`cols=`, or give a dtype for every column.

```pycon
>>> csv_path = os.path.join(os.path.dirname(SqlDaf.db_path), 'marks.csv')
>>> open(csv_path, 'w').write('ballot_id,n\nb1,3\nb2,12\nb3,7\n')
28
>>> marks_sdaf = SqlDaf.from_csv(csv_path, keyfield='ballot_id')
>>> marks_sdaf
<BLANKLINE>
| ballot_id | n  |
| --------: | -: |
|        b1 |  3 |
|        b2 | 12 |
|        b3 |  7 |
<BLANKLINE>
%% daf rows=3; cols=2; keyfield='ballot_id'; name='marks'
<BLANKLINE>
>>> marks_sdaf.select_where_sql('n < ?', [5])
<BLANKLINE>
<BLANKLINE>
%% daf rows=0; cols=2; keyfield='ballot_id'; name='daf_tmp_marks_v1'
<BLANKLINE>
>>> marks_sdaf.apply_dtypes(dtypes={'n': int}, cols='n')
<BLANKLINE>
| ballot_id | n  |
| --------: | -: |
|        b1 |  3 |
|        b2 | 12 |
|        b3 |  7 |
<BLANKLINE>
%% daf rows=3; cols=2; keyfield='ballot_id'; name='marks'
<BLANKLINE>
>>> marks_sdaf.select_where_sql('n < ?', [5])
<BLANKLINE>
| ballot_id | n  |
| --------: | -: |
|        b1 |  3 |
<BLANKLINE>
%% daf rows=1; cols=2; keyfield='ballot_id'; name='daf_tmp_marks_v2'
<BLANKLINE>
```

The dtypes are kept with the table, so they come back when it is opened again. A value that
does not convert, such as 'abc' in an int column, stays as it is. So a converted column can
still hold some text.

### Many files into one table

`from_csv()` also takes a glob pattern, such as `chunks/votes_*.csv`, or a list of files, and
loads them all into one table. By default every file must have the first file's header, or
ValueError names both files. With `strict_header=False`, each value goes into the column of
its name in the file's header. So files whose columns come in another order, or that lack a
column, load safely. A missing column is Python NULL. A name that only a later file has
becomes a new column, after the first file's.

```pycon
>>> chunk_dir = os.path.dirname(SqlDaf.db_path)
>>> open(chunk_dir + '/v_0.csv', 'w').write('ballot_id,n\nb1,3\n')
17
>>> open(chunk_dir + '/v_1.csv', 'w').write('n,ballot_id,note\n12,b2,late\n')
28
>>> SqlDaf.from_csv(chunk_dir + '/v_*.csv', name='chunked', strict_header=False).to_daf()
<BLANKLINE>
| ballot_id | n  | note |
| --------: | -: | ---: |
|        b1 |  3 |      |
|        b2 | 12 | late |
<BLANKLINE>
%% daf rows=2; cols=3; keyfield=''; name='chunked'
<BLANKLINE>
```

Placing by name never puts a value in the wrong column. But a misspelled name, or a file of
another kind, becomes a new column, with no error. The strict default stops both.

## Selections

Indexing and the select methods give a selection. It refers to rows of its parent table.

A selection fixes its rows the first time it is read whole: by `to_daf()`, `len()`, `bool()`,
`to_list()` or the like. A change to the parent before then can change which rows it gets.
After that, its rows are fixed. A change to the values in those rows still shows in it.

Reading one row does not fix the rows. `select_record()` and `iloc()` query the selection as it
stands, each time. A negative position in `iloc()` is the exception, since it counts with
`len()` first.

```pycon
>>> big_sdaf = items_sdaf.select_where_sql('qty > ?', [5])
>>> items_sdaf['a1', 'qty'] = 9
>>> big_sdaf.to_daf().col('id')
['a1', 'a2', 'a3']
>>> items_sdaf['a1', 'qty'] = 1
>>> big_sdaf.to_daf().col('qty')
[1, 12, 7]
```

Here a1 joined the selection, since its value changed before the first read. When it changed
again after the read, a1 stayed in, with its new value.

A selection made from a selection not read yet carries that one's query inside its own. A loop
that selects from its own last result, with no read in between, nests one query per step.
SQLite stops at about 400 levels with OperationalError "Recursion limit". It raises at once.
Read the selection every so often, as with `len()`, and the next one starts a fresh query.

`select_where_sql()` is the fastest way to select, since SQLite does all the work.
`select_where()` takes a Python function of a row, as on a Daf, and calls it for every row.

### One row

To read one row, use `select_record()` by key, or `iloc()` by position. Each is one query,
returns a dict and makes no selection. On a selection, they see only the selection's rows.

```pycon
>>> items_sdaf.select_record('a3')
{'id': 'a3', 'grp': 'y', 'qty': 7}
>>> items_sdaf.select_where_sql('grp = ?', ['y']).select_record('a1')
{}
```

An int in brackets is always a position, even when the keyfield holds ints. Look up an int
key with `select_record()`.

## Changing rows and values

Rows are added and values set as on a Daf. Each change is written to the file at once.

```pycon
>>> items_sdaf.append({'id': 'a4', 'grp': 'y', 'qty': 2})
<BLANKLINE>
| id | grp | qty |
| -: | --: | --: |
| a1 |   x |   1 |
| a2 |   x |  12 |
| a3 |   y |   7 |
| a4 |   y |   2 |
<BLANKLINE>
%% daf rows=4; cols=3; keyfield='id'; name='items'
<BLANKLINE>
>>> items_sdaf['a2', 'qty'] = 13
>>> items_sdaf.select_record('a2')
{'id': 'a2', 'grp': 'x', 'qty': 13}
```

A change made through a selection depends on the operation:

| Operation on a selection | Effect |
| :----------------------- | :----- |
| `sel[key, col] = v`, `update_record_krow()`, `update_by_keylist()`, `assign_col()` | changes the parent's rows |
| `copy()` | gives an independent SqlDaf |
| `copy(inplace=True)` | the selection becomes independent |
| `append()`, `concat()` | copies the selection, then adds rows to the copy |
| `apply_dtypes()` | copies the selection, then converts the copy |
| a sort | copies the selection, then sorts the copy |
| `delete_records(keys, inplace=True)` | drops those rows from the selection only |

So copy a selection first when you want to change it without touching its parent.

Assigning with brackets needs a str key, as in `sdaf['a2', 'qty'] = 13`. An int in brackets
raises NotImplementedError, since it would be a position in a Daf. To change some cells of the
row with a key, of any type, use `update_record_krow()`.

```pycon
>>> items_sdaf.update_record_krow('a3', {'grp': 'z'})
<BLANKLINE>
| id | grp | qty |
| -: | --: | --: |
| a1 |   x |   1 |
| a2 |   x |  13 |
| a3 |   z |   7 |
| a4 |   y |   2 |
<BLANKLINE>
%% daf rows=4; cols=3; keyfield='id'; name='items'
<BLANKLINE>
>>> items_sdaf.select_record('a3')
{'id': 'a3', 'grp': 'z', 'qty': 7}
```

### Many small writes

Each write is one statement. For many small writes, put them in one `transaction()`. Then the
file is locked once for the block, and the writes are committed together, or undone together
if the block raises.

```pycon
>>> with items_sdaf.transaction():
...     for item_id in ('a1', 'a3'):
...         items_sdaf[item_id, 'qty'] = 0
>>> items_sdaf.col('qty')
[0, 13, 0, 2]
```

### Removing rows

To leave rows out without changing the table, use `delete_records(keys)`. By default it
returns a selection of the other rows, and the table is unchanged.

```pycon
>>> rest_sdaf = items_sdaf.delete_records(['a4'])
>>> rest_sdaf.to_daf().col('id'), len(items_sdaf)
(['a1', 'a2', 'a3'], 4)
```

`delete_records(keys, inplace=True)` on a parent deletes the rows from the table. Avoid it
while selections refer to that table. A selection made before the delete can then show a row
added later, which it never selected. A table that has had rows deleted also cannot be read by
position, so `iloc()` and `sdaf[n]` raise ValueError on it.

## Dict cells and reductions

A dict cell stays as the text that a CSV file holds, as written by Python. Nothing converts
it when it is stored. A reduction by `sparse_row` reads the text of each cell as it goes.

```pycon
>>> votes_sdaf = SqlDaf('votes', cols=['ballot_id', 'precinct', 'votes'], keyfield='ballot_id',
...                     lol=[['b1', 'p1', "{'c1': 1}"], ['b2', 'p1', "{'c1': 1, 'c2': 1}"],
...                          ['b3', 'p2', "{'c2': 1}"]])
>>> votes_sdaf
<BLANKLINE>
| ballot_id | precinct |       votes        |
| --------: | -------: | -----------------: |
|        b1 |       p1 |          {'c1': 1} |
|        b2 |       p1 | {'c1': 1, 'c2': 1} |
|        b3 |       p2 |          {'c2': 1} |
<BLANKLINE>
%% daf rows=3; cols=3; keyfield='ballot_id'; name='votes'
<BLANKLINE>
>>> votes_sdaf.reduce(Daf.sum_da, by='sparse_row', indirect_col='votes')
{'c1': 2, 'c2': 2}
```

A reduction reads the rows in batches, so the table is never all in memory. Memory use still
depends on the result your function builds.

## Groups

`groupby()` gives a selection for each value of a column. It makes an index on that column,
so each group is found without a full scan of the table. The index stays with the table.

```pycon
>>> groups_klodaf = votes_sdaf.groupby('precinct')
>>> list(groups_klodaf.keys())
['p1', 'p2']
>>> groups_klodaf['p2'].to_daf().col('ballot_id')
['b3']
```

`groupsum_daf()` and the count methods do the work in SQLite and give a Daf with one row per
group. That Daf is small only when the groups are few.

## Joins

`join()` joins two tables on their keyfields, in SQL. The other table may be in another file. It
is brought in for the join, as [Tables in other files](#tables-in-other-files) shows. The columns
and the rows come as `Daf.join()` gives them. A cell with no match is Python NULL.

```pycon
>>> names_sdaf = SqlDaf('names', cols=['ballot_id', 'voter'], keyfield='ballot_id',
...                     lol=[['b1', 'Ann'], ['b2', 'Bo']])
>>> names_sdaf
<BLANKLINE>
| ballot_id | voter |
| --------: | ----: |
|        b1 |   Ann |
|        b2 |    Bo |
<BLANKLINE>
%% daf rows=2; cols=2; keyfield='ballot_id'; name='names'
<BLANKLINE>
>>> marks_sdaf.join(names_sdaf, how='left')
<BLANKLINE>
| ballot_id | n  | voter |
| --------: | -: | ----: |
|        b1 |  3 |   Ann |
|        b2 | 12 |    Bo |
|        b3 |  7 |       |
<BLANKLINE>
%% daf rows=3; cols=3; keyfield='ballot_id'; name='daf_tmp_marks_join_names_v9'
<BLANKLINE>
```

## Keeping tables

A sort changes which table the SqlDaf refers to. It makes a sorted TEMP table, and the table in
the file keeps its order. `save()` keeps the sorted copy under a name not in use. `close()`
removes TEMP tables, so save what you want to keep first.

```pycon
>>> items_sdaf.sort_by_colname('qty', reverse=True)
<BLANKLINE>
| id | grp | qty |
| -: | --: | --: |
| a2 |   x |  13 |
| a4 |   y |   2 |
| a1 |   x |   0 |
| a3 |   z |   0 |
<BLANKLINE>
%% daf rows=4; cols=3; keyfield='id'; name='daf_tmp_items_v10'
<BLANKLINE>
>>> items_sdaf.col('id')
['a2', 'a4', 'a1', 'a3']
>>> items_sdaf.save('items_by_qty')
<BLANKLINE>
| id | grp | qty |
| -: | --: | --: |
| a2 |   x |  13 |
| a4 |   y |   2 |
| a1 |   x |   0 |
| a3 |   z |   0 |
<BLANKLINE>
%% daf rows=4; cols=3; keyfield='id'; name='items_by_qty'
<BLANKLINE>
>>> SqlDaf.close()
>>> SqlDaf('items_by_qty').col('id')
['a2', 'a4', 'a1', 'a3']
>>> SqlDaf('items').col('id')
['a1', 'a2', 'a3', 'a4']
```

`save()` raises ValueError for a name in use. To replace a table with its sorted copy, delete
the old table first, as in `SqlDaf('items').delete()`, then save the copy under that name.

`SqlDaf.close()` closes the file for every SqlDaf on it. Permanent tables keep every committed
change. A SqlDaf made before the close can no longer be used. Open the table again by name.

## Read-only tables

A Daf changes only memory. Its file changes only when the Daf is written out. A SqlDaf writes
to its file at once, and nothing in the file shows later that it changed. So a table that other
work depends on can be opened read-only:

```pycon
>>> SqlDaf('items_ro', cols=['id', 'qty'], lol=[['a1', 3], ['a2', 12]], keyfield='id')
<BLANKLINE>
| id | qty |
| -: | --: |
| a1 |   3 |
| a2 |  12 |
<BLANKLINE>
%% daf rows=2; cols=2; keyfield='id'; name='items_ro'
<BLANKLINE>
>>> ro_sdaf = SqlDaf('items_ro', read_only=True)
>>> ro_sdaf
<BLANKLINE>
| id | qty |
| -: | --: |
| a1 |   3 |
| a2 |  12 |
<BLANKLINE>
%% daf rows=2; cols=2; keyfield='id'; name='items_ro'
<BLANKLINE>
>>> ro_sdaf.update_record_krow('a1', {'qty': 4})
Traceback (most recent call last):
...
daffodil.sqldaf.ReadOnlyTableError: A table opened with read_only=True cannot have its data changed. copy() it, and change the copy.
>>> work_sdaf = ro_sdaf.copy()
>>> work_sdaf.update_record_krow('a1', {'qty': 4}).col('qty')
[4, 12]
```

Writing a cell, adding or removing rows or columns, renaming a column, converting with
`apply_dtypes()` and dropping the table raise ReadOnlyTableError. Nothing is changed.
Selections, sorts, joins, `groupby()`, an index and the keyfield work, since they leave the data
as it is. An index does change the file, but not its rows. SQLite itself refuses the change, so
no method can slip past it, `sql()` included.

Read-only holds for the SqlDaf that asked for it, and for its selections. Another SqlDaf opened
on the same table without `read_only` can change it. The read-only one then reads the change:

```pycon
>>> rw_sdaf = SqlDaf('items_ro')
>>> rw_sdaf.read_only, ro_sdaf.read_only
(False, True)
>>> rw_sdaf.update_record_krow('a2', {'qty': 13}).col('qty')
[3, 13]
>>> ro_sdaf.col('qty')
[3, 13]
>>> ro_sdaf.sql("UPDATE items_ro SET qty = 0")
Traceback (most recent call last):
...
daffodil.sqldaf.ReadOnlyTableError: A table opened with read_only=True cannot have its data changed. copy() it, and change the copy.
```

So a stage that only reads opens a shared table read-only, and a stage that must change the
table in its own file opens it without `read_only`. Neither has to close the file first.

## Tables in other files

A job may keep each large table in a file of its own. That makes it plain which tables are
built, and a table is dropped by deleting its file.

**A join with a table in another file** is written as for a Daf. The other table is copied into a
TEMP table on this table's connection, with its key index, and joined there. A TEMP table is in no
database file, so neither file changes. The copy lives until the join is first read, and is then
deleted. A join never read loses its copy when it is gone, or when the file is closed. A chain of
joins works the same way. The result is kept in a file of its own with `save(db_path=)`:

```pycon
>>> job_dir = os.path.dirname(SqlDaf.db_path)
>>> ballots_sdaf = SqlDaf('ballots', cols=['ballot_id', 'qty'], keyfield='ballot_id',
...                       lol=[['b1', 3], ['b2', 12], ['b3', 7]])
>>> voters_sdaf = SqlDaf('voters', db_path=job_dir + '/voters.db', keyfield='ballot_id',
...                      cols=['ballot_id', 'voter'], lol=[['b1', 'Ann'], ['b3', 'Cy']])
>>> ballots_voters_sdaf = ballots_sdaf.join(voters_sdaf, how='left')
>>> ballots_voters_sdaf.save('ballots_voters', db_path=job_dir + '/ballots_voters.db')
<BLANKLINE>
| ballot_id | qty | voter |
| --------: | --: | ----: |
|        b1 |   3 |   Ann |
|        b2 |  12 |       |
|        b3 |   7 |    Cy |
<BLANKLINE>
%% daf rows=3; cols=3; keyfield='ballot_id'; name='ballots_voters'
<BLANKLINE>

No copy is left on the connection:

>>> [row_t[0] for row_t in ballots_sdaf.conn.execute("SELECT name FROM sqlite_temp_master WHERE name LIKE '%voters%'")]
[]
```

`concat()` and keys from a SqlDaf in another file, as in `select_records_daf()`, bring it in the
same way. For keys, only the key column is copied. A permanent table is copied in SQL, which took
0.7 s for 1,000,000 rows of 5 columns, with its key index. The join then took as long as in one
file. A selection or a TEMP table of another connection is copied through Python, which is slower.
Inside a transaction, where SQLite cannot attach a file, the copy also goes through Python.

**`from_db()`** copies a table into the working file, `SqlDaf.db_path`, as a permanent table, as
`from_csv()` loads a CSV file. Use it to build a file named for the table a stage makes, with each
input copied into it. Its rows keep their order, and its keyfield and dtypes come with it.

The other file is only read. It is attached for the copy alone, which is one SQL statement, and
then detached. So any number of files can be copied in, and a change to the copy never reaches
the file it came from.

To keep a result in a file of its own, give `save()` that file. The table is written there,
and the SqlDaf then refers to it there.

```pycon
>>> job_dir = os.path.dirname(SqlDaf.db_path)
>>> SqlDaf('items_by_qty').save('items', db_path=job_dir + '/items.db')
<BLANKLINE>
| id | grp | qty |
| -: | --: | --: |
| a2 |   x |  13 |
| a4 |   y |   2 |
| a1 |   x |   0 |
| a3 |   z |   0 |
<BLANKLINE>
%% daf rows=4; cols=3; keyfield='id'; name='items'
<BLANKLINE>
>>> SqlDaf('items', db_path=job_dir + '/items.db').col('id')
['a2', 'a4', 'a1', 'a3']
>>> back_sdaf = SqlDaf.from_db(job_dir + '/items.db', 'items', new_name='items_back')
>>> back_sdaf.db_path == SqlDaf.db_path, back_sdaf.col('id')
(True, ['a2', 'a4', 'a1', 'a3'])
```

Neither works inside a transaction. SQLite cannot detach a file used in an open transaction,
so both raise ValueError there, and nothing is attached.

## How cells are stored

A SqlDaf stores each cell as SQLite holds it: an int as an integer, a float as a real, and a str
as text. A bool is stored as 0 or 1, as daffodil writes bools. A missing value, NULL, is stored as
SQL NULL, and reads and prints as NULL again. A dict or a list is stored as its text, the
`str()` that `to_csv_file()` writes for it.

```pycon
>>> kinds_daf = Daf(cols=['id', 'n', 'x', 'ok', 'p', 'cv', 'ls'], keyfield='id', lol=[
...     ['b1', 3, 2.5, 1, 'P1', {'c0': 1, 'c1': ''}, [1, 'two']],
...     ['b2', '', '', 0, '', {}, []]])
>>> kinds_daf
<BLANKLINE>
| id | n  |  x  | ok | p  |         cv          |     ls     |
| -: | -: | --: | -: | -: | ------------------: | ---------: |
| b1 |  3 | 2.5 |  1 | P1 | {'c0': 1, 'c1': ''} | [1, 'two'] |
| b2 |    |     |  0 |    |                  {} |         [] |
<BLANKLINE>
%% daf rows=2; cols=7; keyfield='id'; name=''
>>> kinds_sdaf = SqlDaf.from_daf(kinds_daf, name='kinds')
>>> kinds_sdaf
<BLANKLINE>
| id | n  |  x  | ok | p  |         cv          |     ls     |
| -: | -: | --: | -: | -: | ------------------: | ---------: |
| b1 |  3 | 2.5 |  1 | P1 | {'c0': 1, 'c1': ''} | [1, 'two'] |
| b2 |    |     |  0 |    |                  {} |         [] |
<BLANKLINE>
%% daf rows=2; cols=7; keyfield='id'; name='kinds'
>>> kinds_sdaf.sql('SELECT id, typeof(n), typeof(x), typeof(ok), typeof(p), typeof(cv), typeof(ls) FROM kinds')
<BLANKLINE>
| id | typeof(n) | typeof(x) | typeof(ok) | typeof(p) | typeof(cv) | typeof(ls) |
| -: | --------: | --------: | ---------: | --------: | ---------: | ---------: |
| b1 |   integer |      real |    integer |      text |       text |       text |
| b2 |      null |      null |    integer |      null |       text |       text |
<BLANKLINE>
%% daf rows=2; cols=7; keyfield=''; name=''
```

The table prints the same as the Daf. `typeof()` shows what SQLite holds. A NULL is SQL NULL, and
the dict and list are text. Reading a row gives that text back:

```pycon
>>> kinds_sdaf.select_record('b1')
{'id': 'b1', 'n': 3, 'x': 2.5, 'ok': 1, 'p': 'P1', 'cv': "{'c0': 1, 'c1': ''}", 'ls': "[1, 'two']"}
```

With NULL stored as SQL NULL, SQL's own sums and counts skip it, as the Daf methods do:

```pycon
>>> kinds_sdaf.sql('SELECT SUM(n) AS sum_n, COUNT(n) AS count_n, AVG(x) AS avg_x FROM kinds')
<BLANKLINE>
| sum_n | count_n | avg_x |
| ----: | ------: | ----: |
|     3 |       1 |   2.5 |
<BLANKLINE>
%% daf rows=1; cols=3; keyfield=''; name=''
```

The text of a dict or list is daffodil's own form, with bools as 0 and 1 and missing values as
''. SQLite's JSON functions read it, so a condition or a query can look inside it. A Python
function in `select_where()` or `apply()` gets it as text, and decodes it if it needs to:

```pycon
>>> kinds_sdaf.select_where_sql("json_extract(cv, '$.c0') = ?", [1]).keys()
['b1']
>>> kinds_sdaf.sql("SELECT id, json_extract(cv, '$.c0') AS c0, json_extract(ls, '$[1]') AS ls1 FROM kinds")
<BLANKLINE>
| id | c0 | ls1 |
| -: | -: | --: |
| b1 |  1 | two |
| b2 |    |     |
<BLANKLINE>
%% daf rows=2; cols=3; keyfield=''; name=''
```

A table loaded from a CSV file starts as text, as a Daf read with `from_csv()` does. An empty
field is stored as SQL NULL. `apply_dtypes()` then converts the columns once, in SQL. A dict or
list dtype is recorded and converts nothing, since the text is the value. After it, the table
holds exactly what the direct write stored:

```pycon
>>> csv_path = os.path.join(os.path.dirname(SqlDaf.db_path), 'kinds.csv')
>>> kinds_daf.to_csv_file(csv_path)
'.../kinds.csv'
>>> print(open(csv_path).read())
id,n,x,ok,p,cv,ls
b1,3,2.5,1,P1,"{'c0': 1, 'c1': ''}","[1, 'two']"
b2,,,0,,{},[]
>>> loaded_sdaf = SqlDaf.from_csv(csv_path, name='kinds_csv', keyfield='id')
>>> loaded_sdaf.sql('SELECT id, typeof(n), typeof(x), typeof(ok), typeof(p), typeof(cv), typeof(ls) FROM kinds_csv')
<BLANKLINE>
| id | typeof(n) | typeof(x) | typeof(ok) | typeof(p) | typeof(cv) | typeof(ls) |
| -: | --------: | --------: | ---------: | --------: | ---------: | ---------: |
| b1 |      text |      text |       text |      text |       text |       text |
| b2 |      null |      null |       text |      null |       text |       text |
<BLANKLINE>
%% daf rows=2; cols=7; keyfield=''; name=''
>>> loaded_sdaf.apply_dtypes(dtypes={'n': int, 'x': float, 'ok': bool, 'cv': dict, 'ls': list}, cols=['n', 'x', 'ok', 'cv', 'ls'])
<BLANKLINE>
| id | n  |  x  | ok | p  |         cv          |     ls     |
| -: | -: | --: | -: | -: | ------------------: | ---------: |
| b1 |  3 | 2.5 |  1 | P1 | {'c0': 1, 'c1': ''} | [1, 'two'] |
| b2 |    |     |  0 |    |                  {} |         [] |
<BLANKLINE>
%% daf rows=2; cols=7; keyfield='id'; name='kinds_csv'
<BLANKLINE>
>>> loaded_sdaf.sql('SELECT id, typeof(n), typeof(x), typeof(ok), typeof(p), typeof(cv), typeof(ls) FROM kinds_csv')
<BLANKLINE>
| id | typeof(n) | typeof(x) | typeof(ok) | typeof(p) | typeof(cv) | typeof(ls) |
| -: | --------: | --------: | ---------: | --------: | ---------: | ---------: |
| b1 |   integer |      real |    integer |      text |       text |       text |
| b2 |      null |      null |    integer |      null |       text |       text |
<BLANKLINE>
%% daf rows=2; cols=7; keyfield=''; name=''
>>> loaded_sdaf.to_daf().lol == kinds_sdaf.to_daf().lol
True
```

In memory, a Daf decodes the dict and list text. `to_daf()` carries the dtypes, and the Daf's
`apply_dtypes()` turns the text back into a dict and a list, so the round trip gives back the
Daf it started from:

```pycon
>>> back_daf = loaded_sdaf.to_daf().apply_dtypes(cols=['cv', 'ls'])
>>> back_daf.select_record('b1')
{'id': 'b1', 'n': 3, 'x': 2.5, 'ok': 1, 'p': 'P1', 'cv': {'c0': 1, 'c1': ''}, 'ls': [1, 'two']}
>>> back_daf.lol == kinds_daf.lol
True
```

Writing a dict or list cell from Python, as with `append()` or `update_record_krow()`, stores its
text too. Only a dict or a list itself is turned into text. A tuple, or a subclass of dict such as
OrderedDict, is refused by SQLite, with sqlite3.ProgrammingError or InterfaceError:

```pycon
>>> kinds_sdaf.append({'id': 'b3', 'n': 4, 'x': 0.5, 'ok': 1, 'p': 'P2', 'cv': {'c2': 2}, 'ls': ['q']}).select_record('b3')
{'id': 'b3', 'n': 4, 'x': 0.5, 'ok': 1, 'p': 'P2', 'cv': "{'c2': 2}", 'ls': "['q']"}
```

## Missing values

There are two kinds of NULL. Python NULL is daffodil's missing value: the empty string '',
which daffodil names NULL, as in `from daffodil.daf import NULL`. SQL NULL is the database's.
SqlDaf writes Python NULL as SQL NULL, and reads SQL NULL back as Python NULL. A Python None
is stored as SQL NULL too, so it also reads back as Python NULL.

The methods that group, count or select by value treat SQL NULL and the text '' as one value,
as a Daf does. In Python, test a value with `val is NULL`. In your own SQL, test for both, as
`(col IS NULL OR col = '')`.

## Methods a SqlDaf does not have

A SqlDaf has the most common Daf methods, under the same names. Every other Daf method raises
NotImplementedError, which names it. So nothing quietly loads a large table into memory.

```pycon
>>> marks_sdaf = SqlDaf('marks')
>>> marks_sdaf.transpose()
Traceback (most recent call last):
...
NotImplementedError: transpose() is not supported on a SqlDaf yet. Use sql(), or to_daf() on a few rows.
```

For anything not covered, `sql()` runs a statement of your own. It gives the whole result as a
Daf, so select only what you need.

```pycon
>>> marks_sdaf.sql("SELECT COUNT(*) AS n FROM marks WHERE n > 5").lol
[[2]]
```

A change to a table's columns made with `sql()` is not seen by a SqlDaf already open on it.
Use `insert_col()`, `drop_cols()` and `rename_cols()` for that.

## Code that runs on a Daf or a SqlDaf

A SqlDaf is a Daf, so code written for a Daf calls the SqlDaf methods when given one. Some ways
of writing it are fast on both, and some fill memory or fail on a SqlDaf. These patterns give the
same rows on both. On a SqlDaf they run in SQL and read only the rows they return.

- **Compare two tables by selecting rows.** `select_records_daf(other, inverse=True)` gives the
  rows of this table whose keys the other lacks. On a SqlDaf it is a join, so no key comes into
  Python. Pass `silent_error=True`, or a key of the other table that this one lacks raises
  KeyError, on both. Avoid making key lists of both and comparing them.
- **Use `keys()` only where a list of keys is the result you want.** It reads every key into
  memory. It takes `astype='view'` and `silent_error`, as a Daf's does, so Daf code runs
  unchanged, but on a SqlDaf the view is built from all the keys. Better ways read no key:
  give the table itself as the keys, as in `b.select_records_daf(a)`; test one key with
  `key in table`; read one row with `table.select_record(key)`.
- **Sample from the first rows.** `daf[0:3].keys()` reads 3 rows. `daf.keys()[0:3]` reads every key first.
- **Count with `len()` of a selection.** Avoid a list made only to take its length.
- **Split a table with `select_by_dict()` and `inverse=True`.** No row is copied, and the rows
  stay in table order. A row with Python NULL in the column falls on the inverse side.
  Avoid `groupby()` and then `append()` of the groups: an empty Daf that appends a SqlDaf fails.
- **Select first, then `to_daf()`.** `to_daf()` reads every row into memory. On a Daf it gives
  the Daf itself, so `select_records_daf(keys).to_daf().apply_dtypes()` reads either kind.
- **Summarize each group with grouped counts, not a loop over the groups.**
  `valuecounts_for_colname1_groupedby_colname2('is_bmd', 'batchid')` counts the values of one
  column in every group at once. A group's size is the sum of its counts, and its distinct
  values are the keys. A loop over `groupby()` reads each group: on a SqlDaf that is a query per
  group. On AuditEngine's pilot d05, 133,422 rows in 11,157 batches, the loop took 565 s on a
  SqlDaf and the grouped counts 1.64 s. On a Daf both took about 0.7 s. Both classes give the
  groups and their values in the order first seen.
- **Write a table into a report with `to_md()`.** `rows[0:50].to_md()` gives those 50 rows and
  no summary line. An f-string gives `str()`, which keeps 10 rows and adds the summary line, with
  the name of the table. A selection of a SqlDaf names its TEMP table there.
- **Respect a read-only table, and work on a copy.** Open a table that others depend on with
  `read_only=True`, and make the changes on `copy()`, a TEMP table of its own. Reading,
  selecting, grouping and joining need no copy. Only a change to the data does:

  ```python
  bif_sdaf = SqlDaf('full_bif', read_only=True)
  p1_sdaf = bif_sdaf.select_by_dict({'precinct': 'P1'})   # no copy
  work_sdaf = p1_sdaf.copy()                              # changes go here
  work_sdaf.insert_col('sheet0', default=0)
  work_sdaf.save('full_bif_p1', db_path='p1.db')          # keep it, if wanted
  ```
  On a Daf the same lines run, and `copy()` gives rows of its own, as a change needs there too.
- **Join tables in their own files directly.** A job that keeps each table in a file of its own
  opens each where it is. A join brings the other table in for itself, and the files are only
  read. Keep the result with `save(db_path=)`. To build a file named for the table a stage makes,
  copy each input into it with `from_db()`.

```pycon
>>> votes_daf = Daf(cols=['ballot_id', 'corrupted'], keyfield='ballot_id',
...                 lol=[['b1', 0], ['b2', 1], ['b3', ''], ['b4', 0]])
>>> votes_daf
<BLANKLINE>
| ballot_id | corrupted |
| --------: | --------: |
|        b1 |         0 |
|        b2 |         1 |
|        b3 |           |
|        b4 |         0 |
<BLANKLINE>
%% daf rows=4; cols=2; keyfield='ballot_id'; name=''
<BLANKLINE>
>>> images_daf = Daf(cols=['ballot_id', 'pages'], keyfield='ballot_id',
...                  lol=[['b2', 2], ['b4', 1], ['b9', 2]])
>>> images_daf
<BLANKLINE>
| ballot_id | pages |
| --------: | ----: |
|        b2 |     2 |
|        b4 |     1 |
|        b9 |     2 |
<BLANKLINE>
%% daf rows=3; cols=2; keyfield='ballot_id'; name=''
<BLANKLINE>
>>> votes_sdaf = SqlDaf.from_daf(votes_daf, name='votes_cb')
>>> images_sdaf = SqlDaf.from_daf(images_daf, name='images_cb')
>>> votes_daf.valuecounts_for_colname1_groupedby_colname2('ballot_id', 'corrupted') == \
...     votes_sdaf.valuecounts_for_colname1_groupedby_colname2('ballot_id', 'corrupted')
True
>>> for votes, images in [(votes_daf, images_daf), (votes_sdaf, images_sdaf)]:
...     votes_only = votes.select_records_daf(images, inverse=True, silent_error=True)
...     images_only = images.select_records_daf(votes, inverse=True, silent_error=True)
...     print(votes_only.keys(), images_only.keys(), len(votes) - len(votes_only),
...           votes[0:2].keys(), votes.select_by_dict({'corrupted': 1}, inverse=True).keys())
['b1', 'b3'] ['b9'] 2 ['b1', 'b2'] ['b1', 'b3', 'b4']
['b1', 'b3'] ['b9'] 2 ['b1', 'b2'] ['b1', 'b3', 'b4']
```

## Differences from Daf

- **Types.** A column holds what SQLite stores, such as int, float, text or SQL NULL. A bool
  is stored as 0 or 1. CSV values are text until `apply_dtypes()` converts them. A dict or list
  cell stays as its text.
- **Selections.** A Daf selection holds its rows at once. A SqlDaf selection fixes them when
  first read whole, by `to_daf()`, `len()` or the like. Reading one row fixes nothing.
- **Changes through a selection.** Setting a value changes the parent, as on a Daf. Adding
  rows, converting and sorting copy the selection first, and leave the parent as it was.
- **Sorting.** A Daf sorts its own rows. A SqlDaf sorts into a new TEMP table, and the table
  in the file keeps its order.
- **Rows in a loop.** `for row in sdaf` and `iter_dict()` give copies of the rows. Changing a
  row in the loop does not change the table.
- **Mixed types in a sort.** Numbers sort before text, where a Daf raises TypeError.
- **Keys.** An int in brackets is always a position. Assigning with an int raises
  NotImplementedError. Use `update_record_krow()` for a row by key. A Python NULL key, the empty string, is not supported. A join assumes
  the keys are unique, and does not check.
- **Positions after a delete.** A table that has had rows deleted cannot be read by position.
- **Printing.** `str()`, `print()` and an f-string show the same Markdown table as a Daf of the
  same rows, reading only the rows shown. So does `repr()`, at the prompt.
- **Methods.** Methods not written for SQL raise NotImplementedError.

## Performance

These timings were taken on the 4-vCPU machine this library is developed on, with Python 3.13.
They come from short scripts that are not part of the repository. Each table has short text
and int columns.

| Operation | Rows | Daf | SqlDaf |
| :-------- | ---: | --: | -----: |
| sort by one text column, 6 columns | 1,000,000 | 1.0 s | 2.4 s |
| group into 500 groups and sum each | 100,000 | 0.73 s | 2.3 s |
| drop one of 6 columns | 1,000,000 | 0.30 s | 0.73 s |
| remove repeated keys, 500 keys | 1,000,000 | 0.23 s | 1.5 s |
| one row by key, `select_record()` | 200,000 | 3.4 µs | 39 µs |

A query passes through sqlite3, which costs about 12 to 24 µs each time. So work in whole
statements where you can. Many rows in one statement cost far less per row than one row per
statement. Put many small writes in one `transaction()`.

The [SqlDaf API][daffodil.sqldaf.SqlDaf] lists every method, grouped by what it does.
