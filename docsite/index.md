<img src="images/daffodil_logo.png" alt="" width="90" align="right">

# Daffodil

Daffodil gives Python fast, lightweight 2-D data tables. A table is a list of rows, and each
row is a plain Python list. Use it where data arrives and is processed row by row: reading
files, building records, cleaning, reshaping, and writing them out again.

![The Daffodil data model: lol, hd, kd and dtypes](images/data_model.svg)

## Install

```bash
pip install daffodil
```

Python 3.10 or later. The table code is pure Python. pandas, NumPy and a few other packages
are needed only for the features that use them. See [Install](guide/getting-started.md#install).

## Quick start

Make a table, with `id` as the keyfield, the column whose values name the rows:

```pycon
>>> from daffodil.daf import Daf
>>> daf = Daf(cols=['id', 'item', 'qty'], keyfield='id')
```

Add rows. A dict is placed by column name, and a list goes in column order. `append()` adds
the row to the table and returns the table, so the prompt shows it each time:

```pycon
>>> daf.append({'id': 'a1', 'item': 'pen', 'qty': 3})
<BLANKLINE>
| id | item | qty |
| -: | ---: | --: |
| a1 |  pen |   3 |
<BLANKLINE>
%% daf rows=1; cols=3; keyfield='id'; name=''
<BLANKLINE>
>>> daf.append(['b2', 'ink', 10])
<BLANKLINE>
| id | item | qty |
| -: | ---: | --: |
| a1 |  pen |   3 |
| b2 |  ink |  10 |
<BLANKLINE>
%% daf rows=2; cols=3; keyfield='id'; name=''
<BLANKLINE>
```

In a program, nothing is shown. You write `daf.append(row)`, and the table grows.

Select with brackets, as `daf[rows, cols]`. A string is a key for rows and a name for columns.
A number is a position. A selection is a smaller table:

```pycon
>>> daf['b2']                   # the row with key 'b2'
<BLANKLINE>
| id | item | qty |
| -: | ---: | --: |
| b2 |  ink |  10 |
<BLANKLINE>
%% daf rows=1; cols=3; keyfield='id'; name=''
<BLANKLINE>
>>> daf[:, 'qty']               # every row, the qty column
<BLANKLINE>
| qty |
| --: |
|   3 |
|  10 |
<BLANKLINE>
%% daf rows=2; cols=1; keyfield=''; name=''
<BLANKLINE>
```

To get plain values out of a selection, ask for them:

```pycon
>>> daf['b2', 'qty'].to_value()   # one cell
10
>>> daf[:, 'qty'].to_list()       # one column
[3, 10]
>>> daf.select_record('a1')       # one row, as a dict
{'id': 'a1', 'item': 'pen', 'qty': 3}
```

Select rows by a condition. The function gets each row as a dict:

```pycon
>>> daf.select_where(lambda row: row['qty'] > 5)
<BLANKLINE>
| id | item | qty |
| -: | ---: | --: |
| b2 |  ink |  10 |
<BLANKLINE>
%% daf rows=1; cols=3; keyfield='id'; name=''
<BLANKLINE>
```

The [Guide](guide/getting-started.md) starts from here, and the [cheatsheet](cheatsheet.md) has it all on one page.

## The data model

The rows are kept in `lol`, a list of lists. A row holds only its cells. The column names are
kept once, in the header dict `hd`, which maps each name to its column number. A table can
have a keyfield. Then the key dict `kd` maps each key to its row number, so a lookup by key
is one dict access.

A cell can hold any Python object: text, a number, a list, a dict, or even another Daf. There
is no conversion on the way in or out. The data stays the Python objects you put there.

## Why not pandas?

pandas stores a table as columns, in NumPy arrays. That suits heavy numeric work on columns.
It costs more for data that is built and used as rows. pandas has no cheap way to add one row,
so the usual pattern collects the rows in Python, then converts them all at once. Daffodil needs
no conversion step.

In one benchmark, of 1,000 rows of 1,000 columns, mostly large ints:

| How the table is built | Total time | Peak memory | Kept memory |
|---|--:|--:|--:|
| pandas: collect a dict per row, then `pd.DataFrame()` | 543 ms | 89 MB | 8 MB |
| pandas: collect a list per row, then `pd.DataFrame()` | 479 ms | 60 MB | 8 MB |
| Daffodil: `append()` each row to the Daf | 173 ms | 36 MB | 36 MB |

Daffodil built the table in about a third of the time, with a lower peak. pandas kept less,
because it stores these ints as 8-byte numbers. With small ints or text, both kept the same.
Results depend on the values, the shape and the versions.
[Performance](guide/performance.md) gives the details, and the script to run it again.

**A selection shares the rows.** Selecting rows copies no data, so it is fast. A change to a
cell through the selection shows in the original. See [copy()][daffodil.daf.Daf.copy].

**Fewer surprises with mixed data.** A column can mix text, numbers and empty cells. Nothing
is coerced to NaN or to a common dtype behind your back.

## When to use pandas instead

If the work is mostly math over whole numeric columns, pandas, NumPy or Polars are faster.
Daffodil converts to them when needed, with `to_pandas_df()` and `to_numpy()`. Build and
clean the data as rows in Daffodil, then hand the numeric columns over for the number crunching.

## Tables larger than memory

SqlDaf stores a table in a SQLite database file, so its rows stay out of Python memory. It
supports tables larger than the memory available, and many operations run directly in SQLite.
A small selection comes into memory as a Daf with `to_daf()`.

A Daf in memory is faster. Whole-table operations ran 2 to 6 times faster in our timings, and
a lookup by key about 11 times faster. So use a Daf for a table that fits in memory.

SqlDaf has the most common Daf methods, under the same names. Some behave differently, because
the data is in a database. Any other Daf method raises NotImplementedError. The guide page
[Tables larger than memory](guide/sqldaf.md) shows how to use it.

## Next

- [Getting started](guide/getting-started.md), then the rest of the Guide in the menu.
- [Coming from pandas](guide/from-pandas.md): the same tasks side by side.
- [Daf overview](api/daf/index.md), and the Daf API by section, for every method.
- [SqlDaf overview](api/sqldaf/index.md), for tables in SQLite.
- [Changelog](https://github.com/raylutz/daffodil/blob/main/CHANGELOG.md), and the code on
  [GitHub](https://github.com/raylutz/daffodil).
