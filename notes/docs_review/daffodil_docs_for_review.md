# Daffodil documentation, for review

Daffodil is a small, fast, pure-Python library of 2-D tables. The main class is Daf: a list of rows,
each a plain Python list, with the column names in a dict. KeyedList is a row that can be read by name.
SqlDaf, new in version 0.8.0, keeps the rows in a SQLite table, for tables too large for memory.

This file holds all of the user documentation, in this order:

1. README.md, the front page on GitHub and PyPI.
2. Every page of the docs site, in the order of its menu: the Home page, the cheatsheet, the Guide
   pages, then the API reference for Daf, SqlDaf and KeyedList.

Every code example in the Guide pages and the docstrings, and every line of code in the cheatsheet,
runs as a test. In the API pages, a directive of the site is replaced by the signature and docstring
of the method it names.

## What to review

- Can a new user understand what Daffodil is, when to use it, and get started from the Guide alone?
- Is the mental model easy to assemble: what a selection shares, what makes a copy, what changes the original?
- Is anything missing, wrong, confusing, or explained twice in a way that could drift apart?
- Do the docstrings teach: what each method does, when to use it, what it returns, what can surprise a caller?
- Is the cheatsheet complete and explicit enough for an AI model writing code with Daffodil?
- Are the names consistent across pages: methods, parameters, and the terms for the same idea?
- Writing style the project follows: short sentences with one idea each, no em dashes, no semicolons,
  few parenthetical asides, plain words. Flag lines that break these, or that read as choppy.

Give findings as a numbered list, the most important first. For each: the page, the method or
heading, the problem, and a suggested rewrite. This review is of the documentation. Where you are
unsure how the code behaves, say so, and ask for the behavior to be checked and documented.

---


---

# README.md

<img src="https://raw.githubusercontent.com/raylutz/daffodil/main/docsite/images/daffodil_logo.png" alt="Daffodil logo" width="110">

## Python Daffodil

Daffodil (DAtaFrames For Optimized Data Inspection and Logical processing) gives Python fast,
lightweight 2-D data tables. A table is a list of rows, and each row is a plain Python list.
Use it where data arrives and is processed row by row: reading files, building records,
cleaning, reshaping, and writing them out again.

**Documentation: https://raylutz.github.io/daffodil/**

Status: alpha, and used in production for several years. The changes of each release are in
the [changelog](https://github.com/raylutz/daffodil/blob/main/CHANGELOG.md).

### Install

```bash
pip install daffodil
```

Python 3.10 or later. The table code is pure Python. pandas, NumPy and a few other packages
are needed only for the features that use them, such as `to_pandas_df()`, and are not
installed with daffodil. The [Install section](https://raylutz.github.io/daffodil/guide/getting-started/#install)
of the Guide lists them.

### The data model

![The Daffodil data model: lol, hd, kd and dtypes](https://raw.githubusercontent.com/raylutz/daffodil/main/docsite/images/data_model.png)

The rows are kept in `lol`, a list of lists. The column names are kept once, in the header
dict `hd`, which maps each name to its position. A table can have a keyfield, a column whose
values identify the rows, and then the key dict `kd` maps each key to its row. A cell can hold
any Python object: text, a number, a list, a dict, or even another table.

### Quick start

```python
from daffodil.daf import Daf

daf = Daf(cols=['id', 'item', 'qty'], keyfield='id')
daf.append({'id': 'a1', 'item': 'pen', 'qty': 3})
daf.append(['b2', 'ink', 10])

daf['b2']                     # the row with key 'b2'
daf[0]                        # the first row
daf[:, 'qty']                 # the qty column
daf['b2', 'qty'].to_value()   # 10
daf.select_where(lambda row: row['qty'] > 5)
print(daf)                    # a Markdown table
```

### Why Daffodil

- **Adding a row is cheap.** It is appended to a Python list. pandas has no cheap way to add
  a row, so code that uses it often collects the rows first and converts them at the end.
- **No conversion.** The data stays the Python objects you put in, so there is no cost to
  get it in or out.
- **A selection shares the rows.** Selecting rows copies no data, and lookups by key are dict
  lookups.
- **Mixed data is fine.** A column can mix text, numbers and empty cells, and nothing is
  coerced to NaN or to a common type.
- **Tables print as Markdown,** ready for a terminal or a report, and can be read back.
- **Small and pure Python,** so it starts fast, which matters in short-lived cloud functions.

For heavy math over whole numeric columns, pandas, NumPy or Polars are faster. Build and clean
the data as rows in Daffodil, then hand the numeric columns over with `to_pandas_df()` or
`to_numpy()`. The [documentation](https://raylutz.github.io/daffodil/) compares the two, with
measurements.

### Tables larger than memory

SqlDaf stores a table in a SQLite database file, so its rows stay out of Python memory. It
supports tables larger than the memory available, and many operations run directly in SQLite.
A small selection comes into memory as a Daf with `to_daf()`.

A Daf in memory is faster. Whole-table operations ran 2 to 6 times faster in our timings, and
a lookup by key about 11 times faster. So use a Daf for a table that fits in memory.

SqlDaf has the most common Daf methods, under the same names. Some behave differently, because
the data is in a database. Any other Daf method raises NotImplementedError. The guide page
[Tables larger than memory](https://raylutz.github.io/daffodil/guide/sqldaf/) shows how to use it.

### Learn more

- [Cheatsheet](https://raylutz.github.io/daffodil/cheatsheet/), everything on one page
- [Getting started](https://raylutz.github.io/daffodil/guide/getting-started/)
- [Building tables](https://raylutz.github.io/daffodil/guide/building-tables/), including the
  fast way to build rows in a loop
- [Selecting and indexing](https://raylutz.github.io/daffodil/guide/selecting/)
- [Columns, keys and schemas](https://raylutz.github.io/daffodil/guide/columns-keys-schemas/)
- [Types and conversion](https://raylutz.github.io/daffodil/guide/types/)
- [Apply, reduce and formulas](https://raylutz.github.io/daffodil/guide/apply-reduce-formulas/)
- [Joins](https://raylutz.github.io/daffodil/guide/joins/)
- [Markdown reports](https://raylutz.github.io/daffodil/guide/markdown/)
- [Coming from pandas](https://raylutz.github.io/daffodil/guide/from-pandas/)
- [Tables larger than memory](https://raylutz.github.io/daffodil/guide/sqldaf/), with SqlDaf
- [The Daf API](https://raylutz.github.io/daffodil/api/daf/), every method with examples
- [The SqlDaf API](https://raylutz.github.io/daffodil/api/sqldaf/)

Questions and ideas are welcome as [GitHub issues](https://github.com/raylutz/daffodil/issues).
Daffodil is released under the MIT license.

---

# Docs page: Home (index.md)

<img src="images/daffodil_logo.png" alt="" width="90" align="right">

## Daffodil

Daffodil gives Python fast, lightweight 2-D data tables. A table is a list of rows, and each
row is a plain Python list. Use it where data arrives and is processed row by row: reading
files, building records, cleaning, reshaping, and writing them out again.

![The Daffodil data model: lol, hd, kd and dtypes](images/data_model.svg)

### Install

```bash
pip install daffodil
```

Python 3.10 or later. The table code is pure Python. pandas, NumPy and a few other packages
are needed only for the features that use them. See [Install](guide/getting-started.md#install).

### Quick start

```python
from daffodil.daf import Daf

daf = Daf(cols=['id', 'item', 'qty'], keyfield='id')
daf.append({'id': 'a1', 'item': 'pen', 'qty': 3})
daf.append(['b2', 'ink', 10])

daf['b2']                     # the row with key 'b2'
daf[0]                        # the first row
daf[:, 'qty']                 # the qty column
daf['b2', 'qty'].to_value()   # 10
daf.select_where(lambda row: row['qty'] > 5)
print(daf)                    # a Markdown table
```

The [Guide](guide/getting-started.md) starts from here, and the [cheatsheet](cheatsheet.md) has it all on one page.

### The data model

The rows are kept in `lol`, a list of lists. A row holds only its cells. The column names are
kept once, in the header dict `hd`, which maps each name to its column number. A table can
have a keyfield. Then the key dict `kd` maps each key to its row number, so a lookup by key
is one dict access.

A cell can hold any Python object: text, a number, a list, a dict, or even another Daf. There
is no conversion on the way in or out. The data stays the Python objects you put there.

### Why not pandas?

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

### When to use pandas instead

If the work is mostly math over whole numeric columns, pandas, NumPy or Polars are faster.
Daffodil converts to them when needed, with `to_pandas_df()` and `to_numpy()`. Build and
clean the data as rows in Daffodil, then hand the numeric columns over for the number crunching.

### Tables larger than memory

SqlDaf stores a table in a SQLite database file, so its rows stay out of Python memory. It
supports tables larger than the memory available, and many operations run directly in SQLite.
A small selection comes into memory as a Daf with `to_daf()`.

A Daf in memory is faster. Whole-table operations ran 2 to 6 times faster in our timings, and
a lookup by key about 11 times faster. So use a Daf for a table that fits in memory.

SqlDaf has the most common Daf methods, under the same names. Some behave differently, because
the data is in a database. Any other Daf method raises NotImplementedError. The guide page
[Tables larger than memory](guide/sqldaf.md) shows how to use it.

### Next

- [Getting started](guide/getting-started.md), then the rest of the Guide in the menu.
- [Coming from pandas](guide/from-pandas.md): the same tasks side by side.
- [Daf overview](api/daf/index.md), and the Daf API by section, for every method.
- [SqlDaf overview](api/sqldaf/index.md), for tables in SQLite.
- [Changelog](https://github.com/raylutz/daffodil/blob/main/CHANGELOG.md), and the code on
  [GitHub](https://github.com/raylutz/daffodil).

---

# Docs page: Cheatsheet (cheatsheet.md)

---
name: daffodil
description: Cheatsheet for Daffodil, the Python package of 2-D tables built on lists (`pip install daffodil`, `from daffodil.daf import Daf`). Use it to write or read code that creates, selects, builds, changes or exports Daf tables.
---

## Daffodil cheatsheet

Daffodil keeps a table as a list of rows, each a plain Python list, with the column names in a
dict. Every line of code in the task tables is run as a test. The docs are at https://raylutz.github.io/daffodil/.
This file is generated from notes/scripts/cheatsheet_content.py. Edit that file, not this one.

### Setup

Each task table below starts from this code, with the default modes. The lines within one
table run in order, so a line may use what the line above it made. The tables do not depend on
each other. The SqlDaf table is one sequence, from top to bottom.

```python
from daffodil.daf import Daf, NULL
from daffodil.keyedlist import KeyedList
daf = Daf(cols=['id', 'item', 'g', 'qty'], keyfield='id',
          lol=[['a1', 'pen', 'x', 3], ['b2', 'ink', 'y', 10], ['c3', 'pad', 'x', 7], ['d4', 'cap', 'y', 1]])
other = Daf(cols=['id', 'price'], keyfield='id', lol=[['a1', 1.5], ['b2', 4.0]])
lod = [{'id': 'a1', 'qty': 3}, {'id': 'b2', 'qty': 10}]
types = {'id': str, 'item': str, 'g': str, 'qty': int}
k, n, row = 'a1', 2, ['e5', 'tag', 'x', 2]
open('data.csv', 'w').write('id,qty\na1,3\nb2,10\n')
```

### Indexing at a glance

```
daf[rows, cols]     rows and columns
daf[rows]           rows, all columns
daf[:, cols]        columns, all rows
```

|          You write           |                  As rows (first)                  |               As columns (second)               |
| :--------------------------- | :------------------------------------------------ | :---------------------------------------------- |
| `:`                          | all rows                                          | all columns                                     |
| `2`                          | row 2, by position                                | column 2, by position                           |
| `-1`                         | the last row                                      | the last column                                 |
| `2:5`, `2:`, `:5`, `::2`     | a slice of rows                                   | a slice of columns                              |
| `range(2, 5)`                | rows 2, 3, 4                                      | columns 2, 3, 4                                 |
| `[0, 4, 2]`                  | these rows, in this order                         | these columns, in this order                    |
| `[range(0, 2), range(4, 6)]` | rows 0, 1, 4, 5                                   | columns 0, 1, 4, 5                              |
| `'a1'`                       | the row with key `'a1'`                           | the column named `'a1'`                         |
| `['c3', 'a1']`               | the rows with these keys, in this order           | the columns with these names, in this order     |
| `('a1', 'c3')`               | the rows from key a1 through key c3, in row order | the columns from a1 through c3, in column order |
| `('a1', None)`, `('a1',)`    | key a1 to the end                                 | name a1 to the end                              |
| `(None, 'c3')`               | the start through key c3                          | the start through name c3                       |
| `[]`                         | no rows                                           | no columns                                      |

- Selecting whole rows, as `daf[rows]` or `daf[rows, :]`, copies no data: the new Daf shares the rows with the original, so a change to a cell in one shows in the other. Selecting some of the columns makes new row lists. Use `copy('editable')` for row lists of your own.
- A list or dict held in a cell is the same object in every selection and in `copy('editable')`. Only `copy('deep')` copies it.
- An integer alone, or in a list, slice or range, is a position. A key or a column name that is an integer needs `select_krows()` or `select_kcols()`.
- The two ends of a tuple are always keys or names, of any type, as in `daf[(10, 20), :]` for int keys. The tuple takes the rows between them in the table's order. It does not compare the values. If the second end comes first in the table, the result is empty.
- Rows by key need a keyfield.
- A tuple of row keys needs the column part, even if it is `:`, as in `daf[('a1', 'c3'), :]`. A tuple of two standing alone is read as `[rows, cols]`.
- A list holds integers, or ranges, or strings, not a mix. `None` alone is not a selector.
- The result is a Daf. `.to_value()`, `.to_list()` or `.to_dict()` give plain values, or set `daf.retmode = 'val'`.
- The same selectors set values: `daf[rows, cols] = value`.

### Create

|                              Code                               |            What it does             |
| :-------------------------------------------------------------- | :---------------------------------- |
| `daf = Daf(cols=['id', 'qty'], lol=[['a1', 3]], keyfield='id')` | from lists                          |
| `daf = Daf.from_lod(lod, keyfield='id')`                        | from a list of dicts                |
| `daf = Daf.from_csv('data.csv')`                                | from a CSV file; the cells are text |
| `daf = Daf(cols=['id', 'qty'])`                                 | empty, to append to                 |

`Daf(schema=MySchema)` takes the columns, dtypes, keyfield and defaults from a `@schemaclass`.

### Look

|      Code       |   What it does   |
| :-------------- | :--------------- |
| `print(daf)`    | a Markdown table |
| `len(daf)`      | number of rows   |
| `daf.shape()`   | (rows, cols)     |
| `daf.columns()` | column names     |
| `daf.keys()`    | keyfield values  |
| `daf[:5]`       | the first 5 rows |

### Rows

|          Code          |             What it does              |
| :--------------------- | :------------------------------------ |
| `daf[2]`               | row 2                                 |
| `daf[-3:]`             | the last 3 rows                       |
| `daf[[0, 2]]`          | rows 0 and 2                          |
| `daf['a1']`            | the row with key 'a1'                 |
| `daf[['a1', 'c3']]`    | the rows with these keys              |
| `daf[('a1', 'c3'), :]` | keys a1 through c3; the `:` is needed |
| `daf[('c3', None), :]` | key c3 to the end                     |

### Columns and cells

|          Code           |          What it does          |
| :---------------------- | :----------------------------- |
| `daf[:, 'qty']`         | the qty column                 |
| `daf[:, ['id', 'qty']]` | two columns, in this order     |
| `daf[:, 1:3]`           | columns 1 and 2                |
| `daf[2, 'qty']`         | one cell, by position and name |
| `daf['b2', 'qty']`      | one cell, by key and name      |

### Values out of a selection

|             Code              |                          What it does                           |
| :---------------------------- | :-------------------------------------------------------------- |
| `daf['b2', 'qty'].to_value()` | the value: 10                                                   |
| `daf[:, 'qty'].to_list()`     | a column as a list                                              |
| `daf['b2'].to_dict()`         | a row as a dict                                                 |
| `daf.col('qty')`              | a column as a list, without building a Daf                      |
| `daf.retmode = 'val'`         | from now on, `daf[...]` of one cell, row or column gives values |
| `daf.retmode = 'obj'`         | back to Dafs, the default                                       |

### Rows by condition

|                      Code                      |              What it does               |
| :--------------------------------------------- | :-------------------------------------- |
| `daf.select_where(lambda row: row['qty'] > 5)` | rows where the function is true         |
| `daf.select_by_dict({'g': 'x'})`               | rows equal to these values; much faster |
| `daf.select_by_dict([{'g': 'x'}, {'g': 'y'}])` | rows equal to any of these              |
| `daf.select_krows(['a1'], inverse=True)`       | all rows but these keys                 |

### Set values

|                    Code                    |                       What it does                        |
| :----------------------------------------- | :-------------------------------------------------------- |
| `daf[2, 'qty'] = 0`                        | one cell                                                  |
| `daf['b2', 'qty'] = 0`                     | one cell, by key                                          |
| `daf[:, 'qty'] = [1, 2, 3, 4]`             | a column, from a list                                     |
| `daf[:, 'qty'] = 0`                        | a column, one value                                       |
| `daf[0] = ['a1', 'pen', 'x', 5]`           | a row, from a list                                        |
| `daf[0] = {'id': 'a1', 'qty': 5}`          | a row, by name; the columns it lacks become NULL          |
| `daf.update_record_krow('b2', {'qty': 4})` | some cells of the row with this key. An int is a key here |
| `daf.update_record_irow(1, {'qty': 4})`    | some cells of the row at this position                    |

### Build in a loop

|                     Code                     |                What it does                 |
| :------------------------------------------- | :------------------------------------------ |
| `daf.append({'id': 'e5', 'qty': 2})`         | a dict, placed by name                      |
| `daf.append(['e5', 'tag', 'x', 2])`          | a list, in column order                     |
| `daf.append(lol=[['e5', 'tag', 'x', 2]])`    | several lists                               |
| `row = daf.default_record(astype=KeyedList)` | a new row: the Daf's columns, with defaults |
| `row['qty'] = 2`                             | fill it by name, in any order               |
| `daf.append(row, fast=True)`                 | append it with no copy                      |

Without `fast`, every row is checked and copied.

With `fast=True`, a KeyedList must come from `default_record(astype=KeyedList)` of the same Daf. A dict or list is checked fully if it is the first row of an empty Daf, and otherwise only for its length. Nothing is copied, so do not change a list after appending it.

### Reshape

|                    Code                    |            What it does             |
| :----------------------------------------- | :---------------------------------- |
| `daf.sort_by_colname('qty', reverse=True)` | sort the rows of this Daf, in place |
| `daf.insert_col('note', ['', '', '', ''])` | add a column                        |
| `daf.select_cols(exclude_cols=['g'])`      | drop columns                        |
| `daf.transpose()`                          | a new Daf, with rows as columns     |
| `daf.copy('editable')`                     | a copy with its own rows            |
| `daf.rename_cols({'qty': 'n'})`            | rename columns                      |

### Types

|                         Code                          |                    What it does                     |
| :---------------------------------------------------- | :-------------------------------------------------- |
| `daf.apply_dtypes(dtypes=types)`                      | convert text to types, in place; names every column |
| `daf.apply_dtypes(dtypes={'qty': int}, cols=['qty'])` | convert only some columns                           |

NULL, a missing value, is `''`, the empty string. dtypes describe the columns, and do not force them. A cell can hold any Python object.

### Apply, reduce, group

|                               Code                                |              What it does               |
| :---------------------------------------------------------------- | :-------------------------------------- |
| `daf.apply(lambda row: {'id': row['id'], 'q2': row['qty'] * 2})`  | a new table, row by row                 |
| `daf.reduce(Daf.sum_da, cols=['qty'])`                            | column sums, as a dict                  |
| `daf.groupby('g')`                                                | a KeyedList of Dafs, one for each value |
| `daf.groupby_cols_reduce(['g'], Daf.sum_da, reduce_cols=['qty'])` | sums for each group                     |
| `daf.valuecounts_for_colname('g')`                                | counts: `{'x': 2, 'y': 2}`              |

### Joins and export

|             Code              |            What it does             |
| :---------------------------- | :---------------------------------- |
| `daf.join(other, how='left')` | by key: inner, left, right or outer |
| `daf.to_lod()`                | a list of dicts                     |
| `daf.to_md(max_rows=10)`      | Markdown text                       |
| `daf.to_csv_file('out.csv')`  | a CSV file                          |
| `daf.to_pandas_df()`          | a pandas DataFrame                  |

### Tables larger than memory

|                            Code                             |                       What it does                       |
| :---------------------------------------------------------- | :------------------------------------------------------- |
| `from daffodil.sqldaf import SqlDaf`                        | SqlDaf keeps the rows in a SQLite table                  |
| `SqlDaf.db_path = 'job.db'`                                 | the database file, set once                              |
| `sdaf = SqlDaf.from_daf(daf, name='items')`                 | a new table, items, from the Daf, with its keyfield      |
| `sdaf = SqlDaf('items')`                                    | open the table again by its name                         |
| `csv_sdaf = SqlDaf.from_csv('data.csv', keyfield='id')`     | a new table, data, from a CSV file. Every cell is text   |
| `csv_sdaf.apply_dtypes(dtypes={'qty': int}, cols='qty')`    | convert numbers before any numeric work                  |
| `sdaf.select_record('a1')`                                  | one row by key, as a dict                                |
| `sdaf.iloc(0)`                                              | one row by position, as a dict                           |
| `sdaf['a1', 'qty'].to_value()`                              | one value. Brackets give a SqlDaf                        |
| `sdaf.update_record_krow('a1', {'qty': 4})`                 | change some cells of the row with this key               |
| `sdaf.select_where_sql('qty > ?', [5])`                     | a selection by a SQL condition                           |
| `sdaf.select_by_dict({'g': 'x'}).to_daf()`                  | a selection, read into memory as a Daf                   |
| `sdaf.groupby('g')`                                         | a KeyedList of selections, with an index on g            |
| `sdaf.sql('SELECT g, SUM(qty) AS n FROM items GROUP BY g')` | any SQL. The whole result comes back as a Daf            |
| `sdaf.copy(name='items_copy').save()`                       | an independent copy, kept in the file                    |
| `sdaf.sort_by_colname('qty')`                               | sort. The SqlDaf now refers to a sorted TEMP table       |
| `SqlDaf.close()`                                            | close the file. Every SqlDaf on it can no longer be used |

The lines run in order. They start from the Daf `daf`, with the columns id, item, g and qty and the keyfield id, and from the file data.csv, with the columns id and qty.

Brackets return a SqlDaf, even for one cell. Use `.to_value()` for a value, and `select_record()` or `iloc()` for a row.

An int in brackets is always a position when reading. Assigning with an int, as in `sdaf[20, 'v'] = 1`, raises NotImplementedError. Look up an int key with `select_record()`, and change some of its cells with `update_record_krow()`.

A selection fixes its rows the first time `to_daf()`, `len()` or the like reads it whole. After that its rows are fixed, and changes to their values show in it. `select_record()` and `iloc()` with a position of 0 or more read one row, and fix nothing.

Setting a value through a selection changes the parent's row. `append()`, `concat()`, `apply_dtypes()` and a sort on a selection copy it first, and leave the parent as it was.

`to_daf()`, `col()`, `keys()`, `to_lod()` and `sql()` bring their whole result into memory. Select the rows first.

`copy()` and a sort make TEMP tables, which `close()` removes. `save()` keeps one in the file, under a name not in use.

Daf methods not written for SQL raise NotImplementedError. A Daf is faster for a table that fits in memory.

### From pandas

|         pandas          |                    Daffodil                    |
| :---------------------- | :--------------------------------------------- |
| `df['qty']`             | `daf[:, 'qty']` or `daf.col('qty')`            |
| `df.loc[k]`             | `daf.select_record(k)`                         |
| `df.iloc[n]`            | `daf[n]`                                       |
| `df.head(n)`            | `daf[:n]`                                      |
| `df.tail(n)`            | `daf[-n:]`                                     |
| `df[df.qty > 5]`        | `daf.select_where(lambda row: row['qty'] > 5)` |
| `pd.concat([df, new])`  | `daf.append(row)`                              |
| `df.merge(other)`       | `daf.join(other)`                              |
| `df.sort_values('qty')` | `daf.copy().sort_by_colname('qty')`            |
| `df.T`                  | `daf.transpose()`                              |

Each line is the same task, not the same expression. `daf[x]` selects rows, where `df[x]` selects a column.

`select_record(k)` gives a dict. `daf[k]` gives a Daf, and there an int `k` is a position.

`sort_by_colname()` sorts the Daf itself, so copy first to keep the original order.

`join()` is one to one, on the keyfields. See the Joins page of the Guide.

### Rules

- Import with `from daffodil.daf import Daf`. Name a table `daf`, as pandas uses `df`.
- A Daf is `lol`, a list of rows, each a plain list, plus `hd`, a dict of column name to position.
- `daf[rows, cols]` selects. An integer is a position. A string is a key of the keyfield for rows, and a column name for columns. A tuple of two is an inclusive range, in row order, and its ends are keys of any type.
- A selection returns a Daf. Use `.to_value()`, `.to_list()` or `.to_dict()` for plain values, or set `daf.retmode = 'val'`.
- A selection of rows shares the rows with the original: changing a cell in it changes the original. A selection of columns makes new row lists.
- `copy()` shares the row lists by default. Use `copy('editable')` for row lists of your own, or `copy('deep')` when a list or dict in a cell must be copied too.
- `for row in daf` gives each row as a new dict. Assigning to it does not change the table. Use `iter_klist()`, `apply_in_place()` or `update_record_krow()` to change values.
- `append()` does not check keys, so a key can repeat. A lookup of a repeated key finds its last row.
- A write with `[]` to a key cell does not update the key index. Call `daf.set_keyfield(daf.keyfield)` after changing keys, before the next lookup.
- `apply_dtypes()` leaves text that does not convert as it is, with no error. Use `check_schema()` to find it.
- `sort_by_colname()` sorts the Daf itself. `transpose()` returns a new Daf.
- `append()`, `apply_dtypes()` and most changing methods return the Daf itself, changed in place.
- A missing value is NULL, the empty string `''`, not None or NaN.
- The keyfield column stays in the table. Key lookups are dict lookups, built when first needed.
- Rows are cheap and columns cost more. Name the columns you need, as with `cols=`, rather than dropping the others.

### Common mistakes

- `daf[:-5]` is all but the last 5 rows. The last 5 are `daf[-5:]`.
- `apply_dtypes(dtypes=...)` without `cols=` must name every column. To convert some, add `cols=`.
- `groupby_cols_reduce()` without `reduce_cols=` reduces nothing.
- A KeyedList row is not a dict: `isinstance(row, dict)` is False, and `json.dumps(row)` fails. Use `row.to_dict()`.
- With `append(row, fast=True)` the table keeps the list itself. Do not fill one list in place and append it again.
- `daf[('a1', 'c3')]` without `, :` is read as `[rows, cols]`. Write `daf[('a1', 'c3'), :]`.
- Methods that do not exist, from old examples: `add_idx()` is `insert_idx_col()`, `to_csv()` is `to_csv_file()`.

---

# Docs page: Guide / Getting started (guide/getting-started.md)

## Getting started

### Install

```bash
pip install daffodil
```

Daffodil needs Python 3.10 or later. The table code is pure Python. Installing it also
installs two small pure-Python packages: Markdown, and xlsx2csv for Excel files.

Some features need another package, which is not installed with daffodil:

| Feature | Package | Install |
| :------ | :------ | :------ |
| `to_pandas_df()`, `from_pandas_df()` | pandas | `pip install daffodil[pandas]` |
| `to_numpy()`, `from_numpy()`, `to_donpa()`, `sum_np()` | NumPy | `pip install daffodil[numpy]` |
| faster reading of list and dict cells | pyjson5 | `pip install daffodil[pyjson5]` |
| `from_csv()` of an `http` or `https` address | requests | `pip install requests` |
| `from_csv()` and `to_csv_file()` with an `s3://` name | boto3 | `pip install boto3` |
| `from_pdf()`, which is experimental | pdfplumber | `pip install daffodil[pdf]` |

A method that needs a missing package raises ImportError, or a RuntimeError that names it.

### Make a table

A Daf is a list of rows, each a plain Python list, and a dict of the column names.

```pycon
>>> from daffodil.daf import Daf
>>> daf = Daf(cols=['id', 'item', 'qty'], lol=[['a1', 'pen', 3], ['b2', 'ink', 10]], keyfield='id')
>>> print(daf.to_md())
| id | item | qty |
| -: | ---: | --: |
| a1 |  pen |   3 |
| b2 |  ink |  10 |
```

`keyfield` names the column whose values identify the rows. It is optional.

From a list of dicts, the columns are the keys:

```pycon
>>> d2 = Daf.from_lod([{'id': 'a1', 'qty': 3}, {'id': 'b2', 'qty': 10}])
>>> d2.columns()
['id', 'qty']
```

From a CSV file, `Daf.from_csv('file.csv')` reads the header line as the column names. The
values are read as text. See [Types and conversion](types.md) to convert them.

### Look at it

```pycon
>>> len(daf), daf.shape(), daf.columns()
(2, (2, 3), ['id', 'item', 'qty'])
>>> daf.keys()
['a1', 'b2']
>>> bool(Daf())
False
```

`print(daf)` and `daf.to_md()` give a Markdown table. See [Markdown reports](markdown.md).

### Read values

A selection with `[]` returns a new Daf. Turn it into plain values with `to_value()`,
`to_list()` or `to_dict()`, or read them with the methods that never build a table.

```pycon
>>> daf[1, 'qty'].to_value()
10
>>> daf[0].to_list()
['a1', 'pen', 3]
>>> daf.select_record('b2')
{'id': 'b2', 'item': 'ink', 'qty': 10}
>>> daf.col('qty')
[3, 10]
```

See [Selecting and indexing](selecting.md) for all the ways to select.

### Add rows

```pycon
>>> _ = daf.append({'id': 'c3', 'item': 'pad', 'qty': 7})
>>> _ = daf.append(['d4', 'cap', 1])
>>> daf.col('id')
['a1', 'b2', 'c3', 'd4']
```

A dict is placed by column name, and a list is in column order. `append()` returns the Daf,
so these examples assign it to `_`, to keep the output short. See
[Building tables](building-tables.md) for the fast ways to build a table in a loop.

### Change values

```pycon
>>> daf[0, 'qty'] = 4
>>> daf.select_record('a1')['qty']
4
```

### The usual pattern

Read a file, go through it row by row, and build a new table. This is where Daffodil is
strongest: appending a row is cheap, and nothing is converted.

In the loop, each `row` is a new dict. Assigning a new value to it does not change the
source table. To change values in place, use `update_record_krow()`, `apply_in_place()`, or
`iter_klist()`, whose rows write through to the table.

```pycon
>>> source = Daf.from_lod([{'name': 'pen', 'price': '1.50'}, {'name': 'ink', 'price': '4.00'}])
>>> out = Daf(cols=['name', 'price_cents'])
>>> for row in source:
...     _ = out.append({'name': row['name'], 'price_cents': round(float(row['price']) * 100)})
>>> out.to_lod()
[{'name': 'pen', 'price_cents': 150}, {'name': 'ink', 'price_cents': 400}]
```

The same, with `apply()`, which calls a function for each row and builds the new table:

```pycon
>>> def to_cents(row):
...     return {'name': row['name'], 'price_cents': round(float(row['price']) * 100)}
>>> source.apply(to_cents).to_lod()
[{'name': 'pen', 'price_cents': 150}, {'name': 'ink', 'price_cents': 400}]
```

### Get the data out

```pycon
>>> out.to_lod()[0]
{'name': 'pen', 'price_cents': 150}
>>> out.lol
[['pen', 150], ['ink', 400]]
```

- `to_csv_file(path)` writes a CSV file.
- `to_pandas_df()` and `to_numpy()` hand the data to pandas or NumPy, for heavy numeric work.

---

# Docs page: Guide / Building tables (guide/building-tables.md)

## Building tables

Daffodil is built for tables that grow a row at a time. Appending a row adds it to a Python
list, so it is cheap. pandas has no cheap way to add a row, which is why code that uses pandas
often collects the rows in a list of dicts first and converts them at the end.

### append()

`append()` takes a dict, a KeyedList, a list of values, a list of dicts, or another Daf.

```pycon
>>> from daffodil.daf import Daf
>>> daf = Daf(cols=['id', 'qty'], keyfield='id')
>>> _ = daf.append({'qty': 3, 'id': 'a1'})      # dict: by name
>>> _ = daf.append(['b2', 10])                   # list: column order
>>> _ = daf.append([{'id': 'c3', 'qty': 7}])     # list of dicts
>>> _ = daf.append(lol=[['d4', 1], ['e5', 2]])   # several lists
>>> daf.col('id')
['a1', 'b2', 'c3', 'd4', 'e5']
```

`append()` returns the Daf, so these examples assign it to `_`.

- A dict that lacks a column gets NULL there, the empty string. A key that is not a column is
  dropped.
- A list shorter than the columns is padded with NULL. A longer one raises `ValueError`.
- The table gets its own copy of each row, so you can reuse your list or dict afterwards.

The key is not checked, which keeps appending fast. A key that is already there is added
again. With `respect_kd=True`, the row that has the same key is replaced instead:

```pycon
>>> _ = daf.append({'id': 'a1', 'qty': 30}, respect_kd=True)
>>> daf.select_record('a1'), len(daf)
({'id': 'a1', 'qty': 30}, 5)
```

### Rows to fill in: default_record()

`default_record()` gives a new record with the Daf's columns, in order, each set to the
schema's default, or to NULL if there is no schema. Fill it in, then append it.

With `astype=KeyedList`, the record is a KeyedList that shares the Daf's column names,
`row.hd is daf.hd`. Assigning to a key writes to that column's position, so the order in which
you fill it does not matter. And `append()` knows that its keys are the columns, so it skips
the check of its columns.

```pycon
>>> from daffodil.keyedlist import KeyedList
>>> marks = Daf(cols=['ballot', 'contest', 'x', 'y'])
>>> for ballot, contest, x, y in [('b1', 'mayor', 10, 20), ('b2', 'mayor', 11, 21)]:
...     row = marks.default_record(astype=KeyedList)
...     row['y'], row['x'] = y, x                   # by name, in any order
...     row['ballot'], row['contest'] = ballot, contest
...     _ = marks.append(row)
>>> marks.lol
[['b1', 'mayor', 10, 20], ['b2', 'mayor', 11, 21]]
```

### append(fast=True)

With `fast=True` you promise that each row is complete and in column order, and you give
the row to the table: nothing is copied. A few checks remain, and a row that fails one raises
`ValueError`, with a message that says which row and what is wrong.

| Row | Checked |
|---|---|
| KeyedList | It must share the Daf's column names, as one from `default_record(astype=KeyedList)`, `iloc()` or `iter_klist()` does. Nothing else is checked. |
| dict, the first row of an empty Daf | Its keys must be the columns, in order. |
| dict, a later row | Its number of keys must be the number of columns. |
| list | Its number of values must be the number of columns. |

```pycon
>>> fast = Daf(cols=['ballot', 'x'])
>>> for ballot, x in [('b1', 10), ('b2', 11)]:
...     row = fast.default_record(astype=KeyedList)
...     row['ballot'], row['x'] = ballot, x
...     _ = fast.append(row, fast=True)
>>> fast.lol
[['b1', 10], ['b2', 11]]
>>> fast.append(['b3'], fast=True)
Traceback (most recent call last):
    ...
ValueError: append(fast=True): row 2 has 1 values for 2 columns. With fast=True a list must have one value for each column, in column order. Leave out fast to pad a short list with NULL.
```

Two things `fast=True` cannot catch:

- A later dict or list in the wrong order, with the right number of values. Its values land
  in the wrong columns. A KeyedList from `default_record()` cannot have this problem.
- A list changed after it was appended. The table holds the list itself. Giving a variable a
  new list each time is fine. Filling one list in place and appending it again is not: every
  row is then that same list.

Measured at 1,000 columns, per row, including building the row: about 90 µs as a dict, 24 µs
as a KeyedList from `default_record()`, and 1.7 µs as that KeyedList with `fast=True`.

### from_lod()

`Daf.from_lod()` makes a Daf from a list of dicts. Without `cols`, the columns are all the
keys of all the dicts, in the order they first appear, and a dict that lacks a key gets NULL
there.

```pycon
>>> Daf.from_lod([{'a': 1}, {'a': 2, 'b': 5}]).lol
[[1, ''], [2, 5]]
```

With `fast=True`, every dict must have the same keys in the same order. The first dict is
checked fully, and every dict for its number of keys. For dicts you built yourself in a loop,
it is several times faster.

### Other ways to grow a table

- `daf.append(other_daf)` adds the rows of a Daf with the same columns. So does `concat()`.
- `insert_irow()` inserts a row at a position.
- `extend()` adds several rows, as `append()` does with a list of dicts or `lol=`.

---

# Docs page: Guide / Selecting and indexing (guide/selecting.md)

## Selecting and indexing

### The two coordinates

<!-- indexing reference: generated by notes/scripts/build_cheatsheet.py -->
```
daf[rows, cols]     rows and columns
daf[rows]           rows, all columns
daf[:, cols]        columns, all rows
```

|          You write           |                  As rows (first)                  |               As columns (second)               |
| :--------------------------- | :------------------------------------------------ | :---------------------------------------------- |
| `:`                          | all rows                                          | all columns                                     |
| `2`                          | row 2, by position                                | column 2, by position                           |
| `-1`                         | the last row                                      | the last column                                 |
| `2:5`, `2:`, `:5`, `::2`     | a slice of rows                                   | a slice of columns                              |
| `range(2, 5)`                | rows 2, 3, 4                                      | columns 2, 3, 4                                 |
| `[0, 4, 2]`                  | these rows, in this order                         | these columns, in this order                    |
| `[range(0, 2), range(4, 6)]` | rows 0, 1, 4, 5                                   | columns 0, 1, 4, 5                              |
| `'a1'`                       | the row with key `'a1'`                           | the column named `'a1'`                         |
| `['c3', 'a1']`               | the rows with these keys, in this order           | the columns with these names, in this order     |
| `('a1', 'c3')`               | the rows from key a1 through key c3, in row order | the columns from a1 through c3, in column order |
| `('a1', None)`, `('a1',)`    | key a1 to the end                                 | name a1 to the end                              |
| `(None, 'c3')`               | the start through key c3                          | the start through name c3                       |
| `[]`                         | no rows                                           | no columns                                      |

- Selecting whole rows, as `daf[rows]` or `daf[rows, :]`, copies no data: the new Daf shares the rows with the original, so a change to a cell in one shows in the other. Selecting some of the columns makes new row lists. Use `copy('editable')` for row lists of your own.
- A list or dict held in a cell is the same object in every selection and in `copy('editable')`. Only `copy('deep')` copies it.
- An integer alone, or in a list, slice or range, is a position. A key or a column name that is an integer needs `select_krows()` or `select_kcols()`.
- The two ends of a tuple are always keys or names, of any type, as in `daf[(10, 20), :]` for int keys. The tuple takes the rows between them in the table's order. It does not compare the values. If the second end comes first in the table, the result is empty.
- Rows by key need a keyfield.
- A tuple of row keys needs the column part, even if it is `:`, as in `daf[('a1', 'c3'), :]`. A tuple of two standing alone is read as `[rows, cols]`.
- A list holds integers, or ranges, or strings, not a mix. `None` alone is not a selector.
- The result is a Daf. `.to_value()`, `.to_list()` or `.to_dict()` give plain values, or set `daf.retmode = 'val'`.
- The same selectors set values: `daf[rows, cols] = value`.
<!-- end of indexing reference -->

### Square brackets

`daf[rows, cols]` selects rows and columns. `daf[rows]` selects rows with all their columns. Each
part can be a position, a slice, a range, a list, a key or a column name. The result is a
new Daf.

```pycon
>>> from daffodil.daf import Daf
>>> daf = Daf(cols=['id', 'a', 'b', 'c'], keyfield='id',
...          lol=[['r1', 1, 2, 3], ['r2', 4, 5, 6], ['r3', 7, 8, 9], ['r4', 10, 11, 12]])
>>> daf[1].lol
[['r2', 4, 5, 6]]
>>> daf[-1].lol
[['r4', 10, 11, 12]]
>>> daf[:2].lol
[['r1', 1, 2, 3], ['r2', 4, 5, 6]]
>>> daf[-2:].lol
[['r3', 7, 8, 9], ['r4', 10, 11, 12]]
>>> daf[[0, 3], 'a'].to_list()
[1, 10]
>>> daf[:, ['c', 'a']].lol[:2]
[[3, 1], [6, 4]]
```

Keys and names:

```pycon
>>> daf['r3'].lol
[['r3', 7, 8, 9]]
>>> daf[['r4', 'r1'], 'b'].to_list()
[11, 2]
```

A tuple of two keys, or of two names, is an inclusive range. `None` stands for the start or
the end. A range of row keys needs the column part, even if it is `:`, because a tuple of two
items standing alone is read as `[rows, cols]`.

The range follows the order of the rows, not the order of the keys. It starts at the row of
the first key and ends at the row of the second. The ends of a tuple are always keys, so a
tuple of ints works for int keys.

```pycon
>>> daf[('r2', 'r3'), :].lol
[['r2', 4, 5, 6], ['r3', 7, 8, 9]]
>>> daf[(None, 'r2'), 'id'].to_list()
['r1', 'r2']
>>> daf[('r3', None), 'id'].to_list()
['r3', 'r4']
>>> daf[0, ('a', 'b')].to_list()
[1, 2]
```

Here the keys are not in sorted order. The range from b to a takes b, z and a, the rows
between them in the table. The range from a to b is empty, as a comes after b in the table:

```pycon
>>> unsorted = Daf(cols=['id', 'v'], keyfield='id', lol=[['m', 1], ['b', 2], ['z', 3], ['a', 4]])
>>> unsorted[('b', 'a'), :].col('id')
['b', 'z', 'a']
>>> unsorted[('a', 'b'), :].col('id')
[]
>>> by_number = Daf(cols=['n', 'v'], keyfield='n', lol=[[10, 'x'], [20, 'y'], [30, 'z']])
>>> by_number[(20, 30), :].col('n')
[20, 30]
```

Outside a tuple, an integer in `[]` is always a position. To reach a key or a column name
that is an integer, use `select_krows()` and `select_kcols()`.

### Getting values out

A selection is a Daf. Turn it into values with `to_value()` for one cell, `to_list()` for one
row or column, and `to_dict()` for one row. Or read the values directly, with the methods
below, which build no table and are faster.

```pycon
>>> daf[2, 'b'].to_value()
8
>>> daf.col('b')
[2, 5, 8, 11]
>>> daf.select_record('r2')
{'id': 'r2', 'a': 4, 'b': 5, 'c': 6}
>>> daf.iloc(0)
{'id': 'r1', 'a': 1, 'b': 2, 'c': 3}
```

With `retmode = 'val'`, a selection of one cell, one row or one column returns the value or a
list instead of a Daf:

```pycon
>>> daf.retmode = 'val'
>>> daf[2, 'b'], daf['r1'], daf[:, 'a']
(8, ['r1', 1, 2, 3], [1, 4, 7, 10])
>>> daf.retmode = 'obj'
```

### Selecting by condition

`select_where()` keeps the rows for which a function is true. The function gets each row as a
KeyedList, which reads like a dict:

```pycon
>>> daf.select_where(lambda row: row['a'] > 5).col('id')
['r3', 'r4']
```

For a test of equality, `select_by_dict()` is much faster, as it compares the cells without
calling a function for each row. A list of dicts means any of them:

```pycon
>>> daf.select_by_dict({'b': 5}).col('id')
['r2']
>>> daf.select_by_dict([{'b': 5}, {'b': 11}]).col('id')
['r2', 'r4']
```

`inverse=True` keeps the rows that do not match, in `select_by_dict()`, `select_irows()` and
`select_krows()`:

```pycon
>>> daf.select_krows(['r1', 'r4'], inverse=True).col('id')
['r2', 'r3']
```

### Setting values

The same selectors set values. A single value fills the selection. A list fills it in order.
A dict given for a row is placed by column name, and a column it lacks becomes NULL.

```pycon
>>> daf[0, 'a'] = 100
>>> daf[1] = ['r2', 40, 50, 60]
>>> daf[:, 'c'] = 0
>>> daf.lol[:2]
[['r1', 100, 2, 0], ['r2', 40, 50, 0]]
```

### Do selections copy the rows?

A selection of rows is a new Daf with a new list of rows, but each row in it is the same list
as in the original. That is why it is fast. It also means that changing a cell in a selection
changes the original:

```pycon
>>> sel = daf.select_where(lambda row: row['id'] == 'r1')
>>> sel[0, 'b'] = 99
>>> daf.select_record('r1')['b']
99
```

| Selection | The rows of the result |
|---|---|
| `daf[rows]`, `select_irows`, `select_krows`, `select_records_daf` | shared |
| `select_where`, `split_where`, `select_by_dict` | shared |
| `groupby`, `multi_groupby`, `groupby_cols`, `group_where`, `copy()` | shared |
| a selection of columns: `daf[:, cols]`, `select_cols`, `select_kcols`, `select_icols` | new |
| `groupby`, `multi_groupby` and `groupby_cols` with only some columns kept | new |

To change a selection without changing the original, copy it first with
`copy('editable')`, which gives each row its own list. Adding a column to a selection, as with
`insert_col()`, copies the shared rows first, so the original keeps its shape.

New row lists still hold the same cell values. A list or dict in a cell is one object, seen
from the original, from any selection, and from `copy('editable')`. Changing it in place
shows everywhere. Only `copy('deep')` copies it:

```pycon
>>> tagged = Daf(cols=['id', 'tags'], lol=[['a', ['x']]])
>>> cols_only = tagged[:, ['tags']]
>>> cols_only[0, 'tags'].to_value().append('y')
>>> tagged.lol
[['a', ['x', 'y']]]
>>> deep = tagged.copy('deep')
>>> deep[0, 'tags'].to_value().append('z')
>>> tagged.lol
[['a', ['x', 'y']]]
```

The same holds for `to_lod()`, `to_dict()`, the rows of a loop, and a KeyedList: each is new,
but its values are the cells of the table.

### Changing a key

The key index maps each key to its row. It is built at the first lookup, and a write with
`[]` does not update it. So after you change a key cell, or replace a row, the old key still
finds the row, and the new key finds nothing. Call `set_keyfield()` with the same keyfield to
have the index rebuilt:

```pycon
>>> stock = Daf(cols=['id', 'qty'], keyfield='id', lol=[['a1', 3], ['b2', 10]])
>>> stock.select_record('a1')
{'id': 'a1', 'qty': 3}
>>> stock[0, 'id'] = 'z9'
>>> stock.select_record('z9')
{}
>>> _ = stock.set_keyfield(stock.keyfield)
>>> stock.select_record('z9')
{'id': 'z9', 'qty': 3}
```

A selection that shares the row has its own index, so do the same on each Daf that looks up
keys after the change.

---

# Docs page: Guide / Columns, keys and schemas (guide/columns-keys-schemas.md)

## Columns, keys and schemas

### Column names

The column names are kept once, in the header dict `hd`, which maps each name to its
position. The rows hold only their cells.

```pycon
>>> from daffodil.daf import Daf
>>> daf = Daf(cols=['id', 'qty'], lol=[['a1', 3], ['b2', 10]])
>>> daf.hd
{'id': 0, 'qty': 1}
```

Names must be unique and hashable. A name must not contain a double underscore, `__`, which is
reserved for the encoding that stores any name in SQLite. Strings are the most convenient, because `[]` reads an
integer as a position. A column named `5` is reached by name with `select_kcols()`, not with
`daf[:, 5]`.

A Daf with no names can be given spreadsheet names, or names of your own, with `set_cols()`:

```pycon
>>> raw = Daf(lol=[[1, 2], [3, 4]])
>>> raw.columns()
[]
>>> _ = raw.set_cols()
>>> raw.columns()
['A', 'B']
>>> _ = raw.set_cols(['x', 'y'])
>>> raw.col('y')
[2, 4]
```

### Column names from a CSV file

`Daf.from_csv()` takes the column names from the first line. A blank name becomes `Unnamed`
followed by its position. A repeated name gets an underscore and its position.
`from_csv_buff()` does the same for text or bytes in memory:

```pycon
>>> Daf.from_csv_buff('id,,qty,qty\n1,2,3,4\n').columns()
['id', 'Unnamed1', 'qty', 'qty_3']
```

With `noheader=True`, the first line is data, and the Daf has no names until you set them:

```pycon
>>> n = Daf.from_csv_buff('1,2\n3,4\n', noheader=True)
>>> n.columns(), n.lol
([], [['1', '2'], ['3', '4']])
```

The values from a CSV file are text. See [Types and conversion](types.md).

### Rows by position

Row positions belong to the table, not to the row. After a sort or a selection, position 0
is whatever row is first. To find a row whatever its position, use a keyfield.

### The keyfield

The keyfield is a column whose values identify the rows, like a primary key. It stays in the
table as an ordinary column. Daffodil builds a key dict, `kd`, from it when a lookup first
needs it, so a table that is only appended to never builds one.

```pycon
>>> daf = Daf(cols=['id', 'qty'], lol=[['a1', 3], ['b2', 10]], keyfield='id')
>>> daf.select_record('b2')
{'id': 'b2', 'qty': 10}
>>> daf['b2', 'qty'].to_value()
10
>>> daf.keys()
['a1', 'b2']
```

The keyfield values must be hashable, and should be unique. Several columns can form the
key together, as a tuple:

```pycon
>>> votes = Daf(cols=['precinct', 'contest', 'n'], lol=[['p1', 'mayor', 40], ['p1', 'council', 31]],
...             keyfield=('precinct', 'contest'))
>>> votes.select_record(('p1', 'council'))
{'precinct': 'p1', 'contest': 'council', 'n': 31}
```

#### A key column with repeated values

`append()` does not check the key, which keeps it fast. So data with repeated keys can be
read in. Then the key index holds each key once, and points to the last row with it:

- A lookup of the key, as `select_record()` or `daf[key]`, finds the last row.
- `keys()` gives each key once, in the order of its first row.
- `join()` matches each row with the last row of that key in the other table.

```pycon
>>> dup = Daf(cols=['id', 'qty'], lol=[['a1', 3], ['b2', 10], ['a1', 5]], keyfield='id')
>>> dup.select_record('a1'), dup.keys(), len(dup)
({'id': 'a1', 'qty': 5}, ['a1', 'b2'], 3)
```

To find repeated keys before you rely on the key:

1. Read the data with no keyfield.
2. Count the distinct values against the rows, as below.
3. Fix or remove the rows with repeated keys.
4. Then set the keyfield with `set_keyfield()`.

```pycon
>>> daf = Daf(cols=['id', 'qty'], lol=[['a1', 3], ['b2', 10], ['a1', 5]])
>>> ids = daf.col('id')
>>> len(ids), len(set(ids))
(3, 2)
>>> daf.valuecounts_for_colname('id')
{'a1': 2, 'b2': 1}
```

If no column can be the key, add one that numbers the rows with `insert_idx_col()`.

### Schemas

A schema declares the columns, their intended types and their defaults in one place. Write it
as a class with type annotations, decorated with `@schemaclass`:

```pycon
>>> from daffodil.lib.schemaclass import schemaclass
>>> @schemaclass
... class Ballot:
...     __keyfield__ = 'ballot_id'
...     ballot_id: str = ''
...     contest: str = ''
...     page: int = 0
>>> b = Daf(schema=Ballot)
>>> b.columns(), b.keyfield
(['ballot_id', 'contest', 'page'], 'ballot_id')
>>> b.dtypes
{'ballot_id': <class 'str'>, 'contest': <class 'str'>, 'page': <class 'int'>}
```

A schema fills only what the Daf does not already have: the columns, the dtypes and the
keyfield. It does not check or convert the cells.

`default_record()` makes a new record with the defaults, for the Daf's own columns:

```pycon
>>> b.default_record()
{'ballot_id': '', 'contest': '', 'page': 0}
>>> Daf(cols=['page', 'ballot_id'], schema=Ballot).default_record()
{'page': 0, 'ballot_id': ''}
```

A Daf that keeps only some of the schema's columns gets records that fit it. See
[Building tables](building-tables.md) for `default_record(astype=KeyedList)`, which is the
fast way to build rows in a loop.

A schema can also be a Daf with one row for each column, with the columns `Name`, `dtype`
and `Default`.

---

# Docs page: Guide / Types and conversion (guide/types.md)

## Types and conversion

A cell can hold any Python object: a number, text, a list, a dict, a set, or another Daf.
Nothing limits what a cell holds, and nothing is converted unless you ask.

### dtypes

`dtypes` is a dict of column name to type. It says how to convert the columns when you ask,
as when reading a CSV file. It is not a schema that checks or forces the cells, and a Daf
built with `dtypes` keeps its cells as they are.

Three steps are separate:

| Step | What it does |
| :--- | :----------- |
| set `dtypes`, or call `set_dtypes()` | declares the types. Nothing is converted |
| `apply_dtypes()` | tries to convert the cells |
| `check_schema()` | finds the cells that do not fit their dtypes, and raises |

Conversion is permissive. Text that is not a number stays text, with no error. So after
`apply_dtypes()`, an `int` column can still hold some text. Call `check_schema()` when you
need to know that every cell fits.

### Converting text from a CSV file

A CSV file gives text. `apply_dtypes()` converts the columns in place, and returns the Daf:

```pycon
>>> from daffodil.daf import Daf
>>> daf = Daf.from_csv_buff('id,qty,price,tags\na1,3,1.5,"[1, 2]"\nb2,,2.0,[]\n')
>>> daf.lol
[['a1', '3', '1.5', '[1, 2]'], ['b2', '', '2.0', '[]']]
>>> _ = daf.apply_dtypes(dtypes={'id': str, 'qty': int, 'price': float, 'tags': list})
>>> daf.lol
[['a1', 3, 1.5, [1, 2]], ['b2', '', 2.0, []]]
```

- An empty cell is NULL, the empty string, and stays NULL, except in a `bool` column. See
  Missing values below.
- A list or dict column is read back from its text, which may be JSON or PYON.
- Text that does not make an `int` or a `float` is kept as it is.
- `cols=` converts only the named columns. Converting a column costs time, so convert only
  the columns you use.

```pycon
>>> e = Daf.from_csv_buff('id,qty,note\na1,3,x\n')
>>> _ = e.apply_dtypes(dtypes={'id': str, 'qty': int, 'note': str}, cols=['qty'])
>>> e.lol
[['a1', 3, 'x']]
```

Load the file as text, then call `apply_dtypes()` for the columns you need. `from_csv()` and
`from_csv_buff()` still take `dtypes=`, and convert as they load, but that is deprecated.

### bool columns

A `bool` column gets 0 or 1. The usual words for true and false are read in lower case,
capitalized or upper case. An empty cell becomes 0, unlike in an `int` or `float` column.
Other text stays as it is. Note that `'False'` gives 0, where Python's `bool('False')` is True.

```pycon
>>> flags = Daf(cols=['ok'], lol=[[''], ['0'], ['1'], ['False'], ['True'], ['no'], ['maybe']])
>>> _ = flags.apply_dtypes(dtypes={'ok': bool})
>>> flags.col('ok')
[0, 0, 1, 0, 1, 0, 'maybe']
```

The words for 0 are `0`, `false`, `no`, `n`, `f` and `off`. The words for 1 are `1`, `true`,
`yes`, `y`, `t` and `on`.

### Missing values

A missing value is NULL, the empty string `''`. It prints as nothing, so a table with missing
values stays easy to read.

- The sum methods skip it.
- The value counts include it, as a count for `''`, unless you pass `omit_nulls=True`.
- `to_pandas_df()` keeps it as `''`, so a number column with NULL becomes an object column.
  Pass `default=None` to get NaN in its place. That writes None into the Daf itself. See
  [to_pandas_df()][daffodil.lib.daf_pandas._to_pandas_df] for a route that leaves the Daf alone.
- `to_numpy()` makes one array of one dtype, so a column with NULL becomes text.

```pycon
>>> from daffodil.daf import NULL
>>> NULL == ''
True
>>> gaps = Daf(cols=['g', 'n'], lol=[['x', 1], ['', 2], ['x', '']])
>>> gaps.sum(['n'])
{'n': 3.0}
>>> gaps.valuecounts_for_colname('g'), gaps.valuecounts_for_colname('g', omit_nulls=True)
({'x': 2, '': 1}, {'x': 2})
```

### Writing a CSV file

`to_csv_file(path)` writes a CSV file, and `to_csv_buff()` returns the text. A list, dict,
set or tuple in a cell is written as PYON, the Python form of the value, and is read back by
`apply_dtypes()`:

```pycon
>>> f = Daf(cols=['id', 'tags'], lol=[['a', [1, 2]], ['b', {'k': (1, 2)}]])
>>> print(f.to_csv_buff())
id,tags
a,"[1, 2]"
b,"{'k': (1, 2)}"
```

PYON is like JSON, but it can hold sets, tuples, and dicts with keys that are not text. See
[PYON](https://github.com/raylutz/pyon/blob/main/README.md).

### To and from other forms

| To | From |
|---|---|
| `to_lod()`: a list of dicts | `Daf.from_lod()` |
| `to_dod()`: a dict of dicts, by key | `Daf.from_dod()` |
| `to_cols_dol()`: a dict of column lists | `Daf.from_cols_dol()` |
| `to_json()` | `Daf.from_json()` |
| `to_md()`: a Markdown table | `Daf.from_md()` |
| `to_pandas_df()` | `Daf.from_pandas_df()` |
| `to_numpy()` | `Daf.from_numpy()` |
| `to_csv_file()`, `to_csv_buff()` | `Daf.from_csv()`, `Daf.from_csv_buff()`, `Daf.from_excel_buff()`, `Daf.from_ods()` |

---

# Docs page: Guide / Apply, reduce and formulas (guide/apply-reduce-formulas.md)

## Apply, reduce and formulas

### Rows, not columns

Daffodil keeps a table as rows, where pandas and Polars keep columns. Appending rows and
going through them one at a time suit that storage. Selecting whole rows reuses their row
lists, and copies no cells. Inserting or deleting a row in the middle moves the rows after
it in the outer list, so it costs more in a large table.

Working with columns costs more, as each column is spread across all the rows. Most of the
time you do not need to: rather than dropping the columns you do not want, name the ones you
do. Give `cols=` to `reduce()`. In an `apply()` function, read only the fields you need. To
show some columns, select them before `to_md()`. For heavy math over whole numeric columns,
hand them to NumPy or pandas, with `to_numpy()` or `to_pandas_df()`.

### apply(): a new table, row by row

`apply()` calls a function on each row, and makes a new table of what it returns. The row is a
dict, and the function returns a dict:

```pycon
>>> from daffodil.daf import Daf
>>> daf = Daf(cols=['g', 'a', 'b'], lol=[['x', 1, 2], ['y', 3, ''], ['x', 5, 6]])
>>> doubled = daf.apply(lambda row: {'g': row['g'], 'a2': row['a'] * 2})
>>> doubled.columns(), doubled.lol
(['g', 'a2'], [['x', 2], ['y', 6], ['x', 10]])
```

`apply_in_place()` changes the rows of the table instead of making a new one.

### reduce(): one record from many rows

`reduce()` folds the rows into one record with a function. `Daf.sum_da` adds up each column:

```pycon
>>> daf.reduce(Daf.sum_da, cols=['a', 'b'])
{'g': '', 'a': 9, 'b': 8}
>>> daf.sum(['a', 'b'])
{'a': 9.0, 'b': 8.0}
```

NULL cells are skipped, so the missing `b` of the second row adds nothing.

`sum()` and `reduce(Daf.sum_da)` differ when a column holds text. `sum()` converts text
that is a number, such as `'2'`, and raises `ValueError` for other text. `Daf.sum_da` skips
every value that is not a number, and never raises:

```pycon
>>> mixed = Daf(cols=['n'], lol=[[1], ['2'], [''], [3.5]])
>>> mixed.sum(['n']), mixed.reduce(Daf.sum_da, cols=['n'])
({'n': 6.5}, {'n': 4.5})
```

#### Your own reduction

The function gets each row, the result so far, and the columns. It returns the new result.
The row is a KeyedList, which reads like a dict. The result has one value for each column,
and starts at 0 unless you give `initial_da`. This one counts the cells that are not NULL:

```pycon
>>> from daffodil.daf import NULL
>>> def count_filled(row, result, cols):
...     for col in cols:
...         if row[col] is not NULL:
...             result[col] += 1
...     return result
>>> daf.reduce(count_filled, cols=['a', 'b'])
{'g': '', 'a': 3, 'b': 2}
```

A key of the result that is not a column is dropped. For a result of another shape, write a
loop over the rows.

### Groups

`groupby()` makes a Daf for each value of a column. `groupsum_daf()` adds up the other columns
of each group. `groupby_cols_reduce()` groups by several columns and reduces each group with
any function.

```pycon
>>> {key: group.lol for key, group in daf.groupby('g').items()}
{'x': [['x', 1, 2], ['x', 5, 6]], 'y': [['y', 3, '']]}
>>> daf.groupsum_daf('g', reduce_cols=['a', 'b']).lol
[['x', 6, 8], ['y', 3, 0]]
>>> daf.groupby_cols_reduce(['g'], Daf.sum_da, reduce_cols=['a']).lol
[['x', 6], ['y', 3]]
>>> daf.valuecounts_for_colname('g')
{'x': 2, 'y': 1}
```

### Data in many files

A table can describe a set of files, one row for each chunk of the data. `manifest_apply()`
applies a function to every row of every chunk, and `manifest_reduce()` reduces them all to
one record. Each chunk is read in turn, so the whole data never has to fit in memory, and the
chunks can be shared out to run in parallel.

### Spreadsheet-like formulas

`apply_formulas()` fills cells from a second Daf of the same shape that holds formulas, as a
spreadsheet does. A formula is Python, evaluated for its cell. An empty formula leaves its
cell alone. In a formula:

- `$d` is the Daf.
- `$r` is the row of the cell, and `$c` its column.

So a formula is absolute unless it uses `$r` or `$c`, the reverse of a spreadsheet. The
formulas are evaluated again until no value changes. If the values still change after 100
passes, it raises `RuntimeError`. This is a limit on the passes, not a check for circular
formulas. A circle that settles raises nothing.

The formulas run with Python's `eval()`, so use only formulas you trust. If a formula raises,
the cells already changed stay changed.

This sums each row into the last column, and each column into the last row:

```pycon
>>> data = Daf(cols=['A', 'B', 'C'], lol=[[1, 2, 0], [4, 5, 0], [7, 8, 0], [0, 0, 0]])
>>> row_sum, col_sum = "sum($d[$r, :$c])", "sum($d[:$r, $c])"
>>> formulas = Daf(cols=['A', 'B', 'C'], lol=[['', '', row_sum],
...                                           ['', '', row_sum],
...                                           ['', '', row_sum],
...                                           [col_sum, col_sum, col_sum]])
>>> _ = data.apply_formulas(formulas)
>>> data.lol
[[1, 2, 3], [4, 5, 9], [7, 8, 15], [12, 15, 27]]
```

The formulas themselves are a table, so they can be written in a spreadsheet, saved as CSV, and
read with `Daf.from_csv()`.

---

# Docs page: Guide / Joins (guide/joins.md)

## Joins

`join()` combines two tables by their keys. Each table needs a keyfield of one column. A
composite keyfield is not supported. A row of one table matches the row of the other with the
same key.

The keys should be unique in both tables. The join is one to one, not many to many as in SQL.
If a key repeats in the other table, each match gets its last row. If it repeats in the first
table, each of those rows is in the result.

```pycon
>>> from daffodil.daf import Daf
>>> stock = Daf(cols=['id', 'name', 'qty'], keyfield='id', name='stock',
...             lol=[['a1', 'pen', 3], ['b2', 'ink', 10], ['c3', 'pad', 7]])
>>> prices = Daf(cols=['id', 'price', 'qty'], keyfield='id', name='prices',
...              lol=[['a1', 1.5, 100], ['b2', 4.0, 50], ['d4', 2.0, 5]])
```

### Join types

`how=` chooses which rows are kept:

- `inner`, the default: only the keys in both tables.
- `left`: every row of the first table, with the matching rows of the second.
- `right`: every row of the second table, with the matching rows of the first.
- `outer`: every row of both.

A side with no matching row gets NULL. Pass `fill=None` to get `None` instead. The rows of the
first table come first, in its order. In a right or outer join, the rows that only the second
table has come after them.

```pycon
>>> stock.join(prices).col('id')
['a1', 'b2']
>>> stock.join(prices, how='outer').lol
[['a1', 'pen', 3, 1.5, 100], ['b2', 'ink', 10, 4.0, 50], ['c3', 'pad', 7, '', ''], ['d4', '', '', 2.0, 5]]
```

### Columns with the same name

The key column appears once. Another column that both tables have is kept twice, with the
table's name as a suffix, or `_daf1` and `_daf2` for tables with no name:

```pycon
>>> stock.join(prices).columns()
['id', 'name', 'qty_stock', 'price', 'qty_prices']
>>> one = Daf(cols=['id', 'q'], keyfield='id', lol=[['a', 1]])
>>> two = Daf(cols=['id', 'q'], keyfield='id', lol=[['a', 2]])
>>> one.join(two).columns()
['id', 'q_daf1', 'q_daf2']
```

List a column in `shared_fields` to keep it once instead. Its value comes from the first table:

```pycon
>>> joined = stock.join(prices, shared_fields=['qty'])
>>> joined.columns(), joined.lol[0]
(['id', 'name', 'qty', 'price'], ['a1', 'pen', 3, 1.5])
```

A row that only the second table has gets NULL in a shared column, so the second table's value
is lost. Here d4 had a qty of 5. This is a known fault. Leave out `shared_fields` when such rows
matter.

```pycon
>>> stock.join(prices, how='outer', shared_fields=['qty']).lol[-1]
['d4', '', '', 2.0]
```

The key of such a row is lost too when the two keyfields have different names. Name them the
same for a right or outer join.

For other names, give a `custom_translator_daf`, which maps each column of each table to its
name in the result. See [join()][daffodil.daf.Daf.join] in the API.

---

# Docs page: Guide / Tables larger than memory (guide/sqldaf.md)

## Tables larger than memory

A [SqlDaf][daffodil.sqldaf.SqlDaf] stores a table in a SQLite database file, so its rows stay
out of Python memory. It supports tables larger than the memory available, and many
operations run directly in SQLite. It has many of the Daf methods, under the same names. Some
of them behave differently, because the data is in a database. [Differences from
Daf](#differences-from-daf) lists them.

Prefer an ordinary Daf when the table, and any copies you make of it, fit comfortably in
memory. A Daf is faster. Use SqlDaf when memory becomes the limit.
[Performance](#performance) gives timings.

### Making and opening a table

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
>>> len(items_sdaf)
3
```

A table in the file opens again by its name alone. Its keyfield and dtypes come back with it,
from a record that SqlDaf keeps in the same file. `from_daf()` makes a table from a Daf.

```pycon
>>> SqlDaf('items').keyfield
'id'
```

### Three kinds of result

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

### Loading a CSV file

`from_csv()` reads a file one row at a time, so the file is never all in memory. The table is
named after the file. CSV values are loaded as text, as `Daf.from_csv()` reads them without
dtypes.

Convert the number columns before filtering, sorting or adding them up. Until then, '12' is
text, not the number 12, and a condition such as `n < 5` finds nothing. `apply_dtypes()`
converts a column in the table itself. As with a Daf, name the columns to convert with
`cols=`, or give a dtype for every column.

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

The dtypes are kept with the table, so they come back when it is opened again. A value that
does not convert, such as 'abc' in an int column, stays as it is. So a converted column can
still hold some text.

### Selections

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

`select_where_sql()` is the fastest way to select, since SQLite does all the work.
`select_where()` takes a Python function of a row, as on a Daf, and calls it for every row.

#### One row

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

### Changing rows and values

Rows are added and values set as on a Daf. Each change is written to the file at once.

```pycon
>>> _ = items_sdaf.append({'id': 'a4', 'grp': 'y', 'qty': 2})
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
>>> _ = items_sdaf.update_record_krow('a3', {'grp': 'z'})
>>> items_sdaf.select_record('a3')
{'id': 'a3', 'grp': 'z', 'qty': 7}
```

#### Many small writes

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

#### Removing rows

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

### Dict cells and reductions

A dict cell stays as the text that a CSV file holds, as written by Python. Nothing converts
it when it is stored. A reduction by `sparse_row` reads the text of each cell as it goes.

```pycon
>>> votes_sdaf = SqlDaf('votes', cols=['ballot_id', 'precinct', 'votes'], keyfield='ballot_id',
...                     lol=[['b1', 'p1', "{'c1': 1}"], ['b2', 'p1', "{'c1': 1, 'c2': 1}"],
...                          ['b3', 'p2', "{'c2': 1}"]])
>>> votes_sdaf.reduce(Daf.sum_da, by='sparse_row', indirect_col='votes')
{'c1': 2, 'c2': 2}
```

A reduction reads the rows in batches, so the table is never all in memory. Memory use still
depends on the result your function builds.

### Groups

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

### Joins

`join()` joins two tables of the same file on their keyfields, in SQL. The columns and the
rows come as `Daf.join()` gives them. A cell with no match is Python NULL.

```pycon
>>> names_sdaf = SqlDaf('names', cols=['ballot_id', 'voter'], keyfield='ballot_id',
...                     lol=[['b1', 'Ann'], ['b2', 'Bo']])
>>> marks_sdaf.join(names_sdaf, how='left').to_daf().lol
[['b1', 3, 'Ann'], ['b2', 12, 'Bo'], ['b3', 7, '']]
```

### Keeping tables

A sort changes which table the SqlDaf refers to. It makes a sorted TEMP table, and the table in
the file keeps its order. `save()` keeps the sorted copy under a name not in use. `close()`
removes TEMP tables, so save what you want to keep first.

```pycon
>>> _ = items_sdaf.sort_by_colname('qty', reverse=True)
>>> items_sdaf.col('id')
['a2', 'a4', 'a1', 'a3']
>>> _ = items_sdaf.save('items_by_qty')
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

### Missing values

There are two kinds of NULL. Python NULL is daffodil's missing value: the empty string '',
which daffodil names NULL, as in `from daffodil.daf import NULL`. SQL NULL is the database's.
SqlDaf writes Python NULL as SQL NULL, and reads SQL NULL back as Python NULL. A Python None
is stored as SQL NULL too, so it also reads back as Python NULL.

The methods that group, count or select by value treat SQL NULL and the text '' as one value,
as a Daf does. In Python, test a value with `val is NULL`. In your own SQL, test for both, as
`(col IS NULL OR col = '')`.

### Methods a SqlDaf does not have

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

### Differences from Daf

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
- **Methods.** Methods not written for SQL raise NotImplementedError.

### Performance

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

---

# Docs page: Guide / Markdown reports (guide/markdown.md)

## Markdown reports

Daffodil shows a table as Markdown. The same text reads well in a terminal, and drops into a
Markdown report or a page like this one.

```pycon
>>> from daffodil.daf import Daf
>>> daf = Daf(cols=['id', 'item', 'qty'], keyfield='id', name='stock',
...          lol=[[f'r{i}', 'x' * i, i] for i in range(1, 9)])
>>> print(daf.to_md(max_rows=4))
| id  |   item   | qty |
| --: | -------: | --: |
|  r1 |        x |   1 |
|  r2 |       xx |   2 |
| ... |      ... | ... |
|  r7 |  xxxxxxx |   7 |
|  r8 | xxxxxxxx |   8 |
```

`print(daf)` shows the table, followed by a line that describes it.

### Options of to_md()

- `max_rows`, `max_cols`: keep the first and last rows or columns, and put `...` between.
- `just`: the alignment of each column, one character each: `<` left, `^` center, `>` right.
- `shorten_text`, `max_text_len`: shorten long text, keeping its two ends.
- `smart_fmt`: fewer digits after the decimal point, in a column of numbers.
- `include_summary`: add the line that describes the table, for `from_md()`.

```pycon
>>> print(Daf(cols=['n', 't'], lol=[[1.23456, 'a' * 100]]).to_md(smart_fmt=True, max_text_len=20))
|  n  |          t           |
| --: | -------------------: |
| 1.2 | aaaaaaaaa..aaaaaaaaa |
>>> print(Daf(cols=['a', 'b'], lol=[[1, 'x']]).to_md(just='<>'))
| a  | b  |
| :- | -: |
| 1  |  x |
```

### Reading Markdown back

Markdown is a report format first. A table can be read back, but only some tables come back
whole. With `include_summary=True`, the table ends with a line that gives its keyfield and
name. `Daf.from_md()` reads the table back, with them.

For the most faithful copy, pass `shorten_text=False`, leave `smart_fmt`, `max_rows` and
`max_cols` at their defaults, and include the summary. Even then:

- The cells come back as text, as from a CSV file, and the dtypes are not kept. See
  [Types and conversion](types.md).
- A newline in a cell comes back as `<br>`.
- A `|` in a cell ends the table at the row before it, with no error. This is a known fault.
- A composite keyfield is not read back, and raises `KeyError`. This is a known fault too.

For data that must come back exactly, use `to_json()` and `from_json()`, which keep numbers,
text and the keyfield. A CSV file keeps the text exactly, and `apply_dtypes()` restores the types.

```pycon
>>> text = daf.to_md(include_summary=True, shorten_text=False)
>>> text.splitlines()[-1]
"%% daf rows=8; cols=3; keyfield='id'; name='stock'"
>>> back = Daf.from_md(text)
>>> back.columns(), back.keyfield, back.name
(['id', 'item', 'qty'], 'id', 'stock')
>>> back.lol[1]
['r2', 'xx', '2']
```

`Daf.dodaf_to_md()` writes several named tables to one Markdown text, and
`Daf.dodaf_from_md()` reads them back as a dict of Dafs.

---

# Docs page: Guide / Coming from pandas (guide/from-pandas.md)

## Coming from pandas

Daffodil keeps rows where pandas keeps columns. So building, selecting and changing rows is
cheap, and math over a whole numeric column is better done in pandas or NumPy. See the
[Home page](../index.md) for when each fits, with measurements.

Some differences to know first:

- `daf[x]` selects rows, where `df[x]` selects a column. A column is `daf[:, 'name']`, or as a
  list, `daf.col('name')`.
- An integer in `[]` is a position, as with `iloc`. Only the ends of a tuple range are keys. A key is a value of the keyfield
  column, which stays in the table, unlike the pandas index.
- A selection of rows shares the rows with the original. Changing a cell in it changes the
  original. See [Selecting and indexing](selecting.md).
- A cell can hold anything, and columns are not converted unless you ask. A missing value is
  `''`, not NaN.
- A table prints as Markdown.

### The same task in each

Each line is the same task, not the same expression. The result type, and whether the table
is changed, can differ. The notes say how.

| pandas | Daffodil | Notes |
|---|---|---|
| `pd.DataFrame(lod)` | `Daf.from_lod(lod)` | |
| `pd.read_csv(path)` | `Daf.from_csv(path)` | the cells are text until `apply_dtypes()` |
| `df.to_csv(path)` | `daf.to_csv_file(path)` | |
| `df.to_dict('records')` | `daf.to_lod()` | |
| `df.shape`, `len(df)` | `daf.shape()`, `len(daf)` | |
| `df.empty` | `not daf` | |
| `df.columns` | `daf.columns()` | |
| `df.dtypes` | `daf.dtypes` | describes, does not enforce |
| `df.astype(...)` | `daf.apply_dtypes(dtypes=...)` | in place |
| `df.head(n)`, `df.tail(n)` | `daf[:n]`, `daf[-n:]` | |
| `df.iloc[n]` | `daf[n]` | a Daf. `.to_dict()` for a dict |
| `df.loc[k]` | `daf.select_record(k)` | a dict. Needs a keyfield. `daf[k]` also works, but there an int `k` is a position |
| `df.set_index('id')` | `daf.set_keyfield('id')` | the column stays in the table |
| `df['c']` | `daf[:, 'c']`, or `daf.col('c')` for a list | |
| `df[['a', 'b']]` | `daf[:, ['a', 'b']]` | |
| `df[df.qty > 5]` | `daf.select_where(lambda row: row['qty'] > 5)` | |
| `df[df.c.isin(vals)]` | `daf.select_by_dict([{'c': v} for v in vals])` | one pass, no list of bools |
| `df.drop(columns=[...])` | `daf.select_cols(exclude_cols=[...])` | |
| `df[~df.id.isin(keys)]` | `daf.select_krows(keys, inverse=True)` | |
| `df.loc[len(df)] = row`, `pd.concat` | `daf.append(row)` | cheap |
| `pd.concat([df1, df2])` | `d1.append(d2)` | |
| `df.insert(i, 'c', vals)` | `daf.insert_col('c', vals, icol=i)` | |
| (none) | `daf.insert_irow(i, row)` | insert a row anywhere |
| `df['c'] = vals` | `daf[:, 'c'] = vals` | |
| `df.rename(columns=...)` | `daf.rename_cols({...})` | |
| `df.sort_values('c')` | `daf.copy().sort_by_colname('c')` | `sort_by_colname()` sorts the Daf itself. Copy first to keep the order |
| `df.T` | `daf.transpose()` | a new Daf |
| `df.apply(f, axis=1)` | `daf.apply(f)` | the row is a dict, and `f` returns one |
| `df.agg(...)` | `daf.reduce(f)` | |
| `df.groupby('g')` | `daf.groupby('g')` | a KeyedList of Dafs, used as a dict |
| `df.groupby('g')[cols].sum()` | `daf.groupby_cols_reduce(['g'], Daf.sum_da, reduce_cols=cols)` | name the columns to add |
| `df.groupby([...])[cols].agg(f)` | `daf.groupby_cols_reduce([...], f, reduce_cols=cols)` | |
| `df.value_counts('c')` | `daf.valuecounts_for_colname('c')` | a dict |
| `df.merge(other, on='id')` | `daf.join(other)` | on the keyfields, one to one. See [Joins](joins.md) |
| `df.replace(old, new)` | `daf.replace_in_columns(cols=None, find_values=[old], replacement=new)` | whole cells equal to `old`. `find_replace()` replaces a whole cell that matches a regex |
| `df.to_markdown()` | `daf.to_md()` | |
| `df.to_numpy()` | `daf.to_numpy()` | |

### A short example

```pycon
>>> from daffodil.daf import Daf
>>> daf = Daf.from_lod([{'id': 'a1', 'g': 'x', 'qty': 3}, {'id': 'b2', 'g': 'y', 'qty': 10},
...                    {'id': 'c3', 'g': 'x', 'qty': 7}], keyfield='id')
>>> daf[:2].col('id')
['a1', 'b2']
>>> daf.select_where(lambda row: row['qty'] > 5).col('id')
['b2', 'c3']
>>> daf.groupby_cols_reduce(['g'], Daf.sum_da, reduce_cols=['qty']).lol
[['x', 10], ['y', 10]]
>>> daf.sort_by_colname('qty', reverse=True).col('id')
['b2', 'c3', 'a1']
```

---

# Docs page: Guide / Performance (guide/performance.md)

## Performance

These measurements compare building a table row by row in pandas and in Daffodil. They hold
for the workloads, values and versions given here. Other data can give other results. For
SqlDaf against Daf, see [Tables larger than memory](sqldaf.md#performance).

### Building a table row by row

pandas has no cheap way to add a row. pandas removed `DataFrame.append()` in version 2.0.
What is left copies or reallocates the frame on each row. Adding 5,000 rows one at a time
took 2.4 s with `pd.concat()`. So the usual pattern is to collect the rows in a list of
dicts, then convert it all at once with `pd.DataFrame()`. Daffodil needs no conversion step.
Collect the rows in a list and wrap it, or append each row to the Daf.

The table below builds 1,000 rows of 1,000 columns: one string column and 999 int columns.
This is the shape of the older benchmarks. The ints run from 0 to 1,000,002, so each one is
its own Python object, as in most real data.

| How the table is built | Build | Convert | Total | Peak memory | Kept memory |
|---|--:|--:|--:|--:|--:|
| pandas: append a dict per row, then `pd.DataFrame()` | 223 ms | 320 ms | 543 ms | 89 MB | 8 MB |
| pandas: append a list per row, then `pd.DataFrame()` | 169 ms | 310 ms | 479 ms | 60 MB | 8 MB |
| pandas: append to a list per column, then `pd.DataFrame()` | 257 ms | 397 ms | 654 ms | 53 MB | 8 MB |
| Daffodil: `append()` each row to the Daf | 173 ms | 0 ms | 173 ms | 36 MB | 36 MB |
| Daffodil: append a list per row, then `Daf(lol=...)` | 173 ms | 0 ms | 173 ms | 36 MB | 36 MB |

Each case gets the rows one at a time, in a loop. The pandas cases must first collect the
rows in Python, then convert them. Daffodil appends each row to the table itself.

What the numbers show:

- **No conversion.** Appending each row to a Daf took a third of the time of the pandas
  patterns, or less, for this shape. That is as fast as collecting the rows in a plain list.
- **Lower peak memory.** pandas holds the collected rows and the new frame at once. Its peak
  was 53 to 89 MB here, against 36 MB for Daffodil.
- **Kept memory depends on the values.** A string cost the same in both, as pandas kept the
  string column as Python objects too. Small ints also cost the same, since Python shares
  one object for each value from -5 to 256. With ints from 0 to 99, both kept 8 MB here, and
  with every column text, both kept 65 MB. Large ints and floats are where pandas keeps
  less. It stores them as 8-byte numbers, while each one in a Daf is a Python object of 24 to
  32 bytes, plus its 8-byte place in the row. That is why the table above shows 36 MB
  against 8 MB. A frame can still lose its 8-byte columns. Built from a 2-D NumPy array of
  mixed rows, or transposed with `df.T` when its columns have different dtypes, its columns
  become Python objects, and the frame is as large as the Daf or larger.
- **The shape matters.** A tall, narrow table has more rows to append and fewer cells per
  row to convert. For 200,000 rows of 10 columns, `append()` took 0.9 s, against 1.1 to
  1.5 s for the pandas patterns. Collecting the rows in a list and wrapping it took 0.7 s.
  The difference is the cost of each call to `append()`, about 1 µs.
- **NumPy values make pandas slower.** Values often come out of NumPy or pandas as NumPy
  scalars rather than Python ints. For 1,000 dicts of 1,000 such values, `pd.DataFrame()`
  took 1.75 s, against 0.27 s for the same values as Python ints. `Daf.from_lod()` took
  0.11 s either way. pandas 1.5.3, 2.1.4 and 2.3.3 all gave the same times.

The conversion is paid again every time data goes into pandas and back out. Turning the
1,000 by 1,000 frame back into a list of lists took another 101 ms. Daffodil started as the fix
for exactly that. In its author's programs, moving data in and out of pandas was among the
slowest steps.

### Selections

Selecting rows builds a new list that points to the same
row objects. No data is copied. Selecting 100,000 of 200,000 rows took 10 ms. Change a cell
through the selection, and the original table sees it. A copy is made only when you ask for
one, at the level you choose. See [copy()][daffodil.daf.Daf.copy].

### How these were measured

Times are the best of three runs on Python 3.10 and pandas 2.3.3, on 2026-10-06 and
2026-10-07. The script is `notes/scripts/bench_build_rows.py` in the repository. Set
`BENCH_ROWS=1000` and `BENCH_INTS=999` for the shape above. Older benchmarks, with more operations, are in
`docs/daf_benchmarks.md` in the repository.

---

# Docs page: Guide / Profiling mode (profiling.md)

## Profiling mode

The profiling mode shows how a program uses Daf: how big its tables get, what is done
with them, and which lines of the program do it. Use it to decide where Daffodil needs to be
faster, or where a program uses it in a costly way.

### Turning it on

Set an environment variable. The program needs no change.

```bash
DAFFODIL_PROFILE=1 DAFFODIL_PROFILE_STAGE=tabulate python my_program.py
```

When the program ends, the profile is written to `daffodil_profile_<stage>_<host>_<pid>.md`
in `DAFFODIL_PROFILE_DIR`, the current directory by default. The report is printed to
stderr. `DAFFODIL_PROFILE_FILE` also writes the report to a file.

Or turn it on in code:

```python
from daffodil.lib import daf_profile

daf_profile.start(stage='tabulate')
run_the_work()
print(daf_profile.report())
daf_profile.stop()
```

### A pipeline of stages

A pipeline whose stages run as separate programs gets one profile file per stage. Give each
stage its name, and write the files to one directory:

```bash
export DAFFODIL_PROFILE=1 DAFFODIL_PROFILE_DIR=profile_run
DAFFODIL_PROFILE_STAGE=load     python stage_load.py
DAFFODIL_PROFILE_STAGE=tabulate python stage_tabulate.py
python -m daffodil.lib.daf_profile combine profile_run -o report.md
```

The report has a section for all the stages together, and one for each stage. `--data
combined.md` also writes the combined tables.

### The profile is Daf tables

The totals are kept as Daf tables, and a profile file is those tables in Markdown, written
with `Daf.dodaf_to_md()`. Every table has a `stage` column.

| Table | One row for each | Columns |
|---|---|---|
| `info` | process | host, pid, Python and daffodil versions, start time, seconds |
| `tables` | stage, creation line and way of making a table | tables, keyed, most rows and columns, tables in each size band |
| `ops` | stage, creation line, way of making, and method | calls on those tables |
| `methods` | stage and method | calls, seconds, rows at each call in size bands, most rows |
| `sites` | stage, call site and method | calls, seconds |

`daf_profile.tables()` returns them for the current process, `load(path)` reads a file, and
`combine(runs)` adds the tables of several runs. Counts and times are added. The most rows
and columns take the largest. `combine()` uses `groupby_cols_reduce()` to do it.

A call site is the module and line, as in `auditengine.tabulate:212`, so the same line has the
same name in every run. A script run directly is named by its file name.

### What it records

It keeps running totals, not a log of every call, so memory stays small.

- **Each table.** The line that created it, and how: `Daf()`, `from_csv`, `select_where` and
  so on. The most rows and columns it had, whether it had a keyfield, and a count of each
  method called on it. When the table is freed, this summary is added to the totals for its
  creation line. The median size is reported as the size band that holds it, since bands
  can be combined across runs and a running sample cannot.
- **Each method.** Calls, total time, and the rows of the table at each call, in size bands.
- **Each call site.** Calls and total time, by method.

Only outer calls are counted. When a Daf method calls another one inside, the inner call is
not counted. A table that is built by 10,000 calls to `append()` is one table, with
`append 10,000` in its list of operations.

### The report

The report lists the runs, then has four Markdown tables, for all stages and for each stage.

1. **Tables by the line that created them.** Tables made at the same line are grouped, so a
   loop that makes 3,000 small tables is one line, with the median and the largest size.
2. **Table sizes.** How many tables reached each size, by rows and by columns.
3. **Methods.** The methods by total time, with the rows at each call in size bands.
4. **Busiest call sites.** The lines of the program that spent the most time in Daf.

### Costs and limits

- When it is off, nothing is wrapped, and Daf runs at full speed.
- Each counted call costs about 3 µs, and each call that makes a new table about 15 µs more.
  A call that Daf makes inside another one costs about 0.4 µs. So the times of cheap calls
  are inflated. Counts and sizes are exact.
- Only calls to Daf methods are seen. A loop that reads `daf.lol` directly is not.
- A child process that ends with `os._exit()`, as multiprocessing workers do, writes nothing
  at exit. Call `daf_profile.dump()` at the end of the worker instead.
- Lambda and other hosts that freeze or kill the process are not handled yet.

### API


##### start(stage: str = '', data_dir: str | None = None, report_path: str | None = None, print_report: bool = False) -> None

Start profiling: wrap the public methods of Daf and begin counting.

Calling it again while profiling is on does nothing. The totals are kept until
`reset()`.

Args:
    cls: The class to profile. Daf if not given.
    stage: A name for this run, such as a stage of a pipeline. It fills the `stage`
        column of the tables, so the runs of several stages can be combined.
    data_dir: When the program ends, write the tables to
        `daffodil_profile_<stage>_<host>_<pid>.md` in this directory.
    report_path: When the program ends, write the report to this file. `{pid}` in the
        name is replaced with the process id.
    print_report: When the program ends, print the report to stderr.



##### stop() -> None

Stop profiling and put the original methods back. The totals are kept.



##### tables() -> Dict[str, Any]

Return the totals so far as Daf tables, keyed by 'info', 'tables', 'ops', 'methods' and
'sites'. Tables that are still alive are included, as if they were freed now.

Returns:
    A dict of Daf tables. See the module notes for the columns.



##### dump(path: str | None = None) -> str

Write the tables so far to one Markdown file, and return the path.

Args:
    path: The file. Without it, `daffodil_profile_<stage>_<host>_<pid>.md` in the
        current directory.

Returns:
    The path written.



##### load(path: str) -> Dict[str, Any]

Read the tables that `dump()` wrote.

Args:
    path: The file.

Returns:
    A dict of Daf tables, with numbers converted back from text.



##### combine(runs: List[Dict[str, Any]], by_stage: bool = True) -> Dict[str, Any]

Add up the tables of several runs.

Args:
    runs: The tables of each run, as from `tables()` or `load()`.
    by_stage: If True, keep a row for each stage. If False, add the stages together
        and set `stage` to 'all'.

Returns:
    A dict of Daf tables of the same form.



##### report(dodaf: Dict[str, Any] | None = None, path: str | None = None, top: int = 30) -> str

Return a readable report as Markdown, and write it to a file if `path` is given.

Args:
    dodaf: The tables to report, as from `tables()`, `load()` or `combine()`. Without
        it, the totals of this process so far.
    path: Write the report to this file too. `{pid}` is replaced with the process id.
    top: The most lines to show in each section.

Returns:
    The report as Markdown text.



##### reset() -> None

Clear all totals. Profiling stays on or off as it was.



##### is_active() -> bool

True while profiling is on.

---

# Docs page: Daf API (api/daf/index.md)

## Daf overview

![The Daffodil data model: lol, hd, kd and dtypes](../../images/data_model.svg)


##### Daf

A table of data, stored as a list of rows.

Daf is a small, fast, pure Python table. Use it to read, reshape and write
2-D data without the weight of pandas. It is not meant for heavy numeric work.

The rows are a list of lists named `lol`. A row holds only values. The column
names are kept once, in a header dict named `hd` that maps each name to its
position. That makes rows cheap to build, append and copy.

A Daf may have a keyfield. This is a column, or a tuple of columns, whose
values name the rows. The key index is built the first time a key is needed.
Methods that add, remove or reorder rows keep it up to date, or mark it to be rebuilt.
Changing a key value inside a row does not. See `__setitem__()`.

A missing value is `NULL`, which is the empty string. It prints as nothing.

A cell can hold any Python object, such as a number, text, a list, a dict, a set or
an object of your own. Nothing limits what a cell can hold. The `dtypes` are not a
schema that checks or forces the cells. They say how to convert columns, as when
`apply_dtypes()` turns the text of a CSV file into numbers. A Daf built with `dtypes`
keeps its cells as they are.

Rows are returned as dicts or as [KeyedList][daffodil.keyedlist.KeyedList]
objects. See [retmode][daffodil.daf.Daf.retmode] and
[itermode][daffodil.daf.Daf.itermode].

Examples:
    >>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'], keyfield='id')
    >>> d.columns()
    ['id', 'v']
    >>> d.select_record(2)
    {'id': 2, 'v': 'b'}
    >>> d.shape()
    (2, 2)
    >>> e = Daf(lol=[[[1, 2], {'k': 1}]], cols=['a', 'b'], dtypes={'a': int, 'b': str})
    >>> e.iloc(0)
    {'a': [1, 2], 'b': {'k': 1}}


### Names in the signatures

The signatures use short names for types. Each is `T_` and then letters for the type, read
from the outside in. They are defined in `daffodil.lib.daf_types`.

| Letter | Means |
| :----- | :---- |
| `l` | a list |
| `d` | a dict with str keys |
| `id` | a dict with int keys |
| `t` | a tuple |
| `s`, `i`, `f`, `b` | str, int, float, bool |
| `a` | any value |
| `o` | "of", between two parts |

So `T_ls` is a list of str, `T_da` is a dict of any values, `T_lola` is a list of lists of any
values, the rows of a Daf, and `T_dols` is a dict of lists of str. A few names do not follow
the scheme:

| Name | Means |
| :--- | :---- |
| `T_cs` | column names: a list of str, the keys of a dict, or a dict |
| `T_ma` | a row as a dict or a [KeyedList][daffodil.keyedlist.KeyedList] |
| `T_klodaf` | a KeyedList of Dafs, one for each group, as `groupby()` gives |
| `T_dtype_dict` | a dict of column name to type, as `dtypes` |
| `T_lor` | a list of ranges |

A default of `_MISSING` means that no value was given. It lets `None` be a real value that you
pass. For example, `to_value()` with no `default` raises for an empty selection, and with
`default=None` it returns None.

### Constructor


##### __init__(lol: T_lola | None = None, hd: T_di | None = None, kd: Dict[str | int, int] | None = None, cols: T_cs | None = None, dtypes: T_dtype_dict | None = None, schema: type | Daf | None = None, keyfield: str | int | T_ta | T_la = '', name: str = '', use_copy: bool = False, disp_cols: T_cs | None = None, retmode: str = 'obj', itermode: str = 'dict', attrs: T_da | None = None)

Create a Daf from rows, column names and options.

Every argument is optional, so `Daf()` makes an empty table. The usual call
gives the rows and the column names.

The rows are not copied. The `lol` list you pass in becomes the data of the
Daf, so changing one changes the other. The same holds for `hd`, `kd` and
`attrs`. Pass `use_copy=True` to deep copy `lol` first.

The `dtypes` can only name columns. If `dtypes` has a name that is not a column, the Daf
keeps a new dict without it, and your dict is not changed. A dict that is already right is adopted.

If `cols` is given, it sets the column names. If it is not given, the keys of
`dtypes` set them. Otherwise `hd` is used. A name that is empty or repeated
is made unique, so `['a', 'a']` becomes `['a', 'a_1']`.

With no names at all, the Daf has no columns and `lol` is only a list of rows.
Call `set_cols()` to name them later.

Args:
    lol: Rows, as a list of lists. Adopted, not copied.
    hd: Header dict that maps column name to position.
    kd: Key index to adopt when no keyfield is set.
    cols: Column names. These win over `hd` and `dtypes`.
    dtypes: Type for each column, used when converting, as from the text of a CSV file. It does not limit what a cell can hold. Entries for names that are not columns are left out.
    schema: A `@schemaclass` or a schema Daf that supplies columns and defaults.
    keyfield: Column, or tuple or list of columns, whose values identify rows. Each must be a column. See `set_keyfield()`.
    name: Free text name of this Daf.
    use_copy: If True, deep copy `lol` instead of adopting it.
    disp_cols: Column names to show when the Daf is printed.
    retmode: Whether a one cell result is returned as a Daf or a bare value.
    itermode: Whether iteration yields dicts or KeyedList objects.
    attrs: Free form dict of extra information. Adopted, not copied.

A `keyfield` must be made of columns, or the Daf raises `KeyError`. A Daf with no column
names yet can still be given one, for the columns that come later, as in
`Daf(keyfield='id')` followed by `append()` of dicts.

Raises:
    TypeError: `disp_cols` is not a list, a tuple or None.
    KeyError: The `keyfield` is not a column, or one of its columns is not.

Examples:
    >>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'], keyfield='id')
    >>> d.num_rows()
    2
    >>> Daf(lol=[[1, 2]], cols=['a', 'a']).columns()
    ['a', 'a_1']


### Class constants


##### COPY_NONE

A constant: 0



##### COPY_ROWS

A constant: 32



##### COPY_HD

A constant: 4



##### COPY_KD

A constant: 16



##### COPY_DTYPES

A constant: 8



##### COPY_ATTRS

A constant: 1



##### COPY_OUTER

A constant: 2



##### copy_level_default

A constant: 'sortable'



##### RETMODE_OBJ

A constant: 'obj'



##### RETMODE_VAL

A constant: 'val'



##### ITERMODE_DICT

A constant: 'dict'



##### ITERMODE_KEYEDLIST

A constant: 'keyedlist'



##### TYPE_MAP

A constant: {'int': int, 'float': float, 'str': str, 'bool': bool, 'list': list, 'dict': dict}


---

# Docs page: Daf API / Structure, columns and keys (api/daf/structure.md)

## Structure, columns and keys

The shape of a Daf, its column names, its keyfield, copying, and iteration.

### Size and shape


##### num_rows() -> int

Return the number of rows.

Returns:
    The number of rows.

Examples:
    >>> Daf(cols=['x'], lol=[[1], [2], [3]]).num_rows()
    3
    >>> Daf().num_rows()
    0



##### num_cols() -> int

Count the columns by looking at the rows.

This does not use the column names. It returns the length of the longest of
the first 10 rows, and 0 if there are no rows. For a table with equal row
lengths that is the real width. Use `len(d.columns())` to count the names, and
`is_rectangular()` to check the rows.

Returns:
    The number of columns.

Examples:
    >>> Daf(lol=[[1, 2, 3]], cols=['a', 'b']).num_cols()
    3
    >>> Daf(cols=['a', 'b']).num_cols()
    0



##### len() -> int

Return the number of rows.

This does the same as `num_rows()` and `len(d)`.

Returns:
    The number of rows.

Examples:
    >>> d = Daf(cols=['x'], lol=[[1], [2], [3]])
    >>> d.len(), len(d)
    (3, 3)



##### __len__() -> int

Return the number of rows, so `len(d)` works.

Returns:
    The number of rows.



##### shape() -> Tuple[int, int]

Return the number of rows and columns as a tuple.

This is a method, not a property, because the column count is worked out from
the rows. It is the same as `(num_rows(), num_cols())`. A Daf with column
names but no rows has shape `(0, 0)`.

Returns:
    A tuple of the number of rows and the number of columns.

Examples:
    >>> Daf(lol=[[1, 2], [3, 4], [5, 6]], cols=['a', 'b']).shape()
    (3, 2)



##### is_rectangular() -> bool

Check whether every row has the same length.

With column names, every row must be as long as the number of names. Without
names, every row must be as long as the first row. An empty Daf is rectangular.

This looks at every row. Use it when you do not trust the source of the data.
`num_cols()` only samples the first rows.

Returns:
    True if all rows have the expected length.

Examples:
    >>> Daf(lol=[[1, 2], [3]], cols=['a', 'b']).is_rectangular()
    False



##### force_rectangular() -> 'Daf'

Pad short rows with empty strings, in place.

Use this when a source leaves off trailing empty cells. An xlsx file read by
`xlsx_to_csv()` does that. The target width is the number of column names. With
no names, it is the length of the longest row.

Rows that are too long are not cut. That would hide damage, such as an
unquoted comma that split a value. They raise an error instead.

Returns:
    This Daf, which has been changed.

Raises:
    ValueError: A row is longer than the number of columns. Nothing is changed.

Examples:
    >>> Daf(lol=[[1, 2], [3]], cols=['a', 'b']).force_rectangular()
    <BLANKLINE>
    | a  | b  |
    | -: | -: |
    |  1 |  2 |
    |  3 |    |
    <BLANKLINE>
    %% daf rows=2; cols=2; keyfield=''; name=''
    <BLANKLINE>



##### __bool__() -> bool

Say whether the Daf holds data.

`bool(d)` and `if d:` are true when the Daf has at least one row with a value.
A Daf that has column names but no rows is false. Daffodil uses this as its
test for an empty table.

Returns:
    True if there is data.

Examples:
    >>> bool(Daf(cols=['a']))
    False
    >>> bool(Daf(lol=[[1]], cols=['a']))
    True


### Column names


##### columns() -> T_ls

Return the column names.

The list is a new copy, so changing it does not change the Daf. Use
`set_cols()` or `rename_cols()` to change the names.

Returns:
    The column names, in order.

Examples:
    >>> Daf(cols=['a', 'b']).columns()
    ['a', 'b']



##### calc_cols(include_cols: Iterable | None = None, exclude_cols: Iterable | None = None, include_types: List[Type] | None = None, exclude_types: List[Type] | None = None) -> Iterable

Work out a list of column names from rules.

Use it to choose the columns for `apply` or `reduce`. The rules are applied
in this order: include by name, exclude by name, include by type and exclude
by type. The result keeps the order of the Daf. A single name may be given
as a string.

With a group by operation, leave the group by columns out of the list.

Args:
    include_cols: Keep only these column names.
    exclude_cols: Drop these column names.
    include_types: Keep only columns whose dtype is in this list.
    exclude_types: Drop columns whose dtype is in this list.

Returns:
    The selected column names.

Raises:
    RuntimeError: A type rule is given and no dtypes are set.

Examples:
    >>> d = Daf(lol=[[1, 2, 3]], cols=['a', 'b', 'c'])
    >>> d.calc_cols(exclude_cols='b')
    ['a', 'c']



##### rename_cols(from_to_dict: T_ds) -> 'Daf'

Rename columns in place.

Names that are not in the mapping stay as they are. Names in the mapping that
are not in the Daf are ignored. The dtypes are renamed too.

The keyfield follows the rename. If its column is renamed, the keyfield has the
new name, and key lookups keep working.

Args:
    from_to_dict: Maps old names to new names.

Returns:
    This Daf, which has been changed.

Examples:
    >>> d = Daf(lol=[[1, 2]], cols=['a', 'b'])
    >>> d.rename_cols({'b': 'c'}).columns()
    ['a', 'c']
    >>> k = Daf(lol=[[1, 'x']], cols=['id', 'v'], keyfield='id')
    >>> k.rename_cols({'id': 'ident'}).keyfield
    'ident'



##### set_cols(new_cols: T_ls | None = None, sanitize_cols: bool = True, unnamed_prefix: str = 'col') -> 'Daf'

Set the column names, in place.

The new names are given by position, so the first name goes to the first
column. Without a list, the names are A, B, C and so on, like a spreadsheet.

With `sanitize_cols` on, a repeated name gets a suffix, so `['a', 'a']` becomes
`['a', 'a_1']`. An empty name becomes the prefix and its position, like `col2`. The
prefix is short, because these names are printed. Names that come from parsing a header, in the
constructor and in `from_md()`, use `Unnamed` instead, which says that there is no name.
The dtypes are renamed by position as well.

The keyfield follows the names, as the dtypes do. If its column gets a new name, the
keyfield has that name. If the Daf had no column names yet, the keyfield is kept
when it is one of the new names, so it can be given before the names are known.
Otherwise it is cleared.

Args:
    new_cols: The names, in order. If None, spreadsheet names are made.
    sanitize_cols: If True, make the names valid and unique.
    unnamed_prefix: The start of a name made for an empty one.

Returns:
    This Daf, which has been changed.

Raises:
    AttributeError: The number of names is not the number of columns.

Examples:
    >>> Daf(lol=[[1, 2, 3]]).set_cols().columns()
    ['A', 'B', 'C']
    >>> Daf(lol=[[1, 2, 3]]).set_cols(['a', 'a', '']).columns()
    ['a', 'a_1', 'col2']
    >>> d = Daf(lol=[[1, 'x']], keyfield='id').set_cols(['id', 'v'])
    >>> d.keyfield
    'id'
    >>> d = Daf(lol=[[1, 'x']], cols=['a', 'b'], keyfield='a').set_cols(['id', 'v'])
    >>> d.keyfield
    'id'


### Keyfield


##### keys(silent_error: bool = True, astype: str = 'list') -> T_la | T_kva

Return the row keys.

The keys come from the key index. There is one for each distinct value of the
keyfield, in the order of the first row that has it. With a composite keyfield
each key is a tuple. The key index is built here if it is needed.

A repeated key appears once. The index then points to the last row that has it,
so a lookup of that key finds the last row. So `len(keys())` is less than
`len(daf)` when keys repeat.

Without a keyfield the answer is an empty list, unless `silent_error` is False.
Then it raises an error. An index passed in as `kd` is not used by this method.

With `astype='view'` you get a view of the key index, not a copy. It keeps the
old keys after the Daf changes, so use it right away.

Args:
    silent_error: If False, raise an error when there is no keyfield.
    astype: `list` for a new list, or `view` for a view of the index.

Returns:
    The keys.

Raises:
    KeysDisabledError: There is no keyfield and no key index, and `silent_error` is False.
    ValueError: `astype` is neither `list` nor `view`.

Examples:
    >>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'], keyfield='id')
    >>> d.keys()
    [1, 2]
    >>> dup = Daf(cols=['id', 'q'], lol=[['a', 1], ['b', 2], ['a', 3]], keyfield='id')
    >>> dup.keys(), dup.select_record('a')
    (['a', 'b'], {'id': 'a', 'q': 3})



##### set_keyfield(keyfield: str | T_ta | T_la = '', silent_error: bool = False, force_kd_rebuild: bool = False) -> 'Daf'

Choose the column, or columns, that identify the rows.

Give a column name, or a tuple or list of names for a composite key. A
composite key is a tuple of those values. An empty keyfield turns key lookups
off. The key index is built later, when it is first needed.

The Daf must have column names first. With none, nothing happens.

A name that is not a column raises `KeyError`. Pass `silent_error=True` to store it
anyway, as older versions did. Then key lookups find nothing. A key that is not
unique is not checked. A lookup finds the last row that has it.

A Daf that has column names and no rows can have a keyfield. It applies to the rows that are
added later.

Args:
    keyfield: Column name, or a tuple or list of names. Empty to turn off.
    silent_error: If True, store a name that is not a column, and do not raise.
    force_kd_rebuild: If True, build the key index now.

Returns:
    This Daf, which has been changed.

Raises:
    KeyError: The keyfield is not a column, and `silent_error` is not True.

Examples:
    >>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'])
    >>> d.set_keyfield('id').keys()
    [1, 2]
    >>> d.set_keyfield(['id', 'v']).keys()
    [(1, 'a'), (2, 'b')]
    >>> e = Daf(cols=['id', 'v']).set_keyfield('id')
    >>> e.keyfield
    'id'
    >>> e.append({'id': 7, 'v': 'x'}).keys()
    [7]



##### get_existing_keys(keylist: T_ls) -> T_ls

Keep the keys that are in the Daf.

Use it to find out which of a list of keys have a row. The order of the list
is kept. The result is empty if the Daf has no keyfield.

Args:
    keylist: The keys to check.

Returns:
    The keys from the list that have a row.

Examples:
    >>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'], keyfield='id')
    >>> d.get_existing_keys([1, 5, 2])
    [1, 2]



##### __contains__(key: Any) -> bool

Test whether a key is in the keyfield column.

Use it as `key in my_daf`. For a composite keyfield, the key is a tuple.
An empty Daf contains nothing.

Args:
    key: The key to look for.

Returns:
    True if a row has this key.

Raises:
    KeyError: The Daf has rows but no keyfield, and no key index was adopted.

Examples:
    >>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'], keyfield='id')
    >>> 2 in d, 5 in d
    (True, False)


### Copying and initializing


##### copy(level: str | T_copybits | None = None, name: str = '', deep: bool = False, for_sorting: bool = False, inplace: bool = False) -> 'Daf'

Make a copy of the Daf. You choose what the copy shares with the original.

Unless `inplace=True`, the copy is a new object of the same class. What it holds is shared
with the original unless you ask for its own. Sharing is what makes daffodil
fast, because nothing is copied that does not have to be. Pick the cheapest
`level` that is safe for what you do next.

The level is one of four names, or a sum of the `COPY_` constants of `Daf`.
Without a level, the class setting `copy_level_default` is used. It is `sortable`.
For copies with their own row lists by default, set it on a subclass, as in
`class SafeDaf(Daf): copy_level_default = 'editable'`. The table gives the cost for
200,000 rows of 50 columns.

| Level      | Parts copied                                          | Cost            |
|------------|-------------------------------------------------------|-----------------|
| `shallow`  | `attrs` and `disp_cols`                               | 0.00002 s       |
| `sortable` | `shallow`, the row list, `hd`, `dtypes`, the key index | 0.002 s, 2 MB   |
| `editable` | `sortable`, plus a new list for every row             | 0.5 s, 93 MB    |
| `deep`     | everything that is a container                        | 4 s, 107 MB     |

To choose your own mix, add these constants of `Daf` with `|`. A part without its bit is
shared. A copy is always a new object, so there is no bit for that. The two bits `COPY_ROWS`
and `COPY_OUTER` imply others: `COPY_ROWS` adds `COPY_OUTER`, and `COPY_OUTER` adds
`COPY_KD`. They are added for you, because a row list that is copied while the key index
is shared would leave the index wrong for one of the two. The name `deep` cannot be
combined with any bit.

| Constant      | Value | Parts copied                                              |
|---------------|------:|-----------------------------------------------------------|
| `COPY_NONE`   |     0 | nothing. Every part is shared.                            |
| `COPY_ATTRS`  |     1 | the `attrs` dict, deep copied, and the `disp_cols` list   |
| `COPY_OUTER`  |     2 | the list that holds the rows. The rows are still shared   |
| `COPY_HD`     |     4 | the header dict                                           |
| `COPY_DTYPES` |     8 | the dtypes dict                                           |
| `COPY_KD`     |    16 | the key index, and a `keyfield` that is a list            |
| `COPY_ROWS`   |    32 | a new list for every row                                  |

With `shallow`, the copy shares the row list, the rows, `hd` and `dtypes` with the
original. Reading is safe. Sorting with `sort_by_colname()` is safe too, because it
builds a new row list. Appending a row, sorting the row list in place or dropping
a column is seen by the original, and can leave it with a key index that is wrong.

With `sortable`, the copy has its own row list, so you can append, extend, insert,
remove, sort or reverse rows, rename columns, change the keyfield or dtypes, and
use `drop_cols()`. The key index is cleared on the copy and is rebuilt on first use.
The rows are still shared. Changing values is seen by the original, whichever way you do it.
That makes a selection a live view for editing. Adding a column is not seen by the original.
`insert_col()`, `insert_icol()`, `insert_idx_col()` and `assign_col()` with a new name
copy shared rows first.

With `editable`, each row is a new list. You can also add columns and change cells.
The cells themselves are shared. That is safe for text and numbers, and not for a
list or dict held in a cell.

With `deep`, nothing that can be changed is shared. Text and numbers are not copied,
because they cannot change. A list or dict held in a cell is copied.

The copy has no name unless you give one with `name`, so that two tables do not
claim to be the same one. The other settings, such as `md_max_rows`, the
`schema` and the retmode and itermode, are taken over. A `schema` is shared.

With `inplace=True`, the Daf copies into itself and returns itself, instead of making a
new one. Use it on a selection, to make it a table of its own: its rows then no longer
share anything with the Daf they came from. The default level is then `deep`. `editable`
may be asked for, to copy only the rows. A lower level would leave the rows shared, so it
raises ValueError. The name stays, unless `name` gives a new one.

The arguments `deep` and `for_sorting` are deprecated. Use `level='deep'` and
`level='sortable'` instead. They still work: `deep=True` gives the level `deep`, and
`for_sorting=True` gives `sortable`. A higher level wins if you give both. A call like
`copy(True)` still means `deep`. `copy(False)` uses the class setting.

Args:
    level: A name, a sum of the `COPY_` constants, or None for the class setting `copy_level_default`.
    name: The name of the copy. It is empty if not given.
    deep: Deprecated. Use `level='deep'`. True gives the level `deep`.
    for_sorting: Deprecated. Use `level='sortable'`. True gives the level `sortable`.
    inplace: If True, copy into this Daf and return it. The level must copy the rows.

Returns:
    The new Daf, or this one if `inplace`.

Raises:
    ValueError: `level` is a name that is not one of the four, or `inplace` with a level that does not copy the rows.

Examples:
    >>> d = Daf(lol=[[2, 'b'], [1, 'a']], cols=['id', 'v'], name='orig')
    >>> c = d.copy(name='work')
    >>> c.name, d.name
    ('work', 'orig')
    >>> _ = c.append([3, 'c'])
    >>> d.num_rows()
    2
    >>> s = d.copy('shallow')
    >>> _ = s.append([3, 'c'])
    >>> d.num_rows()
    3
    >>> d = Daf(lol=[[2, 'b'], [1, 'a']], cols=['id', 'v'])
    >>> e = d.copy('editable')
    >>> e[0, 'v'] = 'X'
    >>> d.iloc(0)
    {'id': 2, 'v': 'b'}
    >>> m = d.copy(Daf.COPY_OUTER | Daf.COPY_HD)
    >>> m[0, 'v'] = 'Y'
    >>> d.iloc(0)
    {'id': 2, 'v': 'Y'}
    >>> sel = d.select_where(lambda row: row['id'] == 1)
    >>> sel.copy(inplace=True) is sel
    True
    >>> d[1, 'v'] = 'Z'
    >>> sel.lol
    [[1, 'a']]



##### clone_empty(lol: T_lola | None = None, cols: T_ls | None = None, name: str = '') -> 'Daf'

Make a new Daf with the same layout and no rows.

This is [`copy()`][daffodil.daf.Daf.copy] with the layout bits and then no rows. The
new Daf is an object of the same class. It has the same column names, keyfield,
dtypes, `attrs`, `schema` and display settings. It has its own `hd`, `dtypes`,
`attrs` and key index. The name is not carried over. Give `name` to set it.

Give `lol` to fill the new Daf with rows. They are adopted, not copied.

If you give `cols` that differ from the columns of the original, the columns are
new. The `dtypes` keep the entries for columns that are still there. The keyfield
is cleared if any of its columns is gone. The `schema` and the `disp_cols` describe
the old columns, so they are not carried over. If `cols` are the same as the
columns of the original, everything is carried over.

Args:
    lol: Rows for the new Daf. If None, it has no rows.
    cols: Column names to use instead of the existing ones.
    name: The name of the new Daf.

Returns:
    The new Daf.

Examples:
    >>> d = Daf(lol=[[1, 'a']], cols=['id', 'v'], keyfield='id')
    >>> c = d.clone_empty()
    >>> c.columns(), c.keyfield, c.num_rows()
    (['id', 'v'], 'id', 0)



##### set_lol(new_lol: T_lola) -> 'Daf'

Replace the rows with a new list of lists.

The list is adopted, not copied. The column names, the keyfield and the other
settings stay. The key index is rebuilt when it is next needed.

Args:
    new_lol: The new rows.

Returns:
    This Daf, which has been changed.

Examples:
    >>> d = Daf(lol=[[1, 'a']], cols=['id', 'v'], keyfield='id')
    >>> d.set_lol([[5, 'q'], [6, 'r']]).keys()
    [5, 6]


### Iteration and comparison


##### retmode

What a selection like `daf[row, col]` returns: a Daf, or plain values.

'obj' is the default. Every selection returns a new Daf, so you can keep calling Daf
methods on the result.

'val' returns plain Python values when the selection is one cell, one row or one
column. You get a single value, or a list. A selection of several rows and several
columns is still a Daf.

A one-row result is the row itself, not a copy. Changing it changes this Daf.

In 'val' mode, one row or one cell is read straight from the rows, with no Daf
built on the way. This is when the row is an int position or a str key, and the
column, if given, is an int position or a str name. On a 10-row table it took
about 0.6 µs by position and 3.3 µs by key, against 12 to 29 µs through a Daf.

Use 'val' when you read values for your own code. Use 'obj' when you want to keep
working with Daf methods.

The mode belongs to the Daf you set it on. A Daf made by a selection starts again at
'obj'. A copy keeps the mode.

Set it with `daf.retmode = 'val'`, or with the retmode argument when you create the
Daf. Any other value raises ValueError.

Returns:
    str: 'obj' or 'val'.

Examples:
    >>> daf = Daf(cols=['x', 'y'], lol=[[1, 2], [3, 4]])
    >>> type(daf[0, 'x']).__name__
    'Daf'
    >>> daf.retmode = 'val'
    >>> daf[0, 'x']
    1
    >>> daf[1, :]
    [3, 4]
    >>> daf[:, 'y']
    [2, 4]



##### itermode

What each row is when you loop over a Daf: a dict, or a KeyedList.

Your code reads a row the same way in both modes, by column name, as in `row['qty']`.
The difference is what the row is.

'dict' is the default. Each row is a new dict of column name to value. The values are
copied into it, so changing it does not change the Daf.

'keyedlist' gives each row as a [KeyedList][daffodil.keyedlist.KeyedList]. It points
into the table instead of copying it. See there for how it works, what changes the
Daf, and when it is faster.

The mode is used by `for row in daf`, and by methods that loop over the rows, such as
`reduce()`. The methods `iter_dict()`, `iter_klist()` and `iter_list()` ignore it and
always give their own kind of row. A Daf made by a selection starts again at 'dict'.

Set it with `daf.itermode = 'keyedlist'`, or with the itermode argument when you
create the Daf. Any other value raises ValueError.

Returns:
    str: 'dict' or 'keyedlist'.

Examples:
    >>> daf = Daf(cols=['x', 'y'], lol=[[1, 2], [3, 4]])
    >>> for row in daf:
    ...     row['y'] = 0
    >>> daf
    <BLANKLINE>
    | x  | y  |
    | -: | -: |
    |  1 |  2 |
    |  3 |  4 |
    <BLANKLINE>
    %% daf rows=2; cols=2; keyfield=''; name=''
    <BLANKLINE>
    >>> daf.itermode = 'keyedlist'
    >>> for row in daf:
    ...     row['y'] = 0
    >>> daf
    <BLANKLINE>
    | x  | y  |
    | -: | -: |
    |  1 |  0 |
    |  3 |  0 |
    <BLANKLINE>
    %% daf rows=2; cols=2; keyfield=''; name=''
    <BLANKLINE>



##### __iter__() -> Iterator[Dict[str, Any] | KeyedList]

Loop over the rows, as in `for row in daf`.

Each row is a dict or a KeyedList, depending on itermode. For plain lists, use
`iter_list()`.

In the default itermode, each row is a new dict. Assigning a new value to it does not
change the Daf. To change values in a loop, use `iter_klist()`, whose rows write
through, or `apply_in_place()`. A list or dict held in a cell is the same object in
the dict and the Daf, so changing it in place does show.

Each loop gets a new iterator, which goes through the rows once. It goes through
the list of rows it started with: rows appended during the loop are seen, but a
new list put in place, as by a sort, is not.

Returns:
    Iterator: The rows, as dicts or KeyedList objects.



##### iter_dict() -> Iterator[Dict[str, Any]]

Loop over the rows as dicts, whatever the itermode is.

Each dict is a copy, so changing it does not change the Daf.

Returns:
    Iterator[Dict[str, Any]]: Each row as a dict of column name to value.

Examples:
    >>> d = Daf(cols=['x', 'y'], lol=[[1, 2], [3, 4]])
    >>> list(d.iter_dict())
    [{'x': 1, 'y': 2}, {'x': 3, 'y': 4}]
    >>> for row in d.iter_dict():
    ...     row['y'] = 0
    >>> d
    <BLANKLINE>
    | x  | y  |
    | -: | -: |
    |  1 |  2 |
    |  3 |  4 |
    <BLANKLINE>
    %% daf rows=2; cols=2; keyfield=''; name=''
    <BLANKLINE>



##### iter_klist() -> Iterator[KeyedList]

Loop over the rows as [KeyedList][daffodil.keyedlist.KeyedList] objects, whatever the
itermode is.

See [KeyedList][daffodil.keyedlist.KeyedList] for how these rows work.

Returns:
    Iterator[KeyedList]: Each row as a KeyedList.

Examples:
    >>> d = Daf(cols=['x', 'y'], lol=[[1, 2], [3, 4]])
    >>> [row['x'] for row in d.iter_klist()]
    [1, 3]
    >>> for row in d.iter_klist():
    ...     row['y'] = 0
    >>> d
    <BLANKLINE>
    | x  | y  |
    | -: | -: |
    |  1 |  0 |
    |  3 |  0 |
    <BLANKLINE>
    %% daf rows=2; cols=2; keyfield=''; name=''
    <BLANKLINE>



##### iter_list() -> Iterator[list]

Loop over the rows as the plain lists that the Daf stores.

Values are read by position, not by column name. These are the Daf's own row lists,
not copies, so assigning to one changes the Daf.

Returns:
    Iterator[list]: Each row as a list.

Examples:
    >>> d = Daf(cols=['x', 'y'], lol=[[1, 2], [3, 4]])
    >>> [row[0] for row in d.iter_list()]
    [1, 3]
    >>> for row in d.iter_list():
    ...     row[1] = 9
    >>> d
    <BLANKLINE>
    | x  | y  |
    | -: | -: |
    |  1 |  9 |
    |  3 |  9 |
    <BLANKLINE>
    %% daf rows=2; cols=2; keyfield=''; name=''
    <BLANKLINE>



##### __eq__(other: object) -> bool

Compare two Daf instances.

Two Daf instances are equal when their rows, their column names and their
keyfield are equal. The order of the columns matters. The name, dtypes,
attrs and modes are not compared. Anything that is not a Daf is not equal.

Args:
    other: The object to compare with.

Returns:
    True if equal.

Examples:
    >>> a = Daf(lol=[[1]], cols=['x'])
    >>> a == Daf(lol=[[1]], cols=['x'])
    True
    >>> a == Daf(lol=[[1]], cols=['x'], keyfield='x')
    False

---

# Docs page: Daf API / Types and schema (api/daf/types.md)

## Types and schema

Column dtypes and schema support.

### Dtypes


##### set_dtypes(default_type: Type = str, typ_to_cols_dict: Dict[Type, T_ls] | None = None) -> 'Daf'

Set the dtype of every column, from a default and a few exceptions.

Use it when most columns have one type and a few differ. The result is stored
in `dtypes`. This does not convert any data. Call `apply_dtypes()` for that.

If most columns differ, assign a dict to `dtypes` yourself.

Args:
    default_type: The type of every column that is not listed.
    typ_to_cols_dict: Maps a type to the names of the columns of that type.

Returns:
    This Daf, which has been changed.

Raises:
    NotImplementedError: The Daf has no column names.

Examples:
    >>> d = Daf(lol=[['1', '2', 'x']], cols=['a', 'b', 'c'])
    >>> d.set_dtypes(str, {int: ['a', 'b']}).dtypes
    {'a': <class 'int'>, 'b': <class 'int'>, 'c': <class 'str'>}



##### apply_dtypes(dtypes: T_dtype_dict | None = None, cols: str | T_ls | None = None, unflatten: bool = True, from_str: bool = True, default_type: Type = str, silent_error: bool = False) -> 'Daf'

Convert the columns to their dtypes, in place.

A CSV file is read as text, because that is the fastest way to load it. Call
this method to turn the columns you need into numbers, lists and so on. It
changes the cells where they are and does not make a new table.

Three steps are separate in daffodil:

| Step | What it does |
|---|---|
| set `dtypes`, or `set_dtypes()` | declares the types of the columns. Nothing is converted |
| `apply_dtypes()` | tries to convert the cells |
| `check_schema()` | finds the cells that do not fit their dtypes |

Conversion is permissive. Text that is not a number stays as it is, and no error is
raised. So a column with an `int` dtype can still hold some text after this call.

Conversion costs time, about 0.04 s for each column of 200,000 rows, so convert
only the columns that you use. Pass `cols` to name them. A column that is not
in `cols` is not converted, is not checked, and keeps its entry in the `dtypes`
of the Daf. Without `cols`, every column of the Daf is converted, and each one
needs a dtype.

The `dtypes` argument is a dict that maps a column name to a type, or one type
for all columns. It may hold more columns than the Daf. Without `cols`, it
replaces the `dtypes` of the Daf. With `cols`, it only supplies the types of
those columns, and the other entries of the `dtypes` of the Daf stay as they
are. If neither the argument nor the Daf has dtypes, nothing happens. With no
columns defined, the names are taken from the dtypes.

Without `cols`, a `dtypes` that leaves out a column of the Daf raises
`ValueError`, unless `silent_error` is True. Then the column gets `default_type`,
which is `str` and is not converted, and the `dtypes` of the Daf are replaced by
the dict with that column added. That replaces the types of the columns that you
meant to convert later. To convert some columns and not others, use `cols`.

Types must be plain types such as `int`, `float`, `bool`, `str`, `list`,
`dict`, `tuple` or `set`. Annotations such as `List[str]` do not work.

By default, columns of type `str` are skipped. The cells are assumed to be text
already. Pass `from_str=False` when they may hold other values. Then each cell
is converted with `str()`. Columns of type `list` or `dict` are read from
their text. Pass `unflatten=False` to leave them as text. That text is read
about eight times faster when pyjson5 is installed, as with
`pip install daffodil[pyjson5]`.

An empty cell, such as two commas next to each other in a CSV file, is NULL,
which is the empty string. It stays NULL when its column is converted to `int` or
`float`, so a number column can hold NULL. A `bool` column is the exception: an
empty cell becomes 0 there.

A `bool` column gets 0 or 1. The text `0`, `false`, `no`, `n`, `f` and `off` give 0,
and `1`, `true`, `yes`, `y`, `t` and `on` give 1, in lower case, capitalized or upper
case. Other text stays as it is. So `'False'` gives 0, where Python's own `bool('False')`
is True. None gives 0, and any other value gives 0 or 1 by its truth. The sum methods, `sum()`, `sum_np()`
and `Daf.sum_da()`, skip a NULL cell. Python's own `sum()` of such a column
raises `TypeError`, so use the methods of the Daf.

A cell that cannot be converted to an `int` or a `float` keeps its text, so a bad
value is still there to be found, as with `list` and `dict`. No error is raised
for text. Call `check_schema()` to find such cells. A later step, such as a sum or a sort, may raise one. A cell that is
not text and not a number, such as a list in an `int` column, cannot be given to
`int()`, and raises `TypeError`. The `dtypes` do not limit what a cell can hold,
so leave such a column out of the conversion. Whole number text of any size is
converted exactly. Text with a decimal point or an exponent is converted to an
`int` by way of a float, which cuts the decimal part.

This method does the common conversions and keeps them simple. For your own
rules, convert the columns yourself and then say what the types are. Use
`apply_to_col()` for one column, or `apply_in_place()` for several. Your
function can raise, collect the bad values, or use any default. Then set
`dtypes`. This does not convert anything. See the last example.

Args:
    dtypes: Maps column names to types, or a single type for all columns.
    cols: The columns to convert, or one column name. Others are left alone.
    unflatten: If True, read list and dict columns from their text.
    from_str: If True, the cells are text, so `str` columns are not converted.
    default_type: The type to use for a column that has no dtype.
    silent_error: If False, raise an error when a column has no dtype.

Returns:
    This Daf, which has been changed.

Raises:
    ValueError: A column to convert has no dtype and `silent_error` is False.
    KeyError: A name in `cols` is not a column of the Daf.
    TypeError: A cell in an `int` or `float` column is not text and not a number.

Examples:
    >>> d = Daf(lol=[['1', '2.5', 'x']], cols=['a', 'b', 'c'])
    >>> d.apply_dtypes(dtypes={'a': int, 'b': float, 'c': str})
    <BLANKLINE>
    | a  |  b  | c  |
    | -: | --: | -: |
    |  1 | 2.5 |  x |
    <BLANKLINE>
    %% daf rows=1; cols=3; keyfield=''; name=''
    <BLANKLINE>

    An empty cell stays empty, and the sum methods skip it:

    >>> d = Daf.from_csv_buff('a,b\n1,\n,3\n4,5\n')
    >>> d.apply_dtypes(dtypes={'a': int, 'b': int})
    <BLANKLINE>
    | a  | b  |
    | -: | -: |
    |  1 |    |
    |    |  3 |
    |  4 |  5 |
    <BLANKLINE>
    %% daf rows=3; cols=2; keyfield=''; name=''
    <BLANKLINE>
    >>> d.sum()
    {'a': 5, 'b': 8}

    Convert only some columns with `cols`. The others stay text, and the
    `dtypes` of the Daf are not changed:

    >>> d = Daf(lol=[['1', '2.5', '7']], cols=['a', 'b', 'n'],
    ...         dtypes={'a': int, 'b': float, 'n': int})
    >>> _ = d.apply_dtypes(cols=['a'])
    >>> d.iloc(0)
    {'a': 1, 'b': '2.5', 'n': '7'}
    >>> d.dtypes == {'a': int, 'b': float, 'n': int}
    True
    >>> _ = d.apply_dtypes(cols=['n'])
    >>> d.iloc(0)
    {'a': 1, 'b': '2.5', 'n': 7}

    A `bool` column, and text that does not convert:

    >>> b = Daf(cols=['ok', 'n'], lol=[['', '1'], ['0', 'x'], ['True', '3'], ['maybe', '']])
    >>> _ = b.apply_dtypes(dtypes={'ok': bool, 'n': int})
    >>> b.lol
    [[0, 1], [0, 'x'], [1, 3], ['maybe', '']]

    Your own conversion, here one that records the values that fail:

    >>> bad = []
    >>> def to_int(value):
    ...     try:
    ...         return int(value)
    ...     except ValueError:
    ...         bad.append(value)
    ...         return ''
    >>> d = Daf(lol=[['1'], ['x'], ['3']], cols=['n'])
    >>> _ = d.apply_to_col('n', to_int)
    >>> d.dtypes = {'n': int}
    >>> d
    <BLANKLINE>
    | n  |
    | -: |
    |  1 |
    |    |
    |  3 |
    <BLANKLINE>
    %% daf rows=3; cols=1; keyfield=''; name=''
    <BLANKLINE>
    >>> bad
    ['x']



##### flatten(convert_bool_to_int: bool = True, use_pyon: bool = True) -> 'Daf'

Turn list and dict cells into text, in place.

You rarely need this. `to_csv_buff()` and `to_csv_file()` write every cell as
text already, so a separate pass is not needed.

Only the columns whose dtype is `list` or `dict` are changed. Each cell
becomes its `str()` text. A column of dtype `bool` becomes 0 and 1. Without
dtypes nothing happens.

Args:
    convert_bool_to_int: If True, write bool columns as 0 and 1.
    use_pyon: Kept for compatibility. Only True is supported.

Returns:
    This Daf, which has been changed.

Raises:
    ValueError: `use_pyon` is False.

Examples:
    >>> d = Daf(lol=[[[1, 2], True]], cols=['a', 'b'], dtypes={'a': list, 'b': bool})
    >>> d.flatten()
    <BLANKLINE>
    |   a    | b  |
    | -----: | -: |
    | [1, 2] |  1 |
    <BLANKLINE>
    %% daf rows=1; cols=2; keyfield=''; name=''
    <BLANKLINE>



##### strip(chrs: str = ' ') -> 'Daf'

Remove characters from both ends of every text cell, in place.

Each character in `chrs` is removed on its own, so `'()"'` removes any mix of
parentheses and quotes. Cells that are not text, and empty cells, are skipped.

Args:
    chrs: The characters to remove.

Returns:
    This Daf, which has been changed.

Examples:
    >>> Daf(lol=[[' a ', 3, '("x")']], cols=['p', 'q', 'r']).strip(' ()"')
    <BLANKLINE>
    | p  | q  | r  |
    | -: | -: | -: |
    |  a |  3 |  x |
    <BLANKLINE>
    %% daf rows=1; cols=3; keyfield=''; name=''
    <BLANKLINE>


### Schema


##### apply_schema(schema: type | Daf | None = None) -> Daf

Attach a schema to this Daf and fill in what the Daf is missing.

A schema describes columns: their names, types and default values. Use one
when many tables share a layout, or to make new records with the defaults.
The constructor calls this method, so `Daf(schema=...)` applies the schema.

Two kinds of schema are accepted.

A class decorated with `@schemaclass`. The annotated attributes give the
column names and types. Their values are the defaults. A `__keyfield__`
attribute gives the keyfield.

A schema Daf, with one row for each column of the table. It must have a
`Name` column. It may have a `dtype` column, which holds one of `str`, `int`,
`float`, `bool`, `list` or `dict`. A `Default` column gives the defaults. Its
`attrs` may hold a `keyfield`.

Only gaps are filled. The column names, the dtypes and the keyfield are taken
from the schema only if the Daf has none. The rows are never changed and
nothing is validated. Call `check_schema()` to check the cells.

The other columns of a schema Daf are for building input forms. They are not
used by daffodil. The `Type` column says what kind of form control to use.

    checkbox   One or more checkboxes. Value lists the labels.
                   checkbox+buttons adds Set and Clear buttons.
                   checkbox+values allows values that differ from the labels.
    date       A text box with a calendar button.
    label      Read only text.
    radio      Like checkbox, but only one can be chosen.
    select     A dropdown or list box. Value lists the options.
                   select+multi allows several choices.
                   select+values allows values that differ from the labels.
    text       A one line text box. Value is the initial text.
    textarea   A multi line text box. Size is columns x rows, such as 80x6.

Args:
    schema: The schema to apply. If None, the schema already attached is used.

Returns:
    This Daf, which has been changed.

Raises:
    TypeError: The schema is neither a `@schemaclass` nor a schema Daf. Nothing is stored.
    RuntimeError: A `dtype` in a schema Daf is not one of the supported names.

Examples:
    >>> from daffodil.daf import Daf
    >>> from daffodil.lib.schemaclass import schemaclass
    >>> @schemaclass
    ... class Person:
    ...     name: str = ''
    ...     age: int = 0
    >>> d = Daf(schema=Person)
    >>> d.columns()
    ['name', 'age']

    On a Daf with rows and no names, it fills the names and dtypes, and converts nothing:

    >>> raw = Daf(lol=[['ann', '7']])
    >>> _ = raw.apply_schema(Person)
    >>> raw.columns(), raw.lol
    (['name', 'age'], [['ann', '7']])
    >>> d.dtypes
    {'name': <class 'str'>, 'age': <class 'int'>}



##### attach_schema(schema: type) -> Daf

Attach a `@schemaclass` to this Daf.

This is the strict form of `apply_schema()`, for a schema class only. It stores
the schema, then fills in the column names, the dtypes and the keyfield, each
only if the Daf has none. The rows are never changed and nothing is validated.

Args:
    schema: A class decorated with `@schemaclass`.

Returns:
    This Daf, which has been changed.

Raises:
    TypeError: The class is not a `@schemaclass`.

Examples:
    >>> from daffodil.daf import Daf
    >>> from daffodil.lib.schemaclass import schemaclass
    >>> @schemaclass
    ... class Person:
    ...     name: str = ''
    ...     age: int = 0
    >>> Daf().attach_schema(Person).columns()
    ['name', 'age']



##### check_schema(max_report: int = 10) -> Daf

Check that every cell fits the dtypes and the required columns. Raise if one does not.

Loading does not check the data, so that it stays fast. `apply_dtypes()` leaves a
cell that does not convert as text, and an empty cell is NULL. Call this method
when you want to find such cells, for example after a load while developing, or
in a test. It is never called by daffodil itself.

The dtypes are those of the Daf, which `apply_schema()` takes from the schema. A
cell must be of its dtype, or NULL. NULL is allowed in a column of any dtype,
unless the column is required, and it is false when tested. A `float` column may
also hold an `int`. A `bool` column holds 0 or 1. `apply_dtypes()` keeps an empty
cell as NULL, except in a `bool` column, where it becomes 0. A column of a dtype
not listed here is not checked.

The required columns come from the `Required` column of an attached schema Daf.
A value such as `yes`, `true` or `1` marks the column as required. A required
column must exist, and none of its cells may be NULL. A `@schemaclass` has no
required columns.

Args:
    max_report: How many bad cells the error lists. The error also gives the total.

Returns:
    This Daf, unchanged, when every cell fits.

Raises:
    ValueError: A cell does not fit its dtype, a required cell is NULL, or a
        required column is missing.

Examples:
    >>> from daffodil.daf import Daf
    >>> schema = Daf.from_md('''
    ... | Name | dtype | Required |
    ... |:-----|:------|:---------|
    ... | id   | str   | yes      |
    ... | n    | int   |          |
    ... ''')
    >>> d = Daf.from_csv_buff('id,n\na,1\n,x\n')
    >>> d = d.apply_schema(schema).apply_dtypes(dtypes=d.dtypes)
    >>> d.check_schema()
    Traceback (most recent call last):
    ...
    ValueError: check_schema(): 2 cells do not fit the schema:
      row 1, column 'id': required, but empty
      row 1, column 'n': 'x' is not int



##### default_record(astype: type = dict) -> Any

Make a new record for this Daf: its columns, with the defaults of the attached schema.

Use it to start a record that you fill in, then append. Each call returns a new
record, so changing it does not change the next one. Nothing is converted or validated.

The defaults are worked out the first time, and kept until the columns or the schema of
the Daf are replaced. A default that is changed on the schema class after that is not seen.

The record follows the columns of the Daf, in their order, however they were set. A
column gets the schema's default if the schema has one, and NULL otherwise. A field of
the schema that is not a column is left out. So a Daf that keeps only some of the
schema's columns gets records that fit it. A Daf with no schema gets NULL in every
column. A Daf with no columns takes its columns from the schema.

With `astype=KeyedList` the record is a KeyedList that shares the `hd` of this Daf,
so `record.hd is daf.hd`. Fill it by name, in any order: assigning to a key writes to
that column's position. `append()` then knows the record fits, and skips the check of
its columns.

For a schema Daf, the `Name` column gives the fields and the `Default` column gives
the values. A missing `Default` column gives empty strings.

Args:
    astype: `dict`, the default, or `KeyedList`.

Returns:
    A dict or a KeyedList of each column and its default.

Raises:
    AttributeError: The Daf has no columns and no schema.
    RuntimeError: A schema Daf has no `Name` column.
    TypeError: The attached schema is of an unsupported kind, or `astype` is not dict or KeyedList.

Examples:
    >>> from daffodil.daf import Daf
    >>> from daffodil.keyedlist import KeyedList
    >>> from daffodil.lib.schemaclass import schemaclass
    >>> @schemaclass
    ... class Person:
    ...     name: str = ''
    ...     age: int = 0
    >>> Daf(schema=Person).default_record()
    {'name': '', 'age': 0}
    >>> Daf(cols=['age', 'city'], schema=Person).default_record()
    {'age': 0, 'city': ''}
    >>> d = Daf(schema=Person)
    >>> row = d.default_record(astype=KeyedList)
    >>> row['age'] = 41
    >>> row.hd is d.hd, row.to_dict()
    (True, {'name': '', 'age': 41})

---

# Docs page: Daf API / Import and export (api/daf/convert.md)

## Import and export

Creating a Daf from other data and files, and converting it back out.

### Python structures


##### from_lod(records_lod: T_loda, keyfield: str | int | T_ta | T_la = '', dtypes: T_dtype_dict | None = None, name: str = '', cols: T_ls | None = None, ignore_extra_keys: bool = False, fast: bool = False) -> 'Daf'

Make a Daf from a list of dicts, one dict for each row.

Without `cols` or `dtypes`, the columns are all the keys of all the dicts, in the
order in which they first appear. The keys of the first dict come first. A dict
that lacks a key gets NULL there. A key that first appears in a later dict adds a
column at the right, and the rows before it get NULL there. You do not need to
look through the data first. Empty dicts and items that are not dicts are skipped.

If `cols` is given, those are the columns. Otherwise the keys of `dtypes` are
the columns. You chose the columns, so a dict with a key that is not one of them
raises `ValueError`, which names the keys, because that value would be lost.
Pass `ignore_extra_keys=True` to leave such keys out on purpose, as when you
pick a few columns of wide records.

With `fast=True` the first dict is checked fully, and every dict for its number of
keys. Use it when you built every dict
yourself, with the same keys in the same order, as in a loop. Each row is the values
of a dict, in the dict's own order. The columns are `cols`, or the keys of `dtypes`,
or the keys of the first dict. With `cols` or `dtypes`, the first dict must have the
columns as its keys, in order, or it raises `ValueError`. A later dict with the wrong
number of keys raises `ValueError`. A later dict with its keys in another order is added
anyway, and its values land in the wrong columns.

Args:
    records_lod: The rows, as dicts.
    keyfield: Column, or tuple or list of columns, whose values identify rows.
    dtypes: Type for each column. When given, it also selects the columns.
    name: Name of the new Daf.
    cols: Column names to use, in order.
    ignore_extra_keys: With `cols` or `dtypes`, leave out keys that are not columns, and do not raise.
    fast: If True, take each dict's values as a row. Check the first dict fully, and every dict's length. See above.

Returns:
    The new Daf.

Raises:
    ValueError: `cols` or `dtypes` is given, a dict has a key that is not one of those columns,
        and `ignore_extra_keys` is not True.

Examples:
    >>> Daf.from_lod([{'a': 1, 'b': 2}, {'a': 3}])
    <BLANKLINE>
    | a  | b  |
    | -: | -: |
    |  1 |  2 |
    |  3 |    |
    <BLANKLINE>
    %% daf rows=2; cols=2; keyfield=''; name=''
    <BLANKLINE>
    >>> Daf.from_lod([{'a': 1}, {'a': 2, 'b': 5}, {'c': 9}])
    <BLANKLINE>
    | a  | b  | c  |
    | -: | -: | -: |
    |  1 |    |    |
    |  2 |  5 |    |
    |    |    |  9 |
    <BLANKLINE>
    %% daf rows=3; cols=3; keyfield=''; name=''
    <BLANKLINE>
    >>> Daf.from_lod([{'a': 1, 'b': 2}], cols=['b', 'a'])
    <BLANKLINE>
    | b  | a  |
    | -: | -: |
    |  2 |  1 |
    <BLANKLINE>
    %% daf rows=1; cols=2; keyfield=''; name=''
    <BLANKLINE>
    >>> Daf.from_lod([{'a': 1, 'b': 2, 'c': 3}], cols=['a', 'b'], ignore_extra_keys=True)
    <BLANKLINE>
    | a  | b  |
    | -: | -: |
    |  1 |  2 |
    <BLANKLINE>
    %% daf rows=1; cols=2; keyfield=''; name=''
    <BLANKLINE>



##### to_lod() -> T_loda

Make a list of dicts, one dict for each row.

The dicts are new, but the values in them are the same objects as in the Daf.
An empty Daf gives an empty list. See `iter_dict()` to go through rows one at
a time without building the whole list.

Returns:
    The rows as dicts.

Examples:
    >>> d = Daf(lol=[[1, 'a'], [2, ['x']]], cols=['id', 'v'])
    >>> lod = d.to_lod()
    >>> lod
    [{'id': 1, 'v': 'a'}, {'id': 2, 'v': ['x']}]
    >>> lod[1]['v'].append('y')
    >>> d.iloc(1)
    {'id': 2, 'v': ['x', 'y']}



##### from_lot(records_lot: T_lota, cols: T_ls | None = None, dtypes: T_dtype_dict | None = None, keyfield: str | int | T_ta | T_la = '', name: str = '') -> 'Daf'

Make a Daf from a list of tuples, one tuple for each row.

Without `cols` the Daf has no column names, as with `Daf(lol=...)`. Call
`set_cols()` to give it spreadsheet names such as `A` and `B`, or to name them. A
Daf with no names cannot give its rows as dicts or KeyedList objects.

Args:
    records_lot: The rows, as tuples.
    cols: Column names. They must be as many as the items in each tuple.
    dtypes: Type for each column.
    keyfield: Column, or tuple or list of columns, whose values identify rows.
    name: Name of the new Daf.

Returns:
    The new Daf.

Raises:
    ValueError: A tuple does not have as many items as there are columns.

Examples:
    >>> Daf.from_lot([(1, 'a'), (2, 'b')], cols=['id', 'v'])
    <BLANKLINE>
    | id | v  |
    | -: | -: |
    |  1 |  a |
    |  2 |  b |
    <BLANKLINE>
    %% daf rows=2; cols=2; keyfield=''; name=''
    <BLANKLINE>
    >>> Daf.from_lot([(1, 'a')]).columns()
    []
    >>> Daf.from_lot([(1, 'a')]).set_cols().columns()
    ['A', 'B']



##### from_dod(dod: T_doda, keyfield: str = 'rowkey', dtypes: T_dtype_dict | None = None, ignore_extra_keys: bool = False) -> 'Daf'

Make a Daf from a dict of dicts, where the outer key names the row.

A dict of dicts usually does not repeat the row key inside each row. A Daf
table always has it as a column. If the inner dicts lack the `keyfield`
column, it is added from the outer keys. The new Daf has that keyfield.

If `dtypes` is given, its keys are the columns, and it must name the `keyfield`
column too. An inner dict with a key that `dtypes` does not name raises
`ValueError`, which names the keys, because that value would be lost. Pass
`ignore_extra_keys=True` to leave such keys out on purpose.

Use `to_dod()` to go back.

Args:
    dod: A dict that maps a row key to a dict of that row.
    keyfield: The column that holds the outer key.
    dtypes: Type for each column. When given, it also selects the columns.
    ignore_extra_keys: With `dtypes`, leave out keys that it does not name, and do not raise.

Returns:
    The new Daf.

Raises:
    ValueError: `dtypes` is given, an inner dict has a key that it does not name, and
        `ignore_extra_keys` is not True.
    KeyError: `dtypes` does not name the `keyfield` column.

Examples:
    >>> d = Daf.from_dod({'r0': {'x': 1}, 'r1': {'x': 2}})
    >>> d
    <BLANKLINE>
    | rowkey | x  |
    | -----: | -: |
    |     r0 |  1 |
    |     r1 |  2 |
    <BLANKLINE>
    %% daf rows=2; cols=2; keyfield='rowkey'; name=''
    <BLANKLINE>
    >>> dod = {'r0': {'x': 1, 'id': 7}, 'r1': {'x': 2, 'id': 8}}
    >>> Daf.from_dod(dod, dtypes={'rowkey': str, 'x': int}, ignore_extra_keys=True)
    <BLANKLINE>
    | rowkey | x  |
    | -----: | -: |
    |     r0 |  1 |
    |     r1 |  2 |
    <BLANKLINE>
    %% daf rows=2; cols=2; keyfield='rowkey'; name=''
    <BLANKLINE>



##### to_dod(remove_keyfield: bool = True) -> T_doda

Make a dict of dicts, where the keyfield value names each row.

The keyfield column is left out of the inner dicts by default. Pass
`remove_keyfield=False` to keep it there too. The Daf must have a keyfield.
Without one, a `KeysDisabledError` is raised, unless the Daf has no rows.

Args:
    remove_keyfield: If True, do not repeat the key inside each inner dict.

Returns:
    A dict that maps each key to a dict of that row.

Raises:
    KeysDisabledError: The Daf has no keyfield.

Examples:
    >>> d = Daf(lol=[[1, 'a']], cols=['id', 'v'], keyfield='id')
    >>> d.to_dod()
    {1: {'v': 'a'}}
    >>> d.to_dod(remove_keyfield=False)
    {1: {'id': 1, 'v': 'a'}}



##### from_cols_dol(cols_dol: T_dola, keyfield: str = '', dtypes: T_dtype_dict | None = None) -> 'Daf'

Make a Daf from a dict of lists, where each list is a column.

All the lists must be as long as each other. If one differs in length from the
first, a `ValueError` is raised, because the rows could not be made without
losing values or guessing them.

Args:
    cols_dol: Maps a column name to the list of its values.
    keyfield: Column, or tuple or list of columns, whose values identify rows.
    dtypes: Type for each column.

Returns:
    The new Daf.

Raises:
    ValueError: A list has a different length from the first.

Examples:
    >>> d = Daf.from_cols_dol({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> d
    <BLANKLINE>
    | A  | B  |
    | -: | -: |
    |  1 |  4 |
    |  2 |  5 |
    |  3 |  6 |
    <BLANKLINE>
    %% daf rows=3; cols=2; keyfield=''; name=''
    <BLANKLINE>



##### to_cols_dol() -> dict

Make a dict of lists, one list of values for each column.

Returns:
    A dict that maps each column name to the list of its values.

Examples:
    >>> Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v']).to_cols_dol()
    {'id': [1, 2], 'v': ['a', 'b']}



##### from_lod_to_cols(lod: T_loda, cols: List | None = None, keyfield: str = '', dtypes: T_dtype_dict | None = None) -> 'Daf'

Make a Daf in which each dict becomes a column, not a row.

The keys of the dicts become the first column, and each dict adds one more
column after it. Use this to compare several results side by side, such as
the same features measured in several tries. It is a transpose of
`from_lod()`.

Without `cols`, the first column is named `key` and the others `A`, `B` and
so on. The names in `cols` include the first column. The keyfield is not set
unless `keyfield` is given.

Args:
    lod: The dicts. They should all have the same keys.
    cols: Column names, starting with the name of the column of keys.
    keyfield: Column whose values identify rows.
    dtypes: Type for each column of the dicts, before the change. It must name every key of the dicts.

Returns:
    The new Daf.

Raises:
    ValueError: `dtypes` is given and a dict has a key that it does not name. The keys would be
        matched to the wrong values. The message mentions `ignore_extra_keys`, which this method does not take.
        Name all the keys.

Examples:
    >>> d = Daf.from_lod_to_cols([{'A': 1, 'B': 2}, {'A': 4, 'B': 5}], cols=['Feature', 'T1', 'T2'])
    >>> d
    <BLANKLINE>
    | Feature | T1 | T2 |
    | ------: | -: | -: |
    |       A |  1 |  4 |
    |       B |  2 |  5 |
    <BLANKLINE>
    %% daf rows=2; cols=3; keyfield=''; name=''
    <BLANKLINE>



##### to_attrib_dict() -> dict

Make a dict with the column names and the rows.

This is deprecated. It keeps the rows as they are, not copied. Use
`to_lod()` or `to_cols_dol()` instead.

Returns:
    A dict with the keys `cols` and `lol`.

Examples:
    >>> Daf(lol=[[1, 'a']], cols=['id', 'v']).to_attrib_dict()
    {'cols': ['id', 'v'], 'lol': [[1, 'a']]}


### Values and lists


##### to_value(default: Any = _MISSING, astype: Callable | str | None = None) -> Any

Get the one value of a Daf that has one row and one column.

Use it on the result of a selection, as in `my_daf[1, 'n'].to_value()`.

Args:
    default: Returned if the Daf is not one cell. Without it, an error is raised.
    astype: A type or function to convert the value with.

Returns:
    The value.

Raises:
    ValueError: The Daf is not one cell and no default is given.

Examples:
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
    >>> d[1, 'n'].to_value()
    20



##### to_list(unique: bool = False, flatten: bool = False, omit_nulls: bool = False, default: Any = _MISSING, astype: Callable | str | type | None = None) -> list

Get the values of a Daf that has one row or one column, as a list.

Use it on the result of a selection, as in `my_daf[:, 'v'].to_list()`. A column
is read from the Daf, so use `col()` if you only need the list. An empty
Daf gives an empty list. A Daf with more than one row and more than one column
is not accepted.

Args:
    unique: If True, leave out repeated values and keep the order.
    flatten: If True, join items that are lists into one list.
    omit_nulls: If True, leave out the empty values.
    default: Replaces NULL, None and NaN values. This may itself be None.
    astype: A type or function to convert each value with.

Returns:
    The list.

Raises:
    ValueError: The Daf has more than one row and more than one column.

Examples:
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
    >>> d[:, 'v'].to_list()
    ['a', 'b', 'c']
    >>> d[1].to_list()
    [2, 'b', 20]



##### to_lota() -> T_lota

Make a list of tuples, one tuple for each row.

This is handy for making composite keys. The tuples are new, but the values in them
are the same objects as in the Daf.

Returns:
    The rows as tuples.

Examples:
    >>> Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v']).to_lota()
    [(1, 'a'), (2, 'b')]



##### to_dict() -> T_da

Get the one row of a Daf as a dict.

Use it on the result of a selection, as in `my_daf[1].to_dict()`. A column is
not turned into a dict. Use `to_list()` for that. An empty Daf gives an empty
dict. A Daf with no column names raises `KeysDisabledError`.

Returns:
    The row, as a dict that maps column names to values.

Raises:
    KeysDisabledError: The Daf has a row and no column names.
    ValueError: The Daf has more than one row.

Examples:
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
    >>> d[1].to_dict()
    {'id': 2, 'v': 'b', 'n': 20}



##### to_klist(irow: int = 0) -> KeyedList

Get one row as a [KeyedList][daffodil.keyedlist.KeyedList].

The KeyedList shares the row and the column names with the Daf, so it costs
little. A negative position counts from the end. A position that is out of range
raises `IndexError`. A Daf with no rows gives an empty KeyedList.

Args:
    irow: The row position.

Returns:
    The row.

Raises:
    IndexError: The position is out of range, for a Daf that has rows.

Examples:
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
    >>> d.to_klist(1)['v']
    'b'


### CSV, Excel and OpenDocument


##### from_csv(source: str | Path, kwargs: Any = {}) -> 'Daf'

Read a CSV file, a web address or an S3 object into a Daf.

The source may be a path, a `Path`, an `http` or `https` address, or an
`s3://bucket/key` name. The `requests` and `boto3` packages are imported only
when they are needed.

Every value is read as text. Call `apply_dtypes()` on the result to convert
them. Giving `dtypes` here also converts them, but that is deprecated. The other keyword arguments, such as `keyfield`, are those of
`from_csv_buff()`. The length of each row is not checked. See `from_csv_buff()`.

Args:
    source: A file path, an `http` address or an `s3://` name.
    **kwargs: Passed on to `from_csv_buff()`.

Returns:
    The new Daf.

Raises:
    RuntimeError: The download fails, a needed package is missing, or the
        local file cannot be read or parsed. The message says which.

Examples:
    >>> import os, tempfile
    >>> path = os.path.join(tempfile.mkdtemp(), 'x.csv')
    >>> with open(path, 'w') as f:
    ...     n = f.write('id,v\n1,a\n')
    >>> Daf.from_csv(path)
    <BLANKLINE>
    | id | v  |
    | -: | -: |
    |  1 |  a |
    <BLANKLINE>
    %% daf rows=1; cols=2; keyfield=''; name=''
    <BLANKLINE>



##### from_csv_buff(csv_buff: bytes | str | Iterator[str], keyfield: str = '', dtypes: T_dtype_dict | None = None, noheader: bool = False, user_format: bool = False, sep: str = ',', unflatten: bool = True, include_cols: T_ls | None = None, name: str = '') -> 'Daf'

Make a Daf from CSV text, bytes or an iterator of lines.

The first row holds the column names, unless `noheader` is True. Quoted fields
may hold commas. Empty rows at the end are dropped. Cells are text, unless
`dtypes` is given. Then the cells are converted, and list and dict columns
are read from their text unless `unflatten` is False.

Converting here is deprecated. It does no more than calling `apply_dtypes()` on the
result, which is clearer: a reader only reads, and the types are applied where they
are needed. Pass no `dtypes`, and call `apply_dtypes()` afterwards. The conversion
still works, for now.

Empty text, or a source with no rows, gives an empty Daf with no columns, as
`from_md()` does. Check `len()` of the result if an empty source would be an error.

A `keyfield` that is not a column of the file raises `KeyError`.

The length of each row is not checked, so that a read costs no more than it must.
A row with a missing cell, or with an extra one from an unquoted comma, is kept as
it is, and a later call may fail with an error that does not mention it. If you do
not trust the source, call `is_rectangular()`. It looks at every row. To pad short
rows, call `force_rectangular()`. It raises `ValueError` for a row that is too long.

Args:
    csv_buff: The CSV, as text, bytes or an iterator of lines.
    keyfield: Column, or tuple or list of columns, whose values identify rows.
    dtypes: Type for each column. Deprecated: call `apply_dtypes()` on the result instead.
    noheader: If True, the first row is data, and the Daf has no column names.
    user_format: If True, skip comment lines and blank lines.
    sep: The character that separates fields.
    unflatten: If True, read list and dict columns from their text.
    include_cols: Keep only these columns, in this order. The cells of the other
        columns are never kept, so a wide file costs little memory.
    name: Name of the new Daf.

Returns:
    The new Daf.

Raises:
    KeyError: A name in `include_cols`, or the `keyfield`, is not in the header of the file.
    ValueError: `include_cols` is given with `noheader=True`, because the names
        of the columns come from the header.

Examples:
    >>> d = Daf.from_csv_buff('id,v\n1,a\n2,"b,c"\n')
    >>> d
    <BLANKLINE>
    | id |  v  |
    | -: | --: |
    |  1 |   a |
    |  2 | b,c |
    <BLANKLINE>
    %% daf rows=2; cols=2; keyfield=''; name=''
    <BLANKLINE>
    >>> Daf.from_csv_buff('id,v\n1,a\n').apply_dtypes(dtypes={'id': int, 'v': str})
    <BLANKLINE>
    | id | v  |
    | -: | -: |
    |  1 |  a |
    <BLANKLINE>
    %% daf rows=1; cols=2; keyfield=''; name=''
    <BLANKLINE>
    >>> d = Daf.from_csv_buff('a,b,c\n1,2,3\n4,5,6\n', include_cols=['c', 'a'])
    >>> d
    <BLANKLINE>
    | c  | a  |
    | -: | -: |
    |  3 |  1 |
    |  6 |  4 |
    <BLANKLINE>
    %% daf rows=2; cols=2; keyfield=''; name=''
    <BLANKLINE>
    >>> d = Daf.from_csv_buff('id,v\n1,a\n2\n')
    >>> d.is_rectangular()
    False
    >>> d.force_rectangular()
    <BLANKLINE>
    | id | v  |
    | -: | -: |
    |  1 |  a |
    |  2 |    |
    <BLANKLINE>
    %% daf rows=2; cols=2; keyfield=''; name=''
    <BLANKLINE>



##### from_csv_file(filepath: str, keyfield: str = '', dtypes: T_dtype_dict | None = None, noheader: bool = False, user_format: bool = False, sep: str = ',', unflatten: bool = True, include_cols: T_ls | None = None, name: str = '') -> 'Daf'

Read a CSV file into a Daf. Deprecated, use `from_csv()`.

This now calls `from_csv()`, so it reads the file as UTF-8, and a file that cannot be
read raises `RuntimeError`. It used to print a message and return None, and to read
the file with the encoding of the machine.

Args:
    filepath: Path of the file.
    keyfield: Column, or tuple or list of columns, whose values identify rows.
    dtypes: Type for each column. Deprecated: call `apply_dtypes()` on the result instead.
    noheader: If True, the first row is data, and the Daf has no column names.
    user_format: If True, skip comment lines and blank lines.
    sep: The character that separates fields.
    unflatten: If True, read list and dict columns from their text.
    include_cols: Keep only these columns, in this order. The cells of the other
        columns are never kept, so a wide file costs little memory.
    name: Name of the new Daf.

Returns:
    The new Daf.

Raises:
    RuntimeError: The file cannot be read or parsed.

Examples:
    >>> import os, tempfile
    >>> path = os.path.join(tempfile.mkdtemp(), 'x.csv')
    >>> with open(path, 'w') as f:
    ...     n = f.write('id,v\n1,a\n')
    >>> Daf.from_csv_file(path)
    <BLANKLINE>
    | id | v  |
    | -: | -: |
    |  1 |  a |
    <BLANKLINE>
    %% daf rows=1; cols=2; keyfield=''; name=''
    <BLANKLINE>



##### to_csv_file(file_path: str | Path = '', line_terminator: str | None = None, include_header: bool = True) -> str

Write the Daf to a CSV file.

The file has a header row of column names unless `include_header` is False.
Each cell is written as text, so lists and dicts are written in their `str()`
form. A NULL cell is written as nothing. The default line ending is `\r\n`.

Args:
    file_path: Where to write the file.
    line_terminator: The line ending. If None, `\r\n` is used.
    include_header: If True, write the column names first.

Returns:
    The path that was written.

Examples:
    >>> import os, tempfile
    >>> path = os.path.join(tempfile.mkdtemp(), 'x.csv')
    >>> d = Daf(lol=[[1, 'a']], cols=['id', 'v'])
    >>> d.to_csv_file(path) == path
    True



##### to_csv_buff(line_terminator: str | None = None, include_header: bool = True) -> T_buff

Make CSV text from the Daf.

The text can be saved to a file or uploaded. There is no need to call
`flatten()` first. Each cell is written with `str()`, so a list or dict is
written as its Python text, with single quotes. The bool `True` is written as
`True`. A NULL cell is written as nothing. The text is not JSON.

A column converted by `apply_dtypes()` to `bool` holds 0 and 1, so it is written
as 0 and 1. A column of Python bools is written as `True` and `False`. Both read
back as 0 and 1 when the column has dtype `bool`. The cells are written as they
are held, with no conversion, since any pass over them would slow every save.

A dict whose keys are not text, such as `{1: 'a'}`, is written with its keys as they are.

Args:
    line_terminator: The line ending. If None, `\r\n` is used.
    include_header: If True, write the column names first.

Returns:
    The CSV text.

Examples:
    >>> d = Daf(lol=[[1, 'a']], cols=['id', 'v'])
    >>> d.to_csv_buff(line_terminator='\n')
    'id,v\n1,a\n'
    >>> Daf(lol=[[{1: 'a'}]], cols=['x']).to_csv_buff(line_terminator='\n')
    "x\n{1: 'a'}\n"



##### buff_to_file(buff: T_buff, file_path: str | Path, fmt: str = '.csv') -> str

Write text or bytes to a file.

This is a static method, so call it as `Daf.buff_to_file(buff, path)`. It is
the helper that `to_csv_file()` uses.

Args:
    buff: The text or bytes to write.
    file_path: Where to write.
    fmt: The file format, such as `.csv`.

Returns:
    The path that was written.

Examples:
    >>> import os, tempfile
    >>> path = os.path.join(tempfile.mkdtemp(), 'x.csv')
    >>> Daf.buff_to_file('id,v\n1,a\n', path) == path
    True
    >>> open(path).read()
    'id,v\n1,a\n'



##### from_excel_buff(excel_buff: bytes, keyfield: str = '', dtypes: T_dtype_dict | None = None, noheader: bool = False, user_format: bool = False, unflatten: bool = True) -> 'Daf'

Make a Daf from the bytes of an xlsx file.

The first sheet is turned into CSV by the `xlsx2csv` package. The CSV is then
read like any CSV text. All values start as text, so give `dtypes` or call
`apply_dtypes()` to convert them.

Short rows are not padded here. `xlsx2csv` pads them. See `xlsx_to_csv()`.

Args:
    excel_buff: The bytes of the xlsx file.
    keyfield: Column whose values identify rows.
    dtypes: Type for each column. Deprecated: call `apply_dtypes()` on the result instead.
    noheader: If True, the first row is data, not column names.
    user_format: If True, skip comment lines and blank lines.
    unflatten: If True, read list and dict columns from their text.

Returns:
    The new Daf.

Examples:
    >>> import io, xlsxwriter
    >>> buff = io.BytesIO()
    >>> workbook = xlsxwriter.Workbook(buff, {'in_memory': True})
    >>> sheet = workbook.add_worksheet()
    >>> for irow, row in enumerate([['id', 'v'], [1, 'a'], [2, 'b']]):
    ...     for icol, value in enumerate(row):
    ...         _ = sheet.write(irow, icol, value)
    >>> workbook.close()
    >>> d = Daf.from_excel_buff(buff.getvalue(), keyfield='id')
    >>> d
    <BLANKLINE>
    | id | v  |
    | -: | -: |
    |  1 |  a |
    |  2 |  b |
    <BLANKLINE>
    %% daf rows=2; cols=2; keyfield='id'; name=''
    <BLANKLINE>
    >>> Daf.from_excel_buff(buff.getvalue(), dtypes={'id': int, 'v': str})
    <BLANKLINE>
    | id | v  |
    | -: | -: |
    |  1 |  a |
    |  2 |  b |
    <BLANKLINE>
    %% daf rows=2; cols=2; keyfield=''; name=''
    <BLANKLINE>



##### from_ods(source: str | Path | bytes, sheet: int | str = 0, keyfield: str = '', dtypes: T_dtype_dict | None = None, noheader: bool = False, unflatten: bool = True, name: str = '') -> 'Daf'

Make a Daf from one sheet of an OpenDocument spreadsheet, an .ods file.

Use it for a table kept in LibreOffice Calc. Only the first sheet is read, unless
you give `sheet`. Only the content of the cells is read: styles, comments, links
and formulas are left out. No package is needed, since the file is read with the
standard library. See `lib/daf_ods.py`.

All values start as text, as from a CSV file, so give `dtypes` or call
`apply_dtypes()` to convert them. A number, date or boolean cell gives its stored
value, such as `168.5` or `2026-09-01`, not the text shown, such as `$168.50`.
A cell with several lines gives them joined with a newline.

Rows are padded with NULL to the width of the widest row. A blank row inside the
table is kept as a row of NULL, so the rows line up with the spreadsheet. Drop such
rows yourself if they must not be used. The blank rows at the end are left out.

Args:
    source: The path of the file, or its bytes.
    sheet: The position of the sheet, from 0, or its name.
    keyfield: Column whose values identify rows.
    dtypes: Type for each column. Deprecated: call `apply_dtypes()` on the result instead.
    noheader: If True, the first row is data, not column names.
    unflatten: If True, read list and dict columns from their text.
    name: The name of the new Daf.

Returns:
    The new Daf.

Raises:
    IndexError: There is no sheet at that position.
    KeyError: There is no sheet with that name.


### JSON


##### to_json(concise: bool = True) -> str

Make JSON text that holds the whole Daf.

The text holds the rows, the column names, the dtypes, the keyfield, the name,
the attrs and the display columns. With `concise=True`, the parts that are
empty are left out. Use `from_json()` to read it back.

Types are written by name. Only `int`, `float`, `str`, `bool`, `list` and `dict`
are read back as types. Other names, such as `date`, come back as text.

Cells must be JSON values. A tuple comes back as a list, and a set raises a
`TypeError`. A NaN is written as `NaN`, which some JSON readers reject.

Args:
    concise: If True, leave out the parts that are empty.

Returns:
    The JSON text.

Examples:
    >>> Daf(lol=[[1]], cols=['a']).to_json()
    '{"lol": [[1]], "hd": {"a": 0}}'



##### from_json(json_str: str) -> 'Daf'

Make a Daf from JSON text made by `to_json()`.

Args:
    json_str: The JSON text.

Returns:
    The new Daf.

Examples:
    >>> d = Daf(lol=[[1, 'a']], cols=['id', 'v'], keyfield='id')
    >>> Daf.from_json(d.to_json()) == d
    True


### Markdown


##### from_md(md_str: str) -> 'Daf'

Make a Daf from a Markdown table. This reads what `to_md()` writes.

Only the first table in the text is read. Text before it, such as headings and
prose, is skipped. A table row starts and ends with `|`. The table must have
a header row and then a separator row such as `| -: | -: |`. The columns are
the header cells. A table with no header row is not read, and raises `RuntimeError`.

A Daf with rows and no column names is written by `to_md()` with the spreadsheet
names `A`, `B` and so on in the header. Reading that text gives a Daf with those
names, so the round trip adds column names. The text of a table cannot tell these
names from names that you chose.

A footer line that starts with `%% daf ` and follows the table may give the
keyfield and name, as `to_md(include_summary=True)` writes. Every cell is read as
text. Convert them with `apply_dtypes()`. Text with no table, or empty text,
gives an empty Daf.

Markdown is a report format, so a round trip through `to_md()` is not exact:

- Text that `to_md()` shortened, rows and columns it left out, and numbers that
  `smart_fmt` rounded stay that way.
- A newline in a cell is written as `<br>`, and is read back as `<br>`.
- A `|` in a cell is written as `\|`, which this method does not read. The table ends
  at the row before it, with no error. A `|` in the first row gives no rows at all.
- A composite keyfield in the footer is not read back, and raises `KeyError`.

Args:
    md_str: The Markdown text.

Returns:
    The new Daf.

Raises:
    RuntimeError: A table is found but it has no header row and separator row.

Examples:
    >>> from daffodil.daf import Daf
    >>> text = "| id | v |\n| -: | -: |\n|  1 | a |\n|  2 | b |\n\n%% daf rows=2; cols=2; keyfield='id'; name='nm'\n"
    >>> d = Daf.from_md(text)
    >>> d
    <BLANKLINE>
    | id | v  |
    | -: | -: |
    |  1 |  a |
    |  2 |  b |
    <BLANKLINE>
    %% daf rows=2; cols=2; keyfield='id'; name='nm'
    <BLANKLINE>
    >>> Daf.from_md(Daf(lol=[[1, 'a']]).to_md()).columns()
    ['A', 'B']



##### dodaf_to_md(dodaf: Dict[str, Any], report_header: str = '', to_md_kwargs: Any = {}) -> str

Make one Markdown report from a dict of Daf instances.

This is a static method. The report has an optional first level heading, then
a second level section for each key, in the order of the dict. Each section
holds that Daf as a table, from its own `to_md()`. A Daf with no rows still gets
its section, with a line `*(no rows)*`, so you can see which sections were empty.
Use `dodaf_from_md()` to read the report back.

Args:
    dodaf: A dict that maps a section name to a Daf.
    report_header: The title of the report. Left out if empty.
    **to_md_kwargs: Passed on to the `to_md()` of each Daf, such as `max_rows`.

Returns:
    The Markdown text.

Examples:
    >>> from daffodil.daf import Daf
    >>> print(Daf.dodaf_to_md({'first': Daf(lol=[[1]], cols=['a'])}, report_header='Report'))
    # Report
    <BLANKLINE>
    <BLANKLINE>
    ## first
    <BLANKLINE>
    <BLANKLINE>
    | a  |
    | -: |
    |  1 |
    <BLANKLINE>
    <BLANKLINE>



##### dodaf_from_md(md_str: str, header_level: int = 2) -> Dict[str, Any]

Read a report made by `dodaf_to_md()` back into a dict of Daf instances.

A section starts at a line that begins with the marker of `header_level`,
which is `## ` by default. Text before the first section, such as the report
title, is skipped. Each section becomes a Daf, read with `from_md()`, so the
cells are text. The section title is the dict key and the `name` of the Daf, unless the table's
footer gives a name. The footer wins for the name, never for the key. Empty
text gives an empty dict.

Args:
    md_str: The Markdown text.
    header_level: The number of `#` that start a section heading.

Returns:
    A dict that maps each section title to a Daf.

Examples:
    >>> from daffodil.daf import Daf
    >>> report = Daf.dodaf_to_md({'first': Daf(lol=[[1]], cols=['a'])})
    >>> back = Daf.dodaf_from_md(report)
    >>> back['first']
    <BLANKLINE>
    | a  |
    | -: |
    |  1 |
    <BLANKLINE>
    %% daf rows=1; cols=1; keyfield=''; name='first'
    <BLANKLINE>


### Pandas and NumPy


##### from_pandas_df(df: T_df, keyfield: str = '', name: str = '', use_csv: bool = False, dtypes: T_dtype_dict | None = None) -> Daf

Make a Daf from a Pandas DataFrame or Series.

The values become plain Python values. The column names come from the
DataFrame. The index is not kept. The dtypes are worked out from the Pandas
dtypes, but the values are not converted to them. A Series becomes one row,
with the index labels as column names. Its dtypes are the Python types of its
values, one for each label.

With `use_csv=True` the DataFrame is turned into CSV text first, and that is
read back. This can be faster for some frames.

Args:
    df: The DataFrame or Series.
    keyfield: Column, or tuple or list of columns, whose values identify rows.
    name: Name of the new Daf.
    use_csv: If True, convert by way of CSV text.
    dtypes: Deprecated, and it is not used. The dtypes come from `df`. It will be removed. A
        `DeprecationWarning` is given if it is passed.

Returns:
    The new Daf.

Examples:
    >>> import pandas as pd
    >>> from daffodil.daf import Daf
    >>> df = pd.DataFrame({'id': [1, 2], 'v': ['a', 'b']})
    >>> d = Daf.from_pandas_df(df, keyfield='id')
    >>> d
    <BLANKLINE>
    | id | v  |
    | -: | -: |
    |  1 |  a |
    |  2 |  b |
    <BLANKLINE>
    %% daf rows=2; cols=2; keyfield='id'; name=''
    <BLANKLINE>



##### to_pandas_df(cols: T_ls | None = None, use_csv: bool = False, use_donpa: bool = False, default: Any = _MISSING, defaulting_cols: T_ls | None = None) -> Any

Make a Pandas DataFrame from the Daf.

Daffodil stores a missing value as an empty string. Pandas treats that as
text, so a column with empty cells is of type `object`, even if the rest are
numbers. Text that looks like a number stays text. Convert columns afterwards
if you want other types.

Giving `default` replaces empty and None cells in the Daf itself, before the
conversion. That changes the Daf you called it on. Copy it first if you need
it unchanged. `default` cannot be used with `use_csv`. With `use_donpa`, the default
replaces the cells in the arrays only. The Daf is not changed, and it goes to every
included column, as `defaulting_cols` is not used on that route.

Args:
    cols: Names, or positions, of the columns to include. If None, all columns.
    use_csv: If True, convert by way of CSV text.
    use_donpa: If True, convert by way of `to_donpa()`. This suits numeric columns.
    default: Value that replaces empty and None cells. This changes the Daf, except with `use_donpa`.
    defaulting_cols: The columns that `default` applies to. If None, all included columns.

Returns:
    The DataFrame.

Raises:
    NotImplementedError: `default` is given with `use_csv=True`.

Examples:
    >>> from daffodil.daf import Daf
    >>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'])
    >>> d.to_pandas_df().values.tolist()
    [[1, 'a'], [2, 'b']]



##### from_numpy(npa: Any, keyfield: str = '', cols: T_la | None = None, name: str = '') -> 'Daf'

Make a Daf from a NumPy array.

The values become plain Python values. A two dimensional array gives one row
for each row of the array. A one dimensional array gives a single row.

An array has one dtype. `np.array([[1, 'x']])` makes every value text, before this
method sees it, so the Daf gets text too. An array with `dtype=object` keeps each
value as the Python object it was, such as an int, a str or None.

Args:
    npa: The NumPy array.
    keyfield: Column, or tuple or list of columns, whose values identify rows.
    cols: Column names.
    name: Name of the new Daf.

Returns:
    The new Daf.

Examples:
    >>> import numpy as np
    >>> Daf.from_numpy(np.array([[1, 2], [3, 4]]), cols=['a', 'b'])
    <BLANKLINE>
    | a  | b  |
    | -: | -: |
    |  1 |  2 |
    |  3 |  4 |
    <BLANKLINE>
    %% daf rows=2; cols=2; keyfield=''; name=''
    <BLANKLINE>



##### to_numpy() -> T_npa

Make a NumPy array of the rows.

Column names, the keyfield and any dtypes are not kept. NumPy picks one type
for the whole array. If the Daf mixes numbers and text, every value becomes
text.

Returns:
    The NumPy array.

Examples:
    >>> Daf(lol=[[1, 2.5]], cols=['a', 'b']).to_numpy().tolist()
    [[1.0, 2.5]]



##### to_donpa(colnames: T_ls | None = None, default: Any = _MISSING) -> T_donpa

Make a dict of NumPy arrays, one array for each column.

This is a light form of a DataFrame. The arrays can be used in vector
operations, such as `donpa['D_pct'] = donpa['D_votes'] / donpa['RV_total']`.
Each column gets its own type, so numbers and text can be mixed.

Args:
    colnames: The columns to include. If None, all columns.
    default: A value that replaces each NULL, None and NaN cell, in the arrays only.
        The Daf is not changed. It applies to every column in `colnames`, so
        give only the numeric columns, or a text column gets it too.

Returns:
    A dict that maps each column name to an array.

Notes:
    A column that mixes numbers and NULL becomes a text array, because NumPy
    has one type for an array. Give a numeric `default` to keep it numeric.
    A column with a `default` is also faster to make than one with blanks and
    no default, because the array is built from numbers.

Raises:
    ColumnNotFoundError: A name in `colnames` is not a column. This is a `KeyError`,
        and also a `RuntimeError`.

Examples:
    >>> Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v']).to_donpa(['id'])['id'].tolist()
    [1, 2]
    >>> d = Daf(lol=[[1, 5], [2, ''], [None, 7]], cols=['n', 'm'])
    >>> d.to_donpa(['m'])['m'].tolist()
    ['5', '', '7']
    >>> d.to_donpa(['n', 'm'], default=0)['m'].tolist()
    [5, 0, 7]
    >>> d
    <BLANKLINE>
    |  n   | m  |
    | ---: | -: |
    |    1 |  5 |
    |    2 |    |
    | None |  7 |
    <BLANKLINE>
    %% daf rows=3; cols=2; keyfield=''; name=''
    <BLANKLINE>


### Other sources


##### from_directory(source: str | Path, schema: type | None = None, recursive: bool = True, file_pat: str | None = None, include_dirs: bool = False) -> 'Daf'

Make a Daf that lists the files in a folder.

Each row describes one file. The columns are the path, the folder, the name,
the name without its extension, the extension, the size in bytes, and the
modified and changed times. On Linux and macOS the `ctime` is the time of the last
change to the file's metadata, such as its permissions, not the time it was
created. Paths use `/` on every system.

With `include_dirs=True`, the folders are listed too, each before the files in the
same folder. A folder has an `is_dir` of 1, a size of 0 and no extension. Without
it, no folder is listed and `is_dir` is always 0.

The rows are in the order in which the file system lists the files. That order is
not sorted, and it differs between machines. Sort the result if the order matters.

The `schema` chooses the columns of the result. A `@schemaclass` that lists
only some of the fields above keeps only those. Other columns of the schema
get their defaults. The new Daf has no keyfield. Files that cannot be read
are skipped.

Only local folders are supported.

Args:
    source: The folder to list.
    schema: A `@schemaclass` that chooses the columns. If None, the built in one is used.
    recursive: If True, list files in all sub folders. Otherwise only the folder itself.
    file_pat: A regular expression. Only names that match it are listed. Case is ignored.
    include_dirs: If True, list the folders as well as the files.

Returns:
    The new Daf.

Examples:
    >>> import os, tempfile
    >>> folder = tempfile.mkdtemp()
    >>> os.mkdir(os.path.join(folder, 'sub'))
    >>> for name in ('b.txt', 'a.csv', os.path.join('sub', 'c.csv')):
    ...     with open(os.path.join(folder, name), 'w') as f:
    ...         n = f.write('hello')
    >>> d = Daf.from_directory(folder)
    >>> sorted(d.col('basename')), d.col('size')
    (['a.csv', 'b.txt', 'c.csv'], [5, 5, 5])
    >>> sorted(Daf.from_directory(folder, recursive=False).col('basename'))
    ['a.csv', 'b.txt']
    >>> sorted(Daf.from_directory(folder, file_pat=r'\.csv$').col('basename'))
    ['a.csv', 'c.csv']
    >>> sorted(Daf.from_directory(folder, include_dirs=True).col('basename'))
    ['a.csv', 'b.txt', 'c.csv', 'sub']



##### from_pdf

A constant: from_pdf


`from_pdf()` reads one kind of county report, of election participation by precinct, from a
PDF file, with pdfplumber. It is experimental and written for that report. Its interface may change, so it
is not documented here.


##### from_googlesheet(spreadsheet_id: str, sheetname: str = 'Sheet1', service_account_file: str) -> 'Daf'

Read a Google Sheet into a Daf. This is not implemented.

It always raises `NotImplementedError`. It holds the place of the interface, which
takes the path of a service account file.

Args:
    spreadsheet_id: The ID of the Google Sheet.
    sheetname: The name of the sheet.
    service_account_file: The path of the Google service account file with the credentials.

Raises:
    NotImplementedError: Always.

Examples:
    >>> Daf.from_googlesheet('some-id', service_account_file='key.json')
    Traceback (most recent call last):
        ...
    NotImplementedError: from_googlesheet() is not implemented yet.



##### to_googlesheet(spreadsheet_id: str, sheetname: str = 'Sheet1', service_account_file: str) -> 'Daf'

Write the Daf to a Google Sheet. This is not implemented.

It always raises `NotImplementedError`. It holds the place of the interface, which
takes the path of a service account file.

Args:
    spreadsheet_id: The ID of the Google Sheet.
    sheetname: The name of the sheet.
    service_account_file: The path of the Google service account file with the credentials.

Raises:
    NotImplementedError: Always.

Examples:
    >>> Daf(cols=['x'], lol=[[1]]).to_googlesheet('some-id', service_account_file='key.json')
    Traceback (most recent call last):
        ...
    NotImplementedError: to_googlesheet() is not implemented yet.

---

# Docs page: Daf API / Selecting and indexing (api/daf/select.md)

## Selecting and indexing

Reading rows, columns and cells by index, key, or condition.

### Indexing with brackets


##### __getitem__(slice_spec: slice | int | str | T_li | T_ls | range | T_lor | Tuple[slice | int | str | T_li | T_ls | range | T_lor | Tuple[str, str], slice | int | str | T_li | T_ls | range | T_lor | Tuple[str, str]]) -> Any

Select rows, columns or cells, as in `my_daf[rows, cols]`.

The selector is `[rows]` or `[rows, cols]`. With one selector, all columns are
returned. Use `:` for all. A selector is one of these.

    integer       A position. Negative counts from the end.
    slice         `2:5`, `:3`, `::2`, as for a Python list.
    list          A list of positions, in the order given, or a list of ranges.
    range         A range of positions.
    string        A key of the keyfield for rows, or a column name for columns.
    list of str   Several keys, or several column names, in the order given.
    tuple         An inclusive range of keys, or of column names. Use None to
                  start at the first or to end at the last, as in `(None, 'r3')`.

A tuple of two items is read as `[rows, cols]` when it stands alone. To give a
range of row keys, add the column selector, as in `my_daf[('r1', 'r3'), :]`.

Integers are always positions. A keyfield or column names that are integers
cannot be used in brackets. Use `select_krows()` and `select_kcols()` instead.

The result is a new Daf. Its rows are shared with this Daf when you select
rows, so changing a cell in the result changes it here too. Selecting columns
makes new row lists, so replacing a cell in the result does not change this Daf.
A list or dict held in a cell is still the same object in both. Use `copy('deep')`
when those must be separate too. The keyfield and dtypes carry
over if their columns are still there. See [retmode][daffodil.daf.Daf.retmode]
for getting a bare value or list when the result is one cell, row or column.

A column slice works as it does for a Python list.

Args:
    slice_spec: A row selector, or a tuple of a row selector and a column selector.

Returns:
    A new Daf, or a value or list if `retmode` is `val`.

Raises:
    IndexError: A row or column position is out of range.
    KeyError: A key or column name is not found.
    KeysDisabledError: Rows are selected by key and there is no keyfield.
    TypeError: A selector is None, or has a type that is not accepted.

Examples:
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
    >>> d[1]
    <BLANKLINE>
    | id | v  | n  |
    | -: | -: | -: |
    |  2 |  b | 20 |
    <BLANKLINE>
    %% daf rows=1; cols=3; keyfield='id'; name=''
    <BLANKLINE>
    >>> d[1:]
    <BLANKLINE>
    | id | v  | n  |
    | -: | -: | -: |
    |  2 |  b | 20 |
    |  3 |  c | 30 |
    <BLANKLINE>
    %% daf rows=2; cols=3; keyfield='id'; name=''
    <BLANKLINE>
    >>> d[:, 'v']
    <BLANKLINE>
    | v  |
    | -: |
    |  a |
    |  b |
    |  c |
    <BLANKLINE>
    %% daf rows=3; cols=1; keyfield=''; name=''
    <BLANKLINE>
    >>> d[[2, 0], ['n', 'id']]
    <BLANKLINE>
    | n  | id |
    | -: | -: |
    | 30 |  3 |
    | 10 |  1 |
    <BLANKLINE>
    %% daf rows=2; cols=2; keyfield='id'; name=''
    <BLANKLINE>
    >>> d[1, 'n'].to_value()
    20
    >>> d[(1, 2), :]
    <BLANKLINE>
    | id | v  | n  |
    | -: | -: | -: |
    |  1 |  a | 10 |
    |  2 |  b | 20 |
    <BLANKLINE>
    %% daf rows=2; cols=3; keyfield='id'; name=''
    <BLANKLINE>



##### __setitem__(slice_spec: slice | int | str | range | T_li | T_ls | T_lb | T_lor | Tuple[slice | int | str | range | T_lor | T_li | T_ls | T_lb | Tuple[Any, Any], slice | int | str | range | T_lor | T_li | T_ls | T_lb | Tuple[Any, Any]], value: Any) -> 'Daf'

Assign values to a selection, as in `my_daf[rows, cols] = value`.

The selector is the same as for `[]`. The Daf is changed in place.

A single value fills every cell of the selection. A list fills the selection
in order. A dict given for a row sets that row from the dict. The cells whose
columns are not in the dict become NULL. See `set_irows_icols()` for what
happens when the source and the selection differ in size.

A `str` or `bytes` is one value, not a list of characters, so
`my_daf[:, 'v'] = 'xyz'` sets every row of `v` to `xyz`. A list assigned to
several whole rows is copied, so each row has its own list.

A write does not update the key index (kd). Writing into a keyfield cell, or
writing a whole row, which includes the key, leaves the kd as it was: a lookup
still finds the row by its old key, and not by the new one. So don't change keys
this way. If you must, call `set_keyfield()` with the same keyfield afterwards,
which invalidates the kd, so it is rebuilt when it is next needed. Writing other
columns is safe.

A selection that shares the row has a key index of its own. Resetting one Daf does
not reset the other. Call `set_keyfield()` on each Daf that will look up keys.

A single name, as in `my_daf['x'] = value`, selects the row with that key. It is
not a column, as it would be in pandas. Use `my_daf[:, 'x']` for a column, or
`assign_col()` to add one. A name that is not a row key raises an error that
says this.

Args:
    slice_spec: A row selector, or a tuple of a row selector and a column selector.
    value: The value, list, dict or Daf to assign.

Raises:
    KeyError: A row key or a column name is not found.
    KeysDisabledError: A row is selected by key, and the Daf has no keyfield.

Examples:
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
    >>> d[1, 'v'] = 'z'
    >>> d[:, 'n'] = [1, 2, 3]
    >>> d
    <BLANKLINE>
    | id | v  | n  |
    | -: | -: | -: |
    |  1 |  a |  1 |
    |  2 |  z |  2 |
    |  3 |  c |  3 |
    <BLANKLINE>
    %% daf rows=3; cols=3; keyfield='id'; name=''
    <BLANKLINE>
    >>> d[0] = {'v': 'q'}
    >>> d.iloc(0)
    {'id': '', 'v': 'q', 'n': ''}

    A key changed after the index is built is found only after `set_keyfield()`:

    >>> k = Daf(cols=['id', 'q'], lol=[['a', 1], ['b', 2]], keyfield='id')
    >>> k.select_record('a')
    {'id': 'a', 'q': 1}
    >>> k[0, 'id'] = 'z'
    >>> k.select_record('z')
    {}
    >>> _ = k.set_keyfield(k.keyfield)
    >>> k.select_record('z')
    {'id': 'z', 'q': 1}



##### set_irows_icols(irows: slice | int | range | T_li | Iterable | None, icols: slice | int | range | T_li | None, value: Any) -> 'Daf'

Set values at row positions and column positions, in place.

This is what `my_daf[rows, cols] = value` calls, after the selectors are turned
into positions. `irows` and `icols` are an integer, a slice, a range or a list.
If `icols` is None, the whole row is set. If `irows` is None, nothing is
set. Use `slice(None)` for all rows.

What is set depends on the value.

    a single value    fills every selected cell. A str or bytes is a single value.
    a list            fills a selection in order. For several whole rows, each
                      row becomes a copy of the list.
    a dict            sets a row. The cells of columns that the dict lacks become NULL.
    a Daf             is copied as a block, from its top left corner.

Nothing is checked, and no error is raised for a size mismatch. A smaller source
fills the top left of the selection and leaves the rest. A larger one fills the
selection and the extra values are ignored. Check the sizes first if it matters.

Args:
    irows: The row positions.
    icols: The column positions.
    value: The value, list, dict or Daf to set.

Returns:
    This Daf, which has been changed.

Examples:
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'])
    >>> d.set_irows_icols([0, 1], [1, 2], 'Z')
    <BLANKLINE>
    | id | v  | n  |
    | -: | -: | -: |
    |  1 |  Z |  Z |
    |  2 |  Z |  Z |
    |  3 |  c | 30 |
    <BLANKLINE>
    %% daf rows=3; cols=3; keyfield=''; name=''
    <BLANKLINE>
    >>> d.set_irows_icols(2, None, ['x', 'y', 'z']).iloc(2)
    {'id': 'x', 'v': 'y', 'n': 'z'}


### Rows


##### select_krows(krows: slice | str | T_la | int | T_lota | Tuple[Any, Any] | Iterable | None, inverse: bool = False, silent_error: bool = False) -> 'Daf'

Select rows by key. Rows with those keys are kept, or dropped if `inverse` is True.

This is the same as `my_daf[keys]`, with a choice to drop rows and to ignore
keys that are not found. It works with integer keys, which brackets cannot.
The result is a new Daf with its own outer row list. Each row is the same list as
in this Daf. Use `copy('editable')` before changing cells in one without the other.

A bare tuple means an inclusive range of keys, so `(1, 2)` is the rows from key
1 through key 2. For a composite keyfield, give a list of tuples.

Args:
    krows: A key, a list of keys, or a tuple that gives a range of keys.
    inverse: If True, drop the selected rows and keep the others.
    silent_error: If True, keys that are not found are ignored.

Returns:
    The new Daf.

Raises:
    KeysDisabledError: The Daf has no keyfield and no key index.
    KeyError: A key is not found and `silent_error` is False.

Examples:
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
    >>> d.select_krows([3, 1])
    <BLANKLINE>
    | id | v  | n  |
    | -: | -: | -: |
    |  3 |  c | 30 |
    |  1 |  a | 10 |
    <BLANKLINE>
    %% daf rows=2; cols=3; keyfield='id'; name=''
    <BLANKLINE>
    >>> d.select_krows([1, 2], inverse=True)
    <BLANKLINE>
    | id | v  | n  |
    | -: | -: | -: |
    |  3 |  c | 30 |
    <BLANKLINE>
    %% daf rows=1; cols=3; keyfield='id'; name=''
    <BLANKLINE>
    >>> d.select_krows([1, 9], silent_error=True)
    <BLANKLINE>
    | id | v  | n  |
    | -: | -: | -: |
    |  1 |  a | 10 |
    <BLANKLINE>
    %% daf rows=1; cols=3; keyfield='id'; name=''
    <BLANKLINE>



##### select_irows(irows: slice | int | T_li | range | T_lor | Iterable | None, inverse: bool = False, invert: bool = False) -> 'Daf'

Select rows by position. Those rows are kept, or dropped if `inverse` is True.

This is the same as `my_daf[rows]`, with a choice to drop rows. It is cheap.
The result is a new Daf with its own outer row list. Each row is the same list as
in this Daf, so changing a cell in the result
changes it here too. This is also so when the selection is empty and `inverse`
is True, which keeps all the rows. The keyfield, dtypes and column names carry over.

Args:
    irows: A position, a slice, a range, a list of positions, or a list of ranges.
    inverse: If True, drop the selected rows and keep the others.
    invert: The old name of `inverse`, still accepted.

Returns:
    The new Daf.

Raises:
    IndexError: A single position is out of range.

Examples:
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
    >>> d.select_irows(1)
    <BLANKLINE>
    | id | v  | n  |
    | -: | -: | -: |
    |  2 |  b | 20 |
    <BLANKLINE>
    %% daf rows=1; cols=3; keyfield='id'; name=''
    <BLANKLINE>
    >>> d.select_irows(1, inverse=True)
    <BLANKLINE>
    | id | v  | n  |
    | -: | -: | -: |
    |  1 |  a | 10 |
    |  3 |  c | 30 |
    <BLANKLINE>
    %% daf rows=2; cols=3; keyfield='id'; name=''
    <BLANKLINE>
    >>> d.select_irows(1, invert=True)
    <BLANKLINE>
    | id | v  | n  |
    | -: | -: | -: |
    |  1 |  a | 10 |
    |  3 |  c | 30 |
    <BLANKLINE>
    %% daf rows=2; cols=3; keyfield='id'; name=''
    <BLANKLINE>
    >>> d.select_irows(slice(1, None))
    <BLANKLINE>
    | id | v  | n  |
    | -: | -: | -: |
    |  2 |  b | 20 |
    |  3 |  c | 30 |
    <BLANKLINE>
    %% daf rows=2; cols=3; keyfield='id'; name=''
    <BLANKLINE>



##### irow(irow: int = 0, include_cols: T_ls | None = None) -> T_da

Get one row as a dict, by position.

This is `iloc()` with the default `rtype`.

Args:
    irow: The row position.
    include_cols: Only these columns are included.

Returns:
    The row as a dict. A negative position counts from the end. A Daf with no rows
    gives an empty dict.

Raises:
    IndexError: The position is out of range, for a Daf that has rows.

Examples:
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
    >>> d.irow(1, include_cols=['n'])
    {'n': 20}



##### irow_la(irow: int) -> T_la

Get one row as a list, by position.

The list is the row itself, not a copy. Changing it changes the Daf. Use
`iloc()` with `rtype='list'` for a copy.

Args:
    irow: The row position.

Returns:
    The row.

Raises:
    IndexError: The position is out of range.

Examples:
    >>> d = Daf(lol=[['r1', 'a', 10], ['r2', 'b', ''], ['r3', 'a', 30]], cols=['id', 'v', 'n'], keyfield='id')

    >>> d.irow_la(1)
    ['r2', 'b', '']
    >>> row = d.irow_la(0)
    >>> row[1] = 'Z'
    >>> d.iloc(0)
    {'id': 'r1', 'v': 'Z', 'n': 10}
    >>> d.irow_la(9)
    Traceback (most recent call last):
        ...
    IndexError: list index out of range



##### iloc(irow: int = 0, include_cols: T_ls | None = None, rtype: str = 'dict') -> T_ma | T_la

Get one row by position, as a dict, a KeyedList or a list.

A negative position counts from the end, so `-1` is the last row. A position that
is out of range raises `IndexError`. A Daf with no rows, and a row with no cells,
give an empty dict, or an empty KeyedList. A dict or a KeyedList needs column
names, so a Daf that has rows and no column names raises `KeysDisabledError`. Name
them with `set_cols()`, or ask for `rtype='list'`.

Args:
    irow: The row position.
    include_cols: Only these columns are included. This applies to a dict.
    rtype: `dict` for a new dict, `klist` for a KeyedList that shares the row, or `list` for a copy of the row.

Returns:
    The row.

Raises:
    IndexError: The position is out of range, for a Daf that has rows.
    KeysDisabledError: `rtype` is `dict` or `klist`, and the Daf has rows and no column names.
    ValueError: `rtype` is not one of the three names.

Examples:
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
    >>> d.iloc(1)
    {'id': 2, 'v': 'b', 'n': 20}
    >>> d.iloc(1, rtype='list')
    [2, 'b', 20]
    >>> d.iloc(-1)
    {'id': 3, 'v': 'c', 'n': 30}



##### select_record(key: str | int | T_ta, silent_error: bool = True) -> T_da

Get one row as a dict, by its key.

If the key is not found, the answer is an empty dict, unless `silent_error` is
False. Then a `KeyError` is raised. A typo in a key gives an empty dict, so
check it, or pass `silent_error=False`. An empty Daf gives an empty dict.
For a composite keyfield the key is a tuple.

Args:
    key: The key of the row.
    silent_error: If False, raise an error when the key is not found.

Returns:
    The row as a dict, or an empty dict.

Raises:
    KeysDisabledError: The Daf has rows but no keyfield and no key index.
    KeyError: The key is not found and `silent_error` is False.

Examples:
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
    >>> d.select_record(2)
    {'id': 2, 'v': 'b', 'n': 20}
    >>> d.select_record(9)
    {}



##### select_records_daf(keys_ls: 'T_ls | Iterable | Daf', inverse: bool = False, silent_error: bool = False) -> 'Daf'

Select several rows by key and return them as a Daf.

This is `select_krows()` with a friendlier answer for an empty list of keys.
No keys gives an empty Daf, or all the rows if `inverse` is True.

The keys may be another Daf. Its keys are then used, through a view of its key index,
so no list of them is made. That other Daf must have a keyfield. Its rows are not
changed. The keys come in the order of its key index, so the rows selected come in
that order too. A key that repeats in this Daf selects only its last row. See `keys()`.

The rows are shared with this Daf, as in `select_krows()`. The new Daf has its own
row list, also with no keys and `inverse` True, so adding a row to one does not
add it to the other.

Args:
    keys_ls: The keys of the rows, or a Daf whose keys are used.
    inverse: If True, drop the selected rows and keep the others.
    silent_error: If True, keys that are not found are ignored.

Returns:
    The new Daf.

Raises:
    KeysDisabledError: This Daf, or a Daf given as the keys, has no keyfield and no key index.
    KeyError: A key is not found and `silent_error` is False.

Examples:
    >>> d = Daf(lol=[['r1', 'a', 10], ['r2', 'b', ''], ['r3', 'a', 30]], cols=['id', 'v', 'n'], keyfield='id')

    >>> d.select_records_daf(['r3', 'r1'])
    <BLANKLINE>
    | id | v  | n  |
    | -: | -: | -: |
    | r3 |  a | 30 |
    | r1 |  a | 10 |
    <BLANKLINE>
    %% daf rows=2; cols=3; keyfield='id'; name=''
    <BLANKLINE>
    >>> d.select_records_daf([])
    <BLANKLINE>
    <BLANKLINE>
    %% daf rows=0; cols=0; keyfield='id'; name=''
    <BLANKLINE>
    >>> kept = d.select_records_daf([], inverse=True)
    >>> kept == d
    True
    >>> _ = kept.append(['r4', 'd', 40])
    >>> d.num_rows(), kept.num_rows()
    (3, 4)
    >>> kept[0, 'v'] = 'Q'
    >>> d.iloc(0)['v']
    'Q'

    The keys of another Daf, without making a list of them:

    >>> other = Daf(lol=[['r3', 1], ['r9', 2]], cols=['id', 'w'], keyfield='id')
    >>> d.select_records_daf(other, silent_error=True).to_lod()
    [{'id': 'r3', 'v': 'a', 'n': 30}]


### Columns


##### select_kcols(kcols: slice | str | T_la | int | Tuple[Any, Any] | None, inverse: bool = False, flip: bool = False, silent_error: bool = False) -> 'Daf'

Select columns by name. Columns with those names are kept, or dropped if `inverse` is True.

This is the same as `my_daf[:, names]`, with more choices. The result is
a new Daf with new rows, in the order of the names you give. With `flip=True`
the selected columns become rows, and the result has no column names and no
keyfield. This costs no more than selecting the columns.

Selecting columns copies data, so it is not cheap. For `apply` and `reduce`,
use their `cols` argument instead.

Args:
    kcols: A name, a list of names, or a tuple that gives a range of names.
    inverse: If True, drop the named columns and keep the others.
    flip: If True, turn the selected columns into rows.
    silent_error: If True, names that are not found are ignored.

Returns:
    The new Daf.

Raises:
    KeysDisabledError: The Daf has no column names.
    KeyError: A name is not found and `silent_error` is False.

Examples:
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
    >>> d.select_kcols(['n', 'id'])
    <BLANKLINE>
    | n  | id |
    | -: | -: |
    | 10 |  1 |
    | 20 |  2 |
    | 30 |  3 |
    <BLANKLINE>
    %% daf rows=3; cols=2; keyfield='id'; name=''
    <BLANKLINE>
    >>> d.select_kcols('v', inverse=True).columns()
    ['id', 'n']
    >>> d.select_kcols('v', flip=True)
    <BLANKLINE>
    | A  | B  | C  |
    | -: | -: | -: |
    |  a |  b |  c |
    <BLANKLINE>
    %% daf rows=1; cols=3; keyfield=''; name=''
    <BLANKLINE>



##### select_icols(icols: slice | int | T_li | range | T_lor | None, flip: bool = False) -> 'Daf'

Select columns by position. Those columns are kept, in the order given.

This is the same as `my_daf[:, cols]`. It makes new rows, so it is not cheap.
For `apply` and `reduce`, use their `cols` argument instead. A slice works as
it does for a Python list. The keyfield and dtypes carry over if their columns
are kept. With `flip=True` the columns become rows, and the result has no
column names, no dtypes and no keyfield.

With `flip=True` the columns are turned into rows as they are selected. That costs less than
selecting them and then calling `transpose()`. In one test with 5 of 50 columns and 20,000
rows it took 0.003 s, against 0.034 s.

Args:
    icols: A position, a slice, a range, a list of positions, or a list of ranges.
    flip: If True, turn the selected columns into rows.

Returns:
    The new Daf.

Raises:
    IndexError: A position is beyond the end of some row.

Examples:
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
    >>> d.select_icols([2, 0])
    <BLANKLINE>
    | n  | id |
    | -: | -: |
    | 10 |  1 |
    | 20 |  2 |
    | 30 |  3 |
    <BLANKLINE>
    %% daf rows=3; cols=2; keyfield='id'; name=''
    <BLANKLINE>
    >>> d.select_icols(slice(-2, None)).columns()
    ['v', 'n']
    >>> d.select_icols([0, 1], flip=True)
    <BLANKLINE>
    | A  | B  | C  |
    | -: | -: | -: |
    |  1 |  2 |  3 |
    |  a |  b |  c |
    <BLANKLINE>
    %% daf rows=2; cols=3; keyfield=''; name=''
    <BLANKLINE>



##### select_cols(cols: T_ls | None = None, exclude_cols: T_ls | None = None) -> 'Daf'

Make a new Daf with only some columns, chosen by name.

The columns come in the order of `cols`, as with `select_kcols()` and
`my_daf[:, cols]`. A name given twice is used once. A name that is not a column
raises `KeyError`. With no `cols`, all columns are kept, in the order of this
Daf. Then `exclude_cols` leaves out the columns you name. A name in
`exclude_cols` that is not a column is ignored.

This copies data, so it is not cheap. For `apply` and `reduce`, use their
`cols` argument instead. The keyfield carries over if its column is kept. A Daf
with no column names gives rows with no columns.

Args:
    cols: The names of the columns to keep, in the order you want. If empty, all columns.
    exclude_cols: The names of the columns to leave out.

Returns:
    The new Daf.

Raises:
    KeyError: A name in `cols` is not a column.

Examples:
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
    >>> d.select_cols(['n', 'id']).columns()
    ['n', 'id']
    >>> d.select_cols(exclude_cols=['v']).columns()
    ['id', 'n']



##### col(colname: str, unique: bool = False, omit_nulls: bool = False, silent_error: bool = False, astype: Callable | str | type | None = None, indirect_col: str | None = None, default: Any | None = '') -> list

Get one column as a list, by name.

This does not make a Daf first, as `my_daf[:, 'v'].to_list()` does.

With `indirect_col`, a name that is not a column is read from the dict held in
that column, row by row. A row that lacks it gets `default`.

Args:
    colname: The column name.
    unique: If True, leave out repeated values and keep the order.
    omit_nulls: If True, leave out the empty values.
    silent_error: If True, a column that is not found gives an empty list.
    astype: A type or function to convert each value with.
    indirect_col: A column that holds a dict, to read the name from.
    default: The value for a row that lacks the name. It is used only with `indirect_col`.

Returns:
    The values of the column.

Raises:
    ColumnNotFoundError: The column is not found and `silent_error` is False. This is a
        `KeyError`, and also a `RuntimeError`, which this method raised in earlier versions.
    RuntimeError: The name is empty.

Examples:
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
    >>> d.col('v')
    ['a', 'b', 'c']
    >>> d.col('n', astype=str)
    ['10', '20', '30']



##### col_to_la(colname: str, unique: bool = False, omit_nulls: bool = False, silent_error: bool = False, astype: Callable | str | type | None = None, indirect_col: str | None = None, default: Any | None = '') -> list

Get one column as a list, by name.

This does the same as `col()`. See that method.

Args:
    colname: The column name.
    unique: If True, leave out repeated values and keep the order.
    omit_nulls: If True, leave out the empty values.
    silent_error: If True, a column that is not found gives an empty list.
    astype: A type or function to convert each value with.
    indirect_col: A column that holds a dict, to read the name from.
    default: The value for a row that lacks the name. It is used only with `indirect_col`.

Returns:
    The values of the column.

Raises:
    ColumnNotFoundError: The column is not found and `silent_error` is False. This is a
        `KeyError`, and also a `RuntimeError`, which this method raised in earlier versions.
    RuntimeError: The name is empty.

Examples:
    >>> d = Daf(lol=[['r1', 'a', 10], ['r2', 'b', ''], ['r3', 'a', 30]], cols=['id', 'v', 'n'], keyfield='id')

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



##### icol(icol: int) -> list

Get one column as a list, by position.

A negative position counts from the end, so `-1` is the last column. A position
that is out of range raises `IndexError`. A Daf with no rows gives an empty list.

Args:
    icol: The column position.

Returns:
    The values of the column.

Raises:
    IndexError: The position is out of range, for a Daf that has rows.

Examples:
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
    >>> d.icol(1)
    ['a', 'b', 'c']
    >>> d.icol(-1)
    [10, 20, 30]



##### icol_to_la(icol: int, unique: bool = False, omit_nulls: bool = False) -> list

Get one column as a list, by position, with options.

A negative position counts from the end, so `-1` is the last column. A position
that is out of range raises `IndexError`. A Daf with no rows gives an empty list.

Args:
    icol: The column position.
    unique: If True, leave out repeated values and keep the order.
    omit_nulls: If True, leave out the empty values.

Returns:
    The values of the column.

Raises:
    IndexError: The position is out of range, for a Daf that has rows.

Examples:
    >>> d = Daf(lol=[['r1', 'a', 10], ['r2', 'b', ''], ['r3', 'a', 30]], cols=['id', 'v', 'n'], keyfield='id')

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


### By condition


##### select_where(where: Callable, indirect_col: str | None = None) -> 'Daf'

Select the rows for which a function is true.

The function gets each row, as a [KeyedList][daffodil.keyedlist.KeyedList].
Read cells by column name, as in `row['n']`. Values are used as stored, so
convert text first if the Daf was read from a CSV.

With `indirect_col`, a name that is not a column is looked up in the dict
held in that column.

If the test only compares columns to values, use `select_by_dict()`. It
compares the cells by position and does not call a function for each row. For
200,000 rows by 50 columns it took 0.012 s, against 0.15 s for this method.

The new Daf shares the selected rows with this one. The keyfield and dtypes
carry over.

To test a value against a list, a set or another table, write the test in the function.
There is no need to build a list of bools first. Build a set
of the values before the call, so that each lookup is fast and the set is built once. The
function can use `and`, `or`, `not` and any other Python. It is called once for each row, so
for a test on one column of a large Daf it is not the fastest way. For a test of equality,
or of "any of these values", use `select_by_dict()` with a list of dicts, as in
`select_by_dict([{'name': n} for n in names])`. It took 0.009 s where `select_where()` took 0.15 s,
for 200,000 rows and 100 names. A comprehension over `col()`, followed by `select_irows()`, is
also faster than a function.

Args:
    where: A function that takes a row and returns True to keep it.
    indirect_col: A column that holds a dict, to read names that are not columns from.

Returns:
    The new Daf.

Examples:
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
    >>> d.select_where(lambda row: row['n'] > 10)
    <BLANKLINE>
    | id | v  | n  |
    | -: | -: | -: |
    |  2 |  b | 20 |
    |  3 |  c | 30 |
    <BLANKLINE>
    %% daf rows=2; cols=3; keyfield='id'; name=''
    <BLANKLINE>
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30], [4, 'a', 40]], cols=['id', 'v', 'n'])
    >>> keep = {'a', 'c'}
    >>> d.select_where(lambda row: row['v'] in keep)
    <BLANKLINE>
    | id | v  | n  |
    | -: | -: | -: |
    |  1 |  a | 10 |
    |  3 |  c | 30 |
    |  4 |  a | 40 |
    <BLANKLINE>
    %% daf rows=3; cols=3; keyfield=''; name=''
    <BLANKLINE>
    >>> d.select_where(lambda row: row['v'] not in keep)
    <BLANKLINE>
    | id | v  | n  |
    | -: | -: | -: |
    |  2 |  b | 20 |
    <BLANKLINE>
    %% daf rows=1; cols=3; keyfield=''; name=''
    <BLANKLINE>
    >>> d.select_where(lambda row: row['v'] in keep and row['n'] > 15)
    <BLANKLINE>
    | id | v  | n  |
    | -: | -: | -: |
    |  3 |  c | 30 |
    |  4 |  a | 40 |
    <BLANKLINE>
    %% daf rows=2; cols=3; keyfield=''; name=''
    <BLANKLINE>
    >>> d.select_where(lambda row: row['id'] % 2 == 0 or row['v'] == 'c')
    <BLANKLINE>
    | id | v  | n  |
    | -: | -: | -: |
    |  2 |  b | 20 |
    |  3 |  c | 30 |
    |  4 |  a | 40 |
    <BLANKLINE>
    %% daf rows=3; cols=3; keyfield=''; name=''
    <BLANKLINE>

    The values can come from another Daf. Make the set once, outside the function:

    >>> other = Daf(lol=[['a'], ['c']], cols=['v'], keyfield='v')
    >>> other_keys = set(other.keys())
    >>> d.select_where(lambda row: row['v'] in other_keys)
    <BLANKLINE>
    | id | v  | n  |
    | -: | -: | -: |
    |  1 |  a | 10 |
    |  3 |  c | 30 |
    |  4 |  a | 40 |
    <BLANKLINE>
    %% daf rows=3; cols=3; keyfield=''; name=''
    <BLANKLINE>

    For a large Daf, the faster form picks the positions from the column:

    >>> d.select_irows([irow for irow, v in enumerate(d.col('v')) if v in keep])
    <BLANKLINE>
    | id | v  | n  |
    | -: | -: | -: |
    |  1 |  a | 10 |
    |  3 |  c | 30 |
    |  4 |  a | 40 |
    <BLANKLINE>
    %% daf rows=3; cols=3; keyfield=''; name=''
    <BLANKLINE>



##### select_where_idxs(where: Callable) -> T_li

Get the positions of the rows for which a function is true.

The function gets each row as a [KeyedList][daffodil.keyedlist.KeyedList], whatever the
`itermode` is, as in `select_where()`. Read cells by column name, as in `row['n']`.
For a test that only compares columns to values, `select_by_dict()` is faster.

The positions are a snapshot. Inserting, removing or sorting rows makes them point
to other rows. Use them right away. To keep a reference to a row for longer, keep its key.

Args:
    where: A function that takes a row and returns True to keep it.

Returns:
    The row positions.

Raises:
    KeysDisabledError: The Daf has rows and no column names.

Examples:
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
    >>> d.select_where_idxs(lambda row: row['n'] > 10)
    [1, 2]



##### split_where(where: Callable, indirect_col: str | None = None) -> Tuple['Daf', 'Daf']

Split the rows in two, by a function that is true or false for each row.

This is deprecated. Use `group_where()` with a function that returns True or False.
It gives the two groups under the keys True and False. A side with no rows has no
group, so read each side with `.get()`.

The function gets each row, as in `select_where()`. Both new Daf instances
share their rows with this one, so changing a cell in one changes it here
too. The keyfield and dtypes carry over to both.

Args:
    where: A function that takes a row and returns True or False.
    indirect_col: A column that holds a dict, to read names that are not columns from.

Returns:
    A tuple of the Daf of the rows where the function is true and the Daf of the others.

Examples:
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
    >>> big, small = d.split_where(lambda row: row['n'] > 10)
    >>> big
    <BLANKLINE>
    | id | v  | n  |
    | -: | -: | -: |
    |  2 |  b | 20 |
    |  3 |  c | 30 |
    <BLANKLINE>
    %% daf rows=2; cols=3; keyfield='id'; name=''
    <BLANKLINE>
    >>> small
    <BLANKLINE>
    | id | v  | n  |
    | -: | -: | -: |
    |  1 |  a | 10 |
    <BLANKLINE>
    %% daf rows=1; cols=3; keyfield='id'; name=''
    <BLANKLINE>



##### select_by_dict(selector_da: T_da | T_loda, expectmax: int = -1, inverse: bool = False, keyfield: str | int | T_ta = '') -> 'Daf'

Select the rows that match every field of a dict, or any one of a list of dicts.

With a dict, a row matches if each key is a column whose value in that row
equals the value given. With a list of dicts, a row matches if it matches at
least one of the dicts. So a dict is an AND of equalities, and a list is an OR of
those. With `inverse=True` the rows that do not match are returned. The cells are
compared by position, so no row is turned into a dict.

To select the rows whose cell is any of several values, give one small dict for
each value, as in `[{'name': n} for n in names]`. To select by several columns
together, give a dict for each combination, as in `[{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]`.
The dicts are grouped by their keys, and the values of each group are put in a set, so
a row costs about one lookup however many dicts there are. This is much faster than
`select_where()` with a function. The result is the same as comparing with `==`.
A value or a cell that cannot be hashed, such as a list or a dict held in a cell,
is compared with `==` as well. It never raises an error, and it takes a slower
path for that call.

A value in a dict is only ever compared for equality. A list or a set given as a value
matches a cell that holds an equal list or set, and not a cell that is one of its items.
To say "any of these", use a list of dicts.

An empty list matches no row. An empty dict matches every row. A dict that names a
column that is not there matches no row, and the other dicts still apply.

A Daf with rows and no column names raises `KeysDisabledError`. The new Daf is a live
view of this one. It has its own outer row list, and its rows are the same lists as the
rows of this Daf. Changing a value in
it changes this Daf. Adding a column with `insert_col()` or `insert_idx_col()` does not,
because those copy shared rows first. Use `copy('editable')` for rows of your own.

Args:
    selector_da: A dict of column names and the values they must have, or a list or tuple of such dicts.
    expectmax: If this is not -1 and more rows match, raise `LookupError`.
    inverse: If True, return the rows that do not match.
    keyfield: The keyfield of the new Daf. If empty, the keyfield of this Daf.

Returns:
    The new Daf.

Raises:
    LookupError: More than `expectmax` rows match.
    KeysDisabledError: The Daf has rows and no column names.
    TypeError: `selector_da` is not a dict, or a list or tuple of dicts.

Examples:
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
    >>> d.select_by_dict({'v': 'b'})
    <BLANKLINE>
    | id | v  | n  |
    | -: | -: | -: |
    |  2 |  b | 20 |
    <BLANKLINE>
    %% daf rows=1; cols=3; keyfield='id'; name=''
    <BLANKLINE>
    >>> d.select_by_dict({'v': 'b'}, inverse=True)
    <BLANKLINE>
    | id | v  | n  |
    | -: | -: | -: |
    |  1 |  a | 10 |
    |  3 |  c | 30 |
    <BLANKLINE>
    %% daf rows=2; cols=3; keyfield='id'; name=''
    <BLANKLINE>
    >>> d.select_by_dict([{'v': 'a'}, {'v': 'c'}])
    <BLANKLINE>
    | id | v  | n  |
    | -: | -: | -: |
    |  1 |  a | 10 |
    |  3 |  c | 30 |
    <BLANKLINE>
    %% daf rows=2; cols=3; keyfield='id'; name=''
    <BLANKLINE>
    >>> d.select_by_dict([{'v': 'a', 'n': 10}, {'v': 'c', 'n': 99}])
    <BLANKLINE>
    | id | v  | n  |
    | -: | -: | -: |
    |  1 |  a | 10 |
    <BLANKLINE>
    %% daf rows=1; cols=3; keyfield='id'; name=''
    <BLANKLINE>
    >>> names = ['a', 'b']
    >>> d.select_by_dict([{'v': name} for name in names], inverse=True)
    <BLANKLINE>
    | id | v  | n  |
    | -: | -: | -: |
    |  3 |  c | 30 |
    <BLANKLINE>
    %% daf rows=1; cols=3; keyfield='id'; name=''
    <BLANKLINE>



##### select_first_row_by_dict(selector_da: T_da, inverse: bool = False) -> T_ma

Get the first row that matches every field of a dict.

The matching rule is that of `select_by_dict()`. With `inverse=True` it is the
first row that does not match.

Args:
    selector_da: The column names and the values they must have.
    inverse: If True, find the first row that does not match.

Returns:
    The row, as a dict or a KeyedList according to `itermode`. An empty dict if none matches.

Examples:
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
    >>> d.select_first_row_by_dict({'v': 'b'})
    {'id': 2, 'v': 'b', 'n': 20}
    >>> d.select_first_row_by_dict({'v': 'zz'})
    {}


### Index translation


##### krows_to_irows(krows: slice | str | T_la | int | Tuple[Any, Any] | T_lota | Iterable | None, inverse: bool = False, silent_error: bool = False) -> slice | int | T_li | range

Turn row keys into row positions.

This is the step that lets `my_daf['r2']` work. Call it when you need the
positions themselves. The Daf must have a keyfield, or a key index passed in
as `kd`.

The keys may be a key, a list of keys, or a tuple that gives an inclusive range
of keys. A range is made from positions, so the keys must be in the same order
as the rows.

The first lookup builds the key index, by reading the keyfield column. In one test with
200,000 rows that took 0.05 s, and a later lookup took about 2 microseconds. Adding or
removing rows clears the index, and the next lookup builds it again.

Args:
    krows: A key, a list of keys, or a tuple that gives a range of keys.
    inverse: If True, return the positions of the rows that are not selected.
    silent_error: If True, keys that are not found are ignored.

Returns:
    The row positions.

Raises:
    KeysDisabledError: There is no keyfield and no key index, or the keyfield is not a column.
    KeyError: A key is not found and `silent_error` is False.

Examples:
    >>> d = Daf(lol=[['r1', 'a', 10], ['r2', 'b', ''], ['r3', 'a', 30]], cols=['id', 'v', 'n'], keyfield='id')

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



##### kcols_to_icols(kcols: str | T_ls | slice | int | T_li | Tuple[Any, Any] | Iterable | None = None, inverse: bool = False, silent_error: bool = False) -> slice | int | T_li | range | None

Turn column names into column positions.

This is the step that lets `my_daf[:, 'v']` work. A name that looks like an
integer is still read as a name.

Args:
    kcols: A name, a list of names, or a tuple that gives an inclusive range of names.
    inverse: If True, return the positions of the columns that are not selected.
    silent_error: If True, names that are not found are ignored.

Returns:
    The column positions. With no column names the result is empty, or all
    positions if `inverse` is True.

Raises:
    KeyError: A name is not found and `silent_error` is False.

Examples:
    >>> d = Daf(lol=[['r1', 'a', 10], ['r2', 'b', ''], ['r3', 'a', 30]], cols=['id', 'v', 'n'], keyfield='id')

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



##### gkeys_to_idxs(keydict: Dict[str | int, int], gkeys: str | T_ls | slice | int | T_li | Tuple[Any, Any] | T_lota | Iterable | None = None, inverse: bool = False, silent_error: bool = False, axis: str = 'rowkeys', name: str = 'unspecified') -> slice | int | T_li | range

Turn keys into positions, using a dict of key to position.

This is the step under `krows_to_irows()` and `kcols_to_icols()`. It is a
static method. It is for internal use.

A key gives a list with one position. A list of keys gives their positions in
that order. A tuple of two keys is an inclusive range, and `None` in it means
the start or the end. A tuple of one key means that key to the end. A slice is
taken as positions. With `inverse`, the positions that are not selected are
returned.

Args:
    keydict: Maps each key to its position.
    gkeys: The keys.
    inverse: If True, return the positions that are not selected.
    silent_error: If True, keys that are not found are ignored.
    axis: A label for error messages.
    name: A label for error messages.

Returns:
    The positions, as a list or a slice.

Raises:
    KeysDisabledError: The dict is empty, as it is for a Daf with no rows.
    KeyError: A key is not found and `silent_error` is False.
    TypeError: The keys are None, or a tuple of a length other than 1 or 2.

Examples:
    >>> Daf.gkeys_to_idxs({'a': 0, 'b': 1, 'c': 2}, ['c', 'a'])
    [2, 0]
    >>> Daf.gkeys_to_idxs({'a': 0, 'b': 1, 'c': 2}, ('a', 'b'))
    slice(0, 2, 1)

---

# Docs page: Daf API / Modifying (api/daf/modify.md)

## Modifying

Adding, changing, removing and reordering records and columns.

### Appending


##### append(data_item: 'Daf' | T_loda | T_da | T_la | KeyedList | None = None, respect_kd: bool = False, lol: T_lola | None = None, la: T_la | None = None, fast: bool = False) -> 'Daf'

Add one row, or several, to the end of the Daf.

This is the general way to add data. What it does depends on what you give it.
Give exactly one of `data_item`, `lol` and `la`.

A dict or a [KeyedList][daffodil.keyedlist.KeyedList] is one row. It is placed
by column name, so its keys may be in any order. A missing key gets NULL. A key
that is not a column is dropped.

A list of values is one row, in column order. A short list is padded with NULL.
A list with more values than there are columns raises `ValueError`. With no
columns defined, the list is added as it is. A list of lists given as
`data_item` is therefore one row whose cells are lists, which fits only if there
are enough columns.

To say what you mean, use the keywords. `lol` is several rows, each a list in
column order, as in `append(lol=[[2, 'b'], [3, 'c']])`. `la` is one row, even if
its items are lists, as in `append(la=[[2, 'b'], [3, 'c']])`. Each row of `lol`
follows the length rule above.

A list of dicts is several rows. See `extend()`.

A Daf is several rows. See `concat()`. Its columns must match.

An empty dict, list or Daf adds nothing.

By default the keyfield is not checked, so a key that is already present is
added again. This keeps appending fast. Pass `respect_kd=True` to replace the
row that has the same key instead. That looks the key up on every call, so
it costs more when you add many rows one at a time.

With columns defined, a list is added as a copy, so you can reuse it. With no
columns, the list itself is added.

A KeyedList made by this Daf, as by `default_record(astype=KeyedList)`, `iloc()` or
`iter_klist()`, shares its `hd`. Its keys are then the columns, so its columns are not
checked. Its values are still copied.

With `fast=True` nothing is copied. A KeyedList must share the `hd` of this Daf, and
then nothing else is checked. A dict or a list is checked fully if it is the first row
added to an empty Daf: a dict must have the columns as its keys, in order, and a list one
value for each column. Every later dict or list is checked only for its number of values.
A row that fails raises `ValueError`.
Use it when you build each row yourself in column order, as in a loop, best with
`default_record()`. A list, or the value list of a KeyedList, becomes the row itself,
so do not change it afterwards. Giving a variable a new list each time is fine.
Filling one list in place and appending it again is not: every row is then that same
list. A dict gives its values in its own order. A later row with the right number of
values in another order is added anyway, and its values land in the wrong columns. If you want every row
checked, leave out `fast`. `fast` makes no difference to the first row of a Daf with
no columns, to a Daf or a list of dicts, or with `respect_kd=True` and a keyfield.

None, an empty dict, an empty list and an empty Daf add nothing.

Args:
    data_item: The row or rows to add.
    respect_kd: If True, replace the row that has the same key. If False, the default, add it.
    lol: Several rows, each a list of values in column order.
    la: One row, as a list of values in column order. Its items are not read as rows.
    fast: If True, add the row with no copy. Check the first row of an empty Daf fully, and every row's length. See above.

Returns:
    This Daf, which has been changed.

Raises:
    TypeError: More than one of `data_item`, `lol` and `la` is given, or `data_item` is not a supported type.
    ValueError: A list has more values than there are columns.

Examples:
    >>> d = Daf(lol=[[1, 'a']], cols=['id', 'v'], keyfield='id')
    >>> d.append({'v': 'b', 'id': 2})
    <BLANKLINE>
    | id | v  |
    | -: | -: |
    |  1 |  a |
    |  2 |  b |
    <BLANKLINE>
    %% daf rows=2; cols=2; keyfield='id'; name=''
    <BLANKLINE>
    >>> d.append([3, 'c'])
    <BLANKLINE>
    | id | v  |
    | -: | -: |
    |  1 |  a |
    |  2 |  b |
    |  3 |  c |
    <BLANKLINE>
    %% daf rows=3; cols=2; keyfield='id'; name=''
    <BLANKLINE>
    >>> d.append(lol=[[4, 'd'], [5, 'e']])
    <BLANKLINE>
    | id | v  |
    | -: | -: |
    |  1 |  a |
    |  2 |  b |
    |  3 |  c |
    |  4 |  d |
    |  5 |  e |
    <BLANKLINE>
    %% daf rows=5; cols=2; keyfield='id'; name=''
    <BLANKLINE>
    >>> d.append({'id': 2, 'v': 'new'}, respect_kd=True)
    <BLANKLINE>
    | id |  v  |
    | -: | --: |
    |  1 |   a |
    |  2 | new |
    |  3 |   c |
    |  4 |   d |
    |  5 |   e |
    <BLANKLINE>
    %% daf rows=5; cols=2; keyfield='id'; name=''
    <BLANKLINE>
    >>> e = Daf(lol=[[1, 'a']], cols=['id', 'v'])
    >>> e.append(None).append({}).append([])
    <BLANKLINE>
    | id | v  |
    | -: | -: |
    |  1 |  a |
    <BLANKLINE>
    %% daf rows=1; cols=2; keyfield=''; name=''
    <BLANKLINE>



##### concat(other_instance: 'Daf', respect_kd: bool = False) -> 'Daf'

Add the rows of another Daf to the end of this one.

Use this to combine two tables with the same columns. By default every row is
appended, so a key that exists in both tables appears twice afterwards. Pass
`respect_kd=True` to get an upsert instead.

Args:
    other_instance: Daf whose rows are added. It is not changed.
    respect_kd: If False, the default, append every row without looking at the keys. If True,
        replace the row that has the same key and append the rows with new keys.

Returns:
    This Daf, which has been changed in place. The rows are deep copies.

Raises:
    KeyError: The column names of the two Daf instances differ.
    ValueError: With `respect_kd=True`, both Daf instances have a keyfield and the
        keyfields differ.

Notes:
    The columns must be equal, including their order.
    With `respect_kd=True`, this Daf's keyfield decides the key. If it has no
    keyfield, the rows are simply appended.
    A key repeated in the other Daf replaces the earlier row, so the last one wins.
    If the other Daf is empty, nothing happens.

Examples:
    >>> a = Daf(lol=[[1, 'a1'], [2, 'a2']], cols=['id', 'v'], keyfield='id')
    >>> b = Daf(lol=[[2, 'b2'], [3, 'b3']], cols=['id', 'v'])
    >>> a.concat(b)
    <BLANKLINE>
    | id | v  |
    | -: | -: |
    |  1 | a1 |
    |  2 | a2 |
    |  2 | b2 |
    |  3 | b3 |
    <BLANKLINE>
    %% daf rows=4; cols=2; keyfield='id'; name=''
    <BLANKLINE>
    >>> a = Daf(lol=[[1, 'a1'], [2, 'a2']], cols=['id', 'v'], keyfield='id')
    >>> a.concat(b, respect_kd=True)
    <BLANKLINE>
    | id | v  |
    | -: | -: |
    |  1 | a1 |
    |  2 | b2 |
    |  3 | b3 |
    <BLANKLINE>
    %% daf rows=3; cols=2; keyfield='id'; name=''
    <BLANKLINE>



##### extend(records_lod: T_loda | None = None, respect_kd: bool = False, lol: T_lola | None = None) -> 'Daf'

Append several records, given as a list of dicts or as a list of lists.

Give a list of dicts as `records_lod`. Each dict is a row, placed by column name,
as in `append()`. A Daf with no columns takes them from the first dict. An empty
list, or a list that holds one empty dict, adds nothing.

Give a list of lists as `lol`, as in `extend(lol=[[2, 'b'], [3, 'c']])`. Each list is a row, in
column order. A short list is padded with NULL. A list with more values than there
are columns raises `ValueError`, and then no row is added. A Daf with no columns
takes the lists as they are. The keyword says that these are several rows, and not
one row whose cells are lists. For that, see `append(la=...)`.

Without `respect_kd`, a key that already exists is added again. Use
`respect_kd=True` when the keyfield must stay unique.

Args:
    records_lod: The records, as dicts.
    respect_kd: If True, replace the row that has the same key. If False, the default, add it.
    lol: The rows, as lists of values in column order.

Returns:
    This Daf, which has been changed.

Raises:
    TypeError: Both `records_lod` and `lol` are given, or a row of `lol` is not a list.
    ValueError: A row of `lol` has more values than there are columns.

Examples:
    >>> d = Daf(lol=[[1, 'a']], cols=['id', 'v'], keyfield='id')
    >>> d.extend([{'id': 2, 'v': 'b'}, {'id': 1, 'v': 'new'}], respect_kd=True)
    <BLANKLINE>
    | id |  v  |
    | -: | --: |
    |  1 | new |
    |  2 |   b |
    <BLANKLINE>
    %% daf rows=2; cols=2; keyfield='id'; name=''
    <BLANKLINE>
    >>> d.extend(lol=[[3, 'c'], [4]])
    <BLANKLINE>
    | id |  v  |
    | -: | --: |
    |  1 | new |
    |  2 |   b |
    |  3 |   c |
    |  4 |     |
    <BLANKLINE>
    %% daf rows=4; cols=2; keyfield='id'; name=''
    <BLANKLINE>



##### record_append(record: T_da | KeyedList, respect_kd: bool = True) -> 'Daf'

Add one row that is given as a dict or a KeyedList.

This is the single row case of `append()`. The row is placed by column name,
a missing key gets NULL, and a key that is not a column is dropped. An empty
Daf takes its columns from the first row. An empty record adds nothing.

With a keyfield, the row that has the same key is replaced, and a new key is
added at the end. This is the default here, unlike `append()`. Pass
`respect_kd=False` to always add. The key index is kept up to date, so adding
many rows one at a time stays fast.

A plain list is not accepted. Use `append()` for that.

Args:
    record: The row, as a dict or a [KeyedList][daffodil.keyedlist.KeyedList].
    respect_kd: If True, the default, replace the row that has the same key. Otherwise add it.

Returns:
    This Daf, which has been changed.

Examples:
    >>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'], keyfield='id')
    >>> d.record_append({'id': 2, 'v': 'new'})
    <BLANKLINE>
    | id |  v  |
    | -: | --: |
    |  1 |   a |
    |  2 | new |
    <BLANKLINE>
    %% daf rows=2; cols=2; keyfield='id'; name=''
    <BLANKLINE>
    >>> d.record_append({'id': 2, 'v': 'dup'}, respect_kd=False)
    <BLANKLINE>
    | id |  v  |
    | -: | --: |
    |  1 |   a |
    |  2 | new |
    |  2 | dup |
    <BLANKLINE>
    %% daf rows=3; cols=2; keyfield='id'; name=''
    <BLANKLINE>


### Records


##### assign_record(record: T_da) -> 'Daf'

Put one row in the Daf by its key, replacing a row that has the same key.

The row is a dict. If a row with that key exists, the whole row is replaced.
The cells of columns that the dict lacks become NULL. If the key is new, the
row is added at the end. This is an upsert for one row. For that, `append()`
with `respect_kd=True` also works. Use `update_by_keylist()` to change only
some cells.

Args:
    record: The row, as a dict. It must have the keyfield.

Returns:
    This Daf, which has been changed.

Raises:
    KeysDisabledError: The Daf has no keyfield.

Examples:
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
    >>> _ = d.assign_record({'id': 2, 'v': 'new'})
    >>> d
    <BLANKLINE>
    | id |  v  | n  |
    | -: | --: | -: |
    |  1 |   a | 10 |
    |  2 | new |    |
    |  3 |   c | 30 |
    <BLANKLINE>
    %% daf rows=3; cols=3; keyfield='id'; name=''
    <BLANKLINE>



##### assign_record_irow(irow: int | None = None, record: T_da | None = None) -> 'Daf'

Put one row in the Daf by position, replacing the row there.

The row is a dict. The whole row is replaced, and the cells of columns that
the dict lacks become NULL. Use `update_record_irow()` to change only some cells.

With the default position, `None`, the row is added at the end. So it is with a
position beyond the last row, and with any position when the Daf has no rows.
A negative position counts from the end, as in a list, so `-1` replaces the last
row. This is also what `my_daf[-1] = {...}` does.

This does not update the key index (kd), so don't change a key with it. See
[`__setitem__`][daffodil.daf.Daf.__setitem__].

Args:
    irow: The row position. None, the default, adds the row at the end.
    record: The row, as a dict. If None, nothing happens.

Returns:
    This Daf, which has been changed.

Raises:
    IndexError: A negative position is before the first row, for a Daf that has rows.

Examples:
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
    >>> _ = d.assign_record_irow(1, {'v': 'q'})
    >>> d.iloc(1)
    {'id': '', 'v': 'q', 'n': ''}
    >>> _ = d.assign_record_irow(-1, {'id': 9})
    >>> d.iloc(-1)
    {'id': 9, 'v': '', 'n': ''}
    >>> _ = d.assign_record_irow(record={'id': 10})
    >>> d.iloc(-1), len(d)
    ({'id': 10, 'v': '', 'n': ''}, 4)



##### update_by_keylist(keylist: T_ls | None = None, record: T_da | None = None) -> 'Daf'

Change some cells in the rows that have the given keys.

Only the columns that are keys of the dict are changed. Other cells keep
their values. A key that is not found is skipped. This is a bulk form of
`update_record_krow()`, for rows that all take the same values. The row positions
do not change, so the key index stays valid.

Args:
    keylist: The keys of the rows to change.
    record: The new values, as a dict of column name and value. Names that are not columns are ignored.

Returns:
    This Daf, which has been changed. With no keyfield, rows, keys or record, nothing happens.

Examples:
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
    >>> d.update_by_keylist([1, 3, 9], {'v': 'q'})
    <BLANKLINE>
    | id | v  | n  |
    | -: | -: | -: |
    |  1 |  q | 10 |
    |  2 |  b | 20 |
    |  3 |  q | 30 |
    <BLANKLINE>
    %% daf rows=3; cols=3; keyfield='id'; name=''
    <BLANKLINE>



##### update_record_irow(irow: int = -1, record: T_da | None = None) -> 'Daf'

Change some cells in the row at a position.

Only the columns that are keys of the dict are changed. Other cells keep their
values. A negative position counts from the end, as in a list, so the default,
`-1`, is the last row. A position that is out of range raises `IndexError`, and
nothing is changed. A Daf with no rows or no columns, and a record of `None`, change nothing.

This does not update the key index (kd), so don't change a key with it. See
[`__setitem__`][daffodil.daf.Daf.__setitem__].

Args:
    irow: The row position. -1, the default, is the last row.
    record: The new values, as a dict of column name and value. Names that are not columns are ignored.

Returns:
    This Daf, which has been changed.

Raises:
    IndexError: The position is out of range, for a Daf that has rows and columns.

Examples:
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
    >>> _ = d.update_record_irow(1, {'v': 'q'})
    >>> d.iloc(1)
    {'id': 2, 'v': 'q', 'n': 20}
    >>> _ = d.update_record_irow(record={'n': 99})
    >>> d.iloc(-1)
    {'id': 3, 'v': 'c', 'n': 99}



##### update_record_krow(key: Any, record: T_da | None = None) -> 'Daf'

Change some cells in the row that has this key.

Only the columns that are keys of the dict are changed. Other cells keep their
values. It is the form of `update_record_irow()` that finds the row by its key.
An int is always a key here, never a position, so it suits a keyfield of ints.
For many rows that take the same values, use `update_by_keylist()`.

As with `update_record_irow()`, this does not update the key index (kd), so don't
change a key with it. See [`__setitem__`][daffodil.daf.Daf.__setitem__].
A record of `None` changes nothing.

Args:
    key: The key of the row to change.
    record: The new values, as a dict of column name and value. Names that are not columns are ignored.

Returns:
    This Daf, which has been changed.

Raises:
    KeyError: No row has this key.
    ValueError: The Daf has no keyfield.

Examples:
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
    >>> _ = d.update_record_krow(2, {'v': 'q'})
    >>> d.select_record(2)
    {'id': 2, 'v': 'q', 'n': 20}



##### insert_irow(irow: int | None = None, row: T_la | T_da | None = None, default: Any = '') -> 'Daf'

Insert a row at a position and move the later rows down.

The row is a list of values, or a dict that is placed by column name. A short
list is filled out with `default`. With the default position, `None`, or with a
position beyond the last row, the row is added at the end. A negative position,
such as `-1`, also adds it at the end. This differs from `list.insert()`, which
puts the item before the last one. Prefer `None`. The key index is rebuilt when
it is next needed.

Args:
    irow: The row position. None, the default, adds the row at the end.
    row: The row, as a list or a dict.
    default: The value for cells that a short list does not reach.

Returns:
    This Daf, which has been changed.

Raises:
    TypeError: The row is not a list or a dict. This includes the default, None.

Examples:
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
    >>> d.insert_irow(1, {'id': 9, 'v': 'z'})
    <BLANKLINE>
    | id | v  | n  |
    | -: | -: | -: |
    |  1 |  a | 10 |
    |  9 |  z |    |
    |  2 |  b | 20 |
    |  3 |  c | 30 |
    <BLANKLINE>
    %% daf rows=4; cols=3; keyfield='id'; name=''
    <BLANKLINE>


### Columns


##### assign_col(colname: str, la: T_la | None = None, default: Any = '') -> 'Daf'

Fill a column by name with the values of a list, or add it if it is new.

This is `my_daf[:, colname] = values`, and it also adds a column. A list that
is too short is filled out with `default`. With no list, every cell gets
`default`. If the column is the keyfield, the key index is rebuilt when it is
next needed.

Args:
    colname: The column name.
    la: The values, one for each row.
    default: The value for rows that the list does not reach.

Returns:
    This Daf, which has been changed.

Examples:
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
    >>> d.assign_col('w', default=0).columns()
    ['id', 'v', 'n', 'w']
    >>> d.col('w')
    [0, 0, 0]



##### assign_icol(icol: int = -1, col_la: T_la | None = None, default: Any = '') -> 'Daf'

Fill a column by position with the values of a list.

A list that is too short is filled out with `default`. With no list, every cell
gets `default`. With `icol=-1` a new column is added at the right. If the Daf has
column names, the new column gets the next spreadsheet name, such as `C`, made
unique. Use `insert_col()` to add a column with a name of your own.

Args:
    icol: The column position. -1 adds a column at the right.
    col_la: The values, one for each row.
    default: The value for rows that the list does not reach.

Returns:
    This Daf, which has been changed.

Examples:
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
    >>> _ = d.assign_icol(1, ['x', 'y'], default='D')
    >>> d.col('v')
    ['x', 'y', 'D']



##### insert_col(colname: str, col_la: T_la | None = None, icol: int = -1, default: Any = '') -> 'Daf'

Add a named column at a position, or overwrite it if the name exists.

A list that is too short is filled out with `default`. With no list, every cell
gets `default`, so this can add a constant column. If the name already exists
the column is overwritten, and `icol` is ignored. An empty name does nothing.
Use `set_keyfield()` if the column is to be the keyfield.

When a column is added and the rows are shared with another Daf, as after a
selection, this Daf first takes its own copies of the rows. The other Daf is
not changed.

Args:
    colname: The name of the column.
    col_la: The values, one for each row.
    icol: The column position. -1 adds the column at the right.
    default: The value for rows that the list does not reach.

Returns:
    This Daf, which has been changed.

Examples:
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
    >>> d.insert_col('w', ['x', 'y', 'z'], icol=1).columns()
    ['id', 'w', 'v', 'n']
    >>> d.insert_col('k', default=5).col('k')
    [5, 5, 5]



##### insert_icol(icol: int = -1, col_la: T_la | None = None, colname: str = '', default: Any = '') -> 'Daf'

Insert a column at a position and move the later columns right.

A list that is too short is filled out with `default`. With `icol=-1`, or a
position beyond the last column, the column is added at the right. Give
`colname` to name it. Without a name, if the Daf has column names, the column
gets the next spreadsheet name, such as `C`, made unique. The dtypes are not
changed. Use `set_keyfield()` if the column is to be the keyfield.

If the rows are shared with another Daf, as after a selection, this Daf first
takes its own copies of the rows. The other Daf is not changed.

Args:
    icol: The column position. -1 adds the column at the right.
    col_la: The values, one for each row.
    colname: The name of the new column.
    default: The value for rows that the list does not reach.

Returns:
    This Daf, which has been changed.

Examples:
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
    >>> d.insert_icol(1, ['x', 'y', 'z'], colname='new').columns()
    ['id', 'new', 'v', 'n']



##### insert_idx_col(colname: str = 'idx', icol: int = 0, startat: int = 0) -> 'Daf'

Insert a column of row numbers.

This is `insert_col()`, so rows shared with another Daf are copied first and
the other Daf is not changed.

Args:
    colname: The name of the new column.
    icol: The column position.
    startat: The number of the first row.

Returns:
    This Daf, which has been changed.

Examples:
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
    >>> d.insert_idx_col().col('idx')
    [0, 1, 2]



##### set_col_irows(colname: str, irows: T_li, val: Any) -> 'Daf'

Set one value in the given rows of a named column.

This is deprecated. Use `my_daf[irows, colname] = value`, which does the same. A
column name that is not found raises `KeyError`. Row positions that are out of
range are skipped.

Args:
    colname: The column name.
    irows: The row positions.
    val: The value to set.

Returns:
    This Daf, which has been changed.

Raises:
    KeyError: The column name is not found.

Examples:
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
    >>> d.set_col_irows('v', [0, 2], 'Z').col('v')
    ['Z', 'b', 'Z']



##### set_icol(icol: int, val: Any) -> 'Daf'

Set one value in every row of a column, by position.

This is `my_daf[:, icol] = value`.

Args:
    icol: The column position.
    val: The value to set.

Returns:
    This Daf, which has been changed.

Raises:
    IndexError: The position is beyond the end of a row.

Examples:
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
    >>> d.set_icol(1, 'Z').col('v')
    ['Z', 'Z', 'Z']



##### set_icol_irows(icol: int, irows: T_li, val: Any) -> 'Daf'

Set one value in the given rows of a column, by position.

This is `my_daf[irows, icol] = value`. Row positions that are out of range
are skipped.

Args:
    icol: The column position.
    irows: The row positions.
    val: The value to set.

Returns:
    This Daf, which has been changed.

Examples:
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
    >>> _ = d.set_icol_irows(1, [0, 9, -1], 'Z')
    >>> d.col('v')
    ['Z', 'b', 'c']


### Removing


##### delete_records(keys_ls: 'T_ls | Iterable | Daf', inplace: bool = False, silent_error: bool = False) -> 'Daf'

Remove the rows with these keys.

By default, this Daf is left as it is, and a new Daf without the rows is returned, as
`select_records_daf(keys, inverse=True)` gives. That is the usual way in daffodil, and
the new Daf shares the rows that remain.

With `inplace=True`, the rows are deleted from this Daf itself, and it is returned. The
row list is rebuilt once, in place, so it costs one pass however many rows go. The
remaining rows keep their order, and the key index is rebuilt when next needed. That is
safe for a Daf in memory: a selection made from it earlier holds its own row list, so it
still has the deleted rows. On a SqlDaf parent table it is not safe in the same way. See
`SqlDaf.delete_records()` in sqldaf.py.

The keys may be a list, any iterable, or another Daf, whose keys are then used through a
view of its key index, as in `select_records_daf()`.

Args:
    keys_ls: The keys of the rows to remove, or a Daf whose keys are used.
    inplace: If True, delete from this Daf and return it. If False, return a new Daf.
    silent_error: If True, keys that are not found are ignored.

Returns:
    The Daf without the rows: this one if `inplace`, otherwise a new one.

Raises:
    KeysDisabledError: This Daf, or a Daf given as the keys, has no keyfield and no key index.
    KeyError: A key is not found and `silent_error` is False. Nothing is removed.

Examples:
    >>> d = Daf(lol=[['r1', 'a'], ['r2', 'b'], ['r3', 'c']], cols=['id', 'v'], keyfield='id')
    >>> d.delete_records(['r2']).to_lod()
    [{'id': 'r1', 'v': 'a'}, {'id': 'r3', 'v': 'c'}]
    >>> len(d)
    3
    >>> _ = d.delete_records(['r2', 'r9'], inplace=True, silent_error=True)
    >>> d.to_lod()
    [{'id': 'r1', 'v': 'a'}, {'id': 'r3', 'v': 'c'}]



##### remove_key(keyval: str | int | T_la | T_ta | None, silent_error: bool = False) -> 'Daf'

Make a new Daf without the row that has the given key.

This is deprecated. Use `select_krows(key, inverse=True)`, which does the same.

This does not remove the row from this Daf. It returns a new Daf that leaves
the row out, and this Daf is unchanged. Keep the result, as in
`d = d.remove_key(2)`.

The new Daf shares the surviving rows with this Daf. Changing a cell in one
changes it in the other. Call `copy('editable')` if you need row lists of your own.

A tuple is a range of keys, as in `remove_key((2, 3))`. With a composite keyfield, a
tuple as long as the keyfield, with no tuples inside it, is one key, as in
`remove_key(('a', 1))`. A tuple of key tuples is a range, and a list of key tuples
is a list of keys.

Args:
    keyval: The key of the row to leave out.
    silent_error: If True, a key that is not found is ignored.

Returns:
    The new Daf. It has the same keyfield.

Raises:
    KeysDisabledError: The Daf has no keyfield and no key index.
    KeyError: The key is not found and `silent_error` is False.

Examples:
    >>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'], keyfield='id')
    >>> d.remove_key(1)
    <BLANKLINE>
    | id | v  |
    | -: | -: |
    |  2 |  b |
    <BLANKLINE>
    %% daf rows=1; cols=2; keyfield='id'; name=''
    <BLANKLINE>
    >>> d.num_rows()
    2
    >>> c = Daf(lol=[['a', 1], ['a', 2]], cols=['g', 'n'], keyfield=('g', 'n'))
    >>> c.remove_key(('a', 1))
    <BLANKLINE>
    | g  | n  |
    | -: | -: |
    |  a |  2 |
    <BLANKLINE>
    %% daf rows=1; cols=2; keyfield='('g', 'n')'; name=''
    <BLANKLINE>



##### remove_keylist(keylist: T_ls, silent_error: bool = False) -> 'Daf'

Make a new Daf without the rows that have the given keys.

This is deprecated. Use `select_krows(keys, inverse=True)`, which does the same.

This does not remove the rows from this Daf. It returns a new Daf that leaves
them out, and this Daf is unchanged. See `remove_key()` for how the rows are
shared.

Args:
    keylist: The keys of the rows to leave out.
    silent_error: If True, keys that are not found are ignored.

Returns:
    The new Daf. It has the same keyfield.

Raises:
    KeysDisabledError: The Daf has no keyfield and no key index.
    KeyError: A key is not found and `silent_error` is False.

Examples:
    >>> d = Daf(lol=[[1, 'a'], [2, 'b'], [3, 'c']], cols=['id', 'v'], keyfield='id')
    >>> d.remove_keylist([1, 3])
    <BLANKLINE>
    | id | v  |
    | -: | -: |
    |  2 |  b |
    <BLANKLINE>
    %% daf rows=1; cols=2; keyfield='id'; name=''
    <BLANKLINE>
    >>> d2 = Daf(lol=[[1, 'a', 0], [2, 'b', 1]], cols=['p', 'q', 'r'], keyfield=['p', 'q'])
    >>> d2.remove_keylist([(1, 'a')])
    <BLANKLINE>
    | p  | q  | r  |
    | -: | -: | -: |
    |  2 |  b |  1 |
    <BLANKLINE>
    %% daf rows=1; cols=3; keyfield='['p', 'q']'; name=''
    <BLANKLINE>



##### remove_dups(keyfield: str | T_ta | T_la = '') -> Tuple['Daf', 'Daf']

Split the rows into those with a unique key and those with a repeated key.

Only the key columns are compared, not the whole row. For each key, the last row
is kept as the unique one. The earlier rows with that key are the duplicates.
The unique rows are in the order in which their keys first appear.

Without `keyfield`, the keyfield of this Daf is used. If there is none, a
`KeysDisabledError` is raised. This Daf is not changed. Both results share their
rows with it, and both have the key columns as their keyfield.

Args:
    keyfield: The column, or tuple or list of columns, that identifies a row.
        If empty, the keyfield of this Daf.

Returns:
    A tuple of the Daf of unique rows and the Daf of duplicate rows.

Raises:
    KeysDisabledError: There is no `keyfield` and this Daf has none.
    KeyError: A key column is not found.

Examples:
    >>> d = Daf(lol=[[1, 'a'], [2, 'b'], [1, 'c']], cols=['id', 'v'])
    >>> unique, dups = d.remove_dups('id')
    >>> unique
    <BLANKLINE>
    | id | v  |
    | -: | -: |
    |  1 |  c |
    |  2 |  b |
    <BLANKLINE>
    %% daf rows=2; cols=2; keyfield='id'; name=''
    <BLANKLINE>
    >>> dups
    <BLANKLINE>
    | id | v  |
    | -: | -: |
    |  1 |  a |
    <BLANKLINE>
    %% daf rows=1; cols=2; keyfield='id'; name=''
    <BLANKLINE>



##### drop_cols(exclude_cols: T_ls | None = None) -> 'Daf'

Remove columns from this Daf, in place.

The rows are rebuilt without those columns, so this copies all the data. Avoid
it for large tables. Instead, name only the columns you need: with `cols=` in
`reduce()`, by reading only those fields in an `apply()` function, or with
`select_kcols()` for a new Daf.

The column names and dtypes are updated. A name that is not a column is ignored.
If a column of the keyfield is dropped, the keyfield is cleared, as in the other
methods that remove a key column. Set it again with `set_keyfield()`. With no
names, nothing happens.

Args:
    exclude_cols: The names of the columns to remove.

Returns:
    This Daf, which has been changed.

Examples:
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
    >>> d.drop_cols(['v']).columns()
    ['id', 'n']


### Find and replace


##### find_replace(find_pat: str, replace_val: Any) -> 'Daf'

Replace every cell that matches a pattern, in place.

Each cell is converted to text and searched with the regular expression
`find_pat`. If it matches anywhere in the text, the whole cell is replaced by
`replace_val`. It is not a substitution inside the text. Every column is
searched, including numbers, and the key index is rebuilt when it is next
needed.

Args:
    find_pat: A regular expression.
    replace_val: The value that replaces a matching cell.

Returns:
    This Daf, which has been changed.

Examples:
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
    >>> _ = d.find_replace(r'^[ab]$', 'HIT')
    >>> d.col('v')
    ['HIT', 'HIT', 'c']



##### replace_in_columns(cols: T_lsi | None, find_values: List[Any] | None = None, replacement: Any = _MISSING) -> 'Daf'

Replace listed values with another value, in some columns, in place.

A cell is replaced when it equals any value in `find_values`. A typical use
is `['', None]` to fill empty cells. Columns may be given by name or by
position, or all columns if `cols` is None. With no `find_values` nothing
happens. The key index is rebuilt if the keyfield may have changed.

Args:
    cols: Column names or positions. If None, all columns.
    find_values: The values to look for.
    replacement: The value to put in their place. This is required.

Returns:
    This Daf, which has been changed.

Raises:
    ValueError: No `replacement` is given.
    KeyError: A column name is not found.
    TypeError: A column is neither a name nor a position.

Examples:
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
    >>> d.replace_in_columns(['v'], ['a', 'c'], '-').col('v')
    ['-', 'b', '-']


### Sorting


##### sort_by_colname(colname: str, reverse: bool = False, length_priority: bool = False, as_str: bool = False) -> 'Daf'

Sort the rows by one column, in place.

Make a copy first if you need the original order. An empty cell sorts before
a cell with content, unlike a spreadsheet. Values in the column must be
comparable with each other. A column that mixes None or text with numbers
raises `TypeError`, which names the column. Use `as_str=True` to sort by the text
of each value instead.

With `length_priority`, a shorter text sorts before a longer one, so numbers
that are stored as text sort as numbers. Without it, `'10'`, `'100'`, `'8'`
are in text order. `length_priority` needs text, so for real numbers use it
with `as_str=True`. Then whole numbers sort in numeric order, and negative and
decimal numbers sort in text order.

Args:
    colname: The column to sort by.
    reverse: If True, sort from high to low.
    length_priority: If True, sort by length first, then by value.
    as_str: If True, sort by the text of each value, and None sorts as an empty cell.

Returns:
    This Daf, which has been changed.

Raises:
    KeyError: The column name is not found.
    TypeError: The column holds values that cannot be compared, and `as_str` is False.

Examples:
    >>> d = Daf(lol=[['10'], ['99'], ['8'], ['100']], cols=['a'])
    >>> d.sort_by_colname('a', length_priority=True).col('a')
    ['8', '10', '99', '100']
    >>> d = Daf(lol=[[10], [None], [2], ['']], cols=['n'])
    >>> d.sort_by_colname('n', as_str=True, length_priority=True).col('n')
    [None, '', 2, 10]



##### sort_by_colnames(colnames: T_ls, reverse: bool = False, length_priority: bool = False, as_str: bool = False) -> 'Daf'

Sort the rows by several columns, in place.

The first column is the main sort key. The others break ties. See
`sort_by_colname()` for the rules of ordering, for `length_priority` and for `as_str`.
Calling `sort_by_colname()` for each column, last column first, gives the same
order.

Args:
    colnames: The columns to sort by, main key first.
    reverse: If True, sort from high to low.
    length_priority: If True, sort by length first, then by value.
    as_str: If True, sort by the text of each value, and None sorts as an empty cell.

Returns:
    This Daf, which has been changed.

Raises:
    KeyError: A column name is not found.
    TypeError: A column holds values that cannot be compared, and `as_str` is False.

Examples:
    >>> d = Daf(lol=[[2, 'b'], [1, 'z'], [1, 'a']], cols=['p', 'q'])
    >>> d.sort_by_colnames(['p', 'q'])
    <BLANKLINE>
    | p  | q  |
    | -: | -: |
    |  1 |  a |
    |  1 |  z |
    |  2 |  b |
    <BLANKLINE>
    %% daf rows=3; cols=2; keyfield=''; name=''
    <BLANKLINE>


### Reshaping and splitting


##### transpose(new_keyfield: str = '', new_cols: T_la | None = None, include_header: bool = False) -> 'Daf'

Turn rows into columns and columns into rows.

The result has one row for each column of this Daf. With `include_header=True`,
the first column of the result holds the column names of this Daf, and the
names given in `new_cols` or the default names must then include that column.
The default cols are `A`, `B` and so on, one for each row of this Daf. With
`include_header=True` they start with `key`. If you pass `new_cols`, give one
name for each row of this Daf, plus one with `include_header`. The data is
copied.

Args:
    new_keyfield: The keyfield of the result.
    new_cols: The cols of the result.
    include_header: If True, the column names become the first column.

Every row must be as long as the columns. Check with `is_rectangular()`, and pad
short rows with `force_rectangular()`.

Returns:
    The new Daf.

Raises:
    ValueError: A row is not as long as the columns.

Examples:
    >>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'])
    >>> d.transpose(include_header=True)
    <BLANKLINE>
    | key | A  | B  |
    | --: | -: | -: |
    |  id |  1 |  2 |
    |   v |  a |  b |
    <BLANKLINE>
    %% daf rows=2; cols=3; keyfield=''; name=''
    <BLANKLINE>
    >>> d.transpose().columns()
    ['A', 'B']
    >>> d.transpose(include_header=True).columns()
    ['key', 'A', 'B']



##### wide_to_narrow(id_cols: T_ls, varname_colname: str = 'varname', value_colname: str = 'value') -> 'Daf'

Turn columns into rows. This is called melt or unpivot.

Each column that is not an id column gives one row for each row of this Daf.
The row holds the id values, the name of the column, and its value.

The default name of the column of names is `varname`. `narrow_to_wide()` looks for
`variable` by default. So to go back, pass `varname_colname='variable'` here, or
`varname_col='varname'` there.

Args:
    id_cols: The columns that identify a row. They are kept as they are.
    varname_colname: The name of the new column that holds the old column names.
    value_colname: The name of the new column that holds the values.

Returns:
    The new Daf.

Raises:
    TypeError: `id_cols` is not a list.

Examples:
    >>> d = Daf(lol=[['x', 1, 2], ['y', 3, 4]], cols=['id', 'a', 'b'])
    >>> d.wide_to_narrow(['id'])
    <BLANKLINE>
    | id | varname | value |
    | -: | ------: | ----: |
    |  x |       a |     1 |
    |  x |       b |     2 |
    |  y |       a |     3 |
    |  y |       b |     4 |
    <BLANKLINE>
    %% daf rows=4; cols=3; keyfield=''; name=''
    <BLANKLINE>

    A round trip, with the names matched:

    >>> narrow = d.wide_to_narrow(['id'], varname_colname='variable')
    >>> narrow.narrow_to_wide(['id']).lol
    [['x', 1, 2], ['y', 3, 4]]



##### narrow_to_wide(id_cols: T_cs, varname_col: str = 'variable', value_col: str = 'value', wide_cols: T_cs | None = None) -> 'Daf'

Turn rows into columns. This is called pivot or spread.

The rows of one id may be anywhere in the table. The result has one row for
each id, in the order in which the ids first appear. The columns are the id
columns and then the names found in `varname_col`, in the order first seen. An
id that has no value for a name gets NULL. If an id has the same name more
than once, the last value is kept.

With `wide_cols`, those are the columns after the ids, in that order. A
name that is not listed is left out.

The default `varname_col` is `variable`, but `wide_to_narrow()` names that column
`varname` by default. Match the names to go back and forth.

An empty Daf gives an empty Daf.

Args:
    id_cols: The columns that identify a row.
    varname_col: The column whose values become the new column names.
    value_col: The column whose values fill the new columns.
    wide_cols: The names to use as columns, in order. If None, all names found.

Returns:
    The new Daf.

Examples:
    >>> d = Daf(lol=[['x', 'a', 1], ['y', 'a', 3], ['x', 'b', 2], ['y', 'b', 4]], cols=['id', 'variable', 'value'])
    >>> d.narrow_to_wide(['id'])
    <BLANKLINE>
    | id | a  | b  |
    | -: | -: | -: |
    |  x |  1 |  2 |
    |  y |  3 |  4 |
    <BLANKLINE>
    %% daf rows=2; cols=3; keyfield=''; name=''
    <BLANKLINE>
    >>> d.narrow_to_wide(['id'], wide_cols=['b', 'a'])
    <BLANKLINE>
    | id | b  | a  |
    | -: | -: | -: |
    |  x |  2 |  1 |
    |  y |  4 |  3 |
    <BLANKLINE>
    %% daf rows=2; cols=3; keyfield=''; name=''
    <BLANKLINE>



##### split_daf_into_ranges(chunk_ranges: List[Tuple[int, int]]) -> List['Daf']

Split the rows into several Daf instances, by position ranges.

Each range is `(start, end)`, and `end` is not included. The new Daf instances
share their rows with this one.

Args:
    chunk_ranges: The ranges of row positions.

Returns:
    A list of Daf instances, one for each range.

Examples:
    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
    >>> parts = d.split_daf_into_ranges([(0, 2), (2, 3)])
    >>> parts[0]
    <BLANKLINE>
    | id | v  | n  |
    | -: | -: | -: |
    |  1 |  a | 10 |
    |  2 |  b | 20 |
    <BLANKLINE>
    %% daf rows=2; cols=3; keyfield='id'; name=''
    <BLANKLINE>
    >>> parts[1]
    <BLANKLINE>
    | id | v  | n  |
    | -: | -: | -: |
    |  3 |  c | 30 |
    <BLANKLINE>
    %% daf rows=1; cols=3; keyfield='id'; name=''
    <BLANKLINE>



##### split_daf_into_chunks_lodaf(max_chunk_size: int) -> List['Daf']

Split the rows evenly into Daf instances of at most a given size.

The sizes are as equal as possible, so some chunks are smaller than the
maximum. None is larger. The chunks share their rows with this Daf.

Args:
    max_chunk_size: The most rows in one chunk.

Returns:
    A list of Daf instances.

Examples:
    >>> d = Daf(lol=[[i] for i in range(7)], cols=['a'])
    >>> [len(part) for part in d.split_daf_into_chunks_lodaf(3)]
    [3, 2, 2]

---

# Docs page: Daf API / Apply, reduce and group (api/daf/apply.md)

## Apply, reduce and group

Running functions over rows and columns, and grouping.

### Apply


##### apply(func: Callable[..., Any], by: str = 'row', keylist: T_la | T_lota | None = None, kwargs: Any = {}) -> 'Daf'

Apply a function to each row and collect the results in a new Daf.

Use it to convert one table into another, with a plain Python function that
turns one row into the new row.
This Daf is not changed. Only the new Daf gets rows.

The function gets a row and returns what to add to the new Daf:

- A dict adds one row.
- A list of dicts adds a row for each dict, so one row can become several.
- None, an empty dict or an empty list adds nothing, so the row is left out.

The new Daf takes its columns from the first row returned. A later dict is
placed by column name, and a column it lacks is NULL. A key that the first row
did not have is dropped without an error. So when the columns are known, have
the function return all of them in every row. The new Daf has no keyfield.

With `by='table'` the function gets the whole Daf, and its result is returned
as it is. Use that to run any function on the table. `by='col'` is not
supported.

The function gets the rows as dicts, or as KeyedList objects if `itermode` is
`keyedlist`. The extra keyword arguments are passed on to it. To work on only
some rows, give the `keylist`, or select them first.

Args:
    func: The function to apply. It takes a row, or the Daf, and the keyword arguments.
    by: `row` to apply to each row, or `table` to apply to the whole Daf.
    keylist: Keys of the rows to include. All rows if None. This may be removed.
    **kwargs: Keyword arguments passed on to the function.

Returns:
    The new Daf, or the result of the function for `by='table'`.

Raises:
    NotImplementedError: `by` is `col`, or is not recognized.

Examples:
    >>> d = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
    >>> d.apply(lambda row: {'g': row['g'].upper(), 'z': row['y'] * 2})
    <BLANKLINE>
    | g  | z  |
    | -: | -: |
    |  A | 20 |
    |  B | 40 |
    |  A | 60 |
    <BLANKLINE>
    %% daf rows=3; cols=2; keyfield=''; name=''
    <BLANKLINE>
    >>> d.apply(lambda row: row if row['y'] > 15 else None)
    <BLANKLINE>
    | g  | x  | y  |
    | -: | -: | -: |
    |  b |  2 | 20 |
    |  a |  3 | 30 |
    <BLANKLINE>
    %% daf rows=2; cols=3; keyfield=''; name=''
    <BLANKLINE>
    >>> d.apply(lambda row: [{'g': row['g'], 'part': 1}, {'g': row['g'], 'part': 2}] if row['x'] == 2 else None)
    <BLANKLINE>
    | g  | part |
    | -: | ---: |
    |  b |    1 |
    |  b |    2 |
    <BLANKLINE>
    %% daf rows=2; cols=2; keyfield=''; name=''
    <BLANKLINE>



##### apply_in_place(func: Callable[..., T_ma | None], by: str = 'row', rowkeys: T_la | T_lota | None = None, kwargs: Any = {}) -> 'Daf'

Apply a function to each row and store the results in this Daf.

With `by='row'` the function gets each row as a dict. It must return a dict.
Its values are written back by column name. A column that the dict lacks
keeps its value, and a key that is not a column is ignored. The order of
the keys does not matter, and the row keeps its length.

With `by='row_klist'` the function gets each row as a
[KeyedList][daffodil.keyedlist.KeyedList]. It changes the row and returns
nothing, so no dict is built for each row. The cost of `row` grows with the
number of columns, because each row is turned into a dict of all its columns.
The cost of `row_klist` does not. In a test that changed one column, the two
took the same time at about 6 columns. At 1,000 columns and 2,000 rows,
`row_klist` took 0.005 s and `row` took 0.245 s. With 3 columns and 200,000
rows, `row` was faster, 0.25 s against 0.30 s. Add or delete no keys in the
KeyedList, because that changes the length of the row, not the columns.

The key index is rebuilt when it is next needed. Use `apply()` to get a new Daf.

Args:
    func: The function to apply to each row. It takes a row and the keyword arguments.
    by: `row` or `row_klist`.
    rowkeys: Keys of the rows to include. All rows if None. The Daf needs a keyfield.
    **kwargs: Keyword arguments passed on to the function.

Returns:
    This Daf, which has been changed.

Raises:
    ValueError: With `by='row'` the function returned None.
    NotImplementedError: `by` is not `row` or `row_klist`.

Examples:
    >>> d = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
    >>> _ = d.apply_in_place(lambda row: {**row, 'y': row['y'] + 1})
    >>> d.col('y')
    [11, 21, 31]
    >>> _ = d.apply_in_place(lambda row: row.__setitem__('y', 0), by='row_klist')
    >>> d.col('y')
    [0, 0, 0]



##### update_row(row: T_ma, da: T_da) -> T_ma

Update a row with the items of a dict, and return the row.

This is a small helper to use inside `apply()`, as in
`d.apply(lambda row: Daf.update_row(row, {'z': 0}))`. It is a static method.

Args:
    row: The row, as a dict. It is changed.
    da: The items to put in the row.

Returns:
    The same row.

Examples:
    >>> Daf.update_row({'a': 1}, {'b': 2})
    {'a': 1, 'b': 2}



##### apply_colwise(target_col: str, func: Callable[[T_da], Any], default: Any = 0) -> 'Daf'

Compute one column from the other columns of each row, in place.

The function gets each row, as a dict, and returns the value for `target_col`.
If the function raises an error for a row, that row gets `default`. If
`target_col` is not a column, it is added at the right, and the rows are
filled with `default` first. Use it for a ratio or a total of other columns.

Args:
    target_col: The column to store the result in.
    func: A function that takes a row and returns the value.
    default: The value for a row where the function raises an error.

Returns:
    This Daf, which has been changed.

Examples:
    >>> d = Daf(lol=[[1, 2], [3, 0]], cols=['a', 'b'])
    >>> d.apply_colwise('ratio', lambda row: row['a'] / row['b'], default=-1.0)
    <BLANKLINE>
    | a  | b  | ratio |
    | -: | -: | ----: |
    |  1 |  2 |   0.5 |
    |  3 |  0 |  -1.0 |
    <BLANKLINE>
    %% daf rows=2; cols=3; keyfield=''; name=''
    <BLANKLINE>



##### apply_to_col(col: str, func: Callable, kwargs: Any = {}) -> 'Daf'

Replace each value of a column by the result of a function, in place.

Args:
    col: The column name.
    func: A function that takes a value and returns the new value.
    **kwargs: Keyword arguments passed on to the function with each value.

Returns:
    This Daf, which has been changed.

Examples:
    >>> d = Daf(lol=[[1, 5], [2, 6]], cols=['a', 'b'])
    >>> _ = d.apply_to_col('b', lambda value: value * 2)
    >>> d.col('b')
    [10, 12]
    >>> _ = d.apply_to_col('b', lambda value, factor: value * factor, factor=10)
    >>> d.col('b')
    [100, 120]



##### apply_formulas(formulas_daf: 'Daf') -> 'Daf'

Fill cells from spreadsheet like formulas, in place.

`formulas_daf` is a Daf of the same shape. Each cell holds a Python
expression as text. An empty cell is skipped. The result of each expression is
stored in the same cell of this Daf. The formulas are evaluated again and again
until no cell changes, so a cell may use the result of another. If the cells
still change after 100 passes, it raises `RuntimeError`. This is a limit on the
passes, not a check for circular formulas. A circle of formulas that settles, such
as two cells that copy each other, raises nothing.

In a formula, `$d` is this Daf, `$r` is the row number of the cell and `$c` is
its column number. References are absolute unless you build them from `$r` and
`$c`. So `sum($d[$r, :$c])` is the sum of the cells to the left in the same row.

Other examples:

    $d[14,20]+$d[15,25]       the sum of two cells
    max(0,$d[($r-1),$c])      the cell above, but not below 0
    $d[($r-1),$c] * 0.15      15 percent of the cell above

Warning: the formulas are run with `eval()`. Never use formulas from a source
you do not trust.

An error in a formula prints the cell and the formula, and is raised again. The
`retmode` of this Daf is restored, and the key index is rebuilt when it is next
needed. The cells that were already changed stay changed.

Args:
    formulas_daf: The formulas, with the same shape as this Daf.

Returns:
    This Daf, which has been changed.

Raises:
    RuntimeError: The shapes differ, or the formulas never settle.

Examples:
    >>> d = Daf(cols=['A', 'B', 'C'], lol=[[1, 2, 0], [4, 5, 0], [7, 8, 0], [0, 0, 0]])
    >>> f = Daf(cols=['A', 'B', 'C'], lol=[
    ...     ['', '', 'sum($d[$r,:$c])'],
    ...     ['', '', 'sum($d[$r,:$c])'],
    ...     ['', '', 'sum($d[$r,:$c])'],
    ...     ['sum($d[:$r,$c])', 'sum($d[:$r,$c])', 'sum($d[:$r,$c])']])
    >>> _ = d.apply_formulas(f)
    >>> d
    <BLANKLINE>
    | A  | B  | C  |
    | -: | -: | -: |
    |  1 |  2 |  3 |
    |  4 |  5 |  9 |
    |  7 |  8 | 15 |
    | 12 | 15 | 27 |
    <BLANKLINE>
    %% daf rows=4; cols=3; keyfield=''; name=''
    <BLANKLINE>


### Reduce


##### reduce(func: Callable[..., Any], by: str = 'row', cols: T_cs | None = None, initial_da: T_ma | None = None, indirect_col: str = '', silent_error: bool = False, kwargs: Any = {}) -> T_ma | T_la

Combine all the rows into one result, using a function.

The function gets each row, and the result so far, and returns the new result.
A sum or a count is a reduction. With `by='row'`, the function is called as
`func(row, result, cols=cols, **kwargs)`. The result starts as 0 for each of
the columns, or as `initial_da`, which is changed in place. It returns a dict with
every column. Columns that are not in `cols` are empty. A key of the result that
is not a column of the Daf is dropped, so a reduction by row gives one value for
each column. For a result of another shape, use `by='table'`, or a loop.

A plain sum, with `sum_da()` or `sum_kl()` and no other options, adds down each
column instead of across each row. That is about 5 times faster, and gives the same
totals. The function is then not called for each row.

With `by='row'`, each row is a new [KeyedList][daffodil.keyedlist.KeyedList]
that points at the row of the table. It reads like a dict, as in `row['x']` or
`row.get('x', 0)`. Writing to it changes the table. A function that needs a real
dict, for example to pass to `json.dumps()`, can call `dict(row)`. A row that is
shorter or longer than the columns gets a KeyedList of a copy, padded with NULL
or cut to fit.

The standard functions are `sum_kl()` and `count_values_kl()`. They read the
KeyedList directly. `Daf.sum_da` and `Daf.count_values_da` are replaced by them
here, since they give the same results.

With `by='col'` the function gets each column as a list and the result so far
as a list. With `by='table'` it gets this Daf and `cols`, and its result is
returned as it is. With `by='sparse_row'`, rows are read from the dict in
`indirect_col`, and the result starts as `initial_da` or an empty dict.

An error in the function stops the reduction. With `silent_error=True` the
row is skipped. The result may then be missing rows, with no sign of it. An
empty Daf gives an empty dict for `by='row'`.

To reduce part of the table, select the rows or columns first, or give `cols`.

Args:
    func: The reduction function. For example `Daf.sum_da`.
    by: `row`, `col`, `table` or `sparse_row`.
    cols: The columns included in the reduction. All columns if None.
    initial_da: The result to start from, instead of zeros.
    indirect_col: A column that holds a dict. Required for `sparse_row`.
    silent_error: If True, a row for which the function raises is skipped.
    **kwargs: Keyword arguments passed on to the function.

Returns:
    A dict for `row`, `table` and `sparse_row`. A list for `col`.

Raises:
    ValueError: `sparse_row` is used with no `indirect_col`.
    NotImplementedError: `by` is not recognized.

Examples:
    >>> d = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
    >>> d.reduce(Daf.sum_da, cols=['x', 'y'])
    {'g': '', 'x': 6, 'y': 60}



##### daf_sum(by: str = 'row', cols: T_cs | None = None, indirect_col: str = '', kwargs: Any = {}) -> T_ma

Add up the columns, using `reduce()` and `sum_kl()`.

Cells that cannot be added, such as text, are skipped with no error. See
[sum_kl][daffodil.daf.Daf.sum_kl] for what that means for a bad number. A column
that is not in `cols` is empty in the result.

Args:
    by: How the function is applied. See `reduce()`.
    cols: The columns to sum. All if None.
    indirect_col: A column that holds a dict. Required for `sparse_row`.
    **kwargs: Keyword arguments passed on to `reduce()`.

Returns:
    A dict with a total for each column.

Examples:
    >>> d = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
    >>> d.daf_sum(cols=['y'])
    {'g': '', 'x': '', 'y': 60}



##### sum(colnames_ls: T_cs | None = None, numeric_only: bool = False) -> dict

Total the columns, and return a dict of the totals.

Each total starts as a float. Empty cells are skipped. Text that is a number,
such as `'2'`, is converted and added. Other text raises `ValueError`, so give
`colnames_ls` to leave those columns out.

`reduce(Daf.sum_da)` adds differently. It skips every value that is not a number,
including the text `'2'`, and never raises. So the two can give different totals
for a column that holds text. See the example. With `numeric_only`, and dtypes of `int` or `float`, only
those columns are totaled, and a cell that is not a number counts as 0. The
totals are converted to the dtypes, if the Daf has them.

Args:
    colnames_ls: The columns to total. All if None.
    numeric_only: If True, total only the columns with an `int` or `float` dtype.

Returns:
    A dict that maps each column name to its total.

Raises:
    ValueError: A cell cannot be converted to a number.

Examples:
    >>> Daf(lol=[[1, 10], [2, 20]], cols=['x', 'y']).sum()
    {'x': 3.0, 'y': 30.0}
    >>> mixed = Daf(cols=['n'], lol=[[1], ['2'], [''], [3.5]])
    >>> mixed.sum(['n']), mixed.reduce(Daf.sum_da, cols=['n'])
    ({'n': 6.5}, {'n': 4.5})



##### sum_np(colnames_ls: T_cs | None = None) -> dict

Total the columns with NumPy, and return a dict of the totals.

This needs NumPy. A blank, `None` or NaN cell counts as 0, as it does in `sum()`.
Use `colnames_ls` to pass only the columns that hold numbers. A column that holds
text raises `TypeError` and names the column.

Args:
    colnames_ls: The columns to total. All if None.

Returns:
    A dict that maps each column name to its total. An empty Daf gives an empty dict.

Raises:
    KeyError: A name in `colnames_ls` is not a column.
    TypeError: A column holds text or other values that cannot be added.

Examples:
    >>> Daf(lol=[[1, 10], [2, 20]], cols=['x', 'y']).sum_np()
    {'x': 3, 'y': 30}
    >>> Daf(lol=[[1, 10], [2, '']], cols=['x', 'y']).sum_np()
    {'x': 3, 'y': 10}



##### sum_kl(row_kl: Any, reduction_da: T_ma, cols: T_cs | None = None, astype: Type | None = None, is_sparse: bool = False, diagnose: bool = False) -> T_ma

Add the values of a row to a running total. This is the standard sum for `reduce()`.

It gives the same result as `sum_da()`, and is faster on the rows that
`reduce()` gives it, which are [KeyedList][daffodil.keyedlist.KeyedList]
objects. It reads the values of the row directly, instead of looking up each
one through the row. `reduce()` uses it in place of `sum_da()`, so a caller
that passes `Daf.sum_da` gets this speed too.

A value that cannot be added, such as text or an empty cell, is skipped, with no
error. That keeps the sum fast, and lets a sum of all columns pass over the text
columns. But a bad number, such as `abc` in an `int` column, is then left out of
the total with no sign. Call `check_schema()` first to find such cells.

A column that the row does not have adds 0. With `astype` or `is_sparse`, or a
row that is not a KeyedList, the work is passed to `sum_da()`.

Args:
    row_kl: The current row.
    reduction_da: The running total. It is changed.
    cols: The columns to add. All if None.
    astype: A type to convert each value to before adding.
    is_sparse: If True, the row may hold only some of the columns.
    diagnose: Not used.

Returns:
    The running total.

Examples:
    >>> from daffodil.keyedlist import KeyedList
    >>> Daf.sum_kl(KeyedList(['x', 'y'], [1, 'abc']), {'x': 10, 'y': 0}, cols=['x', 'y'])
    {'x': 11, 'y': 0}



##### sum_da(row_da: T_ma, reduction_da: T_ma, cols: T_cs | None = None, astype: Type | None = None, is_sparse: bool = False, diagnose: bool = False) -> T_ma

Add the values of a row to a running total. Use it with `reduce()`.

This is a static method. A value that cannot be added, such as text or an empty
cell, is skipped with no error. See [sum_kl][daffodil.daf.Daf.sum_kl] for what that
means for a bad number. The total is changed and returned. With `astype`, each value
is first converted to that type. `reduce()` uses `sum_kl()` in its place.

Args:
    row_da: The current row.
    reduction_da: The running total. It is changed.
    cols: The columns to add. All if None.
    astype: A type to convert each value to before adding.
    is_sparse: If True, the row may hold only some of the columns.
    diagnose: Not used.

Returns:
    The running total.

Examples:
    >>> Daf.sum_da({'x': 1, 'y': 2}, {'x': 10, 'y': 0}, cols=['x', 'y'])
    {'x': 11, 'y': 2}



##### diff_da(d1_da: T_ma, d2_da: T_ma, keys: T_ls | str | None = None) -> T_da

Subtract one dict from another, for the keys you name.

This is a static method. A key that is missing, or an empty value, counts as
0. Keys that you do not name are left out. The values must be numbers.

Args:
    d1_da: The first dict.
    d2_da: The dict to subtract.
    keys: The key or keys to include. If None, the result is empty.

Returns:
    A dict of the differences.

Examples:
    >>> Daf.diff_da({'a': 5, 't': 'x'}, {'a': 2, 't': 'y'}, keys=['a'])
    {'a': 3}



##### count_values_kl(row_kl: Any, reduction_da: T_da, cols: T_cs, omit_nulls: bool = False) -> T_da

Add one row to running counts of the values in each column. This is the standard count for `reduce()`.

It gives the same result as `count_values_da()`, and is faster on the rows that
`reduce()` gives it, which are [KeyedList][daffodil.keyedlist.KeyedList]
objects. `reduce()` uses it in place of `count_values_da()`.

A cell that holds a dict is passed to `count_values_da()`, one column at a
time, as is a row that is not a KeyedList. A value that cannot be a dict key,
such as a list, raises TypeError. Leave a column of lists out with `cols`.

Args:
    row_kl: The current row.
    reduction_da: The running counts. They are changed.
    cols: The columns to count.
    omit_nulls: If True, empty cells are not counted.

Returns:
    The running counts.

Raises:
    TypeError: A cell holds a value that cannot be counted, such as a list.

Examples:
    >>> from daffodil.keyedlist import KeyedList
    >>> Daf.count_values_kl(KeyedList(['g'], ['a']), {'g': {'a': 2}}, ['g'])
    {'g': {'a': 3}}



##### count_values_da(row_da: T_ma, reduction_da: T_da, cols: T_cs, omit_nulls: bool = False) -> T_da

Add one row to running counts of the values in each column. Use it with `reduce()`.

This is a static method. The counts are a dict that maps each column to a dict
of value and count. The counts are changed and returned. A cell that holds a
dict is added to the counts for that column with `sum_da()`. Use `omit_nulls`
to skip empty cells.

A value that cannot be a dict key, such as a list, raises TypeError. There is
no check for it beforehand, to keep the count fast. Leave a column of lists
out with `cols`.
`reduce()` uses `count_values_kl()` in its place.

Args:
    row_da: The current row.
    reduction_da: The running counts. They are changed.
    cols: The columns to count.
    omit_nulls: If True, empty cells are not counted.

Returns:
    The running counts.

Raises:
    TypeError: A cell holds a value that cannot be counted, such as a list.

Examples:
    >>> Daf.count_values_da({'g': 'a'}, {}, ['g'])
    {'g': {'a': 1}}



##### sum_dodis(this_dodi: T_dodi, accum_dodi: T_dodi) -> None

Add one dict of dicts of numbers into another, in place.

This is a static method. For each key, the numbers of the inner dicts are
added with `sum_da()`. A key that is new is stored as it is, not copied.

Args:
    this_dodi: The counts to add.
    accum_dodi: The running totals. They are changed.

Examples:
    >>> total = {'c': {'x': 2, 'y': 1}}
    >>> Daf.sum_dodis({'c': {'x': 1}}, total)
    >>> total
    {'c': {'x': 3, 'y': 1}}


### Group


##### groupby(colname: str = '', colnames: T_ls | None = None, omit_nulls: bool = False, cols: T_cs | None = None) -> T_klodaf

Split the Daf into several Daf instances, one for each value of a column.

The result is a [KeyedList][daffodil.keyedlist.KeyedList] of Daf, a `klodaf`. It
works as a dict: `groups[key]`, `keys()`, `items()`, `len()` and `in`. Its values
also have positions, so `groups.values()[0]` is the first group. Each key is a value
found in the column, in the order first seen. Each value is a Daf of the rows that
have it, with all columns, or only the columns in `cols`.

With all columns, a group holds the same row lists as this Daf. Nothing is copied,
so it is fast and uses little memory. But a cell changed in a group is also changed
here. To keep this Daf as it is, group a copy made with `copy(level='editable')`.
Adding a column to a group makes new rows, and this Daf does not get the column.

Use `cols` when you need only a few of many columns. The rows are then new lists of
the columns kept. The other columns are never copied, and this Daf is not changed.
It is much faster for a wide table.

With several columns, as a list, the keys are tuples of their values. See
`groupby_cols()`. With `omit_nulls`, rows that have an empty value in the
column are left out.

Args:
    colname: The column to group by.
    colnames: Several columns to group by. Use this or `colname`.
    omit_nulls: If True, leave out rows that have an empty value.
    cols: The columns that each group keeps, in this order. If None, all columns.
        A keyfield is kept only if all its columns are kept.

Returns:
    A KeyedList that maps each value, or tuple of values, to a Daf.

Raises:
    KeyError: The group column, or a name in `cols`, is not a column.

Examples:
    >>> d = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
    >>> groups = d.groupby('g')
    >>> list(groups)
    ['a', 'b']
    >>> groups['a']
    <BLANKLINE>
    | g  | x  | y  |
    | -: | -: | -: |
    |  a |  1 | 10 |
    |  a |  3 | 30 |
    <BLANKLINE>
    %% daf rows=2; cols=3; keyfield=''; name=''
    <BLANKLINE>
    >>> d.groupby('g', cols=['y'])['a']
    <BLANKLINE>
    | y  |
    | -: |
    | 10 |
    | 30 |
    <BLANKLINE>
    %% daf rows=2; cols=1; keyfield=''; name=''
    <BLANKLINE>



##### groupby_cols(colnames: T_ls, cols: T_cs | None = None) -> T_klodaf

Split the Daf by the values of several columns.

The result is a KeyedList of Daf, as `groupby()` gives. Each key is a tuple of the
values in the columns, even for one column. Each value is a Daf of the rows that have them, with all
columns, or only the columns in `cols`. With all columns, the rows are not
copied. With `cols`, they are new lists of the columns kept, and the other
columns are never copied.

Args:
    colnames: The columns to group by.
    cols: The columns that each group keeps, in this order. If None, all columns.
        A keyfield is kept only if all its columns are kept.

Returns:
    A KeyedList that maps each tuple of values to a Daf.

Raises:
    KeyError: A name in `colnames` or `cols` is not a column.

Examples:
    >>> d = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
    >>> list(d.groupby_cols(['g']))
    [('a',), ('b',)]
    >>> d.groupby_cols(['g'], cols=['y'])[('a',)]
    <BLANKLINE>
    | y  |
    | -: |
    | 10 |
    | 30 |
    <BLANKLINE>
    %% daf rows=2; cols=1; keyfield=''; name=''
    <BLANKLINE>



##### group_where(where: Callable[[Any], Any], indirect_col: str | None = None) -> T_klodaf

Group the rows by the result of a function.

The function gets each row and returns a key, or a list of keys, or None. A row
goes into the group of each key. A list of keys puts the same row in several
groups. None leaves the row out. The groups are Daf instances in a KeyedList, as
`groupby()` gives. The rows are not copied, so changing a cell in a group changes it
here too.

To split the rows in two, have the function return True or False. A side with no
rows then has no group, so read each side with `.get()`.

Args:
    where: The function. It gets a row and returns None, a key, or an iterable of keys.
    indirect_col: A column that holds a dict, to read names that are not columns from.

Returns:
    A KeyedList that maps each key to a Daf.

Examples:
    >>> d = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
    >>> groups = d.group_where(lambda row: [row['g'], 'all'])
    >>> {key: len(group) for key, group in groups.items()}
    {'a': 2, 'all': 3, 'b': 1}



##### groupby_cols_reduce(groupby_colnames: T_ls, func: Callable[..., Any], by: str = 'row', reduce_cols: T_la | None = None, diagnose: bool = False, kwargs: Any = {}) -> 'Daf'

Group the rows by several columns and reduce each group to one row.

Use it to total the numbers for each combination of a few identifying columns.
The result has one row for each combination. It has the group columns first and
then the `reduce_cols`. It has no keyfield. For each group, `func` is used as in
`reduce()`.

With `reduce_cols`, and `by` of `row` or `sparse_row`, each group holds only those
columns, and the `indirect_col` if there is one. A function that reads another
column of the row will not find it.

For example, group by gender, religion and zip code, and sum the counts of
several causes in each group. The number of rows is the number of combinations
that occur.

Args:
    groupby_colnames: The columns that identify a group.
    func: The reduction function. See `reduce()`.
    by: How the function is applied. See `reduce()`.
    reduce_cols: The columns to reduce.
    diagnose: If True, print progress messages.
    **kwargs: Keyword arguments passed on to the function.

Returns:
    The Daf with one row for each group.

Examples:
    >>> d = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
    >>> d.groupby_cols_reduce(['g'], Daf.sum_da, reduce_cols=['y'])
    <BLANKLINE>
    | g  | y  |
    | -: | -: |
    |  a | 40 |
    |  b | 20 |
    <BLANKLINE>
    %% daf rows=2; cols=2; keyfield=''; name=''
    <BLANKLINE>



##### groupby_reduce(colname: str, func: Callable[..., Any], by: str = 'row', reduce_cols: T_cs | None = None, diagnose: bool = False, kwargs: Any = {}) -> 'Daf'

Group the rows by one column and reduce each group to one row.

The result has one row for each value in the column, and its keyfield is that
column. The columns in `reduce_cols` hold the reduced values. Other columns are
empty. For each group, `func` is used as in `reduce()`.

With `reduce_cols`, and `by` of `row` or `sparse_row`, each group holds only those
columns, and the `indirect_col` if there is one. The other columns are never copied,
which is much faster for a wide table. A function that reads another column of the
row will not find it. The result still has all the columns of this Daf, and the
columns that are not reduced are empty.

Args:
    colname: The column to group by.
    func: The reduction function. See `reduce()`.
    by: How the function is applied. See `reduce()`.
    reduce_cols: The columns to reduce. All except `colname` if None.
    diagnose: If True, print progress messages.
    **kwargs: Keyword arguments passed on to the function.

Returns:
    The Daf with one row for each group.

Examples:
    >>> d = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
    >>> d.groupby_reduce('g', Daf.sum_da, reduce_cols=['y'])
    <BLANKLINE>
    | g  | x  | y  |
    | -: | -: | -: |
    |  a |    | 40 |
    |  b |    | 20 |
    <BLANKLINE>
    %% daf rows=2; cols=3; keyfield='g'; name=''
    <BLANKLINE>



##### multi_groupby(groupby_colnames: T_cs, colnames: T_cs | None = None, omit_nulls: bool = False) -> Dict[str, Dict[str, 'Daf']]

Group the rows by each of several columns, one column at a time.

The result is a dict of dicts. The first key is the column. The second key is a
value in that column. Each innermost value is a Daf of the rows that have it,
with all columns, or only the columns in `colnames`.
This is not a grouping by combinations. Use `groupby_cols()` for that.
The groups are not reduced.

With all columns, the groups hold the same row lists as this Daf, as in
`groupby()`. A cell changed in a group is also changed here, and in the groups of
the other columns. To keep this Daf as it is, group a copy made with
`copy(level='editable')`.

Use `colnames` when you need only a few of many columns. The rows are then new lists
of the columns kept. The other columns are never copied, and this Daf is not changed.
It is much faster for a wide table.

Args:
    groupby_colnames: The columns to group by, one at a time.
    colnames: The columns that each group keeps, in this order. If None, all columns.
        A keyfield is kept only if all its columns are kept.
    omit_nulls: If True, leave out rows that have an empty value.

Returns:
    A dict that maps each column to a dict of value and Daf.

Raises:
    KeyError: A group column, or a name in `colnames`, is not a column.

Examples:
    >>> d = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
    >>> groups = d.multi_groupby(['g', 'x'])
    >>> list(groups), list(groups['g'])
    (['g', 'x'], ['a', 'b'])
    >>> d.multi_groupby(['g'], colnames=['y'])['g']['a']
    <BLANKLINE>
    | y  |
    | -: |
    | 10 |
    | 30 |
    <BLANKLINE>
    %% daf rows=2; cols=1; keyfield=''; name=''
    <BLANKLINE>



##### multi_groupby_reduce(colnames: T_cs, func: Callable[..., Any], by: str = 'row', reduce_cols: T_cs | None = None, diagnose: bool = False, kwargs: Any = {}) -> Dict[str, 'Daf']

Group by each of several columns and reduce each group to one row.

This is `multi_groupby()` followed by `groupby_reduce()` for each column. The
result is a dict. Each key is a column. Each value is a Daf with one row for
each value of that column, and that column as its keyfield.

With `reduce_cols`, and `by` of `row` or `sparse_row`, each group holds only those
columns, and the `indirect_col` if there is one. A function that reads another
column of the row will not find it. The result still has all the columns of this
Daf, and the columns that are not reduced are empty.

Args:
    colnames: The columns to group by, one at a time.
    func: The reduction function. See `reduce()`.
    by: How the function is applied. See `reduce()`.
    reduce_cols: The columns to reduce.
    diagnose: If True, print progress messages.
    **kwargs: Keyword arguments passed on to the function.

Returns:
    A dict that maps each column to its Daf.

Examples:
    >>> d = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
    >>> d.multi_groupby_reduce(['g'], Daf.sum_da, reduce_cols=['y'])['g']
    <BLANKLINE>
    | g  | x  | y  |
    | -: | -: | -: |
    |  a |    | 40 |
    |  b |    | 20 |
    <BLANKLINE>
    %% daf rows=2; cols=3; keyfield='g'; name=''
    <BLANKLINE>



##### reduce_dodaf_to_daf(colname: str, func: Callable[..., Any], grouped_dodaf: T_dodaf, reduce_cols: T_la | None = None, diagnose: bool = False, all_cols: T_ls | None = None, kwargs: Any = {}) -> 'Daf'

Reduce each Daf in a dict to one row, and join the rows in a Daf.

This is the second half of `groupby_reduce()`. The dict maps the values of a
column to Daf instances. Each Daf is reduced with `reduce()`. The value is
stored in the column `colname` of the row. The result has `colname` as its keyfield.

Args:
    colname: The column that holds the value of each group.
    func: The reduction function. See `reduce()`.
    grouped_dodaf: A dict that maps each value to a Daf.
    reduce_cols: The columns to reduce. All columns except `colname` if None.
    diagnose: If True, print progress messages.
    all_cols: The columns of the result, in order. Use it when each group holds only
        some columns, so that the rows have the same columns as for whole groups. The
        columns that are not reduced are then empty.
    **kwargs: Keyword arguments passed on to the function.

Returns:
    The Daf with one row for each group.

Examples:
    >>> groups = {'a': Daf(lol=[[1], [2]], cols=['n']), 'b': Daf(lol=[[3]], cols=['n'])}
    >>> result = Daf.reduce_dodaf_to_daf('g', Daf.sum_da, groups)
    >>> result
    <BLANKLINE>
    | n  | g  |
    | -: | -: |
    |  3 |  a |
    |  3 |  b |
    <BLANKLINE>
    %% daf rows=2; cols=2; keyfield='g'; name=''
    <BLANKLINE>



##### groupsum_daf(colname: str, by: str = 'row', reduce_cols: T_cs | None = None) -> 'Daf'

Group by a column and add up the other columns of each group.

This is `groupby_reduce()` with `sum_da()`.

Args:
    colname: The column to group by.
    by: How the function is applied. See `reduce()`.
    reduce_cols: The columns to add.

Returns:
    The Daf with one row for each group.

Examples:
    >>> d = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
    >>> d.groupsum_daf('g', reduce_cols=['y'])
    <BLANKLINE>
    | g  | x  | y  |
    | -: | -: | -: |
    |  a |    | 40 |
    |  b |    | 20 |
    <BLANKLINE>
    %% daf rows=2; cols=3; keyfield='g'; name=''
    <BLANKLINE>



##### multi_groupsum(colnames: T_cs | None = None, by: str = 'row', reduce_cols: T_cs | None = None) -> Dict[str, 'Daf']

Group by each of several columns, one at a time, and add up the columns.

This is `multi_groupby_reduce()` with `sum_da()`.

Args:
    colnames: The columns to group by. These are required.
    by: How the function is applied. See `reduce()`.
    reduce_cols: The columns to add.

Returns:
    A dict that maps each column to a Daf with one row for each value.

Raises:
    ValueError: No `colnames` are given.

Examples:
    >>> d = Daf(lol=[['a', 'x', 1], ['a', 'y', 2], ['b', 'x', 3]], cols=['g', 'k', 'n'])
    >>> sums = d.multi_groupsum(colnames=['g', 'k'], reduce_cols=['n'])
    >>> sums['g']
    <BLANKLINE>
    | g  | k  | n  |
    | -: | -: | -: |
    |  a |    |  3 |
    |  b |    |  3 |
    <BLANKLINE>
    %% daf rows=2; cols=3; keyfield='g'; name=''
    <BLANKLINE>
    >>> sums['k']
    <BLANKLINE>
    | g  | k  | n  |
    | -: | -: | -: |
    |    |  x |  4 |
    |    |  y |  2 |
    <BLANKLINE>
    %% daf rows=2; cols=3; keyfield='k'; name=''
    <BLANKLINE>
    >>> d.multi_groupsum()
    Traceback (most recent call last):
        ...
    ValueError: multi_groupsum: colnames is required


### Manifest processing


##### manifest_apply(func: Callable[[T_da, T_la | None], Tuple[T_da, 'Daf']], load_func: Callable[[T_ma], 'Daf'], save_func: Callable[[T_da, 'Daf'], str], by: str = 'row', cols: T_la | None = None, kwargs: Any = {}) -> 'Daf'

Run a function on each chunk that a manifest lists, and save the results.

A manifest is a Daf in which each row describes one chunk of data. For each
row, `load_func` loads the chunk as a Daf. `func` is applied to it with
`by='table'`, so it gets the loaded Daf and the keyword `cols`. It returns a
tuple of a dict that describes the result chunk and the new Daf. `save_func`
saves the new Daf. The result manifest has one row for each dict.

Args:
    func: Gets a loaded Daf. Returns a dict that describes the result and the new Daf.
    load_func: Loads the chunk that a manifest row describes.
    save_func: Saves a new Daf, given the dict that describes it.
    by: Must be `table`. The default is `row`, which applies `func` to each row, so always pass `by='table'`.
    cols: Passed to `func` as the keyword `cols`.
    **kwargs: Keyword arguments passed on to `func`.

Returns:
    The manifest of the result chunks.

Examples:
    >>> chunks = {'a': Daf(cols=['n'], lol=[[1], [2]]), 'b': Daf(cols=['n'], lol=[[10]])}
    >>> saved = []
    >>> def load(spec):
    ...     return chunks[spec['chunk']]
    >>> def double(daf, cols=None):
    ...     new = Daf(cols=['n'], lol=[[row['n'] * 2] for row in daf])
    ...     return {'rows': len(new)}, new
    >>> def save(spec, daf):
    ...     saved.append(daf.to_lod())
    ...     return 'saved'
    >>> manifest = Daf(cols=['chunk'], lol=[['a'], ['b']])
    >>> result = manifest.manifest_apply(double, load, save, by='table')
    >>> result
    <BLANKLINE>
    | rows |
    | ---: |
    |    2 |
    |    1 |
    <BLANKLINE>
    %% daf rows=2; cols=1; keyfield=''; name=''
    <BLANKLINE>
    >>> saved
    [[{'n': 2}, {'n': 4}], [{'n': 20}]]



##### manifest_reduce(func: Callable[..., Any], load_func: Callable[[T_ma], 'Daf'] | None = None, by: str = 'row', cols: T_la | None = None, kwargs: Any = {}) -> T_da

Reduce the chunks that a manifest lists into one row.

Each chunk is loaded with `load_func` and reduced with `reduce()`. The
reductions are put in a Daf, and that is reduced again with the same function.
This works for functions such as `sum_da()` that can be combined in this way.

Args:
    func: The reduction function. See `reduce()`.
    load_func: Loads the chunk that a manifest row describes. It is required.
    by: How the function is applied. See `reduce()`.
    cols: The columns to reduce. All columns if None.
    **kwargs: Keyword arguments passed on to the function.

Returns:
    The reduced row, as a dict.

Raises:
    ValueError: No `load_func` is given.

Examples:
    >>> store = {'c1': Daf(lol=[[1, 2], [3, 4]], cols=['a', 'b']), 'c2': Daf(lol=[[10, 20]], cols=['a', 'b'])}
    >>> manifest = Daf(lol=[['c1'], ['c2']], cols=['chunk'])
    >>> manifest.manifest_reduce(Daf.sum_da, load_func=lambda spec: store[spec['chunk']])
    {'a': 14, 'b': 26}



##### manifest_process(func: Callable[..., T_da], kwargs: Any = {}) -> 'Daf'

Call a function once for each chunk that a manifest lists.

The function gets the manifest row as a dict, and does its own loading and
saving. It returns a dict of information about what it did. These dicts are
collected as the rows of the result.

Args:
    func: Gets a manifest row and returns a dict of results.
    **kwargs: Keyword arguments passed on to the function.

Returns:
    A Daf with one row for each returned dict.

Examples:
    >>> manifest = Daf(lol=[['c1'], ['c2']], cols=['chunk'])
    >>> manifest.manifest_process(lambda spec: {'chunk': spec['chunk'], 'seen': True})
    <BLANKLINE>
    | chunk | seen |
    | ----: | ---: |
    |    c1 | True |
    |    c2 | True |
    <BLANKLINE>
    %% daf rows=2; cols=2; keyfield=''; name=''
    <BLANKLINE>


### Regex and settings


##### set_col2_from_col1_using_regex_select(col1: str, col2: str = '', regex: str = '') -> 'Daf'

Fill a column with the part of another column that a regex selects, in place.

`regex` must have one pair of parentheses around the part to keep. A cell that
does not match gives an empty cell. Give `regex` as a keyword. Without `col2`,
`col1` is changed. If `col2` is not a column, it is added at the right.

Args:
    col1: The column to read.
    col2: The column to write. Defaults to `col1`.
    regex: A regular expression with one group.

Returns:
    This Daf, which has been changed.

Examples:
    >>> d = Daf(lol=[[1, 'ab12'], [2, 'cd34']], cols=['id', 's'])
    >>> _ = d.set_col2_from_col1_using_regex_select('s', regex=r'(\d+)')
    >>> d.col('s')
    ['12', '34']



##### apply_replace_regex(col: str, col2: str = '', replace_regex: str = '') -> 'Daf'

Change a column with a pattern of the form `/find/replace/`, in place.

The pattern has three parts that are separated by `/`. The first is a regular
expression. The second is what replaces the part it finds. The groups it
finds can be used as `\1`. A pattern with an empty second part removes the text.

    /find//                       remove
    /find/replace/                replace
    /pre(select)post/\1/          keep only the selected part
    /pre(select)post/a\1b/        keep it, with new text around it

The result goes to `col2`, or to `col` if there is no `col2`. A column that is
not found does nothing. If `col2` is not a column, it is added at the right.

Args:
    col: The column to read.
    col2: The column to write. Defaults to `col`.
    replace_regex: The pattern.

Returns:
    This Daf, which has been changed.

Examples:
    >>> d = Daf(lol=[[1, 'ab12']], cols=['id', 's'])
    >>> d.apply_replace_regex('s', replace_regex='/ab//').col('s')
    ['12']



##### alter_daf_per_setting(settingsdict: T_da, setting_name: str, setting_select_dict: dict, silent_error: bool = False) -> 'Daf'

Change this Daf with the replace patterns found in a settings dict.

The setting is a dict, or a list of dicts. Each has a `spec_name`, a `colname`
and a `replace_regex`. The ones whose fields match `setting_select_dict` are
used. For each, `apply_replace_regex()` changes the column. Use it to give
different edits to different files, such as fixing ids in one source.

A setting that is None or empty changes nothing. A name that is not in the
settings dict raises `KeyError`, unless `silent_error` is True.

Args:
    settingsdict: A dict of settings.
    setting_name: The key of the setting in the dict.
    setting_select_dict: Selects which specs apply, such as `{'spec_name': 'file1.csv'}`.
    silent_error: If True, a missing setting changes nothing.

Returns:
    This Daf, which has been changed.

Raises:
    KeyError: The setting is missing and `silent_error` is False.

Examples:
    >>> d = Daf(lol=[['04_1'], ['05_2']], cols=['ballot_id'])
    >>> spec = {'spec_name': 'a.zip', 'colname': 'ballot_id', 'replace_regex': r'/^04_/14_/'}
    >>> d.alter_daf_per_setting({'fix': [spec]}, 'fix', {'spec_name': 'a.zip'}).col('ballot_id')
    ['14_1', '05_2']



##### alter_daf_per_alter_specs_daf(alter_specs_daf: 'Daf') -> 'Daf'

Change this Daf with the replace patterns listed in a Daf.

Each row of `alter_specs_daf` has a `colname` and a `replace_regex`. The
pattern is applied to the column with `apply_replace_regex()`. Give it only
the rows that apply, for example by selecting on `spec_name` first.

Args:
    alter_specs_daf: The specs, with the columns `colname` and `replace_regex`.

Returns:
    This Daf, which has been changed.

Examples:
    >>> d = Daf(lol=[['04_1'], ['05_2']], cols=['ballot_id'])
    >>> specs = Daf.from_lod([{'colname': 'ballot_id', 'replace_regex': r'/^04_/14_/'}])
    >>> d.alter_daf_per_alter_specs_daf(specs).col('ballot_id')
    ['14_1', '05_2']


### Other


##### cols_to_dol(colname1: str, colname2: str) -> T_dola

Make a lookup from the values of one column to the values of another.

For each value in `colname1`, the result lists the different values that appear
with it in `colname2`, in the order first seen. Use it to see how two columns
relate. The values must be hashable. If a name is not a column, or the Daf is
empty, the result is empty.

Args:
    colname1: The column of keys.
    colname2: The column of values.

Returns:
    A dict that maps each value of the first column to a list of values of the second.

Examples:
    >>> d = Daf(lol=[['a', 'b'], ['b', 'd'], ['a', 'f'], ['b', 'd']], cols=['c1', 'c2'])
    >>> d.cols_to_dol('c1', 'c2')
    {'a': ['b', 'f'], 'b': ['d']}



##### insert_dif_row(irow1: int, irow2: int | None = None, irow_insert: int | None = None, cols: T_ls | None = None) -> 'Daf'

Insert a row that holds the difference of two rows.

The difference is the first row minus the second row, for the numeric columns
you name. Other columns of the new row are empty. A cell that is empty counts
as 0. By default the second row is the one after the first, and the new row goes
between them. Columns that hold text must not be in `cols`.

Args:
    irow1: The position of the first row.
    irow2: The position of the second row. If None, the row after the first.
    irow_insert: Where to insert the new row. If None, at `irow2`.
    cols: The columns to subtract. If None, all columns.

Returns:
    This Daf, which has been changed.

Examples:
    >>> d = Daf(lol=[[1, 10], [3, 14], [6, 20]], cols=['a', 'n'])
    >>> d.insert_dif_row(0)
    <BLANKLINE>
    | a  | n  |
    | -: | -: |
    |  1 | 10 |
    | -2 | -4 |
    |  3 | 14 |
    |  6 | 20 |
    <BLANKLINE>
    %% daf rows=4; cols=2; keyfield=''; name=''
    <BLANKLINE>



##### insert_dif_rows(irows_rli: T_rli | None = None, cols: T_ls | None = None, offset: int = 0) -> 'Daf'

Insert a difference row after each of several rows.

For each position, a row is inserted that holds that row minus the next row,
as in `insert_dif_row()`. Do not list the last row. By default all rows are
used. With `offset=1` the new row goes after the next row, not between them.

Args:
    irows_rli: The positions of the first rows. If None, every row but the last.
    cols: The columns to subtract. If None, all columns.
    offset: 0 inserts between the two rows. 1 inserts after the second.

Returns:
    This Daf, which has been changed.

Examples:
    >>> d = Daf(lol=[[1, 10], [3, 14], [6, 20]], cols=['a', 'n'])
    >>> d.insert_dif_rows()
    <BLANKLINE>
    | a  | n  |
    | -: | -: |
    |  1 | 10 |
    | -2 | -4 |
    |  3 | 14 |
    | -3 | -6 |
    |  6 | 20 |
    <BLANKLINE>
    %% daf rows=5; cols=2; keyfield=''; name=''
    <BLANKLINE>

---

# Docs page: Daf API / Joins and annotation (api/daf/join.md)

## Joins and annotation

Combining two tables.

### Join


##### join(other_daf: 'Daf', how: str = 'inner', shared_fields: T_ls | None = None, tag_other: bool = False, custom_translator_daf: 'Daf' | None = None, diagnose: bool = False, name: str = '', fill: Any = NULL) -> 'Daf'

Join two Daf instances on their keyfields, as in SQL.

Both need a keyfield, and it must be a single column. A composite keyfield, a
tuple of columns, raises `KeyError`. A row of this Daf is joined with the row of
`other_daf` that has the same key.

The join is one to one. It is not a many to many join, as in SQL. Each row of this
Daf finds at most one row of the other, through its key index. If a key repeats in
the other Daf, every match gets its last row. If a key repeats in this Daf, each of
those rows is in the result. Check the keys first if they may repeat. See `keys()`.

The types of join are:

    inner    only the keys that are in both.
    left     all keys of this Daf.
    right    all keys of the other Daf.
    outer    all keys of both.

Columns that only one Daf has are in the result. A column that both have
is given the name of its source as a suffix, such as `name_daf1`, unless it is
in `shared_fields`. The names are `daf1` and `daf2` if the instances have no
names. For other names use `custom_translator_daf`, which you can start from
`derive_join_translator()`.

When a key has no match, its cells from the other side are NULL, the empty string. Pass
`fill=None` to get `None` there, as earlier versions did.
The keyfield of the result is the keyfield of this Daf. The result is a new Daf.

A row that only the other Daf has, in a right or outer join, comes after the rows
of this Daf. Two of its cells can be lost:

- A column in `shared_fields` takes its value from this Daf only. Such a row has no
  row here, so the cell is NULL, though the other Daf has a value.
- The key is NULL when the two keyfields have different names.

These are known faults. Name the keyfields the same, and leave
out `shared_fields`, when such rows matter.

Args:
    other_daf: The Daf to join with.
    how: `inner`, `left`, `right` or `outer`.
    shared_fields: Columns that both have and that appear only once.
    tag_other: If True, every column of the other Daf, except shared ones, gets a suffix.
    custom_translator_daf: A translator that sets all the names.
    diagnose: If True, print progress messages.
    name: The name of the result.
    fill: The value for cells that have no match.

Returns:
    The joined Daf.

Raises:
    ValueError: `how` is not one of the four.
    KeysDisabledError: A Daf has no keyfield.
    KeyError: A keyfield is a tuple.

Examples:
    >>> a = Daf(lol=[[1, 'Alice'], [2, 'Bob']], cols=['id', 'name'], keyfield='id')
    >>> b = Daf(lol=[[1, 50], [3, 70]], cols=['id', 'salary'], keyfield='id')
    >>> a.join(b)
    <BLANKLINE>
    | id | name  | salary |
    | -: | ----: | -----: |
    |  1 | Alice |     50 |
    <BLANKLINE>
    %% daf rows=1; cols=3; keyfield='id'; name=''
    <BLANKLINE>
    >>> a.join(b, how='left')
    <BLANKLINE>
    | id | name  | salary |
    | -: | ----: | -----: |
    |  1 | Alice |     50 |
    |  2 |   Bob |        |
    <BLANKLINE>
    %% daf rows=2; cols=3; keyfield='id'; name=''
    <BLANKLINE>
    >>> a.join(b, how='left', fill=None)
    <BLANKLINE>
    | id | name  | salary |
    | -: | ----: | -----: |
    |  1 | Alice |     50 |
    |  2 |   Bob |   None |
    <BLANKLINE>
    %% daf rows=2; cols=3; keyfield='id'; name=''
    <BLANKLINE>



##### join_records(records: List[T_ma | None], translator_daf: 'Daf', join_names_ls: T_ls | None = None, fill: Any = NULL) -> T_da

Combine one record from each table into one record, using a translator.

This is a static method, and the step that `join()` repeats. A record may be
None, which gives `fill` for the columns of that side. A column that a record
does not have also gets `fill`.

Args:
    records: The two records, in the order of the source names.
    translator_daf: The translator, as from `derive_join_translator()`.
    join_names_ls: The two source names. Needed only if the translator names more than two sources.
    fill: The value for a column that has no record, or that the record lacks. NULL by default.

Returns:
    The combined record, as a dict.

Raises:
    ValueError: The translator names more than two sources and `join_names_ls` is not given.

Examples:
    >>> tr = Daf.derive_join_translator_daf('id', 'id', ['id', 'v'], ['id', 'w'], 'L', 'R')
    >>> Daf.join_records([{'id': 1, 'v': 'a'}, {'id': 1, 'w': 'b'}], tr)
    {'id': 1, 'v': 'a', 'w': 'b'}
    >>> Daf.join_records([{'id': 1, 'v': 'a'}, None], tr)
    {'id': 1, 'v': 'a', 'w': ''}
    >>> Daf.join_records([{'id': 1, 'v': 'a'}, None], tr, fill=0)
    {'id': 1, 'v': 'a', 'w': 0}



##### derive_join_translator(other_daf: 'Daf', shared_fields: T_ls | None = None, omit_other_cols: T_ls | None = None, tag_other: bool = False) -> 'Daf'

Work out how the columns of two Daf instances are named in a join.

The translator is a Daf with a row for each column of the result. Its
columns are `resolved_colname`, `source_name`, `source_colname` and
`is_keyfield`. A column name that both Daf instances have is given a suffix
with the name of its source, such as `name_daf1`, unless it is a shared
field. The names of the instances are `daf1` and `daf2` if they have none.
The keyfields are always shared.

`join()` calls this. Call it yourself to see the names, or to edit the
translator and give it back as `custom_translator_daf`.

Args:
    other_daf: The Daf to join with.
    shared_fields: Columns that both have and that appear only once.
    omit_other_cols: Columns of the other Daf to leave out.
    tag_other: If True, every column of the other Daf, except shared ones, gets a suffix.

Returns:
    The translator Daf. Its keyfield is `resolved_colname`.

Examples:
    >>> a = Daf(lol=[[1, 'x']], cols=['id', 'name'], keyfield='id')
    >>> b = Daf(lol=[[1, 'y']], cols=['id', 'name'], keyfield='id')
    >>> a.derive_join_translator(b).col('resolved_colname')
    ['id', 'name_daf1', 'name_daf2']



##### derive_join_translator_daf(self_keyfield: str, other_keyfield: str, self_cols: T_cs, other_cols: T_cs, self_name: str = '', other_name: str = '', shared_fields: T_cs | None = None, omit_other_cols: T_cs | None = None, tag_other: bool = False) -> 'Daf'

Work out a join translator from column names, with no Daf instances.

This is the form that does not need two Daf instances, so SQL joins can use
it. See `derive_join_translator()` for what the translator holds.

Args:
    self_keyfield: The keyfield of the first table.
    other_keyfield: The keyfield of the other table.
    self_cols: The column names of the first table.
    other_cols: The column names of the other table.
    self_name: The name of the first table.
    other_name: The name of the other table.
    shared_fields: Columns that both have and that appear only once. The list is not changed.
    omit_other_cols: Columns of the other table to leave out.
    tag_other: If True, every column of the other table, except shared ones, gets a suffix.

Returns:
    The translator Daf. Its keyfield is `resolved_colname`.

Examples:
    >>> tr = Daf.derive_join_translator_daf('id', 'id', ['id', 'v'], ['id', 'w'], 'L', 'R')
    >>> tr.columns()
    ['resolved_colname', 'source_name', 'source_colname', 'is_keyfield']
    >>> tr
    <BLANKLINE>
    | resolved_colname | source_name | source_colname | is_keyfield |
    | ---------------: | ----------: | -------------: | ----------: |
    |               id |           L |             id |        True |
    |                v |           L |              v |       False |
    |                w |           R |              w |       False |
    <BLANKLINE>
    %% daf rows=3; cols=4; keyfield='resolved_colname'; name=''
    <BLANKLINE>
    >>> tagged = Daf.derive_join_translator_daf('id', 'id', ['id', 'v'], ['id', 'w'], 'L', 'R', tag_other=True)
    >>> tagged.col('resolved_colname')
    ['id', 'v', 'w_R']
    >>> omitted = Daf.derive_join_translator_daf('id', 'id', ['id', 'v'], ['id', 'w', 'x'], 'L', 'R', omit_other_cols=['x'])
    >>> omitted.col('resolved_colname')
    ['id', 'v', 'w']


### Annotate


##### annotate_daf(other_daf: 'Daf', my_to_other_dict: T_ds) -> 'Daf'

Copy fields from another Daf into this one, row by row, matching on the key.

Both Daf instances need a keyfield. For each row here, the row with the same
key in `other_daf` is found, and each field of this row named in
`my_to_other_dict` gets the value of the other field. A key that is missing
in `other_daf` raises `KeyError`. A field that is not a column here is added as
a new column at the right.

Args:
    other_daf: The Daf to copy from.
    my_to_other_dict: Maps the field to set here to the field to read there.

Returns:
    This Daf, which has been changed.

Raises:
    KeyError: A keyfield is not set, or a key is not found in `other_daf`.

Examples:
    >>> a = Daf(lol=[[1, 'x'], [2, 'y']], cols=['id', 'v'], keyfield='id')
    >>> o = Daf(lol=[[1, 'P'], [2, 'Q']], cols=['id', 'w'], keyfield='id')
    >>> a.annotate_daf(o, {'v': 'w'})
    <BLANKLINE>
    | id | v  |
    | -: | -: |
    |  1 |  P |
    |  2 |  Q |
    <BLANKLINE>
    %% daf rows=2; cols=2; keyfield='id'; name=''
    <BLANKLINE>

---

# Docs page: Daf API / Statistics and reporting (api/daf/report.md)

## Statistics and reporting

Value counts, statistics, and Markdown output.

### Value counts


##### valuecounts_for_colname(colname: str, sort: bool = False, reverse: bool = True, omit_nulls: bool = False) -> T_di

Count how often each value occurs in one column.

With `sort=True` the dict is ordered from the most common value to the least.
Use `reverse=False` for the other way. A column that does not exist gives an
empty dict. An empty cell is counted as the empty string, unless
`omit_nulls` is True.

Args:
    colname: The column to count.
    sort: If True, order by count.
    reverse: With `sort`, True puts the most common first.
    omit_nulls: If True, leave out the count of empty cells.

Returns:
    A dict that maps each value to its count.

Examples:
    >>> d = Daf(lol=[['a'], ['b'], ['a'], ['']], cols=['g'])
    >>> d.valuecounts_for_colname('g', sort=True, omit_nulls=True)
    {'a': 2, 'b': 1}



##### valuecounts_for_colnames_ls(colnames_ls: T_cs | None = None, sort: bool = False, reverse: bool = True, omit_nulls: bool = False) -> T_dodi

Count how often each value occurs, in each of several columns.

Args:
    colnames_ls: The columns to count. All if None.
    sort: If True, order each count by size.
    reverse: With `sort`, True puts the most common first.
    omit_nulls: If True, leave out the count of empty cells.

Returns:
    A dict that maps each column to a dict of value and count.

Examples:
    >>> d = Daf(lol=[['a', 'x'], ['b', 'x']], cols=['g', 'h'])
    >>> d.valuecounts_for_colnames_ls()
    {'g': {'a': 1, 'b': 1}, 'h': {'x': 2}}



##### valuecounts_for_colname_selectedby_colname(colname: str, selectedby_colname: str, selectedby_colvalue: str, sort: bool = False, reverse: bool = True) -> T_di

Count the values of a column, in the rows where another column has a value.

Args:
    colname: The column to count.
    selectedby_colname: The column to test.
    selectedby_colvalue: Only rows where that column equals this are counted.
    sort: If True, order the counts by size.
    reverse: With `sort`, True puts the most common first.

Returns:
    A dict that maps each value to its count. It is empty if a column does not exist.

Examples:
    >>> d = Daf(lol=[['a', 'x'], ['b', 'x'], ['a', 'y']], cols=['g', 'h'])
    >>> d.valuecounts_for_colname_selectedby_colname('g', 'h', 'x')
    {'a': 1, 'b': 1}



##### valuecounts_for_colnames_ls_selectedby_colname(colnames_ls: T_cs | None = None, selectedby_colname: str = '', selectedby_colvalue: str = '', sort: bool = False, reverse: bool = True) -> T_dodi

Count the values of several columns, in the rows where another column has a value.

Args:
    colnames_ls: The columns to count. All if None.
    selectedby_colname: The column to test.
    selectedby_colvalue: Only rows where that column equals this are counted.
    sort: If True, order the counts by size.
    reverse: With `sort`, True puts the most common first.

Returns:
    A dict that maps each column to a dict of value and count.

Examples:
    >>> d = Daf(lol=[['a', 'x', 1], ['a', 'y', 2], ['b', 'x', 3]], cols=['g', 'k', 'n'])
    >>> d.valuecounts_for_colnames_ls_selectedby_colname(['k'], 'g', 'a')
    {'k': {'x': 1, 'y': 1}}
    >>> d.valuecounts_for_colnames_ls_selectedby_colname(['g', 'k'], 'g', 'a')
    {'g': {'a': 2}, 'k': {'x': 1, 'y': 1}}



##### valuecounts_for_colname1_groupedby_colname2(colname1: str, groupedby_colname2: str, sort: bool = False, reverse: bool = True) -> T_dodi

Count the values of one column, for each value of another.

The data is read once. Use it to see whether two columns relate one to one: each
group should then hold a single value.

Args:
    colname1: The column whose values are counted.
    groupedby_colname2: The column whose values form the groups.
    sort: If True, order each count by size.
    reverse: With `sort`, True puts the most common first.

Returns:
    A dict that maps each value of the second column to a dict of value and count.

Examples:
    >>> d = Daf(lol=[['a', 'x'], ['b', 'x'], ['a', 'y']], cols=['g', 'h'])
    >>> d.valuecounts_for_colname1_groupedby_colname2('g', 'h')
    {'x': {'a': 1, 'b': 1}, 'y': {'a': 1}}



##### value_counts_daf(colname: str, sort: bool = False, reverse: bool = True, include_total: bool = False, omit_nulls: bool = False) -> 'Daf'

Make a Daf that lists each value of a column and its count.

The columns are the name of the column, and `counts`. There is no keyfield.
With `include_total` a last row holds the total.

Args:
    colname: The column to count.
    sort: If True, order by count.
    reverse: With `sort`, True puts the most common first.
    include_total: If True, add a row with the total.
    omit_nulls: If True, leave out the count of empty cells.

Returns:
    The new Daf.

Examples:
    >>> d = Daf(lol=[['a'], ['b'], ['a']], cols=['g'])
    >>> d.value_counts_daf('g', sort=True)
    <BLANKLINE>
    | g  | counts |
    | -: | -----: |
    |  a |      2 |
    |  b |      1 |
    <BLANKLINE>
    %% daf rows=2; cols=2; keyfield=''; name=''
    <BLANKLINE>



##### daf_valuecount(by: str = 'row', cols: T_cs | None = None) -> T_ma

Count how often each value occurs, in each column, using `reduce()` and `count_values_kl()`.

Args:
    by: How the function is applied. See `reduce()`.
    cols: The columns to count. All if None.

Returns:
    A dict that maps each column to a dict of value and count. A column that is
    not in `cols` is empty.

Examples:
    >>> d = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
    >>> d.daf_valuecount(cols=['g'])['g']
    {'a': 2, 'b': 1}


### Statistics


##### gen_stats_daf(col_def_lot: T_lota) -> T_doda

Work out statistics for columns, given a profile for each.

`col_def_lot` has a tuple for each column of interest. The tuple is the column
name, a type, a format, and a profile. The profile is one of `index`,
`attrib`, `file_paths`, `scalar` or `localidx`, and chooses what is measured.
An index looks for repeats, an attribute counts the values, and a scalar gives
the minimum, maximum, mean and standard deviation. The type and format are not
used.

Args:
    col_def_lot: A list of tuples of column name, type, format and profile.

Returns:
    A dict that maps each column name to its statistics, as a dict.

Raises:
    NotImplementedError: A profile is not one of the five.

Examples:
    >>> d = Daf(lol=[[1], [3]], cols=['n'])
    >>> d.gen_stats_daf([('n', int, '', 'scalar')])['n']['mean']
    2


### Markdown and text


##### to_md(max_rows: int = 0, max_cols: int = 0, just: str = '', shorten_text: bool = True, max_text_len: int = 80, smart_fmt: bool = False, include_summary: bool = False, disp_cols: T_cs | None = None, header: T_cs | None = None) -> str

Make a Markdown table of the Daf.

Without limits the whole table is written. With `max_rows` or `max_cols`, the
first and last are kept, and the middle is replaced by `...`. Text longer
than `max_text_len` is shortened by cutting out its middle. `just` has one
character for each column: `<` left, `^` center, `>` right. The default is
right. With no column names, `A`, `B` and so on are used. With `include_summary`,
the Markdown can be read back with `from_md()`, as text.

Use `max_rows` and `max_cols` together, or neither. With only `max_cols`, a row of
`...` is added under the header by mistake.

Args:
    max_rows: The most rows to show. 0 for all.
    max_cols: The most columns to show. 0 for all.
    just: The justification of each column.
    shorten_text: If True, shorten text that is longer than `max_text_len`.
    max_text_len: The longest text to show in full.
    smart_fmt: If True, show numbers with fewer decimal places.
    include_summary: If True, add a line with the size, keyfield and name.
    disp_cols: Column names to show instead of the real ones.
    header: A header to use instead.

Returns:
    The Markdown text.

Examples:
    >>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'])
    >>> print(d.to_md())
    | id | v  |
    | -: | -: |
    |  1 |  a |
    |  2 |  b |
    <BLANKLINE>



##### to_md_cols(max_rows: int = 0, max_cols: int = 0, just: str = '', shorten_text: bool = True, max_text_len: int = 80, smart_fmt: bool = False, include_summary: bool = False, disp_cols: T_ls | None = None) -> str

Make a Markdown table in which each row of the Daf is a column.

There is no header row and no separator row. The first column holds the column names of the Daf.
Use it for a Daf with few rows and many columns. Some Markdown renderers, such as Python-Markdown,
show this text as plain text, because they need a header and a separator row to make a table.

Args:
    max_rows: The most rows to show. 0 for all.
    max_cols: The most columns to show. 0 for all.
    just: The justification of each column.
    shorten_text: If True, shorten text that is longer than `max_text_len`.
    max_text_len: The longest text to show in full.
    smart_fmt: If True, show numbers with fewer decimal places.
    include_summary: Not used.
    disp_cols: Column names to show instead of the real ones.

Returns:
    The Markdown text.

Examples:
    >>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['x', 'y'])
    >>> print(d.to_md_cols(), end='')
    | x  | 1  | 2  |
    | y  | a  | b  |



##### md_daf_table_snippet() -> str

Make a short Markdown table of the Daf, with a summary line.

This is what `str()` shows. It keeps at most `md_max_rows` rows and
`md_max_cols` columns, 10 by default. Longer text is shortened to 80
characters.

If the Daf has no rows, it is empty, and it shows no header. The summary line says
`rows=0; cols=0`, because the number of columns is counted from the rows.
`columns()` still gives the names.

Returns:
    The Markdown text.

Examples:
    >>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['x', 'y'])
    >>> print(d.md_daf_table_snippet(), end='')
    | x  | y  |
    | -: | -: |
    |  1 |  a |
    |  2 |  b |
    <BLANKLINE>
    %% daf rows=2; cols=2; keyfield=''; name=''

    An empty Daf has no rows and no columns, so only the summary line shows. This is so even if it has column names:

    >>> Daf(cols=['x', 'y'])
    <BLANKLINE>
    <BLANKLINE>
    %% daf rows=0; cols=0; keyfield=''; name=''
    <BLANKLINE>



##### daf_to_lol_summary(max_rows: int = 10, max_cols: int = 10, disp_cols: T_cs | None = None) -> T_lola

Make a list of lists for display, with the column names first.

If there are more rows or columns than the limits, the first and last are
kept and the middle is replaced by `...`. The limit is the number of data rows
kept, and an odd limit keeps the extra row at the start. A limit of 0 means no
limit, so with only `max_cols` all the rows are kept. The rows are not copied.

Args:
    max_rows: The most rows to keep. 0 for no limit.
    max_cols: The most columns to keep. 0 for no limit.
    disp_cols: Column names to use instead of the real ones.

Returns:
    The rows, with a header row first if there are column names.

Examples:
    >>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['x', 'y'])
    >>> d.daf_to_lol_summary()
    [['x', 'y'], [1, 'a'], [2, 'b']]
    >>> big = Daf(lol=[[i, i] for i in range(20)], cols=['x', 'y'])
    >>> big.daf_to_lol_summary(max_rows=4)
    [['x', 'y'], [0, 0], [1, 1], ['...', '...'], [18, 18], [19, 19]]



##### dict_to_md(da: T_da, cols: T_ls | None = None, just: str = '<<') -> str

Show a dict as a two column Markdown table, for looking at it.

This is a static method. Use it as `print(Daf.dict_to_md(my_da))`. The keys are
in the first column and the values in the second.

Args:
    da: The dict.
    cols: The two column names. Default `key` and `value`.
    just: The justification of the two columns.

Returns:
    The Markdown text.

Examples:
    >>> print(Daf.dict_to_md({'a': 1, 'b': 'two'}))
    | key | value |
    | :-- | :---- |
    | a   | 1     |
    | b   | two   |
    <BLANKLINE>



##### __str__() -> str

Show the Daf as a Markdown table.

The table shows at most `md_max_rows` rows and `md_max_cols` columns, 10 by
default. Larger tables show the first five and the last five, with `...`
between them. A summary line with the size and keyfield follows the table.
Use `md_daf_table_snippet()` for control over the output.

Returns:
    The Markdown text.



##### __repr__() -> str

Show the Daf as a Markdown table, for the Python prompt.

This is the same text as `str()`, with a newline in front, so the table
starts at the left margin when it is echoed at the prompt.

Returns:
    The Markdown text.



##### __format__(format_spec: str) -> str

Format a Daf that holds one cell.

With no format spec this is the same as `str()`. With a spec, the Daf must
have exactly one cell. A number is formatted with the spec. Anything else
is converted with `str()`.

Args:
    format_spec: A format spec such as `.2f`.

Returns:
    The formatted text.

Raises:
    ValueError: A spec is given and the Daf does not have exactly one cell.

Examples:
    >>> format(Daf(lol=[[3.14159]], cols=['a']), '.2f')
    '3.14'

---

# Docs page: SqlDaf API (api/sqldaf/index.md)

## SqlDaf overview


##### SqlDaf

A Daf whose rows live in a SQLite table.

Use it for a table too large to hold in memory. The rows stay on disk in a SQLite database
file, and SQL does the work. When the rows you need are few, pull them into an ordinary Daf
with `to_daf()`, and work on them there.

A SqlDaf is a Daf, so `isinstance(sdaf, Daf)` is True. But it has only the methods
written for SQL. Every other Daf method raises NotImplementedError, which names it. So
nothing quietly loads a large table into memory. For anything not covered yet, `sql()`
runs a SQL statement of your own.

The database is the file named by the class setting `SqlDaf.db_path`. Set it once, for
example once for each job, and every SqlDaf opens that file. A `db_path` given to the
constructor overrides it for that table. With neither, the constructor raises ValueError.
The class setting `engine` is 'sqlite', the only engine so far.

Make a table with the constructor, as a Daf is made: give its name, its columns and its
rows. Open a table that is already in the file by its name alone. Its keyfield and dtypes
come back as they were, from the metadata that SqlDaf keeps in the table daf_tables of the
same file. A table made by other code has no such record, so give them to the constructor.
`SqlDaf.from_csv()` and `SqlDaf.from_daf()` load a file or a Daf into a new table.

A SqlDaf is a parent or a child.

- A parent holds its rows, in a table of its own: one made by the constructor, a loader or
  `copy()`, or one already in the file.
- A child is a selection. It holds references to rows of its parent, in order, as a
  selection of a Daf holds the same row lists. `select_where()`, `select_by_dict()`,
  `select_where_sql()` and indexing, as in `sdaf[2:5]` or `sdaf['b2']`, each make one. A
  child of a child is a child of the same parent. `is_child` tells them apart, and
  `parent_sdaf` is the parent of a child.

Making a child does not read its rows. Some methods run a query at once for their
arguments: `select_krows()` checks the keys, `select_by_dict()` with `expectmax` finds
its rows to count them, and `groupby()` finds the groups. The rows of a child are fixed the first time
it is read whole: by `to_daf()`, `len()`, `bool()`, `to_dict()`, `to_list()` or
`to_value()`. After that, it always means the same rows, as a selection of a Daf does.
`select_record()` and `iloc()` read one row and fix nothing, unless `iloc()` is given a
negative position, which needs `len()`. A change to the values in those
rows shows in it, since it refers to them. `copy(inplace=True)` gives it rows of its own.

Every SqlDaf on one file shares one connection, which it holds in its attributes. So any two
of them can be joined, and a child of one can be used with another. They also share its
transactions: see `transaction()`. `SqlDaf.close()` closes the file, for every SqlDaf on it.
Otherwise it stays open until the program ends. Each statement is committed when it
finishes. So committed changes to permanent tables remain, whether the file is closed or the
program ends. `delete()` deletes a table from the file.

A table made by `copy()`, or by reading a join, is a TEMP table. It belongs to the
connection, and is removed when the connection closes. `save()` makes it a permanent table in the
file, under a name, to be opened again later.

Do not delete rows from a parent that has children. `len()` of a child counts its
references, so it would still count a deleted row, though `to_daf()` leaves it out. Worse,
SQLite gives the daf_rid of a deleted last row to the next row added. A child made before
the delete then shows that new row, which it never selected. Leave rows out with a selection
instead, as in a Daf.

A parent that has had rows deleted cannot be read by position. Positions in a parent are
looked up by daf_rid, and a delete leaves a gap there. So `iloc()`, `iloc_exists()` and
`sdaf[n]` or `sdaf[n:m]` on it raise ValueError. Read positions from `copy()`, which numbers
the rows again, or select the rows you want. A selection numbers its own rows, so positions
in it always work.

To read one row, use `select_record()` by key or `iloc()` by position. Each is one query,
returns a dict, and makes no selection.

The methods, by what they do:

- Size: `len()`, `bool()`, `num_rows()`, `num_cols()`, `shape()` and `columns()`.
- Looping, in batches: `for row in sdaf`, `iter_dict()`, `iter_klist()` and `iter_list()`.
- One row: `select_record()` by key, and `iloc()` by position.
- Columns: `col()`, `col_to_la()`, `icol()`, `icol_to_la()` and `cols_to_dol()`.
- Selecting: `select_where()`, `select_by_dict()`, `select_where_sql()`, `split_where()`,
  `select_cols()` and indexing.
- By key: `key in sdaf`, `keys()`, `select_records_daf()`, `select_krows()`,
  `get_existing_keys()` and `delete_records()`.
- The first match: `select_first_row_by_dict()`.
- Groups and counts: `groupby()`, `groupby_cols()`, `group_where()`, `groupsum_daf()`,
  `sum()`, `daf_valuecount()`, `value_counts_daf()` and the `valuecounts_for_*` methods.
- Functions over rows: `apply()` and `reduce()`.
- Adding rows: `append()`, `record_append()`, `extend()`, `assign_record()` and `concat()`.
- Changing values: `sdaf[key, col] = v`, `sdaf[key] = row`, `sdaf[:, col] = values`,
  `update_record_krow()`, `update_by_keylist()`, `assign_col()`, `apply_to_col()` and `apply_in_place()`.
- Columns and layout: `insert_col()`, `drop_cols()`, `rename_cols()`, `set_cols()`,
  `set_dtypes()`, `attach_schema()` and `clone_empty()`.
- Repeated keys: `remove_dups()`. Sorting: `sort_by_colname()` and `sort_by_colnames()`.
- Joining: `join()`. Choosing columns: `calc_cols()`.
- Into memory: `to_daf()`, `to_dict()`, `to_list()`, `to_value()`, `to_lod()`, `to_dod()`
  and `to_json()`.
- Display and files: `to_md()`, `to_md_cols()`, `to_csv_buff()` and `to_csv_file()`.
- Tables: `from_csv()`, `from_daf()`, `copy()`, `save()`, `delete()` and `table_exists()`.
- The database: `close()`, `transaction()` and `sql()`, which runs any SQL statement.
- Keys and types: `set_keyfield()`, `ensure_key_index()` and `apply_dtypes()`.

Work with large tables in whole statements. One query costs about 12 to 24 us, against
under 1 us for a lookup in memory, because each one passes through sqlite3. Many rows in one
statement are far cheaper per row.

There are two kinds of NULL. Python NULL is daffodil's missing value: the empty string '',
which daffodil names NULL, as in `from daffodil.daf import NULL`. SQL NULL is the database's.
A Python None written to a SqlDaf is stored as SQL NULL too, so it reads back as Python NULL. SqlDaf writes Python NULL as SQL NULL, and reads SQL NULL back as
Python NULL. A table written by other code may hold the text '' instead, which also reads as
Python NULL. So the methods that group, count or select by value treat SQL NULL and '' as the
same value, as a Daf does. In Python, test a value with `val is NULL`. This works because
CPython keeps one shared empty string, and it is faster than `val == ''`. In your own SQL,
test for both, as `(col IS NULL OR col = '')`.

Avoid a Python NULL key, the empty string in the keyfield. If you allow one, the results of
SqlDaf are undefined. The key methods compare keys with = in SQL, and SQL NULL equals
nothing, so such a row is not found by key, `select_records_daf()`, `delete_records()` or
`join()`. The unique index also lets several SQL NULL keys in. A Daf treats a Python NULL key
as a key like any other, so the two differ. Give every row a real key. Making the key index
raises ValueError on a Python NULL key, and so does `concat()` of one.

Until `apply_dtypes()` converts them, values loaded from a CSV file are text, in SQL too.
Text is never less than a number in SQLite, so `WHERE n < 5` finds nothing on a text
column, and MAX() gives the last in text order.

The `name` is the name of the table in the database. A child gets a generated name, such as
daf_tmp_marks_v3, which names its reference table once its rows are found.

Examples:
    >>> import tempfile
    >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
    >>> marks_sdaf = SqlDaf('marks', cols=['ballot_id', 'v'], lol=[['b1', 1], ['b2', '']], keyfield='ballot_id')
    >>> marks_sdaf.sql('SELECT ballot_id FROM marks WHERE v IS NULL').to_list()
    ['b2']
    >>> SqlDaf('marks').keyfield
    'ballot_id'


### Constructor


##### __init__(name: str, cols: T_ls | None = None, lol: Iterable[Iterable[Any]] | None = None, keyfield: str = '', dtypes: T_dtype_dict | None = None, db_path: str = '') -> None

Make a new table, or open one that is in the database file.

With `cols`, a new table is made in the file, with the rows of `lol`, read one at a
time, so an iterator of any length works. The table, its rows and its metadata are made
in one transaction: if any row fails, nothing is made. A Python NULL cell is stored
as SQL NULL. The table gets a hidden first column, daf_rid, an INTEGER PRIMARY KEY, which
numbers the rows in order. It is not one of the `columns()`. The columns are declared
with no SQL type, so an int stays an int, and text stays text. A cell that holds a list
or a dict cannot be stored yet.

Without `cols`, the table of that name is opened. Its keyfield and dtypes are those
recorded when it was made, unless they are given here. Nothing is read yet.

The keyfield is only recorded. Its unique index is made when a key is first used, as a
Daf builds its key index.

Args:
    name: The name of the table.
    cols: The column names of a new table, in order. None to open a table that is there.
    lol: The rows of a new table, each the values in the order of `cols`.
    keyfield: The column that holds the key of each row. For a table that is opened, the recorded one if empty.
    dtypes: The types of the columns, kept for `apply_dtypes()`. Nothing is converted.
    db_path: The database file. The class setting `SqlDaf.db_path` if empty.

Raises:
    ValueError: There is no database path, a new table's name is taken, a column is named daf_rid, or `lol` has no `cols`.
    KeyError: There is no table of that name to open, or the keyfield is not a column.
    FileNotFoundError: A table is to be opened, and the database file is not there.
    NotImplementedError: The class setting `engine` is not 'sqlite'.


### Loading a table


##### from_csv(path: str, name: str = '', keyfield: str = '', delimiter: str = ',', db_path: str = '') -> SqlDaf

Load a CSV file into a new table, without building it in memory.

This only reads, as `Daf.from_csv()` does without dtypes. Every value is stored as the
text of the file. Call `apply_dtypes()` afterwards to convert the columns you need.

The first line gives the column names. The rest is read one row at a time, so the file
is never all in memory. An empty cell is stored as SQL NULL, and comes back as NULL.
A row shorter than the header is padded with Python NULL, and a longer one raises ValueError.
As with the constructor, if any row fails, no table is made.

Args:
    path: The CSV file.
    name: The name of the new table. The file name without its extension if empty.
    keyfield: The column that holds the key of each row.
    delimiter: The field separator.
    db_path: The database file. The class setting `SqlDaf.db_path` if empty.

Returns:
    A SqlDaf for the new table.

Raises:
    ValueError: A table of that name exists, a column is named daf_rid, or a row is longer than the header.
    KeyError: The keyfield is not one of the columns.



##### from_daf(data_daf: Daf, name: str, keyfield: str = '', db_path: str = '') -> SqlDaf

Write an in-memory Daf into a new table.

Each value is stored as it is, so an int stays an int. A NULL cell is stored as SQL NULL.
The new SqlDaf keeps the Daf's dtypes, for `apply_dtypes()`. The keyfield is the Daf's,
unless one is given. As with the constructor, if any row fails, no table is made.

Args:
    data_daf: The Daf to write. It is not changed.
    name: The name of the new table.
    keyfield: The column that holds the key of each row. The Daf's keyfield if empty.
    db_path: The database file. The class setting `SqlDaf.db_path` if empty.

Returns:
    A SqlDaf for the new table.

Raises:
    ValueError: A table of that name exists, or a column is named daf_rid.
    KeyError: The keyfield is not one of the columns.



##### table_exists(name: str, db_path: str = '') -> bool

Say whether the database file has a table of this name.

Use it to decide whether to open a table or to build it, as from its CSV files.
It returns False if the database file does not exist, and does not create the file.

Args:
    name: The name of the table.
    db_path: The database file. The class setting `SqlDaf.db_path` if empty.

Returns:
    True if the table is there.

Raises:
    ValueError: There is no database path.


### Tables


##### name

The name of the table in the database. `copy(name=)` and `save(name)` give a new one.



##### is_child

True for a child, a selection that refers to rows of its parent. False for a parent, which holds its own rows.



##### copy(level: str | T_copybits | None = None, name: str = '', deep: bool = False, for_sorting: bool = False, inplace: bool = False) -> SqlDaf

Copy the rows into a TEMP table of their own, as `Daf.copy()` copies a Daf.

The copy is a parent, with daf_rid, the rows in order, and this SqlDaf's keyfield and
dtypes. Nothing ties it to the rows it came from: a change to one does not show in the
other. A SQL cell holds a plain value, so every level gives what `deep` gives in a Daf.
The level is only checked, as `Daf.copy()` checks it.

With `inplace=True`, this SqlDaf becomes the copy, and is returned. Use it on a child, to
make it a table of its own, independent of its parent. As in a Daf, a level that would
not copy the rows, such as `sortable`, raises ValueError then.

The copy is a TEMP table, so it is removed when the connection closes. `save()` keeps it in the
file.

Args:
    level: As in `Daf.copy()`. With `inplace`, it must copy the rows: `editable` or `deep`.
    name: The name of the copy. Generated if empty, as in daf_tmp_marks_v4.
    deep: Deprecated, as in `Daf.copy()`.
    for_sorting: Deprecated, as in `Daf.copy()`.
    inplace: If True, this SqlDaf becomes the copy, and is returned.

Returns:
    The copy, a new SqlDaf, or this one if `inplace`.

Raises:
    ValueError: A table of that name exists, `level` is not a level, or `inplace` with a level that does not copy the rows.

Examples:
    >>> import tempfile
    >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
    >>> marks_sdaf = SqlDaf('marks', cols=['ballot_id', 'v'], lol=[['b1', 1], ['b2', 0]])
    >>> sel_sdaf = marks_sdaf.select_where_sql('v = 1')
    >>> sel_sdaf.copy(inplace=True).is_child
    False
    >>> _ = marks_sdaf.sql("UPDATE marks SET v = 9")
    >>> sel_sdaf.to_daf().lol
    [['b1', 1]]



##### save(name: str = '') -> SqlDaf

Keep this table in the database file, as a permanent table, to be opened again later.

The name is the one this SqlDaf has, unless one is given: the name given to the
constructor, or to `copy(name=)`. A table with only a generated name, such as
daf_tmp_marks_v4, needs one. The keyfield and dtypes are recorded with it, so
`SqlDaf(name)` opens it as it was.

- A TEMP table, made by `copy()`, a sort or a join, is moved into the file. The TEMP
  table is removed.
- A permanent table saved under its own name is left as it is, and its keyfield and
  dtypes are recorded again.
- A permanent table saved under a new name is copied to it, and this SqlDaf then refers
  to the copy. The first table stays.

A child has no rows of its own. Make it a table first, with `copy(inplace=True)`.

A name already in use raises ValueError. To replace a table with a sorted or changed copy,
delete the old table first, as in `SqlDaf('items').delete()`, then save the copy under its name.

Args:
    name: The name to keep it under. This SqlDaf's name if empty.

Returns:
    This SqlDaf, as a permanent table.

Raises:
    ValueError: This is a child, the name is a generated one, or a table of that name exists.

Examples:
    >>> import tempfile
    >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
    >>> marks_sdaf = SqlDaf('marks', cols=['ballot_id', 'v'], lol=[['b1', 1], ['b2', 0]], keyfield='ballot_id')
    >>> agreed_sdaf = marks_sdaf.select_where_sql('v = 1').copy(name='agreed').save()
    >>> SqlDaf('agreed').keyfield, len(SqlDaf('agreed'))
    ('ballot_id', 1)



##### delete() -> None

Delete this table from the database file, with its metadata.

The table is dropped. Inside a `transaction()` block that then raises, the drop is
undone with the rest of the block. Otherwise it is final.
A child has no table of its own to delete. Delete its parent, or let the child go.
After this, the SqlDaf, and any child of it, can no longer be read.

Raises:
    ValueError: This is a child.


### The database


##### close(db_path: str = '') -> None

Close a database file: the connection that every SqlDaf on it shares.

It closes a file, not one table, so call it on the class, as `SqlDaf.close()`. Called on
an instance, it still closes the file of `db_path`, or of the class setting, not the
file that instance uses.

The permanent tables in the file stay, with every committed change.
`SqlDaf(name)` opens them again, on a new connection.
TEMP tables, such as those made by `copy()` or a sort, are removed.
Every SqlDaf that was on the file can no longer be used. sqlite3 raises ProgrammingError.
A file that is not open is left as it is.

Without `close()`, the file stays open until the program ends.
Each statement is committed when it finishes, or at the end of a `transaction()` block.
So a committed change is kept either way.

Args:
    db_path: The database file. The class setting `SqlDaf.db_path` if empty.

Raises:
    ValueError: There is no database path.



##### transaction() -> Iterator[None]

Run the statements inside the `with` block as one transaction, on this database file.

A transaction belongs to the file, not to this table. Every SqlDaf on the file shares one
connection, so the work of any of them inside the block is committed or undone with the
rest. Wrap the writes of a stage that must succeed or fail together in one block.
Methods that change several things run their own `transaction()`. Inside a block, it
becomes part of the block. Do not run BEGIN, COMMIT or ROLLBACK through `sql()`: they
would end a block that another SqlDaf opened.

Everything in the block is committed together when it ends, or undone together if it
raises. It also makes a run of small statements faster, because the file is locked once
for the block instead of once per statement. On 1,000,000 rows, a lookup by key took
12.4 us inside a transaction, against 24.1 us outside.

While the block is open, no other connection can write to the database file. So keep it
to the work that needs it.

A `transaction()` inside another one becomes a savepoint: if the inner block raises, only
its own work is undone.

Examples:
    >>> import tempfile
    >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
    >>> marks_sdaf = SqlDaf('marks', cols=['ballot_id', 'v'], lol=[['b1', 1]])
    >>> with marks_sdaf.transaction():
    ...     _ = marks_sdaf.sql("UPDATE marks SET v = 2")
    >>> marks_sdaf.to_daf().lol
    [['b1', 2]]



##### sql(stmt: str, params: Iterable[Any] = ()) -> Daf

Run a SQL statement of your own, and return its result as a Daf.

This is the fallback for anything SqlDaf does not do yet. The statement runs on this
SqlDaf's connection, so it can use every table in the file, and the TEMP tables of the
connection. Refer to this table by `sdaf.name`.

Pass values as `params`, with `?` in the statement, instead of putting them in the text.

A SELECT gives a Daf of its rows, with its column names. SQL NULL becomes Python NULL. A
statement that returns no rows, such as UPDATE or CREATE, gives an empty Daf. Outside
`transaction()`, it is committed when it finishes.

The whole result is pulled into memory, so select only what you need. Do not run BEGIN,
COMMIT or ROLLBACK here. Use `transaction()`, which every SqlDaf on the file shares.

A change to a table's columns made here, such as ALTER TABLE, is not seen by a SqlDaf
already open on that table. Open it again by name to see the new columns. Nor does it
update the keyfield and dtypes that SqlDaf records. For column changes, prefer
`insert_col()`, `drop_cols()` and `rename_cols()`, which keep both up to date.

Args:
    stmt: One SQL statement.
    params: Values for the `?` marks in the statement, in order.

Returns:
    A Daf of the rows the statement returns.


### Keys and types


##### set_keyfield(keyfield: str = '') -> SqlDaf

Set the column that holds the key of each row.

As in a Daf, nothing is built yet. The unique index on the column is made when a key is
first used, by `ensure_key_index()`. An empty keyfield clears it. Avoid Python NULL in the
column: a Python NULL key gives undefined results, as SqlDaf explains.

On a permanent table with a record in the metadata, the record is changed too, so the
table opens with this keyfield next time. A table made by other code has no record until
`save()` makes one.

Args:
    keyfield: The column. '' for none.

Returns:
    This SqlDaf.

Raises:
    KeyError: The keyfield is not a column.



##### key_index_name() -> str

The name of the unique index on the keyfield, as in daf_ix_marks_ballot_id.



##### ensure_key_index() -> SqlDaf

Make the unique index on the keyfield of a parent, if it is not there yet.

A Daf builds its key index the first time a key is needed. A SqlDaf does the same: the
methods that use keys call this first. On 1,000,000 rows it took 0.50 s, about what a
Daf takes. After that, a lookup by key uses the index instead of reading every row.

The index is unique. A Daf keeps the last row of a repeated key without a word. A SqlDaf
raises instead, so a duplicate is never missed. Once the index is there, adding a row
with a key that is already present raises too.

A Python NULL key, SQL NULL or the text '', raises ValueError when the index is made.
SqlDaf's results with one would be undefined, and the unique index does not stop several
SQL NULL keys. The check uses the new index, so it costs about 0.3 ms. A Daf accepts a
Python NULL key.

Returns:
    This SqlDaf.

Raises:
    ValueError: There is no keyfield, this is a selection, or a key is a Python NULL. Nothing is made.
    sqlite3.IntegrityError: A key is repeated. Nothing is made.



##### apply_dtypes(dtypes: T_dtype_dict | None = None, cols: str | T_ls | None = None, silent_error: bool = False) -> SqlDaf

Convert the columns to their dtypes, in place, in the database.

New dtypes given here are kept in the metadata record of the table, if it has one.

A table loaded from a CSV file holds text. Call this to turn the columns you need into
numbers. Each value is converted by the same function `Daf.apply_dtypes()` uses, so the
results are the same as in memory.

- '12' becomes 12, and '2.5' with dtype int becomes 2.
- A value that does not convert, such as 'abc', stays as it is.
- Python NULL stays Python NULL, except in a bool column, where it becomes 0, as in a Daf.
- A str column is not touched.

Conversion does not check that every value got the requested type. A converted column
can still hold some text. `silent_error` does not change that. It is about a column that
has no dtype, not about a value that does not convert.

All the columns are converted in one pass over the table, in one transaction.
On 1,000,000 rows, two int columns took 1.8 s.

Only int, float, bool and str can be converted yet. A list or a dict column stays as the text
of the file, as Python writes it. `reduce(by='sparse_row')` reads that text as it goes.

On a selection, the rows are first copied into a TEMP table of its own, as
`copy(inplace=True)` does, and the copy is converted. The parent is not changed. A Daf
selection shares its rows, so there the parent's rows are converted too. Converting is
costly, so convert a selection rather than a large parent that is not needed otherwise.

Args:
    dtypes: The types to convert to, by column. It may name columns that are not there. This SqlDaf's dtypes if None.
    cols: Convert only these columns, or this one column. All the columns of the dtypes if None.
    silent_error: If False, a column to convert that has no dtype raises ValueError. If True, it is left as it is.

Returns:
    This SqlDaf, whose table has been changed.

Raises:
    KeyError: A name in `cols` is not a column.
    ValueError: A column to convert has no dtype, and `silent_error` is False.
    NotImplementedError: A dtype is not int, float, bool or str.

---

# Docs page: SqlDaf API / Reading (api/sqldaf/read.md)

## Reading

Size, looping, single rows and columns, and moving rows into memory.

### Size


##### __len__() -> int

Count the rows.

For a parent, this runs a COUNT. For a child, its rows are found first, and then its
references are counted, which is one lookup. A row deleted from the parent is still
counted, so do not delete rows from a parent that has children.

Returns:
    The number of rows.



##### len() -> int

Count the rows, as `len(sdaf)` does. It is here so that code written for a Daf, which calls `daf.len()`, runs on a SqlDaf.



##### __bool__() -> bool

Say whether there is at least one row.

This stops at the first row, so it is fast on a large table. A child has its rows found
first, and then its references are checked, as `len()` counts them.

Returns:
    True if there is a row.



##### num_rows() -> int

Count the rows, as `len()` does.



##### num_cols() -> int

Count the columns.

The hidden daf_rid column is not counted.

Returns:
    The number of columns.



##### shape() -> Tuple[int, int]

Give the number of rows and the number of columns, as `Daf.shape()` does.

A table with no rows gives (0, 0), as in a Daf. The rows are counted with `len()`.

Returns:
    A tuple of the number of rows and the number of columns.


### Looping


##### __iter__() -> Iterator[T_da]

Loop over the rows as dicts, as in `for row in sdaf`.

Rows are read in order, in batches, so a table of any size fits in memory. Each dict is a
copy: changing it does not change the table. A Daf can give KeyedList rows instead, by its
itermode. A SqlDaf always gives dicts. Use `iter_klist()` for KeyedList rows.

Do not change the table inside the loop. Whether the loop sees the change is not defined.

Returns:
    The rows, as dicts.



##### iter_dict() -> Iterator[T_da]

Loop over the rows as dicts, as `Daf.iter_dict()` does.

Rows are read in order, in batches. SQL NULL comes back as Python NULL. Each dict is a
copy, so changing it does not change the table.

Returns:
    Each row as a dict of column name to value.



##### iter_klist() -> Iterator[KeyedList]

Loop over the rows as [KeyedList][daffodil.keyedlist.KeyedList] objects.

A row reads by column name, as in a Daf. But each row is a copy. In a Daf, assigning to a
KeyedList row changes the Daf. Here it changes only the copy.

Returns:
    Each row as a KeyedList.



##### iter_list() -> Iterator[T_la]

Loop over the rows as lists, with the values in column order.

Each list is a copy. In a Daf, these are the Daf's own row lists, and assigning to one
changes the Daf. Here it changes only the copy.

Returns:
    Each row as a list.


### One row


##### select_record(key: Any, silent_error: bool = True) -> T_da

Read one row by its key, as a dict, as `Daf.select_record()` does, in one query.

The keyfield's unique index is made first if it is not there yet, so the lookup does not
read every row. It took about 18 us on 1,000,000 rows, or 9 us inside `transaction()`.

On a selection, only the selection's rows are searched. A key that is in the parent but
not in the selection is missing. The call makes no selection. On a selection whose rows
are not yet found, it does not fix them: they are still found when it is first read.

Args:
    key: The key of the row.
    silent_error: If True, a missing key gives an empty dict. If False, it raises KeyError.

Returns:
    The row as a dict, or an empty dict.

Raises:
    KeyError: The key is missing, and `silent_error` is False.
    ValueError: There is no keyfield.



##### iloc(irow: int = 0, include_cols: T_ls | None = None, rtype: str = 'dict') -> T_da | T_la

Read one row by position, as `Daf.iloc()` does, in one query.

On a selection, the position is within the selection's own rows, in its order.
The call makes no selection. On a selection whose rows are not yet found, it does not
fix them: they are still found when the selection is first read.
In a parent with daf_rid, or a child whose rows are found, the position is looked up directly. Otherwise SQLite
steps over the rows before it, which takes longer the further in it is. A negative
position counts from the end, which costs a `len()` first.

Args:
    irow: The position of the row. Negative counts from the end.
    include_cols: Only these columns, in this order. All if None.
    rtype: 'dict' for a dict, or 'list' for a list of the values.

Returns:
    The row. Python NULL for SQL NULL.

Raises:
    IndexError: There is no row at that position.
    KeyError: A name in `include_cols` is not a column.
    NotImplementedError: `rtype` is not 'dict' or 'list'.
    ValueError: This is a parent that has had rows deleted. See SqlDaf.



##### iloc_exists(irow: int) -> bool

Say whether there is a row at this position, from 0.

Args:
    irow: The position, from 0.

Returns:
    True if a row is there.

Raises:
    ValueError: This is a parent that has had rows deleted. See SqlDaf.


### Columns


##### col(colname: str, unique: bool = False, omit_nulls: bool = False, silent_error: bool = False, astype: Callable | str | type | None = None, indirect_col: str | None = None, default: Any = NULL) -> T_la

Get one column as a list, by name, as `Daf.col()` does.

Only that column is read into memory, or only `indirect_col` for a value inside a dict
column. The options act as in a Daf. With `omit_nulls`, both SQL NULL and the text '' are
left out, since both read as Python NULL.

Args:
    colname: The column, or the key inside `indirect_col`.
    unique: If True, leave out repeated values and keep the order.
    omit_nulls: If True, leave out Python NULL.
    silent_error: If True, a name that is not a column gives an empty list.
    astype: A type or function to convert each value with.
    indirect_col: A column that holds a dict, to read `colname` from.
    default: The value for a dict that lacks `colname`.

Returns:
    The values of the column, in row order.

Raises:
    ColumnNotFoundError: The name is not a column, and `silent_error` is False.



##### col_to_la(colname: str, unique: bool = False, omit_nulls: bool = False, silent_error: bool = False, astype: Callable | str | type | None = None, indirect_col: str | None = None, default: Any = NULL) -> T_la

Get one column as a list, by name. This does the same as `col()`. See that method.

Args:
    colname: The column, or the key inside `indirect_col`.
    unique: If True, leave out repeated values and keep the order.
    omit_nulls: If True, leave out Python NULL.
    silent_error: If True, a name that is not a column gives an empty list.
    astype: A type or function to convert each value with.
    indirect_col: A column that holds a dict, to read `colname` from.
    default: The value for a dict that lacks `colname`.

Returns:
    The values of the column, in row order.



##### icol(icol: int) -> T_la

Get one column as a list, by its place among the columns, as `Daf.icol()` does.

The place is looked up in the column names, so it does not depend on the rows. A
negative place counts from the end. A place out of range raises IndexError, unless the
table has no rows, as in a Daf.

Args:
    icol: The place of the column, from 0.

Returns:
    The values of the column, in row order.

Raises:
    IndexError: The place is out of range, and the table has rows.



##### icol_to_la(icol: int, unique: bool = False, omit_nulls: bool = False) -> T_la

Get one column as a list, by its place among the columns, with options.

See `icol()` for the place, and `col()` for the options.

Args:
    icol: The place of the column, from 0.
    unique: If True, leave out repeated values and keep the order.
    omit_nulls: If True, leave out Python NULL.

Returns:
    The values of the column, in row order.

Raises:
    IndexError: The place is out of range, and the table has rows.



##### cols_to_dol(colname1: str, colname2: str) -> Dict[Any, T_la]

Map each value of one column to the values seen with it in another, as `Daf.cols_to_dol()` does.

Only the two columns are read into memory. If a name is not a column, or there are no
rows, the result is empty.

Args:
    colname1: The column of keys.
    colname2: The column of values.

Returns:
    A dict from each value of the first column to the values of the second, in the order first seen.


### Into memory


##### to_daf() -> Daf

Pull every row into an ordinary Daf in memory.

Use it once the rows you need are few enough to hold. The rows come in order. A SQL NULL
comes back as Python NULL. The new Daf has this SqlDaf's name, keyfield and dtypes. Nothing
ties it to the table afterwards: a change to one does not show in the other. A child has
its rows found first.

Returns:
    A Daf of every row.



##### to_dict() -> T_da

Read a table of one row as a dict, as `Daf.to_dict()` does.

A child has its rows found first. For a row by key or by position, `select_record()` and
`iloc()` are cheaper: they make no selection.

Returns:
    The row as a dict. An empty dict if there are no rows.

Raises:
    ValueError: There is more than one row.



##### to_list() -> T_la

Read a table of one row, or of one column, as a list, as `Daf.to_list()` does.

One column gives every value of the column, in order, so select first. A child has its
rows found first.

Returns:
    The values of the row, or of the column.

Raises:
    ValueError: There is more than one row and more than one column.



##### to_value() -> Any

Read a table of one row and one column as its value, as `Daf.to_value()` does.

A child has its rows found first.

Returns:
    The value. Python NULL for SQL NULL.

Raises:
    ValueError: The table is not one row and one column.



##### to_lod() -> List[T_da]

Make a list of dicts, one for each row, as `Daf.to_lod()` does.

Every row is read into memory. Use `iter_dict()` to go through the rows one at a time.

Returns:
    The rows as dicts, in order.



##### to_dod(remove_keyfield: bool = True) -> Dict[Any, T_da]

Make a dict of dicts, named by the keyfield, as `Daf.to_dod()` does.

Every row is read into memory, through `to_daf()`.

Args:
    remove_keyfield: If True, leave the keyfield out of the inner dicts.

Returns:
    Each row as a dict, under its key.

Raises:
    KeysDisabledError: There is no keyfield, and there are rows.



##### to_json(concise: bool = True) -> str

Make JSON text that holds the whole table, as `Daf.to_json()` does.

Every row is read into memory, through `to_daf()`. The text holds the rows, the column
names, the dtypes, the keyfield and the name. `Daf.from_json()` reads it back as a Daf.

Args:
    concise: If True, leave out the parts that are empty.

Returns:
    The JSON text.


### Display and files


##### to_md(max_rows: int = 0, max_cols: int = 0, just: str = '', shorten_text: bool = True, max_text_len: int = 80, smart_fmt: bool = False, include_summary: bool = False, disp_cols: Any = None, header: Any = None) -> str

Make a Markdown table, as `Daf.to_md()` does.

Set `max_rows` for a large table. Then only the first and last rows are read. Without it,
every row is read into memory.

Args:
    max_rows: The most rows to show, the first and the last. 0 for all.
    max_cols: The most columns to show, the first and the last. 0 for all.
    just: The justification of each column: '<', '^' or '>'.
    shorten_text: If True, shorten long text, keeping the ends.
    max_text_len: The length past which text is shortened.
    smart_fmt: If True, limit the digits after the point in numbers.
    include_summary: If True, add a line with the shape, keyfield and name.
    disp_cols: Names to show in place of the column names.
    header: A header to use in place of the column names.

Returns:
    The Markdown text.



##### to_md_cols(max_rows: int = 0, max_cols: int = 0, just: str = '', shorten_text: bool = True, max_text_len: int = 80, smart_fmt: bool = False, include_summary: bool = False, disp_cols: T_ls | None = None) -> str

Make a Markdown table with each row as a column, as `Daf.to_md_cols()` does.

Set `max_rows` for a large table. Then only the first and last rows are read.

Args:
    max_rows: The most rows to show, the first and the last. 0 for all.
    max_cols: The most columns to show, the first and the last. 0 for all.
    just: The justification of each column: '<', '^' or '>'.
    shorten_text: If True, shorten long text, keeping the ends.
    max_text_len: The length past which text is shortened.
    smart_fmt: If True, limit the digits after the point in numbers.
    include_summary: Not used, as in a Daf.
    disp_cols: Names to show in place of the column names.

Returns:
    The Markdown text.



##### daf_to_lol_summary(max_rows: int = 10, max_cols: int = 10, disp_cols: Any = None) -> T_lola

Give the header and the rows to display, as `Daf.daf_to_lol_summary()` does.

With `max_rows`, only the first and last rows are read, around a row of '...'. Without
it, every row is read.

Args:
    max_rows: The most rows to show. 0 for all.
    max_cols: The most columns to show. 0 for all.
    disp_cols: Names to show in place of the column names.

Returns:
    The header, then the rows.



##### to_csv_buff(line_terminator: str | None = None, include_header: bool = True) -> T_buff

Make CSV text from the rows, as `Daf.to_csv_buff()` does.

The whole text is built in memory, so use it for a selection of modest size. To write a
large table, use `to_csv_file()`, which writes as it reads.

Args:
    line_terminator: The line ending. If None, `\r\n` is used.
    include_header: If True, write the column names first.

Returns:
    The CSV text.



##### to_csv_file(file_path: str | Path = '', line_terminator: str | None = None, include_header: bool = True) -> str

Write the rows to a CSV file, as `Daf.to_csv_file()` does.

A local file is written as the rows are read, 10,000 at a time, so a table larger than
memory can be written. The text is the same as a Daf of the same rows would write: each
cell with `str()`, Python NULL and SQL NULL as nothing, and `\r\n` line endings by default.
A path that starts with s3 is not streamed: the text is built in memory and uploaded, as a Daf does.

Args:
    file_path: Where to write the file.
    line_terminator: The line ending. If None, `\r\n` is used.
    include_header: If True, write the column names first.

Returns:
    The path that was written.

Examples:
    >>> import os, tempfile
    >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
    >>> marks_sdaf = SqlDaf.from_daf(Daf(lol=[['b1', 1], ['b2', '']], cols=['ballot_id', 'v']), name='marks')
    >>> path = marks_sdaf.to_csv_file(os.path.dirname(SqlDaf.db_path) + '/marks.csv')
    >>> open(path, newline='').read()
    'ballot_id,v\r\nb1,1\r\nb2,\r\n'

---

# Docs page: SqlDaf API / Selecting and indexing (api/sqldaf/select.md)

## Selecting and indexing

Each selection is a child: it refers to rows of its parent, and runs no query until it is read.

### Indexing with brackets


##### __getitem__(sel: Any) -> SqlDaf

Select rows by position or by key, as `daf[n]`, `daf[n:m]` and `daf[key]` do.

Each gives a selection, not an ordinary Daf, so a large slice stays in the database.
Read it with `to_daf()`, or `to_dict()` for one row. For one row as a dict, `iloc()` and
`select_record()` are cheaper: they make no selection.

- An int is always a position, from 0, even when the keyfield holds ints. Negative counts
  from the end. A position past the end raises IndexError, as in a Daf. For an int key,
  use `select_record()`, or `select_records_daf()` for several.
- A slice of positions, as in `sdaf[10:20]`. A step is not supported yet. A slice past
  the end is empty, as in a Daf.
- A str is a key. The keyfield's unique index is made first. A missing key raises
  KeyError, as in a Daf.
- A list of keys gives their rows in the order of the list, as `select_records_daf()`
  does. A missing key raises KeyError, as in a Daf.

Rows and columns together, as in `sdaf[key, 'col']` or `sdaf[:, ['a', 'b']]`, give new
rows with only those columns, as `select_cols()` makes them. The rows part takes any form
above, or `:` for all rows. The columns part is a name, a list of names, or `:` for all.
Read one value with `to_value()`, as in a Daf.

A value read this way makes a TEMP table, which costs about 1 ms. In a loop over many
keys, `select_record()` is about 30 times faster.

A negative position, or a slice with a negative end, costs a `len()` first. In a parent
with daf_rid, or a child whose rows are found, positions are looked up directly. Otherwise
SQLite steps over the rows before them.

Args:
    sel: An int, a slice, a key or a list of keys. Or a pair of rows and columns.

Returns:
    A selection. New rows of the chosen columns, for a pair.

Raises:
    IndexError: An int position past the end.
    KeyError: A key that is missing, or a name that is not a column.
    NotImplementedError: Any other form, such as a list of positions, or a slice with a step.
    ValueError: An int or a slice, on a parent that has had rows deleted. See SqlDaf.

Examples:
    >>> import tempfile
    >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
    >>> marks_sdaf = SqlDaf('marks', cols=['ballot_id', 'v'], lol=[['b1', 1], ['b2', 0], ['b3', 1]], keyfield='ballot_id')
    >>> marks_sdaf[['b3', 'b1']].to_daf().to_lod()
    [{'ballot_id': 'b3', 'v': 1}, {'ballot_id': 'b1', 'v': 1}]
    >>> marks_sdaf['b2', 'v'].to_value()
    0


### By condition


##### select_where(where: Callable[[Any], Any], indirect_col: str | None = None, cols: T_ls | None = None) -> SqlDaf

Select the rows for which a Python function is true, as `Daf.select_where()` does.

The function gets each row as a [KeyedList][daffodil.keyedlist.KeyedList], with Python NULL for
a missing value, and reads cells by name, as in `row['n']`. It is registered with SQLite
and called for each row when the selection is first read. If it raises, that error is
raised, as itself.

Name the columns it reads in `cols`. It is then given only those, which is faster. A row
of every column costs about 1 us. If the test only compares columns to values, use
`select_by_dict()` or `select_where_sql()`, which call no Python function.

Args:
    where: A function of a row, true for the rows to keep.
    indirect_col: Not supported on a SqlDaf yet.
    cols: The columns the function reads. All if None.

Returns:
    A selection.

Raises:
    KeyError: A name in `cols` is not a column.
    NotImplementedError: `indirect_col` is given.

Examples:
    >>> import tempfile
    >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
    >>> marks_sdaf = SqlDaf.from_daf(Daf(lol=[['b1', 1], ['b2', 0]], cols=['ballot_id', 'v']), name='marks')
    >>> marks_sdaf.select_where(lambda row: row['v'] == 1, cols=['v']).to_list()
    ['b1', 1]



##### select_where_sql(where_sql: str, params: Iterable[Any] = ()) -> SqlDaf

Select the rows for which a SQL condition is true.

The condition is the text after WHERE. Name the columns as they are, in double quotes if
they need it, and pass values as `params`, with `?` in the text. Python NULL in `params`
becomes SQL NULL. Test for a missing value with `IS NULL`. This is the fastest way to select,
since SQLite does all the work.

Nothing runs yet. The rows are found when the selection is first read. See SqlDaf.

Args:
    where_sql: The condition, as SQL text.
    params: Values for the `?` marks in it, in order.

Returns:
    A selection.

Examples:
    >>> import tempfile
    >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
    >>> marks_sdaf = SqlDaf.from_daf(Daf(lol=[['b1', 1], ['b2', 0]], cols=['ballot_id', 'v']), name='marks')
    >>> marks_sdaf.select_where_sql('"v" = ?', [0]).to_list()
    ['b2', 0]



##### select_by_dict(selector_da: T_da | List[T_da], expectmax: int = -1, inverse: bool = False, keyfield: str = '') -> SqlDaf

Select the rows that match every field of a dict, or any one of a list of dicts.

It works as `Daf.select_by_dict()` does, with the SQL written for you. A dict is an AND of
equalities, and a list of dicts is an OR of those. With `inverse=True`, the rows that do
not match. An empty dict matches every row, and an empty list none. A dict that names a
column that is not there matches no row. Python NULL as a value matches a missing value.

Values are compared with SQL's IS, which treats SQL NULL as a value, so the inverse keeps rows
with a missing value. As in Python, 1 equals 1.0, and the text '1' does not equal 1.

Nothing runs yet, unless `expectmax` is given: then its rows are found at once, to
count the matches.

Args:
    selector_da: A dict of column and value, or a list of such dicts.
    expectmax: If not -1, and more rows match, raise LookupError.
    inverse: If True, select the rows that do not match.
    keyfield: The keyfield of the selection. This SqlDaf's if empty.

Returns:
    A selection.

Raises:
    LookupError: More than `expectmax` rows match.
    TypeError: `selector_da` is not a dict, or a list or tuple of dicts.

Examples:
    >>> import tempfile
    >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
    >>> marks_sdaf = SqlDaf.from_daf(Daf(lol=[['b1', 1], ['b2', '']], cols=['ballot_id', 'v']), name='marks')
    >>> marks_sdaf.select_by_dict({'v': ''}).to_list()
    ['b2', '']



##### split_where(where: Callable[[Any], Any], indirect_col: str | None = None, cols: T_ls | None = None) -> Tuple[SqlDaf, SqlDaf]

Split the rows in two, by a Python function that is true or false for each row.

This is deprecated, as `Daf.split_where()` is. Use `group_where()` with a function that
returns True or False.

It works as `Daf.split_where()` does, with the rows staying in the database. It is two
calls of `select_where()`: one keeps the rows where the function is true, and the other
the rest. Nothing runs yet. Each side is found when it is first read, and the function is
called again for each side that is read. So it should give the same answer each time.

Name the columns the function reads in `cols`, as in `select_where()`. If the test is
that columns equal values, two calls of `select_by_dict()`, the second with `inverse=True`,
are much faster, since they call no function. On 1,000,000 rows, reading both sides took
1.79 s here, and 0.21 s that way.

Args:
    where: A function of a row, true for the rows of the first side.
    indirect_col: Not supported on a SqlDaf yet.
    cols: The columns the function reads. All if None.

Returns:
    A tuple of the selection of the rows where the function is true and the selection of the others.

Raises:
    KeyError: A name in `cols` is not a column.
    NotImplementedError: `indirect_col` is given.

Examples:
    >>> import tempfile
    >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
    >>> marks_sdaf = SqlDaf.from_daf(Daf(lol=[['b1', 10], ['b2', 20], ['b3', 30]], cols=['ballot_id', 'n']), name='marks')
    >>> big_sdaf, small_sdaf = marks_sdaf.split_where(lambda row: row['n'] > 10, cols=['n'])
    >>> big_sdaf.to_daf().lol, small_sdaf.to_daf().lol
    ([['b2', 20], ['b3', 30]], [['b1', 10]])



##### select_first_row_by_dict(selector_da: T_da, inverse: bool = False) -> T_da

Get the first row that matches every field of a dict, as `Daf.select_first_row_by_dict()` does.

The matching rule is that of `select_by_dict()`. The search stops at the first match.

Args:
    selector_da: The column names and the values they must have.
    inverse: If True, find the first row that does not match.

Returns:
    The row as a dict, or an empty dict if none matches.


### Columns


##### select_cols(cols: T_ls | None = None, exclude_cols: T_ls | None = None) -> SqlDaf

Make a new table with only some columns, chosen by name, as `Daf.select_cols()` does.

The columns come in the order of `cols`. A name given twice is used once. A name that is
not a column raises KeyError. With no `cols`, all columns are kept, in this table's order.
Then `exclude_cols` leaves out the columns you name. A name in `exclude_cols` that is not
a column is ignored. The rows keep their order.

The result is new rows, as in a Daf, not a selection of this table's rows. It is resolved
the first time it is read, into a TEMP table, as a join is. The keyfield and dtypes carry
over for the columns kept.

Args:
    cols: The names of the columns to keep, in the order you want. If empty, all columns.
    exclude_cols: The names of the columns to leave out.

Returns:
    A SqlDaf of the chosen columns, resolved when first read.

Raises:
    KeyError: A name in `cols` is not a column.

Examples:
    >>> import tempfile
    >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
    >>> marks_sdaf = SqlDaf('marks', cols=['ballot_id', 'v', 'n'], lol=[['b1', 1, 10], ['b2', 0, 20]], keyfield='ballot_id')
    >>> marks_sdaf.select_cols(['n', 'ballot_id']).to_daf().lol
    [[10, 'b1'], [20, 'b2']]
    >>> marks_sdaf.select_cols(exclude_cols=['v']).columns()
    ['ballot_id', 'n']



##### calc_cols(include_cols: Iterable | None = None, exclude_cols: Iterable | None = None, include_types: List[Type] | None = None, exclude_types: List[Type] | None = None) -> T_ls

Choose column names by rules, as `Daf.calc_cols()` does.

It reads only the column names and the dtypes, never the rows.

Args:
    include_cols: Keep only these names.
    exclude_cols: Leave out these names.
    include_types: Keep only columns whose dtype is one of these.
    exclude_types: Leave out columns whose dtype is one of these.

Returns:
    The chosen column names, in column order.


### By key


##### __contains__(key: Any) -> bool

Say whether a row has this key, as in `key in sdaf`.

It is one lookup on the keyfield, through its index. A table with no rows contains
nothing, as in a Daf.

Args:
    key: The key to look for.

Returns:
    True if a row has this key.

Raises:
    ValueError: There is no keyfield, and there are rows.



##### keys() -> T_la

Get the keys, as a list of the keyfield's values in row order, as `Daf.keys()` does.

Every key is read into memory. To pass many keys to `select_records_daf()` or
`delete_records()` without reading them, give the SqlDaf itself: its keys are then used
in SQL. Without a keyfield, as in a Daf, the answer is an empty list.

Returns:
    The keys, or an empty list if there is no keyfield.



##### get_existing_keys(keylist: T_la) -> T_la

Keep the keys of a list that have a row, in the order of the list, as `Daf.get_existing_keys()` does.

The keys are looked up in SQL, all at once. Without a keyfield, the result is empty, as in a Daf.

Args:
    keylist: The keys to check.

Returns:
    The keys from the list that have a row.



##### select_krows(krows: Any, inverse: bool = False, silent_error: bool = False) -> SqlDaf

Select rows by key, as `Daf.select_krows()` does.

- A key, or a list or other iterable of keys, selects those rows, in the order of the keys.
  This is `select_records_daf()`.
- A tuple of two keys selects every row from the first key to the last, both included, in
  row order. Either may be None, for the start or the end. A tuple of one key runs to the end.

A key range is found by the place of its keys among the rows, not by row positions. So it
works on a table that has had rows deleted.

A slice is not supported. In a Daf, a slice here means row positions. Use `sdaf[n:m]`.

Args:
    krows: A key, a list of keys, or a tuple of a first and last key.
    inverse: If True, select the rows that are not selected.
    silent_error: If True, keys that are not found are ignored. A key of a range must be found.

Returns:
    A selection.

Raises:
    KeyError: A key is not found.
    NotImplementedError: `krows` is a slice.
    TypeError: `krows` is None, or a tuple that is empty or longer than two.
    ValueError: There is no keyfield.



##### select_records_daf(keys_ls: Any, inverse: bool = False, silent_error: bool = False) -> SqlDaf

Select the rows with these keys, as `Daf.select_records_daf()` does.

The keys may be a list or any iterable, an in-memory Daf, or another SqlDaf of the same
database file. A SqlDaf's keys are used in SQL, as a join, so no key comes into Python. The other forms are written to a TEMP table first. The selected rows come in
the order of the keys, as in a Daf. With `inverse=True`, the rows whose key is not among
them, in this table's order.

A Python NULL key gives undefined results: see SqlDaf.

As in a Daf, a key that is not found raises KeyError, unless `silent_error` is True. That
check runs at once, as one query. The selection itself runs when first read.

Args:
    keys_ls: The keys: a list, an iterable, a Daf or a SqlDaf.
    inverse: If True, select the rows whose key is not among them.
    silent_error: If True, keys that are not found are ignored.

Returns:
    A selection.

Raises:
    KeyError: A key is not found, and `silent_error` is False.
    ValueError: This SqlDaf, or a SqlDaf given as the keys, has no keyfield.

Examples:
    >>> import tempfile
    >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
    >>> marks_daf = Daf(lol=[['b1', 1], ['b2', 0], ['b3', 1]], cols=['ballot_id', 'v'], keyfield='ballot_id')
    >>> marks_sdaf = SqlDaf.from_daf(marks_daf, name='marks')
    >>> marks_sdaf.select_records_daf(['b3', 'b1']).to_daf().to_lod()
    [{'ballot_id': 'b3', 'v': 1}, {'ballot_id': 'b1', 'v': 1}]

---

# Docs page: SqlDaf API / Modifying (api/sqldaf/modify.md)

## Modifying

Adding rows, changing values, and changing the columns.

### Changing values


##### __setitem__(slice_spec: Any, value: Any) -> None

Assign by key and column name, as in `sdaf[key, 'v'] = 1`.

These forms are supported:

- `sdaf[key, col] = value` sets one cell. A list of one value gives that value, as in a Daf.
- `sdaf[key] = row` sets the whole row with that key, from a dict or a list. Columns the
  row lacks become Python NULL, so a dict without the keyfield empties the key, which
  raises ValueError.
- `sdaf[:, col] = values` sets a column. A list goes in row order. A short list leaves the
  rows past its end as they were, and a long one is cut. A single value sets every row.

A row is found by its key, never by its position, so these work after a delete.
The key must be a str. An int in brackets is a position in a Daf, so an int row raises
NotImplementedError, even when the keyfield holds ints. To change a row with an int key,
use `update_record_krow()`. On a selection, the parent's rows
change, as a Daf selection shares its rows.

Args:
    slice_spec: A key, or a tuple of a key or `:` and a column name.
    value: The value, row or list to assign.

Raises:
    KeyError: The key or the column is not found.
    NotImplementedError: Any other form, such as an int row or several columns.
    ValueError: The change gives a Python NULL key. Nothing changes.
    sqlite3.IntegrityError: The change gives a repeated key. Nothing changes.



##### assign_record(record: T_da) -> SqlDaf

Put one row in by its key, replacing the row that has that key, as `Daf.assign_record()` does.

The whole row is replaced, in its place. Columns the dict lacks become Python NULL. A new
key is added at the end. Use `update_record_krow()` to change only some cells.

Args:
    record: The row, as a dict. It must have the keyfield.

Returns:
    This SqlDaf, which has been changed.

Raises:
    KeyError: The record lacks the keyfield.
    ValueError: There is no keyfield, or the key is a Python NULL.



##### update_record_krow(key: Any, record: T_da | None = None) -> SqlDaf

Change some cells in the row that has this key, as `Daf.update_record_krow()` does.

Only the columns named in the dict change. An int is always a key here, never a
position, so it suits a keyfield of ints. It is one UPDATE. On a selection, the
parent's row changes, and only a row of the selection is found.
For many rows that take the same values, use `update_by_keylist()`.

Args:
    key: The key of the row to change.
    record: The new values, by column name. Names that are not columns are ignored.

Returns:
    This SqlDaf, which has been changed. A record of None changes nothing.

Raises:
    KeyError: No row has this key.
    ValueError: There is no keyfield, or the change gives a Python NULL key. Nothing changes.
    sqlite3.IntegrityError: The change gives a repeated key. Nothing changes.

Examples:
    >>> import tempfile
    >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
    >>> ids_sdaf = SqlDaf('ids', cols=['id', 'v'], lol=[[10, 'a'], [20, 'b']], keyfield='id')
    >>> _ = ids_sdaf.update_record_krow(20, {'v': 'q'})
    >>> ids_sdaf.select_record(20)
    {'id': 20, 'v': 'q'}



##### update_by_keylist(keylist: T_la | None = None, record: T_da | None = None) -> SqlDaf

Change some cells in the rows that have the given keys, as `Daf.update_by_keylist()` does.

Only the columns named in the dict change. A key that is not found is skipped. It is one
UPDATE for all the keys. For one row, `update_record_krow()` raises KeyError for a missing key. On a selection, the parent's rows change, as a Daf selection
shares its rows.

Args:
    keylist: The keys of the rows to change.
    record: The new values, by column name. Names that are not columns are ignored.

Returns:
    This SqlDaf, which has been changed. With no keyfield, keys or record, nothing happens.

Raises:
    ValueError: The change gives a Python NULL key. Nothing changes.
    sqlite3.IntegrityError: The change gives a repeated key. Nothing changes.



##### assign_col(colname: str, la: T_la | None = None, default: Any = NULL) -> SqlDaf

Fill a column by name with the values of a list, or add it if it is new, as `Daf.assign_col()` does.

The values go to the rows in order. A short list is filled out with `default`, and a long
one is cut. With no list, every row gets `default`. A new column is added at the end, with
ALTER TABLE, on a parent only.

Args:
    colname: The column name.
    la: The values, in row order.
    default: The value for rows that the list does not reach.

Returns:
    This SqlDaf, which has been changed.

Raises:
    NotImplementedError: The column is new, and this is a selection.
    ValueError: The change gives a Python NULL key. Nothing changes.
    sqlite3.IntegrityError: The change gives a repeated key. Nothing changes.



##### apply_to_col(col: str, func: Callable, kwargs: Any = {}) -> SqlDaf

Replace each value of a column by the result of a function, in place, as `Daf.apply_to_col()` does.

It is one UPDATE, with the function registered in SQLite. The function gets Python NULL
for SQL NULL. An error it raises is raised as itself, and nothing changes.

Args:
    col: The column.
    func: Called as func(value, **kwargs). It returns the new value.
    **kwargs: Passed on to func.

Returns:
    This SqlDaf, which has been changed.

Raises:
    ColumnNotFoundError: The name is not a column.
    ValueError: The change gives a Python NULL key. Nothing changes.
    sqlite3.IntegrityError: The change gives a repeated key. Nothing changes.



##### apply_in_place(func: Callable[..., Any], by: str = 'row', rowkeys: T_la | None = None, kwargs: Any = {}) -> SqlDaf

Apply a function to each row and store the results, as `Daf.apply_in_place()` does.

With `by='row'`, the function gets each row as a dict and returns a dict. Its values are
written back by column name. A column the dict lacks keeps its value. With
`by='row_klist'`, the function gets a [KeyedList][daffodil.keyedlist.KeyedList], changes
it, and returns nothing. The whole row is written back.

Rows are read and written in batches, in one transaction. If the function raises, nothing
changes. With `rowkeys`, only those rows change, and a key that is not found is skipped,
as in a Daf. On a selection, the parent's rows change.

Args:
    func: The function to apply to each row.
    by: 'row' or 'row_klist'.
    rowkeys: The keys of the rows to change. All rows if None.
    **kwargs: Passed on to func.

Returns:
    This SqlDaf, which has been changed.

Raises:
    NotImplementedError: `by` is not 'row' or 'row_klist'.
    ValueError: With `by='row'` the function returned None, or a key became a Python NULL.
    sqlite3.IntegrityError: A key became repeated. Nothing changes.


### Adding rows


##### append(data_item: Any = None, respect_kd: bool = False, lol: T_lola | None = None, la: T_la | None = None, fast: bool = False) -> SqlDaf

Add one row, or several, to the end, as `Daf.append()` does.

- A dict or a KeyedList is one row, placed by column name.
- A list of values is one row, in column order. A short list is padded with Python NULL.
- A list of dicts is several rows. See `extend()`.
- A Daf or a SqlDaf is several rows. See `concat()`.
- `lol=` is several rows, each a list. `la=` is one row.

Each call is one transaction, and each commit writes to disk. On 1,000,000 rows, one
call took 9.7 ms on its own, and 55 us inside `transaction()`. So add many rows in one
call, with `extend()`, or put a loop of single rows inside `transaction()`.

Keys are unique in a SqlDaf. A key that is already there raises sqlite3.IntegrityError,
and nothing is added. With `respect_kd=True`, the row with that key is replaced, in its place. A
Python NULL key raises ValueError, as in `concat()`.

On a selection, the selection gets rows of its own first, as `concat()` does. Its parent
is not changed.

Args:
    data_item: A dict, a KeyedList, a list, a list of dicts, or a Daf.
    respect_kd: If True, replace the row that has the same key.
    lol: Several rows, each a list in column order.
    la: One row, a list in column order.
    fast: Accepted for Daf code. Every row is checked anyway.

Returns:
    This SqlDaf, which has been changed.

Raises:
    TypeError: More than one of `data_item`, `lol` and `la`, or a row of an unknown type.
    ValueError: A row has more values than there are columns, or a key is a Python NULL.
    sqlite3.IntegrityError: A key would be repeated. Nothing is added.



##### record_append(record: T_da | KeyedList, respect_kd: bool = True) -> SqlDaf

Add one row given as a dict or a KeyedList, as `Daf.record_append()` does.

By default the row with the same key is replaced, in its place. This is the default
here, unlike `append()`. An empty record adds nothing.

Args:
    record: The row, as a dict or a [KeyedList][daffodil.keyedlist.KeyedList].
    respect_kd: If True, the default, replace the row that has the same key.

Returns:
    This SqlDaf, which has been changed.

Raises:
    TypeError: The record is not a dict or a KeyedList.
    ValueError: The key is a Python NULL.
    sqlite3.IntegrityError: The key is there, and `respect_kd` is False. Nothing is added.



##### extend(records_lod: List[T_da] | None = None, respect_kd: bool = False, lol: T_lola | None = None) -> SqlDaf

Add several rows, as dicts or as lists, as `Daf.extend()` does.

Each dict is placed by column name. Each list of `lol` is in column order. All rows are
added in one transaction. If one fails, none is added.

Args:
    records_lod: The rows, as dicts.
    respect_kd: If True, replace the row that has the same key.
    lol: The rows, as lists in column order.

Returns:
    This SqlDaf, which has been changed.

Raises:
    TypeError: Both `records_lod` and `lol` are given, or a row of `lol` is not a list.
    ValueError: A row has more values than there are columns, or a key is a Python NULL.
    sqlite3.IntegrityError: A key would be repeated. Nothing is added.



##### concat(other_instance: Daf, respect_kd: bool = False) -> SqlDaf

Add the rows of another Daf or SqlDaf to the end of this one, as `Daf.concat()` does.

The other must have the same columns, in the same order, as in a Daf. A SqlDaf of the
same database file is copied in SQL, in its row order, so no row comes into Python. An in-memory
Daf is written row by row. All of it is one transaction.

If this is a child, it first becomes a table of its own, by `copy(inplace=True)`, as
appending to a selection of a Daf adds only to that selection. The parent is not
changed.

A Daf appends a row whose key is already there without a word. A SqlDaf with a keyfield
has a unique index on it, so a repeated key raises instead, and nothing is added. A
Python NULL key raises ValueError, and nothing is added.

Args:
    other_instance: The Daf or SqlDaf whose rows are added. It is not changed.
    respect_kd: If True, a row whose key is there replaces that row, in its place.

Returns:
    This SqlDaf.

Raises:
    KeyError: The columns differ.
    ValueError: A key added is a Python NULL. Nothing is added.
    sqlite3.IntegrityError: A key would be repeated. Nothing is added.
    NotImplementedError: The other is a SqlDaf this connection cannot read.


### Removing rows


##### delete_records(keys_ls: Any, inplace: bool = False, silent_error: bool = False) -> SqlDaf

Remove the rows with these keys, as `Daf.delete_records()` does. A Python NULL key gives
undefined results: see SqlDaf.

By default nothing is changed: a selection without those rows is returned, as
`select_records_daf(keys, inverse=True)` gives. That is always safe.

With `inplace=True`, the rows are removed from this SqlDaf, and it is returned.

- On a selection, its references to those rows are removed, and the rest renumbered. The
  parent is not changed, so this is safe, as in a Daf.
- On a parent, the rows are deleted from the database. After that, it cannot be read
  by position: see SqlDaf. A selection made from it earlier still counts a deleted
  row in `len()`, though `to_daf()` leaves it out. So delete in place from a parent
  only when nothing else uses it.

The keys may be in any form that `select_records_daf()` takes. A key that is not found
raises KeyError, and nothing is removed, unless `silent_error` is True.

Args:
    keys_ls: The keys of the rows to remove, in any form `select_records_daf()` takes.
    inplace: If True, remove the rows from this SqlDaf and return it. If False, return a selection.
    silent_error: If True, keys that are not found are ignored.

Returns:
    This SqlDaf if `inplace`, otherwise a selection without the rows.

Raises:
    KeyError: A key is not found, and `silent_error` is False. Nothing is removed.
    ValueError: This SqlDaf, or a SqlDaf given as the keys, has no keyfield.



##### remove_dups(keyfield: str = '') -> Tuple[SqlDaf, SqlDaf]

Split the rows into those with a unique key and those with a repeated key, as `Daf.remove_dups()` does.

Only the key column is compared, not the whole row.
For each key, the last row is the unique one. The earlier rows with that key are the duplicates.
The unique rows come in the order in which their keys first appear.
The duplicates keep this table's order.

Both are selections of this table's rows, as a Daf shares its rows with both.
Both have the key column as their keyfield.
SQL NULL and '' are one key here, as Python NULL is one key in a Daf.
No key index is made, so the column may hold repeated keys.
Each side is one GROUP BY, run when it is first read.

A key of several columns is not supported, since a SqlDaf keyfield is one column.

Args:
    keyfield: The column that identifies a row. If empty, the keyfield of this SqlDaf.

Returns:
    A tuple of the selection of unique rows and the selection of duplicate rows.

Raises:
    KeysDisabledError: There is no `keyfield`, and this SqlDaf has none.
    KeyError: The key column is not found.
    NotImplementedError: The key is several columns.

Examples:
    >>> import tempfile
    >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
    >>> marks_sdaf = SqlDaf('marks', cols=['id', 'v'], lol=[[1, 'a'], [2, 'b'], [1, 'c']])
    >>> unique_sdaf, dups_sdaf = marks_sdaf.remove_dups('id')
    >>> unique_sdaf.to_daf().lol, dups_sdaf.to_daf().lol
    ([[1, 'c'], [2, 'b']], [[1, 'a']])


### Columns and layout


##### insert_col(colname: str, col_la: T_la | None = None, icol: int = -1, default: Any = NULL) -> SqlDaf

Add a named column, or fill it if the name is there, as `Daf.insert_col()` does.

A new column goes at the end, since ALTER TABLE adds columns only there.
Any other position raises NotImplementedError.
To read the columns in another order, use `select_cols()`. It gives new rows, in the order you name.

A list that is too short is filled out with `default`.
With no list, every row gets `default`.
If the name is there, its values are replaced, and `icol` is ignored.
An empty name does nothing.
A new column is added on a parent only, as `assign_col()` adds it.

Args:
    colname: The name of the column.
    col_la: The values, in row order.
    icol: The position. Only the end is supported: -1, or the number of columns.
    default: The value for rows that the list does not reach.

Returns:
    This SqlDaf, which has been changed.

Raises:
    NotImplementedError: A new column at a position other than the end, or on a selection.

Examples:
    >>> import tempfile
    >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
    >>> marks_sdaf = SqlDaf('marks', cols=['ballot_id', 'v'], lol=[['b1', 1], ['b2', 0]], keyfield='ballot_id')
    >>> marks_sdaf.insert_col('w', ['x']).to_daf().lol
    [['b1', 1, 'x'], ['b2', 0, '']]



##### drop_cols(exclude_cols: T_ls | None = None) -> SqlDaf

Remove columns from the table, in place, as `Daf.drop_cols()` does.

Each column goes with ALTER TABLE DROP COLUMN, which rewrites the table.
On 1,000,000 rows of 6 columns, one drop took 0.73 s, twice the time of `copy()`.
A name that is not a column is ignored. With no names, nothing happens.
The dtypes lose the columns dropped.

If the keyfield is dropped, it is cleared, as in a Daf, and its index goes with it.
Set a keyfield again with `set_keyfield()`.

A selection has its parent's columns, so a drop on one raises NotImplementedError.
A selection made before the drop still names the dropped column.
Reading it then raises sqlite3.OperationalError, so make it again.

Args:
    exclude_cols: The names of the columns to remove.

Returns:
    This SqlDaf, which has been changed.

Raises:
    NotImplementedError: This is a selection.
    sqlite3.OperationalError: An index of your own is on a column dropped. Nothing changes.

Examples:
    >>> import tempfile
    >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
    >>> marks_sdaf = SqlDaf('marks', cols=['ballot_id', 'v', 'n'], lol=[['b1', 1, 10]], keyfield='ballot_id')
    >>> marks_sdaf.drop_cols(['v']).columns()
    ['ballot_id', 'n']



##### rename_cols(from_to_dict: T_ds) -> SqlDaf

Rename columns in place, as `Daf.rename_cols()` does.

Each is ALTER TABLE RENAME COLUMN, which changes only the schema.
On 1,000,000 rows it took 15 ms.
Names that are not in the mapping stay as they are.
Names in the mapping that are not columns are ignored.
Names may be swapped, as in `{'a': 'b', 'b': 'a'}`.
The dtypes are renamed too.

The keyfield follows the rename, as in a Daf.
Its index is made again under the new name, the next time a key is used.

A Daf lets two columns get the same name. A SqlDaf raises then, and nothing changes.
A selection has its parent's columns, so a rename on one raises NotImplementedError.
A selection made before the rename still names the old columns.
Reading it then raises sqlite3.OperationalError, so make it again.

Args:
    from_to_dict: Maps old names to new names.

Returns:
    This SqlDaf, which has been changed.

Raises:
    NotImplementedError: This is a selection.
    ValueError: A new name is reserved by SqlDaf. Nothing changes.
    sqlite3.OperationalError: Two columns would have the same name. Nothing changes.

Examples:
    >>> import tempfile
    >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
    >>> marks_sdaf = SqlDaf('marks', cols=['id', 'v'], lol=[['b1', 1]], keyfield='id')
    >>> marks_sdaf.rename_cols({'id': 'ballot_id'}).columns(), marks_sdaf.keyfield
    (['ballot_id', 'v'], 'ballot_id')



##### set_cols(new_cols: T_ls | None = None, sanitize_cols: bool = True, unnamed_prefix: str = 'col') -> SqlDaf

Set the column names by position, in place, as `Daf.set_cols()` does.

The names are made as in a Daf.
Without a list, they are A, B, C and so on.
With `sanitize_cols`, a repeated name gets a suffix, and an empty name becomes the prefix and its position.
Each name that changes is renamed as `rename_cols()` renames it, so the rows stay as they are.
The keyfield and the dtypes follow the names.

Args:
    new_cols: The names, in order. If None, spreadsheet names are made.
    sanitize_cols: If True, make the names unique and not empty.
    unnamed_prefix: The start of a name made for an empty one.

Returns:
    This SqlDaf, which has been changed.

Raises:
    AttributeError: The number of names is not the number of columns, as in a Daf.
    NotImplementedError: This is a selection.

Examples:
    >>> import tempfile
    >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
    >>> SqlDaf('t', cols=['a', 'b', 'c'], lol=[[1, 2, 3]]).set_cols(['x', 'x', '']).columns()
    ['x', 'x_1', 'col2']



##### set_dtypes(default_type: Type = str, typ_to_cols_dict: Dict[Type, T_ls] | None = None) -> SqlDaf

Set the dtype of every column, from a default and a few exceptions, as `Daf.set_dtypes()` does.

No value is converted. `apply_dtypes()` does that.
On a permanent table with a record, the record is changed too, as `set_keyfield()` changes it.
The record can store these dtypes: int, float, bool, str, list and dict.

Args:
    default_type: The type of every column that is not listed.
    typ_to_cols_dict: Maps a type to the names of the columns of that type.

Returns:
    This SqlDaf, which has been changed.

Raises:
    NotImplementedError: A type the record cannot hold, on a table with a record. Nothing changes.

Examples:
    >>> import tempfile
    >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
    >>> SqlDaf('t', cols=['a', 'b'], lol=[['1', 'x']]).set_dtypes(str, {int: ['a']}).dtypes
    {'a': <class 'int'>, 'b': <class 'str'>}



##### attach_schema(schema: type) -> SqlDaf

Attach a `@schemaclass` to this SqlDaf, as `Daf.attach_schema()` does.

It stores the schema, then fills in the dtypes and the keyfield, each only if there is none.
No value is converted, and nothing is checked against the rows.
Call `apply_dtypes()` to convert.
On a permanent table with a record, the record is changed too, as `set_keyfield()` changes it.

A Daf with no columns takes them from the schema.
A SqlDaf gets its columns when its table is made, so one with no columns raises NotImplementedError.

Args:
    schema: A class decorated with `@schemaclass`.

Returns:
    This SqlDaf, which has been changed.

Raises:
    TypeError: The class is not a `@schemaclass`.
    KeyError: The schema's keyfield is not a column.
    NotImplementedError: This SqlDaf has no columns, or a dtype the record cannot hold.

Examples:
    >>> import tempfile
    >>> from daffodil.lib.schemaclass import schemaclass
    >>> @schemaclass
    ... class Mark:
    ...     __keyfield__ = 'ballot_id'
    ...     ballot_id: str = ''
    ...     v: int = 0
    >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
    >>> marks_sdaf = SqlDaf('marks', cols=['ballot_id', 'v'], lol=[['b1', '1']]).attach_schema(Mark)
    >>> marks_sdaf.keyfield, marks_sdaf.dtypes
    ('ballot_id', {'ballot_id': <class 'str'>, 'v': <class 'int'>})



##### clone_empty(lol: T_lola | None = None, cols: T_ls | None = None, name: str = '') -> SqlDaf

Make a new table with the same layout, as `Daf.clone_empty()` does.

It has the same columns, keyfield and dtypes, and no rows unless `lol` is given.
It is a TEMP table, as `copy()` makes, so `save()` keeps it.
With `cols` that differ, it has those columns.
The dtypes then keep the entries for the columns still there.
The keyfield is cleared if its column is gone.

Args:
    lol: Rows for the new table, written as the constructor writes them. None for no rows.
    cols: Column names to use instead of these.
    name: The name of the new table. Generated if empty, as in daf_tmp_marks_v4.

Returns:
    The new SqlDaf.

Examples:
    >>> import tempfile
    >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
    >>> marks_sdaf = SqlDaf('marks', cols=['ballot_id', 'v'], lol=[['b1', 1]], keyfield='ballot_id')
    >>> empty_sdaf = marks_sdaf.clone_empty()
    >>> empty_sdaf.columns(), empty_sdaf.keyfield, len(empty_sdaf)
    (['ballot_id', 'v'], 'ballot_id', 0)


### Sorting


##### sort_by_colname(colname: str, reverse: bool = False, length_priority: bool = False, as_str: bool = False) -> SqlDaf

Sort the rows by one column, in place, as `Daf.sort_by_colname()` does.

See `sort_by_colnames()` for how the rows are sorted and where they go.

Args:
    colname: The column to sort by.
    reverse: If True, sort from high to low.
    length_priority: If True, sort by length first, then by value.
    as_str: If True, sort by the text of each value, and Python NULL sorts as ''.

Returns:
    This SqlDaf, now a sorted TEMP table.

Raises:
    KeyError: The column name is not found.

Examples:
    >>> import tempfile
    >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
    >>> SqlDaf('t', cols=['a'], lol=[['10'], ['99'], ['8']]).sort_by_colname('a', length_priority=True).col('a')
    ['8', '10', '99']



##### sort_by_colnames(colnames: T_ls, reverse: bool = False, length_priority: bool = False, as_str: bool = False) -> SqlDaf

Sort the rows by several columns, in place, as `Daf.sort_by_colnames()` does.

The first column is the main sort key. The others break ties.
Rows that tie on every column keep their order, as Python's sort keeps them.

Where the sorted rows go:

The rows are copied in sorted order into a TEMP table, and this SqlDaf becomes it.
This is `copy(inplace=True)` with an ORDER BY, in one statement.
The table it came from is not changed, so its other selections stay as they were.
A Daf selection sorts its own list of rows, and leaves its parent alone too.
On a permanent table, the table in the file keeps its order. Use `save()` to keep the sorted copy.
On 1,000,000 rows of 6 columns, a sort by one text column took 2.4 s, against 1.0 s for a Daf.

How values compare:

- Python NULL, SQL NULL or '', sorts first, as an empty cell sorts first in a Daf.
- Text compares by code point, as Python compares it.
- `length_priority` sorts by the length of the text first, then by the text. That gives
  numeric order for whole numbers held as text, when none is negative or has a leading zero.
  It is not a numeric sort for negative numbers or decimals.
- `as_str` sorts by the text of each value. A float such as 1e20 is written differently by SQLite.
- A column that mixes numbers and text sorts every number before all text.
  A Daf raises TypeError on such a column instead.

Args:
    colnames: The columns to sort by, main key first.
    reverse: If True, sort from high to low.
    length_priority: If True, sort by length first, then by value.
    as_str: If True, sort by the text of each value, and Python NULL sorts as ''.

Returns:
    This SqlDaf, now a sorted TEMP table.

Raises:
    KeyError: A column name is not found.

Examples:
    >>> import tempfile
    >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
    >>> SqlDaf('t', cols=['p', 'q'], lol=[[2, 'b'], [1, 'z'], [1, 'a']]).sort_by_colnames(['p', 'q']).to_daf().lol
    [[1, 'a'], [1, 'z'], [2, 'b']]

---

# Docs page: SqlDaf API / Apply, reduce, group and join (api/sqldaf/apply.md)

## Apply, reduce, group and join

Functions over rows, groups and counts, and joins.

### Functions over rows


##### apply(func: Callable[..., Any], by: str = 'row', keylist: T_la | None = None, kwargs: Any = {}) -> Any

Apply a Python function to each row, and collect the results in a new table, as `Daf.apply()` does.

The function gets each row as a dict, with Python NULL for a missing value, and returns what to
add to the new table, by the rules of `Daf.apply()`:

- A dict, or a KeyedList, adds one row.
- A list of them adds a row for each, so one row can become several.
- None, an empty dict or an empty list adds nothing, so the row is left out.

The new table takes its columns from the first row returned. A later row is placed by
column name. A column it lacks gets Python NULL, and a name the first row did not have is dropped.
So have the function return every column in every row. The new table has no keyfield
and no dtypes, as the new Daf of `Daf.apply()` has none.

The rows are read 10,000 at a time, in order, and the results are written as they come,
all in one transaction. So a table larger than memory can be converted, and memory does
not grow. If the function raises, that error is raised, as itself, and nothing is kept.
The new table is a TEMP table, as `copy()` makes. Keep it with `save(name)`.

On 300,000 rows that each became 4, this took 23.1 s, most of it in the function. A
registered SQL function cannot return a row, which is why the function runs in Python.

With `by='table'` the function gets this SqlDaf, and its result is returned as it is.

Args:
    func: The function. It takes a row and the keyword arguments, or this SqlDaf for `by='table'`.
    by: `row` to apply it to each row, or `table` to apply it to the whole SqlDaf.
    keylist: Not supported on a SqlDaf. Select the rows first.
    **kwargs: Keyword arguments passed on to the function.

Returns:
    The new table, or the result of the function for `by='table'`.

Raises:
    NotImplementedError: `keylist` is given, or `by` is not `row` or `table`.
    TypeError: The function returns something other than a dict, a KeyedList, a list of them, or nothing.

Examples:
    >>> import tempfile
    >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
    >>> votes_sdaf = SqlDaf.from_daf(Daf(lol=[['b1', 'c0,c1'], ['b2', '']], cols=['ballot_id', 'contests']), name='votes')
    >>> def split(row):
    ...     return [{'ballot_id': row['ballot_id'], 'contest': c} for c in row['contests'].split(',') if c]
    >>> votes_sdaf.apply(split).to_daf().lol
    [['b1', 'c0'], ['b1', 'c1']]



##### reduce(func: Callable[..., Any], by: str = 'row', cols: T_ls | None = None, initial_da: T_da | None = None, indirect_col: str = '', silent_error: bool = False, kwargs: Any = {}) -> Any

Combine all the rows into one result, using a function, as `Daf.reduce()` does.

The function is called in Python for each row, with the same arguments and the same
result as in `Daf.reduce()`. A sum or a count is a reduction, as with `Daf.sum_da`.
The rows are read 10,000 at a time, in order, so the table is never all in memory.
Memory use still depends on the result the function builds.

- With `by='row'`, each row is a [KeyedList][daffodil.keyedlist.KeyedList] of its
  values. Unlike in a Daf, writing to it does not change the table.
- With `by='sparse_row'`, the function gets the dict held in `indirect_col`, JSON or
  PYON text decoded as in a Daf, and the result starts as `initial_da` or an empty dict.
- With `by='table'`, the function gets this SqlDaf and `cols`.

For a plain total of columns, `sum()` is faster: SQLite adds them, and no row comes
into Python. `by='col'` is not supported, since it needs each whole column in memory.

Args:
    func: The reduction function, such as `Daf.sum_da`.
    by: `row`, `sparse_row` or `table`.
    cols: The columns included in the reduction. All columns if None.
    initial_da: The result to start from, instead of zeros.
    indirect_col: A column that holds a dict. Required for `sparse_row`.
    silent_error: If True, a row for which the function raises is skipped.
    **kwargs: Keyword arguments passed on to the function.

Returns:
    A dict for `row` and `sparse_row`. The function's result for `table`.

Raises:
    ValueError: `sparse_row` is used with no `indirect_col`.
    NotImplementedError: `by` is `col`, or not recognized.

Examples:
    >>> import tempfile
    >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
    >>> votes_sdaf = SqlDaf('votes', cols=['ballot_id', 'v', 'concise_votes'],
    ...                     lol=[['b1', 1, "{'c0b': 1, 'c0v': 1}"], ['b2', 2, '{"c0b": 1, "c1b": 1}']])
    >>> votes_sdaf.reduce(Daf.sum_da, cols=['v'])
    {'ballot_id': '', 'v': 3, 'concise_votes': ''}
    >>> votes_sdaf.reduce(Daf.sum_da, by='sparse_row', indirect_col='concise_votes')
    {'c0b': 2, 'c0v': 1, 'c1b': 1}


### Groups


##### groupby(colname: str = '', colnames: T_ls | None = None, omit_nulls: bool = False, cols: T_ls | None = None) -> KeyedList

Group the rows by the values of a column, as `Daf.groupby()` does.

The result is a KeyedList of selections, one for each value, in the order first seen. It
works as a dict, as the result of `Daf.groupby()` does. One query finds the values. Each
group is then `select_by_dict()` of its value, so Python NULL is a value like any other. Nothing
is read until a group is read.

The first groupby on a column makes an index on it, in the parent table, unless an index
already starts with it. Each group is then a lookup. Without the index, reading each
group would need a full scan of the table. On 100,000 rows in 500 groups, summing every group took 2.3 s, against
14.8 s without it. The index stays with the table, and `drop_cols()` or `rename_cols()`
of the column drops it.

Args:
    colname: The column to group by.
    colnames: Several columns to group by. See `groupby_cols()`.
    omit_nulls: If True, leave out the rows with Python NULL in the column.
    cols: Not supported on a SqlDaf yet.

Returns:
    A KeyedList that maps each value to a selection.

Raises:
    KeyError: The column is not a column.
    NotImplementedError: `cols` is given.

Examples:
    >>> import tempfile
    >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
    >>> bif_sdaf = SqlDaf.from_daf(Daf(lol=[['b1', 'P1'], ['b2', 'P2'], ['b3', 'P1']], cols=['ballot_id', 'precinct']), name='bif')
    >>> groups = bif_sdaf.groupby('precinct')
    >>> {key: group.to_daf().lol for key, group in groups.items()}
    {'P1': [['b1', 'P1'], ['b3', 'P1']], 'P2': [['b2', 'P2']]}



##### groupby_cols(colnames: T_ls, cols: T_ls | None = None, omit_nulls: bool = False) -> KeyedList

Group the rows by the values of several columns, as `Daf.groupby_cols()` does.

Each key is a tuple of the values, even for one column. Each group is a selection, as in
`groupby()`.

Args:
    colnames: The columns to group by.
    cols: Not supported on a SqlDaf yet.
    omit_nulls: If True, leave out the rows with Python NULL in any of the columns.

Returns:
    A KeyedList that maps each tuple of values to a selection.

Raises:
    KeyError: A name in `colnames` is not a column.
    NotImplementedError: `cols` is given.



##### group_where(where: Callable[[Any], Any], indirect_col: str | None = None, cols: T_ls | None = None) -> KeyedList

Group the rows by the keys a Python function returns, as `Daf.group_where()` does.

The function gets each row as a KeyedList, as in `select_where()`, and returns a key, a
list of keys, or None. A row goes into the group of each key, so a row can be in any
number of groups, and None leaves it out. As in `Daf.group_where()`, any iterable other
than a str is a list of keys, a tuple too. So to use a tuple as one key, return a list
of tuples.

The function runs once for each row, in one pass, and each (key, row) pair is written to
a TEMP table. Each group is then a selection of the rows listed under its key, in the
order of this SqlDaf. Nothing more is read until a group is read. The result is a
KeyedList of selections, in the order the keys were first returned.

Name the columns the function reads in `cols`, as in `select_where()`.

Args:
    where: A function of a row: a key, a list of keys, or None.
    indirect_col: Not supported on a SqlDaf yet.
    cols: The columns the function reads. All if None.

Returns:
    A KeyedList that maps each key to a selection.

Raises:
    KeyError: A name in `cols` is not a column.
    NotImplementedError: `indirect_col` is given.

Examples:
    >>> import tempfile
    >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
    >>> bif_sdaf = SqlDaf.from_daf(Daf(lol=[['b1', 'P1'], ['b2', 'P2'], ['b3', 'P1']], cols=['ballot_id', 'precinct']), name='bif')
    >>> groups = bif_sdaf.group_where(lambda row: [row['precinct'], 'all'], cols=['precinct'])
    >>> {key: len(group) for key, group in groups.items()}
    {'P1': 2, 'all': 3, 'P2': 1}



##### groupsum_daf(colname: str, by: str = 'row', reduce_cols: T_ls | None = None) -> Daf

Group by a column and add up the other columns of each group, as `Daf.groupsum_daf()` does.

One GROUP BY does it all in SQLite. The result is a Daf in memory, with one row for each
value, in the order first seen, and `colname` as its keyfield. As in a
Daf, only numbers are added: text, even text of digits, counts as 0, and a column with
no numbers in a group totals 0. A column not in `reduce_cols` is Python NULL.

Args:
    colname: The column to group by.
    by: Only `row` is supported.
    reduce_cols: The columns to add. All but `colname` if None.

Returns:
    A Daf in memory, with one row for each group.

Raises:
    KeyError: `colname` is not a column.
    NotImplementedError: `by` is not `row`.

Examples:
    >>> import tempfile
    >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
    >>> nums_sdaf = SqlDaf.from_daf(Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y']), name='nums')
    >>> nums_sdaf.groupsum_daf('g', reduce_cols=['y']).lol
    [['a', '', 40], ['b', '', 20]]



##### sum(colnames_ls: T_ls | None = None, numeric_only: bool = False) -> T_da

Total the columns, as `Daf.sum()` does, and return a dict of the totals.

Numbers are totalled by SQLite. A column that also holds text has its text cells read
into Python and added as `Daf.sum()` adds them: text that is not a number raises
ValueError, or counts as 0 with `numeric_only`. Python NULL is skipped. Each total starts as a
float, and is converted to the dtype of its column if there is one.

Args:
    colnames_ls: The columns to total. All if None.
    numeric_only: If True, total only the columns with an int or float dtype, and count text that is not a number as 0.

Returns:
    A dict that maps each column name to its total.

Raises:
    ValueError: A text cell cannot be converted to a number.

Examples:
    >>> import tempfile
    >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
    >>> SqlDaf.from_daf(Daf(lol=[[1, 10], [2, 20]], cols=['x', 'y']), name='nums').sum()
    {'x': 3.0, 'y': 30.0}


### Counts


##### valuecounts_for_colname(colname: str, sort: bool = False, reverse: bool = True, omit_nulls: bool = False) -> T_di

Count how often each value occurs in one column, as `Daf.valuecounts_for_colname()` does.

One GROUP BY counts them in SQLite, so no column is read into Python. The values are in
the order first seen, or by count with `sort`. Python NULL is counted as Python NULL, unless
`omit_nulls`. A column that does not exist gives an empty dict.

Args:
    colname: The column to count.
    sort: If True, order by count.
    reverse: With `sort`, True puts the most common first.
    omit_nulls: If True, leave out the count of Python NULL.

Returns:
    A dict that maps each value to its count.

Examples:
    >>> import tempfile
    >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
    >>> bif_sdaf = SqlDaf.from_daf(Daf(lol=[['b1', 'P1'], ['b2', 'P2'], ['b3', 'P1']], cols=['ballot_id', 'precinct']), name='bif')
    >>> bif_sdaf.valuecounts_for_colname('precinct', sort=True)
    {'P1': 2, 'P2': 1}



##### valuecounts_for_colnames_ls(colnames_ls: T_ls | None = None, sort: bool = False, reverse: bool = True, omit_nulls: bool = False) -> Dict[str, T_di]

Count how often each value occurs, in each of several columns, as `Daf.valuecounts_for_colnames_ls()` does.

It is one GROUP BY for each column. See `valuecounts_for_colname()`.

Args:
    colnames_ls: The columns to count. All if None.
    sort: If True, order each count by size.
    reverse: With `sort`, True puts the most common first.
    omit_nulls: If True, leave out the count of Python NULL.

Returns:
    A dict that maps each column to a dict of value and count.



##### valuecounts_for_colname_selectedby_colname(colname: str, selectedby_colname: str, selectedby_colvalue: Any, sort: bool = False, reverse: bool = True) -> T_di

Count the values of a column, in the rows where another column has a value, as the Daf method does.

Args:
    colname: The column to count.
    selectedby_colname: The column to test.
    selectedby_colvalue: Only rows where that column has this value are counted. Python NULL matches Python NULL.
    sort: If True, order the counts by size.
    reverse: With `sort`, True puts the most common first.

Returns:
    A dict that maps each value to its count. It is empty if a column does not exist.



##### valuecounts_for_colnames_ls_selectedby_colname(colnames_ls: T_ls | None = None, selectedby_colname: str = '', selectedby_colvalue: Any = '', sort: bool = False, reverse: bool = True) -> Dict[str, T_di]

Count the values of several columns, in the rows where another column has a value, as the Daf method does.

Args:
    colnames_ls: The columns to count. All if None.
    selectedby_colname: The column to test.
    selectedby_colvalue: Only rows where that column has this value are counted.
    sort: If True, order the counts by size.
    reverse: With `sort`, True puts the most common first.

Returns:
    A dict that maps each column to a dict of value and count.



##### valuecounts_for_colname1_groupedby_colname2(colname1: str, groupedby_colname2: str, sort: bool = False, reverse: bool = True) -> Dict[Any, T_di]

Count the values of one column, for each value of another, as the Daf method does.

One GROUP BY of the two columns counts them all. The groups, and the values in each, are
in the order first seen. Use it to see whether two columns relate one to one: each group
should then hold a single value.

Args:
    colname1: The column whose values are counted.
    groupedby_colname2: The column whose values form the groups.
    sort: If True, order each count by size.
    reverse: With `sort`, True puts the most common first.

Returns:
    A dict that maps each value of the second column to a dict of value and count.



##### value_counts_daf(colname: str, sort: bool = False, reverse: bool = True, include_total: bool = False, omit_nulls: bool = False) -> Daf

Count the values of a column into a Daf in memory, with one row per value, as `Daf.value_counts_daf()` does.

The counts come from `valuecounts_for_colname()`. The Daf has the columns `colname` and
counts, and a row for each value.

Args:
    colname: The column to count.
    sort: If True, order by count.
    reverse: With `sort`, True puts the most common first.
    include_total: If True, add a row with the total.
    omit_nulls: If True, leave out the count of Python NULL.

Returns:
    A Daf in memory, with a row for each value.



##### daf_valuecount(by: str = 'row', cols: T_ls | None = None) -> T_da

Count how often each value occurs, in each column, as `Daf.daf_valuecount()` does.

It gives the same result: a column that is not in `cols` gets Python NULL in place of its
counts, and a table with no rows gives an empty dict. Each column is one GROUP BY.

Args:
    by: Only `row` is supported.
    cols: The columns to count. All if None.

Returns:
    A dict that maps each column to a dict of value and count.

Raises:
    NotImplementedError: `by` is not `row`.


### Joins


##### join(other_daf: SqlDaf, how: str = 'inner', shared_fields: T_ls | None = None, tag_other: bool = False, custom_translator_daf: Daf | None = None, diagnose: bool = False, name: str = '', fill: Any = NULL) -> SqlDaf

Join two SqlDaf tables on their keyfields, as `Daf.join()` does, in SQL.

The columns are named exactly as `Daf.join()` names them, by the same translator: a
column that both have gets the name of its source as a suffix, such as `name_audit`,
unless it is in `shared_fields`, and `tag_other` suffixes every column of the other. The
names are those of the parent tables, or `daf1` and `daf2` if they are the same. The rows
come as in `Daf.join()`: in this table's order, and for `right` and `outer`, the other
table's unmatched rows after, in its order. A cell with no match is Python NULL. The keyfield of
the result is this table's.

The keyfields must be unique, as in `Daf.join()`. A repeated key is not checked, and the
result is then undefined. Check the keys first if they may repeat. A Python NULL key is
undefined too: see SqlDaf.

A row found only in the other table, in a right or outer join, takes its key and its
shared fields from the other table. `Daf.join()` does not do this in every case yet.
That is a known bug in `Daf.join()`.

Nothing runs yet. The join is resolved the first time it is read, into a TEMP table of
new rows, as `Daf.join()` makes new rows. A selection made from it resolves it first.

Args:
    other_daf: The SqlDaf to join with, in the same database file.
    how: `inner`, `left`, `right` or `outer`.
    shared_fields: Columns that both have and that appear only once.
    tag_other: If True, every column of the other, except shared ones, gets a suffix.
    custom_translator_daf: Not supported on a SqlDaf yet.
    diagnose: Not used.
    name: The name of the result table. Generated if empty.
    fill: The value for cells that have no match. Only Python NULL is supported.

Returns:
    A SqlDaf of the joined rows, resolved when first read.

Raises:
    ValueError: `how` is not one of the four, or a table has no keyfield.
    NotImplementedError: `custom_translator_daf` or a `fill` other than Python NULL, or a SqlDaf this connection cannot read.

Examples:
    >>> import tempfile
    >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
    >>> a_sdaf = SqlDaf.from_daf(Daf(lol=[[1, 'Alice'], [2, 'Bob']], cols=['id', 'name'], keyfield='id'), name='a')
    >>> b_sdaf = SqlDaf.from_daf(Daf(lol=[[1, 50], [3, 70]], cols=['id', 'salary'], keyfield='id'), name='b')
    >>> a_sdaf.join(b_sdaf, how='left').to_daf().lol
    [[1, 'Alice', 50], [2, 'Bob', '']]

---

# Docs page: KeyedList API / KeyedList (api/keyedlist.md)

## KeyedList


##### KeyedList

A row that reads like a dict but points at a list instead of copying its values.

A KeyedList pairs a list of values with an index of the keys. You read it by key, as in
`row['qty']`, the same way you read a dict. But the values are not copied into it. It
adopts the list you give it. It can also share one index of keys with many other
KeyedLists, which is what a table does with the column names of its rows.

Daffodil gives you its rows as KeyedLists when `itermode` is 'keyedlist', and from
`iter_klist()` and `iloc()`. Each one points at a row of the table.

How it differs from a dict:

- Writing to it writes to the list it adopted. If that list is a row of a Daf, the Daf
  changes. A dict row never changes the table.
- `values()` returns the adopted list itself, not a copy. Use `to_dict()` for an
  independent copy.
- Creating one is cheap if you give it an existing index. A dict copies every value in.
- Keys must be unique, and keys and values must be the same length.

Ways to create one:

- `KeyedList(keys, values)`: adopts the values list. No copy.
- `KeyedList(hd, values)`: the same, but shares an hd that you already have, a dict of
  each key and its position. This is the fastest, and it is what Daf does for its rows:
  each row shares the `hd` of the Daf. Then `row.hd is daf.hd`.
- `KeyedList(a_dict)`: copies the values out of the dict. Slower than the forms above.
- `KeyedList(keys, default=0)`: every key gets the same default value.
- `KeyedList(other_klist)`: shares the values list and copies the index.

Reading and writing:

- Read with `row['a']`, `row.get('a', default)`, `'a' in row` or `row.items()`. A list of
  keys, as in `row[['a', 'b']]`, returns a list of values and skips keys that are missing.
- Assigning to an existing key replaces the value in the list.
- Assigning to a new key adds the key, and adds the value to the end of the list. If the
  list is a row of a Daf, that row is now longer than the others.
- Deleting a key removes its value from the list.
- A KeyedList that shares an index copies it before it adds a key. So adding a key to
  one row does not give that key to the other rows.

Speed:

Looping over a table as KeyedLists can be faster than looping over dicts, even if you
never assign to a row. All the rows share one index, so no row builds its own. The gain
grows with the number of columns. In a test with 50,000 rows of 400 columns, reading
one field in a loop took 0.047 s with KeyedLists and 0.583 s with dicts. With 5 columns
the two were about even.

Because a KeyedList points into the table, assigning to it changes the table. If you
need a copy, use `to_dict()`, or loop with `iter_dict()` on the Daf.

Examples:
    >>> values = [1, 2, 3]
    >>> klist = KeyedList(['a', 'b', 'c'], values)
    >>> klist['b']
    2
    >>> klist['b'] = 20
    >>> values
    [1, 20, 3]
    >>> klist.to_dict()
    {'a': 1, 'b': 20, 'c': 3}
    >>> klist.get('z', 0)
    0

    A row of a Daf is a KeyedList that points at the row in the table:

    >>> from daffodil.daf import Daf
    >>> daf = Daf(cols=['x', 'y'], lol=[[1, 2], [3, 4]])
    >>> for row in daf.iter_klist():
    ...     row['y'] = 0
    >>> daf
    <BLANKLINE>
    | x  | y  |
    | -: | -: |
    |  1 |  0 |
    |  3 |  0 |
    <BLANKLINE>
    %% daf rows=2; cols=2; keyfield=''; name=''
    <BLANKLINE>

##### __getitem__(key: TKey | List[TKey]) -> Any

Get the value for a key, or a list of values for a list of keys.

Args:
    key: A key, or a list of keys.

Returns:
    The value. For a list of keys, a list of values in that order. A key of the list that is
    missing is skipped, so the list can be shorter.

Raises:
    KeyError: A single key is not found.
    ValueError: The key is not a list and cannot be hashed.

Examples:
    >>> klist = KeyedList(['a', 'b', 'c'], [1, 2, 3])
    >>> klist['b']
    2
    >>> klist[['c', 'a']]
    [3, 1]
    >>> klist[['a', 'zz']]
    [1]

##### __setitem__(key: TKey, value: Any) -> None

Set the value for a key, as `row[key] = value`.

A key that exists gets the new value in the list that this KeyedList points at. If
that list is a row of a Daf, the Daf changes. A key that is new is added at the end of
the keys, and its value at the end of the list, so a row of a Daf is then longer than
the others. The index is copied before it gets a new key, so other KeyedLists that
shared it do not get the key.

Args:
    key: The key.
    value: The value.

Examples:
    >>> row = [1, 2]
    >>> klist = KeyedList(['a', 'b'], row)
    >>> klist['b'] = 20
    >>> klist['c'] = 30
    >>> row
    [1, 20, 30]
    >>> list(klist)
    ['a', 'b', 'c']

##### __delitem__(key: TKey) -> None

Delete a key and its value, as `del row[key]`.

The value is removed from the list that this KeyedList points at, and the later
values move down. If that list is a row of a Daf, the row is then shorter than the
others. The index is rebuilt, and this KeyedList no longer shares it.

Args:
    key: The key to delete.

Raises:
    KeyError: The key is not found.

Examples:
    >>> row = [1, 2, 3]
    >>> klist = KeyedList(['a', 'b', 'c'], row)
    >>> del klist['b']
    >>> row, list(klist)
    ([1, 3], ['a', 'c'])

##### __len__() -> int

Get the number of values.

Returns:
    The number of values in the list that this KeyedList points at.

Examples:
    >>> len(KeyedList(['a', 'b'], [1, 2]))
    2

##### __iter__() -> Iterator[TKey]

Loop over the keys, as a dict does.

Returns:
    An iterator of the keys, in the order of their values.

Examples:
    >>> list(KeyedList(['a', 'b'], [1, 2]))
    ['a', 'b']

##### __contains__(key: object) -> bool

Test whether a key is one of the keys, as `key in row`.

Args:
    key: The key to look for.

Returns:
    True if the key is found.

Examples:
    >>> 'a' in KeyedList(['a', 'b'], [1, 2])
    True
    >>> 'z' in KeyedList(['a', 'b'], [1, 2])
    False

##### keys() -> KeysView[TKey]

Get the keys, in the order of their values.

This is a keys view of the index itself. When the index is shared with other
KeyedLists, as it is for the rows of a Daf, it is not a copy. For a list of the
keys, use `list(row.keys())`.

Returns:
    The keys.

Examples:
    >>> KeyedList(['a', 'b'], [1, 2]).keys()
    dict_keys(['a', 'b'])

##### set_values(new_values: List[Any]) -> None

Point this KeyedList at a different list of values.

This replaces the list that the KeyedList adopted. It does not write into the old
list. If the old list was a row of a Daf, the row is not changed, and this KeyedList
stops being a view of it. The new list is adopted, not copied. To change the values
in the old list, assign to the keys.

Args:
    new_values: The new list of values, one for each key.

Raises:
    TypeError: `new_values` is not a list. A tuple is not accepted.
    ValueError: The list does not have one value for each key.

Examples:
    >>> row = [1, 2]
    >>> klist = KeyedList(['a', 'b'], row)
    >>> klist.set_values([10, 20])
    >>> klist.values(), row
    ([10, 20], [1, 2])
    >>> klist.set_values([1])
    Traceback (most recent call last):
        ...
    ValueError: values length must match keys

##### values(astype: Callable | str | type | None = None) -> List[Any]

Get the values as a list.

With no `astype`, or with `list`, this is the list that the KeyedList adopted, not
a copy. If it is a row of a Daf, changing it changes the Daf. With any other
`astype` it is a new list, and the original is not changed. An empty cell, which is
NULL in daffodil, is not converted. It stays as the empty string.

Args:
    astype: How to convert each value. A type such as `int`, a function, or one of
        the names `'int'`, `'str'`, `'float'` and `'bool'`. None does not convert.

Returns:
    The values.

Raises:
    ValueError: `astype` is a name that is not supported, or a value cannot be
        converted by `int` or `float`. Another converter raises its own error.

Examples:
    >>> klist = KeyedList(['a', 'b'], ['1', '2'])
    >>> klist.values()
    ['1', '2']
    >>> klist.values(int)
    [1, 2]
    >>> klist.values('float')
    [1.0, 2.0]
    >>> KeyedList(['a', 'b'], ['1', '']).values(int)
    [1, '']
    >>> klist.values() is klist.values()
    True
    >>> klist.values(int) is klist.values(int)
    False
    >>> klist.values('date')
    Traceback (most recent call last):
        ...
    ValueError: astype not supported: date

##### items() -> Iterator[Tuple[TKey, Any]]

Loop over the (key, value) pairs, as `dict.items()` does.

This is an iterator, not a view. It can be used once. To keep the pairs, use
`list(row.items())`.

Returns:
    An iterator of (key, value) tuples.

Examples:
    >>> list(KeyedList(['a', 'b'], [1, 2]).items())
    [('a', 1), ('b', 2)]
    >>> pairs = KeyedList(['a'], [1]).items()
    >>> list(pairs), list(pairs)
    ([('a', 1)], [])

##### get(key: TKey, default: Any = None) -> Any

Get the value for a key, or a default if the key is not found.

Args:
    key: The key.
    default: What to return if the key is not found.

Returns:
    The value, or `default`.

Raises:
    TypeError: The key cannot be hashed, such as a list. Unlike `row[...]`, a list
        of keys is not accepted here.

Examples:
    >>> klist = KeyedList(['a'], [1])
    >>> klist.get('a'), klist.get('z'), klist.get('z', 0)
    (1, None, 0)

##### update(other: KeyedList | Dict[Any, Any]) -> None

Set many keys from a dict or from another KeyedList.

Each key of `other` is assigned as in `row[key] = value`. A key that exists gets
the new value in the list. A key that is new is added at the end of the keys and of
the list. If the list is a row of a Daf, that row is then longer than the others,
so use `update()` only with keys that exist. `other` is not changed.

Args:
    other: A dict or a KeyedList.

Examples:
    >>> klist = KeyedList(['a', 'b'], [1, 2])
    >>> klist.update({'b': 20, 'c': 30})
    >>> klist.to_dict()
    {'a': 1, 'b': 20, 'c': 30}

##### to_dict() -> Dict[TKey, Any]

Make a dict of the keys and values.

The dict is independent of this KeyedList. Later writes to the row do not change
it, and changing it does not change the row. The values are not copied deeply, so a
list or dict held in a cell is the same object in both.

Returns:
    A new dict.

Examples:
    >>> klist = KeyedList(['a', 'b'], [1, 2])
    >>> as_dict = klist.to_dict()
    >>> klist['a'] = 99
    >>> as_dict
    {'a': 1, 'b': 2}

##### __repr__() -> str

Show the keys and values as a dict.

Returns:
    The text of the dict.

Examples:
    >>> KeyedList(['a', 'b'], [1, 2])
    {'a': 1, 'b': 2}

##### __bool__() -> bool

return true if there is something in the values list.

##### to_json() -> str

Make a JSON string that holds the keys and the values.

The text has the marker `__KeyedList__`, the keys with their positions under `hd`,
and the `values`. Every value must be one that `json` can write. JSON keys are text,
so a key that is not a string, such as an int, comes back from `from_json()` as a
string.

Returns:
    The JSON text.

Raises:
    TypeError: A value cannot be written as JSON.

Examples:
    >>> KeyedList(['a', 'b'], [1, 'x']).to_json()
    '{"__KeyedList__": true, "hd": {"a": 0, "b": 1}, "values": [1, "x"]}'

##### from_json(json_str: str) -> KeyedList

Make a KeyedList from the text that `to_json()` writes.

Args:
    json_str: The JSON text.

Returns:
    The new KeyedList. It has its own list of values.

Raises:
    ValueError: The text is not JSON, or the JSON does not have the `__KeyedList__`
        marker.

Examples:
    >>> klist = KeyedList(['a', 'b'], [1, 'x'])
    >>> KeyedList.from_json(klist.to_json()).to_dict()
    {'a': 1, 'b': 'x'}
    >>> KeyedList.from_json('{"a": 1}')
    Traceback (most recent call last):
        ...
    ValueError: Invalid JSON string for KeyedList
