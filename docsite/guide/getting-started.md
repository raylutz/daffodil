# Getting started

## Install

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

## Make a table

A Daf is a list of rows, each a plain Python list, and a dict of the column names. Give it the
column names and the rows:

```pycon
>>> from daffodil.daf import Daf
>>> daf = Daf(cols=['id', 'item', 'qty'], lol=[['a1', 'pen', 3], ['b2', 'ink', 10]], keyfield='id')
>>> daf
<BLANKLINE>
| id | item | qty |
| -: | ---: | --: |
| a1 |  pen |   3 |
| b2 |  ink |  10 |
<BLANKLINE>
%% daf rows=2; cols=3; keyfield='id'; name=''
<BLANKLINE>
```

The prompt shows a Daf as a Markdown table, then a summary line, which starts with `%%`.

`keyfield` names the column whose values identify the rows, here `id`. It lets you find a row
by its key, as `daf['b2']`. It is optional.

From a list of dicts, the columns are the keys, in the order first seen. A dict that lacks a
key gets NULL there, the empty string:

```pycon
>>> prices = Daf.from_lod([{'id': 'a1', 'price': 1.5}, {'id': 'b2', 'price': 4.0, 'note': 'sale'}])
>>> prices
<BLANKLINE>
| id | price | note |
| -: | ----: | ---: |
| a1 |   1.5 |      |
| b2 |   4.0 | sale |
<BLANKLINE>
%% daf rows=2; cols=3; keyfield=''; name=''
<BLANKLINE>
```

From a CSV file, `Daf.from_csv('file.csv')` reads the header line as the column names. The
values are read as text. See [Types and conversion](types.md) to convert them.

## Look at it

```pycon
>>> len(daf)                    # the number of rows
2
>>> daf.shape()                 # rows and columns
(2, 3)
>>> daf.columns()
['id', 'item', 'qty']
>>> daf.keys()                  # the values of the keyfield, in row order
['a1', 'b2']
```

A Daf with rows is true, and one with no rows is false, even if it has columns. So `if daf:`
asks whether there are any rows:

```pycon
>>> bool(daf)
True
>>> empty = Daf(cols=['id', 'item', 'qty'])
>>> bool(empty), len(empty)
(False, 0)
```

`print(daf)` shows the same table as the prompt. For a report, `daf.to_md()` gives the table
as text, with every row. See [Markdown reports](markdown.md).

## Read values

Select with brackets, as `daf[rows, cols]`. The first part picks rows and the second picks
columns. Leave out the second part, or give `:`, for all the columns.

- A number is a position, from 0. A string is a key for rows, and a column name for columns.
- A slice, as `0:2`, and a list, as `['a1', 'b2']`, select several.

```pycon
>>> daf[1]                      # the row at position 1
<BLANKLINE>
| id | item | qty |
| -: | ---: | --: |
| b2 |  ink |  10 |
<BLANKLINE>
%% daf rows=1; cols=3; keyfield='id'; name=''
<BLANKLINE>
>>> daf['a1']                   # the row with key 'a1'
<BLANKLINE>
| id | item | qty |
| -: | ---: | --: |
| a1 |  pen |   3 |
<BLANKLINE>
%% daf rows=1; cols=3; keyfield='id'; name=''
<BLANKLINE>
>>> daf[:, 'qty']               # all rows, the qty column
<BLANKLINE>
| qty |
| --: |
|   3 |
|  10 |
<BLANKLINE>
%% daf rows=2; cols=1; keyfield=''; name=''
<BLANKLINE>
```

A selection is itself a Daf. Turn it into plain values with `to_value()`, `to_list()` or
`to_dict()`:

```pycon
>>> daf['b2', 'qty'].to_value()     # one cell
10
>>> daf[:, 'qty'].to_list()         # one column
[3, 10]
>>> daf['a1'].to_dict()             # one row
{'id': 'a1', 'item': 'pen', 'qty': 3}
```

Some methods give plain values directly, without making a Daf first. They are faster when that
is all you need:

```pycon
>>> daf.select_record('b2')
{'id': 'b2', 'item': 'ink', 'qty': 10}
>>> daf.col('qty')
[3, 10]
```

A selection of whole rows shares those rows with the table. Nothing is copied, so it is fast,
and a change made through the selection shows in the table:

```pycon
>>> first = daf[0]
>>> first[0, 'qty'] = 30
>>> daf.select_record('a1')
{'id': 'a1', 'item': 'pen', 'qty': 30}
>>> first[0, 'qty'] = 3             # back to 3, for the examples below
```

A selection of some columns builds new rows, so it shares nothing. Use `copy('editable')` for
rows of your own. See [Selecting and indexing](selecting.md) for every way to select.

## Add rows

A dict is placed by column name, in any order. A list goes in column order. `append()` adds
to the table and returns it, so the prompt shows the table:

```pycon
>>> daf.append({'qty': 7, 'id': 'c3', 'item': 'pad'})
<BLANKLINE>
| id | item | qty |
| -: | ---: | --: |
| a1 |  pen |   3 |
| b2 |  ink |  10 |
| c3 |  pad |   7 |
<BLANKLINE>
%% daf rows=3; cols=3; keyfield='id'; name=''
<BLANKLINE>
>>> daf.append(['d4', 'cap', 1])
<BLANKLINE>
| id | item | qty |
| -: | ---: | --: |
| a1 |  pen |   3 |
| b2 |  ink |  10 |
| c3 |  pad |   7 |
| d4 |  cap |   1 |
<BLANKLINE>
%% daf rows=4; cols=3; keyfield='id'; name=''
<BLANKLINE>
```

In a program, `daf.append(row)` on its own line adds the row, and nothing is shown. See
[Building tables](building-tables.md) for the fast ways to build a table in a loop.

## Change values

Assign to a selection, with the same brackets: one cell, a column, or a row.

```pycon
>>> daf['a1', 'qty'] = 4                    # one cell, by key and column
>>> daf[:, 'item'] = ['pens', 'ink', 'pads', 'caps']    # a column, from a list
>>> daf['d4'] = ['d4', 'cap', 2]            # a row, by key
>>> daf
<BLANKLINE>
| id | item | qty |
| -: | ---: | --: |
| a1 | pens |   4 |
| b2 |  ink |  10 |
| c3 | pads |   7 |
| d4 |  cap |   2 |
<BLANKLINE>
%% daf rows=4; cols=3; keyfield='id'; name=''
<BLANKLINE>
```

A list for a column has one value for each row, in row order. A list for a row has one value
for each column. To change a few cells of one row by name, `update_record_krow('a1', {'qty': 5})`
is clearer.

## The usual pattern

Read a file, go through it row by row, and build a new table. This is where Daffodil is
strongest: appending a row is cheap, and nothing is converted.

In the loop, each `row` is a new dict. Assigning a new value to it does not change the
source table. To change values in place, use `update_record_krow()`, `apply_in_place()`, or
`iter_klist()`, whose rows write through to the table.

```pycon
>>> source = Daf.from_lod([{'name': 'pen', 'price': '1.50'}, {'name': 'ink', 'price': '4.00'}])
>>> source
<BLANKLINE>
| name | price |
| ---: | ----: |
|  pen |  1.50 |
|  ink |  4.00 |
<BLANKLINE>
%% daf rows=2; cols=2; keyfield=''; name=''
<BLANKLINE>
>>> def to_cents_daf(source):
...     out = Daf(cols=['name', 'price_cents'])
...     for row in source:
...         out.append({'name': row['name'], 'price_cents': round(float(row['price']) * 100)})
...     return out
>>> to_cents_daf(source)
<BLANKLINE>
| name | price_cents |
| ---: | ----------: |
|  pen |         150 |
|  ink |         400 |
<BLANKLINE>
%% daf rows=2; cols=2; keyfield=''; name=''
<BLANKLINE>
```

The same, with `apply()`, which calls a function for each row and builds the new table:

```pycon
>>> def to_cents(row):
...     return {'name': row['name'], 'price_cents': round(float(row['price']) * 100)}
>>> source.apply(to_cents)
<BLANKLINE>
| name | price_cents |
| ---: | ----------: |
|  pen |         150 |
|  ink |         400 |
<BLANKLINE>
%% daf rows=2; cols=2; keyfield=''; name=''
<BLANKLINE>
```

## Get the data out

```pycon
>>> out = source.apply(to_cents)
>>> out.to_lod()                # a list of dicts
[{'name': 'pen', 'price_cents': 150}, {'name': 'ink', 'price_cents': 400}]
>>> out.lol                     # the rows themselves
[['pen', 150], ['ink', 400]]
```

- `to_csv_file(path)` writes a CSV file.
- `to_pandas_df()` and `to_numpy()` hand the data to pandas or NumPy, for heavy numeric work.
