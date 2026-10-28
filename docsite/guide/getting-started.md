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
>>> d2
<BLANKLINE>
| id | qty |
| -: | --: |
| a1 |   3 |
| b2 |  10 |
<BLANKLINE>
%% daf rows=2; cols=2; keyfield=''; name=''
<BLANKLINE>
>>> d2.columns()
['id', 'qty']
```

From a CSV file, `Daf.from_csv('file.csv')` reads the header line as the column names. The
values are read as text. See [Types and conversion](types.md) to convert them.

## Look at it

```pycon
>>> len(daf), daf.shape(), daf.columns()
(2, (2, 3), ['id', 'item', 'qty'])
>>> daf.keys()
['a1', 'b2']
>>> bool(Daf())
False
```

`print(daf)` and `daf.to_md()` give a Markdown table. See [Markdown reports](markdown.md).

## Read values

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

## Add rows

```pycon
>>> daf.append({'id': 'c3', 'item': 'pad', 'qty': 7})
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
>>> daf.col('id')
['a1', 'b2', 'c3', 'd4']
```

A dict is placed by column name, and a list is in column order. `append()` returns the Daf,
so these examples assign it to `_`, to keep the output short. See
[Building tables](building-tables.md) for the fast ways to build a table in a loop.

## Change values

```pycon
>>> daf[0, 'qty'] = 4
>>> daf.select_record('a1')['qty']
4
```

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

## Get the data out

```pycon
>>> out.to_lod()[0]
{'name': 'pen', 'price_cents': 150}
>>> out.lol
[['pen', 150], ['ink', 400]]
```

- `to_csv_file(path)` writes a CSV file.
- `to_pandas_df()` and `to_numpy()` hand the data to pandas or NumPy, for heavy numeric work.
