# Getting started

## Install

```bash
pip install daffodil
```

Daffodil needs Python 3.10 or later. It is pure Python. pandas and NumPy are needed only to
convert to and from them.

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
>>> _ = daf.append({'id': 'c3', 'item': 'pad', 'qty': 7})
>>> _ = daf.append(['d4', 'cap', 1])
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

## Get the data out

```pycon
>>> out.to_lod()[0]
{'name': 'pen', 'price_cents': 150}
>>> out.lol
[['pen', 150], ['ink', 400]]
```

- `to_csv_file(path)` writes a CSV file.
- `to_pandas_df()` and `to_numpy()` hand the data to pandas or NumPy, for heavy numeric work.
