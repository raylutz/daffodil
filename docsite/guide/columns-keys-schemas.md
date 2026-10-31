# Columns, keys and schemas

## Column names

The column names are kept once, in the header dict `hd`, which maps each name to its
position. The rows hold only their cells.

```pycon
>>> from daffodil.daf import Daf
>>> daf = Daf(cols=['id', 'qty'], lol=[['a1', 3], ['b2', 10]])
>>> daf
<BLANKLINE>
| id | qty |
| -: | --: |
| a1 |   3 |
| b2 |  10 |
<BLANKLINE>
%% daf rows=2; cols=2; keyfield=''; name=''
<BLANKLINE>
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
>>> raw
<BLANKLINE>
| A  | B  |
| -: | -: |
|  1 |  2 |
|  3 |  4 |
<BLANKLINE>
%% daf rows=2; cols=2; keyfield=''; name=''
<BLANKLINE>
>>> raw.columns()
[]
>>> raw.set_cols()
<BLANKLINE>
| A  | B  |
| -: | -: |
|  1 |  2 |
|  3 |  4 |
<BLANKLINE>
%% daf rows=2; cols=2; keyfield=''; name=''
<BLANKLINE>
>>> raw.columns()
['A', 'B']
>>> raw.set_cols(['x', 'y'])
<BLANKLINE>
| x  | y  |
| -: | -: |
|  1 |  2 |
|  3 |  4 |
<BLANKLINE>
%% daf rows=2; cols=2; keyfield=''; name=''
<BLANKLINE>
>>> raw.col('y')
[2, 4]
```

## Column names from a CSV file

`Daf.from_csv()` takes the column names from the first line. A blank name becomes `Unnamed`
followed by its position. A repeated name gets an underscore and its position.
`from_csv_buff()` does the same for text or bytes in memory:

```pycon
>>> Daf.from_csv_buff('id,,qty,qty\n1,2,3,4\n').columns()
['id', 'Unnamed1', 'qty', 'qty_3']
```

With `noheader=True`, the first line is data, and the Daf has no names until you set them:

```pycon
>>> n_daf = Daf.from_csv_buff('1,2\n3,4\n', noheader=True)
>>> n_daf
<BLANKLINE>
| A  | B  |
| -: | -: |
|  1 |  2 |
|  3 |  4 |
<BLANKLINE>
%% daf rows=2; cols=2; keyfield=''; name=''
<BLANKLINE>
>>> n_daf.columns(), n_daf.lol
([], [['1', '2'], ['3', '4']])
```

The values from a CSV file are text. See [Types and conversion](types.md).

## Rows by position

Row positions belong to the table, not to the row. After a sort or a selection, position 0
is whatever row is first. To find a row whatever its position, use a keyfield.

## The keyfield

The keyfield is a column whose values identify the rows, like a primary key. It stays in the
table as an ordinary column. Daffodil builds a key dict, `kd`, from it when a lookup first
needs it, so a table that is only appended to never builds one.

```pycon
>>> daf = Daf(cols=['id', 'qty'], lol=[['a1', 3], ['b2', 10]], keyfield='id')
>>> daf
<BLANKLINE>
| id | qty |
| -: | --: |
| a1 |   3 |
| b2 |  10 |
<BLANKLINE>
%% daf rows=2; cols=2; keyfield='id'; name=''
<BLANKLINE>
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
>>> # Create a table of sales by store and item, keyed by both columns.
>>> sales = Daf(cols=['store', 'item', 'qty'], lol=[['s1', 'apples', 40], ['s1', 'pears', 31]],
...             keyfield=('store', 'item'))
>>> sales
<BLANKLINE>
| store |  item  | qty |
| ----: | -----: | --: |
|    s1 | apples |  40 |
|    s1 |  pears |  31 |
<BLANKLINE>
%% daf rows=2; cols=3; keyfield='('store', 'item')'; name=''
<BLANKLINE>
>>> # A key of two columns is a tuple.
>>> sales.select_record(('s1', 'pears'))
{'store': 's1', 'item': 'pears', 'qty': 31}
```

### A key column with repeated values

`append()` does not check the key, which keeps it fast. So data with repeated keys can be
read in. Then the key index holds each key once, and points to the last row with it:

- A lookup of the key, as `select_record()` or `daf[key]`, finds the last row.
- `keys()` gives each key once, in the order of its first row.
- `join()` matches each row with the last row of that key in the other table.

```pycon
>>> dup = Daf(cols=['id', 'qty'], lol=[['a1', 3], ['b2', 10], ['a1', 5]], keyfield='id')
>>> dup
<BLANKLINE>
| id | qty |
| -: | --: |
| a1 |   3 |
| b2 |  10 |
| a1 |   5 |
<BLANKLINE>
%% daf rows=3; cols=2; keyfield='id'; name=''
<BLANKLINE>
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
>>> daf
<BLANKLINE>
| id | qty |
| -: | --: |
| a1 |   3 |
| b2 |  10 |
| a1 |   5 |
<BLANKLINE>
%% daf rows=3; cols=2; keyfield=''; name=''
<BLANKLINE>
>>> ids = daf.col('id')
>>> len(ids), len(set(ids))
(3, 2)
>>> daf.valuecounts_for_colname('id')
{'a1': 2, 'b2': 1}
```

If no column can be the key, add one that numbers the rows with `insert_idx_col()`.

## Schemas

A schema declares the columns, their intended types and their defaults in one place. Write it
as a class with type annotations, decorated with `@schemaclass`:

```pycon
>>> from daffodil.lib.schemaclass import schemaclass
>>> @schemaclass
... class Order:
...     __keyfield__ = 'order_id'
...     order_id: str = ''
...     item: str = ''
...     qty: int = 1
>>> order_daf = Daf(schema=Order)
>>> order_daf
<BLANKLINE>
<BLANKLINE>
%% daf rows=0; cols=0; keyfield='order_id'; name=''
<BLANKLINE>
>>> order_daf.columns(), order_daf.keyfield
(['order_id', 'item', 'qty'], 'order_id')
>>> order_daf.dtypes
{'order_id': <class 'str'>, 'item': <class 'str'>, 'qty': <class 'int'>}
```

A schema fills only what the Daf does not already have: the columns, the dtypes and the
keyfield. It does not check or convert the cells.

`default_record()` makes a new record with the defaults, for the Daf's own columns:

```pycon
>>> order_daf.default_record()
{'order_id': '', 'item': '', 'qty': 1}
>>> Daf(cols=['qty', 'order_id'], schema=Order).default_record()
{'qty': 1, 'order_id': ''}
```

A Daf that keeps only some of the schema's columns gets records that fit it. See
[Building tables](building-tables.md) for `default_record(astype=KeyedList)`, which is the
fast way to build rows in a loop.

A schema can also be a Daf with one row for each column, with the columns `Name`, `dtype`
and `Default`.
