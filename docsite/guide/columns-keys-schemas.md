# Columns, keys and schemas

## Column names

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
>>> n = Daf.from_csv_buff('1,2\n3,4\n', noheader=True)
>>> n.columns(), n.lol
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

### A key column with repeated values

`append()` does not check the key, which keeps it fast. So data with repeated keys can be
read in. To find them before you rely on the key:

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

## Schemas

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
