# Selecting and indexing

## Square brackets

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

An integer in `[]` is always a position. To reach a key or a column name that is an integer,
use `select_krows()` and `select_kcols()`.

## Getting values out

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

## Selecting by condition

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

## Setting values

The same selectors set values. A single value fills the selection. A list fills it in order.
A dict given for a row is placed by column name, and a column it lacks becomes NULL.

```pycon
>>> daf[0, 'a'] = 100
>>> daf[1] = ['r2', 40, 50, 60]
>>> daf[:, 'c'] = 0
>>> daf.lol[:2]
[['r1', 100, 2, 0], ['r2', 40, 50, 0]]
```

## Do selections copy the rows?

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
| `groupby_cols`, `group_where`, `copy()` | shared |
| a selection of columns: `daf[:, cols]`, `select_cols`, `select_kcols`, `select_icols` | new |
| `groupby`, `multi_groupby` | new |

To change a selection without changing the original, copy it first with
`copy('editable')`, which gives each row its own list. Adding a column to a selection, as with
`insert_col()`, copies the shared rows first, so the original keeps its shape.
