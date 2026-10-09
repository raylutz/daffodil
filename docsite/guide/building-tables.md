# Building tables

Daffodil is built for tables that grow a row at a time. Appending a row adds it to a Python
list, so it is cheap. pandas has no cheap way to add a row, which is why code that uses pandas
often collects the rows in a list of dicts first and converts them at the end.

## append()

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

## Rows to fill in: default_record()

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

## append(fast=True)

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

## from_lod()

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

## Other ways to grow a table

- `daf.append(other_daf)` adds the rows of a Daf with the same columns. So does `concat()`.
- `insert_irow()` inserts a row at a position.
- `extend()` adds several rows, as `append()` does with a list of dicts or `lol=`.
