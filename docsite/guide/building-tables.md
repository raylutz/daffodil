# Building tables

Daffodil is built for tables that grow a row at a time. Appending a row adds it to a Python
list, so it is cheap. pandas has no cheap way to add a row, which is why code that uses pandas
often collects the rows in a list of dicts first and converts them at the end.

## append()

`append()` adds rows to the end of the table. It takes a dict, a list of values, a list of dicts,
a list of lists with `lol=`, a KeyedList, or another Daf. It changes the table and returns it,
so at the prompt it shows the whole table. Here each call is followed by a read of the row it
added, to keep the output short. In a program, `daf.append(row)` on its own line adds the row,
and nothing is shown.

A dict is placed by column name, in any order:

```pycon
>>> from daffodil.daf import Daf
>>> daf = Daf(cols=['id', 'qty'], keyfield='id')
>>> daf.append({'qty': 3, 'id': 'a1'}).select_record('a1')
{'id': 'a1', 'qty': 3}
```

A dict that lacks a column gets NULL there, the empty string. A key that is not a column is
dropped:

```pycon
>>> daf.append({'id': 'b2', 'color': 'red'}).select_record('b2')
{'id': 'b2', 'qty': ''}
```

A list goes in column order. A list shorter than the columns is padded with NULL:

```pycon
>>> daf.append(['c3', 10]).select_record('c3')
{'id': 'c3', 'qty': 10}
>>> daf.append(['d4']).select_record('d4')
{'id': 'd4', 'qty': ''}
```

A list longer than the columns raises ValueError, since a value would have no column:

```pycon
>>> daf.append(['e5', 1, 'extra'])
Traceback (most recent call last):
...
ValueError: append(): the list has 3 values for 2 columns.
```

Several rows at once: a list of dicts, or a list of lists with `lol=`:

```pycon
>>> daf.append([{'id': 'f6', 'qty': 7}, {'id': 'g7', 'qty': 1}]).keys()
['a1', 'b2', 'c3', 'd4', 'f6', 'g7']
>>> daf.append(lol=[['h8', 2], ['i9', 5]]).keys()
['a1', 'b2', 'c3', 'd4', 'f6', 'g7', 'h8', 'i9']
>>> daf
<BLANKLINE>
| id | qty |
| -: | --: |
| a1 |   3 |
| b2 |     |
| c3 |  10 |
| d4 |     |
| f6 |   7 |
| g7 |   1 |
| h8 |   2 |
| i9 |   5 |
<BLANKLINE>
%% daf rows=8; cols=2; keyfield='id'; name=''
<BLANKLINE>
```

The table gets its own copy of each row, so you can change or reuse your list or dict
afterwards:

```pycon
>>> row = ['j10', 4]
>>> daf.append(row).select_record('j10')
{'id': 'j10', 'qty': 4}
>>> row[1] = 99
>>> daf.select_record('j10')
{'id': 'j10', 'qty': 4}
```

The key is not checked, which keeps appending fast. A key that is already there is added
again, as a second row. A lookup by that key then finds the last one:

```pycon
>>> stock = Daf(cols=['id', 'qty'], lol=[['a1', 3], ['b2', 10]], keyfield='id')
>>> stock.append({'id': 'a1', 'qty': 30})
<BLANKLINE>
| id | qty |
| -: | --: |
| a1 |   3 |
| b2 |  10 |
| a1 |  30 |
<BLANKLINE>
%% daf rows=3; cols=2; keyfield='id'; name=''
<BLANKLINE>
>>> stock.select_record('a1')
{'id': 'a1', 'qty': 30}
```

With `respect_kd=True`, the row that has the same key is replaced instead:

```pycon
>>> stock = Daf(cols=['id', 'qty'], lol=[['a1', 3], ['b2', 10]], keyfield='id')
>>> stock.append({'id': 'a1', 'qty': 30}, respect_kd=True)
<BLANKLINE>
| id | qty |
| -: | --: |
| a1 |  30 |
| b2 |  10 |
<BLANKLINE>
%% daf rows=2; cols=2; keyfield='id'; name=''
<BLANKLINE>
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
>>> def read_marks(readings):
...     marks = Daf(cols=['ballot', 'contest', 'x', 'y'])
...     for ballot, contest, x, y in readings:
...         row = marks.default_record(astype=KeyedList)
...         row['y'], row['x'] = y, x                   # by name, in any order
...         row['ballot'], row['contest'] = ballot, contest
...         marks.append(row)
...     return marks
>>> read_marks([('b1', 'mayor', 10, 20), ('b2', 'mayor', 11, 21)])
<BLANKLINE>
| ballot | contest | x  | y  |
| -----: | ------: | -: | -: |
|     b1 |   mayor | 10 | 20 |
|     b2 |   mayor | 11 | 21 |
<BLANKLINE>
%% daf rows=2; cols=4; keyfield=''; name=''
<BLANKLINE>
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
>>> def read_fast(readings):
...     fast = Daf(cols=['ballot', 'x'])
...     for ballot, x in readings:
...         row = fast.default_record(astype=KeyedList)
...         row['ballot'], row['x'] = ballot, x
...         fast.append(row, fast=True)
...     return fast
>>> fast = read_fast([('b1', 10), ('b2', 11)])
>>> fast
<BLANKLINE>
| ballot | x  |
| -----: | -: |
|     b1 | 10 |
|     b2 | 11 |
<BLANKLINE>
%% daf rows=2; cols=2; keyfield=''; name=''
<BLANKLINE>
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
