# Booleans

In daffodil a boolean is the int 0 or 1. It is never the word True or False. This is a standard
for every daffodil table, in memory, in a CSV file and in SQLite.

## The rule

- A true value is 1 and a false value is 0.
- Converted to `bool`, a cell becomes the int 0 or 1.
- Converted to `str`, a cell becomes `'0'` or `'1'`. A Python True is made an int first, then text.
- Rows you build hold 0 and 1, not True and False.

## Why

Each system spells the words its own way. JSON writes `true` and `false`. Python writes `True`
and `False`. SQLite and Excel use `TRUE` and `FALSE`. Data that holds any of them must be
converted each time it moves from one system to the next, and a missed conversion is a quiet bug.

0 and 1 are the same in all of them. So a table can be written out, read back and compared with
its source as text, with no special case for bools.

A dict or list cell is written as PYON, the Python form of the value. With 0 and 1 in it, that
text is also readable by SQLite's JSON functions. With True in it, it is not:

```pycon
>>> import sqlite3
>>> conn = sqlite3.connect(':memory:')
>>> conn.execute("SELECT json_extract(?, '$.ok')", ("{'ok': 1}",)).fetchone()
(1,)
>>> conn.execute("SELECT json_extract(?, '$.ok')", ("{'ok': True}",)).fetchone()
Traceback (most recent call last):
...
sqlite3.OperationalError: malformed JSON
```

## It works as a boolean

Python treats 0 as false and 1 as true. Code that tests a flag works the same with either:

```pycon
>>> from daffodil.daf import Daf
>>> daf = Daf(cols=['id', 'ok'], lol=[['a1', 1], ['b2', 0], ['c3', 1]])
>>> [row['id'] for row in daf if row['ok']]
['a1', 'c3']
```

And a sum of a flag column counts the true values:

```pycon
>>> sum(daf.col('ok'))
2
```

So there is no gain in storing True and False, as long as the code treats the column as a flag.

## Converting

`apply_dtypes()` with dtype `bool` reads the usual spellings from other systems and gives 0 or 1:

| Gives | Text read |
| :---- | :-------- |
| 0 | empty, `0`, `0.0`, `false`, `no`, `n`, `f`, `off` |
| 1 | `1`, `1.0`, `true`, `yes`, `y`, `t`, `on` |

- Each word is read in lower case, capitalized or upper case, as `no`, `No` and `NO`.
- Other text stays as it is, with no error. That includes mixed case such as `tRuE`, a word with
  a space around it, and other numbers such as `'2'`.
- A value that is not text goes by its truth, so 2 gives 1. None gives 0.
- An empty cell gives 0, unlike in an `int` column, where it stays NULL.

```pycon
>>> flags = Daf(cols=['ok'], lol=[['True'], ['false'], ['1'], ['NO'], ['Off'], ['y'], [''], ['tRuE'], ['2'], [None], [2]])
>>> flags.apply_dtypes(dtypes={'ok': bool})
<BLANKLINE>
|  ok  |
| ---: |
|    1 |
|    0 |
|    1 |
|    0 |
|    0 |
|  ... |
|    0 |
| tRuE |
|    2 |
|    0 |
|    1 |
<BLANKLINE>
%% daf rows=11; cols=1; keyfield=''; name=''
<BLANKLINE>
>>> flags.col('ok')
[1, 0, 1, 0, 0, 1, 0, 'tRuE', '2', 0, 1]
```

Text that stays as it is can be found with `check_schema()`. See below.

To turn a word into the text `'0'` or `'1'`, convert to `bool` first, then to `str`:

```pycon
>>> words = Daf(cols=['ok'], lol=[['YES'], ['off']])
>>> words.apply_dtypes(dtypes={'ok': bool})
<BLANKLINE>
| ok |
| -: |
|  1 |
|  0 |
<BLANKLINE>
%% daf rows=2; cols=1; keyfield=''; name=''
<BLANKLINE>
>>> words.apply_dtypes(dtypes={'ok': str}, from_str=False)
<BLANKLINE>
| ok |
| -: |
|  1 |
|  0 |
<BLANKLINE>
%% daf rows=2; cols=1; keyfield=''; name=''
<BLANKLINE>
>>> words.col('ok')
['1', '0']
```

Converted to `str`, as with `from_str=False`, a bool becomes `'0'` or `'1'`. So it matches the text
of a file that holds 0 and 1:

```pycon
>>> typed = Daf(cols=['ok', 'n'], lol=[[True, 3], [False, 4]])
>>> typed.apply_dtypes(dtypes={'ok': str, 'n': str}, from_str=False)
<BLANKLINE>
| ok | n  |
| -: | -: |
|  1 |  3 |
|  0 |  4 |
<BLANKLINE>
%% daf rows=2; cols=2; keyfield=''; name=''
<BLANKLINE>
>>> typed.lol
[['1', '3'], ['0', '4']]
```

`normalize_val()` turns the bools in a value into 0 and 1, at any depth. It also turns None into
NULL. Use it on a dict or list before it goes into a cell:

```pycon
>>> from daffodil.lib.daf_utils import normalize_val
>>> normalize_val({'ok': True, 'votes': [False, True], 'note': None})
{'ok': 1, 'votes': [0, 1], 'note': ''}
```

## Normalize once, when the row is built

Put each value into daffodil's form once, when the row is built. Then keep it that form in
every table. Daffodil's form is:

- a boolean is 0 or 1;
- a missing value, None included, is NULL, the empty string.

`normalize_val()` does both, at any depth in a dict, list or tuple. Call it on a dict or list
before it goes into a cell. A cell in this form is already PYON and JSON at once, so it is never
converted again. Python reads its text, SQLite's JSON functions read it, and a CSV file holds it
as it is:

```pycon
>>> cell = normalize_val({'ok': True, 'votes': [False, True], 'note': None})
>>> cell
{'ok': 1, 'votes': [0, 1], 'note': ''}
>>> text = str(cell)
>>> conn.execute("SELECT json_extract(?, '$.ok'), json_extract(?, '$.note')", (text, text)).fetchone()
(1, '')
>>> cells = Daf(cols=['id', 'cv'], lol=[['a1', cell]])
>>> back = Daf.from_csv_buff(cells.to_csv_buff()).apply_dtypes(dtypes={'id': str, 'cv': dict})
>>> back.lol
[['a1', {'ok': 1, 'votes': [0, 1], 'note': ''}]]
```

Daffodil does not normalize again later, as on each write or each `apply_dtypes()`. That would cost
a pass over every dict and list cell, every time. Done once, up front, it costs nothing more.

## What daffodil does not do

Daffodil does not check what you store. A dtype only describes a column, and converts nothing.
A True put in a cell stays True until the column is converted, and `to_csv_file()` writes it as
the word. `apply_dtypes()` with dtype `bool` makes it 1:

```pycon
>>> raw = Daf(cols=['id', 'ok'], lol=[['a1', True]])
>>> print(raw.to_csv_buff())
id,ok
a1,True
>>> print(raw.apply_dtypes(dtypes={'id': str, 'ok': bool}).to_csv_buff())
id,ok
a1,1
```

A dict or list cell is not changed by `apply_dtypes()`, so a True or None inside it stays as it
is. Normalize it when the row is built, as above.

Checking every cell would slow down every table, so daffodil leaves it to the code that builds
the rows. Store 0 and 1.

## Type checkers

To a type checker, a flag column holds ints. mypy accepts True where an int is expected, but it
rejects an int where a bool is expected. So annotate a flag from a table as `int`:

```python
def is_counted(ok: int) -> bool:
    return ok == 1
```

An `int` annotation also allows 2 or -1, which are not flags. mypy cannot tell a flag from a
count. Daffodil accepts this small risk for now, in return for data that every system reads the
same way.

To check a table, give the flag columns dtype `bool` and call `check_schema()`. It raises for a
cell such as 2 or `'x'`. It lets True and False pass, since Python counts them as ints.
