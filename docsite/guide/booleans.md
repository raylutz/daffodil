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

`apply_dtypes()` with dtype `bool` reads the usual words, in lower case, capitalized or upper
case, and gives 0 or 1. See [Types and conversion](types.md#bool-columns) for the full list.

```pycon
>>> flags = Daf(cols=['ok'], lol=[['True'], ['false'], ['1'], ['NO']])
>>> _ = flags.apply_dtypes(dtypes={'ok': bool})
>>> flags.col('ok')
[1, 0, 1, 0]
```

Converted to `str`, as with `from_str=False`, a bool becomes `'0'` or `'1'`. So it matches the text
of a file that holds 0 and 1:

```pycon
>>> typed = Daf(cols=['ok', 'n'], lol=[[True, 3], [False, 4]])
>>> _ = typed.apply_dtypes(dtypes={'ok': str, 'n': str}, from_str=False)
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

## What daffodil does not do

Daffodil does not check what you store. A True put in a cell stays True until the column is
converted. `to_csv_file()` then writes it as the word:

```pycon
>>> raw = Daf(cols=['id', 'ok'], lol=[['a1', True]])
>>> print(raw.to_csv_buff())
id,ok
a1,True
```

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
