# Types and conversion

A cell can hold any Python object: a number, text, a list, a dict, a set, or another Daf.
Nothing limits what a cell holds, and nothing is converted unless you ask.

## dtypes

`dtypes` is a dict of column name to type. It says how to convert the columns when you ask,
as when reading a CSV file. It is not a schema that checks or forces the cells, and a Daf
built with `dtypes` keeps its cells as they are.

Three steps are separate:

| Step | What it does |
| :--- | :----------- |
| set `dtypes`, or call `set_dtypes()` | declares the types. Nothing is converted |
| `apply_dtypes()` | tries to convert the cells |
| `check_schema()` | finds the cells that do not fit their dtypes, and raises |

Conversion is permissive. Text that is not a number stays text, with no error. So after
`apply_dtypes()`, an `int` column can still hold some text. Call `check_schema()` when you
need to know that every cell fits.

## Converting text from a CSV file

A CSV file gives text. `apply_dtypes()` converts the columns in place, and returns the Daf:

```pycon
>>> from daffodil.daf import Daf
>>> daf = Daf.from_csv_buff('id,qty,price,tags\na1,3,1.5,"[1, 2]"\nb2,,2.0,[]\n')
>>> daf
<BLANKLINE>
| id | qty | price |  tags  |
| -: | --: | ----: | -----: |
| a1 |   3 |   1.5 | [1, 2] |
| b2 |     |   2.0 |     [] |
<BLANKLINE>
%% daf rows=2; cols=4; keyfield=''; name=''
<BLANKLINE>
>>> daf.lol
[['a1', '3', '1.5', '[1, 2]'], ['b2', '', '2.0', '[]']]
>>> daf.apply_dtypes(dtypes={'id': str, 'qty': int, 'price': float, 'tags': list})
<BLANKLINE>
| id | qty | price |  tags  |
| -: | --: | ----: | -----: |
| a1 |   3 |   1.5 | [1, 2] |
| b2 |     |   2.0 |     [] |
<BLANKLINE>
%% daf rows=2; cols=4; keyfield=''; name=''
<BLANKLINE>
>>> daf.lol
[['a1', 3, 1.5, [1, 2]], ['b2', '', 2.0, []]]
```

- An empty cell is NULL, the empty string, and stays NULL, except in a `bool` column. See
  Missing values below.
- A list or dict column is read back from its text, which may be JSON or PYON.
- Text that does not make an `int` or a `float` is kept as it is.
- `cols=` converts only the named columns. Converting a column costs time, so convert only
  the columns you use.

```pycon
>>> e_daf = Daf.from_csv_buff('id,qty,note\na1,3,x\n')
>>> e_daf
<BLANKLINE>
| id | qty | note |
| -: | --: | ---: |
| a1 |   3 |    x |
<BLANKLINE>
%% daf rows=1; cols=3; keyfield=''; name=''
<BLANKLINE>
>>> e_daf.apply_dtypes(dtypes={'id': str, 'qty': int, 'note': str}, cols=['qty'])
<BLANKLINE>
| id | qty | note |
| -: | --: | ---: |
| a1 |   3 |    x |
<BLANKLINE>
%% daf rows=1; cols=3; keyfield=''; name=''
<BLANKLINE>
>>> e_daf.lol
[['a1', 3, 'x']]
```

Load the file as text, then call `apply_dtypes()` for the columns you need. `from_csv()` and
`from_csv_buff()` still take `dtypes=`, and convert as they load, but that is deprecated.

## bool columns

In daffodil a boolean is the int 0 or 1, never the word True or False. The page
[Booleans](booleans.md) gives the standard and the reasons for it.

A `bool` column gets 0 or 1. The usual words for true and false are read in lower case,
capitalized or upper case. An empty cell becomes 0, unlike in an `int` or `float` column.
Other text stays as it is. Note that `'False'` gives 0, where Python's `bool('False')` is True.

```pycon
>>> flags = Daf(cols=['ok'], lol=[[''], ['0'], ['1'], ['False'], ['True'], ['no'], ['maybe']])
>>> flags
<BLANKLINE>
|  ok   |
| ----: |
|       |
|     0 |
|     1 |
| False |
|  True |
|    no |
| maybe |
<BLANKLINE>
%% daf rows=7; cols=1; keyfield=''; name=''
<BLANKLINE>
>>> flags.apply_dtypes(dtypes={'ok': bool})
<BLANKLINE>
|  ok   |
| ----: |
|     0 |
|     0 |
|     1 |
|     0 |
|     1 |
|     0 |
| maybe |
<BLANKLINE>
%% daf rows=7; cols=1; keyfield=''; name=''
<BLANKLINE>
>>> flags.col('ok')
[0, 0, 1, 0, 1, 0, 'maybe']
```

The words for 0 are `0`, `false`, `no`, `n`, `f` and `off`. The words for 1 are `1`, `true`,
`yes`, `y`, `t` and `on`.

Converted to `str`, a bool becomes `'0'` or `'1'`, so it matches the text of a file that holds 0
and 1:

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

Daffodil does not change a True or False stored in a cell until it is converted. `to_csv_file()`
writes such a cell as `True` or `False`, so store 0 and 1.

## Missing values

A missing value is NULL, the empty string `''`. It prints as nothing, so a table with missing
values stays easy to read.

Daffodil fills a missing cell with NULL, as when a dict given to `append()` leaves out a
column. But it does not turn other values into NULL. A None that you store
stays None, and `from_pandas_df()` keeps NaN as NaN. Neither is NULL, so `is NULL` is False
for both. Replace them yourself if you want NULL.

A None prints as `None`, unlike NULL. `to_csv_file()` writes it as an empty field, so the file
holds NULL. In memory it stays None until its column is converted:

```pycon
>>> from daffodil.daf import NULL
>>> nones = Daf(cols=['id', 'n', 's', 'ok', 'cv'],
...             lol=[['a1', None, None, None, {'x': None}], ['b2', 3, 'x', 1, {'x': 1}]])
>>> nones
<BLANKLINE>
| id |  n   |  s   |  ok  |     cv      |
| -: | ---: | ---: | ---: | ----------: |
| a1 | None | None | None | {'x': None} |
| b2 |    3 |    x |    1 |    {'x': 1} |
<BLANKLINE>
%% daf rows=2; cols=5; keyfield=''; name=''
<BLANKLINE>
>>> nones.iloc(0)['n'] is NULL
False
>>> print(nones.to_csv_buff())
id,n,s,ok,cv
a1,,,,{'x': None}
b2,3,x,1,{'x': 1}
```

`apply_dtypes()` makes None NULL in an `int` or `float` column, and 0 in a `bool` column. A `str`
column is skipped by default, so None stays there. Pass `from_str=False` to make it NULL too. A
None inside a dict or list cell stays None. `normalize_val()` in `daffodil.lib.daf_utils` makes it NULL.

```pycon
>>> nones_dtypes = {'id': str, 'n': int, 's': str, 'ok': bool, 'cv': dict}
>>> nones.apply_dtypes(dtypes=nones_dtypes).iloc(0)
{'id': 'a1', 'n': '', 's': None, 'ok': 0, 'cv': {'x': None}}
>>> nones.apply_dtypes(dtypes=nones_dtypes, from_str=False).iloc(0)
{'id': 'a1', 'n': '', 's': '', 'ok': 0, 'cv': {'x': None}}
```

- The sum methods skip it.
- The value counts include it, as a count for `''`, unless you pass `omit_nulls=True`.
- `to_pandas_df()` keeps it as `''`, so a number column with NULL becomes an object column.
  Pass `default=None` to get NaN in its place. That writes None into the Daf itself. See
  [to_pandas_df()][daffodil.lib.daf_pandas._to_pandas_df] for a route that leaves the Daf alone.
- `to_numpy()` makes one array of one dtype, so a column with NULL becomes text.

```pycon
>>> gaps = Daf(cols=['g', 'n'], lol=[['x', 1], ['', 2], ['x', '']])
>>> gaps
<BLANKLINE>
| g  | n  |
| -: | -: |
|  x |  1 |
|    |  2 |
|  x |    |
<BLANKLINE>
%% daf rows=3; cols=2; keyfield=''; name=''
<BLANKLINE>
>>> gaps.sum(['n'])
{'n': 3.0}
>>> gaps.valuecounts_for_colname('g'), gaps.valuecounts_for_colname('g', omit_nulls=True)
({'x': 2, '': 1}, {'x': 2})
```

### Testing for NULL

Test a cell with `is NULL`. It is faster than `== NULL`, by about three times for a number
cell, since `==` must first try to compare the number with text. It works because Python
keeps one shared empty string.

```pycon
>>> from daffodil.daf import NULL
>>> [val is NULL for val in gaps.col('n')]
[False, False, True]
```

- The import is not needed. `NULL = ''` in your own module gives the same object.
- Do not write `val is ''`. Python warns about `is` with a literal, so use the name.
- `is NULL` misses an empty NumPy string, or an empty str subclass, written into a cell.
  Daffodil's own loaders and converters never make these.

## Writing a CSV file

`to_csv_file(path)` writes a CSV file, and `to_csv_buff()` returns the text. A list, dict,
set or tuple in a cell is written as PYON, the Python form of the value, and is read back by
`apply_dtypes()`:

```pycon
>>> f_daf = Daf(cols=['id', 'tags'], lol=[['a', [1, 2]], ['b', {'k': (1, 2)}]])
>>> f_daf
<BLANKLINE>
| id |     tags      |
| -: | ------------: |
|  a |        [1, 2] |
|  b | {'k': (1, 2)} |
<BLANKLINE>
%% daf rows=2; cols=2; keyfield=''; name=''
<BLANKLINE>
>>> print(f_daf.to_csv_buff())
id,tags
a,"[1, 2]"
b,"{'k': (1, 2)}"
```

PYON is like JSON, but it can hold sets, tuples, and dicts with keys that are not text. See
[PYON](https://github.com/raylutz/pyon/blob/main/README.md).

## To and from other forms

| To | From |
|---|---|
| `to_lod()`: a list of dicts | `Daf.from_lod()` |
| `to_dod()`: a dict of dicts, by key | `Daf.from_dod()` |
| `to_cols_dol()`: a dict of column lists | `Daf.from_cols_dol()` |
| `to_json()` | `Daf.from_json()` |
| `to_md()`: a Markdown table | `Daf.from_md()` |
| `to_pandas_df()` | `Daf.from_pandas_df()` |
| `to_numpy()` | `Daf.from_numpy()` |
| `to_csv_file()`, `to_csv_buff()` | `Daf.from_csv()`, `Daf.from_csv_buff()`, `Daf.from_excel_buff()`, `Daf.from_ods()` |
