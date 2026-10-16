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
>>> daf.lol
[['a1', '3', '1.5', '[1, 2]'], ['b2', '', '2.0', '[]']]
>>> _ = daf.apply_dtypes(dtypes={'id': str, 'qty': int, 'price': float, 'tags': list})
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
>>> e = Daf.from_csv_buff('id,qty,note\na1,3,x\n')
>>> _ = e.apply_dtypes(dtypes={'id': str, 'qty': int, 'note': str}, cols=['qty'])
>>> e.lol
[['a1', 3, 'x']]
```

Load the file as text, then call `apply_dtypes()` for the columns you need. `from_csv()` and
`from_csv_buff()` still take `dtypes=`, and convert as they load, but that is deprecated.

## bool columns

A `bool` column gets 0 or 1. The usual words for true and false are read in lower case,
capitalized or upper case. An empty cell becomes 0, unlike in an `int` or `float` column.
Other text stays as it is. Note that `'False'` gives 0, where Python's `bool('False')` is True.

```pycon
>>> flags = Daf(cols=['ok'], lol=[[''], ['0'], ['1'], ['False'], ['True'], ['no'], ['maybe']])
>>> _ = flags.apply_dtypes(dtypes={'ok': bool})
>>> flags.col('ok')
[0, 0, 1, 0, 1, 0, 'maybe']
```

The words for 0 are `0`, `false`, `no`, `n`, `f` and `off`. The words for 1 are `1`, `true`,
`yes`, `y`, `t` and `on`.

## Missing values

A missing value is NULL, the empty string `''`. It prints as nothing, so a table with missing
values stays easy to read.

Daffodil fills a missing cell with NULL, as when a dict given to `append()` leaves out a
column. But it does not turn other values into NULL. A None that you store
stays None, and `from_pandas_df()` keeps NaN as NaN. Neither is NULL, so `is NULL` is False
for both. Replace them yourself if you want NULL.

- The sum methods skip it.
- The value counts include it, as a count for `''`, unless you pass `omit_nulls=True`.
- `to_pandas_df()` keeps it as `''`, so a number column with NULL becomes an object column.
  Pass `default=None` to get NaN in its place. That writes None into the Daf itself. See
  [to_pandas_df()][daffodil.lib.daf_pandas._to_pandas_df] for a route that leaves the Daf alone.
- `to_numpy()` makes one array of one dtype, so a column with NULL becomes text.

```pycon
>>> gaps = Daf(cols=['g', 'n'], lol=[['x', 1], ['', 2], ['x', '']])
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
>>> f = Daf(cols=['id', 'tags'], lol=[['a', [1, 2]], ['b', {'k': (1, 2)}]])
>>> print(f.to_csv_buff())
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
