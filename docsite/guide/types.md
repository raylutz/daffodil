# Types and conversion

A cell can hold any Python object: a number, text, a list, a dict, a set, or another Daf.
Nothing limits what a cell holds, and nothing is converted unless you ask.

## dtypes

`dtypes` is a dict of column name to type. It says how to convert the columns when you ask,
as when reading a CSV file. It is not a schema that checks or forces the cells, and a Daf
built with `dtypes` keeps its cells as they are.

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

- An empty cell is NULL, the empty string, and stays NULL. See Missing values below.
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

`from_csv()` and `from_csv_buff()` also take `dtypes=`, to set them as the file is read.

## Missing values

A missing value is NULL, the empty string `''`. It prints as nothing, so a table with missing
values stays easy to read. The sums and counts of Daffodil skip it. Converting to NumPy or
pandas turns it into their missing value.

```pycon
>>> from daffodil.daf import NULL
>>> NULL == ''
True
```

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
| `to_csv_file()`, `to_csv_buff()` | `Daf.from_csv()`, `Daf.from_csv_buff()`, `Daf.from_excel_buff()` |
