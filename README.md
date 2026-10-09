<img src="https://raw.githubusercontent.com/raylutz/daffodil/main/docsite/images/daffodil_logo.png" alt="Daffodil logo" width="110">

# Python Daffodil

Daffodil (DAtaFrames For Optimized Data Inspection and Logical processing) gives Python fast,
lightweight 2-D data tables. A table is a list of rows, and each row is a plain Python list.
Use it where data arrives and is processed row by row: reading files, building records,
cleaning, reshaping, and writing them out again.

**Documentation: https://raylutz.github.io/daffodil/**

Status: alpha, and used in production for several years. The changes of each release are in
the [changelog](https://github.com/raylutz/daffodil/blob/main/CHANGELOG.md).

## Install

```bash
pip install daffodil
```

Python 3.10 or later. Daffodil is pure Python. pandas and NumPy are needed only to convert to
and from them.

## The data model

![The Daffodil data model: lol, hd, kd and dtypes](https://raw.githubusercontent.com/raylutz/daffodil/main/docsite/images/data_model.png)

The rows are kept in `lol`, a list of lists. The column names are kept once, in the header
dict `hd`, which maps each name to its position. A table can have a keyfield, a column whose
values identify the rows, and then the key dict `kd` maps each key to its row. A cell can hold
any Python object: text, a number, a list, a dict, or even another table.

## Quick start

```python
from daffodil.daf import Daf

daf = Daf(cols=['id', 'item', 'qty'], keyfield='id')
daf.append({'id': 'a1', 'item': 'pen', 'qty': 3})
daf.append(['b2', 'ink', 10])

daf['b2']                     # the row with key 'b2'
daf[0]                        # the first row
daf[:, 'qty']                 # the qty column
daf['b2', 'qty'].to_value()   # 10
daf.select_where(lambda row: row['qty'] > 5)
print(daf)                    # a Markdown table
```

## Why Daffodil

- **Adding a row is cheap.** It is appended to a Python list. pandas has no cheap way to add
  a row, so code that uses it often collects the rows first and converts them at the end.
- **No conversion.** The data stays the Python objects you put in, so there is no cost to
  get it in or out.
- **A selection shares the rows.** Selecting rows copies no data, and lookups by key are dict
  lookups.
- **Mixed data is fine.** A column can mix text, numbers and empty cells, and nothing is
  coerced to NaN or to a common type.
- **Tables print as Markdown,** ready for a terminal or a report, and can be read back.
- **Small and pure Python,** so it starts fast, which matters in short-lived cloud functions.

For heavy math over whole numeric columns, pandas, NumPy or Polars are faster. Build and clean
the data as rows in Daffodil, then hand the numeric columns over with `to_pandas_df()` or
`to_numpy()`. The [documentation](https://raylutz.github.io/daffodil/) compares the two, with
measurements.

## Learn more

- [Cheatsheet](https://raylutz.github.io/daffodil/cheatsheet/), everything on one page
- [Getting started](https://raylutz.github.io/daffodil/guide/getting-started/)
- [Building tables](https://raylutz.github.io/daffodil/guide/building-tables/), including the
  fast way to build rows in a loop
- [Selecting and indexing](https://raylutz.github.io/daffodil/guide/selecting/)
- [Columns, keys and schemas](https://raylutz.github.io/daffodil/guide/columns-keys-schemas/)
- [Types and conversion](https://raylutz.github.io/daffodil/guide/types/)
- [Apply, reduce and formulas](https://raylutz.github.io/daffodil/guide/apply-reduce-formulas/)
- [Joins](https://raylutz.github.io/daffodil/guide/joins/)
- [Markdown reports](https://raylutz.github.io/daffodil/guide/markdown/)
- [Coming from pandas](https://raylutz.github.io/daffodil/guide/from-pandas/)
- [The Daf API](https://raylutz.github.io/daffodil/api/daf/), every method with examples

Questions and ideas are welcome as [GitHub issues](https://github.com/raylutz/daffodil/issues).
Daffodil is released under the MIT license.
