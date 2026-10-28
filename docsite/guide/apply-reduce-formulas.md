# Apply, reduce and formulas

## Rows, not columns

Daffodil keeps a table as rows, where pandas and Polars keep columns. Appending rows and
going through them one at a time suit that storage. Selecting whole rows reuses their row
lists, and copies no cells. Inserting or deleting a row in the middle moves the rows after
it in the outer list, so it costs more in a large table.

Working with columns costs more, as each column is spread across all the rows. Most of the
time you do not need to: rather than dropping the columns you do not want, name the ones you
do. Give `cols=` to `reduce()`. In an `apply()` function, read only the fields you need. To
show some columns, select them before `to_md()`. For heavy math over whole numeric columns,
hand them to NumPy or pandas, with `to_numpy()` or `to_pandas_df()`.

## apply(): a new table, row by row

`apply()` calls a function on each row, and makes a new table of what it returns. The row is a
dict, and the function returns a dict:

```pycon
>>> from daffodil.daf import Daf
>>> daf = Daf(cols=['g', 'a', 'b'], lol=[['x', 1, 2], ['y', 3, ''], ['x', 5, 6]])
>>> daf
<BLANKLINE>
| g  | a  | b  |
| -: | -: | -: |
|  x |  1 |  2 |
|  y |  3 |    |
|  x |  5 |  6 |
<BLANKLINE>
%% daf rows=3; cols=3; keyfield=''; name=''
<BLANKLINE>
>>> doubled = daf.apply(lambda row: {'g': row['g'], 'a2': row['a'] * 2})
>>> doubled.columns(), doubled.lol
(['g', 'a2'], [['x', 2], ['y', 6], ['x', 10]])
```

`apply_in_place()` changes the rows of the table instead of making a new one.

## reduce(): one record from many rows

`reduce()` folds the rows into one record with a function. `Daf.sum_da` adds up each column:

```pycon
>>> daf.reduce(Daf.sum_da, cols=['a', 'b'])
{'g': '', 'a': 9, 'b': 8}
>>> daf.sum(['a', 'b'])
{'a': 9.0, 'b': 8.0}
```

NULL cells are skipped, so the missing `b` of the second row adds nothing.

`sum()` and `reduce(Daf.sum_da)` differ when a column holds text. `sum()` converts text
that is a number, such as `'2'`, and raises `ValueError` for other text. `Daf.sum_da` skips
every value that is not a number, and never raises:

```pycon
>>> mixed = Daf(cols=['n'], lol=[[1], ['2'], [''], [3.5]])
>>> mixed
<BLANKLINE>
|  n  |
| --: |
|   1 |
|   2 |
|     |
| 3.5 |
<BLANKLINE>
%% daf rows=4; cols=1; keyfield=''; name=''
<BLANKLINE>
>>> mixed.sum(['n']), mixed.reduce(Daf.sum_da, cols=['n'])
({'n': 6.5}, {'n': 4.5})
```

### Your own reduction

The function gets each row, the result so far, and the columns. It returns the new result.
The row is a KeyedList, which reads like a dict. The result has one value for each column,
and starts at 0 unless you give `initial_da`. This one counts the cells that are not NULL:

```pycon
>>> from daffodil.daf import NULL
>>> def count_filled(row, result, cols):
...     for col in cols:
...         if row[col] is not NULL:
...             result[col] += 1
...     return result
>>> daf.reduce(count_filled, cols=['a', 'b'])
{'g': '', 'a': 3, 'b': 2}
```

A key of the result that is not a column is dropped. For a result of another shape, write a
loop over the rows.

## Groups

`groupby()` makes a Daf for each value of a column. `groupsum_daf()` adds up the other columns
of each group. `groupby_cols_reduce()` groups by several columns and reduces each group with
any function.

```pycon
>>> {key: group.lol for key, group in daf.groupby('g').items()}
{'x': [['x', 1, 2], ['x', 5, 6]], 'y': [['y', 3, '']]}
>>> daf.groupsum_daf('g', reduce_cols=['a', 'b']).lol
[['x', 6, 8], ['y', 3, 0]]
>>> daf.groupby_cols_reduce(['g'], Daf.sum_da, reduce_cols=['a']).lol
[['x', 6], ['y', 3]]
>>> daf.valuecounts_for_colname('g')
{'x': 2, 'y': 1}
```

## Data in many files

A table can describe a set of files, one row for each chunk of the data. `manifest_apply()`
applies a function to every row of every chunk, and `manifest_reduce()` reduces them all to
one record. Each chunk is read in turn, so the whole data never has to fit in memory, and the
chunks can be shared out to run in parallel.

## Spreadsheet-like formulas

`apply_formulas()` fills cells from a second Daf of the same shape that holds formulas, as a
spreadsheet does. A formula is Python, evaluated for its cell. An empty formula leaves its
cell alone. In a formula:

- `$d` is the Daf.
- `$r` is the row of the cell, and `$c` its column.

So a formula is absolute unless it uses `$r` or `$c`, the reverse of a spreadsheet. The
formulas are evaluated again until no value changes. If the values still change after 100
passes, it raises `RuntimeError`. This is a limit on the passes, not a check for circular
formulas. A circle that settles raises nothing.

The formulas run with Python's `eval()`, so use only formulas you trust. If a formula raises,
the cells already changed stay changed.

This sums each row into the last column, and each column into the last row:

```pycon
>>> data = Daf(cols=['A', 'B', 'C'], lol=[[1, 2, 0], [4, 5, 0], [7, 8, 0], [0, 0, 0]])
>>> data
<BLANKLINE>
| A  | B  | C  |
| -: | -: | -: |
|  1 |  2 |  0 |
|  4 |  5 |  0 |
|  7 |  8 |  0 |
|  0 |  0 |  0 |
<BLANKLINE>
%% daf rows=4; cols=3; keyfield=''; name=''
<BLANKLINE>
>>> row_sum, col_sum = "sum($d[$r, :$c])", "sum($d[:$r, $c])"
>>> formulas = Daf(cols=['A', 'B', 'C'], lol=[['', '', row_sum],
...                                           ['', '', row_sum],
...                                           ['', '', row_sum],
...                                           [col_sum, col_sum, col_sum]])
>>> formulas
<BLANKLINE>
|        A         |        B         |        C         |
| ---------------: | ---------------: | ---------------: |
|                  |                  | sum($d[$r, :$c]) |
|                  |                  | sum($d[$r, :$c]) |
|                  |                  | sum($d[$r, :$c]) |
| sum($d[:$r, $c]) | sum($d[:$r, $c]) | sum($d[:$r, $c]) |
<BLANKLINE>
%% daf rows=4; cols=3; keyfield=''; name=''
<BLANKLINE>
>>> data.apply_formulas(formulas)
<BLANKLINE>
| A  | B  | C  |
| -: | -: | -: |
|  1 |  2 |  3 |
|  4 |  5 |  9 |
|  7 |  8 | 15 |
| 12 | 15 | 27 |
<BLANKLINE>
%% daf rows=4; cols=3; keyfield=''; name=''
<BLANKLINE>
>>> data.lol
[[1, 2, 3], [4, 5, 9], [7, 8, 15], [12, 15, 27]]
```

The formulas themselves are a table, so they can be written in a spreadsheet, saved as CSV, and
read with `Daf.from_csv()`.
