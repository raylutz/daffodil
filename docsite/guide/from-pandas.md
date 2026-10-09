# Coming from pandas

Daffodil keeps rows where pandas keeps columns. So building, selecting and changing rows is
cheap, and math over a whole numeric column is better done in pandas or NumPy. See the
[Home page](../index.md) for when each fits, with measurements.

Some differences to know first:

- `daf[x]` selects rows, where `df[x]` selects a column. A column is `daf[:, 'name']`, or as a
  list, `daf.col('name')`.
- An integer in `[]` is always a position, as with `iloc`. A key is a value of the keyfield
  column, which stays in the table, unlike the pandas index.
- A selection of rows shares the rows with the original. Changing a cell in it changes the
  original. See [Selecting and indexing](selecting.md).
- A cell can hold anything, and columns are not converted unless you ask. A missing value is
  `''`, not NaN.
- A table prints as Markdown.

## The same task in each

| pandas | Daffodil | Notes |
|---|---|---|
| `pd.DataFrame(lod)` | `Daf.from_lod(lod)` | |
| `pd.read_csv(path)` | `Daf.from_csv(path)` | the cells are text until `apply_dtypes()` |
| `df.to_csv(path)` | `daf.to_csv_file(path)` | |
| `df.to_dict('records')` | `daf.to_lod()` | |
| `df.shape`, `len(df)` | `daf.shape()`, `len(daf)` | |
| `df.empty` | `not daf` | |
| `df.columns` | `daf.columns()` | |
| `df.dtypes` | `daf.dtypes` | describes, does not enforce |
| `df.astype(...)` | `daf.apply_dtypes(dtypes=...)` | in place |
| `df.head(n)`, `df.tail(n)` | `daf[:n]`, `daf[-n:]` | |
| `df.iloc[n]` | `daf[n]` | `.to_dict()` for a dict |
| `df.loc[k]` | `daf[k]` | needs a keyfield. `.to_dict()` for a dict |
| `df.set_index('id')` | `daf.set_keyfield('id')` | the column stays in the table |
| `df['c']` | `daf[:, 'c']`, or `daf.col('c')` for a list | |
| `df[['a', 'b']]` | `daf[:, ['a', 'b']]` | |
| `df[df.qty > 5]` | `daf.select_where(lambda row: row['qty'] > 5)` | |
| `df[df.c.isin(vals)]` | `daf.select_by_dict([{'c': v} for v in vals])` | one pass, no list of bools |
| `df.drop(columns=[...])` | `daf.select_cols(exclude_cols=[...])` | |
| `df[~df.id.isin(keys)]` | `daf.select_krows(keys, inverse=True)` | |
| `df.loc[len(df)] = row`, `pd.concat` | `daf.append(row)` | cheap |
| `pd.concat([df1, df2])` | `d1.append(d2)` | |
| `df.insert(i, 'c', vals)` | `daf.insert_col('c', vals, icol=i)` | |
| (none) | `daf.insert_irow(i, row)` | insert a row anywhere |
| `df['c'] = vals` | `daf[:, 'c'] = vals` | |
| `df.rename(columns=...)` | `daf.rename_cols({...})` | |
| `df.sort_values('c')` | `daf.sort_by_colname('c')` | |
| `df.T` | `daf.transpose()` | |
| `df.apply(f, axis=1)` | `daf.apply(f)` | the row is a dict, and `f` returns one |
| `df.agg(...)` | `daf.reduce(f)` | |
| `df.groupby('g')` | `daf.groupby('g')` | a dict of Dafs |
| `df.groupby('g')[cols].sum()` | `daf.groupby_cols_reduce(['g'], Daf.sum_da, reduce_cols=cols)` | name the columns to add |
| `df.groupby([...])[cols].agg(f)` | `daf.groupby_cols_reduce([...], f, reduce_cols=cols)` | |
| `df.value_counts('c')` | `daf.valuecounts_for_colname('c')` | a dict |
| `df.merge(other, on='id')` | `daf.join(other)` | both need a keyfield. See [Joins](joins.md). |
| `df.replace(...)` | `daf.find_replace(pattern, value)` | |
| `df.to_markdown()` | `daf.to_md()` | |
| `df.to_numpy()` | `daf.to_numpy()` | |

## A short example

```pycon
>>> from daffodil.daf import Daf
>>> daf = Daf.from_lod([{'id': 'a1', 'g': 'x', 'qty': 3}, {'id': 'b2', 'g': 'y', 'qty': 10},
...                    {'id': 'c3', 'g': 'x', 'qty': 7}], keyfield='id')
>>> daf[:2].col('id')
['a1', 'b2']
>>> daf.select_where(lambda row: row['qty'] > 5).col('id')
['b2', 'c3']
>>> daf.groupby_cols_reduce(['g'], Daf.sum_da, reduce_cols=['qty']).lol
[['x', 10], ['y', 10]]
>>> daf.sort_by_colname('qty', reverse=True).col('id')
['b2', 'c3', 'a1']
```
