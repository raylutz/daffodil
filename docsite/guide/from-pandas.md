# Coming from pandas

Daffodil keeps rows where pandas keeps columns. So building, selecting and changing rows is
cheap, and math over a whole numeric column is better done in pandas or NumPy. See the
[Home page](../index.md) for when each fits, with measurements.

Some differences to know first:

- `d[x]` selects rows, where `df[x]` selects a column. A column is `d[:, 'name']`, or as a
  list, `d.col('name')`.
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
| `df.to_csv(path)` | `d.to_csv_file(path)` | |
| `df.to_dict('records')` | `d.to_lod()` | |
| `df.shape`, `len(df)` | `d.shape()`, `len(d)` | |
| `df.empty` | `not d` | |
| `df.columns` | `d.columns()` | |
| `df.dtypes` | `d.dtypes` | describes, does not enforce |
| `df.astype(...)` | `d.apply_dtypes(dtypes=...)` | in place |
| `df.head(n)`, `df.tail(n)` | `d[:n]`, `d[-n:]` | |
| `df.iloc[i]` | `d[i]`, or `d.iloc(i)` for a dict | |
| `df.loc[key]` | `d[key]`, or `d.select_record(key)` for a dict | needs a keyfield |
| `df.set_index('id')` | `d.set_keyfield('id')` | the column stays in the table |
| `df['c']` | `d[:, 'c']`, or `d.col('c')` for a list | |
| `df[['a', 'b']]` | `d[:, ['a', 'b']]` | |
| `df[df.qty > 5]` | `d.select_where(lambda row: row['qty'] > 5)` | |
| `df[df.c.isin(vals)]` | `d.select_by_dict([{'c': v} for v in vals])` | one pass, no list of bools |
| `df.drop(columns=[...])` | `d.select_cols(exclude_cols=[...])` | |
| `df[~df.id.isin(keys)]` | `d.select_krows(keys, inverse=True)` | |
| `df.loc[len(df)] = row`, `pd.concat` | `d.append(row)` | cheap |
| `pd.concat([df1, df2])` | `d1.append(d2)` | |
| `df.insert(i, 'c', vals)` | `d.insert_col('c', vals, icol=i)` | |
| (none) | `d.insert_irow(i, row)` | insert a row anywhere |
| `df['c'] = vals` | `d[:, 'c'] = vals` | |
| `df.rename(columns=...)` | `d.rename_cols({...})` | |
| `df.sort_values('c')` | `d.sort_by_colname('c')` | |
| `df.T` | `d.transpose()` | |
| `df.apply(f, axis=1)` | `d.apply(f)` | the row is a dict, and `f` returns one |
| `df.agg(...)` | `d.reduce(f)` | |
| `df.groupby('g')` | `d.groupby('g')` | a dict of Dafs |
| `df.groupby('g')[cols].sum()` | `d.groupby_cols_reduce(['g'], Daf.sum_da, reduce_cols=cols)` | name the columns to add |
| `df.groupby([...])[cols].agg(f)` | `d.groupby_cols_reduce([...], f, reduce_cols=cols)` | |
| `df.value_counts('c')` | `d.valuecounts_for_colname('c')` | a dict |
| `df.merge(other, on='id')` | `d.join(other)` | both need a keyfield. See [Joins](joins.md). |
| `df.replace(...)` | `d.find_replace(pattern, value)` | |
| `df.to_markdown()` | `d.to_md()` | |
| `df.to_numpy()` | `d.to_numpy()` | |

## A short example

```pycon
>>> from daffodil.daf import Daf
>>> d = Daf.from_lod([{'id': 'a1', 'g': 'x', 'qty': 3}, {'id': 'b2', 'g': 'y', 'qty': 10},
...                   {'id': 'c3', 'g': 'x', 'qty': 7}], keyfield='id')
>>> d[:2].col('id')
['a1', 'b2']
>>> d.select_where(lambda row: row['qty'] > 5).col('id')
['b2', 'c3']
>>> d.groupby_cols_reduce(['g'], Daf.sum_da, reduce_cols=['qty']).lol
[['x', 10], ['y', 10]]
>>> d.sort_by_colname('qty', reverse=True).col('id')
['b2', 'c3', 'a1']
```
