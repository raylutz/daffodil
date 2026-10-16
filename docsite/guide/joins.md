# Joins

`join()` combines two tables by their keys. Each table needs a keyfield of one column. A
composite keyfield is not supported. A row of one table matches the row of the other with the
same key.

The keys should be unique in both tables. The join is one to one, not many to many as in SQL.
If a key repeats in the other table, each match gets its last row. If it repeats in the first
table, each of those rows is in the result.

```pycon
>>> from daffodil.daf import Daf
>>> stock = Daf(cols=['id', 'name', 'qty'], keyfield='id', name='stock',
...             lol=[['a1', 'pen', 3], ['b2', 'ink', 10], ['c3', 'pad', 7]])
>>> prices = Daf(cols=['id', 'price', 'qty'], keyfield='id', name='prices',
...              lol=[['a1', 1.5, 100], ['b2', 4.0, 50], ['d4', 2.0, 5]])
```

## Join types

`how=` chooses which rows are kept:

- `inner`, the default: only the keys in both tables.
- `left`: every row of the first table, with the matching rows of the second.
- `right`: every row of the second table, with the matching rows of the first.
- `outer`: every row of both.

A side with no matching row gets NULL. Pass `fill=None` to get `None` instead. The rows of the
first table come first, in its order. In a right or outer join, the rows that only the second
table has come after them.

```pycon
>>> stock.join(prices).col('id')
['a1', 'b2']
>>> stock.join(prices, how='outer').lol
[['a1', 'pen', 3, 1.5, 100], ['b2', 'ink', 10, 4.0, 50], ['c3', 'pad', 7, '', ''], ['d4', '', '', 2.0, 5]]
```

## Columns with the same name

The key column appears once. Another column that both tables have is kept twice, with the
table's name as a suffix, or `_daf1` and `_daf2` for tables with no name:

```pycon
>>> stock.join(prices).columns()
['id', 'name', 'qty_stock', 'price', 'qty_prices']
>>> one = Daf(cols=['id', 'q'], keyfield='id', lol=[['a', 1]])
>>> two = Daf(cols=['id', 'q'], keyfield='id', lol=[['a', 2]])
>>> one.join(two).columns()
['id', 'q_daf1', 'q_daf2']
```

List a column in `shared_fields` to keep it once instead. Its value comes from the first table:

```pycon
>>> joined = stock.join(prices, shared_fields=['qty'])
>>> joined.columns(), joined.lol[0]
(['id', 'name', 'qty', 'price'], ['a1', 'pen', 3, 1.5])
```

A row that only the second table has takes the shared column from the second table, since
the first has no row for it. Here d4 has the qty 5 of prices:

```pycon
>>> stock.join(prices, how='outer', shared_fields=['qty']).lol[-1]
['d4', '', 5, 2.0]
```

For other names, give a `custom_translator_daf`, which maps each column of each table to its
name in the result. See [join()][daffodil.daf.Daf.join] in the API.
