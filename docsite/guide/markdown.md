# Markdown reports

Daffodil shows a table as Markdown. The same text reads well in a terminal, and drops into a
Markdown report or a page like this one.

```pycon
>>> from daffodil.daf import Daf
>>> daf = Daf(cols=['id', 'item', 'qty'], keyfield='id', name='stock',
...          lol=[[f'r{i}', 'x' * i, i] for i in range(1, 9)])
>>> print(daf.to_md(max_rows=4))
| id  |   item   | qty |
| --: | -------: | --: |
|  r1 |        x |   1 |
|  r2 |       xx |   2 |
| ... |      ... | ... |
|  r7 |  xxxxxxx |   7 |
|  r8 | xxxxxxxx |   8 |
```

`print(daf)` shows the table, followed by a line that describes it.

## Options of to_md()

- `max_rows`, `max_cols`: keep the first and last rows or columns, and put `...` between.
- `just`: the alignment of each column, one character each: `<` left, `^` center, `>` right.
- `shorten_text`, `max_text_len`: shorten long text, keeping its two ends.
- `smart_fmt`: fewer digits after the decimal point, in a column of numbers.
- `col_fmt`: a format for some columns, by name: a format spec such as `'.1%'`, `'smart'`,
  `'smart%'` or a function. Only the text changes, not the Daf.
- `show`: what a plain table hides, the types. Give one word, or a list of them:
  `'dtypes'` puts each column's dtype in its header, as `age: int`, and notes a column whose
  cells hold another type, as `age: int (str)`. `'quotes'` puts each text cell in quotes, so the
  text `'34'` differs from the number 34. `'types'` shows each cell's type name in place of its value.
- `disp_cols`, `header`: other names in the header, for some columns or for all.
- `include_summary`: add the line that describes the table, for [`from_md()`][daffodil.lib.daf_md._from_md].

```pycon
>>> print(Daf(cols=['n', 't'], lol=[[1.23456, 'a' * 100]]).to_md(smart_fmt=True, max_text_len=20))
|  n  |          t           |
| --: | -------------------: |
| 1.2 | aaaaaaaaa..aaaaaaaaa |
>>> print(Daf(cols=['a', 'b'], lol=[[1, 'x']]).to_md(just='<>'))
| a  | b  |
| :- | -: |
| 1  |  x |
```

Format some columns, and see the types. Here the age is still text, as read from a CSV file:

```pycon
>>> sales = Daf(cols=['city', 'share', 'amount'], lol=[['Denver', 0.3412, 12345.5], ['Reno', 0.0001, 9876.25]])
>>> print(sales.to_md(col_fmt={'share': 'smart%', 'amount': ',.2f'}))
|  city  | share |  amount   |
| -----: | ----: | --------: |
| Denver | 34.1% | 12,345.50 |
|   Reno | 0.01% |  9,876.25 |
>>> people = Daf(cols=['id', 'age'], dtypes={'id': str, 'age': int}, lol=[['p1', '34'], ['p2', '52']])
>>> print(people.to_md(show=['dtypes', 'quotes']))
| id: str | age: int (str) |
| ------: | -------------: |
|    'p1' |           '34' |
|    'p2' |           '52' |
>>> print(people.to_md(show='types'))
| id  | age |
| --: | --: |
| str | str |
| str | str |
```

## Reading Markdown back

Markdown is a report format first. A table can be read back, but only some tables come back
whole. With `include_summary=True`, the table ends with a line that gives its keyfield and
name. [`Daf.from_md()`][daffodil.lib.daf_md._from_md] reads the table back, with them.

For the most faithful copy, pass `shorten_text=False`, leave `smart_fmt`, `max_rows` and
`max_cols` at their defaults, and include the summary. Even then:

- The cells come back as text, as from a CSV file. The dtypes come back only from a header
  written with `show='dtypes'`, and nothing is converted. See [Types and conversion](types.md).
- Quoted text, from `show='quotes'`, comes back without its quotes only with the summary line.
- A table shown with `show='types'` holds type names, not values, so it cannot be read back.
- A newline in a cell comes back as `<br>`.

A `|` in a cell, and a composite keyfield, come back as they were.

For data that must come back exactly, use [`to_json()`][daffodil.daf_convert._to_json] and [`from_json()`][daffodil.daf_convert._from_json], which keep numbers,
text and the keyfield. A CSV file keeps the text exactly, and [`apply_dtypes()`][daffodil.daf_dtypes._apply_dtypes] restores the types.

```pycon
>>> text = daf.to_md(include_summary=True, shorten_text=False)
>>> text.splitlines()[-1]
"%% daf rows=8; cols=3; keyfield='id'; name='stock'"
>>> back = Daf.from_md(text)
>>> back
<BLANKLINE>
| id |   item   | qty |
| -: | -------: | --: |
| r1 |        x |   1 |
| r2 |       xx |   2 |
| r3 |      xxx |   3 |
| r4 |     xxxx |   4 |
| r5 |    xxxxx |   5 |
| r6 |   xxxxxx |   6 |
| r7 |  xxxxxxx |   7 |
| r8 | xxxxxxxx |   8 |
<BLANKLINE>
%% daf rows=8; cols=3; keyfield='id'; name='stock'
<BLANKLINE>
>>> back.columns(), back.keyfield, back.name
(['id', 'item', 'qty'], 'id', 'stock')
>>> back.lol[1]
['r2', 'xx', '2']
```

[`Daf.dodaf_to_md()`][daffodil.lib.daf_md.dodaf_to_md] writes several named tables to one Markdown text, and
[`Daf.dodaf_from_md()`][daffodil.lib.daf_md._dodaf_from_md] reads them back as a dict of Dafs.
