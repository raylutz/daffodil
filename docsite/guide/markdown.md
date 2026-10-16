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
- `include_summary`: add the line that describes the table, for `from_md()`.

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

## Reading Markdown back

Markdown is a report format first. A table can be read back, but only some tables come back
whole. With `include_summary=True`, the table ends with a line that gives its keyfield and
name. `Daf.from_md()` reads the table back, with them.

For the most faithful copy, pass `shorten_text=False`, leave `smart_fmt`, `max_rows` and
`max_cols` at their defaults, and include the summary. Even then:

- The cells come back as text, as from a CSV file, and the dtypes are not kept. See
  [Types and conversion](types.md).
- A newline in a cell comes back as `<br>`.

A `|` in a cell, and a composite keyfield, come back as they were.

For data that must come back exactly, use `to_json()` and `from_json()`, which keep numbers,
text and the keyfield. A CSV file keeps the text exactly, and `apply_dtypes()` restores the types.

```pycon
>>> text = daf.to_md(include_summary=True, shorten_text=False)
>>> text.splitlines()[-1]
"%% daf rows=8; cols=3; keyfield='id'; name='stock'"
>>> back = Daf.from_md(text)
>>> back.columns(), back.keyfield, back.name
(['id', 'item', 'qty'], 'id', 'stock')
>>> back.lol[1]
['r2', 'xx', '2']
```

`Daf.dodaf_to_md()` writes several named tables to one Markdown text, and
`Daf.dodaf_from_md()` reads them back as a dict of Dafs.
