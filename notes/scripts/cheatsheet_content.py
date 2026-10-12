# cheatsheet_content.py
#
# The content of docsite/cheatsheet.md. Edit this file, then run
#   uv run python notes/scripts/build_cheatsheet.py
# tests/test_cheatsheet.py runs every snippet against SETUP, and checks that the .md file is up to date.
#
# SECTIONS: (title, rows, notes). A row is (code, what it does). In the 'From pandas' section a row
# is (pandas code, Daffodil code), and only the Daffodil code is run. A row with run=False as a third
# item is shown but not run. Notes are plain sentences under the table.

SETUP = '''
from daffodil.daf import Daf
from daffodil.keyedlist import KeyedList
daf = Daf(cols=['id', 'item', 'g', 'qty'], keyfield='id',
          lol=[['a1', 'pen', 'x', 3], ['b2', 'ink', 'y', 10], ['c3', 'pad', 'x', 7], ['d4', 'cap', 'y', 1]])
other = Daf(cols=['id', 'price'], keyfield='id', lol=[['a1', 1.5], ['b2', 4.0]])
lod = [{'id': 'a1', 'qty': 3}, {'id': 'b2', 'qty': 10}]
types = {'id': str, 'item': str, 'g': str, 'qty': int}
k, n, row = 'a1', 2, ['e5', 'tag', 'x', 2]
open('data.csv', 'w').write('id,qty\\na1,3\\nb2,10\\n')
'''

PANDAS_TITLE = 'From pandas'

SECTIONS = [
    ('Create', [
        ("daf = Daf(cols=['id', 'qty'], lol=[['a1', 3]], keyfield='id')", 'from lists'),
        ("daf = Daf.from_lod(lod, keyfield='id')", 'from a list of dicts'),
        ("daf = Daf.from_csv('data.csv')", 'from a CSV file; the cells are text'),
        ("daf = Daf(cols=['id', 'qty'])", 'empty, to append to'),
    ], ["`Daf(schema=MySchema)` takes the columns, dtypes, keyfield and defaults from a `@schemaclass`."]),

    ('Look', [
        ("print(daf)", 'a Markdown table'),
        ("len(daf)", 'number of rows'),
        ("daf.shape()", '(rows, cols)'),
        ("daf.columns()", 'column names'),
        ("daf.keys()", 'keyfield values'),
        ("daf[:5]", 'the first 5 rows'),
    ], []),

    ('Rows', [
        ("daf[2]", 'row 2'),
        ("daf[-3:]", 'the last 3 rows'),
        ("daf[[0, 2]]", 'rows 0 and 2'),
        ("daf['a1']", "the row with key 'a1'"),
        ("daf[['a1', 'c3']]", 'the rows with these keys'),
        ("daf[('a1', 'c3'), :]", 'keys a1 through c3; the `:` is needed'),
        ("daf[('c3', None), :]", 'key c3 to the end'),
    ], []),

    ('Columns and cells', [
        ("daf[:, 'qty']", 'the qty column'),
        ("daf[:, ['id', 'qty']]", 'two columns, in this order'),
        ("daf[:, 1:3]", 'columns 1 and 2'),
        ("daf[2, 'qty']", 'one cell, by position and name'),
        ("daf['b2', 'qty']", 'one cell, by key and name'),
    ], []),

    ('Values out of a selection', [
        ("daf['b2', 'qty'].to_value()", 'the value: 10'),
        ("daf[:, 'qty'].to_list()", 'a column as a list'),
        ("daf['b2'].to_dict()", 'a row as a dict'),
        ("daf.col('qty')", 'a column as a list, without building a Daf'),
        ("daf.retmode = 'val'", 'from now on, `daf[...]` of one cell, row or column gives values'),
        ("daf.retmode = 'obj'", 'back to Dafs, the default'),
    ], []),

    ('Rows by condition', [
        ("daf.select_where(lambda row: row['qty'] > 5)", 'rows where the function is true'),
        ("daf.select_by_dict({'g': 'x'})", 'rows equal to these values; much faster'),
        ("daf.select_by_dict([{'g': 'x'}, {'g': 'y'}])", 'rows equal to any of these'),
        ("daf.select_krows(['a1'], inverse=True)", 'all rows but these keys'),
    ], []),

    ('Set values', [
        ("daf[2, 'qty'] = 0", 'one cell'),
        ("daf['b2', 'qty'] = 0", 'one cell, by key'),
        ("daf[:, 'qty'] = [1, 2, 3, 4]", 'a column, from a list'),
        ("daf[:, 'qty'] = 0", 'a column, one value'),
        ("daf[0] = ['a1', 'pen', 'x', 5]", 'a row, from a list'),
        ("daf[0] = {'id': 'a1', 'qty': 5}", 'a row, by name; the columns it lacks become NULL'),
    ], []),

    ('Build in a loop', [
        ("daf.append({'id': 'e5', 'qty': 2})", 'a dict, placed by name'),
        ("daf.append(['e5', 'tag', 'x', 2])", 'a list, in column order'),
        ("daf.append(lol=[['e5', 'tag', 'x', 2]])", 'several lists'),
        ("row = daf.default_record(astype=KeyedList)", "a new row: the Daf's columns, with defaults"),
        ("row['qty'] = 2", 'fill it by name, in any order'),
        ("daf.append(row, fast=True)", 'append it with no copy'),
    ], ["Without `fast`, every row is checked and copied.",
        "With `fast=True`, a KeyedList must come from `default_record(astype=KeyedList)` of the same Daf. "
        "A dict or list is checked fully if it is the first row of an empty Daf, and otherwise only for its length. "
        "Nothing is copied, so do not change a list after appending it."]),

    ('Reshape', [
        ("daf.sort_by_colname('qty', reverse=True)", 'sort the rows'),
        ("daf.insert_col('note', ['', '', '', ''])", 'add a column'),
        ("daf.select_cols(exclude_cols=['g'])", 'drop columns'),
        ("daf.transpose()", 'rows become columns'),
        ("daf.copy('editable')", 'a copy with its own rows'),
        ("daf.rename_cols({'qty': 'n'})", 'rename columns'),
    ], []),

    ('Types', [
        ("daf.apply_dtypes(dtypes=types)", 'convert text to types, in place; names every column'),
        ("daf.apply_dtypes(dtypes={'qty': int}, cols=['qty'])", 'convert only some columns'),
    ], ["NULL, a missing value, is `''`, the empty string. dtypes describe the columns, and do not force them. "
        "A cell can hold any Python object."]),

    ('Apply, reduce, group', [
        ("daf.apply(lambda row: {'id': row['id'], 'q2': row['qty'] * 2})", 'a new table, row by row'),
        ("daf.reduce(Daf.sum_da, cols=['qty'])", 'column sums, as a dict'),
        ("daf.groupby('g')", 'a dict of Dafs, one for each value'),
        ("daf.groupby_cols_reduce(['g'], Daf.sum_da, reduce_cols=['qty'])", 'sums for each group'),
        ("daf.valuecounts_for_colname('g')", "counts: `{'x': 2, 'y': 2}`"),
    ], []),

    ('Join and out', [
        ("daf.join(other, how='left')", 'by key: inner, left, right or outer'),
        ("daf.to_lod()", 'a list of dicts'),
        ("daf.to_md(max_rows=10)", 'Markdown text'),
        ("daf.to_csv_file('out.csv')", 'a CSV file'),
        ("daf.to_pandas_df()", 'a pandas DataFrame'),
    ], []),

    (PANDAS_TITLE, [
        ("df['qty']", "daf[:, 'qty']  or  daf.col('qty')"),
        ("df.loc[k]", "daf[k]"),
        ("df.iloc[n]", "daf[n]"),
        ("df.head(n)", "daf[:n]"),
        ("df.tail(n)", "daf[-n:]"),
        ("df[df.qty > 5]", "daf.select_where(lambda row: row['qty'] > 5)"),
        ("pd.concat([df, new])", "daf.append(row)"),
        ("df.merge(other)", "daf.join(other)"),
        ("df.T", "daf.transpose()"),
    ], ["`daf[x]` selects rows, where `df[x]` selects a column."]),
]

# The indexing reference: what each coordinate of daf[rows, cols] can be. Shown, not run.
INDEXING_SYNTAX = """daf[rows, cols]     rows and columns
daf[rows]           rows, all columns
daf[:, cols]        columns, all rows"""

INDEXING = [
    # (you write, as rows, as columns)
    ("`:`", "all rows", "all columns"),
    ("`2`", "row 2, by position", "column 2, by position"),
    ("`-1`", "the last row", "the last column"),
    ("`2:5`, `2:`, `:5`, `::2`", "a slice of rows", "a slice of columns"),
    ("`range(2, 5)`", "rows 2, 3, 4", "columns 2, 3, 4"),
    ("`[0, 4, 2]`", "these rows, in this order", "these columns, in this order"),
    ("`[range(0, 2), range(4, 6)]`", "rows 0, 1, 4, 5", "columns 0, 1, 4, 5"),
    ("`'a1'`", "the row with key `'a1'`", "the column named `'a1'`"),
    ("`['c3', 'a1']`", "the rows with these keys, in this order", "the columns with these names, in this order"),
    ("`('a1', 'c3')`", "keys a1 through c3, inclusive", "names a1 through c3, inclusive"),
    ("`('a1', None)`, `('a1',)`", "key a1 to the end", "name a1 to the end"),
    ("`(None, 'c3')`", "the start through key c3", "the start through name c3"),
    ("`[]`", "no rows", "no columns"),
]

INDEXING_NOTES = [
    "Selecting whole rows, as `daf[rows]` or `daf[rows, :]`, copies no data: the new Daf shares the rows "
    "with the original, so a change to a cell in one shows in the other. Selecting some of the columns "
    "makes new rows. Use `copy('editable')` for rows of your own.",
    "An integer is always a position. A key or a column name that is an integer needs `select_krows()` or `select_kcols()`.",
    "Rows by key need a keyfield.",
    "A tuple of row keys needs the column part, even if it is `:`, as in `daf[('a1', 'c3'), :]`. "
    "A tuple of two standing alone is read as `[rows, cols]`.",
    "A list holds integers, or ranges, or strings, not a mix. `None` alone is not a selector.",
    "The result is a Daf. `.to_value()`, `.to_list()` or `.to_dict()` give plain values, or set `daf.retmode = 'val'`.",
    "The same selectors set values: `daf[rows, cols] = value`.",
]


def indexing_md() -> str:
    """ The indexing reference as Markdown: the syntax, a table made by Daf.to_md(), and the notes. """
    from daffodil.daf import Daf
    table = Daf(cols=['You write', 'As rows (first)', 'As columns (second)'], lol=[list(r) for r in INDEXING]).to_md(
        just='<<<', shorten_text=False)
    notes = ''.join(f'- {note}\n' for note in INDEXING_NOTES)
    return f"```\n{INDEXING_SYNTAX}\n```\n\n{table}\n{notes}"


RULES = [
    "Import with `from daffodil.daf import Daf`. Name a table `daf`, as pandas uses `df`.",
    "A Daf is `lol`, a list of rows, each a plain list, plus `hd`, a dict of column name to position.",
    "`daf[rows, cols]` selects. An integer is always a position. A string is a key of the keyfield for rows, "
    "and a column name for columns. A tuple of two is an inclusive range.",
    "A selection returns a Daf. Use `.to_value()`, `.to_list()` or `.to_dict()` for plain values, "
    "or set `daf.retmode = 'val'`.",
    "A selection of rows shares the rows with the original: changing a cell in it changes the original. "
    "A selection of columns makes new rows.",
    "`append()`, `apply_dtypes()` and most changing methods return the Daf itself, changed in place.",
    "A missing value is NULL, the empty string `''`, not None or NaN.",
    "The keyfield column stays in the table. Key lookups are dict lookups, built when first needed.",
    "Rows are cheap and columns cost more. Name the columns you need, as with `cols=`, rather than dropping the others.",
]

MISTAKES = [
    "`daf[:-5]` is all but the last 5 rows. The last 5 are `daf[-5:]`.",
    "`apply_dtypes(dtypes=...)` without `cols=` must name every column. To convert some, add `cols=`.",
    "`groupby_cols_reduce()` without `reduce_cols=` reduces nothing.",
    "A KeyedList row is not a dict: `isinstance(row, dict)` is False, and `json.dumps(row)` fails. Use `row.to_dict()`.",
    "With `append(row, fast=True)` the table keeps the list itself. Do not fill one list in place and append it again.",
    "`daf[('a1', 'c3')]` without `, :` is read as `[rows, cols]`. Write `daf[('a1', 'c3'), :]`.",
    "Methods that do not exist, from old examples: `add_idx()` is `insert_idx_col()`, `to_csv()` is `to_csv_file()`.",
]
