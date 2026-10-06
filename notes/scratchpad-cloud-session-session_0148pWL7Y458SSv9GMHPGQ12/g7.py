import sys
sys.path.insert(0, '/tmp/claude-0/-home-user-daffodil/95e36c09-64f8-59ba-9bd6-c962257ad9e3/scratchpad')
from setdoc import setdoc
D = {}
EXG = """    >>> d = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])"""

D['Daf.apply_formulas'] = r'''
Fill cells from spreadsheet like formulas, in place.

`formulas_daf` is a Daf of the same shape. Each cell holds a Python
expression as text. An empty cell is skipped. The result of each expression is
stored in the same cell of this Daf. The formulas are evaluated again and again
until no cell changes, so a cell may use the result of another. A circular
set of formulas raises `RuntimeError` after 100 passes.

In a formula, `$d` is this Daf, `$r` is the row number of the cell and `$c` is
its column number. References are absolute unless you build them from `$r` and
`$c`. So `sum($d[$r, :$c])` is the sum of the cells to the left in the same row.

Other examples:

    $d[14,20]+$d[15,25]       the sum of two cells
    max(0,$d[($r-1),$c])      the cell above, but not below 0
    $d[($r-1),$c] * 0.15      15 percent of the cell above

Warning: the formulas are run with `eval()`. Never use formulas from a source
you do not trust.

An error in a formula prints the cell and the formula, and is raised again.
The `retmode` of this Daf is then left as `val`.

Args:
    formulas_daf: The formulas, with the same shape as this Daf.

Raises:
    RuntimeError: The shapes differ, or the formulas never settle.

Examples:
    >>> d = Daf(cols=['A', 'B', 'C'], lol=[[1, 2, 0], [4, 5, 0], [7, 8, 0], [0, 0, 0]])
    >>> f = Daf(cols=['A', 'B', 'C'], lol=[
    ...     ['', '', 'sum($d[$r,:$c])'],
    ...     ['', '', 'sum($d[$r,:$c])'],
    ...     ['', '', 'sum($d[$r,:$c])'],
    ...     ['sum($d[:$r,$c])', 'sum($d[:$r,$c])', 'sum($d[:$r,$c])']])
    >>> d.apply_formulas(f)
    >>> d.lol
    [[1, 2, 3], [4, 5, 9], [7, 8, 15], [12, 15, 27]]
'''

D['Daf.cols_to_dol'] = r'''
Make a lookup from the values of one column to the values of another.

For each value in `colname1`, the result lists the different values that appear
with it in `colname2`, in the order first seen. Use it to see how two columns
relate. The values must be hashable. If a name is not a column, or the Daf is
empty, the result is empty.

Args:
    colname1: The column of keys.
    colname2: The column of values.

Returns:
    A dict that maps each value of the first column to a list of values of the second.

Examples:
    >>> d = Daf(lol=[['a', 'b'], ['b', 'd'], ['a', 'f'], ['b', 'd']], cols=['c1', 'c2'])
    >>> d.cols_to_dol('c1', 'c2')
    {'a': ['b', 'f'], 'b': ['d']}
'''

D['Daf.insert_dif_row'] = r'''
Insert a row that holds the difference of two rows.

The difference is the first row minus the second row, for the numeric columns
you name. Other columns of the new row are empty. A cell that is empty counts
as 0. By default the second row is the one after the first, and the new row goes
between them. Columns that hold text must not be in `cols`.

Args:
    irow1: The position of the first row.
    irow2: The position of the second row. If None, the row after the first.
    irow_insert: Where to insert the new row. If None, at `irow2`.
    cols: The columns to subtract. If None, all columns.

Returns:
    This Daf, which has been changed.

Examples:
    >>> d = Daf(lol=[[1, 10], [3, 14], [6, 20]], cols=['a', 'n'])
    >>> d.insert_dif_row(0).lol
    [[1, 10], [-2, -4], [3, 14], [6, 20]]
'''

D['Daf.insert_dif_rows'] = r'''
Insert a difference row after each of several rows.

For each position, a row is inserted that holds that row minus the next row,
as in `insert_dif_row()`. Do not list the last row. By default all rows are
used. With `offset=1` the new row goes after the next row, not between them.

Args:
    irows_rli: The positions of the first rows. If None, every row but the last.
    cols: The columns to subtract. If None, all columns.
    offset: 0 inserts between the two rows. 1 inserts after the second.

Returns:
    This Daf, which has been changed.

Examples:
    >>> d = Daf(lol=[[1, 10], [3, 14], [6, 20]], cols=['a', 'n'])
    >>> d.insert_dif_rows().lol
    [[1, 10], [-2, -4], [3, 14], [-3, -6], [6, 20]]
'''

D['Daf.annotate_daf'] = r'''
Copy fields from another Daf into this one, row by row, matching on the key.

Both Daf instances need a keyfield. For each row here, the row with the same
key in `other_daf` is found, and each field of this row named in
`my_to_other_dict` gets the value of the other field. A key that is missing
in `other_daf` raises `KeyError`. Name existing columns. A name that is not
a column adds a value to the rows, but not a column name.

Args:
    other_daf: The Daf to copy from.
    my_to_other_dict: Maps the field to set here to the field to read there.

Returns:
    This Daf, which has been changed.

Raises:
    KeyError: A keyfield is not set, or a key is not found in `other_daf`.

Examples:
    >>> a = Daf(lol=[[1, 'x'], [2, 'y']], cols=['id', 'v'], keyfield='id')
    >>> o = Daf(lol=[[1, 'P'], [2, 'Q']], cols=['id', 'w'], keyfield='id')
    >>> a.annotate_daf(o, {'v': 'w'}).lol
    [[1, 'P'], [2, 'Q']]
'''

D['Daf.apply'] = r'''
Apply a function to each row and collect the results in a new Daf.

The function gets a row and returns the new row as a dict. If it returns an
empty dict or None, that row is left out. The new Daf takes its columns from
the first row returned. It has no keyfield. This Daf is not changed.

With `by='table'` the function gets the whole Daf, and its result is returned
as it is. Use that to run any function on the table. `by='col'` is not
supported.

The function gets the rows as dicts, or as KeyedList objects if `itermode` is
`keyedlist`. The extra keyword arguments are passed on to it. To work on only
some rows, give the `keylist`, or select them first.

Args:
    func: The function to apply. It takes a row, or the Daf, and the keyword arguments.
    by: `row` to apply to each row, or `table` to apply to the whole Daf.
    keylist: Keys of the rows to include. All rows if None. This may be removed.
    **kwargs: Keyword arguments passed on to the function.

Returns:
    The new Daf, or the result of the function for `by='table'`.

Raises:
    NotImplementedError: `by` is `col`, or is not recognized.

Examples:
''' + EXG + r'''
    >>> d.apply(lambda row: {'g': row['g'].upper(), 'z': row['y'] * 2}).lol
    [['A', 20], ['B', 40], ['A', 60]]
    >>> d.apply(lambda row: row if row['y'] > 15 else None).lol
    [['b', 2, 20], ['a', 3, 30]]
'''

D['Daf.update_row'] = r'''
Update a row with the items of a dict, and return the row.

This is a small helper to use inside `apply()`, as in
`d.apply(lambda row: Daf.update_row(row, {'z': 0}))`. It is a static method.

Args:
    row: The row, as a dict. It is changed.
    da: The items to put in the row.

Returns:
    The same row.

Examples:
    >>> Daf.update_row({'a': 1}, {'b': 2})
    {'a': 1, 'b': 2}
'''

D['Daf.apply_in_place'] = r'''
Apply a function to each row and store the results in this Daf.

With `by='row'` the function gets each row as a dict. It must return a row. The
values of the returned dict are stored as the new row, in the order of its
keys, so the keys must be the columns in their order. A dict that is
shorter, longer or in a different order puts values in the wrong columns.

With `by='row_klist'` the function gets each row as a
[KeyedList][daffodil.keyedlist.KeyedList]. It changes the row and returns
nothing. It cannot put values in the wrong column.

The key index is rebuilt when it is next needed. Use `apply()` to get a new Daf.

Args:
    func: The function to apply to each row. It takes a row and the keyword arguments.
    by: `row` or `row_klist`.
    rowkeys: Keys of the rows to include. All rows if None. The Daf needs a keyfield.
    **kwargs: Keyword arguments passed on to the function.

Raises:
    ValueError: With `by='row'` the function returned None.
    NotImplementedError: `by` is not `row` or `row_klist`.

Examples:
''' + EXG + r'''
    >>> d.apply_in_place(lambda row: {**row, 'y': row['y'] + 1})
    >>> d.col('y')
    [11, 21, 31]
    >>> d.apply_in_place(lambda row: row.__setitem__('y', 0), by='row_klist')
    >>> d.col('y')
    [0, 0, 0]
'''

D['Daf.manifest_apply'] = r'''
Run a function on each chunk that a manifest lists, and save the results.

A manifest is a Daf in which each row describes one chunk of data. For each
row, `load_func` loads the chunk as a Daf. `func` is applied to it with
`by='table'`, so it gets the loaded Daf and the keyword `cols`. It returns a
tuple of a dict that describes the result chunk and the new Daf. `save_func`
saves the new Daf. The result manifest has one row for each dict.

Args:
    func: Gets a loaded Daf. Returns a dict that describes the result and the new Daf.
    load_func: Loads the chunk that a manifest row describes.
    save_func: Saves a new Daf, given the dict that describes it.
    by: Must be `table`.
    cols: Passed to `func` as the keyword `cols`.
    **kwargs: Keyword arguments passed on to `func`.

Returns:
    The manifest of the result chunks.
'''

D['Daf.manifest_reduce'] = r'''
Reduce the chunks that a manifest lists into one row.

Each chunk is loaded with `load_func` and reduced with `reduce()`. The
reductions are put in a Daf, and that is reduced again with the same function.
This works for functions such as `sum_da()` that can be combined in this way.

Args:
    func: The reduction function. See `reduce()`.
    load_func: Loads the chunk that a manifest row describes. It is required.
    by: How the function is applied. See `reduce()`.
    cols: The columns to reduce. All columns if None.
    **kwargs: Keyword arguments passed on to the function.

Returns:
    The reduced row, as a dict.

Raises:
    ValueError: No `load_func` is given.

Examples:
    >>> store = {'c1': Daf(lol=[[1, 2], [3, 4]], cols=['a', 'b']), 'c2': Daf(lol=[[10, 20]], cols=['a', 'b'])}
    >>> manifest = Daf(lol=[['c1'], ['c2']], cols=['chunk'])
    >>> manifest.manifest_reduce(Daf.sum_da, load_func=lambda spec: store[spec['chunk']])
    {'a': 14, 'b': 26}
'''

D['Daf.manifest_process'] = r'''
Call a function once for each chunk that a manifest lists.

The function gets the manifest row as a dict, and does its own loading and
saving. It returns a dict of information about what it did. These dicts are
collected as the rows of the result.

Args:
    func: Gets a manifest row and returns a dict of results.
    **kwargs: Keyword arguments passed on to the function.

Returns:
    A Daf with one row for each returned dict.

Examples:
    >>> manifest = Daf(lol=[['c1'], ['c2']], cols=['chunk'])
    >>> manifest.manifest_process(lambda spec: {'chunk': spec['chunk'], 'seen': True}).lol
    [['c1', True], ['c2', True]]
'''

D['Daf.groupby'] = r'''
Split the Daf into several Daf instances, one for each value of a column.

The result is a dict. Each key is a value found in the column, in the order
first seen. Each value is a Daf of the rows that have it, with all columns.
The rows are not copied, so changing a cell in a group changes it here too.

With several columns, as a list, the keys are tuples of their values. See
`groupby_cols()`. With `omit_nulls`, rows that have an empty value in the
column are left out.

Args:
    colname: The column to group by.
    colnames: Several columns to group by. Use this or `colname`.
    omit_nulls: If True, leave out rows that have an empty value.

Returns:
    A dict that maps each value, or tuple of values, to a Daf.

Examples:
''' + EXG + r'''
    >>> groups = d.groupby('g')
    >>> list(groups), groups['a'].lol
    (['a', 'b'], [['a', 1, 10], ['a', 3, 30]])
'''

D['Daf.groupby_cols'] = r'''
Split the Daf by the values of several columns.

The result is a dict. Each key is a tuple of the values in the columns, even
for one column. Each value is a Daf of the rows that have them. The rows are
not copied.

Args:
    colnames: The columns to group by.

Returns:
    A dict that maps each tuple of values to a Daf.

Examples:
''' + EXG + r'''
    >>> list(d.groupby_cols(['g']))
    [('a',), ('b',)]
'''

D['Daf.group_where'] = r'''
Group the rows by the result of a function.

The function gets each row and returns a key, or a list of keys, or None. A row
goes into the group of each key. A list of keys puts the same row in several
groups. None leaves the row out. The groups are Daf instances in a dict. The rows
are not copied, so changing a cell in a group changes it here too.

Args:
    where: The function. It gets a row and returns None, a key, or an iterable of keys.
    indirect_col: A column that holds a dict, to read names that are not columns from.

Returns:
    A dict that maps each key to a Daf.

Examples:
''' + EXG + r'''
    >>> groups = d.group_where(lambda row: [row['g'], 'all'])
    >>> {key: len(group) for key, group in groups.items()}
    {'a': 2, 'all': 3, 'b': 1}
'''

D['Daf.groupby_cols_reduce'] = r'''
Group the rows by several columns and reduce each group to one row.

Use it to total the numbers for each combination of a few identifying columns.
The result has one row for each combination. It has the group columns first and
then the `reduce_cols`. It has no keyfield. For each group, `func` is used as in
`reduce()`.

For example, group by gender, religion and zip code, and sum the counts of
several causes in each group. The number of rows is the number of combinations
that occur.

Args:
    groupby_colnames: The columns that identify a group.
    func: The reduction function. See `reduce()`.
    by: How the function is applied. See `reduce()`.
    reduce_cols: The columns to reduce.
    diagnose: If True, print progress messages.
    **kwargs: Keyword arguments passed on to the function.

Returns:
    The Daf with one row for each group.

Examples:
''' + EXG + r'''
    >>> d.groupby_cols_reduce(['g'], Daf.sum_da, reduce_cols=['y']).lol
    [['a', 40], ['b', 20]]
'''

D['Daf.groupby_reduce'] = r'''
Group the rows by one column and reduce each group to one row.

The result has one row for each value in the column, and its keyfield is that
column. The columns in `reduce_cols` hold the reduced values. Other columns are
empty. For each group, `func` is used as in `reduce()`.

Args:
    colname: The column to group by.
    func: The reduction function. See `reduce()`.
    by: How the function is applied. See `reduce()`.
    reduce_cols: The columns to reduce. All except `colname` if None.
    diagnose: If True, print progress messages.
    **kwargs: Keyword arguments passed on to the function.

Returns:
    The Daf with one row for each group.

Examples:
''' + EXG + r'''
    >>> d.groupby_reduce('g', Daf.sum_da, reduce_cols=['y']).lol
    [['a', '', 40], ['b', '', 20]]
'''

D['Daf.multi_groupby'] = r'''
Group the rows by each of several columns, one column at a time.

The result is a dict of dicts. The first key is the column. The second key is a
value in that column. Each innermost value is a Daf of the rows that have it.
This is not a grouping by combinations. Use `groupby_cols()` for that.
The groups are not reduced.

Args:
    groupby_colnames: The columns to group by.
    colnames: Not used.
    omit_nulls: If True, leave out rows that have an empty value.

Returns:
    A dict that maps each column to a dict of value and Daf.

Examples:
''' + EXG + r'''
    >>> groups = d.multi_groupby(['g', 'x'])
    >>> list(groups), list(groups['g'])
    (['g', 'x'], ['a', 'b'])
'''

D['Daf.reduce_dodaf_to_daf'] = r'''
Reduce each Daf in a dict to one row, and join the rows in a Daf.

This is the second half of `groupby_reduce()`. The dict maps the values of a
column to Daf instances. Each Daf is reduced with `reduce()`. The value is
stored in the column `colname` of the row. The result has `colname` as its keyfield.

Args:
    colname: The column that holds the value of each group.
    func: The reduction function. See `reduce()`.
    grouped_dodaf: A dict that maps each value to a Daf.
    reduce_cols: The columns to reduce. All columns except `colname` if None.
    diagnose: If True, print progress messages.
    **kwargs: Keyword arguments passed on to the function.

Returns:
    The Daf with one row for each group.
'''

D['Daf.multi_groupby_reduce'] = r'''
Group by each of several columns and reduce each group to one row.

This is `multi_groupby()` followed by `groupby_reduce()` for each column. The
result is a dict. Each key is a column. Each value is a Daf with one row for
each value of that column, and that column as its keyfield.

Args:
    colnames: The columns to group by, one at a time.
    func: The reduction function. See `reduce()`.
    by: How the function is applied. See `reduce()`.
    reduce_cols: The columns to reduce.
    diagnose: If True, print progress messages.
    **kwargs: Keyword arguments passed on to the function.

Returns:
    A dict that maps each column to its Daf.

Examples:
''' + EXG + r'''
    >>> d.multi_groupby_reduce(['g'], Daf.sum_da, reduce_cols=['y'])['g'].lol
    [['a', '', 40], ['b', '', 20]]
'''

D['Daf.daf_sum'] = r'''
Add up the columns, using `reduce()` and `sum_da()`.

Cells that cannot be added, such as text, are skipped. A column that is not
in `cols` is empty in the result.

Args:
    by: How the function is applied. See `reduce()`.
    cols: The columns to sum. All if None.
    indirect_col: A column that holds a dict. Required for `sparse_row`.
    **kwargs: Keyword arguments passed on to `reduce()`.

Returns:
    A dict with a total for each column.

Examples:
''' + EXG + r'''
    >>> d.daf_sum(cols=['y'])
    {'g': '', 'x': '', 'y': 60}
'''

D['Daf.reduce'] = r'''
Combine all the rows into one result, using a function.

The function gets each row, and the result so far, and returns the new result.
A sum or a count is a reduction. With `by='row'`, the function is called as
`func(row, result, cols=cols, **kwargs)`. The result starts as 0 for each of
the columns, or as `initial_da`. It returns a dict with every column. Columns
that are not in `cols` are empty.

With `by='col'` the function gets each column as a list and the result so far
as a list. With `by='table'` it gets this Daf and `cols`, and its result is
returned as it is. With `by='sparse_row'`, rows are read from the dict in
`indirect_col`, and the result starts as `initial_da` or an empty dict.

An error in the function stops the reduction. With `silent_error=True` the
row is skipped. The result may then be missing rows, with no sign of it. An
empty Daf gives an empty dict for `by='row'`.

To reduce part of the table, select the rows or columns first, or give `cols`.

Args:
    func: The function that combines. For example `Daf.sum_da`.
    by: `row`, `col`, `table` or `sparse_row`.
    cols: The columns included in the reduction. All columns if None.
    initial_da: The result to start from, instead of zeros.
    indirect_col: A column that holds a dict. Required for `sparse_row`.
    silent_error: If True, a row for which the function raises is skipped.
    **kwargs: Keyword arguments passed on to the function.

Returns:
    A dict for `row`, `table` and `sparse_row`. A list for `col`.

Raises:
    ValueError: `sparse_row` is used with no `indirect_col`.
    NotImplementedError: `by` is not recognized.

Examples:
''' + EXG + r'''
    >>> d.reduce(Daf.sum_da, cols=['x', 'y'])
    {'g': '', 'x': 6, 'y': 60}
'''

D['Daf.sum_da'] = r'''
Add the values of a row to a running total. Use it with `reduce()`.

This is a static method. A value that cannot be added, such as text or an empty
cell, is skipped. The total is changed and returned. With `astype`, each value
is first converted to that type.

Args:
    row_da: The current row.
    reduction_da: The running total. It is changed.
    cols: The columns to add. All if None.
    astype: A type to convert each value to before adding.
    is_sparse: If True, the row may hold only some of the columns.
    diagnose: Not used.

Returns:
    The running total.

Examples:
    >>> Daf.sum_da({'x': 1, 'y': 2}, {'x': 10, 'y': 0}, cols=['x', 'y'])
    {'x': 11, 'y': 2}
'''

D['Daf.daf_valuecount'] = r'''
Count how often each value occurs, in each column, using `reduce()`.

Args:
    by: How the function is applied. See `reduce()`.
    cols: The columns to count. All if None.

Returns:
    A dict that maps each column to a dict of value and count. A column that is
    not in `cols` is empty.

Examples:
''' + EXG + r'''
    >>> d.daf_valuecount(cols=['g'])['g']
    {'a': 2, 'b': 1}
'''

D['Daf.groupsum_daf'] = r'''
Group by a column and add up the other columns of each group.

This is `groupby_reduce()` with `sum_da()`.

Args:
    colname: The column to group by.
    by: How the function is applied. See `reduce()`.
    reduce_cols: The columns to add.

Returns:
    The Daf with one row for each group.

Examples:
''' + EXG + r'''
    >>> d.groupsum_daf('g', reduce_cols=['y']).lol
    [['a', '', 40], ['b', '', 20]]
'''

D['Daf.multi_groupsum'] = r'''
Group by each of several columns, one at a time, and add up the columns.

This is `multi_groupby_reduce()` with `sum_da()`.

Args:
    colnames: The columns to group by. These are required.
    by: How the function is applied. See `reduce()`.
    reduce_cols: The columns to add.

Returns:
    A dict that maps each column to a Daf with one row for each value.

Raises:
    ValueError: No `colnames` are given.
'''

D['Daf.set_col2_from_col1_using_regex_select'] = r'''
Fill a column with the part of another column that a regex selects, in place.

`regex` must have one pair of parentheses around the part to keep. A cell that
does not match gives an empty cell. Give `regex` as a keyword. Without `col2`,
`col1` is changed. If `col2` is not a column, the values are added to the
rows, but not a column name.

Args:
    col1: The column to read.
    col2: The column to write. Defaults to `col1`.
    regex: A regular expression with one group.

Examples:
    >>> d = Daf(lol=[[1, 'ab12'], [2, 'cd34']], cols=['id', 's'])
    >>> d.set_col2_from_col1_using_regex_select('s', regex=r'(\d+)')
    >>> d.col('s')
    ['12', '34']
'''

D['Daf.apply_replace_regex'] = r'''
Change a column with a pattern of the form `/find/replace/`, in place.

The pattern has three parts that are separated by `/`. The first is a regular
expression. The second is what replaces the part it finds. The groups it
finds can be used as `\1`. A pattern with an empty second part removes the text.

    /find//                       remove
    /find/replace/                replace
    /pre(select)post/\1/          keep only the selected part
    /pre(select)post/a\1b/        keep it, with new text around it

The result goes to `col2`, or to `col` if there is no `col2`. A column that is
not found does nothing. If `col2` is not a column, the values are added to the
rows, but not a column name.

Args:
    col: The column to read.
    col2: The column to write. Defaults to `col`.
    replace_regex: The pattern.

Returns:
    This Daf, which has been changed.

Examples:
    >>> d = Daf(lol=[[1, 'ab12']], cols=['id', 's'])
    >>> d.apply_replace_regex('s', replace_regex='/ab//').col('s')
    ['12']
'''

D['Daf.alter_daf_per_setting'] = r'''
Change this Daf with the replace patterns found in a settings dict.

The setting is a dict, or a list of dicts. Each has a `spec_name`, a `colname`
and a `replace_regex`. The ones whose fields match `setting_select_dict` are
used. For each, `apply_replace_regex()` changes the column. Use it to give
different edits to different files, such as fixing ids in one source.

A setting that is None or empty changes nothing. A name that is not in the
settings dict raises `KeyError`, unless `silent_error` is True.

Args:
    settingsdict: A dict of settings.
    setting_name: The key of the setting in the dict.
    setting_select_dict: Selects which specs apply, such as `{'spec_name': 'file1.csv'}`.
    silent_error: If True, a missing setting changes nothing.

Returns:
    This Daf, which has been changed.

Raises:
    KeyError: The setting is missing and `silent_error` is False.

Examples:
    >>> d = Daf(lol=[['04_1'], ['05_2']], cols=['ballot_id'])
    >>> spec = {'spec_name': 'a.zip', 'colname': 'ballot_id', 'replace_regex': r'/^04_/14_/'}
    >>> d.alter_daf_per_setting({'fix': [spec]}, 'fix', {'spec_name': 'a.zip'}).col('ballot_id')
    ['14_1', '05_2']
'''

D['Daf.alter_daf_per_alter_specs_daf'] = r'''
Change this Daf with the replace patterns listed in a Daf.

Each row of `alter_specs_daf` has a `colname` and a `replace_regex`. The
pattern is applied to the column with `apply_replace_regex()`. Give it only
the rows that apply, for example by selecting on `spec_name` first.

Args:
    alter_specs_daf: The specs, with the columns `colname` and `replace_regex`.

Returns:
    This Daf, which has been changed.

Examples:
    >>> d = Daf(lol=[['04_1'], ['05_2']], cols=['ballot_id'])
    >>> specs = Daf.from_lod([{'colname': 'ballot_id', 'replace_regex': r'/^04_/14_/'}])
    >>> d.alter_daf_per_alter_specs_daf(specs).col('ballot_id')
    ['14_1', '05_2']
'''

D['Daf.apply_to_col'] = r'''
Replace each value of a column by the result of a function, in place.

Args:
    col: The column name.
    func: A function that takes a value and returns the new value.
    **kwargs: Passed to `map()`.

Examples:
    >>> d = Daf(lol=[[1, 5], [2, 6]], cols=['a', 'b'])
    >>> d.apply_to_col('b', lambda value: value * 2)
    >>> d.col('b')
    [10, 12]
'''

D['Daf.diff_da'] = r'''
Subtract one dict from another, for the keys you name.

This is a static method. A key that is missing, or an empty value, counts as
0. Keys that you do not name are left out. The values must be numbers.

Args:
    d1_da: The first dict.
    d2_da: The dict to subtract.
    keys: The key or keys to include. If None, the result is empty.

Returns:
    A dict of the differences.

Examples:
    >>> Daf.diff_da({'a': 5, 't': 'x'}, {'a': 2, 't': 'y'}, keys=['a'])
    {'a': 3}
'''

D['Daf.count_values_da'] = r'''
Add one row to running counts of the values in each column. Use it with `reduce()`.

This is a static method. The counts are a dict that maps each column to a dict
of value and count. The counts are changed and returned. A cell that holds
a list is collected into a list. A cell that holds a dict is added to the
counts for that column with `sum_da()`. Use `omit_nulls` to skip empty cells.

Args:
    row_da: The current row.
    reduction_da: The running counts. They are changed.
    cols: The columns to count.
    omit_nulls: If True, empty cells are not counted.

Returns:
    The running counts.

Examples:
    >>> Daf.count_values_da({'g': 'a'}, {}, ['g'])
    {'g': {'a': 1}}
'''

D['Daf.sum_dodis'] = r'''
Add one dict of dicts of numbers into another, in place.

This is a static method. For each key, the numbers of the inner dicts are
added with `sum_da()`. A key that is new is stored as it is, not copied.

Args:
    this_dodi: The counts to add.
    accum_dodi: The running totals. They are changed.

Examples:
    >>> total = {'c': {'x': 2, 'y': 1}}
    >>> Daf.sum_dodis({'c': {'x': 1}}, total)
    >>> total
    {'c': {'x': 3, 'y': 1}}
'''

D['Daf.sum'] = r'''
Total the columns, and return a dict of the totals.

Each total starts as a float. Empty cells are skipped. A cell that is text
that is not a number raises `ValueError`, so give `colnames_ls` to leave
those columns out. With `numeric_only`, and dtypes of `int` or `float`, only
those columns are totaled, and a cell that is not a number counts as 0. The
totals are converted to the dtypes, if the Daf has them.

Args:
    colnames_ls: The columns to total. All if None.
    numeric_only: If True, total only the columns with an `int` or `float` dtype.

Returns:
    A dict that maps each column name to its total.

Raises:
    ValueError: A cell cannot be converted to a number.

Examples:
    >>> Daf(lol=[[1, 10], [2, 20]], cols=['x', 'y']).sum()
    {'x': 3.0, 'y': 30.0}
'''

D['Daf.sum_np'] = r'''
Total the columns with NumPy, and return a dict of the totals.

This needs NumPy. Use `colnames_ls` to pass only the columns that hold numbers.
A column with text or empty cells makes NumPy raise an error, so this
does not skip empty cells as `sum()` does.

Args:
    colnames_ls: The columns to total. All if None.

Returns:
    A dict that maps each column name to its total. An empty Daf gives an empty dict.

Examples:
    >>> Daf(lol=[[1, 10], [2, 20]], cols=['x', 'y']).sum_np()
    {'x': 3, 'y': 30}
'''

D['Daf.valuecounts_for_colname'] = r'''
Count how often each value occurs in one column.

With `sort=True` the dict is ordered from the most common value to the least.
Use `reverse=False` for the other way. A column that does not exist gives an
empty dict. An empty cell is counted as the empty string, unless
`omit_nulls` is True.

Args:
    colname: The column to count.
    sort: If True, order by count.
    reverse: With `sort`, True puts the most common first.
    omit_nulls: If True, leave out the count of empty cells.

Returns:
    A dict that maps each value to its count.

Examples:
    >>> d = Daf(lol=[['a'], ['b'], ['a'], ['']], cols=['g'])
    >>> d.valuecounts_for_colname('g', sort=True, omit_nulls=True)
    {'a': 2, 'b': 1}
'''

D['Daf.valuecounts_for_colnames_ls'] = r'''
Count how often each value occurs, in each of several columns.

Args:
    colnames_ls: The columns to count. All if None.
    sort: If True, order each count by size.
    reverse: With `sort`, True puts the most common first.
    omit_nulls: If True, leave out the count of empty cells.

Returns:
    A dict that maps each column to a dict of value and count.

Examples:
    >>> d = Daf(lol=[['a', 'x'], ['b', 'x']], cols=['g', 'h'])
    >>> d.valuecounts_for_colnames_ls()
    {'g': {'a': 1, 'b': 1}, 'h': {'x': 2}}
'''

D['Daf.valuecounts_for_colname_selectedby_colname'] = r'''
Count the values of a column, in the rows where another column has a value.

Args:
    colname: The column to count.
    selectedby_colname: The column to test.
    selectedby_colvalue: Only rows where that column equals this are counted.
    sort: If True, order the counts by size.
    reverse: With `sort`, True puts the most common first.

Returns:
    A dict that maps each value to its count. It is empty if a column does not exist.

Examples:
    >>> d = Daf(lol=[['a', 'x'], ['b', 'x'], ['a', 'y']], cols=['g', 'h'])
    >>> d.valuecounts_for_colname_selectedby_colname('g', 'h', 'x')
    {'a': 1, 'b': 1}
'''

D['Daf.valuecounts_for_colnames_ls_selectedby_colname'] = r'''
Count the values of several columns, in the rows where another column has a value.

Args:
    colnames_ls: The columns to count. All if None.
    selectedby_colname: The column to test.
    selectedby_colvalue: Only rows where that column equals this are counted.
    sort: If True, order the counts by size.
    reverse: With `sort`, True puts the most common first.

Returns:
    A dict that maps each column to a dict of value and count.
'''

D['Daf.valuecounts_for_colname1_groupedby_colname2'] = r'''
Count the values of one column, for each value of another.

The data is read once. Use it to see whether two columns relate one to one: each
group should then hold a single value.

Args:
    colname1: The column whose values are counted.
    groupedby_colname2: The column whose values form the groups.
    sort: If True, order each count by size.
    reverse: With `sort`, True puts the most common first.

Returns:
    A dict that maps each value of the second column to a dict of value and count.

Examples:
    >>> d = Daf(lol=[['a', 'x'], ['b', 'x'], ['a', 'y']], cols=['g', 'h'])
    >>> d.valuecounts_for_colname1_groupedby_colname2('g', 'h')
    {'x': {'a': 1, 'b': 1}, 'y': {'a': 1}}
'''

D['Daf.value_counts_daf'] = r'''
Make a Daf that lists each value of a column and its count.

The columns are the name of the column, and `counts`. There is no keyfield.
With `include_total` a last row holds the total.

Args:
    colname: The column to count.
    sort: If True, order by count.
    reverse: With `sort`, True puts the most common first.
    include_total: If True, add a row with the total.
    omit_nulls: If True, leave out the count of empty cells.

Returns:
    The new Daf.

Examples:
    >>> d = Daf(lol=[['a'], ['b'], ['a']], cols=['g'])
    >>> d.value_counts_daf('g', sort=True).lol
    [['a', 2], ['b', 1]]
'''

D['Daf.gen_stats_daf'] = r'''
Work out statistics for columns, given a profile for each.

`col_def_lot` has a tuple for each column of interest. The tuple is the column
name, a type, a format, and a profile. The profile is one of `index`,
`attrib`, `file_paths`, `scalar` or `localidx`, and chooses what is measured.
An index looks for repeats, an attribute counts the values, and a scalar gives
the minimum, maximum, mean and standard deviation. The type and format are not
used.

Args:
    col_def_lot: A list of tuples of column name, type, format and profile.

Returns:
    A dict that maps each column name to its statistics, as a dict.

Raises:
    NotImplementedError: A profile is not one of the five.

Examples:
    >>> d = Daf(lol=[[1], [3]], cols=['n'])
    >>> d.gen_stats_daf([('n', int, '', 'scalar')])['n']['mean']
    2
'''

D['Daf.transpose'] = r'''
Turn rows into columns and columns into rows.

The result has one row for each column of this Daf. With `include_header=True`,
the first column of the result holds the column names of this Daf, and the
names given in `new_cols` or the default names must then include that column.
The default names are `key`, then `A`, `B` and so on. Without `include_header`
pass `new_cols` that has one name for each row of this Daf, or the names will
be one too many. The data is copied.

Args:
    new_keyfield: The keyfield of the result.
    new_cols: The names of the columns of the result.
    include_header: If True, the column names become the first column.

Returns:
    The new Daf.

Examples:
    >>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'])
    >>> d.transpose(include_header=True).lol
    [['id', 1, 2], ['v', 'a', 'b']]
'''

D['Daf.derive_join_translator'] = r'''
Work out how the columns of two Daf instances are named in a join.

The translator is a Daf with a row for each column of the result. Its
columns are `resolved_colname`, `source_name`, `source_colname` and
`is_keyfield`. A column name that both Daf instances have is given a suffix
with the name of its source, such as `name_daf1`, unless it is a shared
field. The names of the instances are `daf1` and `daf2` if they have none.
The keyfields are always shared.

`join()` calls this. Call it yourself to see the names, or to edit the
translator and give it back as `custom_translator_daf`.

Args:
    other_daf: The Daf to join with.
    shared_fields: Columns that both have and that appear only once.
    omit_other_cols: Columns of the other Daf to leave out.
    tag_other: If True, every column of the other Daf, except shared ones, gets a suffix.

Returns:
    The translator Daf. Its keyfield is `resolved_colname`.

Examples:
    >>> a = Daf(lol=[[1, 'x']], cols=['id', 'name'], keyfield='id')
    >>> b = Daf(lol=[[1, 'y']], cols=['id', 'name'], keyfield='id')
    >>> a.derive_join_translator(b).col('resolved_colname')
    ['id', 'name_daf1', 'name_daf2']
'''

D['Daf.derive_join_translator_daf'] = r'''
Work out a join translator from column names, with no Daf instances.

This is the form that does not need two Daf instances, so SQL joins can use
it. See `derive_join_translator()` for what the translator holds.

Args:
    self_keyfield: The keyfield of the first table.
    other_keyfield: The keyfield of the other table.
    self_cols: The column names of the first table.
    other_cols: The column names of the other table.
    self_name: The name of the first table.
    other_name: The name of the other table.
    shared_fields: Columns that both have and that appear only once. The list is not changed.
    omit_other_cols: Columns of the other table to leave out.
    tag_other: If True, every column of the other table, except shared ones, gets a suffix.

Returns:
    The translator Daf. Its keyfield is `resolved_colname`.
'''

D['Daf.join'] = r'''
Join two Daf instances on their keyfields, as in SQL.

Both need a keyfield, and it must be a single column. A row of this Daf is
joined with the row of `other_daf` that has the same key.

The types of join are:

    inner    only the keys that are in both.
    left     all keys of this Daf.
    right    all keys of the other Daf.
    outer    all keys of both.

Columns that only one Daf has are in the result. A column that both have
is given the name of its source as a suffix, such as `name_daf1`, unless it is
in `shared_fields`. The names are `daf1` and `daf2` if the instances have no
names. For other names use `custom_translator_daf`, which you can start from
`derive_join_translator()`.

When a key has no match, its cells from the other side are `None`, not NULL.
The keyfield of the result is the keyfield of this Daf. The result is a new Daf.

Args:
    other_daf: The Daf to join with.
    how: `inner`, `left`, `right` or `outer`.
    shared_fields: Columns that both have and that appear only once.
    tag_other: If True, every column of the other Daf, except shared ones, gets a suffix.
    custom_translator_daf: A translator that sets all the names.
    diagnose: If True, print progress messages.
    name: The name of the result.

Returns:
    The joined Daf.

Raises:
    ValueError: `how` is not one of the four.
    KeysDisabledError: A Daf has no keyfield.
    KeyError: A keyfield is a tuple.

Examples:
    >>> a = Daf(lol=[[1, 'Alice'], [2, 'Bob']], cols=['id', 'name'], keyfield='id')
    >>> b = Daf(lol=[[1, 50], [3, 70]], cols=['id', 'salary'], keyfield='id')
    >>> a.join(b).lol
    [[1, 'Alice', 50]]
    >>> a.join(b, how='left').lol
    [[1, 'Alice', 50], [2, 'Bob', None]]
'''

D['Daf.join_records'] = r'''
Combine one record from each table into one record, using a translator.

This is a static method, and the step that `join()` repeats. A record may be
None, which gives None for the columns of that side.

Args:
    records: The two records, in the order of the source names.
    translator_daf: The translator, as from `derive_join_translator()`.
    join_names_ls: The two source names. Needed only if the translator names more than two sources.

Returns:
    The combined record, as a dict.

Raises:
    ValueError: The translator names more than two sources and `join_names_ls` is not given.
'''

D['Daf.wide_to_narrow'] = r'''
Turn columns into rows. This is called melt or unpivot.

Each column that is not an id column gives one row for each row of this Daf.
The row holds the id values, the name of the column, and its value.

Args:
    id_cols: The columns that identify a row. They are kept as they are.
    varname_colname: The name of the new column that holds the old column names.
    value_colname: The name of the new column that holds the values.

Returns:
    The new Daf.

Raises:
    TypeError: `id_cols` is not a list.

Examples:
    >>> d = Daf(lol=[['x', 1, 2], ['y', 3, 4]], cols=['id', 'a', 'b'])
    >>> d.wide_to_narrow(['id']).lol
    [['x', 'a', 1], ['x', 'b', 2], ['y', 'a', 3], ['y', 'b', 4]]
'''

D['Daf.narrow_to_wide'] = r'''
Turn rows into columns. This is called pivot or spread.

The rows of one id must be next to each other. A new row of the result starts
whenever the id values change, so rows that are not sorted by id give a wrong
result with no error. The columns of the result are the id columns and then
the names in `varname_col`, in the order first seen. The `wide_cols` argument
is not used.

Args:
    id_cols: The columns that identify a row.
    varname_col: The column whose values become the new column names.
    value_col: The column whose values fill the new columns.
    wide_cols: Not used.

Returns:
    The new Daf.

Examples:
    >>> d = Daf(lol=[['x', 'a', 1], ['x', 'b', 2], ['y', 'a', 3], ['y', 'b', 4]], cols=['id', 'variable', 'value'])
    >>> d.narrow_to_wide(['id']).lol
    [['x', 1, 2], ['y', 3, 4]]
'''

D['Daf.md_daf_table_snippet'] = r'''
Make a short Markdown table of the Daf, with a summary line.

This is what `str()` shows. It keeps at most `md_max_rows` rows and
`md_max_cols` columns, 10 by default. Longer text is shortened to 80
characters.

Returns:
    The Markdown text.
'''

D['Daf.to_md'] = r'''
Make a Markdown table of the Daf.

Without limits the whole table is written. With `max_rows` or `max_cols`, the
first and last are kept, and the middle is replaced by `...`. Text longer
than `max_text_len` is shortened by cutting out its middle. `just` has one
character for each column: `<` left, `^` center, `>` right. The default is
right. With no column names, `A`, `B` and so on are used. With `include_summary`,
the Markdown can be read back with `from_md()`, as text.

Use `max_rows` and `max_cols` together, or neither. With only `max_cols`, a row of
`...` is added under the header by mistake.

Args:
    max_rows: The most rows to show. 0 for all.
    max_cols: The most columns to show. 0 for all.
    just: The justification of each column.
    shorten_text: If True, shorten text that is longer than `max_text_len`.
    max_text_len: The longest text to show in full.
    smart_fmt: If True, show numbers with fewer decimal places.
    include_summary: If True, add a line with the size, keyfield and name.
    disp_cols: Column names to show instead of the real ones.
    header: A header to use instead.

Returns:
    The Markdown text.

Examples:
    >>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'])
    >>> print(d.to_md())
    | id | v |
    | -: | -: |
    |  1 | a |
    |  2 | b |
    <BLANKLINE>
'''

D['Daf.to_md_cols'] = r'''
Make a Markdown table in which each row of the Daf is a column.

There is no header. The first column holds the column names of the Daf. Use it
for a Daf with few rows and many columns.

Args:
    max_rows: The most rows to show. 0 for all.
    max_cols: The most columns to show. 0 for all.
    just: The justification of each column.
    shorten_text: If True, shorten text that is longer than `max_text_len`.
    max_text_len: The longest text to show in full.
    smart_fmt: If True, show numbers with fewer decimal places.
    include_summary: Not used.
    disp_cols: Column names to show instead of the real ones.

Returns:
    The Markdown text.
'''

D['Daf.daf_to_lol_summary'] = r'''
Make a list of lists for display, with the column names first.

If there are more rows or columns than the limits, the first and last are
kept and the middle is replaced by `...`. Set both limits or neither. With only
`max_cols`, a row of `...` is added under the header by mistake. The rows are
not copied.

Args:
    max_rows: The most rows to keep. 0 for no limit.
    max_cols: The most columns to keep. 0 for no limit.
    disp_cols: Column names to use instead of the real ones.

Returns:
    The rows, with a header row first if there are column names.
'''

D['Daf.dict_to_md'] = r'''
Show a dict as a two column Markdown table, for looking at it.

This is a static method. Use it as `print(Daf.dict_to_md(my_da))`. The keys are
in the first column and the values in the second.

Args:
    da: The dict.
    cols: The two column names. Default `key` and `value`.
    just: The justification of the two columns.

Returns:
    The Markdown text.

Examples:
    >>> print(Daf.dict_to_md({'a': 1, 'b': 'two'}))
    | key | value |
    | :-- | :---- |
    | a   | 1     |
    | b   | two   |
    <BLANKLINE>
'''
setdoc('src/daffodil/daf.py', D)
