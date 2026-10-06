import sys
sys.path.insert(0, '/tmp/claude-0/-home-user-daffodil/95e36c09-64f8-59ba-9bd6-c962257ad9e3/scratchpad')
from setdoc import setdoc
D = {}
EX = """    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')"""

D['Daf.assign_record'] = r'''
Put one row in the Daf by its key, replacing a row that has the same key.

The row is a dict. If a row with that key exists, the whole row is replaced.
The cells of columns that the dict lacks become NULL. If the key is new, the
row is added at the end. This is an upsert for one row. For that, `append()`
with `respect_kd=True` also works. Use `update_by_keylist()` to change only
some cells.

Args:
    record: The row, as a dict. It must have the keyfield.

Raises:
    KeysDisabledError: The Daf has no keyfield.

Examples:
''' + EX + r'''
    >>> d.assign_record({'id': 2, 'v': 'new'})
    >>> d.lol
    [[1, 'a', 10], [2, 'new', ''], [3, 'c', 30]]
'''

D['Daf.assign_record_irow'] = r'''
Put one row in the Daf by position, replacing the row there.

The row is a dict. The whole row is replaced, and the cells of columns that
the dict lacks become NULL. With the default position, which is negative, or
with a position beyond the end, the row is added at the end instead. Use
`update_record_irow()` to change only some cells.

Args:
    irow: The row position. A negative or too large position adds the row at the end.
    record: The row, as a dict. If None, nothing happens.

Examples:
''' + EX + r'''
    >>> d.assign_record_irow(1, {'v': 'q'})
    >>> d.lol[1]
    ['', 'q', '']
'''

D['Daf.update_by_keylist'] = r'''
Change some cells in the rows that have the given keys.

Only the columns that are keys of the dict are changed. Other cells keep
their values. A key that is not found is skipped. This is a bulk form of
`my_daf[key, colname] = value`. The row positions do not change, so the key
index stays valid.

Args:
    keylist: The keys of the rows to change.
    record: The new values, as a dict of column name and value. Names that are not columns are ignored.

Returns:
    This Daf, which has been changed. With no keyfield, rows, keys or record, nothing happens.

Examples:
''' + EX + r'''
    >>> d.update_by_keylist([1, 3, 9], {'v': 'q'}).lol
    [[1, 'q', 10], [2, 'b', 20], [3, 'q', 30]]
'''

D['Daf.update_record_irow'] = r'''
Change some cells in the row at a position.

Only the columns that are keys of the dict are changed. Other cells keep their
values. A position that is out of range does nothing.

Args:
    irow: The row position.
    record: The new values, as a dict of column name and value. Names that are not columns are ignored.

Examples:
''' + EX + r'''
    >>> d.update_record_irow(1, {'v': 'q'})
    >>> d.lol[1]
    [2, 'q', 20]
'''

D['Daf.assign_icol'] = r'''
Fill a column by position with the values of a list.

A list that is too short is filled out with `default`. With no list, every cell
gets `default`. With `icol=-1` a new column is added at the right. Its name is
not added, so the Daf has more values in each row than names. Use `insert_col()`
to add a column with a name.

Args:
    icol: The column position. -1 adds a column at the right.
    col_la: The values, one for each row.
    default: The value for rows that the list does not reach.

Examples:
''' + EX + r'''
    >>> d.assign_icol(1, ['x', 'y'], default='D')
    >>> d.col('v')
    ['x', 'y', 'D']
'''

D['Daf.insert_icol'] = r'''
Insert a column at a position and move the later columns right.

A list that is too short is filled out with `default`. With `icol=-1`, or a
position beyond the last column, the column is added at the right. Give
`colname` to name it. Without a name the data is inserted, but the names are
not changed, so rows have more values than names. The dtypes are not changed.
Use `set_keyfield()` if the column is to be the keyfield.

Args:
    icol: The column position. -1 adds the column at the right.
    col_la: The values, one for each row.
    colname: The name of the new column.
    default: The value for rows that the list does not reach.

Returns:
    This Daf, which has been changed.

Examples:
''' + EX + r'''
    >>> d.insert_icol(1, ['x', 'y', 'z'], colname='new').columns()
    ['id', 'new', 'v', 'n']
'''

D['Daf.insert_irow'] = r'''
Insert a row at a position and move the later rows down.

The row is a list of values, or a dict that is placed by column name. A short
list is filled out with `default`. A position beyond the last row adds the row
at the end. The key index is rebuilt when it is next needed.

Args:
    irow: The row position. -1 adds the row at the end.
    row: The row, as a list or a dict.
    default: The value for cells that a short list does not reach.

Returns:
    This Daf, which has been changed.

Examples:
''' + EX + r'''
    >>> d.insert_irow(1, {'id': 9, 'v': 'z'}).lol
    [[1, 'a', 10], [9, 'z', ''], [2, 'b', 20], [3, 'c', 30]]
'''

D['Daf.assign_col'] = r'''
Fill a column by name with the values of a list, or add it if it is new.

This is `my_daf[:, colname] = values`, and it also adds a column. A list that
is too short is filled out with `default`. With no list, every cell gets
`default`. If the column is the keyfield, the key index is rebuilt when it is
next needed.

Args:
    colname: The column name.
    la: The values, one for each row.
    default: The value for rows that the list does not reach.

Returns:
    This Daf, which has been changed.

Examples:
''' + EX + r'''
    >>> d.assign_col('w', default=0).columns()
    ['id', 'v', 'n', 'w']
    >>> d.col('w')
    [0, 0, 0]
'''

D['Daf.insert_col'] = r'''
Add a named column at a position, or overwrite it if the name exists.

A list that is too short is filled out with `default`. With no list, every cell
gets `default`, so this can add a constant column. If the name already exists
the column is overwritten, and `icol` is ignored. An empty name does nothing.
Use `set_keyfield()` if the column is to be the keyfield.

Args:
    colname: The name of the column.
    col_la: The values, one for each row.
    icol: The column position. -1 adds the column at the right.
    default: The value for rows that the list does not reach.

Returns:
    This Daf, which has been changed.

Examples:
''' + EX + r'''
    >>> d.insert_col('w', ['x', 'y', 'z'], icol=1).columns()
    ['id', 'w', 'v', 'n']
    >>> d.insert_col('k', default=5).col('k')
    [5, 5, 5]
'''

D['Daf.insert_idx_col'] = r'''
Insert a column of row numbers.

Args:
    colname: The name of the new column.
    icol: The column position.
    startat: The number of the first row.

Returns:
    This Daf, which has been changed.

Examples:
''' + EX + r'''
    >>> d.insert_idx_col().col('idx')
    [0, 1, 2]
'''

D['Daf.set_col_irows'] = r'''
Set one value in the given rows of a named column.

This is `my_daf[irows, colname] = value`. A column name that is not found does
nothing. Row positions that are out of range are skipped.

Args:
    colname: The column name.
    irows: The row positions.
    val: The value to set.

Returns:
    This Daf, which has been changed.

Examples:
''' + EX + r'''
    >>> d.set_col_irows('v', [0, 2], 'Z').col('v')
    ['Z', 'b', 'Z']
'''

D['Daf.set_icol'] = r'''
Set one value in every row of a column, by position.

This is `my_daf[:, icol] = value`.

Args:
    icol: The column position.
    val: The value to set.

Returns:
    This Daf, which has been changed.

Raises:
    IndexError: The position is beyond the end of a row.

Examples:
''' + EX + r'''
    >>> d.set_icol(1, 'Z').col('v')
    ['Z', 'Z', 'Z']
'''

D['Daf.set_icol_irows'] = r'''
Set one value in the given rows of a column, by position.

This is `my_daf[irows, icol] = value`. Row positions that are out of range
are skipped.

Args:
    icol: The column position.
    irows: The row positions.
    val: The value to set.

Examples:
''' + EX + r'''
    >>> d.set_icol_irows(1, [0, 9, -1], 'Z')
    >>> d.col('v')
    ['Z', 'b', 'c']
'''

D['Daf.find_replace'] = r'''
Replace every cell that matches a pattern, in place.

Each cell is converted to text and searched with the regular expression
`find_pat`. If it matches anywhere in the text, the whole cell is replaced by
`replace_val`. It is not a substitution inside the text. Every column is
searched, including numbers, and the key index is rebuilt when it is next
needed.

Args:
    find_pat: A regular expression.
    replace_val: The value that replaces a matching cell.

Examples:
''' + EX + r'''
    >>> d.find_replace(r'^[ab]$', 'HIT')
    >>> d.col('v')
    ['HIT', 'HIT', 'c']
'''

D['Daf.replace_in_columns'] = r'''
Replace listed values with another value, in some columns, in place.

A cell is replaced when it equals any value in `find_values`. A typical use
is `['', None]` to fill empty cells. Columns may be given by name or by
position, or all columns if `cols` is None. With no `find_values` nothing
happens. The key index is rebuilt if the keyfield may have changed.

Args:
    cols: Column names or positions. If None, all columns.
    find_values: The values to look for.
    replacement: The value to put in their place. This is required.

Returns:
    This Daf, which has been changed.

Raises:
    ValueError: No `replacement` is given.
    KeyError: A column name is not found.
    TypeError: A column is neither a name nor a position.

Examples:
''' + EX + r'''
    >>> d.replace_in_columns(['v'], ['a', 'c'], '-').col('v')
    ['-', 'b', '-']
'''

D['Daf.split_daf_into_ranges'] = r'''
Split the rows into several Daf instances, by position ranges.

Each range is `(start, end)`, and `end` is not included. The new Daf instances
share their rows with this one.

Args:
    chunk_ranges: The ranges of row positions.

Returns:
    A list of Daf instances, one for each range.

Examples:
''' + EX + r'''
    >>> [part.lol for part in d.split_daf_into_ranges([(0, 2), (2, 3)])]
    [[[1, 'a', 10], [2, 'b', 20]], [[3, 'c', 30]]]
'''

D['Daf.split_daf_into_chunks_lodaf'] = r'''
Split the rows evenly into Daf instances of at most a given size.

The sizes are as equal as possible, so some chunks are smaller than the
maximum. None is larger. The chunks share their rows with this Daf.

Args:
    max_chunk_size: The most rows in one chunk.

Returns:
    A list of Daf instances.

Examples:
    >>> d = Daf(lol=[[i] for i in range(7)], cols=['a'])
    >>> [len(part) for part in d.split_daf_into_chunks_lodaf(3)]
    [3, 2, 2]
'''

D['Daf.sort_by_colname'] = r'''
Sort the rows by one column, in place.

Make a copy first if you need the original order. An empty cell sorts before
a cell with content, unlike a spreadsheet. Values in the column must be
comparable with each other, so a column that mixes None and numbers raises
`TypeError`.

With `length_priority`, a shorter text sorts before a longer one, so numbers
that are stored as text sort as numbers. Without it, `'10'`, `'100'`, `'8'`
are in text order.

Args:
    colname: The column to sort by.
    reverse: If True, sort from high to low.
    length_priority: If True, sort by length first, then by value.

Returns:
    This Daf, which has been changed.

Raises:
    KeyError: The column name is not found.

Examples:
    >>> d = Daf(lol=[['10'], ['99'], ['8'], ['100']], cols=['a'])
    >>> d.sort_by_colname('a', length_priority=True).col('a')
    ['8', '10', '99', '100']
'''

D['Daf.sort_by_colnames'] = r'''
Sort the rows by several columns, in place.

The first column is the main sort key. The others break ties. See
`sort_by_colname()` for the rules of ordering and for `length_priority`.
Calling `sort_by_colname()` for each column, last column first, gives the same
order.

Args:
    colnames: The columns to sort by, main key first.
    reverse: If True, sort from high to low.
    length_priority: If True, sort by length first, then by value.

Returns:
    This Daf, which has been changed.

Raises:
    KeyError: A column name is not found.

Examples:
    >>> d = Daf(lol=[[2, 'b'], [1, 'z'], [1, 'a']], cols=['p', 'q'])
    >>> d.sort_by_colnames(['p', 'q']).lol
    [[1, 'a'], [1, 'z'], [2, 'b']]
'''
setdoc('src/daffodil/daf.py', D)
