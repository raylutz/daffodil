import sys
sys.path.insert(0, '/tmp/claude-0/-home-user-daffodil/95e36c09-64f8-59ba-9bd6-c962257ad9e3/scratchpad')
from setdoc import setdoc
D = {}

EX = """    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')"""

D['Daf.__getitem__'] = r'''
Select rows, columns or cells, as in `my_daf[rows, cols]`.

The selector is `[rows]` or `[rows, cols]`. With one selector, all columns are
returned. Use `:` for all. A selector is one of these.

    integer       A position. Negative counts from the end.
    slice         `2:5`, `:3`, `::2`, as for a Python list.
    list          A list of positions, in the order given, or a list of ranges.
    range         A range of positions.
    string        A key of the keyfield for rows, or a column name for columns.
    list of str   Several keys, or several column names, in the order given.
    tuple         An inclusive range of keys, or of column names. Use None to
                  start at the first or to end at the last, as in `(None, 'r3')`.

A tuple of two items is read as `[rows, cols]` when it stands alone. To give a
range of row keys, add the column selector, as in `my_daf[('r1', 'r3'), :]`.

Integers are always positions. A keyfield or column names that are integers
cannot be used in brackets. Use `select_krows()` and `select_kcols()` instead.

The result is a new Daf. Its rows are shared with this Daf when you select
rows, so changing a cell in the result changes it here too. Selecting columns
makes new rows, so the result is independent. The keyfield and dtypes carry
over if their columns are still there. See [retmode][daffodil.daf.Daf.retmode]
for getting a bare value or list when the result is one cell, row or column.

A column slice works as it does for a Python list.

Args:
    slice_spec: A row selector, or a tuple of a row selector and a column selector.

Returns:
    A new Daf, or a value or list if `retmode` is `val`.

Raises:
    IndexError: A row or column position is out of range.
    KeyError: A key or column name is not found.
    KeysDisabledError: Rows are selected by key and there is no keyfield.
    TypeError: A selector is None, or has a type that is not accepted.

Examples:
''' + EX + r'''
    >>> d[1].lol
    [[2, 'b', 20]]
    >>> d[1:].lol
    [[2, 'b', 20], [3, 'c', 30]]
    >>> d[:, 'v'].lol
    [['a'], ['b'], ['c']]
    >>> d[[2, 0], ['n', 'id']].lol
    [[30, 3], [10, 1]]
    >>> d[1, 'n'].to_value()
    20
    >>> d[(1, 2), :].lol
    [[1, 'a', 10], [2, 'b', 20]]
'''

D['Daf.__setitem__'] = r'''
Assign values to a selection, as in `my_daf[rows, cols] = value`.

The selector is the same as for `[]`. The Daf is changed in place.

A single value fills every cell of the selection. A list fills the selection
in order. A dict given for a row sets that row from the dict. The cells whose
columns are not in the dict become NULL. See `set_irows_icols()` for what
happens when the source and the selection differ in size.

Assigning text to several whole rows at once does not work yet. Assign to a
column instead, as in `my_daf[:, 'v'] = 'x'`.

If you change a keyfield cell, the key index is rebuilt when it is next needed.

Args:
    slice_spec: A row selector, or a tuple of a row selector and a column selector.
    value: The value, list, dict or Daf to assign.

Examples:
''' + EX + r'''
    >>> d[1, 'v'] = 'z'
    >>> d[:, 'n'] = [1, 2, 3]
    >>> d.lol
    [[1, 'a', 1], [2, 'z', 2], [3, 'c', 3]]
    >>> d[0] = {'v': 'q'}
    >>> d.lol[0]
    ['', 'q', '']
'''

D['Daf.krows_to_irows'] = r'''
Turn row keys into row positions.

This is the step that lets `my_daf['r2']` work. Call it when you need the
positions themselves. The Daf must have a keyfield, or a key index passed in
as `kd`.

The keys may be a key, a list of keys, or a tuple that gives an inclusive range
of keys. A range is made from positions, so the keys must be in the same order
as the rows.

Args:
    krows: A key, a list of keys, or a tuple that gives a range of keys.
    inverse: If True, return the positions of the rows that are not selected.
    silent_error: If True, keys that are not found are ignored.

Returns:
    The row positions.

Raises:
    KeysDisabledError: There is no keyfield and no key index.
    KeyError: A key is not found and `silent_error` is False.
'''

D['Daf.kcols_to_icols'] = r'''
Turn column names into column positions.

This is the step that lets `my_daf[:, 'v']` work. A name that looks like an
integer is still read as a name.

Args:
    kcols: A name, a list of names, or a tuple that gives an inclusive range of names.
    inverse: If True, return the positions of the columns that are not selected.
    silent_error: If True, names that are not found are ignored.

Returns:
    The column positions. With no column names the result is empty, or all
    positions if `inverse` is True.

Raises:
    KeyError: A name is not found and `silent_error` is False.
'''

D['Daf.select_krows'] = r'''
Select rows by key. Rows with those keys are kept, or dropped if `inverse` is True.

This is the same as `my_daf[keys]`, with a choice to drop rows and to ignore
keys that are not found. It works with integer keys, which brackets cannot.
The rows are shared with this Daf. Use `copy()` if you need to change them
independently.

A bare tuple means an inclusive range of keys, so `(1, 2)` is the rows from key
1 through key 2. For a composite keyfield, give a list of tuples.

Args:
    krows: A key, a list of keys, or a tuple that gives a range of keys.
    inverse: If True, drop the selected rows and keep the others.
    silent_error: If True, keys that are not found are ignored.

Returns:
    The new Daf.

Raises:
    KeysDisabledError: The Daf has no keyfield.
    KeyError: A key is not found and `silent_error` is False.

Examples:
''' + EX + r'''
    >>> d.select_krows([3, 1]).lol
    [[3, 'c', 30], [1, 'a', 10]]
    >>> d.select_krows([1, 2], inverse=True).lol
    [[3, 'c', 30]]
    >>> d.select_krows([1, 9], silent_error=True).lol
    [[1, 'a', 10]]
'''

D['Daf.select_kcols'] = r'''
Select columns by name. Columns with those names are kept, or dropped if `inverse` is True.

This is the same as `my_daf[:, names]`, with more choices. The result is
a new Daf with new rows, in the order of the names you give. With `flip=True`
the selected columns become rows, and the result has no column names and no
keyfield. This costs no more than selecting the columns.

Selecting columns copies data, so it is not cheap. For `apply` and `reduce`,
use their `cols` argument instead.

Args:
    kcols: A name, a list of names, or a tuple that gives a range of names.
    inverse: If True, drop the named columns and keep the others.
    flip: If True, turn the selected columns into rows.
    silent_error: If True, names that are not found are ignored.

Returns:
    The new Daf.

Raises:
    KeysDisabledError: The Daf has no column names.
    KeyError: A name is not found and `silent_error` is False.

Examples:
''' + EX + r'''
    >>> d.select_kcols(['n', 'id']).lol
    [[10, 1], [20, 2], [30, 3]]
    >>> d.select_kcols('v', inverse=True).columns()
    ['id', 'n']
    >>> d.select_kcols('v', flip=True).lol
    [['a', 'b', 'c']]
'''

D['Daf.select_irows'] = r'''
Select rows by position. Those rows are kept, or dropped if `invert` is True.

This is the same as `my_daf[rows]`, with a choice to drop rows. It is cheap.
The new Daf holds the same row lists as this one, so changing a cell in
the result changes it here too. The exception is dropping an empty selection,
which makes a deep copy. The keyfield, dtypes and column names carry over.

Args:
    irows: A position, a slice, a range, a list of positions, or a list of ranges.
    invert: If True, drop the selected rows and keep the others.

Returns:
    The new Daf.

Raises:
    IndexError: A single position is out of range.

Examples:
''' + EX + r'''
    >>> d.select_irows(1).lol
    [[2, 'b', 20]]
    >>> d.select_irows(1, invert=True).lol
    [[1, 'a', 10], [3, 'c', 30]]
    >>> d.select_irows(slice(1, None)).lol
    [[2, 'b', 20], [3, 'c', 30]]
'''

D['Daf.select_icols'] = r'''
Select columns by position. Those columns are kept, in the order given.

This is the same as `my_daf[:, cols]`. It makes new rows, so it is not cheap.
For `apply` and `reduce`, use their `cols` argument instead. A slice works as
it does for a Python list. The keyfield and dtypes carry over if their columns
are kept. With `flip=True` the columns become rows, and the result has no
column names, no dtypes and no keyfield.

Args:
    icols: A position, a slice, a range, a list of positions, or a list of ranges.
    flip: If True, turn the selected columns into rows.

Returns:
    The new Daf.

Raises:
    IndexError: A position is beyond the end of some row.

Examples:
''' + EX + r'''
    >>> d.select_icols([2, 0]).lol
    [[10, 1], [20, 2], [30, 3]]
    >>> d.select_icols(slice(-2, None)).columns()
    ['v', 'n']
    >>> d.select_icols([0, 1], flip=True).lol
    [[1, 2, 3], ['a', 'b', 'c']]
'''

D['Daf.select_record'] = r'''
Get one row as a dict, by its key.

If the key is not found, the answer is an empty dict, unless `silent_error` is
False. Then a `KeyError` is raised. A typo in a key gives an empty dict, so
check it, or pass `silent_error=False`. An empty Daf gives an empty dict.
For a composite keyfield the key is a tuple.

Args:
    key: The key of the row.
    silent_error: If False, raise an error when the key is not found.

Returns:
    The row as a dict, or an empty dict.

Raises:
    KeysDisabledError: The Daf has rows but no keyfield and no key index.
    KeyError: The key is not found and `silent_error` is False.

Examples:
''' + EX + r'''
    >>> d.select_record(2)
    {'id': 2, 'v': 'b', 'n': 20}
    >>> d.select_record(9)
    {}
'''

D['Daf.select_records_daf'] = r'''
Select several rows by key and return them as a Daf.

This is `select_krows()` with a friendlier answer for an empty list of keys.
No keys gives an empty Daf, or all the rows if `inverse` is True.

Args:
    keys_ls: The keys of the rows.
    inverse: If True, drop the selected rows and keep the others.
    silent_error: If True, keys that are not found are ignored.

Returns:
    The new Daf.

Raises:
    KeysDisabledError: The Daf has no keyfield and no key index.
    KeyError: A key is not found and `silent_error` is False.
'''

D['Daf.irow_la'] = r'''
Get one row as a list, by position.

The list is the row itself, not a copy. Changing it changes the Daf. Use
`iloc()` with `rtype='list'` for a copy.

Args:
    irow: The row position.

Returns:
    The row.

Raises:
    IndexError: The position is out of range.
'''

D['Daf.to_value'] = r'''
Get the one value of a Daf that has one row and one column.

Use it on the result of a selection, as in `my_daf[1, 'n'].to_value()`.

Args:
    default: Returned if the Daf is not one cell. Without it, an error is raised.
    astype: A type or function to convert the value with.

Returns:
    The value.

Raises:
    ValueError: The Daf is not one cell and no default is given.

Examples:
''' + EX + r'''
    >>> d[1, 'n'].to_value()
    20
'''

D['Daf.to_list'] = r'''
Get the values of a Daf that has one row or one column, as a list.

Use it on the result of a selection, as in `my_daf[:, 'v'].to_list()`. A column
is read from the Daf, so use `col()` if you only need the list. An empty
Daf gives an empty list. A Daf with more than one row and more than one column
is not accepted.

Args:
    unique: If True, leave out repeated values and keep the order.
    flatten: If True, join items that are lists into one list.
    omit_nulls: If True, leave out the empty values.
    default: Replaces NULL, None and NaN values. This may itself be None.
    astype: A type or function to convert each value with.

Returns:
    The list.

Raises:
    ValueError: The Daf has more than one row and more than one column.

Examples:
''' + EX + r'''
    >>> d[:, 'v'].to_list()
    ['a', 'b', 'c']
    >>> d[1].to_list()
    [2, 'b', 20]
'''

D['Daf.to_lota'] = r'''
Make a list of tuples, one tuple for each row.

This is handy for making composite keys.

Returns:
    The rows as tuples.

Examples:
    >>> Daf(lol=[[1, 'a']], cols=['id', 'v']).to_lota()
    [(1, 'a')]
'''

D['Daf.to_dict'] = r'''
Get the one row of a Daf as a dict.

Use it on the result of a selection, as in `my_daf[1].to_dict()`. A column is
not turned into a dict. Use `to_list()` for that. An empty Daf gives an empty
dict.

Returns:
    The row, as a dict that maps column names to values.

Raises:
    ValueError: The Daf has more than one row.

Examples:
''' + EX + r'''
    >>> d[1].to_dict()
    {'id': 2, 'v': 'b', 'n': 20}
'''

D['Daf.to_klist'] = r'''
Get one row as a [KeyedList][daffodil.keyedlist.KeyedList].

The KeyedList shares the row and the column names with the Daf, so it costs
little. A position that is out of range, or a negative one, gives an empty
KeyedList.

Args:
    irow: The row position.

Returns:
    The row.

Examples:
''' + EX + r'''
    >>> d.to_klist(1)['v']
    'b'
'''

D['Daf.irow'] = r'''
Get one row as a dict, by position.

This is `iloc()` with the default `rtype`.

Args:
    irow: The row position.
    include_cols: Only these columns are included.

Returns:
    The row as a dict. A position that is out of range gives an empty dict.

Examples:
''' + EX + r'''
    >>> d.irow(1, include_cols=['n'])
    {'n': 20}
'''

D['Daf.iloc'] = r'''
Get one row by position, as a dict, a KeyedList or a list.

A negative position, or one that is out of range, gives an empty dict, or an
empty KeyedList. It does not count from the end. Use the row selector `[-1]`
for that. With no column names, the keys are spreadsheet names such as `A`.

Args:
    irow: The row position.
    include_cols: Only these columns are included. This applies to a dict.
    rtype: `dict` for a new dict, `klist` for a KeyedList that shares the row, or `list` for a copy of the row.

Returns:
    The row.

Raises:
    ValueError: `rtype` is not one of the three names.

Examples:
''' + EX + r'''
    >>> d.iloc(1)
    {'id': 2, 'v': 'b', 'n': 20}
    >>> d.iloc(1, rtype='list')
    [2, 'b', 20]
    >>> d.iloc(-1)
    {}
'''

D['Daf.select_by_dict'] = r'''
Select the rows that match every field of a dict.

A row matches if each key of `selector_da` is a column whose value in that row
equals the value given. With `inverse=True` the rows that do not match are
returned. The rows of the new Daf are new lists, so they can be changed
without changing this Daf.

Args:
    selector_da: The column names and the values they must have.
    expectmax: If this is not -1 and more rows match, raise `LookupError`.
    inverse: If True, return the rows that do not match.
    keyfield: The keyfield of the new Daf. If empty, the keyfield of this Daf.

Returns:
    The new Daf.

Raises:
    LookupError: More than `expectmax` rows match.

Examples:
''' + EX + r'''
    >>> d.select_by_dict({'v': 'b'}).lol
    [[2, 'b', 20]]
    >>> d.select_by_dict({'v': 'b'}, inverse=True).lol
    [[1, 'a', 10], [3, 'c', 30]]
'''

D['Daf.select_first_row_by_dict'] = r'''
Get the first row that matches every field of a dict.

The matching rule is that of `select_by_dict()`. With `inverse=True` it is the
first row that does not match.

Args:
    selector_da: The column names and the values they must have.
    inverse: If True, find the first row that does not match.

Returns:
    The row, as a dict or a KeyedList according to `itermode`. An empty dict if none matches.

Examples:
''' + EX + r'''
    >>> d.select_first_row_by_dict({'v': 'b'})
    {'id': 2, 'v': 'b', 'n': 20}
    >>> d.select_first_row_by_dict({'v': 'zz'})
    {}
'''

D['Daf.select_where'] = r'''
Select the rows for which a function is true.

The function gets each row, as a [KeyedList][daffodil.keyedlist.KeyedList].
Read cells by column name, as in `row['n']`. Values are used as stored, so
convert text first if the Daf was read from a CSV.

With `indirect_col`, a name that is not a column is looked up in the dict
held in that column.

The new Daf shares the selected rows with this one. The keyfield and dtypes
carry over.

Args:
    where: A function that takes a row and returns True to keep it.
    indirect_col: A column that holds a dict, to read names that are not columns from.

Returns:
    The new Daf.

Examples:
''' + EX + r'''
    >>> d.select_where(lambda row: row['n'] > 10).lol
    [[2, 'b', 20], [3, 'c', 30]]
'''

D['Daf.select_where_idxs'] = r'''
Get the positions of the rows for which a function is true.

The function gets each row, as in `select_where()`.

Args:
    where: A function that takes a row and returns True to keep it.

Returns:
    The row positions.

Examples:
''' + EX + r'''
    >>> d.select_where_idxs(lambda row: row['n'] > 10)
    [1, 2]
'''

D['Daf.remove_dups'] = r'''
Split the rows into those with a unique key and those with a repeated key.

Only the keyfield is compared, not the whole row. For each key, the last row
is kept as the unique one. The earlier rows with that key are the duplicates.
The unique rows are in the order in which their keys first appear.

You must pass `keyfield`. This method sets the keyfield of this Daf to it, so
the Daf is changed. If you pass nothing, the keyfield is cleared and every row
is returned as a duplicate. If there are no repeats, the first result is this
same Daf, not a copy, and the second is empty.

Args:
    keyfield: The column, or tuple or list of columns, that identifies a row.

Returns:
    A tuple of the Daf of unique rows and the Daf of duplicate rows.

Examples:
    >>> d = Daf(lol=[[1, 'a'], [2, 'b'], [1, 'c']], cols=['id', 'v'])
    >>> unique, dups = d.remove_dups('id')
    >>> unique.lol, dups.lol
    ([[1, 'c'], [2, 'b']], [[1, 'a']])
'''

D['Daf.split_where'] = r'''
Split the rows in two, by a function that is true or false for each row.

The function gets each row, as in `select_where()`. Both new Daf instances
share their rows with this one, so changing a cell in one changes it here
too. The keyfield and dtypes carry over to both.

Args:
    where: A function that takes a row and returns True or False.
    indirect_col: A column that holds a dict, to read names that are not columns from.

Returns:
    A tuple of the Daf of the rows where the function is true and the Daf of the others.

Examples:
''' + EX + r'''
    >>> big, small = d.split_where(lambda row: row['n'] > 10)
    >>> big.lol, small.lol
    ([[2, 'b', 20], [3, 'c', 30]], [[1, 'a', 10]])
'''

D['Daf.col'] = r'''
Get one column as a list, by name.

This does not make a Daf first, as `my_daf[:, 'v'].to_list()` does.

With `indirect_col`, a name that is not a column is read from the dict held in
that column, row by row. A row that lacks it gets `default`.

Args:
    colname: The column name.
    unique: If True, leave out repeated values and keep the order.
    omit_nulls: If True, leave out the empty values.
    silent_error: If True, a column that is not found gives an empty list.
    astype: A type or function to convert each value with.
    indirect_col: A column that holds a dict, to read the name from.
    default: The value for a row that lacks the name. It is used only with `indirect_col`.

Returns:
    The values of the column.

Raises:
    RuntimeError: The column is not found and `silent_error` is False, or the name is empty.

Examples:
''' + EX + r'''
    >>> d.col('v')
    ['a', 'b', 'c']
    >>> d.col('n', astype=str)
    ['10', '20', '30']
'''

D['Daf.col_to_la'] = r'''
Get one column as a list, by name.

This does the same as `col()`. See that method.

Args:
    colname: The column name.
    unique: If True, leave out repeated values and keep the order.
    omit_nulls: If True, leave out the empty values.
    silent_error: If True, a column that is not found gives an empty list.
    astype: A type or function to convert each value with.
    indirect_col: A column that holds a dict, to read the name from.
    default: The value for a row that lacks the name. It is used only with `indirect_col`.

Returns:
    The values of the column.

Raises:
    RuntimeError: The column is not found and `silent_error` is False, or the name is empty.
'''

D['Daf.icol'] = r'''
Get one column as a list, by position.

A position that is negative or out of range gives an empty list.

Args:
    icol: The column position.

Returns:
    The values of the column.

Examples:
''' + EX + r'''
    >>> d.icol(1)
    ['a', 'b', 'c']
'''

D['Daf.icol_to_la'] = r'''
Get one column as a list, by position, with options.

A position that is negative or out of range gives an empty list.

Args:
    icol: The column position.
    unique: If True, leave out repeated values and keep the order.
    omit_nulls: If True, leave out the empty values.

Returns:
    The values of the column.
'''

D['Daf.drop_cols'] = r'''
Remove columns from this Daf, in place.

The rows are rebuilt without those columns, so this copies all the data. Avoid
it for large tables. Use the `cols` argument of `apply` and `reduce`, or
`select_kcols()` to get a new Daf, instead.

The column names and dtypes are updated. A name that is not a column is ignored.
If the keyfield is dropped, the key index is cleared but the keyfield is not.
Set it again with `set_keyfield()`. With no names, nothing happens.

Args:
    exclude_cols: The names of the columns to remove.

Returns:
    This Daf, which has been changed.

Examples:
''' + EX + r'''
    >>> d.drop_cols(['v']).columns()
    ['id', 'n']
'''

D['Daf.select_cols'] = r'''
Make a new Daf with only some columns, chosen by name.

The columns stay in the order of this Daf, not in the order of `cols`. Use
`select_kcols()` if you want the order you give. With no arguments all columns
are kept. A name that is not a column is ignored, so a list of unknown names
gives rows with no columns.

This copies data, so it is not cheap. For `apply` and `reduce`, use their
`cols` argument instead. The keyfield carries over if its column is kept.

Args:
    cols: The names of the columns to keep. If empty, all columns.
    exclude_cols: The names of the columns to leave out.

Returns:
    The new Daf.

Examples:
''' + EX + r'''
    >>> d.select_cols(['n', 'id']).columns()
    ['id', 'n']
    >>> d.select_cols(exclude_cols=['v']).columns()
    ['id', 'n']
'''
setdoc('src/daffodil/daf.py', D)
