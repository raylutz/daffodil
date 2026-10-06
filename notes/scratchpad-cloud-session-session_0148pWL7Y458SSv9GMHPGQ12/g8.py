import sys
sys.path.insert(0, '/tmp/claude-0/-home-user-daffodil/95e36c09-64f8-59ba-9bd6-c962257ad9e3/scratchpad')
from setdoc import setdoc
D = {}
EX = """    >>> d = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'])"""

D['Daf.apply_colwise'] = r'''
Compute one column from the other columns of each row, in place.

The function gets each row, as a dict, and returns the value for `target_col`.
If the function raises an error for a row, that row gets `default`. If
`target_col` is not a column, it is added at the right, and the rows are
filled with `default` first. Use it for a ratio or a total of other columns.

Args:
    target_col: The column to store the result in.
    func: A function that takes a row and returns the value.
    default: The value for a row where the function raises an error.

Returns:
    This Daf, which has been changed.

Examples:
    >>> d = Daf(lol=[[1, 2], [3, 0]], cols=['a', 'b'])
    >>> d.apply_colwise('ratio', lambda row: row['a'] / row['b'], default=-1.0).lol
    [[1, 2, 0.5], [3, 0, -1.0]]
'''

D['Daf.extend'] = r'''
Append several records that are given as a list of dicts.

Each dict is a row, placed by column name, as in `append()`. A Daf with no
columns takes them from the first dict. An empty list, or a list that holds
one empty dict, adds nothing.

Without `respect_kd`, a key that already exists is added again. Use
`respect_kd=True` when the keyfield must stay unique.

Args:
    records_lod: The records, as dicts.
    respect_kd: If True, replace the row that has the same key. Otherwise add it.

Returns:
    This Daf, which has been changed.

Examples:
    >>> d = Daf(lol=[[1, 'a']], cols=['id', 'v'], keyfield='id')
    >>> d.extend([{'id': 2, 'v': 'b'}, {'id': 1, 'v': 'new'}], respect_kd=True).lol
    [[1, 'new'], [2, 'b']]
'''

D['Daf.set_irows_icols'] = r'''
Set values at row positions and column positions, in place.

This is what `my_daf[rows, cols] = value` calls, after the selectors are turned
into positions. `irows` and `icols` are an integer, a slice, a range or a list.
If `icols` is None, the whole row is set. If `irows` is None, nothing is
set. Use `slice(None)` for all rows.

What is set depends on the value.

    a single value    fills every selected cell. A str or bytes is a single value.
    a list            fills a selection in order. For several whole rows, each
                      row becomes a copy of the list.
    a dict            sets a row. The cells of columns that the dict lacks become NULL.
    a Daf             is copied as a block, from its top left corner.

Nothing is checked, and no error is raised for a size mismatch. A smaller source
fills the top left of the selection and leaves the rest. A larger one fills the
selection and the extra values are ignored. Check the sizes first if it matters.

Args:
    irows: The row positions.
    icols: The column positions.
    value: The value, list, dict or Daf to set.

Returns:
    This Daf, which has been changed.

Examples:
''' + EX + r'''
    >>> d.set_irows_icols([0, 1], [1, 2], 'Z').lol
    [[1, 'Z', 'Z'], [2, 'Z', 'Z'], [3, 'c', 30]]
    >>> d.set_irows_icols(2, None, ['x', 'y', 'z']).lol[2]
    ['x', 'y', 'z']
'''

D['Daf.gkeys_to_idxs'] = r'''
Turn keys into positions, using a dict of key to position.

This is the step under `krows_to_irows()` and `kcols_to_icols()`. It is a
static method. It is for internal use.

A key gives a list with one position. A list of keys gives their positions in
that order. A tuple of two keys is an inclusive range, and `None` in it means
the start or the end. A tuple of one key means that key to the end. A slice is
taken as positions. With `inverse`, the positions that are not selected are
returned.

Args:
    keydict: Maps each key to its position.
    gkeys: The keys.
    inverse: If True, return the positions that are not selected.
    silent_error: If True, keys that are not found are ignored.
    axis: A label for error messages.
    name: A label for error messages.

Returns:
    The positions, as a list or a slice.

Raises:
    KeysDisabledError: The dict is empty.
    KeyError: A key is not found and `silent_error` is False.
    TypeError: The keys are None, or a tuple of a length other than 1 or 2.

Examples:
    >>> Daf.gkeys_to_idxs({'a': 0, 'b': 1, 'c': 2}, ['c', 'a'])
    [2, 0]
    >>> Daf.gkeys_to_idxs({'a': 0, 'b': 1, 'c': 2}, ('a', 'b'))
    slice(0, 2, 1)
'''
setdoc('src/daffodil/daf.py', D)
