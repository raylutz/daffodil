import sys
sys.path.insert(0, '/tmp/claude-0/-home-user-daffodil/95e36c09-64f8-59ba-9bd6-c962257ad9e3/scratchpad')
from setdoc import setdoc
D = {}

D['Daf.append'] = r'''
Add one row, or several, to the end of the Daf.

This is the general way to add data. What it does depends on what you give it.

A dict or a [KeyedList][daffodil.keyedlist.KeyedList] is one row. It is placed
by column name, so its keys may be in any order. A missing key gets NULL. A key
that is not a column is dropped.

A list of values is one row, in column order. A short list is padded with NULL.
Extra values are dropped. With no columns defined, the list is added as it is.
A list of lists is not a list of rows. It becomes one row whose cells are lists.

A list of dicts is several rows. See `extend()`.

A Daf is several rows. See `concat()`. Its columns must match.

An empty dict, list or Daf adds nothing.

By default the keyfield is not checked, so a key that is already present is
added again. This keeps appending fast. Pass `respect_kd=True` to replace the
row that has the same key instead. That looks the key up on every call, so
it costs more when you add many rows one at a time.

A list you pass in is added as the row itself, not as a copy.

Args:
    data_item: The row or rows to add.
    respect_kd: If True, replace the row that has the same key. Otherwise add it.

Returns:
    This Daf, which has been changed.

Examples:
    >>> d = Daf(lol=[[1, 'a']], cols=['id', 'v'], keyfield='id')
    >>> d.append({'v': 'b', 'id': 2}).lol
    [[1, 'a'], [2, 'b']]
    >>> d.append([3, 'c']).lol
    [[1, 'a'], [2, 'b'], [3, 'c']]
    >>> d.append({'id': 2, 'v': 'new'}, respect_kd=True).lol
    [[1, 'a'], [2, 'new'], [3, 'c']]
'''

D['Daf.record_append'] = r'''
Add one row that is given as a dict or a KeyedList.

This is the single row case of `append()`. The row is placed by column name,
a missing key gets NULL, and a key that is not a column is dropped. An empty
Daf takes its columns from the first row. An empty record adds nothing.

With a keyfield, the row that has the same key is replaced, and a new key is
added at the end. This is the default here, unlike `append()`. Pass
`respect_kd=False` to always add. The key index is kept up to date, so adding
many rows one at a time stays fast.

A plain list is not accepted. Use `append()` for that.

Args:
    record: The row, as a dict or a [KeyedList][daffodil.keyedlist.KeyedList].
    respect_kd: If True, replace the row that has the same key. Otherwise add it.

Returns:
    This Daf, which has been changed.

Examples:
    >>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'], keyfield='id')
    >>> d.record_append({'id': 2, 'v': 'new'}).lol
    [[1, 'a'], [2, 'new']]
    >>> d.record_append({'id': 2, 'v': 'dup'}, respect_kd=False).lol
    [[1, 'a'], [2, 'new'], [2, 'dup']]
'''

D['Daf.remove_key'] = r'''
Make a new Daf without the row that has the given key.

This does not remove the row from this Daf. It returns a new Daf that leaves
the row out, and this Daf is unchanged. Keep the result, as in
`d = d.remove_key(2)`.

The new Daf shares the surviving rows with this Daf. Changing a cell in one
changes it in the other. Call `copy()` with `deep=True` if you need rows that
are independent.

For a composite keyfield, put the key tuple inside a list, as in
`remove_key([(1, 'a')])`. A bare tuple is read as a range of keys, not as one key.

Args:
    keyval: The key of the row to leave out.
    silent_error: If True, a key that is not found is ignored.

Returns:
    The new Daf. It has the same keyfield.

Raises:
    KeysDisabledError: The Daf has no keyfield.
    KeyError: The key is not found and `silent_error` is False.

Examples:
    >>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'], keyfield='id')
    >>> d.remove_key(1).lol
    [[2, 'b']]
    >>> d.num_rows()
    2
'''

D['Daf.remove_keylist'] = r'''
Make a new Daf without the rows that have the given keys.

This does not remove the rows from this Daf. It returns a new Daf that leaves
them out, and this Daf is unchanged. See `remove_key()` for how the rows are
shared.

Args:
    keylist: The keys of the rows to leave out.
    silent_error: If True, keys that are not found are ignored.

Returns:
    The new Daf. It has the same keyfield.

Raises:
    KeysDisabledError: The Daf has no keyfield.
    KeyError: A key is not found and `silent_error` is False.

Examples:
    >>> d = Daf(lol=[[1, 'a'], [2, 'b'], [3, 'c']], cols=['id', 'v'], keyfield='id')
    >>> d.remove_keylist([1, 3]).lol
    [[2, 'b']]
    >>> d2 = Daf(lol=[[1, 'a', 0], [2, 'b', 1]], cols=['p', 'q', 'r'], keyfield=['p', 'q'])
    >>> d2.remove_keylist([(1, 'a')]).lol
    [[2, 'b', 1]]
'''
setdoc('src/daffodil/daf.py', D)
