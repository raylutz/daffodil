import sys
sys.path.insert(0, '/tmp/claude-0/-home-user-daffodil/95e36c09-64f8-59ba-9bd6-c962257ad9e3/scratchpad')
from setdoc import setdoc

D = {}

D['Daf'] = '''
A table of data, stored as a list of rows.

Daf is a small, fast, pure Python table. Use it to read, reshape and write
2-D data without the weight of pandas. It is not meant for heavy numeric work.

The rows are a list of lists named `lol`. A row holds only values. The column
names are kept once, in a header dict named `hd` that maps each name to its
position. That makes rows cheap to build, append and copy.

A Daf may have a keyfield. This is a column, or a tuple of columns, whose
values name the rows. The key index is built the first time a key is needed.
It is rebuilt after the rows change.

A missing value is `NULL`, which is the empty string. It prints as nothing.

Rows are returned as dicts or as [KeyedList][daffodil.keyedlist.KeyedList]
objects. See [retmode][daffodil.daf.Daf.retmode] and
[itermode][daffodil.daf.Daf.itermode].

Examples:
    >>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'], keyfield='id')
    >>> d.columns()
    ['id', 'v']
    >>> d.select_record(2)
    {'id': 2, 'v': 'b'}
    >>> d.shape()
    (2, 2)
'''

D['Daf.__init__'] = '''
Create a Daf from rows, column names and options.

Every argument is optional, so `Daf()` makes an empty table. The usual call
gives the rows and the column names.

The rows are not copied. The `lol` list you pass in becomes the data of the
Daf, so changing one changes the other. The same holds for `hd`, `kd` and
`attrs`. Pass `use_copy=True` to deep copy `lol` first.

If `cols` is given, it sets the column names. If it is not given, the keys of
`dtypes` set them. Otherwise `hd` is used. A name that is empty or repeated
is made unique, so `['a', 'a']` becomes `['a', 'a_1']`.

With no names at all, the Daf has no columns and `lol` is only a list of rows.
Call `set_cols()` to name them later.

Args:
    lol: Rows, as a list of lists. Adopted, not copied.
    hd: Header dict that maps column name to position.
    kd: Key index to adopt when no keyfield is set.
    cols: Column names. These win over `hd` and `dtypes`.
    dtypes: Type for each column, used when converting from strings.
    schema: A `@schemaclass` or a schema Daf that supplies columns and defaults.
    keyfield: Column, or tuple or list of columns, whose values identify rows.
    name: Free text name of this Daf.
    use_copy: If True, deep copy `lol` instead of adopting it.
    disp_cols: Column names to show when the Daf is printed.
    retmode: Whether a one cell result is returned as a Daf or a bare value.
    itermode: Whether iteration yields dicts or KeyedList objects.
    attrs: Free form dict of extra information. Adopted, not copied.

Raises:
    TypeError: `disp_cols` is not a list, a tuple or None.

Examples:
    >>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'], keyfield='id')
    >>> d.num_rows()
    2
    >>> Daf(lol=[[1, 2]], cols=['a', 'a']).columns()
    ['a', 'a_1']
'''

D['Daf.__bool__'] = '''
Say whether the Daf holds data.

`bool(d)` and `if d:` are true when the Daf has at least one row with a value.
A Daf that has column names but no rows is false. Daffodil uses this as its
test for an empty table.

Returns:
    True if there is data.

Examples:
    >>> bool(Daf(cols=['a']))
    False
    >>> bool(Daf(lol=[[1]], cols=['a']))
    True
'''

D['Daf.__format__'] = '''
Format a Daf that holds one cell.

With no format spec this is the same as `str()`. With a spec, the Daf must
have exactly one cell. A number is formatted with the spec. Anything else
is converted with `str()`.

Args:
    format_spec: A format spec such as `.2f`.

Returns:
    The formatted text.

Raises:
    ValueError: A spec is given and the Daf does not have exactly one cell.

Examples:
    >>> format(Daf(lol=[[3.14159]], cols=['a']), '.2f')
    '3.14'
'''

D['Daf.__eq__'] = '''
Compare two Daf instances.

Two Daf instances are equal when their rows, their column names and their
keyfield are equal. The order of the columns matters. The name, dtypes,
attrs and modes are not compared. Anything that is not a Daf is not equal.

Args:
    other: The object to compare with.

Returns:
    True if equal.

Examples:
    >>> a = Daf(lol=[[1]], cols=['x'])
    >>> a == Daf(lol=[[1]], cols=['x'])
    True
    >>> a == Daf(lol=[[1]], cols=['x'], keyfield='x')
    False
'''

D['Daf.__str__'] = '''
Show the Daf as a Markdown table.

The table shows at most `md_max_rows` rows and `md_max_cols` columns, 10 by
default. Larger tables show the first five and the last five, with `...`
between them. A summary line with the size and keyfield follows the table.
Use `md_daf_table_snippet()` for control over the output.

Returns:
    The Markdown text.
'''

D['Daf.__repr__'] = '''
Show the Daf as a Markdown table, for the Python prompt.

This is the same text as `str()`, with a newline in front, so the table
starts at the left margin when it is echoed at the prompt.

Returns:
    The Markdown text.
'''

D['Daf.__contains__'] = '''
Test whether a key is in the keyfield column.

Use it as `key in my_daf`. For a composite keyfield, the key is a tuple.
An empty Daf contains nothing.

Args:
    key: The key to look for.

Returns:
    True if a row has this key.

Raises:
    KeyError: The Daf has rows but no keyfield, and no key index was adopted.

Examples:
    >>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'], keyfield='id')
    >>> 2 in d, 5 in d
    (True, False)
'''

D['Daf.num_cols'] = '''
Count the columns by looking at the rows.

This does not use the column names. It returns the length of the longest of
the first 10 rows, and 0 if there are no rows. For a table with equal row
lengths that is the real width. Use `len(d.columns())` to count the names, and
`is_rectangular()` to check the rows.

Returns:
    The number of columns.

Examples:
    >>> Daf(lol=[[1, 2, 3]], cols=['a', 'b']).num_cols()
    3
    >>> Daf(cols=['a', 'b']).num_cols()
    0
'''

D['Daf.__len__'] = '''
Return the number of rows, so `len(d)` works.

Returns:
    The number of rows.
'''

D['Daf.num_rows'] = '''
Return the number of rows.

Returns:
    The number of rows.
'''

D['Daf.len'] = '''
Return the number of rows.

This does the same as `num_rows()` and `len(d)`.

Returns:
    The number of rows.
'''

D['Daf.is_rectangular'] = '''
Check whether every row has the same length.

With column names, every row must be as long as the number of names. Without
names, every row must be as long as the first row. An empty Daf is rectangular.

This looks at every row. Use it when you do not trust the source of the data.
`num_cols()` only samples the first rows.

Returns:
    True if all rows have the expected length.

Examples:
    >>> Daf(lol=[[1, 2], [3]], cols=['a', 'b']).is_rectangular()
    False
'''

D['Daf.force_rectangular'] = '''
Pad short rows with empty strings, in place.

Use this when a source leaves off trailing empty cells. An xlsx file read by
`xlsx_to_csv()` does that. The target width is the number of column names. With
no names, it is the length of the longest row.

Rows that are too long are not cut. That would hide damage, such as an
unquoted comma that split a value. They raise an error instead.

Returns:
    This Daf, which has been changed.

Raises:
    ValueError: A row is longer than the number of columns. Nothing is changed.

Examples:
    >>> Daf(lol=[[1, 2], [3]], cols=['a', 'b']).force_rectangular().lol
    [[1, 2], [3, '']]
'''

D['Daf.shape'] = '''
Return the number of rows and columns as a tuple.

This is a method, not a property, because the column count is worked out from
the rows. It is the same as `(num_rows(), num_cols())`. A Daf with column
names but no rows has shape `(0, 0)`.

Returns:
    A tuple of the number of rows and the number of columns.

Examples:
    >>> Daf(lol=[[1, 2], [3, 4], [5, 6]], cols=['a', 'b']).shape()
    (3, 2)
'''

D['Daf.copy'] = '''
Make a copy of the Daf.

How much is copied depends on the options. Choose the cheapest one that is
safe for what you do next.

A plain copy is shallow. The new Daf shares the row list `lol`, the rows and
`hd` with the original. Appending a row to the copy also appends it to the
original, and so does changing a cell. Only the `attrs` are copied.

With `for_sorting=True` the list of rows is copied, but the rows are still
shared. You can sort or reorder the copy without changing the original. A
change inside a row still reaches both.

With `deep=True` nothing is shared.

Args:
    deep: If True, copy everything, so the two Daf instances are independent.
    for_sorting: If True and not deep, copy the list of rows but share the rows.

Returns:
    The new Daf.

Examples:
    >>> d = Daf(lol=[[2, 'b'], [1, 'a']], cols=['id', 'v'])
    >>> shallow = d.copy()
    >>> shallow.lol.append([3, 'c'])
    >>> d.num_rows()
    3
    >>> d.copy(deep=True).lol is d.lol
    False
'''

D['Daf.columns'] = '''
Return the column names.

The list is a new copy, so changing it does not change the Daf. Use
`set_cols()` or `rename_cols()` to change the names.

Returns:
    The column names, in order.

Examples:
    >>> Daf(cols=['a', 'b']).columns()
    ['a', 'b']
'''

D['Daf.isin'] = '''
Make a list of True and False, one per item of the first collection.

An item is True if it is found in the second collection. This is a static
method, so call it as `Daf.isin(a, b)`. It is handy for picking or leaving out
columns by name.

Args:
    listlike1: The items to test, in order.
    listlike2: The collection to look in.

Returns:
    A list of bools, as long as `listlike1`.

Examples:
    >>> Daf.isin(['a', 'b', 'c'], ['b'])
    [False, True, False]
'''

D['Daf.calc_cols'] = '''
Work out a list of column names from rules.

Use it to choose the columns for `apply` or `reduce`. The rules are applied
in this order: include by name, exclude by name, include by type and exclude
by type. The result keeps the order of the Daf. A single name may be given
as a string.

With a group by operation, leave the group by columns out of the list.

Args:
    include_cols: Keep only these column names.
    exclude_cols: Drop these column names.
    include_types: Keep only columns whose dtype is in this list.
    exclude_types: Drop columns whose dtype is in this list.

Returns:
    The selected column names.

Raises:
    RuntimeError: A type rule is given and no dtypes are set.

Examples:
    >>> d = Daf(lol=[[1, 2, 3]], cols=['a', 'b', 'c'])
    >>> d.calc_cols(exclude_cols='b')
    ['a', 'c']
'''

D['Daf.rename_cols'] = '''
Rename columns in place.

Names that are not in the mapping stay as they are. Names in the mapping that
are not in the Daf are ignored. The dtypes are renamed too.

The keyfield is cleared, even if its column was not renamed. Call
`set_keyfield()` afterwards to turn key lookups back on.

Args:
    from_to_dict: Maps old names to new names.

Returns:
    This Daf, which has been changed.

Examples:
    >>> d = Daf(lol=[[1, 2]], cols=['a', 'b'])
    >>> d.rename_cols({'b': 'c'}).columns()
    ['a', 'c']
'''

D['Daf.set_cols'] = '''
Set the column names, in place.

The new names are given by position, so the first name goes to the first
column. Without a list, the names are A, B, C and so on, like a spreadsheet.

With `sanitize_cols` on, a repeated name gets a suffix, so `['a', 'a']` becomes
`['a', 'a_1']`. An empty name becomes the prefix and its position, like `col2`.
The dtypes are renamed by position as well.

The keyfield is cleared. Call `set_keyfield()` afterwards to turn key lookups
back on.

Args:
    new_cols: The names, in order. If None, spreadsheet names are made.
    sanitize_cols: If True, make the names valid and unique.
    unnamed_prefix: The start of a name made for an empty one.

Returns:
    This Daf, which has been changed.

Raises:
    AttributeError: There are fewer names than columns.

Examples:
    >>> Daf(lol=[[1, 2, 3]]).set_cols().columns()
    ['A', 'B', 'C']
    >>> Daf(lol=[[1, 2, 3]]).set_cols(['a', 'a', '']).columns()
    ['a', 'a_1', 'col2']
'''

D['Daf.keys'] = '''
Return the row keys.

The keys are the values of the keyfield column, in row order. With a composite
keyfield each key is a tuple. The key index is built here if it is needed.

Without a keyfield the answer is an empty list, unless `silent_error` is False.
Then it raises an error. An index passed in as `kd` is not used by this method.

With `astype='view'` you get a view of the key index, not a copy. It keeps the
old keys after the Daf changes, so use it right away.

Args:
    silent_error: If False, raise an error when there is no keyfield.
    astype: `list` for a new list, or `view` for a view of the index.

Returns:
    The keys.

Raises:
    KeysDisabledError: There is no keyfield and `silent_error` is False.
    ValueError: `astype` is neither `list` nor `view`.

Examples:
    >>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'], keyfield='id')
    >>> d.keys()
    [1, 2]
'''

D['Daf.set_keyfield'] = '''
Choose the column, or columns, that identify the rows.

Give a column name, or a tuple or list of names for a composite key. A
composite key is a tuple of those values. An empty keyfield turns key lookups
off. The key index is built later, when it is first needed.

The Daf must have column names first. With none, nothing happens.

A name that is not a column is stored anyway unless `silent_error` is False.
Then a `KeyError` is raised. A key that is not unique is not checked. A lookup
finds the last row that has it.

Args:
    keyfield: Column name, or a tuple or list of names. Empty to turn off.
    silent_error: If False, raise an error for a name that is not a column.
    force_kd_rebuild: If True, build the key index now.

Returns:
    This Daf, which has been changed.

Raises:
    KeyError: The keyfield is not a column and `silent_error` is False.

Examples:
    >>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'])
    >>> d.set_keyfield('id').keys()
    [1, 2]
    >>> d.set_keyfield(['id', 'v']).keys()
    [(1, 'a'), (2, 'b')]
'''

D['Daf.get_existing_keys'] = '''
Keep the keys that are in the Daf.

Use it to find out which of a list of keys have a row. The order of the list
is kept. The result is empty if the Daf has no keyfield.

Args:
    keylist: The keys to check.

Returns:
    The keys from the list that have a row.

Examples:
    >>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'], keyfield='id')
    >>> d.get_existing_keys([1, 5, 2])
    [1, 2]
'''

setdoc('src/daffodil/daf.py', D)
