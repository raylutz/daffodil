import sys
sys.path.insert(0, '/tmp/claude-0/-home-user-daffodil/95e36c09-64f8-59ba-9bd6-c962257ad9e3/scratchpad')
from setdoc import setdoc

D = {}
S = {}

S['_apply_schema'] = '''
Attach a schema to this Daf and fill in what the Daf is missing.

A schema describes columns: their names, types and default values. Use one
when many tables share a layout, or to make new records with the defaults.
The constructor calls this method, so `Daf(schema=...)` applies the schema.

Two kinds of schema are accepted.

A class decorated with `@schemaclass`. The annotated attributes give the
column names and types. Their values are the defaults. A `__keyfield__`
attribute gives the keyfield.

A schema Daf, with one row for each column of the table. It must have a
`Name` column. It may have a `dtype` column, which holds one of `str`, `int`,
`float`, `bool`, `list` or `dict`. A `Default` column gives the defaults. Its
`attrs` may hold a `keyfield`.

Only gaps are filled. The column names, the dtypes and the keyfield are taken
from the schema only if the Daf has none. The rows are never changed and
nothing is validated.

The other columns of a schema Daf are for building input forms. They are not
used by daffodil. The `Type` column says what kind of form control to use.

    checkbox   One or more checkboxes. Value lists the labels.
                   checkbox+buttons adds Set and Clear buttons.
                   checkbox+values allows values that differ from the labels.
    date       A text box with a calendar button.
    label      Read only text.
    radio      Like checkbox, but only one can be chosen.
    select     A dropdown or list box. Value lists the options.
                   select+multi allows several choices.
                   select+values allows values that differ from the labels.
    text       A one line text box. Value is the initial text.
    textarea   A multi line text box. Size is columns x rows, such as 80x6.

Args:
    schema: The schema to apply. If None, the schema already attached is used.

Returns:
    This Daf, which has been changed.

Raises:
    TypeError: The schema is neither a `@schemaclass` nor a schema Daf. Nothing is stored.
    RuntimeError: A `dtype` in a schema Daf is not one of the supported names.

Examples:
    >>> from daffodil.lib.schemaclass import schemaclass
    >>> @schemaclass
    ... class Person:
    ...     name: str = ''
    ...     age: int = 0
    >>> d = Daf(schema=Person)
    >>> d.columns()
    ['name', 'age']
    >>> d.dtypes
    {'name': <class 'str'>, 'age': <class 'int'>}
'''

S['_attach_schema'] = '''
Attach a `@schemaclass` to this Daf.

This is the strict form of `apply_schema()`, for a schema class only. It stores
the schema, then fills in the column names, the dtypes and the keyfield, each
only if the Daf has none. The rows are never changed and nothing is validated.

Args:
    schema: A class decorated with `@schemaclass`.

Returns:
    This Daf, which has been changed.

Raises:
    TypeError: The class is not a `@schemaclass`.

Examples:
    >>> from daffodil.lib.schemaclass import schemaclass
    >>> @schemaclass
    ... class Person:
    ...     name: str = ''
    ...     age: int = 0
    >>> Daf().attach_schema(Person).columns()
    ['name', 'age']
'''

S['_default_record'] = '''
Make a new record that holds the defaults of the attached schema.

Use it to start a record that you fill in, then append. Each call returns a
new dict, so changing it does not change the next one. Nothing is converted
or validated.

For a schema Daf the `Name` column gives the keys and the `Default` column
gives the values. A missing `Default` column gives empty strings.

Returns:
    A dict that maps each column name to its default.

Raises:
    AttributeError: No schema is attached.
    RuntimeError: A schema Daf has no `Name` column.
    TypeError: The attached schema is of an unsupported kind.

Examples:
    >>> from daffodil.lib.schemaclass import schemaclass
    >>> @schemaclass
    ... class Person:
    ...     name: str = ''
    ...     age: int = 0
    >>> Daf(schema=Person).default_record()
    {'name': '', 'age': 0}
'''

D['Daf.set_dtypes'] = '''
Set the dtype of every column, from a default and a few exceptions.

Use it when most columns have one type and a few differ. The result is stored
in `dtypes`. This does not convert any data. Call `apply_dtypes()` for that.

If most columns differ, assign a dict to `dtypes` yourself.

Args:
    default_type: The type of every column that is not listed.
    typ_to_cols_dict: Maps a type to the names of the columns of that type.

Returns:
    This Daf, which has been changed.

Raises:
    NotImplementedError: The Daf has no column names.

Examples:
    >>> d = Daf(lol=[['1', '2', 'x']], cols=['a', 'b', 'c'])
    >>> d.set_dtypes(str, {int: ['a', 'b']}).dtypes
    {'a': <class 'int'>, 'b': <class 'int'>, 'c': <class 'str'>}
'''

D['Daf.apply_dtypes'] = '''
Convert the columns to their dtypes, in place.

A CSV file is read as text, because that is the fastest way to load it. Call
this method to turn the columns you need into numbers, lists and so on. It
changes the cells where they are and does not make a new table. Columns you
leave out are not touched.

The `dtypes` argument is a dict that maps a column name to a type, or one type
for all columns. It may hold more columns than the Daf. If it is given, it
replaces `dtypes` of the Daf. If neither is set, nothing happens. With no
columns defined, the names are taken from the dtypes.

Types must be plain types such as `int`, `float`, `bool`, `str`, `list`,
`dict`, `tuple` or `set`. Annotations such as `List[str]` do not work.

By default, columns of type `str` are skipped. The cells are assumed to be text
already. Pass `from_str=False` when they may hold other values. Then each cell
is converted with `str()`. Columns of type `list` or `dict` are read from
their text. Pass `unflatten=False` to leave them as text.

A cell that cannot be converted becomes NULL. An empty cell stays empty. No
error is raised, so check the result if the data is not trusted.

Args:
    dtypes: Maps column names to types, or a single type for all columns.
    unflatten: If True, read list and dict columns from their text.
    from_str: If True, the cells are text, so `str` columns are not converted.
    default_type: The type to use for a column that has no dtype.
    silent_error: If False, raise an error when a column has no dtype.

Returns:
    This Daf, which has been changed.

Raises:
    ValueError: A column has no dtype and `silent_error` is False.

Examples:
    >>> d = Daf(lol=[['1', '2.5', 'x']], cols=['a', 'b', 'c'])
    >>> d.apply_dtypes(dtypes={'a': int, 'b': float, 'c': str}).lol
    [[1, 2.5, 'x']]
    >>> Daf(lol=[['x', '']], cols=['a', 'b']).apply_dtypes(dtypes={'a': int, 'b': int}).lol
    [['', '']]
'''

D['Daf.flatten'] = '''
Turn list and dict cells into text, in place.

You rarely need this. `to_csv_buff()` and `to_csv_file()` write every cell as
text already, so a separate pass is not needed.

Only the columns whose dtype is `list` or `dict` are changed. Each cell
becomes its `str()` text. A column of dtype `bool` becomes 0 and 1. Without
dtypes nothing happens.

Args:
    convert_bool_to_int: If True, write bool columns as 0 and 1.
    use_pyon: Kept for compatibility. Only True is supported.

Returns:
    This Daf, which has been changed.

Raises:
    ValueError: `use_pyon` is False.

Examples:
    >>> d = Daf(lol=[[[1, 2], True]], cols=['a', 'b'], dtypes={'a': list, 'b': bool})
    >>> d.flatten().lol
    [['[1, 2]', 1]]
'''

D['Daf.strip'] = '''
Remove characters from both ends of every text cell, in place.

Each character in `chrs` is removed on its own, so `'()"'` removes any mix of
parentheses and quotes. Cells that are not text, and empty cells, are skipped.

Args:
    chrs: The characters to remove.

Returns:
    This Daf, which has been changed.

Examples:
    >>> Daf(lol=[[' a ', 3, '("x")']], cols=['p', 'q', 'r']).strip(' ()"').lol
    [['a', 3, 'x']]
'''

D['Daf.clone_empty'] = '''
Make a new Daf with the same layout and no rows.

The new Daf has the same column names, keyfield and dtypes. The dtypes dict is
copied, and the `attrs` are deep copied. The name, the key index and the
display settings are not carried over. Set them on the new Daf if you need them.

Give `lol` to fill the new Daf with rows. They are adopted, not copied.

Args:
    lol: Rows for the new Daf. If None, it has no rows.
    cols: Column names to use instead of the existing ones.
    name: The name of the new Daf.

Returns:
    The new Daf.

Examples:
    >>> d = Daf(lol=[[1, 'a']], cols=['id', 'v'], keyfield='id')
    >>> c = d.clone_empty()
    >>> c.columns(), c.keyfield, c.num_rows()
    (['id', 'v'], 'id', 0)
'''

D['Daf.set_lol'] = '''
Replace the rows with a new list of lists.

The list is adopted, not copied. The column names, the keyfield and the other
settings stay. The key index is rebuilt when it is next needed.

Args:
    new_lol: The new rows.

Returns:
    This Daf, which has been changed.

Examples:
    >>> d = Daf(lol=[[1, 'a']], cols=['id', 'v'], keyfield='id')
    >>> d.set_lol([[5, 'q'], [6, 'r']]).keys()
    [5, 6]
'''

setdoc('src/daffodil/daf.py', D)
setdoc('src/daffodil/lib/daf_schema.py', S)
