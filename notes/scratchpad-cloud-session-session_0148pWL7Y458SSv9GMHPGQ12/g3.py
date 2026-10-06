import sys
sys.path.insert(0, '/tmp/claude-0/-home-user-daffodil/95e36c09-64f8-59ba-9bd6-c962257ad9e3/scratchpad')
from setdoc import setdoc

D = {}

D['Daf.from_lod'] = r'''
Make a Daf from a list of dicts, one dict for each row.

The column names are the keys of the first dict. The first dict must have
every key. A later dict that lacks a key gets NULL there. A later dict with an
extra key loses that value. Empty dicts and items that are not dicts are
skipped.

If `cols` is given, those are the columns. Otherwise the keys of `dtypes` are
the columns, and any other keys are left out.

Args:
    records_lod: The rows, as dicts.
    keyfield: Column, or tuple or list of columns, whose values identify rows.
    dtypes: Type for each column. When given, it also selects the columns.
    name: Name of the new Daf.
    cols: Column names to use, in order.

Returns:
    The new Daf.

Examples:
    >>> Daf.from_lod([{'a': 1, 'b': 2}, {'a': 3}]).lol
    [[1, 2], [3, '']]
    >>> Daf.from_lod([{'a': 1, 'b': 2}], cols=['b', 'a']).lol
    [[2, 1]]
'''

D['Daf.from_lot'] = r'''
Make a Daf from a list of tuples, one tuple for each row.

Without `cols` the columns are named `col_0`, `col_1` and so on. This differs
from `set_cols()`, which makes spreadsheet names such as `A` and `B`.

Args:
    records_lot: The rows, as tuples.
    cols: Column names. They must be as many as the items in each tuple.
    dtypes: Type for each column.
    keyfield: Column, or tuple or list of columns, whose values identify rows.
    name: Name of the new Daf.

Returns:
    The new Daf.

Raises:
    ValueError: A tuple does not have as many items as there are columns.

Examples:
    >>> Daf.from_lot([(1, 'a'), (2, 'b')], cols=['id', 'v']).lol
    [[1, 'a'], [2, 'b']]
    >>> Daf.from_lot([(1, 'a')]).columns()
    ['col_0', 'col_1']
'''

D['Daf.to_lod'] = r'''
Make a list of dicts, one dict for each row.

The dicts are new, but the values in them are the same objects as in the Daf.
An empty Daf gives an empty list. See `iter_dict()` to go through rows one at
a time without building the whole list.

Returns:
    The rows as dicts.

Examples:
    >>> Daf(lol=[[1, 'a']], cols=['id', 'v']).to_lod()
    [{'id': 1, 'v': 'a'}]
'''

D['Daf.from_dod'] = r'''
Make a Daf from a dict of dicts, where the outer key names the row.

A dict of dicts usually does not repeat the row key inside each row. A Daf
table always has it as a column. If the inner dicts lack the `keyfield`
column, it is added from the outer keys. The new Daf has that keyfield.

Use `to_dod()` to go back.

Args:
    dod: A dict that maps a row key to a dict of that row.
    keyfield: The column that holds the outer key.
    dtypes: Type for each column.

Returns:
    The new Daf.

Examples:
    >>> d = Daf.from_dod({'r0': {'x': 1}, 'r1': {'x': 2}})
    >>> d.columns(), d.lol
    (['rowkey', 'x'], [['r0', 1], ['r1', 2]])
'''

D['Daf.to_dod'] = r'''
Make a dict of dicts, where the keyfield value names each row.

The keyfield column is left out of the inner dicts by default. Pass
`remove_keyfield=False` to keep it there too. The Daf must have a keyfield.
Without one, a `KeyError` is raised.

Args:
    remove_keyfield: If True, do not repeat the key inside each inner dict.

Returns:
    A dict that maps each key to a dict of that row.

Raises:
    KeyError: The Daf has no keyfield.

Examples:
    >>> d = Daf(lol=[[1, 'a']], cols=['id', 'v'], keyfield='id')
    >>> d.to_dod()
    {1: {'v': 'a'}}
    >>> d.to_dod(remove_keyfield=False)
    {1: {'id': 1, 'v': 'a'}}
'''

D['Daf.from_cols_dol'] = r'''
Make a Daf from a dict of lists, where each list is a column.

All lists should be as long as the first one. If a list is shorter, an
`IndexError` is raised. If it is longer, the extra values are dropped without
a message.

Args:
    cols_dol: Maps a column name to the list of its values.
    keyfield: Column, or tuple or list of columns, whose values identify rows.
    dtypes: Type for each column.

Returns:
    The new Daf.

Examples:
    >>> d = Daf.from_cols_dol({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> d.lol
    [[1, 4], [2, 5], [3, 6]]
'''

D['Daf.to_cols_dol'] = r'''
Make a dict of lists, one list of values for each column.

Returns:
    A dict that maps each column name to the list of its values.

Examples:
    >>> Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v']).to_cols_dol()
    {'id': [1, 2], 'v': ['a', 'b']}
'''

D['Daf.to_attrib_dict'] = r'''
Make a dict with the column names and the rows.

This is deprecated. It keeps the rows as they are, not copied. Use
`to_lod()` or `to_cols_dol()` instead.

Returns:
    A dict with the keys `cols` and `lol`.

Examples:
    >>> Daf(lol=[[1, 'a']], cols=['id', 'v']).to_attrib_dict()
    {'cols': ['id', 'v'], 'lol': [[1, 'a']]}
'''

D['Daf.from_lod_to_cols'] = r'''
Make a Daf in which each dict becomes a column, not a row.

The keys of the dicts become the first column, and each dict adds one more
column after it. Use this to compare several results side by side, such as
the same features measured in several tries. It is a transpose of
`from_lod()`.

Without `cols`, the first column is named `key` and the others `A`, `B` and
so on. The names in `cols` include the first column. The keyfield is not set
unless `keyfield` is given.

Args:
    lod: The dicts. They should all have the same keys.
    cols: Column names, starting with the name of the column of keys.
    keyfield: Column whose values identify rows.
    dtypes: Type for each column of the dicts, before the change.

Returns:
    The new Daf.

Examples:
    >>> d = Daf.from_lod_to_cols([{'A': 1, 'B': 2}, {'A': 4, 'B': 5}], cols=['Feature', 'T1', 'T2'])
    >>> d.lol
    [['A', 1, 4], ['B', 2, 5]]
'''

D['Daf.from_excel_buff'] = r'''
Make a Daf from the bytes of an xlsx file.

The first sheet is turned into CSV by the `xlsx2csv` package. The CSV is then
read like any CSV text. All values start as text, so give `dtypes` or call
`apply_dtypes()` to convert them.

Short rows are not padded here. `xlsx2csv` pads them. See `xlsx_to_csv()`.

Args:
    excel_buff: The bytes of the xlsx file.
    keyfield: Column whose values identify rows.
    dtypes: Type for each column.
    noheader: If True, the first row is data, not column names.
    user_format: If True, skip comment lines and blank lines.
    unflatten: If True, read list and dict columns from their text.

Returns:
    The new Daf.
'''

D['Daf.from_csv'] = r'''
Read a CSV file, a web address or an S3 object into a Daf.

The source may be a path, a `Path`, an `http` or `https` address, or an
`s3://bucket/key` name. The `requests` and `boto3` packages are imported only
when they are needed.

Every value is read as text. Give `dtypes`, or call `apply_dtypes()`, to
convert them. The other keyword arguments, such as `keyfield`, are those of
`from_csv_buff()`.

Args:
    source: A file path, an `http` address or an `s3://` name.
    **kwargs: Passed on to `from_csv_buff()`.

Returns:
    The new Daf.

Raises:
    RuntimeError: The download fails, a needed package is missing, or the
        local file cannot be read or parsed. The message says which.

Examples:
    >>> import os, tempfile
    >>> path = os.path.join(tempfile.mkdtemp(), 'x.csv')
    >>> with open(path, 'w') as f:
    ...     n = f.write('id,v\n1,a\n')
    >>> Daf.from_csv(path).lol
    [['1', 'a']]
'''

D['Daf.from_csv_buff'] = r'''
Make a Daf from CSV text, bytes or an iterator of lines.

The first row holds the column names, unless `noheader` is True. Quoted fields
may hold commas. Empty rows at the end are dropped. Cells are text, unless
`dtypes` is given. Then the cells are converted, and list and dict columns
are read from their text unless `unflatten` is False.

Args:
    csv_buff: The CSV, as text, bytes or an iterator of lines.
    keyfield: Column, or tuple or list of columns, whose values identify rows.
    dtypes: Type for each column.
    noheader: If True, the first row is data, and the Daf has no column names.
    user_format: If True, skip comment lines and blank lines.
    sep: The character that separates fields.
    unflatten: If True, read list and dict columns from their text.
    include_cols: Accepted, but it has no effect.
    name: Name of the new Daf.

Returns:
    The new Daf.

Examples:
    >>> d = Daf.from_csv_buff('id,v\n1,a\n2,"b,c"\n')
    >>> d.columns(), d.lol
    (['id', 'v'], [['1', 'a'], ['2', 'b,c']])
    >>> Daf.from_csv_buff('id,v\n1,a\n', dtypes={'id': int, 'v': str}).lol
    [[1, 'a']]
'''

D['Daf.from_csv_file'] = r'''
Read a CSV file into a Daf. Deprecated, use `from_csv()`.

Unlike `from_csv()`, a file that cannot be read prints a message and returns
None. It does not raise an error. The file is read with the default encoding.

Args:
    filepath: Path of the file.
    keyfield: Column, or tuple or list of columns, whose values identify rows.
    dtypes: Type for each column.
    noheader: If True, the first row is data, and the Daf has no column names.
    user_format: If True, skip comment lines and blank lines.
    sep: The character that separates fields.
    unflatten: If True, read list and dict columns from their text.
    include_cols: Accepted, but it has no effect.
    name: Name of the new Daf.

Returns:
    The new Daf, or None if the file could not be read.
'''

D['Daf.to_csv_file'] = r'''
Write the Daf to a CSV file.

The file has a header row of column names unless `include_header` is False.
Each cell is written as text, so lists and dicts are written in their `str()`
form. A NULL cell is written as nothing. The default line ending is `\r\n`.

Args:
    file_path: Where to write the file.
    line_terminator: The line ending. If None, `\r\n` is used.
    include_header: If True, write the column names first.

Returns:
    The path that was written.

Examples:
    >>> import os, tempfile
    >>> path = os.path.join(tempfile.mkdtemp(), 'x.csv')
    >>> d = Daf(lol=[[1, 'a']], cols=['id', 'v'])
    >>> d.to_csv_file(path) == path
    True
'''

D['Daf.to_csv_buff'] = r'''
Make CSV text from the Daf.

The text can be saved to a file or uploaded. There is no need to call
`flatten()` first. Each cell is written with `str()`, so a list or dict is
written as its Python text, with single quotes. The bool `True` is written as
`True`. A NULL cell is written as nothing. The text is not JSON.

Args:
    line_terminator: The line ending. If None, `\r\n` is used.
    include_header: If True, write the column names first.

Returns:
    The CSV text.

Examples:
    >>> d = Daf(lol=[[1, 'a']], cols=['id', 'v'])
    >>> d.to_csv_buff(line_terminator='\n')
    'id,v\n1,a\n'
'''

D['Daf.buff_to_file'] = r'''
Write text or bytes to a file.

This is a static method, so call it as `Daf.buff_to_file(buff, path)`. It is
the helper that `to_csv_file()` uses.

Args:
    buff: The text or bytes to write.
    file_path: Where to write.
    fmt: The file format, such as `.csv`.

Returns:
    The path that was written.
'''

D['Daf.from_directory'] = r'''
Make a Daf that lists the files in a folder.

Each row describes one file. The columns are the path, the folder, the name,
the name without its extension, the extension, the size in bytes, and the
modified and created times. There is also an `is_dir` column, which is always
0 because folders are not listed. Paths use `/` on every system.

The `schema` chooses the columns of the result. A `@schemaclass` that lists
only some of the fields above keeps only those. Other columns of the schema
get their defaults. The new Daf has no keyfield. Files that cannot be read
are skipped.

The method prints the time it took, which you may not want in a script.
Only local folders are supported.

Args:
    source: The folder to list.
    schema: A `@schemaclass` that chooses the columns. If None, the built in one is used.
    recursive: If True, list files in all sub folders. Otherwise only the folder itself.
    file_pat: A regular expression. Only file names that match it are listed. Case is ignored.

Returns:
    The new Daf.
'''

D['Daf.from_numpy'] = r'''
Make a Daf from a NumPy array.

The values become plain Python values. A two dimensional array gives one row
for each row of the array. A one dimensional array gives a single row.

NumPy arrays hold one type. If the array mixes numbers and text, NumPy has
already turned everything into text before this method sees it.

Args:
    npa: The NumPy array.
    keyfield: Column, or tuple or list of columns, whose values identify rows.
    cols: Column names.
    name: Name of the new Daf.

Returns:
    The new Daf.

Examples:
    >>> import numpy as np
    >>> Daf.from_numpy(np.array([[1, 2], [3, 4]]), cols=['a', 'b']).lol
    [[1, 2], [3, 4]]
'''

D['Daf.to_numpy'] = r'''
Make a NumPy array of the rows.

Column names, the keyfield and any dtypes are not kept. NumPy picks one type
for the whole array. If the Daf mixes numbers and text, every value becomes
text.

Returns:
    The NumPy array.

Examples:
    >>> Daf(lol=[[1, 2.5]], cols=['a', 'b']).to_numpy().tolist()
    [[1.0, 2.5]]
'''

D['Daf.to_donpa'] = r'''
Make a dict of NumPy arrays, one array for each column.

This is a light form of a DataFrame. The arrays can be used in vector
operations, such as `donpa['D_pct'] = donpa['D_votes'] / donpa['RV_total']`.
Each column gets its own type, so numbers and text can be mixed.

Args:
    colnames: The columns to include. If None, all columns.
    default: Passed to `col()`. It does not replace NULL cells.

Returns:
    A dict that maps each column name to an array.

Examples:
    >>> Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v']).to_donpa(['id'])['id'].tolist()
    [1, 2]
'''

D['Daf.from_googlesheet'] = r'''
Read a Google Sheet into a Daf. This is unfinished.

The sheet is read with the Google API. The path of the service account file in
the source is a placeholder, `path/to/your/service_account.json`. Edit it
before you use this method. The columns are named `A`, `B` and so on, and the
first row of the sheet is data.

Args:
    spreadsheet_id: The ID of the Google Sheet.
    sheetname: The name of the sheet.

Returns:
    The new Daf.
'''

D['Daf.to_googlesheet'] = r'''
Write the Daf to a Google Sheet. This is unfinished.

The rows are written without the column names, starting at cell A1. The path
of the service account file in the source is a placeholder,
`path/to/your/service_account.json`. Edit it before you use this method.
A message is printed when the write is done.

Args:
    spreadsheet_id: The ID of the Google Sheet.
    sheetname: The name of the sheet.

Returns:
    This Daf.
'''

D['Daf.to_json'] = r'''
Make JSON text that holds the whole Daf.

The text holds the rows, the column names, the dtypes, the keyfield, the name,
the attrs and the display columns. With `concise=True`, the parts that are
empty are left out. Use `from_json()` to read it back.

Types are written by name. Only `int`, `float`, `str` and `bool` are read back
as types. Other names, such as `list`, come back as text.

Cells must be JSON values. A tuple comes back as a list, and a set raises a
`TypeError`. A NaN is written as `NaN`, which some JSON readers reject.

If the Daf has no dtypes, they are set to an empty dict as a side effect.

Args:
    concise: If True, leave out the parts that are empty.

Returns:
    The JSON text.

Examples:
    >>> Daf(lol=[[1]], cols=['a']).to_json()
    '{"lol": [[1]], "hd": {"a": 0}}'
'''

D['Daf.from_json'] = r'''
Make a Daf from JSON text made by `to_json()`.

Args:
    json_str: The JSON text.

Returns:
    The new Daf.

Examples:
    >>> d = Daf(lol=[[1, 'a']], cols=['id', 'v'], keyfield='id')
    >>> Daf.from_json(d.to_json()) == d
    True
'''

setdoc('src/daffodil/daf.py', D)
