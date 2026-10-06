import sys
sys.path.insert(0, '/tmp/claude-0/-home-user-daffodil/95e36c09-64f8-59ba-9bd6-c962257ad9e3/scratchpad')
from setdoc import setdoc
P = {}
P['_from_pandas_df'] = r'''
Make a Daf from a Pandas DataFrame or Series.

The values become plain Python values. The column names come from the
DataFrame. The index is not kept. The dtypes are worked out from the Pandas
dtypes, but the values are not converted to them. A Series becomes one row,
with the index labels as column names. Its dtypes dict is keyed `col`, which
does not match those names.

With `use_csv=True` the DataFrame is turned into CSV text first, and that is
read back. This can be faster for some frames.

Args:
    df: The DataFrame or Series.
    keyfield: Column, or tuple or list of columns, whose values identify rows.
    name: Name of the new Daf.
    use_csv: If True, convert by way of CSV text.
    dtypes: Accepted, but it is not used. The dtypes come from `df`.

Returns:
    The new Daf.

Examples:
    >>> import pandas as pd
    >>> from daffodil.daf import Daf
    >>> df = pd.DataFrame({'id': [1, 2], 'v': ['a', 'b']})
    >>> d = Daf.from_pandas_df(df, keyfield='id')
    >>> d.lol
    [[1, 'a'], [2, 'b']]
'''
P['_to_pandas_df'] = r'''
Make a Pandas DataFrame from the Daf.

Daffodil stores a missing value as an empty string. Pandas treats that as
text, so a column with empty cells is of type `object`, even if the rest are
numbers. Text that looks like a number stays text. Convert columns afterwards
if you want other types.

Giving `default` replaces empty and None cells in the Daf itself, before the
conversion. That changes the Daf you called it on. Copy it first if you need
it unchanged. `default` cannot be used with `use_csv`.

Args:
    cols: Names, or positions, of the columns to include. If None, all columns.
    use_csv: If True, convert by way of CSV text.
    use_donpa: If True, convert by way of `to_donpa()`. This suits numeric columns.
    default: Value that replaces empty and None cells. This changes the Daf.
    defaulting_cols: The columns that `default` applies to. If None, all included columns.

Returns:
    The DataFrame.

Raises:
    NotImplementedError: `default` is given with `use_csv=True`.

Examples:
    >>> from daffodil.daf import Daf
    >>> d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'])
    >>> d.to_pandas_df().values.tolist()
    [[1, 'a'], [2, 'b']]
'''
setdoc('src/daffodil/lib/daf_pandas.py', P)
