# daf_pandas.py
"""

# Daf -- Daffodil -- python dataframes.

The Daf class provides a lightweight, simple and fast alternative to provide 
2-d data arrays with mixed types.

This file handles indexing with square brackets[] as functions that operate on
a daf instance 'self'.

"""

"""
    MIT License

    Copyright (c) 2024 Ray Lutz

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""


"""
See README file at this location: https://github.com/raylutz/daffodil/blob/main/README.md
"""

from daffodil.lib.daf_types import T_df, T_dtype_dict, T_ls # noqa: F401
                            #, T_li, T_doda, T_lb
                            # T_lola, T_da, T_di, T_loda, T_dola, T_dodi, T_la, T_lota, T_buff, T_ds, 
                     
#import numpy as np
import csv
import io
import pandas as pd
# import daffodil.lib.daf_utils    as daf_utils

from typing import List, Dict, Any, Tuple, Optional, Union, cast, Type, Callable, TYPE_CHECKING # noqa: F401

if TYPE_CHECKING:       # for the annotations only. A real import would be circular.
    from daffodil.daf import Daf

# define a sentinel object to express a missing item where None is a valid value.
from .daf_utils import _MISSING

#==== Pandas
# see daf_pdf.py's _from_pdf() for why @classmethod on a module-level function here is
# deliberate (wired onto Daf in daf.py: `from_pandas_df = daf_pandas._from_pandas_df`), not a
# mistake mypy can see through.
@classmethod  # type: ignore[misc]
def _from_pandas_df(
        cls,
        df: T_df, 
        keyfield: str='', 
        name: str='', 
        use_csv: bool=False, 
        dtypes: Optional[T_dtype_dict]=None
        ) -> 'Daf':  # -> 'Daf'
    """
    Make a Daf from a Pandas DataFrame or Series.

    The values become plain Python values. The column names come from the
    DataFrame. The index is not kept. The dtypes are worked out from the Pandas
    dtypes, but the values are not converted to them. A Series becomes one row,
    with the index labels as column names. Its dtypes are the Python types of its
    values, one for each label.

    With `use_csv=True` the DataFrame is turned into CSV text first, and that is
    read back. This can be faster for some frames.

    Args:
        df: The DataFrame or Series.
        keyfield: Column, or tuple or list of columns, whose values identify rows.
        name: Name of the new Daf.
        use_csv: If True, convert by way of CSV text.
        dtypes: Deprecated, and it is not used. The dtypes come from `df`. It will be removed. A
            `DeprecationWarning` is given if it is passed.

    Returns:
        The new Daf.

    Examples:
        >>> import pandas as pd
        >>> from daffodil.daf import Daf
        >>> df = pd.DataFrame({'id': [1, 2], 'v': ['a', 'b']})
        >>> d = Daf.from_pandas_df(df, keyfield='id')
        >>> d.lol
        [[1, 'a'], [2, 'b']]
    """
    import pandas as pd     # type: ignore
    import warnings

    if dtypes is not None:
        warnings.warn("from_pandas_df(): the dtypes argument is deprecated and is not used. "
                      "The dtypes come from the DataFrame. It will be removed.",
                      DeprecationWarning, stacklevel=2)

    python_dtypes = dtypes_dict_from_dataframe(df)

    if isinstance(df, pd.Series) or not use_csv:

        if isinstance(df, pd.Series):
            rowdict = df.to_dict()
            cols = list(rowdict.keys())
            lol = [list(rowdict.values())]
            python_dtypes = {label: (type(value) if value is not None else str) for label, value in rowdict.items()}
        else:
            cols = list(df.columns)
            lol = df.values.tolist()

        return cls(cols=cols, lol=lol, keyfield=keyfield, name=name, dtypes=python_dtypes)
        
    # first convert the Pandas df to a csv buffer.
    try:
        csv_buff = df.to_csv(None, index=False, quoting=csv.QUOTE_MINIMAL, lineterminator= '\r\n')
    except TypeError:   # pragma: no cover
        # this uses the old version of the lineterminator with an underscore. 
        csv_buff = df.to_csv(None, index=False, quoting=csv.QUOTE_MINIMAL, line_terminator= '\r\n')

    return cls.from_csv_buff(
        csv_buff=csv_buff,
        keyfield=keyfield,
        name=name,
        dtypes=python_dtypes,    
        unflatten=False,  
        )
        

def _to_pandas_df(
    self, 
    cols: Optional[T_ls] = None,
    *,
    use_csv: bool = False, 
    use_donpa: bool = False, 
    default: Any = _MISSING,
    defaulting_cols: Optional[T_ls] = None,
    ) -> Any:
    """
    Make a Pandas DataFrame from the Daf.

    Daffodil stores a missing value as an empty string. Pandas treats that as
    text, so a column with empty cells is of type `object`, even if the rest are
    numbers. Text that looks like a number stays text. Convert columns afterwards
    if you want other types.

    Giving `default` replaces empty and None cells in the Daf itself, before the
    conversion. That changes the Daf you called it on. Copy it first if you need
    it unchanged. `default` cannot be used with `use_csv`. With `use_donpa`, the default
    replaces the cells in the arrays only. The Daf is not changed, and it goes to every
    included column, as `defaulting_cols` is not used on that route.

    Args:
        cols: Names, or positions, of the columns to include. If None, all columns.
        use_csv: If True, convert by way of CSV text.
        use_donpa: If True, convert by way of `to_donpa()`. This suits numeric columns.
        default: Value that replaces empty and None cells. This changes the Daf, except with `use_donpa`.
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
    """

    selected_cols = cols if cols is not None else self.columns()
    default_cols = defaulting_cols if defaulting_cols is not None else selected_cols

    if use_donpa:
        donpa = self.to_donpa(selected_cols, default=default)
        df = pd.DataFrame(donpa)
    elif use_csv:
        csv_buff = self.to_csv_buff()
        sio = io.StringIO(csv_buff)
        df = pd.read_csv(sio, na_filter=False, index_col=False)
        
        # No Daffodil-native defaulting possible in this path
        if default is not _MISSING:
            raise NotImplementedError("default + use_csv=True is not supported with Daffodil-native logic.")
            
    else:
        # Apply default substitution before DataFrame conversion
        if default is not _MISSING:
            self.replace_in_columns(default_cols, ['', None], default)
       
        df = pd.DataFrame(self[:, selected_cols], columns=selected_cols)

    return df
        

def pandas_dtype_dict_to_python(pandas_dtype_dict: Any) -> Dict[Any, type]:
    """ Map each column's pandas dtype to a Python type.
        Kept for compatibility. It uses pandas_dtype_to_python_type(), the same mapping as
        Daf.from_pandas_df().
    """
    return {colname: pandas_dtype_to_python_type(pandas_dtype)
            for colname, pandas_dtype in pandas_dtype_dict.items()}


def python_dtype_to_pandas(python_type: Type) -> Optional[Any]:
    """
    Translate a Python data type to its equivalent Pandas data type.

    Args:
        python_type (Type): The Python data type to translate.

    Returns:
        Optional[Any]: The equivalent Pandas data type.
    """
    import pandas as pd

    dtype_mapping = {
        str: pd.StringDtype(),              # modern pandas string
        int: "int64",                       # or "Int64" if you want nullable
        float: "float64",
        bool: "bool",                       # or "boolean" for nullable
        pd.Timestamp: "datetime64[ns]",
        pd.Timedelta: "timedelta64[ns]",
        pd.Categorical: "category",
    }
 
    return dtype_mapping.get(python_type, None)
    

def dtypes_dict_from_dataframe(df: pd.DataFrame) -> Dict[str, type]:
    import pandas as pd

    # --- Series case (single column) ---
    if isinstance(df, pd.Series):
        dtypes_dict = {
            df.name if df.name is not None else "col":
            pandas_dtype_to_python_type(df.dtype)
        }
        return dtypes_dict

    # --- DataFrame case ---
    dtypes_dict = {}
    for colname, dtype in df.dtypes.items():
        dtypes_dict[colname] = pandas_dtype_to_python_type(dtype)

    return dtypes_dict


def pandas_dtype_to_python_type(dtype: Any) -> type:
    import pandas as pd
    import numpy as np

    # --- already Python types ---
    if dtype is str or dtype is int or dtype is float or dtype is bool:
        return dtype

    # --- string aliases ---
    if isinstance(dtype, str):
        d = dtype.lower()
        if d in ("object", "string"):
            return str
        elif d.startswith("int"):
            return int
        elif d.startswith("float"):
            return float
        elif d in ("bool", "boolean"):
            return bool
        elif "datetime" in d:
            return pd.Timestamp
        elif "timedelta" in d:
            return pd.Timedelta
        else:
            return str

    # --- numpy / pandas numeric + datetime handling ---
    # NOTE: np.timedelta64 MUST be checked before np.integer. numpy considers timedelta64 a
    # subtype of integer (it's stored internally as an integer count of time units, e.g.
    # nanoseconds), so np.issubdtype(timedelta64_dtype, np.integer) returns True. If the integer
    # check were moved above this one, every timedelta64 column would be silently misclassified
    # as int instead of pd.Timedelta. (datetime64 does not have this issue -- only timedelta64.)
    try:
        if np.issubdtype(dtype, np.timedelta64):
            return pd.Timedelta
        elif np.issubdtype(dtype, np.integer):
            return int
        elif np.issubdtype(dtype, np.floating):
            return float
        elif np.issubdtype(dtype, np.bool_):
            return bool
        elif np.issubdtype(dtype, np.datetime64):
            return pd.Timestamp
    except TypeError:
        pass

    # --- pandas extension dtypes ---
    if pd.api.types.is_string_dtype(dtype):
        return str
    elif pd.api.types.is_integer_dtype(dtype):
        return int
    elif pd.api.types.is_float_dtype(dtype):
        return float
    elif pd.api.types.is_bool_dtype(dtype):
        return bool
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return pd.Timestamp
    elif pd.api.types.is_timedelta64_dtype(dtype):
        return pd.Timedelta

    # --- fallback ---
    return str