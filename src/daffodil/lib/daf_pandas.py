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
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from daffodil.lib.daf_types import T_df, T_dtype_dict #T_ls, T_li, T_doda, T_lb
                            # T_lola, T_da, T_di, T_hllola, T_loda, T_dola, T_dodi, T_la, T_lota, T_buff, T_df, T_ds, 
                     
import numpy as np
import csv
import io
import pandas as pd


from typing import List, Dict, Any, Tuple, Optional, Union, cast, Type, Callable #
def fake_function(a: Optional[List[Dict[str, Tuple[int,Union[Any, str, Type, Callable ]]]]] = None) -> Optional[int]:
    return None or cast(int, 0)   # pragma: no cover




#==== Pandas
@classmethod
def _from_pandas_df(
        cls,
        df: T_df, 
        keyfield: str='', 
        name: str='', 
        use_csv: bool=False, 
        dtypes: Optional[T_dtype_dict]=None
        ):
    """
    Convert a Pandas dataframe to daf object
        @@TODO: This does not enforce dtypes are correct.
    """
    import pandas as pd     # type: ignore
    
    python_dtypes = pandas_dtype_to_python(df)

    if isinstance(df, pd.Series) or not use_csv:

        if isinstance(df, pd.Series):
            rowdict = df.to_dict()
            cols = list(rowdict.keys())
            lol = [list(rowdict.values())]
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
        dtypes=python_dtypes,    
        unflatten=False,  
        )
        

def _to_pandas_df(self, use_csv: bool=False) -> Any:

    import pandas as pd     # type: ignore

    if not use_csv:
        columns = self.columns()
        # return pd.DataFrame(self.lol, columns=columns, dtypes=self.dtypes)
        # above results in NotImplementedError: compound dtypes are not implemented in the DataFrame constructor

        return pd.DataFrame(self.lol, columns=columns)
        
    # it seems this may work faster if we first convert the data to a csv_buff internally,
    # and then convert that to a df.

    csv_buff = self.to_csv_buff()
    sio = io.StringIO(csv_buff)            
    df  = pd.read_csv(sio, 
        na_filter=False, 
        index_col=False, 
        #dtype=self.dtypes,
        #sep=sep,
        usecols=None,
        #comment='#', 
        #skip_blank_lines=True
        )
    return df
    

def pandas_dtype_to_python(df: T_df) -> Optional[Any]:
    """
    Translate a Pandas data type to its equivalent Python data type.

    Args:
        pandas_dtype (Any): The Pandas data type to translate.

    Returns:
        Optional[Type]: The equivalent Python data type.
    """


    if isinstance(df, pd.Series):
        # For Series, it is necessary to examine each element
        series_dtypes = dtypes_dict_from_series(df)
        python_dtypes = pandas_dtype_dict_to_python(series_dtypes)
        
    else:
        python_dtypes = pandas_dtype_dict_to_python(df.dtypes.to_dict())
        
    return python_dtypes


def pandas_dtype_dict_to_python(pandas_dtype_dict: Any) -> Optional[Any]:
    """
    Translate a Pandas data type to its equivalent Python data type.

    Args:
        pandas_dtype (Any): The Pandas data type to translate.

    Returns:
        Optional[Type]: The equivalent Python data type.
    """
    python_dtype_dict = {}
    for colname, pandas_dtype in pandas_dtype_dict.items():
        if pandas_dtype == np.object_ or pandas_dtype == str:
            python_dtype_dict[colname] = str
        elif pandas_dtype in (np.int64, np.int32):
            python_dtype_dict[colname] = int
        elif pandas_dtype == np.float64:
            python_dtype_dict[colname] = float
        elif pandas_dtype == np.bool_:
            python_dtype_dict[colname] = bool
        elif pandas_dtype == np.datetime64:
            python_dtype_dict[colname] = pd.Timestamp
        elif pandas_dtype == np.timedelta64:
            python_dtype_dict[colname] = pd.Timedelta
        else:
            print(f"Unknown Pandas dtype for column '{colname}': {pandas_dtype}")
            breakpoint() #perm
            pass
            
    return python_dtype_dict


def python_dtype_to_pandas(python_type: Type) -> Optional[Any]:
    """
    Translate a Python data type to its equivalent Pandas data type.

    Args:
        python_type (Type): The Python data type to translate.

    Returns:
        Optional[Any]: The equivalent Pandas data type.
    """
    dtype_mapping = {
        str: "object",
        int: "int64",
        float: "float64",
        bool: "bool",
        pd.Timestamp: "datetime64[ns]",
        pd.Timedelta: "timedelta64[ns]",
        pd.Categorical: "category",
    }
 
    return dtype_mapping.get(python_type, None)
    

def dtypes_dict_from_series(series: pd.Series) -> Dict[str, type]:
    """
    Construct a dictionary mapping column names to their respective data types from a Pandas Series.

    Args:
        series (pd.Series): The Pandas Series.

    Returns:
        Dict[str, type]: A dictionary mapping column names to their respective data types.
    """
    
    dtypes_dict = {}
    for column_name in series.index:
        dtype = series.dtype
        dtype_name = np.dtype(dtype).name  # Get the name of the NumPy dtype
        dtype_type = type(series[column_name])
        dtypes_dict[column_name] = dtype_type if dtype_name == 'object' else dtype_type.__base__
    return dtypes_dict
    