# test_daf_pandas.py
#
# Tests for daffodil/lib/daf_pandas.py's standalone dtype-mapping helper functions
# (pandas_dtype_to_python_type, python_dtype_to_pandas, pandas_dtype_dict_to_python,
# dtypes_dict_from_dataframe) and the use_csv + default combination in _to_pandas_df.
# Daf.from_pandas_df()/Daf.to_pandas_df() themselves are already covered in test_daf.py.

import pytest
import pandas as pd
import numpy as np

from daffodil.daf import Daf
from daffodil.lib import daf_pandas


# --- pandas_dtype_to_python_type ---

def test_pandas_dtype_to_python_type_already_python_types():
    assert daf_pandas.pandas_dtype_to_python_type(str) is str
    assert daf_pandas.pandas_dtype_to_python_type(int) is int
    assert daf_pandas.pandas_dtype_to_python_type(float) is float
    assert daf_pandas.pandas_dtype_to_python_type(bool) is bool


def test_pandas_dtype_to_python_type_string_aliases():
    assert daf_pandas.pandas_dtype_to_python_type('object') is str
    assert daf_pandas.pandas_dtype_to_python_type('string') is str
    assert daf_pandas.pandas_dtype_to_python_type('int64') is int
    assert daf_pandas.pandas_dtype_to_python_type('float64') is float
    assert daf_pandas.pandas_dtype_to_python_type('bool') is bool
    assert daf_pandas.pandas_dtype_to_python_type('boolean') is bool
    assert daf_pandas.pandas_dtype_to_python_type('datetime64[ns]') is pd.Timestamp
    assert daf_pandas.pandas_dtype_to_python_type('timedelta64[ns]') is pd.Timedelta


def test_pandas_dtype_to_python_type_unknown_string_alias_defaults_to_str():
    assert daf_pandas.pandas_dtype_to_python_type('some_unknown_dtype') is str


def test_pandas_dtype_to_python_type_numpy_dtypes():
    assert daf_pandas.pandas_dtype_to_python_type(np.dtype('int64')) is int
    assert daf_pandas.pandas_dtype_to_python_type(np.dtype('float64')) is float
    assert daf_pandas.pandas_dtype_to_python_type(np.dtype('bool')) is bool
    assert daf_pandas.pandas_dtype_to_python_type(np.dtype('datetime64[ns]')) is pd.Timestamp


def test_pandas_dtype_to_python_type_timedelta64_not_misclassified_as_int():
    # numpy considers timedelta64 a subtype of integer (it's stored as integer ns counts
    # internally) -- this must be checked before np.integer or it gets misclassified.
    assert daf_pandas.pandas_dtype_to_python_type(np.dtype('timedelta64[ns]')) is pd.Timedelta


def test_pandas_dtype_to_python_type_extension_dtype():
    assert daf_pandas.pandas_dtype_to_python_type(pd.StringDtype()) is str


def test_pandas_dtype_to_python_type_nullable_extension_dtypes():
    # np.issubdtype doesn't recognize these pandas-specific extension dtypes (raises TypeError,
    # caught), so they fall through to the pd.api.types.is_* extension-dtype branches.
    assert daf_pandas.pandas_dtype_to_python_type(pd.Int64Dtype()) is int
    assert daf_pandas.pandas_dtype_to_python_type(pd.Float64Dtype()) is float
    assert daf_pandas.pandas_dtype_to_python_type(pd.BooleanDtype()) is bool
    assert daf_pandas.pandas_dtype_to_python_type(pd.CategoricalDtype()) is str


# --- python_dtype_to_pandas ---

def test_python_dtype_to_pandas_basic_types():
    assert daf_pandas.python_dtype_to_pandas(str) == pd.StringDtype()
    assert daf_pandas.python_dtype_to_pandas(int) == 'int64'
    assert daf_pandas.python_dtype_to_pandas(float) == 'float64'
    assert daf_pandas.python_dtype_to_pandas(bool) == 'bool'


def test_python_dtype_to_pandas_datetime_types():
    assert daf_pandas.python_dtype_to_pandas(pd.Timestamp) == 'datetime64[ns]'
    assert daf_pandas.python_dtype_to_pandas(pd.Timedelta) == 'timedelta64[ns]'


def test_python_dtype_to_pandas_unmapped_type_returns_none():
    assert daf_pandas.python_dtype_to_pandas(complex) is None


# --- pandas_dtype_dict_to_python ---

def test_pandas_dtype_dict_to_python_mixed_columns():
    df = pd.DataFrame({'a': [1, 2], 'b': [1.5, 2.5], 'c': ['x', 'y'], 'd': [True, False]})
    result = daf_pandas.pandas_dtype_dict_to_python(df.dtypes.to_dict())
    assert result == {'a': int, 'b': float, 'c': str, 'd': bool}


def test_pandas_dtype_dict_to_python_datetime_column():
    df = pd.DataFrame({'a': pd.to_datetime(['2024-01-01', '2024-01-02'])})
    result = daf_pandas.pandas_dtype_dict_to_python(df.dtypes.to_dict())
    assert result == {'a': pd.Timestamp}


# --- dtypes_dict_from_dataframe ---

def test_dtypes_dict_from_dataframe_basic():
    df = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
    result = daf_pandas.dtypes_dict_from_dataframe(df)
    assert result == {'a': int, 'b': str}


def test_dtypes_dict_from_dataframe_series_with_name():
    series = pd.Series([1, 2, 3], name='myseries')
    result = daf_pandas.dtypes_dict_from_dataframe(series)
    assert result == {'myseries': int}


def test_dtypes_dict_from_dataframe_series_without_name():
    series = pd.Series([1, 2, 3])
    result = daf_pandas.dtypes_dict_from_dataframe(series)
    assert result == {'col': int}


# --- _to_pandas_df: use_csv + default combination ---

def test_to_pandas_df_use_csv_with_default_raises():
    daf = Daf(lol=[[1, ''], [2, 'x']], cols=['id', 'val'])
    with pytest.raises(NotImplementedError):
        daf.to_pandas_df(use_csv=True, default=0)
