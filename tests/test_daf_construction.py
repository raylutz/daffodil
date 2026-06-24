# test_daf_construction.py
#
# Tests for Daf.__init__, copy, from_lot, set_dtypes/apply_dtypes, append/_basic_append,
# insert_dif_row, and the CSV I/O round trip (from_csv, from_csv_file, to_csv_file, to_csv_buff,
# buff_to_file). These are the entry points most heavily exercised via load_data()/save_data()
# style usage in downstream projects.

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import requests

from daffodil.daf import Daf
from daffodil.keyedlist import KeyedList


# =====================================================================
# __init__
# =====================================================================

def test_init_cols_as_single_string():
    daf = Daf(cols='single_col_as_str')
    assert list(daf.hd.keys()) == ['single_col_as_str']


def test_init_disp_cols_invalid_type_raises():
    with pytest.raises(TypeError):
        Daf(disp_cols=123)


def test_init_hd_and_kd_adopted_directly():
    daf = Daf(lol=[[1, 2]], hd={'a': 0, 'b': 1}, kd={'x': 0})
    assert daf.hd == {'a': 0, 'b': 1}
    assert daf._kd == {'x': 0}


def test_init_cols_from_dtypes_when_no_cols_given():
    daf = Daf(dtypes={'a': int, 'b': str})
    assert list(daf.hd.keys()) == ['a', 'b']


def test_init_use_copy_deep_copies_lol():
    original_lol = [[1, 2]]
    daf = Daf(lol=original_lol, use_copy=True)
    daf.lol[0][0] = 99
    assert original_lol[0][0] == 1


# =====================================================================
# copy
# =====================================================================

def test_copy_shallow_independent_attrs():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    daf.attrs = {'meta': 'x'}
    copied = daf.copy()
    copied.attrs['meta'] = 'y'
    assert daf.attrs == {'meta': 'x'}
    assert copied.attrs == {'meta': 'y'}


def test_copy_deep_creates_independent_lol():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    copied = daf.copy(deep=True)
    assert copied.lol is not daf.lol
    copied.lol[0][0] = 99
    assert daf.lol[0][0] == 1


def test_copy_for_sorting_independent_lol_reference():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    copied = daf.copy(for_sorting=True)
    assert copied.lol is not daf.lol


# =====================================================================
# from_lot
# =====================================================================

def test_from_lot_basic():
    lot = [(1, 'Alice', 30), (2, 'Bob', 25)]
    daf = Daf.from_lot(lot)
    assert list(daf.hd.keys()) == ['col_0', 'col_1', 'col_2']
    assert daf.lol == [[1, 'Alice', 30], [2, 'Bob', 25]]


def test_from_lot_with_cols():
    lot = [(1, 'Alice'), (2, 'Bob')]
    daf = Daf.from_lot(lot, cols=['id', 'name'])
    assert list(daf.hd.keys()) == ['id', 'name']


def test_from_lot_empty_returns_empty_daf():
    daf = Daf.from_lot([])
    assert daf.lol == []


# =====================================================================
# set_dtypes / apply_dtypes
# =====================================================================

def test_set_dtypes_with_typ_to_cols_dict():
    daf = Daf(lol=[[1, 'a', 2.5]], cols=['id', 'name', 'val'])
    daf.set_dtypes(default_type=str, typ_to_cols_dict={int: ['id'], float: ['val']})
    assert daf.dtypes == {'id': int, 'name': str, 'val': float}


def test_set_dtypes_no_hd_raises():
    with pytest.raises(NotImplementedError):
        Daf().set_dtypes()


def test_apply_dtypes_mismatch_raises():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    with pytest.raises(ValueError):
        daf.apply_dtypes(dtypes={'id': int}, silent_error=False)


def test_apply_dtypes_converts_columns():
    daf = Daf(lol=[['1', '2.5']], cols=['id', 'val'])
    daf.apply_dtypes(dtypes={'id': int, 'val': float})
    assert daf.lol == [[1, 2.5]]


def test_apply_dtypes_no_dtypes_is_noop():
    daf = Daf(lol=[['1', 'a']], cols=['id', 'name'])
    daf.apply_dtypes()
    assert daf.lol == [['1', 'a']]


def test_apply_dtypes_empty_lol_is_noop():
    daf = Daf(lol=[], cols=['id', 'name'])
    daf.apply_dtypes(dtypes={'id': int, 'name': str})
    assert daf.lol == []


# =====================================================================
# append
# =====================================================================

def test_append_falsy_is_noop():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    daf.append(None)
    daf.append([])
    assert daf.lol == [[1, 'a']]


def test_append_dict():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    daf.append({'id': 2, 'name': 'b'})
    assert daf.lol == [[1, 'a'], [2, 'b']]


def test_append_list_of_dicts():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    daf.append([{'id': 2, 'name': 'b'}, {'id': 3, 'name': 'c'}])
    assert daf.lol == [[1, 'a'], [2, 'b'], [3, 'c']]


def test_append_simple_list_no_hd():
    daf = Daf()
    daf.append([1, 'x'])
    assert daf.lol == [[1, 'x']]
    assert daf.hd == {}


def test_append_simple_list_with_hd():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    daf.append([5, 'e'])
    assert daf.lol == [[1, 'a'], [5, 'e']]


def test_append_keyedlist_matching_hd():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    kl = KeyedList(['id', 'name'], [6, 'f'])
    daf.append(kl)
    assert daf.lol == [[1, 'a'], [6, 'f']]


def test_append_keyedlist_non_matching_hd():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    kl = KeyedList(['name', 'id'], ['g', 7])
    daf.append(kl)
    assert daf.lol == [[1, 'a'], [7, 'g']]


def test_append_daf_concat():
    daf1 = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    daf2 = Daf(lol=[[2, 'b']], cols=['id', 'name'])
    daf1.append(daf2)
    assert daf1.lol == [[1, 'a'], [2, 'b']]


def test_append_unsupported_type_raises():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    with pytest.raises(RuntimeError):
        daf.append(42)


# =====================================================================
# _basic_append
# =====================================================================

def test_basic_append_list_no_hd():
    daf = Daf()
    daf._basic_append([1, 'x'])
    assert daf.lol == [[1, 'x']]


def test_basic_append_dict_no_hd_adopts_structure():
    daf = Daf()
    daf._basic_append({'a': 1, 'b': 2})
    assert daf.hd == {'a': 0, 'b': 1}
    assert daf.lol == [[1, 2]]


def test_basic_append_dict_with_hd_aligns_order():
    daf = Daf(lol=[[1, 2]], cols=['a', 'b'])
    daf._basic_append({'b': 5, 'a': 3})
    assert daf.lol == [[1, 2], [3, 5]]


def test_basic_append_unsupported_type_raises():
    with pytest.raises(TypeError):
        Daf()._basic_append(42)


# =====================================================================
# insert_dif_row / insert_irow (working list/dict paths)
# =====================================================================

def test_insert_dif_row_default_irow2():
    daf = Daf(lol=[[1, 10], [2, 15], [3, 30]], cols=['id', 'val'])
    daf.insert_dif_row(irow1=0)
    assert daf.lol == [[1, 10], [-1, -5], [2, 15], [3, 30]]


def test_insert_dif_row_explicit_cols():
    daf = Daf(lol=[[1, 10], [2, 15], [3, 30]], cols=['id', 'val'])
    daf.insert_dif_row(irow1=0, irow2=2, irow_insert=1, cols=['val'])
    assert daf.lol == [[1, 10], ['', -20], [2, 15], [3, 30]]


def test_insert_irow_with_list():
    daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'name'])
    daf.insert_irow(irow=1, row=[99, 'z'])
    assert daf.lol == [[1, 'a'], [99, 'z'], [2, 'b']]


def test_insert_irow_with_dict():
    daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'name'])
    daf.insert_irow(irow=0, row={'id': 100, 'name': 'y'})
    assert daf.lol == [[100, 'y'], [1, 'a'], [2, 'b']]


# =====================================================================
# CSV I/O: from_csv, from_csv_file, to_csv_file, to_csv_buff, buff_to_file
# =====================================================================

def test_to_csv_buff_basic():
    daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'name'])
    buff = daf.to_csv_buff()
    assert buff == 'id,name\r\n1,a\r\n2,b\r\n'


def test_to_csv_buff_no_header():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    buff = daf.to_csv_buff(include_header=False)
    assert buff == '1,a\r\n'


def test_round_trip_local_file_basic():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'out.csv')
        daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'name'])
        result_path = daf.to_csv_file(p)
        assert result_path == p
        restored = Daf.from_csv(p)
        assert restored.lol == [['1', 'a'], ['2', 'b']]
        assert list(restored.hd.keys()) == ['id', 'name']


def test_round_trip_local_file_with_dtypes():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'out.csv')
        daf = Daf(lol=[[1, 2.5], [3, 4.5]], cols=['a', 'b'], dtypes={'a': int, 'b': float})
        daf.to_csv_file(p)
        restored = Daf.from_csv_buff(open(p).read(), dtypes={'a': int, 'b': float})
        assert restored.lol == [[1, 2.5], [3, 4.5]]


def test_round_trip_complex_cell_via_pyon():
    daf = Daf(lol=[[1, [1, 2, 3]]], cols=['id', 'items'])
    buff = daf.to_csv_buff()
    restored = Daf.from_csv_buff(buff, dtypes={'id': int, 'items': list})
    assert restored.lol == [[1, [1, 2, 3]]]


def test_from_csv_accepts_path_object():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'test.csv'
        p.write_text('a,b\n1,2\n')
        daf = Daf.from_csv(p)
        assert daf.lol == [['1', '2']]


def test_from_csv_local_file_not_found_raises_runtimeerror():
    with pytest.raises(RuntimeError):
        Daf.from_csv('/tmp/this_does_not_exist_xyz_daffodil_test.csv')


def test_from_csv_http_source(monkeypatch):
    mock_response = MagicMock()
    mock_response.iter_lines.return_value = [b'a,b', b'1,2', b'3,4']
    mock_response.raise_for_status.return_value = None

    with patch('requests.get', return_value=mock_response) as mock_get:
        daf = Daf.from_csv('https://example.com/data.csv')
        assert daf.lol == [['1', '2'], ['3', '4']]
        assert list(daf.hd.keys()) == ['a', 'b']
        mock_get.assert_called_once_with('https://example.com/data.csv', stream=True)


def test_from_csv_http_request_exception_raises_runtimeerror():
    with patch('requests.get', side_effect=requests.RequestException('connection failed')):
        with pytest.raises(RuntimeError):
            Daf.from_csv('https://example.com/data.csv')


def test_from_csv_s3_source_without_boto3_raises_runtimeerror():
    # boto3 is not installed in this environment; confirms the ImportError->RuntimeError path.
    with pytest.raises(RuntimeError):
        Daf.from_csv('s3://bucket/key.csv')


def test_buff_to_file_writes_local_file():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'out.csv')
        result = Daf.buff_to_file('a,b\n1,2\n', file_path=p)
        assert result == p
        assert open(p).read() == 'a,b\n1,2\n'


# =====================================================================
# from_csv_file (deprecated; silently returns None on error)
# =====================================================================

def test_from_csv_file_basic():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'out.csv')
        with open(p, 'w') as f:
            f.write('id,name\n1,a\n')
        daf = Daf.from_csv_file(p)
        assert daf.lol == [['1', 'a']]


def test_from_csv_file_missing_file_returns_none():
    result = Daf.from_csv_file('/tmp/this_does_not_exist_xyz_daffodil_test.csv')
    assert result is None
