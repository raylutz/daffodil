# test_daf_coverage_a
# Targeted coverage tests for src/daffodil/daf.py (lines 1-4500).

import sys
import types
from types import MappingProxyType

import numpy as np
import pytest

from daffodil.daf import Daf, KeysDisabledError
from daffodil.keyedlist import KeyedList


@pytest.fixture
def no_breakpoint(monkeypatch):
    """ Some code paths in daf.py call breakpoint() before raising/continuing.
        Neutralize it so tests never drop into a debugger.
    """
    monkeypatch.setattr(sys, 'breakpointhook', lambda *a, **k: None)


def _daf3():
    return Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['a', 'b', 'c'])


# --- __init__: column sanitization

def test_init_sanitize_cols_duplicates():
    daf = Daf(cols=['a', 'a', 'a_1', ''])
    assert daf.columns() == ['a', 'a_1', 'a_1_2', 'Unnamed3']


@pytest.mark.xfail(strict=True, reason="BUG: daf_utils._sanitize_cols renames the dup 'a' at idx 2 to "
                                       "'a_2', colliding with existing 'a_2'; the column is silently dropped")
def test_init_sanitize_cols_collision():
    daf = Daf(cols=['a_2', 'a', 'a'])
    assert len(daf.columns()) == 3
    assert len(set(daf.columns())) == 3


# --- _default_iterator / iter_list

def test_default_iterator_invalid_itermode():
    daf = _daf3()
    daf._itermode = 'bogus'
    with pytest.raises(ValueError, match='Invalid iteration mode'):
        iter(daf)


def test_iter_list_yields_raw_rows():
    daf = _daf3()
    rows = list(daf.iter_list())
    assert rows == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert rows[0] is daf.lol[0]


# --- __format__

def test_format_numeric_and_non_numeric():
    assert format(Daf(lol=[[3.14159]], cols=['a']), '.2f') == '3.14'
    # non-numeric value with a format spec falls back to str(value)
    assert format(Daf(lol=[[[1, 2]]], cols=['a']), '>10') == '[1, 2]'


# --- __contains__

def test_contains_on_empty_daf_is_false():
    assert ('x' in Daf(keyfield='a')) is False


# --- calc_cols with types but no dtypes

def test_calc_cols_include_types_without_dtypes_raises(no_breakpoint):
    daf = _daf3()
    with pytest.raises(RuntimeError):
        daf.calc_cols(include_types=int)


def test_calc_cols_exclude_types_without_dtypes_raises(no_breakpoint):
    daf = _daf3()
    with pytest.raises(RuntimeError):
        daf.calc_cols(exclude_types=int)


# --- _is_keyfield_valid

def test_is_keyfield_valid_bad_type_raises():
    daf = _daf3()
    with pytest.raises(RuntimeError, match='invalid'):
        daf._is_keyfield_valid(1.5)   # type: ignore[arg-type]
    assert daf._is_keyfield_valid(('a', 'b')) is True
    assert daf._is_keyfield_valid(('a', 'zz')) is False


# --- apply_dtypes

def test_apply_dtypes_adopts_hd_from_dtypes():
    daf = Daf(lol=[['1', '2']])
    assert daf.hd == {}
    daf.apply_dtypes(dtypes={'a': int, 'b': int})
    assert daf.columns() == ['a', 'b']
    assert daf.lol == [[1, 2]]


def test_apply_dtypes_partial_dtypes_uses_default_type():
    daf = Daf(lol=[['1', '2']], cols=['a', 'b'])
    daf.apply_dtypes(dtypes={'a': int}, silent_error=True)
    assert daf.dtypes == {'a': int, 'b': str}
    assert daf.lol == [[1, '2']]


def test_apply_dtypes_single_type_for_all_cols():
    daf = Daf(lol=[['1', '2'], ['3', '4']], cols=['a', 'b'])
    daf.dtypes = int
    daf.apply_dtypes()
    assert daf.lol == [[1, 2], [3, 4]]


# --- flatten

@pytest.mark.xfail(strict=True, reason="BUG: flatten() overwrites its use_pyon parameter with "
                                       "'use_pyon = True' (daf.py:1462), so use_pyon=False never JSON-encodes")
def test_flatten_use_pyon_false_json_encodes():
    daf = Daf(lol=[[{'x': 1}, True]], cols=['a', 'b'], dtypes={'a': dict, 'b': bool})
    daf.flatten(use_pyon=False)
    assert daf.lol == [['{"x": 1}', 1]]


# --- to_attrib_dict

def test_to_attrib_dict():
    daf = Daf(lol=[[1, 2], [3, 4]], cols=['a', 'b'])
    assert daf.to_attrib_dict() == {'cols': ['a', 'b'], 'lol': [[1, 2], [3, 4]]}


# --- from_csv: http / s3 sources (mocked, no network)

def test_from_csv_http_without_requests_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, 'requests', None)
    with pytest.raises(RuntimeError, match='requests'):
        Daf.from_csv('https://example.invalid/data.csv')


def _fake_boto3(get_object):
    mod = types.ModuleType('boto3')
    calls = {}

    class _Client:
        def get_object(self, Bucket, Key):
            calls['Bucket'] = Bucket
            calls['Key'] = Key
            return get_object()

    mod.client = lambda name: (calls.__setitem__('service', name), _Client())[1]
    return mod, calls


def test_from_csv_s3_reads_stream(monkeypatch):
    class _Body:
        def iter_lines(self):
            return iter([b'a,b', b'1,2', b'', b'3,4'])

    mod, calls = _fake_boto3(lambda: {'Body': _Body()})
    monkeypatch.setitem(sys.modules, 'boto3', mod)
    daf = Daf.from_csv('s3://my-bucket/path/to/file.csv')
    assert calls == {'service': 's3', 'Bucket': 'my-bucket', 'Key': 'path/to/file.csv'}
    assert daf.columns() == ['a', 'b']
    assert daf.lol == [['1', '2'], ['3', '4']]


def test_from_csv_s3_error_wrapped(monkeypatch):
    def _boom():
        raise ValueError('no such key')

    mod, _ = _fake_boto3(_boom)
    monkeypatch.setitem(sys.modules, 'boto3', mod)
    with pytest.raises(RuntimeError, match='Failed to read CSV from S3: no such key'):
        Daf.from_csv('s3://bucket/key.csv')


# --- from_csv_buff: trailing empty rows

def test_from_csv_buff_strips_trailing_blank_rows():
    daf = Daf.from_csv_buff("a,b\n1,2\n\n\n")
    assert daf.columns() == ['a', 'b']
    assert daf.lol == [['1', '2']]


# --- to_csv_file with Path

def test_to_csv_file_accepts_path(tmp_path):
    daf = Daf(lol=[[1, 2]], cols=['a', 'b'])
    out = tmp_path / 'out.csv'
    result = daf.to_csv_file(out)
    assert result == str(out)
    assert out.read_text().splitlines() == ['a,b', '1,2']


# --- from_directory: unstat-able entries are skipped

def test_from_directory_skips_broken_symlink(tmp_path):
    (tmp_path / 'good.txt').write_text('hello')
    try:
        (tmp_path / 'broken.txt').symlink_to(tmp_path / 'does_not_exist')
    except (OSError, NotImplementedError):
        pytest.skip('symlinks not supported')
    daf = Daf.from_directory(tmp_path)
    assert daf.col('basename') == ['good.txt']
    assert daf.col('size') == [5]


# --- to_donpa with default colnames

def test_to_donpa_all_columns():
    donpa = Daf(lol=[[1, 2], [3, 4]], cols=['a', 'b']).to_donpa()
    assert list(donpa.keys()) == ['a', 'b']
    np.testing.assert_array_equal(donpa['a'], np.array([1, 3]))
    np.testing.assert_array_equal(donpa['b'], np.array([2, 4]))


# --- from_googlesheet / to_googlesheet (mocked google api modules)

def _install_fake_google(monkeypatch, values=None):
    calls = {}

    class _Req:
        def __init__(self, result):
            self.result = result

        def execute(self):
            return self.result

    class _Values:
        def get(self, **kwargs):
            calls['get'] = kwargs
            return _Req({'values': values or []})

        def update(self, **kwargs):
            calls['update'] = kwargs
            return _Req({'updatedCells': 1})

    class _Sheets:
        def values(self):
            return _Values()

    class _Service:
        def spreadsheets(self):
            return _Sheets()

    def build(api, version, credentials=None):
        calls['build'] = (api, version, credentials)
        return _Service()

    class _Credentials:
        @staticmethod
        def from_service_account_file(path, scopes=None):
            calls['creds'] = (path, scopes)
            return 'CREDS'

    discovery = types.ModuleType('googleapiclient.discovery')
    discovery.build = build
    gapi = types.ModuleType('googleapiclient')
    gapi.discovery = discovery
    service_account = types.ModuleType('google.oauth2.service_account')
    service_account.Credentials = _Credentials
    oauth2 = types.ModuleType('google.oauth2')
    oauth2.service_account = service_account
    google = types.ModuleType('google')
    google.oauth2 = oauth2

    monkeypatch.setitem(sys.modules, 'googleapiclient', gapi)
    monkeypatch.setitem(sys.modules, 'googleapiclient.discovery', discovery)
    monkeypatch.setitem(sys.modules, 'google', google)
    monkeypatch.setitem(sys.modules, 'google.oauth2', oauth2)
    monkeypatch.setitem(sys.modules, 'google.oauth2.service_account', service_account)
    return calls


def test_from_googlesheet_mocked(monkeypatch):
    calls = _install_fake_google(monkeypatch, values=[['1', '2', '3'], ['4', '5', '6']])
    daf = Daf.from_googlesheet('SHEET_ID', sheetname='Data')
    assert calls['get'] == {'spreadsheetId': 'SHEET_ID', 'range': 'Data'}
    assert calls['build'] == ('sheets', 'v4', 'CREDS')
    assert daf.columns() == ['A', 'B', 'C']
    assert daf.lol == [['1', '2', '3'], ['4', '5', '6']]


def test_from_googlesheet_mocked_empty(monkeypatch):
    _install_fake_google(monkeypatch, values=[])
    daf = Daf.from_googlesheet('SHEET_ID')
    assert daf.lol == []
    assert daf.columns() == []


def test_to_googlesheet_mocked(monkeypatch, capsys):
    calls = _install_fake_google(monkeypatch)
    daf = _daf3()
    result = daf.to_googlesheet('SHEET_ID', sheetname='Out')
    assert result is daf
    upd = calls['update']
    assert upd['spreadsheetId'] == 'SHEET_ID'
    assert upd['range'] == 'Out!A1:C3'
    assert upd['valueInputOption'] == 'RAW'
    assert upd['body'] == {'values': daf.lol}
    assert 'successfully' in capsys.readouterr().out


# --- record_append

def test_record_append_empty_record_is_noop():
    daf = _daf3()
    assert daf.record_append({}) is daf
    assert len(daf) == 3


def test_record_append_keyedlist_into_empty_daf():
    daf = Daf()
    daf.record_append(KeyedList({'a': 1, 'b': 2}))
    assert daf.hd == {'a': 0, 'b': 1}
    assert type(daf.hd) is dict
    assert daf.lol == [[1, 2]]


def test_record_append_mapping_reordered():
    # non-dict Mapping with keys in a different order works via the reorder path.
    daf = Daf(lol=[[1, 2]], cols=['a', 'b'])
    daf.record_append(MappingProxyType({'b': 20, 'a': 10}))
    assert daf.lol == [[1, 2], [10, 20]]


@pytest.mark.xfail(strict=True, reason="BUG: record_append() with a non-dict Mapping whose keys match "
                                       "hd order hits breakpoint() at daf.py:3231 and leaves rec_la "
                                       "unbound (UnboundLocalError)")
def test_record_append_mapping_same_order(no_breakpoint):
    daf = Daf(lol=[[1, 2]], cols=['a', 'b'])
    daf.record_append(MappingProxyType({'a': 10, 'b': 20}))
    assert daf.lol == [[1, 2], [10, 20]]


# --- _basic_append

def test_basic_append_list_checks_length():
    daf = Daf(lol=[[1, 2]], cols=['a', 'b'])
    daf._basic_append([3, 4])
    assert daf.lol == [[1, 2], [3, 4]]
    with pytest.raises(AssertionError):
        daf._basic_append([5])


# --- remove_keylist

def test_remove_keylist_without_keyfield_raises():
    with pytest.raises(KeysDisabledError):
        _daf3().remove_keylist(['x'])


# --- _adjust_return_val

def test_adjust_return_val_uses_instance_retmode():
    daf = _daf3()   # default retmode is 'obj'
    assert daf._adjust_return_val() is daf
    row = Daf(lol=[[1, 2, 3]], cols=['a', 'b', 'c'])
    row.retmode = row.RETMODE_VAL
    assert row._adjust_return_val() == [1, 2, 3]


# --- set_irows_icols

def test_set_irows_icols_none_irows_is_noop():
    daf = _daf3()
    daf.set_irows_icols(None, [1], 0)
    assert daf.lol == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]


def test_set_irows_icols_short_column_list_partial(no_breakpoint):
    # value list shorter than the number of rows: extra rows are left unchanged.
    daf = _daf3()
    daf.set_irows_icols([0, 1, 2], 1, [100])
    assert daf.lol == [[1, 100, 3], [4, 5, 6], [7, 8, 9]]


def test_set_irows_icols_short_row_list_partial(no_breakpoint):
    daf = _daf3()
    daf.set_irows_icols([0, 1], [0, 1], [100])
    assert daf.lol == [[100, 2, 3], [100, 5, 6], [7, 8, 9]]


@pytest.mark.xfail(strict=True, reason="BUG: set_irows_icols() single row, no cols, Daf value stores the "
                                       "Daf object itself as the row (daf.py:3627)")
def test_set_irows_icols_single_row_from_daf():
    daf = _daf3()
    daf.set_irows_icols(0, None, Daf(lol=[[10, 20, 30]], cols=['a', 'b', 'c']))
    assert daf.lol[0] == [10, 20, 30]
    assert isinstance(daf.lol[0], list)


@pytest.mark.xfail(strict=True, reason="BUG: set_irows_icols() multi rows, no cols, Daf value stores "
                                       "one-row Daf objects as rows (daf.py:3662)")
def test_set_irows_icols_multi_row_from_daf():
    daf = _daf3()
    daf.set_irows_icols([0, 1], None, Daf(lol=[[10, 20, 30], [40, 50, 60]], cols=['a', 'b', 'c']))
    assert daf.lol == [[10, 20, 30], [40, 50, 60], [7, 8, 9]]


@pytest.mark.xfail(strict=True, reason="BUG: set_irows_icols() single col, Daf value: value[i][0] is "
                                       "still a Daf, so Daf objects are stored in cells (daf.py:3691)")
def test_set_irows_icols_single_col_from_daf():
    daf = _daf3()
    daf.set_irows_icols([0, 1], 1, Daf(lol=[[100], [200]], cols=['x']))
    assert daf.lol == [[1, 100, 3], [4, 200, 6], [7, 8, 9]]


@pytest.mark.xfail(strict=True, reason="BUG: daf[rows, cols] = other_daf indexes value by column only "
                                       "(value[source_col]) and stores row Daf objects in cells (daf.py:3717)")
def test_setitem_block_from_daf():
    daf = _daf3()
    daf[0:2, 0:2] = Daf(lol=[[100, 200], [300, 400]], cols=['a', 'b'])
    assert daf.lol == [[100, 200, 3], [300, 400, 6], [7, 8, 9]]


# --- select_krows / select_kcols without keys

def test_select_krows_without_keyfield_raises():
    with pytest.raises(KeysDisabledError):
        _daf3().select_krows(['x'])


def test_select_kcols_without_hd_raises():
    with pytest.raises(KeysDisabledError):
        Daf(lol=[[1, 2]]).select_kcols(['x'])


# --- select_irows inverse paths

def test_select_irows_invert_long_list():
    daf = Daf(lol=[[i] for i in range(20)], cols=['a'])
    result = daf.select_irows(list(range(15)), invert=True)
    assert result.lol == [[15], [16], [17], [18], [19]]
    assert result.columns() == ['a']


def test_select_irows_invert_list_of_ranges():
    daf = Daf(lol=[[i] for i in range(20)], cols=['a'])
    result = daf.select_irows([range(0, 3), range(10, 12)], invert=True)
    expected = [[i] for i in range(20) if not (0 <= i < 3 or 10 <= i < 12)]
    assert result.lol == expected


# --- select_icols

def test_select_icols_list_of_ranges_flip():
    daf = _daf3()
    result = daf.select_icols([range(0, 1), range(2, 3)], flip=True)
    assert result.lol == [[1, 4, 7], [3, 6, 9]]


def test_select_icols_uneven_rows_raises_indexerror():
    daf = Daf(lol=[[1, 2, 3], [4]], cols=['a', 'b', 'c'])
    with pytest.raises(IndexError):
        daf.select_icols([0, 2])


@pytest.mark.xfail(strict=True, reason="BUG: select_icols(slice) on uneven rows swallows IndexError after "
                                       "breakpoint() (daf.py:4185-4187) and then fails with UnboundLocalError")
def test_select_icols_slice_uneven_rows_raises_indexerror(no_breakpoint):
    daf = Daf(lol=[[1, 2, 3], [4]], cols=['a', 'b', 'c'])
    with pytest.raises(IndexError):
        daf.select_icols(slice(0, 3))


# --- select_record

def test_select_record_missing_key_not_silent_raises():
    daf = Daf(lol=[[1, 2]], cols=['a', 'b'], keyfield='a')
    with pytest.raises(KeyError):
        daf.select_record(99, silent_error=False)
    assert daf.select_record(99) == {}


# --- select_records_daf

def test_select_records_daf_empty_keys():
    daf = Daf(lol=[[1, 2], [3, 4]], cols=['a', 'b'], keyfield='a')
    empty = daf.select_records_daf([])
    assert empty.lol == []
    assert empty.columns() == ['a', 'b']
    assert daf.select_records_daf([], inverse=True).lol == [[1, 2], [3, 4]]


# --- irow_la

def test_irow_la_returns_reference():
    daf = _daf3()
    row = daf.irow_la(1)
    assert row == [4, 5, 6]
    assert row is daf.lol[1]
