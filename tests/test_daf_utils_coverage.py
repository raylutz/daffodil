# test_daf_utils_coverage.py
#
# Additional coverage tests for daffodil/lib/daf_utils.py (plus a few for daf_pandas.py),
# targeting branches not exercised by test_daf_utils.py: numpy JSON encoding, error paths,
# regex helpers, s3 helpers (boto3 mocked -- no network), file-like CSV parsing,
# beep/sts/loc helpers, and astype_value string specs.

import io
import os
import sys
import types
from unittest import mock

import numpy as np
import pytest

from daffodil.lib import daf_utils as utils


# =====================================================================
# daf_utils.py
# =====================================================================

# --- NpEncoder / json_encode ---

def test_json_encode_numpy_float32():
    # np.float32 is not a float subclass, so it is routed through NpEncoder.default()
    assert utils.json_encode(np.float32(1.5)) == '1.5'
    assert utils.json_encode({'a': np.float32(0.25)}) == '{"a": 0.25}'


def test_json_encode_numpy_int_and_array():
    assert utils.json_encode([np.int64(3), np.array([1, 2])]) == '[3, [1, 2]]'


def test_json_encode_unsupported_type_raises_typeerror():
    with pytest.raises(TypeError):
        utils.json_encode({'a': object()})


def test_json_encode_nan_raises_valueerror():
    with pytest.raises(ValueError):
        utils.json_encode([float('nan'), 1])


# --- test_strbool ---

def test_test_strbool_nan_is_false():
    assert utils.test_strbool(float('nan')) is False


def test_test_strbool_unsupported_type_raises():
    with pytest.raises(TypeError, match='object'):
        utils.test_strbool(object())


# --- sort_lol_by_cols ---

def test_sort_lol_by_cols_length_priority():
    lol = [['bb', 'x'], ['a', 'z'], ['a', 'y'], ['c', 'w']]
    result = utils.sort_lol_by_cols(lol, [0, 1])
    # shorter strings first in col 0, then col 1 breaks ties.
    assert result == [['a', 'y'], ['a', 'z'], ['c', 'w'], ['bb', 'x']]


def test_sort_lol_by_cols_length_priority_reverse():
    lol = [['a', '1'], ['bb', '2'], ['c', '3']]
    result = utils.sort_lol_by_cols(lol, [0], reverse=True)
    assert result == [['bb', '2'], ['c', '3'], ['a', '1']]


# --- safe_regex_select ---

def test_safe_regex_select_bytes_and_default():
    assert utils.safe_regex_select(b'"id=(\\d+)"', 'x id=42 y') == '42'
    assert utils.safe_regex_select(r'id=(\d+)', 'nothing here', default='none') == 'none'


def test_safe_regex_select_no_capture_group_raises():
    with pytest.raises(IndexError, match='no such group'):
        utils.safe_regex_select(r'abc', 'xxabcxx', default='dflt')


# --- safe_regex_replace ---

def test_safe_regex_replace_malformed_pattern_is_ignored():
    # first and last chars differ -> malformed, skipped; second pattern still applied.
    result = utils.safe_regex_replace(['/foo/bar|', '/baz/qux/'], 'foo baz')
    assert result == 'foo qux'


def test_safe_regex_replace_bytes_items_in_list():
    assert utils.safe_regex_replace([b'/a/b/', b'#c##'], 'aacc') == 'bb'


def test_safe_regex_replace_too_many_separators_raises():
    with pytest.raises(ValueError, match='/a/b/c/'):
        utils.safe_regex_replace('/a/b/c/', 'abc')


# --- convert_type_value / unflatten_val / safe_convert_json_to_obj ---

def test_convert_type_value_unsupported_type_raises():
    with pytest.raises(TypeError, match='cannot convert'):
        utils.convert_type_value('x', set)


def test_unflatten_val_json_only_literal():
    # 'true'/'null' are not Python literals, so safe_eval fails and JSON decoding is used.
    assert utils.unflatten_val('[true, null, 1]') == [True, None, 1]
    assert utils.unflatten_val('{"a": false}') == {'a': False}


def test_unflatten_val_undecodable_returns_stripped_str():
    assert utils.unflatten_val(" '[not valid' ") == '[not valid'
    assert utils.unflatten_val('[oops]') == '[oops]'


def test_safe_convert_json_to_obj_human_json_and_unrecoverable():
    assert utils.safe_convert_json_to_obj("{'a': None}") == {'a': None}
    # unrecoverable: returns the partially-fixed string.
    assert utils.safe_convert_json_to_obj("{'a': None") == '{"a": null'


# --- list_stats attrib with ints ---

def test_list_stats_attrib_all_ints_sets_all_numeric():
    info = utils.list_stats(['1', '2', '2', '', None], 'attrib')
    assert info['profile'] == 'attrib'
    assert info['all_ints'] is True
    assert info['all_numeric'] is True
    assert info['num_missing'] == 2
    assert info['val_counts'] == {'1': 1, '2': 2}


# --- parse_s3path ---

def test_parse_s3path_valid_and_invalid():
    d = utils.parse_s3path('s3://bucket/US/WI/file.csv')
    assert d['bucket'] == 'bucket'
    assert d['key'] == 'US/WI/file.csv'
    assert d['dirname'] == 'WI'
    with pytest.raises(RuntimeError):
        utils.parse_s3path('http://bucket/US/file.csv')


# --- transpose_lol / safe_get_idx ---

def test_transpose_lol_empty():
    assert utils.transpose_lol([]) == []


def test_safe_get_idx_none_entry_uses_default():
    assert utils.safe_get_idx([None, 2], 0, 'd') == 'd'
    assert utils.safe_get_idx([None, 2], 1, 'd') == 2


# --- smart_fmt ---

def test_smart_fmt_digit_like_but_not_int_returned_as_is():
    assert utils.smart_fmt('1-2') == '1-2'
    assert utils.smart_fmt('+-') == '+-'
    assert utils.smart_fmt('-1234') == '-1,234'


# --- beep ---

def test_beep_non_linux_without_winsound_uses_beep_command(monkeypatch):
    monkeypatch.setattr(utils, 'is_linux', lambda: False)
    monkeypatch.setitem(sys.modules, 'winsound', None)   # forces ImportError
    with mock.patch('os.system') as m_system:
        utils.beep(440, 100)
    m_system.assert_called_once_with('beep -f 440 -l 100')


def test_beep_non_linux_with_winsound(monkeypatch):
    fake_winsound = types.SimpleNamespace(Beep=mock.Mock())
    monkeypatch.setattr(utils, 'is_linux', lambda: False)
    monkeypatch.setitem(sys.modules, 'winsound', fake_winsound)
    utils.beep()
    fake_winsound.Beep.assert_called_once_with(1080, 500)


def test_beep_linux_prints_bell(monkeypatch, capsys):
    monkeypatch.setattr(utils, 'is_linux', lambda: True)
    utils.beep()
    assert capsys.readouterr().out == '\a'


# --- sts / caller_loc / prog_loc ---

def test_sts_no_color(capsys):
    result = utils.sts('hello', 3, color='')
    out = capsys.readouterr().out
    assert result == 'hello\n'
    assert out.endswith(': hello\n')
    assert '\033[' not in out


def test_sts_disabled_and_low_verbosity(capsys):
    assert utils.sts('x', 3, enable=False) == ''
    assert utils.sts('y', 0) == 'y\n'
    assert capsys.readouterr().out == ''


def test_caller_loc_and_prog_loc_without_frame_support(monkeypatch):
    import inspect
    monkeypatch.setattr(inspect, 'currentframe', lambda: None)
    assert utils.caller_loc() == ''
    assert utils.prog_loc() == ''


def test_caller_loc_with_no_grandparent_frame(monkeypatch):
    import inspect
    fake = types.SimpleNamespace(f_back=types.SimpleNamespace(f_back=None))
    monkeypatch.setattr(inspect, 'currentframe', lambda: fake)
    assert utils.caller_loc() == ''


def test_prog_loc_reports_this_file():
    assert utils.prog_loc().startswith('[test_daf_utils_coverage.py:')


# --- split_dups_list ---

def test_split_dups_list_dict_input_and_prior_list():
    result = utils.split_dups_list({'a': 1, 'b': 2, 'c': 3}, prior_unique_d=['b'], list_idx=2)
    assert list(result['uniques_d'].keys()) == ['a', 'c']
    assert result['prior_reps_loti'] == [(2, 1)]
    assert result['within_reps_loti'] == []


# --- buff_csv_to_lol ---

def test_buff_csv_to_lol_sep_none_defaults_to_comma():
    assert utils.buff_csv_to_lol('a,b\n1,2\n', sep=None) == [['a', 'b'], ['1', '2']]


def test_buff_csv_to_lol_binary_file_with_peek():
    buff = io.BufferedReader(io.BytesIO(b'a,b\n1,2\n'))
    assert utils.buff_csv_to_lol(buff) == [['a', 'b'], ['1', '2']]


def test_buff_csv_to_lol_text_file_without_peek():
    assert utils.buff_csv_to_lol(io.StringIO('x,y\n3,4\n')) == [['x', 'y'], ['3', '4']]


def test_buff_csv_to_lol_peek_failure_raises_runtimeerror():
    class BadFile:
        def seek(self, pos):
            pass

        def read(self, n):
            raise OSError('disk gone')

    with pytest.raises(RuntimeError, match='disk gone'):
        utils.buff_csv_to_lol(BadFile())


# --- does_s3path_exist / write_buff_to_s3path (boto3 mocked) ---

class _FakeClientError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {'Error': {'Code': code}}


class _FakeEndpointConnectionError(Exception):
    pass


@pytest.fixture
def fake_boto(monkeypatch):
    boto3 = types.ModuleType('boto3')
    boto3.client = mock.Mock()
    boto3.resource = mock.Mock()
    botocore = types.ModuleType('botocore')
    exceptions = types.ModuleType('botocore.exceptions')
    exceptions.ClientError = _FakeClientError
    exceptions.EndpointConnectionError = _FakeEndpointConnectionError
    botocore.exceptions = exceptions
    monkeypatch.setitem(sys.modules, 'boto3', boto3)
    monkeypatch.setitem(sys.modules, 'botocore', botocore)
    monkeypatch.setitem(sys.modules, 'botocore.exceptions', exceptions)
    sleep = mock.Mock()
    monkeypatch.setattr(utils.time, 'sleep', sleep)
    boto3.sleep = sleep
    return boto3


S3PATH = 's3://my-bucket/dir/file.csv'


def test_does_s3path_exist_true(fake_boto):
    client = fake_boto.client.return_value
    assert utils.does_s3path_exist(S3PATH) is True
    client.head_object.assert_called_once_with(Bucket='my-bucket', Key='dir/file.csv')
    client.close.assert_called_once()


@pytest.mark.parametrize('code', ['404', 'NoSuchKey', '403'])
def test_does_s3path_exist_missing(fake_boto, code):
    fake_boto.client.return_value.head_object.side_effect = _FakeClientError(code)
    assert utils.does_s3path_exist(S3PATH) is False


def test_does_s3path_exist_retries_endpoint_error_then_succeeds(fake_boto):
    client = fake_boto.client.return_value
    client.head_object.side_effect = [_FakeEndpointConnectionError(), True]
    assert utils.does_s3path_exist(S3PATH) is True
    assert client.head_object.call_count == 2
    fake_boto.sleep.assert_called_once_with(0.1)


def test_does_s3path_exist_transient_errors_exhaust_retries(fake_boto):
    client = fake_boto.client.return_value
    client.head_object.side_effect = _FakeClientError('500')
    client.close.side_effect = AttributeError   # close failure is tolerated
    assert utils.does_s3path_exist(S3PATH) is False
    assert client.head_object.call_count == 10


def test_write_buff_to_s3path_success(fake_boto, monkeypatch):
    monkeypatch.setattr(utils, 'does_s3path_exist', lambda p: True)
    obj = fake_boto.resource.return_value.Object.return_value
    assert utils.write_buff_to_s3path(S3PATH, b'data', content_type='text/csv') == 1
    fake_boto.resource.return_value.Object.assert_called_once_with('my-bucket', 'dir/file.csv')
    obj.put.assert_called_once_with(Body=b'data', ContentType='text/csv')


def test_write_buff_to_s3path_no_such_bucket_reraises(fake_boto):
    obj = fake_boto.resource.return_value.Object.return_value
    obj.put.side_effect = _FakeClientError('NoSuchBucket')
    with pytest.raises(_FakeClientError):
        utils.write_buff_to_s3path(S3PATH, b'data')


def test_write_buff_to_s3path_retry_then_success(fake_boto, monkeypatch):
    monkeypatch.setattr(utils, 'does_s3path_exist', lambda p: True)
    obj = fake_boto.resource.return_value.Object.return_value
    obj.put.side_effect = [_FakeClientError('SlowDown'), None]
    assert utils.write_buff_to_s3path(S3PATH, b'data') == 1
    assert obj.put.call_count == 2
    fake_boto.sleep.assert_called_once_with(0.5)


def test_write_buff_to_s3path_retries_exhausted_timeout(fake_boto):
    obj = fake_boto.resource.return_value.Object.return_value
    obj.put.side_effect = _FakeClientError('SlowDown')
    with pytest.raises(TimeoutError, match='my-bucket/dir/file.csv'):
        utils.write_buff_to_s3path(S3PATH, b'data')
    # delay goes 0.5, 1.0, ... 4.5 -> 9 attempts before delay reaches max_delay=5
    assert obj.put.call_count == 9


def test_write_buff_to_s3path_waits_for_visibility(fake_boto, monkeypatch):
    exists = mock.Mock(side_effect=[False, False, True])
    monkeypatch.setattr(utils, 'does_s3path_exist', exists)
    assert utils.write_buff_to_s3path(S3PATH, b'data') == 1
    assert exists.call_count == 3


def test_write_buff_to_s3path_never_visible_raises(fake_boto, monkeypatch):
    monkeypatch.setattr(utils, 'does_s3path_exist', lambda p: False)
    with pytest.raises(RuntimeError, match='not found after write_buff_to_s3path'):
        utils.write_buff_to_s3path(S3PATH, b'data')


# --- write_buff_to_fp ---

def test_write_buff_to_fp_s3_binary(monkeypatch):
    m_write = mock.Mock(return_value=1)
    monkeypatch.setattr(utils, 'write_buff_to_s3path', m_write)
    assert utils.write_buff_to_fp(b'\x00\x01', S3PATH, rtype='binary') == S3PATH
    m_write.assert_called_once_with(S3PATH, b'\x00\x01')


def test_write_buff_to_fp_s3_text_is_utf8_encoded(monkeypatch):
    m_write = mock.Mock(return_value=1)
    monkeypatch.setattr(utils, 'write_buff_to_s3path', m_write)
    assert utils.write_buff_to_fp('héllo', S3PATH) == S3PATH
    m_write.assert_called_once_with(S3PATH, 'héllo'.encode('utf-8'))


def test_write_buff_to_fp_local_binary(tmp_path):
    fp = str(tmp_path / 'out.bin')
    assert utils.write_buff_to_fp(b'\x01\x02', fp, rtype='binary') == fp
    assert (tmp_path / 'out.bin').read_bytes() == b'\x01\x02'


def test_write_buff_to_fp_local_binary_open_failure_raises(tmp_path):
    fp = str(tmp_path / 'no_such_dir' / 'out.bin')
    with pytest.raises(FileNotFoundError):
        utils.write_buff_to_fp(b'\x01', fp, rtype='binary')


def test_write_buff_to_fp_empty_buff_not_written(tmp_path):
    fp = str(tmp_path / 'x.csv')
    assert utils.write_buff_to_fp('', fp) == fp
    assert not os.path.exists(fp)


# --- len_slice / slice_to_range ---

def test_len_slice_bad_bounds_raises():
    with pytest.raises(TypeError):
        utils.len_slice(slice('a', 'b'), 5)


def test_slice_to_range_non_int_start_raises():
    with pytest.raises(TypeError):
        utils.slice_to_range(slice('a', 5), 10)


def test_slice_to_range_normal():
    assert utils.slice_to_range(slice(2, 20, 3), 10) == range(2, 10, 3)
    assert utils.slice_to_range(slice(None, None, None), 3) == range(3)


# --- _sanitize_cols ---

def test_sanitize_cols_empty_and_dups():
    assert utils._sanitize_cols([]) == []
    assert utils._sanitize_cols(['a', '', 'a']) == ['a', 'Unnamed1', 'a_2']


# --- extract_docstring_parts ---

def test_extract_docstring_parts():
    def documented():
        """ Do a thing.
            More detail here.
            And more.
        """

    def undocumented():
        pass

    assert utils.extract_docstring_parts(documented) == ('Do a thing', 'More detail here.\nAnd more.')
    assert utils.extract_docstring_parts(undocumented) == ('', '')


# --- astype_value string specs ---

def test_astype_value_string_specs():
    assert utils.astype_value('7', 'int') == 7
    assert utils.astype_value(7, 'str') == '7'
    assert utils.astype_value('2.5', 'float') == 2.5
    assert utils.astype_value('x', 'bool') is True
    assert utils.astype_value(utils.NULL, 'int') == utils.NULL


def test_astype_value_unsupported_string_raises():
    with pytest.raises(ValueError, match='astype not supported'):
        utils.astype_value('1', 'complex')


def test_astype_la_with_string_spec():
    assert utils.astype_la(['1', '2'], 'int') == [1, 2]


# =====================================================================
# daf_pandas.py
# =====================================================================

from daffodil.lib import daf_pandas  # noqa: E402


def test_pandas_dtype_dict_to_python_timedelta():
    result = daf_pandas.pandas_dtype_dict_to_python({'td': np.dtype('m8[ns]'), 'i': np.dtype('int64')})
    import pandas as pd
    assert result == {'td': pd.Timedelta, 'i': int}


def test_pandas_dtype_dict_to_python_category_is_str():
    # same mapping as pandas_dtype_to_python_type(), which Daf.from_pandas_df() uses.
    import pandas as pd
    result = daf_pandas.pandas_dtype_dict_to_python({'cat': pd.CategoricalDtype(['x']), 'f': np.dtype('float64')})
    assert result == {'cat': str, 'f': float}


def test_pandas_dtype_to_python_type_tz_aware_datetime():
    import pandas as pd
    # DatetimeTZDtype is not understood by np.issubdtype (TypeError) -> pandas extension check.
    assert daf_pandas.pandas_dtype_to_python_type(pd.DatetimeTZDtype(tz='UTC')) is pd.Timestamp


def test_pandas_dtype_to_python_type_timedelta_index():
    import pandas as pd
    assert daf_pandas.pandas_dtype_to_python_type(pd.Index([pd.Timedelta(1)])) is pd.Timedelta


# =====================================================================
# daf_md.py
# =====================================================================
# Remaining uncovered lines in daf_md.py (453, 542) are unreachable -- see report; no tests.
