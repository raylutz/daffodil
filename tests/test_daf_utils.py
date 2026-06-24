# test_daf_utils.py
#
# Tests for daffodil/lib/daf_utils.py functions used for parsing nonconformant CSV/xlsx
# files (header gap/repeat fixup, strbool conversion, Excel stringification cleanup).

import pytest

from daffodil.lib import daf_utils as utils


# --- profile_ls_to_lr ---

def test_profile_ls_to_lr_with_repeats():
    input_ls = ['Alice', 'Bob', 'Unnamed2', 'Charlie', 'David', 'Unnamed5', 'Unnamed6']
    result = utils.profile_ls_to_lr(input_ls)
    assert result == [range(0, 1), range(1, 3), range(3, 4), range(4, 7)]


def test_profile_ls_to_lr_no_repeats():
    input_ls = ['Alice', 'Bob', 'Charlie']
    result = utils.profile_ls_to_lr(input_ls)
    assert result == [range(0, 1), range(1, 2), range(2, 3)]


def test_profile_ls_to_lr_empty_input():
    assert utils.profile_ls_to_lr([]) == []


def test_profile_ls_to_lr_custom_repeat_marker():
    input_ls = ['A', 'Skip1', 'Skip2', 'B']
    result = utils.profile_ls_to_lr(input_ls, repeat_startswith='Skip')
    assert result == [range(0, 3), range(3, 4)]


def test_profile_ls_to_lr_ignore_cols():
    input_ls = ['A', 'Ignored', 'B']
    result = utils.profile_ls_to_lr(input_ls, ignore_cols=['Ignored'])
    assert result == [range(0, 1), range(2, 3)]


# --- is_comment_line ---

def test_is_comment_line_blank():
    assert utils.is_comment_line('') is True


def test_is_comment_line_hash():
    assert utils.is_comment_line('# a comment') is True


def test_is_comment_line_quoted_hash():
    assert utils.is_comment_line('"#a quoted comment') is True


def test_is_comment_line_commas_only():
    assert utils.is_comment_line(',,,') is True


def test_is_comment_line_normal_line():
    assert utils.is_comment_line('a,b,c') is False


# --- make_strbool / test_strbool ---

def test_test_strbool_bool_values():
    assert utils.test_strbool(True) is True
    assert utils.test_strbool(False) is False


def test_test_strbool_string_values():
    assert utils.test_strbool('1') is True
    assert utils.test_strbool('true') is True
    assert utils.test_strbool('TRUE') is True
    assert utils.test_strbool('yes') is True
    assert utils.test_strbool('0') is False
    assert utils.test_strbool('no') is False


def test_test_strbool_int_values():
    assert utils.test_strbool(1) is True
    assert utils.test_strbool(0) is False
    assert utils.test_strbool(2) is False


def test_test_strbool_none():
    assert utils.test_strbool(None) is False


def test_test_strbool_nan():
    assert utils.test_strbool(float('nan')) is False


def test_make_strbool_true_values():
    assert utils.make_strbool(True) == '1'
    assert utils.make_strbool('yes') == '1'
    assert utils.make_strbool(1) == '1'


def test_make_strbool_false_values():
    assert utils.make_strbool(False) == '0'
    assert utils.make_strbool('no') == '0'
    assert utils.make_strbool(0) == '0'
    assert utils.make_strbool(None) == '0'


# --- unexcelstringify ---

def test_unexcelstringify_stringified_value():
    assert utils.unexcelstringify('="12345"') == '12345'


def test_unexcelstringify_plain_string_unchanged():
    assert utils.unexcelstringify('plain') == 'plain'


def test_unexcelstringify_empty_string():
    assert utils.unexcelstringify('') == ''


def test_unexcelstringify_none_passthrough():
    assert utils.unexcelstringify(None) is None


# --- get_csv_column_names ---

def test_get_csv_column_names_basic():
    csv_buff = "a,b,c\n1,2,3\n"
    assert utils.get_csv_column_names(csv_buff) == ['a', 'b', 'c']


def test_get_csv_column_names_skips_leading_comment():
    csv_buff = "# a comment\na,b,c\n1,2,3\n"
    assert utils.get_csv_column_names(csv_buff) == ['a', 'b', 'c']


def test_get_csv_column_names_skips_blank_lines():
    csv_buff = "\n\na,b,c\n"
    assert utils.get_csv_column_names(csv_buff) == ['a', 'b', 'c']


def test_get_csv_column_names_no_valid_line_returns_empty():
    csv_buff = "# only comments\n# more comments\n"
    assert utils.get_csv_column_names(csv_buff) == []


# --- precheck_csv_cols ---

def test_precheck_csv_cols_missing_and_extra():
    csv_buff = "a,b,c\n1,2,3\n"
    missing, extra = utils.precheck_csv_cols(csv_buff, expected_cols=['a', 'b', 'd'])
    assert missing == ['d']
    assert extra == ['c']


def test_precheck_csv_cols_exact_match():
    csv_buff = "a,b,c\n1,2,3\n"
    missing, extra = utils.precheck_csv_cols(csv_buff, expected_cols=['a', 'b', 'c'])
    assert missing == []
    assert extra == []


# =====================================================================
# dtype-coercion cluster (apply_dtypes machinery)
# =====================================================================

# --- convert_type_value ---

def test_convert_type_value_null_string():
    assert utils.convert_type_value('', int) == ''


def test_convert_type_value_none():
    assert utils.convert_type_value(None, int) == ''


def test_convert_type_value_to_int():
    assert utils.convert_type_value('5', int) == 5


def test_convert_type_value_to_int_zero_shortcut():
    assert utils.convert_type_value('0', int) == 0


def test_convert_type_value_to_int_from_float_string():
    assert utils.convert_type_value('1.0', int) == 1


def test_convert_type_value_to_float():
    assert utils.convert_type_value('3.7', float) == 3.7


def test_convert_type_value_to_bool_true():
    assert utils.convert_type_value('1', bool) == 1


def test_convert_type_value_to_bool_false():
    assert utils.convert_type_value('0', bool) == 0


def test_convert_type_value_to_list_unflatten():
    assert utils.convert_type_value('[1,2,3]', list) == [1, 2, 3]


def test_convert_type_value_to_str_from_int():
    assert utils.convert_type_value(123, str) == '123'


def test_convert_type_value_to_str_from_bool():
    assert utils.convert_type_value(True, str) == 1


def test_convert_type_value_list_passthrough_no_conversion():
    val = [1, 2]
    assert utils.convert_type_value(val, list) is val


def test_convert_type_value_dict_passthrough_no_conversion():
    val = {'a': 1}
    assert utils.convert_type_value(val, dict) is val


# --- astype_value ---

def test_astype_value_none_passthrough():
    assert utils.astype_value('5', None) == '5'


def test_astype_value_null_sentinel_preserved():
    assert utils.astype_value(utils.NULL, int) is utils.NULL


def test_astype_value_type_object():
    assert utils.astype_value('5', int) == 5


def test_astype_value_callable():
    assert utils.astype_value(5, str) == '5'


def test_astype_value_str_keyword_float():
    assert utils.astype_value('5', 'float') == 5.0


def test_astype_value_unsupported_string_raises():
    with pytest.raises(ValueError):
        utils.astype_value('x', 'not_a_type')


# --- unflatten_val ---

def test_unflatten_val_list():
    assert utils.unflatten_val('[1, 2, 3]') == [1, 2, 3]


def test_unflatten_val_dict():
    assert utils.unflatten_val('{"a": 1}') == {'a': 1}


def test_unflatten_val_plain_string_unchanged():
    assert utils.unflatten_val('hello') == 'hello'


def test_unflatten_val_empty_list_preserved():
    assert utils.unflatten_val('[]') == []


# --- str2bool ---

def test_str2bool_true_values():
    assert utils.str2bool('yes') is True
    assert utils.str2bool('true') is True
    assert utils.str2bool('1') is True
    assert utils.str2bool(True) is True
    assert utils.str2bool(1) is True


def test_str2bool_false_values():
    assert utils.str2bool('no') is False
    assert utils.str2bool('false') is False
    assert utils.str2bool(None) is False
    assert utils.str2bool(False) is False


def test_str2bool_invalid_string_raises():
    with pytest.raises(ValueError):
        utils.str2bool('maybe')


# --- is_numeric ---

def test_is_numeric_int():
    assert utils.is_numeric(5) is True


def test_is_numeric_float():
    assert utils.is_numeric(3.14) is True


def test_is_numeric_numeric_string():
    assert utils.is_numeric('5') is True


def test_is_numeric_currency_string():
    assert utils.is_numeric('$5,000') is True


def test_is_numeric_non_numeric_string():
    assert utils.is_numeric('abc') is False


def test_is_numeric_non_string_non_number():
    assert utils.is_numeric([1, 2]) is False


# --- safe_eval ---

def test_safe_eval_valid_literal():
    assert utils.safe_eval('[1, 2, 3]') == [1, 2, 3]


def test_safe_eval_invalid_returns_none():
    assert utils.safe_eval('not valid python {') is None


# --- json_decode ---

def test_json_decode_empty_string():
    assert utils.json_decode('') == ''


def test_json_decode_valid_json():
    assert utils.json_decode('{"a": 1}') == {'a': 1}


def test_json_decode_invalid_json_returns_empty_string():
    assert utils.json_decode('not json') == ''


# --- safe_convert_json_to_obj ---

def test_safe_convert_json_to_obj_valid_json():
    assert utils.safe_convert_json_to_obj('{"a": 1}') == {'a': 1}


def test_safe_convert_json_to_obj_single_quotes():
    assert utils.safe_convert_json_to_obj("{'a': 1}") == {'a': 1}


def test_safe_convert_json_to_obj_empty_string_returns_empty_dict():
    assert utils.safe_convert_json_to_obj('') == {}


def test_safe_convert_json_to_obj_none_value_converted():
    assert utils.safe_convert_json_to_obj("{'a': None}") == {'a': None}


# --- json_encode / NpEncoder ---

def test_json_encode_none_returns_empty_string():
    assert utils.json_encode(None) == ''


def test_json_encode_basic_dict():
    assert utils.json_encode({'a': 1}) == '{"a": 1}'


def test_json_encode_numpy_int_via_npencoder():
    import numpy as np
    result = utils.json_encode({'a': np.int64(5)})
    assert result == '{"a": 5}'


def test_json_encode_numpy_float_via_npencoder():
    import numpy as np
    result = utils.json_encode({'a': np.float64(5.5)})
    assert result == '{"a": 5.5}'


def test_json_encode_numpy_array_via_npencoder():
    import numpy as np
    result = utils.json_encode({'a': np.array([1, 2, 3])})
    assert result == '{"a": [1, 2, 3]}'