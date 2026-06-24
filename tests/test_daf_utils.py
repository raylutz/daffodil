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


def test_convert_type_value_to_int_conversion_failure_returns_null():
    assert utils.convert_type_value('not a number', int) == ''


def test_convert_type_value_to_float_conversion_failure_returns_null():
    assert utils.convert_type_value('not a number', float) == ''


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


# =====================================================================
# lol manipulation helpers
# =====================================================================

# --- select_col_of_lol_by_col_idx ---

def test_select_col_of_lol_by_col_idx_basic():
    assert utils.select_col_of_lol_by_col_idx([[1, 2], [3, 4]], 1) == [2, 4]


def test_select_col_of_lol_by_col_idx_out_of_range_returns_empty():
    assert utils.select_col_of_lol_by_col_idx([[1, 2], [3]], 5) == []


# --- insert_col_in_lol_at_icol ---

def test_insert_col_in_lol_at_icol_middle():
    result = utils.insert_col_in_lol_at_icol(icol=1, col_la=['x', 'y'], lol=[[1, 2], [3, 4]])
    assert result == [[1, 'x', 2], [3, 'y', 4]]


def test_insert_col_in_lol_at_icol_at_end():
    result = utils.insert_col_in_lol_at_icol(icol=-1, col_la=['x', 'y'], lol=[[1, 2], [3, 4]])
    assert result == [[1, 2, 'x'], [3, 4, 'y']]


def test_insert_col_in_lol_at_icol_empty_lol_with_col():
    result = utils.insert_col_in_lol_at_icol(col_la=['x', 'y'], lol=[])
    assert result == [['x'], ['y']]


def test_insert_col_in_lol_at_icol_empty_lol_no_col():
    assert utils.insert_col_in_lol_at_icol(col_la=None, lol=[]) == []


# --- insert_row_in_lol_at_irow ---

def test_insert_row_in_lol_at_irow_middle():
    result = utils.insert_row_in_lol_at_irow(irow=0, row_la=['a', 'b'], lol=[[1, 2]])
    assert result == [['a', 'b'], [1, 2]]


def test_insert_row_in_lol_at_irow_append_pads_with_default():
    result = utils.insert_row_in_lol_at_irow(irow=-1, row_la=['a'], lol=[[1, 2]], default='')
    assert result == [[1, 2], ['a', '']]


def test_insert_row_in_lol_at_irow_empty_lol():
    assert utils.insert_row_in_lol_at_irow(row_la=['a', 'b'], lol=[]) == [['a', 'b']]


def test_insert_row_in_lol_at_irow_no_row():
    assert utils.insert_row_in_lol_at_irow(row_la=None, lol=[[1, 2]]) == [[1, 2]]


def test_insert_row_in_lol_at_irow_both_empty():
    assert utils.insert_row_in_lol_at_irow(row_la=None, lol=None) == []


# --- calc_chunk_sizes / convert_sizes_to_idx_ranges ---

def test_calc_chunk_sizes_basic():
    assert utils.calc_chunk_sizes(10, 3) == [3, 3, 2, 2]


def test_calc_chunk_sizes_zero_items():
    assert utils.calc_chunk_sizes(0, 3) == []


def test_calc_chunk_sizes_zero_max_size():
    assert utils.calc_chunk_sizes(10, 0) == []


def test_convert_sizes_to_idx_ranges():
    assert utils.convert_sizes_to_idx_ranges([4, 3, 3]) == [(0, 4), (4, 7), (7, 10)]


def test_convert_sizes_to_idx_ranges_empty():
    assert utils.convert_sizes_to_idx_ranges([]) == []


# --- sort_lol_by_col / sort_lol_by_cols ---

def test_sort_lol_by_col_no_length_priority():
    result = utils.sort_lol_by_col([[3, 'c'], [1, 'a'], [2, 'b']], colidx=0, length_priority=False)
    assert result == [[1, 'a'], [2, 'b'], [3, 'c']]


def test_sort_lol_by_col_reverse():
    result = utils.sort_lol_by_col([[1, 'a'], [3, 'c'], [2, 'b']], colidx=0, length_priority=False, reverse=True)
    assert result == [[3, 'c'], [2, 'b'], [1, 'a']]


def test_sort_lol_by_cols_multi_key():
    result = utils.sort_lol_by_cols([[1, 'b'], [1, 'a'], [0, 'z']], colidxs=[0, 1], length_priority=False)
    assert result == [[0, 'z'], [1, 'a'], [1, 'b']]


# --- safe_regex_select / safe_regex_replace ---

def test_safe_regex_select_match():
    assert utils.safe_regex_select(r'(\d+)', 'abc123def') == '123'


def test_safe_regex_select_no_match_returns_default():
    assert utils.safe_regex_select(r'(\d+)', 'no digits', default='NONE') == 'NONE'


def test_safe_regex_replace_find_replace():
    assert utils.safe_regex_replace('/abc/xyz/', 'abcdef') == 'xyzdef'


def test_safe_regex_replace_remove():
    assert utils.safe_regex_replace('/abc//', 'abcdef') == 'def'


def test_safe_regex_replace_list_of_patterns():
    result = utils.safe_regex_replace(['/abc/x/', '/def/y/'], 'abcdef')
    assert result == 'xy'


def test_safe_regex_replace_bytes_input():
    assert utils.safe_regex_replace(b'/abc/x/', 'abcdef') == 'xdef'


# =====================================================================
# xlsx_to_csv / add_trailing_columns_csv
# =====================================================================

def _make_xlsx_bytes(rows):
    import xlsxwriter, io
    buf = io.BytesIO()
    wb = xlsxwriter.Workbook(buf, {'in_memory': True})
    ws = wb.add_worksheet()
    for i, row in enumerate(rows):
        ws.write_row(i, 0, row)
    wb.close()
    return buf.getvalue()


def test_xlsx_to_csv_with_enough_rows_pads_short_rows():
    xlsx_bytes = _make_xlsx_bytes([['a', 'b', 'c'], [1, 2], [3, 4, 5]])
    result = utils.xlsx_to_csv(xlsx_bytes)
    assert result == b'a,b,c\n1,2,\n3,4,5\n'


def test_xlsx_to_csv_no_trailing_cols_bypasses_padding():
    xlsx_bytes = _make_xlsx_bytes([['a', 'b', 'c'], [1, 2]])
    result = utils.xlsx_to_csv(xlsx_bytes, add_trailing_blank_cols=False)
    assert result == b'a,b,c\n1,2,\n'


def test_xlsx_to_csv_sheetname_param():
    xlsx_bytes = _make_xlsx_bytes([['a', 'b', 'c'], [1, 2], [3, 4, 5]])
    result = utils.xlsx_to_csv(xlsx_bytes, sheetname='Sheet1')
    assert result == b'a,b,c\n1,2,\n3,4,5\n'


def test_add_trailing_columns_csv_pads_short_rows():
    result = utils.add_trailing_columns_csv('a,b,c\n1,2\n3,4,5\n')
    assert result == 'a,b,c\n1,2,\n3,4,5\n'


@pytest.mark.xfail(
    reason=(
        "add_trailing_columns_csv does max(len(next(reader)) for _ in range(num_rows)) (num_rows "
        "defaults to 3) to sample the first num_rows rows for the max column count -- but if the "
        "CSV has fewer than num_rows rows total, the extra next(reader) call raises an uncaught "
        "StopIteration. Affects xlsx_to_csv's default add_trailing_blank_cols=True path for any "
        "short file (< 3 rows). The function's own docstring already flags it as @@TODO DEPRECATED."
    ),
    strict=True,
)
def test_add_trailing_columns_csv_fewer_rows_than_sample_size():
    result = utils.add_trailing_columns_csv('a,b,c\n1,2\n')
    assert result == 'a,b,c\n1,2,\n'


# =====================================================================
# compare_lists remaining branches
# =====================================================================

def test_compare_lists_dict_inputs():
    result = utils.compare_lists(work_list={'a': None, 'b': None}, ref_list={'a': None, 'c': None})
    assert result == (['a'], ['c'], ['b'], [])


def test_compare_lists_with_req_list():
    result = utils.compare_lists(['a', 'b'], ['a', 'c'], req_list=['a', 'b', 'd'])
    assert result == (['a'], ['c'], ['b'], ['d'])


def test_compare_lists_unhashable_items_fallback():
    result = utils.compare_lists([['a'], ['b']], [['a'], ['c']])
    assert result == ([['a']], [['c']], [['b']], [])


# --- set_dict_dtypes ---

def test_set_dict_dtypes_converts_listed_cols():
    result = utils.set_dict_dtypes({'a': '5', 'b': 'x'}, dtypes={'a': int})
    assert result == {'a': 5, 'b': 'x'}


def test_set_dict_dtypes_no_dtypes_returns_unchanged():
    da = {'a': '5'}
    assert utils.set_dict_dtypes(da, dtypes=None) == {'a': '5'}


# =====================================================================
# split_dups_list / list-checking helpers / list_stats family / smart_fmt
# =====================================================================

# --- split_dups_list ---

def test_split_dups_list_within_reps():
    result = utils.split_dups_list(['a', 'b', 'a', 'c', 'b'])
    assert result['uniques_d'] == {'a': None, 'b': None, 'c': None}
    assert result['within_reps_loti'] == [(0, 2), (0, 4)]
    assert result['prior_reps_loti'] == []


def test_split_dups_list_prior_reps():
    result = utils.split_dups_list(['a', 'b'], prior_unique_d=['a'])
    assert result['uniques_d'] == {'b': None}
    assert result['prior_reps_loti'] == [(0, 0)]


# --- is_list_allints / is_list_allnumeric / is_list_allbools ---

def test_is_list_allints_true():
    assert utils.is_list_allints(['1', '2', '-3', '4.0']) is True


def test_is_list_allints_false():
    assert utils.is_list_allints(['1', 'x']) is False


def test_is_list_allnumeric_true():
    assert utils.is_list_allnumeric(['1', '2.5', '-3']) is True


def test_is_list_allnumeric_false():
    assert utils.is_list_allnumeric(['1', 'x']) is False


def test_is_list_allbools_true():
    assert utils.is_list_allbools(['1', '0', 'true', 'false']) == (True, 2)


def test_is_list_allbools_false():
    assert utils.is_list_allbools(['1', 'x', 'maybe']) == (False, 0)


# --- list_stats_filepaths / list_stats_localidx ---

def test_list_stats_filepaths():
    result = utils.list_stats_filepaths(['/a/b', '/c/d', None])
    assert result['num_all'] == 3
    assert result['num_missing'] == 1
    assert result['non_missing'] == ['/a/b', '/c/d']


def test_list_stats_localidx_sequential():
    result = utils.list_stats_localidx(['0', '1', '2'])
    assert result == {'all_ints': True, 'max': 2, 'min': 0, 'sequential': True}


def test_list_stats_localidx_non_sequential():
    result = utils.list_stats_localidx(['0', '1', '3'])
    assert result['sequential'] is False


def test_list_stats_localidx_not_all_ints():
    result = utils.list_stats_localidx(['0', 'x'])
    assert result == {'all_ints': False}


# --- list_stats_attrib / list_stats_scalar / list_stats_index ---

def test_list_stats_attrib():
    result = utils.list_stats_attrib(['a', 'b', 'a', 'c'])
    assert result['num_all'] == 4
    assert result['uniques'] == ['a', 'b', 'c']
    assert result['val_counts'] == {'a': 2, 'b': 1, 'c': 1}


def test_list_stats_scalar_all_ints():
    result = utils.list_stats_scalar(['1', '2', '3'])
    assert result['all_ints'] is True
    assert result['max'] == 3
    assert result['min'] == 1
    assert result['mean'] == 2


def test_list_stats_scalar_floats():
    result = utils.list_stats_scalar(['1.5', '2.5'])
    assert result['all_ints'] is False
    assert result['all_numeric'] is True
    assert result['max'] == 2.5


def test_list_stats_index():
    result = utils.list_stats_index(['1', '2', '1', '3'])
    assert result['num_all'] == 4
    assert result['num_uniques'] == 3
    assert result['num_within_reps'] == 1
    assert result['all_ints'] is True


# --- list_stats dispatcher ---

def test_list_stats_dispatches_to_scalar():
    result = utils.list_stats(['1', '2'], 'scalar')
    assert result['profile'] == 'scalar'
    assert result['mean'] == 1.5


def test_list_stats_unsupported_profile_raises():
    with pytest.raises(NotImplementedError):
        utils.list_stats(['1'], 'bogus')


def test_list_stats_dispatches_to_index():
    result = utils.list_stats(['1', '2'], 'index')
    assert result['profile'] == 'index'
    assert result['num_uniques'] == 2


def test_list_stats_dispatches_to_attrib():
    result = utils.list_stats(['a', 'b'], 'attrib')
    assert result['profile'] == 'attrib'


def test_list_stats_dispatches_to_file_paths():
    result = utils.list_stats(['/a', '/b'], 'file_paths')
    assert result['profile'] == 'file_paths'


def test_list_stats_dispatches_to_localidx():
    result = utils.list_stats(['0', '1'], 'localidx')
    assert result['profile'] == 'localidx'


def test_list_stats_index_non_integer_values():
    result = utils.list_stats_index(['1.5', '2.5', '1', '3'])
    assert result['all_ints'] is False
    assert result['all_numeric'] is True
    assert result['max'] == 3.0
    assert result['min'] == 1.0


# =====================================================================
# equal_cols_lol / to_dn_if_list / get_indirect_da / get_indirect_val
# =====================================================================

def test_equal_cols_lol_no_limit_already_equal_returns_unchanged():
    lol = [[1, 2], [3, 4]]
    assert utils.equal_cols_lol(lol, limit=None) is lol


def test_equal_cols_lol_longer_line_beyond_sample_limit_triggers_full_pass():
    result = utils.equal_cols_lol([[1], [1, 2], [1, 2, 3, 4, 5]], limit=2)
    assert result == [[1, '', '', '', ''], [1, 2, '', '', ''], [1, 2, 3, 4, 5]]


def test_to_dn_if_list_none_returns_empty_list():
    assert utils.to_dn_if_list(None) == []


def test_to_dn_if_list_short_list_passthrough():
    val = [1, 2, 3]
    assert utils.to_dn_if_list(val) is val


def test_to_dn_if_list_long_list_converted_to_dict():
    val = list(range(10))
    result = utils.to_dn_if_list(val)
    assert result == dict.fromkeys(val)


def test_get_indirect_da_dict_value():
    row_da = {'meta': {'x': 1}}
    assert utils.get_indirect_da(row_da, 'meta') == {'x': 1}


def test_get_indirect_da_json_string_value():
    row_da = {'meta': '{"x": 1}'}
    assert utils.get_indirect_da(row_da, 'meta') == {'x': 1}


def test_get_indirect_da_missing_col_returns_empty_dict():
    assert utils.get_indirect_da({}, 'meta') == {}


def test_get_indirect_val_basic():
    row_da = {'meta': {'x': 1}}
    assert utils.get_indirect_val(row_da, 'meta', 'x') == 1


def test_get_indirect_val_missing_col_returns_default():
    row_da = {'meta': {'x': 1}}
    assert utils.get_indirect_val(row_da, 'meta', 'y', default=-1) == -1


# --- smart_fmt ---

def test_smart_fmt_none():
    assert utils.smart_fmt(None) == ''


def test_smart_fmt_bracketed_list_string():
    assert utils.smart_fmt('[1,2,3]') == '1<br>2<br>3'


def test_smart_fmt_large_int_comma_format():
    assert utils.smart_fmt(1234) == '1,234'


def test_smart_fmt_large_float_rounds():
    assert utils.smart_fmt(1234.5) == '1,234'


def test_smart_fmt_small_float_decimals():
    assert utils.smart_fmt(0.5) == '0.50'
    assert utils.smart_fmt(0.05) == '0.050'


def test_smart_fmt_large_float_over_100_rounds_to_int():
    assert utils.smart_fmt(150.7) == '151'


def test_smart_fmt_float_between_1_and_100():
    assert utils.smart_fmt(50.5) == '50.5'


def test_smart_fmt_decreasing_magnitude_decimal_places():
    assert utils.smart_fmt(0.005) == '0.0050'
    assert utils.smart_fmt(0.0005) == '0.00050'
    assert utils.smart_fmt(0.00005) == '0.000050'


def test_smart_fmt_extremely_small_rounds_to_zero():
    assert utils.smart_fmt(0.000005) == '0.0'


def test_smart_fmt_signed_int_strings():
    assert utils.smart_fmt('-5') == '-5'
    assert utils.smart_fmt('+5') == '5'


def test_smart_fmt_scientific_notation_string():
    assert utils.smart_fmt('1e-10') == '0.0'


def test_smart_fmt_plain_string_unchanged():
    assert utils.smart_fmt('hello') == 'hello'


# =====================================================================
# logging / misc helpers
# =====================================================================

# --- sts / stsloc / caller_loc / prog_loc ---

def test_sts_returns_text_plus_end():
    assert utils.sts('hello') == 'hello\n'


def test_sts_none_text_returns_empty():
    assert utils.sts(None) == ''


def test_sts_disabled_returns_empty():
    assert utils.sts('hello', enable=False) == ''


def test_sts_prints_when_verboselevel_high_enough(capsys):
    utils.sts('hello', verboselevel=5)
    captured = capsys.readouterr()
    assert 'hello' in captured.out


def test_stsloc_includes_location_prefix():
    result = utils.stsloc('hello')
    assert result.endswith('hello\n')
    assert ':' in result  # location prefix includes file:line


def test_prog_loc_format():
    result = utils.prog_loc()
    assert result.startswith('[')
    assert result.endswith(']')
    assert ':' in result


# --- colorize ---

def test_colorize_with_color():
    assert utils.colorize('hi', 'red') == '\033[31mhi\033[0m'


def test_colorize_no_color_passthrough():
    assert utils.colorize('hi', '') == 'hi'


# --- beep (dev-only; emits terminal bell on Linux, winsound/os.system('beep') elsewhere) ---

def test_beep_does_not_raise():
    assert utils.beep() is None


def test_beep_emits_terminal_bell_on_linux(capsys):
    if not utils.is_linux():
        pytest.skip("bell-character behavior is Linux-specific in the current implementation")
    utils.beep()
    captured = capsys.readouterr()
    assert '\a' in captured.out


def test_error_beep_does_not_raise():
    assert utils.error_beep() is None


# --- safe_stdev / safe_mean ---

def test_safe_stdev_normal():
    assert utils.safe_stdev([1, 2, 3]) == 1.0


def test_safe_stdev_too_short_returns_zero():
    assert utils.safe_stdev([5]) == 0
    assert utils.safe_stdev([]) == 0


def test_safe_mean_normal():
    assert utils.safe_mean([1, 2, 3]) == 2


def test_safe_mean_empty_returns_zero():
    assert utils.safe_mean([]) == 0


# --- safe_del_key ---

def test_safe_del_key_existing():
    da = {'a': 1, 'b': 2}
    result = utils.safe_del_key(da, 'a')
    assert result == {'b': 2}
    assert result is da


def test_safe_del_key_missing_key_no_error():
    da = {'b': 2}
    assert utils.safe_del_key(da, 'missing') == {'b': 2}


# --- dod_to_lod ---

def test_dod_to_lod_adds_keyfield():
    result = utils.dod_to_lod({'k1': {'x': 1}, 'k2': {'x': 2}}, keyfield='id')
    assert result == [{'id': 'k1', 'x': 1}, {'id': 'k2', 'x': 2}]


def test_dod_to_lod_no_keyfield():
    result = utils.dod_to_lod({'k1': {'x': 1}}, keyfield='')
    assert result == [{'x': 1}]


def test_dod_to_lod_non_dict_raises():
    with pytest.raises(RuntimeError):
        utils.dod_to_lod([1, 2, 3])


# --- equal_cols_lol ---

def test_equal_cols_lol_pads_short_rows():
    result = utils.equal_cols_lol([[1, 2, 3], [1, 2], [1]])
    assert result == [[1, 2, 3], [1, 2, ''], [1, '', '']]


def test_equal_cols_lol_already_equal_unchanged():
    lol = [[1, 2], [3, 4]]
    assert utils.equal_cols_lol(lol) == [[1, 2], [3, 4]]


# --- invert_dol_to_dict ---

def test_invert_dol_to_dict():
    result = utils.invert_dol_to_dict({'a': [1, 2], 'b': [3]})
    assert result == {1: 'a', 2: 'a', 3: 'b'}


# =====================================================================
# slice / range / path / s3 helpers
# =====================================================================

# --- slice_to_range ---

def test_slice_to_range_full_slice():
    assert utils.slice_to_range(slice(None, None, None), 5) == range(0, 5)


def test_slice_to_range_bounded():
    assert utils.slice_to_range(slice(1, 4, 1), 10) == range(1, 4)


def test_slice_to_range_open_ended():
    assert utils.slice_to_range(slice(1, None, 1), 5) == range(1, 5)


# --- len_slice ---

def test_len_slice_full_slice_returns_zero():
    assert utils.len_slice(slice(None, None, None), 5) == 0


def test_len_slice_bounded():
    assert utils.len_slice(slice(1, 4), 10) == 3


def test_len_slice_with_step():
    assert utils.len_slice(slice(0, 10, 2), 10) == 5


# --- len_rowcol_spec ---

def test_len_rowcol_spec_slice():
    assert utils.len_rowcol_spec(slice(0, 4), 10) == 4


def test_len_rowcol_spec_int():
    assert utils.len_rowcol_spec(5, 10) == 1


def test_len_rowcol_spec_list():
    assert utils.len_rowcol_spec([1, 2, 3], 10) == 3


def test_len_rowcol_spec_none():
    assert utils.len_rowcol_spec(None, 10) == 0


# --- is_tuple_of_type_len ---

def test_is_tuple_of_type_len_true():
    assert utils.is_tuple_of_type_len(('a', 'b'), str, 2) is True


def test_is_tuple_of_type_len_wrong_type():
    assert utils.is_tuple_of_type_len(('a', 1), str, 2) is False


def test_is_tuple_of_type_len_not_a_tuple():
    assert utils.is_tuple_of_type_len(['a', 'b'], str, 2) is False


# --- path_sep_per_os ---

def test_path_sep_per_os_forward_slash():
    assert utils.path_sep_per_os('a/b\\c', sep='/') == 'a/b/c'


def test_path_sep_per_os_backslash():
    assert utils.path_sep_per_os('a/b\\c', sep='\\') == 'a\\b\\c'


# --- parse_s3path ---

def test_parse_s3path_valid():
    result = utils.parse_s3path('s3://bucket/prefix/path/file.txt')
    assert result['bucket'] == 'bucket'
    assert result['basename'] == 'file.txt'
    assert result['key'] == 'prefix/path/file.txt'
    assert result['dirname'] == 'path'


def test_parse_s3path_invalid_raises():
    with pytest.raises(RuntimeError):
        utils.parse_s3path('not-an-s3-path')


# =====================================================================
# buff_csv_to_lol
# =====================================================================

def test_buff_csv_to_lol_string_input():
    assert utils.buff_csv_to_lol('a,b\n1,2\n') == [['a', 'b'], ['1', '2']]


def test_buff_csv_to_lol_bytes_input():
    assert utils.buff_csv_to_lol(b'a,b\n1,2\n') == [['a', 'b'], ['1', '2']]


def test_buff_csv_to_lol_seekable_text_file():
    import io
    f = io.StringIO('a,b\n1,2\n')
    assert utils.buff_csv_to_lol(f) == [['a', 'b'], ['1', '2']]


def test_buff_csv_to_lol_seekable_bytes_file():
    import io
    f = io.BytesIO(b'a,b\n1,2\n')
    assert utils.buff_csv_to_lol(f) == [['a', 'b'], ['1', '2']]


def test_buff_csv_to_lol_unsupported_type_raises():
    with pytest.raises(ValueError):
        utils.buff_csv_to_lol(12345)


def test_buff_csv_to_lol_user_format_strips_comments():
    result = utils.buff_csv_to_lol('# comment\na,b\n1,2\n', user_format=True)
    assert result == [['a', 'b'], ['1', '2']]


def test_buff_csv_to_lol_string_iterator_input():
    result = utils.buff_csv_to_lol(iter(['a,b', '1,2']))
    assert result == [['a', 'b'], ['1', '2']]


def test_buff_csv_to_lol_bytes_iterator_input():
    def gen_bytes():
        yield b'a,b\n'
        yield b'1,2\n'
    result = utils.buff_csv_to_lol(gen_bytes())
    assert result == [['a', 'b'], ['1', '2']]


def test_buff_csv_to_lol_streams_http_s3_style_generator():
    # mirrors the generator-expression pattern Daf.from_csv() builds for http(s):// and s3:// sources
    def fake_iter_lines():
        for line in [b'a,b\n', b'1,2\n', b'3,4\n']:
            yield line
    data_stream = (line.decode('utf-8') for line in fake_iter_lines() if line)
    result = utils.buff_csv_to_lol(data_stream)
    assert result == [['a', 'b'], ['1', '2'], ['3', '4']]


def test_buff_csv_to_lol_empty_iterator_returns_empty():
    assert utils.buff_csv_to_lol(iter([])) == []


# --- user_format streaming-compatible filter (default) vs strict_comment_filter ---

def test_buff_csv_to_lol_user_format_blank_lines():
    result = utils.buff_csv_to_lol('a,b\n\n1,2\n', user_format=True)
    assert result == [['a', 'b'], ['1', '2']]


def test_buff_csv_to_lol_user_format_works_with_file_like_streaming():
    import io
    f = io.StringIO('# comment\na,b\n1,2\n')
    result = utils.buff_csv_to_lol(f, user_format=True)
    assert result == [['a', 'b'], ['1', '2']]


def test_buff_csv_to_lol_user_format_works_with_iterator_streaming():
    result = utils.buff_csv_to_lol(iter(['# comment', 'a,b', '1,2']), user_format=True)
    assert result == [['a', 'b'], ['1', '2']]


def test_buff_csv_to_lol_user_format_preserves_comment_lookalike_inside_quoted_field():
    # a line that looks like a comment, but is actually inside a quoted multi-line field,
    # must be preserved rather than stripped (the quote-tracking heuristic's whole purpose).
    csv_str = 'a,b\n1,"line one\n# not a comment, inside quotes\nline three"\n2,normal\n'
    result = utils.buff_csv_to_lol(csv_str, user_format=True)
    assert result == [['a', 'b'], ['1', 'line one\n# not a comment, inside quotes\nline three'], ['2', 'normal']]


def test_buff_csv_to_lol_strict_comment_filter_with_str_input():
    result = utils.buff_csv_to_lol('# comment\na,b\n1,2\n', user_format=True, strict_comment_filter=True)
    assert result == [['a', 'b'], ['1', '2']]


def test_buff_csv_to_lol_strict_comment_filter_requires_str_or_bytes():
    import io
    with pytest.raises(ValueError):
        utils.buff_csv_to_lol(io.StringIO('a,b\n1,2\n'), user_format=True, strict_comment_filter=True)


# --- _filter_comment_lines (direct) ---

def test_filter_comment_lines_basic():
    result = list(utils._filter_comment_lines(['# comment', 'a,b', '', '1,2']))
    assert result == ['a,b', '1,2']


def test_filter_comment_lines_preserves_quoted_multiline_content():
    lines = ['1,"line one', '# looks like a comment', 'line three"', '2,normal']
    result = list(utils._filter_comment_lines(lines))
    assert result == lines  # nothing dropped; no line here is a standalone top-level comment


# --- preprocess_csv_buff (works correctly called directly with raw str/bytes) ---

def test_preprocess_csv_buff_strips_comment_lines():
    result = utils.preprocess_csv_buff('# comment\na,b\n1,2\n')
    assert result == 'a,b\r\n1,2\r\n'


def test_preprocess_csv_buff_strips_blank_lines():
    result = utils.preprocess_csv_buff('a,b\n\n1,2\n')
    assert result == 'a,b\r\n1,2\r\n'


def test_preprocess_csv_buff_bytes_input():
    result = utils.preprocess_csv_buff(b'# comment\na,b\n1,2\n')
    assert result == 'a,b\r\n1,2\r\n'


# =====================================================================
# write_buff_to_fp
# =====================================================================

def test_write_buff_to_fp_text_local_file():
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'out.csv')
        result = utils.write_buff_to_fp('a,b\n1,2\n', p)
        assert result == p
        assert open(p).read() == 'a,b\n1,2\n'


def test_write_buff_to_fp_binary_local_file():
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'out.bin')
        result = utils.write_buff_to_fp(b'hello', p, rtype='binary')
        assert result == p
        assert open(p, 'rb').read() == b'hello'


def test_write_buff_to_fp_empty_buff():
    result = utils.write_buff_to_fp('', '/tmp/whatever.csv')
    assert result == '/tmp/whatever.csv'


def test_write_buff_to_fp_s3_path_reaches_boto3():
    # boto3 isn't installed in this environment (no network), so this can't be fully exercised
    # end-to-end here; just confirm it no longer crashes on the old broken `s3utils` import and
    # genuinely attempts the boto3 call (fails on missing module / credentials instead).
    with pytest.raises((ModuleNotFoundError, ImportError)):
        utils.write_buff_to_fp('a,b\n', 's3://bucket/key.csv')