# test_daf_misc.py
#
# Tests for the long tail of smaller Daf methods/properties: itermode/retmode properties,
# dunder methods (__contains__, __str__, __repr__, __format__), isin, to_value, to_klist,
# to_json, extend, drop_cols, set_cols, flatten, keys, select_where, dict_to_md, set_keyfield,
# _rebuild_kd/_build_kd/_get_keyval, update_row, diff_da, sum, the valuecounts_for_* family,
# set_icol/set_icol_irows/set_col_irows, apply_to_col, iloc, and the DafIterator /
# _IndirectRowView helper classes.
#
# This batch surfaced 6 real bugs, now fixed:
#   - keys(astype='view') with no keyfield set raised AttributeError -- () .keys() doesn't
#     exist (() is a tuple, not a dict).
#   - drop_cols() crashed with AttributeError when self.dtypes was None (the normal default),
#     and separately had an inverted filter condition that kept the dtype of the *dropped*
#     column while discarding dtypes for the columns actually being kept.
#   - _IndirectRowView.values()/items(): using `yield` later in the function body makes the
#     whole function a generator, so `return self.row.values()` in the no-indirect-col branch
#     was silently discarded (a generator function always returns a generator object when
#     called, and `return` inside one just ends iteration without producing a usable value).
#   - set_keyfield(''): the if-not-keyfield reset branch didn't return immediately, falling
#     through to _is_keyfield_valid(''), which incorrectly evaluated '' as an invalid column
#     name -- so resetting the keyfield with silent_error=False raised KeyError unexpectedly.
#   - apply_to_col(): self[:, col] returns a Daf (whose iteration yields row dicts), not a flat
#     list of values, so map(func, self[:, col]) was calling func with dicts instead of values.
#   - iloc(rtype='list'): called to_list(irow=..., icol=...), but those parameters were removed
#     from to_list() in a prior refactor (left as "(removed)" in its own docstring) -- this call
#     site was never updated, raising TypeError.

import pytest

from daffodil.daf import Daf, DafIterator, _IndirectRowView
from daffodil.keyedlist import KeyedList


# =====================================================================
# retmode / itermode properties
# =====================================================================

def test_retmode_default_and_setter():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    assert daf.retmode == Daf.RETMODE_OBJ
    daf.retmode = Daf.RETMODE_VAL
    assert daf.retmode == Daf.RETMODE_VAL


def test_retmode_invalid_raises():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    with pytest.raises(ValueError):
        daf.retmode = 'bogus'


def test_itermode_default_and_setter():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    assert daf.itermode == Daf.ITERMODE_DICT
    daf.itermode = Daf.ITERMODE_KEYEDLIST
    assert daf.itermode == Daf.ITERMODE_KEYEDLIST


def test_itermode_invalid_raises():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    with pytest.raises(ValueError):
        daf.itermode = 'bogus'


# =====================================================================
# __contains__ / __str__ / __repr__ / __format__
# =====================================================================

def test_contains_with_keyfield():
    daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'name'], keyfield='id')
    assert 1 in daf
    assert 99 not in daf


def test_contains_no_keyfield_raises():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    with pytest.raises(KeyError):
        1 in daf


def test_str_and_repr_return_strings():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    assert isinstance(str(daf), str)
    assert isinstance(repr(daf), str)


def test_format_with_spec_on_single_cell():
    single = Daf(lol=[[42]], cols=['n'])
    assert '{:.2f}'.format(single) == '42.00'


def test_format_no_spec_uses_str():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    assert format(daf) == str(daf)


# =====================================================================
# isin
# =====================================================================

def test_isin_basic():
    assert Daf.isin([1, 2, 3, 4], [2, 4]) == [False, True, False, True]


def test_isin_large_lists_uses_dict_fromkeys_path():
    # exercises len(listlike1) > 10 and len(listlike2) > 30
    result = Daf.isin(list(range(15)), list(range(35)))
    assert result == [True] * 15


# =====================================================================
# to_value
# =====================================================================

def test_to_value_single_cell():
    daf = Daf(lol=[[42]], cols=['n'])
    assert daf.to_value() == 42


def test_to_value_wrong_shape_raises():
    daf = Daf(lol=[[1, 2]], cols=['a', 'b'])
    with pytest.raises(ValueError):
        daf.to_value()


def test_to_value_wrong_shape_with_default():
    daf = Daf(lol=[[1, 2]], cols=['a', 'b'])
    assert daf.to_value(default=-1) == -1


# =====================================================================
# to_klist
# =====================================================================

def test_to_klist_returns_keyedlist():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    klist = daf.to_klist(0)
    assert klist.to_dict() == {'id': 1, 'name': 'a'}


# =====================================================================
# to_json
# =====================================================================

def test_to_json_basic():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'], dtypes={'id': int, 'name': str})
    result = daf.to_json()
    assert '"lol": [[1, "a"]]' in result
    assert '"dtypes": {"id": "int", "name": "str"}' in result


def test_to_json_concise_strips_empty():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    result = daf.to_json(concise=True)
    assert 'keyfield' not in result  # empty keyfield stripped


# =====================================================================
# extend
# =====================================================================

def test_extend_basic():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    daf.extend([{'id': 2, 'name': 'b'}, {'id': 3, 'name': 'c'}])
    assert daf.lol == [[1, 'a'], [2, 'b'], [3, 'c']]


def test_extend_empty_list_noop():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    daf.extend([])
    assert daf.lol == [[1, 'a']]


# =====================================================================
# drop_cols
# =====================================================================

def test_drop_cols_with_dtypes():
    daf = Daf(lol=[[1, 'a', True]], cols=['id', 'name', 'flag'],
              dtypes={'id': int, 'name': str, 'flag': bool})
    daf.drop_cols(['name'])
    assert daf.lol == [[1, True]]
    assert list(daf.hd.keys()) == ['id', 'flag']
    assert daf.dtypes == {'id': int, 'flag': bool}


def test_drop_cols_no_dtypes_does_not_crash():
    # this is the bug we found: previously raised AttributeError when dtypes was None
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    daf.drop_cols(['name'])
    assert daf.lol == [[1]]
    assert list(daf.hd.keys()) == ['id']


def test_drop_cols_none_is_noop():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    result = daf.drop_cols(None)
    assert result is None
    assert daf.lol == [[1, 'a']]


def test_drop_cols_invalidates_kd_when_keyfield_dropped():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'], keyfield='id')
    daf._rebuild_kd_if_invalidated()
    daf.drop_cols(['id'])
    assert daf._kd == {}


# =====================================================================
# set_cols
# =====================================================================

def test_set_cols_generates_spreadsheet_names_when_none():
    daf = Daf(lol=[[1, 'a']])
    daf.set_cols()
    assert list(daf.hd.keys()) == ['A', 'B']


def test_set_cols_explicit_names():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    daf.set_cols(['new_id', 'new_name'])
    assert list(daf.hd.keys()) == ['new_id', 'new_name']


def test_set_cols_too_few_raises():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    with pytest.raises(AttributeError):
        daf.set_cols(['only_one'])


# =====================================================================
# flatten
# =====================================================================

def test_flatten_list_column_to_pyon():
    daf = Daf(lol=[[1, [1, 2, 3]]], cols=['id', 'items'], dtypes={'id': int, 'items': list})
    daf.flatten()
    assert daf.lol == [[1, '[1, 2, 3]']]


# =====================================================================
# keys() (astype='view' with no keyfield set)
# =====================================================================

def test_keys_no_keyfield_list():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    assert daf.keys() == []


def test_keys_no_keyfield_view_does_not_crash():
    # this is the bug we found: () .keys() doesn't exist (() is a tuple, not a dict)
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    result = daf.keys(astype='view')
    assert list(result) == []


def test_keys_no_keyfield_silent_error_false_raises():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    with pytest.raises(Exception):
        daf.keys(silent_error=False)


def test_keys_with_keyfield():
    daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'name'], keyfield='id')
    assert daf.keys() == [1, 2]
    assert list(daf.keys(astype='view')) == [1, 2]


# =====================================================================
# DafIterator
# =====================================================================

def test_dafiterator_rtype_list():
    daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'name'])
    result = list(DafIterator(daf, rtype=list))
    assert result == [[1, 'a'], [2, 'b']]


def test_dafiterator_rtype_keyedlist():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    result = list(DafIterator(daf, rtype=KeyedList))
    assert result[0].to_dict() == {'id': 1, 'name': 'a'}


def test_dafiterator_unknown_rtype_raises():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    with pytest.raises(NotImplementedError):
        list(DafIterator(daf, rtype=str))


# =====================================================================
# _IndirectRowView
# =====================================================================

def test_indirect_row_view_no_indirect_col():
    row = {'a': 1}
    view = _IndirectRowView(row, None)
    assert view.get('a') == 1
    assert view.get('missing', 'default') == 'default'
    assert list(view.keys()) == ['a']
    assert list(view.values()) == [1]
    assert list(view.items()) == [('a', 1)]
    assert view['a'] == 1
    assert view['missing'] == ''


def test_indirect_row_view_with_indirect_col():
    row = {'a': 1, 'meta': {'b': 2}}
    view = _IndirectRowView(row, 'meta')
    assert view['a'] == 1
    assert view['b'] == 2
    assert view['missing'] == ''
    assert view.get('c', 'default') == 'default'
    assert set(view.keys()) == {'a', 'meta', 'b'}
    assert list(view.values()) == [1, {'b': 2}, 2]
    assert ('a', 1) in list(view.items())


# =====================================================================
# select_where
# =====================================================================

def test_select_where_basic():
    daf = Daf(lol=[[1], [10]], cols=['n'])
    result = daf.select_where(lambda row: row['n'] > 5)
    assert result.lol == [[10]]


def test_select_where_indirect_col():
    daf = Daf(lol=[[1, {'x': 10}], [2, {'x': 1}]], cols=['id', 'meta'])
    result = daf.select_where(lambda row: row['x'] > 5, indirect_col='meta')
    assert result.lol == [[1, {'x': 10}]]


# =====================================================================
# dict_to_md
# =====================================================================

def test_dict_to_md_default_cols():
    result = Daf.dict_to_md({'a': 1, 'b': 2})
    assert '| key' in result
    assert '| a' in result


# =====================================================================
# set_keyfield
# =====================================================================

def test_set_keyfield_reset_to_empty():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'], keyfield='id')
    daf.set_keyfield('', silent_error=False)
    assert daf.keyfield == ''


def test_set_keyfield_reset_to_empty_silent_error_true_default():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'], keyfield='id')
    daf.set_keyfield('')
    assert daf.keyfield == ''


def test_set_keyfield_change_to_valid_column():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'], keyfield='id')
    daf.set_keyfield('name')
    assert daf.keyfield == 'name'


def test_set_keyfield_invalid_raises():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    with pytest.raises(KeyError):
        daf.set_keyfield('bogus', silent_error=False)


def test_set_keyfield_empty_daf_noop():
    daf = Daf()
    result = daf.set_keyfield('x')
    assert result is daf


# =====================================================================
# _rebuild_kd / _build_kd / _get_keyval
# =====================================================================

def test_rebuild_kd_single_keyfield():
    daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'name'], keyfield='id')
    daf._rebuild_kd()
    assert daf._kd == {1: 0, 2: 1}


def test_rebuild_kd_tuple_keyfield():
    daf = Daf(lol=[[1, 'a', 10], [2, 'b', 20]], cols=['id', 'name', 'val'], keyfield=('id', 'name'))
    daf._rebuild_kd()
    assert daf._kd == {(1, 'a'): 0, (2, 'b'): 1}


def test_get_keyval_single_keyfield():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'], keyfield='id')
    assert daf._get_keyval({'id': 5, 'name': 'x'}) == 5


def test_get_keyval_tuple_keyfield():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'], keyfield=('id', 'name'))
    assert daf._get_keyval({'id': 5, 'name': 'x'}) == (5, 'x')


# =====================================================================
# update_row / diff_da / sum
# =====================================================================

def test_update_row():
    result = Daf.update_row({'a': 1, 'b': 2}, {'b': 99, 'c': 3})
    assert result == {'a': 1, 'b': 99, 'c': 3}


def test_diff_da_basic():
    result = Daf.diff_da({'a': 10, 'b': 5}, {'a': 3, 'b': 2}, keys=['a', 'b'])
    assert result == {'a': 7, 'b': 3}


def test_diff_da_missing_keys_default_to_zero():
    result = Daf.diff_da({'a': 10}, {'b': 2}, keys=['a', 'b'])
    assert result == {'a': 10, 'b': -2}


def test_diff_da_string_key():
    result = Daf.diff_da({'a': 10}, {'a': 3}, keys='a')
    assert result == {'a': 7}


def test_sum_all_columns():
    daf = Daf(lol=[[1, 2], [3, 4]], cols=['a', 'b'])
    assert daf.sum() == {'a': 4.0, 'b': 6.0}


def test_sum_specific_columns():
    daf = Daf(lol=[[1, 2], [3, 4]], cols=['a', 'b'])
    assert daf.sum(['a']) == {'a': 4.0}


def test_sum_numeric_only_with_dtypes():
    daf = Daf(lol=[['1', 'x'], ['3', 'y']], cols=['a', 'b'], dtypes={'a': int, 'b': str})
    assert daf.sum(['a', 'b'], numeric_only=True) == {'a': 4}


# =====================================================================
# valuecounts_for_* family
# =====================================================================

def test_valuecounts_for_colname():
    daf = Daf(lol=[['M'], ['F'], ['M']], cols=['gender'])
    assert daf.valuecounts_for_colname('gender') == {'M': 2, 'F': 1}


def test_valuecounts_for_colnames_ls_selectedby_colname():
    daf = Daf(lol=[['M', 'north'], ['F', 'south'], ['M', 'north']], cols=['gender', 'region'])
    result = daf.valuecounts_for_colnames_ls_selectedby_colname(
        ['gender'], selectedby_colname='region', selectedby_colvalue='north')
    assert result == {'gender': {'M': 2}}


def test_valuecounts_for_colname1_groupedby_colname2():
    daf = Daf(lol=[['M', 'north'], ['F', 'south'], ['M', 'north']], cols=['gender', 'region'])
    result = daf.valuecounts_for_colname1_groupedby_colname2('gender', 'region')
    assert result == {'north': {'M': 2}, 'south': {'F': 1}}


def test_valuecounts_for_colname1_groupedby_colname2_missing_col():
    daf = Daf(lol=[['M', 'north']], cols=['gender', 'region'])
    assert daf.valuecounts_for_colname1_groupedby_colname2('missing', 'region') == {}


# =====================================================================
# set_icol / set_icol_irows / set_col_irows / apply_to_col
# =====================================================================

def test_set_icol():
    daf = Daf(lol=[[1, 2], [3, 4]], cols=['a', 'b'])
    daf.set_icol(1, 99)
    assert daf.lol == [[1, 99], [3, 99]]


def test_set_icol_irows_basic():
    daf = Daf(lol=[[1, 2], [3, 4], [5, 6]], cols=['a', 'b'])
    daf.set_icol_irows(1, [0, 2], 0)
    assert daf.lol == [[1, 0], [3, 4], [5, 0]]


def test_set_icol_irows_out_of_range_skipped():
    daf = Daf(lol=[[1, 2]], cols=['a', 'b'])
    daf.set_icol_irows(1, [99], 0)  # out of range, should be silently skipped
    assert daf.lol == [[1, 2]]


def test_set_icol_irows_negative_skipped():
    daf = Daf(lol=[[1, 2]], cols=['a', 'b'])
    daf.set_icol_irows(1, [-1], 0)  # negative, should be silently skipped
    assert daf.lol == [[1, 2]]


def test_set_col_irows_basic():
    daf = Daf(lol=[[1, 2], [3, 4]], cols=['a', 'b'])
    daf.set_col_irows('b', [0], 100)
    assert daf.lol == [[1, 100], [3, 4]]


def test_set_col_irows_missing_col_noop():
    daf = Daf(lol=[[1, 2]], cols=['a', 'b'])
    result = daf.set_col_irows('missing', [0], 100)
    assert result is daf
    assert daf.lol == [[1, 2]]


def test_apply_to_col_basic():
    # this is the bug we found: self[:, col] returns a Daf whose iteration yields row dicts,
    # not raw values, so map(func, self[:, col]) was calling func with dicts instead of values.
    daf = Daf(lol=[[1, 2], [3, 4]], cols=['a', 'b'])
    daf.apply_to_col('a', lambda x: x * 10)
    assert daf.lol == [[10, 2], [30, 4]]


# =====================================================================
# iloc
# =====================================================================

def test_iloc_dict_default():
    daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'name'])
    assert daf.iloc(0) == {'id': 1, 'name': 'a'}


def test_iloc_klist():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    assert daf.iloc(0, rtype='klist').to_dict() == {'id': 1, 'name': 'a'}


def test_iloc_list():
    # this is the bug we found: previously raised TypeError (to_list() no longer accepts irow/icol)
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    assert daf.iloc(0, rtype='list') == [1, 'a']


def test_iloc_negative_returns_empty():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    assert daf.iloc(-1) == {}


def test_iloc_out_of_range_returns_empty():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    assert daf.iloc(99) == {}
    assert daf.iloc(99, rtype='klist').to_dict() == {}


def test_iloc_no_cols_generates_spreadsheet_names():
    daf = Daf(lol=[[1, 'a']])
    assert daf.iloc(0) == {'A': 1, 'B': 'a'}
