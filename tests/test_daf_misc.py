# test_daf_misc.py
#
# Tests for the long tail of smaller Daf methods/properties: itermode/retmode properties,
# dunder methods (__contains__, __str__, __repr__, __format__), to_value, to_klist,
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
    assert result is daf
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


def test_set_cols_too_many_raises():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    with pytest.raises(AttributeError):
        daf.set_cols(['a', 'b', 'c'])
    assert daf.columns() == ['id', 'name']


def test_set_cols_any_length_on_daf_without_columns():
    daf = Daf()
    daf.set_cols(['x', 'y', 'z'])
    assert daf.columns() == ['x', 'y', 'z']


def test_set_cols_too_many_raises_on_empty_daf_with_cols():
    daf = Daf(cols=['a', 'b'])
    with pytest.raises(AttributeError):
        daf.set_cols(['x', 'y', 'z'])


def test_set_cols_keyfield_follows_the_new_names():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'], keyfield='id')
    daf.set_cols(['new_id', 'new_name'])
    assert daf.keyfield == 'new_id'
    assert daf.select_krows([1]).lol == [[1, 'a']]


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
# Final pass: remaining small/easy methods
# =====================================================================

# --- keys() astype error path ---

def test_keys_invalid_astype_raises():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'], keyfield='id')
    with pytest.raises(ValueError):
        daf.keys(astype='bogus')


# --- set_keyfield force_kd_rebuild ---

def test_set_keyfield_force_kd_rebuild():
    daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'name'])
    daf.set_keyfield('id', force_kd_rebuild=True)
    assert daf._kd == {1: 0, 2: 1}


# --- flatten ---

def test_flatten_bool_to_int():
    daf = Daf(lol=[[1, True]], cols=['id', 'flag'], dtypes={'id': int, 'flag': bool})
    daf.flatten(convert_bool_to_int=True)
    assert daf.lol == [[1, 1]]


def test_flatten_bool_unchanged_when_disabled():
    daf = Daf(lol=[[1, True]], cols=['id', 'flag'], dtypes={'id': int, 'flag': bool})
    daf.flatten(convert_bool_to_int=False)
    assert daf.lol == [[1, True]]


def test_flatten_empty_lol_noop():
    daf = Daf(lol=[], cols=['id'])
    assert daf.flatten() is daf


def test_flatten_no_dtypes_noop():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    assert daf.flatten() is daf


# --- from_lod / from_lot ---

def test_from_lod_explicit_cols():
    daf = Daf.from_lod([{'a': 1, 'b': 2}], cols=['a', 'b'])
    assert daf.lol == [[1, 2]]
    assert list(daf.hd.keys()) == ['a', 'b']


def test_from_lot_mismatched_length_raises():
    with pytest.raises(ValueError):
        Daf.from_lot([(1, 2), (3, 4, 5)], cols=['a', 'b'])


# --- find_replace ---

def test_find_replace_basic():
    daf = Daf(lol=[[1, 'hello'], [2, 'world']], cols=['id', 'text'])
    daf.find_replace(r'hel+o', 'REPLACED')
    assert daf.lol == [[1, 'REPLACED'], [2, 'world']]


# --- apply ---

def test_apply_by_row():
    daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'name'])

    def upper_row(row):
        row['name'] = row['name'].upper()
        return row

    result = daf.apply(upper_row, by='row')
    assert result.lol == [[1, 'A'], [2, 'B']]


def test_apply_by_table():
    daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'name'])
    result = daf.apply(lambda d, **kw: d.num_rows(), by='table')
    assert result == 2


def test_apply_by_col_not_implemented():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    with pytest.raises(NotImplementedError):
        daf.apply(lambda x: x, by='col')


def test_apply_invalid_by_not_implemented():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    with pytest.raises(NotImplementedError):
        daf.apply(lambda x: x, by='bogus')


def test_apply_with_keylist_filter():
    daf = Daf(lol=[[1, 'a'], [2, 'b'], [3, 'c']], cols=['id', 'name'], keyfield='id')

    def upper_row(row):
        row['name'] = row['name'].upper()
        return row

    result = daf.apply(upper_row, by='row', keylist=[1, 3])
    assert result.lol == [[1, 'A'], [3, 'C']]


def test_apply_with_large_keylist_uses_dict_path():
    daf = Daf(lol=[[i, 'x'] for i in range(40)], cols=['id', 'name'], keyfield='id')

    def upper_row(row):
        row['name'] = row['name'].upper()
        return row

    result = daf.apply(upper_row, by='row', keylist=list(range(35)))
    assert result.num_rows() == 35


# --- kcols_to_icols ---

def test_kcols_to_icols_no_hd():
    daf = Daf()
    assert daf.kcols_to_icols(['a', 'b']) == []


def test_kcols_to_icols_no_hd_inverse():
    daf = Daf()
    assert daf.kcols_to_icols(['a', 'b'], inverse=True) == range(0)


def test_kcols_to_icols_with_hd():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    assert daf.kcols_to_icols(['name']) == [1]


# --- _basic_get_record ---

def test_basic_get_record_no_hd_raises_and_does_not_name_the_columns():
    daf = Daf(lol=[[1, 'a']])
    with pytest.raises(KeysDisabledError, match='set_cols'):
        daf._basic_get_record(0)
    assert daf.columns() == []


def test_basic_get_record_with_include_cols():
    daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'name'])
    assert daf._basic_get_record(0, include_cols=['name']) == {'name': 'a'}


# --- sum_dodis ---

def test_sum_dodis_mutates_accum_in_place():
    accum = {'k1': {'a': 10}, 'k2': {'c': 5}}
    Daf.sum_dodis({'k1': {'a': 1, 'b': 2}}, accum)
    assert accum == {'k1': {'a': 11, 'b': 2}, 'k2': {'c': 5}}


def test_sum_dodis_new_key_adopted_directly():
    accum = {}
    Daf.sum_dodis({'k1': {'a': 1}}, accum)
    assert accum == {'k1': {'a': 1}}


# --- groupsum_daf / multi_groupsum ---

def test_groupsum_daf():
    daf = Daf(lol=[['M', 1, 10], ['F', 2, 20], ['M', 3, 30]], cols=['gender', 'x', 'y'])
    result = daf.groupsum_daf('gender', reduce_cols=['x', 'y'])
    assert result.lol == [['M', 4, 40], ['F', 2, 20]]


def test_multi_groupsum():
    daf = Daf(lol=[['M', 1, 10], ['F', 2, 20], ['M', 3, 30]], cols=['gender', 'x', 'y'])
    result = daf.multi_groupsum(['gender'], reduce_cols=['x', 'y'])
    assert result['gender'].lol == [['M', 4, 40], ['F', 2, 20]]


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


def test_set_col_irows_missing_col_raises_keyerror():
    daf = Daf(lol=[[1, 2]], cols=['a', 'b'])
    with pytest.raises(KeyError) as info:
        daf.set_col_irows('missing', [0], 100)
    assert info.value.args == ('missing',)
    assert daf.lol == [[1, 2]]


def test_set_col_irows_does_the_same_as_indexing():
    by_method = Daf(lol=[[1, 2], [3, 4], [5, 6]], cols=['a', 'b'])
    by_index = Daf(lol=[[1, 2], [3, 4], [5, 6]], cols=['a', 'b'])
    by_method.set_col_irows('b', [0, 2], 99)
    by_index[[0, 2], 'b'] = 99
    assert by_method.lol == by_index.lol == [[1, 99], [3, 4], [5, 99]]


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


def test_iloc_negative_counts_from_the_end():
    daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'name'])
    assert daf.iloc(-1) == {'id': 2, 'name': 'b'}
    assert daf.iloc(-2) == {'id': 1, 'name': 'a'}
    assert daf.iloc(-1, rtype='list') == [2, 'b']
    assert daf.iloc(-1, rtype='klist').to_dict() == {'id': 2, 'name': 'b'}


def test_iloc_out_of_range_raises_indexerror():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    for position in (1, 99, -2, -99):
        with pytest.raises(IndexError):
            daf.iloc(position)
    with pytest.raises(IndexError):
        daf.iloc(99, rtype='klist')


def test_iloc_on_a_daf_with_no_rows_is_empty():
    assert Daf().iloc(0) == {}
    assert Daf().iloc(-1) == {}
    assert Daf().to_dict() == {}
    assert Daf(cols=['a']).iloc(5, rtype='klist').to_dict() == {}


def test_icol_negative_counts_from_the_end():
    daf = Daf(lol=[[1, 'a', 10], [2, 'b', 20]], cols=['id', 'v', 'n'])
    assert daf.icol(-1) == [10, 20]
    assert daf.icol(-3) == [1, 2]
    assert daf.icol_to_la(-2, unique=True) == ['a', 'b']


def test_icol_out_of_range_raises_indexerror():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'v'])
    for position in (2, 99, -3):
        with pytest.raises(IndexError):
            daf.icol(position)


def test_icol_on_a_daf_with_no_rows_is_empty():
    assert Daf().icol(0) == []
    assert Daf(cols=['a']).icol(3) == []


def test_iloc_no_cols_raises_for_dict_and_klist_and_works_for_list():
    daf = Daf(lol=[[1, 'a']])
    with pytest.raises(KeysDisabledError, match='set_cols'):
        daf.iloc(0)
    with pytest.raises(KeysDisabledError, match='set_cols'):
        daf.iloc(0, rtype='klist')
    assert daf.iloc(0, rtype='list') == [1, 'a']
    assert daf.set_cols().iloc(0) == {'A': 1, 'B': 'a'}


def test_noop_calls_return_self_for_chaining():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'], keyfield='id')
    assert daf.drop_cols([]) is daf
    assert daf.update_by_keylist([], {'name': 'x'}) is daf
    assert daf.insert_col('') is daf
    assert daf.lol == [[1, 'a']]
    assert list(daf.hd) == ['id', 'name']


def test_drop_cols_clears_the_keyfield_when_its_column_is_dropped():
    daf = Daf(lol=[[1, 'a', 10], [2, 'b', 20]], cols=['id', 'v', 'n'], keyfield='id')
    daf.drop_cols(['id'])
    assert daf.columns() == ['v', 'n']
    assert daf.keyfield == ''
    assert daf.keys() == []


def test_drop_cols_clears_a_composite_keyfield_when_one_column_is_dropped():
    daf = Daf(lol=[[1, 'a', 10], [2, 'b', 20]], cols=['id', 'v', 'n'], keyfield=('id', 'v'))
    daf.drop_cols(['v'])
    assert daf.keyfield == ''


def test_drop_cols_keeps_the_keyfield_when_another_column_is_dropped():
    daf = Daf(lol=[[1, 'a', 10], [2, 'b', 20]], cols=['id', 'v', 'n'], keyfield='id')
    daf.drop_cols(['v'])
    assert daf.keyfield == 'id'
    assert daf.keys() == [1, 2]
    assert daf.select_record(2) == {'id': 2, 'n': 20}


# from_lod

def test_from_lod_later_dict_with_fewer_keys_gets_null():
    daf = Daf.from_lod([{'a': 1, 'b': 2}, {'a': 3}])
    assert daf.columns() == ['a', 'b']
    assert daf.lol == [[1, 2], [3, '']]


def test_from_lod_later_dict_with_a_new_key_adds_a_column():
    daf = Daf.from_lod([{'a': 1}, {'a': 3, 'b': 9}])
    assert daf.columns() == ['a', 'b']
    assert daf.lol == [[1, ''], [3, 9]]


def test_from_lod_columns_are_in_the_order_of_first_appearance_and_earlier_rows_are_padded():
    daf = Daf.from_lod([{'a': 1}, {'a': 2, 'b': 5}, {'c': 9}, {'b': 7, 'a': 0, 'd': 4}])
    assert daf.columns() == ['a', 'b', 'c', 'd']
    assert daf.lol == [[1, '', '', ''], [2, 5, '', ''], ['', '', 9, ''], [0, 7, '', 4]]


def test_from_lod_new_key_is_added_even_when_the_dict_lacks_an_old_key():
    # the same number of keys as the first dict, and one of them is new.
    daf = Daf.from_lod([{'a': 1, 'b': 2, 'c': 3}, {'a': 1, 'c': 3, 'd': 9}])
    assert daf.columns() == ['a', 'b', 'c', 'd']
    assert daf.lol == [[1, 2, 3, ''], [1, '', 3, 9]]
    # fewer keys than the first dict, and one of them is new.
    daf = Daf.from_lod([{'a': 1, 'b': 2, 'c': 3}, {'a': 1, 'd': 9}])
    assert daf.lol == [[1, 2, 3, ''], [1, '', '', 9]]


def test_from_lod_the_first_dict_may_be_empty_or_not_a_dict():
    daf = Daf.from_lod([{}, None, {'x': 1}, {'x': 2, 'y': 3}])
    assert daf.columns() == ['x', 'y']
    assert daf.lol == [[1, ''], [2, 3]]


def test_from_lod_with_only_empty_items_has_no_rows_and_no_columns():
    daf = Daf.from_lod([{}, None])
    assert daf.columns() == [] and daf.lol == []


def test_from_lod_uniform_keys_give_the_same_rows_as_before():
    lod = [{'a': i, 'b': i * 2} for i in range(5)]
    daf = Daf.from_lod(lod, keyfield='a')
    assert daf.columns() == ['a', 'b'] and daf.lol == [[i, i * 2] for i in range(5)]
    assert daf.select_krows([3]).lol == [[3, 6]]


def test_from_lod_a_keyfield_that_first_appears_in_a_later_dict_is_found():
    daf = Daf.from_lod([{'a': 1}, {'a': 2, 'id': 'x'}], keyfield='id')
    assert daf.columns() == ['a', 'id'] and daf.keyfield == 'id'


def test_from_lod_with_cols_raises_for_a_key_that_is_not_a_column():
    with pytest.raises(ValueError) as excinfo:
        Daf.from_lod([{'a': 1, 'b': 2}, {'a': 3, 'b': 4, 'c': 5, 'd': 6}], cols=['b', 'a'])
    assert "['c', 'd']" in str(excinfo.value) and 'ignore_extra_keys' in str(excinfo.value)


def test_from_lod_with_dtypes_raises_for_a_key_that_is_not_a_column():
    with pytest.raises(ValueError, match="'c'"):
        Daf.from_lod([{'a': 1, 'b': 2}, {'a': 3, 'b': 4, 'c': 5}], dtypes={'a': int, 'b': int})


def test_from_lod_with_cols_and_ignore_extra_keys_leaves_other_keys_out():
    daf = Daf.from_lod([{'a': 1, 'b': 2}, {'a': 3, 'b': 4, 'c': 5}], cols=['b', 'a'], ignore_extra_keys=True)
    assert daf.columns() == ['b', 'a'] and daf.lol == [[2, 1], [4, 3]]


def test_from_lod_with_dtypes_and_ignore_extra_keys_leaves_other_keys_out():
    daf = Daf.from_lod([{'a': 1, 'b': 2}, {'a': 3, 'b': 4, 'c': 5}], dtypes={'a': int, 'b': int}, ignore_extra_keys=True)
    assert daf.lol == [[1, 2], [3, 4]]


def test_from_lod_with_cols_gives_null_for_a_key_that_a_dict_lacks_and_accepts_all_the_keys():
    daf = Daf.from_lod([{'a': 1}, {'a': 2, 'b': 3}, {}, None], cols=['a', 'b'])
    assert daf.lol == [[1, ''], [2, 3]]


def test_from_lod_ignore_extra_keys_changes_nothing_when_there_are_no_extra_keys_or_no_cols():
    lod = [{'a': 1}, {'a': 2, 'b': 3}]
    assert Daf.from_lod(lod, ignore_extra_keys=True).lol == Daf.from_lod(lod).lol == [[1, ''], [2, 3]]


def test_from_lod_to_cols_with_dtypes_that_leave_out_a_key_raises_and_does_not_mislabel_the_data():
    # before, dtypes naming only y and z put the values of y under the key x, and those of z under y.
    lod = [{'x': 1, 'y': 2, 'z': 3}, {'x': 4, 'y': 5, 'z': 6}]
    with pytest.raises(ValueError, match="'x'"):
        Daf.from_lod_to_cols(lod, dtypes={'y': int, 'z': int})
    assert Daf.from_lod_to_cols(lod, dtypes={'x': int, 'y': int, 'z': int}).lol == [['x', 1, 4], ['y', 2, 5], ['z', 3, 6]]


def test_from_dod_with_dtypes_that_leave_out_the_key_column_raises():
    dod = {'r1': {'a': 1, 'b': 2}, 'r2': {'a': 3, 'b': 4}}
    with pytest.raises(ValueError, match="'id'"):
        Daf.from_dod(dod, keyfield='id', dtypes={'a': int, 'b': int})
    assert Daf.from_dod(dod, keyfield='id', dtypes={'id': str, 'a': int, 'b': int}).lol == [['r1', 1, 2], ['r2', 3, 4]]


def test_from_lod_skips_empty_and_non_dict_items():
    daf = Daf.from_lod([{'a': 1}, {}, None, {'a': 2}])
    assert daf.lol == [[1], [2]]


def test_from_lod_cells_may_be_arrays():
    np = pytest.importorskip('numpy')
    daf = Daf.from_lod([{'a': 1, 'b': 2}, {'a': np.array([1, 2]), 'b': 3}])
    assert daf.num_rows() == 2


def test_apply_dtypes_keeps_the_text_of_a_value_that_cannot_be_converted():
    daf = Daf(lol=[['1', '2.5'], ['x', 'abc'], ['', '']], cols=['n', 'f'])
    daf.apply_dtypes(dtypes={'n': int, 'f': float})
    assert daf.lol == [[1, 2.5], ['x', 'abc'], ['', '']]


def test_apply_dtypes_keeps_every_digit_of_a_large_number():
    daf = Daf(lol=[['12345678901234567890']], cols=['id'])
    daf.apply_dtypes(dtypes={'id': int})
    assert daf.lol == [[12345678901234567890]]


# columns that are added keep the names and the data together

def _named():
    return Daf(lol=[[1, 'ab12'], [2, 'cd34']], cols=['id', 's'], keyfield='id')


def test_assign_icol_append_names_the_new_column():
    daf = _named()
    daf.assign_icol(-1, ['x', 'y'])
    assert daf.columns() == ['id', 's', 'C']
    assert daf.lol == [[1, 'ab12', 'x'], [2, 'cd34', 'y']]
    assert daf.to_lod() == [{'id': 1, 's': 'ab12', 'C': 'x'}, {'id': 2, 's': 'cd34', 'C': 'y'}]


def test_assign_icol_existing_column_changes_no_names():
    daf = _named()
    daf.assign_icol(1, ['x', 'y'])
    assert daf.columns() == ['id', 's']


def test_assign_icol_append_with_a_taken_name_gets_a_suffix():
    daf = Daf(lol=[[1, 2]], cols=['id', 'C'])
    daf.assign_icol(-1, [9])
    assert daf.columns() == ['id', 'C', 'C_1']


def test_insert_icol_without_a_name_names_the_new_column():
    daf = _named()
    daf.insert_icol(1, ['x', 'y'])
    assert daf.columns() == ['id', 'C', 's']
    assert daf.lol == [[1, 'x', 'ab12'], [2, 'y', 'cd34']]


def test_insert_icol_with_a_name_uses_it():
    daf = _named()
    daf.insert_icol(1, ['x', 'y'], colname='mine')
    assert daf.columns() == ['id', 'mine', 's']


def test_insert_icol_on_a_daf_without_names_adds_no_names():
    daf = Daf(lol=[[1, 'a']])
    daf.insert_icol(1, ['x'])
    assert daf.columns() == []
    assert daf.lol == [[1, 'x', 'a']]


def test_annotate_daf_adds_a_new_field_as_a_column():
    daf = _named()
    other = Daf(lol=[[1, 'P'], [2, 'Q']], cols=['id', 'w'], keyfield='id')
    daf.annotate_daf(other, {'newcol': 'w'})
    assert daf.columns() == ['id', 's', 'newcol']
    assert daf.lol == [[1, 'ab12', 'P'], [2, 'cd34', 'Q']]


def test_regex_select_into_a_new_column():
    daf = _named()
    daf.set_col2_from_col1_using_regex_select('s', 'n', regex=r'(\d+)')
    assert daf.columns() == ['id', 's', 'n']
    assert daf.lol == [[1, 'ab12', '12'], [2, 'cd34', '34']]


def test_regex_select_with_an_unknown_col1_adds_no_column():
    daf = _named()
    with pytest.raises(KeyError):
        daf.set_col2_from_col1_using_regex_select('nope', 'n', regex=r'(\d+)')
    assert daf.columns() == ['id', 's']


def test_apply_replace_regex_into_a_new_column():
    daf = _named()
    daf.apply_replace_regex('s', 't', replace_regex='/ab//')
    assert daf.columns() == ['id', 's', 't']
    assert daf.lol == [[1, 'ab12', '12'], [2, 'cd34', 'cd34']]


def test_apply_replace_regex_with_an_unknown_column_adds_no_column():
    daf = _named()
    daf.apply_replace_regex('nope', 't', replace_regex='/a/b/')
    assert daf.columns() == ['id', 's']


# from_csv_buff include_cols

_CSV = 'a,b,c\n1,2,3\n4,5,6\n'


def test_from_csv_buff_include_cols_keeps_the_listed_columns_in_that_order():
    daf = Daf.from_csv_buff(_CSV, include_cols=['c', 'a'])
    assert daf.columns() == ['c', 'a']
    assert daf.lol == [['3', '1'], ['6', '4']]


def test_from_csv_buff_include_cols_one_column():
    daf = Daf.from_csv_buff(_CSV, include_cols=['b'])
    assert daf.columns() == ['b']
    assert daf.lol == [['2'], ['5']]


def test_from_csv_buff_include_cols_as_a_single_name():
    daf = Daf.from_csv_buff(_CSV, include_cols='b')
    assert daf.columns() == ['b']


def test_from_csv_buff_include_cols_unknown_name_raises_and_names_it():
    with pytest.raises(KeyError, match="'zz'"):
        Daf.from_csv_buff(_CSV, include_cols=['a', 'zz'])


def test_from_csv_buff_include_cols_with_noheader_raises():
    with pytest.raises(ValueError, match='noheader'):
        Daf.from_csv_buff(_CSV, include_cols=['a'], noheader=True)


def test_from_csv_buff_include_cols_from_bytes():
    daf = Daf.from_csv_buff(_CSV.encode(), include_cols=['b', 'c'])
    assert daf.lol == [['2', '3'], ['5', '6']]


def test_from_csv_buff_include_cols_from_a_stream_of_lines():
    lines = iter(['a,b,c\n', '1,2,3\n', '4,5,6\n'])
    daf = Daf.from_csv_buff(lines, include_cols=['c'])
    assert daf.lol == [['3'], ['6']]


def test_from_csv_buff_include_cols_with_dtypes_and_keyfield():
    daf = Daf.from_csv_buff(_CSV, include_cols=['c', 'a'], dtypes={'a': int, 'b': int, 'c': int}, keyfield='a')
    assert daf.lol == [[3, 1], [6, 4]]
    assert daf.keys() == [1, 4]


def test_from_csv_buff_include_cols_short_row_gives_null_and_blank_row_stays_empty():
    daf = Daf.from_csv_buff('a,b,c\n1,2,3\n4\n\n7,8,9\n', include_cols=['a', 'c'])
    assert daf.lol[0] == ['1', '3']
    assert daf.lol[1] == ['4', '']
    assert daf.lol[-1] == ['7', '9']


def test_from_csv_buff_include_cols_quoted_fields_and_user_format():
    text = '# a comment\na,b\n"x,y",2\n'
    daf = Daf.from_csv_buff(text, include_cols=['a'], user_format=True)
    assert daf.lol == [['x,y']]


def test_from_csv_buff_include_cols_repeated_header_name_uses_the_first():
    daf = Daf.from_csv_buff('a,a,b\n1,2,3\n', include_cols=['a'])
    assert daf.lol == [['1']]


def test_from_csv_buff_include_cols_header_only_file():
    daf = Daf.from_csv_buff('a,b\n', include_cols=['b'])
    assert daf.columns() == ['b']
    assert daf.num_rows() == 0


def test_from_csv_with_include_cols_reads_a_file(tmp_path):
    path = tmp_path / 'x.csv'
    path.write_text(_CSV)
    daf = Daf.from_csv(path, include_cols=['b'])
    assert daf.lol == [['2'], ['5']]
    daf2 = Daf.from_csv_file(str(path), include_cols=['c', 'b'])
    assert daf2.lol == [['3', '2'], ['6', '5']]


def test_from_csv_buff_include_cols_gives_the_same_rows_as_selecting_afterwards():
    wide = 'c0,c1,c2,c3,c4\n' + '\n'.join(','.join(str(r * 5 + i) for i in range(5)) for r in range(20)) + '\n'
    cols = ['c4', 'c1', 'c3']
    assert Daf.from_csv_buff(wide, include_cols=cols).lol == Daf.from_csv_buff(wide)[:, cols].lol


# from_cols_dol

def test_from_cols_dol_equal_lists():
    daf = Daf.from_cols_dol({'A': [1, 2, 3], 'B': [4, 5, 6]})
    assert daf.columns() == ['A', 'B']
    assert daf.lol == [[1, 4], [2, 5], [3, 6]]


def test_from_cols_dol_shorter_list_raises_and_names_the_column():
    with pytest.raises(ValueError, match="column 'B' has 1 values, but column 'A' has 2"):
        Daf.from_cols_dol({'A': [1, 2], 'B': [3]})


def test_from_cols_dol_longer_list_raises():
    with pytest.raises(ValueError, match="column 'B' has 2 values, but column 'A' has 1"):
        Daf.from_cols_dol({'A': [1], 'B': [3, 4]})


def test_from_cols_dol_rows_are_new_lists_and_keyfield_and_dtypes_work():
    daf = Daf.from_cols_dol({'id': [1, 2], 'v': ['a', 'b']}, keyfield='id', dtypes={'id': int, 'v': str})
    assert daf.keys() == [1, 2]
    assert daf.select_record(2) == {'id': 2, 'v': 'b'}
    assert daf.lol[0] is not daf.lol[1]


def test_from_cols_dol_empty_and_empty_lists():
    assert Daf.from_cols_dol({}).shape() == (0, 0)
    daf = Daf.from_cols_dol({'A': [], 'B': []})
    assert daf.columns() == ['A', 'B']
    assert daf.num_rows() == 0


def test_from_cols_dol_single_column_and_numpy_arrays():
    np = pytest.importorskip('numpy')
    assert Daf.from_cols_dol({'A': [1, 2]}).lol == [[1], [2]]
    daf = Daf.from_cols_dol({'A': np.array([1, 2]), 'B': np.array([3, 4])})
    assert daf.lol == [[1, 3], [2, 4]]


# transpose default cols

def test_transpose_default_cols_one_per_source_row():
    daf = Daf(lol=[[1, 'a'], [2, 'b'], [3, 'c']], cols=['id', 'v'])
    result = daf.transpose()
    assert result.columns() == ['A', 'B', 'C']
    assert result.lol == [[1, 2, 3], ['a', 'b', 'c']]
    assert result.to_lod() == [{'A': 1, 'B': 2, 'C': 3}, {'A': 'a', 'B': 'b', 'C': 'c'}]


def test_transpose_include_header_default_cols_start_with_key():
    daf = Daf(lol=[[1, 'a'], [2, 'b'], [3, 'c']], cols=['id', 'v'])
    result = daf.transpose(include_header=True)
    assert result.columns() == ['key', 'A', 'B', 'C']
    assert result.lol == [['id', 1, 2, 3], ['v', 'a', 'b', 'c']]


def test_transpose_explicit_new_cols_unchanged():
    daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'])
    assert daf.transpose(new_cols=['r0', 'r1']).columns() == ['r0', 'r1']


# daf_to_lol_summary row and col limits

def _summary_ids(max_rows, max_cols=0, num_rows=8):
    daf = Daf(lol=[[i, f'n{i}'] for i in range(num_rows)], cols=['id', 'name'])
    return [row[0] for row in daf.daf_to_lol_summary(max_rows=max_rows, max_cols=max_cols)[1:]]


def test_summary_only_max_cols_keeps_all_rows_with_no_divider():
    daf = Daf(lol=[[i, f'n{i}', i * 2, f'x{i}'] for i in range(8)], cols=['id', 'name', 'val', 'tag'])
    result = daf.daf_to_lol_summary(max_rows=0, max_cols=2)
    assert result[0] == ['id', '...', 'tag']
    assert [row[0] for row in result[1:]] == list(range(8))


def test_summary_max_rows_one_keeps_first_row_and_divider():
    assert _summary_ids(1) == [0, '...']


def test_summary_even_limit_unchanged():
    assert _summary_ids(2) == [0, '...', 7]
    assert _summary_ids(4) == [0, 1, '...', 6, 7]


def test_summary_odd_limit_keeps_extra_row_at_start():
    assert _summary_ids(3) == [0, 1, '...', 7]
    assert _summary_ids(5) == [0, 1, 2, '...', 6, 7]


def test_summary_limit_at_or_over_row_count_keeps_all_rows():
    assert _summary_ids(8) == list(range(8))
    assert _summary_ids(9) == list(range(8))


def test_to_md_only_max_cols_has_no_divider_row():
    daf = Daf(lol=[[i, f'n{i}', i * 2, f'x{i}'] for i in range(3)], cols=['id', 'name', 'val', 'tag'])
    assert daf.to_md(max_cols=2).splitlines()[2].startswith('|  0')


# to_json dtypes

def test_to_json_does_not_change_missing_dtypes():
    daf = Daf(lol=[[1]], cols=['a'])
    assert daf.dtypes is None
    daf.to_json()
    assert daf.dtypes is None


def test_json_round_trip_keeps_list_and_dict_dtypes():
    daf = Daf(lol=[[1, [1, 2], {'k': 1}, 'x', 1.5, True]], cols=list('abcdef'),
              dtypes={'a': int, 'b': list, 'c': dict, 'd': str, 'e': float, 'f': bool})
    again = Daf.from_json(daf.to_json())
    assert again.dtypes == daf.dtypes
    assert again.lol == daf.lol


def test_json_round_trip_then_apply_dtypes_for_list_column():
    daf = Daf(lol=[['1', '[1, 2]']], cols=['a', 'b'], dtypes={'a': int, 'b': list})
    again = Daf.from_json(daf.to_json())
    assert again.apply_dtypes().lol == [[1, [1, 2]]]


def test_json_unknown_dtype_name_comes_back_as_text():
    again = Daf.from_json('{"lol": [["x"]], "hd": {"a": 0}, "dtypes": {"a": "date"}}')
    assert again.dtypes == {'a': 'date'}


# remove_key with a composite keyfield

def _composite_daf() -> Daf:
    return Daf(lol=[['a', 1, 'x'], ['a', 2, 'y'], ['b', 1, 'z'], ['b', 2, 'w']],
               cols=['g', 'n', 'v'], keyfield=('g', 'n'))


def test_remove_key_composite_bare_tuple_is_one_key():
    result = _composite_daf().remove_key(('a', 1))
    assert result.lol == [['a', 2, 'y'], ['b', 1, 'z'], ['b', 2, 'w']]


def test_remove_key_composite_list_of_tuples_is_list_of_keys():
    result = _composite_daf().remove_key([('a', 1), ('b', 2)])
    assert result.lol == [['a', 2, 'y'], ['b', 1, 'z']]


def test_remove_key_composite_tuple_of_tuples_is_a_range():
    result = _composite_daf().remove_key((('a', 2), ('b', 1)))
    assert result.lol == [['a', 1, 'x'], ['b', 2, 'w']]


def test_remove_key_composite_missing_key_raises_or_is_ignored():
    with pytest.raises(KeyError):
        _composite_daf().remove_key(('c', 9))
    assert len(_composite_daf().remove_key(('c', 9), silent_error=True)) == 4


def test_remove_key_single_keyfield_tuple_is_still_a_range():
    daf = Daf(lol=[[1, 'x'], [2, 'y'], [3, 'z'], [4, 'w']], cols=['id', 'v'], keyfield='id')
    assert daf.remove_key((2, 3)).lol == [[1, 'x'], [4, 'w']]


def test_to_json_leaves_a_dict_of_dtypes_alone():
    daf = Daf(lol=[[1]], cols=['a'], dtypes={'a': int})
    daf.to_json()
    assert daf.dtypes == {'a': int}


# sum_np

def test_sum_np_blank_none_and_nan_count_as_zero():
    daf = Daf(lol=[[1, 10, 1.5], [2, '', None], [3, 30, float('nan')]], cols=['x', 'y', 'z'])
    assert daf.sum_np() == {'x': 6, 'y': 40, 'z': 1.5}


def test_sum_np_keeps_int_totals_as_int():
    result = Daf(lol=[[1], [2]], cols=['x']).sum_np()
    assert result == {'x': 3} and isinstance(result['x'], int)


def test_sum_np_subset_of_columns():
    daf = Daf(lol=[[1, 'a', 5], [2, 'b', 6]], cols=['x', 't', 'n'])
    assert daf.sum_np(['n', 'x']) == {'n': 11, 'x': 3}


def test_sum_np_text_column_raises_and_names_the_column():
    daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['x', 't'])
    with pytest.raises(TypeError, match="column 't'"):
        daf.sum_np()
    assert daf.sum_np(['x']) == {'x': 3}


def test_sum_np_unknown_column_raises_keyerror():
    with pytest.raises(KeyError):
        Daf(lol=[[1]], cols=['x']).sum_np(['nope'])


def test_sum_np_empty_daf():
    assert Daf().sum_np() == {}
    assert Daf(cols=['a']).sum_np() == {}


def test_sum_np_does_not_change_the_daf():
    daf = Daf(lol=[[1, '']], cols=['x', 'y'])
    daf.sum_np()
    assert daf.lol == [[1, '']]


# apply_formulas restores retmode after an error

def test_apply_formulas_restores_retmode_after_formula_error(capsys):
    daf = Daf(cols=['A', 'B'], lol=[[1, 2], [3, 4]])
    formulas = Daf(cols=['A', 'B'], lol=[['', '$d[0,0]+nope'], ['', '']])
    assert daf.retmode == Daf.RETMODE_OBJ
    with pytest.raises(NameError):
        daf.apply_formulas(formulas)
    assert daf.retmode == Daf.RETMODE_OBJ
    assert 'Error in formula for cell [0,1]' in capsys.readouterr().out
    assert isinstance(daf[0], Daf)


def test_apply_formulas_restores_retmode_after_circular_formulas():
    daf = Daf(cols=['A', 'B'], lol=[[1, 2], [3, 4]])
    formulas = Daf(cols=['A', 'B'], lol=[['$d[1,1]+1', ''], ['', '$d[0,0]+1']])
    with pytest.raises(RuntimeError, match='excessive evaluation loops'):
        daf.apply_formulas(formulas)
    assert daf.retmode == Daf.RETMODE_OBJ


def test_apply_formulas_restores_retmode_after_success():
    daf = Daf(cols=['A', 'B'], lol=[[1, 2]])
    daf.apply_formulas(Daf(cols=['A', 'B'], lol=[['', '$d[0,0]+10']]))
    assert daf.lol == [[1, 11]]
    assert daf.retmode == Daf.RETMODE_OBJ


# an explicit keyfield is kept when a schema supplies the cols

from daffodil.lib.schemaclass import schemaclass, SchemaBase


@schemaclass
class _KeyedSchema(SchemaBase):
    __keyfield__ = 'ballot_id'
    ballot_id: str = ''
    contest: str = ''
    votes: int = 0


@schemaclass
class _UnkeyedSchema(SchemaBase):
    ballot_id: str = ''
    contest: str = ''


def test_schema_keyfield_used_when_none_given():
    assert Daf(schema=_KeyedSchema).keyfield == 'ballot_id'


def test_explicit_keyfield_beats_schema_keyfield():
    daf = Daf(schema=_KeyedSchema, keyfield='contest')
    assert daf.columns() == ['ballot_id', 'contest', 'votes']
    assert daf.keyfield == 'contest'


def test_explicit_keyfield_kept_when_schema_has_none():
    assert Daf(schema=_UnkeyedSchema, keyfield='contest').keyfield == 'contest'


def test_explicit_keyfield_with_cols_and_schema_unchanged():
    daf = Daf(schema=_KeyedSchema, keyfield='contest', cols=['ballot_id', 'contest', 'votes'])
    assert daf.keyfield == 'contest'


def test_explicit_keyfield_beats_schema_daf_keyfield():
    schema_daf = Daf(cols=['Name', 'dtype'], lol=[['a', 'str'], ['b', 'int']])
    schema_daf.attrs['keyfield'] = 'a'
    daf = Daf(schema=schema_daf, keyfield='b')
    assert daf.columns() == ['a', 'b']
    assert daf.keyfield == 'b'
    assert Daf(schema=schema_daf).keyfield == 'a'


# missing key and column errors

from daffodil.daf import ColumnNotFoundError, KeysDisabledError


def _keyed_daf() -> Daf:
    return Daf(lol=[[1, 'a'], [2, 'b'], [2, 'c']], cols=['id', 'v'], keyfield='id')


def test_col_missing_column_is_keyerror_and_runtimeerror():
    daf = _keyed_daf()
    for exc_type in (ColumnNotFoundError, KeyError, RuntimeError):
        with pytest.raises(exc_type) as info:
            daf.col('nope')
        assert 'nope' in str(info.value)


def test_col_missing_column_silent_error_returns_empty_list():
    assert _keyed_daf().col('nope', silent_error=True) == []


def test_to_donpa_unknown_column_is_columnnotfound():
    with pytest.raises(ColumnNotFoundError):
        _keyed_daf().to_donpa(['nope'])


def test_select_record_missing_key_names_the_key():
    with pytest.raises(KeyError) as info:
        _keyed_daf().select_record(99, silent_error=False)
    assert info.value.args == (99,)
    assert _keyed_daf().select_record(99) == {}


def test_select_by_dict_expectmax_message_has_counts():
    with pytest.raises(LookupError, match=r"2 rows match, more than expectmax=1"):
        _keyed_daf().select_by_dict({'id': 2}, expectmax=1)


def test_to_dod_without_keyfield_raises_keysdisabled():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'v'])
    with pytest.raises(KeysDisabledError, match='to_dod'):
        daf.to_dod()


def test_to_dod_empty_daf_without_keyfield_is_empty_dict():
    assert Daf().to_dod() == {}


# in place methods return the Daf, so calls can be chained

def _chain_daf() -> Daf:
    return Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'], keyfield='id')


@pytest.mark.parametrize('call', [
    lambda d: d.assign_record({'id': 3, 'v': 'c'}),
    lambda d: d.assign_record({'id': 1, 'v': 'z'}),
    lambda d: d.assign_record_irow(0, {'id': 1, 'v': 'z'}),
    lambda d: d.assign_record_irow(0, None),
    lambda d: d.update_record_irow(0, {'v': 'z'}),
    lambda d: d.update_record_irow(1, {'v': 'z'}),
    lambda d: d.update_record_irow(0, None),
    lambda d: d.assign_icol(1, ['x', 'y']),
    lambda d: d.set_icol_irows(1, [0], 'q'),
    lambda d: d.find_replace('a', 'Q'),
    lambda d: d.apply_to_col('v', str.upper),
    lambda d: d.apply_in_place(lambda row: row, by='row'),
    lambda d: d.apply_formulas(Daf(cols=['id', 'v'], lol=[['', ''], ['', '']])),
    lambda d: d.set_col2_from_col1_using_regex_select('v', 'v', '(.)'),
    ])
def test_in_place_methods_return_the_same_daf(call):
    daf = _chain_daf()
    assert call(daf) is daf


def test_in_place_methods_can_be_chained():
    daf = _chain_daf()
    result = daf.assign_icol(1, ['x', 'y']).find_replace('x', 'X').drop_cols(['id'])
    assert result.lol == [['X'], ['y']]


# select_irows takes inverse, and still accepts the old name invert

def _rows_daf() -> Daf:
    return Daf(lol=[[1, 'a'], [2, 'b'], [3, 'c']], cols=['id', 'v'], keyfield='id')


def test_select_irows_inverse_drops_the_selected_rows():
    assert _rows_daf().select_irows([0, 2], inverse=True).lol == [[2, 'b']]
    assert _rows_daf().select_irows(1, inverse=True).lol == [[1, 'a'], [3, 'c']]
    assert _rows_daf().select_irows(slice(0, 2), inverse=True).lol == [[3, 'c']]


def test_select_irows_old_name_invert_still_works():
    assert _rows_daf().select_irows([0, 2], invert=True).lol == [[2, 'b']]
    assert _rows_daf().select_irows(1, invert=True).lol == [[1, 'a'], [3, 'c']]


def test_select_irows_inverse_positional():
    assert _rows_daf().select_irows([0], True).lol == [[2, 'b'], [3, 'c']]


def test_select_irows_empty_selection_with_inverse_keeps_all_rows():
    assert _rows_daf().select_irows([], inverse=True).lol == [[1, 'a'], [2, 'b'], [3, 'c']]
    assert _rows_daf().select_irows([], invert=True).lol == [[1, 'a'], [2, 'b'], [3, 'c']]


def test_readme_examples_for_dropping_rows_and_columns_work():
    daf = _rows_daf()
    assert daf.select_krows(krows=2, inverse=True).lol == [[1, 'a'], [3, 'c']]
    assert daf.select_krows(krows=[1, 3], inverse=True).lol == [[2, 'b']]
    assert daf.select_kcols(['v'], inverse=True).columns() == ['id']


# the respect_kd defaults, as the README and docstrings describe them

def _dup_key_daf() -> Daf:
    return Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'], keyfield='id')


def test_append_extend_concat_add_a_duplicate_key_by_default():
    assert _dup_key_daf().append({'id': 2, 'v': 'NEW'}).lol == [[1, 'a'], [2, 'b'], [2, 'NEW']]
    assert _dup_key_daf().append([2, 'NEW']).lol == [[1, 'a'], [2, 'b'], [2, 'NEW']]
    assert _dup_key_daf().extend([{'id': 2, 'v': 'NEW'}]).lol == [[1, 'a'], [2, 'b'], [2, 'NEW']]
    other = Daf(lol=[[2, 'NEW']], cols=['id', 'v'])
    assert _dup_key_daf().concat(other).lol == [[1, 'a'], [2, 'b'], [2, 'NEW']]


def test_record_append_replaces_a_duplicate_key_by_default():
    assert _dup_key_daf().record_append({'id': 2, 'v': 'NEW'}).lol == [[1, 'a'], [2, 'NEW']]
    assert _dup_key_daf().record_append({'id': 2, 'v': 'NEW'}, respect_kd=False).lol == [[1, 'a'], [2, 'b'], [2, 'NEW']]


def test_respect_kd_true_replaces_a_duplicate_key():
    assert _dup_key_daf().append({'id': 2, 'v': 'NEW'}, respect_kd=True).lol == [[1, 'a'], [2, 'NEW']]
    assert _dup_key_daf().extend([{'id': 2, 'v': 'NEW'}], respect_kd=True).lol == [[1, 'a'], [2, 'NEW']]


# row sharing of the selectors, as the README table says

def _share_daf() -> Daf:
    return Daf(lol=[[1, 'a'], [2, 'b'], [3, 'c']], cols=['id', 'v'], keyfield='id')


@pytest.mark.parametrize('call, shared', [
    (lambda d: d.select_irows([0, 1]),                                  True),
    (lambda d: d[0:2],                                                  True),
    (lambda d: d.select_krows([1, 2]),                                  True),
    (lambda d: d.select_records_daf([1, 2]),                            True),
    (lambda d: d.remove_key(3),                                         True),
    (lambda d: d.select_where(lambda row: row['id'] < 3),               True),
    (lambda d: d.groupby_cols(['v']),                                   True),
    (lambda d: d.copy(),                                                True),
    (lambda d: d.select_by_dict({'id': 1}),                             True),
    (lambda d: d.groupby('v'),                                          False),
    (lambda d: d.select_cols(['id', 'v']),                              False),
    (lambda d: d.copy('editable'),                                      False),
    ])
def test_selector_row_sharing_matches_the_readme_table(call, shared):
    daf = _share_daf()
    result = call(daf)
    if isinstance(result, dict):
        result = next(iter(result.values()))
    result.lol[0][1] = 'CHANGED'
    assert (daf.lol[0][1] == 'CHANGED' or any(row[1] == 'CHANGED' for row in daf.lol)) is shared


# join fills a missing match with NULL, or with the fill value

def _join_pair():
    left = Daf(lol=[[1, 'x'], [2, 'y']], cols=['id', 'v'], keyfield='id')
    right = Daf(lol=[[2, 'p'], [3, 'q']], cols=['id', 'w'], keyfield='id')
    return left, right


def test_join_fills_a_missing_match_with_null_by_default():
    from daffodil.daf import NULL
    left, right = _join_pair()
    result = left.join(right, how='outer')
    assert result.lol == [[1, 'x', ''], [2, 'y', 'p'], [3, '', 'q']]
    assert result.lol[0][2] is NULL
    assert left.join(right, how='left').lol == [[1, 'x', ''], [2, 'y', 'p']]
    assert left.join(right, how='right').lol == [[2, 'y', 'p'], [3, '', 'q']]


def test_join_fill_none_gives_the_old_result():
    left, right = _join_pair()
    assert left.join(right, how='outer', fill=None).lol == [[1, 'x', None], [2, 'y', 'p'], [3, None, 'q']]


def test_join_fill_with_any_value():
    left, right = _join_pair()
    assert left.join(right, how='left', fill=0).lol == [[1, 'x', 0], [2, 'y', 'p']]


def test_join_inner_has_no_fill():
    left, right = _join_pair()
    assert left.join(right, fill=None).lol == [[2, 'y', 'p']]


def test_join_records_fill():
    translator = Daf(cols=['resolved_colname', 'source_name', 'source_colname', 'is_keyfield'],
                     lol=[['id', 'a', 'id', True], ['v', 'a', 'v', False], ['w', 'b', 'w', False]])
    assert Daf.join_records([{'id': 1, 'v': 'x'}, None], translator, ['a', 'b']) == {'id': 1, 'v': 'x', 'w': ''}
    assert Daf.join_records([{'id': 1, 'v': 'x'}, None], translator, ['a', 'b'], fill=None) == {'id': 1, 'v': 'x', 'w': None}
    assert Daf.join_records([{'id': 1}, {'id': 1}], translator, ['a', 'b'], fill='?') == {'id': 1, 'v': '?', 'w': '?'}


# insert_irow rejects a row that is not a list or a dict

@pytest.mark.parametrize('bad_row', ['zz', 5, None, (1, 'a'), {1, 2}])
def test_insert_irow_bad_row_raises_typeerror(bad_row):
    daf = Daf(lol=[[1, 'a']], cols=['id', 'v'])
    with pytest.raises(TypeError, match='insert_irow'):
        daf.insert_irow(0, bad_row)
    assert daf.lol == [[1, 'a']]


def test_insert_irow_list_and_dict_still_work():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'v'])
    daf.insert_irow(0, [0, 'z']).insert_irow(-1, {'id': 9})
    assert daf.lol == [[0, 'z'], [1, 'a'], [9, '']]


# row positions: None adds at the end, a negative position counts from the end

def _pos_daf() -> Daf:
    return Daf(lol=[[1, 'a'], [2, 'b'], [3, 'c']], cols=['id', 'v'], keyfield='id')


def test_assign_record_irow_none_adds_at_the_end():
    assert _pos_daf().assign_record_irow(None, {'id': 9, 'v': 'z'}).lol == [[1, 'a'], [2, 'b'], [3, 'c'], [9, 'z']]
    assert _pos_daf().assign_record_irow(record={'id': 9, 'v': 'z'}).lol == [[1, 'a'], [2, 'b'], [3, 'c'], [9, 'z']]


def test_assign_record_irow_negative_counts_from_the_end():
    assert _pos_daf().assign_record_irow(-1, {'id': 9, 'v': 'z'}).lol == [[1, 'a'], [2, 'b'], [9, 'z']]
    assert _pos_daf().assign_record_irow(-3, {'id': 9, 'v': 'z'}).lol == [[9, 'z'], [2, 'b'], [3, 'c']]


def test_assign_record_irow_before_the_first_row_raises():
    daf = _pos_daf()
    with pytest.raises(IndexError):
        daf.assign_record_irow(-4, {'id': 9, 'v': 'z'})
    assert len(daf) == 3


def test_assign_record_irow_beyond_the_end_adds_a_row():
    assert _pos_daf().assign_record_irow(3, {'id': 9, 'v': 'z'}).lol[-1] == [9, 'z']
    assert _pos_daf().assign_record_irow(99, {'id': 9, 'v': 'z'}).lol[-1] == [9, 'z']
    assert len(_pos_daf().assign_record_irow(99, {'id': 9, 'v': 'z'})) == 4


def test_assign_record_irow_on_a_daf_with_no_rows_adds_a_row():
    daf = Daf(cols=['id', 'v'])
    daf.assign_record_irow(-1, {'id': 1, 'v': 'a'})
    assert daf.lol == [[1, 'a']]


def test_setitem_negative_row_with_dict_replaces_like_a_list():
    by_dict = _pos_daf()
    by_dict[-1] = {'id': 9, 'v': 'z'}
    by_list = _pos_daf()
    by_list[-1] = [9, 'z']
    assert by_dict.lol == by_list.lol == [[1, 'a'], [2, 'b'], [9, 'z']]


def test_setitem_row_beyond_the_end_with_dict_still_adds_a_row():
    daf = _pos_daf()
    daf[5] = {'id': 9, 'v': 'z'}
    assert daf.lol[-1] == [9, 'z'] and len(daf) == 4


def test_update_record_irow_default_and_negative_reach_the_last_row():
    assert _pos_daf().update_record_irow(record={'v': 'Z'}).lol[-1] == [3, 'Z']
    assert _pos_daf().update_record_irow(-1, {'v': 'Z'}).lol[-1] == [3, 'Z']
    assert _pos_daf().update_record_irow(-3, {'v': 'Z'}).lol[0] == [1, 'Z']


def test_update_record_irow_out_of_range_raises_and_changes_nothing():
    for position in (3, 99, -4, -99):
        daf = _pos_daf()
        with pytest.raises(IndexError, match=f'position {position} is out of range for 3 rows'):
            daf.update_record_irow(position, {'v': 'Z'})
        assert daf.lol == _pos_daf().lol


def test_update_record_irow_with_nothing_to_update_changes_nothing():
    assert Daf().update_record_irow(0, {'v': 'Z'}).lol == []
    assert Daf(lol=[[1]]).update_record_irow(0, {'v': 'Z'}).lol == [[1]]        # no column names
    assert _pos_daf().update_record_irow(0, None).lol == _pos_daf().lol


def test_insert_irow_none_and_minus_one_add_at_the_end():
    assert _pos_daf().insert_irow(row=[9, 'z']).lol[-1] == [9, 'z']
    assert _pos_daf().insert_irow(None, [9, 'z']).lol[-1] == [9, 'z']
    assert _pos_daf().insert_irow(-1, [9, 'z']).lol[-1] == [9, 'z']
    assert _pos_daf().insert_irow(1, [9, 'z']).lol[1] == [9, 'z']


# select_cols keeps the order given, and raises for an unknown name

def _cols_daf() -> Daf:
    return Daf(lol=[[1, 'a', 10], [2, 'b', 20]], cols=['id', 'v', 'n'], keyfield='id',
               dtypes={'id': int, 'v': str, 'n': int})


def test_select_cols_keeps_the_order_given():
    result = _cols_daf().select_cols(['n', 'id'])
    assert result.columns() == ['n', 'id']
    assert result.lol == [[10, 1], [20, 2]]
    assert result.dtypes == {'n': int, 'id': int}
    assert result.keyfield == 'id'
    assert result.lol == _cols_daf().select_kcols(['n', 'id']).lol


def test_select_cols_unknown_name_raises_keyerror():
    with pytest.raises(KeyError):
        _cols_daf().select_cols(['n', 'nope'])
    with pytest.raises(KeyError):
        _cols_daf().select_cols(['nope'], exclude_cols=['nope'])


def test_select_cols_a_name_given_twice_is_used_once():
    assert _cols_daf().select_cols(['n', 'id', 'n']).columns() == ['n', 'id']


def test_select_cols_exclude_cols_keeps_table_order_and_ignores_unknown_names():
    assert _cols_daf().select_cols(exclude_cols=['v']).columns() == ['id', 'n']
    assert _cols_daf().select_cols(exclude_cols=['v', 'nope']).columns() == ['id', 'n']
    assert _cols_daf().select_cols(exclude_cols='v').columns() == ['id', 'n']


def test_select_cols_cols_and_exclude_cols_together():
    assert _cols_daf().select_cols(['n', 'v', 'id'], exclude_cols=['v']).columns() == ['n', 'id']


def test_select_cols_single_name_and_all_columns():
    assert _cols_daf().select_cols('n').columns() == ['n']
    assert _cols_daf().select_cols().columns() == ['id', 'v', 'n']


def test_select_cols_dropping_the_keyfield_clears_it():
    assert _cols_daf().select_cols(['n', 'v']).keyfield == ''


def test_select_cols_everything_excluded_gives_rows_with_no_columns():
    result = _cols_daf().select_cols(exclude_cols=['id', 'v', 'n'])
    assert result.lol == [[], []] and result.columns() == []


def test_select_cols_daf_with_no_columns():
    assert Daf().select_cols(['a']).lol == []
    assert Daf(lol=[[1, 2]]).select_cols().lol == [[]]


# append and extend with lists: lol= is several rows, la= is one row

def _two_col_daf() -> Daf:
    return Daf(lol=[[1, 'a']], cols=['id', 'v'])


def test_append_lol_adds_several_rows():
    assert _two_col_daf().append(lol=[[2, 'b'], [3, 'c']]).lol == [[1, 'a'], [2, 'b'], [3, 'c']]
    assert _two_col_daf().append(lol=[[2, 'b'], [3, 'c'], [4, 'd']]).lol == [[1, 'a'], [2, 'b'], [3, 'c'], [4, 'd']]


def test_append_la_adds_one_row_even_if_its_items_are_lists():
    result = _two_col_daf().append(la=[[2, 'b'], [3, 'c']])
    assert result.lol == [[1, 'a'], [[2, 'b'], [3, 'c']]]


def test_append_la_with_more_values_than_columns_raises():
    daf = _two_col_daf()
    with pytest.raises(ValueError, match='3 values for 2 columns'):
        daf.append(la=[[2, 'b'], [3, 'c'], [4, 'd']])
    assert daf.lol == [[1, 'a']]


def test_append_positional_list_longer_than_the_columns_raises():
    daf = _two_col_daf()
    with pytest.raises(ValueError, match='append'):
        daf.append([[2, 'b'], [3, 'c'], [4, 'd']])
    with pytest.raises(ValueError):
        daf.append([2, 'b', 'EXTRA'])
    assert daf.lol == [[1, 'a']]


def test_append_short_list_is_padded_with_null():
    assert _two_col_daf().append([2]).lol == [[1, 'a'], [2, '']]
    assert _two_col_daf().append(la=[2]).lol == [[1, 'a'], [2, '']]


def test_append_list_with_no_columns_is_added_as_it_is():
    daf = Daf()
    daf.append([1, 2, 3])
    daf.append([4, 5, 6, 7])
    assert daf.lol == [[1, 2, 3], [4, 5, 6, 7]]


def test_append_more_than_one_of_data_item_lol_la_raises():
    with pytest.raises(TypeError, match='only one'):
        _two_col_daf().append([2, 'b'], lol=[[3, 'c']])
    with pytest.raises(TypeError):
        _two_col_daf().append(lol=[[3, 'c']], la=[2, 'b'])


def test_append_unsupported_types_raise_typeerror():
    for bad in [(2, 'b'), 'zz', 5, {2, 3}, range(2)]:
        if not bad:
            continue
        with pytest.raises(TypeError, match='append'):
            _two_col_daf().append(bad)


def test_append_nothing_adds_nothing():
    daf = _two_col_daf()
    daf.append().append([]).append({}).append(lol=[]).append(la=[])
    assert daf.lol == [[1, 'a']]


def test_extend_lol_adds_rows_in_column_order():
    daf = _two_col_daf().extend(lol=[[2, 'b'], [3]])
    assert daf.lol == [[1, 'a'], [2, 'b'], [3, '']]


def test_extend_lol_too_long_row_raises_and_adds_nothing():
    daf = _two_col_daf()
    with pytest.raises(ValueError, match='extend'):
        daf.extend(lol=[[2, 'b'], [3, 'c', 'x']])
    assert daf.lol == [[1, 'a']]


def test_extend_lol_row_that_is_not_a_list_raises():
    with pytest.raises(TypeError, match='extend'):
        _two_col_daf().extend(lol=[[2, 'b'], {'id': 3}])


def test_extend_both_records_lod_and_lol_raises():
    with pytest.raises(TypeError, match='only one'):
        _two_col_daf().extend([{'id': 2}], lol=[[3, 'c']])


def test_extend_lol_respect_kd_replaces_the_row_with_the_same_key():
    daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'], keyfield='id')
    daf.extend(lol=[[2, 'NEW'], [3, 'c']], respect_kd=True)
    assert daf.lol == [[1, 'a'], [2, 'NEW'], [3, 'c']]
    assert Daf(lol=[[1, 'a']], cols=['id', 'v'], keyfield='id').extend(lol=[[1, 'dup']]).lol == [[1, 'a'], [1, 'dup']]


def test_extend_lol_keeps_the_key_index_valid():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'v'], keyfield='id')
    daf.keys()
    daf.extend(lol=[[2, 'b']])
    assert daf.keys() == [1, 2]
    assert daf.select_record(2) == {'id': 2, 'v': 'b'}


def test_extend_lol_on_an_empty_daf_with_no_columns():
    daf = Daf()
    daf.extend(lol=[[1, 2], [3, 4]])
    assert daf.lol == [[1, 2], [3, 4]]


def test_extend_list_of_dicts_still_works():
    assert _two_col_daf().extend([{'id': 2, 'v': 'b'}]).lol == [[1, 'a'], [2, 'b']]
    assert _two_col_daf().append([{'id': 2, 'v': 'b'}, {'id': 3, 'v': 'c'}]).lol == [[1, 'a'], [2, 'b'], [3, 'c']]


# a Daf with no column names: from_lot makes none, and dict views raise a clear error

from daffodil.daf import KeysDisabledError


def _unnamed_daf() -> Daf:
    return Daf.from_lot([(1, 'a', 10), (2, 'b', 20)])


def test_from_lot_without_cols_has_no_column_names():
    daf = _unnamed_daf()
    assert daf.columns() == []
    assert daf.lol == [[1, 'a', 10], [2, 'b', 20]]
    assert daf.set_cols().columns() == ['A', 'B', 'C']


def test_from_lot_tuples_of_different_lengths_raise():
    with pytest.raises(ValueError):
        Daf.from_lot([(1, 'a'), (2, 'b', 'c')])


@pytest.mark.parametrize('call', [
    lambda d: d.to_lod(),
    lambda d: list(d.iter_dict()),
    lambda d: list(d.iter_klist()),
    lambda d: [row for row in d],
    lambda d: d.to_cols_dol(),
    lambda d: d.select_where(lambda row: True),
    lambda d: d.select_by_dict({'A': 1}),
    ])
def test_dict_views_of_a_daf_with_no_names_raise_a_clear_error(call):
    with pytest.raises(KeysDisabledError, match='set_cols'):
        call(_unnamed_daf())


def test_list_views_of_a_daf_with_no_names_still_work():
    daf = _unnamed_daf()
    assert list(daf.iter_list()) == [[1, 'a', 10], [2, 'b', 20]]
    assert daf.to_md().splitlines()[0] == '| A | B | C  |'


def test_naming_the_columns_makes_the_dict_views_work():
    daf = _unnamed_daf().set_cols(['id', 'v', 'n'])
    assert daf.to_lod() == [{'id': 1, 'v': 'a', 'n': 10}, {'id': 2, 'v': 'b', 'n': 20}]
    assert list(daf.iter_dict())[0] == {'id': 1, 'v': 'a', 'n': 10}


def test_an_empty_daf_has_empty_dict_views():
    assert Daf().to_lod() == []
    assert list(Daf()) == []
    assert Daf().to_cols_dol() == {}


def test_row_getters_of_a_daf_with_no_names_raise():
    daf = Daf(lol=[[1, 'a']])
    for call in (lambda d: d.to_dict(), lambda d: d.irow(0), lambda d: d.to_klist(0), lambda d: d.iloc(-1)):
        with pytest.raises(KeysDisabledError, match='set_cols'):
            call(daf)
    assert daf.columns() == []             # nothing was made up, and the Daf was not changed.


def test_row_getters_of_an_empty_daf_are_empty():
    assert Daf().to_dict() == {}
    assert Daf().iloc(0) == {}
    assert Daf().irow(0) == {}


def test_to_md_of_a_daf_with_no_names_writes_a_spreadsheet_header():
    assert Daf(lol=[[1, 'a']]).to_md() == '| A | B |\n| -: | -: |\n| 1 | a |\n'
    assert Daf(lol=[[1, 'a']], cols=['id', 'v']).to_md().splitlines()[0] == '| id | v |'


# a blank column name always becomes Unnamed plus its position

def test_constructor_renames_a_single_blank_name():
    assert Daf(lol=[[1, 2, 3]], cols=['B', '', 'C']).columns() == ['B', 'Unnamed1', 'C']


def test_constructor_renames_several_blank_names():
    assert Daf(lol=[[1, 2, 3, 4]], cols=['id', '', 'v', '']).columns() == ['id', 'Unnamed1', 'v', 'Unnamed3']


def test_constructor_leaves_names_without_blanks_alone():
    assert Daf(lol=[[1, 2]], cols=['a', 'b']).columns() == ['a', 'b']
    assert Daf(lol=[[1, 2]], cols=['a', 'a']).columns() == ['a', 'a_1']


def test_set_cols_blank_names_use_the_short_prefix_col_and_the_prefix_can_be_changed():
    assert Daf(lol=[[1, 2, 3, 4]]).set_cols(['id', '', 'v', '']).columns() == ['id', 'col1', 'v', 'col3']
    assert Daf(lol=[[1, 2, 3]]).set_cols(['a', '', 'c'], unnamed_prefix='Unnamed').columns() == ['a', 'Unnamed1', 'c']


def test_from_md_blank_header_cells_use_unnamed():
    text = "| id |   | v |\n| -: | -: | -: |\n| 1 | 2 | 3 |\n"
    assert Daf.from_md(text).columns() == ['id', 'Unnamed1', 'v']


# values that are used as keys of a dict keep their value, such as a blank

def test_value_counts_daf_keeps_a_blank_value():
    daf = Daf(cols=['a'], lol=[['x'], [''], ['x'], ['y']])
    assert daf.value_counts_daf('a').lol == [['x', 2], ['', 1], ['y', 1]]


def test_from_lod_to_cols_keeps_a_blank_key():
    daf = Daf.from_lod_to_cols([{'': 1, 'B': 2}, {'': 4, 'B': 5}], cols=['Feature', 'T1', 'T2'])
    assert daf.lol == [['', 1, 4], ['B', 2, 5]]
    assert daf.columns() == ['Feature', 'T1', 'T2']


def test_from_lod_to_cols_default_names():
    daf = Daf.from_lod_to_cols([{'A': 1}, {'A': 2}])
    assert daf.columns() == ['key', 'A', 'B']
    assert daf.lol == [['A', 1, 2]]


def test_dict_to_md_keeps_a_blank_key():
    assert Daf.dict_to_md({'': 1, 'b': 2}).splitlines()[2].split('|')[1].strip() == ''


# sorting columns that mix values: a clear error, and as_str=True

def _mixed_daf() -> Daf:
    return Daf(lol=[[3, 'b'], [None, 'a'], [1, 'c'], ['', 'd'], [10, 'e'], [2, 'f']], cols=['n', 't'])


def test_sort_by_colname_mixed_column_raises_a_clear_error():
    daf = _mixed_daf()
    with pytest.raises(TypeError, match="sort_by_colname\\(\\): column 'n'.*as_str=True"):
        daf.sort_by_colname('n')
    assert [row[0] for row in daf.lol] == [3, None, 1, '', 10, 2]       # not changed


def test_sort_by_colname_as_str_sorts_by_text_and_none_is_empty():
    daf = _mixed_daf().sort_by_colname('n', as_str=True)
    assert [row[0] for row in daf.lol] == [None, '', 1, 10, 2, 3]


def test_sort_by_colname_as_str_with_length_priority_sorts_whole_numbers_numerically():
    daf = _mixed_daf().sort_by_colname('n', as_str=True, length_priority=True)
    assert [row[0] for row in daf.lol] == [None, '', 1, 2, 3, 10]


def test_sort_by_colname_as_str_reverse():
    daf = _mixed_daf().sort_by_colname('n', as_str=True, length_priority=True, reverse=True)
    assert [row[0] for row in daf.lol] == [10, 3, 2, 1, None, '']


def test_sort_by_colname_length_priority_on_real_numbers_raises_a_clear_error():
    daf = Daf(lol=[[3], [10], [2]], cols=['n'])
    with pytest.raises(TypeError, match='as_str=True'):
        daf.sort_by_colname('n', length_priority=True)
    assert daf.sort_by_colname('n', as_str=True, length_priority=True).col('n') == [2, 3, 10]


def test_sort_by_colname_plain_cases_are_unchanged():
    assert Daf(lol=[[2], [1.5], [3]], cols=['n']).sort_by_colname('n').col('n') == [1.5, 2, 3]
    assert Daf(lol=[['b'], [''], ['a']], cols=['t']).sort_by_colname('t').col('t') == ['', 'a', 'b']
    assert Daf(lol=[['10'], ['9']], cols=['t']).sort_by_colname('t', length_priority=True).col('t') == ['9', '10']


def test_sort_by_colnames_mixed_column_raises_and_as_str_works():
    daf = _mixed_daf()
    with pytest.raises(TypeError, match="sort_by_colnames\\(\\).*as_str=True"):
        daf.sort_by_colnames(['n', 't'])
    daf.sort_by_colnames(['n', 't'], as_str=True, length_priority=True)
    assert [row[0] for row in daf.lol] == [None, '', 1, 2, 3, 10]


def test_sort_by_colnames_as_str_uses_the_second_column_for_ties():
    daf = Daf(lol=[[1, 'b'], [None, 'z'], [1, 'a'], [None, 'y']], cols=['n', 't'])
    daf.sort_by_colnames(['n', 't'], as_str=True)
    assert daf.lol == [[None, 'y'], [None, 'z'], [1, 'a'], [1, 'b']]


# apply_formulas invalidates the key index even when a formula fails

def test_apply_formulas_error_leaves_the_key_index_matching_the_changed_cells():
    daf = Daf(cols=['id', 'n'], lol=[[1, 10], [2, 20], [3, 30]], keyfield='id')
    daf.keys()
    formulas = Daf(cols=['id', 'n'], lol=[['$d[0,0]+100', ''], ['', ''], ['', 'nope+1']])
    with pytest.raises(NameError):
        daf.apply_formulas(formulas)
    assert daf.lol == [[101, 10], [2, 20], [3, 30]]
    assert daf.keys() == [101, 2, 3]
    assert daf.select_record(101) == {'id': 101, 'n': 10}
    assert daf.select_record(1) == {}


def test_apply_formulas_circular_error_also_invalidates_the_key_index():
    daf = Daf(cols=['id', 'n'], lol=[[1, 2], [3, 4]], keyfield='id')
    daf.keys()
    formulas = Daf(cols=['id', 'n'], lol=[['$d[1,0]+1', ''], ['$d[0,0]+1', '']])
    with pytest.raises(RuntimeError):
        daf.apply_formulas(formulas)
    assert set(daf.keys()) == {row[0] for row in daf.lol}


def test_select_irows_inverse_of_nothing_shares_the_rows():
    d = _rows_daf()
    for result in (d.select_irows([], inverse=True), d.select_irows([], invert=True)):
        assert result is not d
        assert result.lol is not d.lol
        assert result.lol == d.lol
        assert result.lol[0] is d.lol[0]
        assert result.keyfield == 'id'
        assert result.hd == d.hd


def test_select_irows_of_nothing_is_empty():
    result = _rows_daf().select_irows([])
    assert result.lol == []
    assert result.keyfield == 'id'
