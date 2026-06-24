# test_daf_transform.py
#
# Tests for sorting, joining, and transforming operations: calc_cols, sort_by_colname,
# sort_by_colnames, split_where, remove_dups, apply_in_place, annotate_daf, update_by_keylist,
# replace_in_columns, join, join_records.
#
# This batch surfaced 4 real bugs, now fixed:
#   - calc_cols(include_cols='x') / calc_cols(exclude_cols='x') (string form) silently did
#     nothing -- the filter line was only reachable when include_cols/exclude_cols was a list.
#   - split_where(..., indirect_col=...) raised UnboundLocalError -- true_daf/false_daf were
#     only constructed in the non-indirect_col branch.
#   - replace_in_columns(cols=None, ...) raised TypeError (`self.keyfield in cols` where
#     cols was None); separately, the keyfield-invalidation check compared a column name
#     string against a list of column indices, which could never match.
#   - replace_in_columns() silently wrote the _MISSING sentinel object into the data when
#     `replacement` wasn't explicitly passed, since nothing ever checked for it. Now raises
#     ValueError instead.

import pytest

from daffodil.daf import Daf, KeysDisabledError


# =====================================================================
# calc_cols
# =====================================================================

def test_calc_cols_no_filters_returns_all():
    daf = Daf(lol=[[1, 'a', True]], cols=['id', 'name', 'flag'])
    assert list(daf.calc_cols()) == ['id', 'name', 'flag']


def test_calc_cols_include_cols_string():
    daf = Daf(lol=[[1, 'a', True]], cols=['id', 'name', 'flag'])
    assert list(daf.calc_cols(include_cols='id')) == ['id']


def test_calc_cols_include_cols_list():
    daf = Daf(lol=[[1, 'a', True]], cols=['id', 'name', 'flag'])
    assert list(daf.calc_cols(include_cols=['id', 'flag'])) == ['id', 'flag']


def test_calc_cols_exclude_cols_string():
    daf = Daf(lol=[[1, 'a', True]], cols=['id', 'name', 'flag'])
    assert list(daf.calc_cols(exclude_cols='id')) == ['name', 'flag']


def test_calc_cols_exclude_cols_list():
    daf = Daf(lol=[[1, 'a', True]], cols=['id', 'name', 'flag'])
    assert list(daf.calc_cols(exclude_cols=['id', 'flag'])) == ['name']


def test_calc_cols_include_types_list():
    daf = Daf(lol=[[1, 'a', True]], cols=['id', 'name', 'flag'],
              dtypes={'id': int, 'name': str, 'flag': bool})
    assert list(daf.calc_cols(include_types=[int, bool])) == ['id', 'flag']


def test_calc_cols_include_types_single():
    daf = Daf(lol=[[1, 'a', True]], cols=['id', 'name', 'flag'],
              dtypes={'id': int, 'name': str, 'flag': bool})
    assert list(daf.calc_cols(include_types=int)) == ['id']


def test_calc_cols_exclude_types():
    daf = Daf(lol=[[1, 'a', True]], cols=['id', 'name', 'flag'],
              dtypes={'id': int, 'name': str, 'flag': bool})
    assert list(daf.calc_cols(exclude_types=[str])) == ['id', 'flag']


def test_calc_cols_no_cols_at_all_returns_empty():
    daf = Daf()
    assert daf.calc_cols() == []


def test_calc_cols_combined_include_and_exclude():
    daf = Daf(lol=[[1, 'a', True]], cols=['id', 'name', 'flag'])
    result = daf.calc_cols(include_cols=['id', 'name', 'flag'], exclude_cols=['name'])
    assert list(result) == ['id', 'flag']


# =====================================================================
# sort_by_colname / sort_by_colnames
# =====================================================================

def test_sort_by_colname_basic():
    daf = Daf(lol=[['10'], ['99'], ['8'], ['100'], ['0']], cols=['n'])
    daf.sort_by_colname('n')
    assert daf.lol == [['0'], ['10'], ['100'], ['8'], ['99']]


def test_sort_by_colname_length_priority():
    daf = Daf(lol=[['10'], ['99'], ['8'], ['100'], ['0']], cols=['n'])
    daf.sort_by_colname('n', length_priority=True)
    assert daf.lol == [['0'], ['8'], ['10'], ['99'], ['100']]


def test_sort_by_colname_reverse():
    daf = Daf(lol=[['10'], ['99'], ['8']], cols=['n'])
    daf.sort_by_colname('n', reverse=True)
    assert daf.lol == [['99'], ['8'], ['10']]


def test_sort_by_colname_empty_noop():
    daf = Daf()
    assert daf.sort_by_colname('n').lol == []


def test_sort_by_colname_single_row_noop():
    daf = Daf(lol=[[5]], cols=['n'])
    assert daf.sort_by_colname('n').lol == [[5]]


def test_sort_by_colnames_multi_column():
    daf = Daf(lol=[[1, 'b'], [1, 'a'], [0, 'z']], cols=['x', 'y'])
    daf.sort_by_colnames(['x', 'y'])
    assert daf.lol == [[0, 'z'], [1, 'a'], [1, 'b']]


# =====================================================================
# split_where
# =====================================================================

def test_split_where_basic():
    daf = Daf(lol=[[1], [10]], cols=['n'])
    true_daf, false_daf = daf.split_where(lambda row: row['n'] > 5)
    assert true_daf.lol == [[10]]
    assert false_daf.lol == [[1]]


def test_split_where_indirect_col():
    # this is the bug we found: previously raised UnboundLocalError
    daf = Daf(lol=[[1, {'x': 10}], [2, {'x': 1}]], cols=['id', 'meta'])
    true_daf, false_daf = daf.split_where(lambda row: row['x'] > 5, indirect_col='meta')
    assert true_daf.lol == [[1, {'x': 10}]]
    assert false_daf.lol == [[2, {'x': 1}]]


def test_split_where_preserves_keyfield_and_dtypes():
    daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'name'], keyfield='id', dtypes={'id': int, 'name': str})
    true_daf, false_daf = daf.split_where(lambda row: row['id'] > 1)
    assert true_daf.keyfield == 'id'
    assert true_daf.dtypes == {'id': int, 'name': str}


# =====================================================================
# remove_dups
# =====================================================================

def test_remove_dups_with_duplicates():
    daf = Daf(lol=[[1, 'a'], [2, 'b'], [1, 'c']], cols=['id', 'name'])
    unique_daf, dups_daf = daf.remove_dups('id')
    assert sorted(unique_daf.lol) == sorted([[1, 'c'], [2, 'b']])
    assert dups_daf.lol == [[1, 'a']]


def test_remove_dups_no_duplicates_returns_self():
    daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'name'])
    unique_daf, dups_daf = daf.remove_dups('id')
    assert unique_daf is daf
    assert dups_daf.lol == []


# =====================================================================
# apply_in_place
# =====================================================================

def test_apply_in_place_by_row():
    daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'name'])

    def upper_name(row):
        row['name'] = row['name'].upper()
        return row

    daf.apply_in_place(func=upper_name, by='row')
    assert daf.lol == [[1, 'A'], [2, 'B']]


def test_apply_in_place_by_row_with_rowkeys_filter():
    daf = Daf(lol=[[1, 'a'], [2, 'b'], [3, 'c']], cols=['id', 'name'], keyfield='id')

    def upper_name(row):
        row['name'] = row['name'].upper()
        return row

    daf.apply_in_place(func=upper_name, by='row', rowkeys=[1, 3])
    assert daf.lol == [[1, 'A'], [2, 'b'], [3, 'C']]


def test_apply_in_place_by_row_klist():
    daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'name'])

    def upper_klist(klist):
        klist['name'] = klist['name'].upper()

    daf.apply_in_place(func=upper_klist, by='row_klist')
    assert daf.lol == [[1, 'A'], [2, 'B']]


def test_apply_in_place_large_rowkeys_uses_dict_path():
    # exercises the len(rowkeys) >= 30 -> dict.fromkeys(rowkeys) branch
    daf = Daf(lol=[[i, 'x'] for i in range(40)], cols=['id', 'name'], keyfield='id')

    def upper_name(row):
        row['name'] = row['name'].upper()
        return row

    daf.apply_in_place(func=upper_name, by='row', rowkeys=list(range(35)))
    assert daf.lol[0] == [0, 'X']
    assert daf.lol[-1] == [39, 'x']


# =====================================================================
# annotate_daf
# =====================================================================

def test_annotate_daf_basic():
    daf = Daf(lol=[[1, ''], [2, '']], cols=['id', 'name'], keyfield='id')
    other = Daf(lol=[[1, 'Alice'], [2, 'Bob']], cols=['id', 'fullname'], keyfield='id')
    result = daf.annotate_daf(other, {'name': 'fullname'})
    assert result.lol == [[1, 'Alice'], [2, 'Bob']]


def test_annotate_daf_self_no_keyfield_raises():
    daf = Daf(lol=[[1, '']], cols=['id', 'name'])
    other = Daf(lol=[[1, 'Alice']], cols=['id', 'fullname'], keyfield='id')
    with pytest.raises(KeyError):
        daf.annotate_daf(other, {'name': 'fullname'})


def test_annotate_daf_other_no_keyfield_raises():
    daf = Daf(lol=[[1, '']], cols=['id', 'name'], keyfield='id')
    other = Daf(lol=[[1, 'Alice']], cols=['id', 'fullname'])
    with pytest.raises(KeyError):
        daf.annotate_daf(other, {'name': 'fullname'})


# =====================================================================
# update_by_keylist
# =====================================================================

def test_update_by_keylist_basic():
    daf = Daf(lol=[[1, 'a', 10], [2, 'b', 20]], cols=['id', 'name', 'val'], keyfield='id')
    daf.update_by_keylist(keylist=[1], record={'val': 99})
    assert daf.lol == [[1, 'a', 99], [2, 'b', 20]]


def test_update_by_keylist_none_keylist_noop():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'], keyfield='id')
    result = daf.update_by_keylist(keylist=None, record={'name': 'x'})
    assert result is None
    assert daf.lol == [[1, 'a']]


def test_update_by_keylist_none_record_noop():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'], keyfield='id')
    result = daf.update_by_keylist(keylist=[1], record=None)
    assert result is None
    assert daf.lol == [[1, 'a']]


# =====================================================================
# replace_in_columns
# =====================================================================

def test_replace_in_columns_cols_none_processes_all():
    # this is the bug we found: previously raised TypeError when cols=None
    daf = Daf(lol=[[1, ''], [2, 'b']], cols=['id', 'name'], keyfield='id')
    result = daf.replace_in_columns(cols=None, find_values=[''], replacement='N/A')
    assert result.lol == [[1, 'N/A'], [2, 'b']]


def test_replace_in_columns_missing_replacement_raises():
    # this is the second bug we found: previously silently wrote the _MISSING sentinel
    daf = Daf(lol=[[1, '']], cols=['id', 'name'])
    with pytest.raises(ValueError):
        daf.replace_in_columns(cols=['name'], find_values=[''])


def test_replace_in_columns_find_values_none_is_noop():
    daf = Daf(lol=[[1, '']], cols=['id', 'name'])
    result = daf.replace_in_columns(cols=['name'], find_values=None)
    assert result.lol == [[1, '']]


def test_replace_in_columns_by_string_colname():
    daf = Daf(lol=[[1, ''], [2, 'b']], cols=['id', 'name'])
    result = daf.replace_in_columns(cols=['name'], find_values=[''], replacement='N/A')
    assert result.lol == [[1, 'N/A'], [2, 'b']]


def test_replace_in_columns_by_int_colidx():
    daf = Daf(lol=[[1, ''], [2, 'b']], cols=['id', 'name'])
    result = daf.replace_in_columns(cols=[1], find_values=[''], replacement='N/A')
    assert result.lol == [[1, 'N/A'], [2, 'b']]


def test_replace_in_columns_unknown_string_col_raises_keyerror():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    with pytest.raises(KeyError):
        daf.replace_in_columns(cols=['missing'], find_values=[''], replacement='x')


def test_replace_in_columns_invalid_col_type_raises_typeerror():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    with pytest.raises(TypeError):
        daf.replace_in_columns(cols=[1.5], find_values=[''], replacement='x')


def test_replace_in_columns_invalidates_kd_when_keyfield_touched():
    daf = Daf(lol=[[1, ''], [2, 'b']], cols=['id', 'name'], keyfield='id')
    daf._rebuild_kd_if_invalidated()
    daf.replace_in_columns(cols=['id'], find_values=[1], replacement=99)
    assert daf._kd == {}  # invalidated


def test_replace_in_columns_does_not_invalidate_kd_when_keyfield_untouched():
    daf = Daf(lol=[[1, ''], [2, 'b']], cols=['id', 'name'], keyfield='id')
    daf._rebuild_kd_if_invalidated()
    kd_before = dict(daf._kd)
    daf.replace_in_columns(cols=['name'], find_values=[''], replacement='N/A')
    assert daf._kd == kd_before


# =====================================================================
# join / join_records
# =====================================================================

def _make_join_dafs():
    left = Daf(lol=[[1, 'a'], [2, 'b'], [3, 'c']], cols=['id', 'name'], keyfield='id', name='left')
    right = Daf(lol=[[1, 100], [2, 200], [4, 400]], cols=['id', 'val'], keyfield='id', name='right')
    return left, right


def test_join_inner():
    left, right = _make_join_dafs()
    result = left.join(right, how='inner')
    assert result.lol == [[1, 'a', 100], [2, 'b', 200]]
    assert list(result.hd.keys()) == ['id', 'name', 'val']


def test_join_left():
    left, right = _make_join_dafs()
    result = left.join(right, how='left')
    assert result.lol == [[1, 'a', 100], [2, 'b', 200], [3, 'c', None]]


def test_join_right():
    left, right = _make_join_dafs()
    result = left.join(right, how='right')
    assert result.lol == [[1, 'a', 100], [2, 'b', 200], [4, None, 400]]


def test_join_outer():
    left, right = _make_join_dafs()
    result = left.join(right, how='outer')
    assert result.lol == [[1, 'a', 100], [2, 'b', 200], [3, 'c', None], [4, None, 400]]


def test_join_invalid_how_raises():
    left, right = _make_join_dafs()
    with pytest.raises(ValueError):
        left.join(right, how='bogus')


def test_join_missing_keyfield_raises():
    left, right = _make_join_dafs()
    no_key = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    with pytest.raises(KeysDisabledError):
        no_key.join(right)


def test_join_conflicting_column_default_suffixes():
    left = Daf(lol=[[1, 'a', 5]], cols=['id', 'name', 'val'], keyfield='id', name='left')
    right = Daf(lol=[[1, 100]], cols=['id', 'val'], keyfield='id', name='right')
    result = left.join(right, how='inner')
    assert result.lol == [[1, 'a', 5, 100]]
    assert list(result.hd.keys()) == ['id', 'name', 'val_left', 'val_right']


def test_join_shared_fields_avoids_suffixing():
    left = Daf(lol=[[1, 'a', 5]], cols=['id', 'name', 'val'], keyfield='id', name='left')
    right = Daf(lol=[[1, 100]], cols=['id', 'val'], keyfield='id', name='right')
    result = left.join(right, how='inner', shared_fields=['val'])
    assert result.lol == [[1, 'a', 5]]
    assert list(result.hd.keys()) == ['id', 'name', 'val']
