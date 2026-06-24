# test_daf_reduction.py
#
# Tests for the reduction operations cluster: reduce(), sum_da(), count_values_da(), groupby(),
# groupby_cols(), groupby_reduce(), multi_groupby(), multi_groupby_reduce(), reduce_dodaf_to_daf().
#
# This batch surfaced a real bug: reduce(by='row')'s default accumulator initialization
# (`dict.fromkeys(cols_iter, 0)`) is sum_da-specific (0 is the correct additive identity for
# summing), but is silently and incorrectly applied to any reduction function, including
# count_values_da() which needs a dict-based accumulator. Worse, even explicitly passing
# initial_da={} as a workaround didn't help, since `if initial_da:` treats an empty dict as
# falsy. Both are now fixed (`if initial_da is not None:`).

import pytest

from daffodil.daf import Daf


# =====================================================================
# reduce()
# =====================================================================

def test_reduce_by_row_sum():
    daf = Daf(lol=[[1, 2], [3, 4], [5, 6]], cols=['a', 'b'])
    result = daf.reduce(func=Daf.sum_da, by='row')
    assert result == {'a': 9, 'b': 12}


def test_reduce_by_row_with_cols_subset():
    daf = Daf(lol=[[1, 2], [3, 4]], cols=['a', 'b'])
    result = daf.reduce(func=Daf.sum_da, by='row', cols=['a'])
    assert result == {'a': 4, 'b': ''}


def test_reduce_by_col():
    daf = Daf(lol=[[1, 2], [3, 4], [5, 6]], cols=['a', 'b'])

    def col_sum(col_la, accum_la):
        accum_la.append(sum(col_la))
        return accum_la

    result = daf.reduce(func=col_sum, by='col')
    assert result == [9, 12]


def test_reduce_by_table():
    daf = Daf(lol=[[1, 2], [3, 4]], cols=['a', 'b'])

    def row_count(daf_obj, cols):
        return daf_obj.num_rows()

    result = daf.reduce(func=row_count, by='table')
    assert result == 2


def test_reduce_by_sparse_row():
    daf = Daf(lol=[[1, {'x': 10, 'y': 20}], [2, {'x': 5}]], cols=['id', 'meta'])
    result = daf.reduce(func=Daf.sum_da, by='sparse_row', indirect_col='meta')
    assert result == {'x': 15, 'y': 20}


def test_reduce_sparse_row_requires_indirect_col():
    daf = Daf(lol=[[1, {'x': 10}]], cols=['id', 'meta'])
    with pytest.raises(ValueError):
        daf.reduce(func=Daf.sum_da, by='sparse_row')


def test_reduce_invalid_by_raises():
    daf = Daf(lol=[[1, 2]], cols=['a', 'b'])
    with pytest.raises(NotImplementedError):
        daf.reduce(func=Daf.sum_da, by='bogus')


def test_reduce_empty_daf_by_row():
    daf = Daf()
    result = daf.reduce(func=Daf.sum_da, by='row')
    assert result == {}


def test_reduce_initial_da_none_vs_explicit_empty_dict():
    # this is the bug we found: an explicitly-passed empty dict must actually be used as the
    # initial accumulator (needed by count_values_da and similar non-additive reductions),
    # not silently replaced by the sum_da-specific dict.fromkeys(cols, 0) default.
    daf = Daf(lol=[[{'M': 2, 'F': 1}], [{'M': 1}]], cols=['counts'])
    result = daf.reduce(func=Daf.count_values_da, by='row', cols=['counts'], initial_da={})
    assert result == {'counts': {'M': 3, 'F': 1}}


# =====================================================================
# sum_da()
# =====================================================================

def test_sum_da_no_cols_sums_all():
    result = Daf.sum_da({'a': 1, 'b': 2}, {'a': 0, 'b': 0}, cols=None)
    assert result == {'a': 1, 'b': 2}


def test_sum_da_no_cols_skips_non_numeric_strings():
    # cols=None path does not attempt any type coercion -- non-numeric values are skipped,
    # and even numeric-looking strings are skipped unless astype is used (see below).
    result = Daf.sum_da({'a': '1', 'b': 'x'}, {'a': 0, 'b': 0}, cols=None)
    assert result == {'a': 0, 'b': 0}


def test_sum_da_explicit_cols():
    result = Daf.sum_da({'a': 1, 'b': 2}, {'a': 0}, cols=['a', 'b'])
    assert result == {'a': 1, 'b': 2}


def test_sum_da_astype_int_coerces_strings():
    result = Daf.sum_da({'a': '5'}, {'a': 0}, cols=['a'], astype=int)
    assert result == {'a': 5}


def test_sum_da_astype_int_skips_unconvertible():
    result = Daf.sum_da({'a': 'x'}, {'a': 0}, cols=['a'], astype=int)
    assert result == {'a': 0}


def test_sum_da_accumulates_across_calls():
    accum = {'a': 0}
    accum = Daf.sum_da({'a': 5}, accum, cols=['a'])
    accum = Daf.sum_da({'a': 3}, accum, cols=['a'])
    assert accum == {'a': 8}


# =====================================================================
# count_values_da()
# =====================================================================

def test_count_values_da_basic():
    result = {}
    for row in [{'gender': 'M'}, {'gender': 'F'}, {'gender': 'M'}]:
        result = Daf.count_values_da(row, result, cols=['gender'])
    assert result == {'gender': {'M': 2, 'F': 1}}


def test_count_values_da_omit_nulls():
    result = {}
    for row in [{'gender': 'M'}, {'gender': ''}, {'gender': 'M'}]:
        result = Daf.count_values_da(row, result, cols=['gender'], omit_nulls=True)
    assert result == {'gender': {'M': 2}}


def test_count_values_da_dict_merge():
    # second row's dict-valued column merges into the first via sum_da
    result = {}
    result = Daf.count_values_da({'counts': {'M': 2, 'F': 1}}, result, cols=['counts'])
    result = Daf.count_values_da({'counts': {'M': 1}}, result, cols=['counts'])
    assert result == {'counts': {'M': 3, 'F': 1}}


def test_count_values_da_via_reduce_end_to_end():
    daf = Daf(lol=[['M'], ['F'], ['M']], cols=['gender'])
    result = daf.reduce(func=Daf.count_values_da, by='row', cols=['gender'], initial_da={})
    assert result == {'gender': {'M': 2, 'F': 1}}


# =====================================================================
# groupby / groupby_cols / groupby_reduce
# =====================================================================

def test_groupby_single_col():
    daf = Daf(lol=[['M', 1, 10], ['F', 2, 20], ['M', 3, 30]], cols=['gender', 'x', 'y'])
    result = daf.groupby('gender')
    assert result['M'].lol == [['M', 1, 10], ['M', 3, 30]]
    assert result['F'].lol == [['F', 2, 20]]


def test_groupby_omit_nulls():
    daf = Daf(lol=[['M', 1], ['', 2]], cols=['gender', 'x'])
    result = daf.groupby('gender', omit_nulls=True)
    assert list(result.keys()) == ['M']


def test_groupby_list_of_one_col_delegates_correctly():
    daf = Daf(lol=[['M', 1], ['F', 2]], cols=['gender', 'x'])
    result = daf.groupby(['gender'])
    assert set(result.keys()) == {('M',), ('F',)}


def test_groupby_cols_multi_column():
    daf = Daf(lol=[['M', 'north', 1], ['F', 'south', 2], ['M', 'north', 3]],
              cols=['gender', 'region', 'x'])
    result = daf.groupby(colnames=['gender', 'region'])
    assert set(result.keys()) == {('M', 'north'), ('F', 'south')}
    assert result[('M', 'north')].lol == [['M', 'north', 1], ['M', 'north', 3]]


def test_groupby_reduce_basic():
    daf = Daf(lol=[['M', 1, 10], ['F', 2, 20], ['M', 3, 30]], cols=['gender', 'x', 'y'])
    result = daf.groupby_reduce(colname='gender', func=Daf.sum_da, reduce_cols=['x', 'y'])
    assert result.lol == [['M', 4, 40], ['F', 2, 20]]
    assert list(result.hd.keys()) == ['gender', 'x', 'y']
    assert result.keyfield == 'gender'


# =====================================================================
# multi_groupby / multi_groupby_reduce
# =====================================================================

def test_multi_groupby_basic():
    daf = Daf(lol=[['M', 'north', 1], ['F', 'south', 2], ['M', 'north', 3]],
              cols=['gender', 'region', 'x'])
    result = daf.multi_groupby(['gender', 'region'])
    assert set(result.keys()) == {'gender', 'region'}
    assert set(result['gender'].keys()) == {'M', 'F'}
    assert set(result['region'].keys()) == {'north', 'south'}


def test_multi_groupby_single_string_colname():
    daf = Daf(lol=[['M', 1], ['F', 2]], cols=['gender', 'x'])
    result = daf.multi_groupby('gender')
    assert set(result.keys()) == {'gender'}


def test_multi_groupby_omit_nulls():
    daf = Daf(lol=[['M', 1], ['', 2]], cols=['gender', 'x'])
    result = daf.multi_groupby(['gender'], omit_nulls=True)
    assert list(result['gender'].keys()) == ['M']


def test_multi_groupby_reduce_basic():
    daf = Daf(lol=[['M', 1, 10], ['F', 2, 20], ['M', 3, 30]], cols=['gender', 'x', 'y'])
    result = daf.multi_groupby_reduce(['gender'], func=Daf.sum_da, reduce_cols=['x', 'y'])
    assert result['gender'].lol == [['M', 4, 40], ['F', 2, 20]]
