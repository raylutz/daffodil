# test_daf_group_scope.py
# Tests for column scoping in groupby(), groupby_cols(), multi_groupby() and the reduce methods.

import pytest

from daffodil.daf import Daf, NULL


def make_daf(keyfield: str = '') -> Daf:
    return Daf(
        lol=[['a', 1, 10, 100], ['b', 2, 20, 200], ['a', 3, 30, 300]],
        cols=['g', 'x', 'y', 'z'],
        keyfield=keyfield,
        dtypes={'g': str, 'x': int, 'y': int, 'z': int},
    )


# groupby

def test_groupby_cols_projects_in_given_order():
    groups = make_daf().groupby('g', cols=['z', 'y'])
    assert groups['a'].columns() == ['z', 'y']
    assert groups['a'].lol == [[100, 10], [300, 30]]
    assert groups['b'].lol == [[200, 20]]


def test_groupby_cols_filters_dtypes():
    groups = make_daf().groupby('g', cols=['y'])
    assert groups['a'].dtypes == {'y': int}


def test_groupby_cols_keyfield_kept_only_if_column_kept():
    d = make_daf(keyfield='x')
    assert d.groupby('g', cols=['x', 'y'])['a'].keyfield == 'x'
    assert d.groupby('g', cols=['y'])['a'].keyfield == ''


def test_groupby_cols_unknown_name_raises():
    with pytest.raises(KeyError):
        make_daf().groupby('g', cols=['nope'])


def test_groupby_cols_does_not_change_original():
    d = make_daf()
    d.groupby('g', cols=['y'])['a'].lol[0][0] = 999
    assert d.lol[0] == ['a', 1, 10, 100]


def test_groupby_all_columns_unchanged():
    groups = make_daf().groupby('g')
    assert groups['a'].lol == [['a', 1, 10, 100], ['a', 3, 30, 300]]


# groupby_cols

def test_groupby_cols_with_cols():
    groups = make_daf().groupby_cols(['g'], cols=['y'])
    assert list(groups) == [('a',), ('b',)]
    assert groups[('a',)].lol == [[10], [30]]


def test_groupby_cols_without_cols_shares_rows():
    d = make_daf()
    groups = d.groupby_cols(['g'])
    assert groups[('b',)].lol[0] is d.lol[1]


def test_groupby_cols_with_cols_does_not_share_rows():
    d = make_daf()
    groups = d.groupby_cols(['g'], cols=['y'])
    assert groups[('b',)].lol[0] is not d.lol[1]


# multi_groupby

def test_multi_groupby_colnames_scopes_columns():
    result = make_daf().multi_groupby(['g'], colnames=['y'])
    assert result['g']['a'].columns() == ['y']
    assert result['g']['a'].lol == [[10], [30]]


def test_multi_groupby_empty_daf():
    assert Daf().multi_groupby(['g']) == {}


# reduce methods give the same result with and without scoping

def sum_da_wide(row_da, reduction_da, cols=None, **kwargs):
    return Daf.sum_da(row_da, reduction_da, cols=cols, **kwargs)


def test_groupby_reduce_scoped_matches_unscoped_values():
    d = make_daf()
    scoped = d.groupby_reduce('g', Daf.sum_da, reduce_cols=['y'])
    assert scoped.columns() == ['g', 'x', 'y', 'z']
    assert scoped.lol == [['a', NULL, 40, NULL], ['b', NULL, 20, NULL]]


def test_groupby_reduce_unknown_reduce_col_is_dropped_from_groups():
    d = make_daf()
    result = d.groupby_reduce('g', Daf.sum_da, reduce_cols=['y', 'nope'])
    assert result.lol == [['a', NULL, 40, NULL], ['b', NULL, 20, NULL]]


def test_groupby_reduce_by_table_is_not_scoped():
    seen = []

    def table_func(group_daf, cols=None, **kwargs):
        seen.append(group_daf.columns())
        return {'y': len(group_daf)}

    d = make_daf()
    d.groupby_reduce('g', table_func, by='table', reduce_cols=['y'])
    assert seen == [['g', 'x', 'y', 'z'], ['g', 'x', 'y', 'z']]


def test_groupby_cols_reduce_scoped():
    d = make_daf()
    result = d.groupby_cols_reduce(['g'], Daf.sum_da, reduce_cols=['y', 'z'])
    assert result.columns() == ['g', 'y', 'z']
    assert result.lol == [['a', 40, 400], ['b', 20, 200]]


def test_multi_groupby_reduce_scoped():
    d = make_daf()
    result = d.multi_groupby_reduce(['g'], Daf.sum_da, reduce_cols=['y'])
    assert result['g'].lol == [['a', NULL, 40, NULL], ['b', NULL, 20, NULL]]


def test_groupsum_daf_scoped():
    d = make_daf()
    assert d.groupsum_daf('g', reduce_cols=['z']).lol == [['a', NULL, NULL, 400], ['b', NULL, NULL, 200]]
