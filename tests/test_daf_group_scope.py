# test_daf_group_scope.py
# Tests for column scoping in groupby(), groupby_cols(), multi_groupby() and the reduce methods.

import pytest

from daffodil.daf import Daf, NULL
from daffodil.keyedlist import KeyedList
from typing import Any
from daffodil.daf import Daf


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


def test_groupby_without_cols_shares_rows():
    data_daf:     Daf             = make_daf()
    groups_dodaf: dict[str, Daf]  = data_daf.groupby('g')
    assert groups_dodaf['a'].lol[0] is data_daf.lol[0]
    assert groups_dodaf['a'].lol[1] is data_daf.lol[2]
    assert groups_dodaf['b'].lol[0] is data_daf.lol[1]


def test_groupby_with_cols_does_not_share_rows():
    data_daf:     Daf             = make_daf()
    groups_dodaf: dict[str, Daf]  = data_daf.groupby('g', cols=['g', 'y'])
    assert groups_dodaf['a'].lol[0] is not data_daf.lol[0]


def test_groupby_cell_change_in_a_group_changes_the_source():
    data_daf:     Daf             = make_daf()
    groups_dodaf: dict[str, Daf]  = data_daf.groupby('g')
    groups_dodaf['a'][0, 'x'] = 99
    assert data_daf.lol[0] == ['a', 99, 10, 100]


def test_groupby_of_an_editable_copy_keeps_the_source():
    data_daf:     Daf             = make_daf()
    groups_dodaf: dict[str, Daf]  = data_daf.copy(level='editable').groupby('g')
    groups_dodaf['a'][0, 'x'] = 99
    assert data_daf.lol[0] == ['a', 1, 10, 100]


def test_groupby_new_column_in_a_group_keeps_the_source_shape():
    data_daf:     Daf             = make_daf()
    groups_dodaf: dict[str, Daf]  = data_daf.groupby('g')
    groups_dodaf['a'].insert_col('w', default=0)
    assert data_daf.columns() == ['g', 'x', 'y', 'z']
    assert data_daf.lol == [['a', 1, 10, 100], ['b', 2, 20, 200], ['a', 3, 30, 300]]


def test_multi_groupby_without_colnames_shares_rows():
    data_daf:      Daf                        = make_daf()
    groups_dododaf: dict[str, dict[str, Daf]] = data_daf.multi_groupby(['g', 'x'])
    assert groups_dododaf['g']['a'].lol[1] is data_daf.lol[2]
    assert groups_dododaf['x'][3].lol[0] is data_daf.lol[2]


def test_multi_groupby_with_colnames_does_not_share_rows():
    data_daf:      Daf                        = make_daf()
    groups_dododaf: dict[str, dict[str, Daf]] = data_daf.multi_groupby(['g'], colnames=['y'])
    assert groups_dododaf['g']['a'].lol[0] is not data_daf.lol[0]


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


# ---- from test_daf_group_template.py ----
# groupby(), groupby_cols() and multi_groupby() make the first group with clone_empty(),
# and each later group as a copy of the first. These tests check that every group is the
# same as one made with clone_empty(), and that the groups do not share what they own.

def _table() -> Daf:
    daf = Daf(lol=[['r1', 'a', 1, 10], ['r2', 'b', 2, 20], ['r3', 'a', 3, 30], ['r4', 'c', 4, 40]],
              cols=['k', 'g', 'x', 'y'], keyfield='k', dtypes={'k': str, 'g': str, 'x': int, 'y': int})
    daf.attrs['note'] = {'deep': [1]}
    return daf


def _layout(daf: Daf) -> tuple:
    return (type(daf), daf.lol, daf.hd, daf.dtypes, daf.keyfield, daf.name, daf.attrs,
            daf.disp_cols, daf.schema, sorted(daf.__dict__))


def _old_way(daf: Daf, groups: dict, cols: Any) -> dict:
    """ What the groups were before: one clone_empty() for each. """
    names, _idxs, all_cols = daf._cols_scope(cols)
    return {key: daf.clone_empty(lol=group.lol) if all_cols else daf.clone_empty(lol=group.lol, cols=names)
            for key, group in groups.items()}


COLS: list = [None, ['x', 'y'], ['k', 'x'], ['y', 'g', 'k', 'x']]


@pytest.mark.parametrize('cols', COLS, ids=repr)
def test_groupby_cols_same_as_clone_empty(cols: Any) -> None:
    daf = _table()
    groups = daf.groupby_cols(['g'], cols=cols)
    old = _old_way(daf, groups, cols)
    assert list(groups) == [('a',), ('b',), ('c',)]
    assert [_layout(g) for g in groups.values()] == [_layout(g) for g in old.values()]


@pytest.mark.parametrize('cols', COLS, ids=repr)
def test_groupby_same_as_clone_empty(cols: Any) -> None:
    daf = _table()
    groups = daf.groupby('g', cols=cols)
    old = _old_way(daf, groups, cols)
    assert [_layout(g) for g in groups.values()] == [_layout(g) for g in old.values()]


@pytest.mark.parametrize('cols', COLS, ids=repr)
def test_multi_groupby_same_as_clone_empty(cols: Any) -> None:
    daf = _table()
    result = daf.multi_groupby(['g', 'x'], colnames=cols)
    for groups in result.values():
        old = _old_way(daf, groups, cols)
        assert [_layout(g) for g in groups.values()] == [_layout(g) for g in old.values()]


def test_groups_own_their_parts() -> None:
    """ A change to one group's columns, attrs or rows is not seen by another group. """
    daf = _table()
    first, second = list(daf.groupby_cols(['g']).values())[:2]
    for part in ('hd', 'dtypes', 'attrs', '_kd', 'lol'):
        assert getattr(first, part) is not getattr(second, part)
    assert first.attrs['note'] is not second.attrs['note']
    first.rename_cols({'x': 'xx'})
    first.append({'k': 'r9', 'g': 'a', 'xx': 9, 'y': 90})
    assert second.columns() == ['k', 'g', 'x', 'y']
    assert second.lol == [['r2', 'b', 2, 20]]
    assert daf.columns() == ['k', 'g', 'x', 'y']


def test_key_index_of_each_group() -> None:
    daf = _table()
    groups = daf.groupby_cols(['g'])
    groups[('a',)].retmode = 'val'
    assert groups[('a',)].keys() == ['r1', 'r3']
    assert groups[('b',)].keys() == ['r2']
    assert groups[('a',)]['r3', 'x'] == 3


def test_no_rows_gives_no_groups() -> None:
    daf = _table().clone_empty()
    assert isinstance(daf.groupby_cols(['g']), KeyedList) and len(daf.groupby_cols(['g'])) == 0
    assert isinstance(daf.groupby('g'), KeyedList) and len(daf.groupby('g')) == 0


# groupby(), groupby_cols() and group_where() give a KeyedList of Daf (Q155)

def test_groupings_are_keyedlists_used_as_dicts() -> None:
    daf = Daf(lol=[['a', 1], ['b', 2], ['a', 3]], cols=['g', 'x'])
    for groups in (daf.groupby('g'), daf.groupby_cols(['g']), daf.group_where(lambda row: row['g'])):
        assert isinstance(groups, KeyedList)
        assert len(groups) == 2
        first_key = list(groups.keys())[0]
        assert groups[first_key] is groups.values()[0]
        assert dict(groups.items()) == groups.to_dict()
    assert daf.groupby('g')['a'].lol == [['a', 1], ['a', 3]]
    assert daf.groupby_cols(['g'])[('b',)].lol == [['b', 2]]


def test_group_where_true_and_false_with_get() -> None:
    daf = Daf(lol=[[1], [2], [3]], cols=['n'])
    groups = daf.group_where(lambda row: row['n'] > 1)
    assert groups.get(True).lol == [[2], [3]] and groups.get(False).lol == [[1]]
    groups = daf.group_where(lambda row: row['n'] > 9)
    assert groups.get(True) is None and len(groups.get(False)) == 3


# groupby_repeated_cols

def make_images_daf() -> Daf:
    """ Rows that share a crc or a hash, with 0 and blank values that must never match. """
    return Daf(
        cols=['ballot_id', 'crc_p1', 'crc_p2', 'hash_p1', 'hash_p2'],
        lol=[['b1', 7,    1, 'h1', NULL],
             ['b2', 0,    1, NULL, NULL],
             ['b3', 7,    1, 'h1', NULL],
             ['b4', 0,    1, NULL, NULL],
             ['b5', NULL, 1, 'h2', 'x'],
             ['b6', NULL, 1, 'h2', 'y'],
             ['b7', 7,    2, 'h1', NULL]],
    )


def test_groupby_repeated_cols_keeps_only_repeated_keys_in_table_order():
    """ A blank in a later key column is a value, so b1, b3 and b7 match on the hashes. """
    images_daf: Daf = make_images_daf()
    repeats_klodaf = images_daf.groupby_repeated_cols(['hash_p1', 'hash_p2'], exclude={'hash_p1': [NULL]})
    assert list(repeats_klodaf.keys()) == [('h1', NULL)]
    assert repeats_klodaf[('h1', NULL)].col('ballot_id') == ['b1', 'b3', 'b7']
    assert repeats_klodaf[('h1', NULL)].lol[0] is images_daf.lol[0]


def test_groupby_repeated_cols_exclude_leaves_out_rows_that_must_never_match():
    images_daf: Daf = make_images_daf()
    every_klodaf = images_daf.groupby_repeated_cols(['crc_p1', 'crc_p2'])
    assert list(every_klodaf.keys()) == [(7, 1), (0, 1), (NULL, 1)]
    repeats_klodaf = images_daf.groupby_repeated_cols(['crc_p1', 'crc_p2'], exclude={'crc_p1': [0, NULL]})
    assert list(repeats_klodaf.keys()) == [(7, 1)]
    assert repeats_klodaf[(7, 1)].col('ballot_id') == ['b1', 'b3']


def test_groupby_repeated_cols_no_repeat_gives_an_empty_result():
    assert len(make_images_daf().groupby_repeated_cols(['ballot_id'])) == 0
    assert not make_images_daf().groupby_repeated_cols(['ballot_id'])


def test_groupby_repeated_cols_unknown_name_raises():
    with pytest.raises(KeyError):
        make_images_daf().groupby_repeated_cols(['nothing'])
    with pytest.raises(KeyError):
        make_images_daf().groupby_repeated_cols(['crc_p1'], exclude={'nothing': [0]})
