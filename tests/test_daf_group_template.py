# test_daf_group_template.py
#
# groupby(), groupby_cols() and multi_groupby() make the first group with clone_empty(),
# and each later group as a copy of the first. These tests check that every group is the
# same as one made with clone_empty(), and that the groups do not share what they own.

from typing import Any

import pytest
from daffodil.daf import Daf


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
    assert daf.groupby_cols(['g']) == {}
    assert daf.groupby('g') == {}
