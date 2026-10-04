# test_daf_copy_levels.py
# Tests for copy(level=...). Each action is applied to a copy at each level. Then the original
# is checked, so the table in the copy() docstring stays true.

import pytest

from daffodil.daf import Daf
from daffodil.keyedlist import KeyedIndex


LEVELS = ['shallow', 'sortable', 'editable', 'deep']


def make_daf() -> Daf:
    daf = Daf(
        lol=[[1, ' a ', 10], [2, ' b ', 20], [3, ' c ', 30]],
        cols=['id', 'v', 'n'],
        keyfield='id',
        dtypes={'id': int, 'v': str, 'n': int},
    )
    daf.keys()      # build the key index.
    return daf


def snapshot(daf: Daf):
    return ([list(row) for row in daf.lol], dict(daf.hd), dict(daf.dtypes or {}), daf.keyfield)


def is_consistent(daf: Daf) -> bool:
    if any(len(row) != len(daf.hd) for row in daf.lol):
        return False
    if daf.keyfield and daf._kd:
        keyidx = daf.hd[daf.keyfield]
        return all(daf.lol[irow][keyidx] == key for key, irow in daf._kd.items())
    return True


# action name: (function applied to the copy, first level at which the original stays safe)
ACTIONS = {
    'append':               (lambda c: c.append([4, 'd', 40]),                  'sortable'),
    'extend':               (lambda c: c.extend([{'id': 4, 'v': 'd', 'n': 40}]),'sortable'),
    'insert_irow':          (lambda c: c.insert_irow(0, [0, 'z', 0]),           'sortable'),
    'remove_key':           (lambda c: c.remove_key(2),                         'shallow'),
    'sort_by_colname':      (lambda c: c.sort_by_colname('v', reverse=True),    'shallow'),
    'lol_sort':             (lambda c: c.lol.sort(key=lambda r: -r[0]),         'sortable'),
    'lol_reverse':          (lambda c: c.lol.reverse(),                         'sortable'),
    'set_lol':              (lambda c: c.set_lol([[9, 'q', 9]]),                'shallow'),
    'set_keyfield':         (lambda c: c.set_keyfield('v'),                     'shallow'),
    'set_cols':             (lambda c: c.set_cols(['x', 'y', 'z']),             'shallow'),
    'rename_cols':          (lambda c: c.rename_cols({'v': 'vv'}),              'shallow'),
    'set_dtypes':           (lambda c: c.set_dtypes({'id': float}),             'shallow'),
    'hd_edit':              (lambda c: c.hd.__setitem__('extra', 3),            'sortable'),
    'dtypes_edit':          (lambda c: c.dtypes.__setitem__('id', float),       'sortable'),
    'drop_cols':            (lambda c: c.drop_cols(['v']),                      'sortable'),
    'assign_col':           (lambda c: c.assign_col('new', [1, 2, 3]),          'sortable'),
    'insert_col':           (lambda c: c.insert_col('new2', [1, 2, 3], 1),      'sortable'),
    'set_icol':             (lambda c: c.set_icol(1, 'X'),                      'sortable'),
    'set_col_irows':        (lambda c: c.set_col_irows('v', [0], 'X'),          'editable'),
    'setitem':              (lambda c: c.__setitem__((0, 'v'), 'X'),            'editable'),
    'replace_in_columns':   (lambda c: c.replace_in_columns(['v'], ' a ', 'Q'), 'sortable'),
    'apply_in_place':       (lambda c: c.apply_in_place(lambda r: {**r, 'v': 'Z'}, by='row'), 'sortable'),
    'strip':                (lambda c: c.strip(),                               'sortable'),
    'cell_edit':            (lambda c: c.lol[0].__setitem__(1, 'X'),            'editable'),
}


# These edit the data structures directly. The library does not repair the copy afterward.
DIRECT_EDITS = {'lol_sort', 'lol_reverse', 'hd_edit', 'dtypes_edit', 'cell_edit'}


@pytest.mark.parametrize('action', list(ACTIONS))
@pytest.mark.parametrize('level', LEVELS)
def test_original_is_safe_from_the_first_safe_level(level, action):
    func, first_safe = ACTIONS[action]
    daf = make_daf()
    before = snapshot(daf)

    copied = daf.copy(level)
    func(copied)

    if LEVELS.index(level) >= LEVELS.index(first_safe):
        assert snapshot(daf) == before
        assert is_consistent(daf)
        if action not in DIRECT_EDITS:
            assert is_consistent(copied)
    else:
        assert snapshot(daf) != before or not is_consistent(daf)


def test_default_level_is_shallow():
    daf = make_daf()
    copied = daf.copy()
    assert copied.lol is daf.lol
    assert copied.hd is daf.hd


def test_shallow_shares_everything_but_attrs():
    daf = make_daf()
    daf.attrs['x'] = [1]
    copied = daf.copy('shallow')
    assert copied.lol is daf.lol and copied.lol[0] is daf.lol[0]
    assert copied.dtypes is daf.dtypes
    assert copied.attrs == daf.attrs and copied.attrs is not daf.attrs


def test_sortable_copies_lol_hd_dtypes_and_shares_rows():
    daf = make_daf()
    copied = daf.copy('sortable')
    assert copied.lol is not daf.lol
    assert copied.hd is not daf.hd and copied.hd == daf.hd
    assert copied.dtypes is not daf.dtypes and copied.dtypes == daf.dtypes
    assert copied.lol[0] is daf.lol[0]


def test_sortable_clears_key_index_and_rebuilds_it_on_use():
    daf = make_daf()
    copied = daf.copy('sortable')
    assert copied._kd == {}
    copied.lol.reverse()
    assert copied.keys() == [3, 2, 1]
    assert copied._kd == {3: 0, 2: 1, 1: 2}
    assert daf._kd == {1: 0, 2: 1, 3: 2}


def test_editable_copies_rows_and_shares_cells():
    daf = make_daf()
    daf.lol[0][1] = ['mutable']
    copied = daf.copy('editable')
    assert copied.lol[0] is not daf.lol[0]
    assert copied.lol[0][1] is daf.lol[0][1]
    assert copied.lol[0][0] is daf.lol[0][0]


def test_deep_copies_containers_and_shares_text_and_numbers():
    daf = Daf(lol=[['abc', 5, [1, 2], {'k': 1}]], cols=['a', 'b', 'c', 'd'])
    copied = daf.copy('deep')
    assert copied.lol is not daf.lol and copied.lol[0] is not daf.lol[0]
    assert copied.hd is not daf.hd
    assert copied.lol[0][0] is daf.lol[0][0]
    assert copied.lol[0][1] is daf.lol[0][1]
    assert copied.lol[0][2] is not daf.lol[0][2]
    assert copied.lol[0][3] is not daf.lol[0][3]


def test_old_arguments_still_work():
    daf = make_daf()
    assert daf.copy(for_sorting=True).lol is not daf.lol
    assert daf.copy(for_sorting=True).lol[0] is daf.lol[0]
    assert daf.copy(deep=True).lol[0] is not daf.lol[0]
    assert daf.copy(True).lol[0] is not daf.lol[0]
    assert daf.copy(False).lol is daf.lol


def test_higher_level_wins_over_old_arguments():
    daf = make_daf()
    assert daf.copy('editable', for_sorting=True).lol[0] is not daf.lol[0]
    assert daf.copy('shallow', deep=True).lol[0] is not daf.lol[0]


def test_unknown_level_raises():
    with pytest.raises(ValueError, match='level must be one of'):
        make_daf().copy('medium')


def test_sortable_with_no_dtypes_and_empty_daf():
    daf = Daf(lol=[[1, 2]], cols=['a', 'b'])
    assert daf.copy('sortable').dtypes == daf.dtypes
    empty = Daf()
    for level in LEVELS:
        assert empty.copy(level).lol == []


def test_sortable_copies_a_keyedindex_header():
    daf = make_daf()
    daf.hd = KeyedIndex(['id', 'v', 'n'])       # type: ignore[assignment]
    copied = daf.copy('sortable')
    assert isinstance(copied.hd, KeyedIndex)
    assert copied.hd is not daf.hd
    copied.hd.append('extra')
    assert 'extra' not in daf.hd
