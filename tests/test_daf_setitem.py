# test_daf_setitem.py
#
# Tests for Daf.set_irows_icols() -- the implementation behind Daf.__setitem__ (daf[irows, icols] =
# value). Covers all branches: single row/single col, single row/no col, multi-row/no col,
# multi-row/single col, and multi-row/multi-col, each across scalar/list/dict/Daf value types.
#
# Of particular note: dict values must be checked for *before* the broader Sequence/list check in
# every branch, since dict is itself Iterable (though not a Sequence). Getting this ordering wrong
# previously caused either silent data corruption (dict assigned as raw row content) or a hang via
# the deliberate breakpoint() tripwire in the except-block fallback. These tests guard against
# regressing that ordering.

import pytest

from daffodil.daf import Daf
from typing import Any
from daffodil.lib.daf_types import T_la
from daffodil.daf import Daf, KeysDisabledError


# --- single row, no column selection (whole-row assignment) ---

def test_setitem_single_row_list_value():
    daf = Daf(lol=[[1, 2], [3, 4]], cols=['a', 'b'])
    daf[0, :] = [9, 9]
    assert daf.lol == [[9, 9], [3, 4]]


def test_setitem_single_row_dict_value():
    daf = Daf(lol=[[1, 2], [3, 4]], cols=['a', 'b'])
    daf[0, :] = {'a': 10, 'b': 20}
    assert daf.lol == [[10, 20], [3, 4]]


def test_setitem_single_row_scalar_broadcasts():
    daf = Daf(lol=[[1, 2], [3, 4]], cols=['a', 'b'])
    daf[0, :] = 0
    assert daf.lol == [[0, 0], [3, 4]]


# --- single row, single column (single cell) ---

def test_setitem_single_cell_scalar():
    daf = Daf(lol=[[1, 2], [3, 4]], cols=['a', 'b'])
    daf[0, 0] = 99
    assert daf.lol == [[99, 2], [3, 4]]


def test_setitem_single_cell_dict_value():
    daf = Daf(lol=[[1, 2], [3, 4]], cols=['a', 'b'])
    daf[0, 0] = {'a': 7, 'b': 8}
    assert daf.lol == [[7, 8], [3, 4]]


def test_setitem_single_cell_single_item_list_unwraps():
    daf = Daf(lol=[[1, 2], [3, 4]], cols=['a', 'b'])
    daf[0, 0] = [55]
    assert daf.lol == [[55, 2], [3, 4]]


# --- multiple rows, no column selection ---

def test_setitem_multi_row_list_value_broadcasts_whole_row():
    daf = Daf(lol=[[1, 2], [3, 4], [5, 6]], cols=['a', 'b'])
    daf[[0, 1], :] = [0, 0]
    assert daf.lol == [[0, 0], [0, 0], [5, 6]]


def test_setitem_multi_row_scalar_broadcasts():
    daf = Daf(lol=[[1, 2], [3, 4], [5, 6]], cols=['a', 'b'])
    daf[[0, 1], :] = 7
    assert daf.lol == [[7, 7], [7, 7], [5, 6]]


def test_setitem_multi_row_dict_value_maps_columns():
    # this is the bug we found: dict must not be swallowed by the broader Sequence check,
    # or it gets assigned as the raw row content instead of being mapped to columns.
    daf = Daf(lol=[[1, 2], [3, 4], [5, 6]], cols=['a', 'b'])
    daf[[0, 1], :] = {'a': 9, 'b': 8}
    assert daf.lol == [[9, 8], [9, 8], [5, 6]]


# --- all rows or multiple rows, single column ---

def test_setitem_single_col_list_value_per_row():
    daf = Daf(lol=[[1, 2], [3, 4], [5, 6]], cols=['a', 'b'])
    daf[:, 0] = [100, 200, 300]
    assert daf.lol == [[100, 2], [200, 4], [300, 6]]


def test_setitem_single_col_scalar_broadcasts():
    daf = Daf(lol=[[1, 2], [3, 4], [5, 6]], cols=['a', 'b'])
    daf[:, 0] = 0
    assert daf.lol == [[0, 2], [0, 4], [0, 6]]


def test_setitem_single_col_dict_value_maps_all_matching_cols():
    # dict ignores the narrower icols selection and updates whichever of its own keys match --
    # documented in the source as "this is the same as cols=0 bc dict updates the corresponding cols."
    daf = Daf(lol=[[1, 2], [3, 4], [5, 6]], cols=['a', 'b'])
    daf[:, 0] = {'a': 100, 'b': 200}
    assert daf.lol == [[100, 200], [100, 200], [100, 200]]


# --- multiple rows, multiple columns ---

def test_setitem_multi_multi_scalar_broadcasts():
    daf = Daf(lol=[[1, 2], [3, 4]], cols=['a', 'b'])
    daf[:, :] = 0
    assert daf.lol == [[0, 0], [0, 0]]


def test_setitem_multi_multi_list_value_same_for_each_row():
    daf = Daf(lol=[[1, 2], [3, 4]], cols=['a', 'b'])
    daf[[0, 1], [0, 1]] = [10, 20]
    assert daf.lol == [[10, 20], [10, 20]]


def test_setitem_multi_multi_dict_value_maps_columns():
    # this is the bug we found: previously hung via breakpoint() since dict was swallowed by
    # the broader Iterable check, then tried integer-indexed lookup into the dict and raised
    # KeyError inside a bare except that hit the tripwire.
    daf = Daf(lol=[[1, 2], [3, 4]], cols=['a', 'b'])
    daf[[0, 1], [0, 1]] = {'a': 5, 'b': 6}
    assert daf.lol == [[5, 6], [5, 6]]


# ---- from test_daf_setitem_cell_fast.py ----
# __setitem__ writes one cell with a plain value straight into lol, without parsing
# the selectors. These tests check that it does what the general path does: the same
# table afterwards, the same return value, and the same error.

def _general_path(daf: Daf, slice_spec: Any, value: Any) -> Daf:
    """ What __setitem__ did before the fast path. """
    irows, icols = daf._parse_selectors(slice_spec)
    return daf.set_irows_icols(irows=irows, icols=icols, value=value)


def _outcome(func: Any) -> tuple:
    """ Whether a call returned or raised, and with what error. """
    try:
        func()
        return ('ok',)
    except Exception as exc_info:
        return ('err', type(exc_info), str(exc_info))


def _tables() -> list:
    return [
        Daf(lol=[['r1', 10, 100], ['r2', 20, 200], ['r3', 30, 300]], cols=['key', 'A', 'B'], keyfield='key'),
        Daf(lol=[['r1', 10, 100], ['r2', 20, 200]], cols=['key', 'A', 'B']),
        Daf(lol=[[10, 100], [20, 200]]),
        Daf(lol=[['r1', 10], ['r2']], cols=['key', 'A'], keyfield='key'),
        Daf(cols=['key', 'A'], keyfield='key'),
    ]


SPECS: T_la = [
    (0, 'A'), (1, 'B'), (-1, 'key'), (0, 'zz'), (0, '1'),
    (0, 0), (1, 2), (-1, -1), (0, 5), (5, 0), (-9, 0), (5, 'zz'), ('zz', 'zz'),
    ('r2', 'A'), ('r1', 1), ('zz', 'A'), ('r1', 'zz'), ('r2', -1),
    (True, 0), (0, True),
]

VALUES: T_la = [7, 2.5, 'x', '', True, False]


@pytest.mark.parametrize('value', VALUES, ids=repr)
@pytest.mark.parametrize('spec', SPECS, ids=repr)
@pytest.mark.parametrize('itable', range(len(_tables())))
def test_cell_fast_path_matches_general_path(itable: int, spec: Any, value: Any) -> None:
    fast_daf    = _tables()[itable]
    general_daf = _tables()[itable]
    fast    = _outcome(lambda: fast_daf.__setitem__(spec, value))
    general = _outcome(lambda: _general_path(general_daf, spec, value))
    assert fast == general
    assert fast_daf.lol == general_daf.lol


def test_cell_fast_returns_self() -> None:
    daf = _tables()[0]
    assert daf.__setitem__(('r2', 'A'), 5) is daf
    assert daf.__setitem__((0, 1), 5) is daf


@pytest.mark.parametrize('value', [[4], ('q',), {'A': 9}, None, [1, 2]], ids=repr)
def test_other_values_take_the_general_path(value: Any) -> None:
    """ A one-item list stores its item, a dict sets fields by name; as before. """
    fast_daf    = _tables()[0]
    general_daf = _tables()[0]
    fast    = _outcome(lambda: fast_daf.__setitem__(('r2', 'A'), value))
    general = _outcome(lambda: _general_path(general_daf, ('r2', 'A'), value))
    assert fast == general
    assert fast_daf.lol == general_daf.lol


def test_cell_fast_values() -> None:
    daf = _tables()[0]
    daf['r2', 'A'] = 21
    daf[2, 'B'] = 'z'
    daf[0, 1] = 1.5
    assert daf.lol == [['r1', 1.5, 100], ['r2', 21, 200], ['r3', 30, 'z']]


def test_plus_equals_by_key() -> None:
    daf = _tables()[0]
    daf.retmode = 'val'
    for _ in range(3):
        daf['r3', 'A'] += 2
    assert daf.lol[2] == ['r3', 36, 300]


def test_cell_fast_key_lookup_after_append() -> None:
    """ A key added after the key index was built is found, by reads and writes. """
    daf = _tables()[0]
    daf.retmode = 'val'
    assert daf['r1', 'A'] == 10
    daf.append({'key': 'r4', 'A': 40, 'B': 400})
    daf['r4', 'A'] = 41
    assert daf['r4', 'A'] == 41
    assert daf.lol[3] == ['r4', 41, 400]


def test_cell_fast_key_lookup_after_key_change() -> None:
    """ After a key is changed and the key index is rebuilt, the old key is gone. """
    daf = _tables()[0]
    daf.retmode = 'val'
    assert daf['r1', 'A'] == 10
    daf[0, 'key'] = 'q1'
    daf._rebuild_kd()
    daf['q1', 'A'] = 11
    assert daf.lol[0] == ['q1', 11, 100]
    with pytest.raises(KeyError):
        daf['r1', 'A'] = 12
    with pytest.raises(KeyError):
        daf['r1', 'A']


# ---- from test_daf_setitem_row_key_hint.py ----
# daf['name'] = ... selects a row by key. When the name is not a row key, the error says
# so, and points to daf[:, 'name'] and assign_col() for a column. Only the error path is
# changed. These tests check the messages, and that other errors are as before.

def _keyed() -> Daf:
    return Daf(lol=[['r1', 1], ['r2', 2]], cols=['k', 'x'], keyfield='k')


HINT = r"daf\[name\] selects a row\. For a column use daf\[:, 'flagged'\], or assign_col\('flagged', values\) to add one\."


def test_not_a_row_key() -> None:
    daf = _keyed()
    with pytest.raises(KeyError, match=r"'flagged' is not a row key\. " + HINT) as exc_info:
        daf['flagged'] = [True, False]
    assert type(exc_info.value.__cause__) is KeyError
    assert daf.lol == [['r1', 1], ['r2', 2]]


def test_column_name_that_is_not_a_row_key() -> None:
    daf = Daf(lol=[['r1', 1], ['r2', 2]], cols=['k', 'flagged'], keyfield='k')
    with pytest.raises(KeyError, match=HINT):
        daf['flagged'] = True


def test_no_keyfield() -> None:
    daf = Daf(lol=[['r1', 1], ['r2', 2]], cols=['k', 'x'])
    with pytest.raises(KeysDisabledError, match=r'key lookups are disabled.*' + HINT) as exc_info:
        daf['flagged'] = [True, False]
    assert isinstance(exc_info.value.__cause__, KeysDisabledError)


def test_row_key_still_works() -> None:
    daf = _keyed()
    daf['r2'] = ['r2', 20]
    assert daf.lol == [['r1', 1], ['r2', 20]]


def test_what_the_hint_suggests_works() -> None:
    daf = _keyed()
    daf[:, 'x'] = [5, 6]
    daf.assign_col('flagged', [True, False])
    assert daf.lol == [['r1', 5, True], ['r2', 6, False]]


def test_other_selectors_unchanged() -> None:
    """ A list of keys, a key with a column, or a read, gives the error it gave before. """
    daf = _keyed()
    with pytest.raises(KeyError) as exc_info:
        daf[['zz']] = [1, 2]
    assert 'selects a row' not in str(exc_info.value)
    with pytest.raises(KeyError) as exc_info:
        daf['zz', 'x'] = 1
    assert 'selects a row' not in str(exc_info.value)
    with pytest.raises(KeyError) as exc_info:
        daf['zz']
    assert 'selects a row' not in str(exc_info.value)


# ---- from test_daf_setitem_rows.py ----
# Assigning one value or one list to several whole rows: my_daf[[0, 1]] = value.

def _daf() -> Daf:
    return Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'])


def test_setitem_scalar_number_to_several_rows():
    d = _daf()
    d[[0, 1]] = 5
    assert d.lol == [[5, 5, 5], [5, 5, 5], [3, 'c', 30]]


def test_setitem_scalar_str_to_one_row():
    d = _daf()
    d[2] = 'x'
    assert d.lol == [[1, 'a', 10], [2, 'b', 20], ['x', 'x', 'x']]


def test_setitem_scalar_str_to_several_rows():
    d = _daf()
    d[[0, 1]] = 'x'
    assert d.lol == [['x', 'x', 'x'], ['x', 'x', 'x'], [3, 'c', 30]]


def test_setitem_scalar_str_to_all_rows():
    d = _daf()
    d[:] = 'x'
    assert d.lol == [['x', 'x', 'x']] * 3


def test_setitem_list_to_several_rows_makes_independent_rows():
    d = _daf()
    d[[0, 1]] = [7, 8, 9]
    assert d.lol[0] is not d.lol[1]
    d.lol[0][0] = 'X'
    assert d.lol[1][0] == 7


def test_setitem_scalar_str_to_a_column():
    d = _daf()
    d[:, 'v'] = 'x'
    assert d.col('v') == ['x', 'x', 'x']


def test_setitem_list_to_a_column():
    d = _daf()
    d[:, 'v'] = ['x', 'y', 'z']
    assert d.col('v') == ['x', 'y', 'z']


def test_setitem_str_to_block():
    d = _daf()
    d[[0, 1], ['v', 'n']] = 'ab'
    assert d.lol == [[1, 'ab', 'ab'], [2, 'ab', 'ab'], [3, 'c', 30]]


def test_setitem_str_to_a_column_whole_word():
    d = _daf()
    d[:, 'v'] = 'xyz'
    assert d.col('v') == ['xyz', 'xyz', 'xyz']


def test_setitem_tuple_to_several_rows_is_a_list_per_row():
    d = _daf()
    d[[0, 1]] = (7, 8, 9)
    assert d.lol[:2] == [[7, 8, 9], [7, 8, 9]]
    assert d.lol[0] is not d.lol[1]
