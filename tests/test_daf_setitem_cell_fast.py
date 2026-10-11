# test_daf_setitem_cell_fast.py
#
# __setitem__ writes one cell with a plain value straight into lol, without parsing
# the selectors. These tests check that it does what the general path does: the same
# table afterwards, the same return value, and the same error.

from typing import Any

import pytest
from daffodil.daf import Daf
from daffodil.lib.daf_types import T_la


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
