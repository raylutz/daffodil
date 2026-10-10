# test_daf_getitem_val_fast.py
#
# In val mode, __getitem__ reads one row or one cell straight from lol, without
# building a Daf. These tests check that it gives what the general path gives:
# the same value, the same row object, and the same error.

from typing import Any

import pytest
from daffodil.daf import Daf
from daffodil.lib.daf_types import T_la


def _general_path(daf: Daf, slice_spec: Any) -> Any:
    """ What __getitem__ returned before the fast path: build a Daf, then unpack it. """
    irows, icols = daf._parse_selectors(slice_spec)
    if icols is None:
        ret_daf = daf.select_irows(irows=irows)
    else:
        ret_daf = daf.select_irows(irows=irows).select_icols(icols=icols)
    return ret_daf._adjust_return_val(daf.retmode)


def _outcome(func: Any) -> tuple:
    """ The result of a call, or the type and message of the error it raised. """
    try:
        return ('ok', func())
    except Exception as exc_info:
        return ('err', type(exc_info), str(exc_info))


def _tables() -> list:
    return [
        Daf(lol=[['r1', 10, 100], ['r2', 20, 200], ['r3', 30, 300]], cols=['key', 'A', 'B'], keyfield='key', retmode='val'),
        Daf(lol=[['r1', 10, 100], ['r2', 20, 200]], cols=['key', 'A', 'B'], retmode='val'),
        Daf(lol=[[10, 100], [20, 200]], retmode='val'),
        Daf(lol=[['r1'], ['r2']], cols=['key'], keyfield='key', retmode='val'),
        Daf(lol=[['r1', 10], ['r2']], cols=['key', 'A'], keyfield='key', retmode='val'),
        Daf(lol=[[], [1, 2]], cols=['a', 'b'], retmode='val'),
        Daf(cols=['key', 'A'], keyfield='key', retmode='val'),
    ]


SPECS: T_la = [
    0, 1, -1, 2, 5, -9, True,
    'r1', 'r2', 'zz', '0',
    (0, 'A'), (1, 'B'), (-1, 'key'), (0, 'zz'), (0, '1'),
    (0, 0), (1, 2), (-1, -1), (0, 5), (5, 0),
    ('r2', 'A'), ('r1', 1), ('zz', 'A'), ('r1', 'zz'),
    (0,), (0, 1, 2),
]


@pytest.mark.parametrize('spec', SPECS, ids=repr)
@pytest.mark.parametrize('itable', range(len(_tables())))
def test_val_fast_path_matches_general_path(itable: int, spec: Any) -> None:
    daf = _tables()[itable]
    fast    = _outcome(lambda: daf[spec])
    general = _outcome(lambda: _general_path(daf, spec))
    if fast[0] == 'ok' and isinstance(fast[1], Daf):
        assert isinstance(general[1], Daf)
        assert (fast[1].lol, fast[1].columns()) == (general[1].lol, general[1].columns())
    else:
        assert fast == general


def test_val_fast_row_is_the_row_itself() -> None:
    daf = Daf(lol=[['r1', 10, 100], ['r2', 20, 200]], cols=['key', 'A', 'B'], keyfield='key', retmode='val')
    assert daf[1] is daf.lol[1]
    assert daf['r2'] is daf.lol[1]
    daf[1][1] = 99
    assert daf.lol[1] == ['r2', 99, 200]


def test_val_fast_values() -> None:
    daf = Daf(lol=[['r1', 10, 100], ['r2', 20, 200]], cols=['key', 'A', 'B'], keyfield='key', retmode='val')
    assert daf[0] == ['r1', 10, 100]
    assert daf[1, 'B'] == 200
    assert daf['r1', 'A'] == 10
    assert daf[-1, -1] == 200
    one_col = Daf(lol=[['r1'], ['r2']], cols=['key'], keyfield='key', retmode='val')
    assert one_col[1] == 'r2'


def test_val_fast_key_lookup_after_append() -> None:
    # The key index is rebuilt when stale, as on the general path.
    daf = Daf(lol=[['r1', 10]], cols=['key', 'A'], keyfield='key', retmode='val')
    assert daf['r1', 'A'] == 10
    daf.append({'key': 'r2', 'A': 20})
    assert daf['r2', 'A'] == 20


def test_obj_mode_still_returns_daf() -> None:
    daf = Daf(lol=[['r1', 10, 100]], cols=['key', 'A', 'B'], keyfield='key')
    assert isinstance(daf[0], Daf)
    assert isinstance(daf[0, 'A'], Daf)
