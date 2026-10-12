# test_daf_sum_by_col.py
#
# reduce() with sum_da() or sum_kl(), and so daf_sum() and groupby_cols_reduce(), adds down
# each column with _sum_by_col() instead of across each row. These tests check that the
# totals are the same as the row by row path gives, also for floats, where the order of
# the additions matters.

import random
from decimal import Decimal
from typing import Any

import numpy as np
import pytest
from daffodil.daf import Daf, KeysDisabledError


def _by_row(daf: Daf, cols: Any = None, **kwargs: Any) -> Any:
    """ The row by row path: silent_error=True keeps reduce() off the column path. """
    return daf.reduce(Daf.sum_kl, cols=cols, silent_error=True, **kwargs)


def _tables() -> list:
    random.seed(7)
    floats = [random.uniform(-1e6, 1e6) * 10 ** random.randint(-8, 8) for _ in range(500)]
    return [
        Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y']),
        Daf(lol=[[i, f] for i, f in enumerate(floats)], cols=['i', 'f']),
        Daf(lol=[['r1', 1, 'abc', ''], ['r2', 2.5, '', 'x'], ['r3', True, 4, None]], cols=['k', 'n', 'm', 'g'], keyfield='k'),
        Daf(lol=[['r1', 1, 2], ['r2', 3]], cols=['k', 'n', 'm']),
        Daf(lol=[['r1', 1, 2, 99], ['r2', 3, 4]], cols=['k', 'n', 'm']),
        Daf(lol=[['a', [1], (2,), {'z': 1}], ['b', [3], (4,), {'z': 2}]], cols=['k', 'l', 't', 'd']),
        Daf(lol=[['a', Decimal('1.1'), np.int64(5), np.float32(0.5)], ['b', Decimal('2.2'), np.int64(6), 1.5]], cols=['k', 'dec', 'ni', 'nf']),
        Daf(lol=[['a', True, -0.0], ['b', False, -0.0]], cols=['k', 'b', 'z']),
        Daf(lol=[['a', 1, 'x'], ['b', 'x', 1], ['c', 2, 'y']], cols=['k', 'p', 'q']),
    ]


COLS: list = [None, 'x', ['x', 'y'], ['zz'], 'zz', [], ['f'], ['n', 'm', 'g'], ['l', 't', 'd'],
              ['dec', 'ni', 'nf'], ['b', 'z'], ['p', 'q'], ['q', 'p', 'zz', 'k']]


@pytest.mark.parametrize('cols', COLS, ids=repr)
@pytest.mark.parametrize('itable', range(len(_tables())))
@pytest.mark.parametrize('func', [Daf.sum_da, Daf.sum_kl], ids=['sum_da', 'sum_kl'])
def test_same_totals_as_by_row(func: Any, itable: int, cols: Any) -> None:
    daf = _tables()[itable]
    by_col = daf.reduce(func, cols=cols)
    by_row = _by_row(daf, cols=cols)
    assert by_col == by_row
    assert [type(v) for v in by_col.values()] == [type(v) for v in by_row.values()]
    assert list(by_col) == daf.columns()


def test_float_order_kept() -> None:
    """ The builtin sum() adds floats with compensation from Python 3.12, which gives another total. """
    daf = _tables()[1]
    total = 0
    for row_la in daf.lol:
        total = total + row_la[1] + 0
    assert daf.daf_sum(cols=['f'])['f'] == total


def test_daf_sum_and_groupby_cols_reduce() -> None:
    daf = _tables()[0]
    assert daf.daf_sum(cols=['x', 'y']) == {'g': '', 'x': 6, 'y': 60}
    result = daf.groupby_cols_reduce(['g'], Daf.sum_da, reduce_cols=['x', 'y'])
    assert result.to_lod() == [{'g': 'a', 'x': 4, 'y': 40}, {'g': 'b', 'x': 2, 'y': 20}]


def test_table_unchanged() -> None:
    daf = _tables()[3]
    daf.daf_sum()
    assert daf.lol == [['r1', 1, 2], ['r2', 3]]


def test_other_options_take_the_row_path() -> None:
    daf = _tables()[0]
    assert daf.reduce(Daf.sum_da, cols=['x'], initial_da={'x': 100}) == {'g': '', 'x': 106, 'y': ''}
    assert daf.reduce(Daf.sum_da, cols=['x', 'y'], astype=float) == {'g': '', 'x': 6.0, 'y': 60.0}


def test_empty_and_no_column_names() -> None:
    assert Daf(cols=['a']).daf_sum() == {}
    with pytest.raises(KeysDisabledError):
        Daf(lol=[[1, 2], [3, 4]]).reduce(Daf.sum_da)


# --- groupby_cols_reduce() with a sum: one pass, no Daf for each group ---

def _sum_wrapper(row: Any, reduction_da: Any, cols: Any = None, **kwargs: Any) -> Any:
    """ Not Daf.sum_kl itself, so it takes the old path: a Daf for each group, reduced by row. """
    return Daf.sum_kl(row, reduction_da, cols=cols, **kwargs)


def _state(daf: Daf) -> tuple:
    return (daf.lol, daf.hd, daf.dtypes, daf.keyfield, daf.name, daf.attrs, type(daf))


def _group_table() -> Daf:
    random.seed(3)
    lol = [[f'k{i}', random.choice(['a', 'b', '', 'c']), random.randrange(4),
            random.uniform(-1e3, 1e3) * 10 ** random.randint(-6, 6),
            random.choice([1, 2.5, 'abc', '', None, True, np.int64(3), [1]])] for i in range(2000)]
    return Daf(lol=lol, cols=['id', 'g', 'h', 'f', 'm'], keyfield='id', dtypes={'id': str, 'g': str, 'h': int, 'f': float, 'm': int})


GROUP_CASES: list = [
    (['g'], ['f']), (['g'], ['f', 'm']), (['g', 'h'], ['m', 'f']), (['h'], ['m']), (['g', 'h'], ['f']),
]


@pytest.mark.parametrize('case', GROUP_CASES, ids=repr)
@pytest.mark.parametrize('func', [Daf.sum_da, Daf.sum_kl], ids=['sum_da', 'sum_kl'])
def test_groupby_cols_reduce_sum_same_as_old_path(func: Any, case: tuple) -> None:
    daf = _group_table()
    groupby_colnames, reduce_cols = case
    new = daf.groupby_cols_reduce(groupby_colnames, func, reduce_cols=reduce_cols)
    old = daf.groupby_cols_reduce(groupby_colnames, _sum_wrapper, reduce_cols=reduce_cols)
    assert _state(new) == _state(old)
    assert len(new) > 1


def test_groupby_cols_reduce_sum_empty() -> None:
    daf = _group_table().clone_empty()
    new = daf.groupby_cols_reduce(['g'], Daf.sum_da, reduce_cols=['f'])
    old = daf.groupby_cols_reduce(['g'], _sum_wrapper, reduce_cols=['f'])
    assert _state(new) == _state(old)


def test_groupby_cols_reduce_sum_short_row_raises() -> None:
    for func in (Daf.sum_da, _sum_wrapper):
        daf = Daf(lol=[['a', 1, 2], ['b', 3]], cols=['g', 'x', 'y'])
        with pytest.raises(IndexError):
            daf.groupby_cols_reduce(['g'], func, reduce_cols=['x', 'y'])


@pytest.mark.parametrize('reduce_cols', [None, ['f', 'zz'], ['g', 'f'], ['f', 'f']], ids=repr)
def test_groupby_cols_reduce_other_cases_unchanged(reduce_cols: Any) -> None:
    """ These take the old path, so they give what they gave before. """
    daf = _group_table()
    new = daf.groupby_cols_reduce(['g'], Daf.sum_da, reduce_cols=reduce_cols)
    old = daf.groupby_cols_reduce(['g'], _sum_wrapper, reduce_cols=reduce_cols)
    assert _state(new) == _state(old)
