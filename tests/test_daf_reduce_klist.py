# test_daf_reduce_klist.py
#
# With by='row', reduce() gives the function a new KeyedList for each row, and uses
# sum_kl() and count_values_kl() in place of sum_da() and count_values_da(). These
# tests check that the results are what the dict rows gave before.

from typing import Any

import pytest
from daffodil.daf import Daf, KeysDisabledError
from daffodil.keyedlist import KeyedList
from daffodil.lib.daf_types import T_da


def _reduce_with_dicts(daf: Daf, func: Any, cols: Any = None, initial_da: T_da | None = None, **kwargs: Any) -> T_da:
    """ What reduce(by='row') did before: a dict for each row. """
    if not daf:
        return {}
    if cols is None:
        cols_iter: Any = daf.hd.keys()
    elif isinstance(cols, str):
        cols_iter = [cols]
    else:
        cols_iter = {col: None for col in daf.hd.keys() if col in cols}
    reduction_da = initial_da if initial_da is not None else dict.fromkeys(cols_iter, 0)
    keys = daf.hd.keys()
    for row_la in daf.lol:
        reduction_da = func(dict(zip(keys, row_la)), reduction_da, cols=cols_iter, **kwargs)
    return {key: reduction_da.get(key, '') for key in daf.hd.keys()}


def _tables() -> list:
    return [
        Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y']),
        Daf(lol=[['r1', 1, 'abc', ''], ['r2', 2.5, '', 'x'], ['r3', True, 4, 'x']], cols=['k', 'n', 'm', 'g'], keyfield='k'),
        Daf(lol=[['r1', 1, 2], ['r2', 3]], cols=['k', 'n', 'm'], keyfield='k'),
        Daf(lol=[['r1', 1, 2, 99], ['r2', 3, 4]], cols=['k', 'n', 'm']),
        Daf(lol=[['r1', -1, 0.5]], cols=['k', 'n', 'm']),
    ]


COLS: list = [None, ['x', 'y'], 'x', ['n', 'm'], 'n', ['zz'], 'zz', []]


@pytest.mark.parametrize('cols', COLS, ids=repr)
@pytest.mark.parametrize('itable', range(len(_tables())))
@pytest.mark.parametrize('func', [Daf.sum_da, Daf.sum_kl], ids=['sum_da', 'sum_kl'])
def test_sum_matches_dict_rows(func: Any, itable: int, cols: Any) -> None:
    daf = _tables()[itable]
    assert daf.reduce(func, cols=cols) == _reduce_with_dicts(daf, Daf.sum_da, cols=cols)


@pytest.mark.parametrize('cols', [['x', 'y'], ['n', 'm']], ids=repr)
@pytest.mark.parametrize('itable', range(len(_tables())))
def test_sum_with_astype_matches_dict_rows(itable: int, cols: Any) -> None:
    daf = _tables()[itable]
    assert daf.reduce(Daf.sum_da, cols=cols, astype=float) == _reduce_with_dicts(daf, Daf.sum_da, cols=cols, astype=float)


def test_sum_with_initial_da() -> None:
    daf = _tables()[0]
    assert daf.reduce(Daf.sum_da, cols=['x'], initial_da={'x': 100}) == {'g': '', 'x': 106, 'y': ''}


@pytest.mark.parametrize('omit_nulls', [False, True])
@pytest.mark.parametrize('itable', [0, 1])
@pytest.mark.parametrize('func', [Daf.count_values_da, Daf.count_values_kl], ids=['count_values_da', 'count_values_kl'])
def test_count_values_matches_dict_rows(func: Any, itable: int, omit_nulls: bool) -> None:
    daf = _tables()[itable]
    cols = daf.columns()[1:]
    assert daf.reduce(func, cols=cols, omit_nulls=omit_nulls) == \
        _reduce_with_dicts(daf, Daf.count_values_da, cols=cols, omit_nulls=omit_nulls)


def test_count_values_of_dict_cells() -> None:
    daf = Daf(lol=[['r1', {'a': 1, 'b': 2}], ['r2', {'a': 5}]], cols=['k', 'd'])
    assert daf.reduce(Daf.count_values_kl, cols=['d']) == {'k': '', 'd': {'a': 6, 'b': 2}}


@pytest.mark.parametrize('func', [Daf.count_values_da, Daf.count_values_kl], ids=['count_values_da', 'count_values_kl'])
def test_count_values_skips_list_cells(func: Any) -> None:
    daf = Daf(lol=[['r1', ['a', 'b'], 'x'], ['r2', ['a'], 'x'], ['r3', 'z', 'y']], cols=['k', 'v', 'g'])
    assert daf.reduce(func, cols=['v', 'g']) == {'k': '', 'v': {'z': 1}, 'g': {'x': 2, 'y': 1}}


def test_daf_sum_and_daf_valuecount() -> None:
    daf = _tables()[0]
    assert daf.daf_sum(cols=['x', 'y']) == {'g': '', 'x': 6, 'y': 60}
    assert daf.daf_valuecount(cols=['g'])['g'] == {'a': 2, 'b': 1}


def test_groupby_cols_reduce_unchanged() -> None:
    daf = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
    result = daf.groupby_cols_reduce(['g'], Daf.sum_da, reduce_cols=['x', 'y'])
    assert result.to_lod() == [{'g': 'a', 'x': 4, 'y': 40}, {'g': 'b', 'x': 2, 'y': 20}]


def test_each_row_is_a_new_keyedlist() -> None:
    """ A function that keeps the rows gets each row, not the last row many times. """
    daf = _tables()[0]
    rows: list = []
    def keep(row: Any, reduction_da: T_da, cols: Any) -> T_da:
        rows.append(row)
        return reduction_da
    daf.reduce(keep)
    assert all(type(row) is KeyedList for row in rows)
    assert [dict(row) for row in rows] == daf.to_lod()


def test_writing_to_the_row_changes_the_table() -> None:
    daf = _tables()[0]
    def zero_x(row: Any, reduction_da: T_da, cols: Any) -> T_da:
        row['x'] = 0
        return reduction_da
    daf.reduce(zero_x)
    assert daf.col('x') == [0, 0, 0]


def test_short_row_is_padded_with_null() -> None:
    daf = _tables()[2]
    rows: list = []
    def keep(row: Any, reduction_da: T_da, cols: Any) -> T_da:
        rows.append(dict(row))
        return reduction_da
    daf.reduce(keep)
    assert rows == [{'k': 'r1', 'n': 1, 'm': 2}, {'k': 'r2', 'n': 3, 'm': ''}]
    assert daf.lol[1] == ['r2', 3]


def test_no_column_names_raises() -> None:
    daf = Daf(lol=[[1, 2], [3, 4]])
    with pytest.raises(KeysDisabledError):
        daf.reduce(Daf.sum_da)


def test_silent_error_skips_the_row() -> None:
    daf = _tables()[0]
    def fail_on_b(row: Any, reduction_da: T_da, cols: Any) -> T_da:
        if row['g'] == 'b':
            raise ValueError('b')
        return Daf.sum_kl(row, reduction_da, cols=cols)
    assert daf.reduce(fail_on_b, cols=['x'], silent_error=True) == {'g': '', 'x': 4, 'y': ''}
    with pytest.raises(ValueError):
        daf.reduce(fail_on_b, cols=['x'])


def test_sum_kl_called_directly_with_a_dict() -> None:
    assert Daf.sum_kl({'x': 1, 'y': 'abc'}, {'x': 10, 'y': 0}, cols=['x', 'y']) == {'x': 11, 'y': 0}
    assert Daf.count_values_kl({'g': 'a'}, {}, ['g']) == {'g': {'a': 1}}
