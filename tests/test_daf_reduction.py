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
import random
from decimal import Decimal
from typing import Any
import numpy as np
from daffodil.daf import Daf, KeysDisabledError
from daffodil.lib import daf_utils
from daffodil.lib.daf_utils import NULL
from daffodil.keyedlist import KeyedList
from daffodil.lib.daf_types import T_da


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


# The rows of a sparse sum. A key first appears in another order than the cols below.
# Text, None and a float show that a value is skipped or added just as sum_da() does.
SPARSE_SUM_DAF = Daf(lol=[['b1', "{'c1v': 1, 'c0b': 1, 'note': 'x'}"],
                          ['b2', "{'c0b': 2, 'c0v': None, 'c2v': 0.1}"],
                          ['b3', NULL],
                          ['b4', "{'c2v': 0.2, 'c1v': 'y', 'c0v': 3}"]],
                     cols=['ballot_id', 'cv'])


def sparse_sum_by_sum_da(daf: Daf, cols) -> dict:
    """ The sparse sum as sum_da() makes it, one row at a time, with each key tested against cols. """
    reduction_da: dict = {}
    for row_da in daf:
        reduction_da = Daf.sum_da(daf_utils.get_indirect_da(row_da, 'cv'), reduction_da, cols=cols, is_sparse=True)
    return reduction_da


@pytest.mark.parametrize('cols', [
    None,                                           # every key, with no test.
    ['c2v', 'c0b'],                                 # fewer columns than a row has keys: a loop over the columns.
    ['c2v', 'c0b', 'c1v', 'c0v', 'note', 'zz'],     # more: every key is added, and the columns are picked.
    'c0b',                                          # one column, by name.
    ])
def test_reduce_sparse_row_plain_sum_matches_sum_da(cols):
    """ The totals are those of sum_da(). With cols, they are in the order of cols, with only the keys found. """
    result_da: dict = SPARSE_SUM_DAF.reduce(Daf.sum_da, by='sparse_row', indirect_col='cv', cols=cols)
    expect_da: dict = sparse_sum_by_sum_da(SPARSE_SUM_DAF, [cols] if isinstance(cols, str) else cols)
    assert result_da == expect_da
    if cols is None:
        assert list(result_da) == list(expect_da)
    else:
        assert list(result_da) == [col for col in ([cols] if isinstance(cols, str) else cols) if col in expect_da]


def test_reduce_sparse_row_plain_sum_values():
    """ Text and None are skipped. The floats are added in row order. """
    assert SPARSE_SUM_DAF.reduce(Daf.sum_da, by='sparse_row', indirect_col='cv') == \
        {'c1v': 1, 'c0b': 3, 'c2v': 0.1 + 0.2, 'c0v': 3}
    assert SPARSE_SUM_DAF.reduce(Daf.sum_da, by='sparse_row', indirect_col='cv', cols=['c0v', 'c1v', 'zz']) == \
        {'c0v': 3, 'c1v': 1}
    assert Daf(cols=['ballot_id', 'cv']).reduce(Daf.sum_da, by='sparse_row', indirect_col='cv', cols=['c0b']) == {}


def test_sparse_sum_plan_is_chosen_from_the_keys_per_row():
    """ No cols adds every key. Cols no longer than the mean keys per row are looped over. Longer cols are picked at the end. """
    sample_lodict: list = [{'a': 1, 'b': 1, 'c': 1}, {'a': 1}]                 # 2 keys per row.
    assert Daf._sparse_sum_plan(None, sample_lodict) == (None, None)
    assert Daf._sparse_sum_plan(['b', 'a'], sample_lodict) == (('b', 'a'), ['b', 'a'])
    assert Daf._sparse_sum_plan(['c', 'b', 'a'], sample_lodict) == (None, ['c', 'b', 'a'])
    assert Daf._sparse_sum_plan(['a', 'a'], sample_lodict) == (('a',), ['a'])  # a repeated name counts once.


def test_reduce_sparse_row_other_options_keep_the_row_loop():
    """ With initial_da, the totals start from it, and its other keys stay, as before. """
    assert SPARSE_SUM_DAF.reduce(Daf.sum_da, by='sparse_row', indirect_col='cv', cols=['c0b'], initial_da={'k': 5, 'c0b': 10}) == \
        {'k': 5, 'c0b': 13}


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


def test_groupby_reduce_sparse_row_keeps_each_group_concise():
    """ Each group's dict is one cell, with only its own keys. A key the first group lacks is kept. """
    daf = Daf(lol=[['p1', "{'a': 1}"], ['p2', "{'a': 2, 'b': 5}"], ['p1', "{'a': 3}"]], cols=['g', 'cv'])
    result = daf.groupby_reduce('g', Daf.sum_da, by='sparse_row', indirect_col='cv')
    assert result.columns() == ['g', 'cv']
    assert result.keyfield == 'g'
    assert result.lol == [['p1', {'a': 4}], ['p2', {'a': 2, 'b': 5}]]


def test_groupby_reduce_sparse_row_with_reduce_cols_is_in_their_order():
    """ Each group's dict holds the reduce_cols it has, in their order. """
    daf = Daf(lol=[['p1', "{'a': 1, 'c': 7}"], ['p2', "{'a': 2, 'b': 5}"], ['p1', "{'b': 3}"]], cols=['g', 'cv'])
    result = daf.groupby_reduce('g', Daf.sum_da, by='sparse_row', indirect_col='cv', reduce_cols=['b', 'a'])
    assert result.lol == [['p1', {'b': 3, 'a': 1}], ['p2', {'b': 5, 'a': 2}]]


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


# ---- from test_daf_sum_by_col.py ----
# reduce() with sum_da() or sum_kl(), and so daf_sum() and groupby_cols_reduce(), adds down
# each column with _sum_by_col() instead of across each row. These tests check that the
# totals are the same as the row by row path gives, also for floats, where the order of
# the additions matters.

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


# ---- from test_daf_reduce_klist.py ----
# With by='row', reduce() gives the function a new KeyedList for each row, and uses
# sum_kl() and count_values_kl() in place of sum_da() and count_values_da(). These
# tests check that the results are what the dict rows gave before.

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


def _klist_tables() -> list:
    return [
        Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y']),
        Daf(lol=[['r1', 1, 'abc', ''], ['r2', 2.5, '', 'x'], ['r3', True, 4, 'x']], cols=['k', 'n', 'm', 'g'], keyfield='k'),
        Daf(lol=[['r1', 1, 2], ['r2', 3]], cols=['k', 'n', 'm'], keyfield='k'),
        Daf(lol=[['r1', 1, 2, 99], ['r2', 3, 4]], cols=['k', 'n', 'm']),
        Daf(lol=[['r1', -1, 0.5]], cols=['k', 'n', 'm']),
    ]


KLIST_COLS: list = [None, ['x', 'y'], 'x', ['n', 'm'], 'n', ['zz'], 'zz', []]


@pytest.mark.parametrize('cols', KLIST_COLS, ids=repr)
@pytest.mark.parametrize('itable', range(len(_klist_tables())))
@pytest.mark.parametrize('func', [Daf.sum_da, Daf.sum_kl], ids=['sum_da', 'sum_kl'])
def test_sum_matches_dict_rows(func: Any, itable: int, cols: Any) -> None:
    daf = _klist_tables()[itable]
    assert daf.reduce(func, cols=cols) == _reduce_with_dicts(daf, Daf.sum_da, cols=cols)


@pytest.mark.parametrize('cols', [['x', 'y'], ['n', 'm']], ids=repr)
@pytest.mark.parametrize('itable', range(len(_klist_tables())))
def test_sum_with_astype_matches_dict_rows(itable: int, cols: Any) -> None:
    daf = _klist_tables()[itable]
    assert daf.reduce(Daf.sum_da, cols=cols, astype=float) == _reduce_with_dicts(daf, Daf.sum_da, cols=cols, astype=float)


def test_sum_with_initial_da() -> None:
    daf = _klist_tables()[0]
    assert daf.reduce(Daf.sum_da, cols=['x'], initial_da={'x': 100}) == {'g': '', 'x': 106, 'y': ''}


@pytest.mark.parametrize('omit_nulls', [False, True])
@pytest.mark.parametrize('itable', [0, 1])
@pytest.mark.parametrize('func', [Daf.count_values_da, Daf.count_values_kl], ids=['count_values_da', 'count_values_kl'])
def test_count_values_matches_dict_rows(func: Any, itable: int, omit_nulls: bool) -> None:
    daf = _klist_tables()[itable]
    cols = daf.columns()[1:]
    assert daf.reduce(func, cols=cols, omit_nulls=omit_nulls) == \
        _reduce_with_dicts(daf, Daf.count_values_da, cols=cols, omit_nulls=omit_nulls)


def test_count_values_of_dict_cells() -> None:
    daf = Daf(lol=[['r1', {'a': 1, 'b': 2}], ['r2', {'a': 5}]], cols=['k', 'd'])
    assert daf.reduce(Daf.count_values_kl, cols=['d']) == {'k': '', 'd': {'a': 6, 'b': 2}}


@pytest.mark.parametrize('func', [Daf.count_values_da, Daf.count_values_kl], ids=['count_values_da', 'count_values_kl'])
def test_count_values_of_list_cells_raises(func: Any) -> None:
    """ A list cannot be counted. The error names the column. Leaving it out with cols works. """
    daf = Daf(lol=[['r1', ['a', 'b'], 'x'], ['r2', ['a'], 'x'], ['r3', 'z', 'y']], cols=['k', 'v', 'g'])
    with pytest.raises(TypeError, match="column 'v' holds a list, which cannot be counted") as exc_info:
        daf.reduce(func, cols=['v', 'g'])
    assert isinstance(exc_info.value.__cause__, TypeError)
    assert daf.reduce(func, cols=['g']) == {'k': '', 'v': '', 'g': {'x': 2, 'y': 1}}


@pytest.mark.parametrize('name', ['count_values_da', 'count_values_kl'])
def test_count_values_called_directly_names_itself(name: str) -> None:
    with pytest.raises(TypeError, match=name + r"\(\): column 'v'"):
        getattr(Daf, name)(KeyedList(['v'], [[1]]), {}, ['v'])


def test_daf_sum_and_daf_valuecount() -> None:
    daf = _klist_tables()[0]
    assert daf.daf_sum(cols=['x', 'y']) == {'g': '', 'x': 6, 'y': 60}
    assert daf.daf_valuecount(cols=['g'])['g'] == {'a': 2, 'b': 1}


def test_groupby_cols_reduce_unchanged() -> None:
    daf = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
    result = daf.groupby_cols_reduce(['g'], Daf.sum_da, reduce_cols=['x', 'y'])
    assert result.to_lod() == [{'g': 'a', 'x': 4, 'y': 40}, {'g': 'b', 'x': 2, 'y': 20}]


def test_each_row_is_a_new_keyedlist() -> None:
    """ A function that keeps the rows gets each row, not the last row many times. """
    daf = _klist_tables()[0]
    rows: list = []
    def keep(row: Any, reduction_da: T_da, cols: Any) -> T_da:
        rows.append(row)
        return reduction_da
    daf.reduce(keep)
    assert all(type(row) is KeyedList for row in rows)
    assert [dict(row) for row in rows] == daf.to_lod()


def test_writing_to_the_row_changes_the_table() -> None:
    daf = _klist_tables()[0]
    def zero_x(row: Any, reduction_da: T_da, cols: Any) -> T_da:
        row['x'] = 0
        return reduction_da
    daf.reduce(zero_x)
    assert daf.col('x') == [0, 0, 0]


def test_short_row_is_padded_with_null() -> None:
    daf = _klist_tables()[2]
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
    daf = _klist_tables()[0]
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
