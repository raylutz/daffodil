# test_daf_iter_generators.py
#
# iter_dict(), iter_klist() and iter_list() return generators. These tests check the
# rows they give, the error for a Daf with no column names, and how a loop behaves
# when rows are appended, or a new list of rows is put in place, during the loop.

import pytest
from daffodil.daf import Daf, KeysDisabledError
from daffodil.keyedlist import KeyedList


def _daf() -> Daf:
    return Daf(lol=[['a', 1], ['b', 2], ['c', 3]], cols=['k', 'n'], keyfield='k')


def test_iter_dict_rows_are_copies() -> None:
    daf = _daf()
    rows = list(daf.iter_dict())
    assert rows == [{'k': 'a', 'n': 1}, {'k': 'b', 'n': 2}, {'k': 'c', 'n': 3}]
    rows[0]['n'] = 99
    assert daf.lol[0] == ['a', 1]


def test_iter_klist_rows_point_at_the_table() -> None:
    daf = _daf()
    rows = list(daf.iter_klist())
    assert all(type(row) is KeyedList for row in rows)
    assert [dict(row) for row in rows] == daf.to_lod()
    rows[1]['n'] = 20
    assert daf.lol[1] == ['b', 20]


def test_iter_list_rows_are_the_table_rows() -> None:
    daf = _daf()
    assert all(row is daf_row for row, daf_row in zip(daf.iter_list(), daf.lol))


def test_for_row_in_daf_follows_itermode() -> None:
    daf = _daf()
    assert type(next(iter(daf))) is dict
    daf.itermode = 'keyedlist'
    assert type(next(iter(daf))) is KeyedList


@pytest.mark.parametrize('method', ['iter_dict', 'iter_klist'])
def test_no_column_names_raises_at_the_call(method: str) -> None:
    daf = Daf(lol=[[1, 2]])
    with pytest.raises(KeysDisabledError, match='needs column names'):
        getattr(daf, method)()


def test_no_column_names_iter_list_works() -> None:
    assert list(Daf(lol=[[1, 2]]).iter_list()) == [[1, 2]]


@pytest.mark.parametrize('method', ['iter_dict', 'iter_klist', 'iter_list'])
def test_empty_daf(method: str) -> None:
    assert list(getattr(Daf(), method)()) == []


def test_rows_appended_during_the_loop_are_seen() -> None:
    daf = Daf(lol=[[1], [2]], cols=['a'])
    seen = []
    for row in daf:
        seen.append(row['a'])
        if row['a'] < 4:
            daf.append({'a': row['a'] + 2})
    assert seen == [1, 2, 3, 4, 5]


def test_a_sort_during_the_loop_does_not_change_the_loop() -> None:
    daf = Daf(lol=[[3], [1], [2]], cols=['a'])
    seen = []
    for row in daf:
        seen.append(row['a'])
        if len(seen) == 1:
            daf.sort_by_colname('a')
    assert seen == [3, 1, 2]
    assert daf.col('a') == [1, 2, 3]


def test_an_iterator_goes_through_the_rows_once() -> None:
    it = _daf().iter_dict()
    assert len(list(it)) == 3
    assert list(it) == []


def test_each_loop_over_the_daf_starts_again() -> None:
    daf = _daf()
    assert [row['n'] for row in daf] == [row['n'] for row in daf] == [1, 2, 3]


def test_iter_klist_short_row_raises_as_before() -> None:
    daf = Daf(lol=[['a', 1], ['b']], cols=['k', 'n'])
    with pytest.raises(ValueError, match='same length'):
        list(daf.iter_klist())


def test_iter_dict_short_row_gives_fewer_keys_as_before() -> None:
    daf = Daf(lol=[['a', 1], ['b']], cols=['k', 'n'])
    assert list(daf.iter_dict()) == [{'k': 'a', 'n': 1}, {'k': 'b'}]
