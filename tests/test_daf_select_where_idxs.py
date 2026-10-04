# test_daf_select_where_idxs.py
#
# select_where_idxs() gives its function a KeyedList for each row, as select_where() does.

import pytest

from daffodil.daf import Daf, KeysDisabledError
from daffodil.keyedlist import KeyedList


def _daf() -> Daf:
    return Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'a', 30]], cols=['id', 'v', 'n'], keyfield='id')


def test_the_function_gets_a_keyedlist_for_every_row():
    seen = []
    _daf().select_where_idxs(lambda row: seen.append(type(row)) or True)
    assert seen == [KeyedList] * 3


def test_the_function_gets_a_keyedlist_in_every_itermode():
    d = _daf()
    d._itermode = Daf.ITERMODE_DICT
    seen = []
    d.select_where_idxs(lambda row: seen.append(type(row)) or True)
    assert seen == [KeyedList] * 3


def test_positions_of_the_matching_rows():
    d = _daf()
    assert d.select_where_idxs(lambda row: row['v'] == 'a') == [0, 2]
    assert d.select_where_idxs(lambda row: row['n'] > 10) == [1, 2]
    assert d.select_where_idxs(lambda row: row['n'] > 99) == []
    assert d.select_where_idxs(lambda row: True) == [0, 1, 2]


def test_positions_agree_with_select_where():
    d = _daf()
    test = lambda row: row['n'] >= 20 and row['v'] == 'a'
    assert [d.lol[i] for i in d.select_where_idxs(test)] == d.select_where(test).lol


def test_on_a_slice_the_positions_are_those_of_the_slice():
    d = _daf()
    assert d[1:].select_where_idxs(lambda row: row['v'] == 'a') == [1]


def test_empty_daf_gives_no_positions():
    assert Daf(cols=['id', 'v']).select_where_idxs(lambda row: True) == []


def test_daf_with_rows_and_no_column_names_raises():
    with pytest.raises(KeysDisabledError):
        Daf(lol=[[1, 2]]).select_where_idxs(lambda row: True)
