# test_daf_delete_records.py
# Tests for delete_records() (Ray, 2026-10-14, decision 104 and item 2 of the mdaf review: inplace=False by default).

import pytest

from daffodil.daf import Daf, KeysDisabledError


def make_daf() -> Daf:
    return Daf(lol=[['r1', 'a'], ['r2', 'b'], ['r3', 'c'], ['r4', 'd']], cols=['id', 'v'], keyfield='id')


def test_default_returns_a_new_daf_and_leaves_this_one():
    data_daf: Daf = make_daf()
    result_daf: Daf = data_daf.delete_records(['r2', 'r4'])
    assert result_daf.lol == [['r1', 'a'], ['r3', 'c']]
    assert data_daf.lol == make_daf().lol
    assert result_daf.lol[0] is data_daf.lol[0]
    assert result_daf == data_daf.select_records_daf(['r2', 'r4'], inverse=True)


def test_inplace_deletes_from_this_daf_and_returns_it():
    data_daf: Daf = make_daf()
    row_list_la: list = data_daf.lol
    result_daf: Daf = data_daf.delete_records(['r4', 'r2'], inplace=True)
    assert result_daf is data_daf
    assert data_daf.lol == [['r1', 'a'], ['r3', 'c']]
    assert data_daf.lol is row_list_la
    assert data_daf['r3', 'v'].to_value() == 'c'
    assert data_daf.keys() == ['r1', 'r3']


def test_inplace_leaves_an_earlier_selection_whole():
    data_daf: Daf = make_daf()
    sel_daf: Daf = data_daf.select_where(lambda row: True)
    data_daf.delete_records(['r2'], inplace=True)
    assert sel_daf.lol == make_daf().lol


def test_missing_key_raises_and_removes_nothing():
    data_daf: Daf = make_daf()
    with pytest.raises(KeyError):
        data_daf.delete_records(['r2', 'r9'], inplace=True)
    assert data_daf.lol == make_daf().lol
    with pytest.raises(KeyError):
        data_daf.delete_records(['r9'])


def test_silent_error_ignores_missing_keys():
    data_daf: Daf = make_daf()
    assert data_daf.delete_records(['r2', 'r9'], silent_error=True).lol == [['r1', 'a'], ['r3', 'c'], ['r4', 'd']]
    data_daf.delete_records(['r9'], inplace=True, silent_error=True)
    assert data_daf.lol == make_daf().lol


def test_a_daf_as_keys():
    data_daf: Daf = make_daf()
    other_daf: Daf = Daf(lol=[['r3', 0], ['r1', 0]], cols=['id', 'x'], keyfield='id')
    assert data_daf.delete_records(other_daf).lol == [['r2', 'b'], ['r4', 'd']]
    data_daf.delete_records(other_daf, inplace=True)
    assert data_daf.lol == [['r2', 'b'], ['r4', 'd']]
    with pytest.raises(KeysDisabledError):
        data_daf.delete_records(Daf(lol=[['r2']], cols=['id']))


def test_no_keys():
    data_daf: Daf = make_daf()
    assert data_daf.delete_records([]).lol == make_daf().lol
    assert data_daf.delete_records([], inplace=True) is data_daf
    assert data_daf.lol == make_daf().lol
