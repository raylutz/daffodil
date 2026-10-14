# test_daf_select_records_daf_keys.py
# Tests for select_records_daf() given another Daf as the keys (Ray, 2026-10-14).

import pytest

from daffodil.daf import Daf, KeysDisabledError


def make_daf() -> Daf:
    return Daf(lol=[['r1', 'a'], ['r2', 'b'], ['r3', 'c']], cols=['id', 'v'], keyfield='id')


def make_other_daf() -> Daf:
    return Daf(lol=[['r3', 1], ['r9', 2], ['r1', 3]], cols=['id', 'w'], keyfield='id')


def test_a_daf_as_keys_gives_the_same_as_its_keys():
    data_daf: Daf = make_daf()
    other_daf: Daf = make_other_daf()
    assert data_daf.select_records_daf(other_daf, silent_error=True).lol == \
           data_daf.select_records_daf(other_daf.keys(), silent_error=True).lol == [['r3', 'c'], ['r1', 'a']]


def test_a_daf_as_keys_with_inverse():
    data_daf: Daf = make_daf()
    assert data_daf.select_records_daf(make_other_daf(), inverse=True, silent_error=True).lol == [['r2', 'b']]


def test_a_daf_as_keys_missing_key_raises_unless_silent():
    with pytest.raises(KeyError):
        make_daf().select_records_daf(make_other_daf())


def test_a_daf_as_keys_shares_rows_and_leaves_the_other_unchanged():
    data_daf: Daf = make_daf()
    other_daf: Daf = make_other_daf()
    selected_daf: Daf = data_daf.select_records_daf(other_daf, silent_error=True)
    assert selected_daf.lol[0] is data_daf.lol[2]
    assert other_daf.lol == [['r3', 1], ['r9', 2], ['r1', 3]]


def test_a_daf_without_keyfield_as_keys_raises():
    no_key_daf: Daf = Daf(lol=[['r1']], cols=['id'])
    with pytest.raises(KeysDisabledError):
        make_daf().select_records_daf(no_key_daf)


def test_an_empty_daf_as_keys():
    empty_daf: Daf = Daf(cols=['id'], keyfield='id')
    assert make_daf().select_records_daf(empty_daf).lol == []
    assert make_daf().select_records_daf(empty_daf, inverse=True).lol == make_daf().lol
