# test_keyedlist_construction_and_lookup.py
#
# KeyedList tests for the paths that a Daf loop uses for every row. The hd plus row case is tested first in
# __init__, and one hashable key is tried first in __getitem__. These tests pin the results for every input kind.

import pytest

from daffodil.daf import Daf
from daffodil.keyedlist import KeyedList


def _kl() -> KeyedList:
    return KeyedList({'a': 0, 'b': 1, 'c': 2}, [10, 20, 30])


# construction

def test_hd_plus_row_shares_the_row_and_the_index():
    ki = {'a': 0, 'b': 1}
    row = [1, 2]
    kl = KeyedList(ki, row)
    assert kl._values is row and kl.hd is ki


def test_hd_plus_row_still_copies_the_index_before_a_key_is_added():
    ki = {'a': 0, 'b': 1, 'c': 2}
    kl = KeyedList(ki, [1, 2, 3])
    kl['new'] = 4
    assert list(ki) == ['a', 'b', 'c']
    assert list(kl) == ['a', 'b', 'c', 'new']


def test_hd_plus_row_of_the_wrong_length_raises():
    with pytest.raises(ValueError, match='same length'):
        KeyedList({'a': 0, 'b': 1, 'c': 2}, [1, 2])


@pytest.mark.parametrize('args', [({'a': 0, 'b': 1, 'c': 2}, (1, 2, 3)), (5,)])
def test_unsupported_constructor_arguments_raise(args):
    with pytest.raises(ValueError, match='Must provide'):
        KeyedList(*args)


def test_the_other_constructor_forms_still_work():
    assert KeyedList({'a': 1}).values() == [1]
    assert KeyedList({'a': 0}, [5])['a'] == 5           # a dict with a list is an hd: key to position
    assert KeyedList(['a', 'b'], [1, 2]).values() == [1, 2]
    assert KeyedList(['a', 'b'], default=0).values() == [0, 0]
    assert KeyedList(KeyedList({'a': 1})).values() == [1]
    assert KeyedList().values() == []


# lookup

def test_lookup_by_one_key():
    assert _kl()['b'] == 20


def test_lookup_by_a_list_of_keys_skips_the_missing_ones():
    kl = _kl()
    assert kl[['a', 'c']] == [10, 30]
    assert kl[['a', 'zz']] == [10]
    assert kl[[]] == []


@pytest.mark.parametrize('key', ['zz', ('a',), 1, None])
def test_lookup_of_a_missing_key_raises_keyerror(key):
    with pytest.raises(KeyError):
        _kl()[key]


@pytest.mark.parametrize('key', [{'a': 1}, {1}])
def test_lookup_by_a_key_that_cannot_be_hashed_raises_valueerror(key):
    with pytest.raises(ValueError):
        _kl()[key]


def test_the_rows_of_a_daf_loop_read_by_name():
    d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'], keyfield='id')
    assert [kl['v'] for kl in d.iter_klist()] == ['a', 'b']
    assert [kl[['id', 'v']] for kl in d.iter_klist()] == [[1, 'a'], [2, 'b']]


# rows of a Daf share its hd

def test_rows_of_a_daf_share_its_hd():
    d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'])
    row0 = d.iloc(0, rtype='klist')
    assert row0.hd is d.hd
    assert all(row.hd is d.hd for row in d.iter_klist())


def test_adding_or_deleting_a_key_on_a_row_does_not_change_the_daf():
    d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'])
    row = d.iloc(0, rtype='klist')
    row['extra'] = 9
    assert row.hd is not d.hd
    assert d.columns() == ['id', 'v']
    row2 = d.iloc(1, rtype='klist')
    del row2['v']
    assert row2.hd is not d.hd
    assert d.columns() == ['id', 'v'] and d.hd == {'id': 0, 'v': 1}


def test_a_row_made_before_the_columns_change_no_longer_shares_the_hd():
    d = Daf(lol=[[1, 'a']], cols=['id', 'v'])
    row = d.iloc(0, rtype='klist')
    d.insert_col('n', [0])
    assert row.hd is not d.hd


def test_duplicate_keys_raise():
    with pytest.raises(ValueError, match='Duplicate'):
        KeyedList(['a', 'a'], [1, 2])
