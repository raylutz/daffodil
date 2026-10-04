# test_keyedlist_construction_and_lookup.py
#
# KeyedList tests for the paths that a Daf loop uses for every row. The hd plus row case is tested first in
# __init__, and one hashable key is tried first in __getitem__. These tests pin the results for every input kind.

import pytest

from daffodil.daf import Daf
from daffodil.keyedlist import KeyedList, KeyedIndex


def _kl() -> KeyedList:
    return KeyedList(KeyedIndex(['a', 'b', 'c']), [10, 20, 30])


# construction

def test_hd_plus_row_shares_the_row_and_the_index():
    ki = KeyedIndex(['a', 'b'])
    row = [1, 2]
    kl = KeyedList(ki, row)
    assert kl._values is row and kl.hd is ki


def test_hd_plus_row_still_copies_the_index_before_a_key_is_added():
    ki = KeyedIndex(['a', 'b', 'c'])
    kl = KeyedList(ki, [1, 2, 3])
    kl['new'] = 4
    assert list(ki) == ['a', 'b', 'c']
    assert list(kl) == ['a', 'b', 'c', 'new']


def test_hd_plus_row_of_the_wrong_length_raises():
    with pytest.raises(ValueError, match='same length'):
        KeyedList(KeyedIndex(['a', 'b', 'c']), [1, 2])


@pytest.mark.parametrize('args', [(KeyedIndex(['a']),), (KeyedIndex(['a', 'b', 'c']), (1, 2, 3)), (5,)])
def test_unsupported_constructor_arguments_raise(args):
    with pytest.raises(ValueError, match='Must provide'):
        KeyedList(*args)


def test_the_other_constructor_forms_still_work():
    assert KeyedList({'a': 1}).values() == [1]
    assert KeyedList({'a': 1}, [5]).values() == [5]
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
