# test_daf_append_list.py
#
# append() with a list, on a Daf that has columns. The list is added in column order
# without building a dict. These tests pin the behavior that path must keep.

import pytest

from daffodil.daf import Daf


def _daf(keyfield='') -> Daf:
    return Daf(lol=[['a', 1], ['b', 2]], cols=['k', 'v'], keyfield=keyfield)


def test_append_list_adds_a_copy():
    d = _daf()
    row = ['c', 3]
    d.append(row)
    row[1] = 99
    assert d.lol[-1] == ['c', 3]
    assert d.lol[-1] is not row


def test_append_la_adds_a_copy():
    d = _daf()
    row = ['c', 3]
    d.append(la=row)
    assert d.lol[-1] == ['c', 3]
    assert d.lol[-1] is not row


def test_append_short_list_is_padded_with_null():
    d = _daf()
    d.append(['c'])
    assert d.lol[-1] == ['c', '']


def test_append_long_list_raises():
    d = _daf()
    with pytest.raises(ValueError):
        d.append(['c', 3, 'extra'])
    assert d.num_rows() == 2


def test_append_list_with_keyfield_updates_key_lookup():
    d = _daf(keyfield='k')
    assert d.select_record('b') == {'k': 'b', 'v': 2}     # builds the key index
    d.append(['c', 3])
    assert d.select_record('c') == {'k': 'c', 'v': 3}


def test_append_list_with_existing_key_adds_a_second_row():
    d = _daf(keyfield='k')
    d.append(['a', 10])
    assert d.num_rows() == 3
    assert d.lol[-1] == ['a', 10]


def test_append_list_respect_kd_replaces_the_row_with_the_same_key():
    d = _daf(keyfield='k')
    d.append(['a', 10], respect_kd=True)
    assert d.num_rows() == 2
    assert d.lol[0] == ['a', 10]
    d.append(['c', 3], respect_kd=True)
    assert d.num_rows() == 3
    assert d.select_record('c') == {'k': 'c', 'v': 3}


def test_append_list_with_tuple_keyfield():
    d = Daf(lol=[['a', 1, 'x']], cols=['k1', 'k2', 'v'], keyfield=('k1', 'k2'))
    d.append(['b', 2, 'y'])
    assert d.select_record(('b', 2))['v'] == 'y'


def test_append_list_to_daf_without_columns_adds_the_list_itself():
    d = Daf()
    row = [1, 2]
    d.append(row)
    assert d.lol == [[1, 2]]


def test_append_list_matches_append_dict():
    a = _daf(keyfield='k')
    b = _daf(keyfield='k')
    for row in (['c', 3], ['d'], ['a', 7]):
        a.append(row)
        b.append(dict(zip(['k', 'v'], row)))
    assert a.lol == b.lol
    assert a.keys() == b.keys()
