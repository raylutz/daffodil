# test_keyedlist2.py
#
# Supplementary pytest-style tests for keyedlist.py, targeting coverage gaps not
# exercised by the existing (unittest-style) test_keyedlist.py: constructor branches,
# error paths, and _astype_la.

import json

import pytest

from daffodil.keyedlist import KeyedList, _astype_la
from daffodil.lib import daf_utils
from daffodil.daf import Daf
from daffodil.keyedlist import KeyedList


# --- KeyedList construction branches ---

def test_init_from_hd_and_values_length_mismatch():
    hd = {'a': 0, 'b': 1, 'c': 2}
    with pytest.raises(ValueError):
        KeyedList(hd, [1, 2])


def test_init_from_keys_and_values_length_mismatch():
    with pytest.raises(ValueError):
        KeyedList(['a', 'b', 'c'], [1, 2])


def test_init_from_keys_and_default():
    klist = KeyedList(['a', 'b', 'c'], default=0)
    assert len(klist) == 3
    assert klist['a'] == 0
    assert klist['b'] == 0
    assert klist['c'] == 0


def test_init_from_keys_and_default_none():
    klist = KeyedList(['a', 'b'])
    assert klist['a'] is None
    assert klist['b'] is None


def test_init_from_existing_keyedlist():
    original = KeyedList(['a', 'b'], [1, 2])
    copy = KeyedList(original)
    assert copy['a'] == 1
    assert copy['b'] == 2
    assert list(copy.keys()) == ['a', 'b']


def test_init_empty():
    klist = KeyedList()
    assert len(klist) == 0
    assert bool(klist) is False
    assert list(klist.keys()) == []


def test_init_from_dict_and_values_length_mismatch():
    with pytest.raises(ValueError):
        KeyedList({'a': 1, 'b': 2}, [1, 2, 3])


def test_init_from_hd_and_values_happy_path():
    hd = {'a': 0, 'b': 1, 'c': 2}
    values = [1, 2, 3]
    klist = KeyedList(hd, values)
    assert klist['a'] == 1
    assert klist.hd is hd          # reused, not rebuilt
    assert klist.values() is values  # direct reference, not copied


def test_init_invalid_args_raises():
    with pytest.raises(ValueError):
        KeyedList(arg1=42, arg2=None)


# --- __getitem__ ---

def test_getitem_with_list_of_keys():
    klist = KeyedList(['a', 'b', 'c'], [1, 2, 3])
    assert klist[['a', 'c']] == [1, 3]


def test_getitem_with_list_of_keys_skips_missing():
    klist = KeyedList(['a', 'b', 'c'], [1, 2, 3])
    assert klist[['a', 'missing', 'c']] == [1, 3]


def test_getitem_with_unhashable_key_raises():
    klist = KeyedList(['a', 'b'], [1, 2])
    with pytest.raises(ValueError):
        klist[{'not': 'hashable'}]


# --- __setitem__ ---

def test_setitem_extends_with_new_key():
    klist = KeyedList(['a', 'b'], [1, 2])
    klist['c'] = 3
    assert len(klist) == 3
    assert klist['c'] == 3
    assert klist.to_dict() == {'a': 1, 'b': 2, 'c': 3}


def test_setitem_overwrites_existing_key():
    klist = KeyedList(['a', 'b'], [1, 2])
    klist['a'] = 99
    assert klist['a'] == 99
    assert len(klist) == 2


# --- set_values ---

def test_set_values_happy_path():
    klist = KeyedList(['a', 'b', 'c'], [1, 2, 3])
    klist.set_values([10, 20, 30])
    assert klist.values() == [10, 20, 30]


def test_set_values_wrong_type_raises():
    klist = KeyedList(['a', 'b'], [1, 2])
    with pytest.raises(TypeError):
        klist.set_values((1, 2))


def test_set_values_wrong_length_raises():
    klist = KeyedList(['a', 'b'], [1, 2])
    with pytest.raises(ValueError):
        klist.set_values([1, 2, 3])


# --- values(astype=...) ---

def test_values_astype_int():
    klist = KeyedList(['a', 'b'], ['1', '2'])
    assert klist.values(astype=int) == [1, 2]


def test_values_astype_str_keyword():
    klist = KeyedList(['a', 'b'], [1, 2])
    assert klist.values(astype='str') == ['1', '2']


# --- get ---

def test_get_missing_key_returns_default():
    klist = KeyedList(['a', 'b'], [1, 2])
    assert klist.get('missing') is None
    assert klist.get('missing', -1) == -1


def test_get_existing_key():
    klist = KeyedList(['a', 'b'], [1, 2])
    assert klist.get('a') == 1


# --- __repr__ ---

def test_repr():
    klist = KeyedList(['a', 'b'], [1, 2])
    assert repr(klist) == repr({'a': 1, 'b': 2})


# --- to_json / from_json ---

def test_to_json_and_from_json_round_trip():
    klist = KeyedList(['a', 'b'], [1, 2])
    json_str = klist.to_json()
    restored = KeyedList.from_json(json_str)
    assert restored.to_dict() == {'a': 1, 'b': 2}


def test_from_json_invalid_payload_raises():
    bad_json = json.dumps({"not_a_keyedlist": True})
    with pytest.raises(ValueError):
        KeyedList.from_json(bad_json)


# --- _astype_la (module-level function) ---

def test_astype_la_none_passthrough():
    la = [1, 2, 3]
    assert _astype_la(la, None) is la


def test_astype_la_callable():
    assert _astype_la([1, 2, 3], lambda x: x * 2) == [2, 4, 6]


def test_astype_la_type_object():
    assert _astype_la(['1', '2'], int) == [1, 2]


def test_astype_la_str_int():
    assert _astype_la(['1', '2'], 'int') == [1, 2]


def test_astype_la_str_str():
    assert _astype_la([1, 2], 'str') == ['1', '2']


def test_astype_la_str_float():
    assert _astype_la(['1.5', '2.5'], 'float') == [1.5, 2.5]


def test_astype_la_str_bool():
    assert _astype_la([0, 1], 'bool') == [False, True]


def test_astype_la_unsupported_string_raises():
    with pytest.raises(ValueError):
        _astype_la([1, 2], 'not_a_type')


def test_astype_la_unsupported_object_raises():
    with pytest.raises(ValueError):
        _astype_la([1, 2], 42)


# --- a KeyedList that shares an hd must not change it when adding a key ---

def test_adding_a_key_does_not_change_a_shared_index():
    hd = {'a': 0, 'b': 1}
    first, second = KeyedList(hd, [1, 2]), KeyedList(hd, [3, 4])
    first['c'] = 5
    assert list(first.keys()) == ['a', 'b', 'c']
    assert list(second.keys()) == ['a', 'b']
    assert list(hd.keys()) == ['a', 'b']


def test_adding_a_second_key_does_not_copy_the_index_again():
    klist = KeyedList({'a': 0}, [1])
    klist['b'] = 2
    own = klist.hd
    klist['c'] = 3
    assert klist.hd is own
    assert klist.to_dict() == {'a': 1, 'b': 2, 'c': 3}


def test_a_keyedlist_that_built_its_own_index_still_extends_it():
    from daffodil.keyedlist import KeyedList
    klist = KeyedList({'a': 1})
    own = klist.hd
    klist['b'] = 2
    assert klist.hd is own


# ---- from test_keyedlist_astype_null.py ----
# KeyedList.values(astype) and _astype_la() keep an empty cell, which is NULL, as it is. This is the rule of daf_utils.astype_la().

@pytest.mark.parametrize('astype', [int, float, str, bool, 'int', 'float', 'str', 'bool', lambda v: v + v])
def test_an_empty_cell_is_kept_for_every_kind_of_astype(astype):
    assert _astype_la(['1', '', '2'], astype)[1] == ''


def test_astype_la_converts_the_other_cells():
    assert _astype_la(['1', '', '3'], int) == [1, '', 3]
    assert _astype_la(['1.5', ''], 'float') == [1.5, '']
    assert _astype_la([0, '', 1], 'bool') == [False, '', True]


def test_values_keeps_an_empty_cell():
    assert KeyedList(['a', 'b'], ['1', '']).values(int) == [1, '']
    assert KeyedList(['a', 'b'], ['', '2.5']).values('float') == ['', 2.5]


def test_values_of_a_daf_row_with_a_missing_cell():
    from daffodil.daf import Daf
    d = Daf(lol=[['1', ''], ['3', '4']], cols=['a', 'b'])
    assert [row.values(int) for row in d.iter_klist()] == [[1, ''], [3, 4]]


def test_a_cell_that_is_not_empty_and_cannot_be_converted_still_raises():
    with pytest.raises(ValueError):
        KeyedList(['a'], ['x']).values(int)


def test_none_is_not_an_empty_cell():
    with pytest.raises(TypeError):
        KeyedList(['a'], [None]).values(int)


def test_unsupported_astype_still_raises_even_for_an_empty_list():
    with pytest.raises(ValueError, match='astype not supported'):
        _astype_la([], 'date')
    with pytest.raises(ValueError, match='astype not supported'):
        _astype_la([], 42)


def test_astype_la_none_returns_the_same_list():
    la = ['1', '']
    assert _astype_la(la, None) is la


@pytest.mark.parametrize('astype', [int, float, str, bool, 'int', 'float', 'str', 'bool'])
def test_the_keyedlist_version_agrees_with_the_daf_utils_version(astype):
    la = ['1', '', '0', '2']
    assert _astype_la(la, astype) == daf_utils.astype_la(la, astype)


# ---- from test_keyedlist_construction_and_lookup.py ----
# KeyedList tests for the paths that a Daf loop uses for every row. The hd plus row case is tested first in
# __init__, and one hashable key is tried first in __getitem__. These tests pin the results for every input kind.

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
