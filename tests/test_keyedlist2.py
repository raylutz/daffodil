# test_keyedlist2.py
#
# Supplementary pytest-style tests for keyedlist.py, targeting coverage gaps not
# exercised by the existing (unittest-style) test_keyedlist.py: constructor branches,
# error paths, KeyedIndex edge cases, astype_la, and KeyedListEncoder.

import json

import pytest

from daffodil.keyedlist import KeyedList, KeyedIndex, KeyedListEncoder, astype_la


# --- KeyedList construction branches ---

def test_init_from_hd_and_values_length_mismatch():
    hd = KeyedIndex(['a', 'b', 'c'])
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


def test_init_from_keyedindex_and_values_happy_path():
    hd = KeyedIndex(['a', 'b', 'c'])
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


# --- astype_la (module-level function) ---

def test_astype_la_none_passthrough():
    la = [1, 2, 3]
    assert astype_la(la, None) is la


def test_astype_la_callable():
    assert astype_la([1, 2, 3], lambda x: x * 2) == [2, 4, 6]


def test_astype_la_type_object():
    assert astype_la(['1', '2'], int) == [1, 2]


def test_astype_la_str_int():
    assert astype_la(['1', '2'], 'int') == [1, 2]


def test_astype_la_str_str():
    assert astype_la([1, 2], 'str') == ['1', '2']


def test_astype_la_str_float():
    assert astype_la(['1.5', '2.5'], 'float') == [1.5, 2.5]


def test_astype_la_str_bool():
    assert astype_la([0, 1], 'bool') == [False, True]


def test_astype_la_unsupported_string_raises():
    with pytest.raises(ValueError):
        astype_la([1, 2], 'not_a_type')


def test_astype_la_unsupported_object_raises():
    with pytest.raises(ValueError):
        astype_la([1, 2], 42)


# --- KeyedListEncoder ---

def test_keyedlist_encoder_serializes_keyedlist():
    klist = KeyedList(['a', 'b'], [1, 2])
    encoded = json.dumps({'row': klist}, cls=KeyedListEncoder)
    decoded = json.loads(encoded)
    assert decoded['row']['__KeyedList__'] is True
    assert decoded['row']['values'] == [1, 2]


def test_keyedlist_encoder_passthrough_for_other_types():
    with pytest.raises(TypeError):
        json.dumps({'thing': object()}, cls=KeyedListEncoder)


# --- KeyedIndex gaps ---

def test_keyedindex_init_with_none():
    kidx = KeyedIndex(None)
    assert len(kidx) == 0
    assert bool(kidx) is False


def test_keyedindex_init_from_keyedlist():
    klist = KeyedList(['a', 'b'], [1, 2])
    kidx = KeyedIndex(klist)
    assert kidx['a'] == 0
    assert kidx['b'] == 1


def test_keyedindex_init_unsupported_type_raises():
    with pytest.raises(TypeError):
        KeyedIndex(42)


def test_keyedindex_index_method():
    kidx = KeyedIndex(['a', 'b', 'c'])
    assert kidx.index('b') == 1


def test_keyedindex_to_dict():
    kidx = KeyedIndex(['a', 'b'])
    assert kidx.to_dict() == {'a': 0, 'b': 1}
