# test_keyedlist.py

import unittest
import pytest

from daffodil.keyedlist import KeyedList
from daffodil.keyedlist import KeyedIndex


class TestKeyedList(unittest.TestCase):
    def test_initialization_from_keys_and_values(self):
        keys = ['a', 'b', 'c']
        values = [1, 2, 3]
        keyedlist = KeyedList(keys, values)
        self.assertEqual(len(keyedlist), 3)
        self.assertEqual(keyedlist['a'], 1)
        self.assertEqual(keyedlist['b'], 2)
        self.assertEqual(keyedlist['c'], 3)
    
    def test_initialization_from_dict(self):
        da = {'a': 1, 'b': 2, 'c': 3}
        keyedlist = KeyedList(da)
        self.assertEqual(len(keyedlist), 3)
        self.assertEqual(keyedlist['a'], 1)
        self.assertEqual(keyedlist['b'], 2)
        self.assertEqual(keyedlist['c'], 3)
    
    def test_initialization_from_dex_and_values(self):
        dex = {'a': 0, 'b': 1, 'c': 2}
        values = [1, 2, 3]
        keyedlist = KeyedList(dex, values)
        self.assertEqual(len(keyedlist), 3)
        self.assertEqual(keyedlist['a'], 1)
        self.assertEqual(keyedlist['b'], 2)
        self.assertEqual(keyedlist['c'], 3)
    
    def test_get_set_del_item(self):
        keyedlist = KeyedList(['a', 'b', 'c'], [1, 2, 3])
        keyedlist['b'] = 5
        self.assertEqual(keyedlist['b'], 5)
        del keyedlist['c']
        self.assertNotIn('c', keyedlist)
        self.assertEqual(keyedlist.to_dict(), {'a': 1, 'b': 5})
    
    def test_keys_values_items(self):
        keyedlist = KeyedList(['a', 'b', 'c'], [1, 2, 3])
        self.assertEqual(list(keyedlist.keys()), ['a', 'b', 'c'])
        self.assertEqual(list(keyedlist.values()), [1, 2, 3])
        self.assertEqual(list(keyedlist.items()), [('a', 1), ('b', 2), ('c', 3)])
    
    def test_update(self):
        keyedlist = KeyedList(['a', 'b', 'c'], [1, 2, 3])
        keyedlist.update({'b': 5, 'c': 6})
        self.assertEqual(keyedlist['b'], 5)
        self.assertEqual(keyedlist['c'], 6)
    
    def test_to_dict(self):
        keyedlist = KeyedList(['a', 'b', 'c'], [1, 2, 3])
        self.assertEqual(keyedlist.to_dict(), {'a': 1, 'b': 2, 'c': 3})


# Core construction tests


def test_keyedindex_from_list():
    kidx = KeyedIndex(['a', 'b', 'c'])

    assert len(kidx) == 3
    assert kidx['a'] == 0
    assert kidx['b'] == 1
    assert kidx['c'] == 2

def test_keyedindex_from_tuple():
    kidx = KeyedIndex(('x', 'y'))

    assert list(kidx) == ['x', 'y']
    assert kidx['y'] == 1

def test_keyedindex_from_dict_keys():
    d = {'a': 10, 'b': 20}
    kidx = KeyedIndex(d)

    assert list(kidx) == ['a', 'b']
    assert kidx['a'] == 0

def test_keyedindex_from_dict_keys():
    d = {'a': 10, 'b': 20}
    kidx = KeyedIndex(d)

    assert list(kidx) == ['a', 'b']
    assert kidx['a'] == 0

def test_keyedindex_from_keysview():
    d = {'k1': 1, 'k2': 2}
    kidx = KeyedIndex(d.keys())

    assert list(kidx) == ['k1', 'k2']

# Identity reuse

def test_keyedindex_reuse_existing():
    k1 = KeyedIndex(['a', 'b'])
    k2 = KeyedIndex(k1)

    assert k1 is not k2
    assert k1 == k2

# Lookup behavior

def test_keyedindex_contains():
    kidx = KeyedIndex(['a', 'b'])

    assert 'a' in kidx
    assert 'z' not in kidx

def test_keyedindex_getitem_missing():
    kidx = KeyedIndex(['a'])

    with pytest.raises(KeyError):
        _ = kidx['missing']

def test_keyedindex_get_method():
    kidx = KeyedIndex(['a'])

    assert kidx.get('a') == 0
    assert kidx.get('x') is None
    assert kidx.get('x', -1) == -1

# Iteration and ordering

def test_keyedindex_iteration_order():
    keys = ['c', 'a', 'b']
    kidx = KeyedIndex(keys)

    assert list(kidx) == keys


def test_keyedindex_keys_method():
    kidx = KeyedIndex(['a', 'b'])

    assert list(kidx.keys()) == ['a', 'b']


# Length and truthiness
def test_keyedindex_len_and_bool():
    kidx = KeyedIndex(['a'])

    assert len(kidx) == 1
    assert bool(kidx)

    empty = KeyedIndex([])
    assert not empty


# Duplicate handling (critical)

def test_keyedindex_duplicate_keys_raises():
    with pytest.raises(Exception):
        KeyedIndex(['a', 'b', 'a'])

# (Use ValueError if that’s what you enforce.)

# Mixed key types
def test_keyedindex_mixed_types():
    kidx = KeyedIndex(['a', 1, (2, 3)])

    assert kidx['a'] == 0
    assert kidx[1] == 1
    assert kidx[(2, 3)] == 2


# Append behavior
def test_keyedindex_append():
    kidx = KeyedIndex(['a', 'b'])

    kidx.append('c')

    assert kidx['c'] == 2
    assert list(kidx) == ['a', 'b', 'c']


def test_keyedindex_append_duplicate_raises():
    kidx = KeyedIndex(['a'])

    with pytest.raises(Exception):
        kidx.append('a')


# Equality (strict)

def test_keyedindex_equality_same():
    k1 = KeyedIndex(['a', 'b'])
    k2 = KeyedIndex(['a', 'b'])

    assert k1 == k2


def test_keyedindex_equality_different_order():
    k1 = KeyedIndex(['a', 'b'])
    k2 = KeyedIndex(['b', 'a'])

    assert k1 != k2


def test_keyedindex_not_equal_to_dict():
    kidx = KeyedIndex(['a', 'b'])
    d = {'a': 0, 'b': 1}

    assert (kidx == d) is False or (kidx == d) is NotImplemented


# Representation
def test_keyedindex_repr():
    kidx = KeyedIndex(['a', 'b'])

    r = repr(kidx)
    assert 'a' in r
    assert 'b' in r


# Integration with KeyedList assumption

def test_keyedindex_used_for_lookup_alignment():
    keys = ['id', 'grp']
    kidx = KeyedIndex(keys)

    row = [10, 'A']

    assert row[kidx['id']] == 10
    assert row[kidx['grp']] == 'A'

# Optional performance sanity (non-strict)
def test_keyedindex_large_build():
    keys = list(range(10000))
    kidx = KeyedIndex(keys)

    assert kidx[9999] == 9999    

if __name__ == '__main__':
    unittest.main()
