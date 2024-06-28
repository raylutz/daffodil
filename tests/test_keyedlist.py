# test_keyedlist.py

import unittest
from daffodil.keyedlist import KeyedList



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

if __name__ == '__main__':
    unittest.main()
