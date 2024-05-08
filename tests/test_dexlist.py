# test_dexlist.py

import unittest
from daffodil.dexlist import DexList



class TestDexList(unittest.TestCase):
    def test_initialization_from_keys_and_values(self):
        keys = ['a', 'b', 'c']
        values = [1, 2, 3]
        dexlist = DexList(keys, values)
        self.assertEqual(len(dexlist), 3)
        self.assertEqual(dexlist['a'], 1)
        self.assertEqual(dexlist['b'], 2)
        self.assertEqual(dexlist['c'], 3)
    
    def test_initialization_from_dict(self):
        da = {'a': 1, 'b': 2, 'c': 3}
        dexlist = DexList(da)
        self.assertEqual(len(dexlist), 3)
        self.assertEqual(dexlist['a'], 1)
        self.assertEqual(dexlist['b'], 2)
        self.assertEqual(dexlist['c'], 3)
    
    def test_initialization_from_dex_and_values(self):
        dex = {'a': 0, 'b': 1, 'c': 2}
        values = [1, 2, 3]
        dexlist = DexList(dex, values)
        self.assertEqual(len(dexlist), 3)
        self.assertEqual(dexlist['a'], 1)
        self.assertEqual(dexlist['b'], 2)
        self.assertEqual(dexlist['c'], 3)
    
    def test_get_set_del_item(self):
        dexlist = DexList(['a', 'b', 'c'], [1, 2, 3])
        dexlist['b'] = 5
        self.assertEqual(dexlist['b'], 5)
        del dexlist['c']
        self.assertNotIn('c', dexlist)
    
    def test_keys_values_items(self):
        dexlist = DexList(['a', 'b', 'c'], [1, 2, 3])
        self.assertEqual(list(dexlist.keys()), ['a', 'b', 'c'])
        self.assertEqual(list(dexlist.values), [1, 2, 3])
        self.assertEqual(list(dexlist.items()), [('a', 1), ('b', 2), ('c', 3)])
    
    def test_update(self):
        dexlist = DexList(['a', 'b', 'c'], [1, 2, 3])
        dexlist.update({'b': 5, 'c': 6})
        self.assertEqual(dexlist['b'], 5)
        self.assertEqual(dexlist['c'], 6)
    
    def test_to_dict(self):
        dexlist = DexList(['a', 'b', 'c'], [1, 2, 3])
        self.assertEqual(dexlist.to_dict(), {'a': 1, 'b': 2, 'c': 3})

if __name__ == '__main__':
    unittest.main()
