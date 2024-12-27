# test_daf
# copyright (c) 2024 Ray Lutz

import os
import json
import unittest
import numpy as np
import pandas as pd
#from io import BytesIO
from pathlib import Path

import sys
# sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'src'))
# print(sys.path)

# Get the path to the 'src' directory
src_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Append the path only if it's not already in sys.path
if src_path not in sys.path:
    sys.path.append(src_path)

from daffodil.daf import Daf
from daffodil.keyedlist import KeyedList
from daffodil.lib import daf_utils as utils

class TestDaf(unittest.TestCase):

    maxDiff = None
    
    # initialization
    def test_init_default_values(self):
        mydaf = Daf()
        self.assertEqual(mydaf.name, '')
        self.assertEqual(mydaf.keyfield, '')
        self.assertEqual(mydaf.hd, {})
        self.assertEqual(mydaf.lol, [])
        self.assertEqual(mydaf.kd, {})
        self.assertEqual(mydaf.dtypes, {})
        self.assertEqual(mydaf._iter_index, 0)

    def test_init_custom_values(self):
        cols = ['col1', 'col2']
        hd = {'col1': 0, 'col2': 1}
        lol = [[1, 2], [3, 4]]
        kd = {1: 0, 3: 1}
        dtypes = {'col1': int, 'col2': str}
        expected_lol = [[1, '2'], [3, '4']]
        daf = Daf(cols=cols, lol=lol, dtypes=dtypes, name='TestDaf', keyfield='col1').apply_dtypes(from_str=False)
        self.assertEqual(daf.name, 'TestDaf')
        self.assertEqual(daf.keyfield, 'col1')
        self.assertEqual(daf.hd, hd)
        self.assertEqual(daf.lol, expected_lol)
        self.assertEqual(daf.kd, kd)
        self.assertEqual(daf.dtypes, dtypes)
        self.assertEqual(daf._iter_index, 0)

    def test_init_no_cols_but_dtypes(self):
        #cols = ['col1', 'col2']
        hd = {'col1': 0, 'col2': 1}
        lol = [[1, 2], [3, 4]]
        kd = {1: 0, 3: 1}
        dtypes = {'col1': int, 'col2': str}
        expected_lol = [[1, '2'], [3, '4']]
        daf = Daf(lol=lol, dtypes=dtypes, name='TestDaf', keyfield='col1').apply_dtypes(from_str=False)
        self.assertEqual(daf.name, 'TestDaf')
        self.assertEqual(daf.keyfield, 'col1')
        self.assertEqual(daf.hd, hd)
        self.assertEqual(daf.lol, expected_lol)
        self.assertEqual(daf.kd, kd)
        self.assertEqual(daf.dtypes, dtypes)
        self.assertEqual(daf._iter_index, 0)

    # shape
    def test_shape_empty(self):
        # Test shape method with an empty Daf object
        daf = Daf()
        self.assertEqual(daf.shape(), (0, 0))

    def test_shape_non_empty(self):
        # Test shape method with a non-empty Daf object
        data = [[1, 'A'], [2, 'B'], [3, 'C']]
        cols = ['Col1', 'Col2']
        daf = Daf(lol=data, cols=cols)
        self.assertEqual(daf.shape(), (3, 2))

    def test_shape_no_colnames(self):
        # Test shape method with a Daf object initialized without colnames
        data = [[1, 'A'], [2, 'B'], [3, 'C']]
        daf = Daf(lol=data)
        self.assertEqual(daf.shape(), (3, 2))

    def test_shape_empty_data(self):
        # Test shape method with a Daf object initialized with empty data
        cols = ['Col1', 'Col2']
        daf = Daf(cols=cols)
        self.assertEqual(daf.shape(), (0, 0))

    def test_shape_empty_data_specified(self):
        # Test shape method with a Daf object initialized with empty data
        cols = ['Col1', 'Col2']
        daf = Daf(lol=[], cols=cols)
        self.assertEqual(daf.shape(), (0, 0))

    def test_shape_empty_data_specified_empty_col(self):
        # Test shape method with a Daf object initialized with empty data
        cols = ['Col1', 'Col2']
        daf = Daf(lol=[[]], cols=cols)
        self.assertEqual(daf.shape(), (1, 0))

    def test_shape_no_colnames_no_cols(self):
        # Test shape method with a Daf object initialized without colnames
        data = [[], [], []]
        daf = Daf(lol=data)
        self.assertEqual(daf.shape(), (3, 0))

    def test_shape_colnames_no_cols_empty_rows(self):
        # Test shape method with a Daf object initialized without colnames
        data = [[], [], []]
        # cols = ['Col1', 'Col2']
        daf = Daf(lol=data)
        self.assertEqual(daf.shape(), (3, 0))

    # __eq__
    
    def test_eq_different_type(self):
        # Test __eq__ method with a different type
        daf = Daf()
        other = "not a Daf object"
        self.assertFalse(daf == other)

    def test_eq_different_data(self):
        # Test __eq__ method with a Daf object with different data
        daf1 = Daf(lol=[[1, 'A'], [2, 'B'], [3, 'C']], cols=['Col1', 'Col2'], keyfield='Col1')
        daf2 = Daf(lol=[[1, 'X'], [2, 'Y'], [3, 'Z']], cols=['Col1', 'Col2'], keyfield='Col1')
        self.assertFalse(daf1 == daf2)

    def test_eq_different_columns(self):
        # Test __eq__ method with a Daf object with different columns
        daf1 = Daf(lol=[[1, 'A'], [2, 'B'], [3, 'C']], cols=['Col1', 'Col2'], keyfield='Col1')
        daf2 = Daf(lol=[[1, 'A'], [2, 'B'], [3, 'C']], cols=['Col1', 'Col3'], keyfield='Col1')
        self.assertFalse(daf1 == daf2)

    def test_eq_different_keyfield(self):
        # Test __eq__ method with a Daf object with different keyfield
        daf1 = Daf(lol=[[1, 'A'], [2, 'B'], [3, 'C']], cols=['Col1', 'Col2'], keyfield='Col1')
        daf2 = Daf(lol=[[1, 'A'], [2, 'B'], [3, 'C']], cols=['Col1', 'Col2'], keyfield='Col2')
        self.assertFalse(daf1 == daf2)

    def test_eq_equal(self):
        # Test __eq__ method with equal Daf objects
        daf1 = Daf(lol=[[1, 'A'], [2, 'B'], [3, 'C']], cols=['Col1', 'Col2'], keyfield='Col1')
        daf2 = Daf(lol=[[1, 'A'], [2, 'B'], [3, 'C']], cols=['Col1', 'Col2'], keyfield='Col1')
        self.assertTrue(daf1 == daf2)

    # len(daf), .len(), .shape(), .num_cols()
    def test_len(self):
        # Test case with an empty dictionary
        my_daf = Daf()
        assert len(my_daf) == 0
        assert my_daf.len() == 0
        assert my_daf.num_cols() == 0
        assert my_daf.shape() == (0, 0)

        # Test case with a dictionary containing one key-value pair
        my_daf.append({'a': 1, 'b': 2, 'c': 3})
        assert len(my_daf) == 1
        assert my_daf.len() == 1
        assert my_daf.num_cols() == 3
        assert my_daf.shape() == (1, 3)

        # Test case with a dictionary containing multiple key-value pairs
        my_daf.append({'a': 4, 'b': 5, 'c': 6})
        my_daf.append({'a': 7, 'b': 8, 'c': 9})
        assert len(my_daf) == 3
        assert my_daf.len() == 3
        assert my_daf.num_cols() == 3
        assert my_daf.shape() == (3, 3)

        # Test case with a dictionary containing keys of mixed types
        my_daf.append({'a': 1, 2: 'b', 'c': True})
        assert len(my_daf) == 4
        assert my_daf.len() == 4
        assert my_daf.num_cols() == 3
        assert my_daf.shape() == (4, 3)


    # calc_cols
    def test_calc_cols_include_cols(self):
        # Test calc_cols method with include_cols parameter
        data = [
            [1, 'A', True],
            [2, 'B', False],
            [3, 'C', True]
        ]
        columns = ['Col1', 'Col2', 'Col3']
        types = {'Col1': int, 'Col2': str, 'Col3': bool}
        daf = Daf(lol=data, cols=columns, dtypes=types)
        included_cols = daf.calc_cols(include_cols=['Col1', 'Col3'])
        self.assertEqual(included_cols, ['Col1', 'Col3'])

    def test_calc_cols_exclude_cols(self):
        # Test calc_cols method with exclude_cols parameter
        data = [
            [1, 'A', True],
            [2, 'B', False],
            [3, 'C', True]
        ]
        columns = ['Col1', 'Col2', 'Col3']
        types = {'Col1': int, 'Col2': str, 'Col3': bool}
        daf = Daf(lol=data, cols=columns, dtypes=types)
        excluded_cols = daf.calc_cols(exclude_cols=['Col2'])
        self.assertEqual(excluded_cols, ['Col1', 'Col3'])

    def test_calc_cols_include_types(self):
        # Test calc_cols method with include_types parameter
        data = [
            [1, 'A', True],
            [2, 'B', False],
            [3, 'C', True]
        ]
        columns = ['Col1', 'Col2', 'Col3']
        types = {'Col1': int, 'Col2': str, 'Col3': bool}
        daf = Daf(lol=data, cols=columns, dtypes=types)
        included_types = daf.calc_cols(include_types=[int])
        self.assertEqual(included_types, ['Col1'])

    def test_calc_cols_exclude_types(self):
        # Test calc_cols method with exclude_types parameter
        data = [
            [1, 'A', True],
            [2, 'B', False],
            [3, 'C', True]
        ]
        columns = ['Col1', 'Col2', 'Col3']
        types = {'Col1': int, 'Col2': str, 'Col3': bool}
        daf = Daf(lol=data, cols=columns, dtypes=types)
        excluded_types = daf.calc_cols(exclude_types=[str])
        self.assertEqual(excluded_types, ['Col1', 'Col3'])

    def test_calc_cols_complex(self):
        # Test calc_cols method with multiple parameters
        data = [
            [1, 'A', True],
            [2, 'B', False],
            [3, 'C', True]
        ]
        columns = ['Col1', 'Col2', 'Col3']
        types = {'Col1': int, 'Col2': str, 'Col3': bool}
        daf = Daf(lol=data, cols=columns, dtypes=types)
        selected_cols = daf.calc_cols(include_cols=['Col1', 'Col2'],
                                       exclude_cols=['Col2'],
                                       include_types=[int, bool])
        self.assertEqual(selected_cols, ['Col1'])

    # rename_cols
    def test_rename_cols(self):
        # Test rename_cols method to rename columns
        data = [
            [1, 'A', True],
            [2, 'B', False],
            [3, 'C', True]
        ]
        columns = ['Col1', 'Col2', 'Col3']
        types = {'Col1': int, 'Col2': str, 'Col3': bool}
        daf = Daf(lol=data, cols=columns, dtypes=types)
        
        # Rename columns using the provided dictionary
        from_to_dict = {'Col1': 'NewCol1', 'Col3': 'NewCol3'}
        daf.rename_cols(from_to_dict)
        
        # Check if columns are renamed correctly
        expected_columns = ['NewCol1', 'Col2', 'NewCol3']
        self.assertEqual(daf.columns(), expected_columns)

        # Check if dtypes are updated correctly
        expected_types = {'NewCol1': int, 'Col2': str, 'NewCol3': bool}
        self.assertEqual(daf.dtypes, expected_types)

    def test_rename_cols_with_keyfield(self):
        # Test rename_cols method when a keyfield is specified
        data = [
            [1, 'A', True],
            [2, 'B', False],
            [3, 'C', True]
        ]
        columns = ['Col1', 'Col2', 'Col3']
        types = {'Col1': int, 'Col2': str, 'Col3': bool}
        daf = Daf(lol=data, cols=columns, dtypes=types, keyfield='Col1')
        
        # Rename columns using the provided dictionary
        from_to_dict = {'Col1': 'NewCol1', 'Col3': 'NewCol3'}
        daf.rename_cols(from_to_dict)
        
        # Check if keyfield is updated correctly -- requires manual updating of keyfield
        self.assertEqual(daf.keyfield, '')
        

    # set_cols
    def test_set_cols_no_existing_cols(self):
        # Test setting column names when there are no existing columns
        daf = Daf()
        new_cols = ['A', 'B', 'C']
        daf.set_cols(new_cols)
        self.assertEqual(daf.hd, {'A': 0, 'B': 1, 'C': 2})
    
    def test_set_cols_generate_spreadsheet_names(self):
        # Test generating spreadsheet-like column names
        daf = Daf(cols=['col1', 'col2'])
        daf.set_cols()
        self.assertEqual(daf.hd, {'A': 0, 'B': 1})
    
    def test_set_cols_with_existing_cols(self):
        # Test setting column names with existing columns
        daf = Daf(cols=['col1', 'col2'])
        new_cols = ['A', 'B']
        daf.set_cols(new_cols)
        self.assertEqual(daf.hd, {'A': 0, 'B': 1})
    
    def test_set_cols_repair_keyfield(self):
        # Test repairing keyfield if column names are already defined
        daf = Daf(cols=['col1', 'col2'], keyfield='col1')
        new_cols = ['A', 'B']
        daf.set_cols(new_cols)
        self.assertEqual(daf.keyfield, '')
    
    def test_set_cols_update_dtypes(self):
        # Test updating dtypes dictionary with new column names
        daf = Daf(cols=['col1', 'col2'], dtypes={'col1': int, 'col2': str})
        new_cols = ['A', 'B']
        daf.set_cols(new_cols)
        self.assertEqual(daf.dtypes, {'A': int, 'B': str})
    
    def test_set_cols_error_length_mismatch(self):
        # Test error handling when lengths of new column names don't match existing ones
        daf = Daf(cols=['col1', 'col2'])
        new_cols = ['A']  # Length mismatch
        with self.assertRaises(AttributeError):
            daf.set_cols(new_cols)

    def test_set_cols_sanitize(self):
        # sanitizing function
        daf = Daf(cols=['col1', 'col2', 'col3'])
        new_cols = ['A', 'A', '']
        daf.set_cols(new_cols)
        self.assertEqual(daf.columns(), ['A', 'A_1', 'col2'])
    
    def test_set_cols_sanitize_dif_prefix(self):
        # sanitizing function, different prefix
        daf = Daf(cols=['col1', 'col2', 'col3'])
        new_cols = ['A', 'A', '']
        daf.set_cols(new_cols, unnamed_prefix='Unnamed')
        self.assertEqual(daf.columns(), ['A', 'A_1', 'Unnamed2'])
    


    # keys
    def test_keys_no_keyfield(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        daf = Daf(cols=cols, lol=lol, keyfield='', dtypes={'col1': int, 'col2': str})

        result = daf.keys()

        self.assertEqual(result, [])

    def test_keys_with_keyfield(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        result = daf.keys()

        self.assertEqual(result, [1, 2, 3])

    def test_keys_empty_daf(self):
        cols = []
        lol = []
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={})

        result = daf.keys()

        self.assertEqual(result, [])  

    # set_keyfield
    def test_set_keyfield_existing_column(self):
        # Test setting keyfield to an existing column
        daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['ID', 'Value'])
        
        # breakpoint()
        # pass
        
        daf.set_keyfield('ID')
        self.assertEqual(daf.keyfield, 'ID')
    
    def test_set_keyfield_empty_string(self):
        # Test setting keyfield to an empty string
        daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['ID', 'Value'], keyfield='ID')
        daf.set_keyfield('')
        self.assertEqual(daf.keyfield, '')
    
    def test_set_keyfield_nonexistent_column(self):
        # Test trying to set keyfield to a nonexistent column
        daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['ID', 'Value'])
        with self.assertRaises(KeyError):
            daf.set_keyfield('nonexistent_column', silent_error=False)

    # # row_idx_of (DEPRECATED)
    # def test_row_idx_of_existing_key(self):
        # # Test getting row index of an existing key
        # daf = Daf(lol=[['1', 'a'], ['2', 'b']], cols=['ID', 'Value'], keyfield='ID')
        # self.assertEqual(daf.row_idx_of('1'), 0)
    
    # def test_row_idx_of_nonexistent_key(self):
        # # Test getting row index of a nonexistent key
        # daf = Daf(lol=[['1', 'a'], ['2', 'b']], cols=['ID', 'Value'], keyfield='ID')
        # self.assertEqual(daf.row_idx_of('3'), -1)
    
    # def test_row_idx_of_no_keyfield(self):
        # # Test getting row index when no keyfield is defined
        # daf = Daf(lol=[['1', 'a'], ['2', 'b']], cols=['ID', 'Value'])
        # self.assertEqual(daf.row_idx_of('1'), -1)
    
    # def test_row_idx_of_no_kd(self):
        # # Test getting row index when kd is not available
        # daf = Daf(lol=[['1', 'a'], ['2', 'b']], cols=['ID', 'Value'], keyfield='ID')
        # daf.kd = None
        # self.assertEqual(daf.row_idx_of('1'), -1)


    # get_existing_keys
    def test_get_existing_keys_with_existing_keys(self):
        # Test case where all keys in keylist exist in kd
        daf = Daf(lol=[[1, 'a'], [2, 'b'], [3, 'c']], cols=['ID', 'Name'], keyfield='Name')
        existing_keys = daf.get_existing_keys(['a', 'b', 'd'])
        self.assertEqual(existing_keys, ['a', 'b'])

    def test_get_existing_keys_with_no_existing_keys(self):
        # Test case where no keys in keylist exist in kd
        daf = Daf(lol=[[1, 'a'], [2, 'b'], [3, 'c']], cols=['ID', 'Name'], keyfield='Name')
        existing_keys = daf.get_existing_keys(['d', 'e', 'f'])
        self.assertEqual(existing_keys, [])

    def test_get_existing_keys_with_empty_keylist(self):
        # Test case where keylist is empty
        daf = Daf(lol=[[1, 'a'], [2, 'b'], [3, 'c']], cols=['ID', 'Name'], keyfield='Name')
        existing_keys = daf.get_existing_keys([])
        self.assertEqual(existing_keys, [])

    def test_get_existing_keys_with_empty_daf(self):
        # Test case where Daf is empty
        daf = Daf()
        existing_keys = daf.get_existing_keys(['a', 'b', 'c'])
        self.assertEqual(existing_keys, [])



    # calc cols
    def test_calc_cols_with_include_cols(self):
        # Test case where include_cols is provided
        daf = Daf(lol=[[1, 'a'], [2, 'b'], [3, 'c']], cols=['ID', 'Name'], dtypes={'ID': int, 'Name': str})
        self.assertEqual(daf.calc_cols(include_cols=['ID']), ['ID'])

    def test_calc_cols_with_exclude_cols(self):
        # Test case where exclude_cols is provided
        daf = Daf(lol=[[1, 'a'], [2, 'b'], [3, 'c']], cols=['ID', 'Name'], dtypes={'ID': int, 'Name': str})
        self.assertEqual(daf.calc_cols(exclude_cols=['ID']), ['Name'])

    def test_calc_cols_with_include_types(self):
        # Test case where include_types is provided
        daf = Daf(lol=[[1, 'a'], [2, 'b'], [3, 'c']], cols=['ID', 'Name'], dtypes={'ID': int, 'Name': str})
        self.assertEqual(daf.calc_cols(include_types=[str]), ['Name'])

    def test_calc_cols_with_exclude_types(self):
        # Test case where exclude_types is provided
        daf = Daf(lol=[[1, 'a'], [2, 'b'], [3, 'c']], cols=['ID', 'Name'], dtypes={'ID': int, 'Name': str})
        self.assertEqual(daf.calc_cols(exclude_types=[str]), ['ID'])

    def test_calc_cols_with_empty_daf(self):
        # Test case where Daf is empty
        daf = Daf()
        self.assertEqual(daf.calc_cols(), [])
        
    def test_calc_cols_with_include_cols_large(self):
        # Test case where include_cols is provided with more than 10 columns
        daf = Daf(lol=[[1, 'a', True, 1, 'a', True, 1, 'a', True, 1, 'a', True], 
                         [2, 'b', False, 2, 'b', False, 2, 'b', False, 2, 'b', False], 
                         [3, 'c', True, 3, 'c', True, 3, 'c', True, 3, 'c', True]], 
                         cols=['ID', 'Name', 'Flag', 'ID2', 'Name2', 'Flag2', 'ID3', 'Name3', 'Flag3', 'ID4', 'Name4', 'Flag4'], 
                         dtypes={'ID': int, 'Name': str, 'Flag': bool, 
                                 'ID2': int, 'Name2': str, 'Flag2': bool, 
                                 'ID3': int, 'Name3': str, 'Flag3': bool, 
                                 'ID4': int, 'Name4': str, 'Flag4': bool})
        self.assertEqual(daf.calc_cols(include_cols=['ID', 'Name', 'Flag']), ['ID', 'Name', 'Flag'])

    def test_calc_cols_with_include_cols_large_include_large(self):
        # Test case where include_cols is provided with more than 10 columns
        daf = Daf(lol=[[1, 'a', True, 1, 'a', True, 1, 'a', True, 1, 'a', True], 
                         [2, 'b', False, 2, 'b', False, 2, 'b', False, 2, 'b', False], 
                         [3, 'c', True, 3, 'c', True, 3, 'c', True, 3, 'c', True]], 
                         cols=['ID', 'Name', 'Flag', 'ID2', 'Name2', 'Flag2', 'ID3', 'Name3', 'Flag3', 'ID4', 'Name4', 'Flag4'], 
                         dtypes={'ID': int, 'Name': str, 'Flag': bool, 
                                 'ID2': int, 'Name2': str, 'Flag2': bool, 
                                 'ID3': int, 'Name3': str, 'Flag3': bool, 
                                 'ID4': int, 'Name4': str, 'Flag4': bool})
        self.assertEqual(daf.calc_cols(include_cols=['ID', 'Name', 'Flag', 'ID2', 'Name2', 'Flag2', 'ID3', 'Name3', 'Flag3', 'ID4', 'Name4']), ['ID', 'Name', 'Flag', 'ID2', 'Name2', 'Flag2', 'ID3', 'Name3', 'Flag3', 'ID4', 'Name4'])

    def test_calc_cols_with_exclude_cols_large_enclude_large(self):
        # Test case where exclude_cols is provided with more than 10 columns
        daf = Daf(lol=[[1, 'a', True, 1, 'a', True, 1, 'a', True, 1, 'a', True], 
                         [2, 'b', False, 2, 'b', False, 2, 'b', False, 2, 'b', False], 
                         [3, 'c', True, 3, 'c', True, 3, 'c', True, 3, 'c', True]], 
                         cols=['ID', 'Name', 'Flag', 'ID2', 'Name2', 'Flag2', 'ID3', 'Name3', 'Flag3', 'ID4', 'Name4', 'Flag4'], 
                         dtypes={'ID': int, 'Name': str, 'Flag': bool, 
                                 'ID2': int, 'Name2': str, 'Flag2': bool, 
                                 'ID3': int, 'Name3': str, 'Flag3': bool, 
                                 'ID4': int, 'Name4': str, 'Flag4': bool})
        self.assertEqual(daf.calc_cols(exclude_cols=['ID', 'Name', 'Flag', 'ID2', 'Name2', 'Flag2', 'ID3', 'Name3', 'Flag3', 'ID4', 'Name4']), ['Flag4'])

    def test_calc_cols_with_exclude_cols_large(self):
        # Test case where exclude_cols is provided with more than 10 columns
        daf = Daf(lol=[[1, 'a', True, 1, 'a', True, 1, 'a', True, 1, 'a', True], 
                         [2, 'b', False, 2, 'b', False, 2, 'b', False, 2, 'b', False], 
                         [3, 'c', True, 3, 'c', True, 3, 'c', True, 3, 'c', True]], 
                         cols=['ID', 'Name', 'Flag', 'ID2', 'Name2', 'Flag2', 'ID3', 'Name3', 'Flag3', 'ID4', 'Name4', 'Flag4'], 
                         dtypes={'ID': int, 'Name': str, 'Flag': bool, 
                                 'ID2': int, 'Name2': str, 'Flag2': bool, 
                                 'ID3': int, 'Name3': str, 'Flag3': bool, 
                                 'ID4': int, 'Name4': str, 'Flag4': bool})
        self.assertEqual(daf.calc_cols(exclude_cols=['ID', 'Name', 'Flag']), ['ID2', 'Name2', 'Flag2', 'ID3', 'Name3', 'Flag3', 'ID4', 'Name4', 'Flag4'])

    def test_calc_cols_with_include_types_large(self):
        # Test case where include_types is provided with more than 10 columns
        daf = Daf(lol=[[1, 'a', True, 1, 'a', True, 1, 'a', True, 1, 'a', True], 
                         [2, 'b', False, 2, 'b', False, 2, 'b', False, 2, 'b', False], 
                         [3, 'c', True, 3, 'c', True, 3, 'c', True, 3, 'c', True]], 
                         cols=['ID', 'Name', 'Flag', 'ID2', 'Name2', 'Flag2', 'ID3', 'Name3', 'Flag3', 'ID4', 'Name4', 'Flag4'], 
                         dtypes={'ID': int, 'Name': str, 'Flag': bool, 
                                 'ID2': int, 'Name2': str, 'Flag2': bool, 
                                 'ID3': int, 'Name3': str, 'Flag3': bool, 
                                 'ID4': int, 'Name4': str, 'Flag4': bool})
        self.assertEqual(daf.calc_cols(include_types=[int, bool]), ['ID', 'Flag', 'ID2', 'Flag2', 'ID3', 'Flag3', 'ID4', 'Flag4'])

    def test_calc_cols_with_exclude_types_large(self):
        # Test case where exclude_types is provided with more than 10 columns
        daf = Daf(lol=[[1, 'a', True, 1, 'a', True, 1, 'a', True, 1, 'a', True], 
                         [2, 'b', False, 2, 'b', False, 2, 'b', False, 2, 'b', False], 
                         [3, 'c', True, 3, 'c', True, 3, 'c', True, 3, 'c', True]], 
                         cols=['ID', 'Name', 'Flag', 'ID2', 'Name2', 'Flag2', 'ID3', 'Name3', 'Flag3', 'ID4', 'Name4', 'Flag4'], 
                         dtypes={'ID': int, 'Name': str, 'Flag': bool, 
                                 'ID2': int, 'Name2': str, 'Flag2': bool, 
                                 'ID3': int, 'Name3': str, 'Flag3': bool, 
                                 'ID4': int, 'Name4': str, 'Flag4': bool})
        self.assertEqual(daf.calc_cols(exclude_types=[int, bool]), ['Name', 'Name2', 'Name3', 'Name4'])
       
    def test_calc_cols_with_exclude_types_large_exclude_nonlist(self):
        # Test case where exclude_types is provided with more than 10 columns
        daf = Daf(lol=[[1, 'a', True, 1, 'a', True, 1, 'a', True, 1, 'a', True], 
                         [2, 'b', False, 2, 'b', False, 2, 'b', False, 2, 'b', False], 
                         [3, 'c', True, 3, 'c', True, 3, 'c', True, 3, 'c', True]], 
                         cols=['ID', 'Name', 'Flag', 'ID2', 'Name2', 'Flag2', 'ID3', 'Name3', 'Flag3', 'ID4', 'Name4', 'Flag4'], 
                         dtypes={'ID': int, 'Name': str, 'Flag': bool, 
                                 'ID2': int, 'Name2': str, 'Flag2': bool, 
                                 'ID3': int, 'Name3': str, 'Flag3': bool, 
                                 'ID4': int, 'Name4': str, 'Flag4': bool})
        self.assertEqual(daf.calc_cols(exclude_types=int), ['Name', 'Flag', 'Name2', 'Flag2', 'Name3', 'Flag3', 'Name4', 'Flag4'])

    # from/to cases
    def test_from_lod(self):
        records_lod = [ {'col1': 1, 'col2': 2}, 
                        {'col1': 11, 'col2': 12}, 
                        {'col1': 21, 'col2': 22}]
                        
        keyfield = 'col1'
        dtypes = {'col1': int, 'col2': int}
        daf = Daf.from_lod(records_lod, keyfield=keyfield, dtypes=dtypes)

        self.assertEqual(daf.name, '')
        self.assertEqual(daf.keyfield, keyfield)
        self.assertEqual(daf.hd, {'col1': 0, 'col2': 1})
        self.assertEqual(daf.lol, [[1, 2], [11, 12], [21, 22]])
        self.assertEqual(daf.kd, {1: 0, 11: 1, 21: 2})
        self.assertEqual(daf.dtypes, dtypes)
        self.assertEqual(daf._iter_index, 0)

    def test_from_lod_no_records_but_dtypes(self):
        records_lod = []
                        
        keyfield = 'col1'
        dtypes = {'col1': int, 'col2': int}
        daf = Daf.from_lod(records_lod, keyfield=keyfield, dtypes=dtypes)

        self.assertEqual(daf.name, '')
        self.assertEqual(daf.keyfield, keyfield)
        self.assertEqual(daf.hd, {'col1': 0, 'col2': 1})
        self.assertEqual(daf.lol, [])
        self.assertEqual(daf.kd, {})
        self.assertEqual(daf.dtypes, dtypes)
        self.assertEqual(daf._iter_index, 0)

    def test_from_lod_no_records_no_dtypes_but_keyfield(self):
        records_lod = []
                        
        keyfield = 'col1'
        dtypes = {}
        daf = Daf.from_lod(records_lod, keyfield=keyfield, dtypes=dtypes)

        self.assertEqual(daf.name, '')
        self.assertEqual(daf.keyfield, keyfield)
        self.assertEqual(daf.hd, {})
        self.assertEqual(daf.lol, [])
        self.assertEqual(daf.kd, {})
        self.assertEqual(daf.dtypes, dtypes)
        self.assertEqual(daf._iter_index, 0)


    def test_from_lod_no_records_no_dtypes_no_keyfield(self):
        records_lod = []
                        
        keyfield = ''
        dtypes = {}
        daf = Daf.from_lod(records_lod, keyfield=keyfield, dtypes=dtypes)

        self.assertEqual(daf.name, '')
        self.assertEqual(daf.keyfield, '')
        self.assertEqual(daf.hd, {})
        self.assertEqual(daf.lol, [])
        self.assertEqual(daf.kd, {})
        self.assertEqual(daf.dtypes, dtypes)
        self.assertEqual(daf._iter_index, 0)


    def test_from_hllola(self):
        header_list = ['col1', 'col2']
        data_list = [[1, 'a'], [2, 'b'], [3, 'c']]
        hllola = (header_list, data_list)
        keyfield = 'col1'
        dtypes = {'col1': int, 'col2': str}

        daf = Daf.from_hllola(hllola, keyfield=keyfield, dtypes=dtypes)

        self.assertEqual(daf.name, '')
        self.assertEqual(daf.keyfield, keyfield)
        self.assertEqual(daf.hd, {'col1': 0, 'col2': 1})
        self.assertEqual(daf.lol, [[1, 'a'], [2, 'b'], [3, 'c']])
        self.assertEqual(daf.kd, {1: 0, 2: 1, 3: 2})
        self.assertEqual(daf.dtypes, dtypes)
        self.assertEqual(daf._iter_index, 0)

    def test_to_hllola(self):
        cols    = ['col1', 'col2']
        lol     = [[1, 'a'], [2, 'b'], [3, 'c']]
        daf    = Daf(cols=cols, lol=lol)

        expected_hllola = (['col1', 'col2'], [[1, 'a'], [2, 'b'], [3, 'c']])
        actual_hllola = daf.to_hllola()

        self.assertEqual(actual_hllola, expected_hllola)


    # from_dod
    def test_from_dod_with_keyfield(self):
        # Test case where dod is provided with keyfield specified
        dod = {
            'row_0': {'rowkey': 'row_0', 'data1': 1, 'data2': 2},
            'row_1': {'rowkey': 'row_1', 'data1': 11, 'data2': 22},
        }
        daf = Daf.from_dod(dod, keyfield='rowkey')
        self.assertEqual(daf.columns(), ['rowkey', 'data1', 'data2'])
        self.assertEqual(daf.lol, [['row_0', 1, 2], ['row_1', 11, 22]])
        self.assertEqual(daf.keyfield, 'rowkey')

    def test_from_dod_without_keyfield(self):
        # Test case where dod is provided without keyfield in rows
        dod = {
            'row_0': {'data1': 1, 'data2': 2},
            'row_1': {'data1': 11, 'data2': 22},
        }
        daf = Daf.from_dod(dod, keyfield='rowkey')
        self.assertEqual(daf.columns(), ['rowkey', 'data1', 'data2'])
        self.assertEqual(daf.lol, [['row_0', 1, 2], ['row_1', 11, 22]])

    def test_from_dod_with_dtypes(self):
        # Test case where dod is provided with dtypes specified
        dod = {
            'row_0': {'rowkey': 'row_0', 'data1': 1, 'data2': 2},
            'row_1': {'rowkey': 'row_1', 'data1': 11, 'data2': 22},
        }
        dtypes = {'rowkey': str, 'data1': int, 'data2': float}
        daf = Daf.from_dod(dod, keyfield='rowkey', dtypes=dtypes)
        self.assertEqual(daf.columns(), ['rowkey', 'data1', 'data2'])
        self.assertEqual(daf.lol, [['row_0', 1, 2.0], ['row_1', 11, 22.0]])

    def test_from_dod_with_empty_dod(self):
        # Test case where an empty dod is provided
        dod = {}
        daf = Daf.from_dod(dod)
        self.assertEqual(daf.columns(), [])
        self.assertEqual(daf.lol, [])

    # to_dod
    def test_to_dod_with_remove_keyfield_true(self):
        # Test case where keyfield column is removed (default behavior)
        daf = Daf(lol=[['1', 'a', True], 
                         ['2', 'b', False]], cols=['ID', 'Name', 'Flag'], keyfield='ID')
        expected_dod = {'1': {'Name': 'a', 'Flag': True}, '2': {'Name': 'b', 'Flag': False}}
        dod = daf.to_dod()
        self.assertEqual(dod, expected_dod)

    def test_to_dod_with_remove_keyfield_false(self):
        # Test case where keyfield column is retained
        daf = Daf(lol=[['1', 'a', True], ['2', 'b', False]], cols=['ID', 'Name', 'Flag'], keyfield='ID')
        expected_dod = {'1': {'ID': '1', 'Name': 'a', 'Flag': True}, '2': {'ID': '2', 'Name': 'b', 'Flag': False}}
        dod = daf.to_dod(remove_keyfield=False)
        self.assertEqual(dod, expected_dod)

    def test_to_dod_with_empty_daf(self):
        # Test case where Daf is empty
        daf = Daf()
        expected_dod = {}
        dod = daf.to_dod()
        self.assertEqual(dod, expected_dod)

    # to_cols_dol()
    def test_to_cols_dol_with_data(self):
        # Test case where Daf contains data
        daf = Daf(lol=[[1, 'a', True], [2, 'b', False]], cols=['ID', 'Name', 'Flag'])
        expected_dol = {'ID': [1, 2], 'Name': ['a', 'b'], 'Flag': [True, False]}
        dol = daf.to_cols_dol()
        self.assertEqual(dol, expected_dol)

    def test_to_cols_dol_with_empty_daf(self):
        # Test case where Daf is empty
        daf = Daf()
        expected_dol = {}
        dol = daf.to_cols_dol()
        self.assertEqual(dol, expected_dol)


    # from_excel_buff
    def test_from_excel_buff(self):
        # Load the test data from file
        subdir = "test_data"
        
        current_dir = Path(__file__).resolve().parent
        self.test_data_dir = current_dir / subdir

        excel_file_path = self.test_data_dir / "excel_test_1.xlsx"
        with excel_file_path.open("rb") as f:
            excel_data = f.read()

        # Test reading Excel data into Daf
        my_daf = Daf.from_excel_buff(excel_data)
        
        # Assert Daf properties
        self.assertEqual(my_daf.len(), 3)
        self.assertEqual(my_daf.num_cols(), 3)
        self.assertEqual(my_daf.columns(), ['ID', 'Name', 'Age'])
        self.assertEqual(my_daf.lol, [['1', 'John', '30'], ['2', 'Alice', '25'], ['3', 'Bob', '35']])


    def test_to_csv_file(self):
        # Determine the path to the test data directory
        current_dir = Path(__file__).resolve().parent
        self.test_data_dir = current_dir / "test_data"

        # Create a sample Daf object
        daf = Daf(
            lol=[
                ['ID', 'Name', 'Age'],
                [1, 'John', 30],
                [2, 'Alice', 25],
                [3, 'Bob', 35]
            ],
            keyfield='ID'
        )

        # Define the output CSV file path
        csv_file_path = self.test_data_dir / "test_output.csv"

        # Write Daf content to CSV file
        output_file_path = daf.to_csv_file(file_path=str(csv_file_path))

        # Assert that the output file path matches the expected path
        self.assertEqual(output_file_path, str(csv_file_path))

        # Assert that the CSV file has been created
        self.assertTrue(csv_file_path.exists())

        # Optionally, you can also assert the content of the CSV file
        # Here, you can read the content of the CSV file and compare it with the expected content

        # Clean up: Delete the CSV file after the test
        # os.remove(csv_file_path)
        
        
    # from_pandas_df
    def test_from_pandas_df_with_dataframe(self):
        # Mock a Pandas DataFrame with various data types
        df_mock = pd.DataFrame({
            'ID': [1, 2, 3],
            'Name': ['John', 'Alice', 'Bob'],
            'Age': [30, 25, 35],
            'IsAdult': [True, False, True],
            'Grade': [3.5, 4.2, 2.8]
        })

        # Call the method under test
        daf = Daf.from_pandas_df(df_mock)

        # Assert that the Daf object is created with the correct properties
        self.assertEqual(daf.len(), 3)
        self.assertEqual(daf.num_cols(), 5)  # Number of columns including all data types
        self.assertEqual(daf.columns(), ['ID', 'Name', 'Age', 'IsAdult', 'Grade'])
        self.assertEqual(daf.lol, [
            [1, 'John', 30, True, 3.5],
            [2, 'Alice', 25, False, 4.2],
            [3, 'Bob', 35, True, 2.8]
        ])
        
    def test_from_pandas_df_with_series(self):
        # Mock a Pandas Series
        series_mock = pd.DataFrame([
            {'ID': 1, 'Name': 'John', 'Age': 30},
            {'ID': 2, 'Name': 'Alice', 'Age': 25},
            {'ID': 3, 'Name': 'Bob', 'Age': 35}
        ]).iloc[0]

        # Call the method under test
        daf = Daf.from_pandas_df(series_mock)

        # Assert that the Daf object is created with the correct properties
        self.assertEqual(daf.len(), 1)
        self.assertEqual(daf.num_cols(), 3)
        self.assertEqual(daf.columns(), ['ID', 'Name', 'Age'])
        self.assertEqual(daf.lol, [[1, 'John', 30]])
        

    def test_from_pandas_df_with_dataframe_using_csv(self):
        # Mock a Pandas DataFrame
        df_mock = pd.DataFrame([
            {'ID': 1, 'Name': 'John', 'Age': 30},
            {'ID': 2, 'Name': 'Alice', 'Age': 25},
            {'ID': 3, 'Name': 'Bob', 'Age': 35}
        ])

        # Call the method under test
        daf = Daf.from_pandas_df(df_mock, use_csv=True).apply_dtypes(dtypes={'ID': int, 'Name': str, 'Age': int})

        # Assert that the Daf object is created with the correct properties
        self.assertEqual(daf.len(), 3)
        self.assertEqual(daf.num_cols(), 3)
        self.assertEqual(daf.columns(), ['ID', 'Name', 'Age'])
        self.assertEqual(daf.lol, [[1, 'John', 30], [2, 'Alice', 25], [3, 'Bob', 35]])

    def test_from_pandas_df_with_series_using_csv(self):
        # Mock a Pandas Series
        series_mock = pd.DataFrame([
            {'ID': 1, 'Name': 'John',  'Age': 30},
            {'ID': 2, 'Name': 'Alice', 'Age': 25},
            {'ID': 3, 'Name': 'Bob',   'Age': 35}
        ]).iloc[0]

        # Call the method under test
        daf = Daf.from_pandas_df(series_mock, use_csv=True)

        # Assert that the Daf object is created with the correct properties
        self.assertEqual(daf.len(), 1)
        self.assertEqual(daf.num_cols(), 3)
        self.assertEqual(daf.columns(), ['ID', 'Name', 'Age'])
        self.assertEqual(daf.lol, [[1, 'John', 30]])
        
        
    # test_daf_to_pandas
    def test_daf_to_pandas(self):
        # Create a Daf object with sample data
        daf = Daf(
            lol=[
                [1, 'John', 30, True, 3.5],
                [2, 'Alice', 25, False, 4.2],
                [3, 'Bob', 35, True, 2.8]
            ],
            cols=['ID', 'Name', 'Age', 'IsAdult', 'Grade']
        )

        # Convert Daf to Pandas DataFrame
        df = daf.to_pandas_df()

        # Expected DataFrame
        expected_df = pd.DataFrame({
            'ID': [1, 2, 3],
            'Name': ['John', 'Alice', 'Bob'],
            'Age': [30, 25, 35],
            'IsAdult': [True, False, True],
            'Grade': [3.5, 4.2, 2.8]
        })

        # Assert that the generated DataFrame is equal to the expected DataFrame
        pd.testing.assert_frame_equal(df, expected_df)

    def test_daf_to_pandas_using_csv(self):
        # Create a Daf object with sample data
        daf = Daf(
            lol=[
                [1, 'John', 30, True, 3.5],
                [2, 'Alice', 25, False, 4.2],
                [3, 'Bob', 35, True, 2.8]
            ],
            cols=['ID', 'Name', 'Age', 'IsAdult', 'Grade']
        )

        # Convert Daf to Pandas DataFrame
        df = daf.to_pandas_df(use_csv=True)

        # Expected DataFrame
        expected_df = pd.DataFrame({
            'ID': [1, 2, 3],
            'Name': ['John', 'Alice', 'Bob'],
            'Age': [30, 25, 35],
            'IsAdult': [True, False, True],
            'Grade': [3.5, 4.2, 2.8]
        })

        # Assert that the generated DataFrame is equal to the expected DataFrame
        pd.testing.assert_frame_equal(df, expected_df)

    # from numpy
    def test_from_numpy_all_integers(self):
        # Create a numpy array of all integers
        npa_int = np.array([[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]])

        # Call the method under test
        daf_int = Daf.from_numpy(npa_int)

        # Assert that the Daf object is created with the correct properties
        self.assertEqual(daf_int.len(), 3)
        self.assertEqual(daf_int.num_cols(), 3)
        self.assertEqual(daf_int.columns(), [])
        self.assertEqual(daf_int.lol, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def test_from_numpy_all_floats(self):
        # Create a numpy array of all floats
        npa_float = np.array([[1.0, 2.5, 3.7],
                               [4.2, 5.6, 6.9],
                               [7.3, 8.1, 9.4]])

        # Call the method under test
        daf_float = Daf.from_numpy(npa_float)

        # Assert that the Daf object is created with the correct properties
        self.assertEqual(daf_float.len(), 3)
        self.assertEqual(daf_float.num_cols(), 3)
        self.assertEqual(daf_float.columns(), [])
        self.assertEqual(daf_float.lol, [[1.0, 2.5, 3.7], [4.2, 5.6, 6.9], [7.3, 8.1, 9.4]])

    
    def test_from_numpy_with_2d_array(self):
        # Create a 2D numpy array
        npa = np.array([[1, 'John', 30], [2, 'Alice', 25], [3, 'Bob', 35]])

        # Call the from_numpy method
        daf = Daf.from_numpy(npa)

        # Check if the Daf object is created with the correct properties
        
        # Numpy arrays are homogeneous, meaning all elements in a numpy array 
        # must have the same data type. If you attempt to create a numpy array 
        # with elements of different data types, numpy will automatically cast 
        # them to a single data type that can accommodate all elements. This can 
        # lead to loss of information if the original data types are different.

        # For example, if you try to create a numpy array with both integers and 
        # strings, numpy will cast all elements to a common data type, such as Unicode strings.
        
        self.assertEqual(daf.len(), 3)
        self.assertEqual(daf.num_cols(), 3)
        self.assertEqual(daf.columns(), [])
        self.assertEqual(daf.lol, [['1', 'John', '30'], ['2', 'Alice', '25'], ['3', 'Bob', '35']])

    def test_from_numpy_with_1d_array(self):
        # Create a 1D numpy array
        npa = np.array(['John', 'Alice', 'Bob'])

        # Call the from_numpy method
        daf = Daf.from_numpy(npa)
        
        # Check if the Daf object is created with the correct properties
        self.assertEqual(daf.len(), 1)
        self.assertEqual(daf.num_cols(), 3)
        self.assertEqual(daf.columns(), [])
        self.assertEqual(daf.lol, [['John', 'Alice', 'Bob']])

    def test_from_numpy_with_keyfield_and_cols(self):
        # Create a 2D numpy array
        npa = np.array([[1, 'John', 30], [2, 'Alice', 25], [3, 'Bob', 35]])

        # Specify keyfield and columns
        keyfield = 'ID'
        cols = ['ID', 'Name', 'Age']

        # Call the from_numpy method
        daf = Daf.from_numpy(npa, keyfield=keyfield, cols=cols)

        # Check if the Daf object is created with the correct properties
        self.assertEqual(daf.keyfield, keyfield)
        self.assertEqual(daf.columns(), cols)

    # to_numpy
    def test_to_numpy_all_integers(self):
        # Create a Daf object with all integers
        daf_int = Daf(cols=['A', 'B', 'C'],
                        lol=[[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]])

        # Call the method under test
        npa_int = daf_int.to_numpy()

        # Assert that the numpy array is created with the correct values
        expected_npa_int = np.array([[1, 2, 3],
                                     [4, 5, 6],
                                     [7, 8, 9]])
        self.assertTrue(np.array_equal(npa_int, expected_npa_int))

    def test_to_numpy_all_floats(self):
        # Create a Daf object with all floats
        daf_float = Daf(cols=['X', 'Y', 'Z'],
                          lol=[[1.0, 2.5, 3.7],
                               [4.2, 5.6, 6.9],
                               [7.3, 8.1, 9.4]])

        # Call the method under test
        npa_float = daf_float.to_numpy()

        # Assert that the numpy array is created with the correct values
        expected_npa_float = np.array([[1.0, 2.5, 3.7],
                                       [4.2, 5.6, 6.9],
                                       [7.3, 8.1, 9.4]])
        self.assertTrue(np.array_equal(npa_float, expected_npa_float))



    # append
    def test_append_without_record(self):
        daf = Daf()

        daf.append({})

        self.assertEqual(daf, Daf())

    def test_append_without_keyfield(self):
        daf = Daf()
        record_da = {'col1': 1, 'col2': 'b'}

        daf.append(record_da)

        self.assertEqual(daf.name, '')
        self.assertEqual(daf.keyfield, '')
        self.assertEqual(daf.hd, {'col1': 0, 'col2': 1})
        self.assertEqual(daf.lol, [[1, 'b']])
        self.assertEqual(daf.kd, {})
        self.assertEqual(daf.dtypes, {})
        self.assertEqual(daf._iter_index, 0)

    def test_append_list_without_keyfield_but_cols(self):
        daf = Daf(cols=['col1', 'col2'])
        record_la = [1, 'b']

        daf.append(record_la)

        self.assertEqual(daf.name, '')
        self.assertEqual(daf.keyfield, '')
        self.assertEqual(daf.hd, {'col1': 0, 'col2': 1})
        self.assertEqual(daf.lol, [[1, 'b']])
        self.assertEqual(daf.kd, {})
        self.assertEqual(daf.dtypes, {})
        self.assertEqual(daf._iter_index, 0)

    def test_append_list_without_keyfield_no_cols(self):
        daf = Daf()
        record_la = [1, 'b']

        daf.append(record_la)

        self.assertEqual(daf.name, '')
        self.assertEqual(daf.keyfield, '')
        self.assertEqual(daf.hd, {})
        self.assertEqual(daf.lol, [[1, 'b']])
        self.assertEqual(daf.kd, {})
        self.assertEqual(daf.dtypes, {})
        self.assertEqual(daf._iter_index, 0)

    def test_append_with_keyfield(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b']]
        #kd = {1: 0, 2: 1}
        dtypes = {'col1': int, 'col2': str}
        daf = Daf(cols=cols, lol=lol, dtypes=dtypes, keyfield='col1')

        record_da = {'col1': 3, 'col2': 'c'}

        daf.append(record_da)

        self.assertEqual(daf.lol, [[1, 'a'], [2, 'b'], [3, 'c']])

        # append record with same keyfield will modify in place
        record_da = {'col1': 3, 'col2': 'd'}

        daf.append(record_da)

        self.assertEqual(daf.name, '')
        self.assertEqual(daf.keyfield, 'col1')
        self.assertEqual(daf.columns(), cols)
        self.assertEqual(daf.lol, [[1, 'a'], [2, 'b'], [3, 'd']])
        self.assertEqual(daf.kd, {1: 0, 2: 1, 3: 2})
        self.assertEqual(daf.dtypes, dtypes)
        self.assertEqual(daf._iter_index, 0)


    def test_extend_without_keyfield(self):
        daf = Daf()
        records_lod = [{'col1': 1, 'col2': 'b'}, {'col1': 2, 'col2': 'c'}]

        daf.extend(records_lod)

        self.assertEqual(daf.name, '')
        self.assertEqual(daf.keyfield, '')
        self.assertEqual(daf.hd, {'col1': 0, 'col2': 1})
        self.assertEqual(daf.lol, [[1, 'b'], [2, 'c']])
        self.assertEqual(daf.kd, {})
        self.assertEqual(daf.dtypes, {})
        self.assertEqual(daf._iter_index, 0)

    def test_extend_using_append_without_keyfield(self):
        daf = Daf()
        cols = ['col1', 'col2']
        records_lod = [{'col1': 1, 'col2': 'b'}, {'col1': 2, 'col2': 'c'}]

        daf.append(records_lod)

        self.assertEqual(daf.name, '')
        self.assertEqual(daf.keyfield, '')
        self.assertEqual(daf.columns(), cols)
        self.assertEqual(daf.lol, [[1, 'b'], [2, 'c']])
        self.assertEqual(daf.kd, {})
        self.assertEqual(daf.dtypes, {})
        self.assertEqual(daf._iter_index, 0)

    def test_extend_with_keyfield(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b']]
        dtypes = {'col1': int, 'col2': str}
        daf = Daf(cols=cols, lol=lol, dtypes=dtypes, keyfield='col1')

        records_lod = [{'col1': 3, 'col2': 'c'}, {'col1': 4, 'col2': 'd'}]

        daf.extend(records_lod)

        self.assertEqual(daf.name, '')
        self.assertEqual(daf.keyfield, 'col1')
        self.assertEqual(daf.columns(), cols)
        self.assertEqual(daf.lol, [[1, 'a'], [2, 'b'], [3, 'c'], [4, 'd']])
        self.assertEqual(daf.kd, {1: 0, 2: 1, 3: 2, 4: 3})
        self.assertEqual(daf.dtypes, dtypes)
        self.assertEqual(daf._iter_index, 0)
        

    def test_extend_using_append_with_keyfield(self):
        cols = ['col1', 'col2']
        hd = {'col1': 0, 'col2': 1}
        lol = [[1, 'a'], [2, 'b']]
        dtypes = {'col1': int, 'col2': str}
        daf = Daf(cols=cols, lol=lol, dtypes=dtypes, keyfield='col1')

        records_lod = [{'col1': 3, 'col2': 'c'}, {'col1': 4, 'col2': 'd'}]

        daf.append(records_lod)

        self.assertEqual(daf.name, '')
        self.assertEqual(daf.keyfield, 'col1')
        self.assertEqual(daf.hd, hd)
        self.assertEqual(daf.lol, [[1, 'a'], [2, 'b'], [3, 'c'], [4, 'd']])
        self.assertEqual(daf.kd, {1: 0, 2: 1, 3: 2, 4: 3})
        self.assertEqual(daf.dtypes, dtypes)
        self.assertEqual(daf._iter_index, 0)
        

    def test_concat_without_keyfield(self):
        daf1 = Daf()
        daf2 = Daf()

        cols = ['col1', 'col2']
        hd = {'col1': 0, 'col2': 1}
        lol1 = [[1, 'a'], [2, 'b']]
        lol2 = [['x', 'y'], ['z', 'w']]
        daf1 = Daf(cols=cols, lol=lol1, keyfield='', dtypes={'col1': str, 'col2': str}).apply_dtypes(from_str=False)
        daf2 = Daf(cols=cols, lol=lol2, keyfield='', dtypes={'col1': str, 'col2': str}).apply_dtypes(from_str=False)

        daf1.concat(daf2)

        self.assertEqual(daf1.name, '')
        self.assertEqual(daf1.keyfield, '')
        self.assertEqual(daf1.hd, hd)
        self.assertEqual(daf1.lol, [['1', 'a'], ['2', 'b'], ['x', 'y'], ['z', 'w']])
        self.assertEqual(daf1.kd, {})
        self.assertEqual(daf1.dtypes, {'col1': str, 'col2': str})
        self.assertEqual(daf1._iter_index, 0)

    def test_concat_without_keyfield_self_empty(self):
        daf1 = Daf()
        daf2 = Daf()

        cols = ['col1', 'col2']
        hd = {'col1': 0, 'col2': 1}
        lol1 = []
        lol2 = [['x', 'y'], ['z', 'w']]
        daf1 = Daf(           lol=lol1, keyfield='')
        daf2 = Daf(cols=cols, lol=lol2, keyfield='')

        daf1.concat(daf2)

        self.assertEqual(daf1.name, '')
        self.assertEqual(daf1.keyfield, '')
        self.assertEqual(daf1.hd, hd)
        self.assertEqual(daf1.lol, [['x', 'y'], ['z', 'w']])
        self.assertEqual(daf1.kd, {})
        self.assertEqual(daf1.dtypes, {})
        self.assertEqual(daf1._iter_index, 0)

    def test_concat_using_append_without_keyfield(self):
        daf1 = Daf()
        daf2 = Daf()

        cols = ['col1', 'col2']
        hd = {'col1': 0, 'col2': 1}
        lol1 = [[1, 'a'], [2, 'b']]
        lol2 = [['x', 'y'], ['z', 'w']]
        
        # breakpoint() #temp
        daf1 = Daf(cols=cols, lol=lol1, keyfield='', dtypes={'col1': str, 'col2': str}).apply_dtypes(from_str=False)
        daf2 = Daf(cols=cols, lol=lol2, keyfield='', dtypes={'col1': str, 'col2': str}).apply_dtypes(from_str=False)

        daf1.append(daf2)

        self.assertEqual(daf1.name, '')
        self.assertEqual(daf1.keyfield, '')
        self.assertEqual(daf1.hd, hd)
        self.assertEqual(daf1.lol, [['1', 'a'], ['2', 'b'], ['x', 'y'], ['z', 'w']])
        self.assertEqual(daf1.kd, {})
        self.assertEqual(daf1.dtypes, {'col1': str, 'col2': str})
        self.assertEqual(daf1._iter_index, 0)

    def test_concat_with_keyfield(self):
        cols = ['col1', 'col2']
        hd = {'col1': 0, 'col2': 1}
        lol1 = [[1, 'a'], [2, 'b']]
        lol2 = [[3, 'c'], [4, 'd']]
        daf1 = Daf(cols=cols, lol=lol1, keyfield='col1', dtypes={'col1': int, 'col2': str})
        daf2 = Daf(cols=cols, lol=lol2, keyfield='col1', dtypes={'col1': int, 'col2': str})

        daf1.concat(daf2)

        self.assertEqual(daf1.name, '')
        self.assertEqual(daf1.keyfield, 'col1')
        self.assertEqual(daf1.hd, hd)
        self.assertEqual(daf1.lol, [[1, 'a'], [2, 'b'], [3, 'c'], [4, 'd']])
        self.assertEqual(daf1.kd, {1: 0, 2: 1, 3: 2, 4: 3})
        self.assertEqual(daf1.dtypes, {'col1': int, 'col2': str})
        self.assertEqual(daf1._iter_index, 0)

    def test_concat_using_append_with_keyfield(self):
        cols = ['col1', 'col2']
        hd = {'col1': 0, 'col2': 1}
        lol1 = [[1, 'a'], [2, 'b']]
        lol2 = [[3, 'c'], [4, 'd']]
        daf1 = Daf(cols=cols, lol=lol1, keyfield='col1', dtypes={'col1': int, 'col2': str})
        daf2 = Daf(cols=cols, lol=lol2, keyfield='col1', dtypes={'col1': int, 'col2': str})

        daf1.append(daf2)

        self.assertEqual(daf1.name, '')
        self.assertEqual(daf1.keyfield, 'col1')
        self.assertEqual(daf1.hd, hd)
        self.assertEqual(daf1.lol, [[1, 'a'], [2, 'b'], [3, 'c'], [4, 'd']])
        self.assertEqual(daf1.kd, {1: 0, 2: 1, 3: 2, 4: 3})
        self.assertEqual(daf1.dtypes, {'col1': int, 'col2': str})
        self.assertEqual(daf1._iter_index, 0)
        

    # remove_key
    def test_remove_key_existing_key(self):
        cols = ['col1', 'col2']
        hd = {'col1': 0, 'col2': 1}
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        keyval = 2
        new_daf = daf.remove_key(keyval)

        self.assertEqual(new_daf.name, '')
        self.assertEqual(new_daf.keyfield, 'col1')
        self.assertEqual(new_daf.hd, hd)
        self.assertEqual(new_daf.lol, [[1, 'a'], [3, 'c']])
        self.assertEqual(new_daf.kd, {1: 0, 3: 1})
        self.assertEqual(new_daf.dtypes, {'col1': int, 'col2': str})
        self.assertEqual(new_daf._iter_index, 0)

    def test_remove_key_keyfield_notdefined(self):
        cols = ['col1', 'col2']
        hd = {'col1': 0, 'col2': 1}
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        daf = Daf(cols=cols, lol=lol, keyfield='', dtypes={'col1': int, 'col2': str})

        keyval = 4
        new_daf = daf.remove_key(keyval, silent_error=True)

        self.assertEqual(new_daf.name, '')
        self.assertEqual(new_daf.keyfield, '')
        self.assertEqual(new_daf.hd, hd)
        self.assertEqual(new_daf.lol, [[1, 'a'], [2, 'b'], [3, 'c']])
        self.assertEqual(new_daf.kd, {})
        self.assertEqual(new_daf.dtypes, {'col1': int, 'col2': str})
        self.assertEqual(new_daf._iter_index, 0)

    def test_remove_key_nonexistent_key_silent_error(self):
        cols = ['col1', 'col2']
        hd = {'col1': 0, 'col2': 1}
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        keyval = 4
        new_daf = daf.remove_key(keyval, silent_error=True)

        self.assertEqual(new_daf.name, '')
        self.assertEqual(new_daf.keyfield, 'col1')
        self.assertEqual(new_daf.hd, hd)
        self.assertEqual(new_daf.lol, [[1, 'a'], [2, 'b'], [3, 'c']])
        self.assertEqual(new_daf.kd, {1: 0, 2: 1, 3: 2})
        self.assertEqual(new_daf.dtypes, {'col1': int, 'col2': str})
        self.assertEqual(new_daf._iter_index, 0)

    def test_remove_key_nonexistent_key_raise_error(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        keyval = 4
        with self.assertRaises(KeyError):
            daf.remove_key(keyval, silent_error=False)


    # remove_keylist
    def test_remove_keylist_existing_keys(self):
        cols = ['col1', 'col2']
        hd = {'col1': 0, 'col2': 1}
        lol = [ [1, 'a'], 
                [2, 'b'], 
                [3, 'c'], 
                [4, 'd']]
                
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        keylist = [2, 4]
        
        new_daf = daf.remove_keylist(keylist)

        self.assertEqual(new_daf.name, '')
        self.assertEqual(new_daf.keyfield, 'col1')
        self.assertEqual(new_daf.hd, hd)
        self.assertEqual(new_daf.lol, [[1, 'a'], [3, 'c']])
        self.assertEqual(new_daf.kd, {1: 0, 3: 1})
        self.assertEqual(new_daf.dtypes, {'col1': int, 'col2': str})

    def test_remove_keylist_nonexistent_keys_silent_error(self):
        cols = ['col1', 'col2']
        hd = {'col1': 0, 'col2': 1}
        lol = [[1, 'a'], [2, 'b'], [3, 'c'], [4, 'd']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        keylist = [4, 5, 6]
        new_daf = daf.remove_keylist(keylist, silent_error=True)

        self.assertEqual(new_daf.name, '')
        self.assertEqual(new_daf.keyfield, 'col1')
        self.assertEqual(new_daf.hd, hd)
        self.assertEqual(new_daf.lol, [[1, 'a'], [2, 'b'], [3, 'c']])
        self.assertEqual(new_daf.kd, {1: 0, 2: 1, 3: 2})
        self.assertEqual(new_daf.dtypes, {'col1': int, 'col2': str})
        self.assertEqual(new_daf._iter_index, 0)

    def test_remove_keylist_nonexistent_keys_raise_error(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c'], [4, 'd']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        keylist = [4, 5, 6]

        with self.assertRaises(KeyError):
            daf.remove_keylist(keylist, silent_error=False)

    # select_record
    def test_select_record_existing_key(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        key = 2
        record_da = daf.select_record(key)

        self.assertEqual(record_da, {'col1': 2, 'col2': 'b'})

    def test_select_record_nonexistent_key(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        key = 4
        record_da = daf.select_record(key)

        self.assertEqual(record_da, {})

    def test_select_record_no_keyfield(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        daf = Daf(cols=cols, lol=lol, keyfield='', dtypes={'col1': int, 'col2': str})

        key = 'col1'

        with self.assertRaises(RuntimeError):
            daf.select_record(key)

    # iloc / irow
    def test_iloc_existing_row_idx(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        row_idx = 1
        record_da = daf.iloc(row_idx)

        self.assertEqual(record_da, {'col1': 2, 'col2': 'b'})

    def test_iloc_nonexistent_row_idx(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        row_idx = 4
        record_da = daf.iloc(row_idx)

        self.assertEqual(record_da, {})

    def test_iloc_negative_row_idx(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        row_idx = -1
        record_da = daf.irow(row_idx)

        self.assertEqual(record_da, {})

    def test_irow_existing_row_idx(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        row_idx = 1
        record_da = daf.irow(row_idx)

        self.assertEqual(record_da, {'col1': 2, 'col2': 'b'})

    def test_irow_nonexistent_row_idx(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        row_idx = 4
        record_da = daf.iloc(row_idx)

        self.assertEqual(record_da, {})

    def test_irow_negative_row_idx(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        row_idx = -1
        record_da = daf.irow(row_idx)

        self.assertEqual(record_da, {})

    # select_by_dict_to_lod
    def test_select_by_dict_to_lod_existing_selector_da(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c'], [4, 'b']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        selector_da = {'col2': 'b'}
        result_lod = daf.select_by_dict(selector_da).to_lod()

        expected_lod = [{'col1': 2, 'col2': 'b'}, {'col1': 4, 'col2': 'b'}]
        self.assertEqual(result_lod, expected_lod)

    def test_select_by_dict_to_lod_nonexistent_selector_da(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c'], [4, 'b']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        selector_da = {'col2': 'd'}
        result_lod = daf.select_by_dict(selector_da).to_lod()

        self.assertEqual(result_lod, [])

    def test_select_by_dict_to_lod_with_expectmax(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c'], [4, 'b']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        selector_da = {'col2': 'b'}
        expectmax = 1
        with self.assertRaises(LookupError):  # You should replace this with the actual exception that should be raised
            daf.select_by_dict(selector_da, expectmax=expectmax).to_lod()

    # select_by_dict
    def test_select_by_dict_existing_selector_da(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c'], [4, 'b']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        selector_da = {'col2': 'b'}
        result_daf = daf.select_by_dict(selector_da)

        expected_hd = {'col1': 0, 'col2': 1}
        expected_lol = [[2, 'b'], [4, 'b']]
        expected_kd = {2: 0, 4: 1}
        expected_dtypes = {'col1': int, 'col2': str}

        self.assertEqual(result_daf.name, '')
        self.assertEqual(result_daf.keyfield, 'col1')
        self.assertEqual(result_daf.hd, expected_hd)
        self.assertEqual(result_daf.lol, expected_lol)
        self.assertEqual(result_daf.kd, expected_kd)
        self.assertEqual(result_daf.dtypes, expected_dtypes)
        self.assertEqual(result_daf._iter_index, 0)

    def test_select_by_dict_nonexistent_selector_da(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c'], [4, 'b']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        selector_da = {'col2': 'd'}
        result_daf = daf.select_by_dict(selector_da)

        expected_hd = {'col1': 0, 'col2': 1}
        expected_lol = []
        expected_kd = {}
        expected_dtypes = {'col1': int, 'col2': str}

        self.assertEqual(result_daf.name, '')
        self.assertEqual(result_daf.keyfield, 'col1')
        self.assertEqual(result_daf.hd, expected_hd)
        self.assertEqual(result_daf.lol, expected_lol)
        self.assertEqual(result_daf.kd, expected_kd)
        self.assertEqual(result_daf.dtypes, expected_dtypes)
        self.assertEqual(result_daf._iter_index, 0)

    def test_select_by_dict_with_expectmax(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c'], [4, 'b']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        selector_da = {'col2': 'b'}
        expectmax = 1
        with self.assertRaises(LookupError):  # You should replace this with the actual exception that should be raised
            daf.select_by_dict(selector_da, expectmax=expectmax)

    # select_first_row_by_dict
    def test_select_first_row_by_dict_matching(self):
        # Test case where the first row matching the selector dictionary is found
        daf = Daf(lol=[[1, 'John'], [2, 'Jane'], [3, 'Doe']], cols=['ID', 'Name'])
        selected_row = daf.select_first_row_by_dict({'ID': 2})
        self.assertEqual(selected_row, {'ID':2, 'Name':'Jane'})

    def test_select_first_row_by_dict_no_match(self):
        # Test case where no row matches the selector dictionary
        daf = Daf(lol=[[1, 'John'], [2, 'Jane'], [3, 'Doe']], cols=['ID', 'Name'])
        selected_row = daf.select_first_row_by_dict({'ID': 4})
        self.assertEqual(selected_row, {})

    def test_select_first_row_by_dict_inverse_matching(self):
        # Test case where the first row not matching the selector dictionary is found
        daf = Daf(lol=[[1, 'John'], [2, 'Jane'], [3, 'Doe']], cols=['ID', 'Name'])
        selected_row = daf.select_first_row_by_dict({'ID': 2}, inverse=True)
        self.assertEqual(selected_row, {'ID':1, 'Name':'John'})

    def test_select_first_row_by_dict_empty_daf(self):
        # Test case where Daf is empty
        daf = Daf(lol=[], cols=['ID', 'Name'])
        selected_row = daf.select_first_row_by_dict({'ID': 2})
        self.assertEqual(selected_row, {})


    # col / col_to_la
    def test_col_existing_colname(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        colname = 'col2'
        result_la = daf.col(colname)

        expected_la = ['a', 'b', 'c']
        self.assertEqual(result_la, expected_la)

    def test_col_nonexistent_colname(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        colname = 'col3'
        with self.assertRaises(RuntimeError):
            result_la = daf.col(colname)
            result_la = result_la # fool linter

        #self.assertEqual(result_la, [])

    def test_col_empty_colname(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        colname = ''
        with self.assertRaises(RuntimeError):
            result_la = daf.col(colname)
            result_la = result_la # fool linter

        #self.assertEqual(result_la, [])

    def test_col_nonexistent_colname_silent(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        colname = 'col3'
        result_la = daf.col(colname, silent_error=True)

        self.assertEqual(result_la, [])

    # drop_cols
    def test_drop_cols_existing_cols(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 'a', 'x'], [2, 'b', 'y'], [3, 'c', 'z']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': str})

        cols_to_drop = ['col2', 'col3']
        daf.drop_cols(cols_to_drop)

        expected_hd = {'col1': 0}
        expected_lol = [[1], [2], [3]]

        self.assertEqual(daf.hd, expected_hd)
        self.assertEqual(daf.lol, expected_lol)

    def test_drop_cols_nonexistent_cols(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 'a', 'x'], [2, 'b', 'y'], [3, 'c', 'z']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': str})

        cols_to_drop = ['col4', 'col5']
        daf.drop_cols(cols_to_drop)

        expected_hd = {'col1': 0, 'col2': 1, 'col3': 2}
        expected_lol = [[1, 'a', 'x'], [2, 'b', 'y'], [3, 'c', 'z']]

        self.assertEqual(daf.hd, expected_hd)
        self.assertEqual(daf.lol, expected_lol)

    def test_drop_cols_empty_cols(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 'a', 'x'], [2, 'b', 'y'], [3, 'c', 'z']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': str})

        cols_to_drop = []
        daf.drop_cols(cols_to_drop)

        expected_hd = {'col1': 0, 'col2': 1, 'col3': 2}
        expected_lol = [[1, 'a', 'x'], [2, 'b', 'y'], [3, 'c', 'z']]

        self.assertEqual(daf.hd, expected_hd)
        self.assertEqual(daf.lol, expected_lol)

    # assign_col
    def test_assign_col_existing_colname(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 'a', 'x'], [2, 'b', 'y'], [3, 'c', 'z']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': str})

        colname = 'col2'
        new_values = ['A', 'B', 'C']
        daf.assign_col(colname, new_values)

        expected_hd = {'col1': 0, 'col2': 1, 'col3': 2}
        expected_lol = [[1, 'A', 'x'], [2, 'B', 'y'], [3, 'C', 'z']]

        self.assertEqual(daf.hd, expected_hd)
        self.assertEqual(daf.lol, expected_lol)

    def test_assign_col_nonexistent_colname(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 'a', 'x'], [2, 'b', 'y'], [3, 'c', 'z']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': str})

        colname = 'col4'
        new_values = ['A', 'B', 'C']
        daf.assign_col(colname, new_values)
        
        # will insert new col if col not exist.

        expected_hd = {'col1': 0, 'col2': 1, 'col3': 2, 'col4': 3}
        expected_lol = [[1, 'a', 'x', 'A'], [2, 'b', 'y', 'B'], [3, 'c', 'z', 'C']]

        self.assertEqual(daf.hd, expected_hd)
        self.assertEqual(daf.lol, expected_lol)

    def test_assign_col_empty_colname(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 'a', 'x'], [2, 'b', 'y'], [3, 'c', 'z']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': str})

        colname = ''
        new_values = ['A', 'B', 'C']
        daf.assign_col(colname, new_values)

        expected_hd = {'col1': 0, 'col2': 1, 'col3': 2}
        expected_lol = [[1, 'a', 'x'], [2, 'b', 'y'], [3, 'c', 'z']]

        self.assertEqual(daf.hd, expected_hd)
        self.assertEqual(daf.lol, expected_lol)

    # cols_to_dol
    def test_cols_to_dol_valid_cols(self):
        cols = ['col1', 'col2', 'col3']
        lol = [['a', 'b', 'c'], ['b', 'd', 'e'], ['a', 'f', 'g'], ['b', 'd', 'm']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        colname1 = 'col1'
        colname2 = 'col2'
        result_dola = daf.cols_to_dol(colname1, colname2)

        expected_dola = {'a': ['b', 'f'], 'b': ['d']}
        self.assertEqual(result_dola, expected_dola)

    def test_cols_to_dol_invalid_colname(self):
        cols = ['col1', 'col2', 'col3']
        lol = [['a', 'b', 'c'], ['b', 'd', 'e'], ['a', 'f', 'g'], ['b', 'd', 'm']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        colname1 = 'col1'
        colname2 = 'col4'
        result_dola = daf.cols_to_dol(colname1, colname2)

        self.assertEqual(result_dola, {})

    def test_cols_to_dol_empty_daf(self):
        cols = []
        lol = []
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={})

        colname1 = 'col1'
        colname2 = 'col2'
        result_dola = daf.cols_to_dol(colname1, colname2)

        self.assertEqual(result_dola, {})

    def test_cols_to_dol_single_column(self):
        cols = ['col1']
        lol = [['a'], ['b'], ['a'], ['b']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str})

        colname1 = 'col1'
        colname2 = 'col2'
        result_dola = daf.cols_to_dol(colname1, colname2)

        self.assertEqual(result_dola, {})

    # bool
    def test_bool_empty_daf(self):
        cols = []
        lol = []
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={})

        result = bool(daf)

        self.assertFalse(result)

    def test_bool_nonempty_daf(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        result = bool(daf)

        self.assertTrue(result)

    def test_bool_daf_with_empty_lol(self):
        cols = ['col1', 'col2']
        lol = []
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        result = bool(daf)

        self.assertFalse(result)

    # len
    def test_len_empty_daf(self):
        cols = []
        lol = []
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={})

        result = len(daf)

        self.assertEqual(result, 0)

    def test_len_nonempty_daf(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        result = len(daf)

        self.assertEqual(result, 3)

    def test_len_daf_with_empty_lol(self):
        cols = ['col1', 'col2']
        lol = []
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        result = len(daf)

        self.assertEqual(result, 0)
        
    # columns
    def test_columns_empty_daf(self):
        cols = []
        lol = []
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={})

        result = daf.columns()

        self.assertEqual(result, [])

    def test_columns_nonempty_daf(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 'a', 'x'], [2, 'b', 'y'], [3, 'c', 'z']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': str})

        result = daf.columns()

        self.assertEqual(result, ['col1', 'col2', 'col3'])
        
    # clone_empty
    
    def test_clone_empty_from_empty_instance(self):
        old_instance = Daf()
        result = Daf.clone_empty(old_instance)

        self.assertEqual(result.name, '')
        self.assertEqual(result.keyfield, '')
        self.assertEqual(result.hd, {})
        self.assertEqual(result.lol, [])
        self.assertEqual(result.kd, {})
        self.assertEqual(result.dtypes, {})
        self.assertEqual(result._iter_index, 0)

    def test_clone_empty_from_nonempty_instance(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        old_instance = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})
        result = Daf.clone_empty(old_instance)

        self.assertEqual(result.name, old_instance.name)
        self.assertEqual(result.keyfield, old_instance.keyfield)
        self.assertEqual(result.hd, old_instance.hd)
        self.assertEqual(result.lol, [])
        self.assertEqual(result.kd, {})
        self.assertEqual(result.dtypes, old_instance.dtypes)
        self.assertEqual(result._iter_index, 0)

    def test_clone_empty_from_none(self):
        old_instance = None
        result = Daf.clone_empty(old_instance)

        self.assertEqual(result.name, '')
        self.assertEqual(result.keyfield, '')
        self.assertEqual(result.hd, {})
        self.assertEqual(result.lol, [])
        self.assertEqual(result.kd, {})
        self.assertEqual(result.dtypes, {})
        self.assertEqual(result._iter_index, 0)

    # to_lod
    def test_to_lod_empty_daf(self):
        cols = []
        lol = []
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={})

        result = daf.to_lod()

        self.assertEqual(result, [])

    def test_to_lod_nonempty_daf(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        result = daf.to_lod()

        expected_lod = [{'col1': 1, 'col2': 'a'}, {'col1': 2, 'col2': 'b'}, {'col1': 3, 'col2': 'c'}]
        self.assertEqual(result, expected_lod)

    # select_records
    def test_select_records_empty_daf(self):
        cols = []
        lol = []
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={})

        keys_ls = [1, 2, 3]
        result = daf.select_records_daf(keys_ls)

        self.assertEqual(result.name,   '')
        self.assertEqual(result.keyfield, 'col1')
        self.assertEqual(result.hd,     {})
        self.assertEqual(result.lol,    [])
        self.assertEqual(result.kd,     {})
        self.assertEqual(result.dtypes,  {})

    def test_select_records_nonempty_daf(self):
        cols    = ['col1', 'col2']
        lol = [ [1, 'a'], 
                [2, 'b'], 
                [3, 'c']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        keys_ls = [2, 1]
        result = daf.select_records_daf(keys_ls)

        expected_lol = [[2, 'b'], [1, 'a']]
        self.assertEqual(result.name, daf.name)
        self.assertEqual(result.keyfield, daf.keyfield)
        self.assertEqual(result.hd, daf.hd)
        self.assertEqual(result.lol, expected_lol)
        self.assertEqual(result.dtypes, daf.dtypes)

    def test_select_records_empty_keys(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        dtypes={'col1': int, 'col2': str}
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes=dtypes)

        # empty keys changed to select all records.
        keys_ls = []
        result = daf.select_records_daf(keys_ls)

        self.assertEqual(result.name, '')
        self.assertEqual(result.keyfield, 'col1')
        self.assertEqual(result.hd, {'col1': 0, 'col2': 1})
        self.assertEqual(result.lol, [[1, 'a'], [2, 'b'], [3, 'c']])
        self.assertEqual(result.kd, {1:0, 2:1, 3:2})
        self.assertEqual(result.dtypes, dtypes)

    def test_select_records_daf_without_inverse(self):
        # Initialize test data
        cols = ['ID', 'Name', 'Age']
        lol = [
            [1, 'John', 30],
            [2, 'Alice', 25],
            [3, 'Bob', 35]
        ]
        daf = Daf(lol=lol, cols=cols, keyfield='Name')  # Initialize Daf with test data

        # Test without inverse
        keys_ls = ['John', 'Alice']  # Define the keys list
        expected_lol =[
            [1, 'John', 30],
            [2, 'Alice', 25],
        ]

        result_daf = daf.select_records_daf(keys_ls)  # Call the method
        self.assertEqual(result_daf.lol, expected_lol)  # Check if the selected row indices are correct

    def test_select_records_daf_with_inverse(self):
        # Initialize test data
        cols = ['ID', 'Name', 'Age']
        lol = [
            [1, 'John', 30],
            [2, 'Alice', 25],
            [3, 'Bob', 35]
        ]
        daf = Daf(lol=lol, cols=cols, keyfield='Name')  # Initialize Daf with test data

        # Test with inverse
        keys_ls = ['John', 'Alice']  # Define the keys list
        expected_lol =[
            [3, 'Bob', 35]
        ]
        result_daf = daf.select_records_daf(keys_ls, inverse=True)  # Call the method with inverse=True
        self.assertEqual(result_daf.lol, expected_lol)  # Check if the selected row indices are correct


    # assign_record
    def test_assign_record_empty_daf(self):
        cols = []
        lol = []
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={})

        record_da = {'col1': 1, 'col2': 'a'}
        daf.assign_record(record_da)

        expected_lol = [[1, 'a']]
        self.assertEqual(daf.lol, expected_lol)

    def test_assign_record_nonempty_daf_add_new_record(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        record_da = {'col1': 4, 'col2': 'd'}
        daf.assign_record(record_da)

        expected_lol = [[1, 'a'], [2, 'b'], [3, 'c'], [4, 'd']]
        self.assertEqual(daf.lol, expected_lol)

    def test_assign_record_nonempty_daf_update_existing_record(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        record_da = {'col1': 2, 'col2': 'x'}
        daf.assign_record(record_da)

        expected_lol = [[1, 'a'], [2, 'x'], [3, 'c']]
        self.assertEqual(daf.lol, expected_lol)

    def test_assign_record_missing_keyfield(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        record_da = {'col2': 'x'}
        with self.assertRaises(RuntimeError):
            daf.assign_record(record_da)

    def test_assign_record_fields_not_equal_to_columns(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        record_da = {'col1': 4, 'col2': 'd', 'col3': 'extra'}
        with self.assertRaises(RuntimeError):
            daf.assign_record(record_da)

    # assign_record_irow
    def test_assign_record_irow_empty_daf(self):
        cols = []
        lol = []
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={})

        record_da = {'col1': 1, 'col2': 'a'}
        daf.assign_record_irow(irow=0, record=record_da)

        expected_lol = [[1, 'a']]
        self.assertEqual(daf.lol, expected_lol)

    def test_assign_record_irow_nonempty_daf_add_new_record(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        record_da = {'col1': 4, 'col2': 'd'}
        daf.assign_record_irow(irow=3, record=record_da)

        expected_lol = [[1, 'a'], [2, 'b'], [3, 'c'], [4, 'd']]
        self.assertEqual(daf.lol, expected_lol)

    def test_assign_record_irow_nonempty_daf_update_existing_record(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        record_da = {'col1': 2, 'col2': 'x'}
        daf.assign_record_irow(irow=1, record=record_da)

        expected_lol = [[1, 'a'], [2, 'x'], [3, 'c']]
        self.assertEqual(daf.lol, expected_lol)

    def test_assign_record_irow_invalid_irow(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], 
               [2, 'b'], 
               [3, 'c']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        record_da = {'col1': 4, 'col2': 'd'}
        
        daf.assign_record_irow(irow=5, record=record_da)

        expected_lol = [[1, 'a'], 
                        [2, 'b'], 
                        [3, 'c'],
                        [4, 'd'],
                        ]
        self.assertEqual(daf.lol, expected_lol)

    def test_assign_record_irow_missing_record(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        daf.assign_record_irow(irow=1, record=None)

        expected_lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        self.assertEqual(daf.lol, expected_lol)

   # update_record_irow
    def test_update_record_irow_empty_daf(self):
        cols = []
        lol = []
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={})

        record_da = {'col1': 1, 'col2': 'a'}
        daf.update_record_irow(irow=0, record=record_da)

        self.assertEqual(daf.lol, [])

    def test_update_record_irow_nonempty_daf_update_existing_record(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        record_da = {'col1': 2, 'col2': 'x', 'col3': 'extra'}
        daf.update_record_irow(irow=1, record=record_da)

        expected_lol = [[1, 'a'], [2, 'x'], [3, 'c']]
        self.assertEqual(daf.lol, expected_lol)

    def test_update_record_irow_invalid_irow(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        record_da = {'col1': 4, 'col2': 'd'}
        daf.update_record_irow(irow=5, record=record_da)

        self.assertEqual(daf.lol, lol)

    def test_update_record_irow_missing_record(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        daf.update_record_irow(irow=1, record=None)

        expected_lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        self.assertEqual(daf.lol, expected_lol)

    # def test_update_record_irow_missing_hd(self):
        # cols = ['col1', 'col2']
        # hd = {'col1': 0, 'col2': 1}
        # lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        # daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        # record_da = {'col1': 2, 'col2': 'x'}
        # daf.update_record_da_irow(irow=1, record_da=record_da)

        # self.assertEqual(daf.lol, lol)

    # icol_to_la
    def test_icol_to_la_valid_icol(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 'a', True], [2, 'b', False], [3, 'c', True]]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': bool})

        result_la = daf.icol_to_la(1)

        expected_la = ['a', 'b', 'c']
        self.assertEqual(result_la, expected_la)

    def test_icol_to_la_invalid_icol(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 'a', True], [2, 'b', False], [3, 'c', True]]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': bool})

        result_la = daf.icol_to_la(3)

        self.assertEqual(result_la, [])

    def test_icol_to_la_empty_daf(self):
        daf = Daf()

        result_la = daf.icol_to_la(0)

        self.assertEqual(result_la, [])

    def test_icol_to_la_empty_column(self):
        cols = ['col1', 'col2', 'col3']
        lol = []
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': bool})

        result_la = daf.icol_to_la(0)

        self.assertEqual(result_la, [])

    def test_icol_to_la_unique(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 'a', True], [2, 'b', False], [3, 'a', True]]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': bool})

        result_la = daf.icol_to_la(1, unique=True)

        expected_la = ['a', 'b']
        self.assertEqual(result_la, expected_la)
        

    def test_icol_to_la_omit_nulls_true_with_null(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 'a', True], [2, 'b', False], [3, '', True]]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': bool})

        result_la = daf.icol_to_la(1, omit_nulls=True)

        expected_la = ['a', 'b']
        self.assertEqual(result_la, expected_la)
        

    def test_icol_to_la_omit_nulls_false_with_null(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 'a', True], [2, 'b', False], [3, '', True]]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': bool})

        result_la = daf.icol_to_la(1, omit_nulls=False)

        expected_la = ['a', 'b', '']
        self.assertEqual(result_la, expected_la)
        

    # assign_icol
    def test_assign_icol_valid_icol_col_la(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 'a', True], [2, 'b', False], [3, 'c', True]]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': bool})

        col_la = [4, 'd', False]
        daf.assign_icol(icol=1, col_la=col_la)

        expected_lol = [[1, 4, True], [2, 'd', False], [3, False, True]]
        self.assertEqual(daf.lol, expected_lol)

    def test_assign_icol_valid_icol_default(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 'a', True], 
               [2, 'b', False], 
               [3, 'c', True]]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': bool})

        daf.assign_icol(icol=1, default='x')

        expected_lol = [[1, 'x', True], 
                        [2, 'x', False], 
                        [3, 'x', True]]
        self.assertEqual(daf.lol, expected_lol)

    def test_assign_icol_valid_append_icol_col_la(self):
        cols = ['col1', 'col2', 'col3']
        lol = [ [1, 'a', True], 
                [2, 'b', False], 
                [3, 'c', True]]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': bool})

        col_la = [4, 'd', False]
        daf.assign_icol(icol=-1, col_la=col_la)

        expected_lol = [[1, 'a', True, 4], 
                        [2, 'b', False, 'd'], 
                        [3, 'c', True, False]]
        self.assertEqual(daf.lol, expected_lol)

    # def test_assign_icol_invalid_icol_col_la(self):
        # cols = ['col1', 'col2', 'col3']
        # hd = {'col1': 0, 'col2': 1, 'col3': 2}
        # lol = [[1, 'a', True], [2, 'b', False], [3, 'c', True]]
        # daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': bool})

        # col_la = [4, 'd', False]
        # daf.assign_icol(icol=3, col_la=col_la)

        # self.assertEqual(daf.lol, lol)

    def test_assign_icol_empty_daf(self):
        daf = Daf()

        col_la = [4, 'd', False]
        daf.assign_icol(icol=1, col_la=col_la)

        self.assertEqual(daf.lol, [])

    # insert_icol
    def test_insert_icol_valid_icol_col_la(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 'a', True], [2, 'b', False], [3, 'c', True]]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': bool})

        col_la = [4, 'd', False]
        daf.insert_icol(icol=1, col_la=col_la)

        expected_lol = [[1, 4, 'a', True], [2, 'd', 'b', False], [3, False, 'c', True]]
        self.assertEqual(daf.lol, expected_lol)

    def test_insert_icol_valid_append_col_la(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 'a', True], [2, 'b', False], [3, 'c', True]]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': bool})

        col_la = [4, 'd', False]
        daf.insert_icol(icol=-1, col_la=col_la)

        expected_lol = [[1, 'a', True, 4], [2, 'b', False, 'd'], [3, 'c', True, False]]
        self.assertEqual(daf.lol, expected_lol)

    def test_insert_icol_invalid_icol_col_la(self):
        cols = ['col1', 'col2', 'col3']
        lol = [ [1, 'a', True], 
                [2, 'b', False], 
                [3, 'c', True]]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': bool})

        col_la = [4, 'd', False]
        daf.insert_icol(icol=3, col_la=col_la)

        result_lol = [ [1, 'a', True, 4], 
                [2, 'b', False, 'd'], 
                [3, 'c', True,  False]]

        self.assertEqual(daf.lol, result_lol)

    def test_insert_icol_empty_daf(self):
        daf = Daf()

        col_la = [4, 'd', False]
        daf.insert_icol(icol=1, col_la=col_la)

        self.assertEqual(daf.lol, [[4], ['d'], [False]])

    # insert_col
    def test_insert_col_valid_colname_col_la(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 'a', True], [2, 'b', False], [3, 'c', True]]
        daf = Daf(cols=cols,  lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': bool})

        colname = 'new_col'
        col_la = [4, 'd', False]
        daf.insert_col(colname=colname, col_la=col_la, icol=1)

        expected_lol = [[1, 4, 'a', True], [2, 'd', 'b', False], [3, False, 'c', True]]
        expected_hd = {'col1': 0, 'new_col': 1, 'col2': 2, 'col3': 3}
        self.assertEqual(daf.lol, expected_lol)
        self.assertEqual(daf.hd, expected_hd)
        

    def test_insert_col_existing_colname_col_la(self):
        # insert a column that already exists and it will overwrite the contents of that column
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 'a', True], [2, 'b', False], [3, 'c', True]]
        daf = Daf(cols=cols,  lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': bool})

        colname = 'col2'
        col_la = [4, 'd', False]
        daf.insert_col(colname=colname, col_la=col_la, icol=1)

        expected_lol = [[1, 4, True], [2, 'd', False], [3, False, True]]
        expected_hd = {'col1': 0, 'col2': 1, 'col3': 2}
        self.assertEqual(daf.lol, expected_lol)
        self.assertEqual(daf.hd, expected_hd)
        

    def test_insert_col_valid_colname_append_col_la(self):
        cols = ['col1', 'col2', 'col3']
        lol = [ [1, 'a', True], 
                [2, 'b', False], 
                [3, 'c', True]]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': bool})

        colname = 'new_col'
        col_la = [4, 'd', False]
        daf.insert_col(colname=colname, col_la=col_la, icol=-1)

        expected_lol = [[1, 'a', True,  4], 
                        [2, 'b', False, 'd'], 
                        [3, 'c', True,  False]]
        expected_hd = {'col1': 0, 'col2': 1, 'col3': 2, 'new_col': 3}
        self.assertEqual(daf.lol, expected_lol)
        self.assertEqual(daf.hd, expected_hd)
        

    def test_insert_col_valid_colname_invalid_icol_col_la(self):
        cols = ['col1', 'col2', 'col3']
        lol = [ [1, 'a', True], 
                [2, 'b', False], 
                [3, 'c', True]]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': bool})

        colname = 'new_col'
        col_la = [4, 'd', False]
        daf.insert_col(colname=colname, col_la=col_la, icol=3)

        expected_hd = {'col1': 0, 'col2': 1, 'col3': 2, 'new_col': 3}
        expected_lol = [ [1, 'a', True,     4], 
                         [2, 'b', False,    'd'], 
                         [3, 'c', True,     False]]

        self.assertEqual(daf.lol, expected_lol)
        self.assertEqual(daf.hd, expected_hd)
        

    def test_insert_col_valid_colname_empty_daf(self):
        daf = Daf()

        colname = 'new_col'
        col_la = [4, 'd', False]
        daf.insert_col(colname=colname, col_la=col_la, icol=1)

        self.assertEqual(daf.lol, [[4], ['d'], [False]])
        self.assertEqual(daf.hd, {'new_col': 0})
        

    def test_insert_col_empty_colname(self):
        cols = ['col1', 'col2', 'col3']
        hd = {'col1': 0, 'col2': 1, 'col3': 2}
        lol = [[1, 'a', True], [2, 'b', False], [3, 'c', True]]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': bool})

        daf.insert_col(colname='', col_la=[4, 'd', False], icol=1)

        self.assertEqual(daf.lol, lol)
        self.assertEqual(daf.hd, hd)
        

    # insert_idx_col
    def test_insert_idx_col_valid_icol_startat(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 'a', True], [2, 'b', False], [3, 'c', True]]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': bool})

        colname = 'idx'
        daf.insert_idx_col(colname=colname, icol=1, startat=10)

        expected_lol = [[1, 10, 'a', True], [2, 11, 'b', False], [3, 12, 'c', True]]
        expected_hd = {'col1': 0, 'idx': 1, 'col2': 2, 'col3': 3}
        self.assertEqual(daf.lol, expected_lol)
        self.assertEqual(daf.hd, expected_hd)
        

    def test_insert_idx_col_valid_icol_default_startat(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 'a', True], [2, 'b', False], [3, 'c', True]]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': bool})

        colname = 'idx'
        daf.insert_idx_col(colname=colname, icol=1)

        expected_lol = [[1, 0, 'a', True], [2, 1, 'b', False], [3, 2, 'c', True]]
        expected_hd = {'col1': 0, 'idx': 1, 'col2': 2, 'col3': 3}
        self.assertEqual(daf.lol, expected_lol)
        self.assertEqual(daf.hd, expected_hd)

    def test_insert_idx_col_valid_append_default_startat(self):
        daf = Daf()

        colname = 'idx'
        daf.insert_idx_col(colname=colname)

        expected_lol = []
        expected_hd = {'idx': 0}
        self.assertEqual(daf.lol, expected_lol)
        self.assertEqual(daf.hd, expected_hd)

    def test_insert_idx_col_empty_colname(self):
        cols = ['col1', 'col2', 'col3']
        hd = {'col1': 0, 'col2': 1, 'col3': 2}
        lol = [[1, 'a', True], [2, 'b', False], [3, 'c', True]]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': bool})

        daf.insert_idx_col(colname='', icol=1, startat=10)

        self.assertEqual(daf.lol, lol)
        self.assertEqual(daf.hd, hd)

    # select_cols
    def test_select_cols_with_cols(self):
        # Test case where columns are selected based on the cols parameter
        daf = Daf(lol=[[1, 'a', True], [2, 'b', False], [3, 'c', True]], cols=['ID', 'Name', 'Flag'], dtypes={'ID': int, 'Name': str, 'Flag': bool})
        new_daf = daf.select_cols(cols=['ID', 'Flag'])
        self.assertEqual(new_daf.columns(), ['ID', 'Flag'])
        self.assertEqual(new_daf.lol,  [[1, True], [2, False], [3, True]])
        
        # verify that the original is unchanged
        self.assertEqual(daf.columns(), ['ID', 'Name', 'Flag'])
        self.assertEqual(daf.lol,  [[1, 'a', True], [2, 'b', False], [3, 'c', True]])
        

    def test_select_cols_with_exclude_cols(self):
        # Test case where columns are selected based on the exclude_cols parameter
        daf = Daf(lol=[[1, 'a', True], [2, 'b', False], [3, 'c', True]], cols=['ID', 'Name', 'Flag'], dtypes={'ID': int, 'Name': str, 'Flag': bool})
        new_daf = daf.select_cols(exclude_cols=['Name'])
        self.assertEqual(new_daf.columns(), ['ID', 'Flag'])
        self.assertEqual(new_daf.lol,  [[1, True], [2, False], [3, True]])

    def test_select_cols_with_empty_params(self):
        # Test case where no cols or exclude_cols are provided
        daf = Daf(lol=[[1, 'a', True], [2, 'b', False], [3, 'c', True]], cols=['ID', 'Name', 'Flag'], dtypes={'ID': int, 'Name': str, 'Flag': bool})
        new_daf = daf.select_cols()
        self.assertEqual(new_daf, daf)

    def test_select_cols_with_empty_daf(self):
        # Test case where Daf is empty
        daf = Daf()
        new_daf = daf.select_cols(cols=['ID', 'Name'])
        self.assertEqual(new_daf.columns(), [])


    # unified sum
    def test_sum_all_columns(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': int, 'col3': int})

        result_sum = daf.sum()
        expected_sum = {'col1': 12, 'col2': 15, 'col3': 18}
        self.assertEqual(result_sum, expected_sum)

    def test_sum_selected_columns(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': int, 'col3': int})

        result_sum = daf.sum(colnames_ls=['col1', 'col3'])
        expected_sum = {'col1': 12, 'col3': 18}
        self.assertEqual(result_sum, expected_sum)

    def test_sum_numeric_only(self):
        cols = ['col1', 'col2', 'col3']
        lol = [['a', 2, 3], ['b', 5, 6], ['c', 8, 9]]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': int, 'col3': int})

        result_sum = daf.sum(numeric_only=True)
        expected_sum = {'col1': '0.0', 'col2': 15, 'col3': 18}
        self.assertEqual(result_sum, expected_sum)

    def test_sum_empty_daf(self):
        daf = Daf()

        result_sum = daf.sum()
        expected_sum = {}
        self.assertEqual(result_sum, expected_sum)

    # unified sum_np
    def test_sum_np_all_columns(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': int, 'col3': int})

        result_sum = daf.sum_np()
        expected_sum = {'col1': 12, 'col2': 15, 'col3': 18}
        self.assertEqual(result_sum, expected_sum)

    def test_sum_np_selected_columns(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': int, 'col3': int})

        result_sum = daf.sum_np(colnames_ls=['col1', 'col3'])
        expected_sum = {'col1': 12, 'col3': 18}
        self.assertEqual(result_sum, expected_sum)

    def test_sum_np_empty_daf(self):
        daf = Daf()

        result_sum = daf.sum_np()
        expected_sum = {}
        self.assertEqual(result_sum, expected_sum)


    # daf_sum
    def test_daf_sum_all_columns(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': int, 'col3': int})

        result_sum = daf.daf_sum()
        expected_sum = {'col1': 12, 'col2': 15, 'col3': 18}
        self.assertEqual(result_sum, expected_sum)

    def test_daf_sum_selected_columns(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': int, 'col3': int})

        result_sum = daf.daf_sum(cols=['col1', 'col3'])
        expected_sum = {'col1': 12, 'col2': '', 'col3': 18}
        self.assertEqual(result_sum, expected_sum)

    def test_daf_sum_include_types_int(self):
        cols = ['col1', 'col2', 'col3']
        dtypes_dict = {'col1': str, 'col2': int, 'col3': int}
        lol = [['a', 2, 3], ['b', 5, 6], ['c', 8, 9]]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes=dtypes_dict)

        reduce_cols = daf.calc_cols(include_types=int)
        result_sum = daf.daf_sum(cols=reduce_cols)
        expected_sum = {'col1': '', 'col2': 15, 'col3': 18}
        self.assertEqual(result_sum, expected_sum)

    def test_daf_sum_include_types_int_and_float(self):
        cols = ['col1', 'col2', 'col3']
        dtypes_dict = {'col1': str, 'col2': int, 'col3': float}
        lol = [['a', 2, 3.2], ['b', 5, 6.1], ['c', 8, 9.4]]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes=dtypes_dict)

        reduce_cols = daf.calc_cols(include_types=[int, float])
        result_sum = daf.daf_sum(cols=reduce_cols)
        expected_sum = {'col1': '', 'col2': 15, 'col3': 18.7}
        self.assertAlmostEqual(result_sum['col2'], expected_sum['col2'], places=2)
        self.assertAlmostEqual(result_sum['col3'], expected_sum['col3'], places=2)
        #self.assertEqual(result_sum, expected_sum)

    def test_daf_sum_exclude_type_str(self):
        cols = ['col1', 'col2', 'col3']
        dtypes_dict = {'col1': str, 'col2': int, 'col3': int}
        lol = [['a', 2, 3], ['b', 5, 6], ['c', 8, 9]]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes=dtypes_dict)

        reduce_cols = daf.calc_cols(exclude_types=[str, bool, list])
        result_sum = daf.daf_sum(cols=reduce_cols)
        expected_sum = {'col1': '', 'col2': 15, 'col3': 18}
        self.assertEqual(result_sum, expected_sum)

    def test_daf_sum_empty_daf(self):
        daf = Daf()

        result_sum = daf.daf_sum()
        expected_sum = {}
        self.assertEqual(result_sum, expected_sum)

    # valuecounts_for_colname
    def test_valuecounts_for_colname(self):
        cols = ['col1', 'col2', 'col3']
        lol = [['a', 'x', 'y'], ['b', 'x', 'z'], ['a', 'y', 'y'], ['c', 'z', 'z']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        result_valuecounts = daf.valuecounts_for_colname('col2')
        expected_valuecounts = {'x': 2, 'y': 1, 'z': 1}
        self.assertEqual(result_valuecounts, expected_valuecounts)

    def test_valuecounts_for_colname_sort(self):
        cols = ['col1', 'col2', 'col3']
        lol = [['a', 'x', 'y'], ['b', 'x', 'z'], ['a', 'y', 'y'], ['c', 'z', 'z']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        result_valuecounts = daf.valuecounts_for_colname('col2', sort=True)
        expected_valuecounts = {'x': 2, 'z': 1, 'y': 1}
        self.assertEqual(result_valuecounts, expected_valuecounts)

    def test_valuecounts_for_colname_reverse_sort(self):
        cols = ['col1', 'col2', 'col3']
        lol = [['a', 'x', 'y'], ['b', 'x', 'z'], ['a', 'y', 'y'], ['c', 'z', 'z']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        result_valuecounts = daf.valuecounts_for_colname('col2', sort=True, reverse=True)
        expected_valuecounts = {'x': 2, 'y': 1, 'z': 1}
        self.assertEqual(result_valuecounts, expected_valuecounts)

    def test_valuecounts_for_colname_empty_daf(self):
        daf = Daf()

        result_valuecounts = daf.valuecounts_for_colname('col2')
        expected_valuecounts = {}
        self.assertEqual(result_valuecounts, expected_valuecounts)

    # valuecounts_for_colnames_ls
    def test_valuecounts_for_colnames_ls(self):
        cols = ['col1', 'col2', 'col3']
        lol = [['a', 'x', 'y'], ['b', 'x', 'z'], ['a', 'y', 'y'], ['c', 'z', 'z']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        result_valuecounts = daf.valuecounts_for_colnames_ls(['col2', 'col3'])
        expected_valuecounts = {'col2': {'x': 2, 'y': 1, 'z': 1}, 'col3': {'y': 2, 'z': 2}}
        self.assertEqual(result_valuecounts, expected_valuecounts)

    def test_valuecounts_for_colnames_ls_sort(self):
        cols = ['col1', 'col2', 'col3']
        lol = [['a', 'x', 'y'], ['b', 'x', 'z'], ['a', 'y', 'y'], ['c', 'z', 'z']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        result_valuecounts = daf.valuecounts_for_colnames_ls(['col2', 'col3'], sort=True)
        expected_valuecounts = {'col2': {'x': 2, 'z': 1, 'y': 1}, 'col3': {'y': 2, 'z': 2}}
        self.assertEqual(result_valuecounts, expected_valuecounts)

    def test_valuecounts_for_colnames_ls_reverse_sort(self):
        cols = ['col1', 'col2', 'col3']
        lol = [['a', 'x', 'y'], ['b', 'x', 'z'], ['a', 'y', 'y'], ['c', 'z', 'z']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        result_valuecounts = daf.valuecounts_for_colnames_ls(['col2', 'col3'], sort=True, reverse=True)
        expected_valuecounts = {'col2': {'x': 2, 'y': 1, 'z': 1}, 'col3': {'z': 2, 'y': 2}}
        self.assertEqual(result_valuecounts, expected_valuecounts)

    def test_valuecounts_for_colnames_ls_empty_daf(self):
        daf = Daf()

        result_valuecounts = daf.valuecounts_for_colnames_ls(['col2', 'col3'])
        expected_valuecounts = {'col2': {}, 'col3': {}}
        self.assertEqual(result_valuecounts, expected_valuecounts)

    def test_valuecounts_for_colnames_ls_all_columns(self):
        cols = ['col1', 'col2', 'col3']
        lol = [['a', 'x', 'y'], ['b', 'x', 'z'], ['a', 'y', 'y'], ['c', 'z', 'z']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        result_valuecounts = daf.valuecounts_for_colnames_ls()
        expected_valuecounts = {'col1': {'a': 2, 'b': 1, 'c': 1},
                                'col2': {'x': 2, 'y': 1, 'z': 1},
                                'col3': {'y': 2, 'z': 2}}
        self.assertEqual(result_valuecounts, expected_valuecounts)

    # valuecounts_for_colname_selectedby_colname
    def test_valuecounts_for_colname_selectedby_colname(self):
        cols = ['col1', 'col2', 'col3']
        lol = [['a', 'x', 'y'], ['b', 'x', 'z'], ['a', 'y', 'y'], ['c', 'z', 'z']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        result_valuecounts = daf.valuecounts_for_colname_selectedby_colname('col2', 'col1', 'a')
        expected_valuecounts = {'x': 1, 'y': 1}
        self.assertEqual(result_valuecounts, expected_valuecounts)

    def test_valuecounts_for_colname_selectedby_colname_sort(self):
        cols = ['col1', 'col2', 'col3']
        lol = [['a', 'x', 'y'], ['b', 'x', 'z'], ['a', 'y', 'y'], ['c', 'z', 'z']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        result_valuecounts = daf.valuecounts_for_colname_selectedby_colname('col2', 'col1', 'a', sort=True)
        expected_valuecounts = {'y': 1, 'x': 1}
        self.assertEqual(result_valuecounts, expected_valuecounts)

    def test_valuecounts_for_colname_selectedby_colname_reverse_sort(self):
        cols = ['col1', 'col2', 'col3']
        lol = [['a', 'x', 'y'], ['b', 'x', 'z'], ['a', 'y', 'y'], ['c', 'z', 'z']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        result_valuecounts = daf.valuecounts_for_colname_selectedby_colname('col2', 'col1', 'a', sort=True, reverse=True)
        expected_valuecounts = {'x': 1, 'y': 1}
        self.assertEqual(result_valuecounts, expected_valuecounts)

    def test_valuecounts_for_colname_selectedby_colname_not_found(self):
        cols = ['col1', 'col2', 'col3']
        lol = [['a', 'x', 'y'], ['b', 'x', 'z'], ['a', 'y', 'y'], ['c', 'z', 'z']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        result_valuecounts = daf.valuecounts_for_colname_selectedby_colname('col2', 'col1', 'not_found')
        expected_valuecounts = {}
        self.assertEqual(result_valuecounts, expected_valuecounts)

    def test_valuecounts_for_colname_selectedby_colname_empty_daf(self):
        daf = Daf()

        result_valuecounts = daf.valuecounts_for_colname_selectedby_colname('col2', 'col1', 'a')
        expected_valuecounts = {}
        self.assertEqual(result_valuecounts, expected_valuecounts)

    # valuecounts_for_colnames_ls_selectedby_colname
    def test_valuecounts_for_colnames_ls_selectedby_colname(self):
        cols = ['col1', 'col2', 'col3']
        lol = [ ['a', 'x', 'y'], 
                ['b', 'x', 'z'], 
                ['c', 'y', 'y'], 
                ['d', 'z', 'z']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        result_valuecounts = daf.valuecounts_for_colnames_ls_selectedby_colname(
            colnames_ls=['col2', 'col3'],
            selectedby_colname='col1',
            selectedby_colvalue='a'
        )
        expected_valuecounts = {'col2': {'x': 1}, 'col3': {'y': 1}}
        self.assertEqual(result_valuecounts, expected_valuecounts)

    def test_valuecounts_for_colnames_ls_selectedby_colname_sort(self):
        cols = ['col1', 'col2', 'col3']
        lol = [ ['a', 'x', 'y'], 
                ['b', 'x', 'z'], 
                ['c', 'y', 'y'], 
                ['d', 'z', 'z']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        result_valuecounts = daf.valuecounts_for_colnames_ls_selectedby_colname(
            colnames_ls=['col1', 'col2'],
            selectedby_colname='col3',
            selectedby_colvalue='y',
            sort=True
        )
        expected_valuecounts = {'col1': {'a': 1, 'c': 1}, 'col2': {'x': 1, 'y': 1}}
        self.assertEqual(result_valuecounts, expected_valuecounts)

    def test_valuecounts_for_colnames_ls_selectedby_colname_reverse_sort(self):
        cols = ['col1', 'col2', 'col3']
        lol = [ ['a', 'x', 'y'], 
                ['b', 'x', 'z'], 
                ['a', 'y', 'y'], 
                ['c', 'z', 'z']]
        daf = Daf(cols=cols, lol=lol, dtypes={'col1': str, 'col2': str, 'col3': str}, keyfield='')

        result_valuecounts = daf.valuecounts_for_colnames_ls_selectedby_colname(
            colnames_ls=['col2', 'col3'],
            selectedby_colname='col1',
            selectedby_colvalue='a',
            sort=True,
            reverse=True
        )
        expected_valuecounts = {'col2': {'x': 1, 'y': 1}, 'col3': {'y': 2}}
        self.assertEqual(result_valuecounts, expected_valuecounts)

    def test_valuecounts_for_colnames_ls_selectedby_colname_not_found(self):
        cols = ['col1', 'col2', 'col3']
        lol = [ ['a', 'x', 'y'], 
                ['b', 'x', 'z'], 
                ['c', 'y', 'y'], 
                ['d', 'z', 'z']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        result_valuecounts = daf.valuecounts_for_colnames_ls_selectedby_colname(
            colnames_ls=['col2', 'col3'],
            selectedby_colname='col1',
            selectedby_colvalue='not_found'
        )
        expected_valuecounts = {'col2': {}, 'col3': {}}
        self.assertEqual(result_valuecounts, expected_valuecounts)

    def test_valuecounts_for_colnames_ls_selectedby_colname_empty_daf(self):
        daf = Daf()

        result_valuecounts = daf.valuecounts_for_colnames_ls_selectedby_colname(
            colnames_ls=['col2', 'col3'],
            selectedby_colname='col1',
            selectedby_colvalue='a'
        )
        expected_valuecounts = {'col2': {}, 'col3': {}}
        self.assertEqual(result_valuecounts, expected_valuecounts)

    # aluecounts_for_colname1_groupedby_colname2
    def test_valuecounts_for_colname1_groupedby_colname2(self):
        cols = ['col1', 'col2', 'col3']
        lol = [ ['a', 'x', 'y'], 
                ['b', 'x', 'z'], 
                ['c', 'y', 'y'], 
                ['d', 'z', 'z']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        result_valuecounts = daf.valuecounts_for_colname1_groupedby_colname2(
            colname1='col1',
            groupedby_colname2='col2'
        )
        expected_valuecounts = {'x': {'a': 1, 'b': 1}, 'y': {'c': 1}, 'z': {'d': 1}}
        self.assertEqual(result_valuecounts, expected_valuecounts)

    def test_valuecounts_for_colname1_groupedby_colname2_sort(self):
        cols = ['col1', 'col2', 'col3']
        lol = [ ['a', 'x', 'y'], 
                ['b', 'x', 'z'], 
                ['c', 'y', 'y'], 
                ['d', 'z', 'z']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        result_valuecounts = daf.valuecounts_for_colname1_groupedby_colname2(
            colname1='col1',
            groupedby_colname2='col2',
            sort=True
        )
        expected_valuecounts = {'x': {'a': 1, 'b': 1}, 'y': {'c': 1}, 'z': {'d': 1}}
        self.assertEqual(result_valuecounts, expected_valuecounts)

    def test_valuecounts_for_colname1_groupedby_colname2_reverse_sort(self):
        cols = ['col1', 'col2', 'col3']
        lol = [ ['a', 'x', 'y'], 
                ['b', 'x', 'z'], 
                ['c', 'y', 'y'], 
                ['d', 'z', 'z']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        result_valuecounts = daf.valuecounts_for_colname1_groupedby_colname2(
            colname1='col1',
            groupedby_colname2='col2',
            sort=True,
            reverse=True
        )
        expected_valuecounts = {'x': {'a': 1, 'b': 1}, 'y': {'c': 1}, 'z': {'d': 1}}
        self.assertEqual(result_valuecounts, expected_valuecounts)

    def test_valuecounts_for_colname1_groupedby_colname2_not_found(self):
        cols = ['col1', 'col2', 'col3']
        lol = [['a', 'x', 'y'], ['b', 'x', 'z'], ['a', 'y', 'y'], ['c', 'z', 'z']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        result_valuecounts = daf.valuecounts_for_colname1_groupedby_colname2(
            colname1='col1',
            groupedby_colname2='not_found'
        )
        expected_valuecounts = {}
        self.assertEqual(result_valuecounts, expected_valuecounts)

    def test_valuecounts_for_colname1_groupedby_colname2_empty_daf(self):
        daf = Daf()

        result_valuecounts = daf.valuecounts_for_colname1_groupedby_colname2(
            colname1='col1',
            groupedby_colname2='col2'
        )
        expected_valuecounts = {}
        self.assertEqual(result_valuecounts, expected_valuecounts)

    # groupby
    def test_groupby(self):
        cols = ['col1', 'col2', 'col3']
        lol = [ ['a', 'x', 'y'], 
                ['b', 'x', 'z'], 
                ['a', 'y', 'y'], 
                ['c', 'z', 'z']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        result_dodaf = daf.groupby(colname='col2')

        lolx = [ ['a', 'x', 'y'], 
                 ['b', 'x', 'z']]
        loly = [ ['a', 'y', 'y']]
        lolz = [['c', 'z', 'z']]


        daf_x = Daf(cols=cols, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str}, lol=lolx)
        daf_y = Daf(cols=cols, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str}, lol=loly)
        daf_z = Daf(cols=cols, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str}, lol=lolz)


        expected_dodaf = {
            'x': daf_x,
            'y': daf_y,
            'z': daf_z
        }

        for colvalue, expected_daf in expected_dodaf.items():
            result_daf = result_dodaf[colvalue]
            self.assertEqual(result_daf.columns(), expected_daf.columns())
            self.assertEqual(result_daf.to_lod(), expected_daf.to_lod())

    def test_groupby_empty_daf(self):
        daf = Daf()

        result_dodaf = daf.groupby(colname='col1')

        expected_dodaf = {}
        self.assertEqual(result_dodaf, expected_dodaf)

    def test_groupby_colname_not_found(self):
        cols = ['col1', 'col2', 'col3']
        lol = [['a', 'x', 'y'], ['b', 'x', 'z'], ['a', 'y', 'y'], ['c', 'z', 'z']]
        daf = Daf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        with self.assertRaises(KeyError):
            daf.groupby(colname='not_found')

    # test __get_item__
    def test_getitem_single_row_0(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.daf_instance[0]
        expected_lol = [[1, 2, 3]]
        self.assertEqual(result.lol, expected_lol)

    def test_getitem_single_row_1(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.daf_instance[1]
        expected_lol = [[4, 5, 6]]
        self.assertEqual(result.lol, expected_lol)

    def test_getitem_single_row_2(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.daf_instance[2]
        expected_lol = [[7, 8, 9]]
        self.assertEqual(result.lol, expected_lol)

    def test_getitem_single_row_minus_1(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.daf_instance[-1]
        expected_lol = [[7, 8, 9]]
        self.assertEqual(result.lol, expected_lol)

    def test_getitem_single_row_minus_2(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.daf_instance[-2]
        expected_lol = [[4, 5, 6]]
        self.assertEqual(result.lol, expected_lol)

    def test_getitem_single_cell_01(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.daf_instance[0,1]
        expected_lol = [[2]]
        self.assertEqual(result.lol, expected_lol)

    def test_getitem_single_cell_10(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.daf_instance[1,0]
        expected_lol = [[4]]
        self.assertEqual(result.lol, expected_lol)

    def test_getitem_single_cell_11(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.daf_instance[1,1]
        expected_lol = [[5]]
        self.assertEqual(result.lol, expected_lol)

    def test_getitem_single_row_with_cols(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.daf_instance[1, :]
        expected_lol = [[4, 5, 6]]
        self.assertEqual(result.lol, expected_lol)

    def test_getitem_single_col(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.daf_instance[:, 1]
        expected_lol = [[2], [5], [8]]
        self.assertEqual(result.lol, expected_lol)

    def test_getitem_single_colname(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.daf_instance[:, 'B']
        expected_lol = [[2], [5], [8]]
        self.assertEqual(result.lol, expected_lol)

    def test_getitem_single_col_with_rows(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.daf_instance[:, 1:2]
        expected_lol = [[2], [5], [8]]
        self.assertEqual(result.lol, expected_lol)

    def test_getitem_single_col_with_reduced_rows(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.daf_instance[0:2, 1:2]
        expected_lol = [[2], [5]]
        self.assertEqual(result.lol, expected_lol)

    def test_getitem_rows_and_cols(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.daf_instance[1:3, 0:2]
        expected_result = Daf(lol=[[4, 5], [7, 8]], cols=['A', 'B'])
        self.assertEqual(result, expected_result)

    def test_getitem_col_idx_list(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.daf_instance[:, [0,2]]
        expected_result = Daf(lol=[[1, 3], [4, 6], [7, 9]], cols=['A', 'C'])
        
        # print(f"{result=}\n")
        # print(f"{expected_result=}\n")
        self.assertEqual(result, expected_result)

    def test_getitem_col_name_list(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.daf_instance[:, ['A','C']]
        expected_result = Daf(lol=[[1, 3], [4, 6], [7, 9]], cols=['A', 'C'])
        self.assertEqual(result, expected_result)

    def test_getitem_row_idx_list(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.daf_instance[[0,2]]
        expected_result = Daf(lol=[[1, 2, 3], [7, 8, 9]], cols=['A', 'B', 'C'])
        self.assertEqual(result, expected_result)

    def test_getitem_row_name_list(self):
        self.daf_instance = Daf(lol=[['a', 1, 2, 3], ['b', 4, 5, 6], ['c', 7, 8, 9]], 
                                    cols=['k', 'A', 'B', 'C'], keyfield='k')
        result = self.daf_instance[['a','c']]
        expected_result = Daf(lol=[['a', 1, 2, 3], ['c', 7, 8, 9]], cols=['k', 'A', 'B', 'C'], keyfield='k')
        self.assertEqual(result, expected_result)

    # getitem retmode==val
    def test_getitem_single_row_val(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'], retmode='val')
        result = self.daf_instance[1]
        expected_val = [4, 5, 6]
        self.assertEqual(result, expected_val)

    def test_getitem_single_row_minus_1_val(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'], retmode='val')
        result = self.daf_instance[-1]
        expected = [7, 8, 9]
        self.assertEqual(result, expected)

    def test_getitem_single_row_minus_2_val(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'], retmode='val')
        result = self.daf_instance[-2]
        expected = [4, 5, 6]
        self.assertEqual(result, expected)

    def test_getitem_single_cell_01_val(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'], retmode='val')
        result = self.daf_instance[0,1]
        expected_val = 2
        self.assertEqual(result, expected_val)

    def test_getitem_single_cell_10_val(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'], retmode='val')
        result = self.daf_instance[1,0]
        expected_val = 4
        self.assertEqual(result, expected_val)

    def test_getitem_single_cell_11_val(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'], retmode='val')
        result = self.daf_instance[1,1]
        expected_val = 5
        self.assertEqual(result, expected_val)

    def test_getitem_single_row_with_cols_val(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'], retmode='val')
        result = self.daf_instance[1, :]
        expected_val = [4, 5, 6]
        self.assertEqual(result, expected_val)

    def test_getitem_single_col_val(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'], retmode='val')
        result = self.daf_instance[:, 1]
        expected_val = [2, 5, 8]
        self.assertEqual(result, expected_val)

    def test_getitem_single_colname_val(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'], retmode='val')
        result = self.daf_instance[:, 'B']
        expected_val = [2, 5, 8]
        self.assertEqual(result, expected_val)

    def test_getitem_single_col_with_rows_val(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'], retmode='val')
        result = self.daf_instance[:, 1:2]
        expected_val = [2, 5, 8]
        self.assertEqual(result, expected_val)

    def test_getitem_single_col_with_reduced_rows_val(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'], retmode='val')
        result = self.daf_instance[0:2, 1:2]
        expected_val = [2, 5]
        self.assertEqual(result, expected_val)

    def test_getitem_rows_and_cols_val(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'], retmode='val')
        result = self.daf_instance[1:3, 0:2]
        expected_result = Daf(lol=[[4, 5], [7, 8]], cols=['A', 'B'])
        self.assertEqual(result, expected_result)

    def test_getitem_col_idx_list_val(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'], retmode='val')
        result = self.daf_instance[:, [0,2]]
        expected_result = Daf(lol=[[1, 3], [4, 6], [7, 9]], cols=['A', 'C'])
        
        # print(f"{result=}\n")
        # print(f"{expected_result=}\n")
        self.assertEqual(result, expected_result)

    def test_getitem_col_name_list_val(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'], retmode='val')
        result = self.daf_instance[:, ['A','C']]
        expected_result = Daf(lol=[[1, 3], [4, 6], [7, 9]], cols=['A', 'C'])
        self.assertEqual(result, expected_result)

    def test_getitem_row_idx_list_val(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'], retmode='val')
        result = self.daf_instance[[0,2]]
        expected_result = Daf(lol=[[1, 2, 3], [7, 8, 9]], cols=['A', 'B', 'C'])
        self.assertEqual(result, expected_result)

    def test_getitem_row_name_list_val(self):
        self.daf_instance = Daf(lol=[['a', 1, 2, 3], ['b', 4, 5, 6], ['c', 7, 8, 9]], 
                                    cols=['k', 'A', 'B', 'C'], keyfield='k', retmode='val')
        result = self.daf_instance[['a','c']]
        expected_result = Daf(lol=[['a', 1, 2, 3], ['c', 7, 8, 9]], cols=['k', 'A', 'B', 'C'], keyfield='k')
        self.assertEqual(result, expected_result)


    # getitem with .to_...()
    def test_getitem_single_row_to_val(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.daf_instance[1].to_list()
        expected_val = [4, 5, 6]
        self.assertEqual(result, expected_val)

    def test_getitem_single_row_minus_1_to_val(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.daf_instance[-1].to_list()
        expected = [7, 8, 9]
        self.assertEqual(result, expected)

    def test_getitem_single_row_minus_2_to_val(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.daf_instance[-2].to_list()
        expected = [4, 5, 6]
        self.assertEqual(result, expected)

    def test_getitem_single_cell_01_to_val(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.daf_instance[0,1].to_value()
        expected_val = 2
        self.assertEqual(result, expected_val)

    def test_getitem_single_cell_10_to_val(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.daf_instance[1,0].to_value()
        expected_val = 4
        self.assertEqual(result, expected_val)

    def test_getitem_single_cell_11_to_val(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.daf_instance[1,1].to_value()
        expected_val = 5
        self.assertEqual(result, expected_val)

    def test_getitem_single_row_with_cols_to_val(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.daf_instance[1, :].to_list()
        expected_val = [4, 5, 6]
        self.assertEqual(result, expected_val)

    def test_getitem_single_col_to_val(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.daf_instance[:, 1].to_list()
        expected_val = [2, 5, 8]
        self.assertEqual(result, expected_val)

    def test_getitem_single_colname_to_val(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.daf_instance[:, 'B'].to_list()
        expected_val = [2, 5, 8]
        self.assertEqual(result, expected_val)

    def test_getitem_single_col_with_rows_to_val(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.daf_instance[:, 1:2].to_list()
        expected_val = [2, 5, 8]
        self.assertEqual(result, expected_val)

    def test_getitem_single_col_with_reduced_rows_to_val(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.daf_instance[0:2, 1:2].to_list()
        expected_val = [2, 5]
        self.assertEqual(result, expected_val)


    def test_getitem_single_row_to_dict(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.daf_instance[1].to_dict()
        expected_val = {'A':4, 'B':5, 'C':6}
        self.assertEqual(result, expected_val)

    def test_getitem_single_row_minus_1_to_dict(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.daf_instance[-1].to_dict()
        expected_val = {'A':7, 'B':8, 'C':9}
        self.assertEqual(result, expected_val)

    def test_getitem_single_row_minus_2_to_dict(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.daf_instance[-2].to_dict()
        expected_val = {'A':4, 'B':5, 'C':6}
        self.assertEqual(result, expected_val)

    def test_getitem_single_cell_01_to_dict(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.daf_instance[0,1].to_dict()
        expected_val = {'B':2}
        self.assertEqual(result, expected_val)

    def test_getitem_single_cell_10_to_dict(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.daf_instance[1,0].to_dict()
        expected_val = {'A':4}
        self.assertEqual(result, expected_val)

    def test_getitem_single_cell_11_to_dict(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.daf_instance[1,1].to_dict()
        expected_val = {'B':5}
        self.assertEqual(result, expected_val)

    def test_getitem_single_row_with_cols_to_dict(self):
        self.daf_instance = Daf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.daf_instance[1, :].to_dict()
        expected_val = {'A':4, 'B':5, 'C':6}
        self.assertEqual(result, expected_val)






    # test transpose
    def test_transpose(self):
        self.daf_instance = Daf(lol=[[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]], cols=['A', 'B', 'C', 'D'], keyfield='A')
        result = self.daf_instance.transpose(new_keyfield='x', new_cols=['x', 'y', 'z'])
        expected_result = Daf(lol=[[1, 4, 7], [2, 5, 8], [3, 6, 9], [4, 7, 10]], keyfield='x', cols=['x', 'y', 'z']) 
        self.assertEqual(result, expected_result)


    def test_split_daf_into_chunks_lodaf(self):
        self.daf_instance = Daf(
            lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]],
            cols=['A', 'B', 'C']
            )
        max_chunk_size = 2  # Set the maximum chunk size for testing

        # Call the method to split the Daf into chunks
        chunks_lodaf = self.daf_instance.split_daf_into_chunks_lodaf(max_chunk_size)

        # Check if the length of each chunk is within the specified max_chunk_size
        for chunk in chunks_lodaf:
            self.assertLessEqual(len(chunk), max_chunk_size)

        # Check if the sum of the lengths of all chunks equals the length of the original Daf
        total_length = sum(len(chunk) for chunk in chunks_lodaf)
        self.assertEqual(total_length, len(self.daf_instance))

    # __set_item__
    def test_set_item_row_list(self):
        # Assign an entire row using a list
        # Example: Create a Daf instance with some sample data
        self.daf_instance = Daf(
            lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            cols=['A', 'B', 'C']
        )
        self.daf_instance[1] = {'A': 10, 'B': 20, 'C': 30}
        expected_result = Daf(lol=[[1, 2, 3], [10, 20, 30], [7, 8, 9]], cols=['A', 'B', 'C'])
        self.assertEqual(self.daf_instance, expected_result)

    def test_set_item_row_value(self):
        # Assign an entire row using a single value
        # Example: Create a Daf instance with some sample data
        self.daf_instance = Daf(
            lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            cols=['A', 'B', 'C']
        )
        self.daf_instance[2] = 100
        expected_result = Daf(lol=[[1, 2, 3], [4, 5, 6], [100, 100, 100]], cols=['A', 'B', 'C'])
        self.assertEqual(self.daf_instance, expected_result)

    def test_set_item_cell_value(self):
        # Assign a specific cell with a value
        # Example: Create a Daf instance with some sample data
        self.daf_instance = Daf(
            lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            cols=['A', 'B', 'C']
        )
        self.daf_instance[0, 'B'] = 50
        expected_result = Daf(lol=[[1, 50, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        self.assertEqual(self.daf_instance, expected_result)

    def test_set_item_cell_list(self):
        # Assign a specific cell with a list
        # Example: Create a Daf instance with some sample data
        self.daf_instance = Daf(
            lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            cols=['A', 'B', 'C']
        )
        
        self.daf_instance[1, 'A'] = [100, 200, 300]
        expected_result = Daf(lol=[[1, 2, 3], [[100, 200, 300], 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        
        self.longMessage = True
        
        if self.daf_instance != expected_result:
            print (f"test_set_item_cell_list result:\n{self.daf_instance}\nexpected{expected_result}")

        self.assertEqual(self.daf_instance, expected_result)
        
    def test_set_item_row_range_list(self):
        # Assign values in a range of columns in a specific row with a list
        # Example: Create a Daf instance with some sample data
        self.daf_instance = Daf(
            lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            cols=['A', 'B', 'C']
        )
        self.daf_instance[1, 1:3] = [99, 88]
        expected_result = Daf(lol=[[1, 2, 3], [4, 99, 88], [7, 8, 9]], cols=['A', 'B', 'C'])
        self.assertEqual(self.daf_instance, expected_result)

    def test_set_item_row_range_value(self):
        # Assign a single value in a range of columns in a specific row
        # Example: Create a Daf instance with some sample data
        self.daf_instance = Daf(
            lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            cols=['A', 'B', 'C']
        )
        self.daf_instance[0, 1:3] = 77
        expected_result = Daf(lol=[[1, 77, 77], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        self.assertEqual(self.daf_instance, expected_result)

    def test_set_item_col_list(self):
        # Assign an entire column with a list
        # Example: Create a Daf instance with some sample data
        self.daf_instance = Daf(
            lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            cols=['A', 'B', 'C']
            )
        #breakpoint() #temp
        self.daf_instance[:, 'B'] = [55, 66, 77]
        expected_result = Daf(lol=[[1, 55, 3], [4, 66, 6], [7, 77, 9]], cols=['A', 'B', 'C'])
        if self.daf_instance != expected_result:
            print (f"test_set_item_col_list result:\n{self.daf_instance}\nexpected{expected_result}")

        self.assertEqual(self.daf_instance, expected_result)

    def test_set_item_col_range_list(self):
        # Assign values in a range of rows in a specific column with a list
        # Example: Create a Daf instance with some sample data
        self.daf_instance = Daf(
            lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            cols=['A', 'B', 'C']
        )
        self.daf_instance[1:3, 'B'] = [44, 33]
        expected_result = Daf(lol=[[1, 2, 3], [4, 44, 6], [7, 33, 9]], cols=['A', 'B', 'C'])
        self.assertEqual(self.daf_instance, expected_result)

    # select_where
    def test_select_where_basic_condition(self):
        # Test a basic condition where col1 values are greater than 2
        daf = Daf(cols=['col1', 'col2'], lol=[[1, 4], [2, 5], [3, 6]])

        result_daf = daf.select_where(lambda row: row['col1'] > 2)
        expected_data = Daf(cols=['col1', 'col2'], lol=[[3, 6]])
        self.assertEqual(result_daf, expected_data)

    def test_select_where_invalid_condition_runtime_error(self):
        # Test an invalid condition causing a runtime error
        daf = Daf(cols=['col1', 'col2'], lol=[[1, 4], [2, 5], [3, 6]])

        with self.assertRaises(ZeroDivisionError) as context:
            daf.select_where(lambda row: 1 / 0)

        self.assertIn("division by zero", str(context.exception))

    def test_select_where_empty_result(self):
        # Test a condition that results in an empty Daf
        daf = Daf(cols=['col1', 'col2'], lol=[[1, 4], [2, 5], [3, 6]])

        result_daf = daf.select_where(lambda row: row['col1'] > 10)
        expected_data = Daf(cols=['col1', 'col2'], lol=[])
        self.assertEqual(result_daf, expected_data)

    def test_select_where_complex_condition(self):
        # Test a complex condition involving multiple columns
        daf = Daf(cols=['col1', 'col2'], lol=[[1, 4], [2, 5], [3, 6]])

        result_daf = daf.select_where(lambda row: row['col1'] > 1 and row['col2'] < 6)
        expected_data = Daf(cols=['col1', 'col2'], lol=[[2, 5]])
        self.assertEqual(result_daf, expected_data)

    def test_select_where_complex_condition_indexes(self):
        # Test a complex condition involving multiple columns
        daf = Daf(cols=['col1', 'col2'], lol=[[1, 4], [2, 5], [3, 6]])

        result_daf = daf.select_where(lambda row: bool(list(row.values())[0] > 1 and list(row.values())[1] < 6))
        expected_data = Daf(cols=['col1', 'col2'], lol=[[2, 5]])
        self.assertEqual(result_daf, expected_data)

    # select_where_idxs
    def test_select_where_idxs_basic_condition(self):
        # Test a basic condition where col1 values are greater than 2
        daf = Daf(cols=['col1', 'col2'], lol=[[1, 4], [2, 5], [3, 6]])

        result_idxs_list = daf.select_where_idxs(lambda row: row['col1'] > 2)
        expected_data = [2]
        self.assertEqual(result_idxs_list, expected_data)

    def test_select_where_idxs_invalid_condition_runtime_error(self):
        # Test an invalid condition causing a runtime error
        daf = Daf(cols=['col1', 'col2'], lol=[[1, 4], [2, 5], [3, 6]])

        with self.assertRaises(ZeroDivisionError) as context:
            daf.select_where_idxs(lambda row: 1 / 0)

        self.assertIn("division by zero", str(context.exception))

    def test_select_where_idxs_empty_result(self):
        # Test a condition that results in an empty Daf
        daf = Daf(cols=['col1', 'col2'], lol=[[1, 4], [2, 5], [3, 6]])

        result_list = daf.select_where_idxs(lambda row: row['col1'] > 10)
        expected_data = []
        self.assertEqual(result_list, expected_data)

    def test_select_where_idxs_complex_condition(self):
        # Test a complex condition involving multiple columns
        daf = Daf(cols=['col1', 'col2'], lol=[[1, 4], [2, 5], [3, 6]])

        result_list = daf.select_where_idxs(lambda row: row['col1'] > 1 and row['col2'] < 6)
        expected_data = [1]
        self.assertEqual(result_list, expected_data)

    def test_select_where_idxs_complex_condition_indexes(self):
        # Test a complex condition involving multiple columns
        daf = Daf(cols=['col1', 'col2'], lol=[[1, 4], [2, 5], [3, 6]])

        result_list = daf.select_where_idxs(lambda row: bool(list(row.values())[0] > 1 and list(row.values())[1] < 6))
        expected_data = [1]
        self.assertEqual(result_list, expected_data)


    # test test_from_cols_dol
    def test_from_cols_dol_empty_input(self):
        # Test creating Daf instance from empty cols_dol
        cols_dol = {}
        result_daf = Daf.from_cols_dol(cols_dol)
        expected_daf = Daf()
        self.assertEqual(result_daf, expected_daf)

    def test_from_cols_dol_basic_input(self):
        # Test creating Daf instance from cols_dol with basic data
        cols_dol = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
        result_daf = Daf.from_cols_dol(cols_dol)
        expected_daf = Daf(cols=['A', 'B', 'C'], lol=[[1, 4, 7], [2, 5, 8], [3, 6, 9]])
        self.assertEqual(result_daf, expected_daf)

    def test_from_cols_dol_with_keyfield(self):
        # Test creating Daf instance from cols_dol with keyfield specified
        cols_dol = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
        result_daf = Daf.from_cols_dol(cols_dol, keyfield='A')
        expected_daf = Daf(cols=['A', 'B', 'C'], lol=[[1, 4, 7], [2, 5, 8], [3, 6, 9]], keyfield='A')
        self.assertEqual(result_daf, expected_daf)

    def test_from_cols_dol_with_dtypes(self):
        # Test creating Daf instance from cols_dol with specified dtype
        cols_dol = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
        dtypes = {'A': int, 'B': float, 'C': str}
        result_daf = Daf.from_cols_dol(cols_dol).apply_dtypes(dtypes=dtypes, from_str=False)
        expected_daf = Daf(cols=['A', 'B', 'C'], lol=[[1, 4.0, '7'], [2, 5.0, '8'], [3, 6.0, '9']], dtypes=dtypes)
        self.assertEqual(result_daf, expected_daf)

    # # to_dict
    # def test_to_dict_empty_daf(self):
        # # Test to_dict() on an empty Daf instance
        # daf = Daf()
        # #result_dict = daf.to_dict()
        # #expected_daf = {'cols': [], 'lol': []}
        # self.assertEqual(daf.lol, [])
        # self.assertEqual(daf.kd, {})
        # self.assertEqual(daf.kd, {})

    # def test_to_dict_with_data(self):
        # # Test to_dict() on a Daf instance with data
        # daf = Daf(cols=['A', 'B', 'C'], lol=[[1, 4, 7], [2, 5, 8], [3, 6, 9]])
        # #result_dict = daf.to_dict()
        # expected_daf = Daf('cols'= ['A', 'B', 'C'], 'lol'= [[1, 4, 7], [2, 5, 8], [3, 6, 9]])
        # self.assertEqual(result_daf, expected_daf)

    # def test_to_dict_with_keyfield_and_dtypes(self):
        # # Test to_dict() on a Daf instance with keyfield and dtype
        # daf = Daf(cols=['A', 'B', 'C'], 
                    # lol=[[1, 4, 7], [2, 5, 8], [3, 6, 9]], 
                    # keyfield='A', 
                    # dtypes={'A': int, 'B': float, 'C': int})
        # #result_dict = daf.to_dict()
        # expected_daf = Daf('cols'= ['A', 'B', 'C'], 'lol'= [[1, 4, 7], [2, 5, 8], [3, 6, 9]])
        # self.assertEqual(result_daf, expected_daf)

    # apply_formulas
    def test_apply_formulas_basic_absolute(self):
        # Test apply_formulas with basic example
        example_daf = Daf(cols=['A', 'B', 'C'], lol=[[1, 2, 0], [4, 5, 0], [7, 8, 0], [0, 0, 0]])
        formulas_daf = Daf(cols=['A', 'B', 'C'],
                             lol=[['', '', "$d[0,0]+$d[0,1]"],
                                  ['', '', "$d[1,0]+$d[1,1]"],
                                  ['', '', "$d[2,0]+$d[2,1]"],
                                  ["sum($d[0:3,$c])", "sum($d[0:3,$c])", "sum($d[0:3,$c])"]]
                             )
        expected_daf = Daf(cols=['A', 'B', 'C'], lol=[[1, 2, 3], [4, 5, 9], [7, 8, 15], [12, 15, 27]])

        example_daf.apply_formulas(formulas_daf)
        self.assertEqual(example_daf, expected_daf)

    def test_apply_formulas_basic_relative(self):
        # Test apply_formulas with basic example
        example_daf = Daf(cols=['A', 'B', 'C'], lol=[[1, 2, 0], [4, 5, 0], [7, 8, 0], [0, 0, 0]])
        formulas_daf = Daf(cols=['A', 'B', 'C'],
                             lol=[['', '', "$d[$r,0]+$d[$r,1]"],
                                  ['', '', "$d[$r,($c-2)]+$d[$r,($c-1)]"],
                                  ['', '', "sum($d[$r,0:2])"],
                                  ["sum($d[0:3,$c])", "sum($d[:-1,$c])", "sum($d[:$r,$c])"]]
                             )
        expected_result = Daf(cols=['A', 'B', 'C'], lol=[[1, 2, 3], [4, 5, 9], [7, 8, 15], [12, 15, 27]])

        example_daf.apply_formulas(formulas_daf)
        self.assertEqual(example_daf, expected_result)

    def test_apply_formulas_no_changes(self):
        # Test apply_formulas with no changes expected
        example_daf = Daf(cols=['A', 'B', 'C'], lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0]])
        formulas_daf = Daf(cols=['A', 'B', 'C'], lol=[['', '', ''], ['', '', ''], ['', '', ''], ['', '', '']])
        expected_result = Daf(cols=['A', 'B', 'C'], lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0]])

        example_daf.apply_formulas(formulas_daf)
        self.assertEqual(example_daf, expected_result)

    def test_apply_formulas_excessive_loops(self):
        # Test apply_formulas resulting in excessive loops
        example_daf = Daf(cols=['A', 'B', 'C'], lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0]])
        formulas_daf = Daf(cols=['A', 'B', 'C'],
                             lol=[['', '', "$d[0,0]+$d[0,1]"],
                                  ['', '', "$d[1,0]+$d[1,1]"],
                                  ['$d[2,2]', '', "$d[2,0]+$d[2,1]"],     # this is circular
                                  ["sum($d[0:3,'A'])", "sum($d[0:3,'B'])", "sum($d[0:3,'C'])"]]
                             )

        with self.assertRaises(RuntimeError) as context:
            example_daf.apply_formulas(formulas_daf)

        self.assertIn("apply_formulas is resulting in excessive evaluation loops.", str(context.exception))

    def test_generate_spreadsheet_column_names_list(self):
        # Test for 0 columns
        self.assertEqual(utils._generate_spreadsheet_column_names_list(0), [])

        # Test for 1 column
        self.assertEqual(utils._generate_spreadsheet_column_names_list(1), ['A'])

        # Test for 5 columns
        self.assertEqual(utils._generate_spreadsheet_column_names_list(5), ['A', 'B', 'C', 'D', 'E'])

        # Test for 27 columns
        self.assertEqual(utils._generate_spreadsheet_column_names_list(27), ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'AA'])

        # Test for 52 columns
        self.assertEqual(utils._generate_spreadsheet_column_names_list(52), ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI', 'AJ', 'AK', 'AL', 'AM', 'AN', 'AO', 'AP', 'AQ', 'AR', 'AS', 'AT', 'AU', 'AV', 'AW', 'AX', 'AY', 'AZ'])

        # Test for 53 columns
        self.assertEqual(utils._generate_spreadsheet_column_names_list(53), ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI', 'AJ', 'AK', 'AL', 'AM', 'AN', 'AO', 'AP', 'AQ', 'AR', 'AS', 'AT', 'AU', 'AV', 'AW', 'AX', 'AY', 'AZ', 'BA'])


    # from_lod_to_cols
    def test_from_lod_to_cols_empty_lod(self):
        result = Daf.from_lod_to_cols([], cols=['A', 'B', 'C'], keyfield='Key')
        self.assertEqual(result.columns(), ['A', 'B', 'C'])
        self.assertEqual(result.lol, [])

    def test_from_lod_to_cols_no_cols_specified(self):
        lod = [{'A': 1, 'B': 2, 'C': 3}, {'A': 4, 'B': 5, 'C': 6}, {'A': 7, 'B': 8, 'C': 9}]
        result = Daf.from_lod_to_cols(lod, keyfield='key')
        self.assertEqual(result.columns(), ['key', 'A', 'B', 'C'])
        self.assertEqual(result.lol, [['A', 1, 4, 7], ['B', 2, 5, 8], ['C', 3, 6, 9]])

    def test_from_lod_to_cols_with_cols(self):
        lod = [{'A': 1, 'B': 2, 'C': 3}, {'A': 4, 'B': 5, 'C': 6}, {'A': 7, 'B': 8, 'C': 9}]
        result = Daf.from_lod_to_cols(lod, cols=['Feature', 'Try 1', 'Try 2', 'Try 3'], keyfield='Feature')
        self.assertEqual(result.columns(), ['Feature', 'Try 1', 'Try 2', 'Try 3'])
        self.assertEqual(result.lol, [['A', 1, 4, 7], ['B', 2, 5, 8], ['C', 3, 6, 9]])

    # apply
    def test_apply_row(self):
        daf = Daf.from_lod([  {'a': 1, 'b': 2}, 
                                {'a': 3, 'b': 4}])

        def transform_row(
                row: dict, 
                cols=None,                      # columns included in the reduce operation.
                **kwargs):
            return {'a': row['a'] * 2, 'b': row['b'] * 3}

        result_daf = daf.apply(transform_row, by='row')
        expected_result = Daf.from_lod([{'a': 2, 'b': 6}, {'a': 6, 'b': 12}])

        self.assertEqual(result_daf, expected_result)

    # def test_apply_col(self):
        # daf = Daf.from_lod([  {'a': 1, 'b': 2}, 
                                # {'a': 3, 'b': 4}])

        # def transform_col(col, cols, **kwargs):
            # col[0] = col[0] * 2
            # col[1] = col[1] * 3
            # return col

        # result_daf = daf.apply(transform_col, by='col')
        # expected_result = Daf.from_lod([
                                # {'a': 2, 'b': 4}, 
                                # {'a': 9, 'b': 12}])

        # self.assertEqual(result_daf, expected_result)


    def test_set_col2_from_col1_using_regex_select(self):
        # Initialize an instance of your class
        
        # Set up sample data for testing
        cols_dol = {'col1': ['abc (123)', 'def (456)', 'ghi (789)'],
                'col2': [None, None, None]}
        my_daf = Daf.from_cols_dol(cols_dol)

        # Call the method to apply the regex select
        my_daf.set_col2_from_col1_using_regex_select('col1', 'col2', r'\((\d+)\)')
        
        col2_expected = Daf(lol=[['123'], ['456'], ['789']], cols=['col2'])
        
        # Assert the expected results
        self.assertEqual(my_daf[:, 'col2'], col2_expected)
        

    def test_groupby_cols_reduce(self):

        groupby_colnames = ['gender', 'religion', 'zipcode']
        reduce_colnames  = ['cancer', 'covid19', 'gun', 'auto']
            
        cols = ['gender', 'religion', 'zipcode', 'cancer', 'covid19', 'gun', 'auto']
        lol = [
            ['M', 'C', 90001,  1,  2,  3,  4],
            ['M', 'C', 90001,  5,  6,  7,  8],
            ['M', 'C', 90002,  9, 10, 11, 12],
            ['M', 'C', 90002, 13, 14, 15, 16],
            ['M', 'J', 90001,  1,  2,  3,  4],
            ['M', 'J', 90001, 13, 14, 15, 16],
            ['M', 'J', 90002,  5,  6,  7,  8],
            ['M', 'J', 90002,  9, 10, 11, 12],
            ['M', 'I', 90001, 13, 14, 15, 16],
            ['M', 'I', 90001,  1,  2,  3,  4],
            ['M', 'I', 90002,  4,  3,  2,  1],
            ['M', 'I', 90002,  9, 10, 11, 12],
            ['F', 'C', 90001,  4,  3,  2,  1],
            ['F', 'C', 90001,  5,  6,  7,  8],
            ['F', 'C', 90002,  4,  3,  2,  1],
            ['F', 'C', 90002, 13, 14, 15, 16],
            ['F', 'J', 90001,  4,  3,  2,  1],
            ['F', 'J', 90001,  1,  2,  3,  4],
            ['F', 'J', 90002,  8,  7,  6,  5],
            ['F', 'J', 90002,  1,  2,  3,  4],
            ['F', 'I', 90001,  8,  7,  6,  5],
            ['F', 'I', 90001,  5,  6,  7,  8],
            ['F', 'I', 90002,  8,  7,  6,  5],
            ['F', 'I', 90002, 13, 14, 15, 16],
            ]
            
        data_table_daf = Daf(cols=cols, lol=lol)
            
            
        grouped_and_summed_daf = data_table_daf.groupby_cols_reduce(
            groupby_colnames=groupby_colnames, 
            func = Daf.sum_np,
            by='table',                                     # determines how the func is applied.
            reduce_cols = reduce_colnames,                  # columns included in the reduce operation.
            )

        expected_lol = [
            ['M', 'C', 90001,  6,  8, 10, 12],
            ['M', 'C', 90002, 22, 24, 26, 28],
            ['M', 'J', 90001, 14, 16, 18, 20],
            ['M', 'J', 90002, 14, 16, 18, 20],
            ['M', 'I', 90001, 14, 16, 18, 20],
            ['M', 'I', 90002, 13, 13, 13, 13],
            ['F', 'C', 90001,  9,  9,  9,  9],
            ['F', 'C', 90002, 17, 17, 17, 17],
            ['F', 'J', 90001,  5,  5,  5,  5],
            ['F', 'J', 90002,  9,  9,  9,  9],
            ['F', 'I', 90001, 13, 13, 13, 13],
            ['F', 'I', 90002, 21, 21, 21, 21],
            ]

        self.assertEqual(grouped_and_summed_daf.lol, expected_lol)
        

    def test_is_d1_in_d2(self):
        # Test case where d1 is a subset of d2
        d1 = {'a': 1, 'b': 2}
        d2 = {'a': 1, 'b': 2, 'c': 3}
        assert utils.is_d1_in_d2(d1, d2) == True

        # Test case where d1 is equal to d2
        d1 = {'a': 1, 'b': 2}
        d2 = {'a': 1, 'b': 2}
        assert utils.is_d1_in_d2(d1, d2) == True

        # Test case where d1 is not a subset of d2
        d1 = {'a': 1, 'b': 2}
        d2 = {'a': 1, 'c': 3}
        assert utils.is_d1_in_d2(d1, d2) == False

        # Test case where d1 is an empty dictionary
        d1 = {}
        d2 = {'a': 1, 'b': 2}
        assert utils.is_d1_in_d2(d1, d2) == True

        # Test case where d2 is an empty dictionary
        d1 = {'a': 1, 'b': 2}
        d2 = {}
        assert utils.is_d1_in_d2(d1, d2) == False

        # Test case where both d1 and d2 are empty dictionaries
        d1 = {}
        d2 = {}
        assert utils.is_d1_in_d2(d1, d2) == True

        # Test case with mixed types of keys and values
        d1 = {'a': 1, 2: 'b', 'c': True}
        d2 = {'a': 1, 'b': 2, 'c': True}
        assert utils.is_d1_in_d2(d1, d2) == False

        # Test case where d1 has additional fields not present in d2
        d1 = {'a': 1, 'b': 2, 'd': 4}
        d2 = {'a': 1, 'b': 2}
        assert utils.is_d1_in_d2(d1, d2) == False


    def test_set_lol_with_new_lol(self):
        # Create a Daf instance for testing
        self.cols = ['ID', 'Name', 'Age']
        self.lol = [[1, 'John', 30], [2, 'Alice', 25], [3, 'Bob', 35]]
        self.daf = Daf(cols=self.cols, lol=self.lol)

        # Define a new list-of-lists (lol)
        new_lol = [[4, 'David', 40], [5, 'Eve', 28]]

        # Call the set_lol method with the new lol
        self.daf.set_lol(new_lol)

        # Check if lol is set correctly
        self.assertEqual(self.daf.lol, new_lol)

    def test_set_lol_with_empty_lol(self):
        # Create a Daf instance for testing
        self.cols = ['ID', 'Name', 'Age']
        self.lol = [[1, 'John', 30], [2, 'Alice', 25], [3, 'Bob', 35]]
        self.daf = Daf(cols=self.cols, lol=self.lol)

        # Define an empty lol
        new_lol = []

        # Call the set_lol method with the empty lol
        self.daf.set_lol(new_lol)

        # Check if lol is set correctly to empty list
        self.assertEqual(self.daf.lol, new_lol)

    def test_set_lol_recalculates_kd(self):
        # Create a Daf instance for testing
        self.cols = ['ID', 'Name', 'Age']
        self.lol = [[1, 'John', 30], [2, 'Alice', 25], [3, 'Bob', 35]]
        self.daf = Daf(cols=self.cols, lol=self.lol)

        # Define a new list-of-lists (lol)
        new_lol = [[4, 'David', 40], [5, 'Eve', 28]]

        # Call the set_lol method with the new lol
        self.daf.set_lol(new_lol)

        # Check if kd is recalculated
        self.assertIsNotNone(self.daf.kd)
        

    # wide to narrow
    def test_wide_to_narrow_conversion_basic(self):
        # Initialize a wide Daffodil dataframe for testing
        wide_daf = Daf(cols=['ID', 'A', 'B'], lol=[[1, 10, 100], [2, 20, 200], [3, 30, 300]])
        
        # Test if the method converts a wide DataFrame to a narrow format
        narrow_daf = wide_daf.wide_to_narrow(id_cols=['ID'])
        
        # Check the exact content of the narrow DataFrame
        expected_content = [
            {'ID': 1, 'varname': 'A', 'value': 10},
            {'ID': 1, 'varname': 'B', 'value': 100},
            {'ID': 2, 'varname': 'A', 'value': 20},
            {'ID': 2, 'varname': 'B', 'value': 200},
            {'ID': 3, 'varname': 'A', 'value': 30},
            {'ID': 3, 'varname': 'B', 'value': 300}
        ]
        self.assertEqual(len(narrow_daf), 6)  # Ensure correct number of rows
        
        # Check each row in the narrow DataFrame
        for i, row in enumerate(narrow_daf):
            self.assertDictEqual(row, expected_content[i])
        
        # Add more assertions based on the expected structure of the narrow DataFrame
        
        
    # def test_wide_to_narrow_id_cols_handling_additional_columns(self):
        # # Initialize a wide Daffodil dataframe for testing
        # wide_daf = Daf(cols=['ID', 'A', 'B'], lol=[[1, 10, 100], [2, 20, 200], [3, 30, 300]])
        
        # # Test the handling of identifier columns
        # narrow_daf = wide_daf.wide_to_narrow(id_cols=['ID', 'SomeOtherColumn'])
        # self.assertEqual(len(narrow_daf.columns()), 4)  # ID, SomeOtherColumn, varname, value
        
        # # Add more assertions based on the expected handling of identifier columns
        
    def test_wide_to_narrow_input_data_types_invalid_id_cols(self):
        # Initialize a wide Daffodil dataframe for testing
        wide_daf = Daf(cols=['ID', 'A', 'B'], lol=[[1, 10, 100], [2, 20, 200], [3, 30, 300]])
        
        # Test the handling of different input data types
        with self.assertRaises(TypeError):
            wide_daf.wide_to_narrow(id_cols='ID')  # id_cols should be a list
        
        # Add more assertions based on the expected behavior with different data types

    # narrow_to_wide
    def test_narrow_to_wide_conversion_basic(self):
        # Initialize a narrow Daffodil dataframe for testing
        narrow_daf = Daf(
                cols=['ID', 'varname', 'value'], 
                lol=[[1, 'A', 10], 
                    [1, 'B', 100], 
                    [2, 'A', 20], 
                    [2, 'B', 200], 
                    [3, 'A', 30], 
                    [3, 'B', 300]])
        
        # Test if the method converts a narrow DataFrame to a wide format
        wide_daf = narrow_daf.narrow_to_wide(id_cols=['ID'], varname_col='varname', value_col='value')
        
        # Check the exact content of the wide DataFrame
        expected_content = Daf(cols=['ID', 'A', 'B'], lol=[[1, 10, 100], [2, 20, 200], [3, 30, 300]])
        self.assertEqual(wide_daf, expected_content)
        
        # Add more assertions based on the expected structure of the wide DataFrame
        
    def test_narrow_to_wide_id_cols_handling_additional_columns(self):
        # Initialize a narrow Daffodil dataframe for testing
        narrow_daf = Daf(
                cols=['ID', 'varname', 'value'], 
                lol=[[1, 'A', 10], 
                    [1, 'B', 100], 
                    [2, 'A', 20], 
                    [2, 'B', 200], 
                    [3, 'A', 30], 
                    [3, 'B', 300]])
        
        # Test the handling of identifier columns
        wide_daf = narrow_daf.narrow_to_wide(id_cols=['ID'], varname_col='varname', value_col='value')
        self.assertEqual(len(wide_daf.columns()), 3)  # ID, A, B
        
        # Add more assertions based on the expected handling of identifier columns
        
    #==============================
    def test_reduce_lol_cols(self):
        lol = [['a', 'b', 'c', 'd'], [1, 2, 3, 4], ['x', 'y', 'z', 'w'], [5, 6, 7, 8]]
        reduced_lol = utils.reduce_lol_cols(lol, max_cols=3, divider_str='...')
        assert reduced_lol == [['a', 'b', '...', 'd'], [1, 2, '...', 4], ['x', 'y', '...', 'w'], [5, 6, '...', 8]], \
                  f"Expected: [['a', 'b', '...', 'd'], [1, 2, '...', 4], ['x', 'y', '...', 'w'], [5, 6, '...', 8]], Got: {reduced_lol}"

        

class TestDafIteration(unittest.TestCase):

    def setUp(self):
        self.header = {'name': 0, 'age': 1}
        self.data = [['Alice', 30], ['Bob', 25]]
        self.daf = Daf(lol=self.data, hd=self.header)

    def test_iter_dict_returns_correct_dicts(self):
        expected_results = [
            {'name': 'Alice', 'age': 30},
            {'name': 'Bob', 'age': 25}
        ]

        results = list(self.daf.iter_dict())
        self.assertEqual(results, expected_results)

    def test_iter_klist_returns_correct_keyedlists(self):
        results = list(self.daf.iter_klist())

        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], KeyedList)
        self.assertEqual(results[0]['name'], 'Alice')
        self.assertEqual(results[0]['age'], 30)
        self.assertEqual(results[1]['name'], 'Bob')
        self.assertEqual(results[1]['age'], 25)

    def test_iter_klist_modifies_underlying_data(self):
        for row in self.daf.iter_klist():
            row['age'] += 1

        expected_data = [['Alice', 31], ['Bob', 26]]
        self.assertEqual(self.data, expected_data)

    def test_default_iteration_mode_dict(self):
        self.daf._itermode = Daf.ITERMODE_DICT

        expected_results = [
            {'name': 'Alice', 'age': 30},
            {'name': 'Bob', 'age': 25}
        ]

        results = list(iter(self.daf))
        self.assertEqual(results, expected_results)

    def test_default_iteration_mode_keyedlist(self):
        self.daf._itermode = Daf.ITERMODE_KEYEDLIST

        results = list(iter(self.daf))

        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], KeyedList)
        self.assertEqual(results[0]['name'], 'Alice')
        self.assertEqual(results[0]['age'], 30)
        self.assertEqual(results[1]['name'], 'Bob')
        self.assertEqual(results[1]['age'], 25)

    def test_iteration_resets_after_stop_iteration(self):
        # First iteration
        results_1 = list(iter(self.daf))

        # Second iteration
        results_2 = list(iter(self.daf))

        # Verify that the results of both iterations are the same
        self.assertEqual(results_1, results_2)




class TestDafFlattenMethod(unittest.TestCase):
    
    def test_flatten_no_columns_to_flatten(self):
        instance = Daf(
            lol=[],
            dtypes={},
            hd={}
        )
        result = instance.flatten()
        self.assertEqual(result.lol, [])

    def test_flatten_columns_with_lists_to_flatten(self):
        instance = Daf(
            lol=   [[1, [2, 3]], 
                    [4, [5, 6]]],
            dtypes={'col1': int, 'col2': list},
            hd=    {'col1': 0, 'col2': 1}
        )
        result = instance.flatten()
        self.assertEqual(result.lol, [[1, '[2, 3]'], [4, '[5, 6]']])

    def test_flatten_columns_with_dicts_to_flatten(self):
        
        use_pyon = True
        
        instance = Daf(
            lol=[[1, {'a': 2}], [3, {'b': 4}]],
            dtypes={'col1': int, 'col2': dict},
            hd={'col1': 0, 'col2': 1}
        )
        result = instance.flatten()
        if use_pyon:
            self.assertEqual(result.lol, [[1, "{'a': 2}"], [3, "{'b': 4}"]])
        else:
            self.assertEqual(result.lol, [[1, '{"a": 2}'], [3, '{"b": 4}']])

    def test_flatten_bool_columns_to_int(self):
        instance = Daf(
            lol=[[1, True], [3, False]],
            dtypes={'col1': int, 'col2': bool},
            hd={'col1': 0, 'col2': 1}
        )
        result = instance.flatten()
        self.assertEqual(result.lol, [[1, 1], [3, 0]])

    def test_flatten_bool_columns_to_int_singlar_type(self):
        instance = Daf(
            lol=[[True, True], [True, False]],
            dtypes=bool,
            cols=['col1', 'col2']
        )
        result = instance.flatten()
        self.assertEqual(result.lol, [[1, 1], [1, 0]])

    def test_flatten_mixed_columns(self):
        
        use_pyon = True
        
        instance = Daf(
            lol=[[1, [2, 3], {'a': 4}, True], [4, [5, 6], {'b': 7}, False]],
            dtypes={'col1': int, 'col2': list, 'col3': dict, 'col4': bool},
            hd={'col1': 0, 'col2': 1, 'col3': 2, 'col4': 3}
        )
        result = instance.flatten()
        
        if use_pyon:
            self.assertEqual(result.lol, [
                [1, '[2, 3]', "{'a': 4}", 1],
                [4, '[5, 6]', "{'b': 7}", 0]
            ])
        else:
            self.assertEqual(result.lol, [
                [1, '[2, 3]', '{"a": 4}', 1],
                [4, '[5, 6]', '{"b": 7}', 0]
            ])

    def test_flatten_mixed_columns_missing_first_dtype(self):
        
        use_pyon = True
        
        instance = Daf(
            lol=[[1, [2, 3], {'a': 4}, True], [4, [5, 6], {'b': 7}, False]],
            cols=['col1', 'col2', 'col3', 'col4'],
            dtypes={'col2': list, 'col3': dict, 'col4': bool},
            hd={'col1': 0, 'col2': 1, 'col3': 2, 'col4': 3}
        )
        
        result = instance.flatten()
        if use_pyon:
            self.assertEqual(result.lol, [
                [1, '[2, 3]', "{'a': 4}", 1],
                [4, '[5, 6]', "{'b': 7}", 0]
            ])
        else:
            self.assertEqual(result.lol, [
                [1, '[2, 3]', '{"a": 4}', 1],
                [4, '[5, 6]', '{"b": 7}', 0]
            ])

    # def test_flatten_no_hd_raises_runtime_error(self):
        # instance = Daf(
            # lol=[[1, {'a': 2}], [3, {'b': 4}]],
            # dtypes={'col1': int, 'col2': dict},
            # hd=None
        # )
        # with self.assertRaises(RuntimeError):
            # result = instance.flatten()
            

class TestDafStripMethod(unittest.TestCase):

    def test_strip_default_space(self):
        instance = Daf(
            lol=[['  hello  ', ' world ', 'test']],
            dtypes={},
            hd={'col1': 0, 'col2': 1, 'col3': 2}
        )
        result = instance.strip()
        self.assertEqual(result.lol, [['hello', 'world', 'test']])

    def test_strip_specific_chars(self):
        instance = Daf(
            lol=[['"hello"', '(world)', '[test]']],
            dtypes={},
            hd={'col1': 0, 'col2': 1, 'col3': 2}
        )
        result = instance.strip('"()[]')
        self.assertEqual(result.lol, [['hello', 'world', 'test']])

    def test_strip_mixed_chars(self):
        instance = Daf(
            lol=[['**hello**', '~~world~~', '^^test^^']],
            dtypes={},
            hd={'col1': 0, 'col2': 1, 'col3': 2}
        )
        result = instance.strip('*~^')
        self.assertEqual(result.lol, [['hello', 'world', 'test']])

    def test_strip_no_str_values(self):
        instance = Daf(
            lol=[[123, True, None, 45.67]],
            dtypes={},
            hd={'col1': 0, 'col2': 1, 'col3': 2, 'col4': 3}
        )
        result = instance.strip()
        self.assertEqual(result.lol, [[123, True, None, 45.67]])

    def test_strip_empty_strings(self):
        instance = Daf(
            lol=[['', ' ', '   ']],
            dtypes={},
            hd={'col1': 0, 'col2': 1, 'col3': 2}
        )
        result = instance.strip()
        self.assertEqual(result.lol, [['', '', '']])

    def test_strip_multiple_rows(self):
        instance = Daf(
            lol=[
                ['  hello  ', ' world ', 'test'],
                ['**foo**', '  bar  ', 'baz  '],
                ['~abc~', '~~def~~', 'ghi']
            ],
            dtypes={},
            hd={'col1': 0, 'col2': 1, 'col3': 2}
        )
        result = instance.strip(' *~')
        self.assertEqual(result.lol, [
            ['hello', 'world', 'test'],
            ['foo', 'bar', 'baz'],
            ['abc', 'def', 'ghi']
        ])


class TestDafJsonMethods(unittest.TestCase):

    def test_to_json(self):
        instance = Daf(
            lol=[[1, 2, 3], [4, 5, 6]],
            hd={'col1': 0, 'col2': 1, 'col3': 2},
            kd={1: 0, 4: 1},
            dtypes={'col1': int, 'col2': int, 'col3': int},
            keyfield='col1',
            name='TestDaf',
            retmode='val',
            itermode='keyedlist'
        )
        json_str = instance.to_json()
        expected_dict = {
            'name': 'TestDaf',
            'lol': [[1, 2, 3], [4, 5, 6]],
            'hd': {'col1': 0, 'col2': 1, 'col3': 2},
            'kd': {'1': 0, '4': 1},     # Note json dict must use str keys
            'dtypes': {'col1': 'int', 'col2': 'int', 'col3': 'int'},
            'keyfield': 'col1',
            '_retmode': 'val',
            '_itermode': 'keyedlist',
        }
        self.assertEqual(json.loads(json_str), expected_dict)

    def test_from_json(self):
        json_str = json.dumps({
            'name': 'TestDaf',
            'lol': [[1, 2, 3], [4, 5, 6]],
            'hd': {'col1': 0, 'col2': 1, 'col3': 2},
            'kd': {1: 0, 4: 1},     # will create str keys
            'dtypes': {'col1': 'int', 'col2': 'int', 'col3': 'int'},
            'keyfield': 'col1',
            '_retmode': 'val',
            '_itermode': 'keyedlist',
        })
        instance = Daf.from_json(json_str)
        self.assertEqual(instance.name, 'TestDaf')
        self.assertEqual(instance.lol, [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(instance.hd, {'col1': 0, 'col2': 1, 'col3': 2})
        self.assertEqual(instance.kd, {1: 0, 4: 1})     # but conversion must create ints
        self.assertEqual(instance.dtypes, {'col1': int, 'col2': int, 'col3': int})
        self.assertEqual(instance.keyfield, 'col1')
        self.assertEqual(instance._retmode, 'val')
        self.assertEqual(instance._itermode, 'keyedlist')

    def test_to_json_and_from_json(self):
        original_instance = Daf(
            lol=[[1, 2, 3], [4, 5, 6]],
            hd={'col1': 0, 'col2': 1, 'col3': 2},
            kd={'key1': 0},
            dtypes={'col1': int, 'col2': int, 'col3': int},
            keyfield='col1',
            name='TestDaf',
            retmode='val',
            itermode='keyedlist'
        )
        json_str = original_instance.to_json()
        new_instance = Daf.from_json(json_str)
        self.assertEqual(original_instance.name, new_instance.name)
        self.assertEqual(original_instance.lol, new_instance.lol)
        self.assertEqual(original_instance.hd, new_instance.hd)
        self.assertEqual(original_instance.kd, new_instance.kd)
        self.assertEqual(original_instance.dtypes, new_instance.dtypes)
        self.assertEqual(original_instance.keyfield, new_instance.keyfield)
        self.assertEqual(original_instance._retmode, new_instance._retmode)
        self.assertEqual(original_instance._itermode, new_instance._itermode)



class TestAlterDaf(unittest.TestCase):

    def setUp(self):
        # Setup for Daf object
        self.daf_data = [
            ['01780_00000_983814', '01780_00000', 'Database 1 All Tabulators Results.zip'],
            ['01780_00000_996586', '01780_00000', 'Database 1 All Tabulators Results.zip'],
            ['01780_00000_998212', '01780_00000', 'Database 1 All Tabulators Results.zip'],
            ['04000_00001_000001', '04000_00001', 'Database 1 All Tabulators Results.zip'],
            ['04000_00001_000002', '04000_00001', 'Database 1 All Tabulators Results.zip'],
            ['04000_00001_000003', '04000_00001', 'Database 1 All Tabulators Results.zip'],
        ]
        self.daf_cols = ['ballot_id', 'batchid', 'archive_basename']
        self.biabif_daf = Daf(cols=self.daf_cols, lol=self.daf_data)
        
        self.alter_specs_data = [
            {"spec_name": "Database 2 All Tabulators Results.zip", 
             "colname": "ballot_id", 
             "replace_regex": r"/04000_(\d\d\d\d\d_\d\d\d\d\d\d)/14000_\1/"},
            {"spec_name": "CVR_Export_20230206180329 DB2 Certified.csv", 
             "colname": "ballot_id",
             "replace_regex": r"/04000_(\d\d\d\d\d_\d\d\d\d\d\d)/14000_\1/"}
        ]
        self.alter_specs_daf = Daf.from_lod(self.alter_specs_data)
        self.alter_specs_daf = self.alter_specs_daf.select_by_dict({'spec_name': "Database 2 All Tabulators Results.zip"})
        
        self.expected_data = [
            ['01780_00000_983814', '01780_00000', 'Database 1 All Tabulators Results.zip'],
            ['01780_00000_996586', '01780_00000', 'Database 1 All Tabulators Results.zip'],
            ['01780_00000_998212', '01780_00000', 'Database 1 All Tabulators Results.zip'],
            ['14000_00001_000001', '04000_00001', 'Database 1 All Tabulators Results.zip'],
            ['14000_00001_000002', '04000_00001', 'Database 1 All Tabulators Results.zip'],
            ['14000_00001_000003', '04000_00001', 'Database 1 All Tabulators Results.zip'],
        ]
        self.expected_daf = Daf(cols=self.daf_cols, lol=self.expected_data)

    def test_alter_daf_per_alter_specs_daf(self):
        result_daf = self.biabif_daf.alter_daf_per_alter_specs_daf(self.alter_specs_daf)
        
        self.assertEqual(result_daf.lol, self.expected_daf.lol)
        self.assertEqual(result_daf.hd, self.expected_daf.hd)

    def test_empty_alter_specs(self):
        empty_alter_specs_daf = Daf(cols=self.daf_cols, lol=[])
        result_daf = self.biabif_daf.alter_daf_per_alter_specs_daf(empty_alter_specs_daf)
        
        self.assertEqual(result_daf.lol, self.biabif_daf.lol)
        self.assertEqual(result_daf.hd, self.biabif_daf.hd)

    def test_no_matching_colname(self):
        alter_specs_data_no_match = [
            {"spec_name": "Database 2 All Tabulators Results.zip", 
             "colname": "non_existent_col", 
             "replace_regex": r"/04000_(\d\d\d\d\d_\d\d\d\d\d\d)/14000_\1/"}
        ]
        alter_specs_daf_no_match = Daf.from_lod(alter_specs_data_no_match)
        alter_specs_daf_no_match = alter_specs_daf_no_match.select_by_dict({'spec_name': "Database 2 All Tabulators Results.zip"})
        
        result_daf = self.biabif_daf.alter_daf_per_alter_specs_daf(alter_specs_daf_no_match)
        
        self.assertEqual(result_daf.lol, self.biabif_daf.lol)
        self.assertEqual(result_daf.hd, self.biabif_daf.hd)


if __name__ == '__main__':
    unittest.main()
