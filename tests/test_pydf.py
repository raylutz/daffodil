# test_pydf
# copyright (c) 2024 Ray Lutz

import sys
import unittest
#import numpy as np
sys.path.append('..')

from Pydf.Pydf import Pydf

class TestPydf(unittest.TestCase):

    # initialization
    def test_init_default_values(self):
        pydf = Pydf()
        self.assertEqual(pydf.name, '')
        self.assertEqual(pydf.keyfield, '')
        self.assertEqual(pydf.hd, {})
        self.assertEqual(pydf.lol, [])
        self.assertEqual(pydf.kd, {})
        self.assertEqual(pydf.dtypes, {})
        self.assertEqual(pydf._iter_index, 0)

    def test_init_custom_values(self):
        cols = ['col1', 'col2']
        hd = {'col1': 0, 'col2': 1}
        lol = [[1, 2], [3, 4]]
        kd = {1: 0, 3: 1}
        dtypes = {'col1': int, 'col2': str}
        expected_lol = [[1, '2'], [3, '4']]
        pydf = Pydf(cols=cols, lol=lol, dtypes=dtypes, name='TestPydf', keyfield='col1')
        self.assertEqual(pydf.name, 'TestPydf')
        self.assertEqual(pydf.keyfield, 'col1')
        self.assertEqual(pydf.hd, hd)
        self.assertEqual(pydf.lol, expected_lol)
        self.assertEqual(pydf.kd, kd)
        self.assertEqual(pydf.dtypes, dtypes)
        self.assertEqual(pydf._iter_index, 0)

    def test_init_no_cols_but_dtypes(self):
        #cols = ['col1', 'col2']
        hd = {'col1': 0, 'col2': 1}
        lol = [[1, 2], [3, 4]]
        kd = {1: 0, 3: 1}
        dtypes = {'col1': int, 'col2': str}
        expected_lol = [[1, '2'], [3, '4']]
        pydf = Pydf(lol=lol, dtypes=dtypes, name='TestPydf', keyfield='col1')
        self.assertEqual(pydf.name, 'TestPydf')
        self.assertEqual(pydf.keyfield, 'col1')
        self.assertEqual(pydf.hd, hd)
        self.assertEqual(pydf.lol, expected_lol)
        self.assertEqual(pydf.kd, kd)
        self.assertEqual(pydf.dtypes, dtypes)
        self.assertEqual(pydf._iter_index, 0)

    # shape
    def test_shape_empty(self):
        # Test shape method with an empty Pydf object
        pydf = Pydf()
        self.assertEqual(pydf.shape(), (0, 0))

    def test_shape_non_empty(self):
        # Test shape method with a non-empty Pydf object
        data = [[1, 'A'], [2, 'B'], [3, 'C']]
        cols = ['Col1', 'Col2']
        pydf = Pydf(lol=data, cols=cols)
        self.assertEqual(pydf.shape(), (3, 2))

    def test_shape_no_colnames(self):
        # Test shape method with a Pydf object initialized without colnames
        data = [[1, 'A'], [2, 'B'], [3, 'C']]
        pydf = Pydf(lol=data)
        self.assertEqual(pydf.shape(), (3, 2))

    def test_shape_empty_data(self):
        # Test shape method with a Pydf object initialized with empty data
        cols = ['Col1', 'Col2']
        pydf = Pydf(cols=cols)
        self.assertEqual(pydf.shape(), (0, 0))

    def test_shape_empty_data_specified(self):
        # Test shape method with a Pydf object initialized with empty data
        cols = ['Col1', 'Col2']
        pydf = Pydf(lol=[], cols=cols)
        self.assertEqual(pydf.shape(), (0, 0))

    def test_shape_empty_data_specified_empty_col(self):
        # Test shape method with a Pydf object initialized with empty data
        cols = ['Col1', 'Col2']
        pydf = Pydf(lol=[[]], cols=cols)
        self.assertEqual(pydf.shape(), (1, 0))

    def test_shape_no_colnames_no_cols(self):
        # Test shape method with a Pydf object initialized without colnames
        data = [[], [], []]
        pydf = Pydf(lol=data)
        self.assertEqual(pydf.shape(), (3, 0))

    def test_shape_colnames_no_cols_empty_rows(self):
        # Test shape method with a Pydf object initialized without colnames
        data = [[], [], []]
        # cols = ['Col1', 'Col2']
        pydf = Pydf(lol=data)
        self.assertEqual(pydf.shape(), (3, 0))

    # __eq__
    
    def test_eq_different_type(self):
        # Test __eq__ method with a different type
        pydf = Pydf()
        other = "not a Pydf object"
        self.assertFalse(pydf == other)

    def test_eq_different_data(self):
        # Test __eq__ method with a Pydf object with different data
        pydf1 = Pydf(lol=[[1, 'A'], [2, 'B'], [3, 'C']], cols=['Col1', 'Col2'], keyfield='Col1')
        pydf2 = Pydf(lol=[[1, 'X'], [2, 'Y'], [3, 'Z']], cols=['Col1', 'Col2'], keyfield='Col1')
        self.assertFalse(pydf1 == pydf2)

    def test_eq_different_columns(self):
        # Test __eq__ method with a Pydf object with different columns
        pydf1 = Pydf(lol=[[1, 'A'], [2, 'B'], [3, 'C']], cols=['Col1', 'Col2'], keyfield='Col1')
        pydf2 = Pydf(lol=[[1, 'A'], [2, 'B'], [3, 'C']], cols=['Col1', 'Col3'], keyfield='Col1')
        self.assertFalse(pydf1 == pydf2)

    def test_eq_different_keyfield(self):
        # Test __eq__ method with a Pydf object with different keyfield
        pydf1 = Pydf(lol=[[1, 'A'], [2, 'B'], [3, 'C']], cols=['Col1', 'Col2'], keyfield='Col1')
        pydf2 = Pydf(lol=[[1, 'A'], [2, 'B'], [3, 'C']], cols=['Col1', 'Col2'], keyfield='Col2')
        self.assertFalse(pydf1 == pydf2)

    def test_eq_equal(self):
        # Test __eq__ method with equal Pydf objects
        pydf1 = Pydf(lol=[[1, 'A'], [2, 'B'], [3, 'C']], cols=['Col1', 'Col2'], keyfield='Col1')
        pydf2 = Pydf(lol=[[1, 'A'], [2, 'B'], [3, 'C']], cols=['Col1', 'Col2'], keyfield='Col1')
        self.assertTrue(pydf1 == pydf2)

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
        pydf = Pydf(lol=data, cols=columns, dtypes=types)
        included_cols = pydf.calc_cols(include_cols=['Col1', 'Col3'])
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
        pydf = Pydf(lol=data, cols=columns, dtypes=types)
        excluded_cols = pydf.calc_cols(exclude_cols=['Col2'])
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
        pydf = Pydf(lol=data, cols=columns, dtypes=types)
        included_types = pydf.calc_cols(include_types=[int])
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
        pydf = Pydf(lol=data, cols=columns, dtypes=types)
        excluded_types = pydf.calc_cols(exclude_types=[str])
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
        pydf = Pydf(lol=data, cols=columns, dtypes=types)
        selected_cols = pydf.calc_cols(include_cols=['Col1', 'Col2'],
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
        pydf = Pydf(lol=data, cols=columns, dtypes=types)
        
        # Rename columns using the provided dictionary
        from_to_dict = {'Col1': 'NewCol1', 'Col3': 'NewCol3'}
        pydf.rename_cols(from_to_dict)
        
        # Check if columns are renamed correctly
        expected_columns = ['NewCol1', 'Col2', 'NewCol3']
        self.assertEqual(pydf.columns(), expected_columns)

        # Check if dtypes are updated correctly
        expected_types = {'NewCol1': int, 'Col2': str, 'NewCol3': bool}
        self.assertEqual(pydf.dtypes, expected_types)

    def test_rename_cols_with_keyfield(self):
        # Test rename_cols method when a keyfield is specified
        data = [
            [1, 'A', True],
            [2, 'B', False],
            [3, 'C', True]
        ]
        columns = ['Col1', 'Col2', 'Col3']
        types = {'Col1': int, 'Col2': str, 'Col3': bool}
        pydf = Pydf(lol=data, cols=columns, dtypes=types, keyfield='Col1')
        
        # Rename columns using the provided dictionary
        from_to_dict = {'Col1': 'NewCol1', 'Col3': 'NewCol3'}
        pydf.rename_cols(from_to_dict)
        
        # Check if keyfield is updated correctly
        self.assertEqual(pydf.keyfield, 'NewCol1')
        

    # set_cols
    def test_set_cols_no_existing_cols(self):
        # Test setting column names when there are no existing columns
        pydf = Pydf()
        new_cols = ['A', 'B', 'C']
        pydf.set_cols(new_cols)
        self.assertEqual(pydf.hd, {'A': 0, 'B': 1, 'C': 2})
    
    def test_set_cols_generate_spreadsheet_names(self):
        # Test generating spreadsheet-like column names
        pydf = Pydf(cols=['col1', 'col2'])
        pydf.set_cols()
        self.assertEqual(pydf.hd, {'A': 0, 'B': 1})
    
    def test_set_cols_with_existing_cols(self):
        # Test setting column names with existing columns
        pydf = Pydf(cols=['col1', 'col2'])
        new_cols = ['A', 'B']
        pydf.set_cols(new_cols)
        self.assertEqual(pydf.hd, {'A': 0, 'B': 1})
    
    def test_set_cols_repair_keyfield(self):
        # Test repairing keyfield if column names are already defined
        pydf = Pydf(cols=['col1', 'col2'], keyfield='col1')
        new_cols = ['A', 'B']
        pydf.set_cols(new_cols)
        self.assertEqual(pydf.keyfield, 'A')
    
    def test_set_cols_update_dtypes(self):
        # Test updating dtypes dictionary with new column names
        pydf = Pydf(cols=['col1', 'col2'], dtypes={'col1': int, 'col2': str})
        new_cols = ['A', 'B']
        pydf.set_cols(new_cols)
        self.assertEqual(pydf.dtypes, {'A': int, 'B': str})
    
    def test_set_cols_error_length_mismatch(self):
        # Test error handling when lengths of new column names don't match existing ones
        pydf = Pydf(cols=['col1', 'col2'])
        new_cols = ['A']  # Length mismatch
        with self.assertRaises(AttributeError):
            pydf.set_cols(new_cols)


    # keys
    def test_keys_no_keyfield(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='', dtypes={'col1': int, 'col2': str})

        result = pydf.keys()

        self.assertEqual(result, [])

    def test_keys_with_keyfield(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        result = pydf.keys()

        self.assertEqual(result, [1, 2, 3])

    def test_keys_empty_pydf(self):
        cols = []
        lol = []
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={})

        result = pydf.keys()

        self.assertEqual(result, [])  

    # set_keyfield
    def test_set_keyfield_existing_column(self):
        # Test setting keyfield to an existing column
        pydf = Pydf(lol=[[1, 'a'], [2, 'b']], cols=['ID', 'Value'])
        pydf.set_keyfield('ID')
        self.assertEqual(pydf.keyfield, 'ID')
    
    def test_set_keyfield_empty_string(self):
        # Test setting keyfield to an empty string
        pydf = Pydf(lol=[[1, 'a'], [2, 'b']], cols=['ID', 'Value'], keyfield='ID')
        pydf.set_keyfield('')
        self.assertEqual(pydf.keyfield, '')
    
    def test_set_keyfield_nonexistent_column(self):
        # Test trying to set keyfield to a nonexistent column
        pydf = Pydf(lol=[[1, 'a'], [2, 'b']], cols=['ID', 'Value'])
        with self.assertRaises(KeyError):
            pydf.set_keyfield('nonexistent_column')

    # row_idx_of
    def test_row_idx_of_existing_key(self):
        # Test getting row index of an existing key
        pydf = Pydf(lol=[['1', 'a'], ['2', 'b']], cols=['ID', 'Value'], keyfield='ID')
        self.assertEqual(pydf.row_idx_of('1'), 0)
    
    def test_row_idx_of_nonexistent_key(self):
        # Test getting row index of a nonexistent key
        pydf = Pydf(lol=[['1', 'a'], ['2', 'b']], cols=['ID', 'Value'], keyfield='ID')
        self.assertEqual(pydf.row_idx_of('3'), -1)
    
    def test_row_idx_of_no_keyfield(self):
        # Test getting row index when no keyfield is defined
        pydf = Pydf(lol=[['1', 'a'], ['2', 'b']], cols=['ID', 'Value'])
        self.assertEqual(pydf.row_idx_of('1'), -1)
    
    def test_row_idx_of_no_kd(self):
        # Test getting row index when kd is not available
        pydf = Pydf(lol=[['1', 'a'], ['2', 'b']], cols=['ID', 'Value'], keyfield='ID')
        pydf.kd = None
        self.assertEqual(pydf.row_idx_of('1'), -1)



    # from/to cases
    def test_from_lod(self):
        records_lod = [ {'col1': 1, 'col2': 2}, 
                        {'col1': 11, 'col2': 12}, 
                        {'col1': 21, 'col2': 22}]
                        
        keyfield = 'col1'
        dtypes = {'col1': int, 'col2': int}
        pydf = Pydf.from_lod(records_lod, keyfield=keyfield, dtypes=dtypes)

        self.assertEqual(pydf.name, '')
        self.assertEqual(pydf.keyfield, keyfield)
        self.assertEqual(pydf.hd, {'col1': 0, 'col2': 1})
        self.assertEqual(pydf.lol, [[1, 2], [11, 12], [21, 22]])
        self.assertEqual(pydf.kd, {1: 0, 11: 1, 21: 2})
        self.assertEqual(pydf.dtypes, dtypes)
        self.assertEqual(pydf._iter_index, 0)

    def test_from_hllola(self):
        header_list = ['col1', 'col2']
        data_list = [[1, 'a'], [2, 'b'], [3, 'c']]
        hllola = (header_list, data_list)
        keyfield = 'col1'
        dtypes = {'col1': int, 'col2': str}

        pydf = Pydf.from_hllola(hllola, keyfield=keyfield, dtypes=dtypes)

        self.assertEqual(pydf.name, '')
        self.assertEqual(pydf.keyfield, keyfield)
        self.assertEqual(pydf.hd, {'col1': 0, 'col2': 1})
        self.assertEqual(pydf.lol, [[1, 'a'], [2, 'b'], [3, 'c']])
        self.assertEqual(pydf.kd, {1: 0, 2: 1, 3: 2})
        self.assertEqual(pydf.dtypes, dtypes)
        self.assertEqual(pydf._iter_index, 0)

    def test_to_hllola(self):
        cols    = ['col1', 'col2']
        lol     = [[1, 'a'], [2, 'b'], [3, 'c']]
        pydf    = Pydf(cols=cols, lol=lol)

        expected_hllola = (['col1', 'col2'], [[1, 'a'], [2, 'b'], [3, 'c']])
        actual_hllola = pydf.to_hllola()

        self.assertEqual(actual_hllola, expected_hllola)

    # append
    def test_append_without_keyfield(self):
        pydf = Pydf()
        record_da = {'col1': 1, 'col2': 'b'}

        pydf.append(record_da)

        self.assertEqual(pydf.name, '')
        self.assertEqual(pydf.keyfield, '')
        self.assertEqual(pydf.hd, {'col1': 0, 'col2': 1})
        self.assertEqual(pydf.lol, [[1, 'b']])
        self.assertEqual(pydf.kd, {})
        self.assertEqual(pydf.dtypes, {})
        self.assertEqual(pydf._iter_index, 0)

    def test_append_with_keyfield(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b']]
        #kd = {1: 0, 2: 1}
        dtypes = {'col1': int, 'col2': str}
        pydf = Pydf(cols=cols, lol=lol, dtypes=dtypes, keyfield='col1')

        record_da = {'col1': 3, 'col2': 'c'}

        pydf.append(record_da)

        self.assertEqual(pydf.name, '')
        self.assertEqual(pydf.keyfield, 'col1')
        self.assertEqual(pydf.columns(), cols)
        self.assertEqual(pydf.lol, [[1, 'a'], [2, 'b'], [3, 'c']])
        self.assertEqual(pydf.kd, {1: 0, 2: 1, 3: 2})
        self.assertEqual(pydf.dtypes, dtypes)
        self.assertEqual(pydf._iter_index, 0)


    def test_extend_without_keyfield(self):
        pydf = Pydf()
        records_lod = [{'col1': 1, 'col2': 'b'}, {'col1': 2, 'col2': 'c'}]

        pydf.extend(records_lod)

        self.assertEqual(pydf.name, '')
        self.assertEqual(pydf.keyfield, '')
        self.assertEqual(pydf.hd, {'col1': 0, 'col2': 1})
        self.assertEqual(pydf.lol, [[1, 'b'], [2, 'c']])
        self.assertEqual(pydf.kd, {})
        self.assertEqual(pydf.dtypes, {})
        self.assertEqual(pydf._iter_index, 0)

    def test_extend_using_append_without_keyfield(self):
        pydf = Pydf()
        cols = ['col1', 'col2']
        records_lod = [{'col1': 1, 'col2': 'b'}, {'col1': 2, 'col2': 'c'}]

        pydf.append(records_lod)

        self.assertEqual(pydf.name, '')
        self.assertEqual(pydf.keyfield, '')
        self.assertEqual(pydf.columns(), cols)
        self.assertEqual(pydf.lol, [[1, 'b'], [2, 'c']])
        self.assertEqual(pydf.kd, {})
        self.assertEqual(pydf.dtypes, {})
        self.assertEqual(pydf._iter_index, 0)

    def test_extend_with_keyfield(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b']]
        dtypes = {'col1': int, 'col2': str}
        pydf = Pydf(cols=cols, lol=lol, dtypes=dtypes, keyfield='col1')

        records_lod = [{'col1': 3, 'col2': 'c'}, {'col1': 4, 'col2': 'd'}]

        pydf.extend(records_lod)

        self.assertEqual(pydf.name, '')
        self.assertEqual(pydf.keyfield, 'col1')
        self.assertEqual(pydf.columns(), cols)
        self.assertEqual(pydf.lol, [[1, 'a'], [2, 'b'], [3, 'c'], [4, 'd']])
        self.assertEqual(pydf.kd, {1: 0, 2: 1, 3: 2, 4: 3})
        self.assertEqual(pydf.dtypes, dtypes)
        self.assertEqual(pydf._iter_index, 0)
        

    def test_extend_using_append_with_keyfield(self):
        cols = ['col1', 'col2']
        hd = {'col1': 0, 'col2': 1}
        lol = [[1, 'a'], [2, 'b']]
        dtypes = {'col1': int, 'col2': str}
        pydf = Pydf(cols=cols, lol=lol, dtypes=dtypes, keyfield='col1')

        records_lod = [{'col1': 3, 'col2': 'c'}, {'col1': 4, 'col2': 'd'}]

        pydf.append(records_lod)

        self.assertEqual(pydf.name, '')
        self.assertEqual(pydf.keyfield, 'col1')
        self.assertEqual(pydf.hd, hd)
        self.assertEqual(pydf.lol, [[1, 'a'], [2, 'b'], [3, 'c'], [4, 'd']])
        self.assertEqual(pydf.kd, {1: 0, 2: 1, 3: 2, 4: 3})
        self.assertEqual(pydf.dtypes, dtypes)
        self.assertEqual(pydf._iter_index, 0)
        

    def test_concat_without_keyfield(self):
        pydf1 = Pydf()
        pydf2 = Pydf()

        cols = ['col1', 'col2']
        hd = {'col1': 0, 'col2': 1}
        lol1 = [[1, 'a'], [2, 'b']]
        lol2 = [['x', 'y'], ['z', 'w']]
        pydf1 = Pydf(cols=cols, lol=lol1, keyfield='', dtypes={'col1': str, 'col2': str})
        pydf2 = Pydf(cols=cols, lol=lol2, keyfield='', dtypes={'col1': str, 'col2': str})

        pydf1.concat(pydf2)

        self.assertEqual(pydf1.name, '')
        self.assertEqual(pydf1.keyfield, '')
        self.assertEqual(pydf1.hd, hd)
        self.assertEqual(pydf1.lol, [['1', 'a'], ['2', 'b'], ['x', 'y'], ['z', 'w']])
        self.assertEqual(pydf1.kd, {})
        self.assertEqual(pydf1.dtypes, {'col1': str, 'col2': str})
        self.assertEqual(pydf1._iter_index, 0)

    def test_concat_using_append_without_keyfield(self):
        pydf1 = Pydf()
        pydf2 = Pydf()

        cols = ['col1', 'col2']
        hd = {'col1': 0, 'col2': 1}
        lol1 = [[1, 'a'], [2, 'b']]
        lol2 = [['x', 'y'], ['z', 'w']]
        pydf1 = Pydf(cols=cols, lol=lol1, keyfield='', dtypes={'col1': str, 'col2': str})
        pydf2 = Pydf(cols=cols, lol=lol2, keyfield='', dtypes={'col1': str, 'col2': str})

        pydf1.append(pydf2)

        self.assertEqual(pydf1.name, '')
        self.assertEqual(pydf1.keyfield, '')
        self.assertEqual(pydf1.hd, hd)
        self.assertEqual(pydf1.lol, [['1', 'a'], ['2', 'b'], ['x', 'y'], ['z', 'w']])
        self.assertEqual(pydf1.kd, {})
        self.assertEqual(pydf1.dtypes, {'col1': str, 'col2': str})
        self.assertEqual(pydf1._iter_index, 0)

    def test_concat_with_keyfield(self):
        cols = ['col1', 'col2']
        hd = {'col1': 0, 'col2': 1}
        lol1 = [[1, 'a'], [2, 'b']]
        lol2 = [[3, 'c'], [4, 'd']]
        pydf1 = Pydf(cols=cols, lol=lol1, keyfield='col1', dtypes={'col1': int, 'col2': str})
        pydf2 = Pydf(cols=cols, lol=lol2, keyfield='col1', dtypes={'col1': int, 'col2': str})

        pydf1.concat(pydf2)

        self.assertEqual(pydf1.name, '')
        self.assertEqual(pydf1.keyfield, 'col1')
        self.assertEqual(pydf1.hd, hd)
        self.assertEqual(pydf1.lol, [[1, 'a'], [2, 'b'], [3, 'c'], [4, 'd']])
        self.assertEqual(pydf1.kd, {1: 0, 2: 1, 3: 2, 4: 3})
        self.assertEqual(pydf1.dtypes, {'col1': int, 'col2': str})
        self.assertEqual(pydf1._iter_index, 0)

    def test_concat_using_append_with_keyfield(self):
        cols = ['col1', 'col2']
        hd = {'col1': 0, 'col2': 1}
        lol1 = [[1, 'a'], [2, 'b']]
        lol2 = [[3, 'c'], [4, 'd']]
        pydf1 = Pydf(cols=cols, lol=lol1, keyfield='col1', dtypes={'col1': int, 'col2': str})
        pydf2 = Pydf(cols=cols, lol=lol2, keyfield='col1', dtypes={'col1': int, 'col2': str})

        pydf1.append(pydf2)

        self.assertEqual(pydf1.name, '')
        self.assertEqual(pydf1.keyfield, 'col1')
        self.assertEqual(pydf1.hd, hd)
        self.assertEqual(pydf1.lol, [[1, 'a'], [2, 'b'], [3, 'c'], [4, 'd']])
        self.assertEqual(pydf1.kd, {1: 0, 2: 1, 3: 2, 4: 3})
        self.assertEqual(pydf1.dtypes, {'col1': int, 'col2': str})
        self.assertEqual(pydf1._iter_index, 0)
        

    # remove_key
    def test_remove_key_existing_key(self):
        cols = ['col1', 'col2']
        hd = {'col1': 0, 'col2': 1}
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        keyval = 2
        pydf.remove_key(keyval)

        self.assertEqual(pydf.name, '')
        self.assertEqual(pydf.keyfield, 'col1')
        self.assertEqual(pydf.hd, hd)
        self.assertEqual(pydf.lol, [[1, 'a'], [3, 'c']])
        self.assertEqual(pydf.kd, {1: 0, 3: 1})
        self.assertEqual(pydf.dtypes, {'col1': int, 'col2': str})
        self.assertEqual(pydf._iter_index, 0)

    def test_remove_key_nonexistent_key_silent_error(self):
        cols = ['col1', 'col2']
        hd = {'col1': 0, 'col2': 1}
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        keyval = 4
        pydf.remove_key(keyval, silent_error=True)

        self.assertEqual(pydf.name, '')
        self.assertEqual(pydf.keyfield, 'col1')
        self.assertEqual(pydf.hd, hd)
        self.assertEqual(pydf.lol, [[1, 'a'], [2, 'b'], [3, 'c']])
        self.assertEqual(pydf.kd, {1: 0, 2: 1, 3: 2})
        self.assertEqual(pydf.dtypes, {'col1': int, 'col2': str})
        self.assertEqual(pydf._iter_index, 0)

    def test_remove_key_nonexistent_key_raise_error(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        keyval = 4
        with self.assertRaises(KeyError):
            pydf.remove_key(keyval, silent_error=False)


    # remove_keylist
    def test_remove_keylist_existing_keys(self):
        cols = ['col1', 'col2']
        hd = {'col1': 0, 'col2': 1}
        lol = [[1, 'a'], [2, 'b'], [3, 'c'], [4, 'd']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        keylist = [2, 4]
        pydf.remove_keylist(keylist)

        self.assertEqual(pydf.name, '')
        self.assertEqual(pydf.keyfield, 'col1')
        self.assertEqual(pydf.hd, hd)
        self.assertEqual(pydf.lol, [[1, 'a'], [3, 'c']])
        self.assertEqual(pydf.kd, {1: 0, 3: 1})
        self.assertEqual(pydf.dtypes, {'col1': int, 'col2': str})
        self.assertEqual(pydf._iter_index, 0)

    def test_remove_keylist_nonexistent_keys_silent_error(self):
        cols = ['col1', 'col2']
        hd = {'col1': 0, 'col2': 1}
        lol = [[1, 'a'], [2, 'b'], [3, 'c'], [4, 'd']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        keylist = [4, 5, 6]
        pydf.remove_keylist(keylist, silent_error=True)

        self.assertEqual(pydf.name, '')
        self.assertEqual(pydf.keyfield, 'col1')
        self.assertEqual(pydf.hd, hd)
        self.assertEqual(pydf.lol, [[1, 'a'], [2, 'b'], [3, 'c']])
        self.assertEqual(pydf.kd, {1: 0, 2: 1, 3: 2})
        self.assertEqual(pydf.dtypes, {'col1': int, 'col2': str})
        self.assertEqual(pydf._iter_index, 0)

    def test_remove_keylist_nonexistent_keys_raise_error(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c'], [4, 'd']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        keylist = [4, 5, 6]

        with self.assertRaises(KeyError):
            pydf.remove_keylist(keylist, silent_error=False)

    # select_record_da
    def test_select_record_da_existing_key(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        key = 2
        record_da = pydf.select_record_da(key)

        self.assertEqual(record_da, {'col1': 2, 'col2': 'b'})

    def test_select_record_da_nonexistent_key(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        key = 4
        record_da = pydf.select_record_da(key)

        self.assertEqual(record_da, {})

    def test_select_record_da_no_keyfield(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='', dtypes={'col1': int, 'col2': str})

        key = 'col1'

        with self.assertRaises(RuntimeError):
            pydf.select_record_da(key)

    # iloc / irow
    def test_iloc_existing_row_idx(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        row_idx = 1
        record_da = pydf.iloc(row_idx)

        self.assertEqual(record_da, {'col1': 2, 'col2': 'b'})

    def test_iloc_nonexistent_row_idx(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        row_idx = 4
        record_da = pydf.iloc(row_idx)

        self.assertEqual(record_da, {})

    def test_iloc_negative_row_idx(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        row_idx = -1
        record_da = pydf.irow(row_idx)

        self.assertEqual(record_da, {})

    def test_irow_existing_row_idx(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        row_idx = 1
        record_da = pydf.irow(row_idx)

        self.assertEqual(record_da, {'col1': 2, 'col2': 'b'})

    def test_irow_nonexistent_row_idx(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        row_idx = 4
        record_da = pydf.iloc(row_idx)

        self.assertEqual(record_da, {})

    def test_irow_negative_row_idx(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        row_idx = -1
        record_da = pydf.irow(row_idx)

        self.assertEqual(record_da, {})

    # select_by_dict_to_lod
    def test_select_by_dict_to_lod_existing_selector_da(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c'], [4, 'b']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        selector_da = {'col2': 'b'}
        result_lod = pydf.select_by_dict_to_lod(selector_da)

        expected_lod = [{'col1': 2, 'col2': 'b'}, {'col1': 4, 'col2': 'b'}]
        self.assertEqual(result_lod, expected_lod)

    def test_select_by_dict_to_lod_nonexistent_selector_da(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c'], [4, 'b']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        selector_da = {'col2': 'd'}
        result_lod = pydf.select_by_dict_to_lod(selector_da)

        self.assertEqual(result_lod, [])

    def test_select_by_dict_to_lod_with_expectmax(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c'], [4, 'b']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        selector_da = {'col2': 'b'}
        expectmax = 1
        with self.assertRaises(LookupError):  # You should replace this with the actual exception that should be raised
            pydf.select_by_dict_to_lod(selector_da, expectmax=expectmax)

    # select_by_dict
    def test_select_by_dict_existing_selector_da(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c'], [4, 'b']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        selector_da = {'col2': 'b'}
        result_pydf = pydf.select_by_dict(selector_da)

        expected_hd = {'col1': 0, 'col2': 1}
        expected_lol = [[2, 'b'], [4, 'b']]
        expected_kd = {2: 0, 4: 1}
        expected_dtypes = {'col1': int, 'col2': str}

        self.assertEqual(result_pydf.name, '')
        self.assertEqual(result_pydf.keyfield, 'col1')
        self.assertEqual(result_pydf.hd, expected_hd)
        self.assertEqual(result_pydf.lol, expected_lol)
        self.assertEqual(result_pydf.kd, expected_kd)
        self.assertEqual(result_pydf.dtypes, expected_dtypes)
        self.assertEqual(result_pydf._iter_index, 0)

    def test_select_by_dict_nonexistent_selector_da(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c'], [4, 'b']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        selector_da = {'col2': 'd'}
        result_pydf = pydf.select_by_dict(selector_da)

        expected_hd = {'col1': 0, 'col2': 1}
        expected_lol = []
        expected_kd = {}
        expected_dtypes = {'col1': int, 'col2': str}

        self.assertEqual(result_pydf.name, '')
        self.assertEqual(result_pydf.keyfield, 'col1')
        self.assertEqual(result_pydf.hd, expected_hd)
        self.assertEqual(result_pydf.lol, expected_lol)
        self.assertEqual(result_pydf.kd, expected_kd)
        self.assertEqual(result_pydf.dtypes, expected_dtypes)
        self.assertEqual(result_pydf._iter_index, 0)

    def test_select_by_dict_with_expectmax(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c'], [4, 'b']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        selector_da = {'col2': 'b'}
        expectmax = 1
        with self.assertRaises(LookupError):  # You should replace this with the actual exception that should be raised
            pydf.select_by_dict(selector_da, expectmax=expectmax)

    # col / col_to_la
    def test_col_existing_colname(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        colname = 'col2'
        result_la = pydf.col(colname)

        expected_la = ['a', 'b', 'c']
        self.assertEqual(result_la, expected_la)

    def test_col_nonexistent_colname(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        colname = 'col3'
        with self.assertRaises(RuntimeError):
            result_la = pydf.col(colname)
            result_la = result_la # fool linter

        #self.assertEqual(result_la, [])

    def test_col_empty_colname(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        colname = ''
        with self.assertRaises(RuntimeError):
            result_la = pydf.col(colname)
            result_la = result_la # fool linter

        #self.assertEqual(result_la, [])

    def test_col_nonexistent_colname_silent(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        colname = 'col3'
        result_la = pydf.col(colname, silent_error=True)

        self.assertEqual(result_la, [])

    # drop_cols
    def test_drop_cols_existing_cols(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 'a', 'x'], [2, 'b', 'y'], [3, 'c', 'z']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': str})

        cols_to_drop = ['col2', 'col3']
        pydf.drop_cols(cols_to_drop)

        expected_hd = {'col1': 0}
        expected_lol = [[1], [2], [3]]

        self.assertEqual(pydf.hd, expected_hd)
        self.assertEqual(pydf.lol, expected_lol)

    def test_drop_cols_nonexistent_cols(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 'a', 'x'], [2, 'b', 'y'], [3, 'c', 'z']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': str})

        cols_to_drop = ['col4', 'col5']
        pydf.drop_cols(cols_to_drop)

        expected_hd = {'col1': 0, 'col2': 1, 'col3': 2}
        expected_lol = [[1, 'a', 'x'], [2, 'b', 'y'], [3, 'c', 'z']]

        self.assertEqual(pydf.hd, expected_hd)
        self.assertEqual(pydf.lol, expected_lol)

    def test_drop_cols_empty_cols(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 'a', 'x'], [2, 'b', 'y'], [3, 'c', 'z']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': str})

        cols_to_drop = []
        pydf.drop_cols(cols_to_drop)

        expected_hd = {'col1': 0, 'col2': 1, 'col3': 2}
        expected_lol = [[1, 'a', 'x'], [2, 'b', 'y'], [3, 'c', 'z']]

        self.assertEqual(pydf.hd, expected_hd)
        self.assertEqual(pydf.lol, expected_lol)

    # assign_col
    def test_assign_col_existing_colname(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 'a', 'x'], [2, 'b', 'y'], [3, 'c', 'z']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': str})

        colname = 'col2'
        new_values = ['A', 'B', 'C']
        pydf.assign_col(colname, new_values)

        expected_hd = {'col1': 0, 'col2': 1, 'col3': 2}
        expected_lol = [[1, 'A', 'x'], [2, 'B', 'y'], [3, 'C', 'z']]

        self.assertEqual(pydf.hd, expected_hd)
        self.assertEqual(pydf.lol, expected_lol)

    def test_assign_col_nonexistent_colname(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 'a', 'x'], [2, 'b', 'y'], [3, 'c', 'z']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': str})

        colname = 'col4'
        new_values = ['A', 'B', 'C']
        pydf.assign_col(colname, new_values)

        expected_hd = {'col1': 0, 'col2': 1, 'col3': 2}
        expected_lol = [[1, 'a', 'x'], [2, 'b', 'y'], [3, 'c', 'z']]

        self.assertEqual(pydf.hd, expected_hd)
        self.assertEqual(pydf.lol, expected_lol)

    def test_assign_col_empty_colname(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 'a', 'x'], [2, 'b', 'y'], [3, 'c', 'z']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': str})

        colname = ''
        new_values = ['A', 'B', 'C']
        pydf.assign_col(colname, new_values)

        expected_hd = {'col1': 0, 'col2': 1, 'col3': 2}
        expected_lol = [[1, 'a', 'x'], [2, 'b', 'y'], [3, 'c', 'z']]

        self.assertEqual(pydf.hd, expected_hd)
        self.assertEqual(pydf.lol, expected_lol)

    # cols_to_dol
    def test_cols_to_dol_valid_cols(self):
        cols = ['col1', 'col2', 'col3']
        lol = [['a', 'b', 'c'], ['b', 'd', 'e'], ['a', 'f', 'g'], ['b', 'd', 'm']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        colname1 = 'col1'
        colname2 = 'col2'
        result_dola = pydf.cols_to_dol(colname1, colname2)

        expected_dola = {'a': ['b', 'f'], 'b': ['d']}
        self.assertEqual(result_dola, expected_dola)

    def test_cols_to_dol_invalid_colname(self):
        cols = ['col1', 'col2', 'col3']
        lol = [['a', 'b', 'c'], ['b', 'd', 'e'], ['a', 'f', 'g'], ['b', 'd', 'm']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        colname1 = 'col1'
        colname2 = 'col4'
        result_dola = pydf.cols_to_dol(colname1, colname2)

        self.assertEqual(result_dola, {})

    def test_cols_to_dol_empty_pydf(self):
        cols = []
        lol = []
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={})

        colname1 = 'col1'
        colname2 = 'col2'
        result_dola = pydf.cols_to_dol(colname1, colname2)

        self.assertEqual(result_dola, {})

    def test_cols_to_dol_single_column(self):
        cols = ['col1']
        lol = [['a'], ['b'], ['a'], ['b']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str})

        colname1 = 'col1'
        colname2 = 'col2'
        result_dola = pydf.cols_to_dol(colname1, colname2)

        self.assertEqual(result_dola, {})

    # bool
    def test_bool_empty_pydf(self):
        cols = []
        lol = []
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={})

        result = bool(pydf)

        self.assertFalse(result)

    def test_bool_nonempty_pydf(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        result = bool(pydf)

        self.assertTrue(result)

    def test_bool_pydf_with_empty_lol(self):
        cols = ['col1', 'col2']
        lol = []
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        result = bool(pydf)

        self.assertFalse(result)

    # len
    def test_len_empty_pydf(self):
        cols = []
        lol = []
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={})

        result = len(pydf)

        self.assertEqual(result, 0)

    def test_len_nonempty_pydf(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        result = len(pydf)

        self.assertEqual(result, 3)

    def test_len_pydf_with_empty_lol(self):
        cols = ['col1', 'col2']
        lol = []
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        result = len(pydf)

        self.assertEqual(result, 0)
        
    # columns
    def test_columns_empty_pydf(self):
        cols = []
        lol = []
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={})

        result = pydf.columns()

        self.assertEqual(result, [])

    def test_columns_nonempty_pydf(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 'a', 'x'], [2, 'b', 'y'], [3, 'c', 'z']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': str})

        result = pydf.columns()

        self.assertEqual(result, ['col1', 'col2', 'col3'])
        
    # clone_empty
    
    def test_clone_empty_from_empty_instance(self):
        old_instance = Pydf()
        result = Pydf.clone_empty(old_instance)

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
        old_instance = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})
        result = Pydf.clone_empty(old_instance)

        self.assertEqual(result.name, old_instance.name)
        self.assertEqual(result.keyfield, old_instance.keyfield)
        self.assertEqual(result.hd, old_instance.hd)
        self.assertEqual(result.lol, [])
        self.assertEqual(result.kd, {})
        self.assertEqual(result.dtypes, old_instance.dtypes)
        self.assertEqual(result._iter_index, 0)

    def test_clone_empty_from_none(self):
        old_instance = None
        result = Pydf.clone_empty(old_instance)

        self.assertEqual(result.name, '')
        self.assertEqual(result.keyfield, '')
        self.assertEqual(result.hd, {})
        self.assertEqual(result.lol, [])
        self.assertEqual(result.kd, {})
        self.assertEqual(result.dtypes, {})
        self.assertEqual(result._iter_index, 0)

    # to_lod
    def test_to_lod_empty_pydf(self):
        cols = []
        lol = []
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={})

        result = pydf.to_lod()

        self.assertEqual(result, [])

    def test_to_lod_nonempty_pydf(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        result = pydf.to_lod()

        expected_lod = [{'col1': 1, 'col2': 'a'}, {'col1': 2, 'col2': 'b'}, {'col1': 3, 'col2': 'c'}]
        self.assertEqual(result, expected_lod)

    # select_records
    def test_select_records_empty_pydf(self):
        cols = []
        lol = []
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={})

        keys_ls = [1, 2, 3]
        result = pydf.select_records_pydf(keys_ls)

        self.assertEqual(result.name,   '')
        self.assertEqual(result.keyfield, 'col1')
        self.assertEqual(result.hd,     {})
        self.assertEqual(result.lol,    [])
        self.assertEqual(result.kd,     {})
        self.assertEqual(result.dtypes,  {})

    def test_select_records_nonempty_pydf(self):
        cols    = ['col1', 'col2']
        lol = [ [1, 'a'], 
                [2, 'b'], 
                [3, 'c']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        keys_ls = [2, 1]
        result = pydf.select_records_pydf(keys_ls)

        expected_lol = [[2, 'b'], [1, 'a']]
        self.assertEqual(result.name, pydf.name)
        self.assertEqual(result.keyfield, pydf.keyfield)
        self.assertEqual(result.hd, pydf.hd)
        self.assertEqual(result.lol, expected_lol)
        self.assertEqual(result.dtypes, pydf.dtypes)

    def test_select_records_empty_keys(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        dtypes={'col1': int, 'col2': str}
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes=dtypes)

        keys_ls = []
        result = pydf.select_records_pydf(keys_ls)

        self.assertEqual(result.name, '')
        self.assertEqual(result.keyfield, 'col1')
        self.assertEqual(result.hd, {'col1': 0, 'col2': 1})
        self.assertEqual(result.lol, [])
        self.assertEqual(result.kd, {})
        self.assertEqual(result.dtypes, dtypes)

    # assign_record
    def test_assign_record_empty_pydf(self):
        cols = []
        lol = []
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={})

        record_da = {'col1': 1, 'col2': 'a'}
        pydf.assign_record_da(record_da)

        expected_lol = [[1, 'a']]
        self.assertEqual(pydf.lol, expected_lol)

    def test_assign_record_nonempty_pydf_add_new_record(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        record_da = {'col1': 4, 'col2': 'd'}
        pydf.assign_record_da(record_da)

        expected_lol = [[1, 'a'], [2, 'b'], [3, 'c'], [4, 'd']]
        self.assertEqual(pydf.lol, expected_lol)

    def test_assign_record_nonempty_pydf_update_existing_record(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        record_da = {'col1': 2, 'col2': 'x'}
        pydf.assign_record_da(record_da)

        expected_lol = [[1, 'a'], [2, 'x'], [3, 'c']]
        self.assertEqual(pydf.lol, expected_lol)

    def test_assign_record_missing_keyfield(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        record_da = {'col2': 'x'}
        with self.assertRaises(RuntimeError):
            pydf.assign_record_da(record_da)

    def test_assign_record_fields_not_equal_to_columns(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        record_da = {'col1': 4, 'col2': 'd', 'col3': 'extra'}
        with self.assertRaises(RuntimeError):
            pydf.assign_record_da(record_da)

    # assign_record_irow
    def test_assign_record_irow_empty_pydf(self):
        cols = []
        lol = []
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={})

        record_da = {'col1': 1, 'col2': 'a'}
        pydf.assign_record_da_irow(irow=0, record_da=record_da)

        expected_lol = [[1, 'a']]
        self.assertEqual(pydf.lol, expected_lol)

    def test_assign_record_irow_nonempty_pydf_add_new_record(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        record_da = {'col1': 4, 'col2': 'd'}
        pydf.assign_record_da_irow(irow=3, record_da=record_da)

        expected_lol = [[1, 'a'], [2, 'b'], [3, 'c'], [4, 'd']]
        self.assertEqual(pydf.lol, expected_lol)

    def test_assign_record_irow_nonempty_pydf_update_existing_record(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        record_da = {'col1': 2, 'col2': 'x'}
        pydf.assign_record_da_irow(irow=1, record_da=record_da)

        expected_lol = [[1, 'a'], [2, 'x'], [3, 'c']]
        self.assertEqual(pydf.lol, expected_lol)

    def test_assign_record_irow_invalid_irow(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], 
               [2, 'b'], 
               [3, 'c']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        record_da = {'col1': 4, 'col2': 'd'}
        
        pydf.assign_record_da_irow(irow=5, record_da=record_da)

        expected_lol = [[1, 'a'], 
                        [2, 'b'], 
                        [3, 'c'],
                        [4, 'd'],
                        ]
        self.assertEqual(pydf.lol, expected_lol)

    def test_assign_record_irow_missing_record_da(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        pydf.assign_record_da_irow(irow=1, record_da=None)

        expected_lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        self.assertEqual(pydf.lol, expected_lol)

   # update_record_irow
    def test_update_record_irow_empty_pydf(self):
        cols = []
        lol = []
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={})

        record_da = {'col1': 1, 'col2': 'a'}
        pydf.update_record_da_irow(irow=0, record_da=record_da)

        self.assertEqual(pydf.lol, [])

    def test_update_record_irow_nonempty_pydf_update_existing_record(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        record_da = {'col1': 2, 'col2': 'x', 'col3': 'extra'}
        pydf.update_record_da_irow(irow=1, record_da=record_da)

        expected_lol = [[1, 'a'], [2, 'x'], [3, 'c']]
        self.assertEqual(pydf.lol, expected_lol)

    def test_update_record_irow_invalid_irow(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        record_da = {'col1': 4, 'col2': 'd'}
        pydf.update_record_da_irow(irow=5, record_da=record_da)

        self.assertEqual(pydf.lol, lol)

    def test_update_record_irow_missing_record_da(self):
        cols = ['col1', 'col2']
        lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        pydf.update_record_da_irow(irow=1, record_da=None)

        expected_lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        self.assertEqual(pydf.lol, expected_lol)

    # def test_update_record_irow_missing_hd(self):
        # cols = ['col1', 'col2']
        # hd = {'col1': 0, 'col2': 1}
        # lol = [[1, 'a'], [2, 'b'], [3, 'c']]
        # pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str})

        # record_da = {'col1': 2, 'col2': 'x'}
        # pydf.update_record_da_irow(irow=1, record_da=record_da)

        # self.assertEqual(pydf.lol, lol)

    # icol_to_la
    def test_icol_to_la_valid_icol(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 'a', True], [2, 'b', False], [3, 'c', True]]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': bool})

        result_la = pydf.icol_to_la(1)

        expected_la = ['a', 'b', 'c']
        self.assertEqual(result_la, expected_la)

    def test_icol_to_la_invalid_icol(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 'a', True], [2, 'b', False], [3, 'c', True]]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': bool})

        result_la = pydf.icol_to_la(3)

        self.assertEqual(result_la, [])

    def test_icol_to_la_empty_pydf(self):
        pydf = Pydf()

        result_la = pydf.icol_to_la(0)

        self.assertEqual(result_la, [])

    def test_icol_to_la_empty_column(self):
        cols = ['col1', 'col2', 'col3']
        lol = []
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': bool})

        result_la = pydf.icol_to_la(0)

        self.assertEqual(result_la, [])

    # assign_icol
    def test_assign_icol_valid_icol_col_la(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 'a', True], [2, 'b', False], [3, 'c', True]]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': bool})

        col_la = [4, 'd', False]
        pydf.assign_icol(icol=1, col_la=col_la)

        expected_lol = [[1, 4, True], [2, 'd', False], [3, False, True]]
        self.assertEqual(pydf.lol, expected_lol)

    def test_assign_icol_valid_icol_default(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 'a', True], 
               [2, 'b', False], 
               [3, 'c', True]]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': bool})

        pydf.assign_icol(icol=1, default='x')

        expected_lol = [[1, 'x', True], 
                        [2, 'x', False], 
                        [3, 'x', True]]
        self.assertEqual(pydf.lol, expected_lol)

    def test_assign_icol_valid_append_icol_col_la(self):
        cols = ['col1', 'col2', 'col3']
        lol = [ [1, 'a', True], 
                [2, 'b', False], 
                [3, 'c', True]]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': bool})

        col_la = [4, 'd', False]
        pydf.assign_icol(icol=-1, col_la=col_la)

        expected_lol = [[1, 'a', True, 4], 
                        [2, 'b', False, 'd'], 
                        [3, 'c', True, False]]
        self.assertEqual(pydf.lol, expected_lol)

    # def test_assign_icol_invalid_icol_col_la(self):
        # cols = ['col1', 'col2', 'col3']
        # hd = {'col1': 0, 'col2': 1, 'col3': 2}
        # lol = [[1, 'a', True], [2, 'b', False], [3, 'c', True]]
        # pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': bool})

        # col_la = [4, 'd', False]
        # pydf.assign_icol(icol=3, col_la=col_la)

        # self.assertEqual(pydf.lol, lol)

    def test_assign_icol_empty_pydf(self):
        pydf = Pydf()

        col_la = [4, 'd', False]
        pydf.assign_icol(icol=1, col_la=col_la)

        self.assertEqual(pydf.lol, [])

    # insert_icol
    def test_insert_icol_valid_icol_col_la(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 'a', True], [2, 'b', False], [3, 'c', True]]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': bool})

        col_la = [4, 'd', False]
        pydf.insert_icol(icol=1, col_la=col_la)

        expected_lol = [[1, 4, 'a', True], [2, 'd', 'b', False], [3, False, 'c', True]]
        self.assertEqual(pydf.lol, expected_lol)

    def test_insert_icol_valid_append_col_la(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 'a', True], [2, 'b', False], [3, 'c', True]]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': bool})

        col_la = [4, 'd', False]
        pydf.insert_icol(icol=-1, col_la=col_la)

        expected_lol = [[1, 'a', True, 4], [2, 'b', False, 'd'], [3, 'c', True, False]]
        self.assertEqual(pydf.lol, expected_lol)

    def test_insert_icol_invalid_icol_col_la(self):
        cols = ['col1', 'col2', 'col3']
        lol = [ [1, 'a', True], 
                [2, 'b', False], 
                [3, 'c', True]]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': bool})

        col_la = [4, 'd', False]
        pydf.insert_icol(icol=3, col_la=col_la)

        result_lol = [ [1, 'a', True, 4], 
                [2, 'b', False, 'd'], 
                [3, 'c', True,  False]]

        self.assertEqual(pydf.lol, result_lol)

    def test_insert_icol_empty_pydf(self):
        pydf = Pydf()

        col_la = [4, 'd', False]
        pydf.insert_icol(icol=1, col_la=col_la)

        self.assertEqual(pydf.lol, [])

    # insert_col
    def test_insert_col_valid_colname_col_la(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 'a', True], [2, 'b', False], [3, 'c', True]]
        pydf = Pydf(cols=cols,  lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': bool})

        colname = 'new_col'
        col_la = [4, 'd', False]
        pydf.insert_col(colname=colname, col_la=col_la, icol=1)

        expected_lol = [[1, 4, 'a', True], [2, 'd', 'b', False], [3, False, 'c', True]]
        expected_hd = {'col1': 0, 'new_col': 1, 'col2': 2, 'col3': 3}
        self.assertEqual(pydf.lol, expected_lol)
        self.assertEqual(pydf.hd, expected_hd)

    def test_insert_col_valid_colname_append_col_la(self):
        cols = ['col1', 'col2', 'col3']
        lol = [ [1, 'a', True], 
                [2, 'b', False], 
                [3, 'c', True]]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': bool})

        colname = 'new_col'
        col_la = [4, 'd', False]
        pydf.insert_col(colname=colname, col_la=col_la, icol=-1)

        expected_lol = [[1, 'a', True,  4], 
                        [2, 'b', False, 'd'], 
                        [3, 'c', True,  False]]
        expected_hd = {'col1': 0, 'col2': 1, 'col3': 2, 'new_col': 3}
        self.assertEqual(pydf.lol, expected_lol)
        self.assertEqual(pydf.hd, expected_hd)

    def test_insert_col_valid_colname_invalid_icol_col_la(self):
        cols = ['col1', 'col2', 'col3']
        lol = [ [1, 'a', True], 
                [2, 'b', False], 
                [3, 'c', True]]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': bool})

        colname = 'new_col'
        col_la = [4, 'd', False]
        pydf.insert_col(colname=colname, col_la=col_la, icol=3)

        expected_hd = {'col1': 0, 'col2': 1, 'col3': 2, 'new_col': 3}
        expected_lol = [ [1, 'a', True,     4], 
                         [2, 'b', False,    'd'], 
                         [3, 'c', True,     False]]

        self.assertEqual(pydf.lol, expected_lol)
        self.assertEqual(pydf.hd, expected_hd)

    def test_insert_col_valid_colname_empty_pydf(self):
        pydf = Pydf()

        colname = 'new_col'
        col_la = [4, 'd', False]
        pydf.insert_col(colname=colname, col_la=col_la, icol=1)

        self.assertEqual(pydf.lol, [])
        self.assertEqual(pydf.hd, {'new_col': 0})

    def test_insert_col_empty_colname(self):
        cols = ['col1', 'col2', 'col3']
        hd = {'col1': 0, 'col2': 1, 'col3': 2}
        lol = [[1, 'a', True], [2, 'b', False], [3, 'c', True]]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': bool})

        pydf.insert_col(colname='', col_la=[4, 'd', False], icol=1)

        self.assertEqual(pydf.lol, lol)
        self.assertEqual(pydf.hd, hd)

    # insert_idx_col
    def test_insert_idx_col_valid_icol_startat(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 'a', True], [2, 'b', False], [3, 'c', True]]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': bool})

        colname = 'idx'
        pydf.insert_idx_col(colname=colname, icol=1, startat=10)

        expected_lol = [[1, 10, 'a', True], [2, 11, 'b', False], [3, 12, 'c', True]]
        expected_hd = {'col1': 0, 'idx': 1, 'col2': 2, 'col3': 3}
        self.assertEqual(pydf.lol, expected_lol)
        self.assertEqual(pydf.hd, expected_hd)

    def test_insert_idx_col_valid_icol_default_startat(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 'a', True], [2, 'b', False], [3, 'c', True]]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': bool})

        colname = 'idx'
        pydf.insert_idx_col(colname=colname, icol=1)

        expected_lol = [[1, 0, 'a', True], [2, 1, 'b', False], [3, 2, 'c', True]]
        expected_hd = {'col1': 0, 'idx': 1, 'col2': 2, 'col3': 3}
        self.assertEqual(pydf.lol, expected_lol)
        self.assertEqual(pydf.hd, expected_hd)

    def test_insert_idx_col_valid_append_default_startat(self):
        pydf = Pydf()

        colname = 'idx'
        pydf.insert_idx_col(colname=colname)

        expected_lol = []
        expected_hd = {'idx': 0}
        self.assertEqual(pydf.lol, expected_lol)
        self.assertEqual(pydf.hd, expected_hd)

    def test_insert_idx_col_empty_colname(self):
        cols = ['col1', 'col2', 'col3']
        hd = {'col1': 0, 'col2': 1, 'col3': 2}
        lol = [[1, 'a', True], [2, 'b', False], [3, 'c', True]]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': str, 'col3': bool})

        pydf.insert_idx_col(colname='', icol=1, startat=10)

        self.assertEqual(pydf.lol, lol)
        self.assertEqual(pydf.hd, hd)


    # unified sum
    def test_sum_all_columns(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': int, 'col3': int})

        result_sum = pydf.sum()
        expected_sum = {'col1': 12, 'col2': 15, 'col3': 18}
        self.assertEqual(result_sum, expected_sum)

    def test_sum_selected_columns(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': int, 'col3': int})

        result_sum = pydf.sum(colnames_ls=['col1', 'col3'])
        expected_sum = {'col1': 12, 'col3': 18}
        self.assertEqual(result_sum, expected_sum)

    def test_sum_numeric_only(self):
        cols = ['col1', 'col2', 'col3']
        lol = [['a', 2, 3], ['b', 5, 6], ['c', 8, 9]]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': int, 'col3': int})

        result_sum = pydf.sum(numeric_only=True)
        expected_sum = {'col1': '0.0', 'col2': 15, 'col3': 18}
        self.assertEqual(result_sum, expected_sum)

    def test_sum_empty_pydf(self):
        pydf = Pydf()

        result_sum = pydf.sum()
        expected_sum = {}
        self.assertEqual(result_sum, expected_sum)

    # unified sum_np
    def test_sum_np_all_columns(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': int, 'col3': int})

        result_sum = pydf.sum_np()
        expected_sum = {'col1': 12, 'col2': 15, 'col3': 18}
        self.assertEqual(result_sum, expected_sum)

    def test_sum_np_selected_columns(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': int, 'col3': int})

        result_sum = pydf.sum_np(colnames_ls=['col1', 'col3'])
        expected_sum = {'col1': 12, 'col3': 18}
        self.assertEqual(result_sum, expected_sum)

    def test_sum_np_empty_pydf(self):
        pydf = Pydf()

        result_sum = pydf.sum_np()
        expected_sum = {}
        self.assertEqual(result_sum, expected_sum)


    # pydf_sum
    def test_pydf_sum_all_columns(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': int, 'col3': int})

        result_sum = pydf.pydf_sum()
        expected_sum = {'col1': 12, 'col2': 15, 'col3': 18}
        self.assertEqual(result_sum, expected_sum)

    def test_pydf_sum_selected_columns(self):
        cols = ['col1', 'col2', 'col3']
        lol = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': int, 'col2': int, 'col3': int})

        result_sum = pydf.pydf_sum(cols=['col1', 'col3'])
        expected_sum = {'col1': 12, 'col3': 18}
        self.assertEqual(result_sum, expected_sum)

    def test_pydf_sum_include_types_int(self):
        cols = ['col1', 'col2', 'col3']
        dtypes_dict = {'col1': str, 'col2': int, 'col3': int}
        lol = [['a', 2, 3], ['b', 5, 6], ['c', 8, 9]]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes=dtypes_dict)

        reduce_cols = pydf.calc_cols(include_types=int)
        result_sum = pydf.pydf_sum(cols=reduce_cols)
        expected_sum = {'col2': 15, 'col3': 18}
        self.assertEqual(result_sum, expected_sum)

    def test_pydf_sum_include_types_int_and_float(self):
        cols = ['col1', 'col2', 'col3']
        dtypes_dict = {'col1': str, 'col2': int, 'col3': float}
        lol = [['a', 2, 3.2], ['b', 5, 6.1], ['c', 8, 9.4]]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes=dtypes_dict)

        reduce_cols = pydf.calc_cols(include_types=[int, float])
        result_sum = pydf.pydf_sum(cols=reduce_cols)
        expected_sum = {'col1': '', 'col2': 15, 'col3': 18.7}
        self.assertAlmostEqual(result_sum['col2'], expected_sum['col2'], places=2)
        self.assertAlmostEqual(result_sum['col3'], expected_sum['col3'], places=2)
        #self.assertEqual(result_sum, expected_sum)

    def test_pydf_sum_exclude_type_str(self):
        cols = ['col1', 'col2', 'col3']
        dtypes_dict = {'col1': str, 'col2': int, 'col3': int}
        lol = [['a', 2, 3], ['b', 5, 6], ['c', 8, 9]]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes=dtypes_dict)

        reduce_cols = pydf.calc_cols(exclude_types=[str, bool, list])
        result_sum = pydf.pydf_sum(cols=reduce_cols)
        expected_sum = {'col2': 15, 'col3': 18}
        self.assertEqual(result_sum, expected_sum)

    def test_pydf_sum_empty_pydf(self):
        pydf = Pydf()

        result_sum = pydf.pydf_sum()
        expected_sum = {}
        self.assertEqual(result_sum, expected_sum)

    # valuecounts_for_colname
    def test_valuecounts_for_colname(self):
        cols = ['col1', 'col2', 'col3']
        lol = [['a', 'x', 'y'], ['b', 'x', 'z'], ['a', 'y', 'y'], ['c', 'z', 'z']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        result_valuecounts = pydf.valuecounts_for_colname('col2')
        expected_valuecounts = {'x': 2, 'y': 1, 'z': 1}
        self.assertEqual(result_valuecounts, expected_valuecounts)

    def test_valuecounts_for_colname_sort(self):
        cols = ['col1', 'col2', 'col3']
        lol = [['a', 'x', 'y'], ['b', 'x', 'z'], ['a', 'y', 'y'], ['c', 'z', 'z']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        result_valuecounts = pydf.valuecounts_for_colname('col2', sort=True)
        expected_valuecounts = {'x': 2, 'z': 1, 'y': 1}
        self.assertEqual(result_valuecounts, expected_valuecounts)

    def test_valuecounts_for_colname_reverse_sort(self):
        cols = ['col1', 'col2', 'col3']
        lol = [['a', 'x', 'y'], ['b', 'x', 'z'], ['a', 'y', 'y'], ['c', 'z', 'z']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        result_valuecounts = pydf.valuecounts_for_colname('col2', sort=True, reverse=True)
        expected_valuecounts = {'x': 2, 'y': 1, 'z': 1}
        self.assertEqual(result_valuecounts, expected_valuecounts)

    def test_valuecounts_for_colname_empty_pydf(self):
        pydf = Pydf()

        result_valuecounts = pydf.valuecounts_for_colname('col2')
        expected_valuecounts = {}
        self.assertEqual(result_valuecounts, expected_valuecounts)

    # valuecounts_for_colnames_ls
    def test_valuecounts_for_colnames_ls(self):
        cols = ['col1', 'col2', 'col3']
        lol = [['a', 'x', 'y'], ['b', 'x', 'z'], ['a', 'y', 'y'], ['c', 'z', 'z']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        result_valuecounts = pydf.valuecounts_for_colnames_ls(['col2', 'col3'])
        expected_valuecounts = {'col2': {'x': 2, 'y': 1, 'z': 1}, 'col3': {'y': 2, 'z': 2}}
        self.assertEqual(result_valuecounts, expected_valuecounts)

    def test_valuecounts_for_colnames_ls_sort(self):
        cols = ['col1', 'col2', 'col3']
        lol = [['a', 'x', 'y'], ['b', 'x', 'z'], ['a', 'y', 'y'], ['c', 'z', 'z']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        result_valuecounts = pydf.valuecounts_for_colnames_ls(['col2', 'col3'], sort=True)
        expected_valuecounts = {'col2': {'x': 2, 'z': 1, 'y': 1}, 'col3': {'y': 2, 'z': 2}}
        self.assertEqual(result_valuecounts, expected_valuecounts)

    def test_valuecounts_for_colnames_ls_reverse_sort(self):
        cols = ['col1', 'col2', 'col3']
        lol = [['a', 'x', 'y'], ['b', 'x', 'z'], ['a', 'y', 'y'], ['c', 'z', 'z']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        result_valuecounts = pydf.valuecounts_for_colnames_ls(['col2', 'col3'], sort=True, reverse=True)
        expected_valuecounts = {'col2': {'x': 2, 'y': 1, 'z': 1}, 'col3': {'z': 2, 'y': 2}}
        self.assertEqual(result_valuecounts, expected_valuecounts)

    def test_valuecounts_for_colnames_ls_empty_pydf(self):
        pydf = Pydf()

        result_valuecounts = pydf.valuecounts_for_colnames_ls(['col2', 'col3'])
        expected_valuecounts = {'col2': {}, 'col3': {}}
        self.assertEqual(result_valuecounts, expected_valuecounts)

    def test_valuecounts_for_colnames_ls_all_columns(self):
        cols = ['col1', 'col2', 'col3']
        lol = [['a', 'x', 'y'], ['b', 'x', 'z'], ['a', 'y', 'y'], ['c', 'z', 'z']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        result_valuecounts = pydf.valuecounts_for_colnames_ls()
        expected_valuecounts = {'col1': {'a': 2, 'b': 1, 'c': 1},
                                'col2': {'x': 2, 'y': 1, 'z': 1},
                                'col3': {'y': 2, 'z': 2}}
        self.assertEqual(result_valuecounts, expected_valuecounts)

    # valuecounts_for_colname_selectedby_colname
    def test_valuecounts_for_colname_selectedby_colname(self):
        cols = ['col1', 'col2', 'col3']
        lol = [['a', 'x', 'y'], ['b', 'x', 'z'], ['a', 'y', 'y'], ['c', 'z', 'z']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        result_valuecounts = pydf.valuecounts_for_colname_selectedby_colname('col2', 'col1', 'a')
        expected_valuecounts = {'x': 1, 'y': 1}
        self.assertEqual(result_valuecounts, expected_valuecounts)

    def test_valuecounts_for_colname_selectedby_colname_sort(self):
        cols = ['col1', 'col2', 'col3']
        lol = [['a', 'x', 'y'], ['b', 'x', 'z'], ['a', 'y', 'y'], ['c', 'z', 'z']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        result_valuecounts = pydf.valuecounts_for_colname_selectedby_colname('col2', 'col1', 'a', sort=True)
        expected_valuecounts = {'y': 1, 'x': 1}
        self.assertEqual(result_valuecounts, expected_valuecounts)

    def test_valuecounts_for_colname_selectedby_colname_reverse_sort(self):
        cols = ['col1', 'col2', 'col3']
        lol = [['a', 'x', 'y'], ['b', 'x', 'z'], ['a', 'y', 'y'], ['c', 'z', 'z']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        result_valuecounts = pydf.valuecounts_for_colname_selectedby_colname('col2', 'col1', 'a', sort=True, reverse=True)
        expected_valuecounts = {'x': 1, 'y': 1}
        self.assertEqual(result_valuecounts, expected_valuecounts)

    def test_valuecounts_for_colname_selectedby_colname_not_found(self):
        cols = ['col1', 'col2', 'col3']
        lol = [['a', 'x', 'y'], ['b', 'x', 'z'], ['a', 'y', 'y'], ['c', 'z', 'z']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        result_valuecounts = pydf.valuecounts_for_colname_selectedby_colname('col2', 'col1', 'not_found')
        expected_valuecounts = {}
        self.assertEqual(result_valuecounts, expected_valuecounts)

    def test_valuecounts_for_colname_selectedby_colname_empty_pydf(self):
        pydf = Pydf()

        result_valuecounts = pydf.valuecounts_for_colname_selectedby_colname('col2', 'col1', 'a')
        expected_valuecounts = {}
        self.assertEqual(result_valuecounts, expected_valuecounts)

    # valuecounts_for_colnames_ls_selectedby_colname
    def test_valuecounts_for_colnames_ls_selectedby_colname(self):
        cols = ['col1', 'col2', 'col3']
        lol = [ ['a', 'x', 'y'], 
                ['b', 'x', 'z'], 
                ['c', 'y', 'y'], 
                ['d', 'z', 'z']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        result_valuecounts = pydf.valuecounts_for_colnames_ls_selectedby_colname(
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
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        result_valuecounts = pydf.valuecounts_for_colnames_ls_selectedby_colname(
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
        pydf = Pydf(cols=cols, lol=lol, dtypes={'col1': str, 'col2': str, 'col3': str}, keyfield='')

        result_valuecounts = pydf.valuecounts_for_colnames_ls_selectedby_colname(
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
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        result_valuecounts = pydf.valuecounts_for_colnames_ls_selectedby_colname(
            colnames_ls=['col2', 'col3'],
            selectedby_colname='col1',
            selectedby_colvalue='not_found'
        )
        expected_valuecounts = {'col2': {}, 'col3': {}}
        self.assertEqual(result_valuecounts, expected_valuecounts)

    def test_valuecounts_for_colnames_ls_selectedby_colname_empty_pydf(self):
        pydf = Pydf()

        result_valuecounts = pydf.valuecounts_for_colnames_ls_selectedby_colname(
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
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        result_valuecounts = pydf.valuecounts_for_colname1_groupedby_colname2(
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
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        result_valuecounts = pydf.valuecounts_for_colname1_groupedby_colname2(
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
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        result_valuecounts = pydf.valuecounts_for_colname1_groupedby_colname2(
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
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        result_valuecounts = pydf.valuecounts_for_colname1_groupedby_colname2(
            colname1='col1',
            groupedby_colname2='not_found'
        )
        expected_valuecounts = {}
        self.assertEqual(result_valuecounts, expected_valuecounts)

    def test_valuecounts_for_colname1_groupedby_colname2_empty_pydf(self):
        pydf = Pydf()

        result_valuecounts = pydf.valuecounts_for_colname1_groupedby_colname2(
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
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        result_dopydf = pydf.groupby(colname='col2')

        lolx = [ ['a', 'x', 'y'], 
                 ['b', 'x', 'z']]
        loly = [ ['a', 'y', 'y']]
        lolz = [['c', 'z', 'z']]


        pydf_x = Pydf(cols=cols, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str}, lol=lolx)
        pydf_y = Pydf(cols=cols, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str}, lol=loly)
        pydf_z = Pydf(cols=cols, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str}, lol=lolz)


        expected_dopydf = {
            'x': pydf_x,
            'y': pydf_y,
            'z': pydf_z
        }

        for colvalue, expected_pydf in expected_dopydf.items():
            result_pydf = result_dopydf[colvalue]
            self.assertEqual(result_pydf.columns(), expected_pydf.columns())
            self.assertEqual(result_pydf.to_lod(), expected_pydf.to_lod())

    def test_groupby_empty_pydf(self):
        pydf = Pydf()

        result_dopydf = pydf.groupby(colname='col1')

        expected_dopydf = {}
        self.assertEqual(result_dopydf, expected_dopydf)

    def test_groupby_colname_not_found(self):
        cols = ['col1', 'col2', 'col3']
        lol = [['a', 'x', 'y'], ['b', 'x', 'z'], ['a', 'y', 'y'], ['c', 'z', 'z']]
        pydf = Pydf(cols=cols, lol=lol, keyfield='col1', dtypes={'col1': str, 'col2': str, 'col3': str})

        with self.assertRaises(KeyError):
            pydf.groupby(colname='not_found')

    # test __get_item__
    def test_getitem_single_row(self):
        self.pydf_instance = Pydf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.pydf_instance[1]
        expected_result = [4, 5, 6]
        self.assertEqual(result, expected_result)

    def test_getitem_single_row_with_cols(self):
        self.pydf_instance = Pydf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.pydf_instance[1, :]
        expected_result = [4, 5, 6]
        self.assertEqual(result, expected_result)

    def test_getitem_single_col(self):
        self.pydf_instance = Pydf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.pydf_instance[:, 1]
        expected_result = [2, 5, 8]
        self.assertEqual(result, expected_result)

    def test_getitem_single_colname(self):
        self.pydf_instance = Pydf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.pydf_instance[:, 'B']
        expected_result = [2, 5, 8]
        self.assertEqual(result, expected_result)

    def test_getitem_single_col_with_rows(self):
        self.pydf_instance = Pydf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.pydf_instance[:, 1:2]
        expected_result = [2, 5, 8]
        self.assertEqual(result, expected_result)

    def test_getitem_single_col_with_reduced_rows(self):
        self.pydf_instance = Pydf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.pydf_instance[0:2, 1:2]
        expected_result = [2, 5]
        self.assertEqual(result, expected_result)

    def test_getitem_rows_and_cols(self):
        self.pydf_instance = Pydf(lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        result = self.pydf_instance[1:3, 0:2]
        expected_result = Pydf(lol=[[4, 5], [7, 8]], cols=['A', 'B'])
        self.assertEqual(result, expected_result)

    # test transpose
    def test_transpose(self):
        self.pydf_instance = Pydf(lol=[[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]], cols=['A', 'B', 'C', 'D'], keyfield='A')
        result = self.pydf_instance.transpose(new_keyfield='x', new_cols=['x', 'y', 'z'])
        expected_result = Pydf(lol=[[1, 4, 7], [2, 5, 8], [3, 6, 9], [4, 7, 10]], keyfield='x', cols=['x', 'y', 'z']) 
        self.assertEqual(result, expected_result)


    def test_split_pydf_into_chunks_lopydf(self):
        self.pydf_instance = Pydf(
            lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]],
            cols=['A', 'B', 'C']
            )
        max_chunk_size = 2  # Set the maximum chunk size for testing

        # Call the method to split the Pydf into chunks
        chunks_lopydf = self.pydf_instance.split_pydf_into_chunks_lopydf(max_chunk_size)

        # Check if the length of each chunk is within the specified max_chunk_size
        for chunk in chunks_lopydf:
            self.assertLessEqual(len(chunk), max_chunk_size)

        # Check if the sum of the lengths of all chunks equals the length of the original Pydf
        total_length = sum(len(chunk) for chunk in chunks_lopydf)
        self.assertEqual(total_length, len(self.pydf_instance))

    # __set_item__
    def test_set_item_row_list(self):
        # Assign an entire row using a list
        # Example: Create a Pydf instance with some sample data
        self.pydf_instance = Pydf(
            lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            cols=['A', 'B', 'C']
        )
        self.pydf_instance[1] = {'A': 10, 'B': 20, 'C': 30}
        expected_result = Pydf(lol=[[1, 2, 3], [10, 20, 30], [7, 8, 9]], cols=['A', 'B', 'C'])
        self.assertEqual(self.pydf_instance, expected_result)

    def test_set_item_row_value(self):
        # Assign an entire row using a single value
        # Example: Create a Pydf instance with some sample data
        self.pydf_instance = Pydf(
            lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            cols=['A', 'B', 'C']
        )
        self.pydf_instance[2] = 100
        expected_result = Pydf(lol=[[1, 2, 3], [4, 5, 6], [100, 100, 100]], cols=['A', 'B', 'C'])
        self.assertEqual(self.pydf_instance, expected_result)

    def test_set_item_cell_value(self):
        # Assign a specific cell with a value
        # Example: Create a Pydf instance with some sample data
        self.pydf_instance = Pydf(
            lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            cols=['A', 'B', 'C']
        )
        self.pydf_instance[0, 'B'] = 50
        expected_result = Pydf(lol=[[1, 50, 3], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        self.assertEqual(self.pydf_instance, expected_result)

    def test_set_item_cell_list(self):
        # Assign a specific cell with a list
        # Example: Create a Pydf instance with some sample data
        self.pydf_instance = Pydf(
            lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            cols=['A', 'B', 'C']
        )
        self.pydf_instance[1, 'A'] = [100, 200, 300]
        expected_result = Pydf(lol=[[1, 2, 3], [[100, 200, 300], 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        
        self.longMessage = True
        
        if self.pydf_instance != expected_result:
            print (f"test_set_item_cell_list result:\n{self.pydf_instance}\nexpected{expected_result}")

        self.assertEqual(self.pydf_instance, expected_result)
        
    def test_set_item_row_range_list(self):
        # Assign values in a range of columns in a specific row with a list
        # Example: Create a Pydf instance with some sample data
        self.pydf_instance = Pydf(
            lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            cols=['A', 'B', 'C']
        )
        self.pydf_instance[1, 1:3] = [99, 88]
        expected_result = Pydf(lol=[[1, 2, 3], [4, 99, 88], [7, 8, 9]], cols=['A', 'B', 'C'])
        self.assertEqual(self.pydf_instance, expected_result)

    def test_set_item_row_range_value(self):
        # Assign a single value in a range of columns in a specific row
        # Example: Create a Pydf instance with some sample data
        self.pydf_instance = Pydf(
            lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            cols=['A', 'B', 'C']
        )
        self.pydf_instance[0, 1:3] = 77
        expected_result = Pydf(lol=[[1, 77, 77], [4, 5, 6], [7, 8, 9]], cols=['A', 'B', 'C'])
        self.assertEqual(self.pydf_instance, expected_result)

    def test_set_item_col_list(self):
        # Assign an entire column with a list
        # Example: Create a Pydf instance with some sample data
        self.pydf_instance = Pydf(
            lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            cols=['A', 'B', 'C']
        )
        self.pydf_instance[:, 'B'] = [55, 66, 77]
        expected_result = Pydf(lol=[[1, 55, 3], [4, 66, 6], [7, 77, 9]], cols=['A', 'B', 'C'])
        if self.pydf_instance != expected_result:
            print (f"test_set_item_col_list result:\n{self.pydf_instance}\nexpected{expected_result}")

        self.assertEqual(self.pydf_instance, expected_result)

    def test_set_item_col_range_list(self):
        # Assign values in a range of rows in a specific column with a list
        # Example: Create a Pydf instance with some sample data
        self.pydf_instance = Pydf(
            lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            cols=['A', 'B', 'C']
        )
        self.pydf_instance[1:3, 'B'] = [44, 33]
        expected_result = Pydf(lol=[[1, 2, 3], [4, 44, 6], [7, 33, 9]], cols=['A', 'B', 'C'])
        self.assertEqual(self.pydf_instance, expected_result)

    # select_where
    def test_select_where_basic_condition(self):
        # Test a basic condition where col1 values are greater than 2
        pydf = Pydf(cols=['col1', 'col2'], lol=[[1, 4], [2, 5], [3, 6]])

        result_pydf = pydf.select_where(lambda row: row['col1'] > 2)
        expected_data = Pydf(cols=['col1', 'col2'], lol=[[3, 6]])
        self.assertEqual(result_pydf, expected_data)

    # def test_select_where_invalid_condition_syntax_error(self):
        # # Test an invalid condition with syntax error
        # pydf = Pydf(cols=['col1', 'col2'], lol=[[1, 4], [2, 5], [3, 6]])

        # with self.assertRaises(SyntaxError) as context:
            # pydf.select_where("row['col1'] >")

        # self.assertIn("invalid syntax", str(context.exception))

    def test_select_where_invalid_condition_runtime_error(self):
        # Test an invalid condition causing a runtime error
        pydf = Pydf(cols=['col1', 'col2'], lol=[[1, 4], [2, 5], [3, 6]])

        with self.assertRaises(ZeroDivisionError) as context:
            pydf.select_where(lambda row: 1 / 0)

        self.assertIn("division by zero", str(context.exception))

    def test_select_where_empty_result(self):
        # Test a condition that results in an empty Pydf
        pydf = Pydf(cols=['col1', 'col2'], lol=[[1, 4], [2, 5], [3, 6]])

        result_pydf = pydf.select_where(lambda row: row['col1'] > 10)
        expected_data = Pydf(cols=['col1', 'col2'], lol=[])
        self.assertEqual(result_pydf, expected_data)

    def test_select_where_complex_condition(self):
        # Test a complex condition involving multiple columns
        pydf = Pydf(cols=['col1', 'col2'], lol=[[1, 4], [2, 5], [3, 6]])

        result_pydf = pydf.select_where(lambda row: row['col1'] > 1 and row['col2'] < 6)
        expected_data = Pydf(cols=['col1', 'col2'], lol=[[2, 5]])
        self.assertEqual(result_pydf, expected_data)

    def test_select_where_complex_condition_indexes(self):
        # Test a complex condition involving multiple columns
        pydf = Pydf(cols=['col1', 'col2'], lol=[[1, 4], [2, 5], [3, 6]])

        result_pydf = pydf.select_where(lambda row: bool(list(row.values())[0] > 1 and list(row.values())[1] < 6))
        expected_data = Pydf(cols=['col1', 'col2'], lol=[[2, 5]])
        self.assertEqual(result_pydf, expected_data)

    # test test_from_cols_dol
    def test_from_cols_dol_empty_input(self):
        # Test creating Pydf instance from empty cols_dol
        cols_dol = {}
        result_pydf = Pydf.from_cols_dol(cols_dol)
        expected_pydf = Pydf()
        self.assertEqual(result_pydf, expected_pydf)

    def test_from_cols_dol_basic_input(self):
        # Test creating Pydf instance from cols_dol with basic data
        cols_dol = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
        result_pydf = Pydf.from_cols_dol(cols_dol)
        expected_pydf = Pydf(cols=['A', 'B', 'C'], lol=[[1, 4, 7], [2, 5, 8], [3, 6, 9]])
        self.assertEqual(result_pydf, expected_pydf)

    def test_from_cols_dol_with_keyfield(self):
        # Test creating Pydf instance from cols_dol with keyfield specified
        cols_dol = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
        result_pydf = Pydf.from_cols_dol(cols_dol, keyfield='A')
        expected_pydf = Pydf(cols=['A', 'B', 'C'], lol=[[1, 4, 7], [2, 5, 8], [3, 6, 9]], keyfield='A')
        self.assertEqual(result_pydf, expected_pydf)

    def test_from_cols_dol_with_dtypes(self):
        # Test creating Pydf instance from cols_dol with specified dtype
        cols_dol = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
        dtypes = {'A': int, 'B': float, 'C': str}
        result_pydf = Pydf.from_cols_dol(cols_dol, dtypes=dtypes)
        expected_pydf = Pydf(cols=['A', 'B', 'C'], lol=[[1, 4.0, '7'], [2, 5.0, '8'], [3, 6.0, '9']], dtypes=dtypes)
        self.assertEqual(result_pydf, expected_pydf)

    # # to_dict
    # def test_to_dict_empty_pydf(self):
        # # Test to_dict() on an empty Pydf instance
        # pydf = Pydf()
        # #result_dict = pydf.to_dict()
        # #expected_pydf = {'cols': [], 'lol': []}
        # self.assertEqual(pydf.lol, [])
        # self.assertEqual(pydf.kd, {})
        # self.assertEqual(pydf.kd, {})

    # def test_to_dict_with_data(self):
        # # Test to_dict() on a Pydf instance with data
        # pydf = Pydf(cols=['A', 'B', 'C'], lol=[[1, 4, 7], [2, 5, 8], [3, 6, 9]])
        # #result_dict = pydf.to_dict()
        # expected_pydf = Pydf('cols'= ['A', 'B', 'C'], 'lol'= [[1, 4, 7], [2, 5, 8], [3, 6, 9]])
        # self.assertEqual(result_pydf, expected_pydf)

    # def test_to_dict_with_keyfield_and_dtypes(self):
        # # Test to_dict() on a Pydf instance with keyfield and dtype
        # pydf = Pydf(cols=['A', 'B', 'C'], 
                    # lol=[[1, 4, 7], [2, 5, 8], [3, 6, 9]], 
                    # keyfield='A', 
                    # dtypes={'A': int, 'B': float, 'C': int})
        # #result_dict = pydf.to_dict()
        # expected_pydf = Pydf('cols'= ['A', 'B', 'C'], 'lol'= [[1, 4, 7], [2, 5, 8], [3, 6, 9]])
        # self.assertEqual(result_pydf, expected_pydf)

    # apply_formulas
    def test_apply_formulas_basic_absolute(self):
        # Test apply_formulas with basic example
        example_pydf = Pydf(cols=['A', 'B', 'C'], lol=[[1, 2, 0], [4, 5, 0], [7, 8, 0], [0, 0, 0]])
        formulas_pydf = Pydf(cols=['A', 'B', 'C'],
                             lol=[['', '', "$d[0,0]+$d[0,1]"],
                                  ['', '', "$d[1,0]+$d[1,1]"],
                                  ['', '', "$d[2,0]+$d[2,1]"],
                                  ["sum($d[0:3,$c])", "sum($d[0:3,$c])", "sum($d[0:3,$c])"]]
                             )
        expected_pydf = Pydf(cols=['A', 'B', 'C'], lol=[[1, 2, 3], [4, 5, 9], [7, 8, 15], [12, 15, 27]])

        example_pydf.apply_formulas(formulas_pydf)
        self.assertEqual(example_pydf, expected_pydf)

    def test_apply_formulas_basic_relative(self):
        # Test apply_formulas with basic example
        example_pydf = Pydf(cols=['A', 'B', 'C'], lol=[[1, 2, 0], [4, 5, 0], [7, 8, 0], [0, 0, 0]])
        formulas_pydf = Pydf(cols=['A', 'B', 'C'],
                             lol=[['', '', "$d[$r,0]+$d[$r,1]"],
                                  ['', '', "$d[$r,($c-2)]+$d[$r,($c-1)]"],
                                  ['', '', "sum($d[$r,0:2])"],
                                  ["sum($d[0:3,$c])", "sum($d[:-1,$c])", "sum($d[:$r,$c])"]]
                             )
        expected_result = Pydf(cols=['A', 'B', 'C'], lol=[[1, 2, 3], [4, 5, 9], [7, 8, 15], [12, 15, 27]])

        example_pydf.apply_formulas(formulas_pydf)
        self.assertEqual(example_pydf, expected_result)

    def test_apply_formulas_no_changes(self):
        # Test apply_formulas with no changes expected
        example_pydf = Pydf(cols=['A', 'B', 'C'], lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0]])
        formulas_pydf = Pydf(cols=['A', 'B', 'C'], lol=[['', '', ''], ['', '', ''], ['', '', ''], ['', '', '']])
        expected_result = Pydf(cols=['A', 'B', 'C'], lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0]])

        example_pydf.apply_formulas(formulas_pydf)
        self.assertEqual(example_pydf, expected_result)

    def test_apply_formulas_excessive_loops(self):
        # Test apply_formulas resulting in excessive loops
        example_pydf = Pydf(cols=['A', 'B', 'C'], lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0]])
        formulas_pydf = Pydf(cols=['A', 'B', 'C'],
                             lol=[['', '', "$d[0,0]+$d[0,1]"],
                                  ['', '', "$d[1,0]+$d[1,1]"],
                                  ['$d[2,2]', '', "$d[2,0]+$d[2,1]"],     # this is circular
                                  ["sum($d[0:3,'A'])", "sum($d[0:3,'B'])", "sum($d[0:3,'C'])"]]
                             )

        with self.assertRaises(RuntimeError) as context:
            example_pydf.apply_formulas(formulas_pydf)

        self.assertIn("apply_formulas is resulting in excessive evaluation loops.", str(context.exception))

    def test_generate_spreadsheet_column_names_list(self):
        # Test for 0 columns
        self.assertEqual(Pydf._generate_spreadsheet_column_names_list(0), [])

        # Test for 1 column
        self.assertEqual(Pydf._generate_spreadsheet_column_names_list(1), ['A'])

        # Test for 5 columns
        self.assertEqual(Pydf._generate_spreadsheet_column_names_list(5), ['A', 'B', 'C', 'D', 'E'])

        # Test for 27 columns
        self.assertEqual(Pydf._generate_spreadsheet_column_names_list(27), ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'AA'])

        # Test for 52 columns
        self.assertEqual(Pydf._generate_spreadsheet_column_names_list(52), ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI', 'AJ', 'AK', 'AL', 'AM', 'AN', 'AO', 'AP', 'AQ', 'AR', 'AS', 'AT', 'AU', 'AV', 'AW', 'AX', 'AY', 'AZ'])

        # Test for 53 columns
        self.assertEqual(Pydf._generate_spreadsheet_column_names_list(53), ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI', 'AJ', 'AK', 'AL', 'AM', 'AN', 'AO', 'AP', 'AQ', 'AR', 'AS', 'AT', 'AU', 'AV', 'AW', 'AX', 'AY', 'AZ', 'BA'])


    # from_lod_to_cols
    def test_from_lod_to_cols_empty_lod(self):
        result = Pydf.from_lod_to_cols([], cols=['A', 'B', 'C'], keyfield='Key')
        self.assertEqual(result.columns(), ['A', 'B', 'C'])
        self.assertEqual(result.lol, [])

    def test_from_lod_to_cols_no_cols_specified(self):
        lod = [{'A': 1, 'B': 2, 'C': 3}, {'A': 4, 'B': 5, 'C': 6}, {'A': 7, 'B': 8, 'C': 9}]
        result = Pydf.from_lod_to_cols(lod, keyfield='key')
        self.assertEqual(result.columns(), ['key', 'A', 'B', 'C'])
        self.assertEqual(result.lol, [['A', 1, 4, 7], ['B', 2, 5, 8], ['C', 3, 6, 9]])

    def test_from_lod_to_cols_with_cols(self):
        lod = [{'A': 1, 'B': 2, 'C': 3}, {'A': 4, 'B': 5, 'C': 6}, {'A': 7, 'B': 8, 'C': 9}]
        result = Pydf.from_lod_to_cols(lod, cols=['Feature', 'Try 1', 'Try 2', 'Try 3'], keyfield='Feature')
        self.assertEqual(result.columns(), ['Feature', 'Try 1', 'Try 2', 'Try 3'])
        self.assertEqual(result.lol, [['A', 1, 4, 7], ['B', 2, 5, 8], ['C', 3, 6, 9]])

    # apply
    def test_apply_row(self):
        pydf = Pydf.from_lod([  {'a': 1, 'b': 2}, 
                                {'a': 3, 'b': 4}])

        def transform_row(
                row: dict, 
                cols=None,                      # columns included in the reduce operation.
                **kwargs):
            return {'a': row['a'] * 2, 'b': row['b'] * 3}

        result_pydf = pydf.apply(transform_row, by='row')
        expected_result = Pydf.from_lod([{'a': 2, 'b': 6}, {'a': 6, 'b': 12}])

        self.assertEqual(result_pydf, expected_result)

    # def test_apply_col(self):
        # pydf = Pydf.from_lod([  {'a': 1, 'b': 2}, 
                                # {'a': 3, 'b': 4}])

        # def transform_col(col, cols, **kwargs):
            # col[0] = col[0] * 2
            # col[1] = col[1] * 3
            # return col

        # result_pydf = pydf.apply(transform_col, by='col')
        # expected_result = Pydf.from_lod([
                                # {'a': 2, 'b': 4}, 
                                # {'a': 9, 'b': 12}])

        # self.assertEqual(result_pydf, expected_result)


    def test_set_col2_from_col1_using_regex_select(self):
        # Initialize an instance of your class
        
        # Set up sample data for testing
        cols_dol = {'col1': ['abc (123)', 'def (456)', 'ghi (789)'],
                'col2': [None, None, None]}
        my_pydf = Pydf.from_cols_dol(cols_dol)

        # Call the method to apply the regex select
        my_pydf.set_col2_from_col1_using_regex_select('col1', 'col2', r'\((\d+)\)')
        
        # Assert the expected results
        self.assertEqual(my_pydf[:, 'col2'], ['123', '456', '789'])
        

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
            
        data_table_pydf = Pydf(cols=cols, lol=lol)
            
            
        grouped_and_summed_pydf = data_table_pydf.groupby_cols_reduce(
            groupby_colnames=groupby_colnames, 
            func = Pydf.sum_np,
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

        self.assertEqual(grouped_and_summed_pydf.lol, expected_lol)
        

        

if __name__ == '__main__':
    unittest.main()
