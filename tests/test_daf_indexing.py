# test_daf_indexing.py

import unittest
from daffodil.daf import Daf, KeyedList  # Assuming your package is named `daffodil`

class TestDaffodilIndexing(unittest.TestCase):
    def setUp(self):
        # Sample data to use in tests
        self.data = [
            ["row1", 2, 3, 4],
            ["row2", 6, 7, 8],
            ["row3", 10, 11, 12],
        ]
        self.cols = ["A", "B", "C", "D"]
        self.keyfield = "A"  # Setting the keyfield to the first column, which contains strings
        self.daf = Daf(lol=self.data, cols=self.cols, keyfield=self.keyfield)

    
    def test_select_cell(self):
        # Test `my_daf["row3", 3]` to select cell at keyfield "row3", col 3
        result = self.daf["row3", 3]
        expected = Daf(lol=[[12]], cols=["D"], keyfield="")  # Keyfield is reset to "" as it is not propagated
        self.assertEqual(result, expected)

    def test_select_cell_value(self):
        # Test `my_daf["row3", 3].to_value()` to get the value at keyfield "row3", col 3
        result = self.daf["row3", 3].to_value()
        expected = 12
        self.assertEqual(result, expected)

    def test_to_dict(self):
        # Test `.to_dict()` method
        result = self.daf["row3"].to_dict()
        expected = {"A": "row3", "B": 10, "C": 11, "D": 12}
        self.assertEqual(result, expected)

    def test_to_list(self):
        # Test `.to_list()` method
        result = self.daf["row3"].to_list()
        expected = ["row3", 10, 11, 12]
        self.assertEqual(result, expected)

    def test_iter_klist(self):
        # Test `.iter_klist()` method
        for i, klist in enumerate(self.daf.iter_klist()):
            expected_keys = ["A", "B", "C", "D"]
            expected_values = self.data[i]

            # Check if it is an instance of KeyedList
            self.assertIsInstance(klist, KeyedList)

            # Verify keys and values
            self.assertEqual(list(klist.keys()), expected_keys)
            self.assertEqual(list(klist.values()), expected_values)

            # Modify the list through the KeyedList
            klist["B"] += 100

        # Ensure changes are reflected in the original data
        self.assertEqual(self.daf["row1", "B"].to_value(), 102)  # First row, column "B" modified
        self.assertEqual(self.daf["row2", "B"].to_value(), 106)  # Second row, column "B" modified

    def test_slice_rows(self):
        # Test slicing rows, e.g., `my_daf[1:3]`
        result = self.daf[1:3]
        expected = Daf(lol=[["row2", 6, 7, 8], ["row3", 10, 11, 12]], cols=self.cols, keyfield=self.keyfield)
        self.assertEqual(result, expected)

    def test_slice_columns(self):
        # Test slicing columns, e.g., `my_daf[:, 1:3]`
        result = self.daf[:, 1:3]
        expected = Daf(lol=[[2, 3], [6, 7], [10, 11]], cols=["B", "C"], keyfield="")
        self.assertEqual(result, expected)

    def test_select_rows_by_key(self):
        # Test selecting specific rows by keyfield values, e.g., `my_daf[["row1", "row3"]]`
        result = self.daf[["row1", "row3"]]
        expected = Daf(lol=[["row1", 2, 3, 4], ["row3", 10, 11, 12]], cols=self.cols, keyfield=self.keyfield)
        self.assertEqual(result, expected)

    def test_select_columns_by_name(self):
        # Test selecting specific columns by names, e.g., `my_daf[:, ["B", "D"]]`
        result = self.daf[:, ["B", "D"]]
        expected = Daf(lol=[[2, 4], [6, 8], [10, 12]], cols=["B", "D"], keyfield="")
        self.assertEqual(result, expected)

    def test_combined_indexing(self):
        # Test combined indexing, e.g., `my_daf[["row1", "row3"], ["B", "D"]]`
        result = self.daf[["row1", "row3"], ["B", "D"]]
        expected = Daf(lol=[[2, 4], [10, 12]], cols=["B", "D"], keyfield="")
        self.assertEqual(result, expected)
        
    def test_row_key_as_tuple(self):
        # Test slicing rows using tuple keys, e.g., `my_daf[("row2", "row3"), :]`
        result = self.daf[("row2", "row3"), :]
        expected = Daf(lol=[["row2", 6, 7, 8], ["row3", 10, 11, 12]], cols=self.cols, keyfield=self.keyfield)
        self.assertEqual(result, expected)

    
    def test_column_key_as_tuple(self):
        # Test slicing columns using tuple keys, e.g., `my_daf[:, ("B", "D")]`
        result = self.daf[:, ("B", "D")]
        expected = Daf(lol=[[2, 4], [6, 8], [10, 12]], cols=["B", "D"], keyfield="")
        self.assertEqual(result, expected)

    def test_keyfield_reset_on_subset(self):
        # Validate keyfield reset when keyfield data is excluded
        result = self.daf[:, ["B", "C"]]
        expected = Daf(lol=[[2, 3], [6, 7], [10, 11]], cols=["B", "C"], keyfield="")
        self.assertEqual(result, expected)

    def test_indexing_with_range(self):
        # Test indexing with range for rows
        result = self.daf[range(1, 3)]
        expected = Daf(lol=[["row2", 6, 7, 8], ["row3", 10, 11, 12]], cols=self.cols, keyfield=self.keyfield)
        self.assertEqual(result, expected)

    def test_full_row_selection(self):
        # Ensure selecting a full row returns the expected Daf
        result = self.daf["row2"]
        expected = Daf(lol=[["row2", 6, 7, 8]], cols=self.cols, keyfield=self.keyfield)
        self.assertEqual(result, expected)
    
    def test_integer_indexing_single_integer(self):
        # Test single integer indexing for rows and columns
        result = self.daf[1, 2]  # Access row 1, column 2
        expected = Daf(lol=[[7]], cols=["C"], keyfield="")
        self.assertEqual(result, expected)

    def test_integer_indexing_range(self):
        # Test integer range indexing for rows and columns
        result = self.daf[1:3, 1:3]  # Access rows 1-2, columns 1-2
        expected = Daf(lol=[[6, 7], [10, 11]], cols=["B", "C"], keyfield="")
        self.assertEqual(result, expected)

    def test_integer_indexing_slice(self):
        # Test slice indexing for rows and columns
        result = self.daf[slice(0, 2), slice(2, 4)]  # Access rows 0-1, columns 2-3
        expected = Daf(lol=[[3, 4], [7, 8]], cols=["C", "D"], keyfield="")
        self.assertEqual(result, expected)

    def test_integer_indexing_list_of_ranges(self):
        # Test list of ranges indexing for rows and columns
        result = self.daf[[range(0, 1), range(2, 3)]]  # Access rows 0,2
        expected = Daf(lol=[
            ["row1", 2, 3, 4],
            ["row3", 10, 11, 12],
            ], cols=["A", "B", "C", "D"], keyfield="A")
        self.assertEqual(result, expected)

    def test_integer_indexing_list_of_integers(self):
        # Test list of integers indexing for rows and columns
        result = self.daf[[0, 2], [1, 3]]  # Access rows 0 and 2, columns 1 and 3
        expected = Daf(lol=[[2, 4], [10, 12]], cols=["B", "D"], keyfield="")
        self.assertEqual(result, expected)
        
    def test_column_indexing_single_integer(self):
        # Test single integer indexing for columns
        result = self.daf[:, 2]  # Access all rows, column 2
        expected = Daf(lol=[[3], [7], [11]], cols=["C"], keyfield="")
        self.assertEqual(result, expected)

    def test_column_indexing_range(self):
        # Test integer range indexing for columns
        result = self.daf[:, 1:3]  # Access all rows, columns 1-2
        expected = Daf(lol=[[2, 3], [6, 7], [10, 11]], cols=["B", "C"], keyfield="")
        self.assertEqual(result, expected)

    def test_column_indexing_slice(self):
        # Test slice indexing for columns
        result = self.daf[:, slice(2, 4)]  # Access all rows, columns 2-3
        expected = Daf(lol=[[3, 4], [7, 8], [11, 12]], cols=["C", "D"], keyfield="")
        self.assertEqual(result, expected)

    def test_column_indexing_list_of_ranges(self):
        # Test list of ranges indexing for columns
        result = self.daf[:, [range(1, 3)]]  # Access all rows, columns 1-2
        expected = Daf(lol=[[2, 3], [6, 7], [10, 11]], cols=["B", "C"], keyfield="")
        self.assertEqual(result, expected)

    def test_column_indexing_list_of_integers(self):
        # Test list of integers indexing for columns
        result = self.daf[:, [0, 2]]  # Access all rows, columns 0 and 2
        expected = Daf(lol=[["row1", 3], ["row2", 7], ["row3", 11]], cols=["A", "C"], keyfield="A")
        self.assertEqual(result, expected)
        
if __name__ == "__main__":
    unittest.main()