# Pydf.py
"""

# Pydf -- Python Dataframes

The Pydf class provides a lightweight, simple and fast alternative to provide 2-d data arrays with mixed types.

"""

"""
    MIT License

    Copyright (c) 2024 Ray Lutz

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""


"""
See README file at this location: https://github.com/raylutz/Pydf/blob/main/README.md
"""

"""
    v0.1.X (pending)
            Started creating separate package, moved comment text to README.md
            For apply_formulas(), added relative row and column references $r and $c plus $d to reference the pydf object.
            Changed the result of a row access, such as $d[$r, :$c] to return a list so it could be compatible with sum.
                use pydf.select_irow() to select a row with dict as the result.
    
    v0.2.0  (2024-02-03) 
            Copied related code to Pydf repo and resolved all imports. All tests running.
            Added option of appending a plain list to pydf instance using .append()
            Added 'omit_nulls' option to col(), col_to_la(), icol_to_la(), valuecounts_for_colname()
            Added ability to groupby multiple cols
            in select_records_pydf(self, keys_ls: T_ls, inverse:bool=False), added inverse boolean.
            Started to add selcols for zero-copy support.
            Added _num_cols() 
            Added unit tests.
            Add groupby_cols_reduce() and sum_np()
            Fixed bug in item setter for str row. 
            
            
    TODO
            Refactor get_item and set_item
    
"""            
    
    
#VERSION  = 'v0.1.X'
#VERSDATE = '2024-01-21'  

import sys
import io
import csv
import copy
import re
import numpy as np
    
sys.path.append('..')

from Pydf.pydf_types import T_ls, T_lola, T_di, T_hllola, T_loda, T_da, T_li, T_dtype_dict, \
                            T_dola, T_dodi, T_la, T_lota, T_doda, T_buff, T_df, T_ds
                     
import Pydf.pydf_utils as utils
import Pydf.pydf_md    as md

from typing import List, Dict, Any, Tuple, Optional, Union, cast, Type, Callable #
def fake_function(a: Optional[List[Dict[str, Tuple[int,Union[Any, str]]]]] = None) -> Optional[int]:
    return None or cast(int, 0)

T_Pydf = Type['Pydf']



class Pydf:

    """ my_pydf = Pydf() """

    def __init__(self, 
            lol:        Optional[T_lola]        = None,     # Optional List[List[Any]] to initialize the data array. 
            cols:       Optional[T_ls]          = None,     # Optional column names to use.
            dtypes:     Optional[T_dtype_dict]  = None,     # Optional dtype_dict describing the desired type of each column.
                                                            #   also used to define column names if provided and cols not provided.
            keyfield:   str                     = '',       # A field of the columns to be used as a key.
            name:       str                     = '',       # An optional name of the Pydf array.
            use_copy:   bool                    = False,    # If True, make a deep copy of the lol data.
            disp_cols:  Optional[T_ls]          = None,     # Optional list of strings to use for display, if initialized.
            sanitize_cols: bool                 = False,    # check for blank or missing cols and make complete and unique.
        ):
        if lol is None:
            lol = []
        if dtypes is None:
            dtypes = {}
        if cols is None:
            cols = []
            
        self.name           = name              # str
        self.keyfield       = keyfield          # str
        self.hd             = {}                # hd_di
        self.selcols_li     = []                # currently selected columns
        
        if use_copy:
            self.lol        = copy.deepcopy(lol)
        else:
            self.lol        = lol
        
        self.kd             = {}
        self.dtypes         = dtypes
        
        self.md_max_rows    = 10    # default number of rows when used with __repr__ and __str__
        self.md_max_cols    = 10    # default number of cols when used with __repr__ and __str__

        # Initialize iterator variables        
        self._iter_index = 0

        if not cols:
            if dtypes:
                self.hd = {col: idx for idx, col in enumerate(dtypes.keys())}
        elif sanitize_cols:        
                # make sure there are no blanks and columns are unique.
                # this does column renaming, and builds hd
                self._sanitize_cols(cols)
        else:
            self._cols_to_hd(cols)
            
        if self.hd and dtypes:
            effective_dtypes = {col: dtypes.get(col, str) for col in self.hd}
        
            # setting dtypes may be better done manually if required.
            if self._num_cols():

                self.lol = utils.apply_dtypes_to_hdlol((self.hd, self.lol), effective_dtypes)[1]
            
        # rebuild kd if possible.
        self._rebuild_kd()
        
            
    #===========================
    # basic attributes and methods

    def __iter__(self):
        self._iter_index = 0
        return self

    def __next__(self) -> Dict[str, int]:
        if self._iter_index < len(self.lol):
            row_dict = dict(zip(self.hd.keys(), self.lol[self._iter_index]))
            self._iter_index += 1
            return row_dict
        else:
            self._iter_index = 0
            raise StopIteration

    def __bool__(self):
        """ test pydf for existance and not empty 
            test exists in test_pydf.py            
        """
        return bool(self._num_cols())


    def __len__(self):
        """ Return the number of rows in the Pydf instance.
            test exists in test_pydf.py            
        """
        return len(self.lol)
        
        
    def len(self):
        return len(self.lol)
        
        
    def shape(self):
        """ return the number of rows and cols in the pydf data array
            number of columns is based on the first record
        """
        # test exists in test_pydf.py
        
        if not len(self): return (0, 0)
        
        return (len(self.lol), self._num_cols()) 
        
        
    def __eq__(self, other):
        # test exists in test_pydf.py            

        if not isinstance(other, Pydf):
            return False

        return (self.lol == other.lol and self.columns() == other.columns() and self.keyfield == other.keyfield)

    
    def __str__(self) -> str:
        return self.md_pydf_table_snippet()
        
        
    def __repr__(self) -> str:
        return "\n"+self.md_pydf_table_snippet()
    

    def _num_cols(self) -> int:
    
        if not self.lol:
            return 0
        return len(self.lol[0])
        

    #===========================
    # column names
    def columns(self):
        """ Return the column names 
        """
        # test exists in test_pydf.py            
        return list(self.hd.keys())
        
        
    def _cols_to_hd(self, cols: T_ls):
        """ rebuild internal hd from cols provided """
        self.hd = {col:idx for idx, col in enumerate(cols)}
        
        
    def _sanitize_cols(self, cols: T_ls):
        # make sure there are no blanks and columns are unique.
        if cols:
            try:
                cols = [col if col else f"Unnamed{idx}" for idx, col in enumerate(cols)] 
            except Exception as err:
                print(f"{err}")
                import pdb; pdb.set_trace() #temp
                pass
            col_hd = {}
            for idx, col in enumerate(cols):
                if col not in col_hd:
                    col_hd[col] = idx
                else:
                    # if not unique, add _NNN after the name.
                    col_hd[f"{col}_{idx}"] = idx
            self.hd = col_hd
            

    def is_in_colnames(self, colname: str) -> bool:    
        return bool(colname in self.hd)
                

    def calc_cols(self, 
            include_cols: Optional[T_la]=None,
            exclude_cols: Optional[T_la]=None,
            include_types: Optional[List[Type]]=None,
            exclude_types: Optional[List[Type]]=None,
           ) -> T_la:
        """ this method helps to calculate the columns to be specified for a apply or reduce operation.
            Can use any combination of listing columns to be included, or excluded by name,
                or included by type.
            If using a groupby function, the cols spec should not include the groupby column(s)
        """
            
            
        # start with all cols.
        selected_cols = list(self.hd.keys())
        if not selected_cols:
            return []
            
        if include_cols:
            if len(include_cols) > 10:
                include_cols_dict = dict.fromkeys(include_cols)
                selected_cols = [col for col in selected_cols if col in include_cols_dict]
            else:
                selected_cols = [col for col in selected_cols if col in include_cols]

        if exclude_cols:
            if len(exclude_cols) > 10:
                exclude_cols_dict = dict.fromkeys(exclude_cols)
                selected_cols = [col for col in selected_cols if col not in exclude_cols_dict]
            else:
                selected_cols = [col for col in selected_cols if col not in exclude_cols]

        if include_types and self.dtypes:
            if not isinstance(include_types, list):
                include_types = [include_types]
            selected_cols = [col for col in selected_cols if self.dtypes.get(col) in include_types]

        if exclude_types and self.dtypes:
            if not isinstance(exclude_types, list):
                include_types = [include_types]
            selected_cols = [col for col in selected_cols if self.dtypes.get(col) not in exclude_types]

        return selected_cols
        
        
    def normalize(self, defined_cols: T_ls):
        # add or subtract columns and place in order per defined_cols.
        if not self:
            return
        
        # from utilities import utils

        for irow, da in enumerate(self):
            record_da = utils.set_cols_da(da, defined_cols)
            self.update_record_da_irow(irow, record_da)
            
        return
        

    @staticmethod
    def _calculate_single_column_name(index: int) -> str:
        """ provide the spreadsheet-style column name for integer offset.
        """
        
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        result = ''
        
        while index >= 0:
            index, remainder = divmod(index, 26)
            result = letters[remainder] + result
            index -= 1
    
        return result
        
    
    @staticmethod
    def _generate_spreadsheet_column_names_list(num_columns: int) -> T_ls:
        """ generate a full list of column names for the num_columns specified 
        """
    
        return [Pydf._calculate_single_column_name(i) for i in range(num_columns)]


    def rename_cols(self, from_to_dict: T_ds):
        """ rename columns using the from_to_dict provided. 
            respects dtypes and rebuilds hd
        """
        # unit tests exist
        
        self.hd     = {from_to_dict.get(col, col):idx for idx, col in enumerate(self.hd.keys())}
        self.dtypes = {from_to_dict.get(col, col):typ for col, typ in self.dtypes.items()}
        if self.keyfield:
            self.keyfield = from_to_dict.get(self.keyfield, self.keyfield)
  

    #===========================
    # keyfield
        
    def keys(self):
        """ return list of keys from kd of keyfield
            test exists in test_pydf.py            
        """
        
        if not self.keyfield:
            return []
        
        return list(self.kd.keys())

    def set_keyfield(self, keyfield:str=''):
        """ set the indexing keyfield to a new column, which must exist
        """
        if keyfield:
            self.keyfield = keyfield
            self._rebuild_kd()
    
        
    def _rebuild_kd(self) -> None:
        """ anytime deletions are performed, the kd must be rebuilt 
            if the keyfield is set.
        """
        
        if self.keyfield and self.keyfield in self.hd:
            col_idx = self.hd[self.keyfield]
            self.kd = Pydf._build_kd(col_idx, self.lol)


    @staticmethod
    def _build_kd(col_idx: int, lol: T_lola) -> T_di:
        """ build key dictionary from col_idx col of lol """
        
        # from utilities import utils

        key_col = utils.select_col_of_lol_by_col_idx(lol, col_idx)
        kd = {key: index for index, key in enumerate(key_col)}
        return kd
        
        
    
    #===========================
    # dtypes
    
    def apply_dtypes(self):
    
        # from utilities import utils
        
        self.lol = utils.apply_dtypes_to_hdlol((self.hd, self.lol), self.dtypes)[1]
        
        
    def unflatten_cols(self, cols: T_ls):
        """ 
            given a pydf and list of cols, 
            convert cols named to either list or dict if col exists and it appears to be 
                stringified using f"{}" functionality.
                
        """

        if not self:
            return    
            
        # from utilities import utils

        self.hd, self.lol = utils.unflatten_hdlol_by_cols((self.hd, self.lol), cols)    


    def unflatten_dirname(self, dirname: str):

        if not self:
            return
        
        from models.BIF import BIF

        cols = BIF.get_dirname_cols_with_format(dirname=dirname, fmt='json')
        if not cols:
            return
        
        self.unflatten_cols(cols)
        
        
    def unflatten_by_dtypes(self):

        if not self or not self.dtypes:
            return
                
        unflatten_cols = self.calc_cols(include_types = [list, dict])
        
        if not unflatten_cols:
            return
       
        self.unflatten_cols(unflatten_cols)
        
        
    def flatten_cols(self, cols: T_ls):
        # given a pydf, convert given list of columns to json.

        if not self:
            return
        
        # from utilities import utils

        for irow, da in enumerate(self):
            record_da = copy.deepcopy(da)
            for col in cols:
                if col in da:
                    record_da[col] = utils.json_encode(record_da[col])        
            self.update_record_da_irow(irow, record_da)        
    
    
    def flatten_by_dtypes(self):

        if not self or not self.dtypes:
            return
                
        flatten_cols = self.calc_cols(include_types = [list, dict])
        
        if not flatten_cols:
            return
       
        self.flatten_cols(cols=flatten_cols)


    def flatten_dirname(self, dirname: str):

        if not self:
            return
            
        # change True/False to '1'/'0' strings.
        from models.BIF import BIF

        strbool_cols = BIF.get_dirname_cols_with_format(dirname=dirname, fmt='strbool')
        self.cols_to_strbool(strbool_cols)
        
        # make sure the df has all the columns and no extra columns.
        self.normalize(defined_cols=BIF.get_dirname_columns(dirname))
        
        # convert objects marked with format 'json' to json strings.
        json_cols = BIF.get_dirname_cols_with_format(dirname=dirname, fmt='json')
        self.flatten_cols(cols=json_cols)
        

    def cols_to_strbool(self, cols: T_ls):
        # given a lod, convert given list of columns to strbool.

        # from utilities import utils

        for irow, da in enumerate(self):
            record_da = {k:utils.make_strbool(da[k]) for k in cols if k in da}
            self.update_record_da_irow(irow, record_da)        

    def _safe_tofloat(val: Any) -> Union[float, str]:
        try:
            return float(val)
        except ValueError:
            return 0.0
    
           
                        
    
    #===========================
    # indexing

    def __getitem__(self, slice_spec: Union[slice, int, Tuple[Union[slice, int], Union[slice, int]]]) -> 'Pydf':
        """ allow selection and slicing using one or two specs:
        
            my_pydf[2, 3]         -- select cell at row 2, col 3 and return value.
            my_pydf[2]            -- select row 2, including all columns, return as a list.
            my_pydf[2, :]         -- same as above
            my_pydf[[2], :]       -- same as above
            my_pydf[[2,5,8]]      -- select rows 2, 5, and 8, including all columns
            my_pydf[[2,5,8], :]   -- same as above.
            my_pydf[:, 3]         -- select only column 3, including all rows. Return as a list.
            my_pydf[:, 'C']       -- select only column named 'C', including all rows, return as a list.
            my_pydf[:, ['C']]     -- same as above
            my_pydf[:, [1,4,7]]   -- select columns 1,4, and 7, including all rows. Return as a pydf
            my_pydf[:, ['C','F']] -- select columns 'C' and 'F' including all rows. Return as a pydf
            my_pydf[[2,5,8], [1,4,7]]     -- select rows 2, 5, and 8, including columns 1,4, and 7.
            my_pydf[[2,5,8], ['C','F']]   -- select rows 2, 5, and 8, including columns 'C' and 'F'
            
            my_pydf[2:4]          -- select rows 2 and 3, including all columns, return as pydf.
            my_pydf[2:4, :]       -- same as above
            my_pydf[:, 3:5]       -- select columns 3 and 4, including all rows, return as pydf.
            
            new row selection using keys:
            my_pydf['row1']       -- select entire row with keyfield 'row1'.
                                        Note this differs from Pandas operation.
            my_pydf['row1','C']   -- select cell at row with keyfield 'row1' at colname 'C'
                                        Similar to dict-of-dict selection dod['row1']['C']
            my_pydf['row1':'row8']                          -- NOT SUPPORTED YET select rows including 'row1' upto but not including 'row8' (7 rows)
            my_pydf[['row1', 'row5, 'row8']]                -- select three rows by list of keyfield names.
            my_pydf['row1':'row8', ['C','F']]               -- NOT SUPPORTED YET select rows including 'row1' upto but not including 'row8' (7 rows) in columns 'C' and 'F'
            my_pydf[['row1', 'row5, 'row8'], ['C','F']]     -- select three rows by list of keyfield names.
        
            returns a consistent pydf instance copied from the original, and with the data specified.
            always returns the simplest object possible.
            if multiple rows or columns are specified, they will be returned in the original orientation.
            if only one cell is selected, return a single value.
            If only one row is selected, return a list. If a dict is required, use select_irow()
            if only one col is selected, return a list.
        """
    
        if isinstance(slice_spec, slice):
            # Handle slicing only rows
            return self._handle_slice(slice_spec, None)
            
        elif isinstance(slice_spec, int):
            # Handle indexing a single row
            irow = slice_spec
            return self._handle_slice(slice(irow, irow + 1), None)
            
        elif isinstance(slice_spec, str):
            # Handle indexing a single row using keyfield
            if not self.keyfield:
                raise RuntimeError("Use of string row spec requires keyfield defined.")
            irow = self.kd.get(slice_spec, -1)
            if irow < 0:
                raise RuntimeError("Row spec not a valid row key.")
            
            return self._handle_slice(slice(irow, irow + 1), None)
            
        elif isinstance(slice_spec, list):
            # single list of row indices
            row_indices = self._row_indices_from_rowlist(slice_spec)            
            return self._handle_slice(row_indices, None)
            
        elif isinstance(slice_spec, tuple) and len(slice_spec) == 2:
            # Handle slicing both rows and columns
            return self._handle_slice(slice_spec[0], slice_spec[1])
        else:
            raise TypeError("Unsupported indexing type. Use slices, integers, or tuples of slices and integers.")
            

    def _handle_slice(self, row_slice: Union[slice, int, None], col_slice: Union[slice, int, str, None]) -> Any:
        """
        Handles the slicing operation for rows and columns in a Pydf instance.

        Args:
        - row_slice (Union[slice, int, None]): The slice specification for rows.
        - col_slice (Union[slice, int, None]): The slice specification for columns.

        Returns:
        - Pydf: A new Pydf instance with sliced data based on the provided row and column specifications.
        or
        a single value if the resulting Pydf is only one cell.
        or a list if a single row or col results
        """
        all_cols = self.columns()
        sliced_cols = all_cols
        row_sliced_lol = self.lol   # default is no change
        
        # handle simple cases first:
        # selecting a row spec is integer.
        if isinstance(row_slice, int):
            irow = row_slice
            # select one cell.
            if isinstance(col_slice, int):
                return self.lol[irow][col_slice]
            elif isinstance(col_slice, str):
                icol = self.hd[col_slice]
                return self.lol[irow][icol]
                
            # full row    
            if col_slice is None:
                la = self.lol[irow]
                
            # part of a row, according to a slice
            elif isinstance(col_slice, slice):
                start_col, stop_col, step_col = self._parse_slice(col_slice, row_or_col='col')
                la = self.lol[irow][start_col:stop_col:step_col]
                
            # part of a row, according to a column list    
            elif isinstance(col_slice, list):
                # the following handles cases of integer list as well as str list of colnames.
                col_indices_li = self._col_indices_from_collist(col_slice)
                la = [self.lol[irow][icol] for icol in col_indices_li]
            return la

        # first handle the rows
        elif row_slice is None:
            # no specs, return the entire array.
            if col_slice is None:
                return self._adjust_return_val()   # is this correct?
            
            # all rows, one column
            elif isinstance(col_slice, int) or isinstance(col_slice, str):
                if isinstance(col_slice, str):
                    icol = self.hd[col_slice]
                else:
                    icol = col_slice
                col_la = [row[icol] for row in self.lol]
                return col_la

            # part of all rows, according to a list    
            elif isinstance(col_slice, list):    
                col_indices_li = self._col_indices_from_collist(col_slice)
                row_col_sliced_lol = [[row[i] for i in col_indices_li] for row in self.lol]
                sliced_cols = [all_cols[i] for i in col_indices_li]

            elif isinstance(col_slice, slice):
                # use normal slice approach
                start_col, stop_col, step_col = self._parse_slice(col_slice, row_or_col='col')
                row_col_sliced_lol = [[row[icol] for icol in range(start_col, stop_col, step_col)] for row in self.lol]
                sliced_cols = [all_cols[icol] for icol in range(start_col, stop_col, step_col)]
                # also respect the column names
                
            sliced_pydf = Pydf(lol=row_col_sliced_lol, cols=sliced_cols, dtypes=self.dtypes, keyfield=self.keyfield)
            return sliced_pydf._adjust_return_val()

        # sliced or listed rows, first reduce the array by rows.
        elif isinstance(row_slice, list):
            row_indices = self._row_indices_from_rowlist(row_slice)            
                
            if row_indices:
                row_sliced_lol = [self.lol[i] for i in row_indices]
            # else:
                # row_sliced_lol = self.lol
                    
        elif isinstance(row_slice, slice):    
            start_row, stop_row, step_row = self._parse_slice(row_slice)
            row_sliced_lol = self.lol[start_row:stop_row:step_row]
        # else:
            # row_sliced_lol = self.lol

        # sliced rows, all columns
        if col_slice is None:
            row_col_sliced_lol = row_sliced_lol
            
        #   one column    
        elif isinstance(col_slice, int) or isinstance(col_slice, str):
            if isinstance(col_slice, str):
                icol = self.hd[col_slice]
            else:
                icol = col_slice
                
            col_la = [row[icol] for row in row_sliced_lol]
            return col_la

        # part of all sliced rows, according to a list    
        elif isinstance(col_slice, list):    
            col_indices_li = self._col_indices_from_collist(col_slice)
            row_col_sliced_lol = [[row[i] for i in col_indices_li] for row in row_sliced_lol]
            # also respect the column names, if they are defined.
            if self.hd:
                sliced_cols = [all_cols[i] for i in col_indices_li]
            else:
                sliced_cols = None            

        elif isinstance(col_slice, slice):
                           
            # use normal slice approach
            start_col, stop_col, step_col = self._parse_slice(col_slice, row_or_col='col')
            row_col_sliced_lol = [[row[icol] for icol in range(start_col, stop_col, step_col)] for row in row_sliced_lol]
            # also respect the column names, if they are defined.
            if self.hd:
                sliced_cols = [self.columns()[icol] for icol in range(start_col, stop_col, step_col)]
            else:
                sliced_cols = None            

        sliced_pydf = Pydf(lol=row_col_sliced_lol, cols=sliced_cols, dtypes=self.dtypes, keyfield=self.keyfield)
        return sliced_pydf._adjust_return_val()


    def _adjust_return_val(self):
        # not sure if this is the correct way to go!
        # particularly with zero copy, this may not be correct.
        #
        # alternative is to use .tolist() and .todict()
        # if the method returns a pydf.
        #
        # @@TODO: Must be fixed for zero_copy
        
        num_cols = self._num_cols()
        num_rows = len(self.lol)
    
        if num_rows == 1 and num_cols == 1:
            # single value, just return it.
            return self.lol[0][0]
            
        elif num_rows == 1 and num_cols > 1:
            # single row, return as list.
            return self.lol[0]
                
        elif num_rows > 1 and num_cols == 1:
            # single column result as a list.
            return self.icol(0)
            
        return self
        

    def _col_indices_from_collist(self, collist) -> T_li:  # col_indices
    
        if not collist:
            return []
    
        first_item = collist[0]
        if isinstance(first_item, str):             # probably list of column names (did not check them all)
            colnames = collist
            col_indices = [self.hd[col] for col in colnames]
        elif isinstance(first_item, int):           # probably list of column indices (did not check them all)
            col_indices = collist
        else:
            raise ValueError("column slice, if a list, must be a list of strings or ints")
        return col_indices
        

    def _row_indices_from_rowlist(self, rowlist) -> T_li: # row_indices
        if not rowlist:
            return []
            
        first_item = rowlist[0]
        if isinstance(first_item, str):
            if not self.keyfield:
                raise RuntimeError("keyfield must be defined to use str row keys")
            row_indices = [self.kd.get(rowkey, -1) for rowkey in rowlist]
        elif isinstance(first_item, int):
            row_indices = rowlist
        else:
            raise RuntimeError("list spec must use str or int values")
            
        return row_indices

    def _parse_slice(self, s: Union[slice, int, None], row_or_col:str='row') -> Tuple[Optional[int], Optional[int], Optional[int]]:
        if isinstance(s, slice):
            start = s.start if s.start is not None else 0
            
            stop = s.stop if s.stop is not None else (self._num_cols() if row_or_col == 'col' else len(self.lol))
            
            step = s.step if s.step is not None else 1
            return start, stop, step
        elif isinstance(s, int):
            return s, s + 1, 1
        elif s is None:
            return None, None, None
        

    def __setitem__(self, slice_spec: Union[int, Tuple[Union[slice, int], Union[slice, int, str]]], value: Any):
        """
        Handles the assignment of values, lists or dicts to Pydf elements.
        
        Can handle the following scenarios:
            my_pydf[3] = list                   -- assign the entire row at index 3 to the list provided
            my_pydf[[3]] = list                 -- same as above
            my_pydf[3, :] = list                -- same as above.
            my_pydf[[3], :] = list              -- same as above.
            my_pydf[3, :] = dict                -- assign the entire row at index 3 to the dict provided, respecting dict keys.
                                                    will only assign those values that have a new value in the provided dict.
            my_pydf[3] = value                  -- assign the entire row at index 3 to the value provided.
            my_pydf[[3]] = value                -- same as above.
            my_pydf[3, :] = value               -- same as above.
            my_pydf[[3], :] = value             -- same as above.
            my_pydf[[3, 5, 8], :] = value       -- set rows 3, 5 and 8 to the value in all columns
            my_pydf[3, 4] = value               -- set one cell 3, 4 to value
            my_pydf[3, 5:20] = value            -- set a single value in row 3, columns 5 through 19
            my_pydf[3, 5:20] = list             -- set row 3 in columns 5 through 19 to values from the the list provided.
            my_pydf[3, [1,4,7]] = list          -- set row 3 in columns 1, 4, and 7 to values from the the list provided.
            my_pydf[:, 4] = list                -- assign the entire column at index 4 to the list provided.
            my_pydf[3:5, 5] = list              -- assign a partial column at index 5 at rows 3 and 4 to the list provided.

            my_pydf[3, 'C'] = value             -- set a value in cell 3, col 'C'
            my_pydf[3, ['C', 'D', 'G'] = list   -- set a row 3 in columns 'C', 'D' and 'G' to the values in list.
            my_pydf[:, 'C'] = list              -- assign the entire column 'C' to the list provided
            my_pydf[3:5, 'C'] = list            -- assign a partial column 'C' to list provided in rows 3 and 4
            my_pydf[:, 4] = value               -- assign the entire column 4 to the value provided.
            
            my_pydf[[3,5,8], :] = pydf          -- set rows 3,5,8 to the data in pydf provided
            my_pydf[[3,5,8], ['C', 'D', 'G']] = pydf   -- set rows 3,5,8 in columns 'C', 'D' and 'G' to the data in pydf provided
            my_pydf['R1']                       -- choose entire row by name
            my_pydf['R1', :]                    -- choose entire row by name
            my_pydf[['R1', 'R2', 'R3']]         -- choose three rows
            my_pydf[['R1', 'R2', 'R3'], :]      -- choose three rows

        Args:
        - slice_spec: The slice_spec (index or slice) indicating the location to assign the value.
        - value: The value to assign.

        Returns:
        - None
        """
        if isinstance(slice_spec, int):
            irow = slice_spec
            # Assigning a row based on a single integer index and a value which is a list.
            # will trigger an error if the list is not the right length for the row.
            if isinstance(value, list):
                self.lol[irow] = value
                
            # if value is a dict, use it and make sure the right columns are assigned per dict keys.    
            elif isinstance(value, dict):
                self.assign_record_da_irow(irow, record_da=value)
            else:
                # set the same value in the row for all columns.
                self.lol[irow] = [value] * len(self.lol[irow])
            
        elif isinstance(slice_spec, list) and isinstance(value, 'Pydf'):
            # assign a number of rows to the data in pydf provided.
            row_indices = self._row_indices_from_rowlist(slice_spec)
            for source_row, irow in enumerate(row_indices):
                self.lol[irow] = value.lol[source_row]

        elif isinstance(slice_spec, tuple):
            row_spec, col_spec = slice_spec
            if isinstance(row_spec, int) or isinstance(row_spec, str):
                if isinstance(row_spec, str) and self.keyfield:
                    irow = self.kd[row_spec]
                else:
                    irow = row_spec
                    
                if isinstance(col_spec, int):
                    # my_pydf[irow, icol] = value       -- set cell irow, icol to value, where irow, icol are integers.
                    self.lol[irow][col_spec] = value
                    
                elif isinstance(col_spec, str):
                    # my_pydf[irow, colname] = value    -- set a value in cell irow, col, where colname is a string.
                    icol = self.hd[col_spec]
                    self.lol[irow][icol] = value
                    
                elif isinstance(col_spec, list) and col_spec:
                    col_indices = self._col_indices_from_collist(col_spec)
                        
                    # assign a number of columns specified in a list of colnames to a single row, from a list with only those columns.
                    for source_col, icol in enumerate(col_indices):
                        self.lol[irow][icol] = value[source_col]                
            
                elif isinstance(col_spec, slice):
                    # my_pydf[irow, start:end] = value  -- set a value in cells in row irow, from columns start to end.
                    # my_pydf[irow, start:end] = list   -- set values from a list in cells in row irow, from columns start to end.
                    col_start, col_stop, col_step = self._parse_slice(col_spec)
                    for idx, icol in enumerate(range(col_start, col_stop, col_step)):   # type: ignore
                        if isinstance(value, list):
                            self.lol[irow][icol] = value[idx]
                        else:
                            self.lol[irow][icol] = value
                            
            elif isinstance(row_spec, slice):
                row_start, row_stop, row_step = self._parse_slice(row_spec)
                
                if isinstance(col_spec, list) and col_spec:
                    col_indices = self._col_indices_from_collist(col_spec)
                         
                    for source_row, irow in enumerate(range(row_start, row_stop, row_step)):
                        for source_col, icol in enumerate(col_indices):
                            self.lol[irow][icol] = value.lol[source_row][source_col]
                        
                
                elif isinstance(col_spec, int) or isinstance(col_spec, str):
                    if isinstance(col_spec, str):
                        icol = self.hd[col_spec]
                    else:
                        icol = col_spec
                
                    for idx, irow in enumerate(range(row_start, row_stop, row_step)):       # type: ignore
                        if isinstance(value, list):
                            self.lol[irow][icol] = value[idx]
                        else:
                            self.lol[irow][icol] = value
                            
            elif isinstance(row_spec, str) and self.keyfield:
                irow = self.kd[row_spec]
                
                if isinstance(col_spec, list) and col_spec:
                    col_indices = self._col_indices_from_collist(col_spec)

                    for source_row, irow in enumerate(row_indices):
                        for source_col, icol in enumerate(col_indices):
                            self.lol[irow][icol] = value.lol[source_row][source_col]
                            
                elif isinstance(col_spec, int) or isinstance(col_spec, str):
                    if isinstance(col_spec, str):
                        icol = self.hd[col_spec]
                    else:
                        icol = col_spec
                
                    for idx, irow in enumerate(range(row_start, row_stop, row_step)):       # type: ignore
                        if isinstance(value, list):
                            self.lol[irow][icol] = value[idx]
                        else:
                            self.lol[irow][icol] = value
                            
                
            elif isinstance(row_spec, list):
                row_indices = self._row_indices_from_rowlist(row_spec)
                
                if isinstance(col_spec, list) and col_spec:
                    col_indices = self._col_indices_from_collist(col_spec)

                    for source_row, irow in enumerate(row_indices):
                        for source_col, icol in enumerate(col_indices):
                            self.lol[irow][icol] = value.lol[source_row][source_col]
                        
                
                elif isinstance(col_spec, int) or isinstance(col_spec, str):
                    if isinstance(col_spec, str):
                        icol = self.hd[col_spec]
                    else:
                        icol = col_spec
                
                    for idx, irow in enumerate(row_indices): 
                        if isinstance(value, list):
                            self.lol[irow][icol] = value[idx]
                        else:
                            self.lol[irow][icol] = value
            else:
                raise ValueError("Unsupported key type for assignment")

            
        else:
            raise ValueError("Unsupported key type for assignment")

        self._rebuild_kd()    
        
    #===========================
    # initializers
    
    def clone_empty(self) -> 'Pydf':
        """ Create Pydf instance from pydf, adopting dict keys as column names
            adopts keyfield but does not adopt kd.
            test exists in test_pydf.py            
         """
        if self is None:
            return Pydf()
        
        return Pydf(cols=self.columns(), keyfield=self.keyfield, dtypes=copy.deepcopy(self.dtypes))
        
        
    def set_lol(self, new_lol: T_lola):
        """ set the lol with the value passed, leaving other settings, but recalculating kd. 
        """
        
        self.lol = new_lol
        self._rebuild_kd()
        

    #===========================
    # convert from
    
    @staticmethod
    def from_lod(
            records_lod:    T_loda,                         # List[List[Any]] to initialize the lol data array.
            keyfield:       str='',                         # set a new keyfield or set no keyfield.
            dtypes:         Optional[T_dtype_dict]=None     # set the data types for each column.
            ) -> 'Pydf':
        """ Create Pydf instance from loda type, adopting dict keys as column names
            Generally, all dicts in records_lod should be the same OR the first one must have all keys
                and others can be missing keys.
        
            test exists in test_pydf.py
            
            my_pydf = Pydf.from_lod(sample_lod)
        """
        if dtypes is None:
            dtypes = {}
        
        if not records_lod:
            return Pydf(keyfield=keyfield, dtypes=dtypes)
        
        cols = list(records_lod[0].keys())
        
        # from utilities import utils
        
        lol = [list(utils.set_cols_da(record_da, cols).values()) for record_da in records_lod]
        
        return Pydf(cols=cols, lol=lol, keyfield=keyfield, dtypes=dtypes)
        
        
    @staticmethod
    def from_dod(
            dod:            T_doda,         # Dict(str, Dict(str, Any))
            keyfield:       str='rowkey',   # The keyfield will be set to the keys of the outer dict.
                                            # this will set the preferred name. Defaults to 'rowkey'
            dtypes:         Optional[T_dtype_dict]=None     # optionally set the data types for each column.
            ) -> 'Pydf':
        """ a dict of dict structure is very similar to a Pydf table, but there is a slight difference.
            a Pydf table always has the keys of the outer dict as items in each table.
            Thus dod1 = {'row_0': {'rowkey': 'row_0', 'data1': 1, 'data2': 2, ... },
                         'row_1': {'rowkey': 'row_1', 'data1': 11, ... },
                         ...
                         }
            is fully compatible because it has a first item which is the rowkey.
            If a dod is passed that does not have this column, then it will be created.
            The 'keyfield' parameter should be set to the name of this column.
            
            A typical dod does not have the row key as part of the data in each row, such as:
            
             dod2 = {'row_0': {'data1': 1, 'data2': 2, ... },
                     'row_1': {'data1': 11, ... },
                     ...
                    }
            
            If dod2 is passed, it will be convered to dod1 and then converted to pydf instance.
            
            
            A Pydf table is able 1/3 the size of an equivalent dod. because the column keys are not repeated.
            
            use to_dod() to recover the original form by setting 'remove_rowkeys'=True if the row keys are
            not required in the dod.
            
        """
        return Pydf.from_lod(utils.dod_to_lod(dod, keyfield=keyfield), dtypes=dtypes)
    
    
    @staticmethod
    def from_cols_dol(cols_dol: T_dola, keyfield: str='', dtypes: Optional[T_dtype_dict]=None) -> 'Pydf':
        """ Create Pydf instance from cols_dol type, adopting dict keys as column names
            and creating columns from each value (list)
            
            my_pydf = Pydf.from_cols_dol({'A': [1,2,3], 'B': [4,5,6], 'C': [7,8,9])
            
            produces:
                my_pydf.columns() == ['A', 'B', 'C']
                my_pydf.lol == [[1,4,7], [2,5,8], [3,6,9]] 
            
            
        """
        if dtypes is None:
            dtypes = {}
        
        if not cols_dol:
            return Pydf(keyfield=keyfield, dtypes=dtypes)
        
        cols = list(cols_dol.keys())
        
        lol = []
        for irow in range(len(cols_dol[cols[0]])):
            row = []
            for col in cols:
                row.append(cols_dol[col][irow])
            lol.append(row)    
        
        return Pydf(cols=cols, lol=lol, keyfield=keyfield, dtypes=dtypes)


    @staticmethod
    def from_excel_buff(
            excel_buff: bytes, 
            keyfield: str='',                       # field to use as unique key, if not ''
            dtypes: Optional[T_dtype_dict]=None,    # dictionary of types to apply if set.
            noheader: bool=False,                   # if True, do not try to initialize columns in header dict.
            user_format: bool=False,                # if True, preprocess the file and omit comment lines.
            unflatten: bool=True,                   # unflatten fields that are defined as dict or list.
            ) -> 'Pydf':
        """ read excel file from a buffer and convert to pydf.
        """
        
        # from utilities import xlsx_utils

        csv_buff = utils.xlsx_to_csv(excel_buff)

        my_pydf  = Pydf.from_csv_buff(
                        csv_buff, 
                        keyfield    = keyfield,         # field to use as unique key, if not ''
                        dtypes      = dtypes,           # dictionary of types to apply if set.
                        noheader    = noheader,         # if True, do not try to initialize columns in header dict.
                        user_format = user_format,      # if True, preprocess the file and omit comment lines.
                        unflatten   = unflatten,        # unflatten fields that are defined as dict or list.
                        )
        
        return my_pydf
    

    @staticmethod
    def from_csv_buff(
            csv_buff: Union[bytes, str],            # The CSV data as bytes or string.
            keyfield: str='',                       # field to use as unique key, if not ''
            dtypes: Optional[T_dtype_dict]=None,    # dictionary of types to apply if set.
            noheader: bool=False,                   # if True, do not try to initialize columns in header dict.
            user_format: bool=False,                # if True, preprocess the file and omit comment lines.
            sep: str=',',                           # field separator.
            unflatten: bool=True,                   # unflatten fields that are defined as dict or list.
            ) -> 'Pydf':
        """
        Convert CSV data in a buffer (bytes or string) to a pydf object

        Args:
            buff (Union[bytes, str]): 
            keyfield: field to use as unique key, if not ''
            dtypes: dictionary of types to apply if set.
            noheader: do not initialize columns from the first (non-comment) line.
            user_format (bool): Whether to preprocess the CSV data (remove comments and blank lines).
            sep (str): The separator used in the CSV data.

        Returns:
            hllola: A tuple containing a header list and a list of lists representing the CSV data.
        """
        
        from models.DB import DB
    
        data_lol = DB.buff_csv_to_lol(csv_buff, user_format=user_format, sep=sep)
        
        cols = []
        if not noheader:
            cols = data_lol.pop(0)        # return the first item and shorten the list.
        
        my_pydf = Pydf(lol=data_lol, cols=cols, keyfield=keyfield, dtypes=dtypes)
        
        if unflatten:
            my_pydf.unflatten_by_dtypes()
   
        return my_pydf
    

    @staticmethod
    def from_lod_to_cols(lod: T_loda, cols:Optional[List]=None, keyfield: str='', dtypes: Optional[T_dtype_dict]=None) -> 'Pydf':
        r""" Create Pydf instance from a list of dictionaries to be placed in columns
            where each column shares the same keys in the first column of the array.
            This transposes the data from rows to columns and adds the new 'cols' header,
            while adopting the keys as the keyfield. dtypes is applied to the columns
            transposition and then to the rows.
            
            If no 'cols' parameter is provided, then it will be the name 'key' 
            followed by normal spreadsheet column names, like 'A', 'B', ... 
            
            Creates a pydf where the first column are the keys from the dicts,
            and each subsequent column are each of the values of the dicts.
            
            my_pydf = Pydf.from_coldicts_lod( 
                cols = ['Feature', 'Try 1', 'Try 2', 'Try 3'],
                lod =       [{'A': 1, 'B': 2, 'C': 3},          # data for Try 1
                             {'A': 4, 'B': 5, 'C': 6},          # data for Try 2
                             {'A': 7, 'B': 8, 'C': 9} ]         # data for Try 3
            
            produces:
                my_pydf.columns() == ['Feature', 'Try 1', 'Try 2', 'Try 3']
                my_pydf.lol ==        [['A',       1,       4,       7], 
                                             ['B',       2,       5,       8], 
                                             ['C',       3,       6,       9]] 
            
            This format is useful for producing reports of several tries 
            with different values for the same attributes placed in columns,
            particularly when there are many features that need to be compared.
            Columns are defined directly from cols parameter.
            
        """
        if dtypes is None:
            dtypes = {}
            
        if cols is None:
            cols = []
        
        if not lod:
            return Pydf(keyfield=keyfield, dtypes=dtypes, cols=cols)
        
        # the following will adopt the dictionary keys as cols.
        # note that dtypes applies to the columns in this orientation.
        rows_pydf = Pydf.from_lod(lod, dtypes=dtypes)
        
        # this transposes the entire dataframe, including the column names, which become the first column
        # in the new orientation, then adds the new column names, if provided. Otherwise they will be
        # defined as ['key', 'A', 'B', ...]
        cols_pydf = rows_pydf.transpose(new_keyfield = keyfield, new_cols = cols, include_header = True)
        
        return cols_pydf

    
    @staticmethod
    def from_pandas_df(df: T_df, keyfield:str='', name:str='', use_csv:bool=False, dtypes: Optional[T_dtype_dict]=None) -> 'Pydf':
        """
        Convert a Pandas dataframe to pydf object
        """
        import pandas as pd     # type: ignore
        
        if not use_csv:
    
            if isinstance(df, pd.Series):
                rowdict = df.to_dict()
                cols = list(rowdict.keys())
                lol = [list(rowdict.values())]
            else:
                cols = list(df.columns)
                lol = df.values.tolist()

            return Pydf(cols=cols, lol=lol, keyfield=keyfield, name=name, dtypes=df.dtypes.to_dict())
            
        # first convert the Pandas df to a csv buffer.
        try:
            csv_buff = df.to_csv(None, index=False, quoting=csv.QUOTE_MINIMAL, lineterminator= '\r\n')
        except TypeError:
            csv_buff = df.to_csv(None, index=False, quoting=csv.QUOTE_MINIMAL, line_terminator= '\r\n')
            
        return Pydf.from_csv_buff(
            csv_buff=csv_buff,
            keyfield=keyfield,
            dtypes=dtypes,    
            unflatten=False,  
            )
            

    @staticmethod
    def from_numpy(npa: Any, keyfield:str='', cols:Optional[T_la]=None, name:str='') -> 'Pydf':
        """
        Convert a Numpy dataframe to pydf object
        The resulting Python list will contain Python native types, not NumPy types. 
        """
        # import numpy as np
        
        lol = npa.tolist()
    
        return Pydf(cols=cols, lol=lol, keyfield=keyfield, name=name)
    

    @staticmethod
    def from_hllola(hllol: T_hllola, keyfield: str='', dtypes: Optional[T_dtype_dict]=None):
        """ Create Pydf instance from hllola type.
            This is used for all DB. loading.
            test exists in test_pydf.py
        """
        
        hl, lol = hllol
        
        return Pydf(lol=lol, cols=hl, keyfield=keyfield, dtypes=dtypes)


    #===========================
    # convert to other format
    
    def to_csv_buff(
            self, 
            line_terminator: Optional[str]=None,
            include_header: bool=True,
            ) -> T_buff:
    
        if not self:
            return ''
        
        if line_terminator is None:
            line_terminator = '\r\n'
    
        f = io.StringIO(newline = '')           # Use newline='' to ensure consistent line endings
        
        csv_writer = csv.writer(f, lineterminator=line_terminator)
        if include_header:
            csv_writer.writerow(self.columns())     # Write the header row
        csv_writer.writerows(self.lol)          # Write the data rows

        buff = f.getvalue()
        f.close()

        return buff   


    def to_dict(self) -> dict:
        """
        Convert Pydf instance to a dictionary representation.
        The dictionary has two keys: 'cols' and 'lol'.

        Example:
        {
            'cols': ['A', 'B', 'C'],
            'lol': [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
        }
        """
        return {'cols': self.columns(), 'lol': self.lol}

    
    def to_lod(self) -> T_loda:
        """ Create lod from pydf
            test exists in test_pydf.py
        """
        
        if not self:
            return []

        cols = self.columns()
        result_lod = [dict(zip(cols, la)) for la in self.lol]
        return result_lod
        
    
    def to_dod(
            self,
            dod:                T_doda,         # Dict(str, Dict(str, Any))
            remove_keyfield:    bool=True,      # by default, the keyfield column is removed.
            ) -> T_doda:
        """ a dict of dict structure is very similar to a Pydf table, but there is a slight difference.
            a Pydf table always has the keys of the outer dict as items in each table.
            Thus dod1 = {'row_0': {'rowkey': 'row_0', 'data1': 1, 'data2': 2, ... },
                         'row_1': {'rowkey': 'row_1', 'data1': 11, ... },
                         ...
                         }
            If a dod is passed that does not have this column, then it will be created.
            The 'keyfield' parameter should be set to the name of this column.
            
            A typical dod does not have the row key as part of the data in each row, such as:
            
             dod2 = {'row_0': {'data1': 1, 'data2': 2, ... },
                     'row_1': {'data1': 11, ... },
                     ...
                    }
            
            If remove_keyfield=True (default) dod2 will be produced, else dod1.
                        
        """
        return utils.lod_to_dod(self.to_lod(), keyfield=self.keyfield, remove_keyfield=remove_keyfield)
    
    
    def to_pandas_df(self, use_csv: bool=False) -> Any:
    
        import pandas as pd     # type: ignore
    
        if not use_csv:
            columns = self.columns()
            # return pd.DataFrame(self.lol, columns=columns, dtypes=self.dtypes)
            # above results in NotImplementedError: compound dtypes are not implemented in the DataFrame constructor

            return pd.DataFrame(self.lol, columns=columns)
            
        else:
            # it seems this may work faster if we first convert the data to a csv_buff internally,
            # and then convert that to a df.
        
            csv_buff = self.to_csv_buff()
            sio = io.StringIO(csv_buff)            
            df  = pd.read_csv(sio, 
                na_filter=False, 
                index_col=False, 
                #dtype=self.dtypes,
                #sep=sep,
                usecols=None,
                #comment='#', 
                #skip_blank_lines=True
                )
        
            return df
            
            

    def to_numpy(self, dtypes:Optional[Type]=None) -> Any:
        """ 
        Convert the core array of a Pydf object to numpy.
        Note: does not convert any column names if they exist.
        Keyfield lookups are lost, if they are defined.
        
        When you create a NumPy array with a specific data type 
        (e.g., int32, float64), NumPy will attempt to coerce or 
        cast the elements to the specified data type. The rules 
        for type casting follow a hierarchy where more general 
        types are converted to more specific types.
        
        examples:
           if dtype is np.int32 and there are some float values, they will be truncated.
           if dtype is np.int32 and there are some string values, they will be converted to an integer if possible.
           if dtype is float64 and there are some integer values, they will be csst to float64 type.
           if casting is not possible, it will raise an error.
        
        """
    
        import numpy as np
        return np.array(self.lol)
        

    def to_hllola(self) -> T_hllola:
        """ Create hllola from pydf 
            test exists in test_pydf.py
        """    
        return (list(self.hd.keys()), self.lol)
        
    #===========================
    # append
        
    def append(self, data_item: Union[T_Pydf, T_loda, T_da, T_la]):
        """ general append method can handle appending one record as T_da or T_la, many records as T_loda or T_pydf
        """
        # test exists in test_pydf.py for all three cases
        
        if not data_item:
            return self
        
        if isinstance(data_item, dict):
            self.record_append(data_item)
            
        elif isinstance(data_item, list):
            if isinstance(data_item[0], dict):
                # lod type
                self.extend(data_item)
            else:
                # simple list.
                if self.hd:
                    # columns are defined, and keyfield might also be defined
                    # create a dict.
                    da = dict(zip(self.hd.keys(), data_item))
                    self.record_append(da)
                else:    
                    # no columns defined, therefore just append to lol.
                    self.lol.append(data_item)
                
        elif isinstance(data_item, Pydf):  # type: ignore
            self.concat(data_item)
        else:    
            raise RuntimeError
        

    def concat(self, other_instance: 'Pydf'):
        """ concatenate records from passed pydf cls to self pydf 
            This directly modifies self
            if keyfield is '', then insert without respect to the key value
            otherwise, allow only one record per key.
            columns must be equal.
            test exists in test_pydf.py
        """
        
        if not other_instance:
            return 
            
        diagnose = False

        if diagnose:
            print(f"self=\n{self}\npydf=\n{other_instance}")
            
        if not self.lol and not self.hd:
            # new pydf, passed pydf
            # but there is only one header and data is lol
            # this saves space.
            
            self.hd = other_instance.hd
            self.lol = other_instance.lol
            self.kd = other_instance.kd
            self.keyfield = other_instance.keyfield
            self._rebuild_kd()   # only if the keyfield is set.
            return 
            
        # fields must match exactly!
        if self.hd != other_instance.hd:
            raise KeyError ("keys in pydf do not match lod keys")
        
        # simply append the rows from pydf.lol to the end of self.lol
        for idx in range(len(other_instance.lol)):
            rec_la = other_instance.lol[idx]
            self.lol.append(rec_la)
        self._rebuild_kd()   # only if the keyfield is set.

        if diagnose:
            print(f"result=\n{self}")
                

    def extend(self, records_lod: T_loda):
        """ append lod of records into pydf 
            This directly modifies pydf
            if keyfield is '', then insert without respect to the key value
            otherwise, allow only one record per key.
            test exists in test_pydf.py
         """
        
        if not records_lod or len(records_lod) == 1 and not records_lod[0]:
            return
            
        if not self.lol and not self.hd:
            # new pydf, adopt structure of lod.
            # but there is only one header and data is lol
            # this saves space.
            
            self.hd = {col_name: index for index, col_name in enumerate(records_lod[0].keys())}
            self.lol = [list(record_da.values()) for record_da in records_lod]
            self._rebuild_kd()   # only if the keyfield is set.
            return
            
        for record_da in records_lod:
            if not record_da:
                # do not append any records that are empty.
                continue
            
            # the following will either append or insert
            # depending on the keyvalue.
            self.record_append(record_da)    
            
            

    def record_append(self, record_da: T_da):
        """ perform append of one record into pydf (T_da is Dict[str, Any]) 
            This directly modifies pydf
            if keyfield is '', then insert without respect to the key value
            otherwise, allow only one record per key.
            
            if the pydf is empty, it will adopt the structure of record_da.
            Each new append will add to the end of the pydf.lol and will
            update the kd.
            
            If the keys in the record_da have a different order, they will
            be reordered and then appended correctly.
        """
            # test exists in test_pydf.py
        
        if not record_da:
            return
            
        if not self.lol and not self.hd:
            # new pydf, adopt structure of lod.
            # but there is only one header and data is lol
            # this saves space.
            
            self.hd = {col_name: index for index, col_name in enumerate(record_da.keys())}
            self.lol = [list(record_da.values())]
            self._rebuild_kd()   # only if the keyfield is set.
            return
            
        # check if fields match exactly.
        reorder = False
        if list(self.hd.keys()) != list(record_da.keys()):
            reorder = True
        
        if reorder:
            # construct a dict with exactly the cols specified.
            # defaults to '' at this point.
            rec_la = [record_da.get(col, '') for col in self.hd]
        else:
            rec_la = list(record_da.values())
            
        if self.keyfield:
            # insert will overwrite any existing key with the same value.
            keyval = record_da[self.keyfield]
            idx = self.kd.get(keyval, -1)
            if idx >= 0:
                self.lol[idx] = rec_la
            else:
                self._basic_append_la(rec_la, keyval)
        else:
            # no keyfield is set, just append to the end.
            self.lol.append(rec_la)


    def _basic_append_la(self, rec_la: T_la, keyval: str):
        """ basic append to the end of the array without any checks
            including appending to kd and la to lol
        """
        self.kd[keyval] = len(self.lol)
        self.lol.append(rec_la)
                

    #=========================
    # remove records per keyfield; drop cols

    def remove_key(self, keyval:str, silent_error=True) -> None:
        """ remove record from pydf using keyfield
            This directly modifies pydf
        """
        # test exists in test_pydf.py

        if not self.keyfield:
            return
        
        try:
            key_idx = self.kd[keyval]   #will raise KeyError if key not exists.
        except KeyError:
            if silent_error: return
            raise
            
        self.lol.pop(key_idx)
        self._rebuild_kd()
        return
        
    
    def remove_keylist(self, keylist: T_ls, silent_error=True):
        """ remove records from pydf using keyfields
            This directly modifies pydf
            test exists in test_pydf.py
        """

        # get the indexes of rows to be deleted.
        idx_li: T_li = []
        for keyval in keylist:
            try:
                idx = self.kd[keyval]
            except KeyError:
                if silent_error: continue
                raise
            idx_li.append(idx)    
                
        # delete records from the end so the indexes are valid after each deletion.
        reverse_sorted_idx_li = sorted(idx_li, reverse=True)

        for idx in reverse_sorted_idx_li:
            self.lol.pop(idx)

        self._rebuild_kd()
        
        
    def get_existing_keys(self, keylist: T_ls) -> T_ls:
        """ check the keylist against the keys defined in a pydf instance. 
        """
    
        return [key for key in keylist if key in self.kd]


    #=========================
    #   selecting
    
    def select_record_da(self, key: str) -> T_da:
        """ Select one record from pydf using the key and return as a single T_da dict.
            test exists in test_pydf.py
        """
        
        if not self.keyfield:
            raise RuntimeError
            
        row_idx = self.kd.get(key, -1)
        if row_idx < 0:
            return {}
        if row_idx >= len(self.lol):
            return {}
        record_da = dict(zip(self.hd, self.lol[row_idx]))
        return record_da
        
        
    def select_irows(self, irows_li: T_li) -> 'Pydf':
        """ Select multiple records from pydf using row indexes and create new pydf.
            
        """
        
        selected_pydf = self.clone_empty()
        
        for row_idx in irows_li:
            record_da = dict(zip(self.hd, self.lol[row_idx]))
        
            selected_pydf.append(record_da)
                      
        return selected_pydf
        
        
    def select_records_pydf(self, keys_ls: T_ls, inverse:bool=False) -> 'Pydf':
        """ Select multiple records from pydf using the keys and return as a single pydf.
            If inverse is true, select records that are not included in the keys.
            
        """
        #    unit tests exist but not for inverse yet.
        
        if not self.keyfield:
            raise RuntimeError
            
        # determine the rows selected.
        if not inverse:
            selected_irows = [self.kd[key] for key in keys_ls if key in self.kd]    
        else:
            if len(keys_ls) > 10 and len(self.kd) > 30:
                keys_d = dict.fromkeys(keys_ls)     # create a dictionary for fast lookup.
                selected_irows = [self.kd[key] for key in self.kd if key not in keys_d]    
            else:
                # short lists work just fine for fast lookups.
                selected_irows = [self.kd[key] for key in self.kd if key not in keys_ls]    

        
        return self.select_irows(selected_irows) 
        
        
    def irow(self, irow: int, include_cols: Optional[T_ls]=None) -> T_da:
        """ alias for iloc 
            test exists in test_pydf.py
        """
        return self.iloc(irow, include_cols)
        

    def iloc(self, irow: int, include_cols: Optional[T_ls]=None) -> T_da:
        """ Select one record from pydf using the idx and return as a single T_da dict
            test exists in test_pydf.py
        """
        
        if irow < 0 or irow >= len(self.lol) or not self.lol or not self.lol[irow]:
            return {}
            
        if self.hd: 
            if not include_cols:    
                return dict(zip(self.hd, self.lol[irow]))
            else:
                return {col:self.lol[irow][self.hd[col]] for col in include_cols if col in self.hd}
                
        colnames = Pydf._generate_spreadsheet_column_names_list(num_columns=len(self.lol[irow]))
        return dict(zip(colnames, self.lol[irow]))
        

    def select_by_dict_to_lod(self, selector_da: T_da, expectmax: int=-1, inverse: bool=False) -> T_loda:
        """ Select rows in pydf which match the fields specified in d, returning lod 
            test exists in test_pydf.py
        """

        # from utilities import utils

        result_lod = [d2 for d2 in self if inverse ^ utils.is_d1_in_d2(d1=selector_da, d2=d2)]
    
        if expectmax != -1 and len(result_lod) > expectmax:
            raise LookupError
            # import pdb; pdb.set_trace() #perm
            # pass
        
        return result_lod


    def select_by_dict(self, selector_da: T_da, expectmax: int=-1, inverse:bool=False, keyfield:str='') -> 'Pydf':
        """ Selects rows in pydf which match the fields specified in d
            and return new pydf, with keyfield set according to 'keyfield' argument.
            test exists in test_pydf.py
        """

        # from utilities import utils

        result_lol = [list(d2.values()) for d2 in self if inverse ^ utils.is_d1_in_d2(d1=selector_da, d2=d2)]
    
        if expectmax != -1 and len(result_lol) > expectmax:
            raise LookupError
            # import pdb; pdb.set_trace() #perm
            # pass
            
        new_keyfield = keyfield or self.keyfield
        
        pydf = Pydf(cols=self.columns(), lol=result_lol, keyfield=new_keyfield, dtypes=self.dtypes)
        
        return pydf
        
        
    def select_first_row_by_dict(self, selector_da: T_da, inverse:bool=False) -> T_da:
        """ Selects the first row in pydf which matches the fields specified in selector_da
            and returns that row. Else returns {}.
            Use inverse to find the first row that does not match.
        """
            
        # test exists in test_pydf.py

        # from utilities import utils

        for d2 in self:
            if inverse ^ utils.is_d1_in_d2(d1=selector_da, d2=d2):
                return d2

        return {}


    def select_where(self, where: Callable) -> 'Pydf':
        """
        Select rows in Pydf based on the provided where condition
        if provided as a string, the variable 'row' is the current row being evaluated.
        if a callable function, then it is passed the row.

        # Example Usage
        
            result_pydf = original_pydf.select_where(lambda row: bool(int(row['colname']) > 5))
        
        """
        result_lol = [list(row.values()) for row in self if where(row)]

        pydf = Pydf(cols=self.columns(), lol=result_lol, keyfield=self.keyfield, dtypes=self.dtypes)

        return pydf    
        

    def select_where_idxs(self, where: Callable) -> T_li:
        """
        Select rows in Pydf based on the provided where condition
        variable 'row' is the current row being evaluated
        and return list of indexes.

        # Examle Usage
            result_pydf = original_pydf.select_where("int(row['colname']) > 5")
        
        """
        return [idx for idx, row in enumerate(self) if where(row)]


    def col(self, colname: str, unique: bool=False, omit_nulls: bool=False, silent_error:bool=False) -> list:
        """ alias for col_to_la()
            can also use column ranges and then transpose()
            test exists in test_pydf.py
        """
        return self.col_to_la(colname, unique, omit_nulls=omit_nulls, silent_error=silent_error)


    def col_to_la(self, colname: str, unique: bool=False, omit_nulls: bool=False, silent_error:bool=False) -> list:
        """ pull out out a column from pydf by colname as a list of any
            does not modify pydf. Using unique requires that the 
            values in the column are hashable.
            test exists in test_pydf.py
        """
        
        if not colname:
            raise RuntimeError("colname is required.")
        if colname not in self.hd:
            if silent_error:
                return []
            raise RuntimeError(f"colname {colname} not defined in this pydf. Use silent_error to return [] in this case.")

        icol = self.hd[colname]
        result_la = self.icol_to_la(icol, unique=unique, omit_nulls=omit_nulls)
        
        return result_la

        
    def icol(self, icol: int) -> list:
        return self.icol_to_la(icol)


    def icol_to_la(self, icol: int, unique: bool=False, omit_nulls: bool=False) -> list:
        """ pull out out a column from pydf by icol idx as a list of any 
            can also use column ranges and then transpose()
            does not modify pydf
            test exists in test_pydf.py
        """
        
        if icol < 0 or not self or icol >= self._num_cols():
            return []
        
        if omit_nulls:
            result_la = [la[icol] for la in self.lol if la[icol]]
        else:
            result_la = [la[icol] for la in self.lol]

        if unique:
            result_la = list(dict.fromkeys(result_la))
            
        return result_la
            
    
    def drop_cols(self, exclude_cols: Optional[T_ls]=None):
        """ given a list of colnames, cols, remove them from pydf array
            alters the pydf
            test exists in test_pydf.py
        """
        
        if exclude_cols:
            keep_idxs_li: T_li = [self.hd[col] for col in self.hd if col not in exclude_cols]
        
        elif self.selcols_li:
            keep_idxs_li = self.selcols_li

        else:
            return
        
        for irow, la in enumerate(self.lol):
            la = [la[idx] for idx in keep_idxs_li]
            self.lol[irow] = la
            
        old_cols = list(self.hd.keys())
        new_cols = [old_cols[idx] for idx in keep_idxs_li]
        self._cols_to_hd(new_cols)
        
        new_dtypes = {col: typ for idx, (col, typ) in enumerate(self.dtypes.items()) if idx not in keep_idxs_li}
        self.dtypes = new_dtypes
        

    def select_cols(self, cols: Optional[T_ls]=None, exclude_cols: Optional[T_ls]=None, zero_copy: bool=False):
        """ given a list of colnames, alter the pydf to select only the cols specified.
            
        """
        
        
        if not cols:
            cols = []
        if not exclude_cols:
            exclude_cols = []
            
        desired_cols = self.calc_cols( 
            include_cols=cols,
            exclude_cols=exclude_cols
            )
    
        selected_cols_li = [self.hd[col] for col in desired_cols if col in self.hd]
        
        if not zero_copy:
            # select from the array and create a new object.
            # this is time consuming.
            for irow, la in enumerate(self.lol):
                la = [la[col_idx] for col_idx in range(len(la)) if col_idx in selected_cols_li]
                self.lol[irow] = la
           
            # fix up the column names  
            old_cols = list(self.hd.keys())
            new_cols = [old_cols[idx] for idx in range(len(old_cols)) if idx in selected_cols_li]
            self._cols_to_hd(new_cols)
            
            self.dtypes = {col: typ for col, typ in self.dtypes.items() if col in new_cols}

        else:
            # zero_copy
            self.selected_cols_li = selected_cols_li
            
        
        

    def from_selected_cols(self, cols: Optional[T_ls]=None, exclude_cols: Optional[T_ls]=None) -> 'Pydf':
        """ given a list of colnames, create a new pydf of those cols.
            creates as new pydf
        """
        
        if not cols:
            cols = []
        if not exclude_cols:
            exclude_cols = []
            
        desired_cols = self.calc_cols(include_cols=cols, exclude_cols=exclude_cols)
    
        selected_idxs = [self.hd[col] for col in desired_cols if col in self.hd]
        
        new_lol = []
        
        for irow, la in enumerate(self.lol):
            la = [la[idx] for idx in range(len(la)) if idx in selected_idxs]
            new_lol.append(la)
            
        old_cols = list(self.hd.keys())
        new_cols = [old_cols[idx] for idx in range(len(old_cols)) if idx in selected_idxs]
        
        new_dtypes = {col: typ for col, typ in self.dtypes.items() if col in new_cols}
        
        return Pydf(lol=new_lol, cols=new_cols, dtypes=new_dtypes)
        

    #=========================
    #   modify records
        
    def assign_record_da(self, record_da: T_da):
        """ Assign one record in pydf using the key using a single T_da dict.
            unit tests exist
        """
        
        if not self.keyfield:
            raise RuntimeError("No keyfield estabished for pydf.")
            
        keyfield = self.keyfield
        if keyfield not in record_da:
            raise RuntimeError("No keyfield in dict.")
            
        if self and list(record_da.keys()) != list(self.hd.keys()):
            raise RuntimeError("record fields not equal to pydf columns")
            
        keyval = record_da[keyfield]
            
        row_idx = self.kd.get(keyval, -1)
        if row_idx < 0 or row_idx >= len(self.lol):
            self.append(record_da)
        else:
            #normal_record_da = Pydf.normalize_record_da(record_da, cols=self.columns(), dtypes=self.dtypes)   
            self.lol[row_idx] = [record_da.get(col, '') for col in self.hd]
        

    def assign_record_da_irow(self, irow: int=-1, record_da: Optional[T_da]=None):
        """ Assign one record in pydf using the iloc using a single T_da dict.
            unit tests exist
        """
        
        if record_da is None:
            return
        
        if irow < 0 or irow >= len(self.lol):
            self.append(record_da)
        else:
            #normal_record_da = Pydf.normalize_record_da(record_da, cols=self.columns(), dtypes=self.dtypes)   
            self.lol[irow] = [record_da.get(col, '') for col in self.hd]
        

    def update_by_keylist(self, keylist: Optional[T_ls]=None, record_da: Optional[T_da]=None):
        """ Update selected records in pydf by keylist using record_da
            only update those columns that have dict keys
            but keep all other dict items intact in that row if not updated.
        """
        
        if record_da is None or not self.lol or not self.hd or not self.keyfield or not keylist:
            return

        for key in keylist:
            self.update_record_da_irow(self.kd.get(key, -1), record_da)
            
        

    def update_record_da_irow(self, irow: int=-1, record_da: Optional[T_da]=None):
        """ Update one record in pydf at iloc using a single T_da dict,
            and only update those columns that have dict keys
            but keep all other dict items intact in that row.
            unit tests exist
        """
        
        if record_da is None or not self.lol or not self.hd:
            return
        
        if irow < 0 or irow >= len(self.lol):
            return
        
        for colname, val in record_da.items():
            icol = self.hd.get(colname, -1)
            if icol >= 0:
                self.lol[irow][icol] = record_da[colname]
        

    def assign_icol(self, icol: int=-1, col_la: Optional[T_la]=None, default: Any=''):
        """ modify icol by index using col_la 
            use default if col_la not long enough to fill all cells.
            Also, if col_la not provided, use default to fill all cells in the column.
            if icol == -1, append column on the right side.
        """
        # from utilities import utils

        self.lol = utils.assign_col_in_lol_at_icol(icol, col_la, lol=self.lol, default=default)
        
        
        
    def insert_icol(self, icol: int=-1, col_la: Optional[T_la]=None, colname: str='', default: Any='', keyfield:str=''):
        """ insert column col_la at icol, shifting other column data. 
            use default if la not long enough
            If icol==-1, insert column at right end.
            unit tests
        """
        
        # from utilities import utils

        self.lol = utils.insert_col_in_lol_at_icol(icol, col_la, lol=self.lol, default=default)
        
        if colname:
            if not self.hd:
                self.hd = {}
            if icol < 0 or icol >= len(self.hd):
                icol = len(self.hd)
            hl = list(self.hd.keys())
            hl.insert(icol, colname)
            self.hd = {k: idx for idx, k in enumerate(hl)}
            
        if keyfield:
            self.keyfield = keyfield
            self._rebuild_kd()

        
    def insert_irow(self, irow: int=-1, row_la: Optional[T_la]=None, default: Any=''):
        """ insert row row_la at irow, shifting other rows down. 
            use default if la not long enough
            If irow > len(pydf), insert row at the end.
            
        """
        
        # from utilities import utils

        self.lol = utils.insert_row_in_lol_at_irow(irow=irow, row_la=row_la, lol=self.lol, default=default)
        
        self._rebuild_kd()


    def assign_col(self, colname: str, la: Optional[T_la]=None, default: Any=''):
        """ modify col by colname using la 
            use default if la not long enough.
            test exists in test_pydf.py
        """
        
        if not colname or colname not in self.hd:
            return []
        icol = self.hd[colname]
        self.assign_icol(icol, la, default)
        

    # def set_col(self, colname: str, val: Any):
        # """ modify col by colname using val """
        
        # if not colname or colname not in self.hd:
            # return
        # icol = self.hd[colname]
        # self.set_icol(icol, val)
        

    def insert_col(self, colname: str, col_la: Optional[T_la]=None, icol:int=-1, default:Any='', keyfield:str=''):
        """ add col by colname and set to la at icol
            if la is not long enough for a full column, use the default.
            if colname exists, overwrite it.
            Can use to set a constant value by not passing col_la and setting default.
            Can assign a new keyfield to this column, if keyfield is not ''
            unit tested
        """
        
        if not colname:
            return
        if not col_la:
            col_la = []
            
        colname_icol = self.hd.get(colname, -1)
        if colname_icol >= 0:
            # column already exists. ignore icol, overwrite data.
            self.assign_col(colname, col_la, default)
            return

        self.insert_icol(icol=icol, col_la=col_la, colname=colname, default=default, keyfield=keyfield)
        
        # hl = list(self.hd.keys())
        # hl.insert(icol, colname)
        # self.hd = {k: idx for idx, k in enumerate(hl)}        
        
    
    def insert_idx_col(self, colname='idx', icol:int=0, startat:int=0):
        """ insert an index column at column icol with name colname with indexes starting at 'startat' 
            unit tested
        """
        
        num_rows = len(self)
        col_la = list(range(startat, startat + num_rows))
    
        self.insert_col(colname, col_la, icol)


    def set_col_irows(self, colname: str, irows: T_li, val: Any):
        """ set a given icol and list of irows to val """
        
        if not colname or colname not in self.hd:
            return
        icol = self.hd[colname]

        self.set_icol_irows(icol, irows, val)
    

    def set_icol(self, icol: int, val: Any):
    
        for irow in range(len(self.lol)):
            self.lol[irow][icol] = val
        

    def set_icol_irows(self, icol: int, irows: T_li, val: Any):
        """ set a given icol and list of irows to val """
        
        for irow in irows:
            if irow >= len(self.lol) or irow < 0:
                continue
        
            self.lol[irow][icol] = val        
    
    
    #=========================
    # find/replace
    
    def find_replace(self, find_pat, replace_val):
        """ scan cells in pydf and if match is found, replace the cell with pattern """

        for row_la in self.lol:
            for i, value in enumerate(row_la):
                if bool(re.search(find_pat, str(value))):
                    row_la[i] = replace_val
        
    

    #=========================
    # split and grouping
    
    def split_pydf_into_ranges(self, chunk_ranges: List[Tuple[int, int]]) -> List['Pydf']:
        """ Given a df and list of (start,end) ranges, split pydf into list of pydf.
        """
        
        chunks_lopydf = [self.select_irows(list(range(start, end))) for start,end in chunk_ranges]
        #chunks_lopydf = [self[start:end] for start, end in chunk_ranges]
        return chunks_lopydf
        
        
    
    def split_pydf_into_chunks_lopydf(self, max_chunk_size: int) -> List['Pydf']:
        """ given a pydf, split it evenly by rows into a list of pydfs.
            size of some pydfs may be less than the max but not over.
        """
        # from utilities import utils
        
        chunk_sizes_list = utils.calc_chunk_sizes(num_items=len(self), max_chunk_size=max_chunk_size)
        chunk_ranges = utils.convert_sizes_to_idx_ranges(chunk_sizes_list)
        chunks_lopydf = self.split_pydf_into_ranges(chunk_ranges)
        return chunks_lopydf

           
    #=========================
    #   sort
            
    def sort_by_colname(self, colname:str, reverse: bool=False, length_priority: bool=False):
        """ sort the data by a given colname, using length priority unless specified.
            sorts in place. Make a copy if you need the original order.
        """
        colidx = self.hd[colname]
        self.lol = utils.sort_lol_by_col(self.lol, colidx, reverse=reverse, length_priority=length_priority)
        self._rebuild_kd()
        
    #=========================
    #   apply formulas

    def apply_formulas(self, formulas_pydf: 'Pydf'):
        r""" apply an array of formulas to the data in pydf
        
        formulas must have the same shape as self pydf instance.
        cells which are empty '' do not function.
        
        formulas are re-evaluated until there are no further changes. Error will result if expressions are circular.
        
        #### Special Notation
        There is only a very few cases of special notation:

        - $d -- references the current pydf instance, a convenient shorthand.
        - $c -- the current cell column index
        - $r -- the current cell row index
        
        Typical cases:
            sum($d[$r, :$c])        sum all cells in the current row from the first column upto but not including the current column
            sum($d[:$r, $c])        sum all cells in the cnrrent column from the first row upto but not including the current row
            $d[14,20]+$d[15,25]     add data in the cell at row 14, col 20 to the data in cell at row 15 and column 25
            max(0,$d[($r-1),$c])    the value prior row in current column unless less than 0, then will enter 0.
            $d[($r-1),$c] * 0.15    15% of the value prior row in current column  

        By using the current cell references, formulas can be the same no matter where they may be written, similar to spreadsheet formulas.
        Typical spreadsheet formulas are treated as relative, and then modified whenever copied for the same relative cell references, unless they are made absolute by using $.
        Here, the references are absolute unless you create a relative reference by relating to the current cell row $r and/or column $c.
        
        Example usage:
            The following example adds rows and columns of a 3 x 2 array of values.
        
            example_pydf = Pydf(cols=['A', 'B', 'C'], lol=[[1, 2, 0],[4, 5, 0],[7, 8, 0],[0, 0, 0]])
            formulas_pydf = Pydf(cols=['A', 'B', 'C'], 
            formulas_pydf = Pydf(cols=['A', 'B', 'C'], 
                    lol=[['',                    '',                    "sum($d[$r,:$c])"],
                         ['',                    '',                    "sum($d[$r,:$c])"],
                         ['',                    '',                    "sum($d[$r,:$c])"],
                         ["sum($d[:$r,$c])",     "sum($d[:$r,$c])",     "sum($d[:$r,$c])"]]
                         )
            expected_result = Pydf(cols=['A', 'B', 'C'], lol=[[1, 2, 3],[4, 5, 9],[7, 8, 15],[12, 15, 27]])
            
        """
        
        # TODO: This algorithm is not optimal. Ideally, a dependency tree would be formed and
        # cells modified in from those with no dependencies to those that depend on others.
        # This issue will not become a concern unless the number of formulas is substantial.
        
        if not self:
            return
        
        if self.shape() != formulas_pydf.shape():
            import pdb; pdb.set_trace() #temp
            
            raise RuntimeError("apply_formulas requires data arrays of the same shape.")
        
        lol_changed = True     # must evaluate at least once.
        loop_limit = 100
        loop_count = 0
        
        # the following deals with $d, $r, $c in the formulas
        parsed_formulas_pydf = formulas_pydf._parse_formulas()
        
        while lol_changed:
            lol_changed = False
            loop_count += 1
            if loop_count > loop_limit:
                raise RuntimeError("apply_formulas is resulting in excessive evaluation loops.")
            
            for irow in range(len(self.lol)):
                for icol in range(self._num_cols()):
                    cell_formula = parsed_formulas_pydf.lol[irow][icol]
                    if not cell_formula:
                        # no formula provided -- do nothing
                        continue
                    try:    
                        new_value = eval(cell_formula)
                    except Exception as err:
                        print(f"Error in formula for cell [{irow},{icol}]: '{cell_formula}': '{err}'")
                        import pdb; pdb.set_trace() #temp
                        raise
                    
                    if new_value != self.lol[irow][icol]:
                        # update the value in the array, and set lol_changed flag
                        self.lol[irow][icol] = new_value
                        lol_changed = True
                    else:
                        continue
        self._rebuild_kd()
        
    def _parse_formulas(self) -> 'Pydf':
    
        # start with unparsed formulas
        parsed_formulas = copy.deepcopy(self)
    
        for irow in range(len(self.lol)):
            for icol in range(self._num_cols()):
                proposed_formula = self.lol[irow][icol]
                if not proposed_formula:
                    # no formula provided.
                    continue
                    
                proposed_formula = proposed_formula.replace('$d', 'self')    
                proposed_formula = proposed_formula.replace('$c', str(icol))    
                proposed_formula = proposed_formula.replace('$r', str(irow))    
                      
                parsed_formulas[irow,icol] = proposed_formula
                
        return parsed_formulas
        
            

    def cols_to_dol(self, colname1: str, colname2: str) -> T_dola:
        """ given a pydf with at least two columns, create a dict of list
            lookup where the key are values in col1 and list of values are 
            unique values in col2. Values in cols must be hashable.
            
        For example, if:

        pydf.lol = [['a', 'b', 'c'], 
                    ['b', 'd', 'e'], 
                    ['a', 'f', 'g'], 
                    ['b', 'd', 'm']]
        pydf.columns = ['col1', 'col2', 'col3']
        pydf.cols_to_dol('col1', 'col2') results in
        {'a': ['b', 'f'], 'b':['d']}
            
            test exists in test_pydf.py
            
        """
        
        if colname1 not in self.hd or colname2 not in self.hd or not self.lol:
            return {}

        colidx1 = self.hd[colname1]
        colidx2 = self.hd[colname2]
        
        
        # first work with dict of dict for speed.
        result_dadn: Dict[Any, Dict[Any, None]] = {}
        
        for la in self.lol:
            val1 = la[colidx1]
            val2 = la[colidx2]
            if val1 not in result_dadn:
                result_dadn[val1] = {val2: None}
            elif val2 not in result_dadn[val1]:
                result_dadn[val1][val2] = None
            # otherwise, it is already in the result.
                
        # now convert dadn to dola

        result_dola = {k: list(d.keys()) for k, d in result_dadn.items()}
                
        return result_dola        

    #===============================
    # apply and reduce

    def apply(self, 
            func: Callable[[Union[T_da, T_Pydf], Optional[T_la]], Union[T_da, T_Pydf]], 
            by: str='row', 
            cols: Optional[T_la]=None,                      # columns included in the apply operation.
            keylist: Optional[T_ls]=None,                   # list of keys of rows to include.
            **kwargs: Any,
            ) -> "Pydf":
        """
        Apply a function to each 'row', 'col', or 'table' in the Pydf and create a new Pydf with the transformed data.
        Note: to apply a function to a portion of the table, first select the columns or rows desired 
                using a selection process.

        Args:
            func (Callable): The function to apply to each 'row', 'col', or 'table'. 
            It should take a row dictionary and any additional parameters.
            by (str): either 'row', 'col' or 'table'
                if by == 'table', function should create a new Pydf instance.
            keylist: Optional[T_ls]=None,                   # list of keys of rows to include.
            **kwargs: Additional parameters to pass to the function.

        Returns:
            Pydf: A new Pydf instance with the transformed data.
        """
        if by == 'table':
            return func(self, **kwargs)
       
        result_pydf = self.clone_empty()

        if by == 'row':
            if keylist is None:
                keylist = []
        
            keylist_or_dict = keylist if not keylist or len(keylist) < 30 else dict.fromkeys(keylist)
            for row in self:
                if self.keyfield and keylist_or_dict and self.keyfield not in keylist_or_dict:
                    continue
                transformed_row = func(row, cols, **kwargs)
                result_pydf.append(transformed_row)
                
        elif by == 'col':
            # this is not working yet, don't know how to handle cols, for example.
            raise NotImplementedError
        
            num_cols = self._num_cols()
            for icol in range(num_cols):
                col_la = self.icol(icol)
                transformed_col = func(col_la, cols, **kwargs)
                result_pydf.insert_icol(icol, transformed_col)
        else:
            raise NotImplementedError
            
        # Rebuild the internal data structure (if needed)
        result_pydf._rebuild_kd()

        return result_pydf
      
      
    def update_row(row, da):
        row.update(da)
        return row
      
        
    def apply_in_place(self, 
            func: Callable[[T_da], T_da], 
            by: str='row', 
            keylist: Optional[T_ls]=None,                   # list of keys of rows to include.
            **kwargs: Any,
            ):
        """
        Apply a function to each 'row', 'col', or 'table' in the pydf.

        Args:
            func (Callable): The function to apply to each 'row' 
            It should take a row dictionary and any additional parameters.
            # by (str): either 'row', 'col' or 'table'
            #     if by == 'table', function should create a new Pydf instance.
            keylist: list of keys of rows to include.
            **kwargs: Additional parameters to pass to the function.

        Modifies self in-place.
        """
        if keylist is None:
            keylist = []
        
        if by == 'row':
            keylist_or_dict = keylist if not keylist or len(keylist) < 30 else dict.fromkeys(keylist)

            for idx, row_da in enumerate(self):
                if self.keyfield and keylist_or_dict and self.keyfield not in keylist_or_dict:
                    continue
                transformed_row_da = func(row_da, **kwargs)
                self.lol[idx] = list(transformed_row_da.values())
                
        else:
            raise NotImplementedError
            
        # Rebuild the internal data structure (if needed)
        self._rebuild_kd()
        
        
    def reduce(self, 
            func: Callable[[T_da, T_da], Union[T_da, T_la]], 
            by: str='row', 
            cols: Optional[T_la]=None,                      # columns included in the reduce operation.
            **kwargs: Any,
            ) -> Union[T_da, T_la]:
        """
        Apply a function to each 'row', 'col', or 'table' and accumulate to a single T_da
        Note: to apply a function to a portion of the table, first select the columns or rows desired 
                using a selection process.

        Args:
            func (Callable): The function to apply to each 'row', 'col', or 'table'. 
            It should take a row dictionary and any additional parameters.
            by (str): either 'row', 'col' or 'table'
                if by == 'table', function should create a new Pydf instance.
            **kwargs: Additional parameters to pass to the function.

        Returns:
            either a dict (by='rows' or 'table') or list (by='cols')
        """
        if by == 'table':
            reduction_da = func(self, cols, **kwargs)
            return reduction_da    

        if by == 'row':
            reduction_da = {}
            for row_da in self:
                reduction_da = func(row_da, reduction_da, cols, **kwargs)
            return reduction_da    
                
        elif by == 'col':
            reduction_la = []
            num_cols = self._num_cols()
            for icol in range(num_cols):
                col_la = self.icol(icol)
                reduction_la = func(col_la, reduction_la, **kwargs)
            return reduction_la

        else:
            raise NotImplementedError
        return [] # for mypy only.
        
        
    def manifest_apply(self, 
            func: Callable[[T_da, Optional[T_la]], Tuple[T_da, 'Pydf']],    # function to apply according to 'by' parameter 
            load_func: Callable[[T_da], 'Pydf'],            # optional function to load data for each manifest entry, defaults to local file system 
            save_func: Callable[[T_da, 'Pydf'], str],       # optional function to save data for each manifest entry, defaults to local file system
            by: str='row',                                  # determines how the func is applied.
            cols: Optional[T_la]=None,                      # columns included in the apply operation.
             **kwargs: Any,
            ) -> "Pydf":
        """
        Given a chunk_manifest_pydf, where each record is a chunk_spec (dict),
        1. load the each chunk using 'load_func(chunk_spec)'
        2. apply 'func' to the loaded Pydf instance to produce (result_chunk_spec, new_pydf) 
        3. save new_pydf using 'save_func(result_chunk_spec)'
        4. append result_chunk_spec to result_manifest_pydf describing the resulting chunks 

        Args:
            func (Callable): The function to apply to each table specified by each record in self.
            load_func (Callable): load specified pydf table based on the chunkspec in each row of self.
            save_func (Callable): save resulting pydf table after operation by func.
            **kwargs: Additional parameters to pass to func

        Returns:
            result_manifest_pydf
            
        Note, this method can be used for transformation, where the same number of transformed chunks exists,
            or, it can be used for reduction.
        if 'reduce' is true, then each chunk returns a pydf with a single row.
            
            
        """

        result_manifest_pydf = Pydf()

        for chunk_spec in self:
            # Apply the function for all chunks specified.
            # Load the specified Pydf table
            loaded_pydf = load_func(chunk_spec)

            # Apply the function to the loaded Pydf
            result_chunk_spec, transformed_pydf = loaded_pydf.apply(func, by=by, cols=cols, **kwargs)
            
            # Save the resulting Pydf table
            save_func(result_chunk_spec, transformed_pydf)
        
            # Update the manifest with information about the resulting chunk
            result_manifest_pydf.append(result_chunk_spec)
            
        return result_manifest_pydf        

    
    def manifest_reduce(self, 
            func: Callable[[T_da, Optional[T_la]], T_da], 
            load_func: Optional[Callable[[T_da], 'Pydf']] = None,
            by: str='row',                                  # determines how the func is applied.
            cols: Optional[T_la]=None,                      # columns included in the reduce operation.
            **kwargs: Any,
            ) -> T_da:
        """
        Apply a reduction function to the tables specified by the chunk manifest.

        Args:
            func (Callable): The function to apply to each table specified by each record in self.
            load_func (Callable): Load specified pydf table based on the chunkspec in each row of self.
            **kwargs: Additional parameters to pass to func

        Returns:
            Pydf: Result of reducing all chunks into a single record.
        """
        first_reduction_pydf = self.clone_empty()

        for chunk_spec in self:
            # Load the specified Pydf table
            loaded_pydf = load_func(chunk_spec)

            # Apply the function to the loaded Pydf
            reduction_da = loaded_pydf.reduce(func, by=by, cols=cols, **kwargs)

            first_reduction_pydf.append(reduction_da)     

        final_reduction_da = first_reduction_pydf.reduce(func, by=by, cols=cols, **kwargs)
        
        return final_reduction_da
        
        
    def manifest_process(self, 
            func: Callable[[T_da, Optional[T_la]], T_da],   # function to run for each hunk specified by the manifest
            **kwargs: Any,
            ) -> 'Pydf':                                    # records describing metadata of each hunk
        """
        Given a chunk_manifest_pydf, where each record is a chunk_spec (dict),
        1. apply 'func' to each chunk specified by the manifest.
        2. func() will load the chunk and save any results.
        3. returns one record for each func() call, add these to the resulting pydf.

        Args:
            func (Callable): The function to apply to each table specified by each record in self.
            cols:            Reduce scope to a set of cols
            **kwargs: Additional parameters to pass to func

        Returns:
            result_pydf
                      
        """

        result_pydf = Pydf()

        for chunk_spec in self:
            # Apply the function for all chunks specified.
            # Load the specified Pydf table
            # Apply the function to the loaded Pydf
            result_da = func(chunk_spec, **kwargs)
            
            # Update the manifest with information about the resulting chunk
            result_pydf.append(result_da)
            
        return result_pydf        

    
    def groupby(self, 
            colname: str='', 
            colnames: Optional[T_ls]=None,
            omit_nulls: bool=False,         # do not group to values in column that are null ('')
            ) -> Union[Dict[str, 'Pydf'], Dict[Tuple[str, ...], 'Pydf']]:
        """ given a pydf, break into a number of pydf's based on one colname or list of colnames specified. 
            For each discrete value in colname(s), create a pydf table with all cols,
            including colname, and return in a dopydf (dict of pydf) structure.
            If list of colnames is provided, dopydf keys are tuples of the values.
        """
        
        if isinstance(colname, list) and not colnames:
            return self.groupby_cols(colnames=colname)
        elif colnames and not colname:
            if len(colnames) > 1:
                return self.groupby_cols(colnames=colnames)
            else:
                colname = colnames[0]
                # can continue below.
        
        result_dopydf: Dict[str, 'Pydf'] = {}
        
        for da in self:
            fieldval = da[colname]
            if omit_nulls and fieldval=='':
                continue
            
            if fieldval not in result_dopydf:
                result_dopydf[fieldval] = self.clone_empty()
                
            this_pydf = result_dopydf[fieldval]
            this_pydf.record_append(record_da=da)
            result_dopydf[fieldval] = this_pydf
    
        return result_dopydf
    

    def groupby_cols(self, colnames: T_ls) -> Dict[Tuple[str, ...], 'Pydf']:
        """ given a pydf, break into a number of pydf's based on colnames specified. 
            For each discrete value in colname, create a pydf table with all cols,
            including colnames, and return in a dopydf (dict of pydf) structure,
            where the keys are a tuple of the column values.
            
            Examine the records to determine what the values are for the colnames specified.
        """
        
        result_dopydf: Dict[Tuple[str, ...], 'Pydf'] = {}
        
        for da in self:
            fieldval_tuple = tuple(da[colname] for colname in colnames)
            if fieldval_tuple not in result_dopydf:
                result_dopydf[fieldval_tuple] = this_pydf = self.clone_empty()
            
            else:
                this_pydf = result_dopydf[fieldval_tuple]
                
            this_pydf.record_append(record_da=da)
            result_dopydf[fieldval_tuple] = this_pydf
    
        return result_dopydf


    def groupby_cols_reduce(self, 
            groupby_colnames: T_ls, 
            func: Callable[[T_da, T_da], Union[T_da, T_la]], 
            by: str='row',                                  # determines how the func is applied.
            reduce_cols: Optional[T_la]=None,               # columns included in the reduce operation.
            **kwargs: Any,
            ) -> 'Pydf':
        """ given a pydf, break into a number of pydf's based on values in groupby_colnames. 
            For each group, apply func. to data in reduce_cols.
            returns pydf with one row per group, and keyfield not set.
            
            Application note:
            This can be commonly used when some colnames are important for grouping, while others
            contain values or numeric data that can be reduced.
            
            For example, consider the array with the following columns:
            
            gender, religion, zipcode, cancer, covid19, gun, auto
            
            The data can be first grouped by the attribute columns gender, religion, zipcode, and then
            then prevalence of difference modes of death can be summed. The result is a pydf with one
            row per unique combination of gender, religion, zipcode. Say we consider just M/F, C/J/I, 
            and two zipcodes 90001, and 90002, this would result in the following rows, where the 
            values in paranthesis are the reduced values for each of the numeric columns, such as the sum.
            
            gender, religion, zipcode, cancer, covid19, gun, auto
            M, C, 9001, (cancer), (covid19), (gun), (auto)
            M, C, 9002, (cancer), (covid19), (gun), (auto)
            M, J, 9001, (cancer), (covid19), (gun), (auto)
            M, J, 9002, (cancer), (covid19), (gun), (auto)
            M, I, 9001, (cancer), (covid19), (gun), (auto)
            M, I, 9002, (cancer), (covid19), (gun), (auto)
            F, C, 9001, (cancer), (covid19), (gun), (auto)
            F, C, 9002, (cancer), (covid19), (gun), (auto)
            F, J, 9001, (cancer), (covid19), (gun), (auto)
            F, J, 9002, (cancer), (covid19), (gun), (auto)
            F, I, 9001, (cancer), (covid19), (gun), (auto)
            F, I, 9002, (cancer), (covid19), (gun), (auto)
            
            
        """
        
        # divide up the table into groups where each group has a unique set of values in groupby_colnames
        grouped_tdopydf = self.groupby_cols(groupby_colnames)
        
        result_pydf = Pydf(cols=groupby_colnames + reduce_cols)
        
        for coltup, this_pydf in grouped_tdopydf.items():
        
            if not this_pydf:
                # nothing found with this combination of groupby cols.
                continue
                
            # apply the reduction function
            reduction_da = this_pydf.reduce(func, by=by, cols=reduce_cols, **kwargs)
            
            # add the groupby cols
            for idx, groupcolname in enumerate(groupby_colnames):
                reduction_da[groupcolname] = coltup[idx]
            
            result_pydf.append(this_pydf)

        return result_pydf
    

    def groupby_reduce(self, 
            colname:str, 
            func: Callable[[T_da, T_da], Union[T_da, T_la]], 
            by: str='row',                                  # determines how the func is applied.
            cols: Optional[T_la]=None,                      # columns included in the reduce operation.
            **kwargs: Any,
            ) -> 'Pydf':
        """ given a pydf, break into a number of pydf's based on colname specified. 
            For each group, apply callable.
            returns pydf with one row per group, with keyfield the groupby value in colname.
            
        """
        
        grouped_dopydf = self.groupby(self, colname)
        result_pydf = Pydf(keyfield = colname)
        
        for colval, this_pydf in grouped_dopydf.items():
        
            # maybe remove colname from cols here
        
            reduction_da = this_pydf.reduce(func, by=by, cols=cols, **kwargs)
            
            # add colname:colval to the dict
            reduction_da = {colname: colval, **reduction_da}
            
            # this will also maintain the kd.
            result_pydf.append(reduction_da)

        return result_pydf

        
    #===================================
    # apply / reduce convenience methods

    def pydf_sum(self, 
            by: str = 'row', 
            cols: Optional[T_la]=None
            ) -> T_da:
            
        return self.reduce(func=Pydf.sum_da, by=by, cols=cols)
    

    def pydf_valuecount(self, 
            by: str = 'row', 
            cols: Optional[T_la]=None
            ) -> T_da:
        """ count values in columns specified and return for each column,
            a dictionary of values and counts for each value in the column
            
            Need a way to specify that blank values will also be counted.
        """
            
        return self.reduce(func=Pydf.count_values_da, by=by, cols=cols)


    def groupsum_pydf(self,
            colname:str, 
            func: Callable[[T_da, T_da, Optional[T_la]], Union[T_da, T_la]], 
            by: str='row',                                  # determines how the func is applied.
            cols: Optional[T_la]=None,                      # columns included in the reduce operation.
            ) -> 'Pydf':
    
        result_pydf = self.groupby_reduce(colname=colname, func=Pydf.sum_da, by=by, cols=cols)
        
        return result_pydf


    def set_col2_from_col1_using_regex_select(self, col1: str, col2: str='', regex: str=''):
    
        """ given two cols that already exist, apply regex select to col1 to create col2
            regex should include parens that enclose the desired portion of col1.
        """
        
        # from utilities import utils
    
        def set_row_col2_from_col1_using_regex_select(row_da: T_da, col1: str, col2: str, regex: str) -> T_da:
            row_da[col2] = utils.safe_regex_select(regex, row_da[col1])
            return row_da
            
        if not col2:
            col2 = col1

        self.apply_in_place(lambda row_da: set_row_col2_from_col1_using_regex_select(row_da, col1, col2, regex)) 


    def apply_to_col(self, col: str, func: Callable, **kwargs):
    
        self[:, col] = list(map(func, self[:, col], **kwargs))
        
    # for example:
    #   my_pydf.apply_to_col(col='colname', func=lambda x: re.sub(r'^\D+', '', x))    

    #====================================
    # reduction atomic functions
    
    # requirements for reduction functions:
    #   1. reduction will produce a single dictionary of results, for each pydf chunk.
    #   2. each atomic function will be staticmethod which accepts a single row dictionary, this_da
    #       and contributes to an accum_da. The accum_da is mutated by each row call.
    #   3. the reduction atomic function must be able to deal with combining results
    #       in a pydf where each record is the result of processing one chunk.
    #   4. each atomic function will also accept a cols parameter which identifies which 
    #       columns are to be included in the reduction, if it is not None or []
    #       Otherwise, all columns will be processed. This columns parameter can be
    #       initialized explicitly or using my_pydf.calc_cols(include_cols, exclude_cols, include_dtypes, excluded_dtypes)
    #   5. Even if columns are reduced, the result of the function will include all columns
    #       and non-specified columns will be initialized to '' empty string. This complies
    #       with design goal of always producing a result that will be useful in a report.
    #   6. Therefore, the reduction result may be appended to the pydf if desired.
    
    

    @staticmethod
    def sum_da(row_da: T_da, accum_da: T_da, cols: Optional[T_la]=None) -> T_da:     # result_da
        """ sum values in row and accum dicts per colunms provided. 
            will safely skip data that can't be summed.
        """
        cols_list = {}
        
        if cols is None:
            cols_list = []
        elif not isinstance(cols, list):
            cols_list = [cols]
        else:
            cols_list = cols
            
        if len(cols_list) > 10:    
            cols_list_or_dict = dict.fromkeys(cols)
        else:
            cols_list_or_dict = cols_list
            
        for key, value in row_da.items():
            if cols_list_or_dict and key not in cols_list_or_dict:
                # accum_da[key] = ''
                continue

            try:
                if key in accum_da:
                    accum_da[key] += value
                else:
                    accum_da[key] = value
            except Exception:
                pass
        return accum_da


    @staticmethod
    def count_values_da(row_da: T_da, result_dodi: T_dodi, cols: Optional[T_la]=None) -> T_dodi:
        """ incrementally build the result_dodi, which is the valuecounts for each item in row_da.
            can be used to calculate valuecounts over all rows and chunks.
            
            row_da may be scalar values (typically strings that are to be counted)
            but may also be a dodi of totals from a set of chunks that are to be combined.
            
            Intended use is to use this to calculate valuecounts by scanning all rows of a pydf.
            Return a dodi.
            Put those in a pydf table and then scan those combined values and create a singular result.
            
            This is a reducing and accumulating operation. Can be used with pydf.reduce()
            
        """
    
        if cols is None:
            cols_dict = {}
        else:
            cols_dict = dict.fromkeys(cols)
        
        for key, val in row_da.items():
            
            if cols_dict and key not in cols_dict:
                continue

            if key not in result_dodi:
                result_dodi[key] = {}
                
            if val and isinstance(val, dict):
                # val is a dict of values determined in another pass.
                Pydf.sum_dodis(val, result_dodi)
                continue
                
            if val not in result_dodi[key]:
                result_dodi[key][val] = 1
            else:    
                result_dodi[key][val] += 1
        
        return result_dodi
        
                        
    @staticmethod
    def sum_dodis(this_dodi: T_dodi, accum_dodi: T_dodi):
        """ add values for matching keys in this_dodi and accum_dodi.
            sum cases where the keys are the same.
        """
    
        for key, this_di in this_dodi.items():
            if key in accum_dodi:
                Pydf.sum_da(this_di, accum_dodi[key])
            else:
                accum_dodi[key] = this_di

                    
    #===============================================
    # functions not following apply or reduce pattern
    
    # this function does not use normal reduction approach.
    def sum(self, colnames_ls: Optional[T_ls]=None, numeric_only: bool=False) -> dict: # sums_di
        """ total the columns in the table specified, and return a dict of {colname: total,...} 
        
        
            unit tests exist
        """


        if colnames_ls is None:
            cleaned_colnames_ls = list(self.hd.keys())
            cleaned_colidxs_li = list(range(len(cleaned_colnames_ls)))
        elif not (numeric_only and self.dtypes):
            cleaned_colnames_ls = [col for col in colnames_ls if col in self.hd]
            cleaned_colidxs_li = [self.hd[col] for col in cleaned_colnames_ls]  
        else:    
            cleaned_colnames_ls = [col for col in colnames_ls if col in self.hd and self.dtypes.get(col) in [int, float]]
            cleaned_colidxs_li = [self.hd[col] for col in cleaned_colnames_ls]  
        
        sums_d_by_colidx = dict.fromkeys(cleaned_colidxs_li, 0.0)
        
        for la in self.lol:
            for colidx in cleaned_colidxs_li:
                if la[colidx]:
                    if numeric_only:
                        sums_d_by_colidx[colidx] += Pydf._safe_tofloat(la[colidx])
                    else:
                        sums_d_by_colidx[colidx] += float(la[colidx])

        try:
            sums_d = {cleaned_colnames_ls[idx]: sums_d_by_colidx[colidx] for idx, colidx in enumerate(cleaned_colidxs_li)}
        except Exception:
            import pdb; pdb.set_trace() #perm ok
            pass 
        sums_d = utils.set_dict_dtypes(sums_d, self.dtypes)  # type: ignore
        
        return sums_d
        

    def sum_np(self, colnames_ls: Optional[T_ls]=None, ) -> dict: # sums_di
        """ total the columns in the table specified, and return a dict of {colname: total,...}
            This uses NumPy and requires that library, but this is about 3x faster.
            If you have mixed types in your Pydf array, then use colnames to subset the
            columns sent to NumPy to those that contain only numerics and blanks.
            For many numeric operations, convert a set of columns to NumPy
            and work directly with NumPy and then convert back. See to_numpy and from_numpy()
        """
        # unit tests exist
        #   need tests for blanks and subsetting columns.
        
        if not self:
            return {}

        if colnames_ls is None:
            to_sum_pydf = self
            colnames_ls = self.columns()
        else:
            to_sum_pydf = self.from_selected_cols(cols=colnames_ls)
            """ given a list of colnames, create a new pydf of those cols.
                creates as new pydf
            """
        
        # convert those columns to an numpy array.
        nparray = to_sum_pydf.to_numpy()
        
        # sum the columns in the array.
        sum_columns = np.sum(nparray, axis=0)
        
        #convert to a dictionary.
        sums_d = dict(zip(colnames_ls, sum_columns.tolist()))
        
        return sums_d

    
    def valuecounts_for_colname(self, 
            colname: str, 
            sort: bool=False, 
            reverse: bool=True,
            omit_nulls: bool=False,
            ) -> T_di:
        """ given a column of enumerated values, count all unique values in the column 
            and return a dict of valuecounts_di, where the key is each of the unique values
            and the value is the count of that value in the column.
            if sort is true, return the dict sorted from most frequent to least frequently
            detected value.
            unit tests exist
        """

        valuecounts_di: T_di = {}
        
        if colname not in self.hd:
            return {}

        icol = self.hd[colname]
        
        for irow in range(len(self.lol)):
            val = self.lol[irow][icol]
            if val not in valuecounts_di:
                valuecounts_di[val] = 1
            else:
                valuecounts_di[val] += 1

        if omit_nulls:
            utils.safe_del_key(valuecounts_di, '') 
                
        if sort:
            valuecounts_di = dict(sorted(valuecounts_di.items(), key=lambda x: x[1], reverse=reverse))

        return valuecounts_di


    def valuecounts_for_colnames_ls(self, colnames_ls: Optional[T_ls]=None, sort: bool=False, reverse: bool=True) -> T_dodi:
        """ return value counts for a set of columns or all columns if no colnames_ls are provided.
            sort if noted from most prevalent to least in each column.
        """
    
        if not colnames_ls:
            colnames_ls = self.columns()
            
        colnames_ls = cast(T_ls, colnames_ls)    

        valuecounts_dodi: T_dodi = {}
        
        for colname in colnames_ls:
            valuecounts_dodi[colname] = \
                self.valuecounts_for_colname(colname, sort=sort, reverse=reverse)

        return valuecounts_dodi
    

    def valuecounts_for_colname_selectedby_colname(
            self, 
            colname: str, 
            selectedby_colname: str, 
            selectedby_colvalue: str,
            sort: bool = False, 
            reverse: bool = True,
            ) -> T_di:
        """ Create valuecounts for each colname for all rows where
            selectedby_colvalue is found in selectedby_colname
        """    
    

        valuecounts_di: T_di = {}
        
        if colname not in self.hd or selectedby_colname not in self.hd:
            return {}

        icol = self.hd[colname]
        selectedby_colidx = self.hd[selectedby_colname]
        
        for irow in range(len(self.lol)):
            val = self.lol[irow][selectedby_colidx]
            if val != selectedby_colvalue:
                continue
            val = self.lol[irow][icol]    
            if val not in valuecounts_di:
                valuecounts_di[val] = 1
            else:
                valuecounts_di[val] += 1

        if sort:
            valuecounts_di = dict(sorted(valuecounts_di.items(), key=lambda x: x[1], reverse=reverse))

        return valuecounts_di
        

    def valuecounts_for_colnames_ls_selectedby_colname(self, 
            colnames_ls: Optional[T_ls]=None,
            selectedby_colname: str = '', 
            selectedby_colvalue: str = '',
            sort: bool = False, 
            reverse: bool = True,
            ) -> T_dodi:
            
        """ Create valuecounts for each column in colnames_ls (or all columns if None)
            when selectedby_colvalue is found in selectedby_colname
            
            
        """    
    

        if not colnames_ls:
            colnames_ls = self.columns()
            
        colnames_ls = cast(T_ls, colnames_ls)    

        valuecounts_dodi: T_dodi = {}
        
        for colname in colnames_ls:
            valuecounts_dodi[colname] = \
                self.valuecounts_for_colname_selectedby_colname(
                        colname,
                        selectedby_colname, 
                        selectedby_colvalue,
                        sort=sort, 
                        reverse=reverse,
                        )

        return valuecounts_dodi


    def valuecounts_for_colname1_groupedby_colname2(self,
            colname1: str,
            groupedby_colname2: str, 
            sort: bool = False, 
            reverse: bool = True,
            ) -> T_dodi:
        """ frequently, two columns are related, and it may be required 
            that one is single-valued for each item in the other.
            This function does a single scan of the data and accumulates
            value counts for colname1 for each value in groupedby colname2.
            Result is dodi, with the first key being the groupby values
            and the second being the values counted in colname1.
        """
            

        valuecounts_dodi: T_dodi = {}
        
        if colname1 not in self.hd or groupedby_colname2 not in self.hd:
            return {}

        icol1 = self.hd[colname1]
        groupedby_col2idx = self.hd[groupedby_colname2]
        
        for row in self.lol:
            groupval = row[groupedby_col2idx]
            val = row[icol1]    
            if groupval not in valuecounts_dodi:
                valuecounts_dodi[groupval] = {}
            valuecounts_di: T_di = valuecounts_dodi[groupval]    
            if val not in valuecounts_di:
                valuecounts_di[val] = 1
            else:
                valuecounts_di[val] += 1

        if sort:
            for group, valuecounts_di in valuecounts_dodi.items():
                valuecounts_dodi[group] = dict(sorted(valuecounts_di.items(), key=lambda x: x[1], reverse=reverse))

        return valuecounts_dodi

    

    def gen_stats_pydf(self, col_def_lot: T_lota) -> T_doda:

        info_dod = {}

        # from utilities import utils

        for col_def_ta in col_def_lot:
            col_name, col_dtype, col_format, col_profile = col_def_ta
            
            col_data_la = self.col(col_name)
            
            info_dod[col_name] = utils.list_stats(col_data_la, profile=col_profile)
            
        return info_dod

   
    def transpose(self, new_keyfield:str='', new_cols:Optional[T_la]=None, include_header:bool = False) -> 'Pydf':
        """ 
        This implementation uses the built-in zip(*self.lol) to transpose the rows and columns efficiently. 
        The resulting transposed data is then used to create a new Pydf instance.
    
        Args:
        - new_cols (list): names of the new columns. If include_header is True, this will be the first column.
        - new_keyfield (str): The new keyfield to be used in the transposed Pydf.
        - include_header (bool): indicates if the column names, if defined, will also be included
                        and will become the first column in the result

        Returns:
        - Pydf: A new Pydf instance with transposed data and optional new keyfield.
        
        """

        if not new_cols:
            new_cols = ['key'] + Pydf._generate_spreadsheet_column_names_list(num_columns=len(self.lol))

        # transpose the array
        new_lol = [list(row) for row in zip(*self.lol)]
        
        if include_header:
            # add a new first column which will be the old column names row.
            # from utilities import utils
            
            new_lol = utils.insert_col_in_lol_at_icol(icol=0, col_la=self.columns(), lol=new_lol)
        
        return Pydf(lol=new_lol, name=self.name, keyfield=new_keyfield, cols=new_cols, use_copy=True)        



    #===============================
    # reporting

    def md_pydf_table_snippet(
            self, 
            ) -> str:
        """ provide an abbreviated md table given a pydf representation """
        
        return self.to_md(
                max_rows        = self.md_max_rows, 
                max_cols        = self.md_max_cols, 
                shorten_text    = True, 
                max_text_len    = 80, 
                smart_fmt       = False, 
                include_summary = True,
                )

    # the following alias is defind at the bottom of this file.
    # Pydf.md_pydf_table = Pydf.to_md

    def to_md(
            self, 
            max_rows:       int     = 0,         # limit the maximum number of row by keeping leading and trailing rows.
            max_cols:       int     = 0,         # limit the maximum number of cols by keeping leading and trailing cols.
            just:           str     = '',        # provide the justification for each column, using <, ^, > meaning left, center, right justified.
            shorten_text:   bool    = True,      # if the text in any field is more than the max_text_len, then shorten by keeping the ends and redacting the center text.
            max_text_len:   int     = 80,        # see above.
            smart_fmt:      bool    = False,     # if columns are numeric, then limit the number of figures right of the decimal to "smart" numbers.
            include_summary: bool   = False,     # include a one-line summary after the table.
            disp_cols:      Optional[T_ls]=None, # use these column names instead of those defined in pydf.
            ) -> str:
        """ provide an full md table given a pydf representation """

        pydf_lol = self.pydf_to_lol_summary(max_rows=max_rows, max_cols=max_cols, disp_cols=disp_cols)
        
        header_exists = bool(self.hd)
        
        mdstr = md.md_lol_table(pydf_lol, 
            header              = None, 
            includes_header     = header_exists, 
            just                = just or ('>' * len(self.hd)), 
            omit_header         = not header_exists, 
            shorten_text        = shorten_text, 
            max_text_len        = max_text_len, 
            smart_fmt           = smart_fmt,
            
            )
        if include_summary:    
            mdstr += f"\n\[{len(self.lol)} rows x {len(self.hd)} cols; keyfield={self.keyfield}; {len(self.kd)} keys ] ({self.__class__.__name__})\n"
        return mdstr

    def pydf_to_lol_summary(self, max_rows: int=10, max_cols: int=10, disp_cols:Optional[T_ls]=None) -> T_lola:
    
        # from utilities import utils

        # first build a basic summary
        if disp_cols:
            colnames_ls = disp_cols
        else:
            colnames_ls = list(self.hd.keys())
            
        result_lol = self.lol
        if colnames_ls:
            result_lol = [colnames_ls] + result_lol

        # no limits, return summary.
        if not max_rows and not max_cols:
            return result_lol

        num_rows    = len(self.lol) if self.lol else 0
        num_cols    = self._num_cols()

        if max_rows and num_rows <= max_rows:
            # Get all the rows, but potentially limit columns
            result_lol = utils.reduce_lol_cols(result_lol, max_cols=max_cols)
        
        else:
            # Get the first and last portion of rows
            
            first_lol   = self.lol[:max_rows//2]
            last_lol    = self.lol[-(max_rows//2):]
            divider_lol = [['...'] * num_cols]
            
            result_lol  = [colnames_ls] + first_lol + divider_lol + last_lol
            result_lol = utils.reduce_lol_cols(result_lol, max_cols=max_cols)

        return result_lol
        
    @staticmethod
    def dict_to_md(da: T_da, cols: Optional[T_ls]=None, just: str='<<') -> str:
        """ this convenience method can be used for interactive inspection of a dict,
            by placing the dict keys in the first column and the values in the second
            column. Much easier to work with than just using print(da).
            
            To use this interactively, use print(Pydf.dict_to_md(my_da))
        """
        if not cols:
            cols = ['key', 'value']
    
        return Pydf.from_lod_to_cols([da], cols=cols).to_md(just=just)
            
        
    #=========================================
    #  Reporting Convenience Methods

    def value_counts_pydf(self, 
            colname: str,                       # column name to include in the value_counts table
            sort: bool=False,                   # sort values in the category
            reverse: bool=True,                 # reverse the sort
            include_total: bool=False,          #
            omit_nulls: bool=False,             # set to true if '' should be omitted.            
            ) -> 'Pydf':
        """ create a values count pydf of the results of value counts analysis of one column, colname.
            The result is a pydf table with two columns.
            Left column has the values, and the right column has the counts for each value.
            column names are [colname, 'counts'] in the result. These can be changed later if they are not 
            output when used with multiple columns may be useful but only if they have the same set of values.
            provides a total line if "include_sum" is true.
        """
            
        value_counts_di   = self.valuecounts_for_colname(colname=colname, sort=sort, reverse=reverse)
        
        if omit_nulls:
            utils.safe_del_key(value_counts_di, '') 

        value_counts_pydf = Pydf.from_lod_to_cols([value_counts_di], cols=[colname, 'counts'])

        if include_total:
            value_counts_pydf.append({colname: ' **Total** ', 'counts': sum(value_counts_pydf[:,'counts'])})
        return value_counts_pydf


Pydf.md_pydf_table = Pydf.to_md            


# DO NOT DELETE THESE LINES.
# these are required to define these.
# these were required before making a full package that is pip loaded.
#T_pydf = Pydf
#T_dopydf = Dict[Union[str, Tuple[str, ...]], Optional[T_pydf]]
