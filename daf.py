# daf.py
"""

# Daffodil -- Python Dataframes

The Daffodil class provides a lightweight, simple and fast alternative to provide 
2-d data arrays with mixed types.

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
See README file at this location: https://github.com/raylutz/Daf/blob/main/README.md
"""

"""
    v0.1.X (pending)
            Started creating separate package, moved comment text to README.md
            For apply_formulas(), added relative row and column references $r and $c plus $d to reference the daf object.
            Changed the result of a row access, such as $d[$r, :$c] to return a list so it could be compatible with sum.
                use daf.select_irow() to select a row with dict as the result.
    
    v0.2.0  (2024-02-28) 
            Copied related code to Pydf repo and resolved all imports. All tests running.
            Added option of appending a plain list to daf instance using .append()
            Added 'omit_nulls' option to col(), col_to_la(), icol_to_la(), valuecounts_for_colname()
            Added ability to groupby multiple cols
            in select_records_daf(self, keys_ls: T_ls, inverse:bool=False), added inverse boolean.
            Started to add selcols for zero-copy support.
            Added _num_cols() 
            Added unit tests.
            Add groupby_cols_reduce() and sum_np()
            Fixed bug in item setter for str row.
            Added demo of making list of file in a folder.
            groupby_cols_reduce() added, unit tests added. Demo added.
            Fix demo to run on windows, mac or linux.
            Add produced test files to gitignore.
            changed _num_cols() to num_cols()
            removed selcols_ls from class. 
            Pulled in from_csv_file()
            Added buff_to_file()
            improved is_d1_in_d2() by using idiom in Python 3.
            moved sanitize_cols to set_cols() method.
                1. read with no headers.
                2. pull off first row using indexing (could add pop_row())
                3. set set_cols with sanitize_cols=True.
            Removed:
                unflatten_dirname() <-- remove?
                flatten_dirname() <-- remove?
                cols_to_strbool() <-- remove?
                insert_icol()
                    keyfield <-- remove!  use set_keyfield instead
                insert_col() (removed keyfield)
                from_selected_cols <-- remove! but check for usage! (use my_daf[:, colnames_ls]
           Refactor get_item and set_item
                started this but complexity not that much better.
                Redid it again after discussion with Jeremy which was helpful.
            
            
            tests added
                initialization from dtypes and no cols.
                set_cols()
                set_keyfield()
                daf_utils.is_d1_in_d2()
                improved set_cols() to test sanitize_cols flag.
                .len(), num_cols(), shape(), len()
                row_idx_of()
                remove_key -- keyfield not set. (remove)
                get_existing_keys
                select_record_da -- row_idx >= len(self.lol)
                _basic_get_record_da -- no hd, include_cols
                select_first_row_by_dict    
                select_where_idxs
                select_cols
                calc_cols
                    no hd defined.
                    include_cols > 10
                    exclude_cols > 10
                    exclude_types
                insert_col()
                    colname already exists.
                from_lod() with empty records_lod
                to_cols_dol()
                set_lol()
                from_pandas_df()
                to_pandas_df()
                from_dod()
                to_dod()
                from_excel_buff()
                
            Added:
                to_cols_dol()
            Moved code for __getitem__ and __setitem__ to 'indexing' and 
                used method equating to introduce them in the class.
            Moved to_pandas and from_pandas to file daf_pandas.py
            Changed constructors from_... from staticmethods to classmethods
            move toward deprecating remove_key() remove_keys()
            add silent_error in gkeys_to_idxs and trigger error if not selected.
            Handle missing keyfield, hd, kd, in inverse mode.
            Added to_value(), to_list(), to_dict() and tests.
            Tested negative indexes in []
            retmode attribute and constructor parameter to set 'val' vs. 'obj' return value.
            moved __getitem__ and __setitem__ back to main class now that they are vastely reduced in complexity.
            Name change from Pydf to Daffodil and resolve issues.

    v0.3.0  (pending) 
            Added fmt parameter so file are saved with proper Content-Type metadata.
            Added 'from .Daf import Daf' to __init__.py to reduce level.  Eventually removed this.
            
    v0.2.2  Changed the file structure to be incompliance with Python traditions.
                user can use 'from daffodil.daf import Daf' and then Daf()
            Moved daf.py containing class Daf to the top level.
            put supporting functions in lib.


            
    TODO
             
            tests: need to add 
                __str__ and __repr__
                .isin <-- keep this?
                normalize() <-- keep this? (not used)
                
                apply_dtypes()
                unflatten_cols()
                unflatten_by_dtypes()
                flatten_cols()
                flatten_by_dtypes()
                from_csv_buff()
                from_numpy()
                to_csv_buff()
                append()  
                    empty data item
                    simple list with columns defined.
                    no columns defined, just append to lol.
                concat()
                    not other instance
                    empty current daf
                    self.hd != other_instance.hd
                extend()
                    no input records
                    some records empty
                record_append()
                    empty record
                    overwrite of existing
                select_records_daf   (remove?)
                    no keyfield
                    inverse
                        len(keys_ls) > 10 and len(self.kd) > 30
                        smaller cases
                iloc                (remove?)
                    no hd
                icol_to_la          (remove?)
                    unique
                    omit_nulls
                assign_record_da          (remove?)
                    no keyfield defined.
                update_by_keylist <-- remove?
                insert_irow()
                set_col_irows() <-- remove.
                set_icol()  <-- remove.
                set_icol_irows() <-- remove.
                find_replace
                sort_by_colname(
                apply_formulas()
                    no self
                    formula shape differs
                    error in formula
                apply
                    keylist without keyfield
                update_row()
                apply_in_place()
                    keylist without keyfield
                reduce 
                    by == 'col'
                manifest_apply()
                manifest_reduce()
                manifest_process()
                groupby()
                groupby_reduce()
                daf_valuecount()
                groupsum_daf()
                set_col2_from_col1_using_regex_select
                    not col2
                apply_to_col
                sum_da 
                    one col
                    cols_list > 10
                count_values_da()
                sum_dodis -- needed?
                valuecounts_for_colname
                    omit_nulls
                valuecounts_for_colnames_ls_selectedby_colname
                    not colnames_ls
                gen_stats_daf()
                to_md()
                dict_to_md() <-- needed?
                value_counts_daf()
                
            daf_utils
                is_linux()
                apply_dtypes_to_hdlol()
                    empty case
                select_col_of_lol_by_col_idx()
                    col_idx out of range.
                unflatten_hdlol_by_cols 
                json_encode
                make_strbool <-- remove?
                test_strbool >-- remove?
                xlsx_to_csv
                add_trailing_columns_csv()
                insert_col_in_lol_at_icol()
                    col_la empty
                insert_row_in_lol_at_irow() unused?
                calc_chunk_sizes
                    not num_items or not max_chunk_size
                sort_lol_by_col
                set_dict_dtypes 
                    various cases
                list_stats
                    REMOVE?
                list_stats_index
                list_stats_attrib
                list_stats_filepaths
                list_stats_scalar
                list_stats_localidx
                is_list_allints
                profile_ls_to_loti
                reduce_lol_cols
                s3path_to_url
                parse_s3path
                transpose_lol
                safe_get_idx
                shorten_str_keeping_ends
                safe_max
                smart_fmt
                str2bool
                safe_del_key
                dod_to_lod
                lod_to_dod
                safe_eval
                safe_min
                safe_stdev
                safe_mean
                sts
                split_dups_list
                clean_numeric_str
                is_numeric
            daf_indexing
                many cases
            daf_md
                not tested at all.
                
"""            
    
    
#VERSION  = 'v0.2.X'
#VERSDATE = '2024-02-28'  

import sys
import io
import csv
import copy
import re
import numpy as np
    
sys.path.append('..')

from lib.daf_types import T_ls, T_lola, T_di, T_hllola, T_loda, T_da, T_li, T_dtype_dict, \
                            T_dola, T_dodi, T_la, T_lota, T_doda, T_buff, T_ds, T_lb # , T_df
                     
import lib.daf_utils    as utils
import lib.daf_md       as md
import lib.daf_pandas   as daf_pandas

from typing import List, Dict, Any, Tuple, Optional, Union, cast, Type, Callable #
def fake_function(a: Optional[List[Dict[str, Tuple[int,Union[Any, str]]]]] = None) -> Optional[int]:
    return None or cast(int, 0)       # pragma: no cover

T_Pydf = Type['Daf']
T_Daf = Type['Daf']


class Daf:
    RETMODE_OBJ  = 'obj'
    RETMODE_VAL  = 'val'
    

    def __init__(self, 
            lol:        Optional[T_lola]        = None,     # Optional List[List[Any]] to initialize the data array. 
            cols:       Optional[T_ls]          = None,     # Optional column names to use.
            dtypes:     Optional[T_dtype_dict]  = None,     # Optional dtype_dict describing the desired type of each column.
                                                            #   also used to define column names if provided and cols not provided.
            keyfield:   str                     = '',       # A field of the columns to be used as a key.
                                                            # can be set even if columns not set yet.
            name:       str                     = '',       # An optional name of the Daf array.
            use_copy:   bool                    = False,    # If True, make a deep copy of the lol data.
            disp_cols:  Optional[T_ls]          = None,     # Optional list of strings to use for display, if initialized.
            retmode:    str                     = 'obj'     # default return value
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
        
        if use_copy:
            self.lol        = copy.deepcopy(lol)
        else:
            self.lol        = lol
        
        self.kd             = {}
        self.dtypes         = dtypes
        
        self.md_max_rows    = 10    # default number of rows when used with __repr__ and __str__
        self.md_max_cols    = 10    # default number of cols when used with __repr__ and __str__

        self._retmode       = retmode      # return mode can be either RETMODE_OBJ or RETMODE_VAL

        # Initialize iterator variables        
        self._iter_index = 0

        if not cols:
            if dtypes:
                self.hd = {col: idx for idx, col in enumerate(dtypes.keys())}
        else:
            self._cols_to_hd(cols)
            if len(cols) != len(self.hd):
                cols = utils._sanitize_cols(cols=cols)
                self._cols_to_hd(cols)
                if len(cols) != len(self.hd):                
                    import pdb; pdb.set_trace() #temp
                    pass
                    raise AttributeError ("AttributeError: cols not unique")
                
        if self.hd and dtypes:
            effective_dtypes = {col: dtypes.get(col, str) for col in self.hd}
        
            # setting dtypes may be better done manually if required.
            if self.num_cols():

                self.lol = utils.apply_dtypes_to_hdlol((self.hd, self.lol), effective_dtypes)[1]
            
        # rebuild kd if possible.
        self._rebuild_kd()
        
            
    #===========================
    # basic attributes and methods
    
    @property
    def retmode(self):
        return self._retmode
        
    @retmode.setter
    def retmode(self, new_retmode):
        if new_retmode in [self.RETMODE_OBJ, self.RETMODE_VAL]:
            self._retmode = new_retmode
        else:
            raise ValueError("Invalid retmode")

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
        """ test daf for existance and not empty 
            test exists in test_daf.py            
        """
        return bool(self.num_cols())


    def __len__(self):
        """ Return the number of rows in the Daf instance.
        """
        # unit tested
        return len(self.lol)
        
        
    def len(self):
        # unit tested

        return len(self.lol)
        
        
    def shape(self):
        """ return the number of rows and cols in the daf data array
            number of columns is based on the first record
        """
        # test exists in test_daf.py
        
        if not len(self): return (0, 0)
        
        return (len(self.lol), self.num_cols()) 
        
        
    def __eq__(self, other):
        # test exists in test_daf.py            

        if not isinstance(other, Daf):
            return False

        return (self.lol == other.lol and self.columns() == other.columns() and self.keyfield == other.keyfield)

    
    def __str__(self) -> str:
        return self.md_daf_table_snippet()
        
        
    def __repr__(self) -> str:
        return "\n"+self.md_daf_table_snippet()
    

    def num_cols(self) -> int:
        """ return 0 if self.lol is empty.
            return the length of the first row otherwise.
            
            this only works well if the array has rows that are all the same length.            
        """
        # unit tested
    
        if not self.lol:
            return 0
        return len(self.lol[0])
        

    #===========================
    # column names
    def columns(self):
        """ Return the column names 
        """
        # test exists in test_daf.py            
        return list(self.hd.keys())
        
        
    def _cols_to_hd(self, cols: T_ls):
        """ rebuild internal hd from cols provided """
        self.hd = {col:idx for idx, col in enumerate(cols)}
        
        
    @staticmethod
    def isin(listlike1: Union[T_da, T_la], listlike2: Union[T_da, T_la]) -> T_lb:
        """ creates a boolean mask (list of bools) for each item in list1 which is in list2
        
            this can be used particularly for omitting columns, like:
            
                my_daf[:, ~my_daf.columns().isin(colnames_to_omit_list)]
            
            can also be used to select columns
            
                my_daf[:, my_daf.columns().isin(colnames_to_keep_list)]

            but this is easier done by providing the list directly
            
                my_daf[:, colnames_to_keep_list]

            as long as the colnames are not numbers, because then the indexing will 
            assume they are column numbers. So this can be a workaround if the colnames
            are numbers and using them directly can be confusing, but mainly it is used
            to exclude columns. Can be also used for rows, but it is best to use 
            direct selection if possible.
            
            This will directly select rows with the keys selected.
            
                my_daf[rowkeys_to_keep_list]
                
            But can also select with a boolean mask, but it is not as efficient.
            
                my_daf[my_daf.keys().isin(rowkeys_to_keep_list)]
                
            However, that may be good if you just want to exclude rows
            
                my_daf[~my_daf.keys().isin(rowkeys_to_keep_list)]
        
        """
        if isinstance(listlike2, list) and len(listlike1) > 10 and len(listlike2) > 30:
            searchable2 = dict.fromkeys(listlike2)
        else:
            searchable2 = listlike2
        
        bool_mask_lb = [col in searchable2 for col in listlike1]
        
        return bool_mask_lb
                

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
                exclude_types = [exclude_types]
            selected_cols = [col for col in selected_cols if self.dtypes.get(col) not in exclude_types]

        return selected_cols
        
        
    def normalize(self, defined_cols: T_ls):
        # add or subtract columns and place in order per defined_cols.
        # UNUSED
        if not self:
            return
        
        # from utilities import utils

        for irow, da in enumerate(self):
            record_da = utils.set_cols_da(da, defined_cols)
            self.update_record_da_irow(irow, record_da)
            
        return self
        

    def rename_cols(self, from_to_dict: T_ds):
        """ rename columns using the from_to_dict provided. 
            respects dtypes and rebuilds hd
        """
        # unit tests exist
        
        self.hd     = {from_to_dict.get(col, col):idx for idx, col in enumerate(self.hd.keys())}
        self.dtypes = {from_to_dict.get(col, col):typ for col, typ in self.dtypes.items()}
        if self.keyfield:
            self.keyfield = from_to_dict.get(self.keyfield, self.keyfield)
            # no need to rebuild the kd, it should be the same.
            
        return self

    def set_cols(self, new_cols: Optional[T_ls]=None, sanitize_cols: bool=True, unnamed_prefix: str='col'):
        """ set the column names of the daf using an ordered list.
        
            if new_cols is None, then we generate spreadsheet colnames like A, B, C... AA, AB, ...
            
            if sanitize_cols is True (default) then check new_cols for any missing or duplicate names.
                if missing, substitute with {unnamed_prefix}{col_idx}
                if duplicated, substitute with prior_name_{col_idx}
        """
        num_cols = self.num_cols() or len(self.hd)
        
        if new_cols is None:
            new_cols = utils._generate_spreadsheet_column_names_list(num_cols)
            
        elif sanitize_cols:
            new_cols = utils._sanitize_cols(new_cols, unnamed_prefix=unnamed_prefix)
        
        if num_cols and len(new_cols) < num_cols:
            raise AttributeError("Length of new_cols not the same as existing cols")
        
        if self.keyfield and self.hd:
            # if column names are already defined (hd) then we need to repair the keyfield.
            keyfield_idx = self.hd[self.keyfield]
            self.keyfield = new_cols[keyfield_idx]
        
        # set new cols to the hd
        self._cols_to_hd(new_cols)
        
        # convert dtypes dict to use the new names.
        if self.dtypes:
            self.dtypes = dict(zip(new_cols, self.dtypes.values()))
            
        return self
            

    #===========================
    # keyfield
        
    def keys(self):
        """ return list of keys from kd of keyfield
            test exists in test_daf.py            
        """
        
        if not self.keyfield:
            return []
        
        return list(self.kd.keys())
        

    def set_keyfield(self, keyfield: str=''):
        """ set the indexing keyfield to a new column
            if keyfield == '', then reset the keyfield and reset the kd.
            if keyfield not in columns, then KeyError
        """
        if keyfield:
            if keyfield not in self.hd:
                raise KeyError
            self.keyfield = keyfield
            self._rebuild_kd()
        else:
            self.keyfield = ''
            self.kd = {}
            
        return self
    
        
    def _rebuild_kd(self) -> None:
        """ anytime deletions are performed, the kd must be rebuilt 
            if the keyfield is set.
        """
        
        if self.keyfield and self.keyfield in self.hd:
            col_idx = self.hd[self.keyfield]
            self.kd = self.__class__._build_kd(col_idx, self.lol)
            
        return self


    @staticmethod
    def _build_kd(col_idx: int, lol: T_lola) -> T_di:
        """ build key dictionary from col_idx col of lol """
        
        key_col = utils.select_col_of_lol_by_col_idx(lol, col_idx)
        kd = {key: index for index, key in enumerate(key_col)}
        return kd
        
        
    def row_idx_of(self, rowkey: str) -> int:
        """ return row_idx of key provided or -1 if not able to do it.
        """
        # unit tested
        
        if not self.keyfield or not self.kd:
            return -1
        return self.kd.get(rowkey, -1)
        

    def get_existing_keys(self, keylist: T_ls) -> T_ls:
        """ check the keylist against the keys defined in a daf instance. 
        """
        # unit tested
    
        return [key for key in keylist if key in self.kd]


    
    #===========================
    # dtypes
    
    def apply_dtypes(self):
        """ convert columns to the datatypes specified in self.dtypes dict """
        
        self.lol = utils.apply_dtypes_to_hdlol((self.hd, self.lol), self.dtypes)[1]
            
        return self
        
        
    def unflatten_cols(self, cols: T_ls):
        """ 
            given a daf and list of cols, 
            convert cols named to either list or dict if col exists and it appears to be 
                stringified using f"{}" functionality.
                
        """

        if not self:
            return    
            
        # from utilities import utils

        self.hd, self.lol = utils.unflatten_hdlol_by_cols((self.hd, self.lol), cols)    
            
        return self


    def unflatten_by_dtypes(self):

        if not self or not self.dtypes:
            return self
                
        unflatten_cols = self.calc_cols(include_types = [list, dict])
        
        if not unflatten_cols:
            return self
       
        self.unflatten_cols(unflatten_cols)
            
        return self
        
        
    def flatten_cols(self, cols: T_ls):
        # given a daf, convert given list of columns to json.

        if not self:
            return self
        
        # from utilities import utils

        for irow, da in enumerate(self):
            record_da = copy.deepcopy(da)
            for col in cols:
                if col in da:
                    record_da[col] = utils.json_encode(record_da[col])        
            self.update_record_da_irow(irow, record_da)        
            
        return self
    
    
    def flatten_by_dtypes(self):

        if not self or not self.dtypes:
            return self
                
        flatten_cols = self.calc_cols(include_types = [list, dict])
        
        if not flatten_cols:
            return self
       
        self.flatten_cols(cols=flatten_cols)
            
        return self


    def _safe_tofloat(val: Any) -> Union[float, str]:
        try:
            return float(val)
        except ValueError:
            return 0.0
    
    #===========================
    # initializers
    
    def clone_empty(self, lol: Optional[T_lola]=None, cols: Optional[T_ls]=None) -> 'Daf':
        """ Create Daf instance from self, adopting dict keys as column names
            adopts keyfield but does not adopt kd.
            test exists in test_daf.py
            if lol is provided, it is used in the new Daf.
         """
        if self is None:
            return Daf()
            
        new_cols = cols if cols else self.columns()
        
        new_daf = Daf(cols=new_cols, lol=lol, keyfield=self.keyfield, dtypes=copy.deepcopy(self.dtypes))
        
        return new_daf
        
        
    def set_lol(self, new_lol: T_lola) -> 'Daf':
        """ set the lol with the value passed, leaving other settings, 
            and recalculating kd if required (i.e. if keyfield is defined).
        """
        
        self.lol = new_lol
        self._rebuild_kd()
        
        return self
        

    #===========================
    # convert from / to other data or files.
    
    # ==== Python lod (list of dictionaries)
    @classmethod
    def from_lod(
            cls,
            records_lod:    T_loda,                         # List[List[Any]] to initialize the lol data array.
            keyfield:       str='',                         # set a new keyfield or set no keyfield.
            dtypes:         Optional[T_dtype_dict]=None     # set the data types for each column.
            ) -> 'Daf':
        """ Create Daf instance from loda type, adopting dict keys as column names
            Generally, all dicts in records_lod should be the same OR the first one must have all keys
                and others can be missing keys.
        
            test exists in test_daf.py
            
            my_daf = Daf.from_lod(sample_lod)
        """
        if dtypes is None:
            dtypes = {}
        
        if not records_lod:
            return cls(keyfield=keyfield, dtypes=dtypes)
        
        cols = list(records_lod[0].keys())
        
        # from utilities import utils
        
        lol = [list(utils.set_cols_da(record_da, cols).values()) for record_da in records_lod]
        
        return cls(cols=cols, lol=lol, keyfield=keyfield, dtypes=dtypes)
        
        
    def to_lod(self) -> T_loda:
        """ Create lod from daf
            test exists in test_daf.py
        """
        
        if not self:
            return []

        cols = self.columns()
        result_lod = [dict(zip(cols, la)) for la in self.lol]
        return result_lod
        
    
    # ==== Python dod (dict of dict)    
    @classmethod
    def from_dod(
            cls,
            dod:            T_doda,         # Dict(str, Dict(str, Any))
            keyfield:       str='rowkey',   # The keyfield will be set to the keys of the outer dict.
                                            # this will set the preferred name. Defaults to 'rowkey'
            dtypes:         Optional[T_dtype_dict]=None     # optionally set the data types for each column.
            ) -> 'Daf':
            
        """ a dict of dict (dod) structure is very similar to a Daf table, but there is a slight difference.
            A dod structure will have a first key which indexes to a specific dict.
            The key in that dict is likely not also found in the "value" dict of the first level, but it might be.
            
            a Daffodil table always has the keys of the outer dict as items in each table.
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
            
            If dod2 is passed, it will be convered to dod1 and then converted to daf instance.
            
            A Daf table is able 1/3 the size of an equivalent dod. because the column keys are not repeated.
            
            use to_dod() to recover the original form by setting 'remove_rowkeys'=True if the row keys are
            not required in the dod.
            
        """
        return cls.from_lod(utils.dod_to_lod(dod, keyfield=keyfield), keyfield=keyfield, dtypes=dtypes)
    
    
    def to_dod(
            self,
            remove_keyfield:    bool=True,      # by default, the keyfield column is removed.
            ) -> T_doda:
        """ a dict of dict structure is very similar to a Daf table, but there is a slight difference.
            a Daf table always has the keys of the outer dict as items in each table.
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
    

    # ==== cols_dol
    @classmethod
    def from_cols_dol(
            cls, 
            cols_dol: T_dola, 
            keyfield: str='', 
            dtypes: Optional[T_dtype_dict]=None,
            ) -> 'Daf':
        """ Create Daf instance from cols_dol type, adopting dict keys as column names
            and creating columns from each value (list)
            
            my_daf = Daf.from_cols_dol({'A': [1,2,3], 'B': [4,5,6], 'C': [7,8,9])
            
            produces:
                my_daf.columns() == ['A', 'B', 'C']
                my_daf.lol == [[1,4,7], [2,5,8], [3,6,9]] 
            
            
        """
        if dtypes is None:
            dtypes = {}
        
        if not cols_dol:
            return cls(keyfield=keyfield, dtypes=dtypes)
        
        cols = list(cols_dol.keys())
        
        lol = []
        for irow in range(len(cols_dol[cols[0]])):
            row = []
            for col in cols:
                row.append(cols_dol[col][irow])
            lol.append(row)    
        
        return cls(cols=cols, lol=lol, keyfield=keyfield, dtypes=dtypes)
        
        
    def to_cols_dol(self) -> dict:
        """ convert daf to dictionary of lists of values, where key is the 
            column name, and the list are the values in that column.
        """
        result_dol = {colname: [] for colname in self.columns()}
        
        for row_da in self:
            for key, val in row_da.items():
                result_dol[key].append(val)
                
        return result_dol
        

    def to_attrib_dict(self) -> dict:
        """
        Convert Daf instance to a dictionary representation.
        The dictionary has two keys: 'cols' and 'lol'.
        
        DEPRECATED

        Example:
        {
            'cols': ['A', 'B', 'C'],
            'lol': [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
        }
        """
        return {'cols': self.columns(), 'lol': self.lol}

    

    @classmethod
    def from_lod_to_cols(
            cls,
            lod: T_loda, 
            cols:Optional[List]=None, 
            keyfield: str='', 
            dtypes: Optional[T_dtype_dict]=None
            ) -> 'Daf':
            
        r""" Create Daf instance from a list of dictionaries to be placed in columns
            where each column shares the same keys in the first column of the array.
            This transposes the data from rows to columns and adds the new 'cols' header,
            while adopting the keys as the keyfield. dtypes is applied to the columns
            transposition and then to the rows.
            
            If no 'cols' parameter is provided, then it will be the name 'key' 
            followed by normal spreadsheet column names, like 'A', 'B', ... 
            
            Creates a daf where the first column are the keys from the dicts,
            and each subsequent column are each of the values of the dicts.
            
            my_daf = Daf.from_coldicts_lod( 
                cols = ['Feature', 'Try 1', 'Try 2', 'Try 3'],
                lod =       [{'A': 1, 'B': 2, 'C': 3},          # data for Try 1
                             {'A': 4, 'B': 5, 'C': 6},          # data for Try 2
                             {'A': 7, 'B': 8, 'C': 9} ]         # data for Try 3
            
            produces:
                my_daf.columns() == ['Feature', 'Try 1', 'Try 2', 'Try 3']
                my_daf.lol ==        [['A',       1,       4,       7], 
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
            return cls(keyfield=keyfield, dtypes=dtypes, cols=cols)
        
        # the following will adopt the dictionary keys as cols.
        # note that dtypes applies to the columns in this orientation.
        rows_daf = cls.from_lod(lod, dtypes=dtypes)
        
        # this transposes the entire dataframe, including the column names, which become the first column
        # in the new orientation, then adds the new column names, if provided. Otherwise they will be
        # defined as ['key', 'A', 'B', ...]
        cols_daf = rows_daf.transpose(new_keyfield = keyfield, new_cols = cols, include_header = True)
        
        return cols_daf


    #==== Excel
    @classmethod
    def from_excel_buff(
            cls,
            excel_buff: bytes, 
            keyfield: str='',                       # field to use as unique key, if not ''
            dtypes: Optional[T_dtype_dict]=None,    # dictionary of types to apply if set.
            noheader: bool=False,                   # if True, do not try to initialize columns in header dict.
            user_format: bool=False,                # if True, preprocess the file and omit comment lines.
            unflatten: bool=True,                   # unflatten fields that are defined as dict or list.
            ) -> 'Daf':
        """ read excel file from a buffer and convert to daf.
        """
        
        # from utilities import xlsx_utils

        csv_buff = utils.xlsx_to_csv(excel_buff)

        my_daf  = cls.from_csv_buff(
                        csv_buff, 
                        keyfield    = keyfield,         # field to use as unique key, if not ''
                        dtypes      = dtypes,           # dictionary of types to apply if set.
                        noheader    = noheader,         # if True, do not try to initialize columns in header dict.
                        user_format = user_format,      # if True, preprocess the file and omit comment lines.
                        unflatten   = unflatten,        # unflatten fields that are defined as dict or list.
                        )
        
        return my_daf
    
    #==== CSV
    @classmethod
    def from_csv_buff(
            cls,
            csv_buff: Union[bytes, str],            # The CSV data as bytes or string.
            keyfield: str='',                       # field to use as unique key, if not ''
            dtypes: Optional[T_dtype_dict]=None,    # dictionary of types to apply if set.
            noheader: bool=False,                   # if True, do not try to initialize columns in header dict.
            user_format: bool=False,                # if True, preprocess the file and omit comment lines.
            sep: str=',',                           # field separator.
            unflatten: bool=True,                   # unflatten fields that are defined as dict or list.
            include_cols: Optional[T_ls]=None,      # include only the columns specified. noheader must be false.
            ) -> 'Daf':
        """
        Convert CSV data in a buffer (bytes or string) to a daf object

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
        
        data_lol = utils.buff_csv_to_lol(csv_buff, user_format=user_format, sep=sep, include_cols=include_cols, dtypes=dtypes)
        
        cols = []
        if not noheader:
            cols = data_lol.pop(0)        # return the first item and shorten the list.
        
        my_daf = cls(lol=data_lol, cols=cols, keyfield=keyfield, dtypes=dtypes)
        
        if unflatten:
            my_daf.unflatten_by_dtypes()
   
        return my_daf
    


    def to_csv_file(
            self,
            file_path: str='',
            line_terminator: Optional[str]=None,
            include_header: bool=True,
            ) -> str:

        buff = self.to_csv_buff(
                line_terminator=line_terminator,
                include_header=include_header,
                )

        self.__class__.buff_to_file(buff, file_path=file_path, fmt='.csv')
        
        return file_path


    def to_csv_buff(
            self, 
            line_terminator: Optional[str]=None,
            include_header: bool=True,
            ) -> T_buff:
        """ this function writes the daf array to a csv buffer, including the header if include_header==True.
            The buffer can be saved to a local file or uploaded to a storage service like s3.
        """
    
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


    @staticmethod
    def buff_to_file(buff: T_buff, file_path: str, fmt:str='.csv'):
    
        return utils.write_buff_to_fp(buff, file_path, fmt=fmt)

    
    #==== Pandas
    #@classmethod
    from_pandas_df = daf_pandas._from_pandas_df
                
    to_pandas_df = daf_pandas._to_pandas_df

        
            
    #==== Numpy
    @classmethod
    def from_numpy(cls, npa: Any, keyfield:str='', cols:Optional[T_la]=None, name:str='') -> 'Daf':
        """
        Convert a Numpy dataframe to daf object
        The resulting Python list will contain Python native types, not NumPy types.
        
        Numpy arrays are homogeneous, meaning all elements in a numpy array 
        must have the same data type. If you attempt to create a numpy array 
        with elements of different data types, numpy will automatically cast 
        them to a single data type that can accommodate all elements. This can 
        lead to loss of information if the original data types are different.
        For example, if you try to create a numpy array with both integers and 
        strings, numpy will cast all elements to a common data type, such as Unicode strings.

        """
        # import numpy as np
        
        if npa.ndim == 1:
            lol = [npa.tolist()]
        else:
            lol = npa.tolist()
    
        return cls(cols=cols, lol=lol, keyfield=keyfield, name=name)
    

    def to_numpy(self) -> Any:
        """ 
        Convert the core array of a Daf object to numpy.
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
        

    @classmethod
    def from_hllola(cls, hllol: T_hllola, keyfield: str='', dtypes: Optional[T_dtype_dict]=None) -> 'Daf':
        """ Create Daf instance from hllola type.
            This is used for all DB. loading.
            test exists in test_daf.py
            
            DEPRECATED
        """
        
        hl, lol = hllol
        
        return cls(lol=lol, cols=hl, keyfield=keyfield, dtypes=dtypes)
        
        
    #==== Googlesheets
    
    @classmethod
    def from_googlesheet(cls, spreadsheet_id: str, sheetname: str = 'Sheet1') -> 'Daf':
        from googleapiclient.discovery import build
        from google.oauth2 import service_account

        """
        Read data from a Google Sheet specified by its ID.
        
        Args:
            spreadsheet_id (str): The ID of the Google Sheet.
            
        Returns:
            Daf instance
            
        """
        
        # Set up credentials for the Google Sheets API
        SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
        SERVICE_ACCOUNT_FILE = 'path/to/your/service_account.json'
        
        creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        service = build('sheets', 'v4', credentials=creds)
        
        # Specify the range from which to read data (all values)
        range_name = sheetname 
        
        # Call the Sheets API to get values from the specified range
        result = service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=range_name
        ).execute()
        
        lol = result.get('values', [])
        
        # if not lol:
            # print('No data found in the Google Sheet.')
            # return None
        # else:
            # return values

        #num_rows = len(lol)
        num_cols = 0 if not lol else len(lol[0])
        
        cols = utils._generate_spreadsheet_column_names_list(num_cols)

        gs_daf = cls(cols=cols, lol=lol)
        
        return gs_daf
        
        
    def to_googlesheet(self, spreadsheet_id: str, sheetname: str = 'Sheet1') -> 'Daf':
        """ export data from daf structure to googlesheet. """
    
        from googleapiclient.discovery import build
        from google.oauth2 import service_account

        # Set up credentials for the Google Sheets API
        SCOPES = ['https://www.googleapis.com/auth/spreadsheets']       # this might be okay.
        SERVICE_ACCOUNT_FILE = 'path/to/your/service_account.json'      # probably wrong.

        creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        service = build('sheets', 'v4', credentials=creds)

        # Define your list of lists array (Daf)
        # self.lol

        # Define the range where you want to write the data (e.g., Sheet1!A1:C4)
        # get the column name of the last column in the array.
        num_cols = self.num_cols()
        last_col_idx = num_cols - 1
        last_spreadsheet_colname = Daf._calculate_single_column_name(last_col_idx)
        
        range_name = f"{sheetname}!A1:{last_spreadsheet_colname}{len(self.lol)}"
        
        # Build the request body
        body = {
            'values': self.lol
        }

        # Call the Sheets API to update the data in the specified range
        request = service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=range_name,
            valueInputOption='RAW',     # Question: does this provide formulas or just numbers.
            body=body
        )

        response = request.execute()
        # parse response and detect if there was an error.
        
        response = response     # fool linter

        print('Data successfully written to Google Sheets.')
        
        return self


    #===========================
    # convert to other format
    
    def to_hllola(self) -> T_hllola:
        """ Create hllola from daf 
            test exists in test_daf.py
            
            DEPRECATED
        """    
        return (list(self.hd.keys()), self.lol)
        
    #===========================
    # append
        
    def append(self, data_item: Union[T_Daf, T_loda, T_da, T_la]):
        """ general append method can handle appending one record as T_da or T_la, many records as T_loda or T_daf
        """
        # test exists in test_daf.py for all three cases
        
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
                
        elif isinstance(data_item, Daf):  # type: ignore
            self.concat(data_item)
        else:    
            raise RuntimeError    # pragma: no cover
            
        return self
        

    def concat(self, other_instance: 'Daf'):
        """ concatenate records from passed daf cls to self daf 
            This directly modifies self
            if keyfield is '', then insert without respect to the key value
            otherwise, allow only one record per key.
            columns must be equal.
            test exists in test_daf.py
        """
        
        if not other_instance:
            return 
            
        diagnose = False

        if diagnose:      # pragma: no cover
            print(f"self=\n{self}\ndaf=\n{other_instance}")
            
        if not self.lol and not self.hd:
            
            self.hd = other_instance.hd
            self.lol = other_instance.lol
            self.kd = other_instance.kd
            self.keyfield = other_instance.keyfield
            self._rebuild_kd()   # only if the keyfield is set.
            return self
            
        # fields must match exactly!
        if self.hd != other_instance.hd:
            raise KeyError ("keys in daf do not match lod keys")
        
        # simply append the rows from daf.lol to the end of self.lol
        for idx in range(len(other_instance.lol)):
            rec_la = other_instance.lol[idx]
            self.lol.append(rec_la)
        self._rebuild_kd()   # only if the keyfield is set.

        if diagnose:  # pragma: no cover
            print(f"result=\n{self}")
            
        return self
                

    def extend(self, records_lod: T_loda):
        """ append lod of records into daf 
            This directly modifies daf
            if keyfield is '', then insert without respect to the key value
            otherwise, allow only one record per key.
            test exists in test_daf.py
         """
        
        if not records_lod or len(records_lod) == 1 and not records_lod[0]:
            return self
            
        if not self.lol and not self.hd:
            # new daf, adopt structure of lod.
            # but there is only one header and data is lol
            # this saves space.
            
            self.hd = {col_name: index for index, col_name in enumerate(records_lod[0].keys())}
            self.lol = [list(record_da.values()) for record_da in records_lod]
            self._rebuild_kd()   # only if the keyfield is set.
            return self
            
        for record_da in records_lod:
            # this test done inside record_append()
            # if not record_da:
                # # do not append any records that are empty.
                # continue
            
            # the following will either append or insert
            # depending on the keyvalue.
            self.record_append(record_da)    
            
        return self            
            

    def record_append(self, record_da: T_da):
        """ perform append of one record into daf (T_da is Dict[str, Any]) 
            This directly modifies daf
            if keyfield is '', then insert without respect to the key value
            otherwise, allow only one record per key.
            
            if the daf is empty, it will adopt the structure of record_da.
            Each new append will add to the end of the daf.lol and will
            update the kd.
            
            If the keys in the record_da have a different order, they will
            be reordered and then appended correctly.
        """
            # test exists in test_daf.py
        
        if not record_da:
            return self
            
        if not self.lol and not self.hd:
            # new daf, adopt structure of lod.
            # but there is only one header and data is lol
            # this saves space.
            
            self.hd = {col_name: index for index, col_name in enumerate(record_da.keys())}
            self.lol = [list(record_da.values())]
            self._rebuild_kd()   # only if the keyfield is set.
            return self
            
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
            
        return self


    def _basic_append_la(self, rec_la: T_la, keyval: str):
        """ basic append to the end of the array without any checks
            including appending to kd and la to lol
        """
        self.kd[keyval] = len(self.lol)
        self.lol.append(rec_la)
            
        return self
                

    #=========================
    # remove records per keyfield; drop cols

    def remove_key(self, keyval:str, silent_error=False) -> None:
        """ remove record from daf using keyfield
            This directly modifies daf
        """
        # test exists in test_daf.py
        
        return self.select_krows(krows=keyval, inverse=True, silent_error=silent_error)

        # if not self.keyfield:
            # return self
        
        # try:
            # key_idx = self.kd[keyval]   #will raise KeyError if key not exists.
        # except KeyError:
            # if silent_error: 
                # return self
            # raise
            
        # self.lol.pop(key_idx)
        # self._rebuild_kd()
        # return self
        
    
    def remove_keylist(self, keylist: T_ls, silent_error=False):
        """ remove records from daf using keyfields
            This directly modifies daf
            test exists in test_daf.py
        """

        return self.select_krows(krows=keylist, inverse=True, silent_error=silent_error)

        # # get the indexes of rows to be deleted.
        # idx_li: T_li = []
        # for keyval in keylist:
            # try:
                # idx = self.kd[keyval]
            # except KeyError:
                # if silent_error: continue
                # raise
            # idx_li.append(idx)    
                
        # # delete records from the end so the indexes are valid after each deletion.
        # reverse_sorted_idx_li = sorted(idx_li, reverse=True)

        # for idx in reverse_sorted_idx_li:
            # self.lol.pop(idx)

        # self._rebuild_kd()
            
        # return self
        
        
    #===========================
    # indexing -- these methods have substantial complexity and are provided
    #               in the companion file indexing.py

    def __getitem__(self,
            slice_spec:   Union[slice, int, str, T_li, T_ls,  
                                Tuple[  Union[slice, int, str, T_li, T_ls, Tuple[str, str]], 
                                        Union[slice, int, str, T_li, T_ls, Tuple[str, str]]]],
            ) -> Any:

        if isinstance(slice_spec, tuple) and len(slice_spec) == 2:
            # Handle parsing slices for  both rows and columns
            row_spec, col_spec = slice_spec
        else:
            row_spec = slice_spec
            col_spec = None
            
        if col_spec == slice(None, None, None):
            col_spec = None
            
        if row_spec == slice(None, None, None):    
            row_spec = None
            
        if isinstance(row_spec, (int, slice)) or utils.is_list_of_type(row_spec, int):
            sel_rows_daf = self.select_irows(irows=row_spec)
            
        elif isinstance(row_spec, str) or utils.is_list_of_type(row_spec, str):
            sel_rows_daf = self.select_krows(krows=row_spec)
        else:
            sel_rows_daf = self

        if col_spec is None:
            ret_daf = sel_rows_daf
        else:
            if isinstance(col_spec, (int, slice)) or  utils.is_list_of_type(col_spec, int):
                ret_daf = sel_rows_daf.select_icols(icols=col_spec)
                
            elif isinstance(col_spec, str) or utils.is_list_of_type(col_spec, str):
                ret_daf = sel_rows_daf.select_kcols(kcols=col_spec)
            else:
                ret_daf = sel_rows_daf
            
        return ret_daf._adjust_return_val(self.retmode)    
    
        
    def __setitem__(self,
            slice_spec:   Union[slice, int, str, T_li, T_ls, T_lb,
                                Tuple[  Union[slice, int, str, T_li, T_ls, T_lb, Tuple[Any, Any]], 
                                        Union[slice, int, str, T_li, T_ls, T_lb, Tuple[Any, Any]]]],
            value: Any,
            ) -> 'Daf':

        if isinstance(slice_spec, tuple) and len(slice_spec) == 2:
            # Handle parsing slices for  both rows and columns
            row_spec, col_spec = slice_spec
        else:
            row_spec = slice_spec
            col_spec = None
            
        if row_spec == slice(None, None, None):
            row_spec = None
            irows = list(range(len(self)))
            
        if isinstance(row_spec, str) or utils.is_list_of_type(row_spec, str) or isinstance(row_spec, tuple):
            irows = self.krows_to_irows(krows = row_spec)
            
        elif isinstance(row_spec, (int, slice)) or utils.is_list_of_type(row_spec, int):
            irows = row_spec
            
        if col_spec and isinstance(col_spec, str) or utils.is_list_of_type(col_spec, str) or isinstance(col_spec, tuple):
            icols = self.kcols_to_icols(kcols = col_spec)
            
        elif isinstance(col_spec, (int, slice)) or utils.is_list_of_type(col_spec, int):
            icols = col_spec
        else:
            icols = None
            
        return self.set_irows_icols(irows=irows, icols=icols, value=value)


    def _adjust_return_val(self, retmode: str = ''):
        """
            There is currently defined two ways to return data from __getitem__ which
            is controlled by the _retmode property setting. 
            
            RETMODE_OBJ: return a full daf object.
            RETMODE_VAL: return just the value, when possible.
            
            It is helpful to just get a single column as a list, for example,
            instead of returning the entire array.
            
            This implementation is after the fact, and results in additional
            processing, but it is also feasible with this design to avoid 
            creating the intervening daf structure and improve efficiency.
        """
        if not retmode:
            retmode = self.retmode
        
        if retmode == self.RETMODE_OBJ:
            # do nothing in this case.
            return self
        
        num_rows, num_cols = self.shape()

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
        

    def set_irows_icols(self, irows: Union[slice, int, T_li, None], icols: Union[slice, int, T_li, None], value) -> 'Daf':
        """ set rows and cols in given daf.
            
            irows, icols: can be either a slice, int, or list of integers. These
                            refer to row/col indices that are inherent in the lol structure.
            
            mutates existing daf
        """
        if icols is None:
            icols = []
        if irows is None:
            irows = []
        
        num_irows = utils.len_rowcol_spec(irows)
        num_icols = utils.len_rowcol_spec(icols)
        
        if num_irows == 1 and isinstance(irows, int):
            irows = [irows]
        if num_icols == 1 and isinstance(icols, int):
            icols = [icols]
            
        if isinstance(irows, slice):
            irows = utils.slice_to_range(irows, len(self))
        if isinstance(icols, slice):
            icols = utils.slice_to_range(icols, self.num_cols())
            
        # special case when cols not specified.    
        if num_irows == 1 and num_icols == 0:
        
            irow = irows[0]
            
            if isinstance(value, list):
                self.lol[irow] = value
            elif isinstance(value, dict):
                self.assign_record_da_irow(irow, record_da=value)
            elif isinstance(value, self.__class__):
                self.lol[irow] = value
            else:
                # set the same value in the row for all columns.
                self.lol[irow] = [value] * len(self.lol[irow])
                
        if num_irows == 1 and num_icols == 1:
        
            irow = irows[0]
            icol = icols[0]
            
            if isinstance(value, dict):
                self.assign_record_da_irow(irow, record_da=value)
            else:    
                self.lol[irow][icol] = value
                
        elif num_irows > 1 and num_icols == 0:
            
            if isinstance(value, list):
                for irow in irows:
                    self.lol[irow] = value
            elif isinstance(value, dict):
                for irow in irows:
                    self.assign_record_da_irow(irow, record_da=value)
            elif isinstance(value, self.__class__):
                for source_row, irow in enumerate(irows):
                    self.lol[irow] = value[source_row]
            else:
                # set the same value in the row for all columns.
                for irow in irows:
                    self.lol[irow] = [value] * len(self.lol[irow])
                
        elif num_irows > 0 and num_icols == 1:
        
            icol = icols[0]
        
            if irows is None:
                irows = range(len(self.lol))
        
            if isinstance(value, list):
                for source_idx, irow in enumerate(irows):
                    try:
                        self.lol[irow][icol] = value[source_idx]
                    except Exception:
                        import pdb; pdb.set_trace() #temp
                    
            # elif isinstance(value, dict):
                # # this is the same as cols=0 bc dict updates the corresponding cols.
                # for irow in irows:
                    # self.assign_record_da_irow(irow, record_da=value)
                
            elif isinstance(value, self.__class__):
                for source_idx, irow in enumerate(irows):
                    self.lol[irow][icol] = value[source_idx][0]
                
            else:
                # set the same value in the row for all selected columns.
                for irow in irows:
                    self.lol[irow][icol] = value
        else:
            if irows is None:
                irows = range(len(self.lol))
        
            if isinstance(value, list):
                for irow in irows:
                    for source_col, icol in enumerate(icols):
                        try:
                            self.lol[irow][icol] = value[source_col]
                        except Exception:
                            import pdb; pdb.set_trace() #temp
                    
            elif isinstance(value, dict):
                # this is the same as cols=0 bc dict updates the corresponding cols.
                for irow in irows:
                    self.assign_record_da_irow(irow, record_da=value)
                
            elif isinstance(value, self.__class__):
                for irow in irows:
                    for source_col, icol in enumerate(icols):
                        self.lol[irow][icol] = value[source_col]
                
            else:
                # set the same value in the row for all selected columns.
                for irow in irows:
                    for icol in icols:
                        self.lol[irow][icol] = value
        return self
    
    
    def krows_to_irows(self, 
            krows: Union[slice, str, T_la, int, Tuple[Any, Any], None],
            inverse: bool = False,
            silent_error: bool=False,
            ) -> Union[slice, int, T_li]:
        """
        If the keyfield is set, then the rows can be selected by providing a
        krows parameter that will index the rows by using values in the keyfield column.
        The keyfield column is read as a list and then converted to a dictionary that
        provides the indexes of the row for each value in that column. Lookups using
        this method are very fast but there is overhead to reading the column and
        creating the dictionary. Therefore, set keyfield to '' to disable row key lookups.        
        """
        if not self.keyfield or not self.kd:
            if inverse:
                return list(range(len(self)))
            else:
                return []
            
        return self.__class__.gkeys_to_idxs(
                    keydict = self.kd,
                    gkeys = krows,
                    inverse = inverse,
                    silent_error=silent_error,
                    )
    
    def kcols_to_icols(self, 
            kcols: Union[str, T_ls, slice, int, T_li, Tuple[Any, Any], None] = None,
            inverse: bool = False,
            silent_error: bool=False,
            ) -> Union[slice, int, T_li, None]:
        """
        If cols are defined is set, then the cols can be selected by providing a
        kcols parameter that will index the cols by using values in the header dict hd.
        This forces an attempt to use the spec as a column name even if it may look
        like an integer.
        """
        if not self.hd:
            if inverse:
                return list(range(self.num_cols()))
            else:
                return []
        
        return self.__class__.gkeys_to_idxs(
                    keydict = self.hd,
                    gkeys = kcols,
                    inverse = inverse,
                    silent_error=silent_error,
                    )
                    
    @staticmethod
    def gkeys_to_idxs(
            keydict: Dict[Union[str, int], int],
            gkeys: Union[str, T_ls, slice, int, T_li, Tuple[Any, Any], None] = None,
            inverse: bool = False,
            silent_error: bool=False,
            ) -> Union[slice, int, T_li, None]:
        """
        If keydict is defined, then the idxs can be selected by providing a
        gkeys parameter that will index the keydict, and return either a 
        slice, int, T_li, or None, which will index the range.
        """
        
        if not keydict:
            return []
            
        elif isinstance(gkeys, (str, int)):
            idxs = []
            idx = keydict.get(gkeys, -1)
            if idx >= 0:
                idxs.append(idx)
            elif not silent_error:
                raise KeyError
            
        elif isinstance(gkeys, list):     # can be list of integer or strings (or anything hashable)
            idxs = []
            for gkey in gkeys:
                idx = keydict.get(gkey, -1)
                if idx >= 0:
                    idxs.append(idx)
                elif not silent_error:
                    raise KeyError
                    
            
        elif isinstance(gkeys, slice):     # can be list of integer or strings (or anything hashable)
            gkeys_range = range(gkeys.start or 0, gkeys.stop or len(keydict), gkeys.step or 1)
            idxs = []
            for gkey in gkeys_range:
                idx = keydict.get(gkey, -1)
                if gkey >= 0:
                    idxs.append(idx)
                elif not silent_error:
                    raise KeyError
         
        elif isinstance(gkeys, tuple):         
            
            # key_range will return a slice.
            start_idx = 0    
            if gkeys:
                start_idx = keydict.get(gkeys[0], 0)
                
            stop_idx = len(keydict)    
            if len(gkeys) > 1:
                stop_idx = keydict.get(gkeys[0], stop_idx - 1) + 1
                    
            idxs_slice = slice(start_idx, stop_idx, 1)
            idxs = idxs_slice
            
        if inverse:
            if isinstance(idxs, slice):
                idxs = utils.slice_to_list(idxs)
            
            idxs = [idx for idx in range(len(keydict)) if idx not in idxs]
            
        return idxs
        

    def select_krows(self, 
            krows: Union[slice, str, T_la, int, Tuple[Any, Any], None], 
            inverse: bool=False,
            silent_error: bool=False,
            ) -> 'Daf':
    
        irows = self.krows_to_irows( 
            krows = krows,
            inverse = inverse,
            silent_error = silent_error,
            )
        return self.select_irows(irows)
    
    
    def select_kcols(self, 
            kcols: Union[slice, str, T_la, int, Tuple[Any, Any], None], 
            inverse: bool=False, 
            flip: bool=False,
            silent_error: bool=False,
            ) -> 'Daf':
    
        icols = self.kcols_to_icols( 
            kcols = kcols,
            inverse = inverse,
            silent_error = silent_error,
            )
        return self.select_icols(icols, flip=flip)
    
    
    def select_irows(self, irows: Union[slice, int, T_li, None]) -> 'Daf':
        """ select rows from daf and return a new instance.
            This is an efficient opeation. The array in the new instance
            uses references to selected rows in the original array.
            
            irows: can be either a slice, int, or list of integers. These
                    refer to row indices that are inherent in the lol structure.
            
            returns a new daf instance cloned from the original.
        """
        row_sliced_lol = self.lol
        
        if isinstance(irows, int):
            # simple single row selection
            row_sliced_lol = [self.lol[irows]]
        
        elif isinstance(irows, list):
            if irows and isinstance(irows[0], int):
                # list of integers:
                row_sliced_lol = [self.lol[i] for i in irows]
            else:
                row_sliced_lol = []
            
        elif isinstance(irows, slice):
            slice_spec = irows
            row_sliced_lol = self.lol[slice_spec]
            
        return self.clone_empty(lol=row_sliced_lol)
    
    
    def select_icols(self, icols: Union[slice, int, T_li, None], flip: bool=False) -> 'Daf':
        """ select cols from daf and return a new instance.
            This is not an efficient operation and can normally be avoided except when:
                reading/writing data, then columns may need to be dropped.
                exporting a portion of the array to NumPy, for example.
                
            instead, use the cols parameter to select the columns included
                in operations like apply() and reduce()
            
            icols: can be either a slice, int, or list of integers. These
                    refer to column indices that are inherent in the lol structure.
                    
            flip: if True, then the columns selected are turned into rows.
                    This is a transposition operation. Otherwise, the columns selected
                    remain as columns in a new daf instance. There is no additional cost
                    to transpose if it is done when the columns are selected.
            
            returns a new daf instance cloned from the original,
                with cols, keyfield, and dtypes adjusted to be reasonable.
                If flip=True, then colnames=[], dtypes={} and keyfield='' (inactive)
                
        """
        orig_cols = list(self.hd.keys())    # may be an empty list if colnames not defined.
        orig_dtypes = self.dtypes or {}
        sliced_cols = []
        
        if isinstance(icols, int):
            # simple single row selection
            icol = icols
            if not flip:
                col_sliced_lol = [[row[icol]] for row in self.lol]
                if orig_cols:
                    sliced_cols = orig_cols[icol]
            else: # flip
                col_sliced_lol = [row[icol] for row in self.lol]
            
        elif isinstance(icols, slice):
            slice_spec = icols
            icols_range = range(slice_spec.start or 0, slice_spec.stop or 0, slice_spec.step or 1)
            
            if not flip:
                col_sliced_lol = [[row[icol] for icol in icols_range]
                                        for row in self.lol]
                if orig_cols:
                    sliced_cols = [orig_cols[icol] for icol in icols_range]
            else: # flip
                col_sliced_lol = [[row[icol] for row in self.lol]
                                        for icol in icols_range]
                
        elif isinstance(icols, list) and icols and isinstance(icols[0], int):
            # list of integers:
            if not flip:
                col_sliced_lol = [[row[icol] for icol in icols]
                                        for row in self.lol]
                if orig_cols:                        
                    sliced_cols = [orig_cols[icol] for icol in icols]
                
            else: # flip
                col_sliced_lol = [[row[icol] for row in self.lol]
                                        for icol in icols]
        else:
            if not flip:
                return self
            else: # flip
                col_sliced_lol = [[row[icol] for row in self.lol]
                                        for icol in range(self.num_cols())]
                sliced_cols = []
            
        # fix up the dtypes and reset the keyfield if it is no longer in the daf.
        if sliced_cols:
            new_dtypes = {col:orig_dtypes[col] for col in orig_dtypes if col in sliced_cols}
            new_keyfield = self.keyfield if self.keyfield in sliced_cols else ''    
        else:
            new_dtypes = {}
            new_keyfield = ''    
            
        new_daf = Daf(  cols=sliced_cols, 
                        lol=col_sliced_lol, 
                        keyfield=new_keyfield,
                        dtypes=new_dtypes,
                        )
        
        return new_daf
    
    #=============================================
    # the following methods might be absorbed into the above.
    #
    
    def select_record_da(self, key: str) -> T_da:
        """ Select one record from daf using the key and return as a single T_da dict.
        
            May be better to simply use 
            
                selected_daf = select_krows(krows=key)
                
            to select one row from the array, and then use:

                selected_daf[0, 'fieldname']   to select fields like you would when using a dict.
        
            test exists in test_daf.py
        """
        
        if not self.keyfield:
            raise RuntimeError
            
        row_idx = self.kd.get(key, -1)
        if row_idx < 0:
            return {}
        
        record_da = self._basic_get_record_da(row_idx)
        
        return record_da
        
    def _basic_get_record_da(self, irow: int, include_cols: Optional[T_ls]=None) -> T_da:
        """ return a record at irow as dict 
            include only "include_cols" if it is defined
            note, this requires that hd is defined.
            if no column header exists, this will generate a new one using spreadsheet convention.
        """
        if not self.hd:
            cols = utils._generate_spreadsheet_column_names_list(num_cols=self.num_cols())
            self.set_cols(cols)
        
        if include_cols:
            return {col:self.lol[irow][self.hd[col]] for col in include_cols if col in self.hd}
        else:
            return dict(zip(self.hd, self.lol[irow]))

        
    # def select_irows(self, irows_li: T_li) -> 'Daf':
        # """ Select multiple records from daf using row indexes and create new daf.
            
        # """
        
        # selected_daf = self.clone_empty()
        
        # for row_idx in irows_li:
            # record_da = self._basic_get_record_da(row_idx)
        
            # selected_daf.append(record_da)
                      
        # return selected_daf
        
        
    def select_records_daf(self, keys_ls: T_ls, inverse:bool=False) -> 'Daf':
        """ Select multiple records from daf using the keys and return as a single daf.
            If inverse is true, select records that are not included in the keys.
            
        """
        return self.select_krows(krows=keys_ls, inverse=inverse)
          
        
    def irow_la(self, irow: int) -> T_la:
        """ return a row as a list.
        """
        return self.lol[irow]
        

    def to_value(self, irow: int=0, icol: int=0, default:Any='') -> Any:
        """ return a single value from an array,
            at default location 0,0 or as specified.
        """
    
        num_rows, num_cols = self.shape()

        if num_rows >= 1 and num_cols >= 1:
            return self.lol[irow][icol]

        return default
        

    def to_list(self, irow: Optional[int]=None, icol: Optional[int]=None, unique=False) -> list:
        """ return data from a daf array as a list
            defaults to the most obvious list if irow and icol not specified.
                from irow 0, if num_rows is 1 and num_cols >= 1
                from icol 0, if num_rows >= 1 and num_cols == 1
            otherwise, choose irow or icol specified.
                if irow, specified, ignore icol.
                if irow=None and icol specified, then use icol.
        """
    
        num_rows, num_cols = self.shape()

        if irow is None and icol is None:
            if num_rows == 1 and num_cols >= 1:
                # single row, return as list.
                result_la = self.lol[0]
                    
            elif num_rows > 1 and num_cols == 1:
                # single column result as a list.
                result_la = self.icol(0)
            else:
                result_la = []
        elif irow is not None and num_rows:
            result_la = self.lol[irow]
        elif icol and num_cols:
            result_la = self.icol(icol)
        else:
            result_la = []
            
        if unique and result_la:
            result_la = list(dict.fromkeys(result_la))
            
        return result_la


    def to_dict(self, irow: int=0, include_cols: Optional[T_ls]=None) -> T_da:
        """ alias for iloc 
            Note that this does not convert a column to a dict. Use to_list to convert a column.
            test exists in test_daf.py
        """
        return self.iloc(irow, include_cols)
        

    def irow(self, irow: int=0, include_cols: Optional[T_ls]=None) -> T_da:
        """ alias for iloc 
            test exists in test_daf.py
        """
        return self.iloc(irow, include_cols)
        

    def iloc(self, irow: int=0, include_cols: Optional[T_ls]=None) -> T_da:
        """ Select one record from daf using the idx and return as a single T_da dict
            test exists in test_daf.py
        """
        
        if irow < 0 or irow >= len(self.lol) or not self.lol or not self.lol[irow]:
            return {}
            
        if self.hd: 
            return self._basic_get_record_da(irow, include_cols)
                
        colnames = utils._generate_spreadsheet_column_names_list(num_cols=len(self.lol[irow]))
        return dict(zip(colnames, self.lol[irow]))
        

    def select_by_dict_to_lod(self, selector_da: T_da, expectmax: int=-1, inverse: bool=False) -> T_loda:
        """ Select rows in daf which match the fields specified in d, returning lod 
            test exists in test_daf.py
            
            DEPRECATE, use select_by_dict().to_lod()
        """
        
        result_lod = self.select_by_dict(selector_da=selector_da, expectmax=expectmax, inverse=inverse).to_lod()

        return result_lod


    def select_by_dict(self, selector_da: T_da, expectmax: int=-1, inverse:bool=False, keyfield:str='') -> 'Daf':
        """ Selects rows in daf which match the fields specified in d
            and return new daf, with keyfield set according to 'keyfield' argument.
            test exists in test_daf.py
        """

        # from utilities import utils

        result_lol = [list(d2.values()) for d2 in self if inverse ^ utils.is_d1_in_d2(d1=selector_da, d2=d2)]
    
        if expectmax != -1 and len(result_lol) > expectmax:
            raise LookupError
            # import pdb; pdb.set_trace() #perm
            # pass
            
        new_keyfield = keyfield or self.keyfield
        
        daf = Daf(cols=self.columns(), lol=result_lol, keyfield=new_keyfield, dtypes=self.dtypes)
        
        return daf
        
        
    def select_first_row_by_dict(self, selector_da: T_da, inverse:bool=False) -> T_da:
        """ Selects the first row in daf which matches the fields specified in selector_da
            and returns that row. Else returns {}.
            Use inverse to find the first row that does not match.
        """
            
        # test exists in test_daf.py

        for d2 in self:
            if inverse ^ utils.is_d1_in_d2(d1=selector_da, d2=d2):
                return d2

        return {}


    def select_where(self, where: Callable) -> 'Daf':
        """
        Select rows in Daf based on the provided where condition
        if provided as a string, the variable 'row' is the current row being evaluated.
        if a callable function, then it is passed the row.

        # Example Usage
        
            result_daf = original_daf.select_where(lambda row: bool(int(row['colname']) > 5))
        
        """
        # unit test exists.

        result_lol = [list(row.values()) for row in self if where(row)]

        daf = Daf(cols=self.columns(), lol=result_lol, keyfield=self.keyfield, dtypes=self.dtypes)

        return daf    
        

    def select_where_idxs(self, where: Callable) -> T_li:
        """
        Select rows in Daf based on the provided where condition
        variable 'row' is the current row being evaluated
        and return list of indexes.

        # Examle Usage
            result_daf = original_daf.select_where("int(row['colname']) > 5")
        
        """
        # unit test exists.
        
        return [idx for idx, row in enumerate(self) if where(row)]


    def col(self, colname: str, unique: bool=False, omit_nulls: bool=False, silent_error:bool=False) -> list:
        """ alias for col_to_la()
            can also use column ranges and then transpose()
            test exists in test_daf.py
        """
        return self.col_to_la(colname, unique, omit_nulls=omit_nulls, silent_error=silent_error)


    def col_to_la(self, colname: str, unique: bool=False, omit_nulls: bool=False, silent_error:bool=False) -> list:
        """ pull out out a column from daf by colname as a list of any
            does not modify daf. Using unique requires that the 
            values in the column are hashable.
            test exists in test_daf.py
        """
        
        if not colname:
            raise RuntimeError("colname is required.")
        if colname not in self.hd:
            if silent_error:
                return []
            raise RuntimeError(f"colname {colname} not defined in this daf. Use silent_error to return [] in this case.")

        icol = self.hd[colname]
        result_la = self.icol_to_la(icol, unique=unique, omit_nulls=omit_nulls)
        
        return result_la

        
    def icol(self, icol: int) -> list:
        return self.icol_to_la(icol)


    def icol_to_la(self, icol: int, unique: bool=False, omit_nulls: bool=False) -> list:
        """ pull out out a column from daf by icol idx as a list of any 
            can also use column ranges and then transpose()
            does not modify daf
            test exists in test_daf.py
        """
        
        if icol < 0 or not self or icol >= self.num_cols():
            return []
        
        if omit_nulls:
            result_la = [la[icol] for la in self.lol if la[icol]]
        else:
            result_la = [la[icol] for la in self.lol]

        if unique:
            result_la = list(dict.fromkeys(result_la))
            
        return result_la
            
    
    def drop_cols(self, exclude_cols: Optional[T_ls]=None):
        """ given a list of colnames, cols, remove them from daf array
            alters the daf and creates a copy of all data.
            
            Note: could provide an option to not create a copy, and use splicing.
            
            test exists in test_daf.py
        """
        
        if exclude_cols:
            keep_idxs_li: T_li = [self.hd[col] for col in self.hd if col not in exclude_cols]
        
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
        

    def select_cols(self, 
            cols: Optional[T_ls]=None, 
            exclude_cols: Optional[T_ls]=None, 
            ) -> 'Daf':
        """ given a list of colnames, alter the daf to select only the cols specified.
            this produces a new daf. Instead of selecting cols in this manner, it is better
            provide cols parameter in any .apply, .reduce, .from_xxx or .to_xxx methods,
            because this operation is not efficient.
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
        
        # select from the array and create a new object.
        # this is time consuming.
        new_lol = []
        for irow, la in enumerate(self.lol):
            la = [la[col_idx] for col_idx in range(len(la)) if col_idx in selected_cols_li]
            new_lol.append(la)
       
        old_cols = self.columns()
        new_cols = [old_cols[idx] for idx in range(len(old_cols)) if idx in selected_cols_li]
        dtypes = {col: typ for col, typ in self.dtypes.items() if col in new_cols}
        
        new_keyfield = self.keyfield if self.keyfield and self.keyfield in new_cols else ''

        new_daf = Daf(lol=new_lol, cols=new_cols, dtypes=dtypes, keyfield=new_keyfield)
    
        return(new_daf)
        

    # def from_selected_cols(self, cols: Optional[T_ls]=None, exclude_cols: Optional[T_ls]=None) -> 'Daf':
        # """ given a list of colnames, create a new daf of those cols.
            # creates as new daf
            
            # use my_daf[:, colnames_ls]
            
        # """
        
        # if not cols:
            # cols = []
        # if not exclude_cols:
            # exclude_cols = []
            
        # desired_cols = self.calc_cols(include_cols=cols, exclude_cols=exclude_cols)
    
        # selected_idxs = [self.hd[col] for col in desired_cols if col in self.hd]
        
        # new_lol = []
        
        # for irow, la in enumerate(self.lol):
            # la = [la[idx] for idx in range(len(la)) if idx in selected_idxs]
            # new_lol.append(la)
            
        # old_cols = list(self.hd.keys())
        # new_cols = [old_cols[idx] for idx in range(len(old_cols)) if idx in selected_idxs]
        
        # new_dtypes = {col: typ for col, typ in self.dtypes.items() if col in new_cols}
        
        # return Daf(lol=new_lol, cols=new_cols, dtypes=new_dtypes)
        

    #=========================
    #   modify records
        
    def assign_record_da(self, record_da: T_da):
        """ Assign one record in daf using the key using a single T_da dict.
            unit tests exist
        """
        
        if not self.keyfield:
            raise RuntimeError("No keyfield estabished for daf.")
            
        keyfield = self.keyfield
        if keyfield not in record_da:
            raise RuntimeError("No keyfield in dict.")
            
        if self and list(record_da.keys()) != list(self.hd.keys()):
            raise RuntimeError("record fields not equal to daf columns")
            
        keyval = record_da[keyfield]
            
        row_idx = self.kd.get(keyval, -1)
        if row_idx < 0 or row_idx >= len(self.lol):
            self.append(record_da)
        else:
            #normal_record_da = Daf.normalize_record_da(record_da, cols=self.columns(), dtypes=self.dtypes)   
            self.lol[row_idx] = [record_da.get(col, '') for col in self.hd]
        

    def assign_record_da_irow(self, irow: int=-1, record_da: Optional[T_da]=None):
        """ Assign one record in daf using the iloc using a single T_da dict.
            unit tests exist
        """
        
        if record_da is None:
            return
        
        if irow < 0 or irow >= len(self.lol):
            self.append(record_da)
        else:
            #normal_record_da = Daf.normalize_record_da(record_da, cols=self.columns(), dtypes=self.dtypes)   
            self.lol[irow] = [record_da.get(col, '') for col in self.hd]
        

    def update_by_keylist(self, keylist: Optional[T_ls]=None, record_da: Optional[T_da]=None):
        """ Update selected records in daf by keylist using record_da
            only update those columns that have dict keys
            but keep all other dict items intact in that row if not updated.
        """
        
        if record_da is None or not self.lol or not self.hd or not self.keyfield or not keylist:
            return

        for key in keylist:
            self.update_record_da_irow(self.kd.get(key, -1), record_da)
            
        

    def update_record_da_irow(self, irow: int=-1, record_da: Optional[T_da]=None):
        """ Update one record in daf at iloc using a single T_da dict,
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
        
        
        
    def insert_icol(self, icol: int=-1, col_la: Optional[T_la]=None, colname: str='', default: Any=''): #, keyfield:str=''):
        """ insert column col_la at icol, shifting other column data. 
            use default if la not long enough
            If icol==-1, insert column at right end.
            use set_keyfield() if this column will become the keyfield.
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
            
        # if keyfield:
            # self.keyfield = keyfield
            # self._rebuild_kd()

        
    def insert_irow(self, irow: int=-1, row_la: Optional[T_la]=None, default: Any=''):
        """ insert row row_la at irow, shifting other rows down. 
            use default if la not long enough
            If irow > len(daf), insert row at the end.
            
        """
        
        # from utilities import utils

        self.lol = utils.insert_row_in_lol_at_irow(irow=irow, row_la=row_la, lol=self.lol, default=default)
        
        self._rebuild_kd()


    def assign_col(self, colname: str, la: Optional[T_la]=None, default: Any=''):
        """ modify col by colname using la 
            use default if la not long enough.
            test exists in test_daf.py
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
        

    def insert_col(
            self, 
            colname: str, 
            col_la: Optional[T_la]=None, 
            icol: int=-1, 
            default: Any='',
            ): #, keyfield:str=''):
        """ add col by colname and set to la at icol
            if la is not long enough for a full column, use the default.
            if colname exists, overwrite it.
            Can use to set a constant value by not passing col_la and setting default.
            use set_keyfield() if this column will become the keyfield.
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

        self.insert_icol(icol=icol, col_la=col_la, colname=colname, default=default) #, keyfield=keyfield)
        
        # hl = list(self.hd.keys())
        # hl.insert(icol, colname)
        # self.hd = {k: idx for idx, k in enumerate(hl)}
        
        return self
        
    
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
        """ scan cells in daf and if match is found, replace the cell with pattern """

        for row_la in self.lol:
            for i, value in enumerate(row_la):
                if bool(re.search(find_pat, str(value))):
                    row_la[i] = replace_val
        
    

    #=========================
    # split and grouping
    
    def split_daf_into_ranges(self, chunk_ranges: List[Tuple[int, int]]) -> List['Daf']:
        """ Given a df and list of (start,end) ranges, split daf into list of daf.
        """
        
        chunks_lodaf = [self.select_irows(list(range(start, end))) for start,end in chunk_ranges]
        #chunks_lodaf = [self[start:end] for start, end in chunk_ranges]
        return chunks_lodaf
        
        
    
    def split_daf_into_chunks_lodaf(self, max_chunk_size: int) -> List['Daf']:
        """ given a daf, split it evenly by rows into a list of dafs.
            size of some dafs may be less than the max but not over.
        """
        # from utilities import utils
        
        chunk_sizes_list = utils.calc_chunk_sizes(num_items=len(self), max_chunk_size=max_chunk_size)
        chunk_ranges = utils.convert_sizes_to_idx_ranges(chunk_sizes_list)
        chunks_lodaf = self.split_daf_into_ranges(chunk_ranges)
        return chunks_lodaf

           
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

    def apply_formulas(self, formulas_daf: 'Daf'):
        r""" apply an array of formulas to the data in daf
        
        formulas must have the same shape as self daf instance.
        cells which are empty '' do not function.
        
        formulas are re-evaluated until there are no further changes. Error will result if expressions are circular.
        
        #### Special Notation
        There is only a very few cases of special notation:

        - $d -- references the current daf instance, a convenient shorthand.
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
        
            example_daf = Daf(cols=['A', 'B', 'C'], lol=[[1, 2, 0],[4, 5, 0],[7, 8, 0],[0, 0, 0]])
            formulas_daf = Daf(cols=['A', 'B', 'C'], 
            formulas_daf = Daf(cols=['A', 'B', 'C'], 
                    lol=[['',                    '',                    "sum($d[$r,:$c])"],
                         ['',                    '',                    "sum($d[$r,:$c])"],
                         ['',                    '',                    "sum($d[$r,:$c])"],
                         ["sum($d[:$r,$c])",     "sum($d[:$r,$c])",     "sum($d[:$r,$c])"]]
                         )
            expected_result = Daf(cols=['A', 'B', 'C'], lol=[[1, 2, 3],[4, 5, 9],[7, 8, 15],[12, 15, 27]])
            
        """
        
        # TODO: This algorithm is not optimal. Ideally, a dependency tree would be formed and
        # cells modified in from those with no dependencies to those that depend on others.
        # This issue will not become a concern unless the number of formulas is substantial.
        
        if not self:
            return
        
        if self.shape() != formulas_daf.shape():
            import pdb; pdb.set_trace() #temp
            
            raise RuntimeError("apply_formulas requires data arrays of the same shape.")
        
        lol_changed = True     # must evaluate at least once.
        loop_limit = 100
        loop_count = 0
        
        # the following deals with $d, $r, $c in the formulas
        parsed_formulas_daf = formulas_daf._parse_formulas()
        
        # we must use RETMODE_VAL for formulas to work easily for users.
        prior_retmode = self.retmode
        self.retmode = self.RETMODE_VAL
        
        while lol_changed:
            lol_changed = False
            loop_count += 1
            if loop_count > loop_limit:
                raise RuntimeError("apply_formulas is resulting in excessive evaluation loops.")
            
            for irow in range(len(self.lol)):
                for icol in range(self.num_cols()):
                    cell_formula = parsed_formulas_daf.lol[irow][icol]
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
                        
        self.retmode = prior_retmode

        self._rebuild_kd()
        
    def _parse_formulas(self) -> 'Daf':
    
        # start with unparsed formulas
        parsed_formulas = copy.deepcopy(self)
    
        for irow in range(len(self.lol)):
            for icol in range(self.num_cols()):
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
        """ given a daf with at least two columns, create a dict of list
            lookup where the key are values in col1 and list of values are 
            unique values in col2. Values in cols must be hashable.
            
        For example, if:

        daf.lol = [['a', 'b', 'c'], 
                    ['b', 'd', 'e'], 
                    ['a', 'f', 'g'], 
                    ['b', 'd', 'm']]
        daf.columns = ['col1', 'col2', 'col3']
        daf.cols_to_dol('col1', 'col2') results in
        {'a': ['b', 'f'], 'b':['d']}
            
            test exists in test_daf.py
            
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

    def apply(
            self, 
            func: Callable[[Union[T_da, T_Daf], Optional[T_la]], Union[T_da, T_Daf]], 
            by: str='row', 
            cols: Optional[T_la]=None,                      # columns included in the apply operation.
            keylist: Optional[T_ls]=None,                   # list of keys of rows to include.
            **kwargs: Any,
            ) -> "Daf":
        """
        Apply a function to each 'row', 'col', or 'table' in the Daf and create a new Daf with the transformed data.
        Note: to apply a function to a portion of the table, first select the columns or rows desired 
                using a selection process.

        Args:
            func (Callable): The function to apply to each 'row', 'col', or 'table'. 
            It should take a row dictionary and any additional parameters.
            by (str): either 'row', 'col' or 'table'
                if by == 'table', function should create a new Daf instance.
            keylist: Optional[T_ls]=None,                   # list of keys of rows to include.
            **kwargs: Additional parameters to pass to the function.

        Returns:
            Daf: A new Daf instance with the transformed data.
        """
        if by == 'table':
            return func(self, **kwargs)
       
        result_daf = self.clone_empty()

        if by == 'row':
            if keylist is None:
                keylist = []
        
            keylist_or_dict = keylist if not keylist or len(keylist) < 30 else dict.fromkeys(keylist)
            for row in self:
                if self.keyfield and keylist_or_dict and self.keyfield not in keylist_or_dict:
                    continue
                transformed_row = func(row, cols, **kwargs)
                result_daf.append(transformed_row)
                
        elif by == 'col':
            # this is not working yet, don't know how to handle cols, for example.
            raise NotImplementedError
        
            num_cols = self.num_cols()
            for icol in range(num_cols):
                col_la = self.icol(icol)
                transformed_col = func(col_la, cols, **kwargs)
                result_daf.insert_icol(icol, transformed_col)
        else:
            raise NotImplementedError
            
        # Rebuild the internal data structure (if needed)
        result_daf._rebuild_kd()

        return result_daf
      
      
    def update_row(row, da):
        row.update(da)
        return row
      
        
    def apply_in_place(
            self, 
            func: Callable[[T_da], T_da], 
            by: str='row', 
            keylist: Optional[T_ls]=None,                   # list of keys of rows to include.
            **kwargs: Any,
            ):
        """
        Apply a function to each 'row', 'col', or 'table' in the daf.

        Args:
            func (Callable): The function to apply to each 'row' 
            It should take a row dictionary and any additional parameters.
            # by (str): either 'row', 'col' or 'table'
            #     if by == 'table', function should create a new Daf instance.
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
        
        
    def reduce(
            self, 
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
                if by == 'table', function should create a new Daf instance.
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
            num_cols = self.num_cols()
            for icol in range(num_cols):
                col_la = self.icol(icol)
                reduction_la = func(col_la, reduction_la, **kwargs)
            return reduction_la

        else:
            raise NotImplementedError
        return [] # for mypy only.
        
        
    def manifest_apply(
            self, 
            func: Callable[[T_da, Optional[T_la]], Tuple[T_da, 'Daf']],    # function to apply according to 'by' parameter 
            load_func: Callable[[T_da], 'Daf'],            # optional function to load data for each manifest entry, defaults to local file system 
            save_func: Callable[[T_da, 'Daf'], str],       # optional function to save data for each manifest entry, defaults to local file system
            by: str='row',                                  # determines how the func is applied.
            cols: Optional[T_la]=None,                      # columns included in the apply operation.
             **kwargs: Any,
            ) -> "Daf":
        """
        Given a chunk_manifest_daf, where each record is a chunk_spec (dict),
        1. load the each chunk using 'load_func(chunk_spec)'
        2. apply 'func' to the loaded Daf instance to produce (result_chunk_spec, new_daf) 
        3. save new_daf using 'save_func(result_chunk_spec)'
        4. append result_chunk_spec to result_manifest_daf describing the resulting chunks 

        Args:
            func (Callable): The function to apply to each table specified by each record in self.
            load_func (Callable): load specified daf table based on the chunkspec in each row of self.
            save_func (Callable): save resulting daf table after operation by func.
            **kwargs: Additional parameters to pass to func

        Returns:
            result_manifest_daf
            
        Note, this method can be used for transformation, where the same number of transformed chunks exists,
            or, it can be used for reduction.
        if 'reduce' is true, then each chunk returns a daf with a single row.
            
            
        """

        result_manifest_daf = Daf()

        for chunk_spec in self:
            # Apply the function for all chunks specified.
            # Load the specified Daf table
            loaded_daf = load_func(chunk_spec)

            # Apply the function to the loaded Daf
            result_chunk_spec, transformed_daf = loaded_daf.apply(func, by=by, cols=cols, **kwargs)
            
            # Save the resulting Daf table
            save_func(result_chunk_spec, transformed_daf)
        
            # Update the manifest with information about the resulting chunk
            result_manifest_daf.append(result_chunk_spec)
            
        return result_manifest_daf        

    
    def manifest_reduce(
            self, 
            func: Callable[[T_da, Optional[T_la]], T_da], 
            load_func: Optional[Callable[[T_da], 'Daf']] = None,
            by: str='row',                                  # determines how the func is applied.
            cols: Optional[T_la]=None,                      # columns included in the reduce operation.
            **kwargs: Any,
            ) -> T_da:
        """
        Apply a reduction function to the tables specified by the chunk manifest.

        Args:
            func (Callable): The function to apply to each table specified by each record in self.
            load_func (Callable): Load specified daf table based on the chunkspec in each row of self.
            **kwargs: Additional parameters to pass to func

        Returns:
            Daf: Result of reducing all chunks into a single record.
        """
        first_reduction_daf = self.clone_empty()

        for chunk_spec in self:
            # Load the specified Daf table
            loaded_daf = load_func(chunk_spec)

            # Apply the function to the loaded Daf
            reduction_da = loaded_daf.reduce(func, by=by, cols=cols, **kwargs)

            first_reduction_daf.append(reduction_da)     

        final_reduction_da = first_reduction_daf.reduce(func, by=by, cols=cols, **kwargs)
        
        return final_reduction_da
        
        
    def manifest_process(
            self, 
            func: Callable[[T_da, Optional[T_la]], T_da],   # function to run for each hunk specified by the manifest
            **kwargs: Any,
            ) -> 'Daf':                                    # records describing metadata of each hunk
        """
        Given a chunk_manifest_daf, where each record is a chunk_spec (dict),
        1. apply 'func' to each chunk specified by the manifest.
        2. func() will load the chunk and save any results.
        3. returns one record for each func() call, add these to the resulting daf.

        Args:
            func (Callable): The function to apply to each table specified by each record in self.
            cols:            Reduce scope to a set of cols
            **kwargs: Additional parameters to pass to func

        Returns:
            result_daf
                      
        """

        result_daf = Daf()

        for chunk_spec in self:
            # Apply the function for all chunks specified.
            # Load the specified Daf table
            # Apply the function to the loaded Daf
            result_da = func(chunk_spec, **kwargs)
            
            # Update the manifest with information about the resulting chunk
            result_daf.append(result_da)
            
        return result_daf        

    
    def groupby(
            self, 
            colname: str='', 
            colnames: Optional[T_ls]=None,
            omit_nulls: bool=False,         # do not group to values in column that are null ('')
            ) -> Union[Dict[str, 'Daf'], Dict[Tuple[str, ...], 'Daf']]:
        """ given a daf, break into a number of daf's based on one colname or list of colnames specified. 
            For each discrete value in colname(s), create a daf table with all cols,
            including colname, and return in a dodaf (dict of daf) structure.
            If list of colnames is provided, dodaf keys are tuples of the values.
        """
        
        if isinstance(colname, list) and not colnames:
            return self.groupby_cols(colnames=colname)
        elif colnames and not colname:
            if len(colnames) > 1:
                return self.groupby_cols(colnames=colnames)
            else:
                colname = colnames[0]
                # can continue below.
        
        result_dodaf: Dict[str, 'Daf'] = {}
        
        for da in self:
            fieldval = da[colname]
            if omit_nulls and fieldval=='':
                continue
            
            if fieldval not in result_dodaf:
                result_dodaf[fieldval] = self.clone_empty()
                
            this_daf = result_dodaf[fieldval]
            this_daf.record_append(record_da=da)
            result_dodaf[fieldval] = this_daf
    
        return result_dodaf
    

    def groupby_cols(self, colnames: T_ls) -> Dict[Tuple[str, ...], 'Daf']:
        """ given a daf, break into a number of daf's based on colnames specified. 
            For each discrete value in colname, create a daf table with all cols,
            including colnames, and return in a dodaf (dict of daf) structure,
            where the keys are a tuple of the column values.
            
            Examine the records to determine what the values are for the colnames specified.
        """
        
        result_dodaf: Dict[Tuple[str, ...], 'Daf'] = {}
        
        for da in self:
            fieldval_tuple = tuple(da[colname] for colname in colnames)
            if fieldval_tuple not in result_dodaf:
                result_dodaf[fieldval_tuple] = this_daf = self.clone_empty()
            
            else:
                this_daf = result_dodaf[fieldval_tuple]
                
            this_daf.record_append(record_da=da)
            result_dodaf[fieldval_tuple] = this_daf
    
        return result_dodaf


    def groupby_cols_reduce(
            self, 
            groupby_colnames: T_ls, 
            func: Callable[[T_da, Union['Daf', T_da]], Union[T_da, T_la, 'Daf']], 
            by: str='row',                                  # determines how the func is applied.
            reduce_cols: Optional[T_la]=None,               # columns included in the reduce operation.
            diagnose: bool = False,
            **kwargs: Any,
            ) -> 'Daf':
            
        """ 
            Given a daf, break into a number of daf's based on values in groupby_colnames. 
            For each group, apply func. to data in reduce_cols.
            returns daf with one row per group, and keyfield not set.
        """
        # unit test exists.
        """
            
            This can be commonly used when some colnames are important for grouping, while others
            contain values or numeric data that can be reduced.
            
            For example, consider the data table with the following columns:
            
            gender, religion, zipcode, cancer, covid19, gun, auto
            
            The data can be first grouped by the attribute columns gender, religion, zipcode, and then
            then prevalence of difference modes of death can be summed. The result is a daf with one
            row per unique combination of gender, religion, zipcode. Say we consider just M/F, C/J/I, 
            and two zipcodes 90001, and 90002, this would result in the following rows, where the 
            values in paranthesis are the reduced values for each of the numeric columns, such as the sum.
            
            In general, the number of rows is reduced to the product of number of unique values in each column
            grouped. In this case, there are 2 genders, 3 religions, and 2 zipcodes, resulting in
            2 * 3 * 2 = 12 rows.
            
            groupby_colnames = ['gender', 'religion', 'zipcode']
            reduce_colnames  = ['cancer', 'covid19', 'gun', 'auto']
            
            grouped_and_summed = data_table.groupby_cols_reduce(
                groupby_colnames=['gender', 'religion', 'zipcode'], 
                func = sum_np(),
                by='table',                                     # determines how the func is applied.
                reduce_cols = reduce_colnames,                  # columns included in the reduce operation.
                )

            
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

            result_lol = [
            ['M', 'C', 90001,  6,  8, 10, 12],
            ['M', 'C', 90002, 21, 24, 26, 18],
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
            
            This reduction can then be further grouped and summed to create reports or to allow for 
            comparison based on any combination of the subgroups.
            
            
        """
        
        # divide up the table into groups where each group has a unique set of values in groupby_colnames
        # import pdb; pdb.set_trace() #temp
        
        if diagnose:  # pragma: no cover
            utils.sts(f"Starting groupby_cols() of {len(self):,} records.", 3)
            
        grouped_tdodaf = self.groupby_cols(groupby_colnames)
        
        if diagnose:  # pragma: no cover
            utils.sts(f"Total of {len(grouped_tdodaf):,} groups. Reduction starting.", 3)
        
        result_daf = Daf(cols=groupby_colnames + reduce_cols)
                
        for coltup, this_daf in grouped_tdodaf.items():
        
            if not this_daf:
                # nothing found with this combination of groupby cols.
                continue
                
            # apply the reduction function
            reduction_da = this_daf.reduce(func, by=by, cols=reduce_cols, **kwargs)
            
            # add back in the groupby cols
            for idx, groupcolname in enumerate(groupby_colnames):
                reduction_da[groupcolname] = coltup[idx]
            
            result_daf.append(reduction_da)

        if diagnose:  # pragma: no cover
            utils.sts(f"Reduction completed: {len(result_daf):,} records.", 3)

        return result_daf
    

    def groupby_reduce(
            self, 
            colname:str, 
            func: Callable[[T_da, T_da], Union[T_da, T_la]], 
            by: str='row',                                  # determines how the func is applied.
            reduce_cols: Optional[T_la]=None,                      # columns included in the reduce operation.
            **kwargs: Any,
            ) -> 'Daf':
        """ given a daf, break into a number of daf's based on colname specified. 
            For each group, apply callable.
            returns daf with one row per group, with keyfield the groupby value in colname.
            
        """
        
        grouped_dodaf = self.groupby(colname)
        result_daf = Daf(keyfield = colname)
        
        for colval, this_daf in grouped_dodaf.items():
        
            # maybe remove colname from cols here
        
            reduction_da = this_daf.reduce(func, by=by, cols=reduce_cols, **kwargs)
            
            # add colname:colval to the dict
            reduction_da = {colname: colval, **reduction_da}
            
            # this will also maintain the kd.
            result_daf.append(reduction_da)

        return result_daf

        
    #===================================
    # apply / reduce convenience methods

    def daf_sum(
            self, 
            by: str = 'row', 
            cols: Optional[T_la]=None
            ) -> T_da:
            
        return self.reduce(func=Daf.sum_da, by=by, cols=cols)
    

    def daf_valuecount(
            self, 
            by: str = 'row', 
            cols: Optional[T_la]=None
            ) -> T_da:
        """ count values in columns specified and return for each column,
            a dictionary of values and counts for each value in the column
            
            Need a way to specify that blank values will also be counted.
        """
            
        return self.reduce(func=self.__class__.count_values_da, by=by, cols=cols)


    def groupsum_daf(
            self,
            colname:str, 
            func: Callable[[T_da, T_da, Optional[T_la]], Union[T_da, T_la]], 
            by: str='row',                                  # determines how the func is applied.
            reduce_cols: Optional[T_la]=None,               # columns included in the reduce operation.
            ) -> 'Daf':
    
        result_daf = self.groupby_reduce(colname=colname, func=self.__class__.sum_da, by=by, reduce_cols=reduce_cols)
        
        return result_daf


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
    #   my_daf.apply_to_col(col='colname', func=lambda x: re.sub(r'^\D+', '', x))    

    #====================================
    # reduction atomic functions
    
    # requirements for reduction functions:
    #   1. reduction will produce a single dictionary of results, for each daf chunk.
    #   2. each atomic function will be staticmethod which accepts a single row dictionary, this_da
    #       and contributes to an accum_da. The accum_da is mutated by each row call.
    #   3. the reduction atomic function must be able to deal with combining results
    #       in a daf where each record is the result of processing one chunk.
    #   4. each atomic function will also accept a cols parameter which identifies which 
    #       columns are to be included in the reduction, if it is not None or []
    #       Otherwise, all columns will be processed. This columns parameter can be
    #       initialized explicitly or using my_daf.calc_cols(include_cols, exclude_cols, include_dtypes, excluded_dtypes)
    #   5. Even if columns are reduced, the result of the function will include all columns
    #       and non-specified columns will be initialized to '' empty string. This complies
    #       with design goal of always producing a result that will be useful in a report.
    #   6. Therefore, the reduction result may be appended to the daf if desired.
    
    

    @staticmethod
    def sum_da(row_da: T_da, accum_da: T_da, cols: Optional[T_la]=None, diagnose:bool=False) -> T_da:     # result_da
        """ sum values in row and accum dicts per colunms provided. 
            will safely skip data that can't be summed.
        """
        diagnose = diagnose
        
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
            
            Intended use is to use this to calculate valuecounts by scanning all rows of a daf.
            Return a dodi.
            Put those in a daf table and then scan those combined values and create a singular result.
            
            This is a reducing and accumulating operation. Can be used with daf.reduce()
            
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
                Daf.sum_dodis(val, result_dodi)
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
                Daf.sum_da(this_di, accum_dodi[key])
            else:
                accum_dodi[key] = this_di

                    
    #===============================================
    # functions not following apply or reduce pattern
    
    # this function does not use normal reduction approach.
    def sum(
            self, 
            colnames_ls: Optional[T_ls]=None, 
            numeric_only: bool=False,
            ) -> dict: # sums_di
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
                        sums_d_by_colidx[colidx] += Daf._safe_tofloat(la[colidx])
                    else:
                        sums_d_by_colidx[colidx] += float(la[colidx])

        try:
            sums_d = {cleaned_colnames_ls[idx]: sums_d_by_colidx[colidx] for idx, colidx in enumerate(cleaned_colidxs_li)}
        except Exception:
            import pdb; pdb.set_trace() #perm ok
            pass 
        sums_d = utils.set_dict_dtypes(sums_d, self.dtypes)  # type: ignore
        
        return sums_d
        

    def sum_np(
            self, 
            colnames_ls: Optional[T_ls]=None, 
            ) -> dict: # sums_di
            
        """ total the columns in the table specified, and return a dict of {colname: total,...}
            This uses NumPy and requires that library, but this is about 3x faster.
            If you have mixed types in your Daf array, then use colnames to subset the
            columns sent to NumPy to those that contain only numerics and blanks.
            For many numeric operations, convert a set of columns to NumPy
            and work directly with NumPy and then convert back. See to_numpy and from_numpy()
        """
        # unit tests exist
        #   need tests for blanks and subsetting columns.
        
        if not self:
            return {}

        if colnames_ls is None:
            to_sum_daf = self
            colnames_ls = self.columns()
        else:
            to_sum_daf = self[:, colnames_ls]
            """ given a list of colnames, create a new daf of those cols.
                creates as new daf
            """
        
        # convert those columns to an numpy array.
        nparray = to_sum_daf.to_numpy()
        
        # sum the columns in the array.
        sum_columns = np.sum(nparray, axis=0)
        
        #convert to a dictionary.
        sums_d = dict(zip(colnames_ls, sum_columns.tolist()))
        
        return sums_d

    
    def valuecounts_for_colname(
            self, 
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


    def valuecounts_for_colnames_ls(
            self, 
            colnames_ls: Optional[T_ls]=None, 
            sort: bool=False, 
            reverse: bool=True
            ) -> T_dodi:
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
        

    def valuecounts_for_colnames_ls_selectedby_colname(
            self, 
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


    def valuecounts_for_colname1_groupedby_colname2(
            self,
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

    

    def gen_stats_daf(self, col_def_lot: T_lota) -> T_doda:

        info_dod = {}

        # from utilities import utils

        for col_def_ta in col_def_lot:
            col_name, col_dtype, col_format, col_profile = col_def_ta
            
            col_data_la = self.col(col_name)
            
            info_dod[col_name] = utils.list_stats(col_data_la, profile=col_profile)
            
        return info_dod

   
    def transpose(self, new_keyfield:str='', new_cols:Optional[T_la]=None, include_header:bool = False) -> 'Daf':
        """ 
        This implementation uses the built-in zip(*self.lol) to transpose the rows and columns efficiently. 
        The resulting transposed data is then used to create a new Daf instance.
    
        Args:
        - new_cols (list): names of the new columns. If include_header is True, this will be the first column.
        - new_keyfield (str): The new keyfield to be used in the transposed Daf.
        - include_header (bool): indicates if the column names, if defined, will also be included
                        and will become the first column in the result

        Returns:
        - Daf: A new Daf instance with transposed data and optional new keyfield.
        
        """

        if not new_cols:
            new_cols = ['key'] + utils._generate_spreadsheet_column_names_list(num_cols=len(self.lol))

        # transpose the array
        new_lol = [list(row) for row in zip(*self.lol)]
        
        if include_header:
            # add a new first column which will be the old column names row.
            # from utilities import utils
            
            new_lol = utils.insert_col_in_lol_at_icol(icol=0, col_la=self.columns(), lol=new_lol)
        
        return Daf(lol=new_lol, name=self.name, keyfield=new_keyfield, cols=new_cols, use_copy=True)        



    #===============================
    # reporting

    def md_daf_table_snippet(
            self, 
            ) -> str:
        """ provide an abbreviated md table given a daf representation """
        
        return self.to_md(
                max_rows        = self.md_max_rows, 
                max_cols        = self.md_max_cols, 
                shorten_text    = True, 
                max_text_len    = 80, 
                smart_fmt       = False, 
                include_summary = True,
                )

    # the following alias is defind at the bottom of this file.
    # Daf.md_daf_table = Daf.to_md

    def to_md(
            self, 
            max_rows:       int     = 0,         # limit the maximum number of row by keeping leading and trailing rows.
            max_cols:       int     = 0,         # limit the maximum number of cols by keeping leading and trailing cols.
            just:           str     = '',        # provide the justification for each column, using <, ^, > meaning left, center, right justified.
            shorten_text:   bool    = True,      # if the text in any field is more than the max_text_len, then shorten by keeping the ends and redacting the center text.
            max_text_len:   int     = 80,        # see above.
            smart_fmt:      bool    = False,     # if columns are numeric, then limit the number of figures right of the decimal to "smart" numbers.
            include_summary: bool   = False,     # include a one-line summary after the table.
            disp_cols:      Optional[T_ls]=None, # use these column names instead of those defined in daf.
            ) -> str:
        """ provide an full md table given a daf representation """

        daf_lol = self.daf_to_lol_summary(max_rows=max_rows, max_cols=max_cols, disp_cols=disp_cols)
        
        header_exists = bool(self.hd)
        
        mdstr = md.md_lol_table(daf_lol, 
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
        

    def daf_to_lol_summary(self, max_rows: int=10, max_cols: int=10, disp_cols:Optional[T_ls]=None) -> T_lola:
    
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
        num_cols    = self.num_cols()

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
            
            To use this interactively, use print(Daf.dict_to_md(my_da))
        """
        if not cols:
            cols = ['key', 'value']
    
        return Daf.from_lod_to_cols([da], cols=cols).to_md(just=just)
            
        
    #=========================================
    #  Reporting Convenience Methods

    def value_counts_daf(self, 
            colname: str,                       # column name to include in the value_counts table
            sort: bool=False,                   # sort values in the category
            reverse: bool=True,                 # reverse the sort
            include_total: bool=False,          #
            omit_nulls: bool=False,             # set to true if '' should be omitted.            
            ) -> 'Daf':
        """ create a values count daf of the results of value counts analysis of one column, colname.
            The result is a daf table with two columns.
            Left column has the values, and the right column has the counts for each value.
            column names are [colname, 'counts'] in the result. These can be changed later if they are not 
            output when used with multiple columns may be useful but only if they have the same set of values.
            provides a total line if "include_sum" is true.
        """
            
        value_counts_di   = self.valuecounts_for_colname(colname=colname, sort=sort, reverse=reverse)
        
        if omit_nulls:
            utils.safe_del_key(value_counts_di, '') 

        value_counts_daf = Daf.from_lod_to_cols([value_counts_di], cols=[colname, 'counts'])

        if include_total:
            value_counts_daf.append({colname: ' **Total** ', 'counts': sum(value_counts_daf[:,'counts'])})
            
        return value_counts_daf

# ALIASES
# these must not be established until after the class is fully defined.
Pydf = Daf
Pydf.pydf_to_lol_summary    = Daf.daf_to_lol_summary
Pydf.split_pydf_into_ranges  = Daf.split_daf_into_ranges
Pydf.select_records_pydf     = Daf.select_records_daf
Pydf.pydf_sum                = Daf.daf_sum
Pydf.pydf_valuecount         = Daf.daf_valuecount
Pydf.groupsum_pydf           = Daf.groupsum_daf
Pydf.gen_stats_pydf          = Daf.gen_stats_daf
Pydf.value_counts_pydf       = Daf.value_counts_daf

Pydf.md_pydf_table = Pydf.to_md            

