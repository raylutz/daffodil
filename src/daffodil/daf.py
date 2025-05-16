# daf.py
"""

# Daffodil -- Python Dataframes

The Daffodil class provides a lightweight, simple and fast alternative to provide 
2-d data arrays with mixed types.

"""

r"""
    MIT License

    Copyright (c) 2025 Ray Lutz

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


r"""
See README file at this location: https://github.com/raylutz/Daf/blob/main/README.md
"""

r"""
    v0.1.X 
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
                unflatten_dirname() <-- remove? (YES)
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
                row_idx_of()   (REMOVE!)
                remove_key -- keyfield not set. (remove)
                get_existing_keys
                select_record -- row_idx >= len(self.lol)
                _basic_get_record -- no hd, include_cols
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

    v0.2.2  Changed the file structure to be incompliance with Python traditions.
                user can use 'from daffodil.daf import Daf' and then Daf()
            Moved daf.py containing class Daf to the top level.
            put supporting functions in lib.
            added narrow_to_wide and wide_to_narrow methods.

    v0.3.0  (2024-04-14) 
            Added fmt parameter so file are saved with proper Content-Type metadata.
            Added 'from .Daf import Daf' to __init__.py to reduce level.  Eventually removed this.
            Added CODE_OF_CONDUCT.md
            Improved performance of daf_sum(), reduce() and sum_da() by avoiding explicit comparisons and leveraging try/except.
            Improved daf_benchmarks.py to help to diagnose nonperformant design of sum_da().
            Added basic indexed manipulation to daf_demo.py
            Changes due to the suggestions by Trey Hunner.
            
            

    v0.4.0  (2024-04-30)
            v0.4.0 Better dtypes support; apply_dtypes(), flatten(), copy()
            added disabling of garbage collection during timing, getting more consistent results, but does not explain anomaly.
            Improved philosophy of apply_dtypes() and flatten()
                Upon loading of csv file, set dtypes and then use my_daf.apply_dtypes()
                Before writing, use my_daf.flatten() to flatten any list or dict types, if applicable.
                
            apply_dtypes() now handles the entire array, and will skip str entries if initially_all_str is True.
                
                unflatten_cols()        DEPRECATED. use apply_dtypes()
                unflatten_by_dtypes()   DEPRECATED. use apply_dtypes()
                flatten_cols()          DEPRECATED. use flatten()
                flatten_by_dtypes()     Renamed:    use flatten()
                
            added optional dtypes parameter in apply_dtypes() which will be used to initialize dtypes in daf object and
                use it to convert types within the array.
            Changed from la type to interable in reduce()
            added disabling of garbage collection in daf_benchmarks.py
            deprecated functions dealing with hdlol type which was a precursor to daf.
            added convert_type_value() to convert a single value to a desired type and unflatten if enabled.
            removed use of set_dict_dtypes from apply_dtypes() and instead it is done on the entire daf array for efficiency.
            added in daf_utils.py unflatten_val(), json_decode(), validate_json_with_error_details, and safe_convert_json_to_obj.
            Added .copy(deep:bool) method to match pandas syntax.
            Added reference to 1994 workshop in flatten() method docstr.
            Changed packaging code from setup.py approach to pyproject, but still not able to correctly import in Lambdas container.

    v0.4.1  (2024-04-30)
            fixed tests to reflect changes to type conversion paradigm.
            Changed apply_dtypes parameter 'initially_all_str' to 'from_str'
            fixed set_dict_dtypes() in the case of dtypes = {}; Changed parameter to 'dtypes' for uniformity.
            set_dict_dtypes() now also modifies types in-place.
            
    v0.4.2  (2024-05-01)
            Modified packaging for package distribution on PyPI to hopefully make it compatible with installing into AWS Lambdas.
                Tried to use pyproject.toml and flit, but flit has poor toml parsing it seems, and could not find a suitable toml file.
                Went back to setup.py and setuptools, but reorganized files into daffodil/src folder which will be included in the distro.
                To use --editable mode for local development, must set PYTHONPATH to refer to the daffodil/src folder.
                In that folder is daffodil/daf.py and daffodil/lib/daf_(name).py 
                To import supporting py files from lib, must use import daffodil.lib.daf_utils as daf_utils, for example.
                
    v0.4.3  (2024-05-02)
            Went back to pyproject.toml approach. Eliminated setup.py.
            Lambdas now importing daffodil

    v0.4.4  (2024-05-02)
            Improved unflattening to handle f-string type flattening, including tuples.
            fixed bug in set_dtypes()
            added daf_utils.safe_eval()
            
    v0.5.0  (2024-05-23)
            Added split_where(self, where: Callable) which makes a single pass and splits the daf array in two
                true_daf, false_daf.
            Added to Daffodil multi_groupby(), reduce_dodaf_to_daf() and multi_groupby_reduce()
            Added class KeyedList() to provide a new data item that functions like a dict but is a dex plus list.
                can result in much better performance by not redistributing values in the dict structure.
                This is not yet integrated into daffodil, but should be.
                
            Removed '_da' from many Daffodil methods and for keyword parameters, to allow future upgrade to KeyList.
                select_record_da()      -> select_record()
                record_append()
                _basic_get_record_da    -> _basic_get_record
                assign_record_da()      -> assign_record()
                assign_record_da_irow   -> assign_record_irow
                update_by_keylist()
                update_record_da_irow   -> update_record_irow
            changed test_daf accordingly.
                
            Added _build_hd() to consistently build header dict structure.
            Added to_json() and from_json() methods to allow generation of custom JSONEncoder.
            Changed nomenclature in KeyedList class from dex to hd.
            Added from_json and to_json to KeyedList class to allow custom JSONEncoder to be developed.
            
            select_record() silently returns {} if self is empty.
            mark chunks saved to local_mirror if not use_lambdas
            
            fixed _itermode vs. itermode.
            Added .strip() method.
            correct icols when providing a single str column name, and when column names have more than one character each.
            Added 'flatten' in '.to_list' method which will combine lol to a single list.
            Added .num_rows() which will more robustly calculate the number of rows in edge cases.
            Fix unflattening issue discovered when running edge_test_utils.py.
            Updated documentation to reflect new approach to dtypes and flattening.
             
    v0.5.1  (2024-05-25)
            changed dependencies in pyproject.toml so they would allow newer versions.
            Upgraded to Python 3.11 and upgraded all libraries to the latest.
            Using venv311
    
    v0.5.2  (2024-05-30)
            Added .iter_dict() and .iter_klist() to force iteration to produce either dicts or KeyedLists.
                Producing KeyedLists means the list is not copied into a dict but can be mutated and the lol will be mutated.
            Correct calculation of slice_len to correct column assignment from another column
                This may still have some ambiguity if a nested list structure is meant to be assigned to an array cell.
                    collist = my_daf[:, 'colname'].to_list()    # this will return a list, but sometimes of only one value.
                    my_daf[:, 'colname2'] = collist             # there is ambiguity here as to whether the list with one
                                                                # item should be placed in the cell or if just the value.
                
    v0.5.3  (2024-06-28)            
            added tests:
                flatten()
                to_json()       not completely working.
                from_json()     not completely working.
            added __format__ to allow use of {:,} and other f-string formatting. Invokes .to_value()    
            added alias for valuecounts_for_colname() to value_counts() to match Pandas syntax.
            
            extend .iloc to support klist and list rtypes.
            Added .to_klist() to return a record as KeyedList type.
            extended .assign_col() to insert a column if the colname does not exist.
            Enhanced KeyedList() to allow both args to be None, and thus initialize to empty KeyedList.
            insert_col_in_lol_at_icol(): 
                fix bug if icol resolves to add a column. --> '>' changed to '>='
                allow empty lol and create a lol with one column if col_la exists.
            Add .iter_list() to allow iteration over just lol without cols defined.
            fixed __format__ so unadorned daf name prints summary. It takes more than {daf:} in fstring to cause formatting.
            Improved robustness of num_cols() to check first few rows.
                TODO: It will probably be better to keep a value of the num cols and not calculate evertime.
            changed name of values in KeyedList to _values and created accessors.
            added support for Iterables passed for row and col selection.
            Added method "remove_dups()" which returns unique records and duplicated records based on keyfield.
            Changed operation of assign_col to append col to right if colname not exist.
            
            worked around error in Pympler.asizeof.asizeof() function, used in daf_benchmarks.
                this appears to be resolved in future updates of pympler.
                
    v0.5.4  (2024-07-02)
            Add sort_by_colnames(self, colnames:T_ls, reverse: bool=False, length_priority: bool=False)
                Add daf_utils.sort_lol_by_cols()
            Add argument 'omit_nulls' to .to_list() method.
            Change references to klist.values to ._values to avoid amiguity with property getter and setters.
            Add annotate_daf(self, other_daf: 'Daf', my_to_other_dict: T_ds) to effectively join two tables.
            Fix value_counts_daf() by adding .to_list for total.
            
    v0.5.5  (2024-12-27)
            add daf_crm.py as demonstration.
                add op_import() 
            add get_csv_column_names() as refactoring in daf_utils for reading csv.
            precheck_csv_cols()
            compare_lists() -- imported from daf_utils
            Introduce ops_daf for running operations, can also use in audit-engine.
                operation descriptions taken from docstring.
            added 'default_type' to apply_dtypes for any cols not specified in passed dtypes.
            Improved preprocessing of csv file when line is commented out and embedded newlines exist in the line.
            Improved Daf.from_lod() by using columns in dtypes dict if provided instead of relying only on first record of lod.
            
            Added indexing with range and T_lor (list of range) types, for both column and row indexing.
            Added __contains__ method to allow " if key in my_daf: " to test if a given key exists. Requires kd exists.
            revised .sum_da() based on feedback from user group.
            Improve formatting of README.md to include tables of examples.
            improve daf_benchmarks.py to use objsize instead of pympler to evaluate memory use.
            Corrected set_keyfield in daffodil to do nothing if daf is empty.
            Added 'sparse_rows' to reduction 'by' type using an indirect_col.
            Improve daf_sum() to support indirect_col.
            Revised apply_in_place to support by='row_klist'. Func will modify row_klist and that will modify the array.
                Changed name of keyword parameter in apply_in_place() from keylist to rowkeys to avoid confusion.
            added astype parameter for to_list() and to_value()
            Introduced standardization around PYON instead of JSON:
                - Easier to convert esp. during serialization using csv.writer().
                - Compatible with more Python data types.
                - Still easy to convert to JSON.
            Copied function create_index_at_cursor() for sql tables in daf_benchmarks.py
            Added daf_sql.py mainly to support benchmarks at this point.
            This will be the last release before sql enhancements.
            
    v0.5.6  (2025-03-04)
            Added from_lot() class method. Perhaps these can be unified in main init function by examining type of the data passed.
            Added join() method, including unit tests.
            Added from_pdf() class method, used to parse PDF files with table structure across multiple pages.
            Added name argument to from_lod()
            Added name argument to from_csv_buff()
            using raw docstring format to avoid complaints of escape characters in derive_join_translator.
            Added 'tag_other' boolean parameter to tag all other column names during join, to support chained joins.
            Simplified translator_daf table so it is easier to produce by hand and use across many tables being joined.
            Added name argument for join() method, to provide the name of the resulting joined instance.
            Improve unit tests for derive_join_translator
            Added 25 tests for various indexing modes. 
                corrected parsing of tuple of strings for krows and kcols.
            Added name argument for clone_empty() method.
            Added omit_other_cols parameter for 'derive_join_translator' method. 
                this can probably displace the "shared_fields" parameter.
                Fixed omit_other_cols so it could be properly omitted.
            concat (which is called by append if a daf array is appended) was not using deep copy when 
                copying in the frame, and this became a real mess. Added copy with deep to concat.
            Added from_csv() which will load csv to Daf from local files, s3 path or http/s path.
                improved operation of streaming from file to avoid buffer recopying.
            from_csv_buff() still exists for those times when a buffer or file-like object already exists.
            Added .attrs dictionary to core dataframe instance definition to allow for descriptors to be 
                provided to users of the dataframe, esp. between when the daf array is defined and built 
                and when it is used and modified.
            Improve file closure by using context manager in from_csv() for local file usage.
            Deprecated use of utils instead of daf_utils.
            
    v0.5.7  (2025-05-16)
            Change SQL identifier escaping from character escaping to quoting.
                Note, this is a breaking change if prior SQL tables are encountered.
            Move to using pytest. Removed some sys.path.append() statements as a result.
            Improve default handling in to_donpa() to allow None to be the default by using a sentinel _MISSING
            Improved .to_list() default handling to allow None as the default, by using sentinel _MISSING.
            Improved .to_pandas_df() default handling
            Added option use_donpa to .to_pandas_df() to improve convertion to pandas df by first converting columns to numpy vectors.
            Added tests for .to_list() including various options.
            Added .replace_in_columns() to Replace all values in `find_values` with `replacement` for the specified columns.
            Added .apply_colwise() Apply a function to each row and store the result in target_col
            Improved .to_pandas_df() 
                to provide an option to convert first to donpa (dict of numpy array)
                to handle conversion of '' or None to a default such as NA or 0.
                improved tests.
            Improved from_pdf() to support complex conversion scenarios.
            
    v0.5.8  (pending)


    TODO
        Consider conversion .to_donpa() which would convert specified columns to individual numpy arrays in dict, where keys are col names.
        Consider method adjust_cols(select_cols, drop_cols, add_cols_dtypes, default_val) which would make a single pass through the
            table to select (keep), drop, or add columns. This can improve performance by avoiding creating a new
            table and potentially nearly doubling the data. Strategy will be to apply() in place so the table does 
            not grow in size, and this mimics the behavior in SQL tables.
    
        Plan:
            create base class for core operations.
            use this to create daf_sql class to handle sql functionality.
            extend daffodil methods to handle sql tables with similar syntax.
    
        offer dropping unexpected columns when doing concat/append ??
            keys mismatch: daf: (['ballot_id', 'style_num', 'precinct', 'contest', 'option', 'has_indication', 'num_marks', 'num_votes', 'pixel_metric_value', 'sm_pmv', 'writein_name', 'overvotes', 'undervotes', 'ssidx', 'delta_y', 'ev_coord_str', 'ev_precinct_id', 'target_pixels', 'gray_eval', 'is_bmd', 'wipmv', 'p', 'x', 'y', 'w', 'h', 'b', 'bmd_str'])
            other_instance:     (['ballot_id', 'style_num', 'precinct', 'contest', 'option', 'has_indication', 'num_marks', 'num_votes', 'pixel_metric_value', 'sm_pmv', 'writein_name', 'overvotes', 'undervotes', 'ssidx', 'delta_y', 'ev_coord_str', 'ev_precinct_id', 'target_pixels', 'gray_eval', 'is_bmd', 'wipmv', 'p', 'x', 'y', 'w', 'h'])
            > daf.py(2003)concat()
    
            No: It will probably be better to keep a value of the num cols and not calculate every time. (not worth it)
            No: add use of pickledjson to express contents of individual cells when not jsonable. (Use Pyon)
            tests: need to add
                __str__ and __repr__
                
                apply_dtypes()
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
                assign_record       (remove?)
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
                apply_dtypes_to_hdlol()     DEPRECATED
                    empty case
                select_col_of_lol_by_col_idx()
                    col_idx out of range.
                json_encode  --> now 'to_json()'
                make_strbool <-- remove? yes, remove.
                test_strbool >-- remove? yes, remove.
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
                profile_ls_to_lr
                reduce_lol_cols
                s3path_to_url
                parse_s3path
                transpose_lol
                safe_get_idx
                shorten_str_keeping_ends
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
            daf_md
                not tested at all.
                
"""     

r"""
    To update the package:
        1. run all tests and demos to verify validity of the version.
            pytest will run all tests.
            demos
                python tests/daf_demo.py
                python tests/daf_benchmarks.py
        2. Increment version number in pyproject.toml
        3. Remove prior release
                rm -r dist
        3. build the release
                pip install --upgrade build
                python -m build
        4. Upload it
                twine upload dist/*
            You will need pypi security token    
                
     
    venv311\Scripts\activate     
"""       
    
    
#VERSION  = 'v0.5.7'
#VERSDATE = '2025-05-16'  

import os
import sys
import io
import csv
import copy
import re
import json
import time
#import numpy as np
#from typing_extensions import deprecated       # can't get this to import correctly
import collections      # to provide collections.abc.Iterable type.
from pathlib import Path
    
# no longer need the following due to using pytest
# sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from daffodil.lib.daf_types import T_ls, T_lola, T_di, T_loda, T_da, T_li, T_dtype_dict, \
                            T_dola, T_dodi, T_la, T_lota, T_doda, T_buff, T_ds, T_lb, T_rli, \
                            T_ta, T_lor, T_kva, T_donpa, T_npa, T_lsi  #, T_hllola
                     
import daffodil.lib.daf_utils    as daf_utils
import daffodil.lib.daf_md       as md
import daffodil.lib.daf_pandas   as daf_pandas
import daffodil.lib.daf_pdf      as daf_pdf

from daffodil.keyedlist import KeyedList


from typing import List, Dict, Any, Tuple, Optional, Union, cast, Type, Callable, Iterable, Iterator #
def fake_function(a: Optional[List[Dict[str, Tuple[int,Union[Any, str]]]]] = None) -> Optional[int]:
    return None or cast(int, 0)       # pragma: no cover

T_Pydf = Type['Daf']
T_Daf = Type['Daf']
T_dodaf = Dict[str, T_Daf]

logs = daf_utils                # alias

# define a sentinel object to express a missing item where None is a valid value.
from daffodil.lib.daf_utils import _MISSING


class Daf:
    RETMODE_OBJ  = 'obj'
    RETMODE_VAL  = 'val'
    
    ITERMODE_DICT = 'dict' 
    ITERMODE_KEYEDLIST = 'keyedlist'
    

    def __init__(self, 
            lol:        Optional[T_lola]        = None,     # used to initialize the data array.
            hd:         Optional[T_di]          = None,     # used to initialize the hd array. If used, then cols not needed.
            kd:         Optional[T_di]          = None,     # used to initialize the kd array if no keyfield is set.
            cols:       Optional[T_ls]          = None,     # Optional column names to use.
            dtypes:     Optional[T_dtype_dict]  = None,     # Optional dtype_dict describing the desired type of each column.
                                                            #   also used to define column names if provided and cols not provided.
            keyfield:   Union[str, int, T_ta, T_la]  = '',  # A field of the columns to be used as a key.
                                                            # can be set even if columns not set yet.
                                                            # can be tuple or list of colnames, and then they are used as tuple keys.
            name:       str                     = '',       # An optional name of the Daf array.
            use_copy:   bool                    = False,    # If True, make a deep copy of the lol data.
            disp_cols:  Optional[T_ls]          = None,     # Optional list of strings to use for display, if initialized.

            retmode:    str                     = 'obj',    # default retmode
            itermode:   str                     = 'dict',   # default itermode, either 'dict' or 'keyedlist'
        ):
        if lol is None:
            lol = []
        if dtypes is None:
            dtypes = {}
        if cols is None:
            cols = []
            
        self.name           = name              # str
        self.keyfield       = keyfield          # str
        
        if hd is not None:
            self.hd         = hd                # hd_di
        else:
            self.hd         = {}
        

        if use_copy:
            self.lol        = copy.deepcopy(lol)
        else:
            self.lol        = lol
            
        if kd is not None:
            self.kd         = kd
        else:
            self.kd         = {}
            
        self.dtypes         = dtypes
        
        self.md_max_rows    = 10    # default number of rows when used with __repr__ and __str__
        self.md_max_cols    = 10    # default number of cols when used with __repr__ and __str__

        self._retmode       = retmode       # retmode can be either RETMODE_OBJ or RETMODE_VAL
        self._itermode      = itermode      # itermode can be either ITERMODE_DICT or ITERMODE_KEYEDLIST
        
        # Initialize iterator variables        
        self._iter_index = 0

        # Initialize metadata storage.
        # These attributes can be set for any instance. They are not automatically propagated 
        #   when copying or slicing. The metadata must be explicitly propagated:
        #       new_daf.attrs = daf.attrs.copy()
        # This can be used for providing information about a dataframe, such as
        #   the 'first_data_colidx' to keep track of the metadata columns vs. data columns.
        
        self.attrs = {}             # Dictionary for storing instance-wide metadata

        if not cols:
            if dtypes and isinstance(dtypes, dict):
                self.hd = type(self)._build_hd(dtypes.keys())
        else:
            if isinstance(cols, str):
                cols = [cols]
            # cols will be sanitized only if necessary.
            self._cols_to_hd(cols)
            if len(cols) != len(self.hd):
                cols = daf_utils._sanitize_cols(cols=cols)
                self._cols_to_hd(cols)
                if len(cols) != len(self.hd):                
                    breakpoint() #perm assertion error
                    pass
                    raise AttributeError ("AttributeError: cols not unique")
                
        # if self.hd and dtypes:
            # effective_dtypes = {col: dtypes.get(col, str) for col in self.hd}
        
            # # setting dtypes may be better done manually if required.
            # if self.num_cols():

                # self.lol = daf_utils.apply_dtypes_to_hdlol((self.hd, self.lol), effective_dtypes, from_str=False)[1]
            
        # rebuild kd if possible, only if keyfield is defined.
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

    @property
    def itermode(self):
        return self._itermode
        
    @itermode.setter
    def itermode(self, new_itermode):
        if new_itermode in [self.ITERMODE_DICT, self.ITERMODE_KEYEDLIST]:
            self._itermode = new_itermode
        else:
            raise ValueError("Invalid itermode")

    def __iter__(self) -> Iterator[Union[Dict[str, Any], KeyedList]]:
        return self._default_iterator()
        
    def _default_iterator(self) -> Iterator[Union[Dict[str, Any], KeyedList]]:
        if self._itermode == self.ITERMODE_DICT:
            return self.iter_dict()
        elif self._itermode == self.ITERMODE_KEYEDLIST:
            return self.iter_klist()
        else:
            raise ValueError(f"Invalid iteration mode: {self._itermode}")

    def iter_dict(self) -> Iterator[Dict[str, Any]]:
        return DafIterator(self, dict)

    def iter_klist(self) -> Iterator[KeyedList]:
        return DafIterator(self, KeyedList)
        
    def iter_list(self) -> Iterator[list]:
        return DafIterator(self, list)

    # def __next__(self) -> Union[Dict[str, int], KeyedList]:
        # if self._iter_index < len(self.lol):
            # if self._itermode == self.ITERMODE_DICT:
                # row_dict = dict(zip(self.hd.keys(), self.lol[self._iter_index]))
                # self._iter_index += 1
                # return row_dict
            # elif self._itermode == self.ITERMODE_KEYEDLIST:
                # row_klist = KeyedList(self.hd, self.lol[self._iter_index])
                # self._iter_index += 1
                # return row_klist
            # else:
                # raise NotImplementedError
        # else:
            # self._iter_index = 0
            # raise StopIteration

    def __bool__(self):
        """ test daf for existance and not empty 
            test exists in test_daf.py 

            if self is a single value, test the value.  # not implemented.
            
        """
        # if len(self.lol) == 1 and self.lol[0] == 1:
            # return bool(self.lol[0][0])
        
        return bool(self.num_cols())


    def __format__(self, format_spec):
        # Assuming the current object is a single cell when called in formatting
        # If a format_spec is provided, use it; otherwise, use __str__
        if format_spec:
            value = self.to_value()
            if isinstance(value, (int, float)):
                return format(value, format_spec)
            return str(value)
        return self.__str__()


    def __eq__(self, other):
        # test exists in test_daf.py            

        if not isinstance(other, Daf):
            return False

        return (self.lol == other.lol and self.columns() == other.columns() and self.keyfield == other.keyfield)

    
    def __str__(self) -> str:
        return self.md_daf_table_snippet()
        
        
    def __repr__(self) -> str:
        return "\n"+self.md_daf_table_snippet()
    
    
    def __contains__(self, key) -> bool:

        if not self.kd:
            raise KeyError("The dictionary 'kd' is not initialized.")
        
        return key in self.kd
        

    #===========================
    # size and shape
    
    # Please note that daffodil supports pre-allocated arrays with None values to speed appending.


    def num_cols(self) -> int:
        """ return 0 if self.lol is empty.
            return the max length of the first up to 10 rows otherwise.
            
            this only works well if the array has rows that are all the same length.            
        """
        # unit tested
    
        if not self.lol:
            return 0
        _, result = daf_utils.min_max_cols_lol(self.lol, limit=10)
        return result
        
        
    def __len__(self):
        """ Return the number of rows in the Daf instance.
        """
        return self.num_rows()
        
        
    def num_rows(self):

        if not self.lol:
            return 0

        return len(self.lol)
        
        
    def len(self):
        return self.num_rows()
        
        
    def shape(self):
        """ return the number of rows and cols in the daf data array
            number of columns is based on the first record
        """
        # test exists in test_daf.py
        
        if not len(self): 
            return (0, 0)
        
        return (self.num_rows(), self.num_cols()) 
        
        
    #===========================
    # copying convenience function to mimic pandas syntax.

    def copy(self, deep: bool = False) -> 'Daf':
        """Creates a copy of the Daf instance, ensuring `attrs` is copied correctly."""
        if deep:
            new_instance = copy.deepcopy(self)  # Fully independent copy
        else:
            new_instance = copy.copy(self)  # Shallow copy           
            new_instance.attrs = self.attrs.copy()  # Ensure metadata is copied but not linked
            
        return new_instance        

    #===========================
    # column names
    @staticmethod
    def _build_hd(keys: Iterable):
        # it is necessary to rebuild hd whenever it the array or cols is changed.
        # this is equivalent to:
        #   {col: idx for idx, col in enumerate(keys)}
        # but this is substantially faster
        # see https://github.com/raylutz/daffodil/issues/6
        # self.hd = {col: idx for idx, col in enumerate(dtypes.keys())}
        
        # usage here: self.hd = type(self)._build_hd(keys)
        
        # TODO: Consider building this so that any duplicates that are
        # encountered will not overwrite the prior key

        return dict(zip(keys, range(len(keys))))

    def columns(self):
        """ Return the column names 
        """
        # test exists in test_daf.py            
        return list(self.hd.keys())
        
        
    def _cols_to_hd(self, cols: T_ls):
        """ rebuild internal hd from cols provided """
        # see https://github.com/raylutz/daffodil/issues/6
        self.hd = type(self)._build_hd(cols)
        #self.hd = {col:idx for idx, col in enumerate(cols)}
        
        
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
            include_cols: Optional[Iterable]=None,
            exclude_cols: Optional[Iterable]=None,
            include_types: Optional[List[Type]]=None,
            exclude_types: Optional[List[Type]]=None,
           ) -> Iterable:
        """ this method helps to calculate the columns to be specified for a apply or reduce operation.
            Can use any combination of listing columns to be included, or excluded by name,
                or included by type.
            If using a groupby function, the cols spec should not include the groupby column(s)
        """
            
            
        # start with all cols.
        selected_cols = self.hd.keys()  # change to columns()
        if not selected_cols:
            return []
            
        if include_cols:
            if isinstance(include_cols, str):
                include_cols = [include_cols]
            # if len(include_cols) > 10:
                # include_cols_dict = dict.fromkeys(include_cols)
                # selected_cols = [col for col in selected_cols if col in include_cols_dict]
            else:
                selected_cols = [col for col in selected_cols if col in include_cols]

        if exclude_cols:
            if isinstance(exclude_cols, str):
                exclude_cols = [exclude_cols]
            # if len(exclude_cols) > 10:
                # exclude_cols_dict = dict.fromkeys(exclude_cols)
                # selected_cols = [col for col in selected_cols if col not in exclude_cols_dict]
            else:
                selected_cols = [col for col in selected_cols if col not in exclude_cols]

        if include_types:
            if not self.dtypes:
                breakpoint() #perm
                pass
                raise RuntimeError
                
            if not isinstance(include_types, list):
                include_types = [include_types]
            selected_cols = [col for col in selected_cols if self.dtypes.get(col) in include_types]

        if exclude_types:
            if not self.dtypes:
                breakpoint() #perm
                pass
                raise RuntimeError
                
            if not isinstance(exclude_types, list):
                exclude_types = [exclude_types]
            selected_cols = [col for col in selected_cols if self.dtypes.get(col) not in exclude_types]

        return selected_cols
        
    # @staticmethod
    # def normalize(da: T_da, defined_cols: T_ls):
        # """ if given a dict, fill out any columns that exist to match 'defined_cols'
        # """

        # if not self:
            # return
        
        # # from utilities import daf_utils

        # for key in defined_cols:
        
            # record_da = daf_utils.set_cols_da(da, defined_cols)
            
            # self.update_record_irow(irow, record_da)
            
        # return self
        

    def rename_cols(self, from_to_dict: T_ds):
        """ rename columns using the from_to_dict provided. 
            respects dtypes and rebuilds hd
            clears keyfield, it must be reset
        """
        # unit tests exist
        
        self.hd         = {from_to_dict.get(col, col):idx for idx, col in enumerate(self.hd.keys())}
        self.dtypes     = {from_to_dict.get(col, col):typ for col, typ in self.dtypes.items()}
        self.keyfield   = ''
            # self.keyfield = from_to_dict.get(self.keyfield, self.keyfield)
            # # no need to rebuild the kd, it should be the same.
            
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
            new_cols = daf_utils._generate_spreadsheet_column_names_list(num_cols)
            
        elif sanitize_cols:
            new_cols = daf_utils._sanitize_cols(new_cols, unnamed_prefix=unnamed_prefix)
        
        if num_cols and len(new_cols) < num_cols:
            raise AttributeError("Length of new_cols not the same as existing cols")
        
        if self.keyfield and isinstance(self.keyfield, str) and self.hd:
            # if column names are already defined (hd) then we need to repair the keyfield.
            # will only be done if keyfield is a simple string or integer column name.
            try:
                keyfield_idx = self.hd.get[self.keyfield]
                self.keyfield = new_cols[keyfield_idx]
            except Exception:
                self.keyfield = ''
        else:
            self.keyfield = ''
        
        # set new cols to the hd
        self._cols_to_hd(new_cols)
        
        # convert dtypes dict to use the new names.
        if self.dtypes:
            self.dtypes = dict(zip(new_cols, self.dtypes.values()))
            
        return self
            

    #===========================
    # keyfield
        
    def keys(self) -> Union[T_la, T_lota]:
        """ return list of keys from kd of keyfield
            may return a list of str or int or list of tuple of (str or int).
        
            test exists in test_daf.py
            
            can also just use daf.kd   <-- to get a dict with keys, useful for "in" operation.
            
        """
        
        if not self.keyfield:
            return []
        
        return list(self.kd.keys())
        

    def set_keyfield(self, keyfield: Union[str, T_ta, T_la]='', silent_error: bool=True):
        """ set the indexing keyfield to a new column
            if keyfield == '', then reset the keyfield and reset the kd.
            if keyfield not in columns, 
                then KeyError, if not silent_error
                else set it anyway.
            Otherwise, set key field
            keyfield can be a tuple or list of colnames.
            
            if self is empty do nothing.
        """
        if not self:
            return self
        
        if keyfield:
            if not self._is_keyfield_valid(keyfield):
                if not silent_error:
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
        
        if self._is_keyfield_valid():
            if isinstance(self.keyfield, (str, int)):
                col_idx = self.hd[self.keyfield]
                self.kd = type(self)._build_kd(col_idx, self.lol)
            else:
                col_idx_list = [self.hd[key_tup] for key_tup in self.keyfield]
                self.kd = type(self)._build_kd(col_idx_list, self.lol)
        return self


    @staticmethod
    def _build_kd(col_idx: Union[int, T_li], lol: T_lola) -> T_di:
        """ build key dictionary from col_idx col of lol """        
        if isinstance(col_idx, int):
            key_col = daf_utils.select_col_of_lol_by_col_idx(lol, col_idx)

            # see https://github.com/raylutz/daffodil/issues/6
            kd = Daf._build_hd(key_col)
            #kd = {key: index for index, key in enumerate(key_col)}
        else:
            col_idx_list = col_idx
            kd = Daf._build_hd(Daf(lol=lol)[:, col_idx_list].to_lota())
        return kd

        
    def _get_keyval(self, data_item):                  
        if isinstance(self.keyfield, (str, int)):
            keyval = data_item[self.keyfield]
        elif isinstance(self.keyfield, (tuple, list)):
            keyval = tuple((data_item[key_tup] for key_tup in self.keyfield))
        return keyval


    def _is_keyfield_valid(self, keyfield: str=''):
        """
            evaluate self.keyfield or passed keyfield instead,
            to confirm that the key is composed of valid colnames,
            either as a single str or int, or as a tuple of str or int.
        """
        
        keyfield_to_test = keyfield or self.keyfield
            
    
        if isinstance(keyfield_to_test, (str, int)):
            return bool(keyfield_to_test in self.hd)
        elif isinstance(keyfield_to_test, (tuple, list)):
            return all(key_tup in self.hd for key_tup in keyfield_to_test)

        raise RuntimeError(f"keyfield '{keyfield_to_test}' invalid")
    

        
    # def row_idx_of(self, rowkey: str) -> int:
        # """ return row_idx of key provided or -1 if not able to do it.
        # """
        # # unit tested
        
        # # for the following, see https://github.com/raylutz/daffodil/issues/7

        # if not self.keyfield or not self.kd:
            # return -1
        # return self.kd.get(rowkey, -1)
        

    def get_existing_keys(self, keylist: T_ls) -> T_ls:
        """ check the keylist against the keys defined in a daf instance. 
        """
        # unit tested
    
        return [key for key in keylist if key in self.kd]


    
    #===========================
    # dtypes
    
    def set_dtypes(self, 
            default_type: Type = str, 
            typ_to_cols_dict: Optional[Dict[Type, T_ls]] = None,
            ) -> 'Daf':
            
        """ set dtypes from default and dol where the key of the dict is the type, 
            and the list is the column names of that type.
            Useful if most columns are the same type and there are only a few exceptions.
            
            Otherwise, simply set my_daf.dtypes = dtypes dict, and then 
            apply_dtypes() when reading files.
            writing files will generally automatically flatten.
            
            This function does NOT apply the types.            
            
            @@TODO May want to provide a simpler function that sets a type to a set of columns.
        """
            
        if not self.hd:
            raise NotImplementedError ("self.hd must be defined to use .set_dtypes()")
        
        dtypes = {}
        
        col_to_typ_dict = daf_utils.invert_dol_to_dict(typ_to_cols_dict)
        
        for colname in self.hd:
            
            if colname in col_to_typ_dict:
                dtypes[colname] = col_to_typ_dict[colname]
            else:
                dtypes[colname] = default_type

        self.dtypes = dtypes
        
        return self
        
    
    
    def apply_dtypes(self, *, 
            dtypes:         Optional[T_dtype_dict]=None, 
            unflatten:      bool=True, 
            from_str:       bool=True,
            default_type:   Type=str,
            ) -> 'Daf':
        """ convert columns of daf array to the datatypes specified in self.dtypes or in passed parameter.
                dtypes can be a dict where each column may have a different type, or it can be a single type.
                dtypes may provide desired new types for only some of the columns.
            columns (self.hd) must be defined.
            unflatten: unflatten from pyon or json to list or dict types
            if from_str is True, (default) assumes that data starts as str, such as when read from csv
                If the csv is first read, data is all delivered as str.
                Columns specified as str type are not converted.
            
            Note, this converts types in place, and does not create a new array.
        """
        
        if dtypes:
            self.dtypes = dtypes
        
        if not self.hd and self.dtypes:
            self._cols_to_hd(dtypes.keys())
            # currently will not overwrite existing cols.
            # (i.e. pass partial dtypes will alter only those cols)
            
        if not self.lol or not self.lol[0]:
            # this can sometimes happen, no worries.
            return self
            
        # first calculate all columns to consider, that are not str or str and unflatten
        cols = self.hd.keys()
        for col in cols:
            if isinstance(self.dtypes, dict):           # this is the normal case, where each column is defined.
                if col not in self.dtypes:
                    desired_type = default_type         # type not specified for a column, but it exists, use default_type
                    self.dtypes[col] = default_type     # make sure self.dtypes is updated.
                desired_type = self.dtypes[col]
            else:
                desired_type = self.dtypes              # only one type is defined.
                
            if (    desired_type is str and from_str or 
                    desired_type in [list, dict] and not unflatten
                ):
                continue
        
            # update this column if needed.
            icol = self.hd[col]
            
            for irow in range(len(self.lol)):
            
                self.lol[irow][icol] = daf_utils.convert_type_value(self.lol[irow][icol], desired_type) # unflatten=True
                        
        return self
        
        
    def flatten(self, convert_bool_to_int=True, use_pyon: bool = True):
        """ convert any columns specified in dtypes as either list or dict, and
            flatten using PYON or JSON encoding. Modifies the array in place. If use_pyon, will 
            flatten arbitrary nested objects, functions or methods.
            
            Note, if using pyon encoding, there is no need to flatten prior to writing the csv data, 
                as it is converted using csv.writer using the __repr__ method for the object.
            
            if desired type is bool, it will convert to int, if convert_bool_to_int is True
            
            This should be envoked just prior to saving the data to a csv file if use_pyon = False.
            Otherwise, there is no need to flatten the data, it will happen automatically.
            
            Historical Reference: https://legacy.python.org/workshops/1994-11/FlattenPython.html
        """
        
        use_pyon = True

        if not self.lol or not self.lol[0] or not self.dtypes:
            # this can sometimes happen, no worries.
            # when daf is constructed internally and has no list or dict types, there is no need to flatten.
            # in this case, the self.dtypes need not be initialized or used until the table is read.
            return self
            
        if not self.hd:     # pragma: no cover
            # should not be the case. Logic error.
            breakpoint() #perm
            pass
            raise RuntimeError
            
        # first calculate all columns to consider
        cols = self.hd.keys()
        for col in cols:
            if isinstance(self.dtypes, dict):
                if col not in self.dtypes:
                    continue
                desired_type = self.dtypes[col]
            else:
                desired_type = self.dtypes
                
            if desired_type in (list, dict):
        
                icol = self.hd[col]
                
                for irow in range(len(self.lol)):
                
                    if use_pyon:
                        self.lol[irow][icol] = f"{self.lol[irow][icol]}"
                    else:    
                        self.lol[irow][icol] = daf_utils.json_encode(self.lol[irow][icol])
                    
            if convert_bool_to_int and desired_type is bool:
                # this should be rare!
                
                icol = self.hd[col]
                
                for irow in range(len(self.lol)):
                
                    self.lol[irow][icol] = int(bool(self.lol[irow][icol]))

        return self


    # def unflatten_cols(self, cols: T_ls):
        # """ 
            # given a daf and list of cols, 
            # convert cols named to either list or dict if col exists and it appears to be 
                # stringified using f"{}" functionality.
                
        # """

        # if not self:
            # return    
            
        # # from utilities import daf_utils

        # self.hd, self.lol = daf_utils.unflatten_hdlol_by_cols((self.hd, self.lol), cols)    
            
        # return self


    # def unflatten_by_dtypes(self):
        # # deprecated. Use unflatten()

        # if not self or not self.dtypes:
            # return self
                
        # unflatten_cols = self.calc_cols(include_types = [list, dict])
        
        # if not unflatten_cols:
            # return self
       
        # self.unflatten_cols(unflatten_cols)
            
        # return self
        
        
    # def flatten_cols(self, cols: T_ls):
        # # given a daf, convert given list of columns to json.

        # if not self:
            # return self
        
        # # from utilities import daf_utils

        # for irow, da in enumerate(self):
            # record_da = copy.deepcopy(da)
            # for col in cols:
                # if col in da:
                    # record_da[col] = daf_utils.json_encode(record_da[col])        
            # self.update_record_irow(irow, record_da)        
            
        # return self
    
    
    # def flatten(self):

        # if not self or not self.dtypes or not self.hd:
            # return self
                
        # flatten_cols = self.calc_cols(include_types = [list, dict])
        
        # if not flatten_cols:
            # return self
       
        # self._flatten_cols(cols=flatten_cols)
            
        # return self
        
        
    def strip(self, chrs: str=' ') -> 'Daf':
    
        """ remove leading or trailing characters in the string chrs from each str value in the array.
            modifies in-place.  Ignores non str values.
        
            each character in chrs treated seperately, such as strip('"()') removes quotes and parens.
        """    
    
        for row_la in self.lol:
            for icol in range(len(row_la)):
                if isinstance(row_la[icol], str) and row_la[icol]:
                    row_la[icol] = row_la[icol].strip(chrs)

        return self
       

    def _safe_tofloat(val: Any) -> Union[float, str]:
        try:
            return float(val)
        except ValueError:
            return 0.0
    
    #===========================
    # initializers
    
    def clone_empty(self, lol: Optional[T_lola]=None, cols: Optional[T_ls]=None, name:str='') -> 'Daf':
        """
        Create a new empty Daf instance from self, adopting column names but not data.
        
        - Adopts `keyfield` but does not adopt `kd` or `attrs` (metadata is not propagated).
        - If `lol` is provided as an argument, it is used in the new Daf.
        - This method creates a fresh instance with the same structure but no metadata.
        
        If metadata needs to be propagated, it must be done manually:
            new_daf.attrs = daf.attrs.copy()
        
        Returns:
            A new `Daf` instance with the same column structure but no data (or using passed data lol).
        """
        if self is None:
            return Daf()
            
        new_cols = cols if cols else self.columns()
        
        new_daf = Daf(cols=new_cols, lol=lol, keyfield=self.keyfield, dtypes=copy.copy(self.dtypes), name=name)
        
        # Ensure metadata attributes are deeply copied
        new_daf.attrs = copy.deepcopy(self.attrs)

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
            records_lod:    T_loda,                             # List[List[Any]] to initialize the lol data array.
            keyfield:       Union[str, int, T_ta, T_la] = '',   # set a new keyfield or set no keyfield.
            dtypes:         Optional[T_dtype_dict]      = None, # set the data types for each column.
            name:           str                         = '',   # Optional name of the daffodil instance.
            ) -> 'Daf':
        """ Create Daf instance from loda type, adopting dict keys as column names
            Generally, all dicts in records_lod should be the same OR the first one must have all keys
                and others can be missing keys.
            However, if dtypes is provided, it will be used to establish the columns.
        
            test exists in test_daf.py
            
            my_daf = Daf.from_lod(sample_lod)
        """
        if dtypes is None:
            dtypes = {}
        
        if not records_lod:
            return cls(keyfield=keyfield, dtypes=dtypes)
        
        if not dtypes:
            cols = list(records_lod[0].keys())
        else:
            cols = list(dtypes.keys())
        
        # is this better than just appending dicts?
        
        lol = [list(daf_utils.set_cols_da(record_da, cols).values()) 
                for record_da in records_lod if record_da and isinstance(record_da, dict)]
        
        return cls(cols=cols, lol=lol, keyfield=keyfield, dtypes=dtypes, name=name)


    # ==== Python lot (list of tuples)
    @classmethod
    def from_lot(
            cls,
            records_lot:    T_lota,                             # List of tuples to initialize the lol data array.
            cols:           Optional[T_ls]              = None, # Optional column names to use.
            dtypes:         Optional[T_dtype_dict]      = None, # Optional dtype_dict describing the desired type of each column.
                                                                #   also used to define column names if provided and cols not provided.
            keyfield:       Union[str, int, T_ta, T_la] = '',   # A field of the columns to be used as a key.
            name:           str                         = '',   # Optional name of the daffodil instance.
            ) -> 'Daf':
        """
        Create Daf instance from LOT (list of tuples), adopting given column names or generating default ones.

        Args:
            records_lot (List[Tuple[Any, ...]]):                    The LOT data.
            columns (Optional[List[str]]):                          Column names for the tuples. If None, default names will be generated.
            keyfield (Union[str, int, Tuple[Any, ...], List[Any]]): Keyfield for the Daf instance.
            dtypes (Optional[Dict[str, type]]):                     Data types for each column.

        Returns:
            Daf: A new Daf instance with data populated from lot.

        Example:
            records_lot = [(1, 'Alice', 30), (2, 'Bob', 25)]
            columns = ['id', 'name', 'age']
            my_daf = Daf.from_lot(records_lot, cols=cols)
        """
        if not records_lot:
            return cls(keyfield=keyfield, dtypes=dtypes)

        if cols is None:
            # Generate default column names if none are provided
            cols = [f'col_{i}' for i in range(len(records_lot[0]))]

        if dtypes is None:
            dtypes = {}

        # Check for mismatch between column names and tuple length
        if any(len(row) != len(cols) for row in records_lot):
            raise ValueError("Each tuple in records_lot must have the same number of elements as the columns.")

        # Convert LOT to LOL (list of lists) for Daf
        lol = [list(row) for row in records_lot]

        return cls(cols=cols, lol=lol, keyfield=keyfield, dtypes=dtypes, name=name)

        
        
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
        return cls.from_lod(daf_utils.dod_to_lod(dod, keyfield=keyfield), keyfield=keyfield, dtypes=dtypes)
    
    
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
        return daf_utils.lod_to_dod(self.to_lod(), keyfield=self.keyfield, remove_keyfield=remove_keyfield)
    

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

        csv_buff = daf_utils.xlsx_to_csv(excel_buff)
        
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
    def from_csv(cls, source: str | Path, **kwargs):
        """
        Load a CSV file from a local file, URL, or S3 path into a Daf array.

        - Supports streaming for large files.
        - `requests` and `boto3` are only imported if needed.
        
        - All data is imported as str types. use apply_dtypes() to convert data types.
        - Does not set the keyfield, 

        Args:
            source (str): File path, URL (http/https), or S3 path (s3://bucket/key).
            **kwargs: Additional arguments passed to from_csv_buff().

        Returns:
            Daf: A Daf array loaded from the CSV.
        """
        if isinstance(source, Path):  # Convert Path object to string
            source = str(source)

        if source.startswith(('http://', 'https://')):  # Handle HTTP(S) URLs
            try:
                import requests  # Import only if needed
                response = requests.get(source, stream=True)
                response.raise_for_status()
                data_stream = (line.decode("utf-8") for line in response.iter_lines() if line)
            except ImportError:
                raise RuntimeError("Missing `requests` module. Install it via `pip install requests`.")
            except requests.RequestException as e:
                raise RuntimeError(f"Failed to download CSV from {source}: {e}")

        elif source.startswith("s3://"):  # Handle S3 Paths with streaming
            try:
                import boto3  # Import only if needed
                s3 = boto3.client("s3")
                bucket, key = source[5:].split("/", 1)  # Extract bucket and key
                obj = s3.get_object(Bucket=bucket, Key=key)
                data_stream = (line.decode("utf-8") for line in obj["Body"].iter_lines() if line)
            except ImportError:
                raise RuntimeError("Missing `boto3` module. Install it via `pip install boto3`.")
            except Exception as e:
                raise RuntimeError(f"Failed to read CSV from S3: {e}")

        else:  # Assume local file with streaming
            try:
                with open(source, "r", encoding="utf-8") as f:  # Use `with` to ensure closure
                    return cls.from_csv_buff(csv_buff=f, **kwargs)
            except Exception as e:
                raise RuntimeError(f"Failed to read local file: {e}")

        new_daf = cls.from_csv_buff(csv_buff=data_stream, **kwargs)
        
        return new_daf

    # STILL ACTIVE -- use when we know the source is a buffer.
    @classmethod
    def from_csv_buff(
            cls,
            csv_buff: Union[bytes, str, Iterator[str]], # Can now accept iterators directly
            keyfield: str='',                           # field to use as unique key, if not ''
            dtypes: Optional[T_dtype_dict]=None,        # dictionary of types to apply if set.
            noheader: bool=False,                       # if True, do not try to initialize columns in header dict.
            user_format: bool=False,                    # if True, preprocess the file and omit comment lines.
            sep: str=',',                               # field separator.
            unflatten: bool=True,                       # unflatten fields that are defined as dict or list.
            include_cols: Optional[T_ls]=None,          # include only the columns specified. noheader must be false.
            name: str = '',                             # name attribute of the Daf array created.
            ) -> 'Daf':
        """
        Convert CSV data in a buffer (string, bytes, or iterator) to a daf object
        
        Note: probably just use from_csv() and then apply_dtypes(), remove 'unflatten',
            however, in the future of a more optimized conversion is written, then python
            types can be created as the csv is scanned rather than scanning again.

        - Fully supports streaming CSVs (does not require full file in memory).
        - Directly reads data into a list of lists (LoL) from an iterator.
        
        Args:
            csv_buff: Union[bytes, str, Iterator[str]], # Can now accept iterators directly
            keyfield: str='',                           # field to use as unique key, if not ''
            dtypes: Optional[T_dtype_dict]=None,        # dictionary of types to apply if set.
            noheader: bool=False,                       # if True, do not try to initialize columns in header dict.
            user_format: bool=False,                    # if True, preprocess the file and omit comment lines.
            sep: str=',',                               # field separator.
            unflatten: bool=True,                       # unflatten fields that are defined as dict or list.
            include_cols: Optional[T_ls]=None,          # include only the columns specified. noheader must be false.
            name: str = '',                             # name attribute of the Daf array created.

        Returns:
            Daf: The loaded Daf array.
        """
        
        # in the case of bytes, this will set up a conversion of the stream.
        # Converts bytes into a file-like object without reading everything at once.
        if isinstance(csv_buff, bytes):
            csv_buff = io.TextIOWrapper(io.BytesIO(csv_buff), encoding="utf-8")

        # the following will be able to stream from the source and convert directly to lol.
        data_lol = daf_utils.buff_csv_to_lol(csv_buff, user_format=user_format, sep=sep, include_cols=include_cols, dtypes=dtypes)
        
        cols = []
        if not noheader:
            cols = data_lol.pop(0)        # return the first item and shorten the list.
        
        my_daf = cls(lol=data_lol, cols=cols, keyfield=keyfield, dtypes=dtypes, name=name)
        
        # the following will act nicely if there is no dtypes defined.
        my_daf.apply_dtypes(unflatten=unflatten)
   
        return my_daf
    

    # DEPRECATED
    @classmethod
    def from_csv_file(
            cls,
            filepath: str,                          # The CSV filename
            keyfield: str='',                       # field to use as unique key, if not ''
            dtypes: Optional[T_dtype_dict]=None,    # dictionary of types to apply if set.
            noheader: bool=False,                   # if True, do not try to initialize columns in header dict.
            user_format: bool=False,                # if True, preprocess the file and omit comment lines.
            sep: str=',',                           # field separator.
            unflatten: bool=True,                   # unflatten fields that are defined as dict or list.
            include_cols: Optional[T_ls]=None,      # include only the columns specified. noheader must be false.
            name: str = '',                         # name attribute of the Daf array created.
            ) -> 'Daf':                             # New daf instance
                
        """ Read a csv file directly into a daf array in memory, per arguments. 
        
            Args:
                filepath: str,                          # The CSV filename
                keyfield: str='',                       # field to use as unique key, if not ''
                dtypes: Optional[T_dtype_dict]=None,    # dictionary of types to apply if set.
                noheader: bool=False,                   # if True, do not try to initialize columns in header dict.
                user_format: bool=False,                # if True, preprocess the file and omit comment lines.
                sep: str=',',                           # field separator.
                unflatten: bool=True,                   # unflatten fields that are defined as dict or list.
                include_cols: Optional[T_ls]=None,      # include only the columns specified. noheader must be false.
                name: str = '',                         # name attribute of the Daf array created.
            Returns
                New daf instance
        
        """
    
        try:
            with open(filepath) as f:
                csv_buff = f.read()
        except Exception as err:
            print(f"Error reading the file: {err}")
            return None

        return Daf.from_csv_buff(
            csv_buff    = csv_buff,                 # The CSV data as bytes or string.
            keyfield    = keyfield,                 # field to use as unique key, if not ''
            dtypes      = dtypes,                   # dictionary of types to apply if set.
            noheader    = noheader,                 # if True, do not try to initialize columns in header dict.
            user_format = user_format,              # if True, preprocess the file and omit comment lines.
            sep         = sep,                      # field separator.
            unflatten   = unflatten,                # unflatten fields that are defined as dict or list.
            include_cols    = include_cols,         # include only the columns specified. noheader must be false.
            name        = name,                     # name attribute of the Daf array created.
            )
 

    def to_csv_file(
            self,
            file_path:          str | Path = '',
            line_terminator:    Optional[str]=None,
            include_header:     bool=True,
            #append_if_exists:   bool=False,
            ) -> str:
             
        if isinstance(file_path, Path):  # Convert Path object to string
            file_path = str(file_path)

        buff = self.to_csv_buff(
                line_terminator=line_terminator,
                include_header=include_header,
                )

        type(self).buff_to_file(buff, file_path=file_path, fmt='.csv') #, append=append_mode)
        
        return file_path


    def to_csv_buff(
            self, 
            line_terminator: Optional[str]=None,
            include_header: bool=True,
            ) -> T_buff:
        """ this function writes the daf array to a csv buffer, including the header if include_header==True.
            The buffer can be saved to a local file or uploaded to a storage service like s3.
            
            Use .flatten() before calling this and use JSON encoding for any objects that are not str, int, float, or bool.
            
            However, this function will automatically flatten any nested objects, according to __repr__ for that object.
                This differs from pure JSON format because:
                    1. it uses single-quotes instead of double quotes around strings.
                    2. it allows non-str keys in dicts.
                    3. it encodes True/False as 'True'/'False' instead of 'true'/'false'
            
        """
    
        if line_terminator is None:
            line_terminator = '\r\n'
    
        f = io.StringIO(newline = '')           # Use newline='' to ensure consistent line endings
        
        csv_writer = csv.writer(f, lineterminator=line_terminator)
        if include_header:
            csv_writer.writerow(self.columns())     # Write the header row
        csv_writer.writerows(self.lol)              # Write the data rows

        buff = f.getvalue()
        f.close()

        return buff   


    @staticmethod
    def buff_to_file(buff: T_buff, file_path: str | Path, fmt:str='.csv'):
    
        return daf_utils.write_buff_to_fp(buff, file_path, fmt=fmt)

    #==== PDF to Daf
    
    from_pdf = daf_pdf._from_pdf
    from_pdf = from_pdf                             # fool linter.
    
    #==== Pandas
    #@classmethod
    from_pandas_df = daf_pandas._from_pandas_df
    from_pandas_df = from_pandas_df                 # fool linter.
                
    to_pandas_df = daf_pandas._to_pandas_df
    to_pandas_df = to_pandas_df                     # fool linter.
        
            
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
    

    def to_numpy(self) -> T_npa:
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
        
    def to_donpa(self, colnames: Optional[T_ls]=None, default: Any = _MISSING) -> T_donpa:
        """
        Convert specified columns of the Daffodil table to a dict of NumPy arrays (donpa).

        Parameters:
            colnames:
                List of column names to include. If None, include all columns.
            default:
                Optional replacement for missing values ('' or None).
                If specified, all missing values in the selected columns will be replaced
                with this value before conversion to NumPy arrays.

        Returns:
            Dict[str, np.ndarray], where each array corresponds to a 1D column vector.

        Notes:
            - Column arrays can be used directly in vector operations:
                e.g., `my_donpa['D_pct'] = my_donpa['D_votes'] / my_donpa['RV_total']`
            - This structure is conceptually similar to a Pandas DataFrame, but lighter-weight and faster
              for numeric computations or export.
        """        
        if colnames is None:
            colnames = self.columns()
        
        import numpy as np
        donpa = {col: np.array(self[:, col].to_list(default=default)) for col in colnames}
        
        return donpa
        
        
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
            
        # NOT OPERATIONAL
            
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
        
        cols = daf_utils._generate_spreadsheet_column_names_list(num_cols)

        gs_daf = cls(cols=cols, lol=lol)
        
        return gs_daf
        
        
    def to_googlesheet(self, spreadsheet_id: str, sheetname: str = 'Sheet1') -> 'Daf':
        """ export data from daf structure to googlesheet. """
        # NOT OPERATIONAL
    
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
    # JSON compatible representation.
    
    # there are some limitations to JSON encoding, as it does not correctly allow
    # for non-str keys of dictionaries. The kd can be corrected if the keyfield is
    # defined and there is a dtypes that defines the type of that column, so it can
    # be corrected if it is not a str. Also, the dtypes dict cannot contain values
    # that are types.

    # Define a mapping of string representations to Python types
    TYPE_MAP = {
        'int': int,
        'float': float,
        'str': str,
        'bool': bool,
        # Add more types as needed
    }

    def to_json(self) -> str:
        # Convert data types to string representations using the TYPE_MAP
        dtypes_str = {k: self.TYPE_MAP.get(v, v).__name__ for k, v in self.dtypes.items()}
        
        # Serialize Daf object to a JSON-compatible dictionary
        daf_dict = {
            'name': self.name,
            'lol': self.lol,
            'hd': self.hd,
            'kd': self.kd,
            'dtypes': dtypes_str,
            'keyfield': self.keyfield,
            '_retmode': self._retmode,
            '_itermode': self._itermode,
        }
        # Convert dictionary to JSON string
        return json.dumps(daf_dict)

    @classmethod
    def from_json(cls, json_str: str) -> 'Daf':
        # Deserialize JSON string into a Daf object
        daf_dict = json.loads(json_str)
        
        # Convert string representations of data types back to actual types
        dtypes = {k: cls.TYPE_MAP.get(v, v) for k, v in daf_dict.get('dtypes', {}).items()}
        
        # if keyfield is not str type, it must be converted bc JSON converts all keys to str.
        keyfield_dtype = dtypes.get(daf_dict.get('keyfield', ''), str)

        kd = daf_dict.get('kd', {})
        if keyfield_dtype is not str:
            kd = {keyfield_dtype(k): v for k, v in kd.items()}
        
        return cls(lol=daf_dict.get('lol', []),
                   hd=daf_dict.get('hd', {}),
                   kd=kd,
                   dtypes=dtypes,
                   keyfield=daf_dict.get('keyfield', ''),
                   name=daf_dict.get('name', ''),
                   retmode=daf_dict.get('_retmode', 'obj'),
                   itermode=daf_dict.get('_itermode', 'dict'))
                   
    #===========================
    # convert to other format
    
    # #@deprecated("use Daf instead")
    # def to_hllola(self) -> T_hllola:
        # """ Create hllola from daf 
            # test exists in test_daf.py
            
            # DEPRECATED
        # """    
        # return (list(self.hd.keys()), self.lol)
        
    #===========================
    # append
        
    def append(self, data_item: Union[T_Daf, T_loda, T_da, T_la, KeyedList]):
        """ general append method can handle appending one record as T_da or T_la, many records as T_loda or T_daf
            if the data item is None or empty, do not append.
        """
        # test exists in test_daf.py for all three cases
        
        diagnose = False
        
        if not data_item:
            return self
        
        if diagnose:
            start_time = time.time()
            logs.sts(f"{logs.prog_loc()} starting append.", 3)
        
        if isinstance(data_item, (dict, KeyedList)):
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
                    self.record_append(da)  # <-- this takes care of respecing the row kd.
                else:    
                    # no columns defined, therefore just append to lol.
                    self.lol.append(data_item)
                
        elif isinstance(data_item, KeyedList):
            # hd matches.
            if self.hd == data_item.hd:
                self.lol.append(data_item.values())        
                if self.kd and self.keyfield:
                    keyval = self._get_keyval(data_item)

                    self.kd[keyval] = len(self.kd)
                    
                    
            else:
                da = data_item.to_dict()
                self.record_append(da)  # <-- this takes care of respecing the row kd.
                
        elif isinstance(data_item, Daf):  # type: ignore
            self.concat(data_item)
        else:    
            raise RuntimeError    # pragma: no cover

        if diagnose:
            logs.sts(f"{logs.prog_loc()} append done. Elapsed: {(time.time() - start_time):.10f}", 3)
            
        return self
        

    def concat(self, other_instance: 'Daf'):
        """Concatenate records from the passed Daf instance into self.
        
        This directly modifies self.
        - If keyfield is '', insert without respecting the key value.
        - Otherwise, allow only one record per key.
        - Columns must be equal.
        - Test exists in test_daf.py.
        """
        
        if not other_instance:
            return 
        
        diagnose = False

        if diagnose:  # pragma: no cover
            print(f"self=\n{self}\ndaf=\n{other_instance}")
        
        if not self.lol and not self.hd:
            self.hd = copy.deepcopy(other_instance.hd)
            self.lol = copy.deepcopy(other_instance.lol)
            self.kd = copy.deepcopy(other_instance.kd)
            self.keyfield = other_instance.keyfield
            self._rebuild_kd()  # Only if the keyfield is set.
            return self
        
        # Fields must match exactly!
        if self.hd != other_instance.hd:
            _, missing_list, extra_list, _ = daf_utils.compare_lists(
                work_list=self.hd, 
                ref_list=other_instance.hd, 
                req_list=None,
            )
            
            logs.sts(f"{logs.prog_loc()} keys mismatch: this_instance ({self.name}):\n({list(self.hd.keys())}) \n"
                  f"other_instance:\n({list(other_instance.hd.keys())})\n"
                  f"missing_list:\n{missing_list}\n"
                  f"extra_list:\n{extra_list}\n", 3
            )
            # breakpoint()  # Permanent assertion break
            # pass
            raise KeyError
        
        # Append deep-copied rows to avoid referencing issues
        for rec_la in other_instance.lol:
            self.lol.append(copy.deepcopy(rec_la))
        
        self._rebuild_kd()  # Only if the keyfield is set.

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
            

    def record_append(self, record: Union[T_da, KeyedList]):
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
        
        if not record:
            return self
            
        if not self.lol and not self.hd:
            # new daf, adopt structure of da or keyedlist.
            # but there is only one header and data is lol
            # this makes daffodil arrays about 2/3 smaller.
            
            if isinstance(record, KeyedList):
                # for keyedlist, simply adopt the hd and the list as first row.
                self.hd = record.hd
                self.lol = [record.values()]
                
            elif isinstance(record, dict):
                self.hd = type(self)._build_hd(record.keys())
                self.lol = [list(record.values())]
                
            self._rebuild_kd()   # functions only if the keyfield is set.
            return self
            
        # check if fields match exactly.
        reorder = False
        if isinstance(record, KeyedList):
            if record.hd != self.hd:
                reorder = True
        elif list(self.hd.keys()) != list(record.keys()):
            reorder = True
        
        if reorder:
            # construct a dict with exactly the cols specified.
            # defaults to '' at this point.
            # this works for both dict and KeyedList
            rec_la = [record.get(col, '') for col in self.hd]
        # not reordering, slightly different    
        elif isinstance(record, KeyedList):
            rec_la = record.values()        # returns a list.
        elif isinstance(record, dict):    
            rec_la = list(record.values())  # must copy into a list.
        else:
            breakpoint() #perm
            
        if self.keyfield:
            keyval = self._get_keyval(record)        
        
            # the following, see https://github.com/raylutz/daffodil/issues/7
            if keyval in self.kd:
                self.lol[self.kd[keyval]] = rec_la
            # idx = self.kd.get(keyval, -1)
            # if idx >= 0:
                # self.lol[idx] = rec_la
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

    def remove_key(self, keyval: Optional[Union[str, int, T_la, T_ta]], silent_error=False) -> None:
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
    # indexing

    def __getitem__(self,
            slice_spec:   Union[slice, int, str, T_li, T_ls, range, T_lor,
                                Tuple[  Union[slice, int, str, T_li, T_ls, range, T_lor, Tuple[str, str]], 
                                        Union[slice, int, str, T_li, T_ls, range, T_lor, Tuple[str, str]]]],
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
            
        if (isinstance(row_spec, (int, slice, range)) or 
                daf_utils.is_list_of_type(row_spec, (int, range))
            ):
            sel_rows_daf = self.select_irows(irows=row_spec)
            
        elif (isinstance(row_spec, str) or 
                daf_utils.is_list_of_type(row_spec, str) or
                daf_utils.is_tuple_of_type_len(row_spec, str, 2)
            ):
            sel_rows_daf = self.select_krows(krows=row_spec)

        else:
            sel_rows_daf = self

        if col_spec is None:
            ret_daf = sel_rows_daf
        else:
            if (isinstance(col_spec, (int, slice, range)) or  
                    daf_utils.is_list_of_type(col_spec, (int, range))
                ):
                ret_daf = sel_rows_daf.select_icols(icols=col_spec)
                
            elif (isinstance(col_spec, str) or 
                    daf_utils.is_list_of_type(col_spec, str) or
                    daf_utils.is_tuple_of_type_len(col_spec, str, 2)
                ):
                ret_daf = sel_rows_daf.select_kcols(kcols=col_spec)
            else:
                ret_daf = sel_rows_daf
            
        return ret_daf._adjust_return_val(self.retmode)    
    
        
    def __setitem__(self,
            slice_spec:   Union[slice, int, str, range, T_li, T_ls, T_lb, T_lor,
                                Tuple[  Union[slice, int, str, range, T_lor, T_li, T_ls, T_lb, Tuple[Any, Any]], 
                                        Union[slice, int, str, range, T_lor, T_li, T_ls, T_lb, Tuple[Any, Any]]]],
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
            
        if (isinstance(row_spec, (str, tuple)) or 
                daf_utils.is_list_of_type(row_spec, str) or
                daf_utils.is_tuple_of_type_len(row_spec, str, 2)):

            irows = self.krows_to_irows(krows = row_spec)
            
        elif isinstance(row_spec, (int, slice, range)) or daf_utils.is_list_of_type(row_spec, (int, range)):
            irows = row_spec
            
        if col_spec and isinstance(col_spec, (str, tuple)) or daf_utils.is_list_of_type(col_spec, str):
            icols = self.kcols_to_icols(kcols = col_spec)
            
        elif isinstance(col_spec, (int, slice, range)) or daf_utils.is_list_of_type(col_spec, (int, range)):
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
        

    def set_irows_icols(self, irows: Union[slice, int, T_li, Iterable, None], icols: Union[slice, int, T_li, None], value) -> 'Daf':
        """ set rows and cols in given daf.
            
            irows, icols: can be either a slice, int, or list of integers. These
                            refer to row/col indices that are inherent in the lol structure.
            
            mutates existing daf
        """
        if icols is None:
            icols = []
        if irows is None:
            irows = []
            
        tot_num_cols = self.num_cols()    
        tot_num_rows = self.num_rows()    
        
        num_irows = daf_utils.len_rowcol_spec(irows, tot_num_rows)
        num_icols = daf_utils.len_rowcol_spec(icols, tot_num_cols)
        
        if num_irows == 1 and isinstance(irows, int):
            irows = [irows]
        if num_icols == 1 and isinstance(icols, int):
            icols = [icols]
            
        if isinstance(irows, slice):
            irows = daf_utils.slice_to_range(irows, len(self))
        if isinstance(icols, slice):
            icols = daf_utils.slice_to_range(icols, self.num_cols())
            
        # special case when cols not specified.    
        if num_irows == 1 and num_icols == 0:
        
            irow = irows[0]
            
            if isinstance(value, list):
                self.lol[irow] = value
            elif isinstance(value, dict):
                self.assign_record_irow(irow, record=value)
            elif isinstance(value, type(self)):
                self.lol[irow] = value
            else:
                # set the same value in the row for all columns.
                self.lol[irow] = [value] * len(self.lol[irow])
                
        if num_irows == 1 and num_icols == 1:
        
            irow = irows[0]
            icol = icols[0]
            
            if isinstance(value, dict):
                self.assign_record_irow(irow, record=value)
            elif isinstance(value, (list, collections.abc.Iterable)) and len(value) == 1:    
                # frequently, we will have a list generated from a selection of a column, and it it has only one value
                # it needs to be entered in the array location, but not as a list.
                # place a list with only one item in a cell can't be done this way:
                
                # my_daf[0,0] = [4]   # this will insert 4 not [4]
            
                self.lol[irow][icol] = value[0]
            else:    
                self.lol[irow][icol] = value
                
        elif num_irows > 1 and num_icols == 0:
            
            if isinstance(value, (list, collections.abc.Iterable)):
                for irow in irows:
                    self.lol[irow] = value
            elif isinstance(value, dict):
                for irow in irows:
                    self.assign_record_irow(irow, record=value)
            elif isinstance(value, type(self)):
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
        
            if isinstance(value, (list, collections.abc.Iterable)):
                for source_idx, irow in enumerate(irows):
                    
                    try:
                        self.lol[irow][icol] = value[source_idx]
                    except Exception:
                        breakpoint() #perm ok
                        pass # size of column being does not match number of rows.
                    
            # elif isinstance(value, dict):
                # # this is the same as cols=0 bc dict updates the corresponding cols.
                # for irow in irows:
                    # self.assign_record_irow(irow, record=value)
                
            elif isinstance(value, type(self)):
                for source_idx, irow in enumerate(irows):
                    self.lol[irow][icol] = value[source_idx][0]
                
            else:
                # set the same value in the row for all selected columns.
                for irow in irows:
                    self.lol[irow][icol] = value
        else:
            if irows is None:
                irows = range(len(self.lol))
        
            if isinstance(value, (list, collections.abc.Iterable)):
                for irow in irows:
                    for source_col, icol in enumerate(icols):
                        try:
                            self.lol[irow][icol] = value[source_col]
                        except Exception:
                            breakpoint() #perm ok
                    
            elif isinstance(value, dict):
                # this is the same as cols=0 bc dict updates the corresponding cols.
                for irow in irows:
                    self.assign_record_irow(irow, record=value)
                
            elif isinstance(value, type(self)):
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
            krows:          Union[slice, str, T_la, int, Tuple[Any, Any], T_lota, Iterable, None],
            inverse:        bool = False,
            silent_error:   bool = False,
            ) -> Union[slice, int, T_li, range]:
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
                return range(len(self))
            else:
                return []
            
        return type(self).gkeys_to_idxs(
                    keydict         = self.kd,
                    gkeys           = krows,
                    inverse         = inverse,
                    silent_error    = silent_error,
                    axis            = 'rowkeys',     # for error message only
                    name            = self.name,
                    )
    
    def kcols_to_icols(self, 
            kcols: Union[str, T_ls, slice, int, T_li, Tuple[Any, Any], Iterable, None] = None,
            inverse: bool = False,
            silent_error: bool=False,
            ) -> Union[slice, int, T_li, range, None]:
        """
        If cols are defined is set, then the cols can be selected by providing a
        kcols parameter that will index the cols by using values in the header dict hd.
        This forces an attempt to use the spec as a column name even if it may look
        like an integer.
        """
        if not self.hd:
            if inverse:
                return range(self.num_cols())
            else:
                return []
        
        return type(self).gkeys_to_idxs(
                    keydict         = self.hd,
                    gkeys           = kcols,
                    inverse         = inverse,
                    silent_error    = silent_error,
                    axis            = 'colnames',     # for error message only
                    name            = self.name,
                    )
                    
    @staticmethod
    def gkeys_to_idxs(
            keydict:    Dict[Union[str, int], int],
            gkeys:      Union[str, T_ls, slice, int, T_li, Tuple[Any, Any], T_lota, Iterable, None] = None,
            inverse:    bool = False,
            silent_error: bool=False,
            axis:       str='rowkeys',              # used for status messages only.
            name:       str='unspecified',          # used for status messages only.
            ) -> Union[slice, int, T_li, range, None]:
        """
        If keydict is defined, then the idxs can be selected by providing a
        gkeys parameter that will index the keydict, and return either a 
        slice, int, T_li, or None, which will index the range.
        """
        
        if not keydict:
            return []
            
        elif isinstance(gkeys, (str, int)):
            idxs = []
            gkey = gkeys
            
            # For the following, see https://github.com/raylutz/daffodil/issues/6
            try:
                idxs.append(keydict[gkey])                
            except KeyError:
                if not silent_error:
                    # logs.sts(f"{logs.prog_loc()} Cannot find key '{gkey}' in {axis} in dataframe '{name}'", 3) 
                    raise 
            
        elif isinstance(gkeys, (list, collections.abc.Iterable)):     # can be list of integer or strings (or anything hashable)
            idxs = []
            for gkey in gkeys:
                # For the following, see https://github.com/raylutz/daffodil/issues/6
                try:
                    idxs.append(keydict[gkey])                
                except KeyError:
                    if not silent_error:
                        logs.sts(f"{logs.prog_loc()} Cannot find key '{gkey}' in {axis} in dataframe '{name}'", 3)
                        #breakpoint()
                        raise 
                    
            
        elif isinstance(gkeys, slice):     # can be list of integer or strings (or anything hashable)
            gkeys_range = range(gkeys.start or 0, gkeys.stop or len(keydict), gkeys.step or 1)
            idxs = []
            for gkey in gkeys_range:
                # For the following, see https://github.com/raylutz/daffodil/issues/6
                try:
                    idxs.append(keydict[gkey])                
                except KeyError:
                    if not silent_error:
                        logs.sts(f"{logs.prog_loc()} Cannot find key '{gkey}' in {axis} in dataframe '{name}'", 3) 
                        raise 
                # if gkey in keydict:
                    # idxs.append(keydict[gkey])                
                # elif not silent_error:
                    # raise KeyError
         
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
                slice_obj = idxs
                this_range = daf_utils.slice_to_range(slice_obj, len(gkeys))
                idxs = [idx for idx in range(len(keydict)) if idx not in this_range]
            else:
                idxs = [idx for idx in range(len(keydict)) if idx not in idxs]
            
        return idxs
        

    def select_krows(self, 
            krows:          Union[slice, str, T_la, int, T_lota, Tuple[Any, Any], Iterable, None], 
            inverse:        bool=False,
            silent_error:   bool=False,
            ) -> 'Daf':
    
        irows = self.krows_to_irows( 
            krows = krows,
            inverse = inverse,
            silent_error = silent_error,
            )
        new_daf = self.select_irows(irows)
        
        return new_daf
    
    
    def select_kcols(self, 
            kcols:          Union[slice, str, T_la, int, Tuple[Any, Any], None], 
            inverse:        bool=False, 
            flip:           bool=False,
            silent_error:   bool=False,
            ) -> 'Daf':
    
        icols = self.kcols_to_icols( 
            kcols = kcols,
            inverse = inverse,
            silent_error = silent_error,
            )
        return self.select_icols(icols, flip=flip)
    
    
    def select_irows(self, irows: Union[slice, int, T_li, range, T_lor, Iterable, None], invert: bool=False) -> 'Daf':
        """ select rows from daf and return a new instance.
            This is an efficient opeation. The array in the new instance
            uses references to selected rows in the original array.
            
            irows: can be either a slice, int, range, or list of integers. These
                    refer to row indices that are inherent in the lol structure.
            
            returns a new daf instance cloned from the original.
        """
        row_sliced_lol = self.lol
        
        if isinstance(irows, int):
            if not invert:
                # simple single row selection
                row_sliced_lol = [self.lol[irows]]
            else:
                row_sliced_lol.pop(irows)
        
        elif irows and isinstance(irows, list):
        
            if daf_utils.is_list_of_type(irows, int):                
                if not invert:
                    row_sliced_lol = [self.lol[i] for i in irows]
                else:
                    if len(irows) > 10:
                        irows_iter = dict.fromkeys(irows)
                    else:
                        irows_iter = irows
                    row_sliced_lol = [self.lol[i] for i in range(len(self.lol)) if i not in irows_iter]
            
            elif daf_utils.is_list_of_type(irows, range):
                rows_lor = irows    # just a name change
                if not invert:
                    row_sliced_lol = [self.lol[i] for irange in rows_lor for i in irange]
                else:
                    row_sliced_lol = [self.lol[i] for i in range(len(self.lol)) if not any(i in r for r in rows_lor)]
                
        elif irows and isinstance(irows, (range, collections.abc.Iterable)):
            if not invert:
                row_sliced_lol = [self.lol[i] for i in irows]
            else:
                row_sliced_lol = [self.lol[i] for i in range(len(self.lol)) if i not in irows]
             
        elif isinstance(irows, slice):
            slice_spec = irows
            if not invert:
                row_sliced_lol = self.lol[slice_spec]
            else:
                row_sliced_lol = [self.lol[i] for i in range(len(self.lol)) if i not in slice_spec]
            
        new_daf = self.clone_empty(lol=row_sliced_lol)
        
        return new_daf
    
    
    def select_icols(self, icols: Union[slice, int, T_li, range, T_lor, None], flip: bool=False) -> 'Daf':
        """ select cols from daf and return a new instance.
            This is not an efficient operation and can normally be avoided except when:
                reading/writing data, then columns may need to be dropped.
                exporting a portion of the array to NumPy, for example.
                
            instead, use the cols parameter to select the columns included
                in operations like apply() and reduce()
            
            icols: can be either a slice, int, range, list of integers, or list of ranges. These
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
            # simple single col selection
            icol = icols
            if not flip:
                col_sliced_lol = [[row[icol]] for row in self.lol]
                if orig_cols:
                    sliced_cols = [orig_cols[icol]]
            else: # flip
                col_sliced_lol = [row[icol] for row in self.lol]
            
        elif isinstance(icols, slice):
            slice_spec = icols
            icols_range = range(slice_spec.start or 0, slice_spec.stop or self.num_cols(), slice_spec.step or 1)
            
            if not flip:
                try:
                    col_sliced_lol = [[row[icol] for icol in icols_range]
                                            for row in self.lol]
                except Exception:
                    breakpoint()    # perm. This should not ever occur.
                    pass
                    
                if orig_cols:
                    sliced_cols = [orig_cols[icol] for icol in icols_range]
            else: # flip
                col_sliced_lol = [[row[icol] for row in self.lol]
                                        for icol in icols_range]
                
        elif icols and (isinstance(icols, list) and daf_utils.is_list_of_type(icols, int) or
                        isinstance(icols, range)):
            # list of integers or range
            if not flip:
                try:
                    col_sliced_lol = [[row[icol] for icol in icols]
                                            for row in self.lol]
                except IndexError:
                    logs.sts("Columns specified don't exist, array may have uneven rows.", 3)
                    raise
                    
                if orig_cols:                        
                    sliced_cols = [orig_cols[icol] for icol in icols]
                
            else: # flip
                col_sliced_lol = [[row[icol] for row in self.lol]
                                        for icol in icols]
                                    
        # this part needs to be tested!                                    
        elif isinstance(icols, list) and icols and daf_utils.is_list_of_type(icols, range):
            # list of ranges:
            cols_lor = icols # name change only.
            if not flip:
                # Flatten ranges into individual column indices
                col_sliced_lol = [[row[icol] for irange in cols_lor for icol in irange]
                                        for row in self.lol]
                if orig_cols:
                    # Also handle orig_cols, slicing them similarly
                    sliced_cols = [orig_cols[icol] for irange in cols_lor for icol in irange]
            
            else:  # flip
                # Flip logic, slice columns and then rows, based on ranges
                col_sliced_lol = [[row[icol] for row in self.lol]
                                        for irange in cols_lor for icol in irange]
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
            if self.keyfield and isinstance(self.keyfield, str):
                new_keyfield = self.keyfield if self.keyfield in sliced_cols else ''
            else:
                new_keyfield = ''
            
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
    
    def select_record(self, key: Union[str, int, T_ta], silent_error: bool=True) -> T_da:
        """ Select one record from daf using the key and return as a single T_da dict.
        
            returns {} if not self
            assertion break if keyfield not defined.
            if key not found, return {} if silent_error, else raise KeyError exception.
                   
            TODO Update for returning KeyedList
            should return daf unless .to_dict() is used?
        """
        
        if not self:
            return {}
        
        if not self.keyfield and not self.kd:
            logs.sts(f"{logs.prog_loc()} LOGIC ERROR. No keyfield and no kd is defined, required for select_record():\n{self}")
            #breakpoint()    # temp
            raise RuntimeError ("No keyfield and no kd is defined, required for select_record()")
            
        if key in self.kd:
            return self._basic_get_record(self.kd[key])

        if silent_error:
            return {}
            
        raise KeyError    

        
    def _basic_get_record(self, irow: int, include_cols: Optional[T_ls]=None) -> T_da:
        """ return a record at irow as dict 
            include only "include_cols" if it is defined
            note, this requires that hd is defined.
            if no column header exists, this will generate a new one using spreadsheet convention.
            
            TODO Update to return KeyedList as an option.
        """
        if not self.hd:
            cols = daf_utils._generate_spreadsheet_column_names_list(num_cols=self.num_cols())
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
            # record = self._basic_get_record(row_idx)
        
            # selected_daf.append(record)
                      
        # return selected_daf
        
        
    def select_records_daf(self, keys_ls: Union[T_ls, Iterable], inverse:bool=False, silent_error: bool=False) -> 'Daf':
        """ Select multiple records from daf using the keys and return as a single daf.
            If inverse is true, select records that are not included in the keys.
            
        """
        return self.select_krows(krows=keys_ls, inverse=inverse, silent_error=silent_error)
          
        
    def irow_la(self, irow: int) -> T_la:
        """ return a row as a list.
        """
        return self.lol[irow]
        

    def to_value(self, 
        irow:       int=0, 
        icol:       int=0, 
        default:    Any='', 
        astype:     Optional[Union[Callable, str]]=None,
        ) -> Any:
        """ return a single value from an array,
            at default location 0,0 or as specified.
        """
    
        num_rows, num_cols = self.shape()

        if num_rows >= 1 and num_cols >= 1:
            return daf_utils.astype_value(self.lol[irow][icol], astype)

        return default
        

    def to_list(self, 
        irow:       Optional[int]=None,   # select a row 
        icol:       Optional[int]=None,   # or column.
        unique:     bool=False,           # reduce to unique values
        flatten:    bool=False,           # if items is the list are lists, combine them into one list.
        omit_nulls: bool=False,           # omit items that are empty strings (nulls).
        default:    Any = _MISSING,       # use this value instead if value is None or '' or NAN (default can be None)
        astype:     Optional[Union[Callable, str]]=None,
        ) -> list:
        """ return data from a daf array as a list
            defaults to the most obvious list if irow and icol not specified.
                from irow 0, if num_rows is 1 and num_cols >= 1
                from icol 0, if num_rows >= 1 and num_cols == 1
            otherwise, choose irow or icol specified.
                if irow, specified, ignore icol.
                if irow=None and icol specified, then use icol.
            Note:
                If neither irow nor icol is specified, and the table has more than one row and more than one column,
                then the result is an empty list. Use irow or icol explicitly to avoid ambiguity.
        """
    
        num_rows, num_cols = self.shape()

        if irow is None and icol is None:
            # no explicit irow or icol specified, this is the normal case
            
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
            
        if flatten:
            new_list = []
            for sublist in result_la:
                if isinstance(sublist, list):
                    new_list += sublist
                else:
                    new_list += [sublist]
            result_la = new_list        
            
        if unique and result_la:
            result_la = list(dict.fromkeys(result_la))
            
        if omit_nulls and '' in result_la:
            result_la = [val for val in result_la if val != '']
            
        if default is not _MISSING and ('' in result_la or None in result_la):
            filtered_result_la = []
            for val in result_la:
                if val is None or val == '' or val != val:  # val != val catches NaN
                    filtered_result_la.append(default)
                else:     
                    filtered_result_la.append(val)
            result_la = filtered_result_la
                    
        return daf_utils.astype_la(result_la, astype)


    def to_lota(self, 
        ) -> T_lota:
        """ return data from a daf array as a list of tuple of any (List[Tuple[Any...]])
            This is convenient for creating compound keys
        """
    
        lota = []
        
        for la in self.lol:
            lota.append(tuple(la))
            
        return lota    



    def to_dict(self, irow: int=0, include_cols: Optional[T_ls]=None) -> T_da:
        """ alias for iloc 
            Note that this does not convert a column to a dict. Use to_list to convert a column.
            test exists in test_daf.py
        """
        return self.iloc(irow, include_cols)
        

    def to_klist(self, irow: int=0) -> KeyedList:
        """ return a row as a klist
        """
        return self.iloc(irow, rtype='klist')
        

    def irow(self, irow: int=0, include_cols: Optional[T_ls]=None) -> T_da:
        """ alias for iloc 
            test exists in test_daf.py
        """
        return self.iloc(irow, include_cols)
        

    def iloc(self, irow: int=0, include_cols: Optional[T_ls]=None, rtype: str='dict') -> Union[T_da, KeyedList]:
        """ Select one record from daf using the idx and return as a single T_da dict
            test exists in test_daf.py
            
            rtype can be 'dict', 'klist', or 'list'
            
        """
        if irow < 0 or irow >= len(self.lol) or not self.lol or not self.lol[irow]:
            if rtype == 'dict':
                return {}
            else:
                return KeyedList()
            
        if rtype == 'klist':
            return KeyedList(self.hd, self.lol[irow])
        
        elif rtype == 'dict':
            if self.hd: 
                return self._basic_get_record(irow, include_cols)
                
            colnames = daf_utils._generate_spreadsheet_column_names_list(num_cols=len(self.lol[irow]))
            return dict(zip(colnames, self.lol[irow]))
            
        elif rtype == 'list':
            return self.to_list( 
                irow = irow, 
                icol = None,
                unique = False,
                flatten = False,
                )        
        

    # def select_by_dict_to_lod(self, selector_da: T_da, expectmax: int=-1, inverse: bool=False) -> T_loda:
        # """ Select rows in daf which match the fields specified in d, returning lod 
            # test exists in test_daf.py
            
            # DEPRECATE, use select_by_dict().to_lod()
        # """
        
        # result_lod = self.select_by_dict(selector_da=selector_da, expectmax=expectmax, inverse=inverse).to_lod()

        # return result_lod


    def select_by_dict(
            self, 
            selector_da:    T_da, 
            expectmax:      int=-1, 
            inverse:        bool=False, 
            keyfield:       Union[str, int, T_ta]='',
            ) -> 'Daf':
        """ Selects rows in daf which match the fields specified in d
            and return new daf, with keyfield set according to 'keyfield' argument.
            test exists in test_daf.py
        """

        # from utilities import daf_utils

        result_lol = [list(d2.values()) for d2 in self if inverse ^ daf_utils.is_d1_in_d2(d1=selector_da, d2=d2)]
    
        if expectmax != -1 and len(result_lol) > expectmax:
            raise LookupError
            # breakpoint() #perm
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
            if inverse ^ daf_utils.is_d1_in_d2(d1=selector_da, d2=d2):
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

        # Example Usage
            result_daf = original_daf.select_where("int(row['colname']) > 5")
            
        Could use .to_index() approach instead?    
        
        """
        # unit test exists.
        
        return [idx for idx, row in enumerate(self) if where(row)]
        

    def remove_dups(self, keyfield: Union[str, T_ta, T_la]='') -> Tuple['Daf', 'Daf']:  # unique_daf, duplicates_daf
        """
        If it is known that duplicates may exist in the array with respect to keyfield,
        remove records that have the same keyfield and return two daf arrays,
        uniques_daf, dups_daf. 
        
        Please note that this does NOT compare all components, only the keyfields.
        
        Returns unique_daf array with keyfield set and duplicates_daf, which may have repeats.
               
        """
        num_rows_orig = self.num_rows()
        
        self.set_keyfield(keyfield=keyfield)
        
        num_rows_after_set_keyfield = self.num_rows()
        
        if num_rows_orig == num_rows_after_set_keyfield:
            return self, Daf()
        
        # key irows of records that are unique
        unique_irows = self.kd.values()
        
        # unset the keyfield
        self.set_keyfield(keyfield='')
        
        # remove duplicated rows and set the keyfield to form unique_daf
        unique_daf = self.select_irows(irows=unique_irows, invert=False)
        unique_daf.set_keyfield(keyfield=keyfield)
        
        # all the other records are duplicates.
        dups_daf   = self.select_irows(irows=unique_irows, invert=True)

        return unique_daf, dups_daf





    def split_where(self, where: Callable) -> Tuple['Daf', 'Daf']:
        """
        Select rows in Daf based on the provided where condition,
        and split into two Daf objects, for True and False.
        
        Note: this does a shallow copy of the lines and any changes to
        the split objects will change the original daf object.

        # Example Usage
        
            true_daf, false_daf = original_daf.split_where(lambda row: bool(int(row['colname']) > 5))
        
        """
        true_lol = []
        false_lol = []
        
        for klist in self.iter_klist():
            if where(klist):
                true_lol.append(klist._values)
            else:
                false_lol.append(klist._values)

        true_daf = Daf(cols=self.columns(), lol=true_lol, keyfield=self.keyfield, dtypes=self.dtypes)
        false_daf = Daf(cols=self.columns(), lol=false_lol, keyfield=self.keyfield, dtypes=self.dtypes)

        return true_daf, false_daf


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
            
            Note, this is a inefficient process that should be avoided. specify columns 
                    to be included in operation instead.
            
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
        dtypes   = {col: typ for col, typ in self.dtypes.items() if col in new_cols}
        
        new_keyfield = self.keyfield \
            if self.keyfield and isinstance(self.keyfield, str) and self.keyfield in new_cols else ''

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
        
    def assign_record(self, record: T_da):
        """ Assign one record in daf using the key using a single T_da dict.
        
            TODO Upate to accept KeyedList
        
            unit tests exist
        """
        
        if not self.keyfield:
            raise RuntimeError("No keyfield estabished for daf.")
            
        keyfield = self.keyfield
        if isinstance(keyfield, str):
            if keyfield not in record:
                raise RuntimeError(f"No keyfield '{keyfield}' in dict.")
        elif isinstance(keyfield, T_ta):
            if not all(keytup in record for keytup in keyfield):
                raise RuntimeError(f"Not all keyfields '{keyfield}' in dict.")
            
        if self and list(record.keys()) != list(self.hd.keys()):
            raise RuntimeError("record fields not equal to daf columns")
        
        keyval = self._get_keyval(record)

        # for the following, see https://github.com/raylutz/daffodil/issues/7
        if keyval in self.kd:
            # assign the record, normalize the fields.
            self.lol[self.kd[keyval]] = [record.get(col, '') for col in self.hd]
        else:
            # otherwise add it.
            self.append(record)
            
        # row_idx = self.kd.get(keyval, -1)
        # if row_idx < 0 or row_idx >= len(self.lol):
            # self.append(record_da)
        # else:
            # #normal_record_da = Daf.normalize_record_da(record_da, cols=self.columns(), dtypes=self.dtypes)   
            # self.lol[row_idx] = [record_da.get(col, '') for col in self.hd]
        

    def assign_record_irow(self, irow: int=-1, record: Optional[T_da]=None):
        """ Assign one record in daf using the iloc using a single T_da dict.
        
            TODO Update to accept KeyedList
        
            unit tests exist
        """
        
        if record is None:
            return
        
        if irow < 0 or irow >= len(self.lol):
            self.append(record)
        else:
            #normal_record_da = Daf.normalize_record_da(record_da, cols=self.columns(), dtypes=self.dtypes)   
            self.lol[irow] = [record.get(col, '') for col in self.hd]
        

    #@deprecated("Use 'my_daf[keylist] = record' syntax")
    def update_by_keylist(self, keylist: Optional[T_ls]=None, record: Optional[T_da]=None):
        """ Update selected records in daf by keylist using record
            only update those columns that have dict keys
            but keep all other dict items intact in that row if not updated.
            
            Equivalent to: my_daf[keylist] = record
            
            DEPRECATE?
        """
        
        if record is None or not self.lol or not self.hd or not self.keyfield or not keylist:
            return

        for key in keylist:
            if key in self.kd:
                self.update_record_irow(self.kd[key], record)
            
        

    def update_record_irow(self, irow: int=-1, record: Optional[T_da]=None):
        """ Update one record in daf at iloc using a single T_da dict,
            and only update those columns that have dict keys
            but keep all other dict items intact in that row.
            
            Equivalent to my_daf[irow] = record
            
            DEPRECATE?
            
            TODO: Allow record to be KeyedList
            
            unit tests exist
        """
        
        if record is None or not self.lol or not self.hd:
            return
        
        if irow < 0 or irow >= len(self.lol):
            return
        
        # see https://github.com/raylutz/daffodil/issues/7
        for colname, val in record.items():
            if colname in self.hd:
                self.lol[irow][self.hd[colname]] = record[colname]
            # icol = self.hd.get(colname, -1)
            # if icol >= 0:
                # self.lol[irow][icol] = record_da[colname]
        

    def assign_icol(
            self, 
            icol: int=-1, 
            col_la: Optional[T_la]=None, 
            default: Any=''
            ):
        """ modify icol by index using col_la 
            use default if col_la not long enough to fill all cells.
            Also, if col_la not provided, use default to fill all cells in the column.
            if icol == -1, append column on the right side.
            
            Equivalent to  my_daf[:, icol] = col_la
            
            Except that icol can be -1 and it will append.
            DEPRECATE?
            
        """
        # from utilities import daf_utils

        self.lol = daf_utils.assign_col_in_lol_at_icol(icol, col_la, lol=self.lol, default=default)
        
        
        
    def insert_icol(
            self, 
            icol:       int=-1, 
            col_la:     Optional[T_la]=None, 
            colname:    str='', 
            default:    Any=''
            ):
        """ insert column col_la at icol, shifting other column data. 
            use default if la not long enough
            If icol==-1, insert column at right end.
            Note: use set_keyfield() if this column is to be the keyfield.
            unit tests
        """
        
        # from utilities import daf_utils

        self.lol = daf_utils.insert_col_in_lol_at_icol(icol, col_la, lol=self.lol, default=default)
        
        if colname:
            if not self.hd:
                self.hd = {}
            if icol < 0 or icol >= len(self.hd):
                icol = len(self.hd)
            hl = list(self.hd.keys())
            hl.insert(icol, colname)
            self.hd = {k: idx for idx, k in enumerate(hl)}
            
        return self

        
    def insert_irow(self, irow: int=-1, row: Optional[Union[T_la, T_da]]=None, default: Any=''):
        """ insert row row_la at irow, shifting other rows down. 
            use default if la not long enough
            If irow > len(daf), insert row at the end.
            
        """
        
        # from utilities import daf_utils
        
        if isinstance(row, list):
        
            row_la = row

        elif isinstance(row, dict):
        
            row_da = row
            # create normalize list
            row_la = [row_da.get(col, '') for col in self.hd] 
        
        self.lol = daf_utils.insert_row_in_lol_at_irow(irow=irow, row_la=row_la, lol=self.lol, default=default)
        
        self._rebuild_kd()


    def assign_col(self, colname: str, la: Optional[T_la]=None, default: Any=''):
        """ modify col by colname using la 
            use default if la not long enough.
            test exists in test_daf.py
            
            Equivalent to my_daf[:, colname] = la
            
            if colname not exists, then add column of that name and initialize.
        """
        
        if colname in self.hd:
            self.assign_icol(self.hd[colname], la, default)
            
        else:
            self.insert_col(
                colname = colname, 
                col_la = la, 
                #icol: int=-1, 
                default = default,
                )
        
        return self

    # def set_col(self, colname: str, val: Any):
        # """ modify col by colname using val """
        
        # if not colname or colname not in self.hd:
            # return
        # icol = self.hd[colname]
        # self.set_icol(icol, val)
        

    def insert_col(
            self, 
            colname:    str,                    # name of the col
            col_la:     Optional[T_la]=None,    # column to insert
            icol:       int=-1,                 # insert at end by default
            default:    Any='',
            ):
            
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

        # see https://github.com/raylutz/daffodil/issues/7
        if colname in self.hd:
            # column already exists. ignore icol, overwrite data.
            self.assign_col(colname, col_la, default)
                   
        # colname_icol = self.hd.get(colname, -1)
        # if colname_icol >= 0:
            # # column already exists. ignore icol, overwrite data.
            # self.assign_col(colname, col_la, default)
            # return
        else:
            self.insert_icol(icol=icol, col_la=col_la, colname=colname, default=default) #, keyfield=keyfield)
        
        # hl = list(self.hd.keys())
        # hl.insert(icol, colname)
        # self.hd = {k: idx for idx, k in enumerate(hl)}
        
        return self
        
    
    def insert_idx_col(self, colname='idx', icol:int=0, startat:int=0) -> 'Daf':
        """ insert an index column at column icol with name colname with indexes starting at 'startat' 
            unit tested
        """
        
        num_rows = len(self)
        col_la = list(range(startat, startat + num_rows))
    
        self.insert_col(colname, col_la, icol)
        
        return self


    def set_col_irows(self, colname: str, irows: T_li, val: Any):
        """ set a given icol and list of irows to val 
        
            Equivalent to my_daf[irows, colname] = val
            
            DEPRECATE?
        
        """
        
        # see https://github.com/raylutz/daffodil/issues/7
        try:
            icol = self.hd[colname]
        except KeyError:
            return self

        self.set_icol_irows(icol, irows, val)
        
        return self
    

    def set_icol(self, icol: int, val: Any):
    
        """ Equivalent to my_daf[:, icol] = val
        
            DEPRECATE?
        """
    
    
        for irow in range(len(self.lol)):
            self.lol[irow][icol] = val
            
        return self
        

    def set_icol_irows(self, icol: int, irows: T_li, val: Any):
        """ set a given icol and list of irows to val 
        
            Equivalent to my_daf[irows, icol] = val
        
            DEPRECATE?
        
        """
        
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
        
    
    def replace_in_columns(
        self,
        cols: Optional[T_lsi],
        find_values: Optional[List[Any]] = None,
        replacement: Any = _MISSING
    ) -> 'Daf':
        """
        Replace all values in `find_values` with `replacement` for the specified columns.

        Parameters:
            cols:
                A list of column names (str) or indices (int).
                If None, all columns will be included.
            find_values:
                A list of values to search for (e.g., ['', None]).
            replacement:
                The value to substitute in place of any matched item.

        Behavior:
            - Column identifiers may be strings (header names) or integers (column indices).
            - If headers exist, string names are looked up via self.hd.
            - If cols is None, all columns [0..num_cols-1] will be modified.
            - Modifies self in place and returns self.

        Raises:
            KeyError if a string column name is not found in the header.
            TypeError if a column identifier is not str or int.
        """
        
        if find_values is None:
            return self
        
        if cols is None:
            # process the entire array
            target_indices = list(range(self.num_cols()))
        else:
            target_indices = []
            for col in cols:
                if isinstance(col, int):
                    target_indices.append(col)
                elif isinstance(col, str):
                    if col in self.hd:
                        target_indices.append(self.hd[col])
                    else:
                        raise KeyError(f"Column name '{col}' not found in header.")
                else:
                    raise TypeError(f"Column specifier must be str or int, got {type(col).__name__}")

        for row in self.lol:
            for col_idx in target_indices:
                if row[col_idx] in find_values:
                    row[col_idx] = replacement

        return self
        
        
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
        # from utilities import daf_utils
        
        chunk_sizes_list = daf_utils.calc_chunk_sizes(num_items=len(self), max_chunk_size=max_chunk_size)
        chunk_ranges = daf_utils.convert_sizes_to_idx_ranges(chunk_sizes_list)
        chunks_lodaf = self.split_daf_into_ranges(chunk_ranges)
        return chunks_lodaf

           
    #=========================
    #   sort
            
    def sort_by_colname(self, colname:str, reverse: bool=False, length_priority: bool=False) -> 'Daf':
        """ sort the data by a given colname, using length priority if specified.
            sorts in place. Make a copy if you need the original order.
            Note, this will place an empty field before fields with contents, which differs from spreadsheet operation.
            
            Length priority will allow proper sorting when values include embedded integers which are not left-padded.
            
                ie.             ['10', '99', '8', '100', '0'] 
                will sort to    ['0', '8', '10', '99', '100']
                rather than     ['0', '10', '100', '8', '99']
            
        """
        if not self or len(self) <= 1:
            return self
        
        try:
            colidx = self.hd[colname]
        except Exception as err:
            print(f"{err}")
            breakpoint() #perm # assertion break
            pass
            
        self.lol = daf_utils.sort_lol_by_col(self.lol, colidx, reverse=reverse, length_priority=length_priority)
        self._rebuild_kd()
        return self
        
        
    def sort_by_colnames(self, colnames:T_ls, reverse: bool=False, length_priority: bool=False) -> 'Daf':
        """ sort the data by multiple colnames, using length priority if specified.
            sorts in place. Make a copy if you need the original order.
            
            Length priority will allow proper sorting when values include embedded integers which are not left-padded.
            
                ie.             ['10', '99', '8', '100', '0'] 
                will sort to    ['0', '8', '10', '99', '100']
                rather than     ['0', '10', '100', '8', '99']
                
            Note: consider deprecating. Can accomplish with some loss of efficiency as:
                sort_by_colname('colname1').sort_by_colname('colname2')
            
        """
        if not self or len(self) <= 1:
            return self
        
        try:
            colidxs = [self.hd[colname] for colname in colnames]
        except Exception as err:
            print(f"{err}")
            breakpoint() #perm # assertion break
            pass
            
        self.lol = daf_utils.sort_lol_by_cols(self.lol, colidxs, reverse=reverse, length_priority=length_priority)
        self._rebuild_kd()
        return self
        
        
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
            breakpoint() #perm assertion error
            
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
                        breakpoint() #perm assertion error
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
        
        
    def insert_dif_row(self, 
            irow1: int, 
            irow2: Optional[int]=None, 
            irow_insert: Optional[int]=None, 
            cols: Optional[T_ls]=None,
            ) -> 'Daf':
        
        if irow2 is None:
            irow2 = irow1 + 1
        if irow_insert is None:
            irow_insert = irow2
        if cols is None:
            cols = self.columns()
            
        # calculate the difference.    
        diff_result_da = type(self).diff_da(self[irow1].to_dict(), self[irow2].to_dict(), keys=cols)
        
        # this function handles normalizing the columns before insertion
        self.insert_irow(irow=irow_insert,  row=diff_result_da)
        
        return self
        
        
    def insert_dif_rows(self, 
            irows_rli: Optional[T_rli]=None, 
            cols: Optional[T_ls]=None, 
            offset: int=0
            ) -> 'Daf':
    
        """ given a list of row indexes irows_li, insert a difference row of that row
            and the next row, for the columns specified by name, either
            between them (offset=0) or after them (offset=1)
            Note: do not include the final row in the list which is compared with the prior list.
        """
        
        if irows_rli is None:
            reversed_irows_rli = range(len(self) - 2, -1, -1)
        else:
            reversed_irows_rli = sorted(irows_rli, reverse=True)
        
        for irow in reversed_irows_rli:
        
            self.insert_dif_row(irow1=irow, irow_insert=irow + 1 + offset, cols=cols)
                
        return self 
        
    #===============================
    # annotate and join
    
    def annotate_daf(self, other_daf: 'Daf', my_to_other_dict: T_ds) -> 'Daf':
        """
            Adopt fields from other_daf for fields in self using my_to_other_dict map.
        """
    
        my_keyfield = self.keyfield
        if not my_keyfield:
            raise KeyError("annotate_daf: self must have keyfield defined")
            
        other_keyfield = other_daf.keyfield
        if not other_keyfield:
            raise KeyError("annotate_daf: other daf array must have keyfield defined")

        for my_rec_klist in self.iter_klist():
        
            rowkey = my_rec_klist[my_keyfield]

            other_rec = other_daf.select_record(rowkey)
            
            for my_field, other_field in my_to_other_dict.items():
                my_rec_klist[my_field] = other_rec[other_field]
            
        return self


    #===============================
    # apply and reduce

    def apply(
            self, 
            func:       Callable[[Union[T_da, T_Daf], Optional[T_la]], Union[T_da, T_Daf]], 
            by:         str='row', 
            keylist:    Optional[Union[T_la, T_lota]]=None,     # list of keys of rows to include (DEPRECATE?)
            **kwargs:   Any,
                # kwargs may commonly include:
                # cols: Optional[T_la]=None,                    # columns included in the apply operation.
            ) -> "Daf":
        """
        Apply a function to each 'row', 'col', or 'table' in the Daf and create a new Daf with the transformed data.
        If the function returns an empty record or None, then do not append.
        
        Note: to apply a function to a portion of the table, first select the columns or rows desired 
                using a selection process.
            Keylist: probably will be deprecated. Select rows prior to invocation.

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
       
        result_daf = Daf()

        if by == 'row':
            if keylist is None:
                keylist = []
        
            keylist_or_dict = keylist if not keylist or len(keylist) < 30 else dict.fromkeys(keylist)
            for row in self:
                if self.keyfield and keylist_or_dict and self.keyfield not in keylist_or_dict:
                    continue
                transformed_row = func(row, **kwargs)
                result_daf.append(transformed_row)          # Will not append an empty row.
                
        elif by == 'col':
            # this is not working yet, don't know how to handle cols, for example.
            raise NotImplementedError
        
            num_cols = self.num_cols()
            for icol in range(num_cols):
                col_la = self.icol(icol)
                transformed_col = func(col_la, **kwargs)
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
            func:       Callable[[Union[T_da, KeyedList]], Union[T_da, None]], 
            by:         str='row', 
            rowkeys:    Optional[Union[T_la, T_lota]]=None,  # list of rowkeys to include.
                        # the above changed from keylist to avoid confusion with KeyedList
            **kwargs:   Any,
            ):
        """
        Apply a function to each 'row', 'row_klist', 'col', or 'table' in the daf.

        Args:
            func (Callable): The function to apply to each 'row' 
                                It should take a row dictionary (or KeyedList) and any additional parameters.
            by (str):       either 'row', 'row_klist', 'col' or 'table'
                                if by == 'row_klist' then the function should modify the KeyedList of that row
                                                        which will mutate the row in the table.
                                if by == 'table', function should create a new Daf instance.
            rowkeys:        list of keys of rows to include (keyfield must be defined).
            **kwargs:       Additional keyword parameters to pass to the function.

        Modifies self in-place.
        """
        if rowkeys is None:
            rowkeys = []
        # here we create a dict if the rowkeys to search are numerous.
        rowkeys_list_or_dict = rowkeys if (not rowkeys or 
                                        isinstance(rowkeys, dict) 
                                        or len(rowkeys) < 30
                                    ) else dict.fromkeys(rowkeys)
        
        if by == 'row':

            for idx, row_da in enumerate(self):
                if rowkeys_list_or_dict and self.keyfield and row_da[self.keyfield] not in rowkeys_list_or_dict:
                    continue
                transformed_row_da = func(row_da, **kwargs)
                self.lol[idx] = list(transformed_row_da.values())

        elif by == 'row_klist':

            for idx, row_klist in enumerate(self.iter_klist()):
                if rowkeys_list_or_dict and self.keyfield and row_klist[self.keyfield] not in rowkeys_list_or_dict:
                    continue
                    
                # func should return nothing and instead mutate row_klist, which will mutate the row in the array.    
                func(row_klist, **kwargs)
                
        else:
            breakpoint()    # perm
            pass
            raise NotImplementedError
            
        # Rebuild the internal data structure (if needed)
        # note, this should not be necessary. apply_in_place should not modify the keyfield column.
        self._rebuild_kd()
        
        
    # def reduce(
            # self, 
            # func: Callable[[T_da, T_da], Union[T_da, T_la]], 
            # by: str='row', 
            # cols: Optional[T_la]=None,                      # columns included in the reduce operation.
            # **kwargs: Any,
            # ) -> Union[T_da, T_la]:
        # """
        # Apply a function to each 'row', 'col', or 'table' and accumulate to a single T_da
        # Note: to apply a function to a portion of the table, first select the columns or rows desired 
                # using a selection process.

        # Args:
            # func (Callable): The function to apply to each 'row', 'col', or 'table'. 
            # It should take a row dictionary and any additional parameters.
            # by (str): either 'row', 'col' or 'table'
                # if by == 'table', function should create a new Daf instance.
            # **kwargs: Additional parameters to pass to the function.

        # Returns:
            # either a dict (by='rows' or 'table') or list (by='cols')
        # """
        # if by == 'table':
            # reduction_da = func(self, cols, **kwargs)
            # return reduction_da    

        # if by == 'row':
            # reduction_da = {}
            # for row_da in self:
                # reduction_da = func(row_da, reduction_da, cols, **kwargs)
            # return reduction_da    
                
        # elif by == 'col':
            # reduction_la = []
            # num_cols = self.num_cols()
            # for icol in range(num_cols):
                # col_la = self.icol(icol)
                # reduction_la = func(col_la, reduction_la, **kwargs)
            # return reduction_la

        # else:
            # raise NotImplementedError
        # return [] # for mypy only.
        
        
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
            this_daf.record_append(da)
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
        
        for kla in self.iter_klist():
            fieldval_tuple = tuple(kla[colname] for colname in colnames)
            if fieldval_tuple not in result_dodaf:
                result_dodaf[fieldval_tuple] = this_daf = self.clone_empty()
            else:
                this_daf = result_dodaf[fieldval_tuple]
                
            this_daf.record_append(kla)
    
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
        # breakpoint() #temp
        
        if diagnose:  # pragma: no cover
            daf_utils.sts(f"Starting groupby_cols() of {len(self):,} records.", 3)
            
        grouped_tdodaf = self.groupby_cols(groupby_colnames)
        
        if diagnose:  # pragma: no cover
            daf_utils.sts(f"Total of {len(grouped_tdodaf):,} groups. Reduction starting.", 3)
        
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
            daf_utils.sts(f"Reduction completed: {len(result_daf):,} records.", 3)

        return result_daf
    

    def groupby_reduce(
            self, 
            colname:        str, 
            func:           Callable[['Daf', T_da], Union[T_da, T_la]], # function reduces one grouped daf to one record. 
            by:             str='row',                                  # determines how the func is applied.
            reduce_cols:    Optional[T_la]=None,                        # columns included in the reduce operation.
            diagnose:       bool=False,
            **kwargs:       Any,
            ) -> 'Daf':
        """ given a daf, break into a number of daf's based on colname specified. 
            For each group, apply callable.
            returns daf with one row per group, with keyfield the groupby value in colname.
            
        """
        
        if diagnose:
            logs.sts(f"{logs.prog_loc()} starting groupby '{colname}' operation", 3)
        grouped_dodaf = self.groupby(colname)
        result_daf = Daf.reduce_dodaf_to_daf(
            func            = func,             # function reduces one grouped daf to one record. 
            colname         = colname, 
            grouped_dodaf   = grouped_dodaf, 
            reduce_cols     = reduce_cols,
            by              = by,
            diagnose        = diagnose,
            **kwargs,
            )
        return result_daf
        
    
    def multi_groupby(
            self, 
            groupby_colnames: T_ls, 
            colnames: Optional[T_ls]=None,
            omit_nulls: bool=False,         # do not group to values in column that are null ('')
            ) -> Dict[str, Dict[str, 'Daf']]:   # result_dododaf
            
        """ given a daf, break into a number of daf's based on a list of colnames specified. 
            For each groupby_colname, for each discrete value, create a daf table with all cols,
            including colname, and provide a dodaf structure for each groupby_colname.
            
            Please note that each daf array which is built the value of each colname is not reduced.
            
            {groupby_colname1:
                {value1: Daf,
                 value2: Daf,
                 ...
                 valuen: Daf,
                }
             groupby_colname2:
                {value1: Daf,
                 value2: Daf,
                 ...
                 valuen: Daf,
                }
            ...
            }
            
        """
        if isinstance(groupby_colnames, str):
            groupby_colnames = [groupby_colnames]
        
        result_dododaf: Dict[str, Dict[str, 'Daf']] = {}
        
        # the following makes a single pass through Daf array, and
        # allocates the record to one daf for each colname, for each colvalue.
        # these daf arrays are not reduced here.
        
        # (TODO this loop would be better to iterate through list items rather than dicts.)
        
        for da in self:
            for col in groupby_colnames:
                if col not in result_dododaf:
                    result_dododaf[col] = {}
            
                fieldval = da[col]
                if omit_nulls and fieldval=='':
                    continue
            
                if fieldval not in result_dododaf[col]:
                    result_dododaf[col][fieldval] = self.clone_empty()
                
                this_daf = result_dododaf[col][fieldval]
                this_daf.record_append(da)
                result_dododaf[col][fieldval] = this_daf
    
        return result_dododaf
        
    

    @staticmethod
    def reduce_dodaf_to_daf(
            colname:        str, 
            func:           Callable[[T_da, T_da], Union[T_da, T_la]], 
            grouped_dodaf:  T_dodaf, 
            reduce_cols:    Optional[T_la]=None,    # columns included in the reduce operation, None = all except for colname.
            diagnose:       bool=False,
            **kwargs:       Any,
            ) -> 'Daf':
        """ after splitting a daf into a number of mutually exclusive daf objects based on the value in colname,
            combine these using reduction function func, resulting in a single record for each Daf array.
            Then append these records into the resulting Daf array, containing one record per value in column 'colname'
            Other than columns specified in 'reduce_cols', other columns are '' unless they are single-valued.
        """
        
        if diagnose:
            logs.sts(f"{logs.prog_loc()} Grouped into {len(grouped_dodaf)} groups.", 3)
            
            # display the first three daf arrays and the last
            
            for group_num, (colval, this_daf) in enumerate(grouped_dodaf.items()):
                if group_num < 3 or group_num >= len(grouped_dodaf) - 1:
                    logs.sts(f"## Group {group_num}: {colname}={colval}\n\n{this_daf}\n\n", 3)
            
        
        result_daf = Daf(keyfield = colname)
        
        for group_num, (colval, this_daf) in enumerate(grouped_dodaf.items()):
        
            # maybe remove colname from cols here
        
            if diagnose:
                logs.sts(f"## Group {group_num}: {colname}={colval}\n\n{this_daf}\n\n", 3)
            reduction_da = this_daf.reduce(func, cols=reduce_cols, **kwargs)
                    # def reduce(
                            # self, 
                            # func: Callable[[T_da, T_da], Union[T_da, T_la]], 
                            # by: str='row', 
                            # cols: Optional[Iterable]=None,                  # columns included in the reduce operation.
                            # initial_da: Optional[T_da]=None,
                            # **kwargs: Any,
                            # ) -> Union[T_da, T_la]:
            
            if diagnose:
                logs.sts(f"Post reduction: {Daf.from_lod([reduction_da])=}", 3)
            
            # add colname:colval to the dict, as it is removed by the reduction func.
            reduction_da[colname] = colval
            
            if diagnose:
                logs.sts(f"Post add colname:colval to the dict: {Daf.from_lod([reduction_da])=}", 3)

            # this will also maintain the kd.
            result_daf.append(reduction_da)

        if diagnose:
            logs.sts(f"{logs.prog_loc()} result_daf:\n\n{result_daf}", 3)

        return result_daf

        
    def multi_groupby_reduce(
            self, 
            colnames:       T_ls, 
            func:           Callable[[T_da, T_da], Union[T_da, T_la]],  # function reduces one grouped daf to one record.
            by:             str='row',                                  # determines how the func is applied.
            reduce_cols:    Optional[T_la]=None,                        # columns included in the reduce operation.
            diagnose:       bool=False,
            **kwargs:       Any,
            ) -> Dict[str, 'Daf']:
        """ given a daf, break into a number of daf's based on colnames specified. 
            For each group, apply callable.
            returns dodaf with one daf per colname.
            In each of those dafs, there is row per value in the colname, with keyfield the groupby value in colname.
        """
        
        if diagnose:
            logs.sts(f"{logs.prog_loc()} starting multi-groupby '{colnames}' operation", 3)
        multi_grouped_dododaf = self.multi_groupby(colnames)
        
        result_dodaf: T_dodaf = {}
        
        for colname, grouped_dodaf in multi_grouped_dododaf.items():
        
            result_dodaf[colname] = Daf.reduce_dodaf_to_daf(
                func            = func,         # function reduces one grouped daf to one record.
                colname         = colname, 
                grouped_dodaf   = grouped_dodaf, 
                reduce_cols     = reduce_cols,
                diagnose        = diagnose,
                **kwargs,
                )
        if diagnose:
            logs.sts(f"{logs.prog_loc()} Grouped into {len(result_dodaf)} groups.", 3)
            
            for group_num, (colval, this_daf) in enumerate(result_dodaf.items()):
                if group_num < 3 or group_num >= len(grouped_dodaf) - 1:
                    logs.sts(f"## Group {group_num}: {colname}={colval}\n\n{this_daf}\n\n", 3)

        return result_dodaf
            

    def apply_colwise(
        self,
        target_col: Union[str, int],
        func: Callable[[T_da], Any],
        *,
        default: Any = 0,
    ) -> "Daf":
        """
        Apply a function to each row and store the result in target_col.

        - If target_col doesn't exist, it is inserted with default values.
        - If func raises an exception, default is used instead.
        - `func` receives a row dict (T_da) and can access any number of fields.

        Returns the modified Daf object (in-place).
        
        # Example:
        # Compute RV_partisanship = REP / (REP + DEM), and default to 0 on error

            my_daf.apply_colwise(
                target_col="RV_partisanship",
                func=lambda row: row["REP"] / (row["REP"] + row["DEM"]),
                default=0.0
            )

        # Behavior:
        # - If 'RV_partisanship' doesn't exist, it's inserted and initialized to 0.0
        # - If REP or DEM are missing or invalid in a row, that row's value stays 0.0
        # - Otherwise, the row gets REP / (REP + DEM)        
        """

        if target_col not in self.columns():
            self.insert_col(target_col, default)

        def row_op(row: T_da) -> T_da:
            try:
                row[target_col] = func(row)
            except Exception:
                row[target_col] = default
                
            return row

        return self.apply_in_place(row_op)

        
    #===================================
    # apply / reduce convenience methods

    def daf_sum2(
            self, 
            by: str = 'row', 
            cols: Optional[T_la]=None,
            **kwargs: Any,
            ) -> T_da:
        # this one to investigate why daf_sum is so slow!
            
        return self.reduce(func=Daf.sum_da2, by=by, cols=cols, **kwargs)
    

    def daf_sum3(
            self, 
            by: str = 'row', 
            cols: Optional[T_la]=None,
            **kwargs: Any,
            ) -> T_da:
        # this one to investigate why daf_sum is so slow!
            
        return self.reduce(func=Daf.sum_da3, by=by, cols=cols, **kwargs)
    

    # def reduce2(
            # self, 
            # func: Callable[[T_da, T_da], Union[T_da, T_la]], 
            # by: str='row', 
            # cols: Optional[T_la]=None,                      # columns included in the reduce operation.
            # **kwargs: Any,
            # ) -> Union[T_da, T_la]:
        # """
        # Apply a function to each 'row', 'col', or 'table' and accumulate to a single T_da
        # Note: to apply a function to a portion of the table, first select the columns or rows desired 
                # using a selection process.

        # Args:
            # func (Callable): The function to apply to each 'row', 'col', or 'table'. 
            # It should take a row dictionary and any additional parameters.
            # by (str): either 'row', 'col' or 'table'
                # if by == 'table', function should create a new Daf instance.
            # **kwargs: Additional parameters to pass to the function.

        # Returns:
            # either a dict (by='rows' or 'table') or list (by='cols')
        # """
        # if by == 'table':
            # reduction_da = func(self, cols, **kwargs)
            # return reduction_da    

        # if by == 'row':

            # if not self:
                # return {}
                
            # allcols = self.hd.keys()

            # if cols is None:
                # cols = allcols
                
            # reduction_da = dict.fromkeys(cols, 0)
    
            # for row in self:
                # for col in cols:
                    # reduction_da[col] += row[col]
            # return reduction_da    
 

            # # for row_da in self:
                # # reduction_da = func(row_da, reduction_da, cols, **kwargs)
            # # return reduction_da    
                
        # elif by == 'col':
            # reduction_la = []
            # num_cols = self.num_cols()
            # for icol in range(num_cols):
                # col_la = self.icol(icol)
                # reduction_la = func(col_la, reduction_la, **kwargs)
            # return reduction_la

        # else:
            # raise NotImplementedError
        # return [] # for mypy only.
        
        
    def daf_sum(
            self, 
            by:             str = 'row',                    # row|col|table|sparse_row
            cols:           Optional[Iterable] = None,
            indirect_col:   str = '',                       # indirect_col is required for lol array.           
            **kwargs:       Any,
            ) -> T_da:
            
        return self.reduce(func=Daf.sum_da, by=by, cols=cols, indirect_col=indirect_col, **kwargs)
    

    def reduce(
            self, 
            func:           Callable[[T_da, T_da], Union[T_da, T_la]], 
            by:             str = 'row',                                # row|col|table|sparse_row
            cols:           Optional[Iterable]=None,                    # columns included in the reduce operation.
            initial_da:     Optional[T_da]=None,
            indirect_col:   str = '',                                   # indirect_col is required for lol array.
            **kwargs:       Any,
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

            if not self:
                return {}
                
            #allcols = self.hd.keys()

            if cols is None:
                cols_iter = self.hd.keys()
            elif isinstance(cols, str):
                cols_iter = [cols]
            # elif isinstance(cols, list) and len(cols) > 10:    
                # cols_iter = dict.fromkeys(cols)
            else:
                cols_iter = {col: None for col in self.hd.keys() if col in cols}
                           
            if initial_da:
                reduction_da = initial_da
            else:
                reduction_da = dict.fromkeys(cols_iter, 0)
    
            for row_da in self:
                try:
                    reduction_da = func(row_da, reduction_da, cols=cols_iter, **kwargs)
                except Exception as err:
                    print(f"err = {err}")
                    breakpoint() #perm: investigate why reduction function not working
                    pass
                    
                # def count_values_da(row_da: T_da, reduction_da: T_da, cols: Iterable, omit_nulls: bool=False) -> T_dodi:
                # def sum_da         (row_da: T_da, reduction_da: T_da, cols: Iterable, astype: Optional[Type]=None, diagnose:bool=False

            # normalize the result so it contains all columns
            result_da = {key: reduction_da.get(key,'') for key in self.hd.keys()}
                
            return result_da    
                
        elif by == 'col':
            reduction_la = []
            num_cols = self.num_cols()
            for icol in range(num_cols):
                col_la = self.icol(icol)
                reduction_la = func(col_la, reduction_la, **kwargs)
            return reduction_la

        elif by == 'sparse_row':
            if not indirect_col:
                raise ValueError("indirect_col must be specified for 'sparse_row' reduction and lol array.")

            reduction_da = initial_da or {}
            
            for row_da in self:
                indirect_val = row_da[indirect_col]
                if isinstance(indirect_val, str):
                    indirect_da = daf_utils.safe_convert_json_to_obj(indirect_val)
                else:
                    indirect_da = indirect_val
                try:
                    
                    reduction_da = func(indirect_da, reduction_da, cols=cols, is_sparse=True, **kwargs)
                    
                except Exception as err:
                    print(f"err = {err}")
                    breakpoint() #perm: investigate why reduction function not working
                    pass
                    
                # def count_values_da(row_da: T_da, reduction_da: T_da, cols: Iterable, omit_nulls: bool=False) -> T_dodi:
                # def sum_da         (row_da: T_da, reduction_da: T_da, cols: Iterable, astype: Optional[Type]=None, diagnose:bool=False

            # # normalize the result so it contains all columns
            # result_da = {key: reduction_da.get(key,'') for key in self.hd.keys()}
                
            return reduction_da    
                


        else:
            raise NotImplementedError
        return [] # for mypy only.
        
        
    @staticmethod
    def sum_da( row_da:         T_da,                       # the current row from the daf array.
                reduction_da:   T_da,                       # an accumulated result. Must be initialized for all columns in cols.
                *,
                cols:           Optional[Iterable]=None,    # defines the active columns. Can be a list, keys(), range, or slice
                astype:         Optional[Type]=None,        # a type like int, float, str to cast the value if it is not that type. Optional.
                is_sparse:      bool=False,
                diagnose:       bool=False
                ) -> T_da:  # result_da
        """ 
            numeric sum of values in row and accum dicts per colunms provided. 
            will safely skip data that can't be summed.
            First three args should be provided by position only to avoid naming issues among different reductions.
        """
        # def sum_da         (row_da: T_da, reduction_da: T_da, cols: Iterable, astype: Optional[Type]=None, diagnose:bool=False

        diagnose = diagnose
        #nan_indicator = ''

        # for col, value in row_da.items():       # doing it this way requires a check for existence in each loop.
            # if col not in cols:                 # this check is not needed in the version below.
                # continue                        # 251 vs 207.

        if cols is None or is_sparse:
            # iterate by cols in row_da
            for col, val in row_da.items():
                
                if cols and col not in cols:
                    continue
                
                try:
                    # note that the "+ 0" in the following expression is important so that
                    # string operands will cause 'TypeError: can only concatenate str (not "int") to str'
                    # This will only invoke the exception when encountering non-numeric data, and does
                    # not require an initial check
                    
                    reduction_da[col] = reduction_da.get(col, 0) + val + 0
                
                #except Exception:
                except (ValueError, TypeError):
                    continue
                except Exception:
                    breakpoint() #perm
                    pass # unexpected exception
                    
            return reduction_da
            
        else:
            for col in cols:
                
                value = row_da.get(col, 0)
                # if value == nan_indicator:        # this makes the loop take 10x longer (2162) (1044% of original)
                # if isinstance(value, str):        # this makes the loop take 42% longer (294)
                # if isinstance(value, str) and value == '':  # same (294)
                # if value is None or isinstance(value, str) and not value:     (350) vs 207 = 69% longer
                # if value == '':                     # this makes the loop take 10x longer (2105) (1044% of original)
                # if isinstance(value, str) and not value:    # this makes the loop take 50% longer (305)
                # if isinstance(value, str):          # this makes the loop take 50% longer (305)
                                                    # but is needed to disallow concatenating strings.
                    # continue
                # if isinstance(value, (int, float, np.int32, np.int64)):
                    # (indent)
                
                # the try/except below is the most time efficient way to handle this while still
                # allowing for astype and nan values. (212 ms for 1000x1000 array)
                # Please note that the cols value is determined
                # prior to entering the function and must contain an iterable, even if all columns
                # are specified.
                
                # writing this loop the other way around, by going through all columns and skipping those not
                # mentioned in cols is also very inefficient.
                    
                # 213 for the version below, which seems like it should be fastest.
                # but it is slightly less advantageous because initial assignment is inside the try/except block.
                
                # try:
                    # if astype:
                        # value = row_da[col]
                        # if astype==int and isinstance(value, (str, float, bool)):
                            # value = int(float(value))
                        # elif astype==float and isinstance(value, (str, int, bool)):    
                            # value = float(value)
                        # elif astype==str and isinstance(value, (float, int, bool)):
                            # value = str(value)
                        # accum_da[col] += value
                    # else:
                        # accum_da[col] += row_da[col]
                # except Exception:
                    # continue

                # this one measured at 230
                # value = row_da[col]
                # try:
                    # if astype:
                        # if astype==int and isinstance(value, (str, float, bool)):
                            # value = int(float(value))
                        # elif astype==float and isinstance(value, (str, int, bool)):    
                            # value = float(value)
                        # elif astype==str and isinstance(value, (float, int, bool)):
                            # value = str(value)
                            
                    # accum_da[col] += value
                    
                # except Exception:
                    # continue

                # this one measured at 209 with all cols and no astype.
                if astype:
                    #value = row_da.get(col, 0)
                    try:
                        if astype is int and isinstance(value, (str, float, bool)):
                            value = int(float(value or 0))
                        elif astype is float and isinstance(value, (str, int, bool)):    
                            value = float(value or 0)
                        elif astype is str and isinstance(value, (float, int, bool)):
                            value = str(value)
                        
                        reduction_da[col] += value
                
                    except ValueError:
                        continue
                    except Exception:
                        breakpoint() #perm
                        pass # unexpected exception
                        
                else:
                    try:
                        # note that the "+ 0" in the following expression is important so that
                        # string operands will cause 'TypeError: can only concatenate str (not "int") to str'
                        # This will only invoke the exception when encountering non-numeric data, and does
                        # not require an initial check
                        
                        reduction_da[col] = reduction_da.get(col, 0) + value + 0
                    
                    #except Exception:
                    except (ValueError, TypeError):
                        continue
                    except Exception:
                        breakpoint() #perm
                        pass # unexpected exception


                    
            return reduction_da

        
    @staticmethod
    def sum_da2(row_da: T_da,                   # the current row from the daf array.
                accum_da: T_da,                 # an accumulated result. Must be initialized for all columns in cols.
                cols: Iterable,                 # defines the active columns. Can be a list, keys(), range, or slice
                astype: Optional[Type]=None,    # a type like int, float, str to cast the value if it is not that type. Optional.
                diagnose:bool=False
                ) -> T_da:     # result_da
        """ sum values in row and accum dicts per colunms provided. 
            will safely skip data that can't be summed.
        """

        diagnose = diagnose
        # nan_indicator = ''

        # for col, value in row_da.items():       # doing it this way requires a check for existence in each loop.
            # if col not in cols:                 # this check is not needed in the version below.
                # continue                        # 251 vs 207.

        for col in cols:
            
            value = row_da[col]
            #if value == nan_indicator:        # this makes the loop take 10x longer (2162) (1044% of original)
            # if isinstance(value, str):        # this makes the loop take 42% longer (294)
            # if isinstance(value, str) and value == '':  # same (294)
            # if value is None or isinstance(value, str) and not value:     (350) vs 207 = 69% longer
            if value == '':                     # this makes the loop take 10x longer (2105) (1044% of original)
            # if isinstance(value, str) and not value:    # this makes the loop take 50% longer (305)
                continue
            
            # the try/except below is the most time efficient way to handle this while still
            # allowing for astype and nan values. (212 ms for 1000x1000 array)
            # Please note that the cols value is determined
            # prior to entering the function and must contain an iterable, even if all columns
            # are specified.
            
            # writing this loop the other way around, by going through all columns and skipping those not
            # mentioned in cols is also very inefficient.
                
            # 213 for the version below, which seems like it should be fastest.
            # but it is slightly less advantageous because initial assignment is inside the try/except block.
            
            # try:
                # if astype:
                    # value = row_da[col]
                    # if astype==int and isinstance(value, (str, float, bool)):
                        # value = int(float(value))
                    # elif astype==float and isinstance(value, (str, int, bool)):    
                        # value = float(value)
                    # elif astype==str and isinstance(value, (float, int, bool)):
                        # value = str(value)
                    # accum_da[col] += value
                # else:
                    # accum_da[col] += row_da[col]
            # except Exception:
                # continue

            # this one measured at 230
            # value = row_da[col]
            # try:
                # if astype:
                    # if astype==int and isinstance(value, (str, float, bool)):
                        # value = int(float(value))
                    # elif astype==float and isinstance(value, (str, int, bool)):    
                        # value = float(value)
                    # elif astype==str and isinstance(value, (float, int, bool)):
                        # value = str(value)
                        
                # accum_da[col] += value
                
            # except Exception:
                # continue

            # this one measured at 209 with all cols and no astype.
            if astype:
                try:
                    if astype is int and isinstance(value, (str, float, bool)):
                        value = int(float(value))
                    elif astype is float and isinstance(value, (str, int, bool)):    
                        value = float(value)
                    elif astype is str and isinstance(value, (float, int, bool)):
                        value = str(value)
                    
                    accum_da[col] += value
            
                except Exception:
                    continue
                    
            else:
                try:
                    accum_da[col] += value
                
                except Exception:
                        continue


                
        return accum_da

        
    @staticmethod
    def sum_da3(row_da: T_da,                   # the current row from the daf array.
                accum_da: T_da,                 # an accumulated result. Must be initialized for all columns in cols.
                cols: Iterable,                 # defines the active columns. Can be a list, keys(), range, or slice
                astype: Optional[Type]=None,    # a type like int, float, str to cast the value if it is not that type. Optional.
                diagnose:bool=False
                ) -> T_da:     # result_da
        """ sum values in row and accum dicts per colunms provided. 
            will safely skip data that can't be summed.
        """

        diagnose = diagnose
        #nan_indicator = ''

        # for col, value in row_da.items():       # doing it this way requires a check for existence in each loop.
            # if col not in cols:                 # this check is not needed in the version below.
                # continue                        # 251 vs 207.

        for col in cols:
            
            value = row_da[col]
            # if value == nan_indicator:        # this makes the loop take 10x longer (2162) (1044% of original)
            # if isinstance(value, str):        # this makes the loop take 42% longer (294)
            # if isinstance(value, str) and value == '':  # same (294)
            # if value is None or isinstance(value, str) and not value:     (350) vs 207 = 69% longer
            # if value == '':                     # this makes the loop take 10x longer (2105) (1044% of original)
            if isinstance(value, str) and not value:    # this makes the loop take 50% longer (305)
                continue
            
            # the try/except below is the most time efficient way to handle this while still
            # allowing for astype and nan values. (212 ms for 1000x1000 array)
            # Please note that the cols value is determined
            # prior to entering the function and must contain an iterable, even if all columns
            # are specified.
            
            # writing this loop the other way around, by going through all columns and skipping those not
            # mentioned in cols is also very inefficient.
                
            # 213 for the version below, which seems like it should be fastest.
            # but it is slightly less advantageous because initial assignment is inside the try/except block.
            
            # try:
                # if astype:
                    # value = row_da[col]
                    # if astype==int and isinstance(value, (str, float, bool)):
                        # value = int(float(value))
                    # elif astype==float and isinstance(value, (str, int, bool)):    
                        # value = float(value)
                    # elif astype==str and isinstance(value, (float, int, bool)):
                        # value = str(value)
                    # accum_da[col] += value
                # else:
                    # accum_da[col] += row_da[col]
            # except Exception:
                # continue

            # this one measured at 230
            # value = row_da[col]
            # try:
                # if astype:
                    # if astype==int and isinstance(value, (str, float, bool)):
                        # value = int(float(value))
                    # elif astype==float and isinstance(value, (str, int, bool)):    
                        # value = float(value)
                    # elif astype==str and isinstance(value, (float, int, bool)):
                        # value = str(value)
                        
                # accum_da[col] += value
                
            # except Exception:
                # continue

            # this one measured at 209 with all cols and no astype.
            if astype:
                # value = row_da[col]
                try:
                    if astype is int and isinstance(value, (str, float, bool)):
                        value = int(float(value))
                    elif astype is float and isinstance(value, (str, int, bool)):    
                        value = float(value)
                    elif astype is str and isinstance(value, (float, int, bool)):
                        value = str(value)
                    
                    accum_da[col] += value
            
                except Exception:
                    continue
                    
            else:
                try:
                    accum_da[col] += value
                
                except Exception:
                        continue


                
        return accum_da

        
    def daf_valuecount(
            self, 
            by: str = 'row', 
            cols: Optional[T_la]=None
            ) -> T_da:
        """ count values in columns specified and return for each column,
            a dictionary of values and counts for each value in the column
            
            Need a way to specify that blank values will also be counted.
        """
            
        return self.reduce(func=type(self).count_values_da, by=by, cols=cols)


    def groupsum_daf(
            self,
            colname:        str, 
            by:             str='row',                      # determines how the func is applied.
            reduce_cols:    Optional[T_la]=None,            # columns included in the reduce operation.
            # keyfield: str=colname,                        # provide this for when no keyfield is desired?
                                                            # current operation sets keyfield to colname.
            ) -> 'Daf':
    
        result_daf = self.groupby_reduce(colname=colname, func=self.__class__.sum_da, by=by, reduce_cols=reduce_cols)
        
        return result_daf


    def multi_groupsum(
            self,
            colnames: T_ls,                                 # colnames to group over individually 
            by: str='row',                                  # determines how the func is applied.
            reduce_cols: Optional[T_la]=None,               # columns included in the reduce operation.
            # keyfield: str=colname,                        # provide this for when no keyfield is desired?
                                                            # current operation sets keyfield to colname.
            ) -> Dict[str, 'Daf']:
    
        result_dodaf = self.multi_groupby_reduce(colnames=colnames, func=self.__class__.sum_da, by=by, reduce_cols=reduce_cols)
        
        return result_dodaf


    def set_col2_from_col1_using_regex_select(self, col1: str, col2: str='', regex: str=''):
    
        """ given two cols that already exist, apply regex select to col1 to create col2
            regex should include parens that enclose the desired portion of col1.
            col2 defaults to col1 if not provided, but you must use keyword regex.
        """
        
        # from utilities import daf_utils
    
        def set_row_col2_from_col1_using_regex_select(row_da: T_da, col1: str, col2: str, regex: str) -> T_da:
            row_da[col2] = daf_utils.safe_regex_select(regex, row_da[col1])
            return row_da
            
        if not col2:
            col2 = col1

        self.apply_in_place(lambda row_da: set_row_col2_from_col1_using_regex_select(row_da, col1, col2, regex)) 


    def apply_replace_regex(self, col: str, col2: str='', replace_regex: str='') -> 'Daf':
    
        """ apply a replace regex pattern, with form '/find/replace/' to col, or 
            by from col to col2. 
            
            remove pattern:                 /find//
            replace pattern:                /find/replace/
            select pattern:                 /pretext(select)posttext/\1/
            select pattern with changes     /pretext(select)posttext/prefix\1suffix/
        
        """
        
        # from utilities import daf_utils
    
        def set_row_col2_from_col_using_regex_replace(row_da: T_da, col: str, col2: str, replace_regex: str) -> T_da:
            if col in row_da:
                row_da[col2] = daf_utils.safe_regex_replace(regex=replace_regex, s=row_da[col])
            return row_da
            
        if not col2:
            col2 = col

        self.apply_in_place(lambda row_da: set_row_col2_from_col_using_regex_replace(row_da, col, col2, replace_regex)) 

        return self
        

    def alter_daf_per_setting(self, settingsdict: T_da, setting_name: str, setting_select_dict: dict) -> 'Daf':

        """ alter a daf using setting_name in settingsdict, selected by setting_select_dict

            Example:
                biabif_daf = daf_utils.alter_daf_per_setting(
                    this_daf            = biabif_daf, 
                    settingsdict        = argsdict, 
                    setting_name        = 'daf_replace_regex_spec', 
                    setting_select_dict = {'spec_name': archive_basename},
                    )
        
                cvrbif_daf = daf_utils.alter_daf_per_setting(
                    this_daf            = cvrbif_daf, 
                    settingsdict        = argsdict, 
                    setting_name        = 'daf_replace_regex_spec', 
                    setting_select_dict = {'spec_name': cvr_fn},
                    )
        
        daf_replace_regex_spec: 
            list of replace regex specs to apply to biabif, cvrbif, and cvrvotes, specific to each 
            archive or cvr file. Can use to alter ballotids if duplicate names do not reference 
            the same ballot. Format is {'spec_name':archive_filename, "colname"="colname str", 
            replace_regex="replace regex str"}. 
            
            Apply replace regex to ballotids in biabif from specified archive, or cvrbif or cvrvotes 
            from specified cvr fn.  replace_regex is like "/findstr/replstr/
            
        The action is to filter to 'spec_name' in colname, and then apply the replace regex.
        
        """

        setting_lod = settingsdict.get(setting_name, None)
        if setting_lod is None:
            print(f"{setting_name = } not found in settingdict")
            breakpoint()  # perm
            pass
            
        if not setting_lod:
            return self     # do nothing.
            
        if isinstance(setting_lod, dict):
            setting_lod = [setting_lod]
            
        try:    
            setting_daf = Daf.from_lod(setting_lod)
        except Exception:
            breakpoint()  # perm (fixed edge case in .from_lod()
            pass
            
        alter_specs_daf = setting_daf.select_by_dict(setting_select_dict)
        
        return self.alter_daf_per_alter_specs_daf(
            alter_specs_daf     = alter_specs_daf,     # daf containing cols: colname, replace_regex
            )



    def alter_daf_per_alter_specs_daf(
            self,                       # daf to alter
            alter_specs_daf: 'Daf',     # daf containing cols: colname, replace_regex
            ) -> 'Daf':
        r""" 
            Given daf array, use alter_specs_daf to alter the array with individual replace_regex specs.
            
            alter_specs_daf should be preselected to retain only those specs (rows) that apply using "spec_name"
            alter_specs_daf has columns:
                spec_name       - used to select the appropriate alter specs.
                                    prior to entry, the daf contains only those records that are appropriate.
                colname         - column to alter
                replace_regex   - string formatted to provide find and replace.
        
            Sometimes, duplicate ballot_id value will be used for different ballots
            This can happen if the ballots are in different src or cvr files.
            Check argsdict[setting_name] to see if it is a single dict or a lod. Convert da to loda
            Match field 'spec_name' in lod with value 'spec_name'.
            in this dict, use colname specified.
            Apply 'replace_regexp' to column specified.
            
            usage:
                if argsdict['daf_replace_regex_spec']:
                    alter_specs_daf = Daf.from_lod(argsdict['daf_replace_regex_spec'])
                    alter_specs_daf = alter_specs_daf.select_dict({'spec_name': spec_name})
                    
                    for alter_spec_da in alter_specs_daf:
                        this_daf.apply_replace_regex(colname=alter_spec_da['colname']
                        
            Example:
                alter_specs_daf = Daf.from_lod(
                    [{  "spec_name": "Database 2 All Tabulators Results.zip", 
                        "colname": "ballot_id", 
                        "replace_regex": r"/04000_(\d\d\d\d\d_\d\d\d\d\d\d)/14000_\1/"
                     },
                     {  "spec_name": "CVR_Export_20230206180329 DB2 Certified.csv", 
                        "colname": "ballot_id",
                        "replace_regex":r"/04000_(\d\d\d\d\d_\d\d\d\d\d\d)/14000_\1/"
                     }])

            biabif_daf = Daf(cols=['ballot_id', 'batchid', 'archive_basename'],
                lol = [['01780_00000_983814', '01780_00000', 'Database 1 All Tabulators Results.zip'],
                       ['01780_00000_996586', '01780_00000', 'Database 1 All Tabulators Results.zip'],
                       ['01780_00000_998212', '01780_00000', 'Database 1 All Tabulators Results.zip'],
                       ['04000_00001_000001', '04000_00001', 'Database 1 All Tabulators Results.zip'],
                       ['04000_00001_000002', '04000_00001', 'Database 1 All Tabulators Results.zip'],
                       ['04000_00001_000003', '04000_00001', 'Database 1 All Tabulators Results.zip'],
                      ],
                )      
                
            result_daf = biabif_daf.alter_daf_per_alter_specs_daf(
                alter_specs_daf=alter_specs_daf)
            
            expected_daf = Daf(cols=['ballot_id', 'batchid', 'archive_basename'],
                lol = [['01780_00000_983814', '01780_00000', 'Database 1 All Tabulators Results.zip'],
                       ['01780_00000_996586', '01780_00000', 'Database 1 All Tabulators Results.zip'],
                       ['01780_00000_998212', '01780_00000', 'Database 1 All Tabulators Results.zip'],
                       ['14000_00001_000001', '04000_00001', 'Database 1 All Tabulators Results.zip'],
                       ['14000_00001_000002', '04000_00001', 'Database 1 All Tabulators Results.zip'],
                       ['14000_00001_000003', '04000_00001', 'Database 1 All Tabulators Results.zip'],
                      ],
                )                  
        """
        for alter_spec_da in alter_specs_daf:

            self.apply_replace_regex(col=alter_spec_da['colname'], replace_regex=alter_spec_da['replace_regex'])
        
        return self



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
    
    

    # @staticmethod
    # def sum_da(row_da: T_da, accum_da: T_da, cols: Optional[T_la]=None, astype:Type=int, diagnose:bool=False) -> T_da:     # result_da
        # """ sum values in row and accum dicts per colunms provided. 
            # will safely skip data that can't be summed.
        # """
        # diagnose = diagnose
        # nan_indicator = ''
        
        # if cols is None:
            # cols_list = []
        # elif not isinstance(cols, list):
            # cols_list = [cols]
        # else:
            # cols_list = cols
            
        # if len(cols_list) > 10:    
            # cols_list_or_dict = dict.fromkeys(cols_list)
        # else:
            # cols_list_or_dict = cols_list
            
        # for key, value in row_da.items():
            # if value == nan_indicator:
                # continue
        
            # if cols_list_or_dict and key not in cols_list_or_dict:
                # # accum_da[key] = ''
                # continue
                
            # if astype==int and isinstance(value, (str, float, bool)):
                # value = int(float(value))
            # elif astype==float and isinstance(value, (str, int, bool)):    
                # value = float(value)
            # elif astype==str and isinstance(value, (float, int, bool)):
                # value = str(value)

            # try:
                # if key in accum_da:
                    # accum_da[key] += value
                # else:
                    # accum_da[key] = value
            # except Exception:
                # pass
        # return accum_da


    @staticmethod
    def diff_da(d1_da: T_da, d2_da: T_da, keys: Optional[T_la]=None) -> T_da:     # result_da
        """ difference of two dictionaries by keys, according to the columns provided.
            if keys not specified or value does not exist in both rows, then do not include a result.
            if value exists in only one row, assume the value in the other row is 0.
        """

        keys_list = {}
        
        if keys is None:
            keys_list = []
        elif not isinstance(keys, list):
            keys_list = [keys]
        else:
            keys_list = keys
        
        # the 'or 0' part handles null string.
        result_da = {key: (d1_da.get(key, 0) or 0) - (d2_da.get(key, 0) or 0)
                        for key in keys_list}
                
        return result_da


    @staticmethod
    def count_values_da(
            row_da: T_da, 
            reduction_da: T_da, 
            cols: Iterable, 
            *, 
            omit_nulls: bool=False,
            
            ) -> T_dodi:
        """ incrementally build the result_dodi, which is the valuecounts for each item in row_da.
            can be used to calculate valuecounts over all rows and chunks.
            
            row_da may be scalar values (typically strings that are to be counted)
            but may also be a dodi of totals from a set of chunks that are to be combined.
            
            Intended use is to use this to calculate valuecounts by scanning all rows of a daf.
            Return a dodi.
            Put those in a daf table and then scan those combined values and create a singular result.
            
            This is a reducing and accumulating operation. Can be used with daf.reduce().
            
            For example:
                value_counts_dodi = {}
                cols_of_interest  = ['gender', 'location']
            
                value_counts_dodi = my_daf.reduce(count_values_da, value_counts_dodi, cols_of_interest, omit_nulls=True)
                
            results in (something, like, this):
            
                value_counts_dodi = {'gender': {'Male': 2435, 'Female': 2489}, 'location':{'north': 2433, 'south': 345}}
            
        """
    
        # if cols is None:
            # cols_dict = {}
        # else:
            # cols_dict = dict.fromkeys(cols)
            
        if not reduction_da:
            reduction_da = {}
            
        result_dodi = reduction_da
        
        for col in cols:

            val = row_da[col]
            
            if omit_nulls and isinstance(val, str) and not val:
                continue
            
            if val and isinstance(val, list):
                # val is a list of values
                if col not in result_dodi:
                    result_dodi[col] = val
                else:    
                    result_dodi[col].append(val)
                continue
                
            if val and isinstance(val, dict):
                # val is a dict of values determined in another pass.
                if col not in result_dodi:
                    result_dodi[col] = val
                else:    
                    result_dodi[col] = Daf.sum_da(val, result_dodi[col])
                continue
                
            if col not in result_dodi or not result_dodi[col]:
                result_dodi[col] = {val: 1}
            elif val not in result_dodi[col]:
                result_dodi[col][val] = 1
            else:    
                result_dodi[col][val] += 1
        
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
                
                

    # def classify_by_logic_spec(self, groupcol, input_da: T_da) -> T_da: # groupname
    
        # for logic_spec_da in self:



                    
    #===============================================
    # functions not following apply or reduce pattern
    
    # this function does not use reduction approach.
    def sum(
            self, 
            colnames_ls: Optional[T_ls]=None, 
            numeric_only: bool=False,
            ) -> dict: # sums_di
        """ total the columns in the table specified, and return a dict of {colname: total,...} 
            unit tests exist
        """


        if colnames_ls is None:
            cleaned_colnames_is = self.hd.keys()
        elif not (numeric_only and self.dtypes):
            cleaned_colnames_is = [col for col in colnames_ls if col in self.hd.keys()]
        else:    
            cleaned_colnames_is = [col for col in colnames_ls if col in self.hd and self.dtypes.get(col) in [int, float]]
        
        sums_d = dict.fromkeys(cleaned_colnames_is, 0.0)
        
        for colname, colidx in self.hd.items():
            if colname not in cleaned_colnames_is:
                # sums_d[colname] = ''
                continue
            for la in self.lol:
                if la[colidx]:
                    if numeric_only:
                        sums_d[colname] += Daf._safe_tofloat(la[colidx])
                    else:
                        sums_d[colname] += float(la[colidx])

        sums_d = daf_utils.set_dict_dtypes(sums_d, dtypes=self.dtypes)
        
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
        
        import numpy as np
        
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
            daf_utils.safe_del_key(valuecounts_di, '') 
                
        if sort:
            valuecounts_di = dict(sorted(valuecounts_di.items(), key=lambda x: x[1], reverse=reverse))

        return valuecounts_di


    def valuecounts_for_colnames_ls(
            self, 
            colnames_ls: Optional[T_ls]=None, 
            sort: bool=False, 
            reverse: bool=True,
            omit_nulls: bool=False,
            ) -> T_dodi:
        """ return value counts for a set of columns or all columns if no colnames_ls are provided.
            sort if noted from most prevalent to least in each column.
            
            Note: This is an inefficient algorithm
        """
    
        if not colnames_ls:
            colnames_ls = self.columns()
            
        colnames_ls = cast(T_ls, colnames_ls)    

        valuecounts_dodi: T_dodi = {}
        
        for colname in colnames_ls:
            valuecounts_dodi[colname] = \
                self.valuecounts_for_colname(colname, sort=sort, reverse=reverse, omit_nulls=omit_nulls)

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

        # from utilities import daf_utils

        for col_def_ta in col_def_lot:
            col_name, col_dtype, col_format, col_profile = col_def_ta
            
            col_data_la = self[:, col_name].to_list()
            
            info_dod[col_name] = daf_utils.list_stats(col_data_la, profile=col_profile)
            
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
            new_cols = ['key'] + daf_utils._generate_spreadsheet_column_names_list(num_cols=len(self.lol))

        # transpose the array
        new_lol = [list(row) for row in zip(*self.lol)]
        
        if include_header:
            # add a new first column which will be the old column names row.
            # from utilities import daf_utils
            
            new_lol = daf_utils.insert_col_in_lol_at_icol(icol=0, col_la=self.columns(), lol=new_lol)
        
        return Daf(lol=new_lol, name=self.name, keyfield=new_keyfield, cols=new_cols, use_copy=True)        

    def derive_join_translator(
        self,
        other_daf: 'Daf',                               # The other Daf instance to join with
        shared_fields: Optional[T_ls] = None,           # Columns shared between tables that do not require renaming
                                                        # keyfields do not need to be added here.
        omit_other_cols: Optional[T_ls]=None,           # cols to omit from other (use instead of shared fields)
        tag_other: bool = False,                        # if True, and col not in shared_fields, add suffix tag to other_daf cols
                                                
        ) -> 'Daf':  # Translator Daf
        r"""
        Derive a Daf translator for resolving column conflicts between two tables.

        Args:
            other_daf: 'Daf',                               # The other Daf instance to join with
            shared_fields: Optional[T_ls] = None,           # Columns shared between tables that do not require renaming
                                                            # keyfields do not need to be added here.
            tag_other: bool = False,                        # if True, and col not in shared_fields, add suffix tag to other_daf cols

        Returns:
            Daf: Translator table with resolved column names and source mapping.

        Example:
            Source Daf objects:

            `my_daf`:
            | id |  name   |  status  |
            | -: | ------: | -------: |
            |  1 |   Alice |   active |
            |  2 |     Bob | inactive |
            |  3 | Charlie |   active |

            \[3 rows x 3 cols; keyfield='id'; 3 keys ] (Daf)

            `other_daf`:
            | id |   address   | salary |
            | -: | ----------: | -----: |
            |  1 | 123 Main St |  50000 |
            |  3 |  456 Elm St |  70000 |
            |  4 |  789 Oak St |  60000 |

            \[3 rows x 3 cols; keyfield='id'; 3 keys ] (Daf)

            Code:
            ```python
            translator_daf = my_daf.derive_join_translator(other_daf, shared_fields=["id"])
            ```

            Resulting Translator Daf:
            | resolved_colname  | source_name | source_colname | is_keyfield |
            | ----------------: | ----------: | -------------: | ----------: |
            | id                | self        | id             | True        |
            | name              | self        | name           | False       |
            | status            | self        | status         | False       |
            | id_other          | other       | id             | False       |
            | address           | other       | address        | False       |
            | salary            | other       | salary         | False       |

            \[6 rows x 4 cols; keyfield='resolved_colname'; 6 keys ] (Daf)
            
        Template for hand-generated custom translator:

        custom_translator = Daf(
            cols = ["resolved_colname",     "source_name", "source_colname", 'is_keyfield'],
            lol  = [
                    ["id",                  'daf1',        "id",                True],
                    ["full_name",           'daf1',        "name",              False],
                    ["residential_address", 'daf2',        "address",           False],
                    ["monthly_salary",      'daf2',        "salary",            False],
            ],
            keyfield = 'resolved_colname'
        )
        
            
            
        """
        
        translator_daf = Daf.derive_join_translator_daf(
            self_keyfield       = self.keyfield,
            other_keyfield      = other_daf.keyfield,
            self_cols           = self.hd.keys(),
            other_cols          = other_daf.hd.keys(),
            self_name           = self.name,
            other_name          = other_daf.name,
            shared_fields       = shared_fields,
            omit_other_cols     = omit_other_cols,
            tag_other           = tag_other,
            )
        return translator_daf
    
    
    @classmethod
    def derive_join_translator_daf(
        cls,
        self_keyfield:      str,                        # pass self.keyfield (daf)          or esc_my_index_col (sql)
        other_keyfield:     str,                        # pass other_daf.keyfield (daf)     or esc_other_index_col (sql)
        self_cols:          Union[list, dict, T_kva],   # pass self.kd.keys() (daf)         or esc_sql_cols (sql)
        other_cols:         Union[list, dict, T_kva],   # pass other_daf.kd.keys() (daf)    or esc_sql_other_cols (sql)
        self_name:          str = '',                   # pass self.name (daf)              or esc_sql_table_name (sql)
        other_name:         str = '',                   # pass other_daf.name (daf)         or esc_sql_other_table_name (sql)
        shared_fields:      Optional[T_ls] = None,      # Columns shared between tables that do not require renaming (esc if sql) 
                                                        # use omit_other_cols instead of shared_fields due to better name.
        omit_other_cols:    Optional[T_ls]=None,        # cols to omit from other (esc if sql)
        tag_other:          bool = False,               # if True, and col not in shared_fields, add suffix tag to other_cols
        
        ) -> 'Daf':  # Translator Daf
        """
        Derive a Daf translator for resolving column conflicts between two tables.
        This classmethod version is suitable for use by sql join.

        Args:
            self_keyfield:      str,                        # pass self.keyfield (daf)          or esc_my_index_col (sql)
            other_keyfield:     str,                        # pass other_daf.keyfield (daf)     or esc_other_index_col (sql)
            self_cols:          Union[list, dict, T_kva],   # pass self.kd.keys() (daf)         or esc_sql_cols (sql)
            other_cols:         Union[list, dict, T_kva],   # pass other_daf.kd.keys() (daf)    or esc_sql_other_cols (sql)
            self_name:          str = '',                   # pass self.name (daf)              or esc_sql_table_name (sql)
            other_name:         str = '',                   # pass other_daf.name (daf)         or esc_sql_other_table_name (sql)
            shared_fields:      Optional[T_ls] = None,      # Columns shared between tables that do not require renaming (esc if sql)
            tag_other:          bool = False,               # if True, and col not in shared_fields, add suffix tag to other_cols

        Returns:
            Daf: Translator table with resolved column names and source mapping.

        Example:
            Source Daf objects:
        """
        
        
        if shared_fields is None:
            shared_fields = []
            
        if omit_other_cols is None:
            omit_other_cols = []
            
        if self_keyfield and self_keyfield not in shared_fields:
            shared_fields.append(self_keyfield)
            
        if other_keyfield and other_keyfield not in shared_fields:
            shared_fields.append(other_keyfield)
        
        shared_fields   = daf_utils.to_dn_if_list(shared_fields)
        self_cols       = daf_utils.to_dn_if_list(self_cols)
        other_cols      = daf_utils.to_dn_if_list(other_cols)

        # Get column names and names from both Dafs  (list of KeyViews of Any)
        colnames_lokva = [self_cols, other_cols]
        
        # Assign suffixes for source tables based on table names or default
        source_suffixes = [
            f"_{self_name}"         if self_name else "_daf1", 
            f"_{other_name}"        if other_name else "_daf2",
            ]
            
        source_names = [
            self_name if self_name else "daf1", 
            other_name if other_name else "daf2",
            ]

        # Identify columns that are common but not shared fields
        common_colnames = [col for col in colnames_lokva[0] 
                                if col in colnames_lokva[1] and col not in shared_fields and col not in omit_other_cols]
        
        # set.intersection(*[set(colnames_lists[0]) for cols in colnames_lists[1]]) - shared_fields

        # Define the Daf structure first
        translator_daf = Daf(cols=["resolved_colname", "source_name", "source_colname", "is_keyfield"])

        # Populate the Daf by appending rows directly
        for idx, (colnames_kva, source_suffix) in enumerate(zip(colnames_lokva, source_suffixes)):
            for col in colnames_kva:
                if idx and (col in shared_fields or col in omit_other_cols):
                    continue
                    
                if (idx and tag_other or 
                        col and col in common_colnames):    
                    resolved_colname = f"{col}{source_suffix}"
                else:
                    resolved_colname = col

                translator_daf.append([
                    resolved_colname,   # resolved_colname
                    #idx,               # source_index
                    source_names[idx],  # source_name
                    col,                # source_colname
                    bool(col == self_keyfield or col == other_keyfield),   # is_keyfield
                ])

        # Set the keyfield for the resulting Daf
        translator_daf.set_keyfield("resolved_colname")
        return translator_daf


    def join(
        self,
        other_daf: 'Daf',                                   # The other Daf instance to join with.
        how: str = 'inner',                                 # Type of join - 'inner', 'left', 'right', 'outer'. Default is 'inner'.
        shared_fields: Optional[T_ls] = None,               # List of fields to ignore in conflict resolution (shared fields)
        tag_other: bool = False,                            # if not a shared field or keyfield, suffix all other_daf fields with _{name}
        custom_translator_daf: Optional['Daf'] = None,      # provide a custom translater to provide all naming details.
        diagnose: bool = False,                             # Enable diagnostic logging.
        name: str='',                                       # name for the joined instance.
    ) -> 'Daf':
        """
        Perform a join operation between two Daf instances using their keyfields.

        Args:
            other_daf: 'Daf',                                   # The other Daf instance to join with.
            how: str = 'inner',                                 # Type of join - 'inner', 'left', 'right', 'outer'. Default is 'inner'.
            shared_fields: Optional[T_ls] = None,               # List of fields to ignore in conflict resolution (shared fields)
            tag_other: bool = False,                            # if not a shared field or keyfield, suffix all other_daf fields with _{name}
            custom_translator_daf: Optional['Daf'] = None,      # provide a custom translater to provide all naming details.
            diagnose: bool = False,                             # Enable diagnostic logging.
            name: str='',                                       # name for the joined instance.

        Returns:
            Daf: A new Daf instance containing the result of the join.
            
        notes:
            1. both dataframes must have keyfield defined. The keyfield will be the field that is used for the join.
            2. dataframes can be named, otherwise, the names default to 'daf1' and 'daf2'
            3. a single join translater can be used for joins of multiple dataframes. In that case, name all dataframes.
            
        """
        if how not in {"inner", "left", "right", "outer"}:
            raise ValueError(f"Unsupported join type: {how}")

        if not self.keyfield or not other_daf.keyfield:
            raise ValueError("Both Daf instances must have a keyfield defined for a join operation.")

        if diagnose:
            logs.sts(f"{logs.prog_loc()} Initiating join:\nself:\n{self}\nother_daf:\n{other_daf}", 3)

        # Derive or use custom translator Daf
        if custom_translator_daf:
            translator_daf = custom_translator_daf
        else:
            translator_daf = self.derive_join_translator(other_daf, shared_fields=shared_fields, tag_other=tag_other)

        if diagnose:
            logs.sts(f"{logs.prog_loc()} Translator Daf:\n{translator_daf}", 3)
            
        join_names_ls = [self.name or 'daf1', other_daf.name or 'daf2']
                
        # we allow the translator to contain records for more than two dafs that may be joined in a chain opeation.
        try:
            eff_translator_daf = translator_daf.select_where(lambda row: bool(row.get('source_name') in join_names_ls))
        except Exception:
            breakpoint() # perm okay
            pass
        
        resolved_colnames = eff_translator_daf[:, "resolved_colname"].to_list()

        # Prepare the resulting Daf
        result_daf = Daf(cols=resolved_colnames, name=name)
        keyfield = self.keyfield    # will be set at the end.

        # Helper function to fetch a record by key, with silent error
        def fetch_record(daf, key):
            return daf.select_record(key, silent_error=True)

        # Track matched keys for outer joins
        matched_keys = set()

        # Perform the join
        for row_da in self:
            key = row_da[keyfield]
            other_row_da = fetch_record(other_daf, key)

            if other_row_da:
                matched_keys.add(key)
                combined_record = Daf.join_records(
                    [row_da, other_row_da], 
                    translator_daf,
                    join_names_ls,
                )
                result_daf.append(combined_record)
            elif how in {"left", "outer"}:
                combined_record = Daf.join_records(
                    [row_da, None], 
                    translator_daf,
                    join_names_ls,
                )
                result_daf.append(combined_record)

        if how in {"right", "outer"}:
            unmatched_keys = [
                key for key in other_daf.kd if key not in matched_keys
            ]
            for key in unmatched_keys:
                other_row_da = fetch_record(other_daf, key)
                combined_record = Daf.join_records(
                    [None, other_row_da], 
                    translator_daf,
                    join_names_ls,
                )
                result_daf.append(combined_record)

        # Set the keyfield in the resulting Daf
        result_daf.set_keyfield(keyfield)

        if diagnose:
            logs.sts(f"{logs.prog_loc()} Resulting Daf:\n{result_daf}", 3)

        return result_daf
        

    @staticmethod
    def join_records(
        records: List[Optional[T_da]],  # List of dictionaries representing the source records
                                        # only two records are supported here.
                                        # sometimes either one can be None if a corresponding record is not available.
                                        
        translator_daf: 'Daf',          # Translator Daf mapping resolved columns to source
        
        join_names_ls: Optional[T_ls]=None ,  # names of the two Daf arrays supplying the records.
                                        #  required only if there are more than two source_names specified in the translator.
                                        # this is used when a single translator is used for chained joins.

    ) -> T_da:
        """
        Combine multiple records using a join translator Daf. This works on one record from each array to be joined.

        Args:
            records (List[Optional[T_da]]): List of source records, indexed by source_index.
            translator_daf (Daf): Translator Daf with resolved column mapping.
            join_names_ls: names of the two Daf arrays supplying the records 
                            required only if there are more than two source_names specified in the translator.


        Returns:
            T_da: Combined record as a dictionary.
        """
        combined_record = {}
        
        if not join_names_ls:
            join_names_ls = translator_daf[:, 'source_name'].to_list(unique=True)
            if len(join_names_ls) > 2 or not join_names_ls:
                raise ValueError ("join_names_ls must be specified as translator_daf has more than two source_names")

        # Iterate through the translator Daf
        for translator_row in translator_daf:
            source_name     = translator_row["source_name"]
            if join_names_ls and source_name not in join_names_ls:
                continue
            source_index    = join_names_ls.index(source_name)
            resolved_col    = translator_row["resolved_colname"]
            source_colname  = translator_row["source_colname"]
            is_keyfield     = translator_row.get("is_keyfield", False)

            # Handle keyfield explicitly
            if is_keyfield:
                # Take the keyfield value from the first non-None record
                for record in records:
                    if record and source_colname in record:
                        combined_record[resolved_col] = record[source_colname]
                        break
                else:
                    combined_record[resolved_col] = None
                continue

            # Handle all other columns
            if records[source_index] is not None:
                combined_record[resolved_col] = records[source_index].get(source_colname, None)
            else:
                combined_record[resolved_col] = None

        return combined_record
            
    #===============================
    # wide to narrow and narrow to wide conversion
    
    def wide_to_narrow(self, 
            id_cols: T_ls,                      # identify the record but are not used to identify values
            varname_colname: str = 'varname',   # narrow format provides varname for each value col.
            value_colname: str = 'value',       # column of values from each value column
            ) -> 'Daf':
        """
        Convert a wide DataFrame to a narrow format.
        This can be called "melt" or "unpivot"
        
        Parameters:
            self (daf.Daf): The wide Daffodil dataframe to convert.
            id_cols (List[str]): Columns to use as identifier variables.
            value_cols (Union[str, List[str]]): Columns to unpivot.
            varval_cols: defaults to 'variable', 'value'
        
        Returns:
            equivalent dataframe in narrow (i.e tidy) format.
        """
        
        narrow_daf = Daf(cols= id_cols + [varname_colname, value_colname])
        
        value_cols = [colname for colname in self.columns() if colname not in id_cols]
        
        for row_da in self:
            # accept id values from original wide rows.
            new_row = {col: row_da[col] for col in id_cols}
            
            # add a new row for each column
            for col in value_cols:
                new_row[varname_colname] = col
                new_row[value_colname] = row_da[col]
                narrow_daf.append(new_row)
                
        return narrow_daf


    def narrow_to_wide(self,
            id_cols: T_ls,
            varname_col: str = 'variable',
            value_col: str = 'value',
            wide_cols: Optional[T_ls]=None,
            ) -> 'Daf':
        """
        Convert a narrow Daf DataFrame to a wide format.
        This can be called "pivot" or "spread"
        
        Parameters:
            self (Daffodil DataFrame): The narrow DataFrame to convert.
            id_cols:        Columns to use as index/identifier.
            varname_col:    Column to pivot into new DataFrame columns.
            value_col:      Column to pivot into new DataFrame values.
        
        Returns:
            Daf: The wide DataFrame.
            
        """
        wide_daf = Daf()
        
        wide_row_da = {}
        for row_da in self:
            index_cols_da = {col: row_da[col] for col in id_cols}
            if not wide_row_da:
                wide_row_da = index_cols_da.copy()
            elif not daf_utils.is_d1_in_d2(index_cols_da, wide_row_da):
                wide_daf.append(wide_row_da)
                wide_row_da = index_cols_da.copy()

            var_name = row_da[varname_col]
            value    = row_da[value_col]
                    
            wide_row_da[var_name] = value
            
        wide_daf.append(wide_row_da)

        return wide_daf
    
    
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
            header:         Optional[T_ls]=None, # use this header instead.
            ) -> str:
        """ provide an full md table given a daf representation """

        daf_lol = self.daf_to_lol_summary(max_rows=max_rows, max_cols=max_cols, disp_cols=disp_cols)
        
        header_exists = bool(self.hd)
        
        mdstr = md.md_lol_table(
            daf_lol, 
            header              = header, 
            includes_header     = header_exists, 
            just                = just or ('>' * len(self.hd)), 
            omit_header         = not header_exists, 
            shorten_text        = shorten_text, 
            max_text_len        = max_text_len, 
            smart_fmt           = smart_fmt,
            
            )
        if include_summary:
            # this is okay with change to tuple keys.
            mdstr += f"\n\\[{len(self.lol):,} rows x {self.num_cols():,} cols; keyfield='{self.keyfield}'; {len(self.kd):,} keys ] ({self.name or type(self).__name__})\n"
        return mdstr
        

    def to_md_cols(
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
        """ treat rows as columns """

        daf_lol = self.daf_to_lol_summary(max_rows=max_rows, max_cols=max_cols, disp_cols=disp_cols)
        
        #header_exists = bool(self.hd)
        #breakpoint() #temp
        
        mdstr = md.md_cols_lol_table(
                cols_lol        = daf_lol, 
                header          = None, 
                just            = just, 
                omit_header     = True, 
                shorten_text    = shorten_text,
                max_text_len    = max_text_len,
                smart_fmt       = smart_fmt,
                )
        # if include_summary:    
            # mdstr += f"\n\[{len(self.lol)} rows x {len(self.hd)} cols; keyfield={self.keyfield}; {len(self.kd)} keys ] ({type(self).__name__})\n"
        return mdstr

    def daf_to_lol_summary(self, max_rows: int=10, max_cols: int=10, disp_cols:Optional[T_ls]=None) -> T_lola:
    
        # from utilities import daf_utils

        # first build a basic summary by adding colnames, if they exist.
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

        num_rows    = self.num_rows()
        num_cols    = self.num_cols()

        if max_rows and num_rows <= max_rows:
            # Get all the rows, but potentially limit columns
            result_lol = daf_utils.reduce_lol_cols(result_lol, max_cols=max_cols)
        
        else:
            # Get the first and last portion of rows
            
            first_lol   = self.lol[:max_rows//2]
            last_lol    = self.lol[-(max_rows//2):]
            divider_lol = [['...'] * num_cols]
            
            result_lol  = [colnames_ls] + first_lol + divider_lol + last_lol
            result_lol  = daf_utils.reduce_lol_cols(result_lol, max_cols=max_cols)

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
            does not set the keyfield
        """
            
        value_counts_di   = self.valuecounts_for_colname(colname=colname, sort=sort, reverse=reverse)
        
        if omit_nulls:
            daf_utils.safe_del_key(value_counts_di, '') 

        value_counts_daf = Daf.from_lod_to_cols([value_counts_di], cols=[colname, 'counts'])

        if include_total:
            value_counts_daf.append({colname: ' **Total** ', 'counts': sum(value_counts_daf[:,'counts'].to_list())})
            
        return value_counts_daf
        

    # # DEPRECATED
    # @classmethod
    # def from_hllola(cls, hllol: T_hllola, keyfield: str='', dtypes: Optional[T_dtype_dict]=None) -> 'Daf':
        # """ Create Daf instance from hllola type.
            # This is used for all DB. loading.
            # test exists in test_daf.py
            
            # DEPRECATED
        # """
        
        # hl, lol = hllol
        
        # return cls(lol=lol, cols=hl, keyfield=keyfield, dtypes=dtypes)
        
        
# ALIASES
# these must not be established until after the class is fully defined.
# Pydf = Daf
# Pydf.pydf_to_lol_summary     = Daf.daf_to_lol_summary
# Pydf.split_pydf_into_ranges  = Daf.split_daf_into_ranges
# Pydf.select_records_pydf     = Daf.select_records_daf
# Pydf.pydf_sum                = Daf.daf_sum
# Pydf.pydf_valuecount         = Daf.daf_valuecount
# Pydf.groupsum_pydf           = Daf.groupsum_daf
# Pydf.gen_stats_pydf          = Daf.gen_stats_daf
# Pydf.value_counts_pydf       = Daf.value_counts_daf

# Daf.groupsum                 = Daf.groupsum_daf
# Daf.value_counts             = Daf.valuecounts_for_colname

# Pydf.md_pydf_table = Pydf.to_md            


class DafIterator:
    def __init__(self, this_daf: Daf, rtype: Type[Union[Dict[str, Any], KeyedList, list]]):
        self.this_daf = this_daf
        self.rtype = rtype
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self) -> Union[Dict[str, Any], KeyedList, list]:
        if self._index < len(self.this_daf.lol):
            row = self.this_daf.lol[self._index]
            self._index += 1
            if self.rtype is dict:
                return dict(zip(self.this_daf.hd.keys(), row))
            elif self.rtype == KeyedList:
                return KeyedList(self.this_daf.hd, row)
            elif self.rtype is list:
                return row
            
            else:
                raise NotImplementedError(f"Unknown return type: {self.rtype}")
        else:
            self._index = 0
            raise StopIteration
            
            
