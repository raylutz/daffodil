# daf_utils.py

# copyright (c) 2024 Ray Lutz

import io
import os
import csv
import re
import math
import operator
import json
import datetime
import statistics
import ast
import platform

import xlsx2csv     # type: ignore
#import numpy as np


from typing import List, Dict, Any, Tuple, Optional, Union, cast, Type, Iterable #, Callable

def fake_function(a: Optional[List[Dict[str, Tuple[int,Union[Any, str, Iterable]]]]] = None) -> Optional[int]:
    return None or cast(int, 0)       # pragma: no cover


from daffodil.lib.daf_types import T_lola, T_loda, T_dtype_dict, T_da, T_ds, T_hdlola, T_la, T_loti, T_ls, T_doda, T_buff, T_li, T_lr
                    

def is_linux() -> bool: 
    return platform.system() == 'Linux'
    

# def apply_dtypes_to_hdlol(hdlol: T_hdlola, dtypes: T_dtype_dict, from_str: bool=True) -> T_hdlola:
    # # do we need this any more? Use my_daf.apply_types()

    # hd, lol = hdlol
    
    # if not lol or not lol[0] or not hd:
        # return (hd, lol)

    # # make a worklist of those fields that need to be checked.
    # dtypes_worklist: List[Tuple[int, Type]] = []
    # # create a list of types
    # for idx, col in enumerate(hd.keys()):
        # desired_type = dtypes.get(col, str)
        # if desired_type == str and from_str:
            # continue
        # dtypes_worklist.append( (idx, desired_type ) )

    # for la in lol:
        # set_type_la(la, dtypes_worklist)
        
    # return (hd, lol)
    

def set_type_la(la: T_la, dtypes_worklist: List[Tuple[int, Type]]) -> T_la:
    """ set the types of each item in place based on a list of types without making a copy,
        and only working on those fields that may need work.
    """
    
    for idx, desired_type in dtypes_worklist:
        value = la[idx]
        if isinstance(value, desired_type):
            continue
        if isinstance(value, str) and not value:
            # value is null str.
            # leave alone.
            continue
        if desired_type in (int, bool):
            try:
                la[idx] = int(value)
            except ValueError:
                try:
                    la[idx] = int(float(value))
                except ValueError:
                    if value in ('False', 'True'):
                        la[idx] = int(eval(value))
                
        elif desired_type == float:
            try:
                la[idx] = float(value)
            except ValueError:
                if value in ('False', 'True'):
                    la[idx] = float(eval(value))
            
        elif desired_type == str:    
            la[idx] = str(value)
    
    return la
    

def set_cols_da(da: T_da, cols: T_ls, default: Any='') -> T_da:
    """ Set keys in dictionary da to be exactly cols
        Use default if key not already in da.
    """
    
    new_da = {k:da.get(k, default) for k in cols}
    return new_da


def select_col_of_lol_by_col_idx(lol: T_lola, col_idx: int) -> T_la:
    """
    select a row and col from lol
    
    Note: this creates a new object.
        
    """
    result_la = []
    try:
        result_la = [la[col_idx] for la in lol]
    except IndexError:
        pass
    
    return result_la
    

def unflatten_hdlol_by_cols(hdlol: T_hdlola, cols: T_ls) -> T_hdlola:
    """ 
        given a lod and list of cols, 
        convert cols named to either list or dict if col exists and it appears to be 
            stringified using f"{}" functionality.
            
    """

    if not hdlol or not hdlol[0] or not hdlol[1]:
        return hdlol
        
    cols_da, lol = hdlol  
        
    for col in cols:
        if col in cols_da:
            col_idx = cols_da[col]
            for la in lol:
                val = la[col_idx]
                if val and isinstance(val, str) and (val[0], val[-1]) in [('[', ']'), ('{', '}'), ('(', ')')]:
                    la[col_idx] = safe_eval(val)
                else:
                    la[col_idx] = val
                    
    return (cols_da, lol)            


class NpEncoder(json.JSONEncoder):
    #This is needed to allow np.int64 to be converted.
    def default(self, obj):
        import numpy as np
        
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def json_encode(data_item: Any, indent: Optional[int]=None) -> str:
    # use ensure_ascii=False
    # encoding="utf-8" is not supported.
    # if indent is left as None, there is no indenting.
    if data_item is None:
        return ''
    return json.dumps(data_item, cls=NpEncoder, default=str, indent=indent, ensure_ascii=False)    


def make_strbool(val: Union[bool, str, int, None]) -> str:
    # make a strbool value like 'is_bmd' and allow both bool or str types.
    
    return '1' if test_strbool(val) else '0'
    
    
def test_strbool(val: Union[bool, str, int, None, object]) -> bool:
    # test a strbool value like 'is_bmd' and allow both bool or str types.
    
    if isinstance(val, bool):
        return val
    elif isinstance(val, str):
        return bool(val.lower() in ('1', 'true', 'yes'))
    elif isinstance(val, int):
        return bool(val == 1)
    elif val is None:
        return False
    elif not bool(val == val):  # nan
        return False
    else:    
        breakpoint() #perm
        error_beep()
        
    return False    # token return for mypy.
    
        
def xlsx_to_csv(xlsx: bytes, sheetname: Optional[str]=None, add_trailing_blank_cols: bool=True) -> bytes:
    """ convert xlsx file in a buffer to csv file in a buffer. 
        Additional blank columns are added so all rows have the same number of values.
        xlsx2csv returns minimal records, stopping when the last value is filled.
    """
    
    buff = io.BytesIO(xlsx)
    buff_out = io.StringIO()
    sts("Converting using xlxs2csv...", 3)
    xlsx2csv.Xlsx2csv(buff, outputencoding="utf-8").convert(buff_out, sheetname=sheetname)
    sts(f"Conversion to buff_out completed: {len(buff_out.getvalue())} bytes.", 3)

    buff_out.seek(0)
    if add_trailing_blank_cols:
        sts("Adding trailing columns...", 3)
        buff_out = io.StringIO(add_trailing_columns_csv(buff_out.getvalue()))
        sts(f"Trailing Columns added: {len(buff_out.getvalue())} bytes.", 3)
    return buff_out.getvalue().encode('utf-8')


def add_trailing_columns_csv(str_csv:str, num_rows:int = 3) -> str:
    """ Takes a csv file in string form and returns the modified csv with equal number of columns for all rows
        Note: This seems like a lot of extra work just to prepare for the csv to be parsed, when we are fully
                parsing here just to add the columns.
        @@TODO -- The function add_trailing_columns_csv() should be DEPRECATED. It will be better to convert csv to lol, and then
                fix lengths before converting lol to df.
                
    """
    
    buff = io.StringIO(str_csv)
    buff.seek(0)
    reader = csv.reader(buff)

    # Get max number of columns
    max_col = max([len(next(reader)) for _ in range(num_rows)])
    buff.seek(0)
    buff_out = io.StringIO()
    writer = csv.writer(buff_out, quoting=csv.QUOTE_MINIMAL, dialect='unix')
    for row in reader:
        trail_col = max_col - len(row)
        list_trail_col = [''] * trail_col
        row = row + list_trail_col
        writer.writerow(row)
    buff_out.seek(0)
    buff.close()
    return buff_out.getvalue()


def is_d1_in_d2(d1: T_da, d2: T_da) -> bool:
    # true if all the fields in d1 are in d2.
    # d2 may have additional fields.
    return d1.items() <= d2.items()

    
def assign_col_in_lol_at_icol(icol: int=-1, col_la: Optional[T_la]=None, lol: Optional[T_lola]=None, default:Any='') -> T_lola:
    """ assign col in lol.
        if icol == -1 or > len(lol[0]) then insert at left end.
        use default value if col_la not long enough.
        
    """
    if not lol:
        return []
    # if not col_la:
        # return lol
        
    if icol < 0 or icol >= len(lol[0]):
        # add col to right side.
        return insert_col_in_lol_at_icol(icol, col_la, lol, default)
    
    # overwrite existing col.
    for irow, row_la in enumerate(lol):
        val = safe_get_idx(col_la, irow, default)
        row_la[icol] = val
        
    return lol
        
        
def insert_col_in_lol_at_icol(icol: int=-1, col_la: Optional[T_la]=None, lol: Optional[T_lola]=None, default: Any='') -> T_lola:
    """ insert col in lol.
        if icol == -1 or > len(lol[0]) then insert at right end.
        use default value if col_la not long enough.
        
    """
    if not lol: 
        if not col_la:
            return []
        else:
            lol = [[val] for val in col_la]
            return lol
    
    # if not col_la:
        # return lol
        
    num_cols = len(lol[0])
        
    if icol < 0 or icol >= num_cols:
        icol = num_cols    
    
    for irow, row_la in enumerate(lol):
        val = safe_get_idx(col_la, irow, default)
        row_la.insert(icol, val)
        
        if len(row_la) == num_cols:
            breakpoint()    # perm Should never happen, insert should add a column.
            pass        
        
    return lol
    
    
def insert_row_in_lol_at_irow(irow: int=-1, row_la: Optional[T_la]=None, lol: Optional[T_lola]=None, default: Any='') -> T_lola:
    """ insert row in lol.
        if irow == -1 or irow > len(lol) then append to the bottom.
        use default value if row_la not long enough.
        
    """
    if not lol and row_la:
        return [row_la]
        
    if not row_la and lol:
        return lol
        
    if not row_la and not lol:
        return []
        
    num_cols = len(lol[0])
    
    if len(row_la) < num_cols:
        row_la += ([default] * (num_cols - len(row_la)))
        
    if irow < 0 or irow > len(lol):
        lol.append(row_la)
    else:
        lol.insert(irow, row_la)    
    
    return lol
    
    
def calc_chunk_sizes(num_items: int, max_chunk_size: int) -> List[int]:
    """ given num_items, divide these into equal
        chunks where each is no larger than chunk_size
        return list of chunk sizes.
    """
    if not num_items or not max_chunk_size:
        return []
        
    eff_max_size = max_chunk_size
        
    num_chunks = num_items // eff_max_size
        
    residue = num_items % max_chunk_size
    if residue:
        num_chunks += 1
    chunk_size = num_items // num_chunks
    residue = num_items % num_chunks

    first_list = [chunk_size + 1] * residue
    second_list = [chunk_size] * (num_chunks - residue)

    chunk_sizes_list = first_list + second_list

    return chunk_sizes_list
    
    
def convert_sizes_to_idx_ranges(sizes_list: List[int]) -> List[Tuple[int, int]]:
    """ 
        given sizes list, convert to list of tuples of ranges,
        (start,end) where end is one past the last item included.
    """
    
    ranges_list: List[Tuple[int, int]] = []

    os = 0
    for size in sizes_list:
        range = (os, os+size)
        os += size
        ranges_list.append(range)

    return ranges_list
    

def sort_lol_by_col(lol:T_lola, colidx: int=0, reverse: bool=False, length_priority:bool=True) -> T_lola:

    if length_priority:
        return sorted(lol, key=lambda x: (len(x[colidx]), x[colidx]), reverse=reverse)
    else:
        return sorted(lol, key=operator.itemgetter(colidx), reverse=reverse)


def sort_lol_by_cols(lol: T_lola, colidxs: T_li, reverse: bool = False, length_priority: bool = True) -> T_lola:
    if length_priority:
        return sorted(lol, key=lambda x: [(len(x[idx]), x[idx]) for idx in colidxs], reverse=reverse)
    else:
        return sorted(lol, key=lambda x: [x[idx] for idx in colidxs], reverse=reverse)
    
    
def safe_regex_select(regex:Union[str, bytes], s:str, default:str='', flags=0) -> str:

    regex_str = regex.decode('utf-8') if isinstance(regex, bytes) else regex
    regex_str = regex_str.strip('"')
    
    match = re.search(regex_str, s, flags=flags)
    if match:
        try:
            valstr = match.group(1)   # type: ignore
        except IndexError:
            breakpoint() #perm
            error_beep()
        return valstr.strip()
    else:
        return default
        

def safe_regex_replace(regex: Union[List[Union[str, bytes]], str, bytes], s: str, flags=re.S) -> str:

    """ apply one or more replac regex patterns.
        replace pattern is /find/replace/
        any single character can be used for separators but must be consistent
        
        use to replace, remove, or select
        
        /find/replace/  -- find pattern 'find' and replace with 'replace'
        /find//         -- remove 'find'
        /pre(find)post/prefix\1suffix/     -- select pattern find and create with prefix and suffix
        
        normal regex patterns apply.
    """        

    regex_str = regex.decode('utf-8') if isinstance(regex, bytes) else regex
    regex_str = regex_str.strip('"')
    
    if isinstance(regex_str, str):
        regex_list = [regex_str]   # form a list
    else:
        regex_list = regex_str
        
    result = s
    
    # apply list of regexes in order provided.
        
    for one_replace_regex in regex_list:
        one_replace_regex = one_replace_regex.strip()
        sep_char = one_replace_regex[0]
        if sep_char == one_replace_regex[-1]:
            one_replace_regex = one_replace_regex[1:-1]   # remove them -- strip() removes too many in remove case, /asdff//
        else:
            sts(f"malformed replace regex: '{one_replace_regex}', ignoring", 3)
            continue    # give up on this pattern.
        try:
            findpat, replacepat = re.split(sep_char, one_replace_regex)
        except Exception as err:
            print(err)
            breakpoint() #perm
            pass
        result = re.sub(findpat, replacepat, result, flags=flags)
        
    return result

        

def set_dict_dtypes(
        da:             T_da,                       # dict in the daf array.
        dtypes:         T_dtype_dict,               # dtypes of each item. May contain more than the items in da
        #unflatten:      bool=True,                  # also unflatten any list or dict items.
        # convert_cols:   Optional[Iterable]=None,    # specify which columns should be converted (non-str desired type)
        # select_cols:    Optional[Iterable]=None,    # initialize the columns to be include in the result. 
        ) -> T_da:
    """ set the types in da according to dtype_dict or leave alone if not found in dtype_dict 
        dtype_dict can contain additional items that are not found in the dict da.
        Note, if type is int and val is '', that is considered okay, and is how missing values are noted.
        This function assumes the dict starts with items in str format.
        if unflatten, then convert strings in from JSON format to dict or list if dict or list spec'd.
        
        convert_cols should be set to all columns that are not str that should be converted.
        Depending on use case, not all columns need to be converted even if they are non-str type.
        This improves performance to not repeatedly check the columns in the loop that are not a concern.
        
        select_cols allows dropping columns not needed simultaneously with converting values.
    
    """
    
    if not dtypes:
        return da
    
    for col, val in da.items():
    
        if col not in dtypes:
             continue
            
        da[col] = convert_type_value(da[col], dtypes[col])    
            
    return da
    
            
def convert_type_value(val: any, desired_type: type, unflatten: bool=True):
    """ given a single value, and a desired type, convert it if possible.
        For list and dict type, if str and JSON, convert to list or dict type if unflatten is True.
    """    
            
    if desired_type != bool and (val in ('', None) or val != val):   # null string means None or NAN
        new_val = ''

    elif desired_type == int:
        if val in ('0', '0.0', 'False', 'FALSE'):
            new_val = 0
        elif val in ('1', '1.0', 'True', 'TRUE'):
            new_val = 1
        else:
            try:
                new_val = int(float(val))
            except ValueError:
                new_val = ''
            
    elif desired_type == float:
        try:
            new_val = float(val)
        except ValueError:
            new_val = ''
                
    elif desired_type == bool:
        # null string means None or NAN
        new_val = 0 if val in ('0', '', None, False, 'False') or val != val else 1
            
    elif desired_type in (list, dict) and isinstance(val, str) and unflatten:
        new_val = unflatten_val(val)

    elif desired_type == str:
        if isinstance(val, str):
            new_val = val
        elif isinstance(val, bool):
            new_val = 1 if val else 0
        else:    
            new_val = f"{val}"
        
    elif desired_type == list and isinstance(val, list) or desired_type == dict and isinstance(val, dict):
        # no conversion required.
        new_val = val

    else:
        breakpoint() #perm
        error_beep()
        pass
                 
    return new_val
    
    
def unflatten_val(val: str) -> Union[str, list, dict]:
    """ convert a str into python object.
    
        allows correct JSON or python objects stringified with f"{obj}"
    
    """


    val = val.strip()
    
    if val and isinstance(val, str) and val[0] in ('[', '{', '('):
        obj_val = json_decode(val)  # this returns '' on failure
        
    if obj_val == '':
        obj_val = safe_eval(val)
        
    if obj_val == '':
        return val
        
    return obj_val
    
    
def safe_eval(value: str) -> Optional[Any]:
    """ un-stringify an object without risk of using eval. """
    
    try:
        parsed_value = ast.literal_eval(value)
        return parsed_value
    except (SyntaxError, ValueError):
        return None


def json_decode(json_str: str) -> Any:
    # minimal wrapper around json.loads to handle edge conditions only.
    # good for machine decoding.
    if not json_str:
        return json_str
    try:
        return json.loads(json_str)
    except json.decoder.JSONDecodeError:
        return ''
        

def safe_convert_json_to_obj(json_str: str, json_name: str='') -> Any:

    # convert directly from JSON to object, except deal with single quotes or None's if found.
    # THIS IS FOR HUMAN_ENTERED JSON WHICH MAY HAVE HUMAN-ERRORS.
    # Do not use this for known-good json.

    try:
        result_obj    = json.loads(json_str or "{}")
        return result_obj
    except json.decoder.JSONDecodeError:
        pass

    #logs.sts(f"{logs.prog_loc()} WARN: Single quotes found in {json_name}: '{json_str}' ", 3)       
        
    json_str      = re.sub(r"'", r'"', json_str)
    json_str      = re.sub(r'None', r'null', json_str)
        
    try:
        result_obj = json.loads(json_str)
        return result_obj
    except json.decoder.JSONDecodeError:
        pass
        return json_str
        
        #print(f"Having trouble with this json_str: '{json_str}'")
        #breakpoint() #perm


def validate_json_with_error_details(json_str: str) -> Tuple[bool, str]:  # valid_flag, error_str

    try:
        json.loads(json_str)
        return True, ''
    except json.JSONDecodeError as e:
        error_message = str(e)
        line_number = error_message.split(" line ")[-1].split(",")[0]
        column_number = error_message.split("column ")[-1].split(" ")[0]
        return False, f"Error on line {line_number}, column {column_number}: {error_message}"
    
        
def list_stats(alist:T_la, profile:str) -> T_da:
    """ 
        given a list as a column of a table and analyze that given column and provide stats relevant for that column
        depending on the profile, which can be 
            'index'         -- should be unique, so analyze for repeats and type, skips
            'attrib'        -- generally an enumeration of either numbers or strings
            'file_paths'    -- list of file paths, usually only 1 or 2.
            'scalar'        -- numerical value that should be analyzed using normal stats (max, min, mean, std).
            'localidx'      -- probably starts at 0 or 1 and should be contiguous.
    """
    
    if profile == 'index':
        info_d = list_stats_index(alist)
                
    elif profile == 'attrib':
        info_d = list_stats_attrib(alist)
        
    elif profile == 'file_paths':
        info_d = list_stats_filepaths(alist)
        
    elif profile == 'scalar':
        info_d = list_stats_scalar(alist)
        
    elif profile == 'localidx':
        info_d = list_stats_localidx(alist)
    else:
        raise NotImplementedError(f"profile '{profile}' not supported.")
        
    info_d['profile'] = profile
        
    return info_d    
    
    
def list_stats_index(alist:T_la) -> T_da: # info_dict
    """ 
        evaluate a list as an index column and provide stats info_d considering values only within this single list.
        returns info_d with fields as follows:
            'uniques': list of all unique values
            'within_reps_idxs': indexes in this list of repeated values other than the first one.
            'num_all': count of all values
            'num_uniques': count of all unique values, including those that may also have repeated values later.
            'num_witnin_reps': count of the repeats that should be removed to get all unique values.
            'num_missing': if any values are None, '', of nan.
            'all_ints': bool
            'all_numeric': bool
            'max': int or float
    
    """    
    info_d: T_da = split_dups_list(alist)  # {'uniques_d':uniques_d, 'within_repd': within_reps_loti, 'prior_reps_loti':prior_reps_loti}
    
    info_d['num_all']           = len(alist)
    info_d['uniques']           = [v for v in list(dict.fromkeys(alist)) if not(v is None or v == '' or v != v)]
    info_d['num_uniques']       = len(info_d['uniques'])
    info_d['num_within_reps']   = len(info_d['within_reps_loti'])
    info_d['num_missing']       = sum(1 for v in alist if v is None or v == '' or v != v)   # v != v checks for nan
    info_d['non_missing']       = non_missing    = [v for v in alist if not(v is None or v == '' or v != v)]
    
    info_d['all_ints'] = is_list_allints(non_missing)
        
    if info_d['all_ints']:
        local_ilist = [int(float(clean_numeric_str(str(i)))) for i in non_missing if is_numeric(i)]
        info_d['all_numeric'] = True
        info_d['max'] = max(local_ilist, default=0)
        info_d['min'] = min(local_ilist, default=0)

    else:
        info_d['all_numeric'] = is_list_allnumeric(non_missing)
        local_flist = [float(clean_numeric_str(str(i))) for i in non_missing if is_numeric(i)]
        info_d['max'] = max(local_flist, default=0)
        info_d['min'] = min(local_flist, default=0)
        
    return info_d
    

def list_stats_attrib(alist:T_la) -> T_da:         
    """ 
        evaluate a list as an attribute and provide stats info_d considering values only within this single list.
        returns info_d with fields as follows:
            'num_all': count of all values
            'uniques': list of all unique values
            'num_uniques': count of all unique values, including those that may also have repeated values later.
            'num_missing': if any values are None, '', of nan.
            'val_counts': dict of each value and the count of that value.
    
    """    

    info_d: T_da = {}
    info_d['num_all']       = len(alist)
    info_d['uniques']       = uniques       = [v for v in list(dict.fromkeys(alist)) if not(v is None or v == '' or v != v)]
    info_d['num_uniques']   = len(uniques)
    info_d['num_missing']   = sum(1 for v in alist if v is None or v == '' or v != v)   # v != v checks for nan
    info_d['non_missing']   = non_missing   = [v for v in alist if not(v is None or v == '' or v != v)]
    info_d['all_ints']      = is_list_allints(non_missing)
        
    if info_d['all_ints']:
        info_d['all_numeric'] = True
    else:
        info_d['all_numeric'] = is_list_allnumeric(non_missing)
        
    info_d['all_bools'], info_d['num_true'] = is_list_allbools(non_missing)
    
    info_d['val_counts'] = {}
    for unique in uniques:
        info_d['val_counts'][unique] = non_missing.count(unique)

    return info_d
    

def list_stats_filepaths(alist:T_la) -> T_da: 

    info_d: T_da = {}
    info_d['num_all']           = len(alist)
    info_d['num_missing']       = sum(1 for v in alist if v is None or v == '' or v != v)   # v != v checks for nan
    info_d['non_missing']       = [v for v in alist if not(v is None or v == '' or v != v)]
    
    return info_d
    

def list_stats_scalar(alist:T_la) -> T_da:

    info_d: T_da = {}
    info_d['num_all']           = len(alist)
    info_d['uniques'] = uniques = [v for v in list(dict.fromkeys(alist)) if not(v is None or v == '' or v != v)]
    info_d['num_uniques']       = len(uniques)
    info_d['num_missing']       = sum(1 for v in alist if v is None or v == '' or v != v)   # v != v checks for nan
    nonmissing                  = \
    info_d['non_missing']       = [v for v in alist if not(v is None or v == '' or v != v)]

    info_d['all_ints']          = is_list_allints(nonmissing)
        
    if not info_d['all_ints']:    
        info_d['all_numeric'] = is_list_allnumeric(nonmissing)
    
    if info_d['all_ints']:
        info_d['all_numeric'] = True
        local_ilist     = [int(float(i)) for i in nonmissing]
        info_d['max']   = max(local_ilist, default=0)
        info_d['min']   = min(local_ilist, default=0)
        info_d['mean']  = safe_mean(local_ilist)
        info_d['stdev'] = safe_stdev(local_ilist)

    elif info_d['all_numeric']:
        local_flist     = [float(i) for i in nonmissing]
        info_d['max']   = max(local_flist, default=0)
        info_d['min']   = min(local_flist, default=0)
        info_d['mean']  = safe_mean(local_flist)
        info_d['stdev'] = safe_stdev(local_flist)
  
    return info_d
    

def list_stats_localidx(alist: T_la) -> T_da:

    info_d = {}

    # this test for integers does not allow any sort of float
    for val in alist:
        if not bool(re.search(r'^\d+$', f"{val}")):
            info_d['all_ints'] = False
            return info_d
            
    info_d['all_ints'] = True
    local_alist = [int(float(i)) for i in alist]
    info_d['max'] = max(local_alist, default=0)
    info_d['min'] = min(local_alist, default=0)
    
    info_d['sequential'] = bool(info_d['max'] - info_d['min'] == len(alist) - 1)
    
    return info_d
    

def is_list_allints(alist: T_la) -> bool:
    """ check a list of values to see if all qualify as integers 
        even though they may be formatted as strings.
        allows leading +/- and trailing .0 with any number of 0's.
    """
    
    for val in alist:
        if not bool(re.search(r'^\s*[+\-]?\d+(\.0*)?\s*$', f"{val}")):
            return False
    
    return True
    
def is_list_allnumeric(alist: T_la) -> bool:
    """ check a list of values to see if all qualify as integers 
        even though they may be formatted as strings.
        allows leading +/- and trailing .0 with any number of 0's.
    """
    
    try:
        [float(i) for i in alist]
        return True
    except Exception:
        return False


def is_list_allbools(alist: T_la) -> Tuple[bool, int]: # allbools, num_true
    """ check a list of values to see if all qualify as bools 
        even though they may be formatted as strings.
    """
    
    num_true = 0
    try:
        for val in alist:
            if str2bool(val):
                num_true += 1
        return True, num_true
    except Exception:
    
        return False, 0
    
def profile_ls_to_loti(
        input_ls: T_ls, 
        repeat_startswith='Unnamed', 
        include_cols: Optional[T_ls]=None,
        ignore_cols: Optional[T_ls]=None,
        ) -> T_loti:
    """ 
        Given a list strings, which are typically the header of a column,
        return a list of tuples of integers loti, that describes the
        column offset of a given starting string, and the length of
        any repeats of that string, or strings marked with the repeat_marker.
        
        for example:
        input_ls = ['Alice', 'Bob', 'Unnamed2', 'Charlie', 'David', 'Unnamed5', 'Unnamed6']
        
        will return:
        output_loti = [(0, 1), (1, 2), (3, 1), (4, 3)]
        
        if either ignore cols or include cols are provided, then  do not profile 
        any columns not included but include the offsets in the profile.
        
    """    
    repeat_count = 0
    result_loti: T_loti = []
    unique_idx  = 0
    
    if ignore_cols is None:
        ignore_cols = []
    
    unique_idx = -1
    for idx, colstr in enumerate(input_ls):
    
        if colstr in ignore_cols:
            continue
    
        if unique_idx == -1:
            repeat_count = 1
            unique_idx = idx
                
        elif colstr.startswith(repeat_startswith):
            repeat_count += 1
            
        else:
            result_loti.append( (unique_idx, repeat_count) )
            unique_idx = idx
            repeat_count = 1
    
    if repeat_count:
        result_loti.append( (unique_idx, repeat_count) )
        
    return result_loti            
            

def profile_ls_to_lr(
        input_ls: T_ls, 
        repeat_startswith='Unnamed', 
        include_cols: Optional[T_ls]=None,
        ignore_cols: Optional[T_ls]=None,
        ) -> T_lr:
    """ 
        Given a list strings, which each are typically the header of a column,
        return a list of ranges T_lr, that describes the
        column offset of a given starting string, and the ending offest of
        any repeats of that string, or strings marked with the repeat_marker.
        
        for example:
        input_ls = ['Alice', 'Bob', 'Unnamed2', 'Charlie', 'David', 'Unnamed5', 'Unnamed6']
        
        will return:
        output_loti = [range(0, 1), range(1, 3), range(3, 4), range(4, 7)]
        
        if either ignore cols or include cols are provided, then  do not profile 
        any columns not included but respect the offsets in the profile.
        These tend to be leading columns that need not be profiled because they
        are known to never be repeated.
        
    """    
    repeat_count = 0
    result_lr: T_lr = []
    unique_idx  = 0
    
    if ignore_cols is None:
        ignore_cols = []
    
    unique_idx = -1
    for idx, colstr in enumerate(input_ls):
    
        if colstr in ignore_cols:
            continue
    
        if unique_idx == -1:
            repeat_count = 1
            unique_idx = idx
                
        elif colstr.startswith(repeat_startswith):
            repeat_count += 1
            
        else:
            result_lr.append( range(unique_idx, unique_idx + repeat_count) )
            unique_idx = idx
            repeat_count = 1
    
    if repeat_count:
        result_lr.append( range(unique_idx, unique_idx + repeat_count) )
        
    return result_lr
            

def reduce_lol_cols(lol: T_lola, max_cols:int=10, divider_str: str='...') -> T_lola:
    """ if input lol is over max_cols, display only max_cols//2 first_cols, a divider col, then same number of last_cols 
    
        does not alter lol
    
    """
    
    if not max_cols or not lol:
        return lol
        
    lol = equal_cols_lol(lol)
    
    num_cols = len(lol[0])

    if num_cols <= max_cols:
        return lol
        
    first_col_num = math.ceil(max_cols/2)
    last_col_num  = max_cols - first_col_num
    
    result_lol = [row_la[:first_col_num] + [divider_str] + row_la[-last_col_num:]
                                for row_la in lol]   
    return result_lol
        

def s3path_to_url(s3path: str) -> str:
    """ construct url from s3path:
    
    s3path = s3://{bucket}/{key}
    url    = https://{bucket}.s3.amazonaws.com/{key}
             https://us-east-1-audit-engine-jobs.s3.amazonaws.com/US/FL/US_FL_Volusia_Primary_20200317/marks/exc_marks.txt
    """
    
    import urllib.parse
    
    s3dict = parse_s3path(s3path)
    bucket = s3dict['bucket']
    key_url = urllib.parse.quote_plus(s3dict['key'], safe='/')
    
    return f"https://{bucket}.s3.amazonaws.com/{key_url}"
    
    
def parse_s3path(s3path: str) -> T_ds:
    """ the s3 path we use is the same as what is used by s3 console.
        format is:
            s3://<bucket>/<prefix>/<basename>
            
        where <prefix>/<basename> is the key.
        note that components contain trailing / separators.
        but the bucket does not.
    """
    s3dict = {}
    match = re.search(r'(.*://)([^/]+)/(.*/)(.*)$', s3path)
    
    if match:
        try:
            s3dict['protocol']      = match[1]      # type: ignore
            s3dict['bucket']        = match[2]      # type: ignore
            s3dict['prefix']        = match[3]      # type: ignore
            s3dict['basename']      = match[4]      # type: ignore
        except Exception:
            s3dict['protocol']      = ''
            s3dict['bucket']        = ''
            s3dict['prefix']        = ''
            s3dict['basename']      = ''
            
        s3dict['key']           = s3dict['prefix'] + s3dict['basename']
        s3dict['dirpath']       = s3dict['protocol'] + s3dict['bucket'] + '/' + s3dict['prefix']
        s3dict['prefix_parts']  = s3dict['prefix'].split('/')       # like: ['US', 'WI', 'US_WI_Dane_General_20201103', ''] Note last field.
        s3dict['dirname']       = s3dict['prefix_parts'][-2] if s3dict['prefix_parts'] else ''
        # 
    
    if (not match or
            s3dict['protocol'] != 's3://' or
            not s3dict['bucket'] or
            not s3dict['key']):
    
        sts(f"{prog_loc()} s3_path format invalid: {s3path}", 3)
        raise RuntimeError
    return s3dict
    

def transpose_lol(lol: T_lola) -> T_lola:

    # given a list of row list, rearrange into list of column lists.
    # all lines must have same number of columns or it raises RuntimeError
    # (ragged right not allowed)

    transposed_lol = [list(i) for i in zip(*lol)]
    
    if lol and transposed_lol and max([len(lst) for lst in lol]) != len(transposed_lol):
        raise RuntimeError (f"{prog_loc()} transpose_lol encountered ragged lol with record lengths: {[len(lst) for lst in lol]}")

    return transposed_lol


def safe_get_idx(lst: Optional[List[Any]], idx: int, default: Optional[Any]=None) -> Any:
    """ similar to .get for dicts.
        attempt to access list item and if it does not exist return default
        also uses default if list entry is None.
    """
    if lst is None:
        return default    
    
    try:
        val = lst[idx]
        if val is None:
            return default
        return val
    except (IndexError, TypeError):
        return default
    

def safe_max(listlike):
    # note! Using try/except in to guard for the length of list is not specific enough. 
    #   We still need failure under other conditions.

    return max(listlike, default=0)
     

def shorten_str_keeping_ends(string: str, limit: int) -> str:
    """ combine multiple lines into a single line, then 
        shorten the line if needed to no more than limit chars
        by removing chars from the middle.
    """

    single_line = re.sub(r'[\n\r]', ' ', string, flags=re.S).strip()
    if len(single_line) > limit:
        length = (limit // 2) - 1
        single_line = single_line[0:length]+'..'+single_line[-length:]
    return single_line


def smart_fmt(val: Union[str, int, float, None]) -> str:
    # provide reasonable formatting for human consumption
    # if val_str a number: 
    #   if > 1000, use comma formatting.
    #   if val_str has decimal:
    #       if val > 1, include at most one decimal digit
    #       if val > 0.1, include at most two decimal digits
    #       if val > 0.01, include at most three decimal digits
    #       if val > 0.001 include at most four decimal digits
    # if val is a string, and starts with [
    #   split strings by ","
    
    if val is None:
        return ''
        
    if isinstance(val, str) and val.startswith('[') and val.endswith(']'):
        val_str = val.strip('[]')
        val_str = '<br>'.join(val_str.split(','))
        return val_str
    
    val_str = f"{val}"
    if bool(re.search(r'^[+\-\d]+$', val_str)):          # values is only digits, +/- (no decimal)
        try:
            return f"{int(val):,}"                           #   use comma format
        except Exception:
            # probably not really a number after all.
            return val_str
            
    elif bool(re.search(r'^[+\-]?\d+\.?\d*(e\-\d+)?$', val_str)):     # has only digits and decimal, +/-, possibly e-xx
        val_f = float(val)
        abs_val_f = abs(val_f)
        if abs_val_f > 100:
            return f"{int(round(val_f)):,}"
        elif abs_val_f > 1:
            return f"{val_f:.1f}"
        elif abs_val_f > 0.1:     
            return f"{val_f:.2f}"
        elif abs_val_f > 0.01:     
            return f"{val_f:.3f}"
        elif abs_val_f > 0.001:     
            return f"{val_f:.4f}"
        elif abs_val_f > 0.0001:     
            return f"{val_f:.5f}"
        elif abs_val_f > 0.00001:     
            return f"{val_f:.6f}"
        else:
            return f"{0.0:.1f}"

    return val_str
    

def str2bool(value: Optional[Any]) -> bool:
    """Parses string to boolean value."""
    if value is None or value == '':
        return False
    if isinstance(value, (bool, int)):
        return bool(value)
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise ValueError(f"Boolean value expected, value:{value}")


def safe_del_key(da: Dict[Any, Any], k:Any): 
    """ delete a key from da if possible. modifies da directly and handles error.
        also returns da convenience value.
    """
    
    try:
        del da[k]
    except KeyError:
        pass
    return da    
    
    
def dod_to_lod(dod: T_doda, keyfield: str='rowkey') -> T_loda:
    """ given a dod, downconvert to lod by 
        adding the dod key as keyfield to each dict, if required.
            will add in the first position
        and creating lod.
    """

    if not isinstance(dod, dict):
        raise RuntimeError("dod_to_lod requires a dict parameter")
        
    if not keyfield:
        return list(dod.values())
    
    lod = []
    for key, d in dod.items():
        if keyfield and keyfield not in d:
            # insert in the first position.
            d = {keyfield: key, **d}
        lod.append(d)
        
    return lod


def lod_to_dod(lod: T_loda, 
        keyfield:           str='rowkey', 
        remove_keyfield:    bool=True,
        ) -> T_doda:
    """ given a lod with common fields, convert to dod,
        where the outer dict key is the field rowkey in the original lod.
        rowkey defaults to 'rowkey'
        
        if remove_rowkey is True (default) remove rowkey item 
        from the inner dictionary. Otherwise, leave each dict alone.
    """
    dod = {}
    for da in lod:
        dod_key = da[keyfield]
        if remove_keyfield:
            del da[keyfield]
        dod[dod_key] = da
    return dod        
    

def safe_min(listlike: List[Any]) -> Any:
    # note! Using try/except in to guard for the length of list is not specific enough. 
    #   We still need failure under other conditions.

    if len(listlike) < 1:
        return 0
    return min(listlike)
    

def safe_stdev(listlike):
    # note! Using try/except in to guard for the length of list is not specific enough. 
    #   We still need failure under other conditions.
    #   statistics library provides only statisticsError which is not specific enough.

    if len(listlike) < 2:
        return 0
    return statistics.stdev(listlike)
    
    
def safe_mean(listlike):
    # note! Using try/except in to guard for the length of list is not specific enough. 
    #   We still need failure under other conditions.
    #   statistics library provides only statisticsError which is not specific enough.

    if len(listlike) < 1:
        return 0
    return statistics.mean(listlike)    

     
def beep(freq: int=1080, ms: int=500):

    if not is_linux():
        try:
            import winsound
        except ImportError:
            import os
            os.system('beep -f %s -l %s' % (freq, ms))
        else:
            winsound.Beep(freq, ms)
            
    
def error_beep():
    beep()
    

def notice_beep(freq: int=1080):
    beep(freq=freq, ms=250)
    beep(freq=freq, ms=250)


def sts(string: str, verboselevel: int=0, end: str='\n', enable: bool=True) -> str:
    """ Append string to logfile report.
        Also return the string so an interal version can be maintained
        for other reporting.
        The standard logger function could be used but we are interested
        in maintaining a log linked with each phase of the process.
        returns the string.
    """
    
    verbose_level = 3

    if string is None or not enable: return ''

    log_str = f"{get_datetime_str()}: {string}"

    if verboselevel >= verbose_level:
        print(log_str, end=end, flush=True)
    return string+end


def get_datetime_str() -> str:

    return f"{datetime.datetime.now()}"
    #return datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    

def split_dups_list(
        alist: Union[T_la, dict], 
        prior_unique_d: Optional[Union[dict, list]]=None,
        list_idx: int=0,         # used if there are multiple lists
        ) -> T_da:  # {'uniques_d':uniques_d, 'within_reps_loti': within_reps_loti, 'prior_reps_loti':prior_reps_loti}

    """ given list of hashable items, 
        return uniques_d, within_dups, prior_dups
        
        Note: if a val is found in prior_unique_d, then it will be removed from
                consideration for a within_reps entry.
        
        order not changed. dups may have more than one duplicate.
        
        Note: this function needs to use indices.
    """
    if prior_unique_d is None:
        prior_unique_d = {}
        
    # allow prior_unique_d to be a list if that is available.
    if isinstance(prior_unique_d, list):
        prior_unique_d = dict.fromkeys(prior_unique_d)

    if isinstance(alist, dict):
        alist = list(alist.keys())
    
    uniques_d: T_da = {}
    within_reps_loti = []
    prior_reps_loti = []
        
    for idx, a in enumerate(alist):
        if a in prior_unique_d:
            prior_reps_loti.append( (list_idx, idx) )
        elif a in uniques_d:
            within_reps_loti.append( (list_idx, idx) )
        else:
             uniques_d[a] = None
    
    return {'uniques_d':uniques_d, 'within_reps_loti': within_reps_loti, 'prior_reps_loti':prior_reps_loti}


def clean_numeric_str(valstr: str) -> str:
    """ Remove commas, dollars, percents from str. 
    """
    return valstr.replace(',', '').replace('$', '').replace('%', '')
    
    
def is_numeric(val: Any) -> bool:
    """ Test if a string could be treated as numeric after ,$% removed
    """
       
    if isinstance(val, (int, float)):
        return True
    if isinstance(val, str):
        # Remove any commas and dollar signs
        cleaned_str = clean_numeric_str(val)

        # Check if the cleaned string is a valid numeric format
        return re.match(r'^[+-]?\d+(\.\d+)?$', cleaned_str) is not None
    return False
    
    
# use this function to print the file and line number in any string to be printed.

def prog_loc() -> str:
    import inspect
    frame = inspect.currentframe()
    if frame is None:
        return ''
    try:
        frameinfo = inspect.getframeinfo(frame.f_back)          # type: ignore
        filename = re.split(r'[\\/]', frameinfo.filename)[-1]
        linenumber = frameinfo.lineno
    finally:
        del frame
    return f"[{filename}:{linenumber}]"
    
    
def buff_csv_to_lol(
        buff: Union[bytes, str], 
        user_format: bool=False, 
        sep=',', 
        include_cols: Optional[T_ls]=None, 
        dtypes: Optional[T_dtype_dict]=None,
        raw: bool=False,
        ) -> T_lola:
    """
    Convert CSV data in a buffer (bytes or string) to a lol data type.

    Args:
        buff (Union[bytes, str]): The CSV data as bytes or string.
        user_format (bool): Whether to preprocess the CSV data (remove comments and blank lines).
        sep (str): The separator used in the CSV data.

    Returns:
        lola: all lines in the file as lol. May be ragged.
    """

    if sep is None:
        sep = ','
        
    if isinstance(buff, bytes):
        buff = buff.decode("utf-8")
        
    if not raw:
        # @@TODO grab the first few lines and get the header section.
        pass
    
        
    if user_format:
        buff = preprocess_csv_buff(buff)  # remove comments, blank lines
        
    sio = io.StringIO(buff)
    csv_reader = csv.reader(sio, delimiter=sep, quoting=csv.QUOTE_MINIMAL)

    data_lol = [row for row in csv_reader]

    return data_lol
    

def preprocess_csv_buff(buff: Union[bytes, str]) -> str:
    """ given a buffer which is csv file read without conversion,
        perform preprocessing to remove comments and blank lines.
        controls in pandas csv do not work very well, such as when
        there is a comma in a comment line.
    """
    if isinstance(buff, bytes):
        buff = buff.decode("utf-8")

    lines = buff.splitlines()
    lines = [line for line in lines if line and not bool(re.search(r'^"?#', line)) and not bool(re.search(r'^,+$', line))]
    buff = '\n'.join(lines)
    
    return buff
    

def write_buff_to_fp(buff: T_buff, file_path: str, fmt='.csv', rtype='.csv', local_mirror: bool=False, if_unmodified: bool=False) -> str: # file_path

    if buff:
        #--- write buffer based on path
        num_written = 0
        s3path = None   # return value when local mirror is enabled. We always want s3path, not local path.
    
        if file_path.startswith('s3'):
            # make sure the encoding matches what might be saved using the fp.write function
            from . import s3utils
            
            if rtype not in ['binary', 'image']:
                buff = cast(str, buff)
                s3buff = buff.encode('utf-8')
            else: 
                s3buff = cast(bytes, buff)    
                
            if if_unmodified:
                num_written = s3utils.write_buff_to_s3path(file_path, s3buff, fmt) #, backup_flag=backup_flag)
            else:
                num_written = s3utils.write_buff_to_s3path_if_modified(file_path, s3buff, fmt) #, backup_flag=backup_flag)
            
            if num_written:
                sts(f"Saved {len(s3buff):,} bytes to {file_path}")
                                
        if not file_path.startswith('s3'):
            file_path = path_sep_per_os(file_path)
            # if backup_flag:
                # backup_path = utils.create_backup_path(file_path)
                # shutil.copyfile(file_path, backup_path)
            if rtype in ['binary', 'image']:
                buff = cast(bytes, buff)
                try:
                    with open(file_path, mode='wb') as file:
                        file.write(buff)
                except Exception:
                    error_beep()
                    breakpoint() #perm
                    pass
            else:
                buff = cast(str, buff)
                with open(file_path, mode='wt', newline='', encoding="utf-8") as file:
                    file.write(buff)
                                        
            sts(f"Saved {len(buff):,} bytes to {file_path}")
    else:
        sts(f"## Data item not written to {s3path or file_path}")
    return s3path or file_path


def path_sep_per_os(path: str, sep: Optional[str]=None) -> str:
    """ based on os.sep setting, correct path to those separators, 
        assuming no / or \\ characters exist in the path otherwise.
    """
    if sep is None:
        sep = os.sep
    if sep == '/':
        return re.sub(r'\\', r'/', path)
    else:
        return re.sub(r'/', r'\\', path)


def is_list_of_type(test_item: Any, of_type: Type) -> bool:

    # test if test_item is T_ls and it is not empty.

    if test_item and isinstance(test_item, list) and isinstance(test_item[0], of_type):
        return True
        

def len_slice(slice_obj: slice, tot_len: int=0):
    """ Calculate the length of the slice, including step 
    
        tot_len should be the total length of the sliced object.    
    """
    
    if slice_obj == slice(None, None, None):
        return 0
        
    start, stop, step = slice_obj.start or 0, slice_obj.stop or tot_len, slice_obj.step or 1
    
    try:
        len_int = (stop - start + step - 1) // step
    except Exception:
        breakpoint() #perm
        
    return len_int
    

def len_rowcol_spec(ispec: Union[slice, int, T_li, None], tot_len: int) -> int:
    """ return the length of a slice, int, li. If None, then len = 0 
    
        returns -1 if the length is not terminated
    """
    
    if isinstance(ispec, slice):
        return len_slice(ispec, tot_len)
    elif isinstance(ispec, int):
        return 1
    elif isinstance(ispec, list):
        return len(ispec)
    else:
        return 0
        
def slice_to_list(slice_obj) -> list:
    return [i for i in range(slice_obj.start or 0, slice_obj.stop or float('inf'), slice_obj.step or 1)]

def slice_to_range(slice_obj, length):
    if slice_obj == slice(None, None, None):
        return range(length)
    else:
        if slice_obj.stop is None:
            stop = length
        else:
            stop = min(slice_obj.stop, length)
        try:
            return range(slice_obj.start or 0, stop, slice_obj.step or 1)
        except Exception:
            breakpoint()    # perm
            pass


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
    

def _generate_spreadsheet_column_names_list(num_cols: int) -> T_ls:
    """ generate a full list of column names for the num_columns specified 
    """

    return [_calculate_single_column_name(i) for i in range(num_cols)]


def _sanitize_cols(cols: T_ls, unnamed_prefix='Unnamed') -> list:
    """ make sure there are no blanks and columns are unique.
        if missing, substitute with {unnamed_prefix}{col_idx}
        if duplicated, substitute with prior_name_{col_idx}
    """
    
    if cols:
        # first make sure all columns have names.
        cols = [col if col else f"{unnamed_prefix}{idx}" for idx, col in enumerate(cols)] 
            
        # next make sure they are all unique    
        col_hd = {}
        for idx, col in enumerate(cols):
            if col not in col_hd:
                col_hd[col] = idx
            else:
                # if not unique, add _NNN after the name.
                col_hd[f"{col}_{idx}"] = idx
        return list(col_hd.keys())


def dict_with_index(iterable) -> Dict[Any, int]:

    return dict(with_index(iterable))
    
            
def with_index(iterable):
    """Like enumerate, but (val, i) tuples instead of (i, val)."""
    for i, item in enumerate(iterable):
        yield (item, i)

            
def invert_dol_to_dict(input_dol:dict) -> dict:
    """ given a dict of lists where no element in any is seen twice,
        create a dict, where the list elements are the key and the
        value is the prior key. This allows reverse lookup of key
        based on values in the list. If some dict lists have duplicates,
        the last item will dominate.
    """
    result_dict: Dict[Any, Any] = {}

    for key, lst in input_dol.items():
        for val in lst:
            result_dict[val] = key

    return result_dict


def min_max_cols_lol(lol: T_lola, start: Optional[int]=None, limit: Optional[int]=None) -> Tuple[int, int]:

    max_cols = 0
    min_cols = None

    for la in lol[start:limit]:
        colnum = len(la)
        if colnum > max_cols:
            max_cols = colnum
        if min_cols is None or colnum < min_cols:
            min_cols = colnum
            
    return min_cols, max_cols
    

def equal_cols_lol(lol: T_lola, limit: int=10, check_all:bool=False) -> T_lola:
    """ Make lol have equal number of columns throughout. 
        Appends columns of '' on the right end.
        Mutates in place.
        Max length usually happens at the start of the file.
        
        1. first searches the lol array up to limit to find a first guess
            as to the max length.
        2. Uses, this, but if a longer ling is encountered, then 
        
    """
    min_cols, max_cols = min_max_cols_lol(lol, limit=limit)
            
    if limit is None and max_cols == min_cols:
        return lol
        
    if limit and not check_all and max_cols == min_cols:
        return lol

    for la in lol:
        colnum = len(la)
        if colnum < max_cols:
            la += [''] * (max_cols - colnum)
            
        if colnum > max_cols:
            # our estimate of the max col was incorrect.
            break
    else: 
        return lol

    # we found a longer line.
    # recursively call this function with no limit, causes full inspection.
    return equal_cols_lol(lol=lol, limit=None)




            
        