# daf_types.py

from typing import List, Dict, Any, Tuple, Type, Optional, Union, Set, KeysView #, TYPE_CHECKING, Callable, cast

import pandas as pd                 # type: ignore
import numpy  as np                 # type: ignore
import datetime

T_df   = pd.DataFrame

# general
T_si  = Union[str, int]
T_sib = Union[str, int, bool]
T_da  = Dict[str, Any]
T_ida = Dict[int, Any]
T_ds  = Dict[str, str]
T_ids = Dict[int, str]
T_idi = Dict[int, int]
T_di  = Dict[str, int]
T_db  = Dict[str, bool]
T_dsi = Dict[str, T_si]
T_dn  = Dict[str, None]
T_ls  = List[str]
T_li  = List[int]
T_lr  = List[range]
T_rli = Union[range, T_li]
T_lsi = List[T_si]
T_la  = List[Any]
T_lf  = List[float]
T_lb  = List[bool]
T_lt  = List[Type]
T_lor = List[range]
T_ss  = Set[str]
T_sa  = Set[Any]
T_dsn = Dict[str, Optional[str]]
T_ta  = Tuple[Any, ...]
T_ts  = Tuple[str, ...]
T_ti  = Tuple[int, ...]

T_num = Union[int, float]

T_dodf = Dict[str, pd.DataFrame]
T_idodf = Dict[int, pd.DataFrame]
T_lodf = List[pd.DataFrame]
T_dododf = Dict[str, T_dodf]

T_doda = Dict[str, T_da]
T_dooda  = Dict[str, Optional[T_da]]
T_idoda = Dict[int, T_da]
T_dodb = Dict[str, T_db]
T_dols = Dict[str, T_ls]
T_dola = Dict[str, T_la]
T_doli = Dict[str, T_li]
T_doti = Dict[str, T_ti]
T_dodi = Dict[str, T_di]
T_dods = Dict[str, T_ds]
T_dodsi = Dict[str, T_dsi]
T_dosa = Dict[str, T_sa]
T_doss = Dict[str, T_ss]
T_dota = Dict[str, T_ta]
T_dododa = Dict[str, T_doda]
T_dodoli = Dict[str, T_dodi]
T_dododsi = Dict[str, T_dodsi]
T_dolsi = Dict[str, T_lsi]
T_dodododa = Dict[str, T_dododa]
T_dododi = Dict[str,T_dodi]

T_kva = KeysView[Union[str, int, tuple]]  # KeysView for keys that can be str, int, or tuple

T_lods = List[T_ds]
T_loda = List[T_da]             # lod is 12x larger than df, but is faster for appends. 3x larger than hdlola
T_lodi = List[T_di]
T_lodsi = List[T_dsi]
T_lols = List[T_ls]
T_loli = List[T_li]
T_lolf = List[T_lf]
T_lola = List[T_la]
T_lota = List[T_ta]
T_loti = List[T_ti]
T_loloda = List[T_loda]
T_lolods = List[T_lods]
T_doloda = Dict[str, T_loda]
T_dolodi = Dict[str, T_lodi]
T_lodoloda = List[T_doloda]
T_lodolodi = List[T_dolodi]     # timing_marks
T_lodola = List[T_dola]
T_lododa = List[T_doda]
T_lololi = List[T_loli]             # tl, br, corners for each side
T_lodoli = List[T_doli]             # snapgrid for one sheet.
T_dolodoli = Dict[str, T_lodoli]    # style snapgrid

#T_hdlola = Tuple[T_di, T_lola]   # deprecated. use Daf. header dict + lol is equivalent to lod but 1/3 the size of lod
#T_hdlota = Tuple[T_di, T_lota]   # deprecated. use Daf. 
#T_hllola = Tuple[T_ls, T_lola]   # deprecated. use Daf. header list + lol allows duplicates in the header.
T_npa   = np.ndarray
#T_hdnpa =  Tuple[T_di, T_npa]    # deprecated. use Daf. header dict + npa is equivalent to lod but 3% the size of lod, but requires same dtype throughout.
T_donpa = Dict[str, T_npa]       # useful representation of columns in Daf array to allow for numpy operations.

T_dtype_dict = Dict[str, Type]
T_dtype = Union[Type, Dict[str, Type]]
T_coldef = List[Tuple[str, Type[Any], str]]
T_region = T_di

T_df_or_lod = Union[T_df, T_loda]   # type: ignore

# pipeline specific
T_dep_dict = Dict[str, Union[str, bool, int]]               # expresses one dependency
    #               {'path': s3path_to_resource,
    #                'origin': bool, If true, dependency does not need to be checked.
    #                'etag': etag if the file exists.
    #                'size': (int) size in bytes.
    #                'timestamp': timestamp in string format.
    #                'preserve': bool
    #                'title': str (outs only) reports included in the final report.
    #               }, ... ]
T_dep_lod = List[T_dep_dict]                                # deps and outs are a list of those dep.
T_funcdict = Dict[str, Any]

T_stagedict = Dict[str, Union[T_dep_lod, str, T_funcdict]]  # {'deps': [{ }], 'outs': [{ }, ], 'status':str, 'type': str, 'funcdict':{}}
T_stages_dod = Dict[str, T_stagedict]                       # {stagename: stagedict, stagename2: stagedict2, ...}
T_pipeline = Dict[str, Union[T_stages_dod, T_lods]]         # {'stages': T_stages_dod, 'reports': T_lods}      

# when we update the pipeline, we gather the metadata from s3dirpaths:
# use T_dodsi
#T_metadata_dod = Dict[str, Optional[Dict[str, Union[str, int]]]]      # {fullpath:{'etag':etag, 'timestamp':timestamp, 'size':size}, ...  }

# archives

T_all_archives_info = Dict[str, Union[T_loda, int, T_ls]]

T_dateobj = datetime.datetime

# T_side_snapgrid_doli = Dict[str, T_li]

# use T_lodoli
# T_sheet_snapgrid_lodoli = List[T_side_snapgrid_doli]        
        # # [
        # # {"lft": [64, 106, ... 3376], "top": [27, 108, ...  1592]}, 
        # # {"lft": [66, 107, ... 3375], "top": [112, 176, ... 1593]}]

# use T_dolodoli
# T_style_to_snapgrid_dolodoli = Dict[str, T_lodoli]


T_stdresults_dodoildi = Dict[str, Dict[str, Union[int, T_ls, T_di]]]
T_combined_stdresults_dododoildi = Dict[str, Dict[str, Dict[str, Union[int, T_ls, T_di]]]]

T_parsed_cvr = Any

T_image = np.ndarray
T_shape = Tuple[int, int]  # rows, cols

T_PDF_style_info_doslodsi = Dict[str, Union[str, T_lodsi]]

T_point_xy = List[int]
T_corner_points_one_p = Dict[str, T_point_xy]
T_corner_points = List[Optional[T_corner_points_one_p]]    # each page, options dict of x,y points.

T_buff = Union[bytes, str]



