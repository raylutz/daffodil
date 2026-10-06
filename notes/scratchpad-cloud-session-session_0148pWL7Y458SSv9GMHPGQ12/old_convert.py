from typing import *
from daffodil.lib.daf_utils import unflatten_val
T=TypeVar('T')
def convert_type_value(val: Any, desired_type: Type[T], unflatten: bool=True) -> Any:
    """ given a single value, and a desired type, convert it if possible.
        For list and dict type, if str and JSON, convert to list or dict type if unflatten is True.
        At this point, desired type must be the origin of any type definitions, such as int, str, float, list, dict, set, tuple.
        
        If there is no value, value retured is ''
        If type is bool, value is 0 or 1 (integers), but source can be '0', '1', '', None, True, False
    """  

    if desired_type is not bool and (val in ('', None) or val != val):   # null string means None or NAN
        new_val: Any = ''

    # intentionally use == here to allow any type of int.
    # if use 'is' (as recommended by linter) it will exclude int32, int64 and other variants.

    elif desired_type == int:                       # noqa: E721 
        if val in ('0', '0.0', 'False', 'FALSE'):
            new_val = 0
        elif val in ('1', '1.0', 'True', 'TRUE'):
            new_val = 1
        else:
            try:
                new_val = int(float(val))
            except ValueError:
                new_val = ''
            
    elif desired_type is float:
        try:
            new_val = float(val)
        except ValueError:
            new_val = ''
                
    elif desired_type is bool:
        # null string means None or NAN
        new_val = 0 if val in ('0', '', None, False, 'False', 'FALSE') or val != val else 1
            
    elif desired_type in (list, dict) and isinstance(val, str) and unflatten:
        new_val = unflatten_val(val)

    elif desired_type is str:
        if isinstance(val, str):
            new_val = val
        elif isinstance(val, bool):
            new_val = int(val)
        else:    
            new_val = f"{val}"
        
    elif desired_type is list and isinstance(val, list) or \
         desired_type is dict and isinstance(val, dict):
        # no conversion required.
        new_val = val

    else:
        raise TypeError(f"convert_type_value(): cannot convert {type(val).__name__} value to {desired_type}")

    return new_val
    
    
