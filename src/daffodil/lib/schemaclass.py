"""
schemaclass.py

Schema support for Daffodil.

Defines the @schemaclass decorator used to describe tabular schemas
(column names, intended types, and default values) without imposing
runtime object semantics.

A schemaclass is not intended to be instantiated. It is used only as a
schema descriptor by consuming code.
"""

import copy
import typing
from typing import Dict, Any, List, TypeVar, Union   # noqa: F401
from ..keyedlist import KeyedList
from . import daf_utils

T_da =  Dict[str, Any]

T = TypeVar("T")

def schemaclass(cls: type[T]) -> type[T]:
    """
    Decorator marking a class as a Daffodil schema.

    The decorated class is used only for its structure:
    - __annotations__ define column names and intended types
    - class attributes define default values

    The class is not intended to be instantiated.
    """

    # ---- basic validation -------------------------------------------------

    ann = getattr(cls, "__annotations__", None)
    if not isinstance(ann, dict) or not ann:
        raise TypeError(
            f"{cls.__name__} must define at least one annotated field"
        )

    for name in ann:
        if not hasattr(cls, name):
            raise TypeError(
                f"{cls.__name__}.{name} must define a default value"
            )

    # ---- prevent instantiation -------------------------------------------

    def _no_init(*args, **kwargs):
        raise TypeError(
            f"{cls.__name__} is a schemaclass and cannot be instantiated"
        )

    cls.__init__ = _no_init

    # ---- helper methods ---------------------------------------------------

    @classmethod
    def default_record(cls, **kwargs: Any) -> T_da:
        """
        Return a new record dict initialized from schema defaults.

        A new dictionary is returned on each call.

        Scalar and immutable defaults are reused directly.
        Mutable defaults (list, dict, set) are shallow-copied so each
        record receives an independent object.

        No validation or type conversion is performed.

        should include 'astype' parameter to allow return
            of dict or T_klist

        """

        # Debug-only validation
        if __debug__:
            cls.validate_keys_debug(kwargs)

        rec: T_da = {}
        
        for name in cls.__annotations__:
            if name in kwargs:
                rec[name] = kwargs[name]
            else:
                val = getattr(cls, name)
                if isinstance(val, (list, dict, set)):
                    rec[name] = copy.copy(val)
                else:
                    rec[name] = val
        return rec


    @classmethod
    def record_from(cls, src: Union[T_da, KeyedList]) -> T_da:
        """
        Create a schema-compatible record from a source mapping.

        The source may be either a conventional dictionary or a KeyedList.
        Only fields defined in the schema are copied. Missing fields retain
        the default values provided by `default_record()`.

        Values copied from `src` are converted according to the schema
        dtype specification using `daf_utils.convert_type_value()`.

        Parameters
        ----------
        src : Union[Dict[str, Any], KeyedList]
            Source record containing field values.

        Returns
        -------
        Dict[str, Any]
            A dictionary representing a normalized record that conforms
            to the schema definition.
        """

        rec: T_da = cls.default_record()
        dtypes = cls.get_dtypes_dict(use_origins=True)

        for k, dtype in dtypes.items():
            if k in src:
                rec[k] = daf_utils.convert_type_value(src[k], dtype)

        return rec

        
    @classmethod
    def get_dtypes_dict(cls, *, use_origins: bool = False) -> Dict[str, type]:
        """
        Return a dtypes dictionary mapping column names to intended types.

        Types describe non-empty values only.
        """
        if not use_origins:
            return dict(cls.__annotations__)
            
        dtypes_dict = {
            name: (typing.get_origin(tp) or tp)
                for name, tp in cls.__annotations__.items()
            }
            
        return dtypes_dict
   
    
    @classmethod
    def get_columns(cls) -> List[str]:
        """
        Return a list of column names.

        Types describe non-empty values only.
        """

        columns_ls: List[str] = list(cls.__annotations__.keys())
        return columns_ls
   
    
    @staticmethod
    def get_pandas_dtypes_from_schema(schema):
        dtypes = {}

        for name, tp in schema.__annotations__.items():
            origin = typing.get_origin(tp) or tp

            if origin in (list, dict):
                dtypes[name] = "object"
            elif origin is int:
                dtypes[name] = "Int64"   # nullable int
            elif origin is str:
                dtypes[name] = "string"  # or object
            elif origin is float:
                dtypes[name] = "Float64"
            else:
                dtypes[name] = "object"

        return dtypes


    @classmethod
    def validate_keys_debug(cls, da: T_da) -> None:
        if not __debug__:
            return

        invalid = [k for k in da if k not in cls.__annotations__]

        assert not invalid, f"{cls.__name__}: invalid keys {invalid}"

    
    # ---- attach helpers to class -----------------------------------------

    cls.default_record  = default_record
    cls.get_dtypes_dict = get_dtypes_dict
    cls.get_columns     = get_columns
    cls.get_pandas_dtypes_from_schema = get_pandas_dtypes_from_schema
    cls.validate_keys_debug = validate_keys_debug
    cls.record_from     = record_from
    
    # ---- marker attribute -------------------------------------------------

    cls.__is_schemaclass__ = True

    return cls
