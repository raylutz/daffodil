# daf_schema.py

import typing
from typing import List, Dict, Any, Tuple, Optional, TypeVar, Union, cast, Type, Callable # noqa: F401
from daffodil.lib.daf_types import T_ls, T_lola, T_da, T_li, T_cs, T_ca, T_ma # noqa: F401

import copy
from ..keyedlist import KeyedList
from . import daf_utils

# from daffodil.daf import Daf 

def _apply_schema(
        self,
        schema=None,
        ): # -> 'Daf':
    """
    Apply schema metadata to this Daf instance.

    Supports:
        - schemaclass
        - schema_daf

    Semantics:

        - attaches schema to self.schema
        - sets cols if not already defined
        - sets dtypes if not already defined
        - sets keyfield if not already defined

    For schema_daf:

        Expected columns may include:

            Name
            dtype
            Default
            Type
            Values
            Description
            Attributes

        Only:
            Name
            dtype

        are currently interpreted structurally here.
    
    Other fields are defined to help with user interface for entering and editing records.

        Type specifies the UI to be used.
        
        Type        Description

        checkbox    One or more checkboxes. 
                        Size:   How many checkboxes will be displayed on each line. 
                        Value:  A comma-separated list of item labels.
                        Modifiers:
                            checkbox+buttons    will add Set and Clear buttons to the basic checkbox type.
                            checkbox+values     allows the definition of values that are different to the displayed text.

        date        A single-line text box and a calendar icon button next to it; clicking on the 
                        button will bring up a calendar from which the user can select a date. 
                        The date can also be typed into the text box.   
                        Size:   The text box width in characters.   
                        Value:  The initial text (unless default column exists).

        label       Read-only label text.
                        Value:  The text of the label.

        radio       Like checkbox except that radio buttons are mutually exclusive; only one can be selected.   
                        radio+values allows the definition of values that are different to the displayed text.

        select      A select box / dropdown.    
                        Size: A fixed size for the box (e.g. 1, or a range e.g. 3..10. To get a dropdown, use size 1.
                                If you specify a range, the box will never be smaller than 3 items, never larger 
                                than 10, and will be 5 high if there are only 5 options.    
                        Value: A comma-separated list of options for the box.
                        Modifiers:
                            select+multi turns multiselect on for the select, to allow Shift+Click and Ctrl+Click 
                                to select (or deselect) multiple items.
                            select+values   allows the definition of values that are different to the displayed text. 
                                You can combine these modifiers e.g. select+multi+values

        text        A one-line text field. 
                        Size:   The text box width in number of characters. 
                        Value:  The initial (default) content when a new topic is created with this form definition 
                                    (unless default column exists).

        textarea        A multi-line text box.  
                        Size:   Size in columns x rows, e.g. 80x6; default size is 40x5.    
                        Value:  The initial text (unless default column exists).     

    Returns:
        self
    """

    # no schema argument in the method call, use defined schema
    if schema is None:
        schema = self.schema

    # there is no schema defined, so give up.
    if schema is None:
        return self

    self.schema = schema

    # ---------------------------------------------------------
    # schemaclass support
    # ---------------------------------------------------------

    if (isinstance(schema, type)
        and getattr(schema, "__is_schemaclass__", False)
        ):

        return self.attach_schema(schema)

    # ---------------------------------------------------------
    # schema_daf support
    # ---------------------------------------------------------

    if isinstance(schema, type(self)):

        schema_cols = schema.columns()

        # ---- cols ----
        schema_Name_ls = schema.col('Name')

        if (
            not self.hd
            and 'Name' in schema_cols
            ):

            self.set_cols(schema_Name_ls)

            #self._rebuild_hd() Done inside the function above.

        # ---- dtypes ----

        elif (
                not self.dtypes
                and 'dtype' in schema_cols
                ):

            schema_dtype_ls = schema.col('dtype')

            dtypes_dict = dict(zip(schema_Name_ls, schema_dtype_ls))

            for field, dtype_name in dtypes_dict.items():

                if dtype_name == 'str' or not dtype_name:
                    dtypes_dict[field] = str

                elif dtype_name == 'int':
                    dtypes_dict[field] = int

                elif dtype_name == 'float':
                    dtypes_dict[field] = float

                elif dtype_name == 'bool':
                    dtypes_dict[field] = bool

                elif dtype_name == 'list':
                    dtypes_dict[field] = list

                elif dtype_name == 'dict':
                    dtypes_dict[field] = dict

                else:

                    raise RuntimeError(
                        f"Unsupported dtype "
                        f"'{dtype_name}' "
                        f"in schema."
                    )

            self.dtypes = dtypes_dict

        # ---- keyfield ----

        schema_keyfield = schema.attrs.get('keyfield', '')

        if (
            not self.keyfield
            and schema_keyfield
            ):

            self.keyfield = schema_keyfield

        return self


def _attach_schema(self, schema: type) -> None:
    """
    Attach a schema_class type schema to the Daf instance.

    Args:
        schema: Schema class.

    Notes:
        Does not modify data or perform validation.
        Modifies dtypes property.
    """
    """
    Attach a schema to this Daf instance without modifying data.

    The schema is remembered for future use and may provide dtypes,
    defaults, and optional metadata such as __keyfield__.

    No column reconciliation, type conversion, or validation is performed.
    """

    # basic validation
    if not getattr(schema, "__is_schemaclass__", False):
        raise TypeError("schema must be a @schemaclass")

    # remember schema
    self.schema = schema

    # ---- cols ----

    # if not col names are set, then use schema for them

    if not self.hd:

        self.set_cols(schema.get_columns())

        # self._rebuild_hd()   done above.

    # ---- dtypes ----

    if not self.dtypes:

        self.dtypes = schema.get_dtypes_dict(
            use_origins=True,
            )

    # ---- keyfield ----

    # adopt keyfield if not already set
    if not self.keyfield and hasattr(schema, "__keyfield__"):
        self.keyfield = schema.__keyfield__
        if self.hd:
            self._invalidate_kd()    # use lazy kd rebuilding
            # self._rebuild_kd()

    return self


def _default_record(self) -> T_da:
    """
    Return a new record initialized from the attached schema.

    Supports:
        - schemaclass
        - schema_daf

    The schema is used only as a source of:
        - column names
        - default values

    No type conversion, validation, or normalization is performed.

    TODO:
        support KeyedList return type.
    """

    if not self.schema:

        raise AttributeError(
            "schema must be defined. "
            "No schema attached to this Daf instance."
        )

    schema = self.schema

    # ---------------------------------------------------------
    # schemaclass support
    # ---------------------------------------------------------

    if (
        isinstance(schema, type)
        and getattr(schema, "__is_schemaclass__", False)
        ):

        return schema.default_record()

    # ---------------------------------------------------------
    # schema_daf support
    # ---------------------------------------------------------

    if isinstance(schema, type(self)):

        schema_cols = schema.columns()

        if 'Name' not in schema_cols:

            raise RuntimeError(
                "schema_daf must define 'Name' column."
            )

        rec: T_da = {}

        schema_Name_ls = schema.col('Name')

        if 'Default' in schema_cols:

            schema_Default_ls = schema.col('Default')

        else:

            schema_Default_ls = [''] * len(schema_Name_ls)

        for field, default in zip(
                schema_Name_ls,
                schema_Default_ls,
                ):

            rec[field] = copy.copy(default)

        return rec

    # ---------------------------------------------------------
    # unsupported
    # ---------------------------------------------------------

    raise TypeError(
        f"Unsupported schema type: {type(schema)}"
    )



#==================================================================

"""
schemaclass.py

Schema support for Daffodil.

Defines the @schemaclass decorator used to describe tabular schemas
(column names, intended types, and default values) without imposing
runtime object semantics.

A schemaclass is not intended to be instantiated. It is used only as a
schema descriptor by consuming code.
"""


T_da =  Dict[str, Any]

T = TypeVar("T")

class SchemaBase:
    """
    Static interface for schemaclasses so mypy can see attributes
    injected by @schemaclass.
    """

    __annotations__: Dict[str, type]

    @classmethod
    def default_record(cls, **kwargs: Any) -> T_da: ...
    
    @classmethod
    def record_from(cls, src: Union[T_da, KeyedList]) -> T_da: ...

    @classmethod
    def get_dtypes_dict(cls, *, use_origins: bool = False) -> Dict[str, type]: ...

    @classmethod
    def get_columns(cls) -> List[str]: ...

    @staticmethod
    def get_pandas_dtypes_from_schema(schema) -> Dict[str, Any]: ...

    @classmethod
    def validate_keys_debug(cls, da: T_da) -> None: ...
    

def schemaclass(cls: type[T]) -> type[T]:
    """
    Decorator marking a class as a Daffodil schema.

    The decorated class is used only for its structure:
    - __annotations__ define column names and intended types
    - class attributes define default values

    The class is not intended to be instantiated.
    """

    # ---- allow inheritance ------------------------------------------------

    merged_ann = {}

    for base_cls in reversed(cls.__mro__):

        if base_cls is SchemaBase:
            continue

        merged_ann.update(
            getattr(base_cls, "__annotations__", {})
        )

    ann = merged_ann
    
    cls.__schema_annotations__ = merged_ann

    # ---- basic validation -------------------------------------------------

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
        
        for name in cls.__schema_annotations__:
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
            return dict(cls.__schema_annotations__)
            
        dtypes_dict = {
            name: (typing.get_origin(tp) or tp)
                for name, tp in cls.__schema_annotations__.items()
            }
            
        return dtypes_dict
   
    
    @classmethod
    def get_columns(cls) -> List[str]:
        """
        Return a list of column names.

        Types describe non-empty values only.
        """

        columns_ls: List[str] = list(cls.__schema_annotations__.keys())
        return columns_ls
   
    
    @staticmethod
    def get_pandas_dtypes_from_schema(schema):
        dtypes = {}

        for name, tp in schema.__schema_annotations__.items():
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

        invalid = [k for k in da if k not in cls.__schema_annotations__]

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



