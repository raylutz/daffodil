# daf_schema.py

import typing
from typing import List, Dict, Any, Tuple, Optional, TypeVar, Union, cast, Type, Callable # noqa: F401
from daffodil.lib.daf_types import T_ls, T_lola, T_da, T_li, T_cs, T_ca, T_ma # noqa: F401
from daffodil.lib.schemaclass import SchemaBase

import copy

from typing import TYPE_CHECKING
if TYPE_CHECKING:       # for the annotations only. A real import would be circular.
    from daffodil.daf import Daf

def _apply_schema(
        self: 'Daf',
        schema: Optional[Union[type, 'Daf']]=None,
        ) -> 'Daf':
    """
    Attach a schema to this Daf and fill in what the Daf is missing.

    A schema describes columns: their names, types and default values. Use one
    when many tables share a layout, or to make new records with the defaults.
    The constructor calls this method, so `Daf(schema=...)` applies the schema.

    Two kinds of schema are accepted.

    A class decorated with `@schemaclass`. The annotated attributes give the
    column names and types. Their values are the defaults. A `__keyfield__`
    attribute gives the keyfield.

    A schema Daf, with one row for each column of the table. It must have a
    `Name` column. It may have a `dtype` column, which holds one of `str`, `int`,
    `float`, `bool`, `list` or `dict`. A `Default` column gives the defaults. Its
    `attrs` may hold a `keyfield`.

    Only gaps are filled. The column names, the dtypes and the keyfield are taken
    from the schema only if the Daf has none. The rows are never changed and
    nothing is validated.

    The other columns of a schema Daf are for building input forms. They are not
    used by daffodil. The `Type` column says what kind of form control to use.

        checkbox   One or more checkboxes. Value lists the labels.
                       checkbox+buttons adds Set and Clear buttons.
                       checkbox+values allows values that differ from the labels.
        date       A text box with a calendar button.
        label      Read only text.
        radio      Like checkbox, but only one can be chosen.
        select     A dropdown or list box. Value lists the options.
                       select+multi allows several choices.
                       select+values allows values that differ from the labels.
        text       A one line text box. Value is the initial text.
        textarea   A multi line text box. Size is columns x rows, such as 80x6.

    Args:
        schema: The schema to apply. If None, the schema already attached is used.

    Returns:
        This Daf, which has been changed.

    Raises:
        TypeError: The schema is neither a `@schemaclass` nor a schema Daf. Nothing is stored.
        RuntimeError: A `dtype` in a schema Daf is not one of the supported names.

    Examples:
        >>> from daffodil.daf import Daf
        >>> from daffodil.lib.schemaclass import schemaclass
        >>> @schemaclass
        ... class Person:
        ...     name: str = ''
        ...     age: int = 0
        >>> d = Daf(schema=Person)
        >>> d.columns()
        ['name', 'age']
        >>> d.dtypes
        {'name': <class 'str'>, 'age': <class 'int'>}
    """

    # no schema argument in the method call, use defined schema
    if schema is None:
        schema = self.schema

    # there is no schema defined, so give up.
    if schema is None:
        return self

    is_schemaclass = isinstance(schema, type) and getattr(schema, "__is_schemaclass__", False)

    if not is_schemaclass and not isinstance(schema, type(self)):
        raise TypeError(
            f"apply_schema: schema must be a @schemaclass or a schema Daf, "
            f"not {type(schema).__name__}"
            )

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

            keyfield = self.keyfield            # set_cols() clears the keyfield, so keep one the caller gave.
            self.set_cols(schema_Name_ls)
            self.keyfield = keyfield

            #self._rebuild_hd() Done inside the function above.

        # ---- dtypes ----

        if (
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


def _attach_schema(self: 'Daf', schema: type) -> 'Daf':
    """
    Attach a `@schemaclass` to this Daf.

    This is the strict form of `apply_schema()`, for a schema class only. It stores
    the schema, then fills in the column names, the dtypes and the keyfield, each
    only if the Daf has none. The rows are never changed and nothing is validated.

    Args:
        schema: A class decorated with `@schemaclass`.

    Returns:
        This Daf, which has been changed.

    Raises:
        TypeError: The class is not a `@schemaclass`.

    Examples:
        >>> from daffodil.daf import Daf
        >>> from daffodil.lib.schemaclass import schemaclass
        >>> @schemaclass
        ... class Person:
        ...     name: str = ''
        ...     age: int = 0
        >>> Daf().attach_schema(Person).columns()
        ['name', 'age']
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
    schema = cast(Type[SchemaBase], schema)

    # remember schema
    self.schema = schema

    # ---- cols ----

    # if not col names are set, then use schema for them

    if not self.hd:

        keyfield = self.keyfield            # set_cols() clears the keyfield, so keep one the caller gave.
        self.set_cols(schema.get_columns())
        self.keyfield = keyfield

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


def _default_record(self: 'Daf') -> T_da:
    """
    Make a new record that holds the defaults of the attached schema.

    Use it to start a record that you fill in, then append. Each call returns a
    new dict, so changing it does not change the next one. Nothing is converted
    or validated.

    For a schema Daf the `Name` column gives the keys and the `Default` column
    gives the values. A missing `Default` column gives empty strings.

    Returns:
        A dict that maps each column name to its default.

    Raises:
        AttributeError: No schema is attached.
        RuntimeError: A schema Daf has no `Name` column.
        TypeError: The attached schema is of an unsupported kind.

    Examples:
        >>> from daffodil.daf import Daf
        >>> from daffodil.lib.schemaclass import schemaclass
        >>> @schemaclass
        ... class Person:
        ...     name: str = ''
        ...     age: int = 0
        >>> Daf(schema=Person).default_record()
        {'name': '', 'age': 0}
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

        return cast(Type[SchemaBase], schema).default_record()

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



