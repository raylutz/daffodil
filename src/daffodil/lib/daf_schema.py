# daf_schema.py

import typing
from typing import List, Dict, Any, Tuple, TypeVar, cast, Type, Callable # noqa: F401
from daffodil.lib.daf_types import T_ls, T_lola, T_da, T_li, T_cs, T_ca, T_ma # noqa: F401
from daffodil.lib.schemaclass import SchemaBase
from daffodil.keyedlist import KeyedList

import copy

from typing import TYPE_CHECKING
if TYPE_CHECKING:       # for the annotations only. A real import would be circular.
    from daffodil.daf import Daf

def _apply_schema(
        self: 'Daf',
        schema: 'type | Daf | None' = None,
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
    nothing is validated. Call `check_schema()` to check the cells.

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
        >>> daf = Daf(schema=Person)
        >>> daf.columns()
        ['name', 'age']

        On a Daf with rows and no names, it fills the names and dtypes, and converts nothing:

        >>> raw = Daf(lol=[['ann', '7']])
        >>> raw
        <BLANKLINE>
        |  A  | B  |
        | --: | -: |
        | ann |  7 |
        <BLANKLINE>
        %% daf rows=1; cols=2; keyfield=''; name=''
        <BLANKLINE>
        >>> _ = raw.apply_schema(Person)
        >>> raw.columns(), raw.lol
        (['name', 'age'], [['ann', '7']])
        >>> daf.dtypes
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


def _default_record(self: 'Daf', astype: type = dict) -> Any:
    """
    Make a new record for this Daf: its columns, with the defaults of the attached schema.

    Use it to start a record that you fill in, then append. Each call returns a new
    record, so changing it does not change the next one. Nothing is converted or validated.

    The defaults are worked out the first time, and kept until the columns or the schema of
    the Daf are replaced. A default that is changed on the schema class after that is not seen.

    The record follows the columns of the Daf, in their order, however they were set. A
    column gets the schema's default if the schema has one, and NULL otherwise. A field of
    the schema that is not a column is left out. So a Daf that keeps only some of the
    schema's columns gets records that fit it. A Daf with no schema gets NULL in every
    column. A Daf with no columns takes its columns from the schema.

    With `astype=KeyedList` the record is a KeyedList that shares the `hd` of this Daf,
    so `record.hd is daf.hd`. Fill it by name, in any order: assigning to a key writes to
    that column's position. `append()` then knows the record fits, and skips the check of
    its columns.

    For a schema Daf, the `Name` column gives the fields and the `Default` column gives
    the values. A missing `Default` column gives empty strings.

    Args:
        astype: `dict`, the default, or `KeyedList`.

    Returns:
        A dict or a KeyedList of each column and its default.

    Raises:
        AttributeError: The Daf has no columns and no schema.
        RuntimeError: A schema Daf has no `Name` column.
        TypeError: The attached schema is of an unsupported kind, or `astype` is not dict or KeyedList.

    Examples:
        >>> from daffodil.daf import Daf
        >>> from daffodil.keyedlist import KeyedList
        >>> from daffodil.lib.schemaclass import schemaclass
        >>> @schemaclass
        ... class Person:
        ...     name: str = ''
        ...     age: int = 0
        >>> Daf(schema=Person).default_record()
        {'name': '', 'age': 0}
        >>> Daf(cols=['age', 'city'], schema=Person).default_record()
        {'age': 0, 'city': ''}
        >>> daf = Daf(schema=Person)
        >>> row = daf.default_record(astype=KeyedList)
        >>> row['age'] = 41
        >>> row.hd is daf.hd, row.to_dict()
        (True, {'name': '', 'age': 41})
    """
    if astype is not dict and astype is not KeyedList:
        raise TypeError(f"default_record(): astype must be dict or KeyedList, not {astype!r}")

    if self.hd:
        # the defaults in column order are worked out once, and kept until the columns or the
        # schema change. Each call copies them, and copies afresh a default that is a list, dict or set.
        template = self.__dict__.get('_default_template')
        if template is None or template[0] is not self.hd or template[1] is not self.schema:
            defaults_da = _schema_defaults(self) if self.schema else {}
            values_la = [defaults_da.get(col, '') for col in self.hd]
            # (position, copier) for each default that is a list, dict or set, so no record shares it.
            mutable_lt = [(pos, copy.copy if type(val) not in (list, dict, set) else type(val))
                          for pos, val in enumerate(values_la) if isinstance(val, (list, dict, set))]
            template = self.__dict__['_default_template'] = (self.hd, self.schema, values_la, mutable_lt)
        values = template[2][:]
        for pos, copier in template[3]:
            values[pos] = copier(values[pos])
        if astype is KeyedList:
            return KeyedList(self.hd, values)               # shares the hd of this Daf
        return dict(zip(self.hd, values))

    defaults_da = _schema_defaults(self) if self.schema else {}

    if not self.schema:
        raise AttributeError(
            "default_record(): the Daf has no columns and no schema."
        )
    if astype is KeyedList:
        return KeyedList(defaults_da)
    return defaults_da


def _schema_defaults(self: 'Daf') -> T_da:
    """ A new dict of each field of the attached schema and its default, in the schema's order. """

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


# The classes a cell may have for each dtype, for check_schema(). A NULL cell is allowed in any column.
_SCHEMA_CELL_CLASSES: Dict[type, Tuple[type, ...]] = {
    str:    (str,),
    int:    (int,),
    float:  (float, int),
    bool:   (int, bool),
    list:   (list,),
    dict:   (dict,),
    }


def _check_schema(self: 'Daf', max_report: int = 10) -> 'Daf':
    """
    Check that every cell fits the dtypes and the required columns. Raise if one does not.

    Loading does not check the data, so that it stays fast. `apply_dtypes()` leaves a
    cell that does not convert as text, and an empty cell is NULL. Call this method
    when you want to find such cells, for example after a load while developing, or
    in a test. It is never called by daffodil itself.

    The dtypes are those of the Daf, which `apply_schema()` takes from the schema. A
    cell must be of its dtype, or NULL. NULL is allowed in a column of any dtype,
    unless the column is required, and it is false when tested. A `float` column may
    also hold an `int`. A `bool` column holds 0 or 1. `apply_dtypes()` keeps an empty
    cell as NULL, except in a `bool` column, where it becomes 0. A column of a dtype
    not listed here is not checked.

    The required columns come from the `Required` column of an attached schema Daf.
    A value such as `yes`, `true` or `1` marks the column as required. A required
    column must exist, and none of its cells may be NULL. A `@schemaclass` has no
    required columns.

    Args:
        max_report: How many bad cells the error lists. The error also gives the total.

    Returns:
        This Daf, unchanged, when every cell fits.

    Raises:
        ValueError: A cell does not fit its dtype, a required cell is NULL, or a
            required column is missing.

    Examples:
        >>> from daffodil.daf import Daf
        >>> schema = Daf.from_md('''
        ... | Name | dtype | Required |
        ... |:-----|:------|:---------|
        ... | id   | str   | yes      |
        ... | n    | int   |          |
        ... ''')
        >>> schema
        <BLANKLINE>
        | Name | dtype | Required |
        | ---: | ----: | -------: |
        |   id |   str |      yes |
        |    n |   int |          |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield=''; name=''
        <BLANKLINE>
        >>> daf = Daf.from_csv_buff('id,n\\na,1\\n,x\\n')
        >>> daf
        <BLANKLINE>
        | id | n  |
        | -: | -: |
        |  a |  1 |
        |    |  x |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield=''; name=''
        <BLANKLINE>
        >>> daf = daf.apply_schema(schema).apply_dtypes(dtypes=daf.dtypes)
        >>> daf.check_schema()
        Traceback (most recent call last):
        ...
        ValueError: check_schema(): 2 cells do not fit the schema:
          row 1, column 'id': required, but empty
          row 1, column 'n': 'x' is not int
    """
    from daffodil.lib.daf_utils import test_strbool, NULL

    required_set: set = set()
    schema = self.schema
    if schema is not None and not isinstance(schema, type) and 'Required' in schema.columns():
        required_set = {name for name, flag in zip(schema.col('Name'), schema.col('Required')) if test_strbool(flag)}

    dtypes = self.dtypes or {}
    # A row is named by its key, or by its position when the key is missing or made of several columns.
    keyfield_idx = self.hd.get(self.keyfield) if isinstance(self.keyfield, str) and self.keyfield else None
    problems_ls: T_ls = []
    num_bad = 0

    for col in required_set:
        if col not in self.hd:
            num_bad += 1
            problems_ls.append(f"column {col!r}: required, but missing")

    check_cols = list(dict.fromkeys(list(dtypes) + sorted(required_set)))
    for col in check_cols:
        icol = self.hd.get(col)
        if icol is None:
            continue
        dtype = dtypes.get(col)
        allowed = _SCHEMA_CELL_CLASSES.get(dtype) if isinstance(dtype, type) else None
        is_required = col in required_set
        for irow, row_la in enumerate(self.lol):
            val = row_la[icol] if icol < len(row_la) else NULL
            if val is NULL:
                if not is_required:
                    continue
                problem = 'required, but empty'
            elif allowed is None or (val.__class__ in allowed and (dtype is not bool or val in (0, 1))):
                continue
            else:
                problem = f"{val!r} is not {dtype.__name__}"     # type: ignore[union-attr]
            num_bad += 1
            if len(problems_ls) < max_report:
                where = f"row {row_la[keyfield_idx]!r}" if keyfield_idx is not None and keyfield_idx < len(row_la) else f"row {irow}"
                problems_ls.append(f"{where}, column {col!r}: {problem}")

    if num_bad:
        shown = '' if num_bad <= len(problems_ls) else f". The first {len(problems_ls)}"
        cells = '1 cell does' if num_bad == 1 else f"{num_bad} cells do"
        raise ValueError(f"check_schema(): {cells} not fit the schema{shown}:\n  " + "\n  ".join(problems_ls))
    return self
