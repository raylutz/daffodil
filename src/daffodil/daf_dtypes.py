# daf_dtypes.py
# copyright (c) 2026 Ray Lutz
"""
Methods of Daf for types and conversion.

Each function here takes the Daf as its first argument, or the class for a classmethod.
It is assigned in the Daf class body under its Daf name, without the underscore.
The docs show it under that name.

Some functions use names defined in daf.py, such as Daf, when they run.
daf.py sets those names here, as listed in _NAMES_FROM_DAF, once they exist.
An import at the top would be circular, since daf.py imports this module.
"""

from __future__ import annotations

from daffodil.lib.daf_types import T_dtype_dict, T_ls
import daffodil.lib.daf_utils    as daf_utils

from typing import Dict, Any, Type, TYPE_CHECKING

if TYPE_CHECKING:       # for mypy. daf.py sets the names below when the program runs.
    from daffodil.daf import Daf


def _set_dtypes(self: 'Daf',
        default_type: Type = str,
        typ_to_cols_dict: Dict[Type, T_ls] | None = None,
        ) -> 'Daf':

    """
    Set the dtype of every column, from a default and a few exceptions.

    Use it when most columns have one type and a few differ. The result is stored
    in `dtypes`. This does not convert any data. Call `apply_dtypes()` for that.

    If most columns differ, assign a dict to `dtypes` yourself.

    Args:
        default_type: The type of every column that is not listed.
        typ_to_cols_dict: Maps a type to the names of the columns of that type.

    Returns:
        This Daf, which has been changed.

    Raises:
        NotImplementedError: The Daf has no column names.

    Examples:
        >>> daf = Daf(lol=[['1', '2', 'x']], cols=['a', 'b', 'c'])
        >>> daf
        <BLANKLINE>
        | a  | b  | c  |
        | -: | -: | -: |
        |  1 |  2 |  x |
        <BLANKLINE>
        %% daf rows=1; cols=3; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.set_dtypes(str, {int: ['a', 'b']}).dtypes
        {'a': <class 'int'>, 'b': <class 'int'>, 'c': <class 'str'>}
    """

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

    dtypes: T_dtype_dict = {}

    col_to_typ_dict = daf_utils.invert_dol_to_dict(typ_to_cols_dict or {})

    for colname in self.hd:

        if colname in col_to_typ_dict:
            dtypes[colname] = col_to_typ_dict[colname]
        else:
            dtypes[colname] = default_type

    self.dtypes = dtypes

    return self


def _apply_dtypes(self: 'Daf', *,
        dtypes:         T_dtype_dict | None = None,
        cols:           str | T_ls | None = None,
        unflatten:      bool=True,
        from_str:       bool=True,
        default_type:   Type=str,
        silent_error:   bool=False,
        ) -> 'Daf':
    r"""
    Convert the columns to their dtypes, in place.

    A CSV file is read as text, because that is the fastest way to load it. Call
    this method to turn the columns you need into numbers, lists and so on. It
    changes the cells where they are and does not make a new table.

    Three steps are separate in daffodil:

    | Step | What it does |
    |---|---|
    | set `dtypes`, or `set_dtypes()` | declares the types of the columns. Nothing is converted |
    | `apply_dtypes()` | tries to convert the cells |
    | `check_schema()` | finds the cells that do not fit their dtypes |

    Conversion is permissive. Text that is not a number stays as it is, and no error is
    raised. So a column with an `int` dtype can still hold some text after this call.

    Conversion costs time, about 0.04 s for each column of 200,000 rows, so convert
    only the columns that you use. Pass `cols` to name them. A column that is not
    in `cols` is not converted, is not checked, and keeps its entry in the `dtypes`
    of the Daf. Without `cols`, every column of the Daf is converted, and each one
    needs a dtype.

    The `dtypes` argument is a dict that maps a column name to a type, or one type
    for all columns. It may hold more columns than the Daf. Without `cols`, it
    replaces the `dtypes` of the Daf. With `cols`, it only supplies the types of
    those columns, and the other entries of the `dtypes` of the Daf stay as they
    are. If neither the argument nor the Daf has dtypes, nothing happens. With no
    columns defined, the names are taken from the dtypes.

    Without `cols`, a `dtypes` that leaves out a column of the Daf raises
    `ValueError`, unless `silent_error` is True. Then the column gets `default_type`,
    which is `str` and is not converted, and the `dtypes` of the Daf are replaced by
    the dict with that column added. That replaces the types of the columns that you
    meant to convert later. To convert some columns and not others, use `cols`.

    Types must be plain types such as `int`, `float`, `bool`, `str`, `list`,
    `dict`, `tuple` or `set`. Annotations such as `List[str]` do not work.

    By default, columns of type `str` are skipped. The cells are assumed to be text
    already. Pass `from_str=False` when they may hold other values. Then each cell
    is converted with `str()`, except a bool, which becomes `'0'` or `'1'`. Columns of type `list` or `dict` are read from
    their text. Pass `unflatten=False` to leave them as text. That text is read
    about eight times faster when pyjson5 is installed, as with
    `pip install daffodil[pyjson5]`.

    An empty cell, such as two commas next to each other in a CSV file, is NULL,
    which is the empty string. It stays NULL when its column is converted to `int` or
    `float`, so a number column can hold NULL. A `bool` column is the exception: an
    empty cell becomes 0 there.

    A `bool` column gets 0 or 1. The text `0`, `false`, `no`, `n`, `f` and `off` give 0,
    and `1`, `true`, `yes`, `y`, `t` and `on` give 1, in lower case, capitalized or upper
    case. Other text stays as it is. So `'False'` gives 0, where Python's own `bool('False')`
    is True. None gives 0, and any other value gives 0 or 1 by its truth. The sum methods, `sum()`, `sum_np()`
    and `Daf.sum_da()`, skip a NULL cell. Python's own `sum()` of such a column
    raises `TypeError`, so use the methods of the Daf.

    A cell that cannot be converted to an `int` or a `float` keeps its text, so a bad
    value is still there to be found, as with `list` and `dict`. No error is raised
    for text. Call `check_schema()` to find such cells. A later step, such as a sum or a sort, may raise one. A cell that is
    not text and not a number, such as a list in an `int` column, cannot be given to
    `int()`, and raises `TypeError`. The `dtypes` do not limit what a cell can hold,
    so leave such a column out of the conversion. Whole number text of any size is
    converted exactly. Text with a decimal point or an exponent is converted to an
    `int` by way of a float, which cuts the decimal part.

    This method does the common conversions and keeps them simple. For your own
    rules, convert the columns yourself and then say what the types are. Use
    `apply_to_col()` for one column, or `apply_in_place()` for several. Your
    function can raise, collect the bad values, or use any default. Then set
    `dtypes`. This does not convert anything. See the last example.

    Args:
        dtypes: Maps column names to types, or a single type for all columns.
        cols: The columns to convert, or one column name. Others are left alone.
        unflatten: If True, read list and dict columns from their text.
        from_str: If True, the cells are text, so `str` columns are not converted.
        default_type: The type to use for a column that has no dtype.
        silent_error: If False, raise an error when a column has no dtype.

    Returns:
        This Daf, which has been changed.

    Raises:
        ValueError: A column to convert has no dtype and `silent_error` is False.
        KeyError: A name in `cols` is not a column of the Daf.
        TypeError: A cell in an `int` or `float` column is not text and not a number.

    Examples:
        >>> daf = Daf(lol=[['1', '2.5', 'x']], cols=['a', 'b', 'c'])
        >>> daf
        <BLANKLINE>
        | a  |  b  | c  |
        | -: | --: | -: |
        |  1 | 2.5 |  x |
        <BLANKLINE>
        %% daf rows=1; cols=3; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.apply_dtypes(dtypes={'a': int, 'b': float, 'c': str})
        <BLANKLINE>
        | a  |  b  | c  |
        | -: | --: | -: |
        |  1 | 2.5 |  x |
        <BLANKLINE>
        %% daf rows=1; cols=3; keyfield=''; name=''
        <BLANKLINE>

        An empty cell stays empty, and the sum methods skip it:

        >>> daf = Daf.from_csv_buff('a,b\n1,\n,3\n4,5\n')
        >>> daf
        <BLANKLINE>
        | a  | b  |
        | -: | -: |
        |  1 |    |
        |    |  3 |
        |  4 |  5 |
        <BLANKLINE>
        %% daf rows=3; cols=2; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.apply_dtypes(dtypes={'a': int, 'b': int})
        <BLANKLINE>
        | a  | b  |
        | -: | -: |
        |  1 |    |
        |    |  3 |
        |  4 |  5 |
        <BLANKLINE>
        %% daf rows=3; cols=2; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.sum()
        {'a': 5, 'b': 8}

        Convert only some columns with `cols`. The others stay text, and the
        `dtypes` of the Daf are not changed:

        >>> daf = Daf(lol=[['1', '2.5', '7']], cols=['a', 'b', 'n'],
        ...         dtypes={'a': int, 'b': float, 'n': int})
        >>> daf
        <BLANKLINE>
        | a  |  b  | n  |
        | -: | --: | -: |
        |  1 | 2.5 |  7 |
        <BLANKLINE>
        %% daf rows=1; cols=3; keyfield=''; name=''
        <BLANKLINE>
        >>> _ = daf.apply_dtypes(cols=['a'])
        >>> daf.iloc(0)
        {'a': 1, 'b': '2.5', 'n': '7'}
        >>> daf.dtypes == {'a': int, 'b': float, 'n': int}
        True
        >>> _ = daf.apply_dtypes(cols=['n'])
        >>> daf.iloc(0)
        {'a': 1, 'b': '2.5', 'n': 7}

        A `bool` column, and text that does not convert:

        >>> b_daf = Daf(cols=['ok', 'n'], lol=[['', '1'], ['0', 'x'], ['True', '3'], ['maybe', '']])
        >>> b_daf
        <BLANKLINE>
        |  ok   | n  |
        | ----: | -: |
        |       |  1 |
        |     0 |  x |
        |  True |  3 |
        | maybe |    |
        <BLANKLINE>
        %% daf rows=4; cols=2; keyfield=''; name=''
        <BLANKLINE>
        >>> _ = b_daf.apply_dtypes(dtypes={'ok': bool, 'n': int})
        >>> b_daf.lol
        [[0, 1], [0, 'x'], [1, 3], ['maybe', '']]

        Your own conversion, here one that records the values that fail:

        >>> bad = []
        >>> def to_int(value):
        ...     try:
        ...         return int(value)
        ...     except ValueError:
        ...         bad.append(value)
        ...         return ''
        >>> daf = Daf(lol=[['1'], ['x'], ['3']], cols=['n'])
        >>> daf
        <BLANKLINE>
        | n  |
        | -: |
        |  1 |
        |  x |
        |  3 |
        <BLANKLINE>
        %% daf rows=3; cols=1; keyfield=''; name=''
        <BLANKLINE>
        >>> _ = daf.apply_to_col('n', to_int)
        >>> daf.dtypes = {'n': int}
        >>> daf
        <BLANKLINE>
        | n  |
        | -: |
        |  1 |
        |    |
        |  3 |
        <BLANKLINE>
        %% daf rows=3; cols=1; keyfield=''; name=''
        <BLANKLINE>
        >>> bad
        ['x']
    """

    """ convert columns of daf array to the datatypes specified in self.dtypes or in passed parameter.
                dtypes can be a dict where each column may have a different type, or it can be a single type.
                dtypes may provide desired new types for only some of the columns.
                types in dtypes must be origin types, like bool, str, int, float, list, dict, tuple, set,
                    and not type annotations such as List[str] etc.
            columns (self.hd) must be defined.
            unflatten: unflatten from pyon or json to list or dict types
            if from_str is True, (default) assumes that data starts as str, such as when read from csv
                If the csv is first read, data is all delivered as str.
                Columns specified as str type are not converted.

            Note, this converts types in place, and does not create a new array.

            if no dtypes is passed and self.dtypes is not defined, do nothing.

            Unless silent_error is True, will check for columns consistency against the dtypes.
            Note: dtypes can be a SUPERSET of the columns. I.e. a single dtypes definition can be
                    used for a number of tables that differ only in which columns are included.
        """

    if cols is not None:
        return self._apply_dtypes_to_cols(cols, dtypes, unflatten, from_str, default_type, silent_error)

    if dtypes:
        self.dtypes = dtypes

    if not self.dtypes:
        return self

    # this adopts the hd from dtypes if not already defined.
    if not self.hd and self.dtypes and isinstance(self.dtypes, dict):
        # self.dtypes, not the (possibly still-None if the caller didn't pass dtypes=)
        # local dtypes param -- the guard above checks self.dtypes specifically.
        self._cols_to_hd(self.dtypes.keys())
        # currently will not overwrite existing cols.
        # (i.e. pass partial dtypes will alter only those cols)

    if (
        not silent_error
        and isinstance(self.dtypes, dict)
        and self.hd
        and not set(self.hd).issubset(self.dtypes)
        ):
        missing = sorted(set(self.hd) - set(self.dtypes))
        raise ValueError(
            "dtypes mismatch. check the schema:\n"
            f"missing dtypes for cols={missing}\n"
            f"dtypes keys={sorted(self.dtypes)},\n"
            f"cols={sorted(self.hd)}"
        )
    if not self.lol or not self.lol[0]:
        # this can sometimes happen, no worries.
        return self

    # first calculate all columns to consider, that are not str or str and unflatten
    for col in self.hd.keys():
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

        # look up the conversion once for the column, not for each cell.
        convert = daf_utils.get_converter(desired_type)

        for row_la in self.lol:
            row_la[icol] = convert(row_la[icol])
        self._invalidate_kd_if_key([icol])          # a key such as '1' may now be 1.

    return self


def _apply_dtypes_to_cols(self: 'Daf',
        cols:           str | T_ls,
        dtypes:         T_dtype_dict | None,
        unflatten:      bool,
        from_str:       bool,
        default_type:   Type,
        silent_error:   bool,
        ) -> 'Daf':
    """
    Convert only the named columns, for `apply_dtypes()`. Internal use.

    The types come from `dtypes` if it is given, and otherwise from the `dtypes` of the Daf.
    A column that is not in `cols` is not converted and not checked, and its entry in the `dtypes` of the
    Daf is not changed. A given `dtypes` only updates the entries of the columns in `cols`.

    Args:
        cols: The columns to convert, or one column name.
        dtypes: Maps column names to types, or a single type for all the columns.
        unflatten: If True, read list and dict columns from their text.
        from_str: If True, the cells are text, so `str` columns are not converted.
        default_type: The type to use for a column that has no dtype.
        silent_error: If False, raise an error when a column to convert has no dtype.

    Returns:
        This Daf, which has been changed.

    Raises:
        KeyError: A name in `cols` is not a column of the Daf.
        ValueError: A column to convert has no dtype and `silent_error` is False.
    """
    cols_ls = [cols] if isinstance(cols, str) else list(cols)

    missing_cols = [col for col in cols_ls if col not in self.hd]
    if missing_cols:
        raise KeyError(f"apply_dtypes(): cols names {missing_cols}, which are not columns of this Daf. "
                       f"The columns are {list(self.hd)}.")

    source = dtypes if dtypes else self.dtypes
    if not source:
        return self                                         # no dtypes at all, so nothing to convert to.

    type_of_col: Dict[str, Any] = {}
    for col in cols_ls:
        if isinstance(source, dict):
            if col in source:
                type_of_col[col] = source[col]
            elif silent_error:
                type_of_col[col] = default_type
            else:
                raise ValueError(f"apply_dtypes(): no dtype for the columns to convert: "
                                 f"{[c for c in cols_ls if c not in source]}. "
                                 f"Add them to dtypes, or pass silent_error=True to use default_type.")
        else:
            type_of_col[col] = source                       # one type for all columns.

    if dtypes or any(not isinstance(self.dtypes, dict) or col not in self.dtypes for col in cols_ls):
        merged_dtypes = dict(self.dtypes) if isinstance(self.dtypes, dict) else {}
        merged_dtypes.update(type_of_col)
        self.dtypes = merged_dtypes

    if not self.lol or not self.lol[0]:
        return self

    for col, desired_type in type_of_col.items():
        if (    desired_type is str and from_str or
                desired_type in [list, dict] and not unflatten
            ):
            continue

        icol = self.hd[col]
        convert = daf_utils.get_converter(desired_type)     # look up the conversion once for the column.
        for row_la in self.lol:
            row_la[icol] = convert(row_la[icol])
        self._invalidate_kd_if_key([icol])          # a key such as '1' may now be 1.

    return self


def _flatten(self: 'Daf', convert_bool_to_int: bool=True, use_pyon: bool = True) -> 'Daf':
    """
    Turn list and dict cells into text, in place.

    You rarely need this. `to_csv_buff()` and `to_csv_file()` write every cell as
    text already, so a separate pass is not needed.

    Only the columns whose dtype is `list` or `dict` are changed. Each cell
    becomes its `str()` text. A column of dtype `bool` becomes 0 and 1. Without
    dtypes nothing happens.

    Args:
        convert_bool_to_int: If True, write bool columns as 0 and 1.
        use_pyon: Kept for compatibility. Only True is supported.

    Returns:
        This Daf, which has been changed.

    Raises:
        ValueError: `use_pyon` is False.

    Examples:
        >>> daf = Daf(lol=[[[1, 2], True]], cols=['a', 'b'], dtypes={'a': list, 'b': bool})
        >>> daf
        <BLANKLINE>
        |   a    |  b   |
        | -----: | ---: |
        | [1, 2] | True |
        <BLANKLINE>
        %% daf rows=1; cols=2; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.flatten()
        <BLANKLINE>
        |   a    | b  |
        | -----: | -: |
        | [1, 2] |  1 |
        <BLANKLINE>
        %% daf rows=1; cols=2; keyfield=''; name=''
        <BLANKLINE>
    """
    if not use_pyon:
        raise ValueError("flatten(): use_pyon=False is not supported. Cells are flattened to PYON.")

    if not self.lol or not self.lol[0] or not self.dtypes:
        # this can sometimes happen, no worries.
        # when daf is constructed internally and has no list or dict types, there is no need to flatten.
        # in this case, the self.dtypes need not be initialized or used until the table is read.
        return self

    if not self.hd:     # pragma: no cover
        # should not be the case. Logic error.
        raise RuntimeError("flatten(): Daf has data and dtypes but no header")

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
                self.lol[irow][icol] = f"{self.lol[irow][icol]}"
            self._invalidate_kd_if_key([icol])

        if convert_bool_to_int and desired_type is bool:
            # this should be rare!

            icol = self.hd[col]

            for irow in range(len(self.lol)):

                self.lol[irow][icol] = int(bool(self.lol[irow][icol]))
            self._invalidate_kd_if_key([icol])

    return self


def _strip(self: 'Daf', chrs: str=' ') -> 'Daf':
    """
    Remove characters from both ends of every text cell, in place.

    Each character in `chrs` is removed on its own, so `'()"'` removes any mix of
    parentheses and quotes. Cells that are not text, and empty cells, are skipped.

    Args:
        chrs: The characters to remove.

    Returns:
        This Daf, which has been changed.

    Examples:
        >>> Daf(lol=[[' a ', 3, '("x")']], cols=['p', 'q', 'r']).strip(' ()"')
        <BLANKLINE>
        | p  | q  | r  |
        | -: | -: | -: |
        |  a |  3 |  x |
        <BLANKLINE>
        %% daf rows=1; cols=3; keyfield=''; name=''
        <BLANKLINE>
    """
    """ remove leading or trailing characters in the string chrs from each str value in the array.
            modifies in-place.  Ignores non str values.

            each character in chrs treated seperately, such as strip('"()') removes quotes and parens.
        """

    for row_la in self.lol:
        for icol in range(len(row_la)):
            if isinstance(row_la[icol], str) and row_la[icol]:
                row_la[icol] = row_la[icol].strip(chrs)

    self._invalidate_kd()       # a key may have been stripped.

    return self
