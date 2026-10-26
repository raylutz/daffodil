# daf_structure.py
# copyright (c) 2026 Ray Lutz
"""
Methods of Daf for columns and shape.

Each function here takes the Daf as its first argument, or the class for a classmethod.
It is assigned in the Daf class body under its Daf name, without the underscore.
The docs show it under that name.

A function that needs a name defined in daf.py, such as Daf, imports it when it runs.
An import at the top would be circular, since daf.py imports this module.
"""

from __future__ import annotations

from daffodil.lib.daf_types import T_ds, T_ls
import daffodil.lib.daf_utils    as daf_utils

from typing import List, Type, TYPE_CHECKING
from collections.abc import Iterable

if TYPE_CHECKING:       # for the annotations only. A real import would be circular.
    from daffodil.daf import Daf


def _is_rectangular(self: 'Daf') -> bool:
    """
    Check whether every row has the same length.

    With column names, every row must be as long as the number of names. Without
    names, every row must be as long as the first row. An empty Daf is rectangular.

    This looks at every row. Use it when you do not trust the source of the data.
    `num_cols()` only samples the first rows.

    Returns:
        True if all rows have the expected length.

    Examples:
        >>> Daf(lol=[[1, 2], [3]], cols=['a', 'b']).is_rectangular()
        False
    """
    if not self.lol:
        return True

    target_len = len(self.hd) if self.hd else len(self.lol[0])

    return all(len(row) == target_len for row in self.lol)


def _force_rectangular(self: 'Daf') -> 'Daf':
    """
    Pad short rows with empty strings, in place.

    Use this when a source leaves off trailing empty cells. An xlsx file read by
    `xlsx_to_csv()` does that. The target width is the number of column names. With
    no names, it is the length of the longest row.

    Rows that are too long are not cut. That would hide damage, such as an
    unquoted comma that split a value. They raise an error instead.

    Each short row is padded in place, so a selection that shares it sees the new cells.
    No row moves, so the key index stays valid.

    Returns:
        This Daf, which has been changed.

    Raises:
        ValueError: A row is longer than the number of columns. Nothing is changed.

    Examples:
        >>> Daf(lol=[[1, 2], [3]], cols=['a', 'b']).force_rectangular()
        <BLANKLINE>
        | a  | b  |
        | -: | -: |
        |  1 |  2 |
        |  3 |    |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield=''; name=''
        <BLANKLINE>
    """
    if not self.lol:
        return self

    target_len = len(self.hd) if self.hd else max((len(row) for row in self.lol), default=0)

    long_rows = [(i, len(row)) for i, row in enumerate(self.lol) if len(row) > target_len]
    if long_rows:
        raise ValueError(
                f"force_rectangular: {len(long_rows)} row(s) exceed the target width of "
                f"{target_len} -- likely real data corruption (e.g. an unquoted comma "
                f"splitting a value), not the xlsx trailing-cell omission this pads for. "
                f"(row index, actual length): {long_rows[:10]}")

    for row in self.lol:
        if len(row) < target_len:
            row.extend([''] * (target_len - len(row)))

    return self


def _calc_cols(self: 'Daf',
        include_cols: Iterable | None = None,
        exclude_cols: Iterable | None = None,
        include_types: Type | List[Type] | None = None,
        exclude_types: Type | List[Type] | None = None,
       ) -> T_ls:
    """
    Work out a list of column names from rules.

    Use it to choose the columns for `apply` or `reduce`. The rules are applied
    in this order: include by name, exclude by name, include by type and exclude
    by type. The result keeps the order of the Daf. A single name may be given
    as a string.

    With a group by operation, leave the group by columns out of the list.

    Args:
        include_cols: Keep only these column names.
        exclude_cols: Drop these column names.
        include_types: Keep only columns whose dtype is in this list.
        exclude_types: Drop columns whose dtype is in this list.

    Returns:
        The selected column names.

    Raises:
        RuntimeError: A type rule is given and no dtypes are set.

    Examples:
        >>> daf = Daf(lol=[[1, 2, 3]], cols=['a', 'b', 'c'])
        >>> daf
        <BLANKLINE>
        | a  | b  | c  |
        | -: | -: | -: |
        |  1 |  2 |  3 |
        <BLANKLINE>
        %% daf rows=1; cols=3; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.calc_cols(exclude_cols='b')
        ['a', 'c']
    """

    """ this method helps to calculate the columns to be specified for a apply or reduce operation.
            Can use any combination of listing columns to be included, or excluded by name,
                or included by type.
            If using a groupby function, the cols spec should not include the groupby column(s)
        """


    # start with all cols.
    selected_cols: List[str] = list(self.hd.keys())  # change to columns()
    if not selected_cols:
        return []

    if include_cols:
        if isinstance(include_cols, str):
            include_cols = [include_cols]
        # if len(include_cols) > 10:
            # include_cols_dict = dict.fromkeys(include_cols)
            # selected_cols = [col for col in selected_cols if col in include_cols_dict]
        selected_cols = [col for col in selected_cols if col in include_cols]

    if exclude_cols:
        if isinstance(exclude_cols, str):
            exclude_cols = [exclude_cols]
        # if len(exclude_cols) > 10:
            # exclude_cols_dict = dict.fromkeys(exclude_cols)
            # selected_cols = [col for col in selected_cols if col not in exclude_cols_dict]
        selected_cols = [col for col in selected_cols if col not in exclude_cols]

    if include_types:
        if not self.dtypes:
            raise RuntimeError("calc_cols(): include_types/exclude_types requires dtypes to be set")

        if not isinstance(include_types, list):
            include_types = [include_types]
        selected_cols = [col for col in selected_cols if self.dtypes.get(col) in include_types]

    if exclude_types:
        if not self.dtypes:
            raise RuntimeError("calc_cols(): include_types/exclude_types requires dtypes to be set")

        if not isinstance(exclude_types, list):
            exclude_types = [exclude_types]
        selected_cols = [col for col in selected_cols if self.dtypes.get(col) not in exclude_types]

    return selected_cols


def _rename_cols(self: 'Daf', from_to_dict: T_ds) -> 'Daf':
    """
    Rename columns in place.

    Names that are not in the mapping stay as they are. Names in the mapping that
    are not in the Daf are ignored. The dtypes are renamed too.

    The keyfield follows the rename. If its column is renamed, the keyfield has the
    new name, and key lookups keep working.

    Args:
        from_to_dict: Maps old names to new names.

    Returns:
        This Daf, which has been changed.

    Examples:
        >>> daf = Daf(lol=[[1, 2]], cols=['a', 'b'])
        >>> daf
        <BLANKLINE>
        | a  | b  |
        | -: | -: |
        |  1 |  2 |
        <BLANKLINE>
        %% daf rows=1; cols=2; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.rename_cols({'b': 'c'}).columns()
        ['a', 'c']
        >>> k_daf = Daf(lol=[[1, 'x']], cols=['id', 'v'], keyfield='id')
        >>> k_daf
        <BLANKLINE>
        | id | v  |
        | -: | -: |
        |  1 |  x |
        <BLANKLINE>
        %% daf rows=1; cols=2; keyfield='id'; name=''
        <BLANKLINE>
        >>> k_daf.rename_cols({'id': 'ident'}).keyfield
        'ident'
    """

    # unit tests exist

    old_to_new      = {col: from_to_dict.get(col, col) for col in self.hd}
    self.hd         = {new_col: idx for idx, new_col in enumerate(old_to_new.values())}
    if self.dtypes:
        self.dtypes = {from_to_dict.get(col, col):typ for col, typ in self.dtypes.items()}
    self._remap_keyfield(old_to_new)

    return self


def _set_cols(self: 'Daf', new_cols: T_ls | None = None, sanitize_cols: bool=True, unnamed_prefix: str='col') -> 'Daf':
    """
    Set the column names, in place.

    The new names are given by position, so the first name goes to the first
    column. Without a list, the names are A, B, C and so on, like a spreadsheet.

    With `sanitize_cols` on, a repeated name gets a suffix, so `['a', 'a']` becomes
    `['a', 'a_1']`. An empty name becomes the prefix and its position, like `col2`. The
    prefix is short, because these names are printed. Names that come from parsing a header, in the
    constructor and in `from_md()`, use `Unnamed` instead, which says that there is no name.
    The dtypes are renamed by position as well.

    The keyfield follows the names, as the dtypes do. If its column gets a new name, the
    keyfield has that name. If the Daf had no column names yet, the keyfield is kept
    when it is one of the new names, so it can be given before the names are known.
    Otherwise it is cleared.

    Args:
        new_cols: The names, in order. If None, spreadsheet names are made.
        sanitize_cols: If True, make the names valid and unique.
        unnamed_prefix: The start of a name made for an empty one.

    Returns:
        This Daf, which has been changed.

    Raises:
        AttributeError: The number of names is not the number of columns.

    Examples:
        >>> Daf(lol=[[1, 2, 3]]).set_cols().columns()
        ['A', 'B', 'C']
        >>> Daf(lol=[[1, 2, 3]]).set_cols(['a', 'a', '']).columns()
        ['a', 'a_1', 'col2']
        >>> daf = Daf(lol=[[1, 'x']], keyfield='id').set_cols(['id', 'v'])
        >>> daf
        <BLANKLINE>
        | id | v  |
        | -: | -: |
        |  1 |  x |
        <BLANKLINE>
        %% daf rows=1; cols=2; keyfield='id'; name=''
        <BLANKLINE>
        >>> daf.keyfield
        'id'
        >>> daf = Daf(lol=[[1, 'x']], cols=['a', 'b'], keyfield='a').set_cols(['id', 'v'])
        >>> daf
        <BLANKLINE>
        | id | v  |
        | -: | -: |
        |  1 |  x |
        <BLANKLINE>
        %% daf rows=1; cols=2; keyfield='id'; name=''
        <BLANKLINE>
        >>> daf.keyfield
        'id'
    """

    num_cols = self.num_cols() or len(self.hd)

    if new_cols is None:
        new_cols = daf_utils._generate_spreadsheet_column_names_list(num_cols)

    elif sanitize_cols:
        new_cols = daf_utils._sanitize_cols(new_cols, unnamed_prefix=unnamed_prefix)

    if num_cols and len(new_cols) != num_cols:
        raise AttributeError("Length of new_cols not the same as existing cols")

    # The keyfield follows the names by position, as the dtypes do. With no names yet, it is kept if it is one of the new names.
    old_cols = list(self.hd)
    self._remap_keyfield(dict(zip(old_cols, new_cols)) if old_cols else {col: col for col in new_cols})

    # set new cols to the hd
    self._cols_to_hd(new_cols)

    # convert dtypes dict to use the new names.
    if self.dtypes:
        self.dtypes = dict(zip(new_cols, self.dtypes.values()))

    return self
