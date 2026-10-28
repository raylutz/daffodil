# daf_convert.py
# copyright (c) 2026 Ray Lutz
"""
Methods of Daf for import and export.

Each function here takes the Daf as its first argument, or the class for a classmethod.
It is assigned in the Daf class body under its Daf name, without the underscore.
The docs show it under that name.

Some functions use names defined in daf.py, such as Daf, when they run.
daf.py sets those names here, as listed in _NAMES_FROM_DAF, once they exist.
An import at the top would be circular, since daf.py imports this module.
"""

from __future__ import annotations

import io
import csv
import json
from pathlib import Path

from daffodil.lib.daf_types import T_buff, T_da, T_doda, T_dola, T_donpa, T_dtype_dict, T_la, T_loda, T_lota, T_ls, T_npa, T_ta
import daffodil.lib.daf_utils    as daf_utils
import daffodil.lib.daf_ods      as daf_ods
from daffodil.lib.daf_utils import _MISSING, NULL
from daffodil.keyedlist import KeyedList

from typing import List, Dict, Any, Tuple, cast, Callable, TYPE_CHECKING
from collections.abc import Iterable, Iterator

if TYPE_CHECKING:       # for mypy. daf.py sets the names below when the program runs.
    from daffodil.daf import Daf, KeysDisabledError, _keys_mismatch, _short_list

# The names of daf.py that the functions here use when they run.
# daf.py sets each one here once it exists. An import at the top would be circular.
_NAMES_FROM_DAF: Tuple[str, ...] = ('Daf', 'KeysDisabledError', '_keys_mismatch', '_short_list')


# ==== Python lod (list of dictionaries)
def _from_lod(
        cls: 'type[Daf]',
        records_lod:    T_loda,                             # List[List[Any]] to initialize the lol data array.
        *,
        keyfield:       str | int | T_ta | T_la = '',   # set a new keyfield or set no keyfield.
        dtypes:         T_dtype_dict | None         = None, # set the data types for each column.
        name:           str                         = '',   # Optional name of the daffodil instance.
        cols:           T_ls | None                 = None, # Optionally use cols to define column names
        ignore_extra_keys: bool                     = False,# With cols or dtypes, leave out keys that are not columns, and do not raise.
        fast:           bool                        = False,# Every dict has all the columns, in column order: no checks.
        ) -> 'Daf':
    """
    Make a Daf from a list of dicts, one dict for each row.

    Without `cols` or `dtypes`, the columns are all the keys of all the dicts, in the
    order in which they first appear. The keys of the first dict come first. A dict
    that lacks a key gets NULL there. A key that first appears in a later dict adds a
    column at the right, and the rows before it get NULL there. You do not need to
    look through the data first. Empty dicts and items that are not dicts are skipped.

    If `cols` is given, those are the columns. Otherwise the keys of `dtypes` are
    the columns. You chose the columns, so a dict with a key that is not one of them
    raises `ValueError`, which names the keys, because that value would be lost.
    Pass `ignore_extra_keys=True` to leave such keys out on purpose, as when you
    pick a few columns of wide records.

    With `fast=True` the first dict is checked fully, and every dict for its number of
    keys. Use it when you built every dict
    yourself, with the same keys in the same order, as in a loop. Each row is the values
    of a dict, in the dict's own order. The columns are `cols`, or the keys of `dtypes`,
    or the keys of the first dict. With `cols` or `dtypes`, the first dict must have the
    columns as its keys, in order, or it raises `ValueError`. A later dict with the wrong
    number of keys raises `ValueError`. A later dict with its keys in another order is added
    anyway, and its values land in the wrong columns.

    Args:
        records_lod: The rows, as dicts.
        keyfield: Column, or tuple or list of columns, whose values identify rows.
        dtypes: Type for each column. When given, it also selects the columns.
        name: Name of the new Daf.
        cols: Column names to use, in order.
        ignore_extra_keys: With `cols` or `dtypes`, leave out keys that are not columns, and do not raise.
        fast: If True, take each dict's values as a row. Check the first dict fully, and every dict's length. See above.

    Returns:
        The new Daf.

    Raises:
        ValueError: `cols` or `dtypes` is given, a dict has a key that is not one of those columns,
            and `ignore_extra_keys` is not True.

    Examples:
        >>> Daf.from_lod([{'a': 1, 'b': 2}, {'a': 3}])
        <BLANKLINE>
        | a  | b  |
        | -: | -: |
        |  1 |  2 |
        |  3 |    |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield=''; name=''
        <BLANKLINE>
        >>> Daf.from_lod([{'a': 1}, {'a': 2, 'b': 5}, {'c': 9}])
        <BLANKLINE>
        | a  | b  | c  |
        | -: | -: | -: |
        |  1 |    |    |
        |  2 |  5 |    |
        |    |    |  9 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield=''; name=''
        <BLANKLINE>
        >>> Daf.from_lod([{'a': 1, 'b': 2}], cols=['b', 'a'])
        <BLANKLINE>
        | b  | a  |
        | -: | -: |
        |  2 |  1 |
        <BLANKLINE>
        %% daf rows=1; cols=2; keyfield=''; name=''
        <BLANKLINE>
        >>> Daf.from_lod([{'a': 1, 'b': 2, 'c': 3}], cols=['a', 'b'], ignore_extra_keys=True)
        <BLANKLINE>
        | a  | b  |
        | -: | -: |
        |  1 |  2 |
        <BLANKLINE>
        %% daf rows=1; cols=2; keyfield=''; name=''
        <BLANKLINE>

        Give the keyfield and a name. The values stay as they are, so text stays text:

        >>> stock = Daf.from_lod([{'id': 'a1', 'qty': '3'}, {'id': 'b2', 'qty': '10'}], keyfield='id', name='stock')
        >>> stock.name, stock.select_record('b2')
        ('stock', {'id': 'b2', 'qty': '10'})

        `dtypes` records the types, and selects the columns. It does not convert. Call `apply_dtypes()`
        for that:

        >>> typed = Daf.from_lod([{'id': 'a1', 'qty': '3'}], dtypes={'id': str, 'qty': int})
        >>> typed.lol
        [['a1', '3']]
        >>> typed.apply_dtypes().lol
        [['a1', 3]]

        With `fast=True`, each dict's values become the row as they are. Every dict must have the keys
        in the same order. Only the first is checked fully:

        >>> Daf.from_lod([{'id': 'a1', 'qty': 3}, {'id': 'b2', 'qty': 10}], fast=True).lol
        [['a1', 3], ['b2', 10]]
    """

    """ Create Daf instance from loda type, adopting dict keys as column names
            Generally, all dicts in records_lod should be the same OR the first one must have all keys
                and others can be missing keys.
            However, if dtypes is provided, it will be used to establish the columns.

            my_daf = Daf.from_lod(sample_lod)
        """
    # test exists in test_daf.py


    if dtypes is None:
        dtypes = {}

    if cols is None:
        cols = []

    if not records_lod:
        return cls(cols=cols, keyfield=keyfield, dtypes=dtypes)

    if fast:
        # the caller promises the same keys, in column order, in every dict. Only the first is checked.
        if not cols:
            cols = list(dtypes.keys()) if dtypes else list(records_lod[0].keys())
        elif list(records_lod[0]) != list(cols):
            raise ValueError(
                f"from_lod(fast=True): dict 0 does not match the columns {_short_list(list(cols))}: "
                f"{_keys_mismatch(records_lod[0], list(cols))}. With fast=True the first dict must have "
                f"the columns as its keys, in order. Leave out fast to place the values by key.")
        num_cols = len(cols)
        lol = [list(record_da.values()) for record_da in records_lod]
        for irow, row_la in enumerate(lol):
            if len(row_la) != num_cols:
                raise ValueError(
                    f"from_lod(fast=True): dict {irow} has {len(row_la)} keys for {num_cols} columns: "
                    f"{_keys_mismatch(records_lod[irow], list(cols))}. With fast=True every dict must have "
                    f"the same keys as the columns, in the same order. Leave out fast to give a missing key NULL.")
        return cls(cols=cols, lol=lol, keyfield=keyfield, dtypes=dtypes, name=name)

    if cols or dtypes:
        # the caller chose the columns. A key that is not one of them would be lost, so stop and say so,
        # unless the caller said to leave such keys out. Each row is built directly, without a dict for each record.
        if not cols:
            cols = list(dtypes.keys())

        if ignore_extra_keys:
            lol = [[record_da.get(col, NULL) for col in cols]
                    for record_da in records_lod if record_da and isinstance(record_da, dict)]
        else:
            colset = set(cols)
            lol = []
            for record_da in records_lod:
                if record_da and isinstance(record_da, dict):
                    if not record_da.keys() <= colset:
                        extra_keys = [key for key in record_da if key not in colset]
                        raise ValueError(
                            f"from_lod: a record has keys that are not in cols or dtypes: {extra_keys[:5]}. "
                            f"Add them to cols, or pass ignore_extra_keys=True to leave them out.")
                    lol.append([record_da.get(col, NULL) for col in cols])

    else:
        # the columns are the keys of all the records, in the order of their first appearance. A record
        # that lacks a column has NULL there. A key that is new adds a column at the right. The rows that
        # were built before are then shorter, and are padded once at the end. Nothing is padded, and no
        # second pass is made, when the first record has all the keys. Checking the keys of each record
        # against a set of the columns is done in C, and building the row directly avoids a dict per record.
        cols    = []
        colset  = set()
        lol     = []
        grew    = False

        for record_da in records_lod:
            if record_da and isinstance(record_da, dict):
                if not record_da.keys() <= colset:
                    for key in record_da:
                        if key not in colset:
                            cols.append(key)
                            colset.add(key)
                    if lol:
                        grew = True
                lol.append([record_da.get(col, NULL) for col in cols])

        if grew:
            num_cols = len(cols)
            for row_la in lol:
                if len(row_la) < num_cols:
                    row_la.extend([NULL] * (num_cols - len(row_la)))

    # following invalidates kd for lazy rebuilding.
    return cls(cols=cols, lol=lol, keyfield=keyfield, dtypes=dtypes, name=name)


# ==== Python lot (list of tuples)
def _from_lot(
        cls: 'type[Daf]',
        records_lot:    T_lota,                             # List of tuples to initialize the lol data array.
        cols:           T_ls | None                 = None, # Optional column names to use.
        dtypes:         T_dtype_dict | None         = None, # Optional dtype_dict describing the desired type of each column.
                                                            #   also used to define column names if provided and cols not provided.
        keyfield:       str | int | T_ta | T_la = '',   # A field of the columns to be used as a key.
        name:           str                         = '',   # Optional name of the daffodil instance.
        ) -> 'Daf':
    """
    Make a Daf from a list of tuples, one tuple for each row.

    Without `cols` the Daf has no column names, as with `Daf(lol=...)`. Call
    `set_cols()` to give it spreadsheet names such as `A` and `B`, or to name them. A
    Daf with no names cannot give its rows as dicts or KeyedList objects.

    Args:
        records_lot: The rows, as tuples.
        cols: Column names. They must be as many as the items in each tuple.
        dtypes: Type for each column.
        keyfield: Column, or tuple or list of columns, whose values identify rows.
        name: Name of the new Daf.

    Returns:
        The new Daf.

    Raises:
        ValueError: A tuple does not have as many items as there are columns.

    Examples:
        >>> Daf.from_lot([(1, 'a'), (2, 'b')], cols=['id', 'v'])
        <BLANKLINE>
        | id | v  |
        | -: | -: |
        |  1 |  a |
        |  2 |  b |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield=''; name=''
        <BLANKLINE>
        >>> Daf.from_lot([(1, 'a')]).columns()
        []
        >>> Daf.from_lot([(1, 'a')]).set_cols().columns()
        ['A', 'B']

        Give the keyfield, the types and a name:

        >>> stock = Daf.from_lot([('a1', 3), ('b2', 10)], cols=['id', 'qty'], keyfield='id', dtypes={'id': str, 'qty': int}, name='stock')
        >>> stock.keyfield, stock.name, stock.select_record('b2')
        ('id', 'stock', {'id': 'b2', 'qty': 10})
    """
    """
        Create Daf instance from LOT (list of tuples), adopting given column names or generating default ones.

        Args:
            records_lot (List[Tuple[Any, ...]]):                    The LOT data.
            columns (List[str] | None):                          Column names for the tuples. If None, default names will be generated.
            keyfield (str | int | Tuple[Any, ...] | List[Any]): Keyfield for the Daf instance.
            dtypes (Dict[str, type] | None):                     Data types for each column.

        Returns:
            Daf: A new Daf instance with data populated from lot.

        Example:
            records_lot = [(1, 'Alice', 30), (2, 'Bob', 25)]
            columns = ['id', 'name', 'age']
            my_daf = Daf.from_lot(records_lot, cols=cols)
        """
    if not records_lot:
        return cls(keyfield=keyfield, dtypes=dtypes)

    if dtypes is None:
        dtypes = {}

    # Check for mismatch between column names and tuple length, or between the tuples.
    num_cols = len(cols) if cols is not None else len(records_lot[0])
    if any(len(row) != num_cols for row in records_lot):
        raise ValueError("Each tuple in records_lot must have the same number of elements as the columns.")

    # Convert LOT to LOL (list of lists) for Daf
    lol = [list(row) for row in records_lot]

    # following invalidates kd for lazy rebuilding.
    return cls(cols=cols, lol=lol, keyfield=keyfield, dtypes=dtypes, name=name)


def _to_daf(self: 'Daf') -> 'Daf':
    """
    Give this Daf itself, so code can end the same way on a Daf or a SqlDaf.

    `SqlDaf.to_daf()` reads the rows of a table into memory. A Daf is in memory already, so
    nothing is copied. That lets one line read a table of either kind, as in
    `table.select_records_daf(keys).to_daf().apply_dtypes()`.

    This Daf is returned, not a copy. A change to the result changes this Daf. A SqlDaf gives
    rows of their own instead. Use `copy('editable')` for rows of your own.

    Returns:
        This Daf.

    Examples:
        >>> daf = Daf(cols=['id', 'v'], lol=[['a1', 1], ['b2', 2]], keyfield='id')
        >>> daf
        <BLANKLINE>
        | id | v  |
        | -: | -: |
        | a1 |  1 |
        | b2 |  2 |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield='id'; name=''
        <BLANKLINE>
        >>> daf.select_records_daf(['b2']).to_daf().lol
        [['b2', 2]]
        >>> daf.to_daf() is daf
        True
    """
    return self


def _to_lod(self: 'Daf') -> T_loda:
    """
    Make a list of dicts, one dict for each row.

    The dicts are new, but the values in them are the same objects as in the Daf.
    An empty Daf gives an empty list. See `iter_dict()` to go through rows one at
    a time without building the whole list.

    Returns:
        The rows as dicts.

    Examples:
        >>> daf = Daf(lol=[[1, 'a'], [2, ['x']]], cols=['id', 'v'])
        >>> daf
        <BLANKLINE>
        | id |   v   |
        | -: | ----: |
        |  1 |     a |
        |  2 | ['x'] |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield=''; name=''
        <BLANKLINE>
        >>> lod = daf.to_lod()
        >>> lod
        [{'id': 1, 'v': 'a'}, {'id': 2, 'v': ['x']}]
        >>> lod[1]['v'].append('y')
        >>> daf.iloc(1)
        {'id': 2, 'v': ['x', 'y']}
    """
    # test exists in test_daf.py

    if not self:
        return []

    if not self.hd:
        raise KeysDisabledError("to_lod(): the Daf has no column names. Call set_cols() to name them.")

    cols = self.columns()
    result_lod = [dict(zip(cols, la)) for la in self.lol]

    return result_lod


# ==== Python dod (dict of dict)
def _from_dod(
        cls: 'type[Daf]',
        dod:            T_doda,         # Dict(str, Dict(str, Any))
        keyfield:       str='rowkey',   # The keyfield will be set to the keys of the outer dict.
                                        # this will set the preferred name. Defaults to 'rowkey'
        dtypes:         T_dtype_dict | None = None,    # optionally set the data types for each column.
        ignore_extra_keys: bool=False,                  # With dtypes, leave out keys that dtypes does not name, and do not raise.
        ) -> 'Daf':

    """
    Make a Daf from a dict of dicts, where the outer key names the row.

    A dict of dicts usually does not repeat the row key inside each row. A Daf
    table always has it as a column. If the inner dicts lack the `keyfield`
    column, it is added from the outer keys. The new Daf has that keyfield.

    If `dtypes` is given, its keys are the columns, and it must name the `keyfield`
    column too. An inner dict with a key that `dtypes` does not name raises
    `ValueError`, which names the keys, because that value would be lost. Pass
    `ignore_extra_keys=True` to leave such keys out on purpose.

    Use `to_dod()` to go back.

    Args:
        dod: A dict that maps a row key to a dict of that row.
        keyfield: The column that holds the outer key.
        dtypes: Type for each column. When given, it also selects the columns.
        ignore_extra_keys: With `dtypes`, leave out keys that it does not name, and do not raise.

    Returns:
        The new Daf.

    Raises:
        ValueError: `dtypes` is given, an inner dict has a key that it does not name, and
            `ignore_extra_keys` is not True.
        KeyError: `dtypes` does not name the `keyfield` column.

    Examples:
        >>> daf = Daf.from_dod({'r0': {'x': 1}, 'r1': {'x': 2}})
        >>> daf
        <BLANKLINE>
        | rowkey | x  |
        | -----: | -: |
        |     r0 |  1 |
        |     r1 |  2 |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield='rowkey'; name=''
        <BLANKLINE>
        >>> dod = {'r0': {'x': 1, 'id': 7}, 'r1': {'x': 2, 'id': 8}}
        >>> Daf.from_dod(dod, dtypes={'rowkey': str, 'x': int}, ignore_extra_keys=True)
        <BLANKLINE>
        | rowkey | x  |
        | -----: | -: |
        |     r0 |  1 |
        |     r1 |  2 |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield='rowkey'; name=''
        <BLANKLINE>

        The outer keys go into a column named by `keyfield`, which becomes the keyfield:

        >>> stock = Daf.from_dod({'a1': {'qty': 3}, 'b2': {'qty': 10}}, keyfield='id')
        >>> stock
        <BLANKLINE>
        | id | qty |
        | -: | --: |
        | a1 |   3 |
        | b2 |  10 |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield='id'; name=''
        <BLANKLINE>
    """

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
    # following invalidates kd for lazy rebuilding.
    return cls.from_lod(daf_utils.dod_to_lod(dod, keyfield=keyfield), keyfield=keyfield, dtypes=dtypes, ignore_extra_keys=ignore_extra_keys)


def _to_dod(
        self: 'Daf',
        remove_keyfield:    bool=True,      # by default, the keyfield column is removed.
        ) -> T_doda:
    """
    Make a dict of dicts, where the keyfield value names each row.

    The keyfield column is left out of the inner dicts by default. Pass
    `remove_keyfield=False` to keep it there too. The Daf must have a keyfield.
    Without one, a `KeysDisabledError` is raised, unless the Daf has no rows.

    Args:
        remove_keyfield: If True, do not repeat the key inside each inner dict.

    Returns:
        A dict that maps each key to a dict of that row.

    Raises:
        KeysDisabledError: The Daf has no keyfield.

    Examples:
        >>> daf = Daf(lol=[[1, 'a']], cols=['id', 'v'], keyfield='id')
        >>> daf
        <BLANKLINE>
        | id | v  |
        | -: | -: |
        |  1 |  a |
        <BLANKLINE>
        %% daf rows=1; cols=2; keyfield='id'; name=''
        <BLANKLINE>
        >>> daf.to_dod()
        {1: {'v': 'a'}}
        >>> daf.to_dod(remove_keyfield=False)
        {1: {'id': 1, 'v': 'a'}}
    """

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
    if self and not self.keyfield:
        raise KeysDisabledError("to_dod(): the Daf has no keyfield.")

    # lod_to_dod() does a single dict-key lookup per row (da[keyfield]), which only makes
    # sense for a simple str keyfield, not daf's own broader composite (tuple/list) keyfield
    # support -- this method is only meaningful when self.keyfield is a plain str.
    return daf_utils.lod_to_dod(self.to_lod(), keyfield=cast(str, self.keyfield), remove_keyfield=remove_keyfield)


# ==== cols_dol
def _from_cols_dol(
        cls: 'type[Daf]',
        cols_dol: T_dola,
        keyfield: str='',
        dtypes: T_dtype_dict | None = None,
        ) -> 'Daf':
    """
    Make a Daf from a dict of lists, where each list is a column.

    All the lists must be as long as each other. If one differs in length from the
    first, a `ValueError` is raised, because the rows could not be made without
    losing values or guessing them.

    Args:
        cols_dol: Maps a column name to the list of its values.
        keyfield: Column, or tuple or list of columns, whose values identify rows.
        dtypes: Type for each column.

    Returns:
        The new Daf.

    Raises:
        ValueError: A list has a different length from the first.

    Examples:
        >>> daf = Daf.from_cols_dol({'A': [1, 2, 3], 'B': [4, 5, 6]})
        >>> daf
        <BLANKLINE>
        | A  | B  |
        | -: | -: |
        |  1 |  4 |
        |  2 |  5 |
        |  3 |  6 |
        <BLANKLINE>
        %% daf rows=3; cols=2; keyfield=''; name=''
        <BLANKLINE>

        Give the keyfield and the types:

        >>> stock = Daf.from_cols_dol({'id': ['a1', 'b2'], 'qty': [3, 10]}, keyfield='id', dtypes={'id': str, 'qty': int})
        >>> stock.select_record('b2'), stock.dtypes
        ({'id': 'b2', 'qty': 10}, {'id': <class 'str'>, 'qty': <class 'int'>})
    """
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
        # following invalidates kd for lazy rebuilding.
        return cls(keyfield=keyfield, dtypes=dtypes)

    cols = list(cols_dol.keys())

    num_rows = len(cols_dol[cols[0]])
    for col in cols:
        if len(cols_dol[col]) != num_rows:
            raise ValueError(
                f"from_cols_dol: column '{col}' has {len(cols_dol[col])} values, "
                f"but column '{cols[0]}' has {num_rows}.")

    # zip turns the columns into rows in C.
    lol = [list(row) for row in zip(*cols_dol.values())]

    # following invalidates kd for lazy rebuilding.
    return cls(cols=cols, lol=lol, keyfield=keyfield, dtypes=dtypes)


def _to_cols_dol(self: 'Daf') -> dict:
    """
    Make a dict of lists, one list of values for each column.

    Returns:
        A dict that maps each column name to the list of its values.

    Examples:
        >>> Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v']).to_cols_dol()
        {'id': [1, 2], 'v': ['a', 'b']}
    """

    """ convert daf to dictionary of lists of values, where key is the
            column name, and the list are the values in that column.
        """
    result_dol: Dict[str, list] = {colname: [] for colname in self.columns()}

    for row_da in self:
        for key, val in row_da.items():
            result_dol[key].append(val)     # type: ignore[index]  # a row's keys are column names (str)

    return result_dol


def _to_attrib_dict(self: 'Daf') -> dict:
    """
    Make a dict with the column names and the rows.

    This is deprecated. It keeps the rows as they are, not copied. Use
    `to_lod()` or `to_cols_dol()` instead.

    Returns:
        A dict with the keys `cols` and `lol`.

    Examples:
        >>> Daf(lol=[[1, 'a']], cols=['id', 'v']).to_attrib_dict()
        {'cols': ['id', 'v'], 'lol': [[1, 'a']]}
    """
    return {'cols': self.columns(), 'lol': self.lol}


def _from_lod_to_cols(
        cls: 'type[Daf]',
        lod:        T_loda,
        cols:       List | None = None,
        keyfield:   str='',
        dtypes:     T_dtype_dict | None = None
        ) -> 'Daf':
    """
    Make a Daf in which each dict becomes a column, not a row.

    The keys of the dicts become the first column, and each dict adds one more
    column after it. Use this to compare several results side by side, such as
    the same features measured in several tries. It is a transpose of
    `from_lod()`.

    Without `cols`, the first column is named `key` and the others `A`, `B` and
    so on. The names in `cols` include the first column. The keyfield is not set
    unless `keyfield` is given.

    Args:
        lod: The dicts. They should all have the same keys.
        cols: Column names, starting with the name of the column of keys.
        keyfield: Column whose values identify rows.
        dtypes: Type for each column of the dicts, before the change. It must name every key of the dicts.

    Returns:
        The new Daf.

    Raises:
        ValueError: `dtypes` is given and a dict has a key that it does not name. The keys would be
            matched to the wrong values. The message mentions `ignore_extra_keys`, which this method does not take.
            Name all the keys.

    Examples:
        >>> daf = Daf.from_lod_to_cols([{'A': 1, 'B': 2}, {'A': 4, 'B': 5}], cols=['Feature', 'T1', 'T2'])
        >>> daf
        <BLANKLINE>
        | Feature | T1 | T2 |
        | ------: | -: | -: |
        |       A |  1 |  4 |
        |       B |  2 |  5 |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield=''; name=''
        <BLANKLINE>

        Name the columns, and make the column of keys the keyfield:

        >>> rows = Daf.from_lod_to_cols([{'a': 1, 'b': 2}, {'a': 3, 'b': 4}], cols=['field', 'r1', 'r2'], keyfield='field')
        >>> rows
        <BLANKLINE>
        | field | r1 | r2 |
        | ----: | -: | -: |
        |     a |  1 |  3 |
        |     b |  2 |  4 |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield='field'; name=''
        <BLANKLINE>
        >>> rows.select_record('b')
        {'field': 'b', 'r1': 2, 'r2': 4}
    """
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
        # following invalidates kd for lazy rebuilding.
        return cls(keyfield=keyfield, dtypes=dtypes, cols=cols)

    # the following will adopt the dictionary keys as cols.
    # note that dtypes applies to the columns in this orientation.
    # following invalidates kd for lazy rebuilding.
    rows_daf = cls.from_lod(lod, dtypes=dtypes)

    # this transposes the data, and puts the keys of the dicts in the first column. The keys are
    # taken from the dicts and not from the column names of rows_daf, because those names are
    # made valid and unique, and a key such as '' would come back as 'Unnamed1'. The keys are data here.
    # The column names are those given, or else ['key', 'A', 'B', ...]
    if not cols:
        cols = ['key'] + daf_utils._generate_spreadsheet_column_names_list(num_cols=len(lod))

    cols_lol = [[key] + list(values) for key, values in zip(lod[0].keys(), zip(*rows_daf.lol))]

    return cls(lol=cols_lol, cols=cols, keyfield=keyfield)


def _from_excel_buff(
        cls: 'type[Daf]',
        excel_buff: bytes,
        keyfield: str='',                       # field to use as unique key, if not ''
        dtypes: T_dtype_dict | None = None,    # dictionary of types to apply if set.
        noheader: bool=False,                   # if True, do not try to initialize columns in header dict.
        user_format: bool=False,                # if True, preprocess the file and omit comment lines.
        unflatten: bool=True,                   # unflatten fields that are defined as dict or list.
        ) -> 'Daf':
    """
    Make a Daf from the bytes of an xlsx file.

    The first sheet is turned into CSV by the `xlsx2csv` package. The CSV is then
    read like any CSV text. All values start as text, so give `dtypes` or call
    `apply_dtypes()` to convert them.

    Short rows are not padded here. `xlsx2csv` pads them. See `xlsx_to_csv()`.

    Args:
        excel_buff: The bytes of the xlsx file.
        keyfield: Column whose values identify rows.
        dtypes: Type for each column. Deprecated: call `apply_dtypes()` on the result instead.
        noheader: If True, the first row is data, not column names.
        user_format: If True, skip comment lines and blank lines.
        unflatten: If True, read list and dict columns from their text.

    Returns:
        The new Daf.

    Examples:
        A small Excel file, made here with xlsxwriter for the example:

        >>> import io, xlsxwriter
        >>> def excel_bytes(rows):
        ...     buff = io.BytesIO()
        ...     workbook = xlsxwriter.Workbook(buff, {'in_memory': True})
        ...     sheet = workbook.add_worksheet()
        ...     for irow, row in enumerate(rows):
        ...         sheet.write_row(irow, 0, row)
        ...     workbook.close()
        ...     return buff.getvalue()
        >>> xlsx = excel_bytes([['id', 'v'], [1, 'a'], [2, 'b']])
        >>> daf = Daf.from_excel_buff(xlsx, keyfield='id')
        >>> daf
        <BLANKLINE>
        | id | v  |
        | -: | -: |
        |  1 |  a |
        |  2 |  b |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield='id'; name=''
        <BLANKLINE>
        >>> Daf.from_excel_buff(xlsx, dtypes={'id': int, 'v': str})
        <BLANKLINE>
        | id | v  |
        | -: | -: |
        |  1 |  a |
        |  2 |  b |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield=''; name=''
        <BLANKLINE>

        With `noheader=True`, the first row is data, and the Daf has no column names:

        >>> Daf.from_excel_buff(excel_bytes([['a1', 3], ['b2', 10]]), noheader=True).lol
        [['a1', '3'], ['b2', '10']]

        With `user_format=True`, comment lines, which start with `#`, and blank lines are skipped:

        >>> Daf.from_excel_buff(excel_bytes([['# stock count'], ['id', 'qty'], [], ['a1', 3]]), user_format=True).lol
        [['a1', '3']]

        A list or dict column is read from its text, for its dtype. With `unflatten=False` it stays text:

        >>> tags = excel_bytes([['id', 'tags'], ['a1', '[1, 2]']])
        >>> Daf.from_excel_buff(tags, dtypes={'id': str, 'tags': list}).lol
        [['a1', [1, 2]]]
        >>> Daf.from_excel_buff(tags, dtypes={'id': str, 'tags': list}, unflatten=False).lol
        [['a1', '[1, 2]']]
    """

    # from utilities import xlsx_utils

    csv_buff = daf_utils.xlsx_to_csv(excel_buff)

    # following invalidates kd for lazy rebuilding.
    my_daf  = cls.from_csv_buff(
                    csv_buff,
                    keyfield    = keyfield,         # field to use as unique key, if not ''
                    dtypes      = dtypes,           # dictionary of types to apply if set.
                    noheader    = noheader,         # if True, do not try to initialize columns in header dict.
                    user_format = user_format,      # if True, preprocess the file and omit comment lines.
                    unflatten   = unflatten,        # unflatten fields that are defined as dict or list.
                    )

    return my_daf


def _from_ods(
        cls: 'type[Daf]',
        source:     str | Path | bytes,
        sheet:      int | str = 0,                  # the position of the sheet, from 0, or its name.
        keyfield:   str = '',                       # field to use as unique key, if not ''
        dtypes:     T_dtype_dict | None = None,     # dictionary of types to apply if set.
        noheader:   bool = False,                   # if True, the first row is data.
        unflatten:  bool = True,                    # unflatten fields that are defined as dict or list.
        name:       str = '',                       # name attribute of the Daf array created.
        ) -> 'Daf':
    """
    Make a Daf from one sheet of an OpenDocument spreadsheet, an .ods file.

    Use it for a table kept in LibreOffice Calc. Only the first sheet is read, unless
    you give `sheet`. Only the content of the cells is read: styles, comments, links
    and formulas are left out. No package is needed, since the file is read with the
    standard library. See `lib/daf_ods.py`.

    All values start as text, as from a CSV file, so give `dtypes` or call
    `apply_dtypes()` to convert them. A number, date or boolean cell gives its stored
    value, such as `168.5` or `2026-09-01`, not the text shown, such as `$168.50`.
    A cell with several lines gives them joined with a newline.

    Rows are padded with NULL to the width of the widest row. A blank row inside the
    table is kept as a row of NULL, so the rows line up with the spreadsheet. Drop such
    rows yourself if they must not be used. The blank rows at the end are left out.

    Args:
        source: The path of the file, or its bytes.
        sheet: The position of the sheet, from 0, or its name.
        keyfield: Column whose values identify rows.
        dtypes: Type for each column. Deprecated: call `apply_dtypes()` on the result instead.
        noheader: If True, the first row is data, not column names.
        unflatten: If True, read list and dict columns from their text.
        name: The name of the new Daf.

    Returns:
        The new Daf.

    Raises:
        IndexError: There is no sheet at that position.
        KeyError: There is no sheet with that name.

    Examples:
        A small .ods file, made here for the example. A real one comes from a file or a download:

        >>> import io, zipfile
        >>> def ods_bytes(sheets):
        ...     ns = ('xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
        ...           'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
        ...           'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"')
        ...     body = ''
        ...     for name, rows in sheets.items():
        ...         body += f'<table:table table:name="{name}">'
        ...         for row in rows:
        ...             cells = ''.join(f'<table:table-cell><text:p>{v}</text:p></table:table-cell>' for v in row)
        ...             body += f'<table:table-row>{cells}</table:table-row>'
        ...         body += '</table:table>'
        ...     content = (f'<office:document-content {ns}><office:body><office:spreadsheet>'
        ...                f'{body}</office:spreadsheet></office:body></office:document-content>')
        ...     buff = io.BytesIO()
        ...     with zipfile.ZipFile(buff, 'w') as odsfile:
        ...         odsfile.writestr('mimetype', 'application/vnd.oasis.opendocument.spreadsheet')
        ...         odsfile.writestr('content.xml', content)
        ...     return buff.getvalue()
        >>> ods = ods_bytes({'stock': [['id', 'qty'], ['a1', '3'], ['b2', '10']], 'notes': [['a1', 'new']]})

        The first sheet, with a keyfield and a name. The cells are read as text:

        >>> stock = Daf.from_ods(ods, keyfield='id', name='stock')
        >>> stock
        <BLANKLINE>
        | id | qty |
        | -: | --: |
        | a1 |   3 |
        | b2 |  10 |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield='id'; name='stock'
        <BLANKLINE>

        Another sheet, by name or by position, whose first row is data:

        >>> Daf.from_ods(ods, sheet='notes', noheader=True).lol
        [['a1', 'new']]
        >>> Daf.from_ods(ods, sheet=1, noheader=True).lol
        [['a1', 'new']]
    """
    data_lol = daf_ods.ods_to_lol(source, sheet=sheet)
    cols: T_ls = []
    if not noheader and data_lol:
        cols = data_lol.pop(0)
    my_daf = cls(lol=data_lol, cols=cols, keyfield=keyfield, dtypes=dtypes, name=name)
    my_daf.apply_dtypes(unflatten=unflatten)
    return my_daf


def _from_csv(cls: 'type[Daf]', source: str | Path, **kwargs: Any) -> 'Daf':
    r"""
    Read a CSV file, a web address or an S3 object into a Daf.

    The source may be a path, a `Path`, an `http` or `https` address, or an
    `s3://bucket/key` name. The `requests` and `boto3` packages are imported only
    when they are needed.

    Every value is read as text. Call `apply_dtypes()` on the result to convert
    them. Giving `dtypes` here also converts them, but that is deprecated. The other keyword arguments, such as `keyfield`, are those of
    `from_csv_buff()`. The length of each row is not checked. See `from_csv_buff()`.

    Args:
        source: A file path, an `http` address or an `s3://` name.
        **kwargs: Passed on to `from_csv_buff()`.

    Returns:
        The new Daf.

    Raises:
        RuntimeError: The download fails, a needed package is missing, or the
            local file cannot be read or parsed. The message says which.

    Examples:
        >>> import os, tempfile
        >>> path = os.path.join(tempfile.mkdtemp(), 'x.csv')
        >>> with open(path, 'w') as f:
        ...     n = f.write('id,v\n1,a\n')
        >>> Daf.from_csv(path)
        <BLANKLINE>
        | id | v  |
        | -: | -: |
        |  1 |  a |
        <BLANKLINE>
        %% daf rows=1; cols=2; keyfield=''; name=''
        <BLANKLINE>
    """
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

    if source.startswith('http'):  # Handle HTTP(S) URLs
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

    # following invalidates kd for lazy rebuilding.
    new_daf = cls.from_csv_buff(csv_buff=data_stream, **kwargs)

    return new_daf


# STILL ACTIVE -- use when we know the source is a buffer.
def _from_csv_buff(
        cls: 'type[Daf]',
        csv_buff: bytes | str | Iterator[str], # Can now accept iterators directly
        keyfield: str='',                           # field to use as unique key, if not ''
        dtypes: T_dtype_dict | None = None,        # dictionary of types to apply if set.
        noheader: bool=False,                       # if True, do not try to initialize columns in header dict.
        user_format: bool=False,                    # if True, preprocess the file and omit comment lines.
        sep: str=',',                               # field separator.
        unflatten: bool=True,                       # unflatten fields that are defined as dict or list.
        include_cols: T_ls | None = None,          # include only the columns specified. noheader must be false.
        name: str = '',                             # name attribute of the Daf array created.
        ) -> 'Daf':
    r"""
    Make a Daf from CSV text, bytes or an iterator of lines.

    The first row holds the column names, unless `noheader` is True. Quoted fields
    may hold commas. Empty rows at the end are dropped. Cells are text, unless
    `dtypes` is given. Then the cells are converted, and list and dict columns
    are read from their text unless `unflatten` is False.

    Converting here is deprecated. It does no more than calling `apply_dtypes()` on the
    result, which is clearer: a reader only reads, and the types are applied where they
    are needed. Pass no `dtypes`, and call `apply_dtypes()` afterwards. The conversion
    still works, for now.

    Empty text, or a source with no rows, gives an empty Daf with no columns, as
    `from_md()` does. Check `len()` of the result if an empty source would be an error.

    A `keyfield` that is not a column of the file raises `KeyError`.

    The length of each row is not checked, so that a read costs no more than it must.
    A row with a missing cell, or with an extra one from an unquoted comma, is kept as
    it is, and a later call may fail with an error that does not mention it. If you do
    not trust the source, call `is_rectangular()`. It looks at every row. To pad short
    rows, call `force_rectangular()`. It raises `ValueError` for a row that is too long.

    Args:
        csv_buff: The CSV, as text, bytes or an iterator of lines.
        keyfield: Column, or tuple or list of columns, whose values identify rows.
        dtypes: Type for each column. Deprecated: call `apply_dtypes()` on the result instead.
        noheader: If True, the first row is data, and the Daf has no column names.
        user_format: If True, skip comment lines and blank lines.
        sep: The character that separates fields.
        unflatten: If True, read list and dict columns from their text.
        include_cols: Keep only these columns, in this order. The cells of the other
            columns are never kept, so a wide file costs little memory.
        name: Name of the new Daf.

    Returns:
        The new Daf.

    Raises:
        KeyError: A name in `include_cols`, or the `keyfield`, is not in the header of the file.
        ValueError: `include_cols` is given with `noheader=True`, because the names
            of the columns come from the header.

    Examples:
        >>> daf = Daf.from_csv_buff('id,v\n1,a\n2,"b,c"\n')
        >>> daf
        <BLANKLINE>
        | id |  v  |
        | -: | --: |
        |  1 |   a |
        |  2 | b,c |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield=''; name=''
        <BLANKLINE>
        >>> Daf.from_csv_buff('id,v\n1,a\n').apply_dtypes(dtypes={'id': int, 'v': str})
        <BLANKLINE>
        | id | v  |
        | -: | -: |
        |  1 |  a |
        <BLANKLINE>
        %% daf rows=1; cols=2; keyfield=''; name=''
        <BLANKLINE>
        >>> daf = Daf.from_csv_buff('a,b,c\n1,2,3\n4,5,6\n', include_cols=['c', 'a'])
        >>> daf
        <BLANKLINE>
        | c  | a  |
        | -: | -: |
        |  3 |  1 |
        |  6 |  4 |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield=''; name=''
        <BLANKLINE>
        >>> daf = Daf.from_csv_buff('id,v\n1,a\n2\n')
        >>> daf
        <BLANKLINE>
        | id | v  |
        | -: | -: |
        |  1 |  a |
        |  2 |    |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield=''; name=''; ragged rows detected
        <BLANKLINE>
        >>> daf.is_rectangular()
        False
        >>> daf.force_rectangular()
        <BLANKLINE>
        | id | v  |
        | -: | -: |
        |  1 |  a |
        |  2 |    |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield=''; name=''
        <BLANKLINE>

        With `user_format=True`, comment lines, which start with `#`, and blank lines are skipped:

        >>> Daf.from_csv_buff('# stock count\nid,qty\n\na1,3\nb2,10\n', user_format=True).lol
        [['a1', '3'], ['b2', '10']]

        Another separator, and no header line:

        >>> Daf.from_csv_buff('a1;3\nb2;10\n', sep=';', noheader=True).lol
        [['a1', '3'], ['b2', '10']]

        Keep only some columns, in a new order, with a keyfield and a name:

        >>> picked = Daf.from_csv_buff('id,qty,note\na1,3,x\n', include_cols=['qty', 'id'], keyfield='id', name='picked')
        >>> picked
        <BLANKLINE>
        | qty | id |
        | --: | -: |
        |   3 | a1 |
        <BLANKLINE>
        %% daf rows=1; cols=2; keyfield='id'; name='picked'
        <BLANKLINE>

        A list column is read from its text, for its dtype. With `unflatten=False` it stays text:

        >>> Daf.from_csv_buff('id,tags\na1,"[1, 2]"\n', dtypes={'id': str, 'tags': list}).lol
        [['a1', [1, 2]]]
        >>> Daf.from_csv_buff('id,tags\na1,"[1, 2]"\n', dtypes={'id': str, 'tags': list}, unflatten=False).lol
        [['a1', '[1, 2]']]
    """

    """
        Convert CSV data in a buffer (string, bytes, or iterator) to a daf object

        Note: probably just use from_csv() and then apply_dtypes(), remove 'unflatten',
            however, in the future of a more optimized conversion is written, then python
            types can be created as the csv is scanned rather than scanning again.

        - Fully supports streaming CSVs (does not require full file in memory).
        - Directly reads data into a list of lists (LoL) from an iterator.

        Args:
            csv_buff: bytes | str | Iterator[str], # Can now accept iterators directly
            keyfield: str='',                           # field to use as unique key, if not ''
            dtypes: T_dtype_dict | None = None,        # dictionary of types to apply if set.
            noheader: bool=False,                       # if True, do not try to initialize columns in header dict.
            user_format: bool=False,                    # if True, preprocess the file and omit comment lines.
            sep: str=',',                               # field separator.
            unflatten: bool=True,                       # unflatten fields that are defined as dict or list.
            include_cols: T_ls | None = None,          # include only the columns specified. noheader must be false.
            name: str = '',                             # name attribute of the Daf array created.

        Returns:
            Daf: The loaded Daf array.
        """

    if include_cols and noheader:
        raise ValueError("from_csv_buff: include_cols needs the header row to find the columns, so noheader must be False.")

    # in the case of bytes, this will set up a conversion of the stream.
    # Converts bytes into a file-like object without reading everything at once.
    if isinstance(csv_buff, bytes):
        csv_buff = io.TextIOWrapper(io.BytesIO(csv_buff), encoding="utf-8")

    # the following will be able to stream from the source and convert directly to lol.
    data_lol = daf_utils.buff_csv_to_lol(csv_buff, user_format=user_format, sep=sep, include_cols=include_cols, dtypes=dtypes)

    # Remove trailing empty list rows sometimes caused by trailing newlines
    while data_lol and data_lol[-1] == []:
        data_lol.pop()

    cols = []
    if not noheader and data_lol:
        cols = data_lol.pop(0)        # return the first item and shorten the list. No rows gives an empty Daf.

    # following invalidates kd for lazy rebuilding.
    my_daf = cls(lol=data_lol, cols=cols, keyfield=keyfield, dtypes=dtypes, name=name)

    # the following will act nicely if there is no dtypes defined.
    my_daf.apply_dtypes(unflatten=unflatten)

    return my_daf


# DEPRECATED
def _from_csv_file(
        cls: 'type[Daf]',
        filepath: str,                          # The CSV filename
        keyfield: str='',                       # field to use as unique key, if not ''
        dtypes: T_dtype_dict | None = None,    # dictionary of types to apply if set.
        noheader: bool=False,                   # if True, do not try to initialize columns in header dict.
        user_format: bool=False,                # if True, preprocess the file and omit comment lines.
        sep: str=',',                           # field separator.
        unflatten: bool=True,                   # unflatten fields that are defined as dict or list.
        include_cols: T_ls | None = None,      # include only the columns specified. noheader must be false.
        name: str = '',                         # name attribute of the Daf array created.
        ) -> 'Daf':                             # New daf instance.
    r"""
    Read a CSV file into a Daf. Deprecated, use `from_csv()`.

    It calls `from_csv()`, so it reads the file as UTF-8, and a file that cannot be read
    raises `RuntimeError`.

    Args:
        filepath: Path of the file.
        keyfield: Column, or tuple or list of columns, whose values identify rows.
        dtypes: Type for each column. Deprecated: call `apply_dtypes()` on the result instead.
        noheader: If True, the first row is data, and the Daf has no column names.
        user_format: If True, skip comment lines and blank lines.
        sep: The character that separates fields.
        unflatten: If True, read list and dict columns from their text.
        include_cols: Keep only these columns, in this order. The cells of the other
            columns are never kept, so a wide file costs little memory.
        name: Name of the new Daf.

    Returns:
        The new Daf.

    Raises:
        RuntimeError: The file cannot be read or parsed.

    Examples:
        >>> import os, tempfile
        >>> path = os.path.join(tempfile.mkdtemp(), 'x.csv')
        >>> with open(path, 'w') as f:
        ...     n = f.write('id,v\n1,a\n')
        >>> Daf.from_csv_file(path)
        <BLANKLINE>
        | id | v  |
        | -: | -: |
        |  1 |  a |
        <BLANKLINE>
        %% daf rows=1; cols=2; keyfield=''; name=''
        <BLANKLINE>

        The options are those of `from_csv_buff()`, which has an example of each. Here a file with a
        comment line, read with a keyfield, one column and a name:

        >>> import os, tempfile
        >>> path = os.path.join(tempfile.mkdtemp(), 'stock.csv')
        >>> with open(path, 'w') as csv_file:
        ...     n = csv_file.write('# stock count\nid;qty;note\na1;3;x\nb2;10;y\n')
        >>> Daf.from_csv_file(path, sep=';', user_format=True, keyfield='id', include_cols=['id', 'qty'], name='stock')
        <BLANKLINE>
        | id | qty |
        | -: | --: |
        | a1 |   3 |
        | b2 |  10 |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield='id'; name='stock'
        <BLANKLINE>
    """
    """ Read a csv file directly into a daf array in memory, per arguments.

            Args:
                filepath: str,                          # The CSV filename
                keyfield: str='',                       # field to use as unique key, if not ''
                dtypes: T_dtype_dict | None = None,    # dictionary of types to apply if set.
                noheader: bool=False,                   # if True, do not try to initialize columns in header dict.
                user_format: bool=False,                # if True, preprocess the file and omit comment lines.
                sep: str=',',                           # field separator.
                unflatten: bool=True,                   # unflatten fields that are defined as dict or list.
                include_cols: T_ls | None = None,      # include only the columns specified. noheader must be false.
                name: str = '',                         # name attribute of the Daf array created.
            Returns
                New daf instance

        """

    # following invalidates kd for lazy rebuilding.
    return Daf.from_csv(
        filepath,                               # The CSV file.
        keyfield    = keyfield,                 # field to use as unique key, if not ''
        dtypes      = dtypes,                   # dictionary of types to apply if set.
        noheader    = noheader,                 # if True, do not try to initialize columns in header dict.
        user_format = user_format,              # if True, preprocess the file and omit comment lines.
        sep         = sep,                      # field separator.
        unflatten   = unflatten,                # unflatten fields that are defined as dict or list.
        include_cols    = include_cols,         # include only the columns specified. noheader must be false.
        name        = name,                     # name attribute of the Daf array created.
        )


def _to_csv_file(
        self: 'Daf',
        file_path:          str | Path = '',
        line_terminator:    str | None = None,
        include_header:     bool=True,
        #append_if_exists:   bool=False,
        ) -> str:
    r"""
    Write the Daf to a CSV file.

    The file has a header row of column names unless `include_header` is False.
    Each cell is written as text, so lists and dicts are written in their `str()`
    form. A NULL cell is written as nothing. The default line ending is `\r\n`.

    Args:
        file_path: Where to write the file.
        line_terminator: The line ending. If None, `\r\n` is used.
        include_header: If True, write the column names first.

    Returns:
        The path that was written.

    Examples:
        >>> import os, tempfile
        >>> path = os.path.join(tempfile.mkdtemp(), 'x.csv')
        >>> daf = Daf(lol=[[1, 'a']], cols=['id', 'v'])
        >>> daf
        <BLANKLINE>
        | id | v  |
        | -: | -: |
        |  1 |  a |
        <BLANKLINE>
        %% daf rows=1; cols=2; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.to_csv_file(path) == path
        True

        Without the header line, and with another line ending:

        >>> import os, tempfile
        >>> path = os.path.join(tempfile.mkdtemp(), 'stock.csv')
        >>> stock = Daf(cols=['id', 'qty'], lol=[['a1', 3], ['b2', 10]])
        >>> stock.to_csv_file(path, line_terminator='\n', include_header=False) == path
        True
        >>> open(path).read()
        'a1,3\nb2,10\n'
    """
    if isinstance(file_path, Path):  # Convert Path object to string
        file_path = str(file_path)

    buff = self.to_csv_buff(
            line_terminator=line_terminator,
            include_header=include_header,
            )

    type(self).buff_to_file(buff, file_path=file_path, fmt='.csv') #, append=append_mode)

    return file_path


def _to_csv_buff(
        self: 'Daf',
        line_terminator: str | None = None,
        include_header: bool=True,
        ) -> str:
    r"""
    Make CSV text from the Daf.

    The text can be saved to a file or uploaded. There is no need to call
    `flatten()` first. Each cell is written with `str()`, so a list or dict is
    written as its Python text, with single quotes. The bool `True` is written as
    `True`. A NULL cell is written as nothing. The text is not JSON.

    A column converted by `apply_dtypes()` to `bool` holds 0 and 1, so it is written
    as 0 and 1. A column of Python bools is written as `True` and `False`. Both read
    back as 0 and 1 when the column has dtype `bool`. The cells are written as they
    are held, with no conversion, since any pass over them would slow every save.

    A dict whose keys are not text, such as `{1: 'a'}`, is written with its keys as they are.

    Args:
        line_terminator: The line ending. If None, `\r\n` is used.
        include_header: If True, write the column names first.

    Returns:
        The CSV text.

    Examples:
        >>> daf = Daf(lol=[[1, 'a']], cols=['id', 'v'])
        >>> daf
        <BLANKLINE>
        | id | v  |
        | -: | -: |
        |  1 |  a |
        <BLANKLINE>
        %% daf rows=1; cols=2; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.to_csv_buff(line_terminator='\n')
        'id,v\n1,a\n'
        >>> Daf(lol=[[{1: 'a'}]], cols=['x']).to_csv_buff(line_terminator='\n')
        "x\n{1: 'a'}\n"

        Without the header line, and with another line ending:

        >>> stock = Daf(cols=['id', 'qty'], lol=[['a1', 3], ['b2', 10]])
        >>> stock.to_csv_buff(include_header=False)
        'a1,3\r\nb2,10\r\n'
        >>> stock.to_csv_buff(line_terminator='\n')
        'id,qty\na1,3\nb2,10\n'
    """
    """ this function writes the daf array to a csv buffer, including the header if include_header==True.
            The buffer can be saved to a local file or uploaded to a storage service like s3.

            There is no need to call .flatten() first. This function flattens any nested
                    objects to PYON, according to __repr__ for that object.
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


def _buff_to_file(buff: T_buff, file_path: str | Path, fmt:str='.csv') -> str:
    r"""
    Write text or bytes to a file.

    This is a static method, so call it as `Daf.buff_to_file(buff, path)`. It is
    the helper that `to_csv_file()` uses.

    Args:
        buff: The text or bytes to write.
        file_path: Where to write.
        fmt: Not used.

    Returns:
        The path that was written.

    Examples:
        >>> import os, tempfile
        >>> path = os.path.join(tempfile.mkdtemp(), 'x.csv')
        >>> Daf.buff_to_file('id,v\n1,a\n', path) == path
        True
        >>> open(path).read()
        'id,v\n1,a\n'
    """
    # write_buff_to_fp() does str-only operations on file_path (.startswith('s3'), a regex
    # substitution) that would raise AttributeError on a real Path object -- str() it here
    # rather than widening write_buff_to_fp() to silently accept something it can't handle.
    return daf_utils.write_buff_to_fp(buff, str(file_path), fmt=fmt)


def _from_directory(
        cls: 'type[Daf]',
        source: str | Path,
        schema: type | None = None,
        recursive: bool = True,
        file_pat: str | None = None,
        include_dirs: bool = False,
    ) -> 'Daf':
    r"""
    Make a Daf that lists the files in a folder.

    Each row describes one file. The columns are the path, the folder, the name,
    the name without its extension, the extension, the size in bytes, and the
    modified and changed times. On Linux and macOS the `ctime` is the time of the last
    change to the file's metadata, such as its permissions, not the time it was
    created. Paths use `/` on every system.

    With `include_dirs=True`, the folders are listed too, each before the files in the
    same folder. A folder has an `is_dir` of 1, a size of 0 and no extension. Without
    it, no folder is listed and `is_dir` is always 0.

    The rows are in the order in which the file system lists the files. That order is
    not sorted, and it differs between machines. Sort the result if the order matters.

    The `schema` chooses the columns of the result. A `@schemaclass` that lists
    only some of the fields above keeps only those. Other columns of the schema
    get their defaults. The new Daf has no keyfield. Files that cannot be read
    are skipped.

    Only local folders are supported.

    Args:
        source: The folder to list.
        schema: A `@schemaclass` that chooses the columns. If None, the built in one is used.
        recursive: If True, list files in all sub folders. Otherwise only the folder itself.
        file_pat: A regular expression. Only names that match it are listed. Case is ignored.
        include_dirs: If True, list the folders as well as the files.

    Returns:
        The new Daf.

    Examples:
        >>> import os, tempfile
        >>> folder = tempfile.mkdtemp()
        >>> os.mkdir(os.path.join(folder, 'sub'))
        >>> for name in ('b.txt', 'a.csv', os.path.join('sub', 'c.csv')):
        ...     with open(os.path.join(folder, name), 'w') as f:
        ...         n = f.write('hello')
        >>> daf = Daf.from_directory(folder)
        >>> sorted(daf.col('basename')), daf.col('size')
        (['a.csv', 'b.txt', 'c.csv'], [5, 5, 5])
        >>> sorted(Daf.from_directory(folder, recursive=False).col('basename'))
        ['a.csv', 'b.txt']
        >>> sorted(Daf.from_directory(folder, file_pat=r'\.csv$').col('basename'))
        ['a.csv', 'c.csv']
        >>> sorted(Daf.from_directory(folder, include_dirs=True).col('basename'))
        ['a.csv', 'b.txt', 'c.csv', 'sub']

        A `@schemaclass` that names only some of the columns keeps only those:

        >>> from daffodil.lib.schemaclass import schemaclass
        >>> @schemaclass
        ... class FileRow:
        ...     basename: str = ''
        ...     size: int = 0
        >>> listed = Daf.from_directory(folder, schema=FileRow)
        >>> listed.columns()
        ['basename', 'size']
    """

    import os
    import re

    from daffodil.lib.schemaclass import schemaclass, SchemaBase

    #=========================
    # default schema
    #=========================

    if schema is None:

        @schemaclass
        class FilesystemSchema(SchemaBase):

            filepath:      str = ''
            dirpath:       str = ''
            basename:      str = ''
            rootname:      str = ''
            extension:     str = ''

            size:          int = 0

            mtime:         float = 0.0
            ctime:         float = 0.0

            is_dir:        int = 0

        schema = FilesystemSchema

    #=========================
    # initialize daf
    #=========================

    daf_obj = cls(schema=schema)

    #=========================
    # helper
    #=========================

    def normalize_path(path: str) -> str:

        return path.replace('\\', '/')

    #=========================
    # traversal
    #=========================

    walker: Iterable[Tuple[str, List[str], List[str]]]
    if recursive:

        walker = os.walk(str(source))

    else:

        full_source = str(source)

        names_ls = os.listdir(full_source)
        dirname_ls = [name for name in names_ls if os.path.isdir(os.path.join(full_source, name))]
        dirname_set = set(dirname_ls)
        basename_ls = [name for name in names_ls if name not in dirname_set]

        walker = [
            (
                full_source,
                dirname_ls,
                basename_ls,
            )
        ]

    #=========================
    # process files
    #=========================

    for dirpath, dirname_ls, basename_ls in walker:

        dirpath = normalize_path(dirpath)

        entries_lot: List[Tuple[str, int]] = [(name, 1) for name in dirname_ls] if include_dirs else []
        entries_lot += [(name, 0) for name in basename_ls]

        for basename, is_dir in entries_lot:

            if file_pat:

                if not re.search(file_pat, basename, flags=re.I):

                    continue

            filepath = normalize_path(
                os.path.join(dirpath, basename)
            )

            try:

                stat_result = os.stat(filepath)

            except Exception:

                continue

            if is_dir:
                rootname, extension = basename, ''
            else:
                rootname, extension = os.path.splitext(basename)

            row_da = daf_obj.default_record()

            row_da['filepath']  = filepath
            row_da['dirpath']   = dirpath
            row_da['basename']  = basename
            row_da['rootname']  = rootname
            row_da['extension'] = extension

            row_da['size']      = 0 if is_dir else stat_result.st_size

            row_da['mtime']     = stat_result.st_mtime
            row_da['ctime']     = stat_result.st_ctime

            row_da['is_dir']    = is_dir

            daf_obj.append(row_da)

    return daf_obj


def _from_numpy(cls: 'type[Daf]', npa: Any, keyfield:str='', cols:T_la | None = None, name:str='') -> 'Daf':
    """
    Make a Daf from a NumPy array.

    The values become plain Python values. A two dimensional array gives one row
    for each row of the array. A one dimensional array gives a single row.

    An array has one dtype. `np.array([[1, 'x']])` makes every value text, before this
    method sees it, so the Daf gets text too. An array with `dtype=object` keeps each
    value as the Python object it was, such as an int, a str or None.

    Args:
        npa: The NumPy array.
        keyfield: Column, or tuple or list of columns, whose values identify rows.
        cols: Column names.
        name: Name of the new Daf.

    Returns:
        The new Daf.

    Examples:
        >>> import numpy as np
        >>> Daf.from_numpy(np.array([[1, 2], [3, 4]]), cols=['a', 'b'])
        <BLANKLINE>
        | a  | b  |
        | -: | -: |
        |  1 |  2 |
        |  3 |  4 |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield=''; name=''
        <BLANKLINE>

        Give the keyfield and a name:

        >>> import numpy as np
        >>> table = Daf.from_numpy(np.array([[1, 2], [3, 4]]), cols=['a', 'b'], keyfield='a', name='grid')
        >>> table.name, table.select_record(3)
        ('grid', {'a': 3, 'b': 4})
    """
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
    if npa.ndim == 1:
        lol = [npa.tolist()]
    else:
        lol = npa.tolist()

    # following invalidates kd for lazy rebuilding.
    return cls(cols=cols, lol=lol, keyfield=keyfield, name=name)


def _to_numpy(self: 'Daf') -> T_npa:
    """
    Make a NumPy array of the rows.

    Column names, the keyfield and any dtypes are not kept. NumPy picks one type
    for the whole array. If the Daf mixes numbers and text, every value becomes
    text.

    Returns:
        The NumPy array.

    Examples:
        >>> Daf(lol=[[1, 2.5]], cols=['a', 'b']).to_numpy().tolist()
        [[1.0, 2.5]]
    """
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


def _to_donpa(self: 'Daf', colnames: T_ls | None = None, default: Any = _MISSING) -> T_donpa:
    """
    Make a dict of NumPy arrays, one array for each column.

    This is a light form of a DataFrame. The arrays can be used in vector
    operations, such as `donpa['D_pct'] = donpa['D_votes'] / donpa['RV_total']`.
    Each column gets its own type, so numbers and text can be mixed.

    Args:
        colnames: The columns to include. If None, all columns.
        default: A value that replaces each NULL, None and NaN cell, in the arrays only.
            The Daf is not changed. It applies to every column in `colnames`, so
            give only the numeric columns, or a text column gets it too.

    Returns:
        A dict that maps each column name to an array.

    Notes:
        A column that mixes numbers and NULL becomes a text array, because NumPy
        has one type for an array. Give a numeric `default` to keep it numeric.
        A column with a `default` is also faster to make than one with blanks and
        no default, because the array is built from numbers.

    Raises:
        ColumnNotFoundError: A name in `colnames` is not a column. This is a `KeyError`,
            and also a `RuntimeError`.

    Examples:
        >>> Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v']).to_donpa(['id'])['id'].tolist()
        [1, 2]
        >>> daf = Daf(lol=[[1, 5], [2, ''], [None, 7]], cols=['n', 'm'])
        >>> daf
        <BLANKLINE>
        |  n   | m  |
        | ---: | -: |
        |    1 |  5 |
        |    2 |    |
        | None |  7 |
        <BLANKLINE>
        %% daf rows=3; cols=2; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.to_donpa(['m'])['m'].tolist()
        ['5', '', '7']
        >>> daf.to_donpa(['n', 'm'], default=0)['m'].tolist()
        [5, 0, 7]
        >>> daf
        <BLANKLINE>
        |  n   | m  |
        | ---: | -: |
        |    1 |  5 |
        |    2 |    |
        | None |  7 |
        <BLANKLINE>
        %% daf rows=3; cols=2; keyfield=''; name=''
        <BLANKLINE>
    """
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

        """
    if colnames is None:
        colnames = self.columns()

    import numpy as np

    if default is _MISSING:
        return {col: np.array(self.col(col)) for col in colnames}

    # replace the missing cells as the column is read, so that NumPy sees only numbers.
    donpa = {}
    for col in colnames:
        donpa[col] = np.array([default if (val is NULL or val is None or val != val) else val
                                for val in self.col(col)])

    return donpa


def _from_googlesheet(cls: 'type[Daf]', spreadsheet_id: str, sheetname: str = 'Sheet1', *, service_account_file: str) -> 'Daf':
    """
    Read a Google Sheet into a Daf. This is not implemented.

    It always raises `NotImplementedError`. It holds the place of the interface, which
    takes the path of a service account file.

    Args:
        spreadsheet_id: The ID of the Google Sheet.
        sheetname: The name of the sheet.
        service_account_file: The path of the Google service account file with the credentials.

    Raises:
        NotImplementedError: Always.

    Examples:
        >>> Daf.from_googlesheet('some-id', service_account_file='key.json')
        Traceback (most recent call last):
            ...
        NotImplementedError: from_googlesheet() is not implemented yet.
    """
    raise NotImplementedError("from_googlesheet() is not implemented yet.")


def _to_googlesheet(self: 'Daf', spreadsheet_id: str, sheetname: str = 'Sheet1', *, service_account_file: str) -> 'Daf':
    """
    Write the Daf to a Google Sheet. This is not implemented.

    It always raises `NotImplementedError`. It holds the place of the interface, which
    takes the path of a service account file.

    Args:
        spreadsheet_id: The ID of the Google Sheet.
        sheetname: The name of the sheet.
        service_account_file: The path of the Google service account file with the credentials.

    Raises:
        NotImplementedError: Always.

    Examples:
        >>> Daf(cols=['x'], lol=[[1]]).to_googlesheet('some-id', service_account_file='key.json')
        Traceback (most recent call last):
            ...
        NotImplementedError: to_googlesheet() is not implemented yet.
    """
    raise NotImplementedError("to_googlesheet() is not implemented yet.")


def _to_json(self: 'Daf', concise: bool=True) -> str:
    """
    Make JSON text that holds the whole Daf.

    The text holds the rows, the column names, the dtypes, the keyfield, the name,
    the attrs and the display columns. With `concise=True`, the parts that are
    empty are left out. Use `from_json()` to read it back.

    Types are written by name. Only `int`, `float`, `str`, `bool`, `list` and `dict`
    are read back as types. Other names, such as `date`, come back as text.

    Cells must be JSON values. A tuple comes back as a list, and a set raises a
    `TypeError`. A NaN is written as `NaN`, which some JSON readers reject.

    Args:
        concise: If True, leave out the parts that are empty.

    Returns:
        The JSON text.

    Examples:
        >>> Daf(lol=[[1]], cols=['a']).to_json()
        '{"lol": [[1]], "hd": {"a": 0}}'

        With `concise=False`, the empty parts are written too:

        >>> Daf(cols=['a'], lol=[[1]]).to_json(concise=False)
        '{"name": "", "lol": [[1]], "hd": {"a": 0}, "dtypes": {}, "keyfield": "", "attrs": {}, "disp_cols": []}'
    """
    # Convert data types to string representations, rather than type objects.
    #   isinstance(v, type) -- Checks if the value is a Python type object (int, float, str, etc.).
    #   v.__name__          -- the type’s name as a string
    #   else str(v)         -- if the type is already a string, for example.
    
    dtypes_str = {
        k: (v.__name__ if isinstance(v, type) else str(v))
            for k, v in (self.dtypes or {}).items()
        }
    # Serialize Daf object to a JSON-compatible dictionary
    daf_dict = {
        'name':         self.name,
        'lol':          self.lol,
        'hd':           self.hd,
        'dtypes':       dtypes_str,
        'keyfield':     self.keyfield,
        'attrs':        self.attrs,
        'disp_cols':    self.disp_cols,
    }
    if concise:
        # strip out empty items.
        # change this if bool or int items are added.
        daf_dict = {k: v for k, v in daf_dict.items() if v}
    
    # Convert dictionary to JSON string
    return json.dumps(daf_dict)


def _from_json(cls: 'type[Daf]', json_str: str) -> 'Daf':
    """
    Make a Daf from JSON text made by `to_json()`.

    Args:
        json_str: The JSON text.

    Returns:
        The new Daf.

    Examples:
        >>> daf = Daf(lol=[[1, 'a']], cols=['id', 'v'], keyfield='id')
        >>> daf
        <BLANKLINE>
        | id | v  |
        | -: | -: |
        |  1 |  a |
        <BLANKLINE>
        %% daf rows=1; cols=2; keyfield='id'; name=''
        <BLANKLINE>
        >>> Daf.from_json(daf.to_json()) == daf
        True
    """
    # Deserialize JSON string into a Daf object
    daf_dict = json.loads(json_str)

    # Convert string representations of data types back to actual types
    dtypes = {k: cls.TYPE_MAP.get(v, v) 
                for k, v in daf_dict.get('dtypes', {}).items()
            }

    # following invalidates kd for lazy rebuilding.
    return cls(lol          = daf_dict.get('lol', []),
                hd          = daf_dict.get('hd', {}),
                dtypes      = dtypes,
                keyfield    = daf_dict.get('keyfield', ''),
                name        = daf_dict.get('name', ''),
                attrs       = daf_dict.get('attrs', {}),
                disp_cols   = daf_dict.get('disp_cols', []),
                )


def _to_value(self: 'Daf',
    # irow:       int=0,
    # icol:       int=0,
    default:    Any = _MISSING,
    astype:     Callable | str | None = None,
    ) -> Any:
    """
    Get the one value of a Daf that has one row and one column.

    Use it on the result of a selection, as in `my_daf[1, 'n'].to_value()`.

    Args:
        default: Returned if the Daf is not one cell. Without it, an error is raised.
        astype: A type or function to convert the value with.

    Returns:
        The value.

    Raises:
        ValueError: The Daf is not one cell and no default is given.

    Examples:
        >>> daf = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
        >>> daf
        <BLANKLINE>
        | id | v  | n  |
        | -: | -: | -: |
        |  1 |  a | 10 |
        |  2 |  b | 20 |
        |  3 |  c | 30 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>
        >>> daf[1, 'n'].to_value()
        20

        Convert the value with `astype`, here from text:

        >>> Daf(cols=['qty'], lol=[['3']]).to_value(astype=int)
        3

        A Daf that is not one cell raises, unless `default` is given. The default may be None:

        >>> print(Daf(cols=['qty'], lol=[[1], [2]]).to_value(default=None))
        None
    """

    num_rows, num_cols = self.shape()

    if num_rows == 1 and num_cols == 1:
        return daf_utils.astype_value(self.lol[0][0], astype)

    if default is not _MISSING:
        return default

    raise ValueError(f"to_value() requires 1×1, got shape {self.shape()}")


def _to_list(self: 'Daf',
    # irow:       int | None = None,   # select a row
    # icol:       int | None = None,   # or column.
    unique:     bool=False,           # reduce to unique values
    flatten:    bool=False,           # if items is the list are lists, combine them into one list.
    omit_nulls: bool=False,           # omit items that are empty strings (nulls).
    default:    Any = _MISSING,       # use this value instead if value is None or '' or NAN (default can be None)
    astype:     Callable | str | type | None = None,
    ) -> list:
    """
    Get the values of a Daf that has one row or one column, as a list.

    Use it on the result of a selection, as in `my_daf[:, 'v'].to_list()`. A column
    is read from the Daf, so use `col()` if you only need the list. An empty
    Daf gives an empty list. A Daf with more than one row and more than one column
    is not accepted.

    Args:
        unique: If True, leave out repeated values and keep the order.
        flatten: If True, join items that are lists into one list.
        omit_nulls: If True, leave out the empty values.
        default: Replaces NULL, None and NaN values. This may itself be None.
        astype: A type or function to convert each value with.

    Returns:
        The list.

    Raises:
        ValueError: The Daf has more than one row and more than one column.

    Examples:
        >>> daf = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
        >>> daf
        <BLANKLINE>
        | id | v  | n  |
        | -: | -: | -: |
        |  1 |  a | 10 |
        |  2 |  b | 20 |
        |  3 |  c | 30 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>
        >>> daf[:, 'v'].to_list()
        ['a', 'b', 'c']
        >>> daf[1].to_list()
        [2, 'b', 20]

        The options, on a column whose cells hold lists, with one empty cell:

        >>> tags = Daf(cols=['tags'], lol=[[['a', 'b']], [''], [['b', 'c']]])
        >>> tags.to_list()
        [['a', 'b'], '', ['b', 'c']]
        >>> tags.to_list(flatten=True)
        ['a', 'b', '', 'b', 'c']
        >>> tags.to_list(flatten=True, unique=True)
        ['a', 'b', '', 'c']
        >>> tags.to_list(omit_nulls=True)
        [['a', 'b'], ['b', 'c']]
        >>> tags.to_list(default=None)
        [['a', 'b'], None, ['b', 'c']]

        Convert each value with `astype`:

        >>> Daf(cols=['n'], lol=[['1'], ['2']]).to_list(astype=int)
        [1, 2]
    """
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

    if num_rows == 1 and num_cols >= 1:
        # single row, return as list.
        result_la = self.lol[0]

    elif num_rows > 1 and num_cols == 1:
        # single column result as a list.
        result_la = self.icol(0)
    elif num_rows == num_cols == 0:
        result_la = []
    else:
        raise ValueError("to_list() requires a 1D Daf (single row or single column)")

    if flatten:
        new_list = []       # perflint-reviewed (use-tuple-over-list)
        for sublist in result_la:
            if isinstance(sublist, list):
                new_list += sublist
            else:
                new_list += [sublist]
        result_la = new_list

    if unique and result_la:
        result_la = list(dict.fromkeys(result_la))

    if omit_nulls and any(val is NULL for val in result_la):
        result_la = [val for val in result_la if val is not NULL]

    if default is not _MISSING and any(val is NULL or val is None for val in result_la):
        filtered_result_la = []
        for val in result_la:
            if val is None or val is NULL or val != val:  # val != val catches NaN
                filtered_result_la.append(default)
            else:
                filtered_result_la.append(val)
        result_la = filtered_result_la

    return daf_utils.astype_la(result_la, astype)   # returns result_la if astype is None.


def _to_lota(self: 'Daf',
    ) -> T_lota:
    """
    Make a list of tuples, one tuple for each row.

    This is handy for making composite keys. The tuples are new, but the values in them
    are the same objects as in the Daf.

    Returns:
        The rows as tuples.

    Examples:
        >>> Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v']).to_lota()
        [(1, 'a'), (2, 'b')]
    """
    """ return data from a daf array as a list of tuple of any (List[Tuple[Any...]])
            This is convenient for creating compound keys
        """

    lota = []

    for la in self.lol:
        lota.append(tuple(la))

    return lota


def _to_dict(self: 'Daf', 
        # irow: int=0, 
        # include_cols: T_ls | None = None,
        ) -> T_da:
    """
    Get the one row of a Daf as a dict.

    Use it on the result of a selection, as in `my_daf[1].to_dict()`. A column is
    not turned into a dict. Use `to_list()` for that. An empty Daf gives an empty
    dict. A Daf with no column names raises `KeysDisabledError`.

    Returns:
        The row, as a dict that maps column names to values.

    Raises:
        KeysDisabledError: The Daf has a row and no column names.
        ValueError: The Daf has more than one row.

    Examples:
        >>> daf = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
        >>> daf
        <BLANKLINE>
        | id | v  | n  |
        | -: | -: | -: |
        |  1 |  a | 10 |
        |  2 |  b | 20 |
        |  3 |  c | 30 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>
        >>> daf[1].to_dict()
        {'id': 2, 'v': 'b', 'n': 20}
    """
    """ alias for iloc
            Note that this does not convert a column to a dict. Use to_list to convert a column.
            test exists in test_daf.py
        """
    if len(self) > 1:
        raise ValueError("Ambiguous 2dim array.")

    return cast(T_da, self.iloc(irow=0, include_cols=None))


def _to_klist(self: 'Daf', irow: int=0) -> KeyedList:
    """
    Get one row as a [KeyedList][daffodil.keyedlist.KeyedList].

    The KeyedList shares the row and the column names with the Daf, so it costs
    little. A negative position counts from the end. A position that is out of range
    raises `IndexError`. A Daf with no rows gives an empty KeyedList.

    Args:
        irow: The row position.

    Returns:
        The row.

    Raises:
        IndexError: The position is out of range, for a Daf that has rows.

    Examples:
        >>> daf = Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'], keyfield='id')
        >>> daf
        <BLANKLINE>
        | id | v  | n  |
        | -: | -: | -: |
        |  1 |  a | 10 |
        |  2 |  b | 20 |
        |  3 |  c | 30 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>
        >>> daf.to_klist(1)['v']
        'b'
    """
    return cast(KeyedList, self.iloc(irow, rtype='klist'))
