# daf_modify.py
# copyright (c) 2026 Ray Lutz
"""
Methods of Daf for modifying.

Each function here takes the Daf as its first argument, or the class for a classmethod.
It is assigned in the Daf class body under its Daf name, without the underscore.
The docs show it under that name.

Some functions use names defined in daf.py, such as Daf, when they run.
daf.py sets those names here, as listed in _NAMES_FROM_DAF, once they exist.
An import at the top would be circular, since daf.py imports this module.
"""

from __future__ import annotations

import copy
import re
import time

from daffodil.lib.daf_types import T_cs, T_da, T_di, T_la, T_li, T_loda, T_lola, T_ls, T_lsi, T_ta
import daffodil.lib.daf_utils    as daf_utils
from daffodil.lib.daf_utils import _MISSING, NULL
from daffodil.keyedlist import KeyedList

from collections.abc import Iterable
from typing import List, Dict, Any, Tuple, cast, TYPE_CHECKING

if TYPE_CHECKING:       # for mypy. daf.py sets the names below when the program runs.
    from daffodil.daf import Daf, KeysDisabledError, _keys_mismatch, _short_list, logs

# The names of daf.py that the functions here use when they run.
# daf.py sets each one here once it exists. An import at the top would be circular.
_NAMES_FROM_DAF: Tuple[str, ...] = ('Daf', 'KeysDisabledError', '_keys_mismatch', '_short_list', 'logs')


def _append(self: 'Daf',
        data_item:  'Daf' | T_loda | T_da | T_la | KeyedList | None = None,
        respect_kd: bool = False,
        *,
        lol:        T_lola | None = None,
        la:         T_la | None = None,
        fast:       bool = False,
        ) -> 'Daf':
    """
    Add one row, or several, to the end of the Daf.

    This is the general way to add data. Give it one of these:

    - **A dict,** or a [KeyedList][daffodil.keyedlist.KeyedList]: one row, placed by column name,
      in any order. A missing key gets NULL, the empty string. A key that is not a column is dropped.
    - **A list of values:** one row, in column order. A short list is padded with NULL. A longer
      one raises ValueError.
    - **A list of dicts:** several rows, as `extend()` adds them.
    - **A Daf** with the same columns: its rows, as `concat()` adds them.
    - **`lol=`,** a list of lists: several rows, each in column order.
    - **`la=`,** a list: one row, even when its values are lists. Use it to say so plainly.

    None, an empty dict, an empty list and an empty Daf add nothing. `append()` changes this Daf and
    returns it, so calls can be chained.

    **Keys.** The keyfield is not checked, so a key that is already there is added again, as a
    second row. This keeps appending fast. With `respect_kd=True`, the row that has the same key is
    replaced instead. That looks the key up on every call, so it costs more for many single rows.

    **Copies.** With columns defined, each row is copied, so you can change or reuse your list or
    dict afterwards. With no columns defined, a list is added as it is.

    **fast=True** adds the row with no copy, and checks it only lightly, for speed in a loop that
    builds each row itself:

    - A KeyedList must share this Daf's column names, as one from `default_record(astype=KeyedList)`,
      `iloc()` or `iter_klist()` does. Nothing else is checked.
    - A dict or list is checked fully if it is the first row of an empty Daf. A dict must have the
      columns as its keys, in order, and a list one value for each column. Every later dict or list
      is checked only for its number of values. A row that fails raises ValueError.
    - The list, or the KeyedList's values, becomes the row itself. Do not change it afterwards, and
      do not fill one list in place and append it again: every row would be that same list.
    - A later dict in another order, with the right number of values, puts its values in the wrong
      columns. Leave out `fast` if rows may come in another order.

    `fast` makes no difference to the first row of a Daf with no columns, to a Daf, to a list of
    dicts, or with `respect_kd=True` and a keyfield.

    Args:
        data_item: The row or rows to add.
        respect_kd: If True, replace the row that has the same key. If False, the default, add it.
        lol: Several rows, each a list of values in column order.
        la: One row, as a list of values in column order. Its items are not read as rows.
        fast: If True, add the row with no copy. Check the first row of an empty Daf fully, and every row's length. See above.

    Returns:
        This Daf, which has been changed.

    Raises:
        TypeError: More than one of `data_item`, `lol` and `la` is given, or `data_item` is not a supported type.
        ValueError: A list has more values than there are columns.
        KeyError: `data_item` is a Daf whose columns differ, in name or in order. Nothing is added.

    Examples:
        The table used in the examples:

        >>> daf = Daf(cols=['id', 'item', 'qty'], keyfield='id')

        Each call is followed by a read of the row it added. `append()` returns the Daf, so at the
        prompt a bare call shows the whole table.

        **A dict,** by column name. A missing column gets NULL, and an unknown key is dropped:

        >>> daf.append({'qty': 3, 'id': 'a1', 'item': 'pen'}).select_record('a1')
        {'id': 'a1', 'item': 'pen', 'qty': 3}
        >>> daf.append({'id': 'b2', 'color': 'red'}).select_record('b2')
        {'id': 'b2', 'item': '', 'qty': ''}

        **A list,** in column order. A short list is padded, and a long one raises:

        >>> daf.append(['c3', 'pad', 7]).select_record('c3')
        {'id': 'c3', 'item': 'pad', 'qty': 7}
        >>> daf.append(['d4']).select_record('d4')
        {'id': 'd4', 'item': '', 'qty': ''}
        >>> daf.append(['e5', 'tag', 1, 'extra'])
        Traceback (most recent call last):
        ...
        ValueError: append(): the list has 4 values for 3 columns.

        **Several rows:** a list of dicts, `lol=`, or another Daf with the same columns:

        >>> daf.append([{'id': 'f6', 'qty': 2}, {'id': 'g7', 'qty': 5}]).keys()
        ['a1', 'b2', 'c3', 'd4', 'f6', 'g7']
        >>> daf.append(lol=[['h8', 'cap', 1], ['i9', 'ink', 4]]).keys()
        ['a1', 'b2', 'c3', 'd4', 'f6', 'g7', 'h8', 'i9']
        >>> more = Daf(cols=['id', 'item', 'qty'], lol=[['j10', 'pen', 6]])
        >>> daf.append(more).keys()
        ['a1', 'b2', 'c3', 'd4', 'f6', 'g7', 'h8', 'i9', 'j10']

        **One row whose values are lists,** with `la=`:

        >>> tags = Daf(cols=['id', 'tags'], keyfield='id')
        >>> tags.append(la=['a1', ['red', 'large']]).select_record('a1')
        {'id': 'a1', 'tags': ['red', 'large']}

        **A repeated key** is added again, unless `respect_kd=True`, which replaces the row:

        >>> stock = Daf(cols=['id', 'qty'], lol=[['a1', 3], ['b2', 10]], keyfield='id')
        >>> stock.append({'id': 'a1', 'qty': 30}, respect_kd=True)
        <BLANKLINE>
        | id | qty |
        | -: | --: |
        | a1 |  30 |
        | b2 |  10 |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield='id'; name=''
        <BLANKLINE>

        **Nothing to add,** as None or an empty dict, list or Daf, changes nothing:

        >>> len(stock.append(None).append({}).append([]).append(Daf()))
        2

        **fast=True,** with a KeyedList from `default_record()`. A row of the wrong length raises:

        >>> from daffodil.keyedlist import KeyedList
        >>> row = stock.default_record(astype=KeyedList)
        >>> row['id'], row['qty'] = 'c3', 7
        >>> stock.append(row, fast=True).select_record('c3')
        {'id': 'c3', 'qty': 7}
        >>> stock.append(['d4'], fast=True)
        Traceback (most recent call last):
        ...
        ValueError: append(fast=True): row 3 has 1 values for 2 columns. With fast=True a list must have one value for each column, in column order. Leave out fast to pad a short list with NULL.
    """

    """ general append method can handle appending one record as T_da or T_la, many records as T_loda or T_daf
            if the data item is None or empty, do not append.
        """
    # test exists in test_daf.py for all three cases

    diagnose = False

    if fast and self.hd and not (respect_kd and self.keyfield):
        # the caller promises complete rows in column order. The first row of an empty Daf is
        # checked fully, every row has its length checked, and nothing is copied.
        row: Any = None
        if lol is not None and data_item is None and la is None:
            if lol and not self.lol:
                self._check_first_fast_row(lol[0])
            num_cols = len(self.hd)
            for irow, row_la in enumerate(lol):
                if len(row_la) != num_cols:
                    fix = ("Leave out fast to pad a short row with NULL." if len(row_la) < num_cols
                           else "A row longer than the columns is an error with or without fast.")
                    raise ValueError(
                        f"append(fast=True){self._err_name()}: row {irow} of lol has {len(row_la)} values "
                        f"for {num_cols} columns. With fast=True each row must have one value for each column, "
                        f"in column order. Nothing was added. {fix}")
            self.lol.extend(lol)
            self._invalidate_kd()
            return self
        if la is not None and data_item is None:
            row = la
        elif data_item is not None and la is None and lol is None:
            row = data_item
            if isinstance(row, list) and row and isinstance(row[0], dict):
                row = None                          # a list of dicts is several rows: the usual path
        if isinstance(row, KeyedList):
            # a KeyedList qualifies only if it shares the hd of this Daf. Then its keys are the
            # columns, and nothing else needs checking.
            if row.hd is not self.hd:
                if list(row.hd) == self._col_names():
                    why = ("its keys are the columns, but it was built separately, or before the columns "
                           "of this Daf changed")
                else:
                    why = _keys_mismatch(row.hd, self._col_names())
                raise ValueError(
                    f"append(fast=True){self._err_name()}: the KeyedList for row {len(self.lol)} does not share "
                    f"the hd of this Daf, so its keys are not known to be the columns: {why}. Make the row "
                    f"with default_record(astype=KeyedList) of this Daf, and set only keys that are columns, "
                    f"or leave out fast to place its values by column name.")
        elif row and isinstance(row, (list, dict)):
            if not self.lol:
                self._check_first_fast_row(row)
            elif len(row) != len(self.hd):          # every row: the number of values
                self._raise_fast_length(row)
        if row:
            if isinstance(row, list):
                self.lol.append(row)
            elif isinstance(row, KeyedList):
                self.lol.append(row.values())
            elif isinstance(row, dict):
                self.lol.append(list(row.values()))
            else:
                row = None
            if row is not None:
                self._invalidate_kd()
                return self

    if (data_item is not None) + (lol is not None) + (la is not None) > 1:
        raise TypeError("append(): give only one of data_item, lol and la.")

    if lol is not None:
        return self.extend(lol=lol, respect_kd=respect_kd)

    if la is not None:
        if not la:
            return self
        return self._append_row_la(la, respect_kd)

    if not data_item:
        return self

    if diagnose:
        start_time = time.time()
        logs.sts(f"{logs.prog_loc()} starting append.", 3)

    if isinstance(data_item, (dict, KeyedList)):
        self.record_append(data_item, respect_kd=respect_kd)

    elif isinstance(data_item, list):
        if isinstance(data_item[0], dict):
            # lod type
            self.extend(data_item, respect_kd=respect_kd)
        else:
            self._append_row_la(data_item, respect_kd)

    elif isinstance(data_item, Daf):  # type: ignore
        self.concat(data_item, respect_kd=respect_kd)

    else:
        raise TypeError(f"append(): data_item must be a dict, KeyedList, list or Daf, not {type(data_item).__name__}.")

    if diagnose:
        logs.sts(f"{logs.prog_loc()} append done. Elapsed: {(time.time() - start_time):.10f}", 3)

    return self


def _append_row_la(self: 'Daf', row_la: T_la, respect_kd: bool) -> 'Daf':
    """
    Add one row that is given as a list of values in column order. Internal use.

    A list with more values than the columns raises `ValueError`. A short list is padded
    with NULL. With no columns defined, the list is added as it is.

    With columns defined, a copy of the list is added, so the caller can reuse it.
    The values are already in column order, so no dict is built. Only an upsert by key,
    `respect_kd=True` with a keyfield, goes through `record_append()`.
    """
    if self.hd:
        num_cols = len(self.hd)
        if len(row_la) > num_cols:
            raise ValueError(f"append(): the list has {len(row_la)} values for {num_cols} columns.")
        if self.keyfield and respect_kd:
            # upsert by key: record_append() finds and replaces the row with the same key.
            da = dict(zip(self.hd.keys(), row_la))
            self.record_append(da, respect_kd=respect_kd)
            return self
        rec_la = list(row_la)
        if len(rec_la) < num_cols:
            rec_la.extend([''] * (num_cols - len(rec_la)))
        self.lol.append(rec_la)
        self._invalidate_kd()       # the kd is rebuilt lazily, only if a keyfield is set.
    else:
        # no columns defined, therefore just append to lol.
        self.lol.append(row_la)

    return self


def _concat(self: 'Daf', other_instance: 'Daf', respect_kd: bool=False) -> 'Daf':
    """
    Add the rows of another Daf to the end of this one.

    Use this to combine two tables with the same columns. By default every row is
    appended, so a key that exists in both tables appears twice afterwards. Pass
    `respect_kd=True` to get an upsert instead.

    Args:
        other_instance: Daf whose rows are added. It is not changed. A SqlDaf is read into memory first.
        respect_kd: If False, the default, append every row without looking at the keys. If True,
            replace the row that has the same key and append the rows with new keys.

    Returns:
        This Daf, which has been changed in place. The rows are deep copies.

    Raises:
        KeyError: The columns differ, in name or in order. The message names the columns found in only one.
        ValueError: With `respect_kd=True`, both Daf instances have a keyfield and the
            keyfields differ.

    Notes:
        The columns must be equal, including their order.
        With `respect_kd=True`, this Daf's keyfield decides the key. If it has no
        keyfield, the rows are simply appended.
        A key repeated in the other Daf replaces the earlier row, so the last one wins.
        If the other Daf is empty, nothing happens.
        A SqlDaf is read with `to_daf()`, so all its rows come into memory. Select the rows first.

    Examples:
        >>> a_daf = Daf(lol=[[1, 'a1'], [2, 'a2']], cols=['id', 'v'], keyfield='id')
        >>> a_daf
        <BLANKLINE>
        | id | v  |
        | -: | -: |
        |  1 | a1 |
        |  2 | a2 |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield='id'; name=''
        <BLANKLINE>
        >>> b_daf = Daf(lol=[[2, 'b2'], [3, 'b3']], cols=['id', 'v'])
        >>> b_daf
        <BLANKLINE>
        | id | v  |
        | -: | -: |
        |  2 | b2 |
        |  3 | b3 |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield=''; name=''
        <BLANKLINE>
        >>> a_daf.concat(b_daf)
        <BLANKLINE>
        | id | v  |
        | -: | -: |
        |  1 | a1 |
        |  2 | a2 |
        |  2 | b2 |
        |  3 | b3 |
        <BLANKLINE>
        %% daf rows=4; cols=2; keyfield='id'; name=''
        <BLANKLINE>
        >>> a_daf = Daf(lol=[[1, 'a1'], [2, 'a2']], cols=['id', 'v'], keyfield='id')
        >>> a_daf
        <BLANKLINE>
        | id | v  |
        | -: | -: |
        |  1 | a1 |
        |  2 | a2 |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield='id'; name=''
        <BLANKLINE>
        >>> a_daf.concat(b_daf, respect_kd=True)
        <BLANKLINE>
        | id | v  |
        | -: | -: |
        |  1 | a1 |
        |  2 | b2 |
        |  3 | b3 |
        <BLANKLINE>
        %% daf rows=3; cols=2; keyfield='id'; name=''
        <BLANKLINE>
    """

    if not other_instance:
        return self

    # A SqlDaf reads its rows into memory here, as new rows, which are adopted as they are.
    # A Daf gives itself, and its rows are copied.
    other_daf: 'Daf' = other_instance.to_daf()
    own_rows: bool = other_daf is not other_instance

    if not self.lol and not self.hd:
        self.hd = copy.deepcopy(other_daf.hd)
        self.lol = other_daf.lol if own_rows else copy.deepcopy(other_daf.lol)
        self.keyfield = other_daf.keyfield
        self._invalidate_kd()    # use lazy kd rebuilding

        return self

    # Fields must match exactly!
    if self.hd != other_daf.hd:
        raise KeyError(daf_utils.columns_mismatch_msg(list(self.hd), list(other_daf.hd), self.name))

    if respect_kd and self.keyfield:
        if other_daf.keyfield and other_daf.keyfield != self.keyfield:
            raise ValueError(f"concat: keyfield mismatch: {self.keyfield!r} versus {other_daf.keyfield!r}")

        self._rebuild_kd_if_invalidated()
        kd = self._kd
        lol = self.lol
        if isinstance(self.keyfield, (str, int)):
            key_idx = self.hd[cast(str, self.keyfield)]
            for rec_la in other_daf.lol:
                rec_la = copy.deepcopy(rec_la)
                keyval = rec_la[key_idx]
                if keyval in kd:
                    lol[kd[keyval]] = rec_la
                else:
                    kd[keyval] = len(lol)
                    lol.append(rec_la)
        else:
            key_idxs = [self.hd[cast(str, key)] for key in self.keyfield]
            for rec_la in other_daf.lol:
                rec_la = copy.deepcopy(rec_la)
                keyval = tuple(rec_la[i] for i in key_idxs)
                if keyval in kd:
                    lol[kd[keyval]] = rec_la  # type: ignore[index]
                else:
                    kd[keyval] = len(lol)  # type: ignore[index]  # composite key is a tuple
                    lol.append(rec_la)
        return self

    # Append deep-copied rows to avoid referencing issues. Rows read from a SqlDaf are new already.
    if own_rows:
        self.lol.extend(other_daf.lol)
    else:
        for rec_la in other_daf.lol:
            self.lol.append(copy.deepcopy(rec_la))

    self._invalidate_kd()    # use lazy kd rebuilding

    return self


def _extend(self: 'Daf',
        records_lod:    T_loda | None = None,
        respect_kd:     bool = False,
        *,
        lol:            T_lola | None = None,
        ) -> 'Daf':
    """
    Append several records, given as a list of dicts or as a list of lists.

    Give a list of dicts as `records_lod`. Each dict is a row, placed by column name,
    as in `append()`. A Daf with no columns takes them from the first dict. An empty
    list, or a list that holds one empty dict, adds nothing.

    Give a list of lists as `lol`, as in `extend(lol=[[2, 'b'], [3, 'c']])`. Each list is a row, in
    column order. A short list is padded with NULL. A list with more values than there
    are columns raises `ValueError`, and then no row is added. A Daf with no columns
    takes the lists as they are. The keyword says that these are several rows, and not
    one row whose cells are lists. For that, see `append(la=...)`.

    Without `respect_kd`, a key that already exists is added again. Use
    `respect_kd=True` when the keyfield must stay unique.

    Args:
        records_lod: The records, as dicts.
        respect_kd: If True, replace the row that has the same key. If False, the default, add it.
        lol: The rows, as lists of values in column order.

    Returns:
        This Daf, which has been changed.

    Raises:
        TypeError: Both `records_lod` and `lol` are given, or a row of `lol` is not a list.
        ValueError: A row of `lol` has more values than there are columns.

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
        >>> daf.extend([{'id': 2, 'v': 'b'}, {'id': 1, 'v': 'new'}], respect_kd=True)
        <BLANKLINE>
        | id |  v  |
        | -: | --: |
        |  1 | new |
        |  2 |   b |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield='id'; name=''
        <BLANKLINE>
        >>> daf.extend(lol=[[3, 'c'], [4]])
        <BLANKLINE>
        | id |  v  |
        | -: | --: |
        |  1 | new |
        |  2 |   b |
        |  3 |   c |
        |  4 |     |
        <BLANKLINE>
        %% daf rows=4; cols=2; keyfield='id'; name=''
        <BLANKLINE>
    """

    if records_lod is not None and lol is not None:
        raise TypeError("extend(): give only one of records_lod and lol.")

    if lol is not None:
        return self._extend_lol(lol, respect_kd)

    if not records_lod or len(records_lod) == 1 and not records_lod[0]:
        # edge case of one record which is empty.
        return self

    if not self.lol and not self.hd:
        # new daf, adopt structure of lod.
        # but there is only one header and data is lol
        # this saves space.
        self.hd = {col_name: index for index, col_name in enumerate(records_lod[0].keys())}
        self.lol = [list(record_da.values()) for record_da in records_lod]
        self._invalidate_kd() # use lazy kd building.
        return self

    for record_da in records_lod:
        self.record_append(record_da, respect_kd=respect_kd)

    if not respect_kd:
        self._invalidate_kd() # use lazy kd building.

    return self


def _extend_lol(self: 'Daf', lol: T_lola, respect_kd: bool) -> 'Daf':
    """
    Add several rows that are lists of values in column order. Internal use.

    Every row is checked before any row is added. A short row is padded with NULL.
    """
    if not lol:
        return self

    num_cols = len(self.hd)

    for row_la in lol:
        if not isinstance(row_la, list):
            raise TypeError(f"extend(): each row of lol must be a list, not {type(row_la).__name__}.")
        if num_cols and len(row_la) > num_cols:
            raise ValueError(f"extend(): a row has {len(row_la)} values for {num_cols} columns.")

    if respect_kd and self.keyfield and num_cols:
        for row_la in lol:
            self.record_append(dict(zip(self.hd, row_la)), respect_kd=True)
        return self

    if num_cols:
        lol = [row_la + [NULL] * (num_cols - len(row_la)) if len(row_la) < num_cols else row_la for row_la in lol]

    self.lol.extend(lol)
    self._invalidate_kd()       # use lazy kd building.

    return self


def _record_append(self: 'Daf', record: T_da | KeyedList, respect_kd: bool=True) -> 'Daf':
    """
    Add one row that is given as a dict or a KeyedList.

    This is the single row case of `append()`. The row is placed by column name,
    a missing key gets NULL, and a key that is not a column is dropped. An empty
    Daf takes its columns from the first row. An empty record adds nothing.

    With a keyfield, the row that has the same key is replaced, and a new key is
    added at the end. This is the default here, unlike `append()`. Pass
    `respect_kd=False` to always add. The key index is kept up to date, so adding
    many rows one at a time stays fast.

    A plain list is not accepted. Use `append()` for that.

    Args:
        record: The row, as a dict or a [KeyedList][daffodil.keyedlist.KeyedList].
        respect_kd: If True, the default, replace the row that has the same key. Otherwise add it.

    Returns:
        This Daf, which has been changed.

    Examples:
        >>> daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'], keyfield='id')
        >>> daf
        <BLANKLINE>
        | id | v  |
        | -: | -: |
        |  1 |  a |
        |  2 |  b |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield='id'; name=''
        <BLANKLINE>
        >>> daf.record_append({'id': 2, 'v': 'new'})
        <BLANKLINE>
        | id |  v  |
        | -: | --: |
        |  1 |   a |
        |  2 | new |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield='id'; name=''
        <BLANKLINE>
        >>> daf.record_append({'id': 2, 'v': 'dup'}, respect_kd=False)
        <BLANKLINE>
        | id |  v  |
        | -: | --: |
        |  1 |   a |
        |  2 | new |
        |  2 | dup |
        <BLANKLINE>
        %% daf rows=3; cols=2; keyfield='id'; name=''
        <BLANKLINE>
    """
        # test exists in test_daf.py

    if not record:
        return self

    if not self.lol and not self.hd:
        # new daf, adopt structure of da or keyedlist.

        if isinstance(record, KeyedList):
            # for keyedlist, take a copy of its hd, and a copy of its values as the first row.
            self.hd = cast(Dict[str, int], dict(record.hd))
            self.lol = [list(record.values())]     # a copy: values() is the KeyedList's own list.

        elif isinstance(record, dict):
            self.hd = type(self)._build_hd(record.keys())
            self.lol = [list(record.values())]

        self._invalidate_kd()    # use lazy kd rebuilding
        # self._rebuild_kd()   # functions only if the keyfield is set.
        return self

    # check that the keys are the columns, in order. A KeyedList that shares the hd of this Daf
    # needs no check, as its keys are the columns. Anything else is compared with the list of
    # column names, which is kept until the columns change.
    if isinstance(record, KeyedList) and record.hd is self.hd and len(record._values) == len(self.hd):
        reorder = False
    else:
        reorder = list(record.keys()) != self._col_names()

    if reorder:
        # construct a dict with exactly the cols specified.
        # defaults to '' at this point.
        # this works for both dict and KeyedList
        rec_la = [record.get(col, '') for col in self.hd]
    # not reordering, slightly different
    elif isinstance(record, KeyedList):
        rec_la = list(record.values())  # a copy: values() is the KeyedList's own list.
    elif isinstance(record, dict):
        rec_la = list(record.values())  # must copy into a list.
    else:
        # any other mapping with keys in column order.
        rec_la = [record[col] for col in self.hd]

    if self.keyfield and respect_kd:
        keyval = self._get_keyval(record)

        self._rebuild_kd_if_invalidated()

        if keyval in self._kd:
            self.lol[self._kd[keyval]] = rec_la
        else:
            self.lol.append(rec_la)
            if isinstance(self._kd, dict):
                self._kd[keyval] = len(self.lol) - 1
            else:
                self._kd.append(keyval)
    else:
        # no keyfield is set, or not respect_kd, just append to the end.
        self.lol.append(rec_la)
        self._invalidate_kd()    # use lazy kd rebuilding only if keyfield != ''

    return self


def _col_names(self: 'Daf') -> T_ls:
    """ The column names as a list, kept until hd is replaced. Internal. Do not change the list. """
    cache = self.__dict__.get('_col_names_cache')
    if cache is None or cache[0] is not self.hd:
        cache = self._col_names_cache = (self.hd, list(self.hd))     # type: ignore[attr-defined]  # set only here.
    return cache[1]


def _check_first_fast_row(self: 'Daf', row: Any) -> None:
    """
    With fast=True, check the first row added to an empty Daf. Internal.

    A dict must have the columns as its keys, in order. A list must have one value for each
    column. A KeyedList is checked by append() itself: it must share the hd of the Daf.

    Raises:
        ValueError: The row does not match the columns.
    """
    num_cols = len(self.hd)
    if isinstance(row, dict):
        if list(row) != self._col_names():
            raise ValueError(
                f"append(fast=True){self._err_name()}: row 0, the first row of an empty Daf, does not match "
                f"the columns {_short_list(self._col_names())}: {_keys_mismatch(row, self._col_names())}. "
                f"With fast=True the first row must have the columns as its keys, in order. Make it with "
                f"default_record(), or leave out fast to place its values by key.")
    elif len(row) != num_cols:
        self._raise_fast_length(row)


def _raise_fast_length(self: 'Daf', row: Any) -> None:
    """ With fast=True, raise for a dict or list row with the wrong number of values. Internal. """
    num_cols = len(self.hd)
    if isinstance(row, dict):
        raise ValueError(
            f"append(fast=True){self._err_name()}: row {len(self.lol)} has {len(row)} keys for {num_cols} "
            f"columns: {_keys_mismatch(row, self._col_names())}. With fast=True a dict must have the "
            f"columns as its keys, in order. Make it with default_record(), or leave out fast to give a "
            f"missing key NULL.")
    fix = ("Leave out fast to pad a short list with NULL." if len(row) < num_cols
           else "A list longer than the columns is an error with or without fast.")
    raise ValueError(
        f"append(fast=True){self._err_name()}: row {len(self.lol)} has {len(row)} values for {num_cols} "
        f"columns. With fast=True a list must have one value for each column, in column order. {fix}")


def _err_name(self: 'Daf') -> str:
    """ ' to Daf name' for an error message, or nothing for a Daf with no name. Internal. """
    return f" to Daf {self.name!r}" if self.name else ""


def _remove_key(self: 'Daf', keyval: str | int | T_la | T_ta | None, silent_error: bool=False) -> 'Daf':
    """
    Make a new Daf without the row that has the given key.

    This is deprecated. Use `select_krows(key, inverse=True)`, which does the same.

    This does not remove the row from this Daf. It returns a new Daf that leaves
    the row out, and this Daf is unchanged. Keep the result, as in
    `daf = daf.remove_key(2)`.

    The new Daf shares the surviving rows with this Daf. Changing a cell in one
    changes it in the other. Call `copy('editable')` if you need row lists of your own.

    A tuple is a range of keys, as in `remove_key((2, 3))`. With a composite keyfield, a
    tuple as long as the keyfield, with no tuples inside it, is one key, as in
    `remove_key(('a', 1))`. A tuple of key tuples is a range, and a list of key tuples
    is a list of keys.

    Args:
        keyval: The key of the row to leave out.
        silent_error: If True, a key that is not found is ignored.

    Returns:
        The new Daf. It has the same keyfield.

    Raises:
        KeysDisabledError: The Daf has no keyfield and no key index.
        KeyError: The key is not found and `silent_error` is False.

    Examples:
        >>> daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'], keyfield='id')
        >>> daf
        <BLANKLINE>
        | id | v  |
        | -: | -: |
        |  1 |  a |
        |  2 |  b |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield='id'; name=''
        <BLANKLINE>
        >>> daf.remove_key(1)
        <BLANKLINE>
        | id | v  |
        | -: | -: |
        |  2 |  b |
        <BLANKLINE>
        %% daf rows=1; cols=2; keyfield='id'; name=''
        <BLANKLINE>
        >>> daf.num_rows()
        2
        >>> c_daf = Daf(lol=[['a', 1], ['a', 2]], cols=['g', 'n'], keyfield=('g', 'n'))
        >>> c_daf
        <BLANKLINE>
        | g  | n  |
        | -: | -: |
        |  a |  1 |
        |  a |  2 |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield='('g', 'n')'; name=''
        <BLANKLINE>
        >>> c_daf.remove_key(('a', 1))
        <BLANKLINE>
        | g  | n  |
        | -: | -: |
        |  a |  2 |
        <BLANKLINE>
        %% daf rows=1; cols=2; keyfield='('g', 'n')'; name=''
        <BLANKLINE>

        A key that is not found raises KeyError, or is ignored with `silent_error=True`:

        >>> stock = Daf(cols=['id', 'qty'], lol=[['a1', 3], ['b2', 10]], keyfield='id')
        >>> stock.remove_key('z9')
        Traceback (most recent call last):
        ...
        KeyError: 'z9'
        >>> stock.remove_key('z9', silent_error=True).keys()
        ['a1', 'b2']
    """

    # test exists in test_daf.py
    if not self.keyfield and not self._kd:
        raise self._no_keys_error('remove_key')

    if (isinstance(keyval, tuple) and isinstance(self.keyfield, (tuple, list))
            and len(keyval) == len(self.keyfield)
            and not any(isinstance(item, (tuple, list)) for item in keyval)):
        keyval = [keyval]       # a composite key, not a range.

    return self.select_krows(krows=keyval, inverse=True, silent_error=silent_error)


def _remove_keylist(self: 'Daf', keylist: T_la, silent_error: bool=False) -> 'Daf':
    """
    Make a new Daf without the rows that have the given keys.

    This is deprecated. Use `select_krows(keys, inverse=True)`, which does the same.

    This does not remove the rows from this Daf. It returns a new Daf that leaves
    them out, and this Daf is unchanged. See `remove_key()` for how the rows are
    shared.

    Args:
        keylist: The keys of the rows to leave out.
        silent_error: If True, keys that are not found are ignored.

    Returns:
        The new Daf. It has the same keyfield.

    Raises:
        KeysDisabledError: The Daf has no keyfield and no key index.
        KeyError: A key is not found and `silent_error` is False.

    Examples:
        >>> daf = Daf(lol=[[1, 'a'], [2, 'b'], [3, 'c']], cols=['id', 'v'], keyfield='id')
        >>> daf
        <BLANKLINE>
        | id | v  |
        | -: | -: |
        |  1 |  a |
        |  2 |  b |
        |  3 |  c |
        <BLANKLINE>
        %% daf rows=3; cols=2; keyfield='id'; name=''
        <BLANKLINE>
        >>> daf.remove_keylist([1, 3])
        <BLANKLINE>
        | id | v  |
        | -: | -: |
        |  2 |  b |
        <BLANKLINE>
        %% daf rows=1; cols=2; keyfield='id'; name=''
        <BLANKLINE>
        >>> d2 = Daf(lol=[[1, 'a', 0], [2, 'b', 1]], cols=['p', 'q', 'r'], keyfield=['p', 'q'])
        >>> d2
        <BLANKLINE>
        | p  | q  | r  |
        | -: | -: | -: |
        |  1 |  a |  0 |
        |  2 |  b |  1 |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield='['p', 'q']'; name=''
        <BLANKLINE>
        >>> d2.remove_keylist([(1, 'a')])
        <BLANKLINE>
        | p  | q  | r  |
        | -: | -: | -: |
        |  2 |  b |  1 |
        <BLANKLINE>
        %% daf rows=1; cols=3; keyfield='['p', 'q']'; name=''
        <BLANKLINE>

        Keys that are not found raise KeyError, or are ignored with `silent_error=True`:

        >>> stock = Daf(cols=['id', 'qty'], lol=[['a1', 3], ['b2', 10], ['c3', 7]], keyfield='id')
        >>> stock.remove_keylist(['a1', 'z9'], silent_error=True).keys()
        ['b2', 'c3']
    """
    # test exists in test_daf.py

    if not self.keyfield and not self._kd:
        raise self._no_keys_error('remove_keylist')

    return self.select_krows(krows=keylist, inverse=True, silent_error=silent_error)


def _delete_records(
        self: 'Daf',
        keys_ls:        'T_ls | Iterable | Daf',    # the keys of the rows to delete, or a Daf whose keys are used.
        inplace:        bool = False,               # if True, delete from this Daf; if False, make a new one.
        silent_error:   bool = False,               # if True, keys that are not found are ignored.
        ) -> 'Daf':
    """
    Remove the rows with these keys.

    By default, this Daf is left as it is, and a new Daf without the rows is returned, as
    `select_records_daf(keys, inverse=True)` gives. That is the usual way in daffodil, and
    the new Daf shares the rows that remain.

    With `inplace=True`, the rows are deleted from this Daf itself, and it is returned. The
    row list is rebuilt once, in place, so it costs one pass however many rows go. The
    remaining rows keep their order, and the key index is rebuilt when next needed. That is
    safe for a Daf in memory: a selection made from it earlier holds its own row list, so it
    still has the deleted rows. On a SqlDaf parent table it is not safe in the same way. See
    `SqlDaf.delete_records()` in sqldaf.py.

    The keys may be a list, any iterable, or another Daf, whose keys are then used through a
    view of its key index, as in `select_records_daf()`.

    Args:
        keys_ls: The keys of the rows to remove, or a Daf whose keys are used.
        inplace: If True, delete from this Daf and return it. If False, return a new Daf.
        silent_error: If True, keys that are not found are ignored.

    Returns:
        The Daf without the rows: this one if `inplace`, otherwise a new one.

    Raises:
        KeysDisabledError: This Daf, or a Daf given as the keys, has no keyfield and no key index.
        KeyError: A key is not found and `silent_error` is False. Nothing is removed.

    Examples:
        >>> daf = Daf(lol=[['r1', 'a'], ['r2', 'b'], ['r3', 'c']], cols=['id', 'v'], keyfield='id')
        >>> daf
        <BLANKLINE>
        | id | v  |
        | -: | -: |
        | r1 |  a |
        | r2 |  b |
        | r3 |  c |
        <BLANKLINE>
        %% daf rows=3; cols=2; keyfield='id'; name=''
        <BLANKLINE>
        >>> daf.delete_records(['r2']).to_lod()
        [{'id': 'r1', 'v': 'a'}, {'id': 'r3', 'v': 'c'}]
        >>> len(daf)
        3
        >>> daf.delete_records(['r2', 'r9'], inplace=True, silent_error=True)
        <BLANKLINE>
        | id | v  |
        | -: | -: |
        | r1 |  a |
        | r3 |  c |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield='id'; name=''
        <BLANKLINE>
        >>> daf.to_lod()
        [{'id': 'r1', 'v': 'a'}, {'id': 'r3', 'v': 'c'}]
    """
    if not inplace:
        return self.select_records_daf(keys_ls, inverse=True, silent_error=silent_error)

    if isinstance(keys_ls, Daf):
        keys_ls = keys_ls.keys(silent_error=False, astype='view')
    if not keys_ls:
        return self

    irows: Any = self.krows_to_irows(keys_ls, silent_error=silent_error)
    drop_irows: set = set(irows if isinstance(irows, list) else ([irows] if isinstance(irows, int) else irows))
    if drop_irows:
        self.lol[:] = [row_la for irow, row_la in enumerate(self.lol) if irow not in drop_irows]
        self._invalidate_kd()
    return self


def _remove_dups(self: 'Daf', keyfield: str | T_ta | T_la='') -> Tuple['Daf', 'Daf']:  # unique_daf, duplicates_daf
    """
    Split the rows into those with a unique key and those with a repeated key.

    Only the key columns are compared, not the whole row. For each key, the last row
    is kept as the unique one. The earlier rows with that key are the duplicates.
    The unique rows are in the order in which their keys first appear.

    Without `keyfield`, the keyfield of this Daf is used. If there is none, a
    `KeysDisabledError` is raised. This Daf is not changed. Both results share their
    rows with it, and both have the key columns as their keyfield.

    Args:
        keyfield: The column, or tuple or list of columns, that identifies a row.
            If empty, the keyfield of this Daf.

    Returns:
        A tuple of the Daf of unique rows and the Daf of duplicate rows.

    Raises:
        KeysDisabledError: There is no `keyfield` and this Daf has none.
        KeyError: A key column is not found.

    Examples:
        >>> daf = Daf(lol=[[1, 'a'], [2, 'b'], [1, 'c']], cols=['id', 'v'])
        >>> daf
        <BLANKLINE>
        | id | v  |
        | -: | -: |
        |  1 |  a |
        |  2 |  b |
        |  1 |  c |
        <BLANKLINE>
        %% daf rows=3; cols=2; keyfield=''; name=''
        <BLANKLINE>
        >>> unique, dups = daf.remove_dups('id')
        >>> unique
        <BLANKLINE>
        | id | v  |
        | -: | -: |
        |  1 |  c |
        |  2 |  b |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield='id'; name=''
        <BLANKLINE>
        >>> dups
        <BLANKLINE>
        | id | v  |
        | -: | -: |
        |  1 |  a |
        <BLANKLINE>
        %% daf rows=1; cols=2; keyfield='id'; name=''
        <BLANKLINE>
    """
    """
        If it is known that duplicates may exist in the array with respect to keyfield,
        remove records that have the same keyfield and return two daf arrays,
        uniques_daf, dups_daf.

        Please note that this does NOT compare all components, only the keyfields.

        Returns unique_daf array with keyfield set and duplicates_daf, which may have repeats.

        """
    key_cols: Any = keyfield or self.keyfield
    if not key_cols:
        raise KeysDisabledError("remove_dups: give a keyfield, or set one on the Daf.")

    # a key index built here, so this Daf keeps its own keyfield and key index.
    if isinstance(key_cols, (str, int)):
        kd = type(self)._build_kd(self.hd[key_cols], self.lol)  # type: ignore[index]  # an int keyfield is a column name here
    else:
        kd = type(self)._build_kd([self.hd[cast(str, col)] for col in key_cols], self.lol)

    # irows of the last record of each key. These are the unique records.
    unique_irows = list(kd.values())

    unique_daf = self.select_irows(irows=unique_irows, inverse=False)
    unique_daf.keyfield = key_cols

    # all the other records are duplicates.
    dups_daf = self.select_irows(irows=unique_irows, inverse=True)
    dups_daf.keyfield = key_cols

    return unique_daf, dups_daf


def _drop_cols(self: 'Daf', exclude_cols: T_ls | None = None) -> 'Daf':
    """
    Remove columns from this Daf, in place.

    The rows are rebuilt without those columns, so this copies all the data. Avoid
    it for large tables. Instead, name only the columns you need: with `cols=` in
    `reduce()`, by reading only those fields in an `apply()` function, or with
    `select_kcols()` for a new Daf.

    The column names and dtypes are updated. A name that is not a column is ignored.
    If a column of the keyfield is dropped, the keyfield is cleared, as in the other
    methods that remove a key column. Set it again with `set_keyfield()`. With no
    names, nothing happens.

    Args:
        exclude_cols: The names of the columns to remove.

    Returns:
        This Daf, which has been changed.

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
        >>> daf.drop_cols(['v']).columns()
        ['id', 'n']
    """

    if exclude_cols:
        keep_idxs_li: T_li = [self.hd[col] for col in self.hd if col not in exclude_cols]

    else:
        return self

    for irow, la in enumerate(self.lol):
        la = [la[idx] for idx in keep_idxs_li]
        self.lol[irow] = la

    old_cols = list(self.hd.keys())
    new_cols = [old_cols[idx] for idx in keep_idxs_li]
    self._cols_to_hd(new_cols)

    if self.dtypes:
        kept_cols = {old_cols[idx] for idx in keep_idxs_li}
        new_dtypes = {col: typ for col, typ in self.dtypes.items() if col in kept_cols}
        self.dtypes = new_dtypes

    # a keyfield that lost a column can no longer find its rows, so clear it, as the other
    # methods that remove a key column do.
    key_cols = [self.keyfield] if isinstance(self.keyfield, (str, int)) else list(self.keyfield or [])
    if key_cols and any(col in exclude_cols for col in key_cols):
        self.keyfield = ''
        self._kd = {}       # _invalidate_kd() leaves the index alone when there is no keyfield.

    return self


def _assign_record(self: 'Daf', record: T_da) -> 'Daf':
    """
    Put one row in the Daf by its key, replacing a row that has the same key.

    The row is a dict. If a row with that key exists, the whole row is replaced.
    The cells of columns that the dict lacks become NULL. If the key is new, the
    row is added at the end. This is an upsert for one row. For that, `append()`
    with `respect_kd=True` also works. Use `update_by_keylist()` to change only
    some cells.

    Args:
        record: The row, as a dict. It must have the keyfield.

    Returns:
        This Daf, which has been changed.

    Raises:
        KeysDisabledError: The Daf has no keyfield.

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
        >>> daf.assign_record({'id': 2, 'v': 'new'})
        <BLANKLINE>
        | id |  v  | n  |
        | -: | --: | -: |
        |  1 |   a | 10 |
        |  2 | new |    |
        |  3 |   c | 30 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>
    """

    if not self.keyfield:
        raise KeysDisabledError("assign_record(): the Daf needs a keyfield to read the key from the record.")

    # test if valid keyfield, produce an error if not.
    self._is_keyfield_valid()

    keyval = self._get_keyval(record)

    self._rebuild_kd_if_invalidated()

    if keyval in self._kd:
        # assign the record, normalize the fields.
        self.lol[self._kd[keyval]] = [record.get(col, '') for col in self.hd]
    else:
        # otherwise add it, and invalidate kd.
        self.append(record)

    return self


def _assign_record_irow(self: 'Daf', irow: int | None = None, record: T_da | None = None) -> 'Daf':
    """
    Put one row in the Daf by position, replacing the row there.

    The row is a dict. The whole row is replaced, and the cells of columns that
    the dict lacks become NULL. Use `update_record_irow()` to change only some cells.

    With the default position, `None`, the row is added at the end. So it is with a
    position beyond the last row, and with any position when the Daf has no rows.
    A negative position counts from the end, as in a list, so `-1` replaces the last
    row. This is also what `my_daf[-1] = {...}` does.

    This does not update the key index (kd), so don't change a key with it. See
    [`__setitem__`][daffodil.daf.Daf.__setitem__].

    Args:
        irow: The row position. None, the default, adds the row at the end.
        record: The row, as a dict. If None, nothing happens.

    Returns:
        This Daf, which has been changed.

    Raises:
        IndexError: A negative position is before the first row, for a Daf that has rows.

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
        >>> daf.assign_record_irow(1, {'v': 'q'})
        <BLANKLINE>
        | id | v  | n  |
        | -: | -: | -: |
        |  1 |  a | 10 |
        |    |  q |    |
        |  3 |  c | 30 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>
        >>> daf.iloc(1)
        {'id': '', 'v': 'q', 'n': ''}
        >>> daf.assign_record_irow(-1, {'id': 9})
        <BLANKLINE>
        | id | v  | n  |
        | -: | -: | -: |
        |  1 |  a | 10 |
        |    |  q |    |
        |  9 |    |    |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>
        >>> daf.iloc(-1)
        {'id': 9, 'v': '', 'n': ''}
        >>> daf.assign_record_irow(record={'id': 10})
        <BLANKLINE>
        | id | v  | n  |
        | -: | -: | -: |
        |  1 |  a | 10 |
        |    |  q |    |
        |  9 |    |    |
        | 10 |    |    |
        <BLANKLINE>
        %% daf rows=4; cols=3; keyfield='id'; name=''
        <BLANKLINE>
        >>> daf.iloc(-1), len(daf)
        ({'id': 10, 'v': '', 'n': ''}, 4)
    """

    if record is None:
        return self

    if irow is not None and irow < 0 and self.lol:
        irow += len(self.lol)               # count from the end, as a list does.
        if irow < 0:
            raise IndexError(f"assign_record_irow(): row position {irow - len(self.lol)} is before the first row.")

    if irow is None or irow < 0 or irow >= len(self.lol):
        self.append(record)
    else:
        #normal_record_da = Daf.normalize_record_da(record_da, cols=self.columns(), dtypes=self.dtypes)
        self.lol[irow] = [record.get(col, '') for col in self.hd]

    return self


def _update_by_keylist(self: 'Daf', keylist: T_ls | None = None, record: T_da | None = None) -> 'Daf':
    """
    Change some cells in the rows that have the given keys.

    Only the columns that are keys of the dict are changed. Other cells keep
    their values. A key that is not found is skipped. This is a bulk form of
    `update_record_krow()`, for rows that all take the same values. The row positions
    do not change, so the key index stays valid.

    Args:
        keylist: The keys of the rows to change.
        record: The new values, as a dict of column name and value. Names that are not columns are ignored.

    Returns:
        This Daf, which has been changed. With no keyfield, rows, keys or record, nothing happens.

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
        >>> daf.update_by_keylist([1, 3, 9], {'v': 'q'})
        <BLANKLINE>
        | id | v  | n  |
        | -: | -: | -: |
        |  1 |  q | 10 |
        |  2 |  b | 20 |
        |  3 |  q | 30 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>
    """

    if record is None or not self.lol or not self.hd or not self.keyfield or not keylist:
        return self

    self._rebuild_kd_if_invalidated()

    for key in keylist:
        if key in self._kd:
            self.update_record_irow(self._kd[key], record)

    # no need to invalidate kd because key is unchanged by definition.

    return self


def _update_record_irow(self: 'Daf', irow: int=-1, record: T_da | None = None) -> 'Daf':
    """
    Change some cells in the row at a position.

    Only the columns that are keys of the dict are changed. Other cells keep their
    values. A negative position counts from the end, as in a list, so the default,
    `-1`, is the last row. A position that is out of range raises `IndexError`, and
    nothing is changed. A Daf with no rows or no columns, and a record of `None`, change nothing.

    This does not update the key index (kd), so don't change a key with it. See
    [`__setitem__`][daffodil.daf.Daf.__setitem__].

    Args:
        irow: The row position. -1, the default, is the last row.
        record: The new values, as a dict of column name and value. Names that are not columns are ignored.

    Returns:
        This Daf, which has been changed.

    Raises:
        IndexError: The position is out of range, for a Daf that has rows and columns.

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
        >>> daf.update_record_irow(1, {'v': 'q'})
        <BLANKLINE>
        | id | v  | n  |
        | -: | -: | -: |
        |  1 |  a | 10 |
        |  2 |  q | 20 |
        |  3 |  c | 30 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>
        >>> daf.iloc(1)
        {'id': 2, 'v': 'q', 'n': 20}
        >>> daf.update_record_irow(record={'n': 99})
        <BLANKLINE>
        | id | v  | n  |
        | -: | -: | -: |
        |  1 |  a | 10 |
        |  2 |  q | 20 |
        |  3 |  c | 99 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>
        >>> daf.iloc(-1)
        {'id': 3, 'v': 'c', 'n': 99}
    """

    if record is None or not self.lol or not self.hd:
        return self

    position = irow
    if irow < 0:
        irow += len(self.lol)               # count from the end, as a list does.

    if irow < 0 or irow >= len(self.lol):
        raise IndexError(f"update_record_irow(): row position {position} is out of range for {len(self.lol)} rows.")

    for colname, val in record.items():
        if colname in self.hd:
            self.lol[irow][self.hd[colname]] = record[colname]          # perflint-reviewed (loop-invariant-statement)

        # icol = self.hd.get(colname, -1)
        # if icol >= 0:
            # self.lol[irow][icol] = record_da[colname]

    return self


def _update_record_krow(self: 'Daf', key: Any, record: T_da | None = None) -> 'Daf':
    """
    Change some cells in the row that has this key.

    Only the columns that are keys of the dict are changed. Other cells keep their
    values. It is the form of `update_record_irow()` that finds the row by its key.
    An int is always a key here, never a position, so it suits a keyfield of ints.
    For many rows that take the same values, use `update_by_keylist()`.

    As with `update_record_irow()`, this does not update the key index (kd), so don't
    change a key with it. See [`__setitem__`][daffodil.daf.Daf.__setitem__].
    A record of `None` changes nothing.

    Args:
        key: The key of the row to change.
        record: The new values, as a dict of column name and value. Names that are not columns are ignored.

    Returns:
        This Daf, which has been changed.

    Raises:
        KeyError: No row has this key.
        ValueError: The Daf has no keyfield.

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
        >>> daf.update_record_krow(2, {'v': 'q'})
        <BLANKLINE>
        | id | v  | n  |
        | -: | -: | -: |
        |  1 |  a | 10 |
        |  2 |  q | 20 |
        |  3 |  c | 30 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>
        >>> daf.select_record(2)
        {'id': 2, 'v': 'q', 'n': 20}
    """

    if record is None:
        return self
    if not self.keyfield:
        raise ValueError("update_record_krow(): the Daf has no keyfield.")

    self._rebuild_kd_if_invalidated()
    irow: int = self._kd[key]                   # a missing key raises KeyError, naming it.

    row_la: T_la = self.lol[irow]
    hd: T_di = self.hd
    colname: str
    val: Any
    for colname, val in record.items():
        if colname in hd:
            row_la[hd[colname]] = val

    return self


def _assign_icol(
        self: 'Daf',
        icol: int=-1,
        col_la: T_la | None = None,
        default: Any=''
        ) -> 'Daf':
    """
    Fill a column by position with the values of a list.

    A list that is too short is filled out with `default`. With no list, every cell
    gets `default`. With `icol=-1` a new column is added at the right. If the Daf has
    column names, the new column gets the next spreadsheet name, such as `C`, made
    unique. Use `insert_col()` to add a column with a name of your own.

    Args:
        icol: The column position. -1 adds a column at the right.
        col_la: The values, one for each row.
        default: The value for rows that the list does not reach.

    Returns:
        This Daf, which has been changed.

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
        >>> daf.assign_icol(1, ['x', 'y'], default='D')
        <BLANKLINE>
        | id | v  | n  |
        | -: | -: | -: |
        |  1 |  x | 10 |
        |  2 |  y | 20 |
        |  3 |  D | 30 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>
        >>> daf.col('v')
        ['x', 'y', 'D']
    """
    # from utilities import daf_utils

    if self.lol and (icol < 0 or icol >= len(self.lol[0])):
        self._own_rows()        # a column is added. Rows shared with another Daf must not grow.

    self.lol = daf_utils.assign_col_in_lol_at_icol(icol, col_la, lol=self.lol, default=default)
    self._invalidate_kd_if_key([icol])

    if self.hd and self.lol and len(self.lol[0]) > len(self.hd):
        # a column was added at the right. Name it, so the names match the data.
        self._cols_to_hd(list(self.hd) + [self._new_colname()])

    return self


def _own_rows(self: 'Daf') -> None:
    """
    Give this Daf its own copy of the rows if any row is shared with another Daf.

    Call this before writing into rows. It is a lazy copy. Nothing is copied when the rows are not shared.
    Internal use.
    """
    if daf_utils.rows_are_shared(self.lol):
        self.lol = [list(row_la) for row_la in self.lol]


def _new_colname(self: 'Daf') -> str:
    """
    Make a name for a new column that has no name: the next spreadsheet name, such as `C`.

    If that name is taken, a suffix is added, such as `C_1`. Internal use.
    """
    base = daf_utils._calculate_single_column_name(len(self.hd))
    name = base
    suffix = 0
    while name in self.hd:
        suffix += 1
        name = f"{base}_{suffix}"
    return name


def _insert_icol(
        self: 'Daf',
        icol:       int=-1,
        col_la:     T_la | None = None,
        colname:    str='',
        default:    Any=''
        ) -> 'Daf':
    """
    Insert a column at a position and move the later columns right.

    A list that is too short is filled out with `default`. With `icol=-1`, or a
    position beyond the last column, the column is added at the right. Give
    `colname` to name it. Without a name, if the Daf has column names, the column
    gets the next spreadsheet name, such as `C`, made unique. The dtypes are not
    changed. Use `set_keyfield()` if the column is to be the keyfield.

    If the rows are shared with another Daf, as after a selection, this Daf first
    takes its own copies of the rows. The other Daf is not changed.

    Args:
        icol: The column position. -1 adds the column at the right.
        col_la: The values, one for each row.
        colname: The name of the new column.
        default: The value for rows that the list does not reach.

    Returns:
        This Daf, which has been changed.

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
        >>> daf.insert_icol(1, ['x', 'y', 'z'], colname='new').columns()
        ['id', 'new', 'v', 'n']

        A list shorter than the rows is filled out with `default`:

        >>> stock = Daf(cols=['id', 'qty'], lol=[['a1', 3], ['b2', 10], ['c3', 7]], keyfield='id')
        >>> stock.insert_icol(1, ['red'], 'color', default='none')
        <BLANKLINE>
        | id | color | qty |
        | -: | ----: | --: |
        | a1 |   red |   3 |
        | b2 |  none |  10 |
        | c3 |  none |   7 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>
    """

    # from utilities import daf_utils

    self._own_rows()        # the insert must not change the rows of another Daf.

    self.lol = daf_utils.insert_col_in_lol_at_icol(icol, col_la, lol=self.lol, default=default)

    if self.hd and not colname:
        colname = self._new_colname()       # the names must match the data.

    if colname:
        if not self.hd:
            self.hd = {}
        if icol < 0 or icol >= len(self.hd):
            icol = len(self.hd)
        hl = list(self.hd.keys())
        hl.insert(icol, colname)
        self.hd = {k: idx for idx, k in enumerate(hl)}

    return self


def _insert_irow(self: 'Daf', irow: int | None = None, row: T_la | T_da | None = None, default: Any='') -> 'Daf':
    """
    Insert a row at a position and move the later rows down.

    The row is a list of values, or a dict that is placed by column name. A short
    list is filled out with `default`. With the default position, `None`, or with a
    position beyond the last row, the row is added at the end. A negative position,
    such as `-1`, also adds it at the end. This differs from `list.insert()`, which
    puts the item before the last one. Prefer `None`. The key index is rebuilt when
    it is next needed.

    Args:
        irow: The row position. None, the default, adds the row at the end.
        row: The row, as a list or a dict.
        default: The value for cells that a short list does not reach.

    Returns:
        This Daf, which has been changed.

    Raises:
        TypeError: The row is not a list or a dict. This includes the default, None.

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
        >>> daf.insert_irow(1, {'id': 9, 'v': 'z'})
        <BLANKLINE>
        | id | v  | n  |
        | -: | -: | -: |
        |  1 |  a | 10 |
        |  9 |  z |    |
        |  2 |  b | 20 |
        |  3 |  c | 30 |
        <BLANKLINE>
        %% daf rows=4; cols=3; keyfield='id'; name=''
        <BLANKLINE>

        A short list is filled out with `default`:

        >>> stock = Daf(cols=['id', 'item', 'qty'], lol=[['a1', 'pen', 3], ['b2', 'ink', 10]], keyfield='id')
        >>> stock.insert_irow(1, ['z0'], default='?')
        <BLANKLINE>
        | id | item | qty |
        | -: | ---: | --: |
        | a1 |  pen |   3 |
        | z0 |    ? |   ? |
        | b2 |  ink |  10 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>
    """

    # from utilities import daf_utils

    if isinstance(row, list):

        row_la = row

    elif isinstance(row, dict):

        row_da = row
        # create normalize list
        row_la = [row_da.get(col, '') for col in self.hd]

    else:
        raise TypeError(f"insert_irow(): row must be a list or a dict, not {type(row).__name__}.")

    self.lol = daf_utils.insert_row_in_lol_at_irow(irow=-1 if irow is None else irow, row_la=row_la, lol=self.lol, default=default)

    self._invalidate_kd()    # use lazy kd rebuilding
    #self._rebuild_kd()
    return self


def _assign_col(self: 'Daf', colname: str, la: T_la | None = None, default: Any='') -> 'Daf':
    """
    Fill a column by name with the values of a list, or add it if it is new.

    This is `my_daf[:, colname] = values`, and it also adds a column. A list that
    is too short is filled out with `default`. With no list, every cell gets
    `default`. If the column is the keyfield, the key index is rebuilt when it is
    next needed.

    Args:
        colname: The column name.
        la: The values, one for each row.
        default: The value for rows that the list does not reach.

    Returns:
        This Daf, which has been changed.

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
        >>> daf.assign_col('w', default=0).columns()
        ['id', 'v', 'n', 'w']
        >>> daf.col('w')
        [0, 0, 0]

        A list shorter than the rows is filled out with `default`. A new name adds a column at the right:

        >>> stock = Daf(cols=['id', 'qty'], lol=[['a1', 3], ['b2', 10], ['c3', 7]], keyfield='id')
        >>> stock.assign_col('qty', [1], default=0).col('qty')
        [1, 0, 0]
        >>> stock.assign_col('note', ['n1', 'n2', 'n3'])
        <BLANKLINE>
        | id | qty | note |
        | -: | --: | ---: |
        | a1 |   1 |   n1 |
        | b2 |   0 |   n2 |
        | c3 |   0 |   n3 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>
    """

    if colname in self.hd:
        self.assign_icol(self.hd[colname], la, default)

    else:
        self.insert_col(
            colname = colname,
            col_la = la,
            #icol: int=-1,
            default = default,
            )
    if colname in self.hd:
        self._invalidate_kd_if_key([self.hd[colname]])  # also for a column of a composite keyfield.

    return self


def _insert_col(
        self: 'Daf',
        colname:    str,                    # name of the col
        col_la:     T_la | None = None,    # column to insert
        icol:       int=-1,                 # insert at end by default
        default:    Any='',
        ) -> 'Daf':

    """
    Add a named column at a position, or overwrite it if the name exists.

    A list that is too short is filled out with `default`. With no list, every cell
    gets `default`, so this can add a constant column. If the name already exists
    the column is overwritten, and `icol` is ignored. An empty name does nothing.
    Use `set_keyfield()` if the column is to be the keyfield.

    When a column is added and the rows are shared with another Daf, as after a
    selection, this Daf first takes its own copies of the rows. The other Daf is
    not changed.

    Args:
        colname: The name of the column.
        col_la: The values, one for each row.
        icol: The column position. -1 adds the column at the right.
        default: The value for rows that the list does not reach.

    Returns:
        This Daf, which has been changed.

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
        >>> daf.insert_col('w', ['x', 'y', 'z'], icol=1).columns()
        ['id', 'w', 'v', 'n']
        >>> daf.insert_col('k', default=5).col('k')
        [5, 5, 5]
    """

    if not colname:
        return self
    if not col_la:
        col_la = []             # perflint-reviewed (use-tuple-over-list)

    if colname in self.hd:
        # column already exists. ignore icol, overwrite data.
        self.assign_col(colname, col_la, default)

    else:
        self.insert_icol(icol=icol, col_la=col_la, colname=colname, default=default) #, keyfield=keyfield)

    return self


def _insert_idx_col(self: 'Daf', colname: str='idx', icol:int=0, startat:int=0) -> 'Daf':
    """
    Insert a column of row numbers.

    This is `insert_col()`, so rows shared with another Daf are copied first and
    the other Daf is not changed.

    Args:
        colname: The name of the new column.
        icol: The column position.
        startat: The number of the first row.

    Returns:
        This Daf, which has been changed.

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
        >>> daf.insert_idx_col().col('idx')
        [0, 1, 2]

        Another name, in another place, numbered from 1:

        >>> stock = Daf(cols=['id', 'qty'], lol=[['a1', 3], ['b2', 10], ['c3', 7]], keyfield='id')
        >>> stock.insert_idx_col('n', icol=1, startat=1)
        <BLANKLINE>
        | id | n  | qty |
        | -: | -: | --: |
        | a1 |  1 |   3 |
        | b2 |  2 |  10 |
        | c3 |  3 |   7 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>
    """

    num_rows = len(self)
    col_la = list(range(startat, startat + num_rows))

    self.insert_col(colname, col_la, icol)

    return self


def _set_col_irows(self: 'Daf', colname: str, irows: T_li, val: Any) -> 'Daf':
    """
    Set one value in the given rows of a named column.

    This is deprecated. Use `my_daf[irows, colname] = value`, which does the same. A
    column name that is not found raises `KeyError`. Row positions that are out of
    range are skipped.

    Args:
        colname: The column name.
        irows: The row positions.
        val: The value to set.

    Returns:
        This Daf, which has been changed.

    Raises:
        KeyError: The column name is not found.

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
        >>> daf.set_col_irows('v', [0, 2], 'Z').col('v')
        ['Z', 'b', 'Z']
    """

    icol = self.hd[colname]

    self.set_icol_irows(icol, irows, val)

    return self


def _set_icol(self: 'Daf', icol: int, val: Any) -> 'Daf':

    """
    Set one value in every row of a column, by position.

    This is `daf[:, icol] = value`. A negative position counts from the end. To give each row a
    value of its own, assign a list, as `daf[:, icol] = values`.

    Args:
        icol: The column position.
        val: The value to set.

    Returns:
        This Daf, which has been changed.

    Raises:
        IndexError: The position is beyond the end of a row.

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
        >>> daf.set_icol(1, 'Z').col('v')
        ['Z', 'Z', 'Z']
    """


    for irow in range(len(self.lol)):
        self.lol[irow][icol] = val

    self._invalidate_kd_if_key([icol])

    return self


def _set_icol_irows(self: 'Daf', icol: int, irows: T_li, val: Any) -> 'Daf':
    """
    Set one value in the given rows of a column, by position.

    This is `my_daf[irows, icol] = value`. Row positions that are out of range
    are skipped.

    Args:
        icol: The column position.
        irows: The row positions.
        val: The value to set.

    Returns:
        This Daf, which has been changed.

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
        >>> daf.set_icol_irows(1, [0, 9, -1], 'Z')
        <BLANKLINE>
        | id | v  | n  |
        | -: | -: | -: |
        |  1 |  Z | 10 |
        |  2 |  b | 20 |
        |  3 |  c | 30 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>
        >>> daf.col('v')
        ['Z', 'b', 'c']
    """

    for irow in irows:
        if irow >= len(self.lol) or irow < 0:
            continue

        self.lol[irow][icol] = val

    self._invalidate_kd_if_key([icol])

    return self


def _find_replace(self: 'Daf', find_pat: str, replace_val: Any) -> 'Daf':
    """
    Replace every cell that matches a pattern, in place.

    Each cell is converted to text and searched with the regular expression
    `find_pat`. If it matches anywhere in the text, the whole cell is replaced by
    `replace_val`. It is not a substitution inside the text. Every column is
    searched, including numbers, and the key index is rebuilt when it is next
    needed.

    Args:
        find_pat: A regular expression.
        replace_val: The value that replaces a matching cell.

    Returns:
        This Daf, which has been changed.

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
        >>> daf.find_replace(r'^[ab]$', 'HIT')
        <BLANKLINE>
        | id |  v  | n  |
        | -: | --: | -: |
        |  1 | HIT | 10 |
        |  2 | HIT | 20 |
        |  3 |   c | 30 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>
        >>> daf.col('v')
        ['HIT', 'HIT', 'c']
    """

    for row_la in self.lol:
        for i, value in enumerate(row_la):
            if bool(re.search(find_pat, str(value))):
                row_la[i] = replace_val

    self._invalidate_kd()

    return self


def _replace_in_columns(
    self: 'Daf',
    cols: T_ls | T_li | T_lsi | None,
    find_values: List[Any] | None = None,
    replacement: Any = _MISSING
) -> 'Daf':
    """
    Replace listed values with another value, in some columns, in place.

    A cell is replaced when it equals any value in `find_values`. A typical use
    is `['', None]` to fill empty cells. Columns may be given by name or by
    position, or all columns if `cols` is None. With no `find_values` nothing
    happens. The key index is rebuilt if the keyfield may have changed.

    Args:
        cols: Column names or positions. If None, all columns.
        find_values: The values to look for.
        replacement: The value to put in their place. This is required.

    Returns:
        This Daf, which has been changed.

    Raises:
        ValueError: No `replacement` is given.
        KeyError: A column name is not found.
        TypeError: A column is neither a name nor a position.

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
        >>> daf.replace_in_columns(['v'], ['a', 'c'], '-').col('v')
        ['-', 'b', '-']
    """

    if find_values is None:
        return self

    if replacement is _MISSING:
        raise ValueError("replace_in_columns() requires an explicit `replacement` value.")

    if cols is None:
        # process the entire array
        target_indices = list(range(self.num_cols()))
    else:
        target_indices = []
        for col in cols:
            if isinstance(col, int):
                target_indices.append(col)
            elif isinstance(col, str):
                if col in self.hd:
                    target_indices.append(self.hd[col])
                else:
                    raise KeyError(f"Column name '{col}' not found in header.")
            else:
                raise TypeError(f"Column specifier must be str or int, got {type(col).__name__}")

    for row in self.lol:
        for col_idx in target_indices:
            if row[col_idx] in find_values:
                row[col_idx] = replacement

    if self.keyfield:
        if not isinstance(self.keyfield, str):
            # composite keyfield (tuple/list of columns) -- conservatively always invalidate,
            # since checking whether any individual component column was touched is more work
            # than it's worth here.
            self._invalidate_kd()
        else:
            keyfield_idx = self.hd.get(self.keyfield)
            if keyfield_idx is None or keyfield_idx in target_indices:
                self._invalidate_kd()

    return self


def _split_daf_into_ranges(self: 'Daf', chunk_ranges: List[Tuple[int, int]]) -> List['Daf']:
    """
    Split the rows into several Daf instances, by position ranges.

    Each range is `(start, end)`, and `end` is not included. The new Daf instances
    share their rows with this one.

    Args:
        chunk_ranges: The ranges of row positions.

    Returns:
        A list of Daf instances, one for each range.

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
        >>> parts = daf.split_daf_into_ranges([(0, 2), (2, 3)])
        >>> parts[0]
        <BLANKLINE>
        | id | v  | n  |
        | -: | -: | -: |
        |  1 |  a | 10 |
        |  2 |  b | 20 |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield='id'; name=''
        <BLANKLINE>
        >>> parts[1]
        <BLANKLINE>
        | id | v  | n  |
        | -: | -: | -: |
        |  3 |  c | 30 |
        <BLANKLINE>
        %% daf rows=1; cols=3; keyfield='id'; name=''
        <BLANKLINE>
    """

    chunks_lodaf = [self.select_irows(list(range(start, end))) for start,end in chunk_ranges]
    #chunks_lodaf = [self[start:end] for start, end in chunk_ranges]
    return chunks_lodaf


def _split_daf_into_chunks_lodaf(self: 'Daf', max_chunk_size: int) -> List['Daf']:
    """
    Split the rows evenly into Daf instances of at most a given size.

    The sizes are as equal as possible, so some chunks are smaller than the
    maximum. None is larger. The chunks share their rows with this Daf.

    Args:
        max_chunk_size: The most rows in one chunk.

    Returns:
        A list of Daf instances.

    Examples:
        >>> daf = Daf(lol=[[i] for i in range(7)], cols=['a'])
        >>> daf
        <BLANKLINE>
        | a  |
        | -: |
        |  0 |
        |  1 |
        |  2 |
        |  3 |
        |  4 |
        |  5 |
        |  6 |
        <BLANKLINE>
        %% daf rows=7; cols=1; keyfield=''; name=''
        <BLANKLINE>
        >>> [len(part) for part in daf.split_daf_into_chunks_lodaf(3)]
        [3, 2, 2]
    """
    # from utilities import daf_utils

    chunk_sizes_list = daf_utils.calc_chunk_sizes(num_items=len(self), max_chunk_size=max_chunk_size)
    chunk_ranges = daf_utils.convert_sizes_to_idx_ranges(chunk_sizes_list)
    chunks_lodaf = self.split_daf_into_ranges(chunk_ranges)
    return chunks_lodaf


def _sort_by_colname(self: 'Daf', colname:str, *, reverse: bool=False, length_priority: bool=False, as_str: bool=False) -> 'Daf':
    """
    Sort the rows by one column, in place.

    Make a copy first if you need the original order. An empty cell sorts before
    a cell with content, unlike a spreadsheet. Values in the column must be
    comparable with each other. A column that mixes None or text with numbers
    raises `TypeError`, which names the column. Use `as_str=True` to sort by the text
    of each value instead.

    With `length_priority`, a shorter text sorts before a longer one, so numbers
    that are stored as text sort as numbers. Without it, `'10'`, `'100'`, `'8'`
    are in text order. `length_priority` needs text, so for real numbers use it
    with `as_str=True`. Then whole numbers sort in numeric order, and negative and
    decimal numbers sort in text order.

    Args:
        colname: The column to sort by.
        reverse: If True, sort from high to low.
        length_priority: If True, sort by length first, then by value.
        as_str: If True, sort by the text of each value, and None sorts as an empty cell.

    Returns:
        This Daf, which has been changed.

    Raises:
        KeyError: The column name is not found.
        TypeError: The column holds values that cannot be compared, and `as_str` is False.

    Examples:
        >>> daf = Daf(lol=[['10'], ['99'], ['8'], ['100']], cols=['a'])
        >>> daf
        <BLANKLINE>
        |  a  |
        | --: |
        |  10 |
        |  99 |
        |   8 |
        | 100 |
        <BLANKLINE>
        %% daf rows=4; cols=1; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.sort_by_colname('a', length_priority=True).col('a')
        ['8', '10', '99', '100']
        >>> daf = Daf(lol=[[10], [None], [2], ['']], cols=['n'])
        >>> daf
        <BLANKLINE>
        |  n   |
        | ---: |
        |   10 |
        | None |
        |    2 |
        |      |
        <BLANKLINE>
        %% daf rows=4; cols=1; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.sort_by_colname('n', as_str=True, length_priority=True).col('n')
        [None, '', 2, 10]

        From high to low:

        >>> stock = Daf(cols=['id', 'qty'], lol=[['a1', 3], ['b2', 1], ['c3', 2]])
        >>> stock.sort_by_colname('qty', reverse=True).col('id')
        ['a1', 'c3', 'b2']
    """
    if not self or len(self) <= 1:
        return self

    colidx = self.hd[colname]

    try:
        self.lol = daf_utils.sort_lol_by_col(self.lol, colidx, reverse=reverse, length_priority=length_priority, as_str=as_str)
    except TypeError as exc_info:
        raise TypeError(
            f"sort_by_colname(): column '{colname}' holds values that cannot be compared, such as numbers "
            f"with text or None. Use as_str=True to sort by the text of the values, or convert the column "
            f"with apply_dtypes().") from exc_info

    self._invalidate_kd()    # use lazy kd rebuilding
    return self


def _sort_by_colnames(self: 'Daf', colnames:T_ls, reverse: bool=False, length_priority: bool=False, as_str: bool=False) -> 'Daf':
    """
    Sort the rows by several columns, in place.

    The first column is the main sort key. The others break ties. See
    `sort_by_colname()` for the rules of ordering, for `length_priority` and for `as_str`.
    Calling `sort_by_colname()` for each column, last column first, gives the same
    order.

    Args:
        colnames: The columns to sort by, main key first.
        reverse: If True, sort from high to low.
        length_priority: If True, sort by length first, then by value.
        as_str: If True, sort by the text of each value, and None sorts as an empty cell.

    Returns:
        This Daf, which has been changed.

    Raises:
        KeyError: A column name is not found.
        TypeError: A column holds values that cannot be compared, and `as_str` is False.

    Examples:
        >>> daf = Daf(lol=[[2, 'b'], [1, 'z'], [1, 'a']], cols=['p', 'q'])
        >>> daf
        <BLANKLINE>
        | p  | q  |
        | -: | -: |
        |  2 |  b |
        |  1 |  z |
        |  1 |  a |
        <BLANKLINE>
        %% daf rows=3; cols=2; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.sort_by_colnames(['p', 'q'])
        <BLANKLINE>
        | p  | q  |
        | -: | -: |
        |  1 |  a |
        |  1 |  z |
        |  2 |  b |
        <BLANKLINE>
        %% daf rows=3; cols=2; keyfield=''; name=''
        <BLANKLINE>

        The options, on a column of names:

        >>> names = Daf(cols=['id', 'name'], lol=[['a', 'bb'], ['b', 'a'], ['c', 'ccc'], ['d', 'b']])
        >>> names.sort_by_colnames(['name']).col('name')
        ['a', 'b', 'bb', 'ccc']
        >>> names.sort_by_colnames(['name'], reverse=True).col('name')
        ['ccc', 'bb', 'b', 'a']

        With `length_priority=True`, shorter values come first, and values of one length by value:

        >>> names.sort_by_colnames(['name'], length_priority=True).col('name')
        ['a', 'b', 'bb', 'ccc']

        A column of numbers mixed with text, or with None, cannot be compared, and raises TypeError.
        With `as_str=True` it sorts by text, where `'10'` comes before `'9'`, and None sorts as an
        empty cell:

        >>> mixed = Daf(cols=['id', 'v'], lol=[['a', 10], ['b', 9], ['c', 'x'], ['d', None]])
        >>> mixed.sort_by_colnames(['v'], as_str=True).col('v')
        [None, 10, 9, 'x']
    """
    if not self or len(self) <= 1:
        return self

    colidxs = [self.hd[colname] for colname in colnames]

    try:
        self.lol = daf_utils.sort_lol_by_cols(self.lol, colidxs, reverse=reverse, length_priority=length_priority, as_str=as_str)
    except TypeError as exc_info:
        raise TypeError(
            f"sort_by_colnames(): one of the columns {list(colnames)} holds values that cannot be compared, "
            f"such as numbers with text or None. Use as_str=True to sort by the text of the values, or convert "
            f"the columns with apply_dtypes().") from exc_info

    #self._rebuild_kd()
    self._invalidate_kd()    # use lazy kd rebuilding
    return self


def _transpose(self: 'Daf', new_keyfield:str='', new_cols:T_la | None = None, include_header:bool = False) -> 'Daf':
    """
    Turn rows into columns and columns into rows.

    The result has one row for each column of this Daf. With `include_header=True`,
    the first column of the result holds the column names of this Daf, and the
    names given in `new_cols` or the default names must then include that column.
    The default cols are `A`, `B` and so on, one for each row of this Daf. With
    `include_header=True` they start with `key`. If you pass `new_cols`, give one
    name for each row of this Daf, plus one with `include_header`. The data is
    copied.

    Args:
        new_keyfield: The keyfield of the result.
        new_cols: The cols of the result.
        include_header: If True, the column names become the first column.

    Every row must be as long as the columns. Check with `is_rectangular()`, and pad
    short rows with `force_rectangular()`.

    Returns:
        The new Daf.

    Raises:
        ValueError: A row is not as long as the columns.

    Examples:
        >>> daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'])
        >>> daf
        <BLANKLINE>
        | id | v  |
        | -: | -: |
        |  1 |  a |
        |  2 |  b |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.transpose(include_header=True)
        <BLANKLINE>
        | key | A  | B  |
        | --: | -: | -: |
        |  id |  1 |  2 |
        |   v |  a |  b |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.transpose().columns()
        ['A', 'B']
        >>> daf.transpose(include_header=True).columns()
        ['key', 'A', 'B']

        Name the columns of the result, and choose its keyfield. Here the first column holds the old
        column names, so they become the keys:

        >>> stock = Daf(cols=['id', 'item', 'qty'], lol=[['a1', 'pen', 3], ['b2', 'ink', 10]])
        >>> by_field = stock.transpose(include_header=True, new_cols=['field', 'r1', 'r2'], new_keyfield='field')
        >>> by_field
        <BLANKLINE>
        | field | r1  | r2  |
        | ----: | --: | --: |
        |    id |  a1 |  b2 |
        |  item | pen | ink |
        |   qty |   3 |  10 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='field'; name=''
        <BLANKLINE>
        >>> by_field.select_record('qty')
        {'field': 'qty', 'r1': 3, 'r2': 10}
    """

    if not self.is_rectangular():
        raise ValueError("transpose(): some rows are not as long as the columns. "
                         "Check with is_rectangular(), or pad short rows with force_rectangular().")

    if not new_cols:
        new_cols = daf_utils._generate_spreadsheet_column_names_list(num_cols=len(self.lol))
        if include_header:
            new_cols = ['key'] + new_cols

    # zip for a small table, numpy object references for a large one.
    new_lol = daf_utils.transpose_rows_lol(self.lol, len(self.lol[0]) if self.lol else 0)

    # instead of adding column for the header row, consider initializing kd from hd and hd from kd.

    # hd, kd = self._kd, self.hd
    # new_keyfield = ''

    if include_header:
        # add a new first column which will be the old column names row.
        # from utilities import daf_utils

        new_lol = daf_utils.insert_col_in_lol_at_icol(icol=0, col_la=self.columns(), lol=new_lol)

    # following invalidates kd for lazy rebuilding.
    return Daf(lol=new_lol, name=self.name, keyfield=new_keyfield, cols=new_cols, use_copy=True)


def _wide_to_narrow(self: 'Daf',
        id_cols: T_ls      ,                # identify the record but are not used to identify values
        varname_colname: str = 'variable',  # narrow format provides varname for each value col.
        value_colname: str = 'value',       # column of values from each value column
        ) -> 'Daf':
    """
    Turn columns into rows. This is called melt or unpivot.

    Each column that is not an id column gives one row for each row of this Daf.
    The row holds the id values, the name of the column, and its value.

    The default names, `variable` and `value`, are those that `narrow_to_wide()` reads, and
    those that pandas `melt()` uses. So the two methods chain with their defaults.

    Args:
        id_cols: The columns that identify a row. They are kept as they are.
        varname_colname: The name of the new column that holds the old column names.
        value_colname: The name of the new column that holds the values.

    Returns:
        The new Daf.

    Raises:
        TypeError: `id_cols` is not a list.

    Examples:
        >>> daf = Daf(lol=[['x', 1, 2], ['y', 3, 4]], cols=['id', 'a', 'b'])
        >>> daf
        <BLANKLINE>
        | id | a  | b  |
        | -: | -: | -: |
        |  x |  1 |  2 |
        |  y |  3 |  4 |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.wide_to_narrow(['id'])
        <BLANKLINE>
        | id | variable | value |
        | -: | -------: | ----: |
        |  x |        a |     1 |
        |  x |        b |     2 |
        |  y |        a |     3 |
        |  y |        b |     4 |
        <BLANKLINE>
        %% daf rows=4; cols=3; keyfield=''; name=''
        <BLANKLINE>

        A round trip:

        >>> daf.wide_to_narrow(['id']).narrow_to_wide(['id']).lol
        [['x', 1, 2], ['y', 3, 4]]

        Name the two new columns:

        >>> votes = Daf(cols=['ballot', 'mayor', 'council'], lol=[['b1', 1, 0], ['b2', 0, 1]])
        >>> votes.wide_to_narrow(['ballot'], varname_colname='contest', value_colname='vote')
        <BLANKLINE>
        | ballot | contest | vote |
        | -----: | ------: | ---: |
        |     b1 |   mayor |    1 |
        |     b1 | council |    0 |
        |     b2 |   mayor |    0 |
        |     b2 | council |    1 |
        <BLANKLINE>
        %% daf rows=4; cols=3; keyfield=''; name=''
        <BLANKLINE>
    """
    if not isinstance(id_cols, list):
        raise TypeError("id_cols must be a list")

    narrow_daf = Daf(cols= id_cols + [varname_colname, value_colname])

    value_cols = [colname for colname in self.columns() if colname not in id_cols]

    for row_da in self:
        # accept id values from original wide rows.
        new_row = {col: row_da[col] for col in id_cols}

        # add a new row for each column
        for col in value_cols:
            new_row[varname_colname] = col
            new_row[value_colname] = row_da[col]
            narrow_daf.append(new_row)

    return narrow_daf


def _narrow_to_wide(self: 'Daf',
        id_cols:        T_cs,
        varname_col:    str = 'variable',
        value_col:      str = 'value',
        wide_cols:      T_cs | None = None,
        ) -> 'Daf':
    """
    Turn rows into columns. This is called pivot or spread.

    The rows of one id may be anywhere in the table. The result has one row for
    each id, in the order in which the ids first appear. The columns are the id
    columns and then the names found in `varname_col`, in the order first seen. An
    id that has no value for a name gets NULL. If an id has the same name more
    than once, the last value is kept.

    With `wide_cols`, those are the columns after the ids, in that order. A
    name that is not listed is left out.

    The default names, `variable` and `value`, are those that `wide_to_narrow()` writes.

    An empty Daf gives an empty Daf.

    Args:
        id_cols: The columns that identify a row.
        varname_col: The column whose values become the new column names.
        value_col: The column whose values fill the new columns.
        wide_cols: The names to use as columns, in order. If None, all names found.

    Returns:
        The new Daf.

    Examples:
        >>> daf = Daf(lol=[['x', 'a', 1], ['y', 'a', 3], ['x', 'b', 2], ['y', 'b', 4]], cols=['id', 'variable', 'value'])
        >>> daf
        <BLANKLINE>
        | id | variable | value |
        | -: | -------: | ----: |
        |  x |        a |     1 |
        |  y |        a |     3 |
        |  x |        b |     2 |
        |  y |        b |     4 |
        <BLANKLINE>
        %% daf rows=4; cols=3; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.narrow_to_wide(['id'])
        <BLANKLINE>
        | id | a  | b  |
        | -: | -: | -: |
        |  x |  1 |  2 |
        |  y |  3 |  4 |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.narrow_to_wide(['id'], wide_cols=['b', 'a'])
        <BLANKLINE>
        | id | b  | a  |
        | -: | -: | -: |
        |  x |  2 |  1 |
        |  y |  4 |  3 |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield=''; name=''
        <BLANKLINE>

        Name the two columns to read, and keep only some of the new columns:

        >>> narrow = Daf(cols=['ballot', 'contest', 'vote'], lol=[
        ...     ['b1', 'mayor', 1], ['b1', 'council', 0], ['b2', 'mayor', 0], ['b2', 'council', 1]])
        >>> narrow.narrow_to_wide(['ballot'], varname_col='contest', value_col='vote')
        <BLANKLINE>
        | ballot | mayor | council |
        | -----: | ----: | ------: |
        |     b1 |     1 |       0 |
        |     b2 |     0 |       1 |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield=''; name=''
        <BLANKLINE>
        >>> narrow.narrow_to_wide(['ballot'], varname_col='contest', value_col='vote', wide_cols=['council'])
        <BLANKLINE>
        | ballot | council |
        | -----: | ------: |
        |     b1 |       0 |
        |     b2 |       1 |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield=''; name=''
        <BLANKLINE>
    """
    if not self.lol:
        return Daf()

    hd          = self.hd
    id_idxs     = [hd[col] for col in id_cols]
    var_idx     = hd[varname_col]
    val_idx     = hd[value_col]

    rows_dod: Dict[Tuple[Any, ...], Dict[Any, Any]] = {}
    names_d: Dict[Any, None] = {}

    for row_la in self.lol:
        id_tup = tuple([row_la[idx] for idx in id_idxs])
        wide_row_d = rows_dod.get(id_tup)
        if wide_row_d is None:
            wide_row_d = rows_dod[id_tup] = {}
        var_name = row_la[var_idx]
        wide_row_d[var_name] = row_la[val_idx]
        names_d[var_name] = None

    names_ls = list(wide_cols) if wide_cols is not None else list(names_d)

    wide_lol = [list(id_tup) + [wide_row_d.get(name, NULL) for name in names_ls]
                for id_tup, wide_row_d in rows_dod.items()]

    return Daf(lol=wide_lol, cols=list(id_cols) + names_ls)
