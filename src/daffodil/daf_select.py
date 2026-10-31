# daf_select.py
# copyright (c) 2026 Ray Lutz
"""
Methods of Daf for selecting and indexing.

Each function here takes the Daf as its first argument, or the class for a classmethod.
It is assigned in the Daf class body under its Daf name, without the underscore.
The docs show it under that name.

Some functions use names defined in daf.py, such as Daf, when they run.
daf.py sets those names here, as listed in _NAMES_FROM_DAF, once they exist.
An import at the top would be circular, since daf.py imports this module.
"""

from __future__ import annotations

from daffodil.lib.daf_types import T_da, T_la, T_li, T_loda, T_lola, T_lor, T_lota, T_ls, T_ma, T_ta
import daffodil.lib.daf_utils    as daf_utils
from daffodil.lib.daf_utils import NULL
from daffodil.keyedlist import KeyedList

from typing import Dict, Any, Tuple, cast, Callable, Literal, overload, TYPE_CHECKING
from collections.abc import Iterable, Sequence

if TYPE_CHECKING:       # for mypy. daf.py sets the names below when the program runs.
    from daffodil.daf import Daf, ColumnNotFoundError, KeysDisabledError, _IndirectRowView, logs

# The names of daf.py that the functions here use when they run.
# daf.py sets each one here once it exists. An import at the top would be circular.
_NAMES_FROM_DAF: Tuple[str, ...] = ('Daf', 'ColumnNotFoundError', 'KeysDisabledError', '_IndirectRowView', 'logs')


def _set_irows_icols(self: 'Daf',
        irows: slice | int | range | T_li | Iterable | None,
        icols: slice | int | range | T_li | None,
        value: Any) -> 'Daf':
    """
    Set values at row positions and column positions, in place.

    This is what `my_daf[rows, cols] = value` calls, after the selectors are turned
    into positions. `irows` and `icols` are an integer, a slice, a range or a list.
    If `icols` is None, the whole row is set. If `irows` is None, nothing is
    set. Use `slice(None)` for all rows.

    What is set depends on the value.

        a single value    fills every selected cell. A str or bytes is a single value.
        a list            fills a selection in order. For several whole rows, each
                          row becomes a copy of the list.
        a dict            sets a row. The cells of columns that the dict lacks become NULL.
        a Daf             is copied as a block, from its top left corner.

    Nothing is checked, and no error is raised for a size mismatch. A smaller source
    fills the top left of the selection and leaves the rest. A larger one fills the
    selection and the extra values are ignored. Check the sizes first if it matters.

    Args:
        irows: The row positions.
        icols: The column positions.
        value: The value, list, dict or Daf to set.

    Returns:
        This Daf, which has been changed.

    Examples:
        Create a small Daf of people, keyed by an int id:

        >>> daf = Daf(cols=['id', 'city', 'age'], lol=[[1, 'Denver', 34], [2, 'Austin', 52], [3, 'Boston', 27]])
        >>> daf
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        |  1 | Denver |  34 |
        |  2 | Austin |  52 |
        |  3 | Boston |  27 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield=''; name=''
        <BLANKLINE>

        Set the city and age of the first two rows to one value:

        >>> daf.set_irows_icols([0, 1], [1, 2], 'unknown')
        <BLANKLINE>
        | id |  city   |   age   |
        | -: | ------: | ------: |
        |  1 | unknown | unknown |
        |  2 | unknown | unknown |
        |  3 |  Boston |      27 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield=''; name=''
        <BLANKLINE>

        Set a whole row from a list, with `None` for every column:

        >>> daf.set_irows_icols(2, None, [4, 'Miami', 60]).iloc(2)
        {'id': 4, 'city': 'Miami', 'age': 60}
    """
    """ set rows and cols in given daf.

            irows, icols: can be either a slice, int, or list of integers. These
                            refer to row/col indices that are inherent in the lol structure.

            mutates existing daf
        """
    if icols is None:
        icols = []
    if irows is None:
        irows = []

    tot_num_cols = self.num_cols()
    tot_num_rows = self.num_rows()

    # irows accepts a bare Iterable per this method's own signature, but len_rowcol_spec
    # only actually measures slice/int/range/list (silently returns 0 for anything else,
    # e.g. a generator) -- in every real caller, irows/icols is one of those four by here.
    num_irows = daf_utils.len_rowcol_spec(cast('slice | int | range | T_li | None', irows), tot_num_rows)
    num_icols = daf_utils.len_rowcol_spec(icols, tot_num_cols)

    if num_irows == 1 and isinstance(irows, int):
        irows = [irows]
    if num_icols == 1 and isinstance(icols, int):
        icols = [icols]

    if isinstance(irows, slice):
        irows = daf_utils.slice_to_range(irows, len(self))
    if isinstance(icols, slice):
        icols = daf_utils.slice_to_range(icols, self.num_cols())

    # By this point every branch above has normalized irows/icols down to range or list[int]
    # only -- slice was just converted to range, None to [], and (unlike icols, which has no
    # equivalent conversion) a lone int irows/icols was already wrapped in a single-item list
    # a few lines up (num_irows/num_icols == 1 and isinstance(..., int)). The declared param
    # type is wider (Iterable, T_li, ...) to accept what callers may pass in, not what
    # remains once this normalization runs.
    irows = cast('range | T_li', irows)
    icols = cast('range | T_li', icols)

    # special case when cols not specified.
    if num_irows == 1 and num_icols == 0:

        irow = irows[0]

        if isinstance(value, list):
            self.lol[irow] = value
        elif isinstance(value, dict):
            self.assign_record_irow(irow, record=value)
        elif isinstance(value, type(self)):
            self.lol[irow] = list(value.lol[0])
        else:
            # set the same value in the row for all columns.
            self.lol[irow] = [value] * len(self.lol[irow])

    elif num_irows == 1 and num_icols == 1:

        irow = irows[0]
        icol = icols[0]

        if isinstance(value, dict):
            self.assign_record_irow(irow, record=value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 1:
            # frequently, we will have a list generated from a selection of a column, and it it has only one value
            # it needs to be entered in the array location, but not as a list.
            # place a list with only one item in a cell can't be done this way:

            # my_daf[0,0] = [4]   # this will insert 4 not [4]

            self.lol[irow][icol] = value[0]
        else:
            self.lol[irow][icol] = value

    elif num_irows > 1 and num_icols == 0:

        if isinstance(value, dict):
            for irow in irows:
                self.assign_record_irow(irow, record=value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            # a str or bytes is a single value, not a sequence of values.
            # each row gets its own copy, so the rows do not share one list.
            for irow in irows:
                self.lol[irow] = list(value)
        elif isinstance(value, type(self)):
            for source_row, irow in zip(value.lol, irows):
                self.lol[irow] = list(source_row)
        else:
            # set the same value in the row for all columns.
            for irow in irows:
                self.lol[irow] = [value] * len(self.lol[irow])

    elif num_irows > 0 and num_icols == 1:

        icol = icols[0]

        if irows is None:
            irows = range(len(self.lol))

        if isinstance(value, dict):
            # this is the same as cols=0 bc dict updates the corresponding cols.
            for irow in irows:
                self.assign_record_irow(irow, record=value)

        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for source_val, irow in zip(value, irows):
                self.lol[irow][icol] = source_val

        elif isinstance(value, type(self)):
            for source_row, irow in zip(value.lol, irows):
                self.lol[irow][icol] = source_row[0]

        else:
            # set the same value in the row for all selected columns.
            for irow in irows:
                self.lol[irow][icol] = value
    else:
        if irows is None:
            irows = range(len(self.lol))

        if isinstance(value, dict):
            # this is the same as cols=0 bc dict updates the corresponding cols.
            for irow in irows:
                self.assign_record_irow(irow, record=value)

        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            # the same list of values is applied to each selected row.
            for irow in irows:
                for source_val, icol in zip(value, icols):
                    self.lol[irow][icol] = source_val

        elif isinstance(value, type(self)):
            for source_row, irow in zip(value.lol, irows):
                for source_val, icol in zip(source_row, icols):
                    self.lol[irow][icol] = source_val

        else:
            # set the same value in the row for all selected columns.
            for irow in irows:
                for icol in icols:
                    self.lol[irow][icol] = value
    return self


def _krows_to_irows(self: 'Daf',
        krows:          slice | str | T_la | int | Tuple[Any, Any] | T_lota | Iterable | None,
        inverse:        bool = False,
        silent_error:   bool = False,
        ) -> slice | int | T_li | range:
    """
    Turn row keys into row positions.

    This is the step that lets `my_daf['r2']` work. Call it when you need the
    positions themselves. The Daf must have a keyfield, or a key index passed in
    as `kd`.

    The keys may be a key, a list of keys, or a tuple that gives an inclusive range
    of keys. A range is made from positions, so the keys must be in the same order
    as the rows.

    The first lookup builds the key index, by reading the keyfield column. In one test with
    200,000 rows that took 0.05 s, and a later lookup took about 2 microseconds. Adding or
    removing rows clears the index, and the next lookup builds it again.

    Args:
        krows: A key, a list of keys, or a tuple that gives a range of keys.
        inverse: If True, return the positions of the rows that are not selected.
        silent_error: If True, keys that are not found are ignored.

    Returns:
        The row positions.

    Raises:
        KeysDisabledError: There is no keyfield and no key index, or the keyfield is not a column.
        KeyError: A key is not found and `silent_error` is False.

    Examples:
        Create a small Daf of people, keyed by id. One age is not known:

        >>> daf = Daf(cols=['id', 'city', 'age'], lol=[['p1', 'Denver', 34], ['p2', 'Austin', ''], ['p3', 'Denver', 27]], keyfield='id')
        >>> daf
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        | p1 | Denver |  34 |
        | p2 | Austin |     |
        | p3 | Denver |  27 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>

        Find the position of one key, of a list of keys, and of a range of keys:

        >>> daf.krows_to_irows('p2')
        [1]
        >>> daf.krows_to_irows(['p3', 'p1'])
        [2, 0]
        >>> daf.krows_to_irows(('p1', 'p2'))
        slice(0, 2, 1)
        >>> daf.krows_to_irows(['p1'], inverse=True)
        [1, 2]
        >>> daf.krows_to_irows(['p9'], silent_error=True)
        []
        >>> daf.krows_to_irows(['p9'])
        Traceback (most recent call last):
            ...
        KeyError: 'p9'
    """
    """
        If the keyfield is set, then the rows can be selected by providing a
        krows parameter that will index the rows by using values in the keyfield column.
        The keyfield column is read as a list and then converted to a dictionary that
        provides the indexes of the row for each value in that column. Lookups using
        this method are very fast but there is overhead to reading the column and
        creating the dictionary. Therefore, set keyfield to '' to disable row key lookups.
        
        raises KeysDisabledError if keyfield is not set.
        """
    if not self.keyfield and not self._kd:
        # if keyfield is unset, kd may still be initialized manually.

        # `raise` was missing here -- constructed but never raised, so this docstring's own
        # "raises KeysDisabledError if keyfield is not set" silently didn't happen; every
        # other identical message elsewhere in this file (e.g. select_krows()) does raise.
        raise self._no_keys_error('krows_to_irows')
        # if inverse:
            # return range(len(self))
        # else:
            # return []

    self._rebuild_kd_if_invalidated()    # Only rebuilds when keyfield is set and `_kd` is empty.

    if not self._kd and self.keyfield and not self._is_keyfield_valid():
        raise KeysDisabledError(self._keyfield_not_a_column_message())

    return type(self).gkeys_to_idxs(
                keydict         = self._kd,
                gkeys           = krows,
                inverse         = inverse,
                silent_error    = silent_error,
                axis            = 'rowkeys',     # for error message only
                name            = self.name,
                )


def _kcols_to_icols(self: 'Daf',
        kcols: str | T_ls | slice | int | T_li | Tuple[Any, Any] | Iterable | None = None,
        inverse: bool = False,
        silent_error: bool=False,
        ) -> slice | int | T_li | range | None:
    """
    Turn column names into column positions.

    This is the step that lets `my_daf[:, 'v']` work. A name that looks like an
    integer is still read as a name.

    Args:
        kcols: A name, a list of names, or a tuple that gives an inclusive range of names.
        inverse: If True, return the positions of the columns that are not selected.
        silent_error: If True, names that are not found are ignored.

    Returns:
        The column positions. With no column names the result is empty, or all
        positions if `inverse` is True.

    Raises:
        KeyError: A name is not found and `silent_error` is False.

    Examples:
        Create a small Daf of people:

        >>> daf = Daf(cols=['id', 'city', 'age'], lol=[['p1', 'Denver', 34], ['p2', 'Austin', ''], ['p3', 'Denver', 27]], keyfield='id')
        >>> daf
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        | p1 | Denver |  34 |
        | p2 | Austin |     |
        | p3 | Denver |  27 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>

        Find the position of one column, of a list of columns, and of a range of columns:

        >>> daf.kcols_to_icols('city')
        [1]
        >>> daf.kcols_to_icols(['age', 'id'])
        [2, 0]
        >>> daf.kcols_to_icols(('id', 'city'))
        slice(0, 2, 1)
        >>> daf.kcols_to_icols('city', inverse=True)
        [0, 2]
        >>> daf.kcols_to_icols('email')
        Traceback (most recent call last):
            ...
        KeyError: 'email'

        A name that is not a column raises KeyError, or is left out with `silent_error=True`:

        >>> daf.kcols_to_icols(['city', 'email'], silent_error=True)
        [1]
    """
    """
        If cols are defined is set, then the cols can be selected by providing a
        kcols parameter that will index the cols by using values in the header dict hd.
        This forces an attempt to use the spec as a column name even if it may look
        like an integer.
        """
    if not self.hd:
        if inverse:
            return range(self.num_cols())
        else:
            return []

    return type(self).gkeys_to_idxs(
                # self.hd's keys (Dict[str, int]) are a subset of what gkeys_to_idxs accepts
                # (Dict[str|int, int]) -- dict is invariant in its key type for mypy, so this
                # narrower-is-fine relationship needs a cast to type-check.
                keydict         = cast('Dict[str | int, int]', self.hd),
                gkeys           = kcols,
                inverse         = inverse,
                silent_error    = silent_error,
                axis            = 'colnames',     # for error message only
                name            = self.name,
                )


def _gkeys_to_idxs(
        keydict:    Dict[str | int, int],
        gkeys:      str | T_ls | slice | int | T_li | Tuple[Any, Any] | T_lota | Iterable | None = None,
        inverse:    bool = False,
        silent_error: bool=False,
        axis:       str='rowkeys',              # used for status messages only.
        name:       str='unspecified',          # used for status messages only.
        ) -> slice | int | T_li | range:
    """
    Turn keys into positions, using a dict of key to position.

    This is the step under `krows_to_irows()` and `kcols_to_icols()`. It is a
    static method. It is for internal use.

    A key gives a list with one position. A list of keys gives their positions in
    that order. A tuple of two keys is an inclusive range, and `None` in it means
    the start or the end. A tuple of one key means that key to the end. A slice is
    taken as positions. With `inverse`, the positions that are not selected are
    returned.

    Args:
        keydict: Maps each key to its position.
        gkeys: The keys.
        inverse: If True, return the positions that are not selected.
        silent_error: If True, keys that are not found are ignored.
        axis: The name of the index, for the error raised when it is empty.
        name: Not used.

    Returns:
        The positions, as a list or a slice.

    Raises:
        KeysDisabledError: The dict is empty, as it is for a Daf with no rows.
        KeyError: A key is not found and `silent_error` is False.
        TypeError: The keys are None, or a tuple of a length other than 1 or 2.

    Examples:
        Look up names in a dict of positions, as the column names of a Daf map to their places:

        >>> Daf.gkeys_to_idxs({'id': 0, 'name': 1, 'age': 2}, ['age', 'id'])
        [2, 0]
        >>> Daf.gkeys_to_idxs({'id': 0, 'name': 1, 'age': 2}, ('id', 'name'))
        slice(0, 2, 1)

        With `inverse=True`, the positions that are not selected. With `silent_error=True`, a key that
        is not found is left out:

        >>> positions = {'id': 0, 'name': 1, 'age': 2}
        >>> Daf.gkeys_to_idxs(positions, ['name'], inverse=True)
        [0, 2]
        >>> Daf.gkeys_to_idxs(positions, ['name', 'email'], silent_error=True)
        [1]

        `axis` names the index in the error for an empty one:

        >>> Daf.gkeys_to_idxs({}, ['id'], axis='rowkeys')
        Traceback (most recent call last):
        ...
        daffodil.daf.KeysDisabledError: gkeys_to_idxs(): the rowkeys index is empty, so there is nothing to look up. A Daf with no rows has no keys.
    """
    """
        If keydict is defined, then the idxs can be selected by providing a
        gkeys parameter that will index the keydict, and return either a
        slice, int, T_li, or None, which will index the range.
        """

    # --- no key system ---
    if not keydict:
        raise KeysDisabledError(f"gkeys_to_idxs(): the {axis} index is empty, so there is nothing to look up. A Daf with no rows has no keys.")

    # --- invalid selector ---
    if gkeys is None:
        raise TypeError("None is not a valid key selector")

    # --- empty selection ---
    idxs: slice | T_li
    if isinstance(gkeys, (list,dict,tuple)) and not gkeys:
        idxs = []

    elif isinstance(gkeys, (str, int)):
        idxs = []
        gkey = gkeys

        # For the following, see https://github.com/raylutz/daffodil/issues/6
        try:
            idxs.append(keydict[gkey])
        except KeyError:
            if not silent_error:
                # logs.sts(f"{logs.prog_loc()} Cannot find key '{gkey}' in {axis} in dataframe '{name}'", 3)
                raise

    elif isinstance(gkeys, tuple):

        n = len(gkeys)

        if n == 0:
            raise TypeError("Empty tuple selector is invalid")

        # --- (start,) → start to end ---
        if n == 1:
            start_key = gkeys[0]

            if start_key is None:
                raise TypeError("Tuple (None,) is ambiguous")

            start_idx = keydict[start_key]
            stop_idx  = len(keydict)

        # --- (start, stop) ---
        elif n == 2:
            start_key, stop_key = gkeys

            # (None, stop) → beginning to stop
            if start_key is None:
                start_idx = 0
            else:
                start_idx = keydict[start_key]

            # (start, None) → start to end
            if stop_key is None:
                stop_idx = len(keydict)
            else:
                stop_idx = keydict[stop_key] + 1   # inclusive

        else:
            raise TypeError("Tuple selector must have length 1 or 2")

        idxs = slice(start_idx, stop_idx, 1)


    elif isinstance(gkeys, Iterable) and not isinstance(gkeys, (str, bytes, dict)):     # can be list of integer or strings (or anything hashable)
        idxs = []
        for one_gkey in gkeys:   # renamed from gkey -- distinct from the str|int `gkey` above
            # For the following, see https://github.com/raylutz/daffodil/issues/6
            try:
                # gkeys may iterate to a tuple or other non-str/int element (per its Union
                # type); keydict only has str|int keys, so such a lookup simply KeyErrors
                # below and is handled the same as any other missing key.
                idxs.append(keydict[cast('str | int', one_gkey)])
            except KeyError:
                if not silent_error:
                    # logs.sts(f"{logs.prog_loc()} Cannot find key '{one_gkey}' in {axis} in dataframe '{name}'", 3)
                    # breakpoint()
                    raise


    elif isinstance(gkeys, slice):
        # slice.start/.stop/.step are ordinal positions into keydict's current iteration
        # order (same convention as integer indices elsewhere) -- NOT keys to look up.
        # For arbitrary (e.g. string) keys there is no well-defined "next key" to support
        # exclusive-stop slicing the way integer positions do; use the (start_key, stop_key)
        # tuple form (inclusive of stop_key) for genuine key-range selection instead.
        #
        # Note: this branch is only reachable via the explicit select_krows()/select_kcols()
        # method calls. The [] / __getitem__/__setitem__ syntax (_parse_selectors) routes ANY
        # slice object straight to the plain-index path without ever calling krows_to_irows/
        # kcols_to_icols/gkeys_to_idxs at all -- regardless of whether the slice's bounds are
        # ints or strings. So `daf['a':'c']` does NOT do key-based slicing (it errors trying
        # to use 'a'/'c' as literal list-slice bounds); only daf.select_krows(slice(...)) /
        # daf.select_kcols(slice(...)) reach this code.
        n = len(keydict)
        start = gkeys.start if gkeys.start is not None else 0
        stop  = gkeys.stop  if gkeys.stop  is not None else n
        step  = gkeys.step  if gkeys.step  is not None else 1
        idxs = slice(start, stop, step)


    if inverse:
        n = len(keydict)
        if isinstance(idxs, slice):
            this_range = daf_utils.slice_to_range(idxs, n)
            idxs = [idx for idx in range(n) if idx not in this_range]
        else:
            drop_set: set = set(idxs)                   # a set, so each test is one lookup, not a scan of the list.
            idxs = [idx for idx in range(n) if idx not in drop_set]

    return idxs


def _select_krows(self: 'Daf',
        krows:          slice | str | T_la | int | T_lota | Tuple[Any, Any] | Iterable | None,
        inverse:        bool=False,
        silent_error:   bool=False,
        ) -> 'Daf':
    """
    Select rows by key. Rows with those keys are kept, or dropped if `inverse` is True.

    This is the same as `my_daf[keys]`, with a choice to drop rows and to ignore
    keys that are not found. It works with integer keys, which brackets cannot.
    The result is a new Daf with its own outer row list. Each row is the same list as
    in this Daf. Use `copy('editable')` before changing cells in one without the other.

    A bare tuple means an inclusive range of keys, so `(1, 2)` is the rows from key
    1 through key 2. For a composite keyfield, give a list of tuples.

    Args:
        krows: A key, a list of keys, or a tuple that gives a range of keys.
        inverse: If True, drop the selected rows and keep the others.
        silent_error: If True, keys that are not found are ignored.

    Returns:
        The new Daf.

    Raises:
        KeysDisabledError: The Daf has no keyfield and no key index.
        KeyError: A key is not found and `silent_error` is False.

    Examples:
        Create a small Daf of people, keyed by an int id. An int here is a key, not a position:

        >>> daf = Daf(cols=['id', 'city', 'age'], lol=[[1, 'Denver', 34], [2, 'Austin', 52], [3, 'Boston', 27]], keyfield='id')
        >>> daf
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        |  1 | Denver |  34 |
        |  2 | Austin |  52 |
        |  3 | Boston |  27 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>

        Select the rows with keys 3 and 1, in that order:

        >>> daf.select_krows([3, 1])
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        |  3 | Boston |  27 |
        |  1 | Denver |  34 |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield='id'; name=''
        <BLANKLINE>

        Select the rows not listed:

        >>> daf.select_krows([1, 2], inverse=True)
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        |  3 | Boston |  27 |
        <BLANKLINE>
        %% daf rows=1; cols=3; keyfield='id'; name=''
        <BLANKLINE>

        Leave out a key that is not found:

        >>> daf.select_krows([1, 9], silent_error=True)
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        |  1 | Denver |  34 |
        <BLANKLINE>
        %% daf rows=1; cols=3; keyfield='id'; name=''
        <BLANKLINE>
    """
    self._rebuild_kd_if_invalidated()

    if not self.keyfield and not self._kd:
        raise self._no_keys_error('select_krows')

    irows = self.krows_to_irows(
        krows = krows,
        inverse = inverse,
        silent_error = silent_error,
        )
    new_daf = self.select_irows(irows)

    return new_daf


def _select_kcols(self: 'Daf',
        kcols:          slice | str | T_la | int | Tuple[Any, Any] | None,
        inverse:        bool=False,
        flip:           bool=False,
        silent_error:   bool=False,
        ) -> 'Daf':
    """
    Select columns by name. Columns with those names are kept, or dropped if `inverse` is True.

    This is the same as `my_daf[:, names]`, with more choices. The result is
    a new Daf with new rows, in the order of the names you give. With `flip=True`
    the selected columns become rows, and the result has no column names and no
    keyfield. This costs no more than selecting the columns.

    Selecting columns copies data, so it is not cheap. For `apply` and `reduce`,
    use their `cols` argument instead.

    Args:
        kcols: A name, a list of names, or a tuple that gives a range of names.
        inverse: If True, drop the named columns and keep the others.
        flip: If True, turn the selected columns into rows.
        silent_error: If True, names that are not found are ignored.

    Returns:
        The new Daf.

    Raises:
        KeysDisabledError: The Daf has no column names.
        KeyError: A name is not found and `silent_error` is False.

    Examples:
        Create a small Daf of people:

        >>> daf = Daf(cols=['id', 'city', 'age'], lol=[[1, 'Denver', 34], [2, 'Austin', 52], [3, 'Boston', 27]], keyfield='id')
        >>> daf
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        |  1 | Denver |  34 |
        |  2 | Austin |  52 |
        |  3 | Boston |  27 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>

        Select two columns by name, in the order given:

        >>> daf.select_kcols(['age', 'id'])
        <BLANKLINE>
        | age | id |
        | --: | -: |
        |  34 |  1 |
        |  52 |  2 |
        |  27 |  3 |
        <BLANKLINE>
        %% daf rows=3; cols=2; keyfield='id'; name=''
        <BLANKLINE>

        Select every column but one:

        >>> daf.select_kcols('city', inverse=True).columns()
        ['id', 'age']

        Select a column and turn it into a row, with `flip=True`:

        >>> daf.select_kcols('city', flip=True)
        <BLANKLINE>
        |   A    |   B    |   C    |
        | -----: | -----: | -----: |
        | Denver | Austin | Boston |
        <BLANKLINE>
        %% daf rows=1; cols=3; keyfield=''; name=''
        <BLANKLINE>

        A name that is not a column raises KeyError, or is left out with `silent_error=True`:

        >>> daf.select_kcols(['id', 'email'], silent_error=True).columns()
        ['id']
    """
    if not self.hd:
        raise KeysDisabledError("select_kcols requires hd.")

    icols = self.kcols_to_icols(
        kcols = kcols,
        inverse = inverse,
        silent_error = silent_error,
        )
    return self.select_icols(icols, flip=flip)


def _select_irows(self: 'Daf', irows: slice | int | T_li | range | T_lor | Iterable | None, inverse: bool=False, invert: bool=False) -> 'Daf':
    """
    Select rows by position. Those rows are kept, or dropped if `inverse` is True.

    This is the same as `my_daf[rows]`, with a choice to drop rows. It is cheap.
    The result is a new Daf with its own outer row list. Each row is the same list as
    in this Daf, so changing a cell in the result
    changes it here too. This is also so when the selection is empty and `inverse`
    is True, which keeps all the rows. The keyfield, dtypes and column names carry over.

    Args:
        irows: A position, a slice, a range, a list of positions, or a list of ranges.
        inverse: If True, drop the selected rows and keep the others.
        invert: The old name of `inverse`, still accepted.

    Returns:
        The new Daf.

    Raises:
        IndexError: A single position is out of range.

    Examples:
        Create a small Daf of people:

        >>> daf = Daf(cols=['id', 'city', 'age'], lol=[[1, 'Denver', 34], [2, 'Austin', 52], [3, 'Boston', 27]], keyfield='id')
        >>> daf
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        |  1 | Denver |  34 |
        |  2 | Austin |  52 |
        |  3 | Boston |  27 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>

        Select the row at position 1, and every row but that one:

        >>> daf.select_irows(1)
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        |  2 | Austin |  52 |
        <BLANKLINE>
        %% daf rows=1; cols=3; keyfield='id'; name=''
        <BLANKLINE>
        >>> daf.select_irows(1, inverse=True)
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        |  1 | Denver |  34 |
        |  3 | Boston |  27 |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield='id'; name=''
        <BLANKLINE>
        >>> daf.select_irows(1, invert=True)
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        |  1 | Denver |  34 |
        |  3 | Boston |  27 |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield='id'; name=''
        <BLANKLINE>

        Select a slice of rows:

        >>> daf.select_irows(slice(1, None))
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        |  2 | Austin |  52 |
        |  3 | Boston |  27 |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield='id'; name=''
        <BLANKLINE>
    """
    """ select rows from daf and return a new instance.
            This is an efficient opeation. The array in the new instance
            uses references to selected rows in the original array.

            irows: can be either a slice, int, range, or list of integers. These
                    refer to row indices that are inherent in the lol structure.

            returns a new daf instance cloned from the original.
        """
    inverse = inverse or invert     # invert is the old name.

    row_sliced_lol = self.lol

    no_rows_specified = bool(not isinstance(irows, int) and not irows)

    if not self.lol or no_rows_specified and not inverse:
        return self.clone_empty()

    if no_rows_specified:
        if inverse:
            return self.clone_empty(lol=list(self.lol))     # drop nothing: a shallow new Daf, rows shared.
        else:
            return self.clone_empty()

    if isinstance(irows, int):
        if not inverse:
            # simple single row selection
            try:
                row_sliced_lol = [self.lol[irows]]
            except IndexError:
                raise
        else:
            # build a fresh list rather than mutating self.lol in place (row_sliced_lol was
            # an alias for self.lol, so .pop() here previously destructively removed the row
            # from the original Daf too). Normalize negative indices the way list.pop() does.
            actual_idx = irows if irows >= 0 else len(self.lol) + irows
            row_sliced_lol = [row for i, row in enumerate(self.lol) if i != actual_idx]

    elif irows and isinstance(irows, list):

        if daf_utils.is_list_of_type(irows, int):
            # is_list_of_type() isn't a TypeGuard -- confirmed at runtime that every element
            # is int, mypy just can't narrow irows: T_li | T_lor from that call itself.
            irows_li = cast(T_li, irows)
            if not inverse:
                # short-circuits on the first mismatch rather than materializing/comparing
                # two full-length lists, so a non-natural-order irows (the common case)
                # bails out almost immediately rather than doing O(num_rows) work regardless
                # -- matters for large arrays (e.g. 500K+ rows) where this check needs to
                # stay cheap even when it doesn't end up taking the fast path.
                if len(irows_li) == len(self.lol) and all(irow == i for i, irow in enumerate(irows_li)):
                    row_sliced_lol = self.lol
                else:
                    row_sliced_lol = [self.lol[i] for i in irows_li]
            else:
                irows_iter: Iterable[int]
                if len(irows_li) > 10:
                    irows_iter = dict.fromkeys(irows_li)
                else:
                    irows_iter = irows_li
                row_sliced_lol = [self.lol[i] for i in range(len(self.lol)) if i not in irows_iter]

        elif daf_utils.is_list_of_type(irows, range):
            rows_lor = cast(T_lor, irows)    # just a name change
            if not inverse:
                row_sliced_lol = [self.lol[i] for irange in rows_lor for i in irange]
            else:
                row_sliced_lol = [self.lol[i] for i in range(len(self.lol)) if not any(i in r for r in rows_lor)]

    elif irows and isinstance(irows, (range, Iterable)):
        irows_ii = cast(Iterable[int], irows)
        if not inverse:
            row_sliced_lol = [self.lol[i] for i in irows_ii]
        else:
            row_sliced_lol = [self.lol[i] for i in range(len(self.lol)) if i not in irows_ii]

    elif isinstance(irows, slice):
        slice_spec = irows
        if not inverse:
            row_sliced_lol = self.lol[slice_spec]
        else:
            slice_range = daf_utils.slice_to_range(slice_spec, len(self.lol))
            row_sliced_lol = [self.lol[i] for i in range(len(self.lol)) if i not in slice_range]

    if row_sliced_lol is self.lol:
        row_sliced_lol = list(self.lol)     # a shallow new Daf: its own row list, rows shared.

    new_daf = self.clone_empty(lol=row_sliced_lol)

    return new_daf


def _select_icols(self: 'Daf', icols: slice | int | T_li | range | T_lor | None, flip: bool=False) -> 'Daf':
    """
    Select columns by position. Those columns are kept, in the order given.

    This is the same as `my_daf[:, cols]`. It makes new rows, so it is not cheap.
    For `apply` and `reduce`, use their `cols` argument instead. A slice works as
    it does for a Python list. The keyfield and dtypes carry over if their columns
    are kept. With `flip=True` the columns become rows, and the result has no
    column names, no dtypes and no keyfield.

    With `flip=True` the columns are turned into rows as they are selected. That costs less than
    selecting them and then calling `transpose()`. In one test with 5 of 50 columns and 20,000
    rows it took 0.003 s, against 0.034 s.

    Args:
        icols: A position, a slice, a range, a list of positions, or a list of ranges.
        flip: If True, turn the selected columns into rows.

    Returns:
        The new Daf.

    Raises:
        IndexError: A position is beyond the end of some row.

    Examples:
        Create a small Daf of people:

        >>> daf = Daf(cols=['id', 'city', 'age'], lol=[[1, 'Denver', 34], [2, 'Austin', 52], [3, 'Boston', 27]], keyfield='id')
        >>> daf
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        |  1 | Denver |  34 |
        |  2 | Austin |  52 |
        |  3 | Boston |  27 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>

        Select columns by position, in the order given:

        >>> daf.select_icols([2, 0])
        <BLANKLINE>
        | age | id |
        | --: | -: |
        |  34 |  1 |
        |  52 |  2 |
        |  27 |  3 |
        <BLANKLINE>
        %% daf rows=3; cols=2; keyfield='id'; name=''
        <BLANKLINE>

        Select the last two columns:

        >>> daf.select_icols(slice(-2, None)).columns()
        ['city', 'age']

        Select two columns and turn them into rows, with `flip=True`:

        >>> daf.select_icols([0, 1], flip=True)
        <BLANKLINE>
        |   A    |   B    |   C    |
        | -----: | -----: | -----: |
        |      1 |      2 |      3 |
        | Denver | Austin | Boston |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield=''; name=''
        <BLANKLINE>
    """

    """ select cols from daf and return a new instance.
            This is not an efficient operation and can normally be avoided except when:
                reading/writing data, then columns may need to be dropped.
                exporting a portion of the array to NumPy, for example.

            instead, use the cols parameter to select the columns included
                in operations like apply() and reduce()

            icols: can be either a slice, int, range, list of integers, or list of ranges. These
                    refer to column indices that are inherent in the lol structure.

            flip: if True, then the columns selected are turned into rows.
                    This is a transposition operation. Otherwise, the columns selected
                    remain as columns in a new daf instance. There is no additional cost
                    to transpose if it is done when the columns are selected.

            returns a new daf instance cloned from the original,
                with cols, keyfield, and dtypes adjusted to be reasonable.
                If flip=True, then colnames=[], dtypes={} and keyfield='' (inactive)

        """

    orig_cols = list(self.hd.keys())    # may be an empty list if colnames not defined.
    orig_dtypes = self.dtypes or {}
    sliced_cols = []

    if isinstance(icols, int):
        # simple single col selection
        icol = icols
        if not flip:
            col_sliced_lol = [[row[icol]] for row in self.lol]
            if orig_cols:
                sliced_cols = [orig_cols[icol]]
        else: # flip
            col_sliced_lol = [row[icol] for row in self.lol]

    elif isinstance(icols, slice):
        slice_spec = icols
        icols_range = range(*slice_spec.indices(self.num_cols()))   # same rule as a Python list slice.

        if not icols_range:
            return Daf()

        if not flip:
            try:
                col_sliced_lol = [[row[icol] for icol in icols_range]
                                        for row in self.lol]
            except IndexError as exc_info:
                raise IndexError("select_icols(): column slice exceeds the length of some rows") from exc_info

            if orig_cols:
                sliced_cols = [orig_cols[icol] for icol in icols_range]
        else: # flip
            col_sliced_lol = [[row[icol] for row in self.lol]
                                    for icol in icols_range]

    elif (daf_utils.is_list_of_type(icols, int) or
            isinstance(icols, list) and not icols or
            isinstance(icols, range)):
        # list of integers or range -- is_list_of_type() isn't a TypeGuard, confirmed at
        # runtime, mypy just can't narrow icols from that call itself.
        icols = cast('T_li | range', icols)
        if not flip:
            try:
                # short-circuits on the first mismatch rather than materializing/comparing
                # two full-length lists, so a non-natural-order icols (the common case)
                # bails out almost immediately rather than doing O(num_cols) work regardless.
                if len(icols) == self.num_cols() and all(icol == i for i, icol in enumerate(icols)):
                    col_sliced_lol = self.lol
                elif len(icols) == 0:
                    col_sliced_lol = []
                else:
                    col_sliced_lol = [[row[icol] for icol in icols]
                                        for row in self.lol]
            except IndexError:
                logs.sts("Columns specified don't exist, array may have uneven rows.", 3)
                raise

            if orig_cols:
                sliced_cols = [orig_cols[icol] for icol in icols]

        else: # flip
            col_sliced_lol = [[row[icol] for row in self.lol]
                                    for icol in icols]

    # this part needs to be tested!
    elif isinstance(icols, list) and icols and daf_utils.is_list_of_type(icols, range):
        # list of ranges:
        cols_lor = cast(T_lor, icols) # name change only.
        if not flip:
            # Flatten ranges into individual column indices
            col_sliced_lol = [[row[icol] for irange in cols_lor for icol in irange]
                                    for row in self.lol]
            if orig_cols:
                # Also handle orig_cols, slicing them similarly
                sliced_cols = [orig_cols[icol] for irange in cols_lor for icol in irange]

        else:  # flip
            # Flip logic, slice columns and then rows, based on ranges
            col_sliced_lol = [[row[icol] for row in self.lol]
                                    for irange in cols_lor for icol in irange]
    else:
        if not flip:
            return self
        else: # flip
            col_sliced_lol = [[row[icol] for row in self.lol]
                                    for icol in range(self.num_cols())]
            sliced_cols = []    # perflint-reviewed (use-tuple-over-list)

    # fix up the dtypes and reset the keyfield if it is no longer in the daf.
    if sliced_cols:
        new_dtypes = {col: orig_dtypes[col] for col in sliced_cols if col in orig_dtypes}
        if self.keyfield and isinstance(self.keyfield, str):
            new_keyfield = self.keyfield if self.keyfield in sliced_cols else ''
        else:
            new_keyfield = ''

    else:
        new_dtypes = {}
        new_keyfield = ''

    new_daf = Daf(  cols=sliced_cols,
                    lol=col_sliced_lol,
                    keyfield=new_keyfield,
                    dtypes=new_dtypes,
                    )

    return new_daf


def _select_record(self: 'Daf', key: str | int | T_ta, silent_error: bool=True) -> T_da:
    """
    Get one row as a dict, by its key.

    If the key is not found, the answer is an empty dict, unless `silent_error` is
    False. Then a `KeyError` is raised. A typo in a key gives an empty dict, so
    check it, or pass `silent_error=False`. An empty Daf gives an empty dict.
    For a composite keyfield the key is a tuple.

    Args:
        key: The key of the row.
        silent_error: If False, raise an error when the key is not found.

    Returns:
        The row as a dict, or an empty dict.

    Raises:
        KeysDisabledError: The Daf has rows but no keyfield and no key index.
        KeyError: The key is not found and `silent_error` is False.

    Examples:
        Create a small Daf of people, keyed by an int id:

        >>> daf = Daf(cols=['id', 'city', 'age'], lol=[[1, 'Denver', 34], [2, 'Austin', 52], [3, 'Boston', 27]], keyfield='id')
        >>> daf
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        |  1 | Denver |  34 |
        |  2 | Austin |  52 |
        |  3 | Boston |  27 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>

        Read the row with key 2 as a dict. A key that is not there gives an empty dict:

        >>> daf.select_record(2)
        {'id': 2, 'city': 'Austin', 'age': 52}
        >>> daf.select_record(9)
        {}

        A key that is not found gives an empty dict, quietly. Pass `silent_error=False` to raise instead:

        >>> stock = Daf(cols=['id', 'qty'], lol=[['a1', 3], ['b2', 10]], keyfield='id')
        >>> stock.select_record('z9')
        {}
        >>> stock.select_record('z9', silent_error=False)
        Traceback (most recent call last):
        ...
        KeyError: 'z9'
    """
    """ Select one record from daf using the key and return as a single T_da dict.

            returns {} if not self
            assertion break if keyfield not defined.
            if key not found, return {} if silent_error, else raise KeyError exception.

            TODO Update for returning KeyedList
            should return daf unless .to_dict() is used?
        """

    if not self:
        return {}

    self._rebuild_kd_if_invalidated()

    if not self.keyfield and not self._kd:
        raise self._no_keys_error('select_record')

    if key in self._kd:
        return self._basic_get_record(self._kd[key])

    if silent_error:
        return {}

    raise KeyError(key)


def _basic_get_record(self: 'Daf', irow: int, include_cols: T_ls | None = None) -> T_da:
    """
    Retrieve a row as a dictionary.

    Args:
        irow: Row index.
        include_cols: Optional subset of columns.

    Returns:
        Dict: Row data.
    """

    """ return a record at irow as dict
            include only "include_cols" if it is defined
            note, this requires that hd is defined.

            TODO Update to return KeyedList as an option.
        """
    if not self.hd:
        raise KeysDisabledError("Getting a row as a dict needs column names. This Daf has none. Call set_cols() to name them.")

    if include_cols:
        return {col:self.lol[irow][self.hd[col]] for col in include_cols if col in self.hd}
    else:
        return dict(zip(self.hd, self.lol[irow]))


def _select_records_daf(self: 'Daf', keys_ls: 'T_ls | Iterable | Daf', inverse:bool=False, silent_error: bool=False) -> 'Daf':
    """
    Select several rows by key and return them as a Daf.

    This is `select_krows()` with a friendlier answer for an empty list of keys.
    No keys gives an empty Daf, or all the rows if `inverse` is True.

    The keys may be another Daf. Its keys are then used, through a view of its key index,
    so no list of them is made. That other Daf must have a keyfield. Its rows are not
    changed. The keys come in the order of its key index, so the rows selected come in
    that order too. A key that repeats in this Daf selects only its last row. See `keys()`.

    The rows are shared with this Daf, as in `select_krows()`. The new Daf has its own
    row list, also with no keys and `inverse` True, so adding a row to one does not
    add it to the other.

    Args:
        keys_ls: The keys of the rows, or a Daf whose keys are used.
        inverse: If True, drop the selected rows and keep the others.
        silent_error: If True, keys that are not found are ignored.

    Returns:
        The new Daf.

    Raises:
        KeysDisabledError: This Daf, or a Daf given as the keys, has no keyfield and no key index.
        KeyError: A key is not found and `silent_error` is False.

    Examples:
        Create a small Daf of people, keyed by id. One age is not known:

        >>> daf = Daf(cols=['id', 'city', 'age'], lol=[['p1', 'Denver', 34], ['p2', 'Austin', ''], ['p3', 'Denver', 27]], keyfield='id')
        >>> daf
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        | p1 | Denver |  34 |
        | p2 | Austin |     |
        | p3 | Denver |  27 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>

        Select the rows with these keys, in the order given. No keys selects no rows:

        >>> daf.select_records_daf(['p3', 'p1'])
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        | p3 | Denver |  27 |
        | p1 | Denver |  34 |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield='id'; name=''
        <BLANKLINE>
        >>> daf.select_records_daf([])
        <BLANKLINE>
        <BLANKLINE>
        %% daf rows=0; cols=0; keyfield='id'; name=''
        <BLANKLINE>
        >>> kept = daf.select_records_daf([], inverse=True)
        >>> kept == daf
        True
        >>> kept.append(['p4', 'Boston', 41])
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        | p1 | Denver |  34 |
        | p2 | Austin |     |
        | p3 | Denver |  27 |
        | p4 | Boston |  41 |
        <BLANKLINE>
        %% daf rows=4; cols=3; keyfield='id'; name=''
        <BLANKLINE>
        >>> daf.num_rows(), kept.num_rows()
        (3, 4)
        >>> kept[0, 'city'] = 'Miami'
        >>> daf.iloc(0)['city']
        'Miami'

        The keys of another Daf, without making a list of them:

        >>> other = Daf(cols=['id', 'visits'], lol=[['p3', 7], ['p9', 12]], keyfield='id')
        >>> other
        <BLANKLINE>
        | id | visits |
        | -: | -----: |
        | p3 |      7 |
        | p9 |     12 |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield='id'; name=''
        <BLANKLINE>
        >>> daf.select_records_daf(other, silent_error=True).to_lod()
        [{'id': 'p3', 'city': 'Denver', 'age': 27}]
    """

    """ Select multiple records from daf using the keys and return as a single daf.
            If inverse is true, select records that are not included in the keys.
            silent_error = True: do not raise an error if any keys are not found.

            This function requires that a keyfield or manual kd exists, otherwise raises KeysDisabledError.
        """
    if isinstance(keys_ls, Daf):
        keys_ls = keys_ls.keys(silent_error=False, astype='view')

    # rudamentary special cases:
    if not keys_ls:
        if inverse:
            return self.clone_empty(lol=list(self.lol))
        else:
            return self.clone_empty()

    return self.select_krows(krows=keys_ls, inverse=inverse, silent_error=silent_error)


def _irow_la(self: 'Daf', irow: int) -> T_la:
    """
    Get one row as a list, by position.

    The list is the row itself, not a copy. Changing it changes the Daf. Use
    `iloc()` with `rtype='list'` for a copy.

    Args:
        irow: The row position.

    Returns:
        The row.

    Raises:
        IndexError: The position is out of range.

    Examples:
        Create a small Daf of people:

        >>> daf = Daf(cols=['id', 'city', 'age'], lol=[['p1', 'Denver', 34], ['p2', 'Austin', ''], ['p3', 'Denver', 27]], keyfield='id')
        >>> daf
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        | p1 | Denver |  34 |
        | p2 | Austin |     |
        | p3 | Denver |  27 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>

        Read the row at position 1, as the Daf's own list:

        >>> daf.irow_la(1)
        ['p2', 'Austin', '']
        >>> row = daf.irow_la(0)
        >>> row[1] = 'Miami'
        >>> daf.iloc(0)
        {'id': 'p1', 'city': 'Miami', 'age': 34}
        >>> daf.irow_la(9)
        Traceback (most recent call last):
            ...
        IndexError: list index out of range
    """
    return self.lol[irow]


def _irow(self: 'Daf', irow: int=0, include_cols: T_ls | None = None) -> T_da:
    """
    Get one row as a dict, by position.

    This is `iloc()` with the default `rtype`. A negative position counts from the end. The dict
    is new, so changing it does not change the Daf.

    Args:
        irow: The row position.
        include_cols: Only these columns are included.

    Returns:
        The row as a dict. A negative position counts from the end. A Daf with no rows
        gives an empty dict.

    Raises:
        IndexError: The position is out of range, for a Daf that has rows.

    Examples:
        Create a small Daf of people:

        >>> daf = Daf(cols=['id', 'city', 'age'], lol=[[1, 'Denver', 34], [2, 'Austin', 52], [3, 'Boston', 27]], keyfield='id')
        >>> daf
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        |  1 | Denver |  34 |
        |  2 | Austin |  52 |
        |  3 | Boston |  27 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>

        Read the row at position 1 as a dict, with only some columns:

        >>> daf.irow(1, include_cols=['age'])
        {'age': 52}
    """
    return cast(T_da, self.iloc(irow, include_cols))


@overload
def _iloc(self: 'Daf', irow: int=0, include_cols: T_ls | None = None, rtype: Literal['dict']='dict') -> T_da: ...
@overload
def _iloc(self: 'Daf', irow: int=0, include_cols: T_ls | None = None, *, rtype: Literal['klist']) -> KeyedList: ...
@overload
def _iloc(self: 'Daf', irow: int=0, include_cols: T_ls | None = None, *, rtype: Literal['list']) -> T_la: ...
@overload
def _iloc(self: 'Daf', irow: int=0, include_cols: T_ls | None = None, rtype: str='dict') -> T_ma | T_la: ...

def _iloc(self: 'Daf', irow: int=0, include_cols: T_ls | None = None, rtype: str='dict') -> T_ma | T_la:
    """
    Get one row by position, as a dict, a KeyedList or a list.

    A negative position counts from the end, so `-1` is the last row. A position that
    is out of range raises `IndexError`. A Daf with no rows, and a row with no cells,
    give an empty dict, or an empty KeyedList. A dict or a KeyedList needs column
    names, so a Daf that has rows and no column names raises `KeysDisabledError`. Name
    them with `set_cols()`, or ask for `rtype='list'`.

    Args:
        irow: The row position.
        include_cols: Only these columns are included. This applies to a dict.
        rtype: `dict` for a new dict, `klist` for a KeyedList that shares the row, or `list` for a copy of the row.

    Returns:
        The row.

    Raises:
        IndexError: The position is out of range, for a Daf that has rows.
        KeysDisabledError: `rtype` is `dict` or `klist`, and the Daf has rows and no column names.
        ValueError: `rtype` is not one of the three names.

    Examples:
        Create a small Daf of people:

        >>> daf = Daf(cols=['id', 'city', 'age'], lol=[[1, 'Denver', 34], [2, 'Austin', 52], [3, 'Boston', 27]], keyfield='id')
        >>> daf
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        |  1 | Denver |  34 |
        |  2 | Austin |  52 |
        |  3 | Boston |  27 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>

        Read the row at position 1 as a dict, as a list, and the last row:

        >>> daf.iloc(1)
        {'id': 2, 'city': 'Austin', 'age': 52}
        >>> daf.iloc(1, rtype='list')
        [2, 'Austin', 52]
        >>> daf.iloc(-1)
        {'id': 3, 'city': 'Boston', 'age': 27}

        Only some columns, in a dict:

        >>> daf.iloc(1, include_cols=['id', 'age'])
        {'id': 2, 'age': 52}

        A KeyedList shares the row with the Daf, so a change to it changes the table:

        >>> row = daf.iloc(1, rtype='klist')
        >>> row['age'] = 35
        >>> daf.iloc(1)
        {'id': 2, 'city': 'Austin', 'age': 35}
    """
    """ Select one record from daf using the idx and return as a single T_da dict
            test exists in test_daf.py

            rtype can be 'dict', 'klist', or 'list'  <-- should be astype

        """
    if self.lol:
        if irow < 0:
            irow += len(self.lol)           # count from the end, as a list does.
        if irow < 0 or irow >= len(self.lol):
            raise IndexError(f"iloc: row position {irow} is out of range for {len(self.lol)} rows.")

    if not self.lol or not self.lol[irow]:
        if rtype == 'dict':
            return {}
        else:
            return KeyedList()

    if rtype in ('klist', 'dict') and not self.hd:
        raise KeysDisabledError(
            "Getting a row as a dict or a KeyedList needs column names. This Daf has none. "
            "Call set_cols() to name them, or ask for rtype='list'.")

    if rtype == 'klist':
        return KeyedList(self.hd, self.lol[irow])        # the row shares the hd of this Daf

    elif rtype == 'dict':
        return self._basic_get_record(irow, include_cols)

    elif rtype == 'list':
        return list(self.lol[irow])

    raise ValueError(f"iloc: unrecognized rtype '{rtype}'")


def _rows_matching_any(self: 'Daf', selectors_loda: T_loda, inverse: bool) -> T_lola:
    """
    Give the rows of this Daf that match at least one dict of a list of dicts, for `select_by_dict()`.

    The dicts are grouped by their keys, and the values of each group go in a set, so a row
    costs about one lookup however many dicts there are. A selector value that cannot be hashed is kept
    in a short list and compared with `==`. A cell that cannot be hashed makes the quick loop raise
    `TypeError`, and then a slower loop gives the same answer. The rows are shared, not copied. Internal use.

    Args:
        selectors_loda: The dicts. A row matches if it matches all the fields of any one of them.
        inverse: If True, give the rows that match none of them.

    Returns:
        The list of the rows.

    Raises:
        TypeError: An item of `selectors_loda` is not a dict.
    """
    hd = self.hd
    lol = self.lol
    groups: Dict[Tuple[Any, ...], Tuple[set, list]] = {}

    for selector_da in selectors_loda:
        if not isinstance(selector_da, dict):
            raise TypeError(f"select_by_dict(): each item of the list must be a dict, not {type(selector_da).__name__}.")
        if any(col not in hd for col in selector_da):
            continue                                            # an unknown column matches nothing.
        keys = tuple(selector_da)
        hashable_set, unhashable_list = groups.setdefault(keys, (set(), []))
        values = tuple(selector_da.values())
        try:
            hashable_set.add(values)
        except TypeError:
            unhashable_list.append(values)

    if not groups:
        return list(lol) if inverse else []                     # an empty list, or only unknown columns, matches nothing.
    if () in groups:
        return [] if inverse else list(lol)                     # an empty dict matches every row.

    plan = [(tuple([hd[col] for col in keys]), hashable_set, unhashable_list)
            for keys, (hashable_set, unhashable_list) in groups.items()]

    if not any(unhashable_list for _, _, unhashable_list in plan):
        try:
            if len(plan) == 1:
                idxs, hashable_set, _ = plan[0]
                if len(idxs) == 1:
                    icol = idxs[0]
                    value_set = {values[0] for values in hashable_set}
                    return [row_la for row_la in lol if (row_la[icol] in value_set) is not inverse]
                return [row_la for row_la in lol if (tuple([row_la[i] for i in idxs]) in hashable_set) is not inverse]
            return [row_la for row_la in lol
                    if any(tuple([row_la[i] for i in idxs]) in hashable_set for idxs, hashable_set, _ in plan) is not inverse]
        except TypeError:
            pass                                                # a cell cannot be hashed. Compare by == below.

    result_lol = []
    for row_la in lol:
        hit = False
        for idxs, hashable_set, unhashable_list in plan:
            probe = tuple([row_la[i] for i in idxs])
            try:
                hit = probe in hashable_set
            except TypeError:
                hit = False                                     # a cell that cannot be hashed equals no value that can.
            if not hit and unhashable_list:
                hit = any(probe == values for values in unhashable_list)
            if hit:
                break
        if hit is not inverse:
            result_lol.append(row_la)
    return result_lol


def _select_by_dict(
        self: 'Daf',
        selector_da:    T_da | T_loda,
        expectmax:      int=-1,
        inverse:        bool=False,
        keyfield:       str | int | T_ta='',
        ) -> 'Daf':
    """
    Select the rows that match every field of a dict, or any one of a list of dicts.

    With a dict, a row matches if each key is a column whose value in that row
    equals the value given. With a list of dicts, a row matches if it matches at
    least one of the dicts. So a dict is an AND of equalities, and a list is an OR of
    those. With `inverse=True` the rows that do not match are returned. The cells are
    compared by position, so no row is turned into a dict.

    To select the rows whose cell is any of several values, give one small dict for
    each value, as in `[{'name': n} for n in names]`. To select by several columns
    together, give a dict for each combination, as in `[{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]`.
    The dicts are grouped by their keys, and the values of each group are put in a set, so
    a row costs about one lookup however many dicts there are. This is much faster than
    `select_where()` with a function. The result is the same as comparing with `==`.
    A value or a cell that cannot be hashed, such as a list or a dict held in a cell,
    is compared with `==` as well. It never raises an error, and it takes a slower
    path for that call.

    A value in a dict is only ever compared for equality. A list or a set given as a value
    matches a cell that holds an equal list or set, and not a cell that is one of its items.
    To say "any of these", use a list of dicts.

    An empty list matches no row. An empty dict matches every row. A dict that names a
    column that is not there matches no row, and the other dicts still apply.

    A Daf with rows and no column names raises `KeysDisabledError`. The new Daf is a live
    view of this one. It has its own outer row list, and its rows are the same lists as the
    rows of this Daf. Changing a value in
    it changes this Daf. Adding a column with `insert_col()` or `insert_idx_col()` does not,
    because those copy shared rows first. Use `copy('editable')` for rows of your own.

    Args:
        selector_da: A dict of column names and the values they must have, or a list or tuple of such dicts.
        expectmax: If this is not -1 and more rows match, raise `LookupError`.
        inverse: If True, return the rows that do not match.
        keyfield: The keyfield of the new Daf. If empty, the keyfield of this Daf.

    Returns:
        The new Daf.

    Raises:
        LookupError: More than `expectmax` rows match.
        KeysDisabledError: The Daf has rows and no column names.
        TypeError: `selector_da` is not a dict, or a list or tuple of dicts.

    Examples:
        Create a small Daf of people:

        >>> daf = Daf(cols=['id', 'city', 'age'], lol=[[1, 'Denver', 34], [2, 'Austin', 52], [3, 'Boston', 27]], keyfield='id')
        >>> daf
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        |  1 | Denver |  34 |
        |  2 | Austin |  52 |
        |  3 | Boston |  27 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>

        Select the rows whose city is Austin, and the rows whose city is not:

        >>> daf.select_by_dict({'city': 'Austin'})
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        |  2 | Austin |  52 |
        <BLANKLINE>
        %% daf rows=1; cols=3; keyfield='id'; name=''
        <BLANKLINE>
        >>> daf.select_by_dict({'city': 'Austin'}, inverse=True)
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        |  1 | Denver |  34 |
        |  3 | Boston |  27 |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield='id'; name=''
        <BLANKLINE>

        Select the rows that match any of several dicts:

        >>> daf.select_by_dict([{'city': 'Denver'}, {'city': 'Boston'}])
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        |  1 | Denver |  34 |
        |  3 | Boston |  27 |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield='id'; name=''
        <BLANKLINE>

        A dict with several fields matches a row that has all of them:

        >>> daf.select_by_dict([{'city': 'Denver', 'age': 34}, {'city': 'Boston', 'age': 99}])
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        |  1 | Denver |  34 |
        <BLANKLINE>
        %% daf rows=1; cols=3; keyfield='id'; name=''
        <BLANKLINE>

        Select the rows whose city is in neither of two cities:

        >>> cities = ['Denver', 'Austin']
        >>> daf.select_by_dict([{'city': city} for city in cities], inverse=True)
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        |  3 | Boston |  27 |
        <BLANKLINE>
        %% daf rows=1; cols=3; keyfield='id'; name=''
        <BLANKLINE>

        Give the new Daf another keyfield:

        >>> customers = Daf(cols=['id', 'plan', 'city'], keyfield='id', lol=[
        ...     ['c1', 'basic', 'Denver'], ['c2', 'pro', 'Denver'], ['c3', 'basic', 'Austin']])
        >>> customers.select_by_dict({'plan': 'basic'}, keyfield='city').keys()
        ['Denver', 'Austin']

        Check how many rows match, with `expectmax`. More than that raises LookupError:

        >>> customers.select_by_dict({'plan': 'pro'}, expectmax=1).keys()
        ['c2']
        >>> customers.select_by_dict({'plan': 'basic'}, expectmax=1)
        Traceback (most recent call last):
        ...
        LookupError: select_by_dict(): 2 rows match, more than expectmax=1.
    """

    """ Selects rows in daf which match the fields specified in selector_da
            and return new daf, with keyfield set according to 'keyfield' argument.
        """
    # test exists in test_daf.py

    # the cells are compared by position, so no row is turned into a dict or a KeyedList.
    if self.lol and not self.hd:
        raise KeysDisabledError("select_by_dict(): this Daf has no column names. Call set_cols() to name them.")

    hd = self.hd
    if not isinstance(selector_da, dict):
        if not isinstance(selector_da, (list, tuple)):
            raise TypeError(f"select_by_dict(): selector_da must be a dict, or a list of dicts, not {type(selector_da).__name__}.")
        result_lol = self._rows_matching_any(selector_da, inverse)
    elif any(col not in hd for col in selector_da):
        result_lol = list(self.lol) if inverse else []          # an unknown column matches nothing.
    else:
        pairs = [(hd[col], val) for col, val in selector_da.items()]
        if not pairs:
            result_lol = [] if inverse else list(self.lol)      # an empty selector matches every row.
        elif len(pairs) == 1:
            icol, val = pairs[0]
            result_lol = [row_la for row_la in self.lol if (row_la[icol] == val) is not inverse]
        else:
            icol, val = pairs[0]
            rest = pairs[1:]
            result_lol = [row_la for row_la in self.lol
                          if (row_la[icol] == val and all(row_la[i] == v for i, v in rest)) is not inverse]

    if expectmax != -1 and len(result_lol) > expectmax:
        raise LookupError(f"select_by_dict(): {len(result_lol)} rows match, more than expectmax={expectmax}.")
        # breakpoint() #perm
        # pass

    new_keyfield = keyfield or self.keyfield

    # following invalidates kd for future lazy rebuild.
    daf = Daf(cols=self.columns(), lol=result_lol, keyfield=new_keyfield, dtypes=self.dtypes)

    return daf


def _select_first_row_by_dict(self: 'Daf', selector_da: T_da, inverse:bool=False) -> T_ma:
    """
    Get the first row that matches every field of a dict.

    The matching rule is that of `select_by_dict()`. With `inverse=True` it is the
    first row that does not match.

    Args:
        selector_da: The column names and the values they must have.
        inverse: If True, find the first row that does not match.

    Returns:
        The row, as a dict or a KeyedList according to `itermode`. An empty dict if none matches.

    Examples:
        Create a small Daf of people:

        >>> daf = Daf(cols=['id', 'city', 'age'], lol=[[1, 'Denver', 34], [2, 'Austin', 52], [3, 'Boston', 27]], keyfield='id')
        >>> daf
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        |  1 | Denver |  34 |
        |  2 | Austin |  52 |
        |  3 | Boston |  27 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>

        Find the first row whose city is Austin. No match gives an empty dict:

        >>> daf.select_first_row_by_dict({'city': 'Austin'})
        {'id': 2, 'city': 'Austin', 'age': 52}
        >>> daf.select_first_row_by_dict({'city': 'Miami'})
        {}

        With `inverse=True`, the first row that does not match:

        >>> customers = Daf(cols=['id', 'plan'], lol=[['c1', 'basic'], ['c2', 'pro'], ['c3', 'basic']])
        >>> customers.select_first_row_by_dict({'plan': 'basic'}, inverse=True)
        {'id': 'c2', 'plan': 'pro'}
    """

    """ Selects the first row in daf which matches the fields specified in selector_da
            and returns that row. Else returns {}.
            Use inverse to find the first row that does not match.
        """

    # test exists in test_daf.py

    for d2 in self:
        if inverse ^ daf_utils.is_d1_in_d2(d1=selector_da, d2=d2):
            return d2

    return {}


def _select_where(self: 'Daf', where: Callable, indirect_col: str | None = None) -> 'Daf':
    """
    Select the rows for which a function is true.

    The function gets each row, as a [KeyedList][daffodil.keyedlist.KeyedList].
    Read cells by column name, as in `row['n']`. Values are used as stored, so
    convert text first if the Daf was read from a CSV.

    With `indirect_col`, a name that is not a column is looked up in the dict
    held in that column.

    If the test only compares columns to values, use `select_by_dict()`. It
    compares the cells by position and does not call a function for each row. For
    200,000 rows by 50 columns it took 0.012 s, against 0.15 s for this method.

    The new Daf shares the selected rows with this one. The keyfield and dtypes
    carry over.

    To test a value against a list, a set or another table, write the test in the function.
    There is no need to build a list of bools first. Build a set
    of the values before the call, so that each lookup is fast and the set is built once. The
    function can use `and`, `or`, `not` and any other Python. It is called once for each row, so
    for a test on one column of a large Daf it is not the fastest way. For a test of equality,
    or of "any of these values", use `select_by_dict()` with a list of dicts, as in
    `select_by_dict([{'name': n} for n in names])`. It took 0.009 s where `select_where()` took 0.15 s,
    for 200,000 rows and 100 names. A comprehension over `col()`, followed by `select_irows()`, is
    also faster than a function.

    Args:
        where: A function that takes a row and returns True to keep it.
        indirect_col: A column that holds a dict, to read names that are not columns from.

    Returns:
        The new Daf.

    Examples:
        Create a small Daf of people:

        >>> daf = Daf(cols=['id', 'city', 'age'], lol=[[1, 'Denver', 34], [2, 'Austin', 52], [3, 'Boston', 27]], keyfield='id')
        >>> daf
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        |  1 | Denver |  34 |
        |  2 | Austin |  52 |
        |  3 | Boston |  27 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>

        Select the people over 30, with a function of the row:

        >>> daf.select_where(lambda row: row['age'] > 30)
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        |  1 | Denver |  34 |
        |  2 | Austin |  52 |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield='id'; name=''
        <BLANKLINE>

        Create a Daf of four people, to select by a set of cities:

        >>> daf = Daf(cols=['id', 'city', 'age'], lol=[[1, 'Denver', 34], [2, 'Austin', 52], [3, 'Boston', 27], [4, 'Denver', 41]])
        >>> daf
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        |  1 | Denver |  34 |
        |  2 | Austin |  52 |
        |  3 | Boston |  27 |
        |  4 | Denver |  41 |
        <BLANKLINE>
        %% daf rows=4; cols=3; keyfield=''; name=''
        <BLANKLINE>

        Select the people in these cities, those not in them, and combine conditions:

        >>> keep = {'Denver', 'Boston'}
        >>> daf.select_where(lambda row: row['city'] in keep)
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        |  1 | Denver |  34 |
        |  3 | Boston |  27 |
        |  4 | Denver |  41 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.select_where(lambda row: row['city'] not in keep)
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        |  2 | Austin |  52 |
        <BLANKLINE>
        %% daf rows=1; cols=3; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.select_where(lambda row: row['city'] in keep and row['age'] > 30)
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        |  1 | Denver |  34 |
        |  4 | Denver |  41 |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.select_where(lambda row: row['id'] % 2 == 0 or row['city'] == 'Boston')
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        |  2 | Austin |  52 |
        |  3 | Boston |  27 |
        |  4 | Denver |  41 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield=''; name=''
        <BLANKLINE>

        The values can come from another Daf. Make the set once, outside the function:

        >>> other = Daf(cols=['city'], lol=[['Denver'], ['Boston']], keyfield='city')
        >>> other
        <BLANKLINE>
        |  city  |
        | -----: |
        | Denver |
        | Boston |
        <BLANKLINE>
        %% daf rows=2; cols=1; keyfield='city'; name=''
        <BLANKLINE>
        >>> other_keys = set(other.keys())
        >>> daf.select_where(lambda row: row['city'] in other_keys)
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        |  1 | Denver |  34 |
        |  3 | Boston |  27 |
        |  4 | Denver |  41 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield=''; name=''
        <BLANKLINE>

        For a large Daf, the faster form picks the positions from the column:

        >>> daf.select_irows([irow for irow, v in enumerate(daf.col('city')) if v in keep])
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        |  1 | Denver |  34 |
        |  3 | Boston |  27 |
        |  4 | Denver |  41 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield=''; name=''
        <BLANKLINE>

        For sparse data, where each row holds a dict in one column, `indirect_col` names that column.
        Then the function can read the dict's keys as if they were columns. A row that lacks a key
        gets NULL:

        >>> customers = Daf(cols=['id', 'cart'], keyfield='id', lol=[
        ...     ['c1', {'apples': 1}], ['c2', {'apples': 1, 'pears': 2}], ['c3', {'pears': 1}]])
        >>> customers.select_where(lambda row: row['apples'] == 1, indirect_col='cart').keys()
        ['c1', 'c2']
    """
    """
        Select rows in Daf based on the provided where condition
        
        # Example Usage

            result_daf = original_daf.select_where(lambda row: bool(int(row['colname']) > 5))

        if indirect_col is set to a column that exists in the daf array, then any 
            column references will be first tried in the non-indirect columns and if not found,
            then the indirect column will be attempted.

        """
    # unit test exists.

    result_lol: T_lola = []

    # --- fast path: no indirect ---
    if not indirect_col:
        for row_kl in self.iter_klist():
            if where(row_kl):
                result_lol.append(row_kl.values())

    # --- indirect path ---
    else:
        for row_kl in self.iter_klist():
            row = _IndirectRowView(row_kl, indirect_col)
            if where(row):
                result_lol.append(row_kl.values())

    daf = Daf(cols=self.columns(), lol=result_lol, keyfield=self.keyfield, dtypes=self.dtypes)

    return daf


def _select_where_idxs(self: 'Daf', where: Callable) -> T_li:
    """
    Get the positions of the rows for which a function is true.

    The function gets each row as a [KeyedList][daffodil.keyedlist.KeyedList], whatever the
    `itermode` is, as in `select_where()`. Read cells by column name, as in `row['n']`.
    For a test that only compares columns to values, `select_by_dict()` is faster.

    The positions are a snapshot. Inserting, removing or sorting rows makes them point
    to other rows. Use them right away. To keep a reference to a row for longer, keep its key.

    Args:
        where: A function that takes a row and returns True to keep it.

    Returns:
        The row positions.

    Raises:
        KeysDisabledError: The Daf has rows and no column names.

    Examples:
        Create a small Daf of people:

        >>> daf = Daf(cols=['id', 'city', 'age'], lol=[[1, 'Denver', 34], [2, 'Austin', 52], [3, 'Boston', 27]], keyfield='id')
        >>> daf
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        |  1 | Denver |  34 |
        |  2 | Austin |  52 |
        |  3 | Boston |  27 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>

        Find the positions of the people over 30:

        >>> daf.select_where_idxs(lambda row: row['age'] > 30)
        [0, 1]
    """

    """
        Select rows in Daf based on the provided where condition
        return list of indexes.

        # Example Usage
            result_daf = original_daf.select_where(lambda row: int(row['colname']) > 5)

        Could use .to_index() approach instead?

        """
    # unit test exists.

    return [idx for idx, row_kl in enumerate(self.iter_klist()) if where(row_kl)]


def _split_where(
        self: 'Daf', 
        where: Callable,
        *,
        indirect_col: str | None = None,
        ) -> Tuple['Daf', 'Daf']:
    """
    Split the rows in two, by a function that is true or false for each row.

    This is deprecated. Use `group_where()` with a function that returns True or False.
    It gives the two groups under the keys True and False. A side with no rows has no
    group, so read each side with `.get()`.

    The function gets each row, as in `select_where()`. Both new Daf instances
    share their rows with this one, so changing a cell in one changes it here
    too. The keyfield and dtypes carry over to both.

    Args:
        where: A function that takes a row and returns True or False.
        indirect_col: A column that holds a dict, to read names that are not columns from.

    Returns:
        A tuple of the Daf of the rows where the function is true and the Daf of the others.

    Examples:
        Create a small Daf of people:

        >>> daf = Daf(cols=['id', 'city', 'age'], lol=[[1, 'Denver', 34], [2, 'Austin', 52], [3, 'Boston', 27]], keyfield='id')
        >>> daf
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        |  1 | Denver |  34 |
        |  2 | Austin |  52 |
        |  3 | Boston |  27 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>

        Split the people into those over 30 and the rest:

        >>> older, younger = daf.split_where(lambda row: row['age'] > 30)
        >>> older
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        |  1 | Denver |  34 |
        |  2 | Austin |  52 |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield='id'; name=''
        <BLANKLINE>
        >>> younger
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        |  3 | Boston |  27 |
        <BLANKLINE>
        %% daf rows=1; cols=3; keyfield='id'; name=''
        <BLANKLINE>

        With `indirect_col`, the function can read the keys of the dict in that column as columns:

        >>> customers = Daf(cols=['id', 'cart'], keyfield='id', lol=[
        ...     ['c1', {'apples': 1}], ['c2', {'apples': 1, 'pears': 2}], ['c3', {'pears': 1}]])
        >>> with_pears, without_pears = customers.split_where(lambda row: row['pears'] != '', indirect_col='cart')
        >>> with_pears.keys(), without_pears.keys()
        (['c2', 'c3'], ['c1'])
    """
    """
        Select rows in Daf based on the provided where condition,
        and split into two Daf objects, for True and False.

        Note: this does a shallow copy of the lines and any changes to
        the split objects will change the original daf object.

        # Example Usage

            true_daf, false_daf = original_daf.split_where(lambda row: bool(int(row['colname']) > 5))

        """
    true_lol = []
    false_lol = []

    if indirect_col:

        for klist in self.iter_klist():

            row = _IndirectRowView(klist, indirect_col)

            if where(row):
                true_lol.append(klist._values)
            else:
                false_lol.append(klist._values)

    else:    

        for klist in self.iter_klist():
            if where(klist):
                true_lol.append(klist._values)
            else:
                false_lol.append(klist._values)

    true_daf = Daf(cols=self.columns(), lol=true_lol, keyfield=self.keyfield, dtypes=self.dtypes)
    false_daf = Daf(cols=self.columns(), lol=false_lol, keyfield=self.keyfield, dtypes=self.dtypes)

    return true_daf, false_daf


def _col(self: 'Daf', 
        colname:        str,
        *,
        unique:         bool=False, 
        omit_nulls:     bool=False, 
        silent_error:   bool=False,
        astype:         Callable | str | type | None = None,
        indirect_col:   str | None = None,
        default:        Any | None='',
        ) -> list:
    """
    Get one column as a list, by name.

    This reads the column directly, without making a Daf first, as `daf[:, 'v'].to_list()` does.
    So it is the fast way to get the values of one column.

    The options leave values out or change them:

    - `unique=True` keeps the first of each value, in order.
    - `omit_nulls=True` leaves out NULL, the empty string.
    - `astype` converts each value with a type or a function.
    - `silent_error=True` gives an empty list for a column that does not exist, where it would raise.

    `indirect_col` is for sparse data, where each row holds a dict in one column, with only the
    keys it has. Then `colname` is a key of those dicts, and a row that lacks it gets `default`. A
    name that is a real column is still read from the column.

    Args:
        colname: The column name.
        unique: If True, leave out repeated values and keep the order.
        omit_nulls: If True, leave out the empty values.
        silent_error: If True, a column that is not found gives an empty list.
        astype: A type or function to convert each value with.
        indirect_col: A column that holds a dict, to read the name from.
        default: The value for a row that lacks the name. It is used only with `indirect_col`.

    Returns:
        The values of the column.

    Raises:
        ColumnNotFoundError: The column is not found and `silent_error` is False. This is a
            `KeyError`, and also a `RuntimeError`, so a handler for either catches it.
        RuntimeError: The name is empty.

    Examples:
        Create a small Daf of customers, with a plan and a cart of items for each. One plan is not known:

        >>> daf = Daf(cols=['id', 'plan', 'cart'], keyfield='id', lol=[
        ...     ['c1', 'basic', {'apples': 1}],
        ...     ['c2', 'pro', {'apples': 1, 'pears': 2}],
        ...     ['c3', '',  {}],
        ...     ['c4', 'basic', {'pears': 1}]])
        >>> daf.col('plan')
        ['basic', 'pro', '', 'basic']

        Each value once, and without the empty ones:

        >>> daf.col('plan', unique=True)
        ['basic', 'pro', '']
        >>> daf.col('plan', omit_nulls=True)
        ['basic', 'pro', 'basic']
        >>> daf.col('plan', unique=True, omit_nulls=True)
        ['basic', 'pro']

        Convert each value, here with a function:

        >>> daf.col('id', astype=str.upper)
        ['C1', 'C2', 'C3', 'C4']

        A column that does not exist raises, or gives an empty list:

        >>> daf.col('email')
        Traceback (most recent call last):
        ...
        daffodil.daf.ColumnNotFoundError: 'email'
        >>> daf.col('email', silent_error=True)
        []

        Read a key from the dict in the cart column. A row without it gets NULL, or `default`:

        >>> daf.col('apples', indirect_col='cart')
        [1, 1, '', '']
        >>> daf.col('pears', indirect_col='cart', default=0)
        [0, 2, 0, 1]
    """

    """ alias for col_to_la()
            can also use column ranges and then transpose()
            test exists in test_daf.py
            silent_error: if colname not found, return []

            Can use my_daf[:, colname].to_list(unique=unique, omit_nulls=omit_nulls)
        """
    return self.col_to_la(colname, 
            unique          = unique, 
            omit_nulls      = omit_nulls, 
            silent_error    = silent_error,
            astype          = astype,
            indirect_col    = indirect_col,
            default         = default,
            )


def _col_to_la(self: 'Daf', 
        colname:        str, 
        *, 
        unique:         bool=False, 
        omit_nulls:     bool=False, 
        silent_error:   bool=False,     # no error if column not found.
        astype:         Callable | str | type | None = None,
        indirect_col:   str | None = None,
        default:        Any | None='',
    ) -> list:
    """
    Get one column as a list, by name.

    This does the same as [`col()`][daffodil.daf_select._col], which shows each option.

    Args:
        colname: The column name.
        unique: If True, leave out repeated values and keep the order.
        omit_nulls: If True, leave out the empty values.
        silent_error: If True, a column that is not found gives an empty list.
        astype: A type or function to convert each value with.
        indirect_col: A column that holds a dict, to read the name from.
        default: The value for a row that lacks the name. It is used only with `indirect_col`.

    Returns:
        The values of the column.

    Raises:
        ColumnNotFoundError: The column is not found and `silent_error` is False. This is a
            `KeyError`, and also a `RuntimeError`, so a handler for either catches it.
        RuntimeError: The name is empty.

    Examples:
        `col_to_la()` is the same method as [col()][daffodil.daf_select._col], which has an example
    of each option:

        >>> daf = Daf(cols=['id', 'plan'], lol=[['c1', 'basic'], ['c2', 'pro'], ['c3', 'basic']])
        >>> daf.col_to_la('plan', unique=True) == daf.col('plan', unique=True)
        True
    """

    if not colname:
        raise RuntimeError("colname is required.")

    # normal column    
    if colname in self.hd:
        icol = self.hd[colname]
        result_la = self.icol_to_la(icol, unique=unique, omit_nulls=omit_nulls)
        result_la = daf_utils.astype_la(result_la, astype)

    elif indirect_col and indirect_col in self.hd:
        result_la = []

        for row_da in self:
            val = daf_utils.get_indirect_val(row_da, indirect_col, colname, default=default)

            if omit_nulls and val is NULL:
                continue

            # this added for numpy compatibility.
            if val is NULL and default is not NULL:
                val = default

            val = daf_utils.astype_value(val, astype)
            result_la.append(val)
        if unique:
            result_la = list(dict.fromkeys(result_la))

    else:
        if silent_error:
            return []
        raise ColumnNotFoundError(colname)

    return result_la


def _icol(self: 'Daf', icol: int) -> list:
    """
    Get one column as a list, by position.

    A negative position counts from the end, so `-1` is the last column. A position
    that is out of range raises `IndexError`. A Daf with no rows gives an empty list.

    Args:
        icol: The column position.

    Returns:
        The values of the column.

    Raises:
        IndexError: The position is out of range, for a Daf that has rows.

    Examples:
        Create a small Daf of people:

        >>> daf = Daf(cols=['id', 'city', 'age'], lol=[[1, 'Denver', 34], [2, 'Austin', 52], [3, 'Boston', 27]], keyfield='id')
        >>> daf
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        |  1 | Denver |  34 |
        |  2 | Austin |  52 |
        |  3 | Boston |  27 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>

        Read the column at position 1, and the last column:

        >>> daf.icol(1)
        ['Denver', 'Austin', 'Boston']
        >>> daf.icol(-1)
        [34, 52, 27]
    """

    return self.icol_to_la(icol)


def _icol_to_la(self: 'Daf', icol: int, unique: bool=False, omit_nulls: bool=False) -> list:
    """
    Get one column as a list, by position, with options.

    A negative position counts from the end, so `-1` is the last column. A position
    that is out of range raises `IndexError`. A Daf with no rows gives an empty list.

    Args:
        icol: The column position.
        unique: If True, leave out repeated values and keep the order.
        omit_nulls: If True, leave out the empty values.

    Returns:
        The values of the column.

    Raises:
        IndexError: The position is out of range, for a Daf that has rows.

    Examples:
        Create a small Daf of people. One age is not known:

        >>> daf = Daf(cols=['id', 'city', 'age'], lol=[['p1', 'Denver', 34], ['p2', 'Austin', ''], ['p3', 'Denver', 27]], keyfield='id')
        >>> daf
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        | p1 | Denver |  34 |
        | p2 | Austin |     |
        | p3 | Denver |  27 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>

        Read a column by position, the last one, and each value once:

        >>> daf.icol_to_la(1)
        ['Denver', 'Austin', 'Denver']
        >>> daf.icol_to_la(-1)
        [34, '', 27]
        >>> daf.icol_to_la(1, unique=True)
        ['Denver', 'Austin']
        >>> daf.icol_to_la(5)
        Traceback (most recent call last):
            ...
        IndexError: icol: column position 5 is out of range for 3 columns.

        Leave out the empty values, and the repeats:

        >>> plans = Daf(cols=['id', 'plan'], lol=[['c1', 'basic'], ['c2', 'pro'], ['c3', ''], ['c4', 'basic']])
        >>> plans.icol_to_la(1, omit_nulls=True)
        ['basic', 'pro', 'basic']
        >>> plans.icol_to_la(1, unique=True, omit_nulls=True)
        ['basic', 'pro']
    """

    if not self:
        return []

    num_cols = self.num_cols()
    if icol < 0:
        icol += num_cols                    # count from the end, as a list does.
    if icol < 0 or icol >= num_cols:
        raise IndexError(f"icol: column position {icol} is out of range for {num_cols} columns.")

    if omit_nulls:
        result_la = [la[icol] for la in self.lol if la[icol] is not NULL]
    else:
        result_la = [la[icol] for la in self.lol]

    if unique:
        result_la = list(dict.fromkeys(result_la))

    return result_la


def _select_cols(self: 'Daf',
        cols: T_ls | None = None,
        exclude_cols: T_ls | None = None,
        ) -> 'Daf':
    """
    Make a new Daf with only some columns, chosen by name.

    The columns come in the order of `cols`, as with `select_kcols()` and
    `my_daf[:, cols]`. A name given twice is used once. A name that is not a column
    raises `KeyError`. With no `cols`, all columns are kept, in the order of this
    Daf. Then `exclude_cols` leaves out the columns you name. A name in
    `exclude_cols` that is not a column is ignored.

    This copies data, so it is not cheap. For `apply` and `reduce`, use their
    `cols` argument instead. The keyfield carries over if its column is kept. A Daf
    with no column names gives rows with no columns.

    Args:
        cols: The names of the columns to keep, in the order you want. If empty, all columns.
        exclude_cols: The names of the columns to leave out.

    Returns:
        The new Daf.

    Raises:
        KeyError: A name in `cols` is not a column.

    Examples:
        Create a small Daf of people:

        >>> daf = Daf(cols=['id', 'city', 'age'], lol=[[1, 'Denver', 34], [2, 'Austin', 52], [3, 'Boston', 27]], keyfield='id')
        >>> daf
        <BLANKLINE>
        | id |  city  | age |
        | -: | -----: | --: |
        |  1 | Denver |  34 |
        |  2 | Austin |  52 |
        |  3 | Boston |  27 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name=''
        <BLANKLINE>

        Select some columns, in the order given, or every column but some:

        >>> daf.select_cols(['age', 'id']).columns()
        ['age', 'id']
        >>> daf.select_cols(exclude_cols=['city']).columns()
        ['id', 'age']
    """

    if not self.hd:
        return Daf(lol=[[] for _ in self.lol])

    if isinstance(cols, str):
        cols = [cols]
    if isinstance(exclude_cols, str):
        exclude_cols = [exclude_cols]
    exclude_set = set(exclude_cols) if exclude_cols else set()

    if cols:
        for col in cols:
            self.hd[col]                # a KeyError for a name that is not a column.
        new_cols = [col for col in dict.fromkeys(cols) if col not in exclude_set]
    else:
        new_cols = [col for col in self.hd if col not in exclude_set]

    idxs = [self.hd[col] for col in new_cols]

    # select from the array and create a new object.
    new_lol = [[la[idx] for idx in idxs] for la in self.lol]

    if self.dtypes:
        dtypes = {col: self.dtypes[col] for col in new_cols if col in self.dtypes}
    else:
        dtypes = None

    new_keyfield = self.keyfield \
        if self.keyfield and isinstance(self.keyfield, str) and self.keyfield in new_cols else ''

    return Daf(lol=new_lol, cols=new_cols, dtypes=dtypes, keyfield=new_keyfield)


def _keyfield_not_a_column_message(self: 'Daf') -> str:
    """
    Say why key lookups fail when the keyfield is set but is not made of columns of this Daf.

    Internal use. It is called only on the failure path.
    """
    if not self.hd:
        return (f"Key lookups are disabled: the keyfield {self.keyfield!r} is set, but this Daf has no column names. "
                f"Call set_cols() to name them.")
    return (f"Key lookups are disabled: the keyfield {self.keyfield!r} is not a column of this Daf. "
            f"The columns are {list(self.hd)}. Use set_keyfield() to choose one.")
