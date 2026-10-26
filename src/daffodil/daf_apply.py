# daf_apply.py
# copyright (c) 2026 Ray Lutz
"""
Methods of Daf for apply, reduce and group.

Each function here takes the Daf as its first argument, or the class for a classmethod.
It is assigned in the Daf class body under its Daf name, without the underscore.
The docs show it under that name.

Some functions use names defined in daf.py, such as Daf, when they run.
daf.py sets those names here, as listed in _NAMES_FROM_DAF, once they exist.
An import at the top would be circular, since daf.py imports this module.
"""

from __future__ import annotations

import copy
import functools
import itertools
import operator

from daffodil.lib.daf_types import T_cs, T_da, T_dodi, T_dola, T_la, T_li, T_lola, T_lota, T_ls, T_ma, T_rli, T_rma, T_ta, T_ts
import daffodil.lib.daf_utils    as daf_utils
from daffodil.lib.daf_utils import NULL
from daffodil.keyedlist import KeyedList

from typing import List, Dict, Any, Tuple, cast, Type, Callable, Literal, overload, TYPE_CHECKING
from collections.abc import Iterable, Iterator

if TYPE_CHECKING:       # for mypy. daf.py sets the names below when the program runs.
    from daffodil.daf import Daf, T_dodaf, T_klodaf, _IndirectRowView, _uncountable_message, logs

# The names of daf.py that the functions here use when they run.
# daf.py sets each one here once it exists. An import at the top would be circular.
_NAMES_FROM_DAF: Tuple[str, ...] = ('Daf', 'T_dodaf', '_IndirectRowView', '_uncountable_message', 'logs')


def _safe_tofloat(val: Any) -> float:
    """
    Safely convert value to float.

    Args:
        val: Value to convert.

    Returns:
        float | str: Converted float or original value on failure.

    Note:
        Internal use.    
    """
    try:
        return float(val)
    except ValueError:
        return 0.0


def _basic_append(self: 'Daf', row: KeyedList | Dict[Any, Any] | list) -> 'Daf':
    # --- list ---
    if isinstance(row, list):
        if self.hd:
            assert len(row) == len(self.hd)
        self.lol.append(row)
        return self

    # --- KeyedList ---
    if isinstance(row, KeyedList):
        if not self.hd:
            self.hd = cast(Dict[str, int], dict(row.hd))
        self.lol.append(row._values)
        return self

    # --- dict ---
    if isinstance(row, dict):
        if not self.hd:
            keys = list(row.keys())
            self.hd = dict(zip(keys, range(len(keys))))

            self.lol.append(list(row.values()))
            return self

        # hd exists → align order
        self.lol.append([row.get(col, '') for col in self.hd])
        return self

    raise TypeError("Unsupported row type for basic_append")


def _apply_formulas(self: 'Daf', formulas_daf: 'Daf') -> 'Daf':
    """
    Fill cells from spreadsheet like formulas, in place.

    `formulas_daf` is a Daf of the same shape. Each cell holds a Python
    expression as text. An empty cell is skipped. The result of each expression is
    stored in the same cell of this Daf. The formulas are evaluated again and again
    until no cell changes, so a cell may use the result of another. If the cells
    still change after 100 passes, it raises `RuntimeError`. This is a limit on the
    passes, not a check for circular formulas. A circle of formulas that settles, such
    as two cells that copy each other, raises nothing.

    In a formula, `$d` is this Daf, `$r` is the row number of the cell and `$c` is
    its column number. References are absolute unless you build them from `$r` and
    `$c`. So `sum($d[$r, :$c])` is the sum of the cells to the left in the same row.

    Other examples:

        $d[14,20]+$d[15,25]       the sum of two cells
        max(0,$d[($r-1),$c])      the cell above, but not below 0
        $d[($r-1),$c] * 0.15      15 percent of the cell above

    Warning: the formulas are run with `eval()`. Never use formulas from a source
    you do not trust.

    An error in a formula prints the cell and the formula, and is raised again. The
    `retmode` of this Daf is restored, and the key index is rebuilt when it is next
    needed. The cells that were already changed stay changed.

    Args:
        formulas_daf: The formulas, with the same shape as this Daf.

    Returns:
        This Daf, which has been changed.

    Raises:
        RuntimeError: The shapes differ, or the formulas never settle.

    Examples:
        >>> daf = Daf(cols=['A', 'B', 'C'], lol=[[1, 2, 0], [4, 5, 0], [7, 8, 0], [0, 0, 0]])
        >>> daf
        <BLANKLINE>
        | A  | B  | C  |
        | -: | -: | -: |
        |  1 |  2 |  0 |
        |  4 |  5 |  0 |
        |  7 |  8 |  0 |
        |  0 |  0 |  0 |
        <BLANKLINE>
        %% daf rows=4; cols=3; keyfield=''; name=''
        <BLANKLINE>
        >>> f_daf = Daf(cols=['A', 'B', 'C'], lol=[
        ...     ['', '', 'sum($d[$r,:$c])'],
        ...     ['', '', 'sum($d[$r,:$c])'],
        ...     ['', '', 'sum($d[$r,:$c])'],
        ...     ['sum($d[:$r,$c])', 'sum($d[:$r,$c])', 'sum($d[:$r,$c])']])
        >>> f_daf
        <BLANKLINE>
        |        A        |        B        |        C        |
        | --------------: | --------------: | --------------: |
        |                 |                 | sum($d[$r,:$c]) |
        |                 |                 | sum($d[$r,:$c]) |
        |                 |                 | sum($d[$r,:$c]) |
        | sum($d[:$r,$c]) | sum($d[:$r,$c]) | sum($d[:$r,$c]) |
        <BLANKLINE>
        %% daf rows=4; cols=3; keyfield=''; name=''
        <BLANKLINE>
        >>> _ = daf.apply_formulas(f_daf)
        >>> daf
        <BLANKLINE>
        | A  | B  | C  |
        | -: | -: | -: |
        |  1 |  2 |  3 |
        |  4 |  5 |  9 |
        |  7 |  8 | 15 |
        | 12 | 15 | 27 |
        <BLANKLINE>
        %% daf rows=4; cols=3; keyfield=''; name=''
        <BLANKLINE>
    """

    # TODO: This algorithm is not optimal. Ideally, a dependency tree would be formed and
    # cells modified in from those with no dependencies to those that depend on others.
    # This issue will not become a concern unless the number of formulas is substantial.

    if not self:
        return self

    if self.shape() != formulas_daf.shape():
        raise RuntimeError("apply_formulas requires data arrays of the same shape.")

    lol_changed = True     # must evaluate at least once.
    loop_limit = 100
    loop_count = 0

    # the following deals with $d, $r, $c in the formulas
    parsed_formulas_daf = formulas_daf._parse_formulas()

    # we must use RETMODE_VAL for formulas to work easily for users.
    prior_retmode = self.retmode
    self.retmode = self.RETMODE_VAL

    try:
        while lol_changed:
            lol_changed = False
            loop_count += 1
            if loop_count > loop_limit:
                raise RuntimeError("apply_formulas is resulting in excessive evaluation loops.")      # perflint-reviewed (loop-invariant-statement)

            for irow in range(len(self.lol)):
                for icol in range(self.num_cols()):
                    cell_formula = parsed_formulas_daf.lol[irow][icol]          # perflint-reviewed (loop-invariant-statement)
                    if not cell_formula:
                        # no formula provided -- do nothing
                        continue
                    try:
                        new_value = eval(cell_formula)
                    except Exception as exc_info:
                        print(f"Error in formula for cell [{irow},{icol}]: '{cell_formula}': '{exc_info}'")
                        raise

                    if new_value != self.lol[irow][icol]:
                        # update the value in the array, and set lol_changed flag
                        self.lol[irow][icol] = new_value
                        lol_changed = True
                    else:
                        continue
    finally:
        self.retmode = prior_retmode
        self._invalidate_kd()       # cells may have changed, even if there was an error.

    #self._rebuild_kd()

    return self


def _parse_formulas(self: 'Daf') -> 'Daf':

    # start with unparsed formulas
    parsed_formulas = copy.deepcopy(self)

    for irow in range(len(self.lol)):
        for icol in range(self.num_cols()):
            proposed_formula = self.lol[irow][icol]
            if not proposed_formula:
                # no formula provided.
                continue

            proposed_formula = proposed_formula.replace('$d', 'self')
            proposed_formula = proposed_formula.replace('$c', str(icol))
            proposed_formula = proposed_formula.replace('$r', str(irow))    # perflint-reviewed (loop-invariant-statement)

            parsed_formulas[irow,icol] = proposed_formula

    return parsed_formulas


def _cols_to_dol(self: 'Daf', colname1: str, colname2: str) -> T_dola:
    """
    Make a lookup from the values of one column to the values of another.

    For each value in `colname1`, the result lists the different values that appear
    with it in `colname2`, in the order first seen. Use it to see how two columns
    relate. The values must be hashable. If a name is not a column, or the Daf is
    empty, the result is empty.

    Args:
        colname1: The column of keys.
        colname2: The column of values.

    Returns:
        A dict that maps each value of the first column to a list of values of the second.

    Examples:
        >>> daf = Daf(lol=[['a', 'b'], ['b', 'd'], ['a', 'f'], ['b', 'd']], cols=['c1', 'c2'])
        >>> daf
        <BLANKLINE>
        | c1 | c2 |
        | -: | -: |
        |  a |  b |
        |  b |  d |
        |  a |  f |
        |  b |  d |
        <BLANKLINE>
        %% daf rows=4; cols=2; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.cols_to_dol('c1', 'c2')
        {'a': ['b', 'f'], 'b': ['d']}
    """

    if colname1 not in self.hd or colname2 not in self.hd or not self.lol:
        return {}

    colidx1 = self.hd[colname1]
    colidx2 = self.hd[colname2]


    # first work with dict of dict for speed.
    result_dadn: Dict[Any, Dict[Any, None]] = {}

    for la in self.lol:
        val1 = la[colidx1]
        val2 = la[colidx2]
        if val1 not in result_dadn:
            result_dadn[val1] = {val2: None}
        elif val2 not in result_dadn[val1]:
            result_dadn[val1][val2] = None
        # otherwise, it is already in the result.

    # now convert dadn to dola

    result_dola = {k: list(d.keys()) for k, d in result_dadn.items()}

    return result_dola


def _insert_dif_row(self: 'Daf',
        irow1: int,
        irow2: int | None = None,
        irow_insert: int | None = None,
        cols: T_ls | None = None,
        ) -> 'Daf':

    """
    Insert a row that holds the difference of two rows.

    The difference is the first row minus the second row, for the numeric columns
    you name. Other columns of the new row are empty. A cell that is empty counts
    as 0. By default the second row is the one after the first, and the new row goes
    between them. Columns that hold text must not be in `cols`.

    Args:
        irow1: The position of the first row.
        irow2: The position of the second row. If None, the row after the first.
        irow_insert: Where to insert the new row. If None, at `irow2`.
        cols: The columns to subtract. If None, all columns.

    Returns:
        This Daf, which has been changed.

    Examples:
        >>> daf = Daf(lol=[[1, 10], [3, 14], [6, 20]], cols=['a', 'n'])
        >>> daf
        <BLANKLINE>
        | a  | n  |
        | -: | -: |
        |  1 | 10 |
        |  3 | 14 |
        |  6 | 20 |
        <BLANKLINE>
        %% daf rows=3; cols=2; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.insert_dif_row(0)
        <BLANKLINE>
        | a  | n  |
        | -: | -: |
        |  1 | 10 |
        | -2 | -4 |
        |  3 | 14 |
        |  6 | 20 |
        <BLANKLINE>
        %% daf rows=4; cols=2; keyfield=''; name=''
        <BLANKLINE>
    """

    if irow2 is None:
        irow2 = irow1 + 1
    if irow_insert is None:
        irow_insert = irow2
    if cols is None:
        cols = self.columns()

    # calculate the difference.
    diff_result_da = type(self).diff_da(self[irow1].to_dict(), self[irow2].to_dict(), keys=cols)

    # this function handles normalizing the columns before insertion
    self.insert_irow(irow=irow_insert,  row=diff_result_da)

    return self


def _insert_dif_rows(self: 'Daf',
        irows_rli: T_rli | None = None,
        cols: T_ls | None = None,
        offset: int=0
        ) -> 'Daf':

    """
    Insert a difference row after each of several rows.

    For each position, a row is inserted that holds that row minus the next row,
    as in `insert_dif_row()`. Do not list the last row. By default all rows are
    used. With `offset=1` the new row goes after the next row, not between them.

    Args:
        irows_rli: The positions of the first rows. If None, every row but the last.
        cols: The columns to subtract. If None, all columns.
        offset: 0 inserts between the two rows. 1 inserts after the second.

    Returns:
        This Daf, which has been changed.

    Examples:
        >>> daf = Daf(lol=[[1, 10], [3, 14], [6, 20]], cols=['a', 'n'])
        >>> daf
        <BLANKLINE>
        | a  | n  |
        | -: | -: |
        |  1 | 10 |
        |  3 | 14 |
        |  6 | 20 |
        <BLANKLINE>
        %% daf rows=3; cols=2; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.insert_dif_rows()
        <BLANKLINE>
        | a  | n  |
        | -: | -: |
        |  1 | 10 |
        | -2 | -4 |
        |  3 | 14 |
        | -3 | -6 |
        |  6 | 20 |
        <BLANKLINE>
        %% daf rows=5; cols=2; keyfield=''; name=''
        <BLANKLINE>
    """

    reversed_irows_rli: range | T_li
    if irows_rli is None:
        reversed_irows_rli = range(len(self) - 2, -1, -1)
    else:
        reversed_irows_rli = sorted(irows_rli, reverse=True)

    for irow in reversed_irows_rli:

        self.insert_dif_row(irow1=irow, irow_insert=irow + 1 + offset, cols=cols)

    return self


def _apply(
        self: 'Daf',
        # called as func(self, **kwargs) for by='table', func(row, **kwargs) for by='row'.
        # by='table' is a pure passthrough of func's return value (any type -- see
        # test_daf_misc.py, where it returns a bare int), so func's return type isn't
        # pinned down here.
        func:       Callable[..., Any],
        by:         str='row',
        keylist:    T_la | T_lota | None = None,     # list of keys of rows to include (DEPRECATE?)
        **kwargs:   Any,
            # kwargs may commonly include:
            # cols: T_la | None = None,                    # columns included in the apply operation.
        ) -> "Daf":
    """
    Apply a function to each row and collect the results in a new Daf.

    Use it to convert one table into another, with a plain Python function that
    turns one row into the new row.
    This Daf is not changed. Only the new Daf gets rows.

    The function gets a row and returns what to add to the new Daf:

    - A dict adds one row.
    - A list of dicts adds a row for each dict, so one row can become several.
    - None, an empty dict or an empty list adds nothing, so the row is left out.

    The new Daf takes its columns from the first row returned. A later dict is
    placed by column name, and a column it lacks is NULL. A key that the first row
    did not have is dropped without an error. So when the columns are known, have
    the function return all of them in every row. The new Daf has no keyfield.

    With `by='table'` the function gets the whole Daf, and its result is returned
    as it is. Use that to run any function on the table. `by='col'` is not
    supported.

    The function gets the rows as dicts, or as KeyedList objects if `itermode` is
    `keyedlist`. The extra keyword arguments are passed on to it. To work on only
    some rows, give the `keylist`, or select them first.

    Args:
        func: The function to apply. It takes a row, or the Daf, and the keyword arguments.
        by: `row` to apply to each row, or `table` to apply to the whole Daf.
        keylist: Keys of the rows to include. All rows if None. This may be removed.
        **kwargs: Keyword arguments passed on to the function.

    Returns:
        The new Daf, or the result of the function for `by='table'`.

    Raises:
        NotImplementedError: `by` is `col`, or is not recognized.

    Examples:
        >>> daf = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
        >>> daf
        <BLANKLINE>
        | g  | x  | y  |
        | -: | -: | -: |
        |  a |  1 | 10 |
        |  b |  2 | 20 |
        |  a |  3 | 30 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.apply(lambda row: {'g': row['g'].upper(), 'z': row['y'] * 2})
        <BLANKLINE>
        | g  | z  |
        | -: | -: |
        |  A | 20 |
        |  B | 40 |
        |  A | 60 |
        <BLANKLINE>
        %% daf rows=3; cols=2; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.apply(lambda row: row if row['y'] > 15 else None)
        <BLANKLINE>
        | g  | x  | y  |
        | -: | -: | -: |
        |  b |  2 | 20 |
        |  a |  3 | 30 |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.apply(lambda row: [{'g': row['g'], 'part': 1}, {'g': row['g'], 'part': 2}] if row['x'] == 2 else None)
        <BLANKLINE>
        | g  | part |
        | -: | ---: |
        |  b |    1 |
        |  b |    2 |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield=''; name=''
        <BLANKLINE>
    """
    if by == 'table':
        # by contract (see docstring) func returns a 'Daf' when by='table', though this
        # is a pure passthrough and isn't actually enforced (a test exercises it with a
        # func returning a bare int).
        return func(self, **kwargs)  # type: ignore[no-any-return]

    result_daf = Daf()

    if by == 'row':
        if keylist is None:
            keylist = []                        # perflint-reviewed

        keylist_or_dict = keylist if not keylist or len(keylist) < 30 else dict.fromkeys(keylist)

        self._rebuild_kd_if_invalidated()

        for row in self:
            # self.keyfield here always names a single field (composite/multi-col keyfields
            # aren't used for this row-membership check).
            if self.keyfield and keylist_or_dict and row[cast(str, self.keyfield)] not in keylist_or_dict:
                continue
            transformed_row = func(row, **kwargs)
            result_daf.append(transformed_row)          # Will not append an empty row.

    elif by == 'col':
        # this is not working yet, don't know how to handle cols, for example.
        raise NotImplementedError

        num_cols = self.num_cols()
        for icol in range(num_cols):
            col_la = self.icol(icol)
            transformed_col = func(col_la, **kwargs)
            result_daf.insert_icol(icol, transformed_col)
    else:
        raise NotImplementedError

    # Rebuild the internal data structure (if needed)
    # result_daf._rebuild_kd()
    result_daf._invalidate_kd()   # use lazy rebuilding of kd.

    return result_daf


def _update_row(row: T_rma, da: T_da) -> T_rma:
    """
    Update a row with the items of a dict, and return the row.

    This is a small helper to use inside `apply()`, as in
    `daf.apply(lambda row: Daf.update_row(row, {'z': 0}))`. It is a static method.

    Args:
        row: The row, as a dict. It is changed.
        da: The items to put in the row.

    Returns:
        The same row.

    Examples:
        >>> Daf.update_row({'a': 1}, {'b': 2})
        {'a': 1, 'b': 2}
    """

    row.update(da)
    return row


def _apply_in_place(
        self: 'Daf',
        # called as func(row, **kwargs) -- see the apply() comment on Callable[...] above.
        func:       Callable[..., T_ma | None],
        by:         str='row',
        rowkeys:    T_la | T_lota | None = None,  # list of rowkeys to include.
                    # the above changed from keylist to avoid confusion with KeyedList
        **kwargs:   Any,
        ) -> 'Daf':
    """
    Apply a function to each row and store the results in this Daf.

    With `by='row'` the function gets each row as a dict. It must return a dict.
    Its values are written back by column name. A column that the dict lacks
    keeps its value, and a key that is not a column is ignored. The order of
    the keys does not matter, and the row keeps its length.

    With `by='row_klist'` the function gets each row as a
    [KeyedList][daffodil.keyedlist.KeyedList]. It changes the row and returns
    nothing, so no dict is built for each row. The cost of `row` grows with the
    number of columns, because each row is turned into a dict of all its columns.
    The cost of `row_klist` does not. In a test that changed one column, the two
    took the same time at about 6 columns. At 1,000 columns and 2,000 rows,
    `row_klist` took 0.005 s and `row` took 0.245 s. With 3 columns and 200,000
    rows, `row` was faster, 0.25 s against 0.30 s. Add or delete no keys in the
    KeyedList, because that changes the length of the row, not the columns.

    The key index is rebuilt when it is next needed. Use `apply()` to get a new Daf.

    Args:
        func: The function to apply to each row. It takes a row and the keyword arguments.
        by: `row` or `row_klist`.
        rowkeys: Keys of the rows to include. All rows if None. The Daf needs a keyfield.
        **kwargs: Keyword arguments passed on to the function.

    Returns:
        This Daf, which has been changed.

    Raises:
        ValueError: With `by='row'` the function returned None.
        NotImplementedError: `by` is not `row` or `row_klist`.

    Examples:
        >>> daf = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
        >>> daf
        <BLANKLINE>
        | g  | x  | y  |
        | -: | -: | -: |
        |  a |  1 | 10 |
        |  b |  2 | 20 |
        |  a |  3 | 30 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield=''; name=''
        <BLANKLINE>
        >>> _ = daf.apply_in_place(lambda row: {**row, 'y': row['y'] + 1})
        >>> daf.col('y')
        [11, 21, 31]
        >>> _ = daf.apply_in_place(lambda row: row.__setitem__('y', 0), by='row_klist')
        >>> daf.col('y')
        [0, 0, 0]
    """
    if rowkeys is None:
        rowkeys = []
    # here we create a dict if the rowkeys to search are numerous.
    rowkeys_list_or_dict = rowkeys if (not rowkeys or
                                    isinstance(rowkeys, dict)
                                    or len(rowkeys) < 30
                                ) else dict.fromkeys(rowkeys)

    if by == 'row':

        self._rebuild_kd_if_invalidated()

        hd = self.hd
        for row_la, row_da in zip(self.lol, self):
            if rowkeys_list_or_dict and self.keyfield and row_da[cast(str, self.keyfield)] not in rowkeys_list_or_dict:
                continue
            transformed_row_da = func(row_da, **kwargs)
            if transformed_row_da is None:
                raise ValueError("apply_in_place: func must return a row for by='row' (None is only valid for by='row_klist')")
            # write the returned values back by column name. The row keeps its length and its
            # column order. A name that is not a column is ignored. A column that is not
            # returned keeps its value.
            for colname, val in transformed_row_da.items():
                icol = hd.get(colname)  # type: ignore[call-overload]  # a row key from a KeyedList is Hashable
                if icol is not None:
                    row_la[icol] = val

    elif by == 'row_klist':

        for idx, row_klist in enumerate(self.iter_klist()):
            if rowkeys_list_or_dict and self.keyfield and row_klist[cast(str, self.keyfield)] not in rowkeys_list_or_dict:
                continue

            # func should return nothing and instead mutate row_klist, which will mutate the row in the array.
            func(row_klist, **kwargs)

    else:
        raise NotImplementedError

    # Rebuild the internal data structure (if needed)
    # note, this should not be necessary. apply_in_place should not modify the keyfield column.
    self._invalidate_kd()   # use lazy rebuilding of kd.

    # self._rebuild_kd()

    return self


def _manifest_apply(
        self: 'Daf',
        func: Callable[[T_da, T_la | None], Tuple[T_da, 'Daf']],    # function to apply according to 'by' parameter
        load_func: Callable[[T_ma], 'Daf'],            # optional function to load data for each manifest entry, defaults to local file system
        save_func: Callable[[T_da, 'Daf'], str],       # optional function to save data for each manifest entry, defaults to local file system
        by: str='row',                                  # determines how the func is applied.
        cols: T_la | None = None,                      # columns included in the apply operation.
         **kwargs: Any,
        ) -> "Daf":
    """
    Run a function on each chunk that a manifest lists, and save the results.

    A manifest is a Daf in which each row describes one chunk of data. For each
    row, `load_func` loads the chunk as a Daf. `func` is applied to it with
    `by='table'`, so it gets the loaded Daf and the keyword `cols`. It returns a
    tuple of a dict that describes the result chunk and the new Daf. `save_func`
    saves the new Daf. The result manifest has one row for each dict.

    Args:
        func: Gets a loaded Daf. Returns a dict that describes the result and the new Daf.
        load_func: Loads the chunk that a manifest row describes.
        save_func: Saves a new Daf, given the dict that describes it.
        by: Must be `table`. The default is `row`, which applies `func` to each row, so always pass `by='table'`.
        cols: Passed to `func` as the keyword `cols`.
        **kwargs: Keyword arguments passed on to `func`.

    Returns:
        The manifest of the result chunks.

    Examples:
        >>> chunks = {'a': Daf(cols=['n'], lol=[[1], [2]]), 'b': Daf(cols=['n'], lol=[[10]])}
        >>> saved = []
        >>> def load(spec):
        ...     return chunks[spec['chunk']]
        >>> def double(daf, cols=None):
        ...     new = Daf(cols=['n'], lol=[[row['n'] * 2] for row in daf])
        ...     return {'rows': len(new)}, new
        >>> def save(spec, daf):
        ...     saved.append(daf.to_lod())
        ...     return 'saved'
        >>> manifest = Daf(cols=['chunk'], lol=[['a'], ['b']])
        >>> manifest
        <BLANKLINE>
        | chunk |
        | ----: |
        |     a |
        |     b |
        <BLANKLINE>
        %% daf rows=2; cols=1; keyfield=''; name=''
        <BLANKLINE>
        >>> result = manifest.manifest_apply(double, load, save, by='table')
        >>> result
        <BLANKLINE>
        | rows |
        | ---: |
        |    2 |
        |    1 |
        <BLANKLINE>
        %% daf rows=2; cols=1; keyfield=''; name=''
        <BLANKLINE>
        >>> saved
        [[{'n': 2}, {'n': 4}], [{'n': 20}]]
    """

    result_manifest_daf = Daf()

    for chunk_spec in self:
        # Apply the function for all chunks specified.
        # Load the specified Daf table
        loaded_daf = load_func(chunk_spec)

        # Apply the function to the loaded Daf. apply()'s by='table' mode is a pure
        # passthrough of func's return value, so at by='table' this really returns
        # func's declared Tuple[T_da, 'Daf'] rather than apply()'s nominal 'Daf'.
        # No caller/test currently exercises manifest_apply -- untested.
        result_chunk_spec, transformed_daf = cast(Tuple[T_da, 'Daf'], loaded_daf.apply(func, by=by, cols=cols, **kwargs))

        # Save the resulting Daf table
        save_func(result_chunk_spec, transformed_daf)

        # Update the manifest with information about the resulting chunk
        result_manifest_daf.append(result_chunk_spec)

    return result_manifest_daf


def _manifest_reduce(
        self: 'Daf',
        # func is passed through to Daf.reduce() (row/reduction/cols/**kwargs), not called
        # directly here, so its exact arity isn't pinned down at this level.
        func: Callable[..., Any],
        load_func: Callable[[T_ma], 'Daf'] | None = None,
        by: str='row',                                  # determines how the func is applied.
        cols: T_la | None = None,                      # columns included in the reduce operation.
        **kwargs: Any,
        ) -> T_da:
    """
    Reduce the chunks that a manifest lists into one row.

    Each chunk is loaded with `load_func` and reduced with `reduce()`. The
    reductions are put in a Daf, and that is reduced again with the same function.
    This works for functions such as `sum_da()` that can be combined in this way.

    Args:
        func: The reduction function. See `reduce()`.
        load_func: Loads the chunk that a manifest row describes. It is required.
        by: How the function is applied. See `reduce()`.
        cols: The columns to reduce. All columns if None.
        **kwargs: Keyword arguments passed on to the function.

    Returns:
        The reduced row, as a dict.

    Raises:
        ValueError: No `load_func` is given.

    Examples:
        >>> store = {'c1': Daf(lol=[[1, 2], [3, 4]], cols=['a', 'b']), 'c2': Daf(lol=[[10, 20]], cols=['a', 'b'])}
        >>> manifest = Daf(lol=[['c1'], ['c2']], cols=['chunk'])
        >>> manifest
        <BLANKLINE>
        | chunk |
        | ----: |
        |    c1 |
        |    c2 |
        <BLANKLINE>
        %% daf rows=2; cols=1; keyfield=''; name=''
        <BLANKLINE>
        >>> manifest.manifest_reduce(Daf.sum_da, load_func=lambda spec: store[spec['chunk']])
        {'a': 14, 'b': 26}
    """
    if load_func is None:
        raise ValueError("manifest_reduce: load_func is required")

    # collects one reduction per chunk; its columns come from the chunks, not the manifest.
    first_reduction_daf = type(self)()

    for chunk_spec in self:
        # Load the specified Daf table
        loaded_daf = load_func(chunk_spec)

        # Apply the function to the loaded Daf
        reduction_ma = loaded_daf.reduce(func, by=by, cols=cols, **kwargs)

        first_reduction_daf.append(reduction_ma)

    final_reduction_ma = first_reduction_daf.reduce(func, by=by, cols=cols, **kwargs)

    return cast(T_da, final_reduction_ma)


def _manifest_process(
        self: 'Daf',
        # called as func(chunk_spec, **kwargs) -- see the apply() comment on Callable[...] above.
        func: Callable[..., T_da],   # function to run for each hunk specified by the manifest
        **kwargs: Any,
        ) -> 'Daf':                                    # records describing metadata of each hunk
    """
    Call a function once for each chunk that a manifest lists.

    The function gets the manifest row as a dict, and does its own loading and
    saving. It returns a dict of information about what it did. These dicts are
    collected as the rows of the result.

    Args:
        func: Gets a manifest row and returns a dict of results.
        **kwargs: Keyword arguments passed on to the function.

    Returns:
        A Daf with one row for each returned dict.

    Examples:
        >>> manifest = Daf(lol=[['c1'], ['c2']], cols=['chunk'])
        >>> manifest
        <BLANKLINE>
        | chunk |
        | ----: |
        |    c1 |
        |    c2 |
        <BLANKLINE>
        %% daf rows=2; cols=1; keyfield=''; name=''
        <BLANKLINE>
        >>> manifest.manifest_process(lambda spec: {'chunk': spec['chunk'], 'seen': True})
        <BLANKLINE>
        | chunk | seen |
        | ----: | ---: |
        |    c1 | True |
        |    c2 | True |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield=''; name=''
        <BLANKLINE>
    """

    result_daf = Daf()

    for chunk_spec in self:
        # Apply the function for all chunks specified.
        # Load the specified Daf table
        # Apply the function to the loaded Daf
        result_da = func(chunk_spec, **kwargs)

        # Update the manifest with information about the resulting chunk
        result_daf.append(result_da)

    return result_daf


def _cols_scope(self: 'Daf', cols: T_cs | None) -> Tuple[List[str], List[int], bool]:
    """
    Work out which columns a grouping keeps: their names, their positions, and whether that is all of them.

    With no `cols`, all columns are kept. A name that is not a column raises `KeyError`. Internal use.
    """
    if not cols:
        return list(self.hd), list(range(len(self.hd))), True

    names = list(cols)
    idxs = [self.hd[name] for name in names]

    return names, idxs, names == list(self.hd)


def _reduce_scope(self: 'Daf', by: str, reduce_cols: T_cs | None, kwargs: Dict[str, Any]) -> T_ls | None:
    """
    Work out the columns that the groups need for a reduction, or None for all columns.

    Only a reduction by row, or by sparse row, looks at just the columns in `reduce_cols`.
    For a sparse row the `indirect_col` is kept as well. A name that is not a column is
    left out, as `reduce()` does. If none are columns, all columns are kept. Internal use.
    """
    if not reduce_cols or by not in ('row', 'sparse_row'):
        return None

    names = [col for col in reduce_cols if col in self.hd]

    indirect_col = kwargs.get('indirect_col')
    if names and indirect_col and indirect_col in self.hd and indirect_col not in names:
        names.append(indirect_col)

    return names or None


def _new_group_dafs(self: 'Daf', groups: Dict[Any, T_lola], names: T_ls, all_cols: bool) -> T_klodaf:
    """
    Make the Daf for each group, with the layout of this Daf, or of the columns kept.

    Each group is a copy of a template, with its own rows. With all columns, the
    template is this Daf. Otherwise it is made once with `clone_empty()`, which cuts
    the dtypes to the columns kept, and clears the keyfield if any key column is not
    kept. The copy is what `clone_empty()` makes, without working out the layout again
    for each group. Internal use.
    """
    template_daf = self if all_cols else self.clone_empty(cols=names)
    copy_bits    = self.COPY_ATTRS | self.COPY_HD | self.COPY_DTYPES | self.COPY_KD    # as clone_empty().

    dodaf: Dict[Any, 'Daf'] = {}
    for groupkey, group_lol in groups.items():
        group_daf = template_daf.copy(copy_bits)
        group_daf.lol = group_lol                           # adopted, as in clone_empty().
        dodaf[groupkey] = group_daf

    return KeyedList(dodaf)


def _groupby(
        self: 'Daf',
        colname: str='',
        colnames: T_ls | None = None,
        omit_nulls: bool=False,         # do not group to values in column that are null ('')
        cols: T_cs | None = None,
        ) -> T_klodaf:

    """
    Split the Daf into several Daf instances, one for each value of a column.

    The result is a [KeyedList][daffodil.keyedlist.KeyedList] of Daf, a `klodaf`. It
    works as a dict: `groups[key]`, `keys()`, `items()`, `len()` and `in`. Its values
    also have positions, so `groups.values()[0]` is the first group. Each key is a value
    found in the column, in the order first seen. Each value is a Daf of the rows that
    have it, with all columns, or only the columns in `cols`.

    With all columns, a group holds the same row lists as this Daf. Nothing is copied,
    so it is fast and uses little memory. But a cell changed in a group is also changed
    here. To keep this Daf as it is, group a copy made with `copy(level='editable')`.
    Adding a column to a group makes new rows, and this Daf does not get the column.

    Use `cols` when you need only a few of many columns. The rows are then new lists of
    the columns kept. The other columns are never copied, and this Daf is not changed.
    It is much faster for a wide table.

    With several columns, as a list, the keys are tuples of their values. See
    `groupby_cols()`. With `omit_nulls`, rows that have an empty value in the
    column are left out.

    Args:
        colname: The column to group by.
        colnames: Several columns to group by. Use this or `colname`.
        omit_nulls: If True, leave out rows that have an empty value.
        cols: The columns that each group keeps, in this order. If None, all columns.
            A keyfield is kept only if all its columns are kept.

    Returns:
        A KeyedList that maps each value, or tuple of values, to a Daf.

    Raises:
        KeyError: The group column, or a name in `cols`, is not a column.

    Examples:
        >>> daf = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
        >>> daf
        <BLANKLINE>
        | g  | x  | y  |
        | -: | -: | -: |
        |  a |  1 | 10 |
        |  b |  2 | 20 |
        |  a |  3 | 30 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield=''; name=''
        <BLANKLINE>
        >>> groups = daf.groupby('g')
        >>> list(groups)
        ['a', 'b']
        >>> groups['a']
        <BLANKLINE>
        | g  | x  | y  |
        | -: | -: | -: |
        |  a |  1 | 10 |
        |  a |  3 | 30 |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.groupby('g', cols=['y'])['a']
        <BLANKLINE>
        | y  |
        | -: |
        | 10 |
        | 30 |
        <BLANKLINE>
        %% daf rows=2; cols=1; keyfield=''; name=''
        <BLANKLINE>
    """

    if isinstance(colname, list) and not colnames:
        return self.groupby_cols(colnames=colname, cols=cols)
    elif colnames and not colname:
        if len(colnames) > 1:
            return self.groupby_cols(colnames=colnames, cols=cols)
        else:
            colname = colnames[0]
            # can continue below.

    if not self.lol:
        return KeyedList()

    names, idxs, all_cols = self._cols_scope(cols)
    group_idx = self.hd[colname]

    groups: Dict[Any, T_lola] = {}

    for row_la in self.lol:
        fieldval = row_la[group_idx]
        if omit_nulls and fieldval is NULL:
            continue

        group_lol = groups.get(fieldval)
        if group_lol is None:
            group_lol = groups[fieldval] = []

        group_lol.append(row_la if all_cols else [row_la[idx] for idx in idxs])

    return self._new_group_dafs(groups, names, all_cols)


def _groupby_cols(self: 'Daf', colnames: T_ls, cols: T_cs | None = None) -> T_klodaf:

    """
    Split the Daf by the values of several columns.

    The result is a KeyedList of Daf, as `groupby()` gives. Each key is a tuple of the
    values in the columns, even for one column. Each value is a Daf of the rows that have them, with all
    columns, or only the columns in `cols`. With all columns, the rows are not
    copied. With `cols`, they are new lists of the columns kept, and the other
    columns are never copied.

    Args:
        colnames: The columns to group by.
        cols: The columns that each group keeps, in this order. If None, all columns.
            A keyfield is kept only if all its columns are kept.

    Returns:
        A KeyedList that maps each tuple of values to a Daf.

    Raises:
        KeyError: A name in `colnames` or `cols` is not a column.

    Examples:
        >>> daf = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
        >>> daf
        <BLANKLINE>
        | g  | x  | y  |
        | -: | -: | -: |
        |  a |  1 | 10 |
        |  b |  2 | 20 |
        |  a |  3 | 30 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield=''; name=''
        <BLANKLINE>
        >>> list(daf.groupby_cols(['g']))
        [('a',), ('b',)]
        >>> daf.groupby_cols(['g'], cols=['y'])[('a',)]
        <BLANKLINE>
        | y  |
        | -: |
        | 10 |
        | 30 |
        <BLANKLINE>
        %% daf rows=2; cols=1; keyfield=''; name=''
        <BLANKLINE>
    """

    names, idxs, all_cols = self._cols_scope(cols)
    group_idxs = [self.hd[colname] for colname in colnames]

    groups: Dict[Tuple[Any, ...], T_lola] = {}

    for row_la in self.lol:
        fieldval_tuple = tuple([row_la[idx] for idx in group_idxs])

        group_lol = groups.get(fieldval_tuple)
        if group_lol is None:
            group_lol = groups[fieldval_tuple] = []

        group_lol.append(row_la if all_cols else [row_la[idx] for idx in idxs])

    return self._new_group_dafs(groups, names, all_cols)


def _group_where(
        self: 'Daf',
        where: Callable[[Any], Any],
        *,
        indirect_col: str | None = None,
    ) -> T_klodaf:

    """
    Group the rows by the result of a function.

    The function gets each row and returns a key, or a list of keys, or None. A row
    goes into the group of each key. A list of keys puts the same row in several
    groups. None leaves the row out. The groups are Daf instances in a KeyedList, as
    `groupby()` gives. The rows are not copied, so changing a cell in a group changes it
    here too.

    To split the rows in two, have the function return True or False. A side with no
    rows then has no group, so read each side with `.get()`.

    Args:
        where: The function. It gets a row and returns None, a key, or an iterable of keys.
        indirect_col: A column that holds a dict, to read names that are not columns from.

    Returns:
        A KeyedList that maps each key to a Daf.

    Examples:
        >>> daf = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
        >>> daf
        <BLANKLINE>
        | g  | x  | y  |
        | -: | -: | -: |
        |  a |  1 | 10 |
        |  b |  2 | 20 |
        |  a |  3 | 30 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield=''; name=''
        <BLANKLINE>
        >>> groups = daf.group_where(lambda row: [row['g'], 'all'])
        >>> {key: len(group) for key, group in groups.items()}
        {'a': 2, 'all': 3, 'b': 1}
    """

    # --- helpers ---

    def _is_iterable(val: Any) -> bool:
        return isinstance(val, Iterable) and not isinstance(val, (str, bytes))

    # --- iterator selection ---

    if indirect_col:
        # reuse existing indirect semantics via wrapper
        def _iter_rows() -> Iterator['_IndirectRowView']:
            for row_kl in self.iter_klist():
                yield _IndirectRowView(row_kl, indirect_col)
        row_iter: Iterator[Any] = _iter_rows()
    else:
        # prefer KeyedList iteration for zero-copy
        row_iter = self.iter_klist()

    # --- grouping ---

    dodaf: Dict[Any, 'Daf'] = {}

    for row in row_iter:

        keys = where(row)

        if keys is None:
            continue

        if not _is_iterable(keys):
            keys = [keys]
        else:
            # if iterable but empty, skip early
            # (avoid constructing list unless needed)
            try:
                iterator = iter(keys)
                first = next(iterator)
            except StopIteration:
                continue
            # reconstruct iterable including first element
            keys = (k for k in (first, *iterator))

        for key in keys:

            sub_daf = dodaf.get(key)
            if sub_daf is None:
                sub_daf = Daf(name=str(key), keyfield=self.keyfield)
                dodaf[key] = sub_daf

            # append original KeyedList row (no copy)
            # if indirect view was used, unwrap to underlying row
            if isinstance(row, _IndirectRowView):
                sub_daf._basic_append(row.row)
            else:
                sub_daf._basic_append(row)


    return KeyedList(dodaf)


def _groupby_repeated_cols(
        self: 'Daf',
        colnames:   T_ls,                                   # the columns that make the key of a row.
        exclude:    Dict[str, T_la] | None = None,          # values of a column that make a row never match.
        ) -> T_klodaf:
    """
    Group the rows whose key repeats, and leave out every key that appears only once.

    Use it to find rows that share a key of one or more columns, such as two images with the
    same hash. Each group holds every row with its key, in table order, so its first row is
    the earliest. The groups come in the order of their first rows. An empty result means no
    key repeats. `groupby_cols()` gives every key, but here a key seen once makes no group.

    Each key is a tuple of the values, even for one column. Python NULL is a value like any
    other, so two rows with a blank in the same key column match. To keep a row from matching
    at all, as when its first key column is 0 or blank, name those values in `exclude`. A row
    is left out if any column named there holds one of its values. The values must match the
    stored type. A Daf read from a CSV file holds text, such as '0', until `apply_dtypes()`.

    The groups share their rows with this Daf, as in `groupby_cols()`. `SqlDaf` has the same
    method, which finds the keys in SQL and gives each group as a selection.

    Args:
        colnames: The columns that make the key of a row.
        exclude: For each column named, the values that keep a row out of every group.

    Returns:
        A KeyedList that maps each repeated key, a tuple of values, to a Daf of its rows.

    Raises:
        KeyError: A name in `colnames` or `exclude` is not a column.

    Examples:
        >>> daf = Daf(cols=['ballot_id', 'crc', 'size'],
        ...           lol=[['b1', 7, 10], ['b2', 0, 10], ['b3', 7, 10], ['b4', 0, 10], ['b5', 8, 10]])
        >>> daf
        <BLANKLINE>
        | ballot_id | crc | size |
        | --------: | --: | ---: |
        |        b1 |   7 |   10 |
        |        b2 |   0 |   10 |
        |        b3 |   7 |   10 |
        |        b4 |   0 |   10 |
        |        b5 |   8 |   10 |
        <BLANKLINE>
        %% daf rows=5; cols=3; keyfield=''; name=''
        <BLANKLINE>
        >>> repeats_klodaf = daf.groupby_repeated_cols(['crc', 'size'], exclude={'crc': [0, '']})
        >>> list(repeats_klodaf.keys())
        [(7, 10)]
        >>> repeats_klodaf[(7, 10)].col('ballot_id')
        ['b1', 'b3']
    """
    key_idx_li: T_li = [self.hd[col] for col in colnames]
    exclude_lot: List[Tuple[int, set]] = [(self.hd[col], set(vals_la)) for col, vals_la in (exclude or {}).items()]

    # ---- every row by its key, in table order. A dict keeps the keys in the order first seen.
    groups_dolola: Dict[Tuple[Any, ...], T_lola] = {}
    row_la: T_la
    for row_la in self.lol:
        if exclude_lot and any(row_la[idx] in vals_set for idx, vals_set in exclude_lot):
            continue
        key_t: Tuple[Any, ...] = tuple([row_la[idx] for idx in key_idx_li])
        groups_dolola.setdefault(key_t, []).append(row_la)

    repeated_dolola: Dict[Tuple[Any, ...], T_lola] = {key_t: rows_lol for key_t, rows_lol in groups_dolola.items() if len(rows_lol) > 1}
    return self._new_group_dafs(repeated_dolola, list(self.hd), all_cols=True)


def _groupby_cols_reduce(
        self: 'Daf',
        groupby_colnames: T_ls,
        # passed through to Daf.reduce() (row/reduction/cols/**kwargs) -- see the
        # manifest_reduce comment on Callable[...] above.
        func: Callable[..., Any],
        by: str='row',                                  # determines how the func is applied.
        reduce_cols: T_la | None = None,               # columns included in the reduce operation.
        diagnose: bool = False,
        **kwargs: Any,
        ) -> 'Daf':

    """
    Group the rows by several columns and reduce each group to one row.

    Use it to total the numbers for each combination of a few identifying columns.
    The result has one row for each combination. It has the group columns first and
    then the `reduce_cols`. It has no keyfield. For each group, `func` is used as in
    `reduce()`.

    With `reduce_cols`, and `by` of `row` or `sparse_row`, each group holds only those
    columns, and the `indirect_col` if there is one. A function that reads another
    column of the row will not find it.

    For example, group by gender, religion and zip code, and sum the counts of
    several causes in each group. The number of rows is the number of combinations
    that occur.

    Args:
        groupby_colnames: The columns that identify a group.
        func: The reduction function. See `reduce()`.
        by: How the function is applied. See `reduce()`.
        reduce_cols: The columns to reduce.
        diagnose: If True, print progress messages.
        **kwargs: Keyword arguments passed on to the function.

    Returns:
        The Daf with one row for each group.

    Examples:
        >>> daf = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
        >>> daf
        <BLANKLINE>
        | g  | x  | y  |
        | -: | -: | -: |
        |  a |  1 | 10 |
        |  b |  2 | 20 |
        |  a |  3 | 30 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.groupby_cols_reduce(['g'], Daf.sum_da, reduce_cols=['y'])
        <BLANKLINE>
        | g  | y  |
        | -: | -: |
        |  a | 40 |
        |  b | 20 |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield=''; name=''
        <BLANKLINE>
    """
    # unit test exists.
    """
            
            This can be commonly used when some colnames are important for grouping, while others
            contain values or numeric data that can be reduced.
            
            For example, consider the data table with the following columns:
            
            gender, religion, zipcode, cancer, covid19, gun, auto
            
            The data can be first grouped by the attribute columns gender, religion, zipcode, and then
            then prevalence of difference modes of death can be summed. The result is a daf with one
            row per unique combination of gender, religion, zipcode. Say we consider just M/F, C/J/I, 
            and two zipcodes 90001, and 90002, this would result in the following rows, where the 
            values in paranthesis are the reduced values for each of the numeric columns, such as the sum.
            
            In general, the number of rows is reduced to the product of number of unique values in each column
            grouped. In this case, there are 2 genders, 3 religions, and 2 zipcodes, resulting in
            2 * 3 * 2 = 12 rows.
            
            groupby_colnames = ['gender', 'religion', 'zipcode']
            reduce_colnames  = ['cancer', 'covid19', 'gun', 'auto']
            
            grouped_and_summed = data_table.groupby_cols_reduce(
                groupby_colnames=['gender', 'religion', 'zipcode'], 
                func = sum_np(),
                by='table',                                     # determines how the func is applied.
                reduce_cols = reduce_colnames,                  # columns included in the reduce operation.
                )

            
            cols = ['gender', 'religion', 'zipcode', 'cancer', 'covid19', 'gun', 'auto']
            lol = [
            ['M', 'C', 90001,  1,  2,  3,  4],
            ['M', 'C', 90001,  5,  6,  7,  8],
            ['M', 'C', 90002,  9, 10, 11, 12],
            ['M', 'C', 90002, 13, 14, 15, 16],
            ['M', 'J', 90001,  1,  2,  3,  4],
            ['M', 'J', 90001, 13, 14, 15, 16],
            ['M', 'J', 90002,  5,  6,  7,  8],
            ['M', 'J', 90002,  9, 10, 11, 12],
            ['M', 'I', 90001, 13, 14, 15, 16],
            ['M', 'I', 90001,  1,  2,  3,  4],
            ['M', 'I', 90002,  4,  3,  2,  1],
            ['M', 'I', 90002,  9, 10, 11, 12],
            ['F', 'C', 90001,  4,  3,  2,  1],
            ['F', 'C', 90001,  5,  6,  7,  8],
            ['F', 'C', 90002,  4,  3,  2,  1],
            ['F', 'C', 90002, 13, 14, 15, 16],
            ['F', 'J', 90001,  4,  3,  2,  1],
            ['F', 'J', 90001,  1,  2,  3,  4],
            ['F', 'J', 90002,  8,  7,  6,  5],
            ['F', 'J', 90002,  1,  2,  3,  4],
            ['F', 'I', 90001,  8,  7,  6,  5],
            ['F', 'I', 90001,  5,  6,  7,  8],
            ['F', 'I', 90002,  8,  7,  6,  5],
            ['F', 'I', 90002, 13, 14, 15, 16],
            ]

            result_lol = [
            ['M', 'C', 90001,  6,  8, 10, 12],
            ['M', 'C', 90002, 21, 24, 26, 18],
            ['M', 'J', 90001, 14, 16, 18, 20],
            ['M', 'J', 90002, 14, 16, 18, 20],
            ['M', 'I', 90001, 14, 16, 18, 20],
            ['M', 'I', 90002, 13, 13, 13, 13],
            ['F', 'C', 90001,  9,  9,  9,  9],
            ['F', 'C', 90002, 17, 17, 17, 17],
            ['F', 'J', 90001,  5,  5,  5,  5],
            ['F', 'J', 90002,  9,  9,  9,  9],
            ['F', 'I', 90001, 13, 13, 13, 13],
            ['F', 'I', 90002, 21, 21, 21, 21],
            ]
            
            This reduction can then be further grouped and summed to create reports or to allow for 
            comparison based on any combination of the subgroups.
            
            
        """

    # A plain sum by row of columns that are not grouped is added up in one pass, with no
    # Daf for each group. It is 3 to 5 times faster, with the same result.
    if func in (Daf.sum_da, Daf.sum_kl) and by == 'row' and reduce_cols and not diagnose \
            and all(col in self.hd and col not in groupby_colnames for col in reduce_cols) \
            and len(set(reduce_cols)) == len(reduce_cols) \
            and kwargs.keys() <= {'astype', 'is_sparse'} and not any(kwargs.values()):
        return self._groupby_cols_sum(groupby_colnames, reduce_cols)

    # divide up the table into groups where each group has a unique set of values in groupby_colnames
    # breakpoint() #temp

    if diagnose:  # pragma: no cover
        daf_utils.sts(f"Starting groupby_cols() of {len(self):,} records.", 3)

    scope = self._reduce_scope(by, reduce_cols, kwargs)     # the groups hold only the columns that are reduced.
    grouped_tdodaf = self.groupby_cols(groupby_colnames, cols=scope)

    if diagnose:  # pragma: no cover
        daf_utils.sts(f"Total of {len(grouped_tdodaf):,} groups. Reduction starting.", 3)

    result_daf = Daf(cols=groupby_colnames + (reduce_cols or []))

    for coltup, this_daf in grouped_tdodaf.items():

        if not this_daf:
            # nothing found with this combination of groupby cols.
            continue

        # apply the reduction function. by='row'/'table' (the only modes that make sense
        # here, to combine with the groupby cols below) always reduce to a single record.
        reduction_da = cast(T_ma, this_daf.reduce(func, by=by, cols=reduce_cols, **kwargs))

        # add back in the groupby cols
        group_t: T_ta = cast(T_ta, coltup)
        for idx, groupcolname in enumerate(groupby_colnames):
            reduction_da[groupcolname] = group_t[idx]

        result_daf.append(reduction_da)

    if diagnose:  # pragma: no cover
        daf_utils.sts(f"Reduction completed: {len(result_daf):,} records.", 3)

    return result_daf


def _groupby_cols_sum(self: 'Daf', groupby_colnames: T_ls, sum_cols: T_ls) -> 'Daf':
    """
    Sum `sum_cols` for each group of `groupby_colnames`, in one pass, for `groupby_cols_reduce()`.

    The result is what `groupby_cols_reduce()` with `sum_kl()` gives: one row for each
    group, in the order first seen, with the group columns and then the sums. Each
    total gets the same additions in the same order, so floats give the same totals.
    A cell that cannot be added is skipped, as `sum_kl()` skips it, and text is
    skipped before trying. A row too short for a group or sum column raises
    IndexError, as the split into groups does.

    On 100,000 rows in 43,242 groups, summing two columns took 0.174 s instead of
    0.767 s. Internal use.

    Args:
        groupby_colnames: The columns that identify a group.
        sum_cols: The columns to sum. They are columns, and not group columns.

    Returns:
        The Daf with one row for each group.
    """
    hd          = self.hd
    group_idxs  = [hd[col] for col in groupby_colnames]
    sum_idxs    = [hd[col] for col in sum_cols]
    num_sums    = len(sum_idxs)

    totals_dola: Dict[Tuple[Any, ...], T_la] = {}

    for row_la in self.lol:
        group_t = tuple([row_la[idx] for idx in group_idxs])
        totals_la = totals_dola.get(group_t)
        if totals_la is None:
            totals_la = totals_dola[group_t] = [0] * num_sums

        for isum in range(num_sums):
            value = row_la[sum_idxs[isum]]          # IndexError for a short row, as the split into groups gives.
            if value.__class__ is str:
                continue
            try:
                totals_la[isum] = totals_la[isum] + value + 0
            except (ValueError, TypeError):
                continue

    return Daf(lol=[[*group_t, *totals_la] for group_t, totals_la in totals_dola.items()],
               cols=groupby_colnames + sum_cols)


def _groupby_reduce(
        self: 'Daf',
        colname:        str,
        # passed through to Daf.reduce() (row/reduction/cols/**kwargs) -- see the
        # manifest_reduce comment on Callable[...] above.
        func:           Callable[..., Any], # function reduces one grouped daf to one record.
        by:             str='row',                                  # determines how the func is applied.
        reduce_cols:    T_cs | None = None,                           # columns included in the reduce operation.
        diagnose:       bool=False,
        **kwargs:       Any,
        ) -> 'Daf':
    """
    Group the rows by one column and reduce each group to one row.

    The result has one row for each value in the column, and its keyfield is that
    column. The columns in `reduce_cols` hold the reduced values. Other columns are
    empty. For each group, `func` is used as in `reduce()`.

    With `by='sparse_row'`, the result stays concise. It has two columns: this column,
    and the column named by `indirect_col`. That cell holds the dict of the group, with
    only the keys found in the group. Each group's dict may have other keys. With
    `reduce_cols` and a plain sum, each dict holds the columns its group has, in the
    order of `reduce_cols`, as `reduce()` describes.

    With `reduce_cols`, and `by` of `row` or `sparse_row`, each group holds only those
    columns, and the `indirect_col` if there is one. The other columns are never copied,
    which is much faster for a wide table. A function that reads another column of the
    row will not find it. The result still has all the columns of this Daf, and the
    columns that are not reduced are empty.

    Args:
        colname: The column to group by.
        func: The reduction function. See `reduce()`.
        by: How the function is applied. See `reduce()`.
        reduce_cols: The columns to reduce. All except `colname` if None.
        diagnose: If True, print progress messages.
        **kwargs: Keyword arguments passed on to the function.

    Returns:
        The Daf with one row for each group.

    Examples:
        >>> daf = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
        >>> daf
        <BLANKLINE>
        | g  | x  | y  |
        | -: | -: | -: |
        |  a |  1 | 10 |
        |  b |  2 | 20 |
        |  a |  3 | 30 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.groupby_reduce('g', Daf.sum_da, reduce_cols=['y'])
        <BLANKLINE>
        | g  | x  | y  |
        | -: | -: | -: |
        |  a |    | 40 |
        |  b |    | 20 |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield='g'; name=''
        <BLANKLINE>
    """

    if diagnose:
        logs.sts(f"{logs.prog_loc()} starting groupby '{colname}' operation", 3)
    # groupby(colname=<str>) (colnames not passed) always takes the Dict[str, 'Daf']
    # branch, never the tuple-keyed Dict[Tuple[str, ...], 'Daf'] one.
    scope = self._reduce_scope(by, reduce_cols, kwargs)     # the groups hold only the columns that are reduced.
    grouped_dodaf = cast(T_dodaf, self.groupby(colname, cols=scope))
    result_daf = Daf.reduce_dodaf_to_daf(
        func            = func,             # function reduces one grouped daf to one record.
        colname         = colname,
        grouped_dodaf   = grouped_dodaf,
        reduce_cols     = list(reduce_cols) if reduce_cols is not None else None,
        by              = by,
        diagnose        = diagnose,
        all_cols        = self.columns() if scope else None,    # the rows keep all the columns, as for whole groups.
        **kwargs,
        )
    return result_daf


def _multi_groupby(
        self: 'Daf',
        groupby_colnames:   T_cs,
        colnames:           T_cs | None = None,
        omit_nulls:         bool=False,         # do not group to values in column that are null ('')
        ) -> Dict[str, T_klodaf]:

    """
    Group the rows by each of several columns, one column at a time.

    The result is a dict of KeyedLists. The first key is the column. The second key is a
    value in that column. Each innermost value is a Daf of the rows that have it,
    with all columns, or only the columns in `colnames`.
    This is not a grouping by combinations. Use `groupby_cols()` for that.
    The groups are not reduced.

    With all columns, the groups hold the same row lists as this Daf, as in
    `groupby()`. A cell changed in a group is also changed here, and in the groups of
    the other columns. To keep this Daf as it is, group a copy made with
    `copy(level='editable')`.

    Use `colnames` when you need only a few of many columns. The rows are then new lists
    of the columns kept. The other columns are never copied, and this Daf is not changed.
    It is much faster for a wide table.

    Args:
        groupby_colnames: The columns to group by, one at a time.
        colnames: The columns that each group keeps, in this order. If None, all columns.
            A keyfield is kept only if all its columns are kept.
        omit_nulls: If True, leave out rows that have an empty value.

    Returns:
        A dict that maps each column to a KeyedList of value and Daf.

    Raises:
        KeyError: A group column, or a name in `colnames`, is not a column.

    Examples:
        >>> daf = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
        >>> daf
        <BLANKLINE>
        | g  | x  | y  |
        | -: | -: | -: |
        |  a |  1 | 10 |
        |  b |  2 | 20 |
        |  a |  3 | 30 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield=''; name=''
        <BLANKLINE>
        >>> groups = daf.multi_groupby(['g', 'x'])
        >>> list(groups), list(groups['g'])
        (['g', 'x'], ['a', 'b'])
        >>> daf.multi_groupby(['g'], colnames=['y'])['g']['a']
        <BLANKLINE>
        | y  |
        | -: |
        | 10 |
        | 30 |
        <BLANKLINE>
        %% daf rows=2; cols=1; keyfield=''; name=''
        <BLANKLINE>
    """

    if isinstance(groupby_colnames, str):
        groupby_colnames = [groupby_colnames]

    if not self.lol:
        return {}

    names, idxs, all_cols = self._cols_scope(colnames)
    group_idxs = [(col, self.hd[col]) for col in groupby_colnames]

    grouped: Dict[str, Dict[Any, T_lola]] = {}

    for row_la in self.lol:
        for col, group_idx in group_idxs:
            col_groups = grouped.get(col)
            if col_groups is None:
                col_groups = grouped[col] = {}

            fieldval = row_la[group_idx]
            if omit_nulls and fieldval is NULL:
                continue

            group_lol = col_groups.get(fieldval)
            if group_lol is None:
                group_lol = col_groups[fieldval] = []

            group_lol.append(row_la if all_cols else [row_la[idx] for idx in idxs])

    return {col: self._new_group_dafs(col_groups, names, all_cols) for col, col_groups in grouped.items()}


def _reduce_dodaf_to_daf(
        colname:        str,
        # passed through to Daf.reduce() (row/reduction/cols/**kwargs) -- see the
        # manifest_reduce comment on Callable[...] above.
        func:           Callable[..., Any],
        grouped_dodaf:  T_dodaf | T_klodaf,
        reduce_cols:    T_la | None = None,    # columns included in the reduce operation, None = all except for colname.
        diagnose:       bool=False,
        all_cols:       T_ls | None = None,    # the columns of the result, if the groups hold only some of them.
        **kwargs:       Any,
        ) -> 'Daf':
    """
    Reduce each Daf in a dict to one row, and join the rows in a Daf.

    This is the second half of `groupby_reduce()`. The dict maps the values of a
    column to Daf instances. Each Daf is reduced with `reduce()`. The value is
    stored in the column `colname` of the row. The result has `colname` as its keyfield.

    Args:
        colname: The column that holds the value of each group.
        func: The reduction function. See `reduce()`.
        grouped_dodaf: A dict or a KeyedList that maps each value to a Daf.
        reduce_cols: The columns to reduce. All columns except `colname` if None.
        diagnose: If True, print progress messages.
        all_cols: The columns of the result, in order. Use it when each group holds only
            some columns, so that the rows have the same columns as for whole groups. The
            columns that are not reduced are then empty.
        **kwargs: Keyword arguments passed on to the function.

    Returns:
        The Daf with one row for each group.

    Examples:
        >>> groups = {'a': Daf(lol=[[1], [2]], cols=['n']), 'b': Daf(lol=[[3]], cols=['n'])}
        >>> result = Daf.reduce_dodaf_to_daf('g', Daf.sum_da, groups)
        >>> result
        <BLANKLINE>
        | n  | g  |
        | -: | -: |
        |  3 |  a |
        |  3 |  b |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield='g'; name=''
        <BLANKLINE>
    """

    if diagnose:
        logs.sts(f"{logs.prog_loc()} Grouped into {len(grouped_dodaf)} groups.", 3)

        # display the first three daf arrays and the last

        for group_num, (colval, this_daf) in enumerate(grouped_dodaf.items()):
            if group_num < 3 or group_num >= len(grouped_dodaf) - 1:
                logs.sts(f"## Group {group_num}: {colname}={colval}\n\n{this_daf}\n\n", 3)


    result_daf = Daf(keyfield = colname)

    for group_num, (colval, this_daf) in enumerate(grouped_dodaf.items()):

        # maybe remove colname from cols here

        if diagnose:
            logs.sts(f"## Group {group_num}: {colname}={colval}\n\n{this_daf}\n\n", 3)
        # by defaults to 'row' (the only mode used here), which always reduces to a
        # single record.
        reduction_da = cast(T_ma, this_daf.reduce(func, cols=reduce_cols, **kwargs))
                # def reduce(
                        # self,
                        # func: Callable[[T_da, T_da], T_da | T_la],
                        # by: str='row',
                        # cols: Iterable | None = None,                  # columns included in the reduce operation.
                        # initial_da: T_da | None = None,
                        # **kwargs: Any,
                        # ) -> T_da | T_la:

        if diagnose:
            logs.sts(f"Post reduction: {Daf.from_lod([cast(T_da, reduction_da)])=}", 3)

        # A sparse reduction stays concise. The dict of each group is one cell, in the column
        # named for the indirect column, and holds only the keys found in the group.
        if kwargs.get('by') == 'sparse_row':
            sparse_record_da: T_da = {colname: colval, kwargs['indirect_col']: reduction_da}
            result_daf.append(sparse_record_da)
            continue

        # add colname:colval to the dict, as it is removed by the reduction func.
        reduction_da[colname] = colval

        if all_cols:
            reduction_da = {col: reduction_da.get(col, NULL) for col in all_cols}

        if diagnose:
            logs.sts(f"Post add colname:colval to the dict: {Daf.from_lod([cast(T_da, reduction_da)])=}", 3)

        # this will also maintain the kd.
        result_daf.append(reduction_da)

    if diagnose:
        logs.sts(f"{logs.prog_loc()} result_daf:\n\n{result_daf}", 3)

    return result_daf


def _multi_groupby_reduce(
        self: 'Daf',
        colnames:       T_cs,
        # passed through to Daf.reduce() (row/reduction/cols/**kwargs) -- see the
        # manifest_reduce comment on Callable[...] above.
        func:           Callable[..., Any],  # function reduces one grouped daf to one record.
        by:             str='row',                                  # determines how the func is applied.
        reduce_cols:    T_cs | None = None,                           # columns included in the reduce operation.
        diagnose:       bool=False,
        **kwargs:       Any,
        ) -> Dict[str, 'Daf']:
    """
    Group by each of several columns and reduce each group to one row.

    This is `multi_groupby()` followed by `groupby_reduce()` for each column. The
    result is a dict. Each key is a column. Each value is a Daf with one row for
    each value of that column, and that column as its keyfield.

    With `reduce_cols`, and `by` of `row` or `sparse_row`, each group holds only those
    columns, and the `indirect_col` if there is one. A function that reads another
    column of the row will not find it. The result still has all the columns of this
    Daf, and the columns that are not reduced are empty.

    Args:
        colnames: The columns to group by, one at a time.
        func: The reduction function. See `reduce()`.
        by: How the function is applied. See `reduce()`.
        reduce_cols: The columns to reduce.
        diagnose: If True, print progress messages.
        **kwargs: Keyword arguments passed on to the function.

    Returns:
        A dict that maps each column to its Daf.

    Examples:
        >>> daf = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
        >>> daf
        <BLANKLINE>
        | g  | x  | y  |
        | -: | -: | -: |
        |  a |  1 | 10 |
        |  b |  2 | 20 |
        |  a |  3 | 30 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.multi_groupby_reduce(['g'], Daf.sum_da, reduce_cols=['y'])['g']
        <BLANKLINE>
        | g  | x  | y  |
        | -: | -: | -: |
        |  a |    | 40 |
        |  b |    | 20 |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield='g'; name=''
        <BLANKLINE>
    """

    if diagnose:
        logs.stsloc(f"starting multi-groupby '{colnames}' operation", 3)
    scope = self._reduce_scope(by, reduce_cols, kwargs)     # the groups hold only the columns that are reduced.
    multi_grouped_dododaf = self.multi_groupby(colnames, colnames=scope)

    result_dodaf: T_dodaf = {}
    reduce_cols_la = list(reduce_cols) if reduce_cols is not None else None
    result_cols = self.columns() if scope else None         # the rows keep all the columns, as for whole groups.

    for colname, grouped_dodaf in multi_grouped_dododaf.items():

        result_dodaf[colname] = Daf.reduce_dodaf_to_daf(
            func            = func,         # function reduces one grouped daf to one record.
            colname         = colname,
            grouped_dodaf   = grouped_dodaf,
            reduce_cols     = reduce_cols_la,
            diagnose        = diagnose,
            all_cols        = result_cols,
            **kwargs,
            )
    if diagnose:
        logs.stsloc(f"Grouped into {len(result_dodaf)} groups.", 3)

        for group_num, (colval, this_daf) in enumerate(result_dodaf.items()):
            if group_num < 3 or group_num >= len(grouped_dodaf) - 1:
                logs.sts(f"## Group {group_num}: {colname}={colval}\n\n{this_daf}\n\n", 3)

    return result_dodaf


def _apply_colwise(
    self: 'Daf',
    target_col: str,
    func: Callable[[T_da], Any],
    *,
    default: Any = 0,
) -> "Daf":
    """
    Compute one column from the other columns of each row, in place.

    The function gets each row, as a dict, and returns the value for `target_col`.
    If the function raises an error for a row, that row gets `default`. If
    `target_col` is not a column, it is added at the right, and the rows are
    filled with `default` first. Use it for a ratio or a total of other columns.

    Args:
        target_col: The column to store the result in.
        func: A function that takes a row and returns the value.
        default: The value for a row where the function raises an error.

    Returns:
        This Daf, which has been changed.

    Examples:
        >>> daf = Daf(lol=[[1, 2], [3, 0]], cols=['a', 'b'])
        >>> daf
        <BLANKLINE>
        | a  | b  |
        | -: | -: |
        |  1 |  2 |
        |  3 |  0 |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.apply_colwise('ratio', lambda row: row['a'] / row['b'], default=-1.0)
        <BLANKLINE>
        | a  | b  | ratio |
        | -: | -: | ----: |
        |  1 |  2 |   0.5 |
        |  3 |  0 |  -1.0 |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield=''; name=''
        <BLANKLINE>
    """

    if target_col not in self.columns():
        self.insert_col(target_col, default=default)

    def row_op(row: T_da) -> T_da:
        try:
            row[target_col] = func(row)
        except Exception:
            row[target_col] = default

        return row

    self.apply_in_place(row_op)

    return self


def _daf_sum(
        self: 'Daf',
        by:             str = 'row',                    # row|col|table|sparse_row
        cols:           T_cs | None = None,
        indirect_col:   str = '',                       # indirect_col is required for lol array.
        **kwargs:       Any,
        ) -> T_da:

    # for the default by='row' (the only mode used by any caller/test), reduce() always
    # returns a single record.
    """
    Add up the columns, using `reduce()` and `sum_kl()`.

    Cells that cannot be added, such as text, are skipped with no error. See
    [sum_kl][daffodil.daf_apply._sum_kl] for what that means for a bad number. A column
    that is not in `cols` is empty in the result.

    Args:
        by: How the function is applied. See `reduce()`.
        cols: The columns to sum. All if None.
        indirect_col: A column that holds a dict. Required for `sparse_row`.
        **kwargs: Keyword arguments passed on to `reduce()`.

    Returns:
        A dict with a total for each column.

    Examples:
        >>> daf = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
        >>> daf
        <BLANKLINE>
        | g  | x  | y  |
        | -: | -: | -: |
        |  a |  1 | 10 |
        |  b |  2 | 20 |
        |  a |  3 | 30 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.daf_sum(cols=['y'])
        {'g': '', 'x': '', 'y': 60}
    """

    return cast(T_da, self.reduce(func=Daf.sum_kl, by=by, cols=cols, indirect_col=indirect_col, **kwargs))


@overload
def _reduce(self: 'Daf', func: Callable[..., Any], by: Literal['row', 'table', 'sparse_row'] = 'row', cols: T_cs | None = None, initial_da: T_ma | None = None,
           indirect_col: str = '', silent_error: bool = False, **kwargs: Any) -> T_da: ...
@overload
def _reduce(self: 'Daf', func: Callable[..., Any], *, by: Literal['col'], cols: T_cs | None = None, initial_da: T_ma | None = None,
           indirect_col: str = '', silent_error: bool = False, **kwargs: Any) -> T_la: ...
@overload
def _reduce(self: 'Daf', func: Callable[..., Any], by: str = 'row', cols: T_cs | None = None, initial_da: T_ma | None = None,
           indirect_col: str = '', silent_error: bool = False, **kwargs: Any) -> T_ma | T_la: ...

def _reduce(
        self: 'Daf',
        # real reduction functions (sum_da, count_values_da, ...) take further keyword-only
        # params beyond the row/reduction pair (cols, astype, omit_nulls, is_sparse,
        # diagnose, ...) passed through via **kwargs below, so the fixed 2-arg Callable
        # forms can't express this.
        func:           Callable[..., Any],
        by:             str = 'row',                                # row|col|table|sparse_row
        cols:           T_cs | None = None,                    # columns included in the reduce operation.
        initial_da:     T_ma | None = None,
        indirect_col:   str = '',                                   # indirect_col is required for lol array.
        silent_error:   bool = False,                               # if True, skip rows where func raises.
        **kwargs:       Any,
        ) -> T_ma | T_la:
    """
    Combine all the rows into one result, using a function.

    The function gets each row, and the result so far, and returns the new result.
    A sum or a count is a reduction. With `by='row'`, the function is called as
    `func(row, result, cols=cols, **kwargs)`. The result starts as 0 for each of
    the columns, or as `initial_da`, which is changed in place. It returns a dict with
    every column. Columns that are not in `cols` are empty. A key of the result that
    is not a column of the Daf is dropped, so a reduction by row gives one value for
    each column. For a result of another shape, use `by='table'`, or a loop.

    A plain sum, with `sum_da()` or `sum_kl()` and no other options, adds down each
    column instead of across each row. That is about 5 times faster, and gives the same
    totals. The function is then not called for each row.

    With `by='row'`, each row is a new [KeyedList][daffodil.keyedlist.KeyedList]
    that points at the row of the table. It reads like a dict, as in `row['x']` or
    `row.get('x', 0)`. Writing to it changes the table. A function that needs a real
    dict, for example to pass to `json.dumps()`, can call `dict(row)`. A row that is
    shorter or longer than the columns gets a KeyedList of a copy, padded with NULL
    or cut to fit.

    The standard functions are `sum_kl()` and `count_values_kl()`. They read the
    KeyedList directly. `Daf.sum_da` and `Daf.count_values_da` are replaced by them
    here, since they give the same results.

    With `by='col'` the function gets each column as a list and the result so far
    as a list. With `by='table'` it gets this Daf and `cols`, and its result is
    returned as it is. With `by='sparse_row'`, rows are read from the dict in
    `indirect_col`, and the result starts as `initial_da` or an empty dict.

    A plain sum of sparse rows, with `sum_da()` or `sum_kl()` and no other options,
    chooses its loop once, from the first 100 rows. No key is tested inside the loop.

    - With no `cols`, every key of each row is added.
    - With `cols` no longer than the mean number of keys in a row, each column is
      looked up in each row.
    - With longer `cols`, every key is added, and the columns are picked at the end.

    The totals are the same in each case. With `cols`, the result holds the columns
    that some row has, in the order of `cols`. A column that no row has is left out,
    not given as 0. To sum nearly every key, give no `cols` and pick the totals needed.

    An error in the function stops the reduction. With `silent_error=True` the
    row is skipped. The result may then be missing rows, with no sign of it. An
    empty Daf gives an empty dict for `by='row'`.

    To reduce part of the table, select the rows or columns first, or give `cols`.

    Args:
        func: The reduction function. For example `Daf.sum_da`.
        by: `row`, `col`, `table` or `sparse_row`.
        cols: The columns included in the reduction. All columns if None.
        initial_da: The result to start from, instead of zeros.
        indirect_col: A column that holds a dict. Required for `sparse_row`.
        silent_error: If True, a row for which the function raises is skipped.
        **kwargs: Keyword arguments passed on to the function.

    Returns:
        A dict for `row`, `table` and `sparse_row`. A list for `col`.

    Raises:
        ValueError: `sparse_row` is used with no `indirect_col`.
        NotImplementedError: `by` is not recognized.

    Examples:
        >>> daf = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
        >>> daf
        <BLANKLINE>
        | g  | x  | y  |
        | -: | -: | -: |
        |  a |  1 | 10 |
        |  b |  2 | 20 |
        |  a |  3 | 30 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.reduce(Daf.sum_da, cols=['x', 'y'])
        {'g': '', 'x': 6, 'y': 60}

        A sparse sum, of the dict in each row:

        >>> votes_daf = Daf(lol=[['b1', {'c1v': 1, 'c0b': 1}], ['b2', {'c0b': 1, 'c2v': 1}]], cols=['ballot_id', 'cv'])
        >>> votes_daf
        <BLANKLINE>
        | ballot_id |          cv          |
        | --------: | -------------------: |
        |        b1 | {'c1v': 1, 'c0b': 1} |
        |        b2 | {'c0b': 1, 'c2v': 1} |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield=''; name=''
        <BLANKLINE>
        >>> votes_daf.reduce(Daf.sum_da, by='sparse_row', indirect_col='cv')
        {'c1v': 1, 'c0b': 2, 'c2v': 1}
        >>> votes_daf.reduce(Daf.sum_da, by='sparse_row', indirect_col='cv', cols=['c2v', 'c0b'])
        {'c2v': 1, 'c0b': 2}
    """
    if by == 'table':
        reduction_ma = func(self, cols, **kwargs)
        return reduction_ma

    if by == 'row':

        if not self:
            return {}

        #allcols = self.hd.keys()

        cols_iter: Iterable[str]
        if cols is None:
            cols_iter = self.hd.keys()
        elif isinstance(cols, str):
            cols_iter = [cols]                              # perflint-reviewed (use-tuple-over-list)
        # elif isinstance(cols, list) and len(cols) > 10:
            # cols_iter = dict.fromkeys(cols)
        else:
            cols_iter = {col: None for col in self.hd.keys() if col in cols}

        if initial_da is not None:
            reduction_ma = initial_da
        else:
            reduction_ma = dict.fromkeys(cols_iter, 0)

        # The standard functions take the KeyedList versions, which give the same result faster.
        if func is Daf.sum_da:
            func = Daf.sum_kl
        elif func is Daf.count_values_da:
            func = Daf.count_values_kl

        # A plain sum adds down each column instead, about 5 times faster, with the same result.
        if func is Daf.sum_kl and initial_da is None and not silent_error \
                and kwargs.keys() <= {'astype', 'is_sparse', 'diagnose'} and not any(kwargs.values()):
            iter(self)                          # raises KeysDisabledError if there are no column names.
            reduction_ma = self._sum_by_col(cols_iter, reduction_ma)
            return {key: reduction_ma.get(key, '') for key in self.hd.keys()}

        # Each row is a new KeyedList that points at the row of the table. A short or
        # long row gets one of a copy, padded with NULL or cut to the number of columns.
        # The KeyedList is made as KeyedList(hd, row_la) would make it, without its checks,
        # since the length is checked here. That saves about 0.2 us per row.
        iter(self)                              # raises KeysDisabledError if there are no column names.
        hd = self.hd
        num_cols = len(hd)
        new_klist = KeyedList.__new__
        for row_la in self.lol:
            if len(row_la) != num_cols:
                row_la = (row_la + [NULL] * num_cols)[:num_cols]
            row_kl = new_klist(KeyedList)
            row_kl.hd = hd
            row_kl._values = row_la
            row_kl._hd_shared = True
            try:
                reduction_ma = func(row_kl, reduction_ma, cols=cols_iter, **kwargs)
            except Exception:
                if not silent_error:
                    raise

        # normalize the result so it contains all columns
        result_ma = {key: reduction_ma.get(key,'') for key in self.hd.keys()}

        return result_ma

    elif by == 'col':
        reduction_la: T_la = []                   # perflint-reviewed (use-tuple-over-list)
        num_cols = self.num_cols()
        for icol in range(num_cols):
            col_la = self.icol(icol)
            reduction_la = func(col_la, reduction_la, **kwargs)
        return reduction_la

    elif by == 'sparse_row':
        if not indirect_col:
            raise ValueError("indirect_col must be specified for 'sparse_row' reduction and lol array.")

        # A plain sum picks its loop once, before the first row.
        if Daf._is_plain_sum(func, initial_da, silent_error, kwargs):
            return Daf._sum_sparse((daf_utils.get_indirect_da(row_ma, indirect_col) for row_ma in self), cols)

        reduction_ma = initial_da or {}

        for row_ma in self:
            indirect_ma = daf_utils.get_indirect_da(row_ma, indirect_col)
            # indirect_val = row_da[indirect_col]
            # if isinstance(indirect_val, str):
            #     indirect_da = daf_utils.safe_convert_json_to_obj(indirect_val)
            # else:
            #     indirect_da = indirect_val
            try:
                reduction_ma = func(indirect_ma, reduction_ma, cols=cols, is_sparse=True, **kwargs)
            except Exception:
                if not silent_error:
                    raise

            # def count_values_da(row_da: T_da, reduction_da: T_da, cols: Iterable, omit_nulls: bool=False) -> T_dodi:
            # def sum_da         (row_da: T_da, reduction_da: T_da, cols: Iterable, astype: Type | None = None, diagnose:bool=False

        # # normalize the result so it contains all columns
        # result_da = {key: reduction_da.get(key,'') for key in self.hd.keys()}

        return reduction_ma



    else:
        raise NotImplementedError
    return [] # for mypy only.


def _sum_by_col(self: 'Daf', cols: Iterable[str], reduction_ma: T_ma) -> T_ma:
    """
    Add up each column in `cols` into `reduction_ma`, as `reduce()` with `sum_kl()` does.

    `sum_kl()` goes across each row. This goes down each column, which gives the same
    additions in the same order, so the same totals, also for floats. A whole column is
    first added in C with `functools.reduce()`. The builtin `sum()` is not used, because
    from Python 3.12 it adds floats with compensated summation, which gives other totals.

    If a cell cannot be added, the column is added again one cell at a time, from the
    start. A cell that cannot be added is then skipped, as `sum_kl()` skips it. Text
    is skipped before trying, since adding it always fails, and an exception is slow.
    A missing cell of a short row is skipped too, as the NULL that `reduce()` pads it with.

    On 100,000 rows, two numeric columns took 0.015 s instead of 0.078 s, and with two
    more columns of mixed text and numbers, 0.034 s instead of 0.173 s. Internal use.

    Args:
        cols: The columns to add.
        reduction_ma: The running totals, which start at 0. It is changed.

    Returns:
        The running totals.
    """
    hd  = self.hd
    lol = self.lol
    add = operator.add

    for col in cols:
        icol  = hd.get(col)
        start = reduction_ma.get(col, 0)
        if icol is None:                    # not a column: sum_kl() adds 0.
            reduction_ma[col] = start
            continue
        try:
            total = functools.reduce(add, map(operator.itemgetter(icol), lol), start) + 0    # + 0 as sum_kl() does.
        except (ValueError, TypeError, IndexError):
            total = start
            for row_la in lol:
                try:
                    value = row_la[icol]
                    if value.__class__ is str:
                        continue
                    total = total + value + 0
                except (ValueError, TypeError, IndexError):
                    continue
        reduction_ma[col] = total

    return reduction_ma


def _sum_da( row_da:         T_ma,                       # the current row from the daf array.
            reduction_da:   T_rma,                      # an accumulated result. Must be initialized for all columns in cols.
            *,
            cols:           T_cs | None = None,    # defines the active columns. Can be a list, keys(), range, or slice
            astype:         Type | None = None,        # a type like int, float, str to cast the value if it is not that type. Optional.
            is_sparse:      bool=False,
            diagnose:       bool=False
            ) -> T_rma: # result_ma -- same object (and type) as reduction_da, returned in place
    """
    Add the values of a row to a running total. Use it with `reduce()`.

    This is a static method. A value that cannot be added, such as text or an empty
    cell, is skipped with no error. See [sum_kl][daffodil.daf_apply._sum_kl] for what that
    means for a bad number. The total is changed and returned. With `astype`, each value
    is first converted to that type. `reduce()` uses `sum_kl()` in its place.

    Args:
        row_da: The current row.
        reduction_da: The running total. It is changed.
        cols: The columns to add. All if None.
        astype: A type to convert each value to before adding.
        is_sparse: If True, the row may hold only some of the columns.
        diagnose: Not used.

    Returns:
        The running total.

    Examples:
        >>> Daf.sum_da({'x': 1, 'y': 2}, {'x': 10, 'y': 0}, cols=['x', 'y'])
        {'x': 11, 'y': 2}
    """
    # def sum_da         (row_da: T_da, reduction_da: T_da, cols: Iterable, astype: Type | None = None, diagnose:bool=False

    diagnose = diagnose
    #nan_indicator = ''

    # for col, value in row_da.items():       # doing it this way requires a check for existence in each loop.
        # if col not in cols:                 # this check is not needed in the version below.
            # continue                        # 251 vs 207.

    if cols is None or is_sparse:
        # iterate by cols in row_da
        for col, val in row_da.items():

            if cols and col not in cols:
                continue

            try:
                # note that the "+ 0" in the following expression is important so that
                # string operands will cause 'TypeError: can only concatenate str (not "int") to str'
                # This will only invoke the exception when encountering non-numeric data, and does
                # not require an initial check

                reduction_da[col] = reduction_da.get(col, 0) + val + 0     # type: ignore[index, call-overload]

            #except Exception:
            except (ValueError, TypeError):
                continue

        return reduction_da

    else:
        for col in cols:

            value = row_da.get(col, 0)
            # if value == nan_indicator:        # this makes the loop take 10x longer (2162) (1044% of original)
            # if isinstance(value, str):        # this makes the loop take 42% longer (294)
            # if isinstance(value, str) and value == '':  # same (294)
            # if value is None or isinstance(value, str) and not value:     (350) vs 207 = 69% longer
            # if value == '':                     # this makes the loop take 10x longer (2105) (1044% of original)
            # if isinstance(value, str) and not value:    # this makes the loop take 50% longer (305)
            # if isinstance(value, str):          # this makes the loop take 50% longer (305)
                                                # but is needed to disallow concatenating strings.
                # continue
            # if isinstance(value, (int, float, np.int32, np.int64)):
                # (indent)

            # the try/except below is the most time efficient way to handle this while still
            # allowing for astype and nan values. (212 ms for 1000x1000 array)
            # Please note that the cols value is determined
            # prior to entering the function and must contain an iterable, even if all columns
            # are specified.

            # writing this loop the other way around, by going through all columns and skipping those not
            # mentioned in cols is also very inefficient.

            # 213 for the version below, which seems like it should be fastest.
            # but it is slightly less advantageous because initial assignment is inside the try/except block.

            # try:
                # if astype:
                    # value = row_da[col]
                    # if astype==int and isinstance(value, (str, float, bool)):
                        # value = int(float(value))
                    # elif astype==float and isinstance(value, (str, int, bool)):
                        # value = float(value)
                    # elif astype==str and isinstance(value, (float, int, bool)):
                        # value = str(value)
                    # accum_da[col] += value
                # else:
                    # accum_da[col] += row_da[col]
            # except Exception:
                # continue

            # this one measured at 230
            # value = row_da[col]
            # try:
                # if astype:
                    # if astype==int and isinstance(value, (str, float, bool)):
                        # value = int(float(value))
                    # elif astype==float and isinstance(value, (str, int, bool)):
                        # value = float(value)
                    # elif astype==str and isinstance(value, (float, int, bool)):
                        # value = str(value)

                # accum_da[col] += value

            # except Exception:
                # continue

            # this one measured at 209 with all cols and no astype.
            if astype:
                #value = row_da.get(col, 0)
                try:
                    if astype is int and isinstance(value, (str, float, bool)):
                        value = int(float(value or 0))
                    elif astype is float and isinstance(value, (str, int, bool)):
                        value = float(value or 0)
                    elif astype is str and isinstance(value, (float, int, bool)):
                        value = str(value)

                    reduction_da[col] += value

                except ValueError:
                    continue

            else:
                try:
                    # note that the "+ 0" in the following expression is important so that
                    # string operands will cause 'TypeError: can only concatenate str (not "int") to str'
                    # This will only invoke the exception when encountering non-numeric data, and does
                    # not require an initial check

                    reduction_da[col] = reduction_da.get(col, 0) + value + 0

                #except Exception:
                except (ValueError, TypeError):
                    continue



        return reduction_da


def _sum_sparse_keys(row_da: T_ma, reduction_da: T_rma, **kwargs: Any) -> T_rma:
    """
    Add every key of a sparse row to the totals, with no test of the key. Internal use.

    A value that cannot be added is skipped, as `sum_da()` skips it.
    Other keywords, such as `is_sparse`, are taken and not used.
    """
    for col, val in row_da.items():
        try:
            reduction_da[col] = reduction_da.get(col, 0) + val + 0     # type: ignore[index, call-overload]
        except (ValueError, TypeError):
            continue
    return reduction_da


def _sum_sparse_cols(row_da: T_ma, reduction_da: T_rma, *, cols: T_ts, **kwargs: Any) -> T_rma:
    """
    Add the columns in `cols` that a sparse row has, looking up each one. Internal use.

    A column the row does not have is skipped, so it is never added as 0.
    A value that cannot be added is skipped, as `sum_da()` skips it.
    """
    get = row_da.get
    for col in cols:
        val = get(col)
        if val is None:
            continue
        try:
            reduction_da[col] = reduction_da.get(col, 0) + val + 0     # type: ignore[index, call-overload]
        except (ValueError, TypeError):
            continue
    return reduction_da


def _sparse_sum_plan(cols: T_cs | str | None, sample_lodict: List[Any]) -> Tuple[T_ts | None, T_ls | None]:
    """
    Choose how a plain sum of sparse rows loops. Internal use.

    Each row is a dict that holds only some keys. The choice is made once, before the loop.

    - With no `cols`, every key of each row is added. No key is tested.
    - With `cols` no longer than the mean number of keys in the sample rows, the
      loop goes over `cols` and looks each one up in the row.
    - With longer `cols`, every key of each row is added, and the columns are picked
      from the totals at the end. Adding a key costs about what testing it would cost,
      so this is faster than testing each key of each row.

    Returns:
        The columns to loop over, or None to add every key.
        The columns to pick at the end, in their order, or None to keep every total.
    """
    if cols is None:
        return None, None
    select_ls: T_ls = [cols] if isinstance(cols, str) else list(dict.fromkeys(cols))
    num_keys: int = sum(len(row_da) for row_da in sample_lodict)
    if len(select_ls) * len(sample_lodict) <= num_keys:
        return tuple(select_ls), select_ls
    return None, select_ls


def _sum_sparse(rows_iter: Iterator[Any], cols: T_cs | str | None) -> T_da:
    """
    Sum sparse rows, each a dict that holds only some keys, as `reduce()` does with `sum_da()`.

    The loop is chosen once, by `_sparse_sum_plan()`, from the first rows.
    The totals are the same as `sum_da()` gives, added in the same order.
    With `cols`, the result holds the columns of `cols` that some row has, in the order of `cols`.
    With no `cols`, it holds every key, in the order the rows first have them. Internal use.
    """
    sample_lodict: List[Any] = list(itertools.islice(rows_iter, Daf._SPARSE_SAMPLE_ROWS))
    loop_cols_t, select_ls = Daf._sparse_sum_plan(cols, sample_lodict)

    reduction_da: T_da = {}
    row_da: Any
    if loop_cols_t is None:
        for row_da in itertools.chain(sample_lodict, rows_iter):
            reduction_da = Daf._sum_sparse_keys(row_da, reduction_da)
    else:
        for row_da in itertools.chain(sample_lodict, rows_iter):
            reduction_da = Daf._sum_sparse_cols(row_da, reduction_da, cols=loop_cols_t)

    if select_ls is None:
        return reduction_da
    selected_da: T_da = {col: reduction_da[col] for col in select_ls if col in reduction_da}
    return selected_da


def _is_plain_sum(func: Callable[..., Any], initial_da: Any, silent_error: bool, kwargs: Dict[str, Any]) -> bool:
    """
    True if a reduction is a plain sum, with `sum_da()` or `sum_kl()` and no other options. Internal use.

    `reduce()` then adds the rows by its own loop, which gives the same totals faster.
    """
    return (func in (Daf.sum_da, Daf.sum_kl) and initial_da is None and not silent_error
            and kwargs.keys() <= {'astype', 'is_sparse', 'diagnose'} and not any(kwargs.values()))


def _sum_kl( row_kl:         Any,                        # the current row, a KeyedList.
            reduction_da:   T_rma,                      # the running total.
            *,
            cols:           T_cs | None = None,         # the columns to add.
            astype:         Type | None = None,         # a type to convert each value to first.
            is_sparse:      bool=False,
            diagnose:       bool=False
            ) -> T_rma:
    """
    Add the values of a row to a running total. This is the standard sum for `reduce()`.

    It gives the same result as `sum_da()`, and is faster on the rows that
    `reduce()` gives it, which are [KeyedList][daffodil.keyedlist.KeyedList]
    objects. It reads the values of the row directly, instead of looking up each
    one through the row. `reduce()` uses it in place of `sum_da()`, so a caller
    that passes `Daf.sum_da` gets this speed too.

    A value that cannot be added, such as text or an empty cell, is skipped, with no
    error. That keeps the sum fast, and lets a sum of all columns pass over the text
    columns. But a bad number, such as `abc` in an `int` column, is then left out of
    the total with no sign. Call `check_schema()` first to find such cells.

    A column that the row does not have adds 0. With `astype` or `is_sparse`, or a
    row that is not a KeyedList, the work is passed to `sum_da()`.

    Args:
        row_kl: The current row.
        reduction_da: The running total. It is changed.
        cols: The columns to add. All if None.
        astype: A type to convert each value to before adding.
        is_sparse: If True, the row may hold only some of the columns.
        diagnose: Not used.

    Returns:
        The running total.

    Examples:
        >>> from daffodil.keyedlist import KeyedList
        >>> Daf.sum_kl(KeyedList(['x', 'y'], [1, 'abc']), {'x': 10, 'y': 0}, cols=['x', 'y'])
        {'x': 11, 'y': 0}
    """
    if row_kl.__class__ is not KeyedList or cols is None or astype is not None or is_sparse:
        return Daf.sum_da(row_kl, reduction_da, cols=cols, astype=astype, is_sparse=is_sparse)

    values_la = row_kl._values
    hd = row_kl.hd
    for col in cols:
        try:
            reduction_da[col] = reduction_da[col] + values_la[hd[col]] + 0
        except (ValueError, TypeError):
            continue
        except KeyError:
            # A column the row or the total does not have yet. Add as sum_da() does.
            value = values_la[hd[col]] if col in hd else 0
            try:
                reduction_da[col] = reduction_da.get(col, 0) + value + 0
            except (ValueError, TypeError):
                continue
    return reduction_da


def _groupsum_daf(
        self: 'Daf',
        colname:        str,
        by:             str='row',                      # determines how the func is applied.
        reduce_cols:    T_cs | None = None,             # columns included in the reduce operation.
        # keyfield: str=colname,                        # provide this for when no keyfield is desired?
                                                        # current operation sets keyfield to colname.
        ) -> 'Daf':

    """
    Group by a column and add up the other columns of each group.

    This is `groupby_reduce()` with `sum_da()`.

    Args:
        colname: The column to group by.
        by: How the function is applied. See `reduce()`.
        reduce_cols: The columns to add.

    Returns:
        The Daf with one row for each group.

    Examples:
        >>> daf = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
        >>> daf
        <BLANKLINE>
        | g  | x  | y  |
        | -: | -: | -: |
        |  a |  1 | 10 |
        |  b |  2 | 20 |
        |  a |  3 | 30 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.groupsum_daf('g', reduce_cols=['y'])
        <BLANKLINE>
        | g  | x  | y  |
        | -: | -: | -: |
        |  a |    | 40 |
        |  b |    | 20 |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield='g'; name=''
        <BLANKLINE>
    """

    result_daf = self.groupby_reduce(colname=colname, func=self.__class__.sum_da, by=by, reduce_cols=reduce_cols)

    return result_daf


def _multi_groupsum(
        self: 'Daf',
        colnames:       T_cs | None = None,             # colnames to group over individually
        by:             str         ='row',             # determines how the func is applied.
        reduce_cols:    T_cs | None = None,             # columns included in the reduce operation.
        # keyfield: str=colname,                        # provide this for when no keyfield is desired?
                                                        # current operation sets keyfield to colname.
        ) -> Dict[str, 'Daf']:

    """
    Group by each of several columns, one at a time, and add up the columns.

    This is `multi_groupby_reduce()` with `sum_da()`.

    Args:
        colnames: The columns to group by. These are required.
        by: How the function is applied. See `reduce()`.
        reduce_cols: The columns to add.

    Returns:
        A dict that maps each column to a Daf with one row for each value.

    Raises:
        ValueError: No `colnames` are given.

    Examples:
        >>> daf = Daf(lol=[['a', 'x', 1], ['a', 'y', 2], ['b', 'x', 3]], cols=['g', 'k', 'n'])
        >>> daf
        <BLANKLINE>
        | g  | k  | n  |
        | -: | -: | -: |
        |  a |  x |  1 |
        |  a |  y |  2 |
        |  b |  x |  3 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield=''; name=''
        <BLANKLINE>
        >>> sums = daf.multi_groupsum(colnames=['g', 'k'], reduce_cols=['n'])
        >>> sums['g']
        <BLANKLINE>
        | g  | k  | n  |
        | -: | -: | -: |
        |  a |    |  3 |
        |  b |    |  3 |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield='g'; name=''
        <BLANKLINE>
        >>> sums['k']
        <BLANKLINE>
        | g  | k  | n  |
        | -: | -: | -: |
        |    |  x |  4 |
        |    |  y |  2 |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield='k'; name=''
        <BLANKLINE>
        >>> daf.multi_groupsum()
        Traceback (most recent call last):
            ...
        ValueError: multi_groupsum: colnames is required
    """

    if colnames is None:
        raise ValueError("multi_groupsum: colnames is required")

    result_dodaf = self.multi_groupby_reduce(colnames=colnames, func=self.__class__.sum_da, by=by, reduce_cols=reduce_cols)

    return result_dodaf


def _set_col2_from_col1_using_regex_select(self: 'Daf', col1: str, col2: str='', regex: str='') -> 'Daf':

    r"""
    Fill a column with the part of another column that a regex selects, in place.

    `regex` must have one pair of parentheses around the part to keep. A cell that
    does not match gives an empty cell. Give `regex` as a keyword. Without `col2`,
    `col1` is changed. If `col2` is not a column, it is added at the right.

    Args:
        col1: The column to read.
        col2: The column to write. Defaults to `col1`.
        regex: A regular expression with one group.

    Returns:
        This Daf, which has been changed.

    Examples:
        >>> daf = Daf(lol=[[1, 'ab12'], [2, 'cd34']], cols=['id', 's'])
        >>> daf
        <BLANKLINE>
        | id |  s   |
        | -: | ---: |
        |  1 | ab12 |
        |  2 | cd34 |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield=''; name=''
        <BLANKLINE>
        >>> _ = daf.set_col2_from_col1_using_regex_select('s', regex=r'(\d+)')
        >>> daf.col('s')
        ['12', '34']
    """

    # from utilities import daf_utils

    def set_row_col2_from_col1_using_regex_select(row_da: T_da, col1: str, col2: str, regex: str) -> T_da:
        row_da[col2] = daf_utils.safe_regex_select(regex, row_da[col1])
        return row_da

    if not col2:
        col2 = col1

    if col2 not in self.hd and col1 in self.hd:
        self.insert_col(col2)       # a new column, so the result has a place to go.

    self.apply_in_place(lambda row_da: set_row_col2_from_col1_using_regex_select(row_da, col1, col2, regex))

    return self


def _apply_replace_regex(self: 'Daf', col: str, col2: str='', replace_regex: str='') -> 'Daf':

    r"""
    Change a column with a pattern of the form `/find/replace/`, in place.

    The pattern has three parts that are separated by `/`. The first is a regular
    expression. The second is what replaces the part it finds. The groups it
    finds can be used as `\1`. A pattern with an empty second part removes the text.

        /find//                       remove
        /find/replace/                replace
        /pre(select)post/\1/          keep only the selected part
        /pre(select)post/a\1b/        keep it, with new text around it

    The result goes to `col2`, or to `col` if there is no `col2`. A column that is
    not found does nothing. If `col2` is not a column, it is added at the right.

    Args:
        col: The column to read.
        col2: The column to write. Defaults to `col`.
        replace_regex: The pattern.

    Returns:
        This Daf, which has been changed.

    Examples:
        >>> daf = Daf(lol=[[1, 'ab12']], cols=['id', 's'])
        >>> daf
        <BLANKLINE>
        | id |  s   |
        | -: | ---: |
        |  1 | ab12 |
        <BLANKLINE>
        %% daf rows=1; cols=2; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.apply_replace_regex('s', replace_regex='/ab//').col('s')
        ['12']
    """

    # from utilities import daf_utils

    def set_row_col2_from_col_using_regex_replace(row_ma: T_ma, col: str, col2: str, replace_regex: str) -> T_ma:
        if col in row_ma:
            row_ma[col2] = daf_utils.safe_regex_replace(regex=replace_regex, s=row_ma[col])
        return row_ma

    if not col2:
        col2 = col

    if col2 not in self.hd and col in self.hd:
        self.insert_col(col2)       # a new column, so the result has a place to go.

    self.apply_in_place(lambda row_ma: set_row_col2_from_col_using_regex_replace(row_ma, col, col2, replace_regex))

    return self


def _alter_daf_per_setting(
        self: 'Daf',
        settingsdict:           T_da,
        setting_name:           str,
        setting_select_dict:    dict,
        silent_error:           bool = False,   # if True, a missing setting does nothing
        ) -> 'Daf':

    """
    Change this Daf with the replace patterns found in a settings dict.

    The setting is a dict, or a list of dicts. Each has a `spec_name`, a `colname`
    and a `replace_regex`. The ones whose fields match `setting_select_dict` are
    used. For each, `apply_replace_regex()` changes the column. Use it to give
    different edits to different files, such as fixing ids in one source.

    A setting that is None or empty changes nothing. A name that is not in the
    settings dict raises `KeyError`, unless `silent_error` is True.

    Args:
        settingsdict: A dict of settings.
        setting_name: The key of the setting in the dict.
        setting_select_dict: Selects which specs apply, such as `{'spec_name': 'file1.csv'}`.
        silent_error: If True, a missing setting changes nothing.

    Returns:
        This Daf, which has been changed.

    Raises:
        KeyError: The setting is missing and `silent_error` is False.

    Examples:
        >>> daf = Daf(lol=[['04_1'], ['05_2']], cols=['ballot_id'])
        >>> daf
        <BLANKLINE>
        | ballot_id |
        | --------: |
        |      04_1 |
        |      05_2 |
        <BLANKLINE>
        %% daf rows=2; cols=1; keyfield=''; name=''
        <BLANKLINE>
        >>> spec = {'spec_name': 'a.zip', 'colname': 'ballot_id', 'replace_regex': r'/^04_/14_/'}
        >>> daf.alter_daf_per_setting({'fix': [spec]}, 'fix', {'spec_name': 'a.zip'}).col('ballot_id')
        ['14_1', '05_2']
    """

    if silent_error:
        setting_lod = settingsdict.get(setting_name)
    else:
        setting_lod = settingsdict[setting_name]

    if not setting_lod:
        return self     # do nothing.

    if isinstance(setting_lod, dict):
        setting_lod = [setting_lod]                 # perflint-reviewed (use-tuple-over-list)

    setting_daf = Daf.from_lod(setting_lod)

    alter_specs_daf = setting_daf.select_by_dict(setting_select_dict)

    return self.alter_daf_per_alter_specs_daf(
        alter_specs_daf     = alter_specs_daf,     # daf containing cols: colname, replace_regex
        )


def _alter_daf_per_alter_specs_daf(
        self: 'Daf',                       # daf to alter
        alter_specs_daf: 'Daf',     # daf containing cols: colname, replace_regex
        ) -> 'Daf':
    """
    Change this Daf with the replace patterns listed in a Daf.

    Each row of `alter_specs_daf` has a `colname` and a `replace_regex`. The
    pattern is applied to the column with `apply_replace_regex()`. Give it only
    the rows that apply, for example by selecting on `spec_name` first.

    Args:
        alter_specs_daf: The specs, with the columns `colname` and `replace_regex`.

    Returns:
        This Daf, which has been changed.

    Examples:
        >>> daf = Daf(lol=[['04_1'], ['05_2']], cols=['ballot_id'])
        >>> daf
        <BLANKLINE>
        | ballot_id |
        | --------: |
        |      04_1 |
        |      05_2 |
        <BLANKLINE>
        %% daf rows=2; cols=1; keyfield=''; name=''
        <BLANKLINE>
        >>> specs = Daf.from_lod([{'colname': 'ballot_id', 'replace_regex': r'/^04_/14_/'}])
        >>> specs
        <BLANKLINE>
        |  colname  | replace_regex |
        | --------: | ------------: |
        | ballot_id |    /^04_/14_/ |
        <BLANKLINE>
        %% daf rows=1; cols=2; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.alter_daf_per_alter_specs_daf(specs).col('ballot_id')
        ['14_1', '05_2']
    """
    for alter_spec_da in alter_specs_daf:

        self.apply_replace_regex(col=alter_spec_da['colname'], replace_regex=alter_spec_da['replace_regex'])

    return self


def _apply_to_col(self: 'Daf', col: str, func: Callable, **kwargs: Any) -> 'Daf':

    """
    Replace each value of a column by the result of a function, in place.

    Args:
        col: The column name.
        func: A function that takes a value and returns the new value.
        **kwargs: Keyword arguments passed on to the function with each value.

    Returns:
        This Daf, which has been changed.

    Examples:
        >>> daf = Daf(lol=[[1, 5], [2, 6]], cols=['a', 'b'])
        >>> daf
        <BLANKLINE>
        | a  | b  |
        | -: | -: |
        |  1 |  5 |
        |  2 |  6 |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield=''; name=''
        <BLANKLINE>
        >>> _ = daf.apply_to_col('b', lambda value: value * 2)
        >>> daf.col('b')
        [10, 12]
        >>> _ = daf.apply_to_col('b', lambda value, factor: value * factor, factor=10)
        >>> daf.col('b')
        [100, 120]
    """

    if kwargs:
        self[:, col] = [func(value, **kwargs) for value in self.col(col)]
    else:
        self[:, col] = list(map(func, self.col(col)))

    if col == self.keyfield or not isinstance(self.keyfield, str):
        self._invalidate_kd()

    return self


def _diff_da(d1_da: T_ma, d2_da: T_ma, keys: T_ls | str | None = None) -> T_da:     # result_da
    """
    Subtract one dict from another, for the keys you name.

    This is a static method. A key that is missing, or an empty value, counts as
    0. Keys that you do not name are left out. The values must be numbers.

    Args:
        d1_da: The first dict.
        d2_da: The dict to subtract.
        keys: The key or keys to include. If None, the result is empty.

    Returns:
        A dict of the differences.

    Examples:
        >>> Daf.diff_da({'a': 5, 't': 'x'}, {'a': 2, 't': 'y'}, keys=['a'])
        {'a': 3}
    """

    keys_ca: T_ls

    if isinstance(keys, str):
        keys_ca = [keys]                      # perflint-reviewed (use-tuple-over-list)
    else:
        keys_ca = keys or []

    # the 'or 0' part handles null string. Always a plain dict comprehension,
    # regardless of whether d1_da/d2_da are dict or KeyedList.
    result_da = {key: (d1_da.get(key, 0) or 0) - (d2_da.get(key, 0) or 0)
                    for key in keys_ca}

    return result_da


def _count_values_da(
        row_da:         T_ma,
        # needs a dict-based accumulator (see test_daf_reduction.py header), but despite the
        # T_dodi ("dict of dict of int") name, values may also be a list or dict copied
        # straight from row_da (see the 'val is a list'/'val is a dict' branches below).
        reduction_da:   T_da,
        cols:           T_cs,
        *,
        omit_nulls: bool=False,

        ) -> T_da:
    """
    Add one row to running counts of the values in each column. Use it with `reduce()`.

    This is a static method. The counts are a dict that maps each column to a dict
    of value and count. The counts are changed and returned. A cell that holds a
    dict is added to the counts for that column with `sum_da()`. Use `omit_nulls`
    to skip empty cells.

    A value that cannot be a dict key, such as a list, raises TypeError. There is
    no check for it beforehand, to keep the count fast. Leave a column of lists
    out with `cols`.
    `reduce()` uses `count_values_kl()` in its place.

    Args:
        row_da: The current row.
        reduction_da: The running counts. They are changed.
        cols: The columns to count.
        omit_nulls: If True, empty cells are not counted.

    Returns:
        The running counts.

    Raises:
        TypeError: A cell holds a value that cannot be counted, such as a list.

    Examples:
        >>> Daf.count_values_da({'g': 'a'}, {}, ['g'])
        {'g': {'a': 1}}
    """

    # if cols is None:
        # cols_dict = {}
    # else:
        # cols_dict = dict.fromkeys(cols)

    if not reduction_da:
        reduction_da = {}

    result_dodi = reduction_da

    # A value that cannot be a dict key, such as a list, raises TypeError. There is no
    # check for it on the normal path. Leave such a column out with `cols`.
    try:
        for col in cols:

            val = row_da[col]

            if omit_nulls and val is NULL:
                continue

            if val and isinstance(val, dict):
                # val is a dict of values determined in another pass.
                # Copy it, since sum_da() below accumulates into result_dodi[col] in place.
                # A count that is missing or empty starts here. reduce() starts each count at 0.
                if not result_dodi.get(col):
                    result_dodi[col] = dict(val)
                else:
                    result_dodi[col] = Daf.sum_da(val, result_dodi[col])
                continue

            if col not in result_dodi or not result_dodi[col]:
                result_dodi[col] = {val: 1}
            elif val not in result_dodi[col]:
                result_dodi[col][val] = 1
            else:
                result_dodi[col][val] += 1

    except TypeError as exc_info:
        raise TypeError(_uncountable_message('count_values_da', col, val)) from exc_info

    return result_dodi


def _count_values_kl(
        row_kl:         Any,
        reduction_da:   T_da,
        cols:           T_cs,
        *,
        omit_nulls:     bool=False,
        ) -> T_da:
    """
    Add one row to running counts of the values in each column. This is the standard count for `reduce()`.

    It gives the same result as `count_values_da()`, and is faster on the rows that
    `reduce()` gives it, which are [KeyedList][daffodil.keyedlist.KeyedList]
    objects. `reduce()` uses it in place of `count_values_da()`.

    A cell that holds a dict is passed to `count_values_da()`, one column at a
    time, as is a row that is not a KeyedList. A value that cannot be a dict key,
    such as a list, raises TypeError. Leave a column of lists out with `cols`.

    Args:
        row_kl: The current row.
        reduction_da: The running counts. They are changed.
        cols: The columns to count.
        omit_nulls: If True, empty cells are not counted.

    Returns:
        The running counts.

    Raises:
        TypeError: A cell holds a value that cannot be counted, such as a list.

    Examples:
        >>> from daffodil.keyedlist import KeyedList
        >>> Daf.count_values_kl(KeyedList(['g'], ['a']), {'g': {'a': 2}}, ['g'])
        {'g': {'a': 3}}
    """
    if row_kl.__class__ is not KeyedList:
        return Daf.count_values_da(row_kl, reduction_da, cols, omit_nulls=omit_nulls)

    if not reduction_da:
        reduction_da = {}

    values_la = row_kl._values
    hd = row_kl.hd
    try:
        for col in cols:
            val = values_la[hd[col]]

            if omit_nulls and val is NULL:
                continue

            if val and isinstance(val, dict):
                reduction_da = Daf.count_values_da(row_kl, reduction_da, [col], omit_nulls=omit_nulls)
                continue

            counts_di = reduction_da.get(col)
            if not counts_di:
                reduction_da[col] = {val: 1}
            else:
                counts_di[val] = counts_di.get(val, 0) + 1

    except TypeError as exc_info:
        raise TypeError(_uncountable_message('count_values_kl', col, val)) from exc_info

    return reduction_da


def _sum_dodis(this_dodi: T_dodi, accum_dodi: T_dodi) -> None:
    """
    Add one dict of dicts of numbers into another, in place.

    This is a static method. For each key, the numbers of the inner dicts are
    added with `sum_da()`. A key that is new is stored as it is, not copied.

    Args:
        this_dodi: The counts to add.
        accum_dodi: The running totals. They are changed.

    Examples:
        >>> total = {'c': {'x': 2, 'y': 1}}
        >>> Daf.sum_dodis({'c': {'x': 1}}, total)
        >>> total
        {'c': {'x': 3, 'y': 1}}
    """

    for key, this_di in this_dodi.items():
        if key in accum_dodi:
            Daf.sum_da(this_di, accum_dodi[key])
        else:
            accum_dodi[key] = this_di


# this function does not use reduction approach.
def _sum(
        self: 'Daf',
        colnames_ls:    T_cs | None = None,    # parameter name now inconsistent.
        numeric_only:   bool = False,
        ) -> dict: # sums_di
    """
    Total the columns, and return a dict of the totals.

    Each total starts as a float. Empty cells are skipped. Text that is a number,
    such as `'2'`, is converted and added. Other text raises `ValueError`, so give
    `colnames_ls` to leave those columns out.

    `reduce(Daf.sum_da)` adds differently. It skips every value that is not a number,
    including the text `'2'`, and never raises. So the two can give different totals
    for a column that holds text. See the example. With `numeric_only`, and dtypes of `int` or `float`, only
    those columns are totaled, and a cell that is not a number counts as 0. The
    totals are converted to the dtypes, if the Daf has them.

    Args:
        colnames_ls: The columns to total. All if None.
        numeric_only: If True, total only the columns with an `int` or `float` dtype.

    Returns:
        A dict that maps each column name to its total.

    Raises:
        ValueError: A cell cannot be converted to a number.

    Examples:
        >>> Daf(lol=[[1, 10], [2, 20]], cols=['x', 'y']).sum()
        {'x': 3.0, 'y': 30.0}
        >>> mixed = Daf(cols=['n'], lol=[[1], ['2'], [''], [3.5]])
        >>> mixed
        <BLANKLINE>
        |  n  |
        | --: |
        |   1 |
        |   2 |
        |     |
        | 3.5 |
        <BLANKLINE>
        %% daf rows=4; cols=1; keyfield=''; name=''
        <BLANKLINE>
        >>> mixed.sum(['n']), mixed.reduce(Daf.sum_da, cols=['n'])
        ({'n': 6.5}, {'n': 4.5})
    """


    if colnames_ls is None:
        cleaned_colnames_cs: T_cs = self.hd.keys()
    elif not (numeric_only and self.dtypes):
        cleaned_colnames_cs = {col:None for col in colnames_ls if col in self.hd.keys()}
    else:
        cleaned_colnames_cs = {col:None for col in colnames_ls if col in self.hd and self.dtypes.get(col) in [int, float]}

    sums_d = dict.fromkeys(cleaned_colnames_cs, 0.0)

    for colname in cleaned_colnames_cs:
        colidx = self.hd[colname]
        for la in self.lol:
            val = la[colidx]
            if val:
                if numeric_only:
                    sums_d[colname] += Daf._safe_tofloat(val)    # perflint-reviewed
                else:
                    sums_d[colname] += float(val)        # perflint-reviewed

    sums_d = daf_utils.set_dict_dtypes(sums_d, dtypes=self.dtypes)

    return sums_d


def _sum_np(
        self: 'Daf',
        colnames_ls: T_cs | None = None,
        ) -> dict: # sums_di

    """
    Total the columns with NumPy, and return a dict of the totals.

    This needs NumPy. A blank, `None` or NaN cell counts as 0, as it does in `sum()`.
    Use `colnames_ls` to pass only the columns that hold numbers. A column that holds
    text raises `TypeError` and names the column.

    Args:
        colnames_ls: The columns to total. All if None.

    Returns:
        A dict that maps each column name to its total. An empty Daf gives an empty dict.

    Raises:
        KeyError: A name in `colnames_ls` is not a column.
        TypeError: A column holds text or other values that cannot be added.

    Examples:
        >>> Daf(lol=[[1, 10], [2, 20]], cols=['x', 'y']).sum_np()
        {'x': 3, 'y': 30}
        >>> Daf(lol=[[1, 10], [2, '']], cols=['x', 'y']).sum_np()
        {'x': 3, 'y': 10}
    """
    # unit tests exist

    import numpy as np

    if not self:
        return {}

    colnames_ls = self.columns() if colnames_ls is None else list(colnames_ls)

    for colname in colnames_ls:
        self.hd[colname]        # a KeyError for a name that is not a column.

    donpa = self.to_donpa(colnames_ls, default=0)

    sums_d = {}
    for colname in colnames_ls:
        try:
            sums_d[colname] = np.sum(donpa[colname]).item()
        except TypeError as exc_info:
            raise TypeError(
                f"sum_np(): column '{colname}' holds text or other values that cannot be added."
                ) from exc_info

    return sums_d
