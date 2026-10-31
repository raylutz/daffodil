# sqldaf_apply.py
# copyright (c) 2026 Ray Lutz
"""
Methods of SqlDaf for apply, reduce, group and join.

Each function here takes the table as its first argument, or the class for a classmethod.
It is assigned in the SqlDaf class body under its name, without the underscore.
The docs show it under that name.

Some functions use names defined in sqldaf.py, such as SqlDaf, when they run.
sqldaf.py sets those names here, as listed in _NAMES_FROM_SQLDAF, once they exist.
An import at the top would be circular, since sqldaf.py imports this module.
"""

from typing import Any, Callable, Dict, Generator, Iterator, List, Tuple, Type, cast, TYPE_CHECKING
from collections.abc import Iterable
from daffodil.daf import Daf, NULL
from daffodil.keyedlist import KeyedList
from daffodil.lib.daf_types import T_ls, T_la, T_lola, T_di, T_da, T_cs, T_ta
import daffodil.lib.daf_utils as daf_utils
from daffodil.lib.daf_sql import sqlite3

if TYPE_CHECKING:       # for mypy. sqldaf.py sets the names below when the program runs.
    from daffodil.sqldaf import SqlDaf, CellReduction, GROUP_INDEX_PREFIX, GroupReduction, ORD_COL, REF_COL, RowReduction, T_klosdaf, _create_table_sql, _indexes_of, is_value_sql, none_to_null, null_to_none, placeholders, quote_name, value_sql

# The names of sqldaf.py that the functions here use when they run.
# sqldaf.py sets each one here once it exists. An import at the top would be circular.
_NAMES_FROM_SQLDAF: Tuple[str, ...] = ('SqlDaf', 'CellReduction', 'GROUP_INDEX_PREFIX', 'GroupReduction', 'ORD_COL', 'REF_COL', 'RowReduction', '_create_table_sql', '_indexes_of', 'is_value_sql', 'none_to_null', 'null_to_none', 'placeholders', 'quote_name', 'value_sql')


def _register_aggregate(self: 'SqlDaf', aggregate_cls: Type[Any], nargs: int) -> str:
    """
    Register a Python aggregate class on the connection, and return the name to call it by in SQL. Internal use.

    SQLite makes one object of the class for each group.
    It calls step() once for each row of the group, and finalize() once at the end.
    An error in any of them is kept and raised as itself, as in `_register_function()`.
    """
    self._conn_da['func_count'] += 1
    func_name: str = f'daf_agg_{self._conn_da["func_count"]}'
    conn_da: T_da = self._conn_da

    class Guarded:
        """ The aggregate, with each error kept for `_execute()`. """
        def __init__(self) -> None:
            try:
                self.inner: Any = aggregate_cls()
            except BaseException as exc_info:
                conn_da['func_error'] = exc_info
                raise

        def step(self, *args: Any) -> None:
            try:
                self.inner.step(*args)
            except BaseException as exc_info:
                conn_da['func_error'] = exc_info
                raise

        def finalize(self) -> Any:
            try:
                return self.inner.finalize()
            except BaseException as exc_info:
                conn_da['func_error'] = exc_info
                raise

    self.conn.create_aggregate(func_name, nargs, Guarded)
    return func_name


def _apply(
        self: 'SqlDaf',
        func:       Callable[..., Any],         # called as func(row, **kwargs), or func(self, **kwargs) for by='table'.
        by:         str = 'row',                # 'row' for each row, 'table' for the whole SqlDaf.
        keylist:    T_la | None = None,         # not supported on a SqlDaf: select the rows first.
        **kwargs:   Any,                        # passed on to func.
        ) -> Any:
    """
    Apply a Python function to each row, and collect the results in a new table, as [`Daf.apply()`][daffodil.daf_apply._apply] does.

    The function gets each row as a dict, with Python NULL for a missing value, and returns what to
    add to the new table, by the rules of [`Daf.apply()`][daffodil.daf_apply._apply]:

    - A dict, or a KeyedList, adds one row.
    - A list of them adds a row for each, so one row can become several.
    - None, an empty dict or an empty list adds nothing, so the row is left out.

    The new table takes its columns from the first row returned. A later row is placed by
    column name. A column it lacks gets Python NULL, and a name the first row did not have is dropped.
    So have the function return every column in every row. The new table has no keyfield
    and no dtypes, as the new Daf of [`Daf.apply()`][daffodil.daf_apply._apply] has none.

    The rows are read 10,000 at a time, in order, and the results are written as they come,
    all in one transaction. So a table larger than memory can be converted, and memory does
    not grow. If the function raises, that error is raised, as itself, and nothing is kept.
    The new table is a TEMP table, as `copy()` makes. Keep it with `save(name)`.

    On 300,000 rows that each became 4, this took 23.1 s, most of it in the function. A
    registered SQL function cannot return a row, which is why the function runs in Python.

    With `by='table'` the function gets this SqlDaf, and its result is returned as it is.

    Args:
        func: The function. It takes a row and the keyword arguments, or this SqlDaf for `by='table'`.
        by: `row` to apply it to each row, or `table` to apply it to the whole SqlDaf.
        keylist: Not supported on a SqlDaf. Select the rows first.
        **kwargs: Keyword arguments passed on to the function.

    Returns:
        The new table, or the result of the function for `by='table'`.

    Raises:
        NotImplementedError: `keylist` is given, or `by` is not `row` or `table`.
        TypeError: The function returns something other than a dict, a KeyedList, a list of them, or nothing.

    Examples:
        Create a small SqlDaf table of people, each with a list of hobbies in one cell:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> people_sdaf = SqlDaf('people', cols=['id', 'name', 'hobbies'], lol=[
        ...     ['p1', 'Ana Lopez', 'chess,hiking'], ['p2', 'Ben Okafor', '']])

        Make a new table with one row for each person and hobby. The function may return several rows, or none:

        >>> def split_hobbies(row):
        ...     return [{'id': row['id'], 'hobby': hobby} for hobby in row['hobbies'].split(',') if hobby]
        >>> people_sdaf.apply(split_hobbies)
        <BLANKLINE>
        | id | hobby  |
        | -: | -----: |
        | p1 |  chess |
        | p1 | hiking |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield=''; name='daf_tmp_people_v1'
        <BLANKLINE>

        With `by='table'`, the function gets the whole SqlDaf, and its result is returned as it is:

        >>> people_sdaf.apply(lambda table: len(table), by='table')
        2
    """
    if by == 'table':
        return func(self, **kwargs)
    if by != 'row':
        raise NotImplementedError(f"apply(by={by!r}) is not supported on a SqlDaf.")
    if keylist is not None:
        raise NotImplementedError("apply(keylist=) is not supported on a SqlDaf. Select the rows first.")

    cols: T_ls = self.columns()
    target: str = self._next_temp_name(self._root().name)
    result_sdaf: SqlDaf | None = None
    out_cols: T_ls = []
    with self.transaction():
        batch_lot: List[Tuple[Any, ...]]
        for batch_lot in self._iter_row_batches():
            out_lol: T_lola = []
            vals_t: Tuple[Any, ...]
            for vals_t in batch_lot:
                result: Any = func(dict(zip(cols, none_to_null(vals_t))), **kwargs)
                if not result:
                    continue
                out_ma: Any
                for out_ma in (result if isinstance(result, (list, tuple)) else (result,)):
                    if not isinstance(out_ma, (dict, KeyedList)):
                        raise TypeError(f"apply: the function returned a {type(out_ma).__name__}. "
                                        "It must return a dict, a KeyedList, a list of them, or None.")
                    if result_sdaf is None:
                        out_cols = cast(T_ls, list(out_ma.keys()))
                        self._execute(_create_table_sql(self.conn, target, out_cols, 'temp'))
                        result_sdaf = self._new_on_conn(target, '', None, is_temp=True)
                    out_lol.append([out_ma.get(col, NULL) for col in out_cols])
            if out_lol and result_sdaf is not None:
                result_sdaf._insert_rows(out_lol)
        if result_sdaf is None:                             # no row returned anything: an empty table.
            self._execute(_create_table_sql(self.conn, target, [], 'temp'))
            result_sdaf = self._new_on_conn(target, '', None, is_temp=True)
    return result_sdaf


def _reduce(
        self: 'SqlDaf',
        func:           Callable[..., Any],         # combines a row with the result so far.
        by:             str = 'row',                # 'row', 'sparse_row' or 'table'.
        cols:           T_cs | None = None,         # the columns included. All if None.
        initial_da:     T_da | None = None,         # the result to start from.
        indirect_col:   str = '',                   # the column that holds a dict, for 'sparse_row'.
        silent_error:   bool = False,               # if True, skip rows where func raises.
        **kwargs:       Any,                        # passed on to func.
        ) -> Any:
    """
    Combine all the rows into one result, using a function, as [`Daf.reduce()`][daffodil.daf_apply._reduce] does.

    The function is called in Python for each row, with the same arguments and the same
    result as in [`Daf.reduce()`][daffodil.daf_apply._reduce]. A sum or a count is a reduction, as with `Daf.sum_da`.
    The rows are read 10,000 at a time, in order, so the table is never all in memory.
    Memory use still depends on the result the function builds.

    - With `by='row'`, each row is a [KeyedList][daffodil.keyedlist.KeyedList] of its
      values. Unlike in a Daf, writing to it does not change the table.
    - With `indirect_col`, or `by='sparse_row'`, the function gets the dict held in `indirect_col`, JSON or
      PYON text decoded as in a Daf, and the result starts as `initial_da` or an empty dict.
      A plain sum chooses its loop once, from the first 100 rows, as [`Daf.reduce()`][daffodil.daf_apply._reduce] describes.
      With `cols`, the result is in the order of `cols`.
    - With `by='table'`, the function gets this SqlDaf and `cols`.

    For a plain total of columns, `sum()` is faster: SQLite adds them, and no row comes
    into Python. `by='col'` is not supported, since it needs each whole column in memory.

    Args:
        func: The reduction function, such as `Daf.sum_da`.
        by: `row`, `sparse_row` or `table`.
        cols: The columns included in the reduction. All columns if None. With many columns, a KeysView or a dict is faster than a list.
        initial_da: The result to start from, instead of zeros.
        indirect_col: A column that holds a dict. Its dicts are the rows, as with `by='sparse_row'`.
        silent_error: If True, a row for which the function raises is skipped.
        **kwargs: Keyword arguments passed on to the function.

    Returns:
        A dict for `row` and `sparse_row`. The function's result for `table`.

    Raises:
        ValueError: `sparse_row` is used with no `indirect_col`.
        NotImplementedError: `by` is `col`, or not recognized.

    Examples:
        Create a small SqlDaf table of customers, with a count of visits and a cart that maps each item to how many:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> carts_sdaf = SqlDaf('carts', cols=['id', 'visits', 'cart'], lol=[
        ...     ['c1', 4, {'apples': 3, 'pears': 2}], ['c2', 7, {'apples': 5, 'plums': 6}]])
        >>> carts_sdaf
        <BLANKLINE>
        | id | visits |           cart            |
        | -: | -----: | ------------------------: |
        | c1 |      4 | {'apples': 3, 'pears': 2} |
        | c2 |      7 | {'apples': 5, 'plums': 6} |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield=''; name='carts'
        <BLANKLINE>

        Total the visits:

        >>> carts_sdaf.reduce(Daf.sum_da, cols=['visits'])
        {'id': '', 'visits': 11, 'cart': ''}

        Total the items in all the carts. Each cart holds only the items bought, so it is a sparse row:

        >>> carts_sdaf.reduce(Daf.sum_da, indirect_col='cart')
        {'apples': 8, 'pears': 2, 'plums': 6}

        Start from a total other than zero, with `initial_da`:

        >>> carts_sdaf.reduce(Daf.sum_da, cols=['visits'], initial_da={'visits': 100})
        {'id': '', 'visits': 111, 'cart': ''}

        A function of your own that fails on a row stops the reduction, unless `silent_error=True`, which skips that row:

        >>> mixed_sdaf = SqlDaf('mixed', cols=['id', 'visits'], lol=[['c1', 4], ['c2', 'unknown'], ['c3', 7]])
        >>> def add_visits(row, result, cols=None):
        ...     result['visits'] = result['visits'] + row['visits']
        ...     return result
        >>> mixed_sdaf.reduce(add_visits, cols=['visits'], silent_error=True)
        {'id': '', 'visits': 11}
    """
    # indirect_col names the column that holds each row's dict, so it means sparse rows, as in Daf.reduce().
    if indirect_col and by == 'row':
        by = 'sparse_row'

    if by == 'table':
        return func(self, cols, **kwargs)
    if by not in ('row', 'sparse_row'):
        raise NotImplementedError(f"reduce(by={by!r}) is not supported on a SqlDaf.")
    if by == 'sparse_row' and not indirect_col:
        raise ValueError("indirect_col must be specified for 'sparse_row' reduction.")
    if by == 'row' and not len(self):
        return {}

    hd: T_di = dict(self.hd)

    # A plain sparse sum picks its loop once, before the first row, as in Daf.reduce().
    if by == 'sparse_row' and Daf._is_plain_sum(func, initial_da, silent_error, kwargs):
        sparse_idx: int = hd[indirect_col]
        rows_iter: Iterator[Any] = (daf_utils.get_indirect_da({indirect_col: none_to_null(vals_t)[sparse_idx]}, indirect_col)
                                    for batch_lot in self._iter_row_batches() for vals_t in batch_lot)
        return Daf._sum_sparse(rows_iter, cols)

    cols_iter: Any = None
    reduction_ma: Any
    if by == 'row':
        cols_iter = hd.keys() if cols is None else {col: None for col in hd if col in cols}
        reduction_ma = initial_da if initial_da is not None else dict.fromkeys(cols_iter, 0)
        if func is Daf.sum_da:                      # as in Daf.reduce(): the same result, faster.
            func = Daf.sum_kl
        elif func is Daf.count_values_da:
            func = Daf.count_values_kl
    else:
        reduction_ma = initial_da or {}
    indirect_idx: int = hd[indirect_col] if by == 'sparse_row' else 0

    batch_lot: List[Tuple[Any, ...]]
    vals_t: Tuple[Any, ...]
    for batch_lot in self._iter_row_batches():
        for vals_t in batch_lot:
            try:
                if by == 'row':
                    reduction_ma = func(KeyedList(hd, none_to_null(vals_t)), reduction_ma, cols=cols_iter, **kwargs)
                else:
                    indirect_ma: Any = daf_utils.get_indirect_da({indirect_col: none_to_null(vals_t)[indirect_idx]}, indirect_col)
                    reduction_ma = func(indirect_ma, reduction_ma, cols=cols, is_sparse=True, **kwargs)
            except Exception:
                if not silent_error:
                    raise
    if by == 'row':
        return {key: reduction_ma.get(key, '') for key in hd}
    return reduction_ma


def _value_counts_lt(self: 'SqlDaf', colnames: T_ls, where_sql: str = '', params: Iterable[Any] = ()) -> List[Tuple[Any, ...]]:
    """
    Count each combination of values of some columns, in one GROUP BY. Internal use.

    Gives (value, ..., count) tuples, with SQL NULL as Python NULL, in the order each combination is
    first seen, as a dict in a Daf fills.
    """
    self._resolve_parent()
    rows_sql, rows_params = self._rows_sql()
    group_sql: str = ', '.join(value_sql(col) for col in colnames)          # '' and SQL NULL count as one value.
    where_part: str = f' WHERE {where_sql}' if where_sql else ''
    cursor: sqlite3.Cursor = self._execute(
            f'SELECT {group_sql}, COUNT(*) FROM ({rows_sql}){where_part} GROUP BY {group_sql} ORDER BY MIN({ORD_COL})',
            rows_params + null_to_none(params))
    return [tuple(none_to_null(row_t)) for row_t in self._fetch(cursor)]


def _valuecounts_for_colname(
        self: 'SqlDaf',
        colname:    str,                            # the column to count.
        sort:       bool = False,                   # if True, order by count.
        reverse:    bool = True,                    # with sort, True puts the most common first.
        omit_nulls: bool = False,                   # if True, leave out the count of Python NULL.
        ) -> Dict[Any, int]:
    """
    Count how often each value occurs in one column, as [`Daf.valuecounts_for_colname()`][daffodil.daf_report._valuecounts_for_colname] does.

    One GROUP BY counts them in SQLite, so no column is read into Python. The values are in
    the order first seen, or by count with `sort`. Python NULL is counted as Python NULL, unless
    `omit_nulls`. A column that does not exist gives an empty dict.

    Args:
        colname: The column to count.
        sort: If True, order by count.
        reverse: With `sort`, True puts the most common first.
        omit_nulls: If True, leave out the count of Python NULL.

    Returns:
        A dict that maps each value to its count.

    Examples:
        Create a small SqlDaf table of people. One city is not known:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> people_sdaf = SqlDaf('people', cols=['id', 'city'], lol=[
        ...     ['p1', 'Denver'], ['p2', 'Austin'], ['p3', 'Denver'], ['p4', ''], ['p5', 'Denver']])

        Count the people in each city, the most common first:

        >>> people_sdaf.valuecounts_for_colname('city', sort=True)
        {'Denver': 3, 'Austin': 1, '': 1}

        Count them with the least common first:

        >>> people_sdaf.valuecounts_for_colname('city', sort=True, reverse=False)
        {'Austin': 1, '': 1, 'Denver': 3}

        Leave out the unknown city:

        >>> people_sdaf.valuecounts_for_colname('city', omit_nulls=True)
        {'Denver': 3, 'Austin': 1}
    """
    if colname not in self.hd:
        return {}
    valuecounts_di: T_di = {val: count for val, count in self._value_counts_lt([colname])}
    if omit_nulls:
        valuecounts_di.pop(NULL, None)
    if sort:
        valuecounts_di = dict(sorted(valuecounts_di.items(), key=lambda item: item[1], reverse=reverse))
    return valuecounts_di


def _valuecounts_for_colnames_ls(
        self: 'SqlDaf',
        colnames_ls:    T_ls | None = None,         # the columns to count. All if None.
        sort:           bool = False,
        reverse:        bool = True,
        omit_nulls:     bool = False,
        ) -> Dict[str, Dict[Any, int]]:
    """
    Count how often each value occurs, in each of several columns, as [`Daf.valuecounts_for_colnames_ls()`][daffodil.daf_report._valuecounts_for_colnames_ls] does.

    It is one GROUP BY for each column. See `valuecounts_for_colname()`.

    Args:
        colnames_ls: The columns to count. All if None.
        sort: If True, order each count by size.
        reverse: With `sort`, True puts the most common first.
        omit_nulls: If True, leave out the count of Python NULL.

    Returns:
        A dict that maps each column to a dict of value and count.

    Examples:
        Create a small SqlDaf table of customers, with a city and a plan. One plan is not known:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> customers_sdaf = SqlDaf('customers', cols=['id', 'city', 'plan'], lol=[
        ...     ['c1', 'Denver', 'basic'], ['c2', 'Austin', 'pro'], ['c3', 'Denver', 'pro'], ['c4', 'Denver', 'basic'], ['c5', 'Denver', '']])

        Count the values of two columns:

        >>> customers_sdaf.valuecounts_for_colnames_ls(['city', 'plan'])
        {'city': {'Denver': 4, 'Austin': 1}, 'plan': {'basic': 2, 'pro': 2, '': 1}}

        Count them least common first, leaving out the unknown plan:

        >>> customers_sdaf.valuecounts_for_colnames_ls(['city', 'plan'], sort=True, reverse=False, omit_nulls=True)
        {'city': {'Austin': 1, 'Denver': 4}, 'plan': {'basic': 2, 'pro': 2}}
    """
    return {colname: self.valuecounts_for_colname(colname, sort=sort, reverse=reverse, omit_nulls=omit_nulls)
            for colname in (colnames_ls or self.columns())}


def _valuecounts_for_colname_selectedby_colname(
        self: 'SqlDaf',
        colname:                str,                # the column to count.
        selectedby_colname:     str,                # the column to test.
        selectedby_colvalue:    Any,                # only rows where that column has this value are counted.
        sort:                   bool = False,
        reverse:                bool = True,
        ) -> Dict[Any, int]:
    """
    Count the values of a column, in the rows where another column has a value, as [`Daf.valuecounts_for_colname_selectedby_colname()`][daffodil.daf_report._valuecounts_for_colname_selectedby_colname] does.

    Args:
        colname: The column to count.
        selectedby_colname: The column to test.
        selectedby_colvalue: Only rows where that column has this value are counted. Python NULL matches Python NULL.
        sort: If True, order the counts by size.
        reverse: With `sort`, True puts the most common first.

    Returns:
        A dict that maps each value to its count. It is empty if a column does not exist.

    Examples:
        Create a small SqlDaf table of customers, with a city and a plan:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> customers_sdaf = SqlDaf('customers', cols=['id', 'city', 'plan'], lol=[
        ...     ['c1', 'Denver', 'basic'], ['c2', 'Austin', 'pro'], ['c3', 'Denver', 'pro'], ['c4', 'Denver', 'basic'], ['c5', 'Denver', 'basic']])
        >>> customers_sdaf
        <BLANKLINE>
        | id |  city  | plan  |
        | -: | -----: | ----: |
        | c1 | Denver | basic |
        | c2 | Austin |   pro |
        | c3 | Denver |   pro |
        | c4 | Denver | basic |
        | c5 | Denver | basic |
        <BLANKLINE>
        %% daf rows=5; cols=3; keyfield=''; name='customers'
        <BLANKLINE>

        Count the plans of the customers in Denver only:

        >>> customers_sdaf.valuecounts_for_colname_selectedby_colname('plan', 'city', 'Denver')
        {'basic': 3, 'pro': 1}

        Count them least common first:

        >>> customers_sdaf.valuecounts_for_colname_selectedby_colname('plan', 'city', 'Denver', sort=True, reverse=False)
        {'pro': 1, 'basic': 3}
    """
    if colname not in self.hd or selectedby_colname not in self.hd:
        return {}
    cond_sql, cond_params = is_value_sql(selectedby_colname, selectedby_colvalue)
    valuecounts_di: T_di = {val: count for val, count in self._value_counts_lt([colname], cond_sql, cond_params)}
    if sort:
        valuecounts_di = dict(sorted(valuecounts_di.items(), key=lambda item: item[1], reverse=reverse))
    return valuecounts_di


def _valuecounts_for_colnames_ls_selectedby_colname(
        self: 'SqlDaf',
        colnames_ls:            T_ls | None = None, # the columns to count. All if None.
        selectedby_colname:     str = '',           # the column to test.
        selectedby_colvalue:    Any = '',           # only rows where that column has this value are counted.
        sort:                   bool = False,
        reverse:                bool = True,
        ) -> Dict[str, Dict[Any, int]]:
    """
    Count the values of several columns, in the rows where another column has a value, as [`Daf.valuecounts_for_colnames_ls_selectedby_colname()`][daffodil.daf_report._valuecounts_for_colnames_ls_selectedby_colname] does.

    Args:
        colnames_ls: The columns to count. All if None.
        selectedby_colname: The column to test.
        selectedby_colvalue: Only rows where that column has this value are counted.
        sort: If True, order the counts by size.
        reverse: With `sort`, True puts the most common first.

    Returns:
        A dict that maps each column to a dict of value and count.

    Examples:
        Create a small SqlDaf table of customers, with a city, a plan and a size of company:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> customers_sdaf = SqlDaf('customers', cols=['id', 'city', 'plan', 'size'], lol=[
        ...     ['c1', 'Denver', 'basic', 'small'], ['c2', 'Austin', 'pro', 'large'], ['c3', 'Denver', 'pro', 'large'],
        ...     ['c4', 'Denver', 'basic', 'small'], ['c5', 'Denver', 'basic', 'large']])

        Count the plans and sizes of the customers in Denver only:

        >>> customers_sdaf.valuecounts_for_colnames_ls_selectedby_colname(['plan', 'size'], 'city', 'Denver')
        {'plan': {'basic': 3, 'pro': 1}, 'size': {'small': 2, 'large': 2}}

        Count them least common first:

        >>> customers_sdaf.valuecounts_for_colnames_ls_selectedby_colname(['plan', 'size'], 'city', 'Denver', sort=True, reverse=False)
        {'plan': {'pro': 1, 'basic': 3}, 'size': {'small': 2, 'large': 2}}
    """
    return {colname: self.valuecounts_for_colname_selectedby_colname(
                colname, selectedby_colname, selectedby_colvalue, sort=sort, reverse=reverse)
            for colname in (colnames_ls or self.columns())}


def _valuecounts_for_colname1_groupedby_colname2(
        self: 'SqlDaf',
        colname1:           str,                    # the column whose values are counted.
        groupedby_colname2: str,                    # the column whose values form the groups.
        sort:               bool = False,
        reverse:            bool = True,
        ) -> Dict[Any, Dict[Any, int]]:
    """
    Count the values of one column, for each value of another, as the Daf method does.

    One GROUP BY of the two columns counts them all. The groups, and the values in each, are
    in the order first seen. Use it to see whether two columns relate one to one: each group
    should then hold a single value.

    Args:
        colname1: The column whose values are counted.
        groupedby_colname2: The column whose values form the groups.
        sort: If True, order each count by size.
        reverse: With `sort`, True puts the most common first.

    Returns:
        A dict that maps each value of the second column to a dict of value and count.

    Examples:
        Create a small SqlDaf table of customers, with a city and a plan:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> customers_sdaf = SqlDaf('customers', cols=['city', 'plan'], lol=[
        ...     ['Denver', 'pro'], ['Denver', 'basic'], ['Austin', 'basic'], ['Denver', 'basic']])

        Count the plans in each city:

        >>> customers_sdaf.valuecounts_for_colname1_groupedby_colname2('plan', 'city')
        {'Denver': {'pro': 1, 'basic': 2}, 'Austin': {'basic': 1}}

        Sort each count, the most common first, or the least common first:

        >>> customers_sdaf.valuecounts_for_colname1_groupedby_colname2('plan', 'city', sort=True)
        {'Denver': {'basic': 2, 'pro': 1}, 'Austin': {'basic': 1}}
        >>> customers_sdaf.valuecounts_for_colname1_groupedby_colname2('plan', 'city', sort=True, reverse=False)
        {'Denver': {'pro': 1, 'basic': 2}, 'Austin': {'basic': 1}}
    """
    if colname1 not in self.hd or groupedby_colname2 not in self.hd:
        return {}
    valuecounts_dodi: Dict[Any, T_di] = {}
    for groupval, val, count in self._value_counts_lt([groupedby_colname2, colname1]):
        valuecounts_dodi.setdefault(groupval, {})[val] = count
    if sort:
        valuecounts_dodi = {groupval: dict(sorted(valuecounts_di.items(), key=lambda item: item[1], reverse=reverse))
                            for groupval, valuecounts_di in valuecounts_dodi.items()}
    return valuecounts_dodi


def _value_counts_daf(
        self: 'SqlDaf',
        colname:        str,                        # the column to count.
        sort:           bool = False,
        reverse:        bool = True,
        include_total:  bool = False,               # if True, add a row with the total.
        omit_nulls:     bool = False,
        ) -> Daf:
    """
    Count the values of a column into a Daf in memory, with one row per value, as [`Daf.value_counts_daf()`][daffodil.daf_report._value_counts_daf] does.

    The counts come from `valuecounts_for_colname()`. The Daf has the columns `colname` and
    counts, and a row for each value.

    Args:
        colname: The column to count.
        sort: If True, order by count.
        reverse: With `sort`, True puts the most common first.
        include_total: If True, add a row with the total.
        omit_nulls: If True, leave out the count of Python NULL.

    Returns:
        A Daf in memory, with a row for each value.

    Examples:
        Create a small SqlDaf table of customers. One plan is not known:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> customers_sdaf = SqlDaf('customers', cols=['id', 'plan'], lol=[
        ...     ['c1', 'basic'], ['c2', 'pro'], ['c3', 'basic'], ['c4', 'team'], ['c5', 'basic'], ['c6', '']])

        Count the plans into a Daf, one row for each:

        >>> customers_sdaf.value_counts_daf('plan')
        <BLANKLINE>
        | plan  | counts |
        | ----: | -----: |
        | basic |      3 |
        |   pro |      1 |
        |  team |      1 |
        |       |      1 |
        <BLANKLINE>
        %% daf rows=4; cols=2; keyfield=''; name=''
        <BLANKLINE>

        Sort it, leave out the unknown plan, and add a total:

        >>> customers_sdaf.value_counts_daf('plan', sort=True, include_total=True, omit_nulls=True)
        <BLANKLINE>
        |   plan    | counts |
        | --------: | -----: |
        |     basic |      3 |
        |       pro |      1 |
        |      team |      1 |
        | **Total** |      5 |
        <BLANKLINE>
        %% daf rows=4; cols=2; keyfield=''; name=''
        <BLANKLINE>

        Sort it with the least common first:

        >>> customers_sdaf.value_counts_daf('plan', sort=True, reverse=False, omit_nulls=True)
        <BLANKLINE>
        | plan  | counts |
        | ----: | -----: |
        |   pro |      1 |
        |  team |      1 |
        | basic |      3 |
        <BLANKLINE>
        %% daf rows=3; cols=2; keyfield=''; name=''
        <BLANKLINE>
    """
    value_counts_daf: Daf = Daf.from_lod_to_cols(
            [self.valuecounts_for_colname(colname, sort=sort, reverse=reverse, omit_nulls=omit_nulls)],
            cols=[colname, 'counts'])
    if include_total:
        value_counts_daf.append({colname: ' **Total** ', 'counts': sum(value_counts_daf.col('counts'))})
    return value_counts_daf


def _daf_valuecount(
        self: 'SqlDaf',
        by:     str = 'row',                        # only 'row', as the default of Daf.
        cols:   T_ls | None = None,                 # the columns to count. All if None.
        ) -> T_da:
    """
    Count how often each value occurs, in each column, as [`Daf.daf_valuecount()`][daffodil.daf_report._daf_valuecount] does.

    It gives the same result: a column that is not in `cols` gets Python NULL in place of its
    counts, and a table with no rows gives an empty dict. Each column is one GROUP BY.

    Args:
        by: Only `row` is supported.
        cols: The columns to count. All if None.

    Returns:
        A dict that maps each column to a dict of value and count.

    Raises:
        NotImplementedError: `by` is not `row`.

    Examples:
        Create a small SqlDaf table of customers:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> customers_sdaf = SqlDaf('customers', cols=['id', 'city', 'plan'], lol=[
        ...     ['c1', 'Denver', 'basic'], ['c2', 'Austin', 'pro'], ['c3', 'Denver', 'basic']])

        Count the values of two columns. A column not counted maps to Python NULL:

        >>> customers_sdaf.daf_valuecount(cols=['city', 'plan'])
        {'id': '', 'city': {'Denver': 2, 'Austin': 1}, 'plan': {'basic': 2, 'pro': 1}}
    """
    if by != 'row':
        raise NotImplementedError(f"daf_valuecount(by={by!r}) is not supported on a SqlDaf.")
    if not self:
        return {}
    count_cols_set: set = set(cols if cols is not None else self.columns())
    return {colname: (self.valuecounts_for_colname(colname) if colname in count_cols_set else NULL)
            for colname in self.columns()}


def _sum(
        self: 'SqlDaf',
        colnames_ls:    T_ls | None = None,         # the columns to total. All if None.
        ) -> T_da:
    """
    Total the columns, as [`Daf.sum()`][daffodil.daf_apply._sum] does, and return a dict of the totals.

    Numbers are totalled by SQLite. A column that also holds text has its text cells read
    into Python and added as [`Daf.sum()`][daffodil.daf_apply._sum] adds them: text that is not a number raises
    ValueError. Python NULL is skipped. Each total starts as a
    float, and is converted to the dtype of its column if there is one.

    Args:
        colnames_ls: The columns to total. All if None. Name only the columns that hold numbers.

    Returns:
        A dict that maps each column name to its total.

    Raises:
        ValueError: A text cell cannot be converted to a number.

    Examples:
        Create a small SqlDaf table of orders:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> orders_sdaf = SqlDaf('orders', cols=['qty', 'amount'], lol=[[2, 25.0], [6, 24.0], [1, 30.5]])

        Total every column:

        >>> orders_sdaf.sum()
        {'qty': 9.0, 'amount': 79.5}

        Total only some columns. A table with a text column needs them named:

        >>> named_sdaf = SqlDaf('named', cols=['id', 'qty', 'amount'], lol=[['o1', 2, 25.0], ['o2', 6, 24.0]],
        ...                     dtypes={'id': str, 'qty': int, 'amount': float})
        >>> named_sdaf.sum(['qty'])
        {'qty': 8}
    """
    sum_cols_ls: T_ls
    if colnames_ls is None:
        sum_cols_ls = self.columns()
    else:
        sum_cols_ls = list(dict.fromkeys(col for col in colnames_ls if col in self.hd))
    sums_d: T_da = dict.fromkeys(sum_cols_ls, 0.0)
    if not sum_cols_ls:
        return sums_d

    self._resolve_parent()
    rows_sql, rows_params = self._rows_sql()
    totals_sql: str = ', '.join(
            f"TOTAL(CASE WHEN typeof({quote_name(col)}) IN ('integer', 'real') THEN {quote_name(col)} END), "
            f"SUM(typeof({quote_name(col)}) = 'text')" for col in sum_cols_ls)
    totals_t: Tuple[Any, ...] = self._fetch(self._execute(f'SELECT {totals_sql} FROM ({rows_sql})', rows_params))[0]
    icol: int
    col: str
    for icol, col in enumerate(sum_cols_ls):
        sums_d[col] = float(totals_t[2 * icol])
        if totals_t[2 * icol + 1]:                      # text cells: add them as Daf.sum() does.
            text_lot: List[Tuple[Any, ...]] = self._fetch(self._execute(
                    f"SELECT {quote_name(col)} FROM ({rows_sql}) WHERE typeof({quote_name(col)}) = 'text'", rows_params))
            sums_d[col] += sum(float(text_t[0]) for text_t in text_lot if text_t[0])
    return daf_utils.set_dict_dtypes(sums_d, dtypes=self.dtypes)


def _groupsum_daf(
        self: 'SqlDaf',
        colname:        str,                        # the column to group by.
        by:             str = 'row',                # 'row', or 'sparse_row' with indirect_col.
        reduce_cols:    T_ls | None = None,         # the columns to add. All but colname if None.
        indirect_col:   str = '',                   # a column that holds a dict, for sparse rows.
        ) -> Daf:
    """
    Group by a column and add up the other columns of each group, as [`Daf.groupsum_daf()`][daffodil.daf_apply._groupsum_daf] does.

    One GROUP BY does it all in SQLite. The result is a Daf in memory, with one row for each
    value, in the order first seen, and `colname` as its keyfield. As in a
    Daf, only numbers are added: text, even text of digits, counts as 0, and a column with
    no numbers in a group totals 0. A column not in `reduce_cols` is Python NULL.

    Empty cells do not change the type of a total. SQL NULL and the text '' are skipped,
    so a column of integers with empty cells totals as an int, as in a Daf.
    A total is a float only when the group holds a float.
    A plain SQL `SUM()` would count '' as 0 and make the total a float, which is why it is not used.

    Args:
        colname: The column to group by.
        by: `row`, or `sparse_row` for the dict in `indirect_col`. Giving `indirect_col` is enough.
        reduce_cols: The columns to add. All but `colname` if None.
        indirect_col: A column that holds a dict. Each group's dicts are added into one, in that column, as JSON text.

    Returns:
        A Daf in memory, with one row for each group.

    Raises:
        KeyError: `colname` is not a column.
        ValueError: `by` is `sparse_row`, and there is no `indirect_col`.
        NotImplementedError: `by` is not `row` or `sparse_row`.

    Examples:
        Create a small SqlDaf table of orders, with the city they were shipped to:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> orders_sdaf = SqlDaf('orders', cols=['city', 'qty', 'amount'], lol=[
        ...     ['Denver', 2, 25.0], ['Austin', 6, 24.0], ['Denver', 1, 30.5]])

        Total the amount for each city. A column not totaled is Python NULL:

        >>> orders_sdaf.groupsum_daf('city', reduce_cols=['amount'])
        <BLANKLINE>
        |  city  | qty | amount |
        | -----: | --: | -----: |
        | Denver |     |   55.5 |
        | Austin |     |   24.0 |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield='city'; name=''
        <BLANKLINE>

        Total the items in the carts of each city, where each cart is a dict of item to count:

        >>> carts_sdaf = SqlDaf('carts', cols=['id', 'city', 'cart'], lol=[
        ...     ['c1', 'Denver', {'apples': 3}], ['c2', 'Denver', {'apples': 5, 'pears': 2}], ['c3', 'Austin', {'pears': 4}]])
        >>> carts_sdaf.groupsum_daf('city', indirect_col='cart')
        <BLANKLINE>
        |  city  |          cart          |
        | -----: | ---------------------: |
        | Austin |            {"pears":4} |
        | Denver | {"apples":8,"pears":2} |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield='city'; name='daf_tmp_carts_v1'
        <BLANKLINE>
    """
    # Sparse rows are summed by groupby_reduce(), which keeps each group's dict in one cell.
    if indirect_col or by == 'sparse_row':
        if not indirect_col:
            raise ValueError("groupsum_daf(by='sparse_row') needs indirect_col.")
        return self.groupby_reduce(colname, Daf.sum_da, reduce_cols=reduce_cols, indirect_col=indirect_col).to_daf()
    if by != 'row':
        raise NotImplementedError(f"groupsum_daf(by={by!r}) is not supported on a SqlDaf.")
    if colname not in self.hd:
        raise KeyError(colname)
    sum_cols_set: set = set(reduce_cols if reduce_cols is not None else self.columns()) - {colname}
    self._resolve_parent()
    rows_sql, rows_params = self._rows_sql()
    select_ls: T_ls = []
    col: str
    for col in self.columns():
        if col == colname:
            select_ls.append(value_sql(col))
        elif col in sum_cols_set:
            select_ls.append(f"SUM(CASE WHEN typeof({quote_name(col)}) IN ('integer', 'real') THEN {quote_name(col)} ELSE 0 END)")
        else:
            select_ls.append('NULL')
    rows_lot: List[Tuple[Any, ...]] = self._fetch(self._execute(
            f'SELECT {", ".join(select_ls)} FROM ({rows_sql}) GROUP BY {value_sql(colname)} ORDER BY MIN({ORD_COL})',
            rows_params))
    return Daf(lol=[none_to_null(row_t) for row_t in rows_lot], cols=self.columns(), keyfield=colname)


def _groupby(
        self: 'SqlDaf',
        colname:        str = '',                   # the column to group by.
        colnames:       T_ls | None = None,         # several columns to group by, as groupby_cols().
        omit_nulls:     bool = False,               # if True, leave out rows with Python NULL in the column.
        cols:           T_ls | None = None,         # not supported on a SqlDaf.
        ) -> KeyedList:
    """
    Group the rows by the values of a column, as [`Daf.groupby()`][daffodil.daf_apply._groupby] does.

    The result is a KeyedList of selections, one for each value, in the order first seen. It
    works as a dict, as the result of [`Daf.groupby()`][daffodil.daf_apply._groupby] does. One query finds the values. Each
    group is then `select_by_dict()` of its value, so Python NULL is a value like any other. Nothing
    is read until a group is read.

    The first groupby on a column makes an index on it, in the parent table, unless an index
    already starts with it. Each group is then a lookup. Without the index, reading each
    group would need a full scan of the table. On 100,000 rows in 500 groups, summing every group took 2.3 s, against
    14.8 s without it. The index stays with the table, and `drop_cols()` or `rename_cols()`
    of the column drops it.

    Args:
        colname: The column to group by.
        colnames: Several columns to group by. See `groupby_cols()`.
        omit_nulls: If True, leave out the rows with Python NULL in the column.
        cols: Not supported on a SqlDaf. It raises NotImplementedError.

    Returns:
        A KeyedList that maps each value to a selection.

    Raises:
        KeyError: The column is not a column.
        NotImplementedError: `cols` is given.

    Examples:
        Create a small SqlDaf table of people. One city is not known:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> people_sdaf = SqlDaf('people', cols=['id', 'city', 'plan'], keyfield='id', lol=[
        ...     ['p1', 'Denver', 'basic'], ['p2', 'Austin', 'pro'], ['p3', 'Denver', 'pro'], ['p4', '', 'basic']])

        Group the people by city. Each group is a selection:

        >>> groups = people_sdaf.groupby('city')
        >>> {city: group_sdaf.keys() for city, group_sdaf in groups.items()}
        {'Denver': ['p1', 'p3'], 'Austin': ['p2'], '': ['p4']}

        Group by two columns together, as tuples:

        >>> list(people_sdaf.groupby(colnames=['city', 'plan']).keys())
        [('Denver', 'basic'), ('Austin', 'pro'), ('Denver', 'pro'), ('', 'basic')]

        Leave out the people with no city, with `omit_nulls=True`:

        >>> list(people_sdaf.groupby('city', omit_nulls=True).keys())
        ['Denver', 'Austin']
    """
    if isinstance(colname, list) and not colnames:
        return self.groupby_cols(colname, cols=cols)
    if colnames and not colname:
        if len(colnames) > 1:
            return self.groupby_cols(colnames, cols=cols)
        colname = colnames[0]
    groups_kl: KeyedList = self.groupby_cols([colname], cols=cols, omit_nulls=omit_nulls)
    return KeyedList([cast(T_ta, key_t)[0] for key_t in groups_kl.keys()], groups_kl.values())


def _groupby_reduce(
        self: 'SqlDaf',
        colname:        str,                        # the column to group by.
        func:           Callable[..., Any],         # reduces the rows of one group, as in Daf.reduce().
        by:             str = 'row',                # 'row' or 'sparse_row', as in Daf.reduce().
        reduce_cols:    T_ls | None = None,         # the columns to reduce. All except colname if None.
        diagnose:       bool = False,               # if True, show the query and the first results.
        batch_size:     int = 10000,                # not used. Kept so that existing calls work.
        **kwargs:       Any,                        # passed on to func, and indirect_col for 'sparse_row'.
        ) -> 'SqlDaf':
    """
    Group the rows by one column and reduce each group to one row, as [`Daf.groupby_reduce()`][daffodil.daf_apply._groupby_reduce] does.

    Use it to turn many rows of one key into one row, such as the order lines of an order
    into one record of the order.

    The groups are reduced inside SQLite, in one query, by a registered aggregate,
    as `groupby_aggregate()` does. No query is made for each group.

    - With `by='row'`, the rows of each group become a Daf of the columns the group holds.
      These are `reduce_cols`, or all columns. The cells are as stored, so a dict cell is text.
      [`Daf.reduce()`][daffodil.daf_apply._reduce] reduces it, so the function is called just as [`Daf.groupby_reduce()`][daffodil.daf_apply._groupby_reduce] calls it.
      A plain `Daf.sum_da` adds down each column there, with no call for each row.
    - With `by='sparse_row'`, the function gets the dict held in the cell of `indirect_col`.
      Its `cols` is `reduce_cols`, and the reduction starts as an empty dict.

    For totals of columns that hold only numbers, `groupsum_daf()` adds them in SQL, several times faster.

    The new table has a row for each value of the column, with the column as its keyfield.
    With `by='row'` it has the columns of this table. The columns not reduced are empty.
    With `by='sparse_row'` it has two: this column, and the column named by `indirect_col`.
    That cell holds the dict of the group, with only the keys found in the group, as compact JSON text.
    With `reduce_cols` and a plain sum, each dict holds the columns its group has, in the order of `reduce_cols`.
    Results are in daffodil form, as `groupby_aggregate()` describes.
    The new table is a TEMP table, as `copy()` makes. Keep it with `save(name)`.

    It differs from [`Daf.groupby_reduce()`][daffodil.daf_apply._groupby_reduce] in three ways:

    - The rows are in the sorted order of the column, not in the order first seen.
      SQL NULL and '' are one group, first.
    - Each group is a plain Daf of the rows, with no keyfield and no dtypes.
    - `by='table'` is not supported.

    Args:
        colname: The column to group by.
        func: The reduction function, such as `Daf.sum_da`. See [`Daf.reduce()`][daffodil.daf_apply._reduce].
        by: `row` or `sparse_row`. See [`Daf.reduce()`][daffodil.daf_apply._reduce].
        reduce_cols: The columns to reduce. All columns except `colname` if None.
        diagnose: If True, show the query and the first rows of the result.
        batch_size: Not used. It is kept so that existing calls work.
        **kwargs: Keyword arguments passed on to the function, and `indirect_col` for `sparse_row`.

    Returns:
        A new SqlDaf with one row for each group.

    Raises:
        KeyError: The column is not a column.
        NotImplementedError: `by` is not `row` or `sparse_row`.

    Examples:
        Create a small SqlDaf table of carts, each a dict of item to count, with the city of the customer:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> carts_sdaf = SqlDaf('carts', cols=['city', 'cart'], lol=[
        ...     ['Denver', {'apples': 3}], ['Austin', {'apples': 5, 'pears': 2}], ['Denver', {'pears': 4}]])

        Total the items for each city, into a new table. Each cart is a sparse row:

        >>> carts_sdaf.groupby_reduce('city', Daf.sum_da, indirect_col='cart')
        <BLANKLINE>
        |  city  |          cart          |
        | -----: | ---------------------: |
        | Austin | {"apples":5,"pears":2} |
        | Denver | {"apples":3,"pears":4} |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield='city'; name='daf_tmp_carts_v1'
        <BLANKLINE>

        Reduce only some columns, with `reduce_cols`. The groups come in sorted order:

        >>> orders_sdaf = SqlDaf('orders', cols=['id', 'city', 'qty'], lol=[['o1', 'Denver', 2], ['o2', 'Austin', 6], ['o3', 'Denver', 1]],
        ...                      dtypes={'id': str, 'city': str, 'qty': int})
        >>> orders_sdaf.groupby_reduce('city', Daf.sum_da, reduce_cols=['qty'])
        <BLANKLINE>
        | id |  city  | qty |
        | -: | -----: | --: |
        |    | Austin |   6 |
        |    | Denver |   3 |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield='city'; name='daf_tmp_orders_v2'
        <BLANKLINE>
    """
    if colname not in self.hd:
        raise KeyError(colname)
    # indirect_col names the column that holds each row's dict, so it means sparse rows, as in reduce().
    if kwargs.get('indirect_col') and by == 'row':
        by = 'sparse_row'
    if by not in ('row', 'sparse_row'):
        raise NotImplementedError(f"groupby_reduce(by={by!r}) is not supported on a SqlDaf. Use 'row' or 'sparse_row'.")

    # The columns each group holds, as in Daf.groupby_reduce().
    scope_ls: T_ls | None = Daf._reduce_scope(self, by, reduce_cols, kwargs)     # it reads only the column names.
    group_cols_ls: T_ls = scope_ls if scope_ls else self.columns()
    func_kwargs: T_da = {key: val for key, val in kwargs.items() if key != 'indirect_col'}

    # The aggregate, as Daf.reduce() calls the function.
    agg_cls: Type[Any]
    if by == 'sparse_row':
        indirect_col: str = kwargs['indirect_col']
        sparse_func: Callable[..., Any] = func
        call_kwargs: T_da = {'cols': reduce_cols, 'is_sparse': True, **func_kwargs}
        select_ls: T_ls | None = None

        # A plain sum picks its loop once, from the first rows, as Daf.reduce() does.
        if Daf._is_plain_sum(func, None, False, func_kwargs):
            batches_iter: Generator[List[Tuple[Any, ...]], None, None] = self._iter_row_batches(Daf._SPARSE_SAMPLE_ROWS, [indirect_col])
            sample_lot: List[Tuple[Any, ...]] = next(batches_iter, [])
            batches_iter.close()                    # the rest of the rows are not read.
            sample_lodict: List[Any] = [daf_utils.get_indirect_da({indirect_col: none_to_null(vals_t)[0]}, indirect_col)
                                        for vals_t in sample_lot]
            loop_cols_t, select_ls = Daf._sparse_sum_plan(reduce_cols, sample_lodict)
            if loop_cols_t is None:
                sparse_func, call_kwargs = Daf._sum_sparse_keys, {}
            else:
                sparse_func, call_kwargs = Daf._sum_sparse_cols, {'cols': loop_cols_t}

        agg_cls = type('BoundCellReduction', (CellReduction,), {
                'func':         staticmethod(sparse_func),
                'col':          indirect_col,
                'call_kwargs':  call_kwargs,
                'sparse':       True,
                'select_cols':  select_ls,
                })
        return self._aggregate_groups([colname], agg_cls, [indirect_col], [indirect_col],
                                      [colname, indirect_col], 'groupby_reduce', diagnose)

    reduced_cols_ls: T_ls = group_cols_ls if reduce_cols is None else [col for col in group_cols_ls if col in reduce_cols]
    result_cols_ls: T_ls = [col for col in group_cols_ls if col != colname]
    agg_cls = type('BoundGroupReduction', (GroupReduction,), {
            'func':         staticmethod(func),
            'read_cols':    group_cols_ls,
            'reduce_cols':  reduced_cols_ls,
            'result_cols':  result_cols_ls,
            'call_kwargs':  func_kwargs,
            })
    return self._aggregate_groups([colname], agg_cls, group_cols_ls, result_cols_ls,
                                  self.columns(), 'groupby_reduce', diagnose)


def _groupby_aggregate(
        self: 'SqlDaf',
        groupby_cols:   str | T_ls,                 # the column, or columns, to group by.
        func:           Callable[..., Any],         # reduces the rows of a group, as in Daf.reduce().
        reduce_cols:    str | T_ls,                 # the columns given to the function.
        result_cols:    T_ls | None = None,         # the columns of the result. reduce_cols if None.
        diagnose:       bool = False,               # if True, show the query and the first results.
        **kwargs:       Any,                        # passed on to func.
        ) -> 'SqlDaf':
    """
    Group the rows and reduce each group with a Python function, in one SQL query.

    The function is registered in SQLite as an aggregate.
    SQLite groups the rows and gives the function each row of a group, in the order of the table.
    The function returns one result for each group.
    So no rows are read into Python, except one at a time through the function.
    It is the same reduction as `groupby_reduce()`, and faster when there are many small groups.

    The function has the form of [`Daf.sum_da()`][daffodil.daf_apply._sum_da]: it takes a row as a dict and the reduction
    so far, and returns the reduction.

    There are two forms, by the number of columns:

    - **One column.** The function gets the cell as the row, if it holds a dict, as text of
      JSON or PYON. This merges or totals concise dicts, which SQL cannot read.
      The result cell is the reduced dict, as compact JSON text.
      A cell that holds a plain value is given as a row of one column, and the result is its
      reduced value. An empty cell is an empty dict.
    - **Several columns.** The function gets a dict of the grouping columns and these
      columns, with each dict cell decoded. The reduction starts with 0 for each column of
      `reduce_cols`, as in [`Daf.reduce()`][daffodil.daf_apply._reduce]. The result has a column for each of `result_cols`.

    In a result, a dict or list is compact JSON text. True and False are 1 and 0, and None is
    NULLSTR, as `normalize_val()` gives them.

    The new table has the grouping columns, then the result columns, and one row for each
    group, in the sorted order of the grouping columns. SQL NULL and '' are one group, first.
    With one grouping column, it is the keyfield. The table is a TEMP table, as `copy()` makes.
    Keep it with `save(name)`.

    For totals of columns that hold only numbers, `groupsum_daf()` adds them in SQL, several times faster.

    An error in the function is raised as itself, not as the short error SQLite reports.

    Args:
        groupby_cols: The column, or columns, to group by.
        func: The reduction function, such as `Daf.sum_da`.
        reduce_cols: The columns given to the function.
        result_cols: The columns of the result, filled from the reduction. `reduce_cols` if None.
        diagnose: If True, show the query and the first rows of the result.
        **kwargs: Keyword arguments passed on to the function.

    Returns:
        A new SqlDaf with one row for each group.

    Raises:
        KeyError: A grouping or reduced column is not a column.

    Examples:
        Create a small SqlDaf table of carts, with a count of visits and a cart of items:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> carts_sdaf = SqlDaf('carts', cols=['city', 'visits', 'cart'], lol=[
        ...     ['Denver', 4, {'apples': 3}], ['Austin', 7, {'apples': 5, 'pears': 2}], ['Denver', 2, {'pears': 4}]])

        Total the carts for each city, inside SQLite:

        >>> carts_sdaf.groupby_aggregate('city', Daf.sum_da, 'cart')
        <BLANKLINE>
        |  city  |          cart          |
        | -----: | ---------------------: |
        | Austin | {"apples":5,"pears":2} |
        | Denver | {"apples":3,"pears":4} |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield='city'; name='daf_tmp_carts_v1'
        <BLANKLINE>

        Give the function two columns, and keep only the visits in the result:

        >>> carts_sdaf.groupby_aggregate('city', Daf.sum_da, ['visits', 'cart'], result_cols=['visits'])
        <BLANKLINE>
        |  city  | visits |
        | -----: | -----: |
        | Austin |      7 |
        | Denver |      6 |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield='city'; name='daf_tmp_carts_v2'
        <BLANKLINE>
    """
    group_cols_ls: T_ls = [groupby_cols] if isinstance(groupby_cols, str) else list(groupby_cols)
    reduce_cols_ls: T_ls = [reduce_cols] if isinstance(reduce_cols, str) else list(reduce_cols)
    result_cols_ls: T_ls = list(result_cols) if result_cols is not None else list(reduce_cols_ls)
    col: str
    for col in group_cols_ls + reduce_cols_ls:
        if col not in self.hd:
            raise KeyError(col)

    # The aggregate, by the number of columns.
    agg_cls: Type[Any]
    arg_cols_ls: T_ls
    if len(reduce_cols_ls) == 1 and len(result_cols_ls) == 1:
        # A plain sum adds every key, with no test, as Daf.reduce() does with no cols.
        plain_sum: bool = Daf._is_plain_sum(func, None, False, kwargs)
        agg_cls = type('BoundCellReduction', (CellReduction,), {
                'func':         staticmethod(Daf._sum_sparse_keys if plain_sum else func),
                'col':          reduce_cols_ls[0],
                'call_kwargs':  {} if plain_sum else {'cols': None, 'is_sparse': True, **kwargs},
                })
        arg_cols_ls = [reduce_cols_ls[0]]
    else:
        read_cols_ls: T_ls = group_cols_ls + [col for col in reduce_cols_ls if col not in group_cols_ls]
        agg_cls = type('BoundRowReduction', (RowReduction,), {
                'func':         staticmethod(func),
                'read_cols':    read_cols_ls,
                'reduce_cols':  reduce_cols_ls,
                'result_cols':  result_cols_ls,
                'call_kwargs':  kwargs,
                })
        arg_cols_ls = read_cols_ls
    return self._aggregate_groups(group_cols_ls, agg_cls, arg_cols_ls, result_cols_ls,
                                  group_cols_ls + result_cols_ls, 'groupby_aggregate', diagnose)


def _aggregate_groups(
        self: 'SqlDaf',
        group_cols:     T_ls,                       # the columns to group by.
        agg_cls:        Type[Any],                  # a bound CellReduction or RowReduction.
        arg_cols:       T_ls,                       # the columns given to the aggregate, one argument each.
        result_cols:    T_ls,                       # the columns the aggregate fills.
        table_cols:     T_ls,                       # the columns of the new table, in order.
        label:          str,                        # the method named in diagnose output.
        diagnose:       bool,                       # if True, show the query and the first results.
        ) -> 'SqlDaf':
    """
    Make a new table with one row for each group, filled by a Python aggregate. Internal use.

    This is the query of `groupby_aggregate()` and `groupby_reduce()`.
    A CellReduction returns the cell itself, and a RowReduction returns a JSON array of the result columns.
    A column of the table that is neither grouped nor filled is empty.
    """
    self._resolve()
    rows_sql, rows_params = self._rows_sql()
    agg_name: str = self._register_aggregate(agg_cls, len(arg_cols))

    # The query. The ORDER BY inside the call gives each group to the function in the order of the table.
    # Each cell is its own argument, so values keep their types and every digit.
    group_names_ls: T_ls = [f'daf_g{idx}' for idx in range(len(group_cols))]
    groups_sql: str = ', '.join(f'{value_sql(col)} AS {name}' for col, name in zip(group_cols, group_names_ls))
    names_sql: str = ', '.join(group_names_ls)
    args_sql: str = ', '.join(quote_name(col) for col in arg_cols)
    grouped_sql: str = (f'SELECT {groups_sql}, {agg_name}({args_sql} ORDER BY {ORD_COL}) AS daf_res '
                        f'FROM ({rows_sql}) GROUP BY {names_sql}')
    if issubclass(agg_cls, CellReduction):
        results_sql: str = "NULLIF(daf_res, '')"
    else:
        results_sql = ', '.join(f"NULLIF(json_extract(daf_res, '$[{idx}]'), '')" for idx in range(len(result_cols)))
    out_cols_ls: T_ls = group_cols + result_cols
    selected_sql: str = f'{names_sql}, {results_sql}' if result_cols else names_sql
    target: str = self._next_temp_name(self._root().name)
    insert_sql: str = (f'INSERT INTO temp.{quote_name(target)} ({", ".join(quote_name(col) for col in out_cols_ls)}) '
                       f'SELECT {selected_sql} FROM ({grouped_sql}) ORDER BY {names_sql}')
    if diagnose:
        daf_utils.sts(f"{label}({group_cols}): {insert_sql}", 3)

    # The new table, filled by the one query.
    with self.transaction():
        self._execute(_create_table_sql(self.conn, target, table_cols, 'temp'))
        self._execute(insert_sql, rows_params)
    keyfield: str = group_cols[0] if len(group_cols) == 1 else ''
    result_sdaf: SqlDaf = self._new_on_conn(target, keyfield, None, is_temp=True)
    if diagnose:
        daf_utils.sts(f"{label}({group_cols}): {len(result_sdaf)} groups. First:\n{result_sdaf[:3].to_daf()}", 3)
    return result_sdaf


def _groupby_cols(
        self: 'SqlDaf',
        colnames:       T_ls,                       # the columns to group by.
        cols:           T_ls | None = None,         # not supported on a SqlDaf.
        omit_nulls:     bool = False,               # if True, leave out rows with Python NULL in any of the columns.
        ) -> KeyedList:
    """
    Group the rows by the values of several columns, as [`Daf.groupby_cols()`][daffodil.daf_apply._groupby_cols] does.

    Each key is a tuple of the values, even for one column. Each group is a selection, as in
    `groupby()`.

    Args:
        colnames: The columns to group by.
        cols: Not supported on a SqlDaf. It raises NotImplementedError.
        omit_nulls: If True, leave out the rows with Python NULL in any of the columns.

    Returns:
        A KeyedList that maps each tuple of values to a selection.

    Raises:
        KeyError: A name in `colnames` is not a column.
        NotImplementedError: `cols` is given.

    Examples:
        Create a small SqlDaf table of customers. One plan is not known:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> customers_sdaf = SqlDaf('customers', cols=['id', 'city', 'plan'], keyfield='id', lol=[
        ...     ['c1', 'Denver', 'basic'], ['c2', 'Austin', 'pro'], ['c3', 'Denver', 'basic'], ['c4', 'Denver', 'pro'], ['c5', 'Denver', '']])

        Group the customers by city and plan. Each key is a tuple:

        >>> groups_klist = customers_sdaf.groupby_cols(['city', 'plan'])
        >>> {key: group_sdaf.keys() for key, group_sdaf in groups_klist.items()}
        {('Denver', 'basic'): ['c1', 'c3'], ('Austin', 'pro'): ['c2'], ('Denver', 'pro'): ['c4'], ('Denver', ''): ['c5']}

        Leave out a row with Python NULL in any of the columns, with `omit_nulls=True`:

        >>> {key: group_sdaf.keys() for key, group_sdaf in customers_sdaf.groupby_cols(['plan'], omit_nulls=True).items()}
        {('basic',): ['c1', 'c3'], ('pro',): ['c2', 'c4']}
    """
    if cols is not None:
        raise NotImplementedError("groupby(cols=) is not supported on a SqlDaf yet. Group, then read the columns you need.")
    col: str
    for col in colnames:
        if col not in self.hd:
            raise KeyError(col)
    self._resolve_parent()
    self._ensure_group_index(colnames)
    rows_sql, rows_params = self._rows_sql()
    group_sql: str = ', '.join(value_sql(col) for col in colnames)          # '' and SQL NULL are one group.
    where_sql: str = (' WHERE ' + ' AND '.join(f'{value_sql(col)} IS NOT NULL' for col in colnames)) if omit_nulls else ''
    values_lot: List[Tuple[Any, ...]] = self._fetch(self._execute(
            f'SELECT {group_sql} FROM ({rows_sql}){where_sql} GROUP BY {group_sql} ORDER BY MIN({ORD_COL})', rows_params))
    keys_lt: List[Tuple[Any, ...]] = [tuple(none_to_null(values_t)) for values_t in values_lot]
    return KeyedList(keys_lt, [self.select_by_dict(dict(zip(colnames, key_t))) for key_t in keys_lt])


def _groupby_repeated_cols(
        self: 'SqlDaf',
        colnames:   T_ls,                                   # the columns that make the key of a row.
        exclude:    Dict[str, T_la] | None = None,          # values of a column that make a row never match.
        ) -> 'T_klosdaf':
    """
    Group the rows whose key repeats, as [`Daf.groupby_repeated_cols()`][daffodil.daf_apply._groupby_repeated_cols] does.

    The repeated keys are found in one query, a GROUP BY that keeps the keys with more than
    one row. So no group is made for a key seen once, and nothing comes into memory but the
    keys. Each group is a selection of the rows with its key, in table order, as in
    `groupby_cols()`. The groups come in the order of their first rows. An empty result means
    no key repeats.

    Python NULL is a value, so two rows with a blank in the same key column match. The text ''
    and SQL NULL are one value. A row is left out if any column named in `exclude` holds one of
    its values. Values compare as stored, so an int 0 does not match the text '0'.

    On the 133,422 rows of a BIF table, finding the repeated keys of one or three columns took
    0.09 to 0.14 s. An index on the key columns is made only when some key repeats, so each
    selection reads its rows by the index.

    Args:
        colnames: The columns that make the key of a row.
        exclude: For each column named, the values that keep a row out of every group.

    Returns:
        A KeyedList that maps each repeated key, a tuple of values, to a selection of its rows.

    Raises:
        KeyError: A name in `colnames` or `exclude` is not a column.

    Examples:
        Create a small SqlDaf table of sign-ups, where some people signed up more than once:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> signups_sdaf = SqlDaf('signups', cols=['id', 'name', 'birth_year', 'status'], keyfield='id', lol=[
        ...     ['s1', 'Ana Lopez', 1990, 'ok'], ['s2', 'Ben Okafor', 1972, 'ok'], ['s3', 'Ana Lopez', 1990, 'spam'],
        ...     ['s4', 'Chen Wu', 1997, 'ok'], ['s5', 'Ben Okafor', 1972, 'ok']])
        >>> signups_sdaf
        <BLANKLINE>
        | id |    name    | birth_year | status |
        | -: | ---------: | ---------: | -----: |
        | s1 |  Ana Lopez |       1990 |     ok |
        | s2 | Ben Okafor |       1972 |     ok |
        | s3 |  Ana Lopez |       1990 |   spam |
        | s4 |    Chen Wu |       1997 |     ok |
        | s5 | Ben Okafor |       1972 |     ok |
        <BLANKLINE>
        %% daf rows=5; cols=4; keyfield='id'; name='signups'
        <BLANKLINE>

        Group the rows whose name and birth year repeat. A pair seen once makes no group:

        >>> {key: group_sdaf.keys() for key, group_sdaf in signups_sdaf.groupby_repeated_cols(['name', 'birth_year']).items()}
        {('Ana Lopez', 1990): ['s1', 's3'], ('Ben Okafor', 1972): ['s2', 's5']}

        Leave out the rows marked spam first, with `exclude`, so that pair no longer repeats:

        >>> {key: group_sdaf.keys() for key, group_sdaf in signups_sdaf.groupby_repeated_cols(['name', 'birth_year'], exclude={'status': ['spam']}).items()}
        {('Ben Okafor', 1972): ['s2', 's5']}
    """
    col: str
    for col in list(colnames) + list(exclude or {}):
        if col not in self.hd:
            raise KeyError(col)
    self._resolve_parent()
    rows_sql, rows_params = self._rows_sql()

    # ---- a row is left out if a column of exclude holds one of its values. Python NULL is ''.
    where_ls: T_ls = []
    where_params_la: T_la = []
    vals_la: T_la
    for col, vals_la in (exclude or {}).items():
        others_la: T_la = [val for val in vals_la if val is not NULL]
        null_sql: str = 'IS NOT NULL' if NULL in vals_la else 'IS NULL'
        not_in_sql: str = f' NOT IN ({placeholders(len(others_la))})' if others_la else ''
        if not_in_sql and NULL in vals_la:
            where_ls.append(f'({value_sql(col)} IS NOT NULL AND {value_sql(col)}{not_in_sql})')
        elif not_in_sql:
            where_ls.append(f'({value_sql(col)} IS NULL OR {value_sql(col)}{not_in_sql})')
        else:
            where_ls.append(f'{value_sql(col)} {null_sql}')
        where_params_la.extend(others_la)
    where_sql: str = (' WHERE ' + ' AND '.join(where_ls)) if where_ls else ''

    group_sql: str = ', '.join(value_sql(col) for col in colnames)          # '' and SQL NULL are one value.
    values_lot: List[Tuple[Any, ...]] = self._fetch(self._execute(
            f'SELECT {group_sql} FROM ({rows_sql}){where_sql} GROUP BY {group_sql} '
            f'HAVING COUNT(*) > 1 ORDER BY MIN({ORD_COL})', tuple(rows_params) + tuple(where_params_la)))
    keys_lt: List[Tuple[Any, ...]] = [tuple(none_to_null(values_t)) for values_t in values_lot]
    if keys_lt:
        self._ensure_group_index(list(colnames))

    # ---- a selection for each key. A column of exclude outside the key is applied to it too.
    groups_l: List[SqlDaf] = [self.select_by_dict(dict(zip(colnames, key_t))) for key_t in keys_lt]
    if where_ls and any(col not in colnames for col in exclude or {}):
        groups_l = [group_sdaf.select_where_sql(' AND '.join(where_ls), where_params_la) for group_sdaf in groups_l]
    return KeyedList(keys_lt, groups_l)


def _ensure_group_index(self: 'SqlDaf', colnames: T_ls) -> None:
    """
    Make an index on the columns a groupby uses, on the parent table, if none starts with them. Internal use.

    Each group is a selection by its value, so without an index each group reads the whole table.
    """
    root_sdaf: SqlDaf = self._root()
    if any(index_cols[:len(colnames)] == colnames for index_cols in _indexes_of(self.conn, root_sdaf.name).values()):
        return
    index_name: str = f'{GROUP_INDEX_PREFIX}{root_sdaf.name}_' + '_'.join(colnames)
    with self.transaction():
        self._execute(f'CREATE INDEX IF NOT EXISTS {quote_name(index_name)} ON {quote_name(root_sdaf.name)} '
                      f'({", ".join(quote_name(col) for col in colnames)})')


def _group_where(
        self: 'SqlDaf',
        where:          Callable[[Any], Any],       # a function of a row: a key, a list of keys, or None.
        *,
        indirect_col:   str | None = None,          # not supported on a SqlDaf.
        cols:           T_ls | None = None,         # the columns the function reads. All if None.
        ) -> KeyedList:
    """
    Group the rows by the keys a Python function returns, as [`Daf.group_where()`][daffodil.daf_apply._group_where] does.

    The function gets each row as a KeyedList, as in `select_where()`, and returns a key, a
    list of keys, or None. A row goes into the group of each key, so a row can be in any
    number of groups, and None leaves it out. As in [`Daf.group_where()`][daffodil.daf_apply._group_where], any iterable other
    than a str is a list of keys, a tuple too. So to use a tuple as one key, return a list
    of tuples.

    The function runs once for each row, in one pass, and each (key, row) pair is written to
    a TEMP table. Each group is then a selection of the rows listed under its key, in the
    order of this SqlDaf. Nothing more is read until a group is read. The result is a
    KeyedList of selections, in the order the keys were first returned.

    Name the columns the function reads in `cols`, as in `select_where()`.

    Args:
        where: A function of a row: a key, a list of keys, or None.
        indirect_col: Not supported on a SqlDaf. It raises NotImplementedError.
        cols: The columns the function reads. All if None.

    Returns:
        A KeyedList that maps each key to a selection.

    Raises:
        KeyError: A name in `cols` is not a column.
        NotImplementedError: `indirect_col` is given.

    Examples:
        Create a small SqlDaf table of people:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> people_sdaf = SqlDaf('people', cols=['id', 'city'], lol=[['p1', 'Denver'], ['p2', 'Austin'], ['p3', 'Denver']])

        Put each person in two groups, their city and 'all'. The function returns the groups of a row:

        >>> groups = people_sdaf.group_where(lambda row: [row['city'], 'all'], cols=['city'])
        >>> {key: len(group_sdaf) for key, group_sdaf in groups.items()}
        {'Denver': 2, 'all': 3, 'Austin': 1}
    """
    if indirect_col is not None:
        raise NotImplementedError("group_where(indirect_col=) is not supported on a SqlDaf yet.")
    func_cols: T_ls = list(cols) if cols is not None else list(self.hd)
    col: str
    for col in func_cols:
        if col not in self.hd:
            raise KeyError(col)
    func_hd: T_di = {col: idx for idx, col in enumerate(func_cols)}
    self._resolve_parent()
    rows_sql, rows_params = self._rows_sql()
    labels: str = self._next_temp_name(self._root().name + '_groups')
    key_ids_da: Dict[Any, int] = {}                     # each key and its number in the TEMP table, in the order first seen.

    with self.transaction():
        self._execute(f'CREATE TEMP TABLE {quote_name(labels)} (kid INTEGER, ref)')
        cursor: sqlite3.Cursor = self._execute(
                f'SELECT {REF_COL}{"".join(", " + quote_name(col) for col in func_cols)} FROM ({rows_sql}) ORDER BY {ORD_COL}',
                rows_params)
        insert_sql: str = f'INSERT INTO {quote_name(labels)} (kid, ref) VALUES (?, ?)'
        while True:
            batch_lot: List[Tuple[Any, ...]] = cursor.fetchmany(10000)
            if not batch_lot:
                break
            pairs_lt: List[Tuple[int, Any]] = []
            row_t: Tuple[Any, ...]
            for row_t in batch_lot:
                keys: Any = where(KeyedList(func_hd, none_to_null(row_t[1:])))
                if keys is None:
                    continue
                key: Any
                for key in (keys if isinstance(keys, Iterable) and not isinstance(keys, (str, bytes)) else (keys,)):    # as in Daf: a tuple is a list of keys.
                    kid: int | None = key_ids_da.get(key)
                    if kid is None:
                        kid = key_ids_da[key] = len(key_ids_da)
                    pairs_lt.append((kid, row_t[0]))
            self.conn.executemany(insert_sql, pairs_lt)
        self._execute(f'CREATE INDEX {quote_name(labels + "_ix")} ON {quote_name(labels)} (kid, ref)')

    member_sql: str = f'{REF_COL} IN (SELECT ref FROM {quote_name(labels)} WHERE kid = ?)'
    return KeyedList(list(key_ids_da), [self.select_where_sql(member_sql, [kid]) for kid in key_ids_da.values()])


def _join(
        self: 'SqlDaf',
        other_daf:              'SqlDaf',           # the SqlDaf to join with, in this file or another.
        how:                    str = 'inner',      # 'inner', 'left', 'right' or 'outer'.
        shared_fields:          T_ls | None = None, # columns that both have and that appear only once.
        tag_other:              bool = False,       # if True, every column of the other gets a suffix.
        custom_translator_daf:  Daf | None = None,  # not supported on a SqlDaf.
        diagnose:               bool = False,       # not used.
        name:                   str = '',           # the name of the result. Generated if empty.
        fill:                   Any = NULL,         # only Python NULL is supported.
        ) -> 'SqlDaf':
    """
    Join two SqlDaf tables on their keyfields, as [`Daf.join()`][daffodil.daf_join._join] does, in SQL.

    The columns are named exactly as [`Daf.join()`][daffodil.daf_join._join] names them, by the same translator: a
    column that both have gets the name of its source as a suffix, such as `name_audit`,
    unless it is in `shared_fields`, and `tag_other` suffixes every column of the other. The
    names are those of the parent tables, or `daf1` and `daf2` if they are the same. The rows
    come as in [`Daf.join()`][daffodil.daf_join._join]: in this table's order, and for `right` and `outer`, the other
    table's unmatched rows after, in its order. A cell with no match is Python NULL. The keyfield of
    the result is this table's.

    The keyfields must be unique, as in [`Daf.join()`][daffodil.daf_join._join]. A repeated key is not checked, and the
    result is then undefined. Check the keys first if they may repeat. A Python NULL key is
    undefined too: see SqlDaf.

    A row found only in the other table, in a right or outer join, takes its key and its
    shared fields from the other table, as in [`Daf.join()`][daffodil.daf_join._join].

    Nothing runs yet. The join is resolved the first time it is read, into a TEMP table of
    new rows, as [`Daf.join()`][daffodil.daf_join._join] makes new rows. A selection made from it resolves it first.

    The other table may be in another database file. It is copied then into a TEMP table on this
    table's connection, with its key index, and the join reads the copy. Neither file changes. The
    copy is deleted when the join is first read, or when the join is gone unread, or when the file
    is closed. A chain of joins keeps each copy until the last join that uses it is read. See the
    Guide page on SqlDaf, Tables in other files.

    Args:
        other_daf: The SqlDaf to join with. It may be in another database file.
        how: `inner`, `left`, `right` or `outer`.
        shared_fields: Columns that both have and that appear only once.
        tag_other: If True, every column of the other, except shared ones, gets a suffix.
        custom_translator_daf: Not supported on a SqlDaf. It raises NotImplementedError.
        diagnose: Not used.
        name: The name of the result table. Generated if empty.
        fill: The value for cells that have no match. Only Python NULL is supported.

    Returns:
        A SqlDaf of the joined rows, resolved when first read.

    Raises:
        ValueError: `how` is not one of the four, or a table has no keyfield.
        NotImplementedError: `custom_translator_daf` or a `fill` other than Python NULL, or a selection or TEMP table of another thread's connection.

    Examples:
        Create two small SqlDaf tables, people and their salaries, keyed by id:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> people_sdaf = SqlDaf('people', cols=['id', 'name'], keyfield='id', lol=[['p1', 'Ana Lopez'], ['p2', 'Ben Okafor']])
        >>> pay_sdaf = SqlDaf('pay', cols=['id', 'salary'], keyfield='id', lol=[['p1', 72000], ['p3', 58000]])

        Join them, keeping every person, with a blank salary where there is none:

        >>> people_sdaf.join(pay_sdaf, how='left')
        <BLANKLINE>
        | id |    name    | salary |
        | -: | ---------: | -----: |
        | p1 |  Ana Lopez |  72000 |
        | p2 | Ben Okafor |        |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield='id'; name='daf_tmp_people_join_pay_v1'
        <BLANKLINE>

        Both tables have a name column. By default, each gets its table's name as a suffix:

        >>> staff_sdaf = SqlDaf('staff', cols=['id', 'name', 'salary'], keyfield='id', lol=[['p1', 'Ana Lopez', 72000]])
        >>> people_sdaf.join(staff_sdaf).columns()
        ['id', 'name_people', 'name_staff', 'salary']

        Take the name once, with `shared_fields`, and name the result:

        >>> people_sdaf.join(staff_sdaf, shared_fields=['name'], name='people_pay')
        <BLANKLINE>
        | id |   name    | salary |
        | -: | --------: | -----: |
        | p1 | Ana Lopez |  72000 |
        <BLANKLINE>
        %% daf rows=1; cols=3; keyfield='id'; name='people_pay'
        <BLANKLINE>

        Give every column of the other table a suffix, with `tag_other=True`:

        >>> people_sdaf.join(staff_sdaf, tag_other=True).columns()
        ['id', 'name_people', 'name_staff', 'salary_staff']
    """
    if how not in ('inner', 'left', 'right', 'outer'):
        raise ValueError(f"join: how='{how}' is not inner, left, right or outer.")
    if custom_translator_daf is not None:
        raise NotImplementedError("join(custom_translator_daf=) is not supported on a SqlDaf yet.")
    if fill is not NULL:
        raise NotImplementedError("join(fill=) other than Python NULL is not supported on a SqlDaf yet.")
    if not isinstance(other_daf, SqlDaf):
        raise NotImplementedError("join: the other must be a SqlDaf. Make one with SqlDaf.from_daf().")
    if not self.keyfield or not other_daf.keyfield:
        raise ValueError("join: both tables need a keyfield.")

    # A table in another file is brought in as a TEMP copy, which lives until the join is first read.
    # A chain of joins also keeps the copies of the unread joins it is made from.
    other_rows_sdaf: SqlDaf = self._readable(other_daf)
    _brought_in_ls: T_ls = list(self._brought_in_ls) + list(other_daf._brought_in_ls)
    if other_rows_sdaf is not other_daf:
        _brought_in_ls.append(other_rows_sdaf.name)

    a_name: str = self._root().name
    b_name: str = other_daf._root().name
    if a_name == b_name:
        a_name, b_name = 'daf1', 'daf2'
    translator_daf: Daf = Daf.derive_join_translator_daf(
            self_keyfield   = self.keyfield,
            other_keyfield  = other_daf.keyfield,
            self_cols       = self.columns(),
            other_cols      = other_daf.columns(),
            self_name       = a_name,
            other_name      = b_name,
            shared_fields   = shared_fields,
            tag_other       = tag_other,
            )
    shared_set: set = set(shared_fields or [])

    out_cols: T_ls = []
    a_exprs_ls: T_ls = []                           # the columns for rows of this table, matched or not.
    b_exprs_ls: T_ls = []                           # the columns for rows found only in the other.
    tr_la: T_la
    for tr_la in translator_daf.lol:
        resolved_col, source_name, source_col, is_keyfield = tr_la
        out_cols.append(resolved_col)
        if source_name == a_name:
            a_exprs_ls.append(f'A.{quote_name(source_col)}')
            if is_keyfield:
                b_exprs_ls.append(f'B.{quote_name(other_daf.keyfield)}')
            elif source_col in shared_set and source_col in other_daf.hd:
                b_exprs_ls.append(f'B.{quote_name(source_col)}')
            else:
                b_exprs_ls.append('NULL')
        else:
            a_exprs_ls.append(f'B.{quote_name(source_col)}')
            b_exprs_ls.append(f'B.{quote_name(source_col)}')

    a_sql, a_params = self._rows_sql()
    b_sql, b_params = other_rows_sdaf._rows_sql()
    on_sql: str = f'A.{quote_name(self.keyfield)} = B.{quote_name(other_daf.keyfield)}'
    named_ls: T_ls = [f'{expr} AS {quote_name(col)}' for expr, col in zip(a_exprs_ls, out_cols)]
    join_kind: str = 'JOIN' if how in ('inner', 'right') else 'LEFT JOIN'
    parts_ls: T_ls = [f'SELECT {", ".join(named_ls)}, 0 AS daf_grp, A.{ORD_COL} AS daf_o '
                      f'FROM ({a_sql}) AS A {join_kind} ({b_sql}) AS B ON {on_sql}']
    params: Tuple[Any, ...] = a_params + b_params
    if how in ('right', 'outer'):
        b_named_ls: T_ls = [f'{expr} AS {quote_name(col)}' for expr, col in zip(b_exprs_ls, out_cols)]
        parts_ls.append(f'SELECT {", ".join(b_named_ls)}, 1 AS daf_grp, B.{ORD_COL} AS daf_o '
                        f'FROM ({b_sql}) AS B WHERE NOT EXISTS (SELECT 1 FROM ({a_sql}) AS A WHERE {on_sql})')
        params = params + b_params + a_params

    cols_sql: str = ', '.join(quote_name(col) for col in out_cols)
    pending_sql: str = (f'SELECT NULL AS {REF_COL}, ROW_NUMBER() OVER (ORDER BY daf_grp, daf_o) AS {ORD_COL}, {cols_sql} '
                        f'FROM ({" UNION ALL ".join(parts_ls)})')

    result_keyfield: str = next((tr_la[0] for tr_la in translator_daf.lol if tr_la[3]), '')
    join_sdaf: SqlDaf = SqlDaf.__new__(SqlDaf)
    join_sdaf.conn           = self.conn
    join_sdaf._conn_da       = self._conn_da
    join_sdaf.db_path        = self.db_path
    join_sdaf._name          = name or self._next_temp_name(f'{a_name}_join_{b_name}')
    join_sdaf.keyfield       = result_keyfield
    join_sdaf.has_rid        = False
    join_sdaf.hd             = {col: idx for idx, col in enumerate(out_cols)}
    join_sdaf.dtypes         = {}
    join_sdaf.is_temp        = True
    join_sdaf.parent_sdaf    = None
    join_sdaf.pending_sql    = pending_sql
    join_sdaf.pending_params = params
    join_sdaf.is_resolved    = False
    if _brought_in_ls:
        self._hold_brought_in(join_sdaf, _brought_in_ls)
    return join_sdaf


def _join_with_residue(
        self: 'SqlDaf',
        other_daf:              'SqlDaf',               # the other SqlDaf, joined by key.
        shared_fields:          T_ls | None = None,     # columns both have that appear once, as in join().
        tag_other:              bool = False,           # suffix the other's columns, as in join().
        custom_translator_daf:  Daf | None = None,      # not supported on a SqlDaf.
        diagnose:               bool = False,           # not used.
        name:                   str = '',               # the name of the joined table.
        ) -> Tuple['SqlDaf', 'SqlDaf', 'SqlDaf']:
    """
    Join two tables by key, and keep the rows of each that have no partner, as [`Daf.join_with_residue()`][daffodil.daf_join._join_with_residue] does.

    It gives three SqlDaf objects: the inner join, as `join()` gives it, the rows of this table
    whose key the other lacks, and the rows of the other table whose key this one lacks. Each
    residue is a selection of its own table, in its own columns, as `select_records_daf()` gives.
    Nothing runs yet. Each result is found when it is first read. Neither table is changed.

    The other table may be in another database file. The join brings it in as `join()` does.
    Each residue reads only the keys of the other table, so a table in another file gives only
    its key column there.

    The rows of both tables stay as they are. Each residue looks up keys, so each table gets the
    unique index on its keyfield in its own file, if it has none yet, as any lookup by key makes it.
    A table opened with `read_only=True` allows that, since an index leaves the data as it is.

    Args:
        other_daf: The SqlDaf to join with, by its keyfield. It may be in another database file.
        shared_fields: Columns that both have and that appear only once in the join.
        tag_other: If True, every column of the other, except shared ones, gets a suffix.
        custom_translator_daf: Not supported on a SqlDaf. It raises NotImplementedError.
        diagnose: Not used.
        name: The name of the joined table. Generated if empty.

    Returns:
        The joined SqlDaf, the residue of this SqlDaf, and the residue of the other.

    Raises:
        ValueError: A table has no keyfield.
        NotImplementedError: `custom_translator_daf` is given.

    Examples:
        Create two small SqlDaf tables, people and their salaries. Each has a key the other lacks:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> people_sdaf = SqlDaf('people', cols=['id', 'name', 'city'], keyfield='id', lol=[
        ...     ['p1', 'Ana Lopez', 'Denver'], ['p2', 'Ben Okafor', 'Austin'], ['p3', 'Chen Wu', 'Boston']])
        >>> pay_sdaf = SqlDaf('pay', cols=['id', 'salary', 'city'], keyfield='id', lol=[
        ...     ['p2', 64000, 'Austin'], ['p3', 58000, 'Boston'], ['p9', 81000, 'Denver']])

        Join them, and keep the rows of each that have no partner:

        >>> joined_sdaf, people_res_sdaf, pay_res_sdaf = people_sdaf.join_with_residue(pay_sdaf, shared_fields=['city'], name='people_pay')
        >>> joined_sdaf
        <BLANKLINE>
        | id |    name    |  city  | salary |
        | -: | ---------: | -----: | -----: |
        | p2 | Ben Okafor | Austin |  64000 |
        | p3 |    Chen Wu | Boston |  58000 |
        <BLANKLINE>
        %% daf rows=2; cols=4; keyfield='id'; name='people_pay'
        <BLANKLINE>
        >>> people_res_sdaf.to_lod()
        [{'id': 'p1', 'name': 'Ana Lopez', 'city': 'Denver'}]
        >>> pay_res_sdaf.to_lod()
        [{'id': 'p9', 'salary': 81000, 'city': 'Denver'}]

        The other table may be in another file:

        >>> pay_path = tempfile.mkdtemp() + '/pay.db'
        >>> far_sdaf = pay_sdaf.copy().save('pay_far', db_path=pay_path)
        >>> [result_sdaf.keys() for result_sdaf in people_sdaf.join_with_residue(far_sdaf, tag_other=True)]
        [['p2', 'p3'], ['p1'], ['p9']]
    """
    if custom_translator_daf is not None:
        raise NotImplementedError("join_with_residue(custom_translator_daf=) is not supported on a SqlDaf.")
    joined_sdaf: SqlDaf = self.join(other_daf, how='inner', shared_fields=shared_fields, tag_other=tag_other,
                                    diagnose=diagnose, name=name)

    # Each residue is a selection of its own table: the rows whose key the other table lacks.
    self_res_sdaf: SqlDaf = self.select_records_daf(other_daf, inverse=True, silent_error=True)
    other_res_sdaf: SqlDaf = other_daf.select_records_daf(self, inverse=True, silent_error=True)
    return joined_sdaf, self_res_sdaf, other_res_sdaf
