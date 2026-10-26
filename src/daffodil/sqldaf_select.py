# sqldaf_select.py
# copyright (c) 2026 Ray Lutz
"""
Methods of SqlDaf for selecting.

Each function here takes the table as its first argument, or the class for a classmethod.
It is assigned in the SqlDaf class body under its name, without the underscore.
The docs show it under that name.

Some functions use names defined in sqldaf.py, such as SqlDaf, when they run.
sqldaf.py sets those names here, as listed in _NAMES_FROM_SQLDAF, once they exist.
An import at the top would be circular, since sqldaf.py imports this module.
"""


from typing import Any, Callable, List, Tuple, Type, TYPE_CHECKING
from collections.abc import Iterable
from daffodil.daf import Daf, NULL
from daffodil.keyedlist import KeyedList
from daffodil.lib.daf_types import T_ls, T_la, T_di, T_da

if TYPE_CHECKING:       # for mypy. sqldaf.py sets the names below when the program runs.
    from daffodil.sqldaf import SqlDaf, ORD_COL, REF_COL, is_value_sql, null_to_none, quote_name

# The names of sqldaf.py that the functions here use when they run.
# sqldaf.py sets each one here once it exists. An import at the top would be circular.
_NAMES_FROM_SQLDAF: Tuple[str, ...] = ('SqlDaf', 'ORD_COL', 'REF_COL', 'is_value_sql', 'null_to_none', 'quote_name')


def _select_where_sql(self: 'SqlDaf', where_sql: str, params: Iterable[Any] = ()) -> 'SqlDaf':
    """
    Select the rows for which a SQL condition is true.

    The condition is the text after WHERE. Name the columns as they are, in double quotes if
    they need it, and pass values as `params`, with `?` in the text. Python NULL in `params`
    becomes SQL NULL. Test for a missing value with `IS NULL`. This is the fastest way to select,
    since SQLite does all the work.

    Nothing runs yet. The rows are found when the selection is first read. See SqlDaf.

    Args:
        where_sql: The condition, as SQL text.
        params: Values for the `?` marks in it, in order.

    Returns:
        A selection.

    Examples:
        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> marks_sdaf = SqlDaf.from_daf(Daf(lol=[['b1', 1], ['b2', 0]], cols=['ballot_id', 'v']), name='marks')
        >>> marks_sdaf.to_daf()
        <BLANKLINE>
        | ballot_id | v  |
        | --------: | -: |
        |        b1 |  1 |
        |        b2 |  0 |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield=''; name='marks'
        <BLANKLINE>
        >>> marks_sdaf.select_where_sql('"v" = ?', [0]).to_list()
        ['b2', 0]
    """
    self._resolve_parent()
    parent_sql, parent_params = self._rows_sql()
    return SqlDaf._new_selection(self, f'SELECT * FROM ({parent_sql}) WHERE {where_sql}',
                                 parent_params + null_to_none(params))


def _select_where(
        self: 'SqlDaf',
        where:          Callable[[Any], Any],       # a function of a row that is true for the rows to keep.
        indirect_col:   str | None = None,          # not supported yet.
        cols:           T_ls | None = None,         # the columns the function reads. All if None.
        ) -> 'SqlDaf':
    """
    Select the rows for which a Python function is true, as `Daf.select_where()` does.

    The function gets each row as a [KeyedList][daffodil.keyedlist.KeyedList], with Python NULL for
    a missing value, and reads cells by name, as in `row['n']`. It is registered with SQLite
    and called for each row when the selection is first read. If it raises, that error is
    raised, as itself.

    Name the columns it reads in `cols`. It is then given only those, which is faster. A row
    of every column costs about 1 us. If the test only compares columns to values, use
    `select_by_dict()` or `select_where_sql()`, which call no Python function.

    Args:
        where: A function of a row, true for the rows to keep.
        indirect_col: Not supported on a SqlDaf yet.
        cols: The columns the function reads. All if None.

    Returns:
        A selection.

    Raises:
        KeyError: A name in `cols` is not a column.
        NotImplementedError: `indirect_col` is given.

    Examples:
        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> marks_sdaf = SqlDaf.from_daf(Daf(lol=[['b1', 1], ['b2', 0]], cols=['ballot_id', 'v']), name='marks')
        >>> marks_sdaf.to_daf()
        <BLANKLINE>
        | ballot_id | v  |
        | --------: | -: |
        |        b1 |  1 |
        |        b2 |  0 |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield=''; name='marks'
        <BLANKLINE>
        >>> marks_sdaf.select_where(lambda row: row['v'] == 1, cols=['v']).to_list()
        ['b1', 1]
    """
    if indirect_col is not None:
        raise NotImplementedError("select_where(indirect_col=) is not supported on a SqlDaf yet.")
    func_cols: T_ls = list(cols) if cols is not None else list(self.hd)
    col: str
    for col in func_cols:
        if col not in self.hd:
            raise KeyError(col)
    func_hd: T_di = {col: idx for idx, col in enumerate(func_cols)}

    def row_test(*values: Any) -> int:
        return 1 if where(KeyedList(func_hd, [NULL if val is None else val for val in values])) else 0

    func_name: str = self._register_function(row_test, len(func_cols))
    return self.select_where_sql(f'{func_name}({", ".join(quote_name(col) for col in func_cols)})')


def _select_by_dict(
        self: 'SqlDaf',
        selector_da:    T_da | List[T_da],          # a dict of column and value, or a list of them.
        expectmax:      int = -1,                   # if not -1, more matches than this raise LookupError.
        inverse:        bool = False,               # if True, the rows that do not match.
        keyfield:       str = '',                   # the keyfield of the selection. This one's if empty.
        ) -> 'SqlDaf':
    """
    Select the rows that match every field of a dict, or any one of a list of dicts.

    It works as `Daf.select_by_dict()` does, with the SQL written for you. A dict is an AND of
    equalities, and a list of dicts is an OR of those. With `inverse=True`, the rows that do
    not match. An empty dict matches every row, and an empty list none. A dict that names a
    column that is not there matches no row. Python NULL as a value matches a missing value.

    Values are compared with SQL's IS, which treats SQL NULL as a value, so the inverse keeps rows
    with a missing value. As in Python, 1 equals 1.0, and the text '1' does not equal 1.

    Nothing runs yet, unless `expectmax` is given: then its rows are found at once, to
    count the matches.

    Args:
        selector_da: A dict of column and value, or a list of such dicts.
        expectmax: If not -1, and more rows match, raise LookupError.
        inverse: If True, select the rows that do not match.
        keyfield: The keyfield of the selection. This SqlDaf's if empty.

    Returns:
        A selection.

    Raises:
        LookupError: More than `expectmax` rows match.
        TypeError: `selector_da` is not a dict, or a list or tuple of dicts.

    Examples:
        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> marks_sdaf = SqlDaf.from_daf(Daf(lol=[['b1', 1], ['b2', '']], cols=['ballot_id', 'v']), name='marks')
        >>> marks_sdaf.to_daf()
        <BLANKLINE>
        | ballot_id | v  |
        | --------: | -: |
        |        b1 |  1 |
        |        b2 |    |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield=''; name='marks'
        <BLANKLINE>
        >>> marks_sdaf.select_by_dict({'v': ''}).to_list()
        ['b2', '']
    """
    dicts_lda: List[T_da]
    if isinstance(selector_da, dict):
        dicts_lda = [selector_da]
    elif isinstance(selector_da, (list, tuple)) and all(isinstance(one_da, dict) for one_da in selector_da):
        dicts_lda = list(selector_da)
    else:
        raise TypeError("select_by_dict: selector_da must be a dict, or a list or tuple of dicts.")

    terms_ls: T_ls = []
    params_la: T_la = []
    one_da: T_da
    for one_da in dicts_lda:
        if any(col not in self.hd for col in one_da):
            terms_ls.append('0')                    # names a column that is not there: matches no row.
            continue
        if not one_da:
            terms_ls.append('1')
            continue
        conds_lt: List[Tuple[str, Tuple[Any, ...]]] = [is_value_sql(col, val) for col, val in one_da.items()]
        terms_ls.append('(' + ' AND '.join(cond_sql for cond_sql, _ in conds_lt) + ')')
        for _, cond_params in conds_lt:
            params_la.extend(cond_params)

    cond_sql: str = ' OR '.join(terms_ls) if terms_ls else '0'
    if inverse:
        cond_sql = f'NOT ({cond_sql})'
    sel_sdaf: SqlDaf = self.select_where_sql(cond_sql, params_la)
    if keyfield:
        sel_sdaf.set_keyfield(keyfield)
    if expectmax != -1 and len(sel_sdaf) > expectmax:
        raise LookupError(f"select_by_dict: more than {expectmax} rows match {selector_da}.")
    return sel_sdaf


def _split_where(
        self: 'SqlDaf',
        where:          Callable[[Any], Any],       # a function of a row that is true for the first side.
        *,
        indirect_col:   str | None = None,          # not supported yet.
        cols:           T_ls | None = None,         # the columns the function reads. All if None.
        ) -> Tuple['SqlDaf', 'SqlDaf']:
    """
    Split the rows in two, by a Python function that is true or false for each row.

    This is deprecated, as `Daf.split_where()` is. Use `group_where()` with a function that
    returns True or False.

    It works as `Daf.split_where()` does, with the rows staying in the database. It is two
    calls of `select_where()`: one keeps the rows where the function is true, and the other
    the rest. Nothing runs yet. Each side is found when it is first read, and the function is
    called again for each side that is read. So it should give the same answer each time.

    Name the columns the function reads in `cols`, as in `select_where()`. If the test is
    that columns equal values, two calls of `select_by_dict()`, the second with `inverse=True`,
    are much faster, since they call no function. On 1,000,000 rows, reading both sides took
    1.79 s here, and 0.21 s that way.

    Args:
        where: A function of a row, true for the rows of the first side.
        indirect_col: Not supported on a SqlDaf yet.
        cols: The columns the function reads. All if None.

    Returns:
        A tuple of the selection of the rows where the function is true and the selection of the others.

    Raises:
        KeyError: A name in `cols` is not a column.
        NotImplementedError: `indirect_col` is given.

    Examples:
        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> marks_sdaf = SqlDaf.from_daf(Daf(lol=[['b1', 10], ['b2', 20], ['b3', 30]], cols=['ballot_id', 'n']), name='marks')
        >>> marks_sdaf.to_daf()
        <BLANKLINE>
        | ballot_id | n  |
        | --------: | -: |
        |        b1 | 10 |
        |        b2 | 20 |
        |        b3 | 30 |
        <BLANKLINE>
        %% daf rows=3; cols=2; keyfield=''; name='marks'
        <BLANKLINE>
        >>> big_sdaf, small_sdaf = marks_sdaf.split_where(lambda row: row['n'] > 10, cols=['n'])
        >>> big_sdaf.to_daf().lol, small_sdaf.to_daf().lol
        ([['b2', 20], ['b3', 30]], [['b1', 10]])
    """
    return (self.select_where(where, indirect_col=indirect_col, cols=cols),
            self.select_where(lambda row: not where(row), indirect_col=indirect_col, cols=cols))


def _select_first_row_by_dict(self: 'SqlDaf', selector_da: T_da, inverse: bool = False) -> T_da:
    """
    Get the first row that matches every field of a dict, as `Daf.select_first_row_by_dict()` does.

    The matching rule is that of `select_by_dict()`. The search stops at the first match.

    Args:
        selector_da: The column names and the values they must have.
        inverse: If True, find the first row that does not match.

    Returns:
        The row as a dict, or an empty dict if none matches.
    """
    found_sdaf: SqlDaf = self.select_by_dict(selector_da, inverse=inverse)
    return found_sdaf.iloc(0) if found_sdaf.iloc_exists(0) else {}       # type: ignore[return-value]


def _select_krows(
        self: 'SqlDaf',
        krows:          Any,                        # a key, a list of keys, or a tuple of a first and last key.
        inverse:        bool = False,               # if True, the rows that are not selected.
        silent_error:   bool = False,               # if True, keys that are not found are ignored.
        ) -> 'SqlDaf':
    """
    Select rows by key, as `Daf.select_krows()` does.

    - A key, or a list or other iterable of keys, selects those rows, in the order of the keys.
      This is `select_records_daf()`.
    - A tuple of two keys selects every row from the first key to the last, both included, in
      row order. Either may be None, for the start or the end. A tuple of one key runs to the end.

    A key range is found by the place of its keys among the rows, not by row positions. So it
    works on a table that has had rows deleted.

    A slice is not supported. In a Daf, a slice here means row positions. Use `sdaf[n:m]`.

    Args:
        krows: A key, a list of keys, or a tuple of a first and last key.
        inverse: If True, select the rows that are not selected.
        silent_error: If True, keys that are not found are ignored. A key of a range must be found.

    Returns:
        A selection.

    Raises:
        KeyError: A key is not found.
        NotImplementedError: `krows` is a slice.
        TypeError: `krows` is None, or a tuple that is empty or longer than two.
        ValueError: There is no keyfield.
    """
    if krows is None:
        raise TypeError("None is not a valid key selector.")
    if isinstance(krows, slice):
        raise NotImplementedError("select_krows() with a slice is not supported on a SqlDaf. "
                                  "In a Daf it means row positions. Use sdaf[n:m].")
    if not isinstance(krows, tuple):
        return self.select_records_daf(krows, inverse=inverse, silent_error=silent_error)

    # A tuple is a range of keys, from the first to the last, in row order.
    if not 1 <= len(krows) <= 2 or krows == (None,):
        raise TypeError(f"select_krows: a key range is a tuple of one or two keys, not {krows!r}.")
    self._resolve()                                 # a child's place numbers stay fixed once found.
    key_col: str = self._need_keyfield('select_krows')
    rows_sql, rows_params = self._rows_sql()
    bounds_ls: List[str] = []
    bounds_params: Tuple[Any, ...] = ()
    op: str
    key: Any
    for op, key in zip(('>=', '<='), krows):
        if key is None:
            continue
        ord_t: Tuple[Any, ...] | None = self._execute(
                f'SELECT {ORD_COL} FROM ({rows_sql}) WHERE {quote_name(key_col)} = ?', rows_params + (key,)).fetchone()
        if ord_t is None:
            raise KeyError(key)
        bounds_ls.append(f'{ORD_COL} {op} ?')
        bounds_params += (ord_t[0],)
    range_sql: str = ' AND '.join(bounds_ls) or '1'
    if inverse:
        range_sql = f'NOT ({range_sql})'
    return SqlDaf._new_selection(self, f'SELECT * FROM ({rows_sql}) WHERE {range_sql}', rows_params + bounds_params)


def _select_records_daf(
        self: 'SqlDaf',
        keys_ls:        Any,                        # the keys: a list, an iterable, a Daf or a SqlDaf.
        inverse:        bool = False,               # if True, the rows whose key is not among them.
        silent_error:   bool = False,               # if True, keys that are not found are ignored.
        ) -> 'SqlDaf':
    """
    Select the rows with these keys, as `Daf.select_records_daf()` does.

    The keys may be a list or any iterable, an in-memory Daf, or another SqlDaf of the same
    database file. A SqlDaf's keys are used in SQL, as a join, so no key comes into Python. The other forms are written to a TEMP table first. The selected rows come in
    the order of the keys, as in a Daf. With `inverse=True`, the rows whose key is not among
    them, in this table's order.

    A Python NULL key gives undefined results: see SqlDaf.

    As in a Daf, a key that is not found raises KeyError, unless `silent_error` is True. That
    check runs at once, as one query. The selection itself runs when first read.

    Args:
        keys_ls: The keys: a list, an iterable, a Daf or a SqlDaf.
        inverse: If True, select the rows whose key is not among them.
        silent_error: If True, keys that are not found are ignored.

    Returns:
        A selection.

    Raises:
        KeyError: A key is not found, and `silent_error` is False.
        ValueError: This SqlDaf, or a SqlDaf given as the keys, has no keyfield.

    Examples:
        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> marks_daf = Daf(lol=[['b1', 1], ['b2', 0], ['b3', 1]], cols=['ballot_id', 'v'], keyfield='ballot_id')
        >>> marks_daf
        <BLANKLINE>
        | ballot_id | v  |
        | --------: | -: |
        |        b1 |  1 |
        |        b2 |  0 |
        |        b3 |  1 |
        <BLANKLINE>
        %% daf rows=3; cols=2; keyfield='ballot_id'; name=''
        <BLANKLINE>
        >>> marks_sdaf = SqlDaf.from_daf(marks_daf, name='marks')
        >>> marks_sdaf.select_records_daf(['b3', 'b1']).to_daf().to_lod()
        [{'ballot_id': 'b3', 'v': 1}, {'ballot_id': 'b1', 'v': 1}]
    """
    self._resolve_parent()
    key_col: str = self._need_keyfield('select_records_daf')
    keys_sql, keys_params = self._key_source_sql(keys_ls)
    if not silent_error:
        missing_key: Any = self._first_missing_key(keys_sql, keys_params)
        if missing_key is not None:
            raise KeyError(missing_key)

    rows_sql, rows_params = self._rows_sql()
    if inverse:
        return SqlDaf._new_selection(self,
                f'SELECT * FROM ({rows_sql}) AS r WHERE NOT EXISTS '
                f'(SELECT 1 FROM ({keys_sql}) AS k WHERE k.daf_key = r.{quote_name(key_col)})',
                rows_params + keys_params)
    return SqlDaf._new_selection(self,
            f'SELECT r.{REF_COL} AS {REF_COL}, k.{ORD_COL} AS {ORD_COL}, {self._cols_sql("r.")} '
            f'FROM ({rows_sql}) AS r JOIN ({keys_sql}) AS k ON r.{quote_name(key_col)} = k.daf_key',
            rows_params + keys_params)


def _calc_cols(
        self: 'SqlDaf',
        include_cols:   Iterable | None = None,
        exclude_cols:   Iterable | None = None,
        include_types:  List[Type] | None = None,
        exclude_types:  List[Type] | None = None,
        ) -> T_ls:
    """
    Choose column names by rules, as `Daf.calc_cols()` does.

    It reads only the column names and the dtypes, never the rows.

    Args:
        include_cols: Keep only these names.
        exclude_cols: Leave out these names.
        include_types: Keep only columns whose dtype is one of these.
        exclude_types: Leave out columns whose dtype is one of these.

    Returns:
        The chosen column names, in column order.
    """
    return Daf.calc_cols(self, include_cols=include_cols, exclude_cols=exclude_cols,          # type: ignore[return-value]
                         include_types=include_types, exclude_types=exclude_types)


def _select_cols(
        self: 'SqlDaf',
        cols:           T_ls | None = None,     # the columns to keep, in this order. All if empty.
        exclude_cols:   T_ls | None = None,     # the columns to leave out.
        ) -> 'SqlDaf':
    """
    Make a new table with only some columns, chosen by name, as `Daf.select_cols()` does.

    The columns come in the order of `cols`. A name given twice is used once. A name that is
    not a column raises KeyError. With no `cols`, all columns are kept, in this table's order.
    Then `exclude_cols` leaves out the columns you name. A name in `exclude_cols` that is not
    a column is ignored. The rows keep their order.

    The result is new rows, as in a Daf, not a selection of this table's rows. It is resolved
    the first time it is read, into a TEMP table, as a join is. The keyfield and dtypes carry
    over for the columns kept.

    Args:
        cols: The names of the columns to keep, in the order you want. If empty, all columns.
        exclude_cols: The names of the columns to leave out.

    Returns:
        A SqlDaf of the chosen columns, resolved when first read.

    Raises:
        KeyError: A name in `cols` is not a column.

    Examples:
        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> marks_sdaf = SqlDaf('marks', cols=['ballot_id', 'v', 'n'], lol=[['b1', 1, 10], ['b2', 0, 20]], keyfield='ballot_id')
        >>> marks_sdaf.to_daf()
        <BLANKLINE>
        | ballot_id | v  | n  |
        | --------: | -: | -: |
        |        b1 |  1 | 10 |
        |        b2 |  0 | 20 |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield='ballot_id'; name='marks'
        <BLANKLINE>
        >>> marks_sdaf.select_cols(['n', 'ballot_id']).to_daf().lol
        [[10, 'b1'], [20, 'b2']]
        >>> marks_sdaf.select_cols(exclude_cols=['v']).columns()
        ['ballot_id', 'n']
    """
    if isinstance(cols, str):
        cols = [cols]
    if isinstance(exclude_cols, str):
        exclude_cols = [exclude_cols]
    exclude_set: set = set(exclude_cols) if exclude_cols else set()
    col: str
    if cols:
        for col in cols:
            self.hd[col]                            # a KeyError for a name that is not a column.
        new_cols: T_ls = [col for col in dict.fromkeys(cols) if col not in exclude_set]
    else:
        new_cols = [col for col in self.hd if col not in exclude_set]

    rows_sql, rows_params = self._rows_sql()
    cols_sql: str = ''.join(f', {quote_name(col)}' for col in new_cols)
    root_name: str = self._root().name
    cols_sdaf: SqlDaf = SqlDaf.__new__(SqlDaf)
    cols_sdaf.conn           = self.conn
    cols_sdaf._conn_da       = self._conn_da
    cols_sdaf.db_path        = self.db_path
    cols_sdaf._name          = self._next_temp_name(root_name)
    cols_sdaf.keyfield       = self.keyfield if self.keyfield in new_cols else ''
    cols_sdaf.has_rid        = False
    cols_sdaf.hd             = {col: idx for idx, col in enumerate(new_cols)}
    cols_sdaf.dtypes         = {col: typ for col, typ in self.dtypes.items() if col in new_cols}
    cols_sdaf.is_temp        = True
    cols_sdaf.parent_sdaf    = None
    cols_sdaf.pending_sql    = (f'SELECT NULL AS {REF_COL}, ROW_NUMBER() OVER (ORDER BY {ORD_COL}) AS {ORD_COL}{cols_sql} '
                                f'FROM ({rows_sql})')
    cols_sdaf.pending_params = rows_params
    cols_sdaf.is_resolved    = False
    return cols_sdaf
