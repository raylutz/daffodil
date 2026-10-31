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
        Create a small SqlDaf table of people. It is stored on disk, so its size is not limited by memory:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> people_sdaf = SqlDaf('people', cols=['id', 'name', 'city', 'age'], keyfield='id', lol=[
        ...     ['p1', 'Ana Lopez', 'Denver', 34], ['p2', 'Ben Okafor', 'Austin', 52], ['p3', 'Chen Wu', 'Boston', 27]])
        >>> people_sdaf
        <BLANKLINE>
        | id |    name    |  city  | age |
        | -: | ---------: | -----: | --: |
        | p1 |  Ana Lopez | Denver |  34 |
        | p2 | Ben Okafor | Austin |  52 |
        | p3 |    Chen Wu | Boston |  27 |
        <BLANKLINE>
        %% daf rows=3; cols=4; keyfield='id'; name='people'
        <BLANKLINE>

        Select the people over 30 with a SQL condition, the value passed as a parameter:

        >>> people_sdaf.select_where_sql('age > ?', [30])
        <BLANKLINE>
        | id |    name    |  city  | age |
        | -: | ---------: | -----: | --: |
        | p1 |  Ana Lopez | Denver |  34 |
        | p2 | Ben Okafor | Austin |  52 |
        <BLANKLINE>
        %% daf rows=2; cols=4; keyfield='id'; name='daf_tmp_people_v1'
        <BLANKLINE>
    """
    self._resolve_parent()
    parent_sql, parent_params = self._rows_sql()
    return SqlDaf._new_selection(self, f'SELECT * FROM ({parent_sql}) WHERE {where_sql}',
                                 parent_params + null_to_none(params))


def _select_where(
        self: 'SqlDaf',
        where:          Callable[[Any], Any],       # a function of a row that is true for the rows to keep.
        indirect_col:   str | None = None,          # not supported on a SqlDaf.
        cols:           T_ls | None = None,         # the columns the function reads. All if None.
        ) -> 'SqlDaf':
    """
    Select the rows for which a Python function is true, as [`Daf.select_where()`][daffodil.daf_select._select_where] does.

    The function gets each row as a [KeyedList][daffodil.keyedlist.KeyedList], with Python NULL for
    a missing value, and reads cells by name, as in `row['n']`. It is registered with SQLite
    and called for each row when the selection is first read. If it raises, that error is
    raised, as itself.

    Name the columns it reads in `cols`. It is then given only those, which is faster. A row
    of every column costs about 1 us. If the test only compares columns to values, use
    `select_by_dict()` or `select_where_sql()`, which call no Python function.

    Args:
        where: A function of a row, true for the rows to keep.
        indirect_col: Not supported on a SqlDaf. It raises NotImplementedError.
        cols: The columns the function reads. All if None.

    Returns:
        A selection.

    Raises:
        KeyError: A name in `cols` is not a column.
        NotImplementedError: `indirect_col` is given.

    Examples:
        Create a small SqlDaf table of people:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> people_sdaf = SqlDaf('people', cols=['id', 'name', 'city', 'age'], keyfield='id', lol=[
        ...     ['p1', 'Ana Lopez', 'Denver', 34], ['p2', 'Ben Okafor', 'Austin', 52], ['p3', 'Chen Wu', 'Boston', 27]])
        >>> people_sdaf
        <BLANKLINE>
        | id |    name    |  city  | age |
        | -: | ---------: | -----: | --: |
        | p1 |  Ana Lopez | Denver |  34 |
        | p2 | Ben Okafor | Austin |  52 |
        | p3 |    Chen Wu | Boston |  27 |
        <BLANKLINE>
        %% daf rows=3; cols=4; keyfield='id'; name='people'
        <BLANKLINE>

        Select the people who live in Austin or Boston, with a Python function. `cols` names the columns the
        function reads, so only those are read:

        >>> people_sdaf.select_where(lambda row: row['city'] in ('Austin', 'Boston'), cols=['city'])
        <BLANKLINE>
        | id |    name    |  city  | age |
        | -: | ---------: | -----: | --: |
        | p2 | Ben Okafor | Austin |  52 |
        | p3 |    Chen Wu | Boston |  27 |
        <BLANKLINE>
        %% daf rows=2; cols=4; keyfield='id'; name='daf_tmp_people_v1'
        <BLANKLINE>
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

    It works as [`Daf.select_by_dict()`][daffodil.daf_select._select_by_dict] does, with the SQL written for you. A dict is an AND of
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
        Create a small SqlDaf table of people, one with no city given:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> people_sdaf = SqlDaf('people', cols=['id', 'name', 'city', 'age'], keyfield='id', lol=[
        ...     ['p1', 'Ana Lopez', 'Denver', 34], ['p2', 'Ben Okafor', 'Austin', 52],
        ...     ['p3', 'Chen Wu', 'Austin', 27], ['p4', 'Dana Smith', '', 41]])
        >>> people_sdaf
        <BLANKLINE>
        | id |    name    |  city  | age |
        | -: | ---------: | -----: | --: |
        | p1 |  Ana Lopez | Denver |  34 |
        | p2 | Ben Okafor | Austin |  52 |
        | p3 |    Chen Wu | Austin |  27 |
        | p4 | Dana Smith |        |  41 |
        <BLANKLINE>
        %% daf rows=4; cols=4; keyfield='id'; name='people'
        <BLANKLINE>

        Select the rows whose city is Austin:

        >>> people_sdaf.select_by_dict({'city': 'Austin'})
        <BLANKLINE>
        | id |    name    |  city  | age |
        | -: | ---------: | -----: | --: |
        | p2 | Ben Okafor | Austin |  52 |
        | p3 |    Chen Wu | Austin |  27 |
        <BLANKLINE>
        %% daf rows=2; cols=4; keyfield='id'; name='daf_tmp_people_v1'
        <BLANKLINE>

        Select the row with no city. Python NULL matches an empty cell:

        >>> people_sdaf.select_by_dict({'city': ''}).keys()
        ['p4']

        Select the rows that do not match, with `inverse=True`:

        >>> people_sdaf.select_by_dict({'city': 'Austin'}, inverse=True).keys()
        ['p1', 'p4']

        Give the selection another keyfield, here the name:

        >>> people_sdaf.select_by_dict({'city': 'Austin'}, keyfield='name').keys()
        ['Ben Okafor', 'Chen Wu']

        Check how many rows match, with `expectmax`. More than that raises:

        >>> people_sdaf.select_by_dict({'city': 'Austin'}, expectmax=1)
        Traceback (most recent call last):
        ...
        LookupError: select_by_dict: more than 1 rows match {'city': 'Austin'}.
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
        indirect_col:   str | None = None,          # not supported on a SqlDaf.
        cols:           T_ls | None = None,         # the columns the function reads. All if None.
        ) -> Tuple['SqlDaf', 'SqlDaf']:
    """
    Split the rows in two, by a Python function that is true or false for each row.

    This is deprecated, as [`Daf.split_where()`][daffodil.daf_select._split_where] is. Use `group_where()` with a function that
    returns True or False.

    It works as [`Daf.split_where()`][daffodil.daf_select._split_where] does, with the rows staying in the database. It is two
    calls of `select_where()`: one keeps the rows where the function is true, and the other
    the rest. Nothing runs yet. Each side is found when it is first read, and the function is
    called again for each side that is read. So it should give the same answer each time.

    Name the columns the function reads in `cols`, as in `select_where()`. If the test is
    that columns equal values, two calls of `select_by_dict()`, the second with `inverse=True`,
    are much faster, since they call no function. On 1,000,000 rows, reading both sides took
    1.79 s here, and 0.21 s that way.

    Args:
        where: A function of a row, true for the rows of the first side.
        indirect_col: Not supported on a SqlDaf. It raises NotImplementedError.
        cols: The columns the function reads. All if None.

    Returns:
        A tuple of the selection of the rows where the function is true and the selection of the others.

    Raises:
        KeyError: A name in `cols` is not a column.
        NotImplementedError: `indirect_col` is given.

    Examples:
        Create a small SqlDaf table of people:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> people_sdaf = SqlDaf('people', cols=['id', 'name', 'city', 'age'], keyfield='id', lol=[
        ...     ['p1', 'Ana Lopez', 'Denver', 34], ['p2', 'Ben Okafor', 'Austin', 52], ['p3', 'Chen Wu', 'Boston', 27]])

        Split the people into those over 30 and the rest, in one call:

        >>> older_sdaf, younger_sdaf = people_sdaf.split_where(lambda row: row['age'] > 30, cols=['age'])
        >>> older_sdaf
        <BLANKLINE>
        | id |    name    |  city  | age |
        | -: | ---------: | -----: | --: |
        | p1 |  Ana Lopez | Denver |  34 |
        | p2 | Ben Okafor | Austin |  52 |
        <BLANKLINE>
        %% daf rows=2; cols=4; keyfield='id'; name='daf_tmp_people_v1'
        <BLANKLINE>
        >>> younger_sdaf
        <BLANKLINE>
        | id |  name   |  city  | age |
        | -: | ------: | -----: | --: |
        | p3 | Chen Wu | Boston |  27 |
        <BLANKLINE>
        %% daf rows=1; cols=4; keyfield='id'; name='daf_tmp_people_v2'
        <BLANKLINE>
    """
    return (self.select_where(where, indirect_col=indirect_col, cols=cols),
            self.select_where(lambda row: not where(row), indirect_col=indirect_col, cols=cols))


def _select_first_row_by_dict(self: 'SqlDaf', selector_da: T_da, inverse: bool = False) -> T_da:
    """
    Get the first row that matches every field of a dict, as [`Daf.select_first_row_by_dict()`][daffodil.daf_select._select_first_row_by_dict] does.

    The matching rule is that of `select_by_dict()`. The search stops at the first match.

    Args:
        selector_da: The column names and the values they must have.
        inverse: If True, find the first row that does not match.

    Returns:
        The row as a dict, or an empty dict if none matches.

    Examples:
        Create a small SqlDaf table of people:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> people_sdaf = SqlDaf('people', cols=['id', 'name', 'city', 'age'], keyfield='id', lol=[
        ...     ['p1', 'Ana Lopez', 'Denver', 34], ['p2', 'Ben Okafor', 'Austin', 52], ['p3', 'Chen Wu', 'Austin', 27]])

        Find the first person in Austin, as a dict:

        >>> people_sdaf.select_first_row_by_dict({'city': 'Austin'})
        {'id': 'p2', 'name': 'Ben Okafor', 'city': 'Austin', 'age': 52}

        Find the first person not in Austin, with `inverse=True`:

        >>> people_sdaf.select_first_row_by_dict({'city': 'Austin'}, inverse=True)
        {'id': 'p1', 'name': 'Ana Lopez', 'city': 'Denver', 'age': 34}

        No match gives an empty dict:

        >>> people_sdaf.select_first_row_by_dict({'city': 'Miami'})
        {}
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
    Select rows by key, as [`Daf.select_krows()`][daffodil.daf_select._select_krows] does.

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

    Examples:
        Create a small SqlDaf table of people:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> people_sdaf = SqlDaf('people', cols=['id', 'name', 'age'], keyfield='id', lol=[
        ...     ['p1', 'Ana Lopez', 34], ['p2', 'Ben Okafor', 52], ['p3', 'Chen Wu', 27], ['p4', 'Dana Smith', 41]])

        Select by a list of keys, in the order given:

        >>> people_sdaf.select_krows(['p3', 'p1'])
        <BLANKLINE>
        | id |   name    | age |
        | -: | --------: | --: |
        | p3 |   Chen Wu |  27 |
        | p1 | Ana Lopez |  34 |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield='id'; name='daf_tmp_people_v2'
        <BLANKLINE>

        Select a range of keys, in row order. One end may be left open:

        >>> people_sdaf.select_krows(('p2', 'p3')).keys()
        ['p2', 'p3']
        >>> people_sdaf.select_krows(('p3',)).keys()
        ['p3', 'p4']
        >>> people_sdaf.select_krows((None, 'p2')).keys()
        ['p1', 'p2']

        Select the rows not listed, with `inverse=True`:

        >>> people_sdaf.select_krows(['p1', 'p2'], inverse=True).keys()
        ['p3', 'p4']

        Ignore keys that are not found, with `silent_error=True`:

        >>> people_sdaf.select_krows(['p1', 'p9'], silent_error=True).keys()
        ['p1']
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
    Select the rows with these keys, as [`Daf.select_records_daf()`][daffodil.daf_select._select_records_daf] does.

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
        Create a small SqlDaf table of people, from a Daf in memory:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> people_daf = Daf(cols=['id', 'name', 'age'], keyfield='id', lol=[
        ...     ['p1', 'Ana Lopez', 34], ['p2', 'Ben Okafor', 52], ['p3', 'Chen Wu', 27]])
        >>> people_sdaf = SqlDaf.from_daf(people_daf, name='people')

        Select the rows with these keys, in the order given:

        >>> people_sdaf.select_records_daf(['p3', 'p1'])
        <BLANKLINE>
        | id |   name    | age |
        | -: | --------: | --: |
        | p3 |   Chen Wu |  27 |
        | p1 | Ana Lopez |  34 |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield='id'; name='daf_tmp_people_v2'
        <BLANKLINE>

        Select the rows whose keys are not listed, with `inverse=True`:

        >>> people_sdaf.select_records_daf(['p1'], inverse=True).keys()
        ['p2', 'p3']

        A key that is not found raises. With `silent_error=True` it is ignored:

        >>> people_sdaf.select_records_daf(['p1', 'p9'], silent_error=True).keys()
        ['p1']

        The keys may come from another table, as its keyfield:

        >>> visits_sdaf = SqlDaf('visits', cols=['id', 'visits'], keyfield='id', lol=[['p2', 7], ['p3', 12]])
        >>> people_sdaf.select_records_daf(visits_sdaf).keys()
        ['p2', 'p3']
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
    Choose column names by rules, as [`Daf.calc_cols()`][daffodil.daf_structure._calc_cols] does.

    It reads only the column names and the dtypes, never the rows.

    Args:
        include_cols: Keep only these names.
        exclude_cols: Leave out these names.
        include_types: Keep only columns whose dtype is one of these.
        exclude_types: Leave out columns whose dtype is one of these.

    Returns:
        The chosen column names, in column order.

    Examples:
        Create a small SqlDaf table of people, with a type for each column:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> people_sdaf = SqlDaf('people', cols=['id', 'name', 'age', 'height_m'], keyfield='id',
        ...     dtypes={'id': str, 'name': str, 'age': int, 'height_m': float}, lol=[['p1', 'Ana Lopez', 34, 1.68]])

        Choose the columns of numbers, by type:

        >>> people_sdaf.calc_cols(include_types=[int, float])
        ['age', 'height_m']

        Choose every column that is not text:

        >>> people_sdaf.calc_cols(exclude_types=[str])
        ['age', 'height_m']

        Choose from a list of names, leaving some out:

        >>> people_sdaf.calc_cols(include_cols=['id', 'age', 'height_m'], exclude_cols=['height_m'])
        ['id', 'age']
    """
    return Daf.calc_cols(self, include_cols=include_cols, exclude_cols=exclude_cols,          # type: ignore[return-value]
                         include_types=include_types, exclude_types=exclude_types)


def _select_cols(
        self: 'SqlDaf',
        cols:           T_ls | None = None,     # the columns to keep, in this order. All if empty.
        exclude_cols:   T_ls | None = None,     # the columns to leave out.
        ) -> 'SqlDaf':
    """
    Make a new table with only some columns, chosen by name, as [`Daf.select_cols()`][daffodil.daf_select._select_cols] does.

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
        Create a small SqlDaf table of people:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> people_sdaf = SqlDaf('people', cols=['id', 'name', 'city', 'age'], keyfield='id', lol=[
        ...     ['p1', 'Ana Lopez', 'Denver', 34], ['p2', 'Ben Okafor', 'Austin', 52]])

        Select some columns, in the order given:

        >>> people_sdaf.select_cols(['age', 'name'])
        <BLANKLINE>
        | age |    name    |
        | --: | ---------: |
        |  34 |  Ana Lopez |
        |  52 | Ben Okafor |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield=''; name='daf_tmp_people_v1'
        <BLANKLINE>

        Select every column but some:

        >>> people_sdaf.select_cols(exclude_cols=['city']).columns()
        ['id', 'name', 'age']
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
