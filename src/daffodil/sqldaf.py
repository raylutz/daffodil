# sqldaf.py
#
# A Daf whose rows live in a SQLite table, and the store that owns the database.
# The design and the decisions behind it are in notes/sql_backing/design_draft.md.

import os
import csv
import sqlite3
import tempfile
import contextlib

from typing import Any, Callable, Dict, Iterator, List, Tuple, Type, TypeAlias      # noqa: F401
from collections.abc import Iterable

from daffodil.daf import Daf, NULL, DaffodilError
from daffodil.keyedlist import KeyedList
from daffodil.lib.daf_types import T_ls, T_la, T_lola, T_di, T_da, T_dtype_dict
import daffodil.lib.daf_utils as daf_utils


RID_COL = 'daf_rid'                 # the INTEGER PRIMARY KEY of each table SqlStore makes
REF_COL = 'daf_ref'                 # the reference to a base row, in the queries of a selection
ORD_COL = 'daf_ord'                 # the order of the rows, in the queries of a selection
RESERVED_COLS = (RID_COL, REF_COL, ORD_COL)
TEMP_PREFIX = 'daf_tmp_'            # the prefix of every TEMP object SqlStore makes
INDEX_PREFIX = 'daf_ix_'            # the prefix of every index SqlStore makes

# dtypes that apply_dtypes() can store. A list, dict and the like would need JSON text, not yet done.
STORABLE_DTYPES = (int, float, bool, str)


class SqlNotSupportedError(DaffodilError, NotImplementedError):
    """
    A Daf method was called on a SqlDaf that has no SQL version of it yet.

    The message names the method. Use `SqlDaf.sql()` instead, or pull the rows into memory
    with `SqlDaf.to_daf()` first.
    """


# ---- the parts that differ between engines (decision 89) ----

def quote_name(name: str) -> str:
    """
    Quote a table or column name for SQLite.

    The name is always quoted, so any text is a valid name, and an embedded double quote is
    doubled. Postgres quotes the same way. MySQL would use backticks.
    """
    return '"' + name.replace('"', '""') + '"'


def placeholders(count: int) -> str:
    """ The parameter marks for `count` values, as in `?, ?, ?`. Postgres and MySQL use %s. """
    return ', '.join(['?'] * count)


def null_to_none(row_la: Iterable[Any]) -> Tuple[Any, ...]:
    """ A row on its way into SQL: NULL becomes None. """
    return tuple(None if val is NULL else val for val in row_la)


def none_to_null(row_t: Tuple[Any, ...]) -> T_la:
    """ A row on its way out of SQL: None becomes NULL. """
    return [NULL if val is None else val for val in row_t]


class SqlStore:
    """
    One SQLite database, and the tables and selections made in it.

    A [SqlDaf][daffodil.sqldaf.SqlDaf] always belongs to a store. Every table it makes or
    selects from is in the store's database, so tables can be joined without attaching
    another file.

    With a path, the database is that file, and its tables are kept after `close()`. With no
    path, the database is a new temporary file, which `close()` deletes. With `conn`, the
    store uses a connection you already have, and `close()` leaves it open. That is how an
    existing database, such as one of AuditEngine's, is used.

    A store owns one connection, and sqlite3 lets a connection be used only by the thread
    that made it. So a store belongs to one thread. Another thread opens its own store on
    the same file.

    Outside `transaction()`, each statement is committed when it finishes, and locks and
    unlocks the file by itself. Use `transaction()` around a run of many small statements,
    such as lookups by key: it halves their cost.

    Use the store as a context manager to close it on the way out.

    Examples:
        >>> with SqlStore() as store:
        ...     marks_sdaf = store.from_daf(Daf(lol=[['b1', 1], ['b2', '']], cols=['ballot_id', 'v']), name='marks')
        ...     len(marks_sdaf)
        2
    """

    def __init__(
            self,
            path:   str = '',                           # the database file. '' makes a temporary file.
            conn:   sqlite3.Connection | None = None,   # an open connection to use instead of a path.
            ) -> None:
        """
        Open a store.

        Args:
            path: The database file, made if it does not exist. Without one, a temporary file is used.
            conn: An open connection to use instead. It is not closed by `close()`.

        Raises:
            ValueError: Both `path` and `conn` were given.
        """
        if path and conn is not None:
            raise ValueError("SqlStore: give a path or a conn, not both.")

        self.temp_path:     str     = ''                # the temporary file to delete at close, if any.
        self.owns_conn:     bool    = conn is None
        self.temp_count:    int     = 0                 # counter for the names of TEMP objects.
        self.savepoint_count: int   = 0                 # counter for the names of nested transactions.
        self.func_count:    int     = 0                 # counter for the names of registered functions.
        self.func_error:    BaseException | None = None # an error raised inside a registered function.

        if conn is not None:
            self.conn: sqlite3.Connection = conn
        else:
            if not path:
                fd, path = tempfile.mkstemp(prefix='daf_store_', suffix='.db')
                os.close(fd)
                self.temp_path = path
            self.conn = sqlite3.connect(path, isolation_level=None)

        self.path: str = path


    def __enter__(self) -> 'SqlStore':
        return self


    def __exit__(self, *exc_info: Any) -> None:
        self.close()


    def close(self) -> None:
        """
        Close the store.

        TEMP objects go when the connection closes. A temporary database file is deleted. A
        connection passed in as `conn` is left open, and its TEMP objects stay until it closes.
        Closing twice does nothing more.
        """
        if self.owns_conn:
            self.conn.close()
        if self.temp_path:
            os.remove(self.temp_path)
            self.temp_path = ''


    @contextlib.contextmanager
    def transaction(self) -> Iterator[None]:
        """
        Run the statements inside the `with` block as one transaction.

        Everything in the block is committed together when it ends, or undone together if it
        raises. It also makes a run of small statements faster, because the file is locked once
        for the block instead of once per statement. On 1,000,000 rows, a lookup by key took
        12.4 us inside a transaction, against 24.1 us outside.

        While the block is open, another process cannot write to the database file. So keep it
        to the work that needs it.

        A `transaction()` inside another one becomes a savepoint: if the inner block raises, only
        its own work is undone.

        Examples:
            >>> with SqlStore() as store:
            ...     marks_sdaf = store.from_daf(Daf(lol=[['b1', 1]], cols=['ballot_id', 'v']), name='marks')
            ...     with store.transaction():
            ...         _ = marks_sdaf.sql("UPDATE marks SET v = 2")
            ...     marks_sdaf.to_daf().lol
            [['b1', 2]]
        """
        if not self.conn.in_transaction:
            self.conn.execute('BEGIN')
            try:
                yield
            except BaseException:
                self.conn.execute('ROLLBACK')
                raise
            self.conn.execute('COMMIT')
        else:
            self.savepoint_count += 1
            savepoint: str = quote_name(f'daf_sp_{self.savepoint_count}')
            self.conn.execute(f'SAVEPOINT {savepoint}')
            try:
                yield
            except BaseException:
                self.conn.execute(f'ROLLBACK TO {savepoint}')
                self.conn.execute(f'RELEASE {savepoint}')
                raise
            self.conn.execute(f'RELEASE {savepoint}')


    def table(
            self,
            name:       str,                            # the name of the table.
            keyfield:   str = '',                       # the column that holds the key of each row.
            dtypes:     T_dtype_dict | None = None,     # the types of the columns, for apply_dtypes().
            ) -> 'SqlDaf':
        """
        Get a SqlDaf for a table that is already in the database.

        Args:
            name: The name of the table.
            keyfield: The column that holds the key of each row, if any.
            dtypes: The types of the columns, kept for `apply_dtypes()`. Nothing is converted.

        Returns:
            A SqlDaf for the table. Nothing is read yet.

        Raises:
            KeyError: There is no table of that name, or the keyfield is not a column.
        """
        return SqlDaf(store=self, name=name, keyfield=keyfield, dtypes=dtypes)


    def table_exists(self, name: str) -> bool:
        """ Say whether the database has a table or view of this name, TEMP ones included. """
        found_t: Tuple[Any, ...] | None = self.conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name = ? UNION ALL SELECT 1 FROM sqlite_temp_master WHERE name = ?",
                (name, name)).fetchone()
        return found_t is not None


    def drop_table(self, name: str, missing_ok: bool = False) -> None:
        """
        Remove a table from the database.

        A selection that still refers to the table is not checked. Reading it afterwards fails.

        Args:
            name: The name of the table.
            missing_ok: If True, a table that is not there is not an error.

        Raises:
            KeyError: There is no table of that name, and `missing_ok` is False.
        """
        if not self.table_exists(name):
            if missing_ok:
                return
            raise KeyError(name)
        self.conn.execute(f'DROP TABLE {quote_name(name)}')


    def _create_table_sql(self, name: str, cols: T_ls) -> str:
        """ Check a new table, and give the SQL that makes it. Internal use. """
        if self.table_exists(name):
            raise ValueError(f"SqlStore: a table named '{name}' already exists.")
        reserved_ls: T_ls = [col for col in cols if col in RESERVED_COLS]
        if reserved_ls:
            raise ValueError(f"SqlStore: {reserved_ls} reserved, and cannot be column names: {list(RESERVED_COLS)}.")

        coldefs_ls: T_ls = [f'{quote_name(RID_COL)} INTEGER PRIMARY KEY'] + [quote_name(col) for col in cols]
        return f'CREATE TABLE {quote_name(name)} ({", ".join(coldefs_ls)})'


    def create_table(
            self,
            name:       str,                            # the name of the new table.
            cols:       T_ls,                           # its column names, in order.
            keyfield:   str = '',                       # the column that holds the key of each row.
            dtypes:     T_dtype_dict | None = None,     # the types of the columns, for apply_dtypes().
            ) -> 'SqlDaf':
        """
        Make an empty table, and return a SqlDaf for it.

        The table gets a hidden first column, `daf_rid`, an INTEGER PRIMARY KEY. It numbers the
        rows in the order they are added, keeps that order, and does not change when the file
        is vacuumed. It is not one of the `columns()`, and `to_daf()` leaves it out.

        The columns are declared with no SQL type, so SQLite keeps each value as it is given:
        an int stays an int, and text stays text. `apply_dtypes()` converts them, as in a Daf.

        A cell that holds a list or a dict cannot be stored yet. sqlite3 raises an error for it.

        Args:
            name: The name of the new table.
            cols: The column names, in order.
            keyfield: The column that holds the key of each row. Its index is made when a key is first needed.
            dtypes: The types of the columns, kept for `apply_dtypes()`. Nothing is converted.

        Returns:
            A SqlDaf for the empty table.

        Raises:
            ValueError: A table of that name exists, or a column is named daf_rid.
            KeyError: The keyfield is not one of the columns.
        """
        if keyfield and keyfield not in cols:
            raise KeyError(keyfield)
        self.conn.execute(self._create_table_sql(name, cols))
        return SqlDaf(store=self, name=name, keyfield=keyfield, dtypes=dtypes)


    def insert_rows(self, name: str, cols: T_ls, rows: Iterable[Iterable[Any]]) -> int:
        """
        Add rows to a table, in one transaction, and return how many were added.

        The rows are read from `rows` as they are needed, so an iterator of any length works.
        A NULL cell is stored as SQL NULL. If any row fails, none of them are added, and the
        error is raised. A key that is repeated fails only if the table has a unique index on
        its keyfield, as it does once a key has been used.

        Args:
            name: The table.
            cols: The columns the values of each row go to, in order.
            rows: The rows, each the values in the order of `cols`.

        Returns:
            The number of rows added.
        """
        stmt: str = (f'INSERT INTO {quote_name(name)} ({", ".join(quote_name(col) for col in cols)}) '
                     f'VALUES ({placeholders(len(cols))})')
        count: int = 0

        def counted(rows: Iterable[Iterable[Any]]) -> Iterator[Tuple[Any, ...]]:
            nonlocal count
            row_la: Iterable[Any]
            for row_la in rows:
                count += 1
                yield null_to_none(row_la)

        with self.transaction():
            self.conn.executemany(stmt, counted(rows))
        return count


    def _load_table(
            self,
            name:       str,                            # the name of the new table.
            cols:       T_ls,                           # its column names, in order.
            keyfield:   str,                            # the column that holds the key of each row, or ''.
            dtypes:     T_dtype_dict | None,            # the types of the columns, for apply_dtypes().
            rows:       Iterable[Tuple[Any, ...]],      # the rows, with None for a missing value.
            ) -> 'SqlDaf':
        """ Make a table and add the rows, in one transaction. If any row fails, nothing is made. Internal use. """
        if keyfield and keyfield not in cols:
            raise KeyError(keyfield)
        table_sql: str = self._create_table_sql(name, cols)
        insert_sql: str = (f'INSERT INTO {quote_name(name)} ({", ".join(quote_name(col) for col in cols)}) '
                           f'VALUES ({placeholders(len(cols))})')
        with self.transaction():
            self.conn.execute(table_sql)
            self.conn.executemany(insert_sql, rows)
        return SqlDaf(store=self, name=name, keyfield=keyfield, dtypes=dtypes)


    def from_csv(
            self,
            path:       str,                            # the CSV file.
            name:       str = '',                       # the table name. The file name without .csv if empty.
            keyfield:   str = '',                       # the column that holds the key of each row.
            delimiter:  str = ',',                      # the field separator.
            ) -> 'SqlDaf':
        """
        Load a CSV file into a new table, without building it in memory.

        This only reads, as `Daf.from_csv()` does without dtypes. Every value is stored as the
        text of the file. Call `apply_dtypes()` afterwards to convert the columns you need.

        The first line gives the column names. The rest is read one row at a time, so memory use
        does not grow with the file. An empty cell is stored as SQL NULL, and comes back as NULL.
        A row shorter than the header is padded with NULL, and a longer one raises ValueError.
        The table and its rows are made in one transaction. If any row fails, no table is made.

        The keyfield is only recorded. Its unique index is made when a key is first used, as a
        Daf builds its key index. A repeated key raises then, not here.

        Args:
            path: The CSV file.
            name: The name of the new table. The file name without its extension if empty.
            keyfield: The column that holds the key of each row.
            delimiter: The field separator.

        Returns:
            A SqlDaf for the new table.

        Raises:
            ValueError: A table of that name exists, a column is named daf_rid, or a row is longer than the header.
            KeyError: The keyfield is not one of the columns.
        """
        name = name or os.path.splitext(os.path.basename(path))[0]

        with open(path, newline='', encoding='utf-8') as csv_f:
            reader: Iterator[T_ls] = csv.reader(csv_f, delimiter=delimiter)
            header_ls: T_ls = next(reader, [])
            cols: T_ls = Daf(cols=header_ls).columns()          # empty and repeated names made unique, as in Daf.
            width: int = len(cols)

            def fitted(reader: Iterator[T_ls]) -> Iterator[Tuple[Any, ...]]:
                row_ls: T_ls
                for row_ls in reader:
                    if len(row_ls) > width:
                        raise ValueError(f"SqlStore.from_csv: a row of {path} has {len(row_ls)} values, "
                                         f"more than the {width} columns.")
                    yield tuple([None if val == '' else val for val in row_ls] + [None] * (width - len(row_ls)))

            return self._load_table(name, cols, keyfield, None, fitted(reader))


    def from_daf(self, data_daf: Daf, name: str, keyfield: str = '') -> 'SqlDaf':
        """
        Write an in-memory Daf into a new table.

        Each value is stored as it is, so an int stays an int. A NULL cell is stored as SQL NULL.
        The new SqlDaf keeps the Daf's dtypes, for `apply_dtypes()`. The keyfield is the Daf's,
        unless one is given. As in `from_csv()`, its index is made when a key is first used, and
        if any row fails, no table is made.

        Args:
            data_daf: The Daf to write. It is not changed.
            name: The name of the new table.
            keyfield: The column that holds the key of each row. The Daf's keyfield if empty.

        Returns:
            A SqlDaf for the new table.

        Raises:
            ValueError: A table of that name exists, or a column is named daf_rid.
            KeyError: The keyfield is not one of the columns.
        """
        keyfield = keyfield or (data_daf.keyfield if isinstance(data_daf.keyfield, str) else '')
        dtypes: T_dtype_dict | None = data_daf.dtypes if isinstance(data_daf.dtypes, dict) and data_daf.dtypes else None
        row_la: T_la
        return self._load_table(name, data_daf.columns(), keyfield, dtypes,
                                (null_to_none(row_la) for row_la in data_daf.lol))


    def next_temp_name(self, parent: str) -> str:
        """ A new name for a TEMP object made from `parent`, as in daf_tmp_marks_v3. """
        self.temp_count += 1
        return f'{TEMP_PREFIX}{parent}_v{self.temp_count}'


    def register_function(self, func: Callable[..., Any], nargs: int) -> str:
        """
        Register a Python function on the connection, and return the name to call it by in SQL.

        If the function raises, SQLite only reports that a function failed. So the error is kept,
        and `execute()` raises it instead, with its own traceback.

        Args:
            func: The function.
            nargs: The number of arguments it takes.

        Returns:
            Its SQL name, as in daf_fn_3.
        """
        self.func_count += 1
        func_name: str = f'daf_fn_{self.func_count}'

        def guarded(*args: Any) -> Any:
            try:
                return func(*args)
            except BaseException as exc_info:
                self.func_error = exc_info
                raise

        self.conn.create_function(func_name, nargs, guarded)
        return func_name


    def execute(self, stmt: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        """
        Run one statement on the connection, and return the cursor.

        It is `conn.execute()`, except that an error raised inside a registered Python function
        is raised as itself, not as the general sqlite3.OperationalError. Values are passed as
        they are. Convert NULL first with `null_to_none()` if they may hold it.

        Args:
            stmt: One SQL statement.
            params: Values for its `?` marks.

        Returns:
            The cursor, ready to fetch from.
        """
        self.func_error = None
        try:
            return self.conn.execute(stmt, tuple(params))
        except sqlite3.OperationalError as exc_info:
            if self.func_error is not None:
                raise self.func_error from exc_info
            raise


    def fetch(self, cursor: sqlite3.Cursor) -> List[Tuple[Any, ...]]:
        """ Fetch every row of a cursor from `execute()`, raising a registered function's error as itself. """
        try:
            return cursor.fetchall()
        except sqlite3.OperationalError as exc_info:
            if self.func_error is not None:
                raise self.func_error from exc_info
            raise


class SqlDaf(Daf):
    """
    A Daf whose rows live in a SQLite table.

    Use it for a table too large to hold in memory. The rows stay on disk in the database of a
    [SqlStore][daffodil.sqldaf.SqlStore], and SQL does the work. When the rows you need are
    few, pull them into an ordinary Daf with `to_daf()`, and work on them there.

    A SqlDaf is a Daf, so `isinstance(sdaf, Daf)` is True. But it has only the methods
    written for SQL. Every other Daf method raises SqlNotSupportedError, which names it. So
    nothing quietly loads a large table into memory. For anything not covered yet, `sql()`
    runs a SQL statement of your own.

    A SqlDaf is either a base table or a selection.

    - A base table holds the rows. Get one from the store: `store.table()`, `store.from_csv()`,
      `store.from_daf()` or `store.create_table()`.
    - A selection holds references to rows of a base table, in order, as a selection of a Daf
      holds the same row lists. `select_where()`, `select_by_dict()`, `select_where_sql()` and
      indexing, as in `sdaf[2:5]` or `sdaf['b2']`, each make one. A selection of a selection
      is a selection of the same base table.

    A selection runs nothing when it is made. Its rows are found, and fixed, the first time it
    is read: by `to_daf()`, `len()`, `bool()`, `to_dict()`, `to_list()` or `to_value()`. After
    that, it always means the same rows, as a selection of a Daf does. A change to the values
    in those rows shows in it, since it refers to them. A selection made from one that is not
    fixed yet is computed from the same conditions, all in one query.

    Do not delete rows from a base table that has selections, or that you read by position.
    Positions use daf_rid, which then has gaps, and `len()` of a selection counts its references,
    so it would still count a deleted row. Leave rows out with a selection instead, as in a Daf.
    `to_daf()` does leave a deleted row out.

    To read one row, use `select_record()` by key or `iloc()` by position. Each is one query,
    returns a dict, and makes no selection.

    The methods so far:

    - `len()`, `bool()`, `columns()` and `num_cols()`.
    - `select_where()`, `select_by_dict()`, `select_where_sql()` and indexing, which select.
    - `select_record()` and `iloc()`, which read one row.
    - `to_daf()`, `to_dict()`, `to_list()` and `to_value()`, which read rows into memory.
    - `materialize()` and `save_as()`, which copy rows into a new table.
    - `apply_dtypes()`, `set_keyfield()` and `ensure_key_index()`.
    - `sql()`, to run any SQL statement.

    Work with large tables in whole statements. One query costs about 12 to 24 us, against
    under 1 us for a lookup in memory, because each one passes through sqlite3. Many rows in one
    statement are far cheaper per row.

    A missing value is NULL in Python, as in any Daf, and SQL NULL in the database. It is
    converted on the way in and on the way out. So in Python test a value with `val is NULL`,
    and in SQL write `col IS NULL`. Do not compare with ''.

    Until `apply_dtypes()` converts them, values loaded from a CSV file are text, in SQL too.
    Text is never less than a number in SQLite, so `WHERE n < 5` finds nothing on a text
    column, and MAX() gives the last in text order.

    The `name` is the name of the table in the database. A selection gets a generated name,
    such as daf_tmp_marks_v3, which names its reference table once it is fixed. It cannot be
    changed.

    Examples:
        >>> with SqlStore() as store:
        ...     marks_sdaf = store.from_daf(Daf(lol=[['b1', 1], ['b2', '']], cols=['ballot_id', 'v']), name='marks')
        ...     marks_sdaf.sql('SELECT ballot_id FROM marks WHERE v IS NULL').to_list()
        ['b2']
    """

    def __init__(
            self,
            store:      SqlStore,                       # the store that owns the table.
            name:       str,                            # the name of the table.
            keyfield:   str = '',                       # the column that holds the key of each row.
            dtypes:     T_dtype_dict | None = None,     # the types of the columns, for apply_dtypes().
            ) -> None:
        """
        Make a SqlDaf for a base table in the store.

        Usually you get one from the store instead, as with `store.table()`.

        Args:
            store: The store that owns the table.
            name: The name of the table.
            keyfield: The column that holds the key of each row, if any.
            dtypes: The types of the columns, kept for `apply_dtypes()`. Nothing is converted.

        Raises:
            KeyError: There is no table of that name, or the keyfield is not a column.
        """
        info_lot: List[Tuple[Any, ...]] = store.conn.execute(f'PRAGMA table_info({quote_name(name)})').fetchall()
        if not info_lot:
            raise KeyError(name)

        all_cols: T_ls = [info_t[1] for info_t in info_lot]
        cols: T_ls = [col for col in all_cols if col != RID_COL]
        if keyfield and keyfield not in cols:
            raise KeyError(keyfield)

        self.store:     SqlStore     = store
        self._name:     str          = name
        self.keyfield:  str          = keyfield
        self.has_rid:   bool         = RID_COL in all_cols
        self.hd:        T_di         = {col: idx for idx, col in enumerate(cols)}
        self.dtypes:    T_dtype_dict = dict(dtypes) if dtypes else {}

        # A base table has no base. A selection has its base table, and either the query that
        # finds its rows, not run yet, or the name of the reference table that holds them.
        self.base_sdaf:     SqlDaf | None   = None
        self.pending_sql:   str             = ''
        self.pending_params: Tuple[Any, ...] = ()
        self.is_fixed:      bool            = True


    @classmethod
    def _new_selection(
            cls,
            parent_sdaf:    'SqlDaf',                   # the SqlDaf the rows are selected from.
            pending_sql:    str,                        # the query that finds the rows, with daf_ref and daf_ord.
            pending_params: Tuple[Any, ...],            # its parameters.
            keyfield:       str | None = None,          # the keyfield of the selection. The parent's if None.
            ) -> 'SqlDaf':
        """ Make a selection, not fixed yet. Internal use. """
        base_sdaf: SqlDaf = parent_sdaf.base_sdaf or parent_sdaf
        base_sdaf._ref_col()                            # raises if the base has no reference that survives VACUUM.
        sel_sdaf: SqlDaf = cls.__new__(cls)
        sel_sdaf.store          = parent_sdaf.store
        sel_sdaf._name          = parent_sdaf.store.next_temp_name(base_sdaf.name)
        sel_sdaf.keyfield       = parent_sdaf.keyfield if keyfield is None else keyfield
        sel_sdaf.has_rid        = False
        sel_sdaf.hd             = dict(parent_sdaf.hd)
        sel_sdaf.dtypes         = dict(parent_sdaf.dtypes)
        sel_sdaf.base_sdaf      = base_sdaf
        sel_sdaf.pending_sql    = pending_sql
        sel_sdaf.pending_params = pending_params
        sel_sdaf.is_fixed       = False
        return sel_sdaf


    @property
    def name(self) -> str:              # type: ignore[override]
        """ The name of the table in the database. It cannot be changed. """
        return self._name


    @property
    def conn(self) -> sqlite3.Connection:
        """ The store's connection. """
        return self.store.conn


    def __repr__(self) -> str:
        kind: str = 'selection' if self.base_sdaf is not None else 'table'
        return f"<SqlDaf {kind} name='{self.name}' rows={len(self)} cols={len(self.hd)} keyfield='{self.keyfield}'>"


    __str__ = __repr__


    # ---- the SQL that gives the rows ----

    def _cols_sql(self, prefix: str = '') -> str:
        """ The columns, quoted, in order, without daf_rid. Internal use. """
        return ', '.join(prefix + quote_name(col) for col in self.hd)


    def _ref_col(self) -> str:
        """
        The column of this base table that a selection refers to rows by. Internal use.

        daf_rid if the table has it. Otherwise the keyfield, with its unique index made first, so
        the references survive VACUUM. Otherwise a selection is not possible.
        """
        if self.has_rid:
            return RID_COL
        if self.keyfield:
            self.ensure_key_index()
            return self.keyfield
        raise SqlNotSupportedError(
                f"A selection of '{self.name}' needs a daf_rid column or a unique keyfield. "
                f"Set a keyfield with unique values, or copy the table into the store with from_daf().")


    def _rows_sql(self) -> Tuple[str, Tuple[Any, ...]]:
        """
        A query that gives every row, as daf_ref, daf_ord and then the columns, not sorted. Internal use.

        For a base table, it reads the table. Its daf_ref is daf_rid, or else the keyfield, or else
        the rowid. Only reading uses the rowid: `_new_selection()` checks for a reference that
        survives VACUUM first. For a fixed selection, it joins the reference table
        to the base table. For a selection not fixed yet, it is the query that finds the rows.
        """
        if self.base_sdaf is None:
            ord_sql: str = quote_name(RID_COL) if self.has_rid else 'rowid'
            ref_sql: str = ord_sql if self.has_rid or not self.keyfield else quote_name(self.keyfield)
            return (f'SELECT {ref_sql} AS {REF_COL}, {ord_sql} AS {ORD_COL}, {self._cols_sql()} '
                    f'FROM {quote_name(self.name)}'), ()
        if not self.is_fixed:
            return self.pending_sql, self.pending_params
        base_ref_col: str = self.base_sdaf._ref_col()
        return (f'SELECT s.ref AS {REF_COL}, s.ord AS {ORD_COL}, {self._cols_sql("b.")} '
                f'FROM {quote_name(self.name)} AS s JOIN {quote_name(self.base_sdaf.name)} AS b '
                f'ON b.{quote_name(base_ref_col)} = s.ref'), ()


    def fix(self) -> 'SqlDaf':
        """
        Find the rows of a selection now, and fix them.

        Reading a selection does this the first time, so it is rarely needed. It writes a TEMP
        table of references, one integer or key per row, in order. On 1,000,000 rows, a selection
        of 773,698 took 0.20 s. A base table, or a selection already fixed, is not changed.

        Returns:
            This SqlDaf.
        """
        if self.is_fixed:
            return self
        self.store.execute(f'CREATE TEMP TABLE {quote_name(self.name)} (ord INTEGER PRIMARY KEY, ref)')
        self.store.execute(f'INSERT INTO {quote_name(self.name)} (ref) '
                           f'SELECT {REF_COL} FROM ({self.pending_sql}) ORDER BY {ORD_COL}',
                           self.pending_params)
        self.is_fixed = True
        self.pending_sql = ''
        self.pending_params = ()
        return self


    def _is_dense(self) -> bool:
        """
        Say whether daf_ord numbers the rows 1, 2, 3 and so on, so a position is a lookup. Internal use.

        It does in a fixed selection, whose reference table numbers its rows as it is written. It
        does in a base table with daf_rid, as long as no row has been deleted from it.
        """
        return self.has_rid if self.base_sdaf is None else self.is_fixed


    def _count_for_position(self) -> int:
        """
        The number of rows, for a negative position. Internal use.

        Where positions are dense, this is the highest daf_ord, one lookup. Otherwise it is `len()`.
        """
        if self.base_sdaf is None and self.has_rid:
            return int(self.conn.execute(f'SELECT COALESCE(MAX({quote_name(RID_COL)}), 0) FROM {quote_name(self.name)}').fetchone()[0])
        return len(self)


    def _select_rows(self, where_sql: str = '', params: Tuple[Any, ...] = (), limit: int = -1, offset: int = 0,
                     cols: T_ls | None = None) -> List[Tuple[Any, ...]]:
        """ Read rows in order, with an optional condition and limit, as tuples with SQL NULL as None. Internal use. """
        rows_sql, rows_params = self._rows_sql()
        cols_sql: str = ', '.join(quote_name(col) for col in (cols if cols is not None else self.hd))
        where_part: str = f' WHERE {where_sql}' if where_sql else ''
        cursor: sqlite3.Cursor = self.store.execute(
                f'SELECT {cols_sql} FROM ({rows_sql}){where_part} ORDER BY {ORD_COL} LIMIT ? OFFSET ?',
                rows_params + params + (limit, offset))
        return self.store.fetch(cursor)


    # ---- reading ----

    def __len__(self) -> int:
        """
        Count the rows.

        For a base table, this runs a COUNT. For a selection, it is fixed first, and then its
        references are counted, which is one lookup. A row deleted from the base table is still
        counted, so do not delete rows from a base table that has selections.

        Returns:
            The number of rows.
        """
        self.fix()
        if self.base_sdaf is None:
            return int(self.conn.execute(f'SELECT COUNT(*) FROM {quote_name(self.name)}').fetchone()[0])
        return int(self.conn.execute(f'SELECT COALESCE(MAX(ord), 0) FROM {quote_name(self.name)}').fetchone()[0])


    def __bool__(self) -> bool:
        """
        Say whether there is at least one row.

        This stops at the first row, so it is fast on a large table. A selection is fixed first,
        and then its references are checked, as `len()` counts them.

        Returns:
            True if there is a row.
        """
        self.fix()
        return self.conn.execute(f'SELECT 1 FROM {quote_name(self.name)} LIMIT 1').fetchone() is not None


    def num_cols(self) -> int:
        """
        Count the columns.

        The hidden daf_rid column is not counted.

        Returns:
            The number of columns.
        """
        return len(self.hd)


    def to_daf(self) -> Daf:
        """
        Pull every row into an ordinary Daf in memory.

        Use it once the rows you need are few enough to hold. The rows come in order. A SQL NULL
        comes back as NULL. The new Daf has this SqlDaf's name, keyfield and dtypes. Nothing
        ties it to the table afterwards: a change to one does not show in the other. A selection
        is fixed first.

        Returns:
            A Daf of every row.
        """
        self.fix()
        lol: T_lola = [none_to_null(row_t) for row_t in self._select_rows()]
        return Daf(lol=lol, cols=self.columns(), keyfield=self.keyfield, name=self.name,
                   dtypes=dict(self.dtypes) if self.dtypes else None)


    def to_dict(self) -> T_da:
        """
        Read a table of one row as a dict, as `Daf.to_dict()` does.

        A selection is fixed first. For a row by key or by position, `select_record()` and
        `iloc()` are cheaper: they make no selection.

        Returns:
            The row as a dict. An empty dict if there are no rows.

        Raises:
            ValueError: There is more than one row.
        """
        self.fix()
        rows_lot: List[Tuple[Any, ...]] = self._select_rows(limit=2)
        if len(rows_lot) > 1:
            raise ValueError("Ambiguous 2dim array.")
        if not rows_lot:
            return {}
        return dict(zip(self.hd, none_to_null(rows_lot[0])))


    def to_list(self) -> T_la:                                              # type: ignore[override]
        """
        Read a table of one row, or of one column, as a list, as `Daf.to_list()` does.

        One column gives every value of the column, in order, so select first. A selection is
        fixed first.

        Returns:
            The values of the row, or of the column.

        Raises:
            ValueError: There is more than one row and more than one column.
        """
        self.fix()
        if len(self.hd) == 1:
            return [NULL if row_t[0] is None else row_t[0] for row_t in self._select_rows()]
        rows_lot: List[Tuple[Any, ...]] = self._select_rows(limit=2)
        if len(rows_lot) > 1:
            raise ValueError("to_list() requires a 1D Daf (single row or single column)")
        return none_to_null(rows_lot[0]) if rows_lot else []


    def to_value(self) -> Any:                                              # type: ignore[override]
        """
        Read a table of one row and one column as its value, as `Daf.to_value()` does.

        A selection is fixed first.

        Returns:
            The value. NULL for SQL NULL.

        Raises:
            ValueError: The table is not one row and one column.
        """
        self.fix()
        rows_lot: List[Tuple[Any, ...]] = self._select_rows(limit=2)
        if len(rows_lot) != 1 or len(self.hd) != 1:
            raise ValueError(f"to_value() requires 1×1, got shape ({len(rows_lot) if len(rows_lot) < 2 else '2+'}, {len(self.hd)})")
        return NULL if rows_lot[0][0] is None else rows_lot[0][0]


    def iloc(
            self,
            irow:           int = 0,                    # the position of the row. Negative counts from the end.
            include_cols:   T_ls | None = None,         # only these columns.
            rtype:          str = 'dict',               # 'dict' or 'list'.
            ) -> T_da | T_la:
        """
        Read one row by position, as `Daf.iloc()` does, in one query.

        This makes no selection, and does not fix a selection it is called on. In a base table
        with daf_rid, or a fixed selection, the position is looked up directly. Otherwise SQLite
        steps over the rows before it, which takes longer the further in it is. A negative
        position counts from the end, which costs a `len()` first.

        Args:
            irow: The position of the row. Negative counts from the end.
            include_cols: Only these columns, in this order. All if None.
            rtype: 'dict' for a dict, or 'list' for a list of the values.

        Returns:
            The row. NULL for SQL NULL.

        Raises:
            IndexError: There is no row at that position.
            KeyError: A name in `include_cols` is not a column.
            SqlNotSupportedError: `rtype` is not 'dict' or 'list'.
        """
        if rtype not in ('dict', 'list'):
            raise SqlNotSupportedError(f"iloc(rtype='{rtype}') is not supported on a SqlDaf yet. Use 'dict' or 'list'.")
        cols: T_ls = list(include_cols) if include_cols is not None else list(self.hd)
        col: str
        for col in cols:
            if col not in self.hd:
                raise KeyError(col)
        position: int = irow if irow >= 0 else self._count_for_position() + irow
        rows_lot: List[Tuple[Any, ...]] = []
        if position >= 0 and self._is_dense():
            rows_lot = self._select_rows(f'{ORD_COL} = ?', (position + 1,), limit=1, cols=cols)
        elif position >= 0:
            rows_lot = self._select_rows(limit=1, offset=position, cols=cols)
        if not rows_lot:
            raise IndexError(f"iloc: row position {irow} is out of range.")
        values_la: T_la = none_to_null(rows_lot[0])
        return dict(zip(cols, values_la)) if rtype == 'dict' else values_la


    def select_record(self, key: Any, silent_error: bool = True) -> T_da:
        """
        Read one row by its key, as a dict, as `Daf.select_record()` does, in one query.

        The keyfield's unique index is made first if it is not there yet, so the lookup does not
        read every row. This makes no selection, and does not fix a selection it is called on.
        It took about 18 us on 1,000,000 rows, or 9 us inside `store.transaction()`.

        Args:
            key: The key of the row.
            silent_error: If True, a missing key gives an empty dict. If False, it raises KeyError.

        Returns:
            The row as a dict, or an empty dict.

        Raises:
            KeyError: The key is missing, and `silent_error` is False.
            ValueError: There is no keyfield.
        """
        if not self.keyfield:
            raise ValueError(f"SqlDaf.select_record: '{self.name}' has no keyfield.")
        (self.base_sdaf or self).ensure_key_index()
        rows_lot: List[Tuple[Any, ...]] = self._select_rows(f'{quote_name(self.keyfield)} IS ?', (key,), limit=1)
        if not rows_lot:
            if silent_error:
                return {}
            raise KeyError(key)
        return dict(zip(self.hd, none_to_null(rows_lot[0])))


    # ---- selecting ----

    def select_where_sql(self, where_sql: str, params: Iterable[Any] = ()) -> 'SqlDaf':
        """
        Select the rows for which a SQL condition is true.

        The condition is the text after WHERE. Name the columns as they are, in double quotes if
        they need it, and pass values as `params`, with `?` in the text. NULL in `params` becomes
        SQL NULL. Test for a missing value with `IS NULL`. This is the fastest way to select,
        since SQLite does all the work.

        Nothing runs yet. The rows are found when the selection is first read. See SqlDaf.

        Args:
            where_sql: The condition, as SQL text.
            params: Values for the `?` marks in it, in order.

        Returns:
            A selection.

        Examples:
            >>> with SqlStore() as store:
            ...     marks_sdaf = store.from_daf(Daf(lol=[['b1', 1], ['b2', 0]], cols=['ballot_id', 'v']), name='marks')
            ...     marks_sdaf.select_where_sql('"v" = ?', [0]).to_list()
            ['b2', 0]
        """
        parent_sql, parent_params = self._rows_sql()
        return SqlDaf._new_selection(self, f'SELECT * FROM ({parent_sql}) WHERE {where_sql}',
                                     parent_params + null_to_none(params))


    def select_where(
            self,
            where:          Callable[[Any], Any],       # a function of a row that is true for the rows to keep.
            indirect_col:   str | None = None,          # not supported yet.
            cols:           T_ls | None = None,         # the columns the function reads. All if None.
            ) -> 'SqlDaf':
        """
        Select the rows for which a Python function is true, as `Daf.select_where()` does.

        The function gets each row as a [KeyedList][daffodil.keyedlist.KeyedList], with NULL for
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
            SqlNotSupportedError: `indirect_col` is given.

        Examples:
            >>> with SqlStore() as store:
            ...     marks_sdaf = store.from_daf(Daf(lol=[['b1', 1], ['b2', 0]], cols=['ballot_id', 'v']), name='marks')
            ...     marks_sdaf.select_where(lambda row: row['v'] == 1, cols=['v']).to_list()
            ['b1', 1]
        """
        if indirect_col is not None:
            raise SqlNotSupportedError("select_where(indirect_col=) is not supported on a SqlDaf yet.")
        func_cols: T_ls = list(cols) if cols is not None else list(self.hd)
        col: str
        for col in func_cols:
            if col not in self.hd:
                raise KeyError(col)
        func_hd: T_di = {col: idx for idx, col in enumerate(func_cols)}

        def row_test(*values: Any) -> int:
            return 1 if where(KeyedList(func_hd, [NULL if val is None else val for val in values])) else 0

        func_name: str = self.store.register_function(row_test, len(func_cols))
        return self.select_where_sql(f'{func_name}({", ".join(quote_name(col) for col in func_cols)})')


    def select_by_dict(                                                     # type: ignore[override]
            self,
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
        column that is not there matches no row. NULL as a value matches a missing value.

        Values are compared with SQL's IS, which treats NULL as a value, so the inverse keeps rows
        with a missing value. As in Python, 1 equals 1.0, and the text '1' does not equal 1.

        Nothing runs yet, unless `expectmax` is given: then the selection is fixed at once, to
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
            >>> with SqlStore() as store:
            ...     marks_sdaf = store.from_daf(Daf(lol=[['b1', 1], ['b2', '']], cols=['ballot_id', 'v']), name='marks')
            ...     marks_sdaf.select_by_dict({'v': ''}).to_list()
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
            terms_ls.append('(' + ' AND '.join(f'{quote_name(col)} IS ?' for col in one_da) + ')')
            params_la.extend(one_da.values())

        cond_sql: str = ' OR '.join(terms_ls) if terms_ls else '0'
        if inverse:
            cond_sql = f'NOT ({cond_sql})'
        sel_sdaf: SqlDaf = self.select_where_sql(cond_sql, params_la)
        if keyfield:
            sel_sdaf.set_keyfield(keyfield)
        if expectmax != -1 and len(sel_sdaf) > expectmax:
            raise LookupError(f"select_by_dict: more than {expectmax} rows match {selector_da}.")
        return sel_sdaf


    def __getitem__(self, sel: Any) -> 'SqlDaf':
        """
        Select rows by position or by key, as `daf[n]`, `daf[n:m]` and `daf[key]` do.

        Each gives a selection, not an ordinary Daf, so a large slice stays in the database.
        Read it with `to_daf()`, or `to_dict()` for one row. For one row as a dict, `iloc()` and
        `select_record()` are cheaper: they make no selection.

        - An int is a position, from 0. Negative counts from the end. A position past the end
          raises IndexError, as in a Daf.
        - A slice of positions, as in `sdaf[10:20]`. A step is not supported yet. A slice past
          the end is empty, as in a Daf.
        - A str is a key. The keyfield's unique index is made first. A missing key raises
          KeyError, as in a Daf.

        A negative position, or a slice with a negative end, costs a `len()` first. In a base
        table with daf_rid, or a fixed selection, positions are looked up directly. Otherwise
        SQLite steps over the rows before them.

        Args:
            sel: An int, a slice or a str.

        Returns:
            A selection.

        Raises:
            IndexError: An int position past the end.
            KeyError: A key that is missing.
            SqlNotSupportedError: Any other form, such as `sdaf[rows, cols]`, or a slice with a step.
        """
        parent_sql, parent_params = self._rows_sql()
        if isinstance(sel, bool) or not isinstance(sel, (int, slice, str)):
            raise SqlNotSupportedError(f"sdaf[{sel!r}] is not supported on a SqlDaf yet. Use an int, a slice or a key.")

        if isinstance(sel, str):
            if not self.keyfield:
                raise SqlNotSupportedError(f"sdaf['{sel}']: '{self.name}' has no keyfield.")
            if not self.select_record(sel):
                raise KeyError(sel)
            return SqlDaf._new_selection(self, f'SELECT * FROM ({parent_sql}) WHERE {quote_name(self.keyfield)} IS ?',
                                         parent_params + (sel,))

        start: int
        stop: int
        if isinstance(sel, int):
            start = sel if sel >= 0 else self._count_for_position() + sel
            if start < 0 or not self.iloc_exists(start):
                raise IndexError(f"sdaf[{sel}]: row position {sel} is out of range.")
            stop = start + 1
        else:
            if sel.step not in (None, 1):
                raise SqlNotSupportedError("A slice with a step is not supported on a SqlDaf yet.")
            needs_len: bool = (sel.start is not None and sel.start < 0) or (sel.stop is not None and sel.stop < 0)
            count: int = self._count_for_position() if needs_len else 0
            start = 0 if sel.start is None else (sel.start if sel.start >= 0 else max(count + sel.start, 0))
            stop = -1 if sel.stop is None else (sel.stop if sel.stop >= 0 else max(count + sel.stop, 0))

        if self._is_dense():
            if stop == -1:
                return SqlDaf._new_selection(self, f'SELECT * FROM ({parent_sql}) WHERE {ORD_COL} > ?',
                                             parent_params + (start,))
            return SqlDaf._new_selection(self, f'SELECT * FROM ({parent_sql}) WHERE {ORD_COL} > ? AND {ORD_COL} <= ?',
                                         parent_params + (start, max(stop, start)))
        limit: int = -1 if stop == -1 else max(stop - start, 0)
        return SqlDaf._new_selection(self, f'SELECT * FROM ({parent_sql}) ORDER BY {ORD_COL} LIMIT ? OFFSET ?',
                                     parent_params + (limit, start))


    def iloc_exists(self, irow: int) -> bool:
        """ Say whether there is a row at this position, from 0. """
        rows_sql, rows_params = self._rows_sql()
        if self._is_dense():
            return self.store.execute(f'SELECT 1 FROM ({rows_sql}) WHERE {ORD_COL} = ?',
                                      rows_params + (irow + 1,)).fetchone() is not None
        cursor: sqlite3.Cursor = self.store.execute(
                f'SELECT 1 FROM ({rows_sql}) ORDER BY {ORD_COL} LIMIT 1 OFFSET ?', rows_params + (irow,))
        return cursor.fetchone() is not None


    # ---- copying ----

    def _copy_to(self, name: str, temp: bool) -> 'SqlDaf':
        """ Copy the rows, in order, into a new base table with daf_rid. Internal use. """
        if self.store.table_exists(name):
            raise ValueError(f"SqlStore: a table named '{name}' already exists.")
        table_sql: str = self.store._create_table_sql(name, list(self.hd)).replace('CREATE TABLE', 'CREATE TEMP TABLE' if temp else 'CREATE TABLE', 1)
        rows_sql, rows_params = self._rows_sql()
        with self.store.transaction():
            self.store.execute(table_sql)
            self.store.execute(f'INSERT INTO {quote_name(name)} ({self._cols_sql()}) '
                               f'SELECT {self._cols_sql()} FROM ({rows_sql}) ORDER BY {ORD_COL}', rows_params)
        return SqlDaf(store=self.store, name=name, keyfield=self.keyfield, dtypes=self.dtypes)


    def materialize(self) -> 'SqlDaf':
        """
        Copy the rows into a new TEMP table, and return it as a base table.

        Use it for a result that is read many times, or changed, without changing the rows it
        came from. The copy is a table of its own: a change to it does not reach the base table,
        nor the other way round. It goes when the store closes. A base table gives itself back.

        Returns:
            A base table of the copied rows, named as in daf_tmp_marks_v4.
        """
        if self.base_sdaf is None:
            return self
        return self._copy_to(self.store.next_temp_name((self.base_sdaf).name), temp=True)


    def save_as(self, name: str) -> 'SqlDaf':
        """
        Copy the rows into a new table with this name, kept in the database file.

        The copy is a table of its own, with the keyfield and dtypes of this one. A change to it
        does not reach the rows it came from, nor the other way round.

        Args:
            name: The name of the new table.

        Returns:
            A base table of the copied rows.

        Raises:
            ValueError: A table of that name exists.
        """
        return self._copy_to(name, temp=False)


    # ---- keys and types ----

    def set_keyfield(self, keyfield: str = '') -> 'SqlDaf':        # type: ignore[override]
        """
        Set the column that holds the key of each row.

        As in a Daf, nothing is built yet. The unique index on the column is made when a key is
        first used, by `ensure_key_index()`. An empty keyfield clears it.

        Args:
            keyfield: The column. '' for none.

        Returns:
            This SqlDaf.

        Raises:
            KeyError: The keyfield is not a column.
        """
        if keyfield and keyfield not in self.hd:
            raise KeyError(keyfield)
        self.keyfield = keyfield
        return self


    def key_index_name(self) -> str:
        """ The name of the unique index on the keyfield, as in daf_ix_marks_ballot_id. """
        return f'{INDEX_PREFIX}{self.name}_{self.keyfield}'


    def ensure_key_index(self) -> 'SqlDaf':
        """
        Make the unique index on the keyfield of a base table, if it is not there yet.

        A Daf builds its key index the first time a key is needed. A SqlDaf does the same: the
        methods that use keys call this first. On 1,000,000 rows it took 0.50 s, about what a
        Daf takes. After that, a lookup by key uses the index instead of reading every row.

        The index is unique. A Daf keeps the last row of a repeated key without a word. A SqlDaf
        raises instead, so a duplicate is never missed. Once the index is there, adding a row
        with a key that is already present raises too.

        Returns:
            This SqlDaf.

        Raises:
            ValueError: There is no keyfield, or this is a selection.
            sqlite3.IntegrityError: A key is repeated. Nothing is made.
        """
        if self.base_sdaf is not None:
            raise ValueError(f"SqlDaf.ensure_key_index: '{self.name}' is a selection. Its base table has the index.")
        if not self.keyfield:
            raise ValueError(f"SqlDaf.ensure_key_index: '{self.name}' has no keyfield.")
        self.conn.execute(
                f'CREATE UNIQUE INDEX IF NOT EXISTS {quote_name(self.key_index_name())} '
                f'ON {quote_name(self.name)} ({quote_name(self.keyfield)})')
        return self


    def apply_dtypes(                                               # type: ignore[override]
            self,
            *,
            dtypes:         T_dtype_dict | None = None,     # the types to convert to. This SqlDaf's dtypes if None.
            cols:           str | T_ls | None = None,       # convert only these columns.
            silent_error:   bool = False,                   # if True, a column with no dtype stays as it is.
            ) -> 'SqlDaf':
        """
        Convert the columns to their dtypes, in place, in the database.

        A table loaded from a CSV file holds text. Call this to turn the columns you need into
        numbers. Each value is converted by the same function `Daf.apply_dtypes()` uses, so the
        results are the same as in memory: '12' becomes 12, '2.5' with dtype int becomes 2, and
        a value that does not convert, such as 'abc', stays as it is. NULL stays NULL, except in
        a bool column, where it becomes 0, as in a Daf. A str column is not touched. All the columns are converted in one pass over the table, in one
        transaction. On 1,000,000 rows, two int columns took 1.8 s.

        Only int, float, bool and str can be stored yet. A list or a dict would need JSON text.

        Args:
            dtypes: The types to convert to, by column. It may name columns that are not there. This SqlDaf's dtypes if None.
            cols: Convert only these columns, or this one column. All the columns of the dtypes if None.
            silent_error: If False, a column to convert that has no dtype raises ValueError. If True, it is left as it is.

        Returns:
            This SqlDaf, whose table has been changed.

        Raises:
            KeyError: A name in `cols` is not a column.
            ValueError: A column to convert has no dtype, and `silent_error` is False.
            SqlNotSupportedError: A dtype is not int, float, bool or str, or this is a selection.
        """
        if self.base_sdaf is not None:
            raise SqlNotSupportedError(f"apply_dtypes() on the selection '{self.name}': convert its base table instead.")
        if dtypes:
            self.dtypes = {**self.dtypes, **dtypes}
        source_dtypes: T_dtype_dict = dtypes or self.dtypes
        if not source_dtypes:
            return self

        cols_ls: T_ls = list(self.hd) if cols is None else ([cols] if isinstance(cols, str) else list(cols))
        col: str
        for col in cols_ls:
            if col not in self.hd:
                raise KeyError(col)
        missing_ls: T_ls = [col for col in cols_ls if col not in source_dtypes]
        if missing_ls and not silent_error:
            raise ValueError(f"SqlDaf.apply_dtypes: no dtype for the columns {missing_ls}. "
                             f"Add them to dtypes, or pass silent_error=True to leave them as they are.")

        convert_dtypes: Dict[str, Any] = {col: source_dtypes[col] for col in cols_ls
                                          if col in source_dtypes and source_dtypes[col] is not str}
        unstorable_ls: T_ls = [col for col, dtype in convert_dtypes.items() if dtype not in STORABLE_DTYPES]
        if unstorable_ls:
            raise SqlNotSupportedError(f"apply_dtypes() cannot store {[convert_dtypes[col] for col in unstorable_ls]} "
                                       f"for the columns {unstorable_ls} yet. Only int, float, bool and str.")
        if not convert_dtypes:
            return self

        set_ls: T_ls = []
        dtype: Any
        for col, dtype in convert_dtypes.items():
            func_name: str = f'daf_to_{dtype.__name__}'
            converter: Callable[[Any], Any] = daf_utils.get_converter(dtype)
            self.conn.create_function(func_name, 1, _none_safe(converter), deterministic=True)
            set_ls.append(f'{quote_name(col)} = {func_name}({quote_name(col)})')

        with self.store.transaction():
            self.conn.execute(f'UPDATE {quote_name(self.name)} SET {", ".join(set_ls)}')
        return self


    def sql(self, stmt: str, params: Iterable[Any] = ()) -> Daf:
        """
        Run a SQL statement of your own, and return its result as a Daf.

        This is the fallback for anything SqlDaf does not do yet. The statement runs on the
        store's connection, so it can use every table in the store. Refer to this table by
        `sdaf.name`. Pass values as `params`, with `?` in the statement, instead of putting
        them in the text.

        A SELECT gives a Daf of its rows, with its column names. A SQL NULL becomes NULL. A
        statement that returns no rows, such as UPDATE or CREATE, gives an empty Daf. Outside
        `store.transaction()`, it is committed when it finishes.

        The whole result is pulled into memory, so select only what you need.

        Args:
            stmt: One SQL statement.
            params: Values for the `?` marks in the statement, in order.

        Returns:
            A Daf of the rows the statement returns.
        """
        cursor: sqlite3.Cursor = self.conn.execute(stmt, null_to_none(params))
        if cursor.description is None:
            return Daf()
        cols: T_ls = [desc_t[0] for desc_t in cursor.description]
        lol: T_lola = [none_to_null(row_t) for row_t in cursor]
        return Daf(lol=lol, cols=cols)


# Type aliases for SqlDaf, as daf_types.py has for other types. A SqlDaf variable ends in _sdaf,
# as in bif_sdaf. They are here, not in daf_types.py, because this module imports that one.
T_sdaf:     TypeAlias = SqlDaf
T_lsdaf:    TypeAlias = List[SqlDaf]
T_dosdaf:   TypeAlias = Dict[str, SqlDaf]


def _none_safe(converter: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """
    Wrap a converter so that it sees NULL where SQL has NULL, as it would in a Daf. Internal use.

    An empty result is stored as SQL NULL again. So NULL stays NULL for int and float, and
    becomes 0 for bool, as `Daf.apply_dtypes()` gives.
    """
    def convert(val: Any) -> Any:
        result: Any = converter(NULL if val is None else val)
        return None if result is NULL else result
    return convert


# ---- block every Daf method that SqlDaf does not define (decision 88) ----

# Daf methods that work unchanged on a SqlDaf, because they read only hd or keyfield.
SAFE_INHERITED_SET = {
    'columns',
    }

# Python internals that a class needs, never blocked.
KEEP_SET = {
    '__init__', '__new__', '__dict__', '__doc__', '__module__', '__weakref__', '__class__',
    '__getattribute__', '__setattr__', '__delattr__', '__init_subclass__', '__annotations__',
    '__hash__', '__subclasshook__',
    }


def _make_blocker(name: str) -> Callable[..., Any]:
    """ A method that raises SqlNotSupportedError, naming `name`. Internal use. """
    def blocker(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise SqlNotSupportedError(
                f"{name}() is not supported on a SqlDaf yet. Use sql(), or to_daf() on a few rows.")
    blocker.__name__ = name
    blocker.__qualname__ = f'SqlDaf.{name}'
    return blocker


def _make_property_blocker(name: str) -> property:
    """ A property that raises SqlNotSupportedError, naming `name`. Internal use. """
    def blocked_get(self: Any) -> Any:
        raise SqlNotSupportedError(f"{name} is not supported on a SqlDaf yet.")
    return property(blocked_get)


BLOCKED_LS: T_ls = []
_attr_name: str
_attr: Any
for _attr_name, _attr in list(vars(Daf).items()):
    if _attr_name in vars(SqlDaf) or _attr_name in SAFE_INHERITED_SET or _attr_name in KEEP_SET:
        continue
    if isinstance(_attr, property):
        setattr(SqlDaf, _attr_name, _make_property_blocker(_attr_name))
        BLOCKED_LS.append(_attr_name)
    elif callable(_attr) or isinstance(_attr, (staticmethod, classmethod)):
        setattr(SqlDaf, _attr_name, _make_blocker(_attr_name))
        BLOCKED_LS.append(_attr_name)
