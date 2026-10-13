# sqldaf.py
#
# A Daf whose rows live in a SQLite table, and the store that owns the database.
# The design and the decisions behind it are in notes/sql_backing/design_draft.md.

import os
import csv
import sqlite3
import tempfile

from typing import Any, Callable, Dict, Iterator, List, Tuple, Type      # noqa: F401
from collections.abc import Iterable

from daffodil.daf import Daf, NULL, DaffodilError
from daffodil.lib.daf_types import T_ls, T_la, T_lola, T_di, T_dtype_dict


RID_COL = 'daf_rid'                 # the INTEGER PRIMARY KEY of each table SqlStore makes
TEMP_PREFIX = 'daf_tmp_'            # the prefix of every TEMP object SqlStore makes

# The SQL type declared for a dtype. A column with any other dtype, or none, is declared with
# no type, so SQLite keeps each value as it is given.
SQL_TYPE_OF_DTYPE: Dict[Any, str] = {
    int:    'INTEGER',
    bool:   'INTEGER',
    float:  'REAL',
    str:    'TEXT',
    }


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


def null_to_none(row_la: T_la) -> Tuple[Any, ...]:
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

    The connection runs in autocommit mode: each statement is committed when it finishes.
    A load of many rows is wrapped in one transaction, which is much faster.

    Use it as a context manager to close it on the way out.

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

        self.temp_path: str     = ''                    # the temporary file to delete at close, if any.
        self.owns_conn: bool    = conn is None
        self.temp_count: int    = 0                     # counter for the names of TEMP objects.

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


    def table(self, name: str, keyfield: str = '') -> 'SqlDaf':
        """
        Get a SqlDaf for a table that is already in the database.

        Args:
            name: The name of the table.
            keyfield: The column that holds the key of each row, if any.

        Returns:
            A SqlDaf for the table. Nothing is read yet.

        Raises:
            KeyError: There is no table of that name.
        """
        return SqlDaf(store=self, name=name, keyfield=keyfield)


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


    def create_table(
            self,
            name:       str,                            # the name of the new table.
            cols:       T_ls,                           # its column names, in order.
            keyfield:   str = '',                       # a column whose values must be unique.
            dtypes:     T_dtype_dict | None = None,     # the type of each column. Untyped if not given.
            ) -> 'SqlDaf':
        """
        Make an empty table, and return a SqlDaf for it.

        The table gets a hidden first column, `daf_rid`, an INTEGER PRIMARY KEY. It numbers the
        rows in the order they are added, keeps that order, and does not change when the file
        is vacuumed. It is not one of the `columns()`, and `to_daf()` leaves it out.

        A column whose dtype is int, bool, float or str is declared INTEGER, REAL or TEXT. SQLite
        then turns text such as '12' into the number 12 as it is stored. A value that does not
        fit, such as 'abc' in an INTEGER column, is kept as text, with no error. A column with
        any other dtype, or none, is declared with no type, and each value is kept as given: an
        int stays an int, and text stays text.

        A cell that holds a list or a dict cannot be stored yet. sqlite3 raises an error for it.

        With a keyfield, the table gets a unique index on it, so adding a row with a key that is
        already there raises `sqlite3.IntegrityError`.

        Args:
            name: The name of the new table.
            cols: The column names, in order.
            keyfield: A column whose values must be unique. None if empty.
            dtypes: The type of each column, as in a Daf.

        Returns:
            A SqlDaf for the empty table.

        Raises:
            ValueError: A table of that name exists, or a column is named daf_rid.
            KeyError: The keyfield is not one of the columns.
        """
        table_sql, index_sql = self._create_table_sql(name, cols, keyfield, dtypes)
        self.conn.execute(table_sql)
        if index_sql:
            self.conn.execute(index_sql)
        return SqlDaf(store=self, name=name, keyfield=keyfield)


    def _create_table_sql(self, name: str, cols: T_ls, keyfield: str, dtypes: T_dtype_dict | None) -> Tuple[str, str]:
        """ Check a new table, and give the SQL that makes it and its unique index, or '' for none. Internal use. """
        if self.table_exists(name):
            raise ValueError(f"SqlStore: a table named '{name}' already exists.")
        if RID_COL in cols:
            raise ValueError(f"SqlStore: '{RID_COL}' is reserved and cannot be a column name.")
        if keyfield and keyfield not in cols:
            raise KeyError(keyfield)

        dtypes = dtypes or {}
        coldefs_ls: T_ls = [f'{quote_name(RID_COL)} INTEGER PRIMARY KEY'] + [
                f'{quote_name(col)} {SQL_TYPE_OF_DTYPE.get(dtypes.get(col), "")}'.rstrip() for col in cols]
        table_sql: str = f'CREATE TABLE {quote_name(name)} ({", ".join(coldefs_ls)})'
        index_sql: str = ''
        if keyfield:
            index_sql = (f'CREATE UNIQUE INDEX {quote_name("daf_ix_" + name + "_" + keyfield)} '
                         f'ON {quote_name(name)} ({quote_name(keyfield)})')
        return table_sql, index_sql


    def _load_table(
            self,
            name:       str,                            # the name of the new table.
            cols:       T_ls,                           # its column names, in order.
            keyfield:   str,                            # a column whose values must be unique, or ''.
            dtypes:     T_dtype_dict | None,            # the type of each column.
            rows:       Iterable[Tuple[Any, ...]],      # the rows, with None for a missing value.
            ) -> 'SqlDaf':
        """
        Make a table, add the rows and then the unique index, all in one transaction. Internal use.

        Building the index after the rows is faster than keeping it up during the load. If any
        row fails, or a key is repeated, nothing is made, and the error is raised.
        """
        table_sql, index_sql = self._create_table_sql(name, cols, keyfield, dtypes)
        insert_sql: str = (f'INSERT INTO {quote_name(name)} ({", ".join(quote_name(col) for col in cols)}) '
                           f'VALUES ({placeholders(len(cols))})')
        self.conn.execute('BEGIN')
        try:
            self.conn.execute(table_sql)
            self.conn.executemany(insert_sql, rows)
            if index_sql:
                self.conn.execute(index_sql)
        except BaseException:
            self.conn.execute('ROLLBACK')
            raise
        self.conn.execute('COMMIT')
        return SqlDaf(store=self, name=name, keyfield=keyfield)


    def insert_rows(self, name: str, cols: T_ls, rows: Iterable[T_la]) -> int:
        """
        Add rows to a table, in one transaction, and return how many were added.

        The rows are read from `rows` as they are needed, so an iterator of any length works.
        A NULL cell is stored as SQL NULL. If any row fails, as for a duplicate key, none of
        them are added, and the error is raised.

        Args:
            name: The table.
            cols: The columns the values of each row go to, in order.
            rows: The rows, each a list of values in the order of `cols`.

        Returns:
            The number of rows added.
        """
        stmt: str = (f'INSERT INTO {quote_name(name)} ({", ".join(quote_name(col) for col in cols)}) '
                     f'VALUES ({placeholders(len(cols))})')
        count: int = 0
        row_la: T_la

        def counted(rows: Iterable[T_la]) -> Iterator[Tuple[Any, ...]]:
            nonlocal count
            for row_la in rows:
                count += 1
                yield null_to_none(row_la)

        self.conn.execute('BEGIN')
        try:
            self.conn.executemany(stmt, counted(rows))
        except BaseException:
            self.conn.execute('ROLLBACK')
            raise
        self.conn.execute('COMMIT')
        return count


    def from_csv(
            self,
            path:       str,                            # the CSV file.
            name:       str = '',                       # the table name. The file name without .csv if empty.
            keyfield:   str = '',                       # a column whose values must be unique.
            dtypes:     T_dtype_dict | None = None,     # the type of each column. Untyped if not given.
            delimiter:  str = ',',                      # the field separator.
            ) -> 'SqlDaf':
        """
        Load a CSV file into a new table, without building it in memory.

        The first line gives the column names. The rest is read one row at a time and added in
        batches, so memory use does not grow with the file. An empty cell is stored as SQL NULL,
        and comes back as NULL. A row shorter than the header is padded with NULL, and a longer
        one raises ValueError. The table, its rows and its unique index are made in one
        transaction. If any row fails, or a key is repeated, no table is made.

        The column types come from `dtypes`, as `create_table()` explains. Without them the
        values stay the text of the file, as `Daf.from_csv()` gives them.

        Args:
            path: The CSV file.
            name: The name of the new table. The file name without its extension if empty.
            keyfield: A column whose values must be unique. A duplicate raises.
            dtypes: The type of each column, as in a Daf.
            delimiter: The field separator.

        Returns:
            A SqlDaf for the new table.

        Raises:
            ValueError: A table of that name exists, a column is named daf_rid, or a row is longer than the header.
            sqlite3.IntegrityError: The keyfield has a duplicate value.
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

            table_sdaf: SqlDaf = self._load_table(name, cols, keyfield, dtypes, fitted(reader))

        return table_sdaf


    def from_daf(self, data_daf: Daf, name: str, keyfield: str = '') -> 'SqlDaf':
        """
        Write an in-memory Daf into a new table.

        The column types come from the Daf's dtypes. The keyfield is the Daf's, unless one is
        given. A NULL cell is stored as SQL NULL. As in `from_csv()`, if any row fails, or a key
        is repeated, no table is made.

        Args:
            data_daf: The Daf to write. It is not changed.
            name: The name of the new table.
            keyfield: A column whose values must be unique. The Daf's keyfield if empty.

        Returns:
            A SqlDaf for the new table.

        Raises:
            ValueError: A table of that name exists, or a column is named daf_rid.
            sqlite3.IntegrityError: The keyfield has a duplicate value.
        """
        keyfield = keyfield or (data_daf.keyfield if isinstance(data_daf.keyfield, str) else '')
        cols: T_ls = data_daf.columns()
        row_la: T_la
        return self._load_table(name, cols, keyfield, data_daf.dtypes, (null_to_none(row_la) for row_la in data_daf.lol))


    def next_temp_name(self, parent: str) -> str:
        """ A new name for a TEMP object made from `parent`, as in daf_tmp_marks_v3. """
        self.temp_count += 1
        return f'{TEMP_PREFIX}{parent}_v{self.temp_count}'


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

    The methods so far:

    - `len()`, `bool()`, `columns()` and `num_cols()`.
    - `to_daf()`, to pull every row into memory.
    - `sql()`, to run any SQL statement.

    Get a SqlDaf from its store: `store.table()`, `store.from_csv()`, `store.from_daf()` or
    `store.create_table()`.

    A missing value is NULL in Python, as in any Daf, and SQL NULL in the database. It is
    converted on the way in and on the way out. So in Python test a value with `val is NULL`,
    and in SQL write `col IS NULL`. Do not compare with ''.

    The `name` is the name of the table in the database. It cannot be changed.

    Examples:
        >>> with SqlStore() as store:
        ...     marks_sdaf = store.from_daf(Daf(lol=[['b1', 1], ['b2', '']], cols=['ballot_id', 'v']), name='marks')
        ...     marks_sdaf.sql('SELECT ballot_id FROM marks WHERE v IS NULL').to_list()
        ['b2']
    """

    def __init__(
            self,
            store:      SqlStore,               # the store that owns the table.
            name:       str,                    # the name of the table.
            keyfield:   str = '',               # the column that holds the key of each row.
            ) -> None:
        """
        Make a SqlDaf for a table in the store.

        Usually you get one from the store instead, as with `store.table()`.

        Args:
            store: The store that owns the table.
            name: The name of the table.
            keyfield: The column that holds the key of each row, if any.

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

        self.store:     SqlStore = store
        self._name:     str      = name
        self.keyfield:  str      = keyfield
        self.has_rid:   bool     = RID_COL in all_cols
        self.hd:        T_di     = {col: idx for idx, col in enumerate(cols)}
        self.dtypes:    T_dtype_dict = {}


    @property
    def name(self) -> str:              # type: ignore[override]
        """ The name of the table in the database. It cannot be changed. """
        return self._name


    @property
    def conn(self) -> sqlite3.Connection:
        """ The store's connection. """
        return self.store.conn


    def __repr__(self) -> str:
        return f"<SqlDaf name='{self.name}' rows={len(self)} cols={len(self.hd)} keyfield='{self.keyfield}'>"


    __str__ = __repr__


    def __len__(self) -> int:
        """
        Count the rows.

        This runs a COUNT in the database, so it reads the table each time.

        Returns:
            The number of rows.
        """
        return int(self.conn.execute(f'SELECT COUNT(*) FROM {quote_name(self.name)}').fetchone()[0])


    def __bool__(self) -> bool:
        """
        Say whether there is at least one row.

        This stops at the first row, so it is fast on a large table.

        Returns:
            True if the table has a row.
        """
        return self.conn.execute(f'SELECT 1 FROM {quote_name(self.name)} LIMIT 1').fetchone() is not None


    def num_cols(self) -> int:
        """
        Count the columns.

        The hidden daf_rid column is not counted.

        Returns:
            The number of columns.
        """
        return len(self.hd)


    def _select_cols_sql(self) -> str:
        """ The column list for a SELECT of every column, in order, without daf_rid. Internal use. """
        return ', '.join(quote_name(col) for col in self.hd)


    def to_daf(self) -> Daf:
        """
        Pull every row into an ordinary Daf in memory.

        Use it once the rows you need are few enough to hold. The rows come in the order they
        were added. A SQL NULL comes back as NULL. The new Daf has this SqlDaf's name and
        keyfield. Nothing ties it to the table afterwards: a change to one does not show in
        the other.

        Returns:
            A Daf of every row.
        """
        order_sql: str = f' ORDER BY {quote_name(RID_COL)}' if self.has_rid else ''
        cursor: sqlite3.Cursor = self.conn.execute(
                f'SELECT {self._select_cols_sql()} FROM {quote_name(self.name)}{order_sql}')
        lol: T_lola = [none_to_null(row_t) for row_t in cursor]
        return Daf(lol=lol, cols=self.columns(), keyfield=self.keyfield, name=self.name)


    def sql(self, stmt: str, params: Iterable[Any] = ()) -> Daf:
        """
        Run a SQL statement of your own, and return its result as a Daf.

        This is the fallback for anything SqlDaf does not do yet. The statement runs on the
        store's connection, so it can use every table in the store. Refer to this table by
        `sdaf.name`. Pass values as `params`, with `?` in the statement, instead of putting
        them in the text.

        A SELECT gives a Daf of its rows, with its column names. A SQL NULL becomes NULL. A
        statement that returns no rows, such as UPDATE or CREATE, gives an empty Daf. It is
        committed when it finishes.

        The whole result is pulled into memory, so select only what you need.

        Args:
            stmt: One SQL statement.
            params: Values for the `?` marks in the statement, in order.

        Returns:
            A Daf of the rows the statement returns.
        """
        cursor: sqlite3.Cursor = self.conn.execute(stmt, tuple(null_to_none(list(params))))
        if cursor.description is None:
            return Daf()
        cols: T_ls = [desc_t[0] for desc_t in cursor.description]
        lol: T_lola = [none_to_null(row_t) for row_t in cursor]
        return Daf(lol=lol, cols=cols)


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
