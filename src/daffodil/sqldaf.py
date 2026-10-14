# sqldaf.py
#
# A Daf whose rows live in a SQLite table, and the store that owns the database.
# The design and the decisions behind it are in notes/sql_backing/design_draft.md.

import os
import csv
import sqlite3
import tempfile
import contextlib

from typing import Any, Callable, Dict, Iterator, List, Tuple, Type      # noqa: F401
from collections.abc import Iterable

from daffodil.daf import Daf, NULL, DaffodilError
from daffodil.lib.daf_types import T_ls, T_la, T_lola, T_di, T_dtype_dict
import daffodil.lib.daf_utils as daf_utils


RID_COL = 'daf_rid'                 # the INTEGER PRIMARY KEY of each table SqlStore makes
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
        if RID_COL in cols:
            raise ValueError(f"SqlStore: '{RID_COL}' is reserved and cannot be a column name.")

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
    - `apply_dtypes()`, to convert columns in place.
    - `set_keyfield()`, and `ensure_key_index()`, which builds the unique index on the keyfield.
    - `to_daf()`, to pull every row into memory.
    - `sql()`, to run any SQL statement.

    Get a SqlDaf from its store: `store.table()`, `store.from_csv()`, `store.from_daf()` or
    `store.create_table()`.

    Work with large tables in whole statements. One lookup by key costs about 12 to 24 us,
    against under 1 us in memory, because each statement passes through sqlite3. Many keys
    in one statement cost 5 to 7 us each.

    A missing value is NULL in Python, as in any Daf, and SQL NULL in the database. It is
    converted on the way in and on the way out. So in Python test a value with `val is NULL`,
    and in SQL write `col IS NULL`. Do not compare with ''.

    Until `apply_dtypes()` converts them, values loaded from a CSV file are text, in SQL too.
    Text is never less than a number in SQLite, so `WHERE n < 5` finds nothing on a text
    column, and MAX() gives the last in text order.

    The `name` is the name of the table in the database. It cannot be changed.

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
        Make a SqlDaf for a table in the store.

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
        Make the unique index on the keyfield, if it is not there yet.

        A Daf builds its key index the first time a key is needed. A SqlDaf does the same: the
        methods that use keys call this first. On 1,000,000 rows it took 0.50 s, about what a
        Daf takes. After that, a lookup by key uses the index instead of reading every row.

        The index is unique. A Daf keeps the last row of a repeated key without a word. A SqlDaf
        raises instead, so a duplicate is never missed. Once the index is there, adding a row
        with a key that is already present raises too.

        Returns:
            This SqlDaf.

        Raises:
            ValueError: There is no keyfield.
            sqlite3.IntegrityError: A key is repeated. Nothing is made.
        """
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
            SqlNotSupportedError: A dtype is not int, float, bool or str.
        """
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


    def _select_cols_sql(self) -> str:
        """ The column list for a SELECT of every column, in order, without daf_rid. Internal use. """
        return ', '.join(quote_name(col) for col in self.hd)


    def to_daf(self) -> Daf:
        """
        Pull every row into an ordinary Daf in memory.

        Use it once the rows you need are few enough to hold. The rows come in the order they
        were added. A SQL NULL comes back as NULL. The new Daf has this SqlDaf's name, keyfield
        and dtypes. Nothing ties it to the table afterwards: a change to one does not show in
        the other.

        Returns:
            A Daf of every row.
        """
        order_sql: str = f' ORDER BY {quote_name(RID_COL)}' if self.has_rid else ''
        cursor: sqlite3.Cursor = self.conn.execute(
                f'SELECT {self._select_cols_sql()} FROM {quote_name(self.name)}{order_sql}')
        lol: T_lola = [none_to_null(row_t) for row_t in cursor]
        return Daf(lol=lol, cols=self.columns(), keyfield=self.keyfield, name=self.name,
                   dtypes=dict(self.dtypes) if self.dtypes else None)


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
