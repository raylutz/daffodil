# sqldaf.py
#
# A Daf whose rows live in a SQLite table.
# The design and the decisions behind it are in notes/sql_backing/design_draft.md.

import os
import csv
import glob
import json
import threading
import weakref
import contextlib
import urllib.parse

from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, Generator, Iterator, List, Literal, Tuple, Type, TypeAlias, cast, overload      # noqa: F401
from collections.abc import Iterable

from daffodil.daf import Daf, NULL, KeysDisabledError
from daffodil.keyedlist import KeyedList
from daffodil.lib.daf_types import T_ls, T_la, T_lola, T_di, T_da, T_ss, T_sa, T_kva, T_dtype_dict, T_copybits
import daffodil.lib.daf_utils as daf_utils
import daffodil.lib.daf_namespace as daf_namespace
from daffodil.lib.daf_sql import sqlite3
import daffodil.sqldaf_read as sqldaf_read
import daffodil.sqldaf_select as sqldaf_select
import daffodil.sqldaf_modify as sqldaf_modify
import daffodil.sqldaf_apply as sqldaf_apply


RID_COL = 'daf_rid'                 # the INTEGER PRIMARY KEY of each table SqlDaf makes
REF_COL = 'daf_ref'                 # the reference to a parent row, in the queries of a selection
ORD_COL = 'daf_ord'                 # the order of the rows, in the queries of a selection
RESERVED_COLS = (RID_COL, REF_COL, ORD_COL)
TEMP_PREFIX = 'daf_tmp_'            # the prefix of every TEMP object SqlDaf makes
INDEX_PREFIX = 'daf_ix_'            # the prefix of every index SqlDaf makes
GROUP_INDEX_PREFIX = 'daf_gx_'      # the prefix of an index groupby() makes on the columns it groups by
META_TABLE = 'daf_tables'           # the table of metadata, one row per table SqlDaf made
SAVEPOINT = 'daf_sp'                # the name of a nested transaction. One name can be nested.
FILE_SCHEMA = 'daf_file'            # the schema of another database file, attached only to copy a table in or out

# dtypes that apply_dtypes() and from_csv() take.
# A str, a dict and a list are stored as their text, so nothing is converted. A Daf decodes a dict or list after to_daf().
STORABLE_DTYPES = (int, float, bool, str, dict, list)
TEXT_DTYPES = (str, dict, list)
MIN_SQLITE_VERSION: Tuple[int, int, int] = (3, 48, 0)     # ORDER BY in aggregates, JSON5, and 1000 function arguments.

# dtypes that the metadata can record, by name.
DTYPE_BY_NAME: Dict[str, Type] = {'int': int, 'float': float, 'bool': bool, 'str': str, 'list': list, 'dict': dict}


T_klosdaf: TypeAlias = "KeyedList[Any, SqlDaf]"     # a KeyedList of SqlDaf, one selection for each group.


# ---- the parts that differ between engines ----

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


# A cell of these classes is stored as its text, as `Daf.to_csv_file()` writes it. SQLite has no such type.
_TEXT_CELL_CLASSES: T_sa = {dict, list}


def null_to_none(row_la: Iterable[Any]) -> Tuple[Any, ...]:
    """
    A row on its way into SQL. Internal use.

    Python NULL becomes None, which SQLite stores as SQL NULL.
    A dict or a list becomes its text, as `Daf.to_csv_file()` writes it.
    A subclass of either, or a tuple, is passed as it is, and SQLite refuses it.
    """
    return tuple(None if val is NULL else str(val) if val.__class__ in _TEXT_CELL_CLASSES else val
                 for val in row_la)


def none_to_null(row_t: Tuple[Any, ...]) -> T_la:
    """ A row on its way out of SQL: None, from SQL NULL, becomes Python NULL. """
    return [NULL if val is None else val for val in row_t]


def value_sql(col: str) -> str:
    """
    A column as SQL, with the text '' read as SQL NULL. Internal use.

    Both are Python NULL when read. SqlDaf writes SQL NULL, but a table written by other code
    may hold ''. Grouping by this puts the two in one group, as a Daf has them.
    """
    return f"NULLIF({quote_name(col)}, '')"


def is_value_sql(col: str, val: Any) -> Tuple[str, Tuple[Any, ...]]:
    """
    A condition that a column holds this value, and its parameters. Internal use.

    Python NULL matches both SQL NULL and the text ''. Any other value matches with IS.
    Both forms can use an index on the column.
    """
    if val is NULL or val is None:
        return f"({quote_name(col)} IS NULL OR {quote_name(col)} = '')", ()
    return f'{quote_name(col)} IS ?', (val,)


#===================================
# Python aggregates for groupby_aggregate()
# Research, trials and timings: notes/engineering_notebook/2026-10-18_1428_sqldaf_group_reductions.md

def aggregate_cell_value(val: Any) -> Any:
    """
    A cell as the reduction function gets it. Internal use.

    SQL NULL is Python NULL. Text of a dict or a list is decoded, as JSON or PYON.
    Other values are kept.
    """
    if val is None:
        cell_a: Any = NULL
    elif val.__class__ is str and val[:1] in ('{', '['):
        cell_a = daf_utils.unflatten_val(val)
    else:
        cell_a = val
    return cell_a


def aggregate_result_cell(val: Any) -> Any:
    """
    A result value as a table cell. Internal use.

    A dict, list or tuple is compact JSON text.
    A bool is 1 or 0, and None is Python NULL, at any depth, by `normalize_val()`.
    """
    normal_a: Any = daf_utils.normalize_val(val)
    if isinstance(normal_a, (dict, list, tuple)):
        cell_a: Any = json.dumps(normal_a, separators=(',', ':'), ensure_ascii=False)
    else:
        cell_a = normal_a
    return cell_a


class CellReduction:
    """
    The aggregate of one column, for `groupby_aggregate()`. Internal use.

    A subclass sets func, col and call_kwargs.
    A cell that holds a dict is given to the function as the row.
    Any other cell is given as a row of one column.
    The result is the reduced dict, as JSON text.
    If the cells were not dicts, the result is the value of the column in the reduced dict.
    With sparse set, the result is always the reduced dict.
    """
    func:           Callable[..., Any]
    col:            str
    call_kwargs:    T_da
    sparse:         bool = False
    select_cols:    T_ls | None = None              # with sparse, the keys kept at the end, in this order.

    def __init__(self) -> None:
        self.reduction_da: T_da = {}
        self.saw_value: bool = False

    def step(self, cell: Any) -> None:
        val: Any = aggregate_cell_value(cell)
        if isinstance(val, dict):
            row_da: T_da = val
        elif val is NULL:
            row_da = {}                                 # an empty cell is an empty dict, as in Daf.reduce().
        else:
            row_da = {self.col: val}
            self.saw_value = True
        self.reduction_da = self.func(row_da, self.reduction_da, **self.call_kwargs)

    def finalize(self) -> Any:
        result_a: Any = self.reduction_da.get(self.col, NULL) if self.saw_value and not self.sparse else self.reduction_da
        if self.sparse and self.select_cols is not None:
            result_a = {col: self.reduction_da[col] for col in self.select_cols if col in self.reduction_da}
        return aggregate_result_cell(result_a)


class RowReduction:
    """
    The aggregate of several columns, for `groupby_aggregate()`. Internal use.

    A subclass sets func, read_cols, reduce_cols, result_cols and call_kwargs.
    Each cell of a row comes as one argument, so a float keeps every digit.
    The row is a dict of the columns, with each cell decoded.
    The result is a JSON array of the result columns, in order.
    """
    func:           Callable[..., Any]
    read_cols:      T_ls
    reduce_cols:    T_ls
    result_cols:    T_ls
    call_kwargs:    T_da

    def __init__(self) -> None:
        self.reduction_da: T_da = dict.fromkeys(self.reduce_cols, 0)       # as Daf.reduce() starts a row reduction.

    def step(self, *values_t: Any) -> None:
        row_da: T_da = {col: aggregate_cell_value(val) for col, val in zip(self.read_cols, values_t)}
        self.reduction_da = self.func(row_da, self.reduction_da, cols=self.reduce_cols, **self.call_kwargs)

    def finalize(self) -> str:
        results_la: T_la = [aggregate_result_cell(self.reduction_da.get(col, NULL)) for col in self.result_cols]
        results_text: str = json.dumps(results_la, separators=(',', ':'), ensure_ascii=False)
        return results_text


class GroupReduction:
    """
    The aggregate of `groupby_reduce()` by row. Internal use.

    A subclass sets func, read_cols, reduce_cols, result_cols and call_kwargs.
    Each cell of a row comes as one argument, as for RowReduction.
    The rows of the group are kept as they are stored, with SQL NULL as Python NULL.
    At the end they are one Daf, reduced by `Daf.reduce()`.
    So the function is called just as `Daf.groupby_reduce()` calls it.
    A plain sum then adds down each column, which is much faster than a call for each row.
    """
    func:           Callable[..., Any]
    read_cols:      T_ls
    reduce_cols:    T_ls
    result_cols:    T_ls
    call_kwargs:    T_da

    def __init__(self) -> None:
        self.rows_lol: T_lola = []

    def step(self, *values_t: Any) -> None:
        self.rows_lol.append([NULL if val is None else val for val in values_t])

    def finalize(self) -> str:
        group_daf: Daf = Daf(lol=self.rows_lol, cols=self.read_cols)
        reduction_da: T_da = cast(T_da, group_daf.reduce(self.func, cols=self.reduce_cols, **self.call_kwargs))
        results_la: T_la = [aggregate_result_cell(reduction_da.get(col, NULL)) for col in self.result_cols]
        results_text: str = json.dumps(results_la, separators=(',', ':'), ensure_ascii=False)
        return results_text


def _connect(path: str, **kwargs: Any) -> sqlite3.Connection:
    """
    Open a connection to a database file, after checking the SQLite version. Internal use.

    SqlDaf needs SQLite MIN_SQLITE_VERSION or later.
    An older one raises RuntimeError here, before any query, with how to get a newer one.
    """
    if sqlite3.sqlite_version_info < MIN_SQLITE_VERSION:
        raise RuntimeError(f"SqlDaf needs SQLite {'.'.join(map(str, MIN_SQLITE_VERSION))} or later. "
                           f"This Python has SQLite {sqlite3.sqlite_version}. "
                           f"Use a Python built with a newer SQLite. Every Python that uv installs has one.")
    conn: sqlite3.Connection = sqlite3.connect(path, **kwargs)
    return conn


def _index_exists_on(conn: sqlite3.Connection, name: str) -> bool:
    """ Say whether the connection sees an index of this name, in main or TEMP. Internal use. """
    return conn.execute("SELECT EXISTS (SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = ? "
                        "UNION ALL SELECT 1 FROM sqlite_temp_master WHERE type = 'index' AND name = ?)", (name, name)).fetchone()[0] == 1


def _indexes_of(conn: sqlite3.Connection, table: str) -> Dict[str, T_ls]:
    """ The indexes on a table, each with its columns in order. Internal use. """
    names_ls: T_ls = [row_t[1] for row_t in conn.execute(f'PRAGMA index_list({quote_name(table)})')]
    return {name: [row_t[2] for row_t in conn.execute(f'PRAGMA index_info({quote_name(name)})')] for name in names_ls}


def _table_exists_on(conn: sqlite3.Connection, name: str, schema: str = '') -> bool:
    """ Say whether the connection sees a table or view of this name: in `schema` only, or in main and TEMP. Internal use. """
    schemas_ls: T_ls = [schema] if schema else ['main', 'temp']
    found_t: Tuple[Any, ...] | None = conn.execute(
            ' UNION ALL '.join(f'SELECT 1 FROM {one_schema}.sqlite_master WHERE name = ?' for one_schema in schemas_ls),
            (name,) * len(schemas_ls)).fetchone()
    return found_t is not None


def _create_table_sql(conn: sqlite3.Connection, name: str, cols: T_ls, schema: str = 'main', check_exists: bool = True) -> str:
    """ Check a new table, and give the SQL that makes it in `schema`, main or temp. Internal use. """
    if check_exists and _table_exists_on(conn, name):
        raise ValueError(f"SqlDaf: a table named '{name}' already exists.")
    reserved_ls: T_ls = [col for col in cols if col in RESERVED_COLS]
    if reserved_ls:
        raise ValueError(f"SqlDaf: {reserved_ls} reserved, and cannot be column names: {list(RESERVED_COLS)}.")
    coldefs_ls: T_ls = [f'{quote_name(RID_COL)} INTEGER PRIMARY KEY'] + [quote_name(col) for col in cols]
    return f'CREATE TABLE {schema}.{quote_name(name)} ({", ".join(coldefs_ls)})'


class ReadOnlyTableError(sqlite3.DatabaseError):
    """
    A change to the data of a table opened with `read_only=True`.

    Writing a cell, adding or removing rows or columns, renaming a column and dropping the table
    raise it. Making an index or setting the keyfield do not, since the data stays the same.
    `copy()` the table, and change the copy.
    """


# What SQLite would do to the rows or columns of a table. The authorizer refuses these on a read-only table.
_DATA_WRITE_ACTIONS: T_sa = {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE, sqlite3.SQLITE_DROP_TABLE}
_READ_ONLY_MESSAGE: str = "A table opened with read_only=True cannot have its data changed. copy() it, and change the copy."


def _read_only_authorizer(read_only_ss: T_ss) -> Callable[..., int]:
    """
    An authorizer that refuses any change to the data of the tables in `read_only_ss`. Internal use.

    It is set on the connection while a read-only SqlDaf uses it. See the property `SqlDaf.conn`.

    SQLite calls it as each statement is compiled. An index, a TEMP table or a write to another table passes.
    For an ALTER TABLE, SQLite gives the schema first and the table second.
    """
    def authorize(action: int, arg1: str | None, arg2: str | None, db_name: str | None, source: str | None) -> int:
        if action == sqlite3.SQLITE_ALTER_TABLE:
            return sqlite3.SQLITE_DENY if arg1 == 'main' and arg2 in read_only_ss else sqlite3.SQLITE_OK
        if action in _DATA_WRITE_ACTIONS and db_name == 'main' and arg1 in read_only_ss:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    return authorize


def _allow_all(action: int, arg1: str | None, arg2: str | None, db_name: str | None, source: str | None) -> int:
    """
    An authorizer that allows every statement, for a SqlDaf that is not read-only. Internal use.

    Setting it, not None, removes a read-only guard. Before Python 3.11, `set_authorizer(None)`
    does not remove the authorizer, and every statement is refused.
    """
    return sqlite3.SQLITE_OK


def _is_refused(exc_info: BaseException) -> bool:
    """ Say whether an error is SQLite refusing a statement, as the read-only authorizer does. Internal use. """
    return isinstance(exc_info, sqlite3.DatabaseError) and not isinstance(exc_info, ReadOnlyTableError) \
        and str(exc_info) == 'not authorized'


@contextlib.contextmanager
def _transaction_on(conn: sqlite3.Connection) -> Iterator[None]:
    """ A transaction on the connection, or a savepoint inside one already open. See `SqlDaf.transaction()`. Internal use. """
    if not conn.in_transaction:
        conn.execute('BEGIN')
        try:
            yield
        except BaseException as exc_info:
            conn.execute('ROLLBACK')
            if _is_refused(exc_info):
                raise ReadOnlyTableError(_READ_ONLY_MESSAGE) from exc_info
            raise
        conn.execute('COMMIT')
    else:
        conn.execute(f'SAVEPOINT {SAVEPOINT}')
        try:
            yield
        except BaseException as exc_info:
            conn.execute(f'ROLLBACK TO {SAVEPOINT}')
            conn.execute(f'RELEASE {SAVEPOINT}')
            if _is_refused(exc_info):
                raise ReadOnlyTableError(_READ_ONLY_MESSAGE) from exc_info
            raise
        conn.execute(f'RELEASE {SAVEPOINT}')


def _load_rows(
        conn:       sqlite3.Connection,             # the connection.
        name:       str,                            # the name of the new table.
        cols:       T_ls,                           # its column names, in order.
        keyfield:   str,                            # the column that holds the key of each row, or ''.
        dtypes:     T_dtype_dict,                   # the types of the columns, for apply_dtypes().
        rows:       Iterable[Tuple[Any, ...]],      # the rows, with None for a missing value.
        ) -> None:
    """ Make a permanent table, add the rows and record its metadata, in one transaction. If any row fails, nothing is made. Internal use. """
    table_sql: str = _create_table_sql(conn, name, cols, 'main')
    insert_sql: str = (f'INSERT INTO main.{quote_name(name)} ({", ".join(quote_name(col) for col in cols)}) '
                       f'VALUES ({placeholders(len(cols))})')
    with _transaction_on(conn):
        conn.execute(table_sql)
        if cols:                                    # a table with no columns has no rows to add.
            conn.executemany(insert_sql, rows)
        _meta_write(conn, name, keyfield, dtypes)


# ---- the metadata of each table, kept in the database ----

def _meta_read(conn: sqlite3.Connection, name: str, schema: str = 'main') -> Tuple[str, T_dtype_dict] | None:
    """ The keyfield and dtypes recorded for a table in `schema`, or None if there is no record. Internal use. """
    if not _table_exists_on(conn, META_TABLE, schema):
        return None
    found_t: Tuple[Any, ...] | None = conn.execute(
            f'SELECT keyfield, dtypes FROM {schema}.{quote_name(META_TABLE)} WHERE name = ?', (name,)).fetchone()
    if found_t is None:
        return None
    names_ds: Dict[str, str] = json.loads(found_t[1]) if found_t[1] else {}
    return found_t[0] or '', {col: DTYPE_BY_NAME[type_name] for col, type_name in names_ds.items()}


def _meta_write(conn: sqlite3.Connection, name: str, keyfield: str, dtypes: T_dtype_dict, schema: str = 'main') -> None:
    """ Record the keyfield and dtypes of a table in `schema`, replacing any record. Run it in the same transaction as the change. Internal use. """
    unknown_ls: T_ls = [col for col, dtype in dtypes.items() if getattr(dtype, '__name__', '') not in DTYPE_BY_NAME]
    if unknown_ls:
        raise NotImplementedError(f"SqlDaf: the dtypes of the columns {unknown_ls} cannot be recorded yet. "
                                  f"Only {list(DTYPE_BY_NAME)}.")
    conn.execute(f'CREATE TABLE IF NOT EXISTS {schema}.{quote_name(META_TABLE)} '
                 f'(name TEXT PRIMARY KEY, keyfield TEXT, dtypes TEXT, made TEXT)')
    conn.execute(f"INSERT INTO {schema}.{quote_name(META_TABLE)} (name, keyfield, dtypes, made) "
                 f"VALUES (?, ?, ?, datetime('now')) "
                 f"ON CONFLICT (name) DO UPDATE SET keyfield = excluded.keyfield, dtypes = excluded.dtypes",
                 (name, keyfield, json.dumps({col: dtype.__name__ for col, dtype in dtypes.items()})))


@contextlib.contextmanager
def _file_schema(conn: sqlite3.Connection, conn_path: str, path: str, read_only: bool = False) -> Iterator[str]:
    """
    The schema by which a connection reads or writes another database file, for one block. Internal use.

    Its own file is main, and nothing is attached.
    Another file is attached for the block only, and detached after it, even on an error.
    With `read_only`, it is attached as a URI with mode=ro, so SQLite refuses any write to it.
    """
    if os.path.realpath(path) == os.path.realpath(conn_path):
        yield 'main'
        return
    if conn.in_transaction:
        raise ValueError(f"SqlDaf: {path} cannot be read or written inside a transaction. "
                         f"Copy a table in or out before the transaction, or after it.")
    attach_path: str = f'file:{urllib.parse.quote(os.path.abspath(path))}?mode=ro' if read_only else path
    conn.execute(f'ATTACH DATABASE ? AS {FILE_SCHEMA}', (attach_path,))
    try:
        yield FILE_SCHEMA
    finally:
        conn.execute(f'DETACH DATABASE {FILE_SCHEMA}')


def _release_brought_in(conn: sqlite3.Connection, counts_di: Dict[str, int], names: Tuple[str, ...]) -> None:
    """
    Give up a join's use of TEMP copies, and drop each that no other unread join uses. Internal use.

    It runs once, when the join is first read, or when the join is gone. A copy that cannot be
    dropped now, as on a closed or busy connection, goes when the connection closes.
    """
    name: str
    for name in names:
        counts_di[name] = counts_di.get(name, 1) - 1
        if counts_di[name] > 0:
            continue
        del counts_di[name]
        try:
            conn.execute(f'DROP TABLE IF EXISTS temp.{quote_name(name)}')
        except sqlite3.Error:
            pass


def _meta_delete(conn: sqlite3.Connection, name: str) -> None:
    """ Remove the record of a table, if there is one. Internal use. """
    if _table_exists_on(conn, META_TABLE, 'main'):
        conn.execute(f'DELETE FROM main.{quote_name(META_TABLE)} WHERE name = ?', (name,))


class SqlDaf(Daf):
    """
    A Daf whose rows live in a SQLite table.

    Use it for a table too large to hold in memory. The rows stay on disk in a SQLite database
    file, and SQL does the work. When the rows you need are few, pull them into an ordinary Daf
    with `to_daf()`, and work on them there.

    A SqlDaf is a Daf, so `isinstance(sdaf, Daf)` is True. But it has only the methods
    written for SQL. Every other Daf method raises NotImplementedError, which names it. So
    nothing quietly loads a large table into memory. For anything not covered yet, `sql()`
    runs a SQL statement of your own.

    The database is the file named by the class setting `SqlDaf.db_path`. Set it once, for
    example once for each job, and every SqlDaf opens that file. A `db_path` given to the
    constructor overrides it for that table. With neither, the constructor raises ValueError.
    The class setting `engine` is 'sqlite', the only engine so far.
    It needs SQLite 3.48 or later. See the guide page on SqlDaf for how to get it.

    Make a table with the constructor, as a Daf is made: give its name, its columns and its
    rows. Open a table that is already in the file by its name alone. Its keyfield and dtypes
    come back as they were, from the metadata that SqlDaf keeps in the table daf_tables of the
    same file. A table made by other code has no such record, so give them to the constructor.
    `SqlDaf.from_csv()` and `SqlDaf.from_daf()` load a file or a Daf into a new table.
    A join, `concat()` and keys with a table in another database file bring it in themselves, as a
    TEMP copy. `SqlDaf.from_db()` copies a table in as a permanent one.

    A SqlDaf is a parent or a child.

    - A parent holds its rows, in a table of its own: one made by the constructor, a loader or
      `copy()`, or one already in the file.
    - A child is a selection. It holds references to rows of its parent, in order, as a
      selection of a Daf holds the same row lists. `select_where()`, `select_by_dict()`,
      `select_where_sql()` and indexing, as in `sdaf[2:5]` or `sdaf['b2']`, each make one. A
      child of a child is a child of the same parent. `is_child` tells them apart, and
      `parent_sdaf` is the parent of a child.

    Making a child does not read its rows. Some methods run a query at once for their
    arguments: `select_krows()` checks the keys, `select_by_dict()` with `expectmax` finds
    its rows to count them, and `groupby()` finds the groups. The rows of a child are fixed the first time
    it is read whole: by `to_daf()`, `len()`, `bool()`, `to_dict()`, `to_list()` or
    `to_value()`. After that, it always means the same rows, as a selection of a Daf does.
    `select_record()` and `iloc()` read one row and fix nothing, unless `iloc()` is given a
    negative position, which needs `len()`. A change to the values in those
    rows shows in it, since it refers to them. `copy(inplace=True)` gives it rows of its own.

    A child made from a child not read yet carries that child's query inside its own. So a chain
    of selections, each made from the last before any is read, nests one query per step. SQLite
    stops at about 400 levels, 413 with SQLite 3.53, with OperationalError "Recursion limit". It
    raises at once, and does not hang. A child that has been read starts a fresh query, so read
    one, as with `len()`, every so often in a loop that selects from its own last result. A chain
    of 2,000 selections, each read as it was made, ran in 1.7 s.

    Every SqlDaf on one file shares one connection, which it holds in its attributes. So any two
    of them can be joined, and a child of one can be used with another. They also share its
    transactions: see `transaction()`. `SqlDaf.close()` closes the file, for every SqlDaf on it.
    Otherwise it stays open until the program ends. Each statement is committed when it
    finishes. So committed changes to permanent tables remain, whether the file is closed or the
    program ends. `delete()` deletes a table from the file.

    A table made by `copy()`, or by reading a join, is a TEMP table. It belongs to the
    connection, and is removed when the connection closes. `save()` makes it a permanent table in the
    file, under a name, to be opened again later.

    Do not delete rows from a parent that has children. `len()` of a child counts its
    references, so it would still count a deleted row, though `to_daf()` leaves it out. Worse,
    SQLite gives the daf_rid of a deleted last row to the next row added. A child made before
    the delete then shows that new row, which it never selected. Leave rows out with a selection
    instead, as in a Daf.

    A parent that has had rows deleted cannot be read by position. Positions in a parent are
    looked up by daf_rid, and a delete leaves a gap there. So `iloc()`, `iloc_exists()` and
    `sdaf[n]` or `sdaf[n:m]` on it raise ValueError. Read positions from `copy()`, which numbers
    the rows again, or select the rows you want. A selection numbers its own rows, so positions
    in it always work.

    To read one row, use `select_record()` by key or `iloc()` by position. Each is one query,
    returns a dict, and makes no selection.

    The methods, by what they do:

    - Size: `len()`, `bool()`, `num_rows()`, `num_cols()`, `shape()` and `columns()`.
    - Looping, in batches: `for row in sdaf`, `iter_dict()`, `iter_klist()` and `iter_list()`.
    - One row: `select_record()` by key, and `iloc()` by position.
    - Columns: `col()`, `col_to_la()`, `icol()`, `icol_to_la()` and `cols_to_dol()`.
    - Selecting: `select_where()`, `select_by_dict()`, `select_where_sql()`, `split_where()`,
      `select_cols()` and indexing.
    - By key: `key in sdaf`, `keys()`, `select_records_daf()`, `select_krows()`,
      `get_existing_keys()` and `delete_records()`.
    - The first match: `select_first_row_by_dict()`.
    - Groups and counts: `groupby()`, `groupby_cols()`, `group_where()`, `groupsum_daf()`,
      `sum()`, `daf_valuecount()`, `value_counts_daf()` and the `valuecounts_for_*` methods.
    - Functions over rows: `apply()` and `reduce()`.
    - Adding rows: `append()`, `record_append()`, `extend()`, `assign_record()` and `concat()`.
    - Changing values: `sdaf[key, col] = v`, `sdaf[key] = row`, `sdaf[:, col] = values`,
      `update_record_krow()`, `update_by_keylist()`, `assign_col()`, `apply_to_col()` and `apply_in_place()`.
    - Columns and layout: `insert_col()`, `drop_cols()`, `rename_cols()`, `set_cols()`,
      `set_dtypes()`, `attach_schema()` and `clone_empty()`.
    - Repeated keys: `remove_dups()`. Sorting: `sort_by_colname()` and `sort_by_colnames()`.
    - Joining: `join()`. Choosing columns: `calc_cols()`.
    - Into memory: `to_daf()`, `to_dict()`, `to_list()`, `to_value()`, `to_lod()`, `to_dod()`
      and `to_json()`.
    - Display and files: `to_md()`, `to_md_cols()`, `to_csv_buff()` and `to_csv_file()`.
    - Tables: `from_csv()`, `from_daf()`, `from_db()`, `copy()`, `save()`, `delete()` and `table_exists()`.
    - The database: `close()`, `transaction()` and `sql()`, which runs any SQL statement.
    - Keys and types: `set_keyfield()`, `ensure_key_index()` and `apply_dtypes()`.

    Work with large tables in whole statements. One query costs about 12 to 24 us, against
    under 1 us for a lookup in memory, because each one passes through sqlite3. Many rows in one
    statement are far cheaper per row.

    There are two kinds of NULL. Python NULL is daffodil's missing value: the empty string '',
    which daffodil names NULL, as in `from daffodil.daf import NULL`. SQL NULL is the database's.
    A Python None written to a SqlDaf is stored as SQL NULL too, so it reads back as Python NULL. SqlDaf writes Python NULL as SQL NULL, and reads SQL NULL back as
    Python NULL. A table written by other code may hold the text '' instead, which also reads as
    Python NULL. So the methods that group, count or select by value treat SQL NULL and '' as the
    same value, as a Daf does. In Python, test a value with `val is NULL`. This works because
    CPython keeps one shared empty string, and it is faster than `val == ''`. In your own SQL,
    test for both, as `(col IS NULL OR col = '')`.

    Avoid a Python NULL key, the empty string in the keyfield. If you allow one, the results of
    SqlDaf are undefined. The key methods compare keys with = in SQL, and SQL NULL equals
    nothing, so such a row is not found by key, `select_records_daf()`, `delete_records()` or
    `join()`. The unique index also lets several SQL NULL keys in. A Daf treats a Python NULL key
    as a key like any other, so the two differ. Give every row a real key. Making the key index
    raises ValueError on a Python NULL key, and so does `concat()` of one.

    Until `apply_dtypes()` converts them, values loaded from a CSV file are text, in SQL too.
    Text is never less than a number in SQLite, so `WHERE n < 5` finds nothing on a text
    column, and MAX() gives the last in text order.

    The `name` is the name of the table in the database. A child gets a generated name, such as
    daf_tmp_marks_v3, which names its reference table once its rows are found.

    Examples:
        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> marks_sdaf = SqlDaf('marks', cols=['ballot_id', 'v'], lol=[['b1', 1], ['b2', '']], keyfield='ballot_id')
        >>> marks_sdaf.to_daf()
        <BLANKLINE>
        | ballot_id | v  |
        | --------: | -: |
        |        b1 |  1 |
        |        b2 |    |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield='ballot_id'; name='marks'
        <BLANKLINE>
        >>> marks_sdaf.sql('SELECT ballot_id FROM marks WHERE v IS NULL').to_list()
        ['b2']
        >>> SqlDaf('marks').keyfield
        'ballot_id'
    """

    engine: str = 'sqlite'
    """ The database engine. Only SQLite so far. """

    db_path: str = ''
    """ The database file every SqlDaf opens, unless its constructor names another. Set it once, at the start. """

    # The connection open on each file, with its shared state, by thread and real path.
    # Every SqlDaf on a file shares it. `close()` closes it.
    _open_conns: ClassVar[Dict[Tuple[int, str], Tuple[sqlite3.Connection, T_da]]] = {}

    def __init__(
            self,
            name:       str,                                    # the name of the table.
            cols:       T_ls | None = None,                     # the columns of a new table. None opens one that is there.
            lol:        Iterable[Iterable[Any]] | None = None,  # the rows of a new table.
            keyfield:   str = '',                               # the column that holds the key of each row.
            dtypes:     T_dtype_dict | None = None,             # the types of the columns, for apply_dtypes().
            db_path:    str = '',                               # the database file. SqlDaf.db_path if empty.
            read_only:  bool = False,                           # if True, a change to the table's data raises ReadOnlyTableError.
            ) -> None:
        """
        Make a new table, or open one that is in the database file.

        With `cols`, a new table is made in the file, with the rows of `lol`, read one at a
        time, so an iterator of any length works. The table, its rows and its metadata are made
        in one transaction: if any row fails, nothing is made. A Python NULL cell is stored
        as SQL NULL. The table gets a hidden first column, daf_rid, an INTEGER PRIMARY KEY, which
        numbers the rows in order. It is not one of the `columns()`. The columns are declared
        with no SQL type, so an int stays an int, and text stays text. A cell that holds a dict
        or a list is stored as its text, as `Daf.to_csv_file()` writes it, and is read back as
        that text. `to_daf().apply_dtypes()` decodes it, with the column's dtype dict or list, as
        for a table loaded from a CSV file.

        Without `cols`, the table of that name is opened. Its keyfield and dtypes are those
        recorded when it was made, unless they are given here. Nothing is read yet.

        The keyfield is only recorded. Its unique index is made when a key is first used, as a
        Daf builds its key index.

        With `read_only=True`, the table's data cannot be changed: writing a cell, adding or
        removing rows or columns, renaming a column and dropping the table raise ReadOnlyTableError.
        Selections, copies, sorts, joins, indexes and the keyfield still work, since they leave the
        data as it is. To change it, `copy()` the table and change the copy. A Daf changes only
        memory until it is written out. A SqlDaf writes to its file at once, and nothing in the
        file shows it later. SQLite enforces it, so no method can slip past it, `sql()` included.

        Read-only holds for this SqlDaf and its selections only. Another SqlDaf opened on the same
        table without `read_only` can change it, and this one then reads the changed rows.

        Args:
            name: The name of the table.
            cols: The column names of a new table, in order. None to open a table that is there.
            lol: The rows of a new table, each the values in the order of `cols`.
            keyfield: The column that holds the key of each row. For a table that is opened, the recorded one if empty.
            dtypes: The types of the columns, kept for `apply_dtypes()`. Nothing is converted.
            db_path: The database file. The class setting `SqlDaf.db_path` if empty.
            read_only: If True, a change to the table's data raises ReadOnlyTableError.

        Raises:
            ValueError: There is no database path, a new table's name is taken, a column is named daf_rid, or `lol` has no `cols`.
            KeyError: There is no table of that name to open, or the keyfield is not a column.
            FileNotFoundError: A table is to be opened, and the database file is not there.
            NotImplementedError: The class setting `engine` is not 'sqlite'.
        """
        path: str = db_path or type(self).db_path
        if not path:
            raise ValueError("SqlDaf: no database. Set SqlDaf.db_path, or give db_path to the constructor.")
        if type(self).engine != 'sqlite':
            raise NotImplementedError(f"SqlDaf: the engine '{type(self).engine}' is not supported yet. Only 'sqlite'.")
        if cols is None and lol is not None:
            raise ValueError("SqlDaf: give the cols of the new table along with its lol.")
        if cols is None and not os.path.exists(path):
            raise FileNotFoundError(path)

        conn, conn_da = type(self)._conn_for(path)
        if cols is not None:
            col_ls: T_ls = list(cols)
            if keyfield and keyfield not in col_ls:
                raise KeyError(keyfield)
            row_la: Iterable[Any]
            _load_rows(conn, name, col_ls, keyfield, dict(dtypes) if dtypes else {},
                       (null_to_none(row_la) for row_la in (lol if lol is not None else [])))
        self._open_on(conn, conn_da, path, name, keyfield, dtypes, is_temp=False, use_meta=cols is None)

        # Read-only holds for this SqlDaf and its selections. See the property conn.
        if read_only:
            self.read_only = True
            conn_da['has_read_only'] = True


    @classmethod
    def _conn_key(cls, path: str) -> Tuple[int, str]:
        """ The key of a file's connection: the thread, since sqlite3 ties a connection to one, and the real path. Internal use. """
        return threading.get_ident(), os.path.realpath(path)


    @classmethod
    def _conn_for(cls, path: str) -> Tuple[sqlite3.Connection, T_da]:
        """ The connection open on this file, and its shared state, opened first if there is none. Internal use. """
        key_t: Tuple[int, str] = cls._conn_key(path)
        if key_t not in SqlDaf._open_conns:
            # URI filenames let from_db() attach a source read-only. A plain path opens as before.
            SqlDaf._open_conns[key_t] = (_connect(path, isolation_level=None, uri=True),
                                         {'func_error': None, 'temp_count': 0, 'func_count': 0, 'no_gaps_ss': set(), 'has_read_only': False, 'guard': None,
                                          'brought_in_di': {}})
        return SqlDaf._open_conns[key_t]


    def _open_on(
            self,
            conn:       sqlite3.Connection,             # the connection to hold.
            conn_da:    T_da,                           # the state shared by every SqlDaf on that connection.
            path:       str,                            # the database file.
            name:       str,                            # the table.
            keyfield:   str,                            # the keyfield, or '' for the recorded one.
            dtypes:     T_dtype_dict | None,            # the dtypes, or None for the recorded ones.
            is_temp:    bool,                           # True for a TEMP table.
            use_meta:   bool,                           # True to fill in the keyfield and dtypes from the metadata.
            ) -> None:
        """ Set this SqlDaf up as a parent on a table the connection sees. Internal use. """
        info_lot: List[Tuple[Any, ...]] = conn.execute(f'PRAGMA table_info({quote_name(name)})').fetchall()
        if not info_lot:
            raise KeyError(name)
        all_cols: T_ls = [info_t[1] for info_t in info_lot]
        cols: T_ls = [col for col in all_cols if col != RID_COL]

        if use_meta:
            meta_t: Tuple[str, T_dtype_dict] | None = _meta_read(conn, name)
            if meta_t is not None:
                keyfield = keyfield or meta_t[0]
                dtypes = dtypes if dtypes is not None else meta_t[1]
        if keyfield and keyfield not in cols:
            raise KeyError(keyfield)

        self.conn                           = conn
        self._conn_da:  T_da            = conn_da       # the state shared by every SqlDaf on this connection.
        self.db_path:   str             = path          # type: ignore[misc]
        self._name:     str             = name
        self.keyfield:  str             = keyfield
        self.has_rid:   bool            = RID_COL in all_cols
        self.hd:        T_di            = {col: idx for idx, col in enumerate(cols)}
        self.dtypes:    T_dtype_dict    = dict(dtypes) if dtypes else {}
        self.is_temp:   bool            = is_temp

        # A parent has no parent. A child has its parent, and either the query that finds its
        # rows, not run yet, or the name of the reference table that holds them. A join not read
        # yet has no parent and a query.
        self.parent_sdaf:   SqlDaf | None   = None
        self.pending_sql:   str             = ''
        self.pending_params: Tuple[Any, ...] = ()
        self.is_resolved:   bool            = True


    def _new_on_conn(self, name: str, keyfield: str, dtypes: T_dtype_dict | None, is_temp: bool) -> 'SqlDaf':
        """ A new parent SqlDaf on a table of this connection. Internal use. """
        new_sdaf: SqlDaf = type(self).__new__(type(self))
        new_sdaf._open_on(self.conn, self._conn_da, self.db_path, name, keyfield, dtypes, is_temp, use_meta=False)
        return new_sdaf


    @classmethod
    def _new_selection(
            cls,
            parent_sdaf:    'SqlDaf',                   # the SqlDaf the rows are selected from.
            pending_sql:    str,                        # the query that finds the rows, with daf_ref and daf_ord.
            pending_params: Tuple[Any, ...],            # its parameters.
            keyfield:       str | None = None,          # the keyfield of the selection. The parent's if None.
            ) -> 'SqlDaf':
        """ Make a child, its rows not found yet. Internal use. """
        root_sdaf: SqlDaf = parent_sdaf._root()
        root_sdaf._resolve()                            # a join not read yet becomes a table first.
        root_sdaf._ref_col()                            # raises if the parent has no reference that survives VACUUM.
        sel_sdaf: SqlDaf = cls.__new__(cls)
        sel_sdaf.conn           = parent_sdaf.conn
        sel_sdaf._conn_da       = parent_sdaf._conn_da
        sel_sdaf.db_path        = parent_sdaf.db_path
        sel_sdaf._name          = parent_sdaf._next_temp_name(root_sdaf.name)
        sel_sdaf.keyfield       = parent_sdaf.keyfield if keyfield is None else keyfield
        sel_sdaf.has_rid        = False
        sel_sdaf.hd             = dict(parent_sdaf.hd)
        sel_sdaf.dtypes         = dict(parent_sdaf.dtypes)
        sel_sdaf.is_temp        = True
        sel_sdaf.parent_sdaf    = root_sdaf
        sel_sdaf.pending_sql    = pending_sql
        sel_sdaf.pending_params = pending_params
        sel_sdaf.is_resolved    = False
        sel_sdaf.read_only      = parent_sdaf.read_only
        return sel_sdaf


    @classmethod
    def from_csv(                                                           # type: ignore[override]
            cls,
            path:       str | Path | List[str | Path],  # the CSV file, a glob pattern, or a list of files.
            name:       str = '',                       # the table name. The first file name without .csv if empty.
            keyfield:   str = '',                       # the column that holds the key of each row.
            delimiter:  str = ',',                      # the field separator.
            db_path:    str = '',                       # the database file. SqlDaf.db_path if empty.
            dtypes:     T_dtype_dict | None = None,     # the types to convert the columns to, as they are loaded.
            strict_header: bool = True,                 # True needs the first file's header in every file. False places values by name.
            ) -> 'SqlDaf':
        """
        Load one or more CSV files into a new table, without building it in memory.

        Give one file, a glob pattern such as `chunks/votes_*.csv`, or a list of files. The
        files of a pattern are loaded in sorted order, and a list in its own order. All the
        rows go into one table. This is the interface of `read_csv()` in dask.

        Each file has the column names in its first line. SQLite puts the values of a row into the
        columns by position, so a file whose columns differ would load into the wrong columns.
        By default, every file must have the names of the first file, in the same order, or
        ValueError names both files. That also stops a file of another kind.

        With `strict_header=False`, a value goes into the table by the name of its column, so files
        whose columns differ in order or in set load safely. The table has the columns of the
        first file, then each new name in the order it is first seen. A file that lacks a column
        gives Python NULL in it. A misspelled name, or a file of another kind, becomes a new column
        and is not flagged.

        Then the first line of every file is read before any row. A file with the table's own columns
        is read as it is. Only a file whose columns differ has its values placed by name. On
        the 20 hunks of a BIF table, 344,347 rows, one hunk in another order, the load took 5.54 s,
        the same as without it. Placing every file by name would cost about 13% more.

        An empty file has no names and no rows, so it is skipped. The rest is read one row at a
        time, so no file is ever all in memory.

        Without `dtypes`, every value is stored as the text of the file. With `dtypes`, the
        columns are converted as the rows are loaded, as `apply_dtypes()` would convert them.
        That saves a second pass over the table. On 200,000 rows with five int columns, the load
        took 1.0 s with `dtypes`, against 1.4 s for a text load and then `apply_dtypes()`.
        A dtype may name a column that is not there. The dtypes are kept for the table. A dict or
        list dtype keeps the text, which a Daf decodes after `to_daf()`.

        An empty cell is stored as SQL NULL, and comes back as NULL. A row shorter than the
        header is padded with Python NULL, and a longer one raises ValueError. As with the
        constructor, if any row fails, no table is made.

        Args:
            path: The CSV file, a glob pattern, or a list of CSV files.
            name: The name of the new table. The name of the first file without its extension if empty.
            keyfield: The column that holds the key of each row.
            delimiter: The field separator.
            db_path: The database file. The class setting `SqlDaf.db_path` if empty.
            dtypes: The type of each column, converted as the rows are loaded. Text is kept if None.
            strict_header: If True, every file must have the first file's names, in order. If False, place each file's values by column name.

        Returns:
            A SqlDaf for the new table.

        Raises:
            FileNotFoundError: A file is missing, or a pattern matches no file.
            ValueError: A table of that name exists, a column is named daf_rid, a row is longer than its header,
                or the column names of a file differ from those of the first file, unless `strict_header=False`.
            KeyError: The keyfield is not one of the columns.
            NotImplementedError: A dtype is not int, float, bool, str, dict or list.
        """
        paths_ls: T_ls = _csv_paths(path)
        name = name or os.path.splitext(os.path.basename(paths_ls[0]))[0]
        dtypes_d: T_dtype_dict = dict(dtypes) if dtypes else {}
        unstorable_ls: T_ls = [col for col, dtype in dtypes_d.items() if dtype not in STORABLE_DTYPES]
        if unstorable_ls:
            raise NotImplementedError(f"SqlDaf.from_csv cannot store {[dtypes_d[col] for col in unstorable_ls]} "
                                      f"for the columns {unstorable_ls} yet. Only int, float, bool, str, dict and list.")

        # ---- the columns of each file, read before any row, so the table has every column
        # Empty and repeated names are made unique, as in Daf. An empty file has none.
        file_cols_lol: List[T_ls] = []
        first_header_ls: T_ls = []
        csv_path: str
        for csv_path in paths_ls:
            with open(csv_path, newline='', encoding='utf-8') as csv_f:
                header_ls: T_ls = next(csv.reader(csv_f, delimiter=delimiter), [])
            file_cols_lol.append(Daf(cols=header_ls).columns() if header_ls else [])
            first_header_ls = first_header_ls or header_ls
            if strict_header and header_ls:
                break                                   # only the first header is needed. Each file is checked as it is read.
        first_idx: int = next((idx for idx, file_cols_ls in enumerate(file_cols_lol) if file_cols_ls), 0)
        first_path: str = paths_ls[first_idx]
        cols: T_ls = list(file_cols_lol[first_idx])
        if not strict_header:
            seen_ss: T_ss = set(cols)
            file_cols_ls: T_ls
            for file_cols_ls in file_cols_lol:
                col: str
                for col in file_cols_ls:
                    if col not in seen_ss:
                        seen_ss.add(col)
                        cols.append(col)
        width: int = len(cols)
        col_idx_di: T_di = {col: idx for idx, col in enumerate(cols)}

        # ---- a converter for each column, as apply_dtypes() uses. None keeps the text.
        converters_l: List[Callable[[Any], Any] | None] = [
                daf_utils.get_converter(dtypes_d[col]) if col in dtypes_d and dtypes_d[col] not in TEXT_DTYPES else None
                for col in cols]

        def rows() -> Iterator[T_la]:
            csv_path: str
            for csv_path in paths_ls:
                with open(csv_path, newline='', encoding='utf-8') as csv_f:
                    reader: Iterator[T_ls] = csv.reader(csv_f, delimiter=delimiter)
                    file_header_ls: T_ls = next(reader, [])
                    if not file_header_ls:
                        continue                        # an empty file
                    file_cols_ls: T_ls = Daf(cols=file_header_ls).columns()

                    # ---- where each value of this file goes, chosen once, before its rows
                    # None keeps the order, for a file with the table's columns.
                    places_li: List[int] | None = None
                    if strict_header and file_header_ls != first_header_ls:
                        raise ValueError(f"SqlDaf.from_csv: the column names of {csv_path} differ from those of {first_path}.")
                    if file_cols_ls != cols:
                        places_li = [col_idx_di[col] for col in file_cols_ls]
                    file_width: int = len(file_cols_ls)

                    row_ls: T_ls
                    for row_ls in reader:
                        if len(row_ls) > file_width:
                            raise ValueError(f"SqlDaf.from_csv: a row of {csv_path} has {len(row_ls)} values, "
                                             f"more than the {file_width} columns.")
                        if places_li is None:
                            row_ls = row_ls + [NULL] * (width - len(row_ls))
                        else:
                            placed_ls: T_ls = [NULL] * width
                            place: int
                            val: str
                            for place, val in zip(places_li, row_ls):
                                placed_ls[place] = val
                            row_ls = placed_ls
                        yield [(NULL if val == '' else val) if converter is None else converter(val)
                               for val, converter in zip(row_ls, converters_l)]

        return cls(name, cols=cols, lol=rows(), keyfield=keyfield, db_path=db_path, dtypes=dtypes_d or None)


    @classmethod
    def from_daf(
            cls,
            data_daf:   Daf,                            # the Daf to write. It is not changed.
            name:       str,                            # the name of the new table.
            keyfield:   str = '',                       # the keyfield. The Daf's if empty.
            db_path:    str = '',                       # the database file. SqlDaf.db_path if empty.
            ) -> 'SqlDaf':
        """
        Write an in-memory Daf into a new table.

        Each value is stored as it is, so an int stays an int. A NULL cell is stored as SQL NULL.
        The new SqlDaf keeps the Daf's dtypes, for `apply_dtypes()`. The keyfield is the Daf's,
        unless one is given. As with the constructor, if any row fails, no table is made.

        Args:
            data_daf: The Daf to write. It is not changed.
            name: The name of the new table.
            keyfield: The column that holds the key of each row. The Daf's keyfield if empty.
            db_path: The database file. The class setting `SqlDaf.db_path` if empty.

        Returns:
            A SqlDaf for the new table.

        Raises:
            ValueError: A table of that name exists, or a column is named daf_rid.
            KeyError: The keyfield is not one of the columns.
        """
        keyfield = keyfield or (data_daf.keyfield if isinstance(data_daf.keyfield, str) else '')
        dtypes: T_dtype_dict | None = data_daf.dtypes if isinstance(data_daf.dtypes, dict) and data_daf.dtypes else None
        return cls(name, cols=data_daf.columns(), lol=data_daf.lol, keyfield=keyfield, dtypes=dtypes, db_path=db_path)


    @classmethod
    def from_db(
            cls,
            path:       str | Path,                     # the database file that holds the table. It is only read.
            name:       str,                            # the table in that file.
            new_name:   str = '',                       # the name of the copy. `name` if empty.
            db_path:    str = '',                       # the working file the copy goes in. SqlDaf.db_path if empty.
            ) -> 'SqlDaf':
        """
        Copy a table from another database file into the working file, and open the copy.

        It makes a permanent table in the working file. Use it to build a file named for the table a
        stage makes, with each input copied into it. A join, `concat()` or keys with a table in another
        file need no copy made this way: they bring the other table in themselves, as a TEMP copy.

        The source file is only read, once, and never changed. It is attached read-only, so SQLite
        refuses any write to it. A change made to the copy stays in the working file. To keep a result in a file of its own, write it with `save(db_path=)`.

        The rows are copied in their order, with the values as stored. The keyfield and dtypes
        are those recorded for the table in the source file, if any. Only this table and its own
        record are copied. The other tables of the source file stay there.

        The source file is attached only during the copy, which is one SQL statement, so no row
        passes through Python. On 1,000,000 rows of 5 columns the copy took 0.62 s. Through Python
        it took 2.54 s. Nothing stays attached, so any number of files can be copied in.

        A source in the working file itself is copied within it, so `new_name` must differ.

        Args:
            path: The database file that holds the table. It is only read.
            name: The table in that file.
            new_name: The name of the copy in the working file. `name` if empty.
            db_path: The working file. The class setting `SqlDaf.db_path` if empty.

        Returns:
            A SqlDaf for the copy, a permanent table in the working file.

        Raises:
            FileNotFoundError: The source file is not there.
            KeyError: The source file has no table of that name.
            ValueError: There is no working file, the working file has a table of the new name,
                or a transaction is open on the working file.

        Examples:
            >>> import tempfile
            >>> job_dir = tempfile.mkdtemp()
            >>> bif_sdaf = SqlDaf.from_daf(Daf(lol=[['b1', 'P1'], ['b2', 'P2']], cols=['ballot_id', 'precinct'], keyfield='ballot_id'),
            ...                            name='bif', db_path=job_dir + '/bif.db')
            >>> bif_sdaf.to_daf()
            <BLANKLINE>
            | ballot_id | precinct |
            | --------: | -------: |
            |        b1 |       P1 |
            |        b2 |       P2 |
            <BLANKLINE>
            %% daf rows=2; cols=2; keyfield='ballot_id'; name='bif'
            <BLANKLINE>
            >>> SqlDaf.db_path = job_dir + '/work.db'
            >>> work_sdaf = SqlDaf.from_db(job_dir + '/bif.db', 'bif')
            >>> work_sdaf.db_path == SqlDaf.db_path, work_sdaf.keyfield, len(work_sdaf)
            (True, 'ballot_id', 2)

            Give the copy another name with `new_name`, and read from a file other than the working one:

            >>> import os, tempfile
            >>> source_path = os.path.join(tempfile.mkdtemp(), 'source.db')
            >>> parts_sdaf = SqlDaf.from_daf(Daf(cols=['id', 'qty'], lol=[['a1', 3], ['b2', 4]]), name='fromdb_parts')
            >>> parts_sdaf.copy().save('parts', db_path=source_path).name
            'parts'
            >>> brought_sdaf = SqlDaf.from_db(source_path, 'parts', new_name='fromdb_brought')
            >>> brought_sdaf.name, len(brought_sdaf)
            ('fromdb_brought', 2)
        """
        working: str = db_path or cls.db_path
        if not working:
            raise ValueError("SqlDaf: no database. Set SqlDaf.db_path, or give db_path.")
        source: str = str(path)
        if not os.path.exists(source):
            raise FileNotFoundError(source)
        target: str = new_name or name

        conn, _conn_da = cls._conn_for(working)
        with _file_schema(conn, working, source, read_only=True) as schema:
            if not _table_exists_on(conn, name, schema):
                raise KeyError(name)
            if _table_exists_on(conn, target, 'main') or _table_exists_on(conn, target, 'temp'):
                raise ValueError(f"SqlDaf.from_db: a table named '{target}' already exists in {working}.")
            all_cols_ls: T_ls = [info_t[1] for info_t in conn.execute(f'PRAGMA {schema}.table_info({quote_name(name)})')]
            cols_ls: T_ls = [col for col in all_cols_ls if col != RID_COL]
            ord_sql: str = quote_name(RID_COL) if RID_COL in all_cols_ls else 'rowid'
            meta_t: Tuple[str, T_dtype_dict] | None = _meta_read(conn, name, schema)
            keyfield, dtypes = meta_t if meta_t is not None else ('', {})
            cols_sql: str = ', '.join(quote_name(col) for col in cols_ls)

            # One statement copies every row, in order. A new daf_rid numbers them from 1.
            with _transaction_on(conn):
                conn.execute(_create_table_sql(conn, target, cols_ls, 'main', check_exists=False))
                if cols_ls:
                    conn.execute(f'INSERT INTO main.{quote_name(target)} ({cols_sql}) '
                                 f'SELECT {cols_sql} FROM {schema}.{quote_name(name)} ORDER BY {ord_sql}')
                _meta_write(conn, target, keyfield, dtypes)
        return cls(target, db_path=working)


    @classmethod
    def table_exists(cls, name: str, db_path: str = '') -> bool:
        """
        Say whether the database file has a table of this name.

        Use it to decide whether to open a table or to build it, as from its CSV files.
        It returns False if the database file does not exist, and does not create the file.

        Args:
            name: The name of the table.
            db_path: The database file. The class setting `SqlDaf.db_path` if empty.

        Returns:
            True if the table is there.

        Raises:
            ValueError: There is no database path.
        """
        path: str = db_path or cls.db_path
        if not path:
            raise ValueError("SqlDaf: no database. Set SqlDaf.db_path, or give db_path.")
        if not os.path.exists(path):
            return False
        conn: sqlite3.Connection = _connect(path)
        try:
            return _table_exists_on(conn, name, 'main')
        finally:
            conn.close()


    @property
    def name(self) -> str:              # type: ignore[override]
        """ The name of the table in the database. `copy(name=)` and `save(name)` give a new one. """
        return self._name


    read_only: bool = False

    # The TEMP copies a join not read yet depends on, brought in from other files. See _readable().
    _brought_in_ls: Tuple[str, ...] = ()
    _release_brought_in: Any = None
    """ True if this SqlDaf was opened with `read_only=True`, or is a selection of one. """

    @property
    def conn(self) -> sqlite3.Connection:
        """
        The connection to the database file, which every SqlDaf on the file shares.

        Every statement a SqlDaf runs reads this first. So while a read-only SqlDaf is open on the
        file, it sets the authorizer for the SqlDaf that runs the statement. A read-only one gets
        one that refuses a change to its own table. Any other gets one that allows every statement.

        The authorizer is set again only when that changes, as between a reader and a writer.
        Setting it makes SQLite prepare its statements again under it. That matters, because SQLite
        checks a statement only when it prepares it, and Python's sqlite3 reuses prepared statements.
        A switch costs about 5 to 15 us.
        """
        conn_da: T_da = self._conn_da
        if conn_da['has_read_only']:
            guard: str | None = self._root()._name if self.read_only else None
            if conn_da['guard'] != guard:
                self._conn.set_authorizer(_read_only_authorizer({guard}) if guard else _allow_all)
                conn_da['guard'] = guard
        return self._conn

    @conn.setter
    def conn(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def _root(self) -> 'SqlDaf':
        """
        The table this SqlDaf reads its rows from: its parent if it is a child, or itself. Internal use.

        It tests the parent against None. `parent or self` would test whether the parent has
        rows, so an empty parent table gave the child itself.
        """
        return self.parent_sdaf if self.parent_sdaf is not None else self


    @property
    def is_child(self) -> bool:
        """ True for a child, a selection that refers to rows of its parent. False for a parent, which holds its own rows. """
        return self.parent_sdaf is not None


    def __repr__(self) -> str:
        """
        Show the rows as a Markdown table, for the Python prompt, as `repr()` of a Daf shows them.

        This is the same text as `str()`, with a newline in front, so the table starts at the left
        margin when it is echoed. `is_child` and `is_temp` tell what kind of SqlDaf it is.

        Returns:
            The Markdown text.
        """
        return '\n' + self.md_daf_table_snippet()


    # ---- text, as a Daf shows it ----

    md_max_rows: int = 10
    """ The number of rows that `str()` shows, half from the start and half from the end. """

    md_max_cols: int = 10
    """ The number of columns that `str()` shows, half from the start and half from the end. """

    # The rows are in SQLite, not in memory. An add-on registered for Daf alone is refused here.
    _rows_in_memory: ClassVar[bool] = False

    #==== reading: functions in sqldaf_read.py

    md_daf_table_snippet = sqldaf_read._md_daf_table_snippet
    _select_rows         = sqldaf_read._select_rows
    to_daf               = sqldaf_read._to_daf
    _write_csv           = sqldaf_read._write_csv
    to_csv_file          = sqldaf_read._to_csv_file
    to_csv_buff          = sqldaf_read._to_csv_buff  # type: ignore[assignment]
    to_dict              = sqldaf_read._to_dict
    to_list              = sqldaf_read._to_list  # type: ignore[assignment]
    to_value             = sqldaf_read._to_value  # type: ignore[assignment]
    iloc                 = sqldaf_read._iloc  # type: ignore[assignment]
    select_record        = sqldaf_read._select_record
    _cols_daf            = sqldaf_read._cols_daf
    col                  = sqldaf_read._col
    col_to_la            = sqldaf_read._col_to_la
    icol                 = sqldaf_read._icol
    icol_to_la           = sqldaf_read._icol_to_la
    cols_to_dol          = sqldaf_read._cols_to_dol
    to_lod               = sqldaf_read._to_lod
    to_dod               = sqldaf_read._to_dod
    to_json              = sqldaf_read._to_json
    daf_to_lol_summary   = sqldaf_read._daf_to_lol_summary
    to_md                = sqldaf_read._to_md
    to_md_cols           = sqldaf_read._to_md_cols
    iloc_exists          = sqldaf_read._iloc_exists

    def __str__(self) -> str:
        """
        Show the rows as a Markdown table, as `str()` of a Daf shows them.

        At most `md_max_rows` rows and `md_max_cols` columns, 10 by default, then the summary line.
        `print()`, an f-string and `repr()` give the same, as for a Daf.

        Returns:
            The Markdown text.
        """
        return self.md_daf_table_snippet()


    def __format__(self, format_spec: str) -> str:
        """
        Format the rows for an f-string, as a Daf formats.

        With no format spec this is `str()`. With one, such as `.2f`, there must be exactly one cell.

        Args:
            format_spec: A format spec such as `.2f`.

        Returns:
            The formatted text.

        Raises:
            ValueError: A spec is given and there is not exactly one cell.
        """
        return Daf.__format__(self, format_spec)


    # ---- the connection ----

    @classmethod
    def close(cls, db_path: str = '') -> None:
        """
        Close a database file: the connection that every SqlDaf on it shares.

        It closes a file, not one table, so call it on the class, as `SqlDaf.close()`. Called on
        an instance, it still closes the file of `db_path`, or of the class setting, not the
        file that instance uses.

        The permanent tables in the file stay, with every committed change.
        `SqlDaf(name)` opens them again, on a new connection.
        TEMP tables, such as those made by `copy()` or a sort, are removed.
        Every SqlDaf that was on the file can no longer be used. sqlite3 raises ProgrammingError.
        A file that is not open is left as it is.

        Without `close()`, the file stays open until the program ends.
        Each statement is committed when it finishes, or at the end of a `transaction()` block.
        So a committed change is kept either way.

        Args:
            db_path: The database file. The class setting `SqlDaf.db_path` if empty.

        Raises:
            ValueError: There is no database path.
        """
        path: str = db_path or cls.db_path
        if not path:
            raise ValueError("SqlDaf.close: no database. Set SqlDaf.db_path, or give db_path.")
        open_t: Tuple[sqlite3.Connection, T_da] | None = SqlDaf._open_conns.pop(cls._conn_key(path), None)
        if open_t is not None:
            open_t[0].close()


    def delete(self) -> None:
        """
        Delete this table from the database file, with its metadata.

        The table is dropped. Inside a `transaction()` block that then raises, the drop is
        undone with the rest of the block. Otherwise it is final.
        A child has no table of its own to delete. Delete its parent, or let the child go.
        After this, the SqlDaf, and any child of it, can no longer be read.

        Raises:
            ValueError: This is a child.
        """
        if self.parent_sdaf is not None:
            raise ValueError(f"SqlDaf.delete: '{self.name}' is a child, with no table of its own. Delete its parent instead.")
        self._resolve()
        schema: str = 'temp' if self.is_temp else 'main'
        with self.transaction():
            self.conn.execute(f'DROP TABLE {schema}.{quote_name(self.name)}')
            if not self.is_temp:
                _meta_delete(self.conn, self.name)


    @contextlib.contextmanager
    def transaction(self) -> Iterator[None]:
        """
        Run the statements inside the `with` block as one transaction, on this database file.

        A transaction belongs to the file, not to this table. Every SqlDaf on the file shares one
        connection, so the work of any of them inside the block is committed or undone with the
        rest. Wrap the writes of a stage that must succeed or fail together in one block.
        Methods that change several things run their own `transaction()`. Inside a block, it
        becomes part of the block. Do not run BEGIN, COMMIT or ROLLBACK through `sql()`: they
        would end a block that another SqlDaf opened.

        Everything in the block is committed together when it ends, or undone together if it
        raises. It also makes a run of small statements faster, because the file is locked once
        for the block instead of once per statement. On 1,000,000 rows, a lookup by key took
        12.4 us inside a transaction, against 24.1 us outside.

        While the block is open, no other connection can write to the database file. So keep it
        to the work that needs it.

        A `transaction()` inside another one becomes a savepoint: if the inner block raises, only
        its own work is undone.

        Examples:
            >>> import tempfile
            >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
            >>> marks_sdaf = SqlDaf('marks', cols=['ballot_id', 'v'], lol=[['b1', 1]])
            >>> marks_sdaf.to_daf()
            <BLANKLINE>
            | ballot_id | v  |
            | --------: | -: |
            |        b1 |  1 |
            <BLANKLINE>
            %% daf rows=1; cols=2; keyfield=''; name='marks'
            <BLANKLINE>

            An UPDATE returns no rows, so `sql()` gives an empty Daf, shown by its summary line:

            >>> with marks_sdaf.transaction():
            ...     marks_sdaf.sql("UPDATE marks SET v = 2")
            <BLANKLINE>
            <BLANKLINE>
            %% daf rows=0; cols=0; keyfield=''; name=''
            <BLANKLINE>
            >>> marks_sdaf.to_daf().lol
            [['b1', 2]]
        """
        with _transaction_on(self.conn):
            yield


    def _next_temp_name(self, parent: str) -> str:
        """ A new name for a TEMP object made from `parent`, as in daf_tmp_marks_v3. Internal use. """
        while True:
            self._conn_da['temp_count'] += 1
            name: str = f'{TEMP_PREFIX}{parent}_v{self._conn_da["temp_count"]}'
            if not _table_exists_on(self.conn, name):
                return name


    def _register_function(self, func: Callable[..., Any], nargs: int) -> str:
        """
        Register a Python function on the connection, and return the name to call it by in SQL. Internal use.

        If the function raises, SQLite only reports that a function failed. So the error is kept,
        where every SqlDaf on the connection sees it, and `_execute()` raises it instead.
        """
        self._conn_da['func_count'] += 1
        func_name: str = f'daf_fn_{self._conn_da["func_count"]}'
        conn_da: T_da = self._conn_da

        def guarded(*args: Any) -> Any:
            try:
                return func(*args)
            except BaseException as exc_info:
                conn_da['func_error'] = exc_info
                raise

        self.conn.create_function(func_name, nargs, guarded)
        return func_name


    #==== apply, reduce, group and join: functions in sqldaf_apply.py

    _register_aggregate                            = sqldaf_apply._register_aggregate
    apply                                          = sqldaf_apply._apply
    reduce                                         = sqldaf_apply._reduce  # type: ignore[assignment]
    _value_counts_lt                               = sqldaf_apply._value_counts_lt
    valuecounts_for_colname                        = sqldaf_apply._valuecounts_for_colname
    valuecounts_for_colnames_ls                    = sqldaf_apply._valuecounts_for_colnames_ls  # type: ignore[assignment]
    valuecounts_for_colname_selectedby_colname     = sqldaf_apply._valuecounts_for_colname_selectedby_colname
    valuecounts_for_colnames_ls_selectedby_colname = sqldaf_apply._valuecounts_for_colnames_ls_selectedby_colname  # type: ignore[assignment]
    valuecounts_for_colname1_groupedby_colname2    = sqldaf_apply._valuecounts_for_colname1_groupedby_colname2
    value_counts_daf                               = sqldaf_apply._value_counts_daf
    daf_valuecount                                 = sqldaf_apply._daf_valuecount  # type: ignore[assignment]
    sum                                            = sqldaf_apply._sum  # type: ignore[assignment]
    groupsum_daf                                   = sqldaf_apply._groupsum_daf  # type: ignore[assignment]
    groupby                                        = sqldaf_apply._groupby  # type: ignore[assignment]
    groupby_reduce                                 = sqldaf_apply._groupby_reduce  # type: ignore[assignment]
    groupby_aggregate                              = sqldaf_apply._groupby_aggregate
    _aggregate_groups                              = sqldaf_apply._aggregate_groups
    groupby_cols                                   = sqldaf_apply._groupby_cols  # type: ignore[assignment]
    groupby_repeated_cols                          = sqldaf_apply._groupby_repeated_cols  # type: ignore[assignment]
    _ensure_group_index                            = sqldaf_apply._ensure_group_index
    group_where                                    = sqldaf_apply._group_where
    join                                           = sqldaf_apply._join  # type: ignore[assignment]

    def _execute(self, stmt: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        """ Run one statement, raising an error from a registered Python function as itself. Internal use. """
        self._conn_da['func_error'] = None
        try:
            return self.conn.execute(stmt, tuple(params))
        except sqlite3.OperationalError as exc_info:
            if self._conn_da['func_error'] is not None:
                raise self._conn_da['func_error'] from exc_info
            raise
        except sqlite3.DatabaseError as exc_info:
            if _is_refused(exc_info):
                raise ReadOnlyTableError(_READ_ONLY_MESSAGE) from exc_info
            raise


    def _fetch(self, cursor: sqlite3.Cursor) -> List[Tuple[Any, ...]]:
        """ Fetch every row of a cursor from `_execute()`, raising a registered function's error as itself. Internal use. """
        try:
            return cursor.fetchall()
        except sqlite3.OperationalError as exc_info:
            if self._conn_da['func_error'] is not None:
                raise self._conn_da['func_error'] from exc_info
            raise


    def _insert_rows(self, rows: Iterable[Iterable[Any]]) -> None:
        """ Add rows to this parent's table, in one transaction. NULL is stored as SQL NULL. Internal use. """
        stmt: str = (f'INSERT INTO {quote_name(self.name)} ({self._cols_sql()}) VALUES ({placeholders(len(self.hd))})')
        row_la: Iterable[Any]
        with self.transaction():
            self.conn.executemany(stmt, (null_to_none(row_la) for row_la in rows))


    def _can_read(self, other_sdaf: 'SqlDaf') -> bool:
        """
        Say whether this connection can read another SqlDaf in SQL, for a join or the like. Internal use.

        On the same connection, it always can. On another connection to the same file, it can if
        the other is a parent on a permanent table, since every connection on a file sees its tables.
        A child or a TEMP table depends on objects that only its own connection sees.
        """
        if other_sdaf.conn is self.conn:
            return True
        return (os.path.realpath(other_sdaf.db_path) == os.path.realpath(self.db_path)
                and other_sdaf.parent_sdaf is None and not other_sdaf.is_temp and other_sdaf.is_resolved)


    def _readable(self, other_sdaf: 'SqlDaf') -> 'SqlDaf':
        """
        The other SqlDaf, or a TEMP copy of it on this connection if this connection cannot read it. Internal use.

        A join, `concat()` and keys from a SqlDaf read the other table in SQL on this connection.
        A table in another file, or a selection or TEMP table of another connection, is brought in
        as a TEMP table here. A TEMP table is in no database file, so neither file changes, and it
        goes when the connection closes. The caller drops it sooner, when it is no longer needed.
        """
        if self._can_read(other_sdaf):
            return other_sdaf
        return self._bring_in(other_sdaf)


    def _bring_in(self, other_sdaf: 'SqlDaf') -> 'SqlDaf':
        """
        Copy another SqlDaf into a TEMP table on this connection, and give a SqlDaf on it. Internal use.

        A permanent table in another file is copied in SQL, with its file attached read-only for
        the copy, as `from_db()` does. That took 0.6 s for 1,000,000 rows of 5 columns. Anything
        else, or any copy inside a transaction, where SQLite cannot attach a file, is copied
        through Python, a batch of rows at a time, from the other's own connection.
        """
        name: str = self._next_temp_name(other_sdaf._root().name)
        cols_ls: T_ls = list(other_sdaf.hd)
        cols_sql: str = ', '.join(quote_name(col) for col in cols_ls)
        table_sql: str = _create_table_sql(self.conn, name, cols_ls, 'temp')
        is_table: bool = other_sdaf.parent_sdaf is None and not other_sdaf.is_temp and other_sdaf.is_resolved
        if is_table and not self.conn.in_transaction:
            order_col: str = RID_COL if other_sdaf.has_rid else 'rowid'
            with _file_schema(self.conn, self.db_path, other_sdaf.db_path, read_only=True) as schema:
                self._execute(table_sql)
                self._execute(f'INSERT INTO {quote_name(name)} ({cols_sql}) '
                              f'SELECT {cols_sql} FROM {schema}.{quote_name(other_sdaf.name)} ORDER BY {order_col}')
        else:
            insert_sql: str = f'INSERT INTO {quote_name(name)} ({cols_sql}) VALUES ({placeholders(len(cols_ls))})'
            try:
                with self.transaction():
                    self._execute(table_sql)
                    batch_lot: List[Tuple[Any, ...]]
                    for batch_lot in other_sdaf._iter_row_batches(cols=cols_ls):
                        self.conn.executemany(insert_sql, batch_lot)
            except sqlite3.ProgrammingError as exc_info:
                # Python's sqlite3 refuses a connection of another thread, or one that is closed.
                raise NotImplementedError(f"'{other_sdaf.name}' is a selection or TEMP table of another thread's "
                                          f"connection, or of a closed one, so it cannot be read here. "
                                          f"Save it in its own thread first.") from exc_info
        copy_sdaf: SqlDaf = self._new_on_conn(name, other_sdaf.keyfield, dict(other_sdaf.dtypes), is_temp=True)
        # A join looks up each key in the copy, so the copy gets its key index, as the table had.
        if copy_sdaf.keyfield:
            copy_sdaf.ensure_key_index()
        return copy_sdaf


    def _hold_brought_in(self, join_sdaf: 'SqlDaf', names: T_ls) -> None:
        """
        Count a join as a user of TEMP copies, and release them when it is read or goes away. Internal use.

        A chain of joins shares the copies of the unread joins in it, so each copy has a count of
        the joins that use it. The join gives them up when it is first read, or when it is gone
        without being read, as the inner join of a chain is. A copy is dropped at a count of zero.
        """
        counts_di: Dict[str, int] = self._conn_da['brought_in_di']
        name: str
        for name in names:
            counts_di[name] = counts_di.get(name, 0) + 1
        join_sdaf._brought_in_ls = tuple(names)
        join_sdaf._release_brought_in = weakref.finalize(join_sdaf, _release_brought_in, self.conn, counts_di, tuple(names))


    # ---- the SQL that gives the rows ----

    def _cols_sql(self, prefix: str = '', lead: str = '') -> str:
        """ The columns, quoted, in order, without daf_rid, after `lead` if there are any. Internal use. """
        if not self.hd:
            return ''
        return lead + ', '.join(prefix + quote_name(col) for col in self.hd)


    def _refs_table_sql(self, name: str) -> str:
        """
        The SQL that makes a child's TEMP table of references, ord and ref. Internal use.

        ref is declared with the type of the parent column it refers by, INTEGER for daf_rid.
        With no type, SQLite cannot use an index on ref to look up a parent row.
        """
        root_sdaf: SqlDaf = self._root()
        ref_col: str = root_sdaf._ref_col()
        ref_type: str = 'INTEGER' if ref_col == RID_COL else next(
                (info_t[2] for info_t in self.conn.execute(f'PRAGMA table_info({quote_name(root_sdaf.name)})') if info_t[1] == ref_col), '')
        return f'CREATE TEMP TABLE {quote_name(name)} (ord INTEGER PRIMARY KEY, ref {ref_type})'.replace(' )', ')')


    def _ensure_ref_index(self) -> None:
        """
        Make an index on ref of this child's table of references, if there is none. Internal use.

        It is made when the child is first used as keys, so a key is found in it by one lookup.
        A child used only to read its rows never pays for it.
        """
        self._execute(f'CREATE INDEX IF NOT EXISTS {quote_name(self.name + "_ref")} ON {quote_name(self.name)} (ref)')


    def _ref_col(self) -> str:
        """
        The column of this parent that a child refers to rows by. Internal use.

        daf_rid if the table has it. Otherwise the keyfield, with its unique index made first, so
        the references survive VACUUM. Otherwise a selection is not possible.
        """
        if self.has_rid:
            return RID_COL
        if self.keyfield:
            self.ensure_key_index()
            return self.keyfield
        raise NotImplementedError(
                f"A selection of '{self.name}' needs a daf_rid column or a unique keyfield. "
                f"Set a keyfield with unique values, or copy the table with copy().")


    def _resolve_parent(self) -> None:
        """ Resolve a join not read yet, this one or the parent of this child, before selecting from it. Internal use. """
        parent_sdaf: SqlDaf = self._root()
        parent_sdaf._resolve()


    def _rows_sql(self) -> Tuple[str, Tuple[Any, ...]]:
        """
        A query that gives every row, as daf_ref, daf_ord and then the columns, not sorted. Internal use.

        For a parent, it reads the table. Its daf_ref is daf_rid, or else the keyfield, or else
        the rowid. Only reading uses the rowid: `_new_selection()` checks for a reference that
        survives VACUUM first. For a child whose rows are found, it joins the reference table to
        the parent. For a child not read yet, it is the query that finds the rows.
        """
        if self.parent_sdaf is None and not self.is_resolved:
            return self.pending_sql, self.pending_params            # a join not resolved yet.
        if self.parent_sdaf is None:
            ord_sql: str = quote_name(RID_COL) if self.has_rid else 'rowid'
            ref_sql: str = ord_sql if self.has_rid or not self.keyfield else quote_name(self.keyfield)
            return (f'SELECT {ref_sql} AS {REF_COL}, {ord_sql} AS {ORD_COL}{self._cols_sql(lead=", ")} '
                    f'FROM {quote_name(self.name)}'), ()
        if not self.is_resolved:
            return self.pending_sql, self.pending_params
        parent_ref_col: str = self.parent_sdaf._ref_col()
        return (f'SELECT s.ref AS {REF_COL}, s.ord AS {ORD_COL}{self._cols_sql("b.", lead=", ")} '
                f'FROM {quote_name(self.name)} AS s JOIN {quote_name(self.parent_sdaf.name)} AS b '
                f'ON b.{quote_name(parent_ref_col)} = s.ref'), ()


    def _resolve(self) -> 'SqlDaf':
        """
        Find the rows of a child, and keep which rows they are. Internal use.

        Reading a child does this the first time. For a child, it writes a TEMP
        table of references, one daf_rid or key per row, in order. The rows stay in the parent.
        On 1,000,000 rows, a selection of 773,698 took 0.20 s. For a join not read yet, it writes
        a TEMP table of the joined rows. A parent, or a child already resolved, is not changed.
        """
        if self.is_resolved:
            return self
        if self.parent_sdaf is None:                                # a join: resolve it into a TEMP table of new rows.
            table_sql: str = _create_table_sql(self.conn, self.name, list(self.hd), 'temp')
            with self.transaction():
                self._execute(table_sql)
                self._execute(f'INSERT INTO {quote_name(self.name)} ({self._cols_sql()}) '
                              f'SELECT {self._cols_sql()} FROM ({self.pending_sql}) ORDER BY {ORD_COL}',
                              self.pending_params)
            self.has_rid = True
            self.is_temp = True
            self.is_resolved = True
            self.pending_sql = ''
            self.pending_params = ()
            if self._release_brought_in is not None:
                self._release_brought_in()             # the join is read, so it needs its copies no more.
            self._brought_in_ls = ()
            return self
        self._execute(self._refs_table_sql(self.name))
        self._execute(f'INSERT INTO {quote_name(self.name)} (ref) '
                      f'SELECT {REF_COL} FROM ({self.pending_sql}) ORDER BY {ORD_COL}',
                      self.pending_params)
        self.is_resolved = True
        self.pending_sql = ''
        self.pending_params = ()
        return self


    def _is_dense(self) -> bool:
        """
        Say whether daf_ord numbers the rows 1, 2, 3 and so on, so a position is a lookup. Internal use.

        It does in a resolved child, whose reference table numbers its rows as it is written. It
        does in a parent with daf_rid, as long as no row has been deleted from it.
        """
        return self.has_rid if self.parent_sdaf is None else self.is_resolved


    def _check_positions(self, method: str) -> None:
        """
        Raise ValueError if this is a parent whose daf_rid has gaps, from deleted rows. Internal use.

        Positions in a parent are looked up by daf_rid, so a gap would give the wrong row. A gap is
        found by comparing MAX(daf_rid) with COUNT(*). A table found without one is kept in a set
        on the connection, so it is checked once. `delete_records()`, and a statement through
        `sql()` that changes rows, take it out of the set, to be checked again.
        """
        if self.parent_sdaf is not None or not self.has_rid:
            return                                      # a child numbers its own rows, and other tables use OFFSET.
        no_gaps_ss: T_ss = self._conn_da['no_gaps_ss']
        if self.name in no_gaps_ss:
            return
        max_rid, count = self.conn.execute(
                f'SELECT COALESCE(MAX({quote_name(RID_COL)}), 0), COUNT(*) FROM {quote_name(self.name)}').fetchone()
        if max_rid == count:
            no_gaps_ss.add(self.name)
        else:
            raise ValueError(f"{method}: '{self.name}' has had rows deleted, so it cannot be read by position. "
                             "Read positions from copy(), which numbers the rows again, or select the rows you want.")


    def _count_for_position(self) -> int:
        """
        The number of rows, for a negative position. Internal use.

        Where positions are dense, this is the highest daf_ord, one lookup. Otherwise it is `len()`.
        """
        if self.parent_sdaf is None and self.has_rid:
            return int(self.conn.execute(f'SELECT COALESCE(MAX({quote_name(RID_COL)}), 0) FROM {quote_name(self.name)}').fetchone()[0])
        return len(self)


    # ---- reading ----

    def __len__(self) -> int:
        """
        Count the rows.

        For a parent, this runs a COUNT. For a child, its rows are found first, and then its
        references are counted, which is one lookup. A row deleted from the parent is still
        counted, so do not delete rows from a parent that has children.

        Returns:
            The number of rows.
        """
        self._resolve()
        if self.parent_sdaf is None:
            return int(self.conn.execute(f'SELECT COUNT(*) FROM {quote_name(self.name)}').fetchone()[0])
        return int(self.conn.execute(f'SELECT COALESCE(MAX(ord), 0) FROM {quote_name(self.name)}').fetchone()[0])


    def len(self) -> int:
        """ Count the rows, as `len(sdaf)` does. It is here so that code written for a Daf, which calls `daf.len()`, runs on a SqlDaf. """
        return len(self)


    def __bool__(self) -> bool:
        """
        Say whether there is at least one row.

        This stops at the first row, so it is fast on a large table. A child has its rows found
        first, and then its references are checked, as `len()` counts them.

        Returns:
            True if there is a row.
        """
        self._resolve()
        return self.conn.execute(f'SELECT 1 FROM {quote_name(self.name)} LIMIT 1').fetchone() is not None


    def num_cols(self) -> int:
        """
        Count the columns.

        The hidden daf_rid column is not counted.

        Returns:
            The number of columns.
        """
        return len(self.hd)


    def _iter_row_batches(self, batch_size: int = 10000, cols: T_ls | None = None) -> Generator[List[Tuple[Any, ...]], None, None]:
        """ Read every row in order, a batch at a time, as tuples with SQL NULL as None. Only `cols` if given. Internal use. """
        self._resolve()
        cols_ls: T_ls = list(self.hd) if cols is None else cols
        if not cols_ls:
            return                                      # no columns to read, so no rows.
        rows_sql, rows_params = self._rows_sql()
        cols_sql: str = ', '.join(quote_name(col) for col in cols_ls)
        cursor: sqlite3.Cursor = self._execute(f'SELECT {cols_sql} FROM ({rows_sql}) ORDER BY {ORD_COL}', rows_params)
        while True:
            try:
                batch_lot: List[Tuple[Any, ...]] = cursor.fetchmany(batch_size)
            except sqlite3.OperationalError as exc_info:
                if self._conn_da['func_error'] is not None:
                    raise self._conn_da['func_error'] from exc_info
                raise
            if not batch_lot:
                return
            yield batch_lot


    # ---- looping over rows ----

    def __iter__(self) -> Iterator[T_da]:                                   # type: ignore[override]
        """
        Loop over the rows as dicts, as in `for row in sdaf`.

        Rows are read in order, in batches, so a table of any size fits in memory. Each dict is a
        copy: changing it does not change the table. A Daf can give KeyedList rows instead, by its
        itermode. A SqlDaf always gives dicts. Use `iter_klist()` for KeyedList rows.

        Do not change the table inside the loop. Whether the loop sees the change is not defined.

        Returns:
            The rows, as dicts.
        """
        return self.iter_dict()


    def iter_dict(self) -> Iterator[T_da]:                                  # type: ignore[override]
        """
        Loop over the rows as dicts, as `Daf.iter_dict()` does.

        Rows are read in order, in batches. SQL NULL comes back as Python NULL. Each dict is a
        copy, so changing it does not change the table.

        Returns:
            Each row as a dict of column name to value.
        """
        cols_ls: T_ls = list(self.hd)
        batch_lot: List[Tuple[Any, ...]]
        for batch_lot in self._iter_row_batches():
            for row_t in batch_lot:
                yield dict(zip(cols_ls, none_to_null(row_t)))


    def iter_klist(self) -> Iterator[KeyedList]:                            # type: ignore[override]
        """
        Loop over the rows as [KeyedList][daffodil.keyedlist.KeyedList] objects.

        A row reads by column name, as in a Daf. But each row is a copy. In a Daf, assigning to a
        KeyedList row changes the Daf. Here it changes only the copy.

        Returns:
            Each row as a KeyedList.
        """
        batch_lot: List[Tuple[Any, ...]]
        for batch_lot in self._iter_row_batches():
            for row_t in batch_lot:
                yield KeyedList(self.hd, none_to_null(row_t))


    def iter_list(self) -> Iterator[T_la]:                                  # type: ignore[override]
        """
        Loop over the rows as lists, with the values in column order.

        Each list is a copy. In a Daf, these are the Daf's own row lists, and assigning to one
        changes the Daf. Here it changes only the copy.

        Returns:
            Each row as a list.
        """
        batch_lot: List[Tuple[Any, ...]]
        for batch_lot in self._iter_row_batches():
            for row_t in batch_lot:
                yield none_to_null(row_t)


    # ---- reading columns ----

    def num_rows(self) -> int:
        """ Count the rows, as `len()` does. """
        return len(self)


    def shape(self) -> Tuple[int, int]:
        """
        Give the number of rows and the number of columns, as `Daf.shape()` does.

        A table with no rows gives (0, 0), as in a Daf. The rows are counted with `len()`.

        Returns:
            A tuple of the number of rows and the number of columns.
        """
        num_rows: int = len(self)
        return (num_rows, len(self.hd)) if num_rows else (0, 0)


    #==== selecting: functions in sqldaf_select.py

    select_where_sql         = sqldaf_select._select_where_sql
    select_where             = sqldaf_select._select_where
    select_by_dict           = sqldaf_select._select_by_dict  # type: ignore[assignment]
    split_where              = sqldaf_select._split_where
    select_first_row_by_dict = sqldaf_select._select_first_row_by_dict
    select_krows             = sqldaf_select._select_krows
    select_records_daf       = sqldaf_select._select_records_daf
    calc_cols                = sqldaf_select._calc_cols  # type: ignore[assignment]
    select_cols              = sqldaf_select._select_cols

    #==== modifying: functions in sqldaf_modify.py

    _drop_group_indexes  = sqldaf_modify._drop_group_indexes
    delete_records       = sqldaf_modify._delete_records
    remove_dups          = sqldaf_modify._remove_dups  # type: ignore[assignment]
    concat               = sqldaf_modify._concat
    _write_rows          = sqldaf_modify._write_rows
    _excluded_sql        = sqldaf_modify._excluded_sql
    append               = sqldaf_modify._append
    record_append        = sqldaf_modify._record_append
    extend               = sqldaf_modify._extend
    assign_record        = sqldaf_modify._assign_record
    update_by_keylist    = sqldaf_modify._update_by_keylist
    update_record_krow   = sqldaf_modify._update_record_krow
    assign_col           = sqldaf_modify._assign_col
    apply_to_col         = sqldaf_modify._apply_to_col
    apply_in_place       = sqldaf_modify._apply_in_place
    _apply_one           = staticmethod(sqldaf_modify._apply_one)
    _check_layout_change = sqldaf_modify._check_layout_change
    insert_col           = sqldaf_modify._insert_col
    insert_idx_col       = sqldaf_modify._insert_idx_col
    drop_cols            = sqldaf_modify._drop_cols
    rename_cols          = sqldaf_modify._rename_cols
    _rename_by_map       = sqldaf_modify._rename_by_map
    set_cols             = sqldaf_modify._set_cols
    set_dtypes           = sqldaf_modify._set_dtypes
    sort_by_colname      = sqldaf_modify._sort_by_colname
    sort_by_colnames     = sqldaf_modify._sort_by_colnames
    attach_schema        = sqldaf_modify._attach_schema

    def __getitem__(self, sel: Any) -> 'SqlDaf':
        """
        Select rows by position or by key, as `daf[n]`, `daf[n:m]` and `daf[key]` do.

        Each gives a selection, not an ordinary Daf, so a large slice stays in the database.
        Read it with `to_daf()`, or `to_dict()` for one row. For one row as a dict, `iloc()` and
        `select_record()` are cheaper: they make no selection.

        - An int is always a position, from 0, even when the keyfield holds ints. Negative counts
          from the end. A position past the end raises IndexError, as in a Daf. For an int key,
          use `select_record()`, or `select_records_daf()` for several.
        - A slice of positions, as in `sdaf[10:20]`. A step is not supported. A slice past
          the end is empty, as in a Daf.
        - A str is a key. The keyfield's unique index is made first. A missing key raises
          KeyError, as in a Daf.
        - A list of keys gives their rows in the order of the list, as `select_records_daf()`
          does. A missing key raises KeyError, as in a Daf.

        Rows and columns together, as in `sdaf[key, 'col']` or `sdaf[:, ['a', 'b']]`, give new
        rows with only those columns, as `select_cols()` makes them. The rows part takes any form
        above, or `:` for all rows. The columns part is a name, a list of names, or `:` for all.
        Read one value with `to_value()`, as in a Daf.

        A value read this way makes a TEMP table, which costs about 1 ms. In a loop over many
        keys, `select_record()` is about 30 times faster.

        A negative position, or a slice with a negative end, costs a `len()` first. In a parent
        with daf_rid, or a child whose rows are found, positions are looked up directly. Otherwise
        SQLite steps over the rows before them.

        Args:
            sel: An int, a slice, a key or a list of keys. Or a pair of rows and columns.

        Returns:
            A selection. New rows of the chosen columns, for a pair.

        Raises:
            IndexError: An int position past the end.
            KeyError: A key that is missing, or a name that is not a column.
            NotImplementedError: Any other form, such as a list of positions, or a slice with a step.
            ValueError: An int or a slice, on a parent that has had rows deleted. See SqlDaf.

        Examples:
            >>> import tempfile
            >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
            >>> marks_sdaf = SqlDaf('marks', cols=['ballot_id', 'v'], lol=[['b1', 1], ['b2', 0], ['b3', 1]], keyfield='ballot_id')
            >>> marks_sdaf.to_daf()
            <BLANKLINE>
            | ballot_id | v  |
            | --------: | -: |
            |        b1 |  1 |
            |        b2 |  0 |
            |        b3 |  1 |
            <BLANKLINE>
            %% daf rows=3; cols=2; keyfield='ballot_id'; name='marks'
            <BLANKLINE>
            >>> marks_sdaf[['b3', 'b1']].to_daf().to_lod()
            [{'ballot_id': 'b3', 'v': 1}, {'ballot_id': 'b1', 'v': 1}]
            >>> marks_sdaf['b2', 'v'].to_value()
            0
        """
        if isinstance(sel, tuple):
            # Rows and columns: choose the rows, then the columns.
            if len(sel) != 2:
                raise NotImplementedError(f"sdaf[{sel!r}] is not supported on a SqlDaf. Give rows and columns.")
            rows_sel, cols_sel = sel
            rows_sdaf: SqlDaf = self if rows_sel == slice(None) else self[rows_sel]
            if cols_sel == slice(None):
                return rows_sdaf.select_cols()
            if not isinstance(cols_sel, (str, list)):
                raise NotImplementedError(f"sdaf[rows, {cols_sel!r}] is not supported on a SqlDaf. "
                                          f"Give a column name, a list of names or ':'.")
            return rows_sdaf.select_cols([cols_sel] if isinstance(cols_sel, str) else cols_sel)
        if isinstance(sel, list) and all(isinstance(key, str) for key in sel):
            return self.select_records_daf(sel)
        if isinstance(sel, bool) or not isinstance(sel, (int, slice, str)):
            raise NotImplementedError(f"sdaf[{sel!r}] is not supported on a SqlDaf yet. "
                                      f"Use an int, a slice, a key or a list of keys.")
        if not isinstance(sel, str):
            self._check_positions(f'sdaf[{sel!r}]')
        self._resolve_parent()
        parent_sql, parent_params = self._rows_sql()

        if isinstance(sel, str):
            if not self.keyfield:
                raise NotImplementedError(f"sdaf['{sel}']: '{self.name}' has no keyfield.")
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
                raise NotImplementedError("A slice with a step is not supported on a SqlDaf yet.")
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


    # ---- keys: selecting, deleting ----

    @overload
    def keys(self, *, silent_error: bool = True, astype: Literal['list'] = 'list') -> T_la: ...
    @overload
    def keys(self, *, silent_error: bool = True, astype: Literal['view']) -> T_kva: ...
    @overload
    def keys(self, *, silent_error: bool = True, astype: str = 'list') -> T_la | T_kva: ...

    def keys(                                                               # type: ignore[override]
            self,
            *,
            silent_error:   bool = True,                # if False, raise when there is no keyfield.
            astype:         str = 'list',               # 'list' for a list, 'view' for a KeysView.
            ) -> T_la | T_kva:
        """
        Get the keys, the keyfield's values in row order, as `Daf.keys()` does.

        Every key is read into memory. This takes the arguments of `Daf.keys()`, so code written
        for a Daf runs on a SqlDaf. With `astype='view'` it gives a KeysView of a dict made from
        the keys, in row order, with a repeated key once. A Daf gives a view of its key index,
        which costs nothing. Here the dict is built, so it holds every key.

        It exists for that compatibility. On a large table there are better ways, which read no
        key into Python and run the same on a Daf:

        - To compare two tables, select the rows: `a.select_records_daf(b, inverse=True, silent_error=True)`.
        - To pass the keys to another method, give the SqlDaf itself, as in `b.select_records_daf(a)`.
        - To test one key, use `key in table`, or `table.select_record(key)` for its row.
        - To sample, take the first rows: `table[0:3].keys()`.
        - To count, use `len()` of the selection.

        Args:
            silent_error: If False, raise an error when there is no keyfield.
            astype: `list` for a new list, or `view` for a KeysView.

        Returns:
            The keys. Without a keyfield, an empty list, as in a Daf.

        Raises:
            KeysDisabledError: There is no keyfield, and `silent_error` is False.
            ValueError: `astype` is neither `list` nor `view`.

        Examples:
            >>> import tempfile
            >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
            >>> marks_sdaf = SqlDaf('marks', cols=['ballot_id', 'v'], lol=[['b1', 1], ['b2', 0], ['b3', 1]], keyfield='ballot_id')
            >>> marks_sdaf.to_daf()
            <BLANKLINE>
            | ballot_id | v  |
            | --------: | -: |
            |        b1 |  1 |
            |        b2 |  0 |
            |        b3 |  1 |
            <BLANKLINE>
            %% daf rows=3; cols=2; keyfield='ballot_id'; name='marks'
            <BLANKLINE>
            >>> marks_sdaf.keys(astype='view')
            dict_keys(['b1', 'b2', 'b3'])
            >>> 'b2' in marks_sdaf, marks_sdaf[0:2].keys(), len(marks_sdaf.select_by_dict({'v': 1}))
            (True, ['b1', 'b2'], 2)

            With no keyfield, the answer is an empty list, or an error with `silent_error=False`:

            >>> plain_sdaf = SqlDaf.from_daf(Daf(cols=['id'], lol=[['a']]), name='keys_plain')
            >>> plain_sdaf.keys()
            []
            >>> plain_sdaf.keys(silent_error=False)
            Traceback (most recent call last):
            ...
            daffodil.daf.KeysDisabledError: keys: this SqlDaf has no keyfield.
        """
        if astype not in ('list', 'view'):
            raise ValueError(f"keys: astype must be 'list' or 'view', not {astype!r}.")
        if not self.keyfield:
            if not silent_error:
                raise KeysDisabledError("keys: this SqlDaf has no keyfield.")
            return []
        rows_sql, rows_params = self._rows_sql()
        cursor: sqlite3.Cursor = self._execute(
                f'SELECT {quote_name(self.keyfield)} FROM ({rows_sql}) ORDER BY {ORD_COL}', rows_params)
        keys_la: T_la = [NULL if row_t[0] is None else row_t[0] for row_t in self._fetch(cursor)]
        if astype == 'view':
            return dict.fromkeys(keys_la).keys()
        return keys_la


    def _key_source_sql(self, keys_ls: Any) -> Tuple[str, Tuple[Any, ...]]:
        """
        A query that gives the keys as daf_key and daf_ord, in order, from any form of keys. Internal use.

        A SqlDaf gives the keyfield of its rows. A table uses its key index. A selection is found
        in by its references, so they get an index the first time it is used as keys. Without it,
        each row scanned every reference: on 133,422 rows against 1,603,908 keys, past 115 s.
        With it, 0.9 s. An in-memory Daf gives its keys, and a list or other iterable its values:
        both are written to a TEMP table of keys first, with an index, so the keys are read once
        and joined in SQL.
        """
        if isinstance(keys_ls, SqlDaf) and not keys_ls.keyfield:
            raise ValueError(f"The keys are the SqlDaf '{keys_ls.name}', which has no keyfield.")
        if isinstance(keys_ls, SqlDaf) and not self._can_read(keys_ls):
            # A SqlDaf in another file gives only its keys, read in batches into a TEMP table of keys below.
            keys_ls = (row_t[0] for batch_lot in keys_ls._iter_row_batches(cols=[keys_ls.keyfield]) for row_t in batch_lot)
        if isinstance(keys_ls, SqlDaf):
            # A selection of keys is found in by its references, so they need an index.
            if keys_ls.parent_sdaf is not None:
                keys_ls._resolve()
                keys_ls._ensure_ref_index()
            # Its table finds each key by the key index, built lazily as for any lookup by key.
            keys_ls._need_keyfield('The keys')
            rows_sql, rows_params = keys_ls._rows_sql()
            return f'SELECT {quote_name(keys_ls.keyfield)} AS daf_key, {ORD_COL} FROM ({rows_sql})', rows_params

        if isinstance(keys_ls, Daf):
            keys_ls = keys_ls.keys(silent_error=False, astype='view')
        if isinstance(keys_ls, (str, bytes)) or not isinstance(keys_ls, Iterable):
            keys_ls = [keys_ls]

        name: str = self._next_temp_name('keys')
        self._execute(f'CREATE TEMP TABLE {quote_name(name)} (ord INTEGER PRIMARY KEY, daf_key)')
        key: Any
        with self.transaction():
            self.conn.executemany(f'INSERT INTO {quote_name(name)} (daf_key) VALUES (?)',
                                  ((None if key is NULL else key,) for key in keys_ls))
            self._execute(f'CREATE INDEX {quote_name(name + "_ix")} ON {quote_name(name)} (daf_key)')
        return f'SELECT daf_key, ord AS {ORD_COL} FROM {quote_name(name)}', ()


    def _need_keyfield(self, method: str) -> str:
        """ The keyfield, with the parent's index made, or ValueError if there is none. Internal use. """
        if not self.keyfield:
            raise ValueError(f"SqlDaf.{method}: '{self.name}' has no keyfield.")
        parent_sdaf: SqlDaf = self._root()
        if parent_sdaf.keyfield == self.keyfield:
            parent_sdaf.ensure_key_index()
        return self.keyfield


    def _first_missing_key(self, keys_sql: str, keys_params: Tuple[Any, ...]) -> Any:
        """ A key of the key source that no row of this SqlDaf has, or None if every one is there. Internal use. """
        rows_sql, rows_params = self._rows_sql()
        found_t: Tuple[Any, ...] | None = self._execute(
                f'SELECT k.daf_key FROM ({keys_sql}) AS k WHERE NOT EXISTS '
                f'(SELECT 1 FROM ({rows_sql}) AS r WHERE r.{quote_name(self.keyfield)} = k.daf_key) LIMIT 1',
                keys_params + rows_params).fetchone()
        return None if found_t is None else (NULL if found_t[0] is None else found_t[0])


    def __contains__(self, key: Any) -> bool:                               # type: ignore[override]
        """
        Say whether a row has this key, as in `key in sdaf`.

        It is one lookup on the keyfield, through its index. A table with no rows contains
        nothing, as in a Daf.

        Args:
            key: The key to look for.

        Returns:
            True if a row has this key.

        Raises:
            ValueError: There is no keyfield, and there are rows.
        """
        if not self.keyfield and not self:
            return False
        return bool(self.select_record(key))


    def get_existing_keys(self, keylist: T_la) -> T_la:                     # type: ignore[override]
        """
        Keep the keys of a list that have a row, in the order of the list, as `Daf.get_existing_keys()` does.

        The keys are looked up in SQL, all at once. Without a keyfield, the result is empty, as in a Daf.

        Args:
            keylist: The keys to check.

        Returns:
            The keys from the list that have a row.
        """
        if not self.keyfield:
            return []
        key_col: str = self._need_keyfield('get_existing_keys')
        keys_sql, keys_params = self._key_source_sql(keylist)
        rows_sql, rows_params = self._rows_sql()
        found_lot: List[Tuple[Any, ...]] = self._execute(
                f'SELECT DISTINCT k.daf_key FROM ({keys_sql}) AS k '
                f'WHERE EXISTS (SELECT 1 FROM ({rows_sql}) AS r WHERE r.{quote_name(key_col)} = k.daf_key)',
                keys_params + rows_params).fetchall()
        found_set: set = {found_t[0] for found_t in found_lot}
        return [key for key in keylist if key in found_set]


    # ---- adding rows one at a time, or a few ----

    def _row_from_item(self, item: Any, method: str) -> T_la:
        """
        One row in column order, placed as `Daf.append()` places it. Internal use.

        A dict or a KeyedList is placed by column name. A missing name gets Python NULL.
        A name that is not a column is dropped. A list is in column order, and a short one is padded.
        """
        if isinstance(item, (dict, KeyedList)):
            return [item[col] if col in item else NULL for col in self.hd]
        if isinstance(item, list):
            if len(item) > len(self.hd):
                raise ValueError(f"{method}: a row has {len(item)} values, but '{self.name}' has {len(self.hd)} columns.")
            return list(item) + [NULL] * (len(self.hd) - len(item))
        raise TypeError(f"{method}: a row must be a dict, a KeyedList or a list, not {type(item).__name__}.")


    # ---- changing values ----

    def _update_scope(self) -> Tuple[str, str, str, Tuple[Any, ...]]:
        """
        Where an UPDATE of this SqlDaf's rows goes. Internal use.

        Gives the table, its reference column, and a condition with its parameters. A parent
        updates its own table. A selection updates its parent's rows that it refers to, as a
        Daf selection shares its rows.
        """
        self._resolve()
        if self.parent_sdaf is None:
            ord_col: str = RID_COL if self.has_rid else 'rowid'
            return self.name, ord_col, '1', ()
        rows_sql, rows_params = self._rows_sql()
        ref_col: str = self.parent_sdaf._ref_col()
        return (self.parent_sdaf.name, ref_col,
                f'{quote_name(ref_col)} IN (SELECT {REF_COL} FROM ({rows_sql}))', rows_params)


    def _after_key_change(self, cols: Iterable[str], method: str) -> None:
        """ Check the keys of the table that was written, if a write reached the keyfield. Internal use. """
        root_sdaf: SqlDaf = self._root()
        if root_sdaf.keyfield and root_sdaf.keyfield in cols:
            root_sdaf._check_no_null_key(method)


    def __setitem__(self, slice_spec: Any, value: Any) -> None:            # type: ignore[override]
        """
        Assign by key and column name, as in `sdaf[key, 'v'] = 1`.

        These forms are supported:

        - `sdaf[key, col] = value` sets one cell. A list of one value gives that value, as in a Daf.
        - `sdaf[key] = row` sets the whole row with that key, from a dict or a list. Columns the
          row lacks become Python NULL, so a dict without the keyfield empties the key, which
          raises ValueError.
        - `sdaf[:, col] = values` sets a column. A list goes in row order. A short list leaves the
          rows past its end as they were, and a long one is cut. A single value sets every row.

        A row is found by its key, never by its position, so these work after a delete.
        The key must be a str. An int in brackets is a position in a Daf, so an int row raises
        NotImplementedError, even when the keyfield holds ints. To change a row with an int key,
        use `update_record_krow()`. On a selection, the parent's rows
        change, as a Daf selection shares its rows.

        Args:
            slice_spec: A key, or a tuple of a key or `:` and a column name.
            value: The value, row or list to assign.

        Raises:
            KeyError: The key or the column is not found.
            NotImplementedError: Any other form, such as an int row or several columns.
            ValueError: The change gives a Python NULL key. Nothing changes.
            sqlite3.IntegrityError: The change gives a repeated key. Nothing changes.
        """
        rows_sel: Any = slice_spec[0] if isinstance(slice_spec, tuple) and len(slice_spec) == 2 else slice_spec
        cols_sel: Any = slice_spec[1] if isinstance(slice_spec, tuple) and len(slice_spec) == 2 else slice(None)
        whole_rows: bool = rows_sel == slice(None)
        whole_cols: bool = cols_sel == slice(None)

        # A column, for every row.
        if whole_rows and isinstance(cols_sel, str):
            if cols_sel not in self.hd:
                raise KeyError(cols_sel)
            if isinstance(value, (list, tuple)):
                self._assign_col_values(cols_sel, list(value), keep_rest=True)
            else:
                self._assign_col_values(cols_sel, None, keep_rest=False, default=value)
            return

        if isinstance(rows_sel, int) and not isinstance(rows_sel, bool):
            raise NotImplementedError(f"sdaf[{slice_spec!r}] = ...: an int row is a position, as in a Daf, and a SqlDaf "
                                      "does not assign by position. For a row by key, of any type, use "
                                      "update_record_krow(key, record).")
        if isinstance(rows_sel, bool) or not isinstance(rows_sel, str) or not (whole_cols or isinstance(cols_sel, str)):
            raise NotImplementedError(f"sdaf[{slice_spec!r}] = ... is not supported on a SqlDaf. "
                                      "Use sdaf[key, col], sdaf[key] or sdaf[:, col].")

        # One cell, or one whole row, by key.
        key_col: str = self._need_keyfield('__setitem__')
        if whole_cols:
            row_la: T_la = self._row_from_item(dict(value) if isinstance(value, KeyedList) else value, '__setitem__')
            values_da: T_da = dict(zip(self.hd, row_la))
        else:
            if cols_sel not in self.hd:
                raise KeyError(cols_sel)
            if isinstance(value, (list, tuple)) and len(value) == 1:
                value = value[0]
            values_da = {cols_sel: value}
        table, _ref_col, scope_sql, scope_params = self._update_scope()
        set_sql: str = ', '.join(f'{quote_name(col)} = ?' for col in values_da)
        with self.transaction():
            cursor: sqlite3.Cursor = self._execute(
                    f'UPDATE {quote_name(table)} SET {set_sql} WHERE {scope_sql} AND {quote_name(key_col)} = ?',
                    tuple(null_to_none(list(values_da.values()))) + scope_params + (rows_sel,))
            if cursor.rowcount == 0:
                raise KeyError(rows_sel)
            self._after_key_change(values_da, '__setitem__')


    def _assign_col_values(self, colname: str, vals_la: T_la | None, keep_rest: bool, default: Any = NULL) -> None:
        """
        Write a column in row order, in one transaction. Internal use.

        Values go to the rows in order, matched by their place in that order, not by daf_rid.
        Rows past the end of the list keep their values if `keep_rest`, or else get `default`.
        With no list, every row gets `default`.
        """
        table, ref_col, scope_sql, scope_params = self._update_scope()
        col_sql: str = quote_name(colname)
        with self.transaction():
            if vals_la is None:
                self._execute(f'UPDATE {quote_name(table)} SET {col_sql} = ? WHERE {scope_sql}',
                              (None if default is NULL else default,) + scope_params)
            else:
                rows_sql, rows_params = self._rows_sql()
                vals_name: str = self._next_temp_name('vals')
                self._execute(f'CREATE TEMP TABLE {quote_name(vals_name)} (ord INTEGER PRIMARY KEY, val)')
                self.conn.executemany(f'INSERT INTO {quote_name(vals_name)} (val) VALUES (?)',
                                      ((None if val is NULL else val,) for val in vals_la))
                # Number the rows in order, then join each number to its value.
                order_sql: str = f'SELECT {REF_COL} AS ref, ROW_NUMBER() OVER (ORDER BY {ORD_COL}) AS rn FROM ({rows_sql})'
                self._execute(f'UPDATE {quote_name(table)} SET {col_sql} = v.val '
                              f'FROM ({order_sql}) AS m JOIN {quote_name(vals_name)} AS v ON v.ord = m.rn '
                              f'WHERE {quote_name(table)}.{quote_name(ref_col)} = m.ref', rows_params)
                if not keep_rest and len(vals_la) < len(self):
                    self._execute(f'UPDATE {quote_name(table)} SET {col_sql} = ? '
                                  f'FROM ({order_sql}) AS m WHERE m.rn > ? '
                                  f'AND {quote_name(table)}.{quote_name(ref_col)} = m.ref',
                                  rows_params + (None if default is NULL else default, len(vals_la)))
                self._execute(f'DROP TABLE {quote_name(vals_name)}')
            self._after_key_change([colname], 'assign_col')


    # ---- column names and layout ----

    def clone_empty(                                                        # type: ignore[override]
            self,
            lol:    T_lola | None = None,           # rows for the new table. None for no rows.
            cols:   T_ls | None = None,             # column names to use instead of these.
            name:   str = '',                       # the name of the new table. Generated if empty.
            ) -> 'SqlDaf':
        """
        Make a new table with the same layout, as `Daf.clone_empty()` does.

        It has the same columns, keyfield and dtypes, and no rows unless `lol` is given.
        It is a TEMP table, as `copy()` makes, so `save()` keeps it.
        With `cols` that differ, it has those columns.
        The dtypes then keep the entries for the columns still there.
        The keyfield is cleared if its column is gone.

        Args:
            lol: Rows for the new table, written as the constructor writes them. None for no rows.
            cols: Column names to use instead of these.
            name: The name of the new table. Generated if empty, as in daf_tmp_marks_v4.

        Returns:
            The new SqlDaf.

        Examples:
            >>> import tempfile
            >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
            >>> marks_sdaf = SqlDaf('marks', cols=['ballot_id', 'v'], lol=[['b1', 1]], keyfield='ballot_id')
            >>> marks_sdaf.to_daf()
            <BLANKLINE>
            | ballot_id | v  |
            | --------: | -: |
            |        b1 |  1 |
            <BLANKLINE>
            %% daf rows=1; cols=2; keyfield='ballot_id'; name='marks'
            <BLANKLINE>
            >>> empty_sdaf = marks_sdaf.clone_empty()
            >>> empty_sdaf.columns(), empty_sdaf.keyfield, len(empty_sdaf)
            (['ballot_id', 'v'], 'ballot_id', 0)

            With rows of its own, other column names, or a name:

            >>> stock_sdaf = SqlDaf.from_daf(Daf(cols=['id', 'qty'], keyfield='id', lol=[['a1', 3]]), name='clone_stock')
            >>> stock_sdaf.clone_empty(lol=[['z9', 1]]).to_daf().lol
            [['z9', 1]]
            >>> stock_sdaf.clone_empty(cols=['k', 'v']).columns()
            ['k', 'v']
            >>> stock_sdaf.clone_empty(name='clone_blank').name
            'clone_blank'
        """
        cols_ls: T_ls = list(self.hd) if not cols else ([cols] if isinstance(cols, str) else list(cols))
        if cols_ls != list(self.hd) and (len(set(cols_ls)) != len(cols_ls) or NULL in cols_ls):
            cols_ls = daf_utils._sanitize_cols(cols=cols_ls)
        root_sdaf: SqlDaf = self._root()
        target: str = name or self._next_temp_name(root_sdaf.name)
        table_sql: str = _create_table_sql(self.conn, target, cols_ls, 'temp')
        with self.transaction():
            self._execute(table_sql)
        new_sdaf: SqlDaf = self._new_on_conn(
                target,
                self.keyfield if self.keyfield in cols_ls else '',
                {col: typ for col, typ in self.dtypes.items() if col in cols_ls},
                is_temp=True)
        if lol:
            new_sdaf._insert_rows(lol)
        return new_sdaf


    # ---- copying and saving ----

    def copy(                                                               # type: ignore[override]
            self,
            level:       str | T_copybits | None = None,  # as in Daf.copy(). With inplace, it must copy the rows.
            name:        str  = '',                         # the name of the copy, a TEMP table. Generated if empty.
            deep:        bool = False,                      # deprecated, as in Daf.copy().
            for_sorting: bool = False,                      # deprecated, as in Daf.copy().
            inplace:     bool = False,                      # if True, copy into this SqlDaf and return it.
            ) -> 'SqlDaf':
        """
        Copy the rows into a TEMP table of their own, as `Daf.copy()` copies a Daf.

        The copy is a parent, with daf_rid, the rows in order, and this SqlDaf's keyfield and
        dtypes. Nothing ties it to the rows it came from: a change to one does not show in the
        other. A SQL cell holds a plain value, so every level gives what `deep` gives in a Daf.
        The level is only checked, as `Daf.copy()` checks it.

        With `inplace=True`, this SqlDaf becomes the copy, and is returned. Use it on a child, to
        make it a table of its own, independent of its parent. As in a Daf, a level that would
        not copy the rows, such as `sortable`, raises ValueError then.

        The copy is a TEMP table, so it is removed when the connection closes. `save()` keeps it in the
        file.

        Args:
            level: As in `Daf.copy()`. With `inplace`, it must copy the rows: `editable` or `deep`.
            name: The name of the copy. Generated if empty, as in daf_tmp_marks_v4.
            deep: Deprecated, as in `Daf.copy()`.
            for_sorting: Deprecated, as in `Daf.copy()`.
            inplace: If True, this SqlDaf becomes the copy, and is returned.

        Returns:
            The copy, a new SqlDaf, or this one if `inplace`.

        Raises:
            ValueError: A table of that name exists, `level` is not a level, or `inplace` with a level that does not copy the rows.

        Examples:
            >>> import tempfile
            >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
            >>> marks_sdaf = SqlDaf('marks', cols=['ballot_id', 'v'], lol=[['b1', 1], ['b2', 0]])
            >>> marks_sdaf.to_daf()
            <BLANKLINE>
            | ballot_id | v  |
            | --------: | -: |
            |        b1 |  1 |
            |        b2 |  0 |
            <BLANKLINE>
            %% daf rows=2; cols=2; keyfield=''; name='marks'
            <BLANKLINE>
            >>> sel_sdaf = marks_sdaf.select_where_sql('v = 1')
            >>> sel_sdaf.copy(inplace=True).is_child
            False
            >>> marks_sdaf.sql("UPDATE marks SET v = 9")
            <BLANKLINE>
            <BLANKLINE>
            %% daf rows=0; cols=0; keyfield=''; name=''
            <BLANKLINE>
            >>> sel_sdaf.to_daf().lol
            [['b1', 1]]

            Give the copy a name, or a level. A level that copies the rows gives a TEMP table of its own:

            >>> sizes_sdaf = SqlDaf.from_daf(Daf(cols=['id', 'qty'], lol=[['a1', 3]]), name='copy_sizes')
            >>> sizes_sdaf.copy(name='copy_sizes_2').name
            'copy_sizes_2'
            >>> sizes_sdaf.copy(level='editable').to_daf().lol
            [['a1', 3]]
        """
        if isinstance(level, bool):                     # an old call, copy(True), meant deep.
            deep, level = deep or level, None
        if inplace and level is None and not deep and not for_sorting:
            level = 'deep'
        if not deep and level is not None and level != 'deep':
            if isinstance(level, str) and level not in Daf._COPY_PRESETS:
                raise ValueError(f"copy: level must be one of {[*Daf._COPY_PRESETS, 'deep']}, not {level!r}")
            bits: int = Daf._COPY_PRESETS[level] if isinstance(level, str) else int(level)
            if inplace and not bits & Daf.COPY_ROWS:
                raise ValueError(f"copy(inplace=True): level {level!r} does not copy the rows. Use 'editable' or 'deep'.")

        root_sdaf: SqlDaf = self._root()
        target: str = name or self._next_temp_name(root_sdaf.name)
        table_sql: str = _create_table_sql(self.conn, target, list(self.hd), 'temp')
        rows_sql, rows_params = self._rows_sql()
        with self.transaction():
            self._execute(table_sql)
            self._execute(f'INSERT INTO {quote_name(target)} ({self._cols_sql()}) '
                          f'SELECT {self._cols_sql()} FROM ({rows_sql}) ORDER BY {ORD_COL}', rows_params)
        if not inplace:
            return self._new_on_conn(target, self.keyfield, self.dtypes, is_temp=True)
        self._name          = target
        self.has_rid        = True
        self.is_temp        = True
        self.parent_sdaf    = None
        self.is_resolved    = True
        self.pending_sql    = ''
        self.pending_params = ()
        return self


    def save(self, name: str = '', db_path: str = '') -> 'SqlDaf':
        """
        Keep this table in the database file, as a permanent table, to be opened again later.

        The name is the one this SqlDaf has, unless one is given: the name given to the
        constructor, or to `copy(name=)`. A table with only a generated name, such as
        daf_tmp_marks_v4, needs one. The keyfield and dtypes are recorded with it, so
        `SqlDaf(name)` opens it as it was.

        - A TEMP table, made by `copy()`, a sort or a join, is moved into the file. The TEMP
          table is removed.
        - A permanent table saved under its own name is left as it is, and its keyfield and
          dtypes are recorded again.
        - A permanent table saved under a new name is copied to it, and this SqlDaf then refers
          to the copy. The first table stays.

        A child has no rows of its own. Make it a table first, with `copy(inplace=True)`.

        With `db_path`, the table is written into that file instead, which is made if it is not
        there. This keeps a result in a file of its own, as for a table copied in by `from_db()`.
        A TEMP table is moved there, and a permanent table is copied, as above. This SqlDaf then
        refers to the table in that file. The file is attached only during the copy, which is one
        SQL statement. On 1,000,000 rows of 8 columns it took 0.64 s.

        A name already in use raises ValueError. To replace a table with a sorted or changed copy,
        delete the old table first, as in `SqlDaf('items').delete()`, then save the copy under its name.

        Args:
            name: The name to keep it under. This SqlDaf's name if empty.
            db_path: The database file to keep it in. This SqlDaf's file if empty.

        Returns:
            This SqlDaf, as a permanent table.

        Raises:
            ValueError: This is a child, the name is a generated one, or a table of that name exists.
                With `db_path`, also when a transaction is open.

        Examples:
            >>> import tempfile
            >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
            >>> marks_sdaf = SqlDaf('marks', cols=['ballot_id', 'v'], lol=[['b1', 1], ['b2', 0]], keyfield='ballot_id')
            >>> marks_sdaf.to_daf()
            <BLANKLINE>
            | ballot_id | v  |
            | --------: | -: |
            |        b1 |  1 |
            |        b2 |  0 |
            <BLANKLINE>
            %% daf rows=2; cols=2; keyfield='ballot_id'; name='marks'
            <BLANKLINE>
            >>> agreed_sdaf = marks_sdaf.select_where_sql('v = 1').copy(name='agreed').save()
            >>> SqlDaf('agreed').keyfield, len(SqlDaf('agreed'))
            ('ballot_id', 1)

            Save under a name, or into another database file, with `db_path`:

            >>> import os, tempfile
            >>> report_sdaf = SqlDaf.from_daf(Daf(cols=['id', 'qty'], lol=[['a1', 3]]), name='save_report')
            >>> report_sdaf.copy().save('save_kept').name
            'save_kept'
            >>> other_path = os.path.join(tempfile.mkdtemp(), 'other.db')
            >>> report_sdaf.copy().save('save_elsewhere', db_path=other_path).name, os.path.exists(other_path)
            ('save_elsewhere', True)
        """
        if self.parent_sdaf is not None:
            raise ValueError(f"SqlDaf.save: '{self.name}' is a child, with no rows of its own. "
                             f"Use copy(inplace=True) first.")
        self._resolve()                                 # a join not read yet.
        target: str = name or self._name
        if not name and self._name.startswith(TEMP_PREFIX):
            raise ValueError(f"SqlDaf.save: '{self._name}' is a generated name. Give the name to save it under.")

        if db_path and os.path.realpath(db_path) != os.path.realpath(self.db_path):
            return self._save_to_file(target, db_path)
        if not self.is_temp and target == self._name:
            with self.transaction():
                _meta_write(self.conn, target, self.keyfield, self.dtypes)
            return self

        if _table_exists_on(self.conn, target, 'main') or (target != self._name and _table_exists_on(self.conn, target, 'temp')):
            raise ValueError(f"SqlDaf.save: a table named '{target}' already exists.")
        source_sql: str = f'{"temp" if self.is_temp else "main"}.{quote_name(self._name)}'
        ord_sql: str = quote_name(RID_COL) if self.has_rid else 'rowid'
        table_sql: str = _create_table_sql(self.conn, target, list(self.hd), 'main', check_exists=False)
        with self.transaction():
            self._execute(table_sql)
            self._execute(f'INSERT INTO main.{quote_name(target)} ({self._cols_sql()}) '
                          f'SELECT {self._cols_sql()} FROM {source_sql} ORDER BY {ord_sql}')
            if self.is_temp:
                self._execute(f'DROP TABLE {source_sql}')
            _meta_write(self.conn, target, self.keyfield, self.dtypes)
        self._name   = target
        self.has_rid = True
        self.is_temp = False
        return self


    def _save_to_file(self, target: str, db_path: str) -> 'SqlDaf':
        """ Write this table into another database file, then refer to it there, for `save(db_path=)`. Internal use. """
        source_sql: str = f'{"temp" if self.is_temp else "main"}.{quote_name(self._name)}'
        ord_sql: str = quote_name(RID_COL) if self.has_rid else 'rowid'
        with _file_schema(self.conn, self.db_path, db_path) as schema:
            if _table_exists_on(self.conn, target, schema):
                raise ValueError(f"SqlDaf.save: a table named '{target}' already exists in {db_path}.")
            with self.transaction():
                self._execute(_create_table_sql(self.conn, target, list(self.hd), schema, check_exists=False))
                if self.hd:
                    self._execute(f'INSERT INTO {schema}.{quote_name(target)} ({self._cols_sql()}) '
                                  f'SELECT {self._cols_sql()} FROM {source_sql} ORDER BY {ord_sql}')
                if self.is_temp:
                    self._execute(f'DROP TABLE {source_sql}')
                _meta_write(self.conn, target, self.keyfield, self.dtypes, schema)

        # From here on, this SqlDaf is the table in that file, on that file's own connection.
        conn, conn_da = type(self)._conn_for(db_path)
        self._open_on(conn, conn_da, db_path, target, self.keyfield, self.dtypes, is_temp=False, use_meta=False)
        return self


    # ---- keys and types ----

    def set_keyfield(self, keyfield: str = '') -> 'SqlDaf':        # type: ignore[override]
        """
        Set the column that holds the key of each row.

        As in a Daf, nothing is built yet. The unique index on the column is made when a key is
        first used, by `ensure_key_index()`. An empty keyfield clears it. Avoid Python NULL in the
        column: a Python NULL key gives undefined results, as SqlDaf explains.

        On a permanent table with a record in the metadata, the record is changed too, so the
        table opens with this keyfield next time. A table made by other code has no record until
        `save()` makes one.

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
        with self.transaction():
            self._meta_update()
        return self


    def _meta_update(self) -> None:
        """ Change the metadata record of this table, if it is a permanent parent with one. Internal use. """
        if self.parent_sdaf is None and not self.is_temp and self.is_resolved and _meta_read(self.conn, self.name) is not None:
            _meta_write(self.conn, self.name, self.keyfield, self.dtypes)


    def key_index_name(self) -> str:
        """ The name of the unique index on the keyfield, as in daf_ix_marks_ballot_id. """
        return f'{INDEX_PREFIX}{self.name}_{self.keyfield}'


    def ensure_key_index(self) -> 'SqlDaf':
        """
        Make the unique index on the keyfield of a parent, if it is not there yet.

        A Daf builds its key index the first time a key is needed. A SqlDaf does the same: the
        methods that use keys call this first. On 1,000,000 rows it took 0.50 s, about what a
        Daf takes. After that, a lookup by key uses the index instead of reading every row.

        The index is unique. A Daf keeps the last row of a repeated key without a word. A SqlDaf
        raises instead, so a duplicate is never missed. Once the index is there, adding a row
        with a key that is already present raises too.

        A Python NULL key, SQL NULL or the text '', raises ValueError when the index is made.
        SqlDaf's results with one would be undefined, and the unique index does not stop several
        SQL NULL keys. The check uses the new index, so it costs about 0.3 ms. A Daf accepts a
        Python NULL key.

        Returns:
            This SqlDaf.

        Raises:
            ValueError: There is no keyfield, this is a selection, or a key is a Python NULL. Nothing is made.
            sqlite3.IntegrityError: A key is repeated. Nothing is made.
        """
        if self.parent_sdaf is not None:
            raise ValueError(f"SqlDaf.ensure_key_index: '{self.name}' is a child. Its parent has the index.")
        if not self.keyfield:
            raise ValueError(f"SqlDaf.ensure_key_index: '{self.name}' has no keyfield.")
        self._resolve()
        index_name: str = self.key_index_name()
        if _index_exists_on(self.conn, index_name):
            return self
        with self.transaction():                        # a Python NULL key leaves no index.
            self._execute(f'CREATE UNIQUE INDEX {quote_name(index_name)} ON {quote_name(self.name)} ({quote_name(self.keyfield)})')
            self._check_no_null_key('ensure_key_index')
        return self


    def _check_no_null_key(self, method: str) -> None:
        """ Raise ValueError if a key of this parent is a Python NULL, SQL NULL or ''. Uses the key index. Internal use. """
        key_sql: str = quote_name(self.keyfield)
        if self.conn.execute(f"SELECT EXISTS (SELECT 1 FROM {quote_name(self.name)} WHERE {key_sql} IS NULL OR {key_sql} = '')").fetchone()[0]:
            raise ValueError(f"SqlDaf.{method}: '{self.name}' has a Python NULL key in '{self.keyfield}'. "
                             "Give every row a real key: SqlDaf's results with a Python NULL key are undefined.")


    def apply_dtypes(                                               # type: ignore[override]
            self,
            *,
            dtypes:         T_dtype_dict | None = None,     # the types to convert to. This SqlDaf's dtypes if None.
            cols:           str | T_ls | None = None,       # convert only these columns.
            from_str:       bool = True,                    # if True, the cells are text, so str columns are not converted.
            silent_error:   bool = False,                   # if True, a column with no dtype stays as it is.
            ) -> 'SqlDaf':
        """
        Convert the columns to their dtypes, in place, in the database.

        New dtypes given here are kept in the metadata record of the table, if it has one.

        A table loaded from a CSV file holds text. Call this to turn the columns you need into
        numbers. Each value is converted by the same function `Daf.apply_dtypes()` uses, so the
        results are the same as in memory.

        - '12' becomes 12, and '2.5' with dtype int becomes 2.
        - A value that does not convert, such as 'abc', stays as it is.
        - Python NULL stays Python NULL, except in a bool column, where it becomes 0, as in a Daf.
        - A str column is not touched, unless `from_str` is False.

        By default the cells are taken to be text already, as from a CSV file, so a str column is
        skipped. Pass `from_str=False` when a str column may hold numbers, as from `from_daf()` of a
        typed Daf. Then each number becomes its text, as `Daf.apply_dtypes()` gives, so 3 becomes '3'
        and 1.5 becomes '1.5'. The table then holds what `from_csv()` of the same rows holds.

        Conversion does not check that every value got the requested type. A converted column
        can still hold some text. `silent_error` does not change that. It is about a column that
        has no dtype, not about a value that does not convert.

        All the columns are converted in one pass over the table, in one transaction.
        On 1,000,000 rows, two int columns took 1.8 s.

        int, float and bool are converted. str, dict and list are recorded, and nothing is converted: the
        stored text is the value. A dict or list cell is the text a Daf's `to_csv_file()` writes. On the
        SqlDaf it stays text. `reduce(by='sparse_row')` reads it as it goes, and after `to_daf()` a Daf's
        `apply_dtypes()` decodes it.

        On a selection, the rows are first copied into a TEMP table of its own, as
        `copy(inplace=True)` does, and the copy is converted. The parent is not changed. A Daf
        selection shares its rows, so there the parent's rows are converted too. Converting is
        costly, so convert a selection rather than a large parent that is not needed otherwise.

        Args:
            dtypes: The types to convert to, by column. It may name columns that are not there. This SqlDaf's dtypes if None.
            cols: Convert only these columns, or this one column. All the columns of the dtypes if None.
            from_str: If True, the cells are text, so str columns are not converted.
            silent_error: If False, a column to convert that has no dtype raises ValueError. If True, it is left as it is.

        Returns:
            This SqlDaf, whose table has been changed.

        Raises:
            KeyError: A name in `cols` is not a column.
            ValueError: A column to convert has no dtype, and `silent_error` is False.
            NotImplementedError: A dtype is not int, float, bool, str, dict or list.
        """
        # The dtypes are recorded with the conversion, in one transaction, so a refused or failed
        # conversion leaves the record and this SqlDaf as they were.
        merged_dtypes: T_dtype_dict = {**self.dtypes, **dtypes} if dtypes else self.dtypes
        source_dtypes: T_dtype_dict = dtypes or merged_dtypes
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

        # A dict or list cell is always stored as text. A str column may hold numbers unless from_str.
        convert_dtypes: Dict[str, Any] = {col: source_dtypes[col] for col in cols_ls
                                          if col in source_dtypes
                                          and (source_dtypes[col] not in TEXT_DTYPES or source_dtypes[col] is str and not from_str)}
        unstorable_ls: T_ls = [col for col, dtype in convert_dtypes.items() if dtype not in STORABLE_DTYPES]
        if unstorable_ls:
            raise NotImplementedError(f"apply_dtypes() cannot store {[convert_dtypes[col] for col in unstorable_ls]} "
                                       f"for the columns {unstorable_ls} yet. Only int, float, bool, str, dict and list.")
        if not convert_dtypes:
            if dtypes:
                self.dtypes = merged_dtypes
                with self.transaction():
                    self._meta_update()
            return self

        if self.parent_sdaf is not None:
            self.copy(inplace=True)                     # a selection becomes a table of its own first.
        set_ls: T_ls = []
        dtype: Any
        for col, dtype in convert_dtypes.items():
            func_name: str = f'daf_to_{dtype.__name__}'
            converter: Callable[[Any], Any] = daf_utils.get_converter(dtype)
            self.conn.create_function(func_name, 1, _none_safe(converter), deterministic=True)
            set_ls.append(f'{quote_name(col)} = {func_name}({quote_name(col)})')

        with self.transaction():
            self.conn.execute(f'UPDATE {quote_name(self.name)} SET {", ".join(set_ls)}')
            self.dtypes = merged_dtypes
            self._meta_update()
        return self


    def sql(self, stmt: str, params: Iterable[Any] = ()) -> Daf:
        """
        Run a SQL statement of your own, and return its result as a Daf.

        This is the way to do anything that SqlDaf has no method for. The statement runs on this
        SqlDaf's connection, so it can use every table in the file, and the TEMP tables of the
        connection. Refer to this table by `sdaf.name`.

        Pass values as `params`, with `?` in the statement, instead of putting them in the text.

        A SELECT gives a Daf of its rows, with its column names. SQL NULL becomes Python NULL. A
        statement that returns no rows, such as UPDATE or CREATE, gives an empty Daf. Outside
        `transaction()`, it is committed when it finishes.

        The whole result is pulled into memory, so select only what you need. Do not run BEGIN,
        COMMIT or ROLLBACK here. Use `transaction()`, which every SqlDaf on the file shares.

        A change to a table's columns made here, such as ALTER TABLE, is not seen by a SqlDaf
        already open on that table. Open it again by name to see the new columns. Nor does it
        update the keyfield and dtypes that SqlDaf records. For column changes, prefer
        `insert_col()`, `drop_cols()` and `rename_cols()`, which keep both up to date.

        Args:
            stmt: One SQL statement.
            params: Values for the `?` marks in the statement, in order.

        Returns:
            A Daf of the rows the statement returns.
        """
        changes_before: int = self.conn.total_changes
        try:
            cursor: sqlite3.Cursor = self.conn.execute(stmt, null_to_none(params))
        except sqlite3.DatabaseError as exc_info:
            if _is_refused(exc_info):
                raise ReadOnlyTableError(_READ_ONLY_MESSAGE) from exc_info
            raise
        result_daf: Daf = Daf()
        if cursor.description is not None:
            cols: T_ls = [desc_t[0] for desc_t in cursor.description]
            lol: T_lola = [none_to_null(row_t) for row_t in cursor]
            result_daf = Daf(lol=lol, cols=cols)
        if self.conn.total_changes != changes_before:
            self._conn_da['no_gaps_ss'].clear()         # rows may have been deleted, so check positions again.
        return result_daf


# Type aliases for SqlDaf, as daf_types.py has for other types. A SqlDaf variable ends in _sdaf,
# as in bif_sdaf. They are here, not in daf_types.py, because this module imports that one.
T_sdaf:      TypeAlias = SqlDaf
T_losdaf:    TypeAlias = List[SqlDaf]
T_dosdaf:    TypeAlias = Dict[str, SqlDaf]


def _csv_paths(path: str | Path | List[str | Path]) -> T_ls:
    """
    The files that `from_csv()` loads, in order. A glob pattern is expanded and sorted. Internal use.
    """
    if isinstance(path, list):
        return [str(one_path) for one_path in path]
    path_s: str = str(path)
    if not glob.has_magic(path_s):
        return [path_s]
    paths_ls: T_ls = sorted(glob.glob(path_s))
    if not paths_ls:
        raise FileNotFoundError(f"SqlDaf.from_csv: no file matches {path_s}")
    return paths_ls


def _none_safe(converter: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """
    Wrap a converter so that it sees Python NULL where SQL has SQL NULL, as it would in a Daf. Internal use.

    An empty result is stored as SQL NULL again. So NULL stays NULL for int and float, and
    becomes 0 for bool, as `Daf.apply_dtypes()` gives.
    """
    def convert(val: Any) -> Any:
        result: Any = converter(NULL if val is None else val)
        return None if result is NULL else result
    return convert


# The names every SqlDaf sets on itself. An add-on may not take one.
_SQLDAF_INSTANCE_ATTRS: Tuple[str, ...] = (
    '_conn', '_conn_da', 'db_path', '_name', 'keyfield', 'has_rid', 'hd', 'dtypes',
    'is_temp', 'parent_sdaf', 'pending_sql', 'pending_params', 'is_resolved',
)


def register_sqldaf_namespace(name: str) -> Callable[[type], type]:
    """
    Add a class of methods to every SqlDaf, under one name, as `register_daf_namespace()` does for a Daf.

    A SqlDaf holds its rows in SQLite, so an add-on written for a Daf, which reads rows in memory,
    does not work on it. An add-on registers a SqlDaf version here, under the same name as its Daf
    version or another. Then `sdaf.<name>` is an object of that class for that SqlDaf, a selection
    included. Without one, `sdaf.<name>` of a Daf add-on raises NotImplementedError.

    A name already in use raises ValueError: a method or attribute of SqlDaf, or another SqlDaf add-on.
    The name of a Daf add-on is free here, for its SqlDaf version.

    Args:
        name: The attribute name, such as `hist`. It must be a public identifier.

    Returns:
        A decorator that registers the class and returns it unchanged.

    Raises:
        ValueError: The name is not a public identifier, or it is already in use on SqlDaf.

    Examples:
        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> @register_sqldaf_namespace('totals')
        ... class SqlTotals:
        ...     def __init__(self, sdaf):
        ...         self._sdaf = sdaf
        ...     def of(self, col):
        ...         return self._sdaf.sum([col])[col]
        >>> sdaf = SqlDaf('items', cols=['id', 'qty'], lol=[['a1', 3], ['b2', 10]], keyfield='id',
        ...               dtypes={'id': str, 'qty': int})
        >>> sdaf
        <BLANKLINE>
        | id | qty |
        | -: | --: |
        | a1 |   3 |
        | b2 |  10 |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield='id'; name='items'
        <BLANKLINE>
        >>> sdaf.totals.of('qty'), sdaf[0:1].totals.of('qty')
        (13, 3)
        >>> del SqlDaf.totals
    """
    return daf_namespace.register_namespace(name, SqlDaf, _SQLDAF_INSTANCE_ATTRS)


# The group modules hold most of the SqlDaf methods, by topic.
# Some of their functions use names defined here, such as SqlDaf, when they run.
# Each module lists those names in _NAMES_FROM_SQLDAF, and they are set in it here, once they all exist.
# A module cannot import them at its top, since this module imports it first.
for _group_module in (sqldaf_read, sqldaf_select, sqldaf_modify, sqldaf_apply):
    for _name in getattr(_group_module, '_NAMES_FROM_SQLDAF', ()):
        setattr(_group_module, _name, globals()[_name])


# ---- block every Daf method that SqlDaf does not define ----

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
    """ A method that raises NotImplementedError, naming `name`. Internal use. """
    def blocker(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
                f"{name}() is not supported on a SqlDaf yet. Use sql(), or to_daf() on a few rows.")
    blocker.__name__ = name
    blocker.__qualname__ = f'SqlDaf.{name}'
    return blocker


def _make_property_blocker(name: str) -> property:
    """ A property that raises NotImplementedError, naming `name`. Internal use. """
    def blocked_get(self: Any) -> Any:
        raise NotImplementedError(f"{name} is not supported on a SqlDaf yet.")
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
