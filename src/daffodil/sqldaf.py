# sqldaf.py
#
# A Daf whose rows live in a SQLite table.
# The design and the decisions behind it are in notes/sql_backing/design_draft.md.

import io
import os
import csv
import json
import threading
import contextlib

from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, Iterator, List, Literal, Tuple, Type, TypeAlias, cast, overload      # noqa: F401
from collections.abc import Iterable

from daffodil.daf import Daf, NULL, ColumnNotFoundError, KeysDisabledError
from daffodil.keyedlist import KeyedList
from daffodil.lib.daf_types import T_ls, T_la, T_lola, T_di, T_da, T_ds, T_ss, T_ta, T_dtype_dict, T_copybits, T_buff
import daffodil.lib.daf_utils as daf_utils
import daffodil.lib.daf_sql as daf_sql
from daffodil.lib.daf_sql import sqlite3


RID_COL = 'daf_rid'                 # the INTEGER PRIMARY KEY of each table SqlDaf makes
REF_COL = 'daf_ref'                 # the reference to a parent row, in the queries of a selection
ORD_COL = 'daf_ord'                 # the order of the rows, in the queries of a selection
RESERVED_COLS = (RID_COL, REF_COL, ORD_COL)
TEMP_PREFIX = 'daf_tmp_'            # the prefix of every TEMP object SqlDaf makes
INDEX_PREFIX = 'daf_ix_'            # the prefix of every index SqlDaf makes
GROUP_INDEX_PREFIX = 'daf_gx_'      # the prefix of an index groupby() makes on the columns it groups by
META_TABLE = 'daf_tables'           # the table of metadata, one row per table SqlDaf made
SAVEPOINT = 'daf_sp'                # the name of a nested transaction. One name can be nested.

# dtypes that apply_dtypes() can store.
# A list or a dict would need JSON text, which is not built.
STORABLE_DTYPES = (int, float, bool, str)
MIN_SQLITE_VERSION: Tuple[int, int, int] = (3, 48, 0)     # ORDER BY in aggregates, JSON5, and 1000 function arguments.

# dtypes that the metadata can record, by name.
DTYPE_BY_NAME: Dict[str, Type] = {'int': int, 'float': float, 'bool': bool, 'str': str, 'list': list, 'dict': dict}


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


def null_to_none(row_la: Iterable[Any]) -> Tuple[Any, ...]:
    """ A row on its way into SQL: Python NULL becomes None, which SQLite stores as SQL NULL. """
    return tuple(None if val is NULL else val for val in row_la)


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
                           f"Install pysqlite3-binary, as in pip install 'daffodil[sqlite]', and SqlDaf uses its newer SQLite.")
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


@contextlib.contextmanager
def _transaction_on(conn: sqlite3.Connection) -> Iterator[None]:
    """ A transaction on the connection, or a savepoint inside one already open. See `SqlDaf.transaction()`. Internal use. """
    if not conn.in_transaction:
        conn.execute('BEGIN')
        try:
            yield
        except BaseException:
            conn.execute('ROLLBACK')
            raise
        conn.execute('COMMIT')
    else:
        conn.execute(f'SAVEPOINT {SAVEPOINT}')
        try:
            yield
        except BaseException:
            conn.execute(f'ROLLBACK TO {SAVEPOINT}')
            conn.execute(f'RELEASE {SAVEPOINT}')
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

def _meta_read(conn: sqlite3.Connection, name: str) -> Tuple[str, T_dtype_dict] | None:
    """ The keyfield and dtypes recorded for a table, or None if there is no record. Internal use. """
    if not _table_exists_on(conn, META_TABLE, 'main'):
        return None
    found_t: Tuple[Any, ...] | None = conn.execute(
            f'SELECT keyfield, dtypes FROM main.{quote_name(META_TABLE)} WHERE name = ?', (name,)).fetchone()
    if found_t is None:
        return None
    names_ds: Dict[str, str] = json.loads(found_t[1]) if found_t[1] else {}
    return found_t[0] or '', {col: DTYPE_BY_NAME[type_name] for col, type_name in names_ds.items()}


def _meta_write(conn: sqlite3.Connection, name: str, keyfield: str, dtypes: T_dtype_dict) -> None:
    """ Record the keyfield and dtypes of a table, replacing any record. Run it in the same transaction as the change. Internal use. """
    unknown_ls: T_ls = [col for col, dtype in dtypes.items() if getattr(dtype, '__name__', '') not in DTYPE_BY_NAME]
    if unknown_ls:
        raise NotImplementedError(f"SqlDaf: the dtypes of the columns {unknown_ls} cannot be recorded yet. "
                                  f"Only {list(DTYPE_BY_NAME)}.")
    conn.execute(f'CREATE TABLE IF NOT EXISTS main.{quote_name(META_TABLE)} '
                 f'(name TEXT PRIMARY KEY, keyfield TEXT, dtypes TEXT, made TEXT)')
    conn.execute(f"INSERT INTO main.{quote_name(META_TABLE)} (name, keyfield, dtypes, made) "
                 f"VALUES (?, ?, ?, datetime('now')) "
                 f"ON CONFLICT (name) DO UPDATE SET keyfield = excluded.keyfield, dtypes = excluded.dtypes",
                 (name, keyfield, json.dumps({col: dtype.__name__ for col, dtype in dtypes.items()})))


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
    - Tables: `from_csv()`, `from_daf()`, `copy()`, `save()`, `delete()` and `table_exists()`.
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

    engine:     str = 'sqlite'          # the database engine. Only SQLite so far.
    db_path:    str = ''                # the database file every SqlDaf opens, unless the constructor names another.

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
            ) -> None:
        """
        Make a new table, or open one that is in the database file.

        With `cols`, a new table is made in the file, with the rows of `lol`, read one at a
        time, so an iterator of any length works. The table, its rows and its metadata are made
        in one transaction: if any row fails, nothing is made. A Python NULL cell is stored
        as SQL NULL. The table gets a hidden first column, daf_rid, an INTEGER PRIMARY KEY, which
        numbers the rows in order. It is not one of the `columns()`. The columns are declared
        with no SQL type, so an int stays an int, and text stays text. A cell that holds a list
        or a dict cannot be stored yet.

        Without `cols`, the table of that name is opened. Its keyfield and dtypes are those
        recorded when it was made, unless they are given here. Nothing is read yet.

        The keyfield is only recorded. Its unique index is made when a key is first used, as a
        Daf builds its key index.

        Args:
            name: The name of the table.
            cols: The column names of a new table, in order. None to open a table that is there.
            lol: The rows of a new table, each the values in the order of `cols`.
            keyfield: The column that holds the key of each row. For a table that is opened, the recorded one if empty.
            dtypes: The types of the columns, kept for `apply_dtypes()`. Nothing is converted.
            db_path: The database file. The class setting `SqlDaf.db_path` if empty.

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


    @classmethod
    def _conn_key(cls, path: str) -> Tuple[int, str]:
        """ The key of a file's connection: the thread, since sqlite3 ties a connection to one, and the real path. Internal use. """
        return threading.get_ident(), os.path.realpath(path)


    @classmethod
    def _conn_for(cls, path: str) -> Tuple[sqlite3.Connection, T_da]:
        """ The connection open on this file, and its shared state, opened first if there is none. Internal use. """
        key_t: Tuple[int, str] = cls._conn_key(path)
        if key_t not in SqlDaf._open_conns:
            SqlDaf._open_conns[key_t] = (_connect(path, isolation_level=None),
                                         {'func_error': None, 'temp_count': 0, 'func_count': 0, 'no_gaps_ss': set()})
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

        self.conn:      sqlite3.Connection  = conn
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
        root_sdaf: SqlDaf = parent_sdaf.parent_sdaf or parent_sdaf
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
        return sel_sdaf


    @classmethod
    def from_csv(                                                           # type: ignore[override]
            cls,
            path:       str,                            # the CSV file.
            name:       str = '',                       # the table name. The file name without .csv if empty.
            keyfield:   str = '',                       # the column that holds the key of each row.
            delimiter:  str = ',',                      # the field separator.
            db_path:    str = '',                       # the database file. SqlDaf.db_path if empty.
            ) -> 'SqlDaf':
        """
        Load a CSV file into a new table, without building it in memory.

        This only reads, as `Daf.from_csv()` does without dtypes. Every value is stored as the
        text of the file. Call `apply_dtypes()` afterwards to convert the columns you need.

        The first line gives the column names. The rest is read one row at a time, so the file
        is never all in memory. An empty cell is stored as SQL NULL, and comes back as NULL.
        A row shorter than the header is padded with Python NULL, and a longer one raises ValueError.
        As with the constructor, if any row fails, no table is made.

        Args:
            path: The CSV file.
            name: The name of the new table. The file name without its extension if empty.
            keyfield: The column that holds the key of each row.
            delimiter: The field separator.
            db_path: The database file. The class setting `SqlDaf.db_path` if empty.

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

            def fitted(reader: Iterator[T_ls]) -> Iterator[T_la]:
                row_ls: T_ls
                for row_ls in reader:
                    if len(row_ls) > width:
                        raise ValueError(f"SqlDaf.from_csv: a row of {path} has {len(row_ls)} values, "
                                         f"more than the {width} columns.")
                    yield [NULL if val == '' else val for val in row_ls] + [NULL] * (width - len(row_ls))

            return cls(name, cols=cols, lol=fitted(reader), keyfield=keyfield, db_path=db_path)


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


    @property
    def is_child(self) -> bool:
        """ True for a child, a selection that refers to rows of its parent. False for a parent, which holds its own rows. """
        return self.parent_sdaf is not None


    def __repr__(self) -> str:
        kind: str = 'child' if self.parent_sdaf is not None else ('TEMP table' if self.is_temp else 'table')
        return f"<SqlDaf {kind} name='{self.name}' rows={len(self)} cols={len(self.hd)} keyfield='{self.keyfield}'>"


    __str__ = __repr__


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
            >>> with marks_sdaf.transaction():
            ...     _ = marks_sdaf.sql("UPDATE marks SET v = 2")
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


    def _register_aggregate(self, aggregate_cls: Type[Any], nargs: int) -> str:
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


    def _execute(self, stmt: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        """ Run one statement, raising an error from a registered Python function as itself. Internal use. """
        self._conn_da['func_error'] = None
        try:
            return self.conn.execute(stmt, tuple(params))
        except sqlite3.OperationalError as exc_info:
            if self._conn_da['func_error'] is not None:
                raise self._conn_da['func_error'] from exc_info
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


    def _check_same_db(self, other_sdaf: 'SqlDaf', method: str) -> None:
        """
        Check that another SqlDaf can be read from this connection, for a join or the like. Internal use.

        On the same connection, it always can. On another connection to the same file, it can if
        it is a parent on a permanent table, since every connection on a file sees its tables.
        A child or a TEMP table depends on objects that only its own connection sees.
        """
        if other_sdaf.conn is self.conn:
            return
        if os.path.realpath(other_sdaf.db_path) != os.path.realpath(self.db_path):
            raise NotImplementedError(f"{method}: the other SqlDaf is in another database file, "
                                      f"{other_sdaf.db_path}. Both must be in one file.")
        if other_sdaf.parent_sdaf is not None or other_sdaf.is_temp or not other_sdaf.is_resolved:
            raise NotImplementedError(f"{method}: '{other_sdaf.name}' is a child or a TEMP table of another connection. "
                                      f"Save it first, or make both from the same table.")


    # ---- the SQL that gives the rows ----

    def _cols_sql(self, prefix: str = '', lead: str = '') -> str:
        """ The columns, quoted, in order, without daf_rid, after `lead` if there are any. Internal use. """
        if not self.hd:
            return ''
        return lead + ', '.join(prefix + quote_name(col) for col in self.hd)


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
        parent_sdaf: SqlDaf = self.parent_sdaf or self
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
            return self
        self._execute(f'CREATE TEMP TABLE {quote_name(self.name)} (ord INTEGER PRIMARY KEY, ref)')
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


    def _select_rows(self, where_sql: str = '', params: Tuple[Any, ...] = (), limit: int = -1, offset: int = 0,
                     cols: T_ls | None = None) -> List[Tuple[Any, ...]]:
        """ Read rows in order, with an optional condition and limit, as tuples with SQL NULL as None. Internal use. """
        if not (cols if cols is not None else self.hd):
            return []                                   # a table with no columns has no rows.
        rows_sql, rows_params = self._rows_sql()
        cols_sql: str = ', '.join(quote_name(col) for col in (cols if cols is not None else self.hd))
        where_part: str = f' WHERE {where_sql}' if where_sql else ''
        cursor: sqlite3.Cursor = self._execute(
                f'SELECT {cols_sql} FROM ({rows_sql}){where_part} ORDER BY {ORD_COL} LIMIT ? OFFSET ?',
                rows_params + params + (limit, offset))
        return self._fetch(cursor)


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


    def to_daf(self) -> Daf:
        """
        Pull every row into an ordinary Daf in memory.

        Use it once the rows you need are few enough to hold. The rows come in order. A SQL NULL
        comes back as Python NULL. The new Daf has this SqlDaf's name, keyfield and dtypes. Nothing
        ties it to the table afterwards: a change to one does not show in the other. A child has
        its rows found first.

        Returns:
            A Daf of every row.
        """
        self._resolve()
        lol: T_lola = [none_to_null(row_t) for row_t in self._select_rows()]
        return Daf(lol=lol, cols=self.columns(), keyfield=self.keyfield, name=self.name,
                   dtypes=dict(self.dtypes) if self.dtypes else None)


    def _iter_row_batches(self, batch_size: int = 10000, cols: T_ls | None = None) -> Iterator[List[Tuple[Any, ...]]]:
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


    def _write_csv(self, text_f: Any, line_terminator: str | None, include_header: bool) -> None:
        """ Write the rows as CSV to an open text file, a batch at a time, as `Daf.to_csv_buff()` writes them. Internal use. """
        csv_writer = csv.writer(text_f, lineterminator='\r\n' if line_terminator is None else line_terminator)
        if include_header:
            csv_writer.writerow([daf_sql.sql_unesc_str(col) for col in self.columns()])     # a name stored with the __HH encoding is written decoded.
        batch_lot: List[Tuple[Any, ...]]
        for batch_lot in self._iter_row_batches():
            csv_writer.writerows(batch_lot)                     # SQL NULL is written as an empty cell, as Python NULL is.


    def to_csv_file(                                                        # type: ignore[override]
            self,
            file_path:          str | Path = '',
            line_terminator:    str | None = None,
            include_header:     bool = True,
            ) -> str:
        r"""
        Write the rows to a CSV file, as `Daf.to_csv_file()` does.

        A local file is written as the rows are read, 10,000 at a time, so a table larger than
        memory can be written. The text is the same as a Daf of the same rows would write: each
        cell with `str()`, Python NULL and SQL NULL as nothing, and `\r\n` line endings by default.
        A column name stored with the `__HH` encoding is written decoded. So a table that another
        program stored with a column `__67roup` is written with the header `group`.
        A path that starts with s3 is not streamed: the text is built in memory and uploaded, as a Daf does.

        Args:
            file_path: Where to write the file.
            line_terminator: The line ending. If None, `\r\n` is used.
            include_header: If True, write the column names first.

        Returns:
            The path that was written.

        Examples:
            >>> import os, tempfile
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
            >>> path = marks_sdaf.to_csv_file(os.path.dirname(SqlDaf.db_path) + '/marks.csv')
            >>> open(path, newline='').read()
            'ballot_id,v\r\nb1,1\r\nb2,\r\n'
        """
        file_path = str(file_path)
        if file_path.startswith('s3'):
            Daf.buff_to_file(self.to_csv_buff(line_terminator=line_terminator, include_header=include_header), file_path)
            return file_path
        with open(daf_utils.path_sep_per_os(file_path), 'w', newline='', encoding='utf-8') as csv_f:
            self._write_csv(csv_f, line_terminator, include_header)
        return file_path


    def to_csv_buff(                                                        # type: ignore[override]
            self,
            line_terminator:    str | None = None,
            include_header:     bool = True,
            ) -> T_buff:
        r"""
        Make CSV text from the rows, as `Daf.to_csv_buff()` does.

        The whole text is built in memory, so use it for a selection of modest size. To write a
        large table, use `to_csv_file()`, which writes as it reads.

        Args:
            line_terminator: The line ending. If None, `\r\n` is used.
            include_header: If True, write the column names first.

        Returns:
            The CSV text.
        """
        text_f = io.StringIO(newline='')
        self._write_csv(text_f, line_terminator, include_header)
        return text_f.getvalue()


    def apply(                                                              # type: ignore[override]
            self,
            func:       Callable[..., Any],         # called as func(row, **kwargs), or func(self, **kwargs) for by='table'.
            by:         str = 'row',                # 'row' for each row, 'table' for the whole SqlDaf.
            keylist:    T_la | None = None,         # not supported on a SqlDaf: select the rows first.
            **kwargs:   Any,                        # passed on to func.
            ) -> Any:
        """
        Apply a Python function to each row, and collect the results in a new table, as `Daf.apply()` does.

        The function gets each row as a dict, with Python NULL for a missing value, and returns what to
        add to the new table, by the rules of `Daf.apply()`:

        - A dict, or a KeyedList, adds one row.
        - A list of them adds a row for each, so one row can become several.
        - None, an empty dict or an empty list adds nothing, so the row is left out.

        The new table takes its columns from the first row returned. A later row is placed by
        column name. A column it lacks gets Python NULL, and a name the first row did not have is dropped.
        So have the function return every column in every row. The new table has no keyfield
        and no dtypes, as the new Daf of `Daf.apply()` has none.

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
            >>> import tempfile
            >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
            >>> votes_sdaf = SqlDaf.from_daf(Daf(lol=[['b1', 'c0,c1'], ['b2', '']], cols=['ballot_id', 'contests']), name='votes')
            >>> votes_sdaf.to_daf()
            <BLANKLINE>
            | ballot_id | contests |
            | --------: | -------: |
            |        b1 |    c0,c1 |
            |        b2 |          |
            <BLANKLINE>
            %% daf rows=2; cols=2; keyfield=''; name='votes'
            <BLANKLINE>
            >>> def split(row):
            ...     return [{'ballot_id': row['ballot_id'], 'contest': c} for c in row['contests'].split(',') if c]
            >>> votes_sdaf.apply(split).to_daf().lol
            [['b1', 'c0'], ['b1', 'c1']]
        """
        if by == 'table':
            return func(self, **kwargs)
        if by != 'row':
            raise NotImplementedError(f"apply(by={by!r}) is not supported on a SqlDaf.")
        if keylist is not None:
            raise NotImplementedError("apply(keylist=) is not supported on a SqlDaf. Select the rows first.")

        cols: T_ls = self.columns()
        target: str = self._next_temp_name((self.parent_sdaf or self).name)
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


    def reduce(                                                             # type: ignore[override]
            self,
            func:           Callable[..., Any],         # combines a row with the result so far.
            by:             str = 'row',                # 'row', 'sparse_row' or 'table'.
            cols:           T_ls | None = None,         # the columns included. All if None.
            initial_da:     T_da | None = None,         # the result to start from.
            indirect_col:   str = '',                   # the column that holds a dict, for 'sparse_row'.
            silent_error:   bool = False,               # if True, skip rows where func raises.
            **kwargs:       Any,                        # passed on to func.
            ) -> Any:
        """
        Combine all the rows into one result, using a function, as `Daf.reduce()` does.

        The function is called in Python for each row, with the same arguments and the same
        result as in `Daf.reduce()`. A sum or a count is a reduction, as with `Daf.sum_da`.
        The rows are read 10,000 at a time, in order, so the table is never all in memory.
        Memory use still depends on the result the function builds.

        - With `by='row'`, each row is a [KeyedList][daffodil.keyedlist.KeyedList] of its
          values. Unlike in a Daf, writing to it does not change the table.
        - With `by='sparse_row'`, the function gets the dict held in `indirect_col`, JSON or
          PYON text decoded as in a Daf, and the result starts as `initial_da` or an empty dict.
        - With `by='table'`, the function gets this SqlDaf and `cols`.

        For a plain total of columns, `sum()` is faster: SQLite adds them, and no row comes
        into Python. `by='col'` is not supported, since it needs each whole column in memory.

        Args:
            func: The reduction function, such as `Daf.sum_da`.
            by: `row`, `sparse_row` or `table`.
            cols: The columns included in the reduction. All columns if None.
            initial_da: The result to start from, instead of zeros.
            indirect_col: A column that holds a dict. Required for `sparse_row`.
            silent_error: If True, a row for which the function raises is skipped.
            **kwargs: Keyword arguments passed on to the function.

        Returns:
            A dict for `row` and `sparse_row`. The function's result for `table`.

        Raises:
            ValueError: `sparse_row` is used with no `indirect_col`.
            NotImplementedError: `by` is `col`, or not recognized.

        Examples:
            >>> import tempfile
            >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
            >>> votes_sdaf = SqlDaf('votes', cols=['ballot_id', 'v', 'concise_votes'],
            ...                     lol=[['b1', 1, "{'c0b': 1, 'c0v': 1}"], ['b2', 2, '{"c0b": 1, "c1b": 1}']])
            >>> votes_sdaf.to_daf()
            <BLANKLINE>
            | ballot_id | v  |    concise_votes     |
            | --------: | -: | -------------------: |
            |        b1 |  1 | {'c0b': 1, 'c0v': 1} |
            |        b2 |  2 | {"c0b": 1, "c1b": 1} |
            <BLANKLINE>
            %% daf rows=2; cols=3; keyfield=''; name='votes'
            <BLANKLINE>
            >>> votes_sdaf.reduce(Daf.sum_da, cols=['v'])
            {'ballot_id': '', 'v': 3, 'concise_votes': ''}
            >>> votes_sdaf.reduce(Daf.sum_da, by='sparse_row', indirect_col='concise_votes')
            {'c0b': 2, 'c0v': 1, 'c1b': 1}
        """
        if by == 'table':
            return func(self, cols, **kwargs)
        if by not in ('row', 'sparse_row'):
            raise NotImplementedError(f"reduce(by={by!r}) is not supported on a SqlDaf.")
        if by == 'sparse_row' and not indirect_col:
            raise ValueError("indirect_col must be specified for 'sparse_row' reduction.")
        if by == 'row' and not len(self):
            return {}

        hd: T_di = dict(self.hd)
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


    def to_dict(self) -> T_da:
        """
        Read a table of one row as a dict, as `Daf.to_dict()` does.

        A child has its rows found first. For a row by key or by position, `select_record()` and
        `iloc()` are cheaper: they make no selection.

        Returns:
            The row as a dict. An empty dict if there are no rows.

        Raises:
            ValueError: There is more than one row.
        """
        self._resolve()
        rows_lot: List[Tuple[Any, ...]] = self._select_rows(limit=2)
        if len(rows_lot) > 1:
            raise ValueError("Ambiguous 2dim array.")
        if not rows_lot:
            return {}
        return dict(zip(self.hd, none_to_null(rows_lot[0])))


    def to_list(self) -> T_la:                                              # type: ignore[override]
        """
        Read a table of one row, or of one column, as a list, as `Daf.to_list()` does.

        One column gives every value of the column, in order, so select first. A child has its
        rows found first.

        Returns:
            The values of the row, or of the column.

        Raises:
            ValueError: There is more than one row and more than one column.
        """
        self._resolve()
        if len(self.hd) == 1:
            return [NULL if row_t[0] is None else row_t[0] for row_t in self._select_rows()]
        rows_lot: List[Tuple[Any, ...]] = self._select_rows(limit=2)
        if len(rows_lot) > 1:
            raise ValueError("to_list() requires a 1D Daf (single row or single column)")
        return none_to_null(rows_lot[0]) if rows_lot else []


    def to_value(self) -> Any:                                              # type: ignore[override]
        """
        Read a table of one row and one column as its value, as `Daf.to_value()` does.

        A child has its rows found first.

        Returns:
            The value. Python NULL for SQL NULL.

        Raises:
            ValueError: The table is not one row and one column.
        """
        self._resolve()
        rows_lot: List[Tuple[Any, ...]] = self._select_rows(limit=2)
        if len(rows_lot) != 1 or len(self.hd) != 1:
            raise ValueError(f"to_value() requires 1×1, got shape ({len(rows_lot) if len(rows_lot) < 2 else '2+'}, {len(self.hd)})")
        return NULL if rows_lot[0][0] is None else rows_lot[0][0]


    @overload                                                               # type: ignore[override]
    def iloc(self, irow: int = 0, include_cols: T_ls | None = None, rtype: Literal['dict'] = 'dict') -> T_da: ...
    @overload
    def iloc(self, irow: int = 0, include_cols: T_ls | None = None, *, rtype: Literal['list']) -> T_la: ...
    @overload
    def iloc(self, irow: int = 0, include_cols: T_ls | None = None, rtype: str = 'dict') -> T_da | T_la: ...

    def iloc(
            self,
            irow:           int = 0,                    # the position of the row. Negative counts from the end.
            include_cols:   T_ls | None = None,         # only these columns.
            rtype:          str = 'dict',               # 'dict' or 'list'.
            ) -> T_da | T_la:
        """
        Read one row by position, as `Daf.iloc()` does, in one query.

        On a selection, the position is within the selection's own rows, in its order.
        The call makes no selection. On a selection whose rows are not yet found, it does not
        fix them: they are still found when the selection is first read.
        In a parent with daf_rid, or a child whose rows are found, the position is looked up directly. Otherwise SQLite
        steps over the rows before it, which takes longer the further in it is. A negative
        position counts from the end, which costs a `len()` first.

        Args:
            irow: The position of the row. Negative counts from the end.
            include_cols: Only these columns, in this order. All if None.
            rtype: 'dict' for a dict, or 'list' for a list of the values.

        Returns:
            The row. Python NULL for SQL NULL.

        Raises:
            IndexError: There is no row at that position.
            KeyError: A name in `include_cols` is not a column.
            NotImplementedError: `rtype` is not 'dict' or 'list'.
            ValueError: This is a parent that has had rows deleted. See SqlDaf.
        """
        self._check_positions('iloc')
        if rtype not in ('dict', 'list'):
            raise NotImplementedError(f"iloc(rtype='{rtype}') is not supported on a SqlDaf yet. Use 'dict' or 'list'.")
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
        read every row. It took about 18 us on 1,000,000 rows, or 9 us inside `transaction()`.

        On a selection, only the selection's rows are searched. A key that is in the parent but
        not in the selection is missing. The call makes no selection. On a selection whose rows
        are not yet found, it does not fix them: they are still found when it is first read.

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
        (self.parent_sdaf or self).ensure_key_index()
        rows_lot: List[Tuple[Any, ...]] = self._select_rows(f'{quote_name(self.keyfield)} IS ?', (key,), limit=1)
        if not rows_lot:
            if silent_error:
                return {}
            raise KeyError(key)
        return dict(zip(self.hd, none_to_null(rows_lot[0])))


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

    def _cols_daf(self, cols_ls: T_ls) -> Daf:
        """ A Daf of only these columns, in row order. Names that are not columns are left out. Internal use. """
        cols_ls = [col for col in dict.fromkeys(cols_ls) if col in self.hd]
        lol: T_lola = []
        batch_lot: List[Tuple[Any, ...]]
        for batch_lot in self._iter_row_batches(cols=cols_ls):
            lol.extend(none_to_null(row_t) for row_t in batch_lot)
        return Daf(lol=lol, cols=cols_ls)


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


    def col(                                                                # type: ignore[override]
            self,
            colname:        str,
            *,
            unique:         bool = False,
            omit_nulls:     bool = False,
            silent_error:   bool = False,
            astype:         Callable | str | type | None = None,
            indirect_col:   str | None = None,
            default:        Any = NULL,
            ) -> T_la:
        """
        Get one column as a list, by name, as `Daf.col()` does.

        Only that column is read into memory, or only `indirect_col` for a value inside a dict
        column. The options act as in a Daf. With `omit_nulls`, both SQL NULL and the text '' are
        left out, since both read as Python NULL.

        Args:
            colname: The column, or the key inside `indirect_col`.
            unique: If True, leave out repeated values and keep the order.
            omit_nulls: If True, leave out Python NULL.
            silent_error: If True, a name that is not a column gives an empty list.
            astype: A type or function to convert each value with.
            indirect_col: A column that holds a dict, to read `colname` from.
            default: The value for a dict that lacks `colname`.

        Returns:
            The values of the column, in row order.

        Raises:
            ColumnNotFoundError: The name is not a column, and `silent_error` is False.
        """
        return self.col_to_la(colname, unique=unique, omit_nulls=omit_nulls, silent_error=silent_error,
                              astype=astype, indirect_col=indirect_col, default=default)


    def col_to_la(                                                          # type: ignore[override]
            self,
            colname:        str,
            *,
            unique:         bool = False,
            omit_nulls:     bool = False,
            silent_error:   bool = False,
            astype:         Callable | str | type | None = None,
            indirect_col:   str | None = None,
            default:        Any = NULL,
            ) -> T_la:
        """
        Get one column as a list, by name. This does the same as `col()`. See that method.

        Args:
            colname: The column, or the key inside `indirect_col`.
            unique: If True, leave out repeated values and keep the order.
            omit_nulls: If True, leave out Python NULL.
            silent_error: If True, a name that is not a column gives an empty list.
            astype: A type or function to convert each value with.
            indirect_col: A column that holds a dict, to read `colname` from.
            default: The value for a dict that lacks `colname`.

        Returns:
            The values of the column, in row order.
        """
        # A plain column is read straight into a list, with the options applied as Daf applies them.
        if colname in self.hd:
            vals_la: T_la = [NULL if row_t[0] is None else row_t[0]
                             for batch_lot in self._iter_row_batches(cols=[colname]) for row_t in batch_lot]
            if omit_nulls:
                vals_la = [val for val in vals_la if val is not NULL]     # the text '' is the same object as NULL.
            if unique:
                vals_la = list(dict.fromkeys(vals_la))
            return daf_utils.astype_la(vals_la, astype)

        # A value inside a dict column, or a missing name, goes through Daf, so the results match.
        return self._cols_daf([indirect_col or '']).col_to_la(colname, unique=unique, omit_nulls=omit_nulls, silent_error=silent_error,
                                                              astype=astype, indirect_col=indirect_col, default=default)


    def icol(self, icol: int) -> T_la:                                      # type: ignore[override]
        """
        Get one column as a list, by its place among the columns, as `Daf.icol()` does.

        The place is looked up in the column names, so it does not depend on the rows. A
        negative place counts from the end. A place out of range raises IndexError, unless the
        table has no rows, as in a Daf.

        Args:
            icol: The place of the column, from 0.

        Returns:
            The values of the column, in row order.

        Raises:
            IndexError: The place is out of range, and the table has rows.
        """
        return self.icol_to_la(icol)


    def icol_to_la(self, icol: int, unique: bool = False, omit_nulls: bool = False) -> T_la:     # type: ignore[override]
        """
        Get one column as a list, by its place among the columns, with options.

        See `icol()` for the place, and `col()` for the options.

        Args:
            icol: The place of the column, from 0.
            unique: If True, leave out repeated values and keep the order.
            omit_nulls: If True, leave out Python NULL.

        Returns:
            The values of the column, in row order.

        Raises:
            IndexError: The place is out of range, and the table has rows.
        """
        cols_ls: T_ls = list(self.hd)
        if not -len(cols_ls) <= icol < len(cols_ls):
            if not self:
                return []
            raise IndexError(f"icol: column {icol} is out of range. '{self.name}' has {len(cols_ls)} columns.")
        return self.col_to_la(cols_ls[icol], unique=unique, omit_nulls=omit_nulls)


    def cols_to_dol(self, colname1: str, colname2: str) -> Dict[Any, T_la]:    # type: ignore[override]
        """
        Map each value of one column to the values seen with it in another, as `Daf.cols_to_dol()` does.

        Only the two columns are read into memory. If a name is not a column, or there are no
        rows, the result is empty.

        Args:
            colname1: The column of keys.
            colname2: The column of values.

        Returns:
            A dict from each value of the first column to the values of the second, in the order first seen.
        """
        if colname1 not in self.hd or colname2 not in self.hd:
            return {}
        return self._cols_daf([colname1, colname2]).cols_to_dol(colname1, colname2)


    # ---- whole-table exports ----

    def to_lod(self) -> List[T_da]:                                         # type: ignore[override]
        """
        Make a list of dicts, one for each row, as `Daf.to_lod()` does.

        Every row is read into memory. Use `iter_dict()` to go through the rows one at a time.

        Returns:
            The rows as dicts, in order.
        """
        return list(self.iter_dict())


    def to_dod(self, remove_keyfield: bool = True) -> Dict[Any, T_da]:     # type: ignore[override]
        """
        Make a dict of dicts, named by the keyfield, as `Daf.to_dod()` does.

        Every row is read into memory, through `to_daf()`.

        Args:
            remove_keyfield: If True, leave the keyfield out of the inner dicts.

        Returns:
            Each row as a dict, under its key.

        Raises:
            KeysDisabledError: There is no keyfield, and there are rows.
        """
        return self.to_daf().to_dod(remove_keyfield=remove_keyfield)


    def to_json(self, concise: bool = True) -> str:                        # type: ignore[override]
        """
        Make JSON text that holds the whole table, as `Daf.to_json()` does.

        Every row is read into memory, through `to_daf()`. The text holds the rows, the column
        names, the dtypes, the keyfield and the name. `Daf.from_json()` reads it back as a Daf.

        Args:
            concise: If True, leave out the parts that are empty.

        Returns:
            The JSON text.
        """
        return self.to_daf().to_json(concise=concise)


    def daf_to_lol_summary(                                                 # type: ignore[override]
            self,
            max_rows:   int = 10,
            max_cols:   int = 10,
            disp_cols:  Any = None,
            ) -> T_lola:
        """
        Give the header and the rows to display, as `Daf.daf_to_lol_summary()` does.

        With `max_rows`, only the first and last rows are read, around a row of '...'. Without
        it, every row is read.

        Args:
            max_rows: The most rows to show. 0 for all.
            max_cols: The most columns to show. 0 for all.
            disp_cols: Names to show in place of the column names.

        Returns:
            The header, then the rows.
        """
        colnames_ls: T_ls = list(disp_cols) if disp_cols else list(self.hd)
        header_lol: T_lola = [colnames_ls] if colnames_ls else []
        num_rows: int = len(self) if max_rows else 0
        if not max_rows or num_rows <= max_rows:
            result_lol: T_lola = header_lol + [none_to_null(row_t) for row_t in self._select_rows()]
            return daf_utils.reduce_lol_cols(result_lol, max_cols=max_cols) if max_cols or max_rows else result_lol

        # Read the head in order, and the tail backwards, so neither steps over the middle rows.
        rows_sql, rows_params = self._rows_sql()
        first_lol: T_lola = [none_to_null(row_t) for row_t in self._select_rows(limit=(max_rows + 1) // 2)]
        last_lot: List[Tuple[Any, ...]] = self._execute(
                f'SELECT {self._cols_sql()} FROM ({rows_sql}) ORDER BY {ORD_COL} DESC LIMIT ?',
                rows_params + (max_rows // 2,)).fetchall()
        last_lol: T_lola = [none_to_null(row_t) for row_t in reversed(last_lot)]
        divider_lol: T_lola = [['...'] * len(self.hd)]
        return daf_utils.reduce_lol_cols(header_lol + first_lol + divider_lol + last_lol, max_cols=max_cols)


    def to_md(                                                              # type: ignore[override]
            self,
            max_rows:           int = 0,
            max_cols:           int = 0,
            just:               str = '',
            shorten_text:       bool = True,
            max_text_len:       int = 80,
            smart_fmt:          bool = False,
            include_summary:    bool = False,
            disp_cols:          Any = None,
            header:             Any = None,
            ) -> str:
        """
        Make a Markdown table, as `Daf.to_md()` does.

        Set `max_rows` for a large table. Then only the first and last rows are read. Without it,
        every row is read into memory.

        Args:
            max_rows: The most rows to show, the first and the last. 0 for all.
            max_cols: The most columns to show, the first and the last. 0 for all.
            just: The justification of each column: '<', '^' or '>'.
            shorten_text: If True, shorten long text, keeping the ends.
            max_text_len: The length past which text is shortened.
            smart_fmt: If True, limit the digits after the point in numbers.
            include_summary: If True, add a line with the shape, keyfield and name.
            disp_cols: Names to show in place of the column names.
            header: A header to use in place of the column names.

        Returns:
            The Markdown text.
        """
        md_str: str = Daf.to_md(self, max_rows=max_rows, max_cols=max_cols, just=just, shorten_text=shorten_text,
                                max_text_len=max_text_len, smart_fmt=smart_fmt, include_summary=False,
                                disp_cols=disp_cols, header=header)
        if include_summary:
            md_str += f"\n%% daf rows={len(self)}; cols={len(self.hd)}; keyfield='{self.keyfield}'; name='{self.name}'\n"
        return md_str


    def to_md_cols(                                                         # type: ignore[override]
            self,
            max_rows:           int = 0,
            max_cols:           int = 0,
            just:               str = '',
            shorten_text:       bool = True,
            max_text_len:       int = 80,
            smart_fmt:          bool = False,
            include_summary:    bool = False,
            disp_cols:          T_ls | None = None,
            ) -> str:
        """
        Make a Markdown table with each row as a column, as `Daf.to_md_cols()` does.

        Set `max_rows` for a large table. Then only the first and last rows are read.

        Args:
            max_rows: The most rows to show, the first and the last. 0 for all.
            max_cols: The most columns to show, the first and the last. 0 for all.
            just: The justification of each column: '<', '^' or '>'.
            shorten_text: If True, shorten long text, keeping the ends.
            max_text_len: The length past which text is shortened.
            smart_fmt: If True, limit the digits after the point in numbers.
            include_summary: Not used, as in a Daf.
            disp_cols: Names to show in place of the column names.

        Returns:
            The Markdown text.
        """
        return Daf.to_md_cols(self, max_rows=max_rows, max_cols=max_cols, just=just, shorten_text=shorten_text,
                              max_text_len=max_text_len, smart_fmt=smart_fmt, include_summary=include_summary,
                              disp_cols=disp_cols)


    # ---- selecting ----

    def select_where_sql(self, where_sql: str, params: Iterable[Any] = ()) -> 'SqlDaf':
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


    def select_where(
            self,
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


    def split_where(                                                        # type: ignore[override]
            self,
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


    def select_first_row_by_dict(self, selector_da: T_da, inverse: bool = False) -> T_da:     # type: ignore[override]
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


    # ---- counts and sums ----

    def _value_counts_lt(self, colnames: T_ls, where_sql: str = '', params: Iterable[Any] = ()) -> List[Tuple[Any, ...]]:
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


    def valuecounts_for_colname(                                            # type: ignore[override]
            self,
            colname:    str,                            # the column to count.
            sort:       bool = False,                   # if True, order by count.
            reverse:    bool = True,                    # with sort, True puts the most common first.
            omit_nulls: bool = False,                   # if True, leave out the count of Python NULL.
            ) -> Dict[Any, int]:
        """
        Count how often each value occurs in one column, as `Daf.valuecounts_for_colname()` does.

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
            >>> import tempfile
            >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
            >>> bif_sdaf = SqlDaf.from_daf(Daf(lol=[['b1', 'P1'], ['b2', 'P2'], ['b3', 'P1']], cols=['ballot_id', 'precinct']), name='bif')
            >>> bif_sdaf.to_daf()
            <BLANKLINE>
            | ballot_id | precinct |
            | --------: | -------: |
            |        b1 |       P1 |
            |        b2 |       P2 |
            |        b3 |       P1 |
            <BLANKLINE>
            %% daf rows=3; cols=2; keyfield=''; name='bif'
            <BLANKLINE>
            >>> bif_sdaf.valuecounts_for_colname('precinct', sort=True)
            {'P1': 2, 'P2': 1}
        """
        if colname not in self.hd:
            return {}
        valuecounts_di: T_di = {val: count for val, count in self._value_counts_lt([colname])}
        if omit_nulls:
            valuecounts_di.pop(NULL, None)
        if sort:
            valuecounts_di = dict(sorted(valuecounts_di.items(), key=lambda item: item[1], reverse=reverse))
        return valuecounts_di


    def valuecounts_for_colnames_ls(                                        # type: ignore[override]
            self,
            colnames_ls:    T_ls | None = None,         # the columns to count. All if None.
            sort:           bool = False,
            reverse:        bool = True,
            omit_nulls:     bool = False,
            ) -> Dict[str, Dict[Any, int]]:
        """
        Count how often each value occurs, in each of several columns, as `Daf.valuecounts_for_colnames_ls()` does.

        It is one GROUP BY for each column. See `valuecounts_for_colname()`.

        Args:
            colnames_ls: The columns to count. All if None.
            sort: If True, order each count by size.
            reverse: With `sort`, True puts the most common first.
            omit_nulls: If True, leave out the count of Python NULL.

        Returns:
            A dict that maps each column to a dict of value and count.
        """
        return {colname: self.valuecounts_for_colname(colname, sort=sort, reverse=reverse, omit_nulls=omit_nulls)
                for colname in (colnames_ls or self.columns())}


    def valuecounts_for_colname_selectedby_colname(                         # type: ignore[override]
            self,
            colname:                str,                # the column to count.
            selectedby_colname:     str,                # the column to test.
            selectedby_colvalue:    Any,                # only rows where that column has this value are counted.
            sort:                   bool = False,
            reverse:                bool = True,
            ) -> Dict[Any, int]:
        """
        Count the values of a column, in the rows where another column has a value, as the Daf method does.

        Args:
            colname: The column to count.
            selectedby_colname: The column to test.
            selectedby_colvalue: Only rows where that column has this value are counted. Python NULL matches Python NULL.
            sort: If True, order the counts by size.
            reverse: With `sort`, True puts the most common first.

        Returns:
            A dict that maps each value to its count. It is empty if a column does not exist.
        """
        if colname not in self.hd or selectedby_colname not in self.hd:
            return {}
        cond_sql, cond_params = is_value_sql(selectedby_colname, selectedby_colvalue)
        valuecounts_di: T_di = {val: count for val, count in self._value_counts_lt([colname], cond_sql, cond_params)}
        if sort:
            valuecounts_di = dict(sorted(valuecounts_di.items(), key=lambda item: item[1], reverse=reverse))
        return valuecounts_di


    def valuecounts_for_colnames_ls_selectedby_colname(                     # type: ignore[override]
            self,
            colnames_ls:            T_ls | None = None, # the columns to count. All if None.
            selectedby_colname:     str = '',           # the column to test.
            selectedby_colvalue:    Any = '',           # only rows where that column has this value are counted.
            sort:                   bool = False,
            reverse:                bool = True,
            ) -> Dict[str, Dict[Any, int]]:
        """
        Count the values of several columns, in the rows where another column has a value, as the Daf method does.

        Args:
            colnames_ls: The columns to count. All if None.
            selectedby_colname: The column to test.
            selectedby_colvalue: Only rows where that column has this value are counted.
            sort: If True, order the counts by size.
            reverse: With `sort`, True puts the most common first.

        Returns:
            A dict that maps each column to a dict of value and count.
        """
        return {colname: self.valuecounts_for_colname_selectedby_colname(
                    colname, selectedby_colname, selectedby_colvalue, sort=sort, reverse=reverse)
                for colname in (colnames_ls or self.columns())}


    def valuecounts_for_colname1_groupedby_colname2(                        # type: ignore[override]
            self,
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


    def value_counts_daf(                                                   # type: ignore[override]
            self,
            colname:        str,                        # the column to count.
            sort:           bool = False,
            reverse:        bool = True,
            include_total:  bool = False,               # if True, add a row with the total.
            omit_nulls:     bool = False,
            ) -> Daf:
        """
        Count the values of a column into a Daf in memory, with one row per value, as `Daf.value_counts_daf()` does.

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
        """
        value_counts_daf: Daf = Daf.from_lod_to_cols(
                [self.valuecounts_for_colname(colname, sort=sort, reverse=reverse, omit_nulls=omit_nulls)],
                cols=[colname, 'counts'])
        if include_total:
            value_counts_daf.append({colname: ' **Total** ', 'counts': sum(value_counts_daf.col('counts'))})
        return value_counts_daf


    def daf_valuecount(                                                     # type: ignore[override]
            self,
            by:     str = 'row',                        # only 'row', as the default of Daf.
            cols:   T_ls | None = None,                 # the columns to count. All if None.
            ) -> T_da:
        """
        Count how often each value occurs, in each column, as `Daf.daf_valuecount()` does.

        It gives the same result: a column that is not in `cols` gets Python NULL in place of its
        counts, and a table with no rows gives an empty dict. Each column is one GROUP BY.

        Args:
            by: Only `row` is supported.
            cols: The columns to count. All if None.

        Returns:
            A dict that maps each column to a dict of value and count.

        Raises:
            NotImplementedError: `by` is not `row`.
        """
        if by != 'row':
            raise NotImplementedError(f"daf_valuecount(by={by!r}) is not supported on a SqlDaf.")
        if not self:
            return {}
        count_cols_set: set = set(cols if cols is not None else self.columns())
        return {colname: (self.valuecounts_for_colname(colname) if colname in count_cols_set else NULL)
                for colname in self.columns()}


    def sum(                                                                # type: ignore[override]
            self,
            colnames_ls:    T_ls | None = None,         # the columns to total. All if None.
            numeric_only:   bool = False,               # if True, total only int and float columns, and text counts as 0.
            ) -> T_da:
        """
        Total the columns, as `Daf.sum()` does, and return a dict of the totals.

        Numbers are totalled by SQLite. A column that also holds text has its text cells read
        into Python and added as `Daf.sum()` adds them: text that is not a number raises
        ValueError, or counts as 0 with `numeric_only`. Python NULL is skipped. Each total starts as a
        float, and is converted to the dtype of its column if there is one.

        Args:
            colnames_ls: The columns to total. All if None.
            numeric_only: If True, total only the columns with an int or float dtype, and count text that is not a number as 0.

        Returns:
            A dict that maps each column name to its total.

        Raises:
            ValueError: A text cell cannot be converted to a number.

        Examples:
            >>> import tempfile
            >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
            >>> SqlDaf.from_daf(Daf(lol=[[1, 10], [2, 20]], cols=['x', 'y']), name='nums').sum()
            {'x': 3.0, 'y': 30.0}
        """
        sum_cols_ls: T_ls
        if colnames_ls is None:
            sum_cols_ls = self.columns()
        elif not (numeric_only and self.dtypes):
            sum_cols_ls = list(dict.fromkeys(col for col in colnames_ls if col in self.hd))
        else:
            sum_cols_ls = list(dict.fromkeys(col for col in colnames_ls if col in self.hd and self.dtypes.get(col) in (int, float)))
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
                to_float: Callable[[Any], float] = Daf._safe_tofloat if numeric_only else float
                text_lot: List[Tuple[Any, ...]] = self._fetch(self._execute(
                        f"SELECT {quote_name(col)} FROM ({rows_sql}) WHERE typeof({quote_name(col)}) = 'text'", rows_params))
                sums_d[col] += sum(to_float(text_t[0]) for text_t in text_lot if text_t[0])
        return daf_utils.set_dict_dtypes(sums_d, dtypes=self.dtypes)


    def groupsum_daf(                                                       # type: ignore[override]
            self,
            colname:        str,                        # the column to group by.
            by:             str = 'row',                # only 'row', as the default of Daf.
            reduce_cols:    T_ls | None = None,         # the columns to add. All but colname if None.
            ) -> Daf:
        """
        Group by a column and add up the other columns of each group, as `Daf.groupsum_daf()` does.

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
            by: Only `row` is supported.
            reduce_cols: The columns to add. All but `colname` if None.

        Returns:
            A Daf in memory, with one row for each group.

        Raises:
            KeyError: `colname` is not a column.
            NotImplementedError: `by` is not `row`.

        Examples:
            >>> import tempfile
            >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
            >>> nums_sdaf = SqlDaf.from_daf(Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y']), name='nums')
            >>> nums_sdaf.to_daf()
            <BLANKLINE>
            | g  | x  | y  |
            | -: | -: | -: |
            |  a |  1 | 10 |
            |  b |  2 | 20 |
            |  a |  3 | 30 |
            <BLANKLINE>
            %% daf rows=3; cols=3; keyfield=''; name='nums'
            <BLANKLINE>
            >>> nums_sdaf.groupsum_daf('g', reduce_cols=['y']).lol
            [['a', '', 40], ['b', '', 20]]
        """
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


    # ---- groups ----

    def groupby(                                                            # type: ignore[override]
            self,
            colname:        str = '',                   # the column to group by.
            colnames:       T_ls | None = None,         # several columns to group by, as groupby_cols().
            omit_nulls:     bool = False,               # if True, leave out rows with Python NULL in the column.
            cols:           T_ls | None = None,         # not supported on a SqlDaf yet.
            ) -> KeyedList:
        """
        Group the rows by the values of a column, as `Daf.groupby()` does.

        The result is a KeyedList of selections, one for each value, in the order first seen. It
        works as a dict, as the result of `Daf.groupby()` does. One query finds the values. Each
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
            cols: Not supported on a SqlDaf yet.

        Returns:
            A KeyedList that maps each value to a selection.

        Raises:
            KeyError: The column is not a column.
            NotImplementedError: `cols` is given.

        Examples:
            >>> import tempfile
            >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
            >>> bif_sdaf = SqlDaf.from_daf(Daf(lol=[['b1', 'P1'], ['b2', 'P2'], ['b3', 'P1']], cols=['ballot_id', 'precinct']), name='bif')
            >>> bif_sdaf.to_daf()
            <BLANKLINE>
            | ballot_id | precinct |
            | --------: | -------: |
            |        b1 |       P1 |
            |        b2 |       P2 |
            |        b3 |       P1 |
            <BLANKLINE>
            %% daf rows=3; cols=2; keyfield=''; name='bif'
            <BLANKLINE>
            >>> groups = bif_sdaf.groupby('precinct')
            >>> {key: group.to_daf().lol for key, group in groups.items()}
            {'P1': [['b1', 'P1'], ['b3', 'P1']], 'P2': [['b2', 'P2']]}
        """
        if isinstance(colname, list) and not colnames:
            return self.groupby_cols(colname, cols=cols)
        if colnames and not colname:
            if len(colnames) > 1:
                return self.groupby_cols(colnames, cols=cols)
            colname = colnames[0]
        groups_kl: KeyedList = self.groupby_cols([colname], cols=cols, omit_nulls=omit_nulls)
        return KeyedList([cast(T_ta, key_t)[0] for key_t in groups_kl.keys()], groups_kl.values())


    def groupby_reduce(                                                     # type: ignore[override]
            self,
            colname:        str,                        # the column to group by.
            func:           Callable[..., Any],         # reduces the rows of one group, as in Daf.reduce().
            by:             str = 'row',                # 'row' or 'sparse_row', as in Daf.reduce().
            reduce_cols:    T_ls | None = None,         # the columns to reduce. All except colname if None.
            diagnose:       bool = False,               # if True, show the query and the first results.
            batch_size:     int = 10000,                # not used. Kept so that existing calls work.
            **kwargs:       Any,                        # passed on to func, and indirect_col for 'sparse_row'.
            ) -> 'SqlDaf':
        """
        Group the rows by one column and reduce each group to one row, as `Daf.groupby_reduce()` does.

        Use it to turn many rows of one key into one row, such as the contests of a ballot
        into one record of the ballot.

        The groups are reduced inside SQLite, in one query, by a registered aggregate,
        as `groupby_aggregate()` does. No query is made for each group.

        - With `by='row'`, the rows of each group become a Daf of the columns the group holds.
          These are `reduce_cols`, or all columns. The cells are as stored, so a dict cell is text.
          `Daf.reduce()` reduces it, so the function is called just as `Daf.groupby_reduce()` calls it.
          A plain `Daf.sum_da` adds down each column there, with no call for each row.
        - With `by='sparse_row'`, the function gets the dict held in the cell of `indirect_col`.
          Its `cols` is `reduce_cols`, and the reduction starts as an empty dict.

        For totals of columns that hold only numbers, `groupsum_daf()` adds them in SQL, several times faster.

        The new table has a row for each value of the column, with the column as its keyfield.
        With `by='row'` it has the columns of this table. The columns not reduced are empty.
        With `by='sparse_row'` it has two: this column, and the column named by `indirect_col`.
        That cell holds the dict of the group, with only the keys found in the group, as compact JSON text.
        Results are in daffodil form, as `groupby_aggregate()` describes.
        The new table is a TEMP table, as `copy()` makes. Keep it with `save(name)`.

        It differs from `Daf.groupby_reduce()` in three ways:

        - The rows are in the sorted order of the column, not in the order first seen.
          SQL NULL and '' are one group, first.
        - Each group is a plain Daf of the rows, with no keyfield and no dtypes.
        - `by='table'` is not supported.

        Args:
            colname: The column to group by.
            func: The reduction function, such as `Daf.sum_da`. See `Daf.reduce()`.
            by: `row` or `sparse_row`. See `Daf.reduce()`.
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
            >>> import tempfile
            >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
            >>> votes_sdaf = SqlDaf('votes', cols=['precinct', 'concise_votes'],
            ...                     lol=[['P2', "{'c0v': 1}"], ['P1', "{'c0v': 1, 'c1v': 1}"], ['P2', "{'c1v': 1}"]])
            >>> votes_sdaf.to_daf()
            <BLANKLINE>
            | precinct |    concise_votes     |
            | -------: | -------------------: |
            |       P2 |           {'c0v': 1} |
            |       P1 | {'c0v': 1, 'c1v': 1} |
            |       P2 |           {'c1v': 1} |
            <BLANKLINE>
            %% daf rows=3; cols=2; keyfield=''; name='votes'
            <BLANKLINE>
            >>> sums_sdaf = votes_sdaf.groupby_reduce('precinct', Daf.sum_da, by='sparse_row', indirect_col='concise_votes')
            >>> sums_sdaf.to_daf().lol
            [['P1', '{"c0v":1,"c1v":1}'], ['P2', '{"c0v":1,"c1v":1}']]
        """
        if colname not in self.hd:
            raise KeyError(colname)
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
            agg_cls = type('BoundCellReduction', (CellReduction,), {
                    'func':         staticmethod(func),
                    'col':          indirect_col,
                    'call_kwargs':  {'cols': reduce_cols, 'is_sparse': True, **func_kwargs},
                    'sparse':       True,
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


    def groupby_aggregate(
            self,
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

        The function has the form of `Daf.sum_da()`: it takes a row as a dict and the reduction
        so far, and returns the reduction.

        There are two forms, by the number of columns:

        - **One column.** The function gets the cell as the row, if it holds a dict, as text of
          JSON or PYON. This merges or totals concise dicts, which SQL cannot read.
          The result cell is the reduced dict, as compact JSON text.
          A cell that holds a plain value is given as a row of one column, and the result is its
          reduced value. An empty cell is an empty dict.
        - **Several columns.** The function gets a dict of the grouping columns and these
          columns, with each dict cell decoded. The reduction starts with 0 for each column of
          `reduce_cols`, as in `Daf.reduce()`. The result has a column for each of `result_cols`.

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
            >>> import tempfile
            >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
            >>> votes_sdaf = SqlDaf('votes', cols=['precinct', 'v', 'concise_votes'],
            ...                     lol=[['P2', 1, "{'c0v': 1}"], ['P1', 1, "{'c0v': 1, 'c1v': 1}"], ['P2', 0, "{'c1v': 1}"]])
            >>> votes_sdaf.to_daf()
            <BLANKLINE>
            | precinct | v  |    concise_votes     |
            | -------: | -: | -------------------: |
            |       P2 |  1 |           {'c0v': 1} |
            |       P1 |  1 | {'c0v': 1, 'c1v': 1} |
            |       P2 |  0 |           {'c1v': 1} |
            <BLANKLINE>
            %% daf rows=3; cols=3; keyfield=''; name='votes'
            <BLANKLINE>
            >>> votes_sdaf.groupby_aggregate('precinct', Daf.sum_da, 'concise_votes').to_daf().lol
            [['P1', '{"c0v":1,"c1v":1}'], ['P2', '{"c0v":1,"c1v":1}']]
            >>> votes_sdaf.groupby_aggregate('precinct', Daf.sum_da, ['v', 'concise_votes'], result_cols=['v']).to_daf().lol
            [['P1', 1], ['P2', 1]]
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
            agg_cls = type('BoundCellReduction', (CellReduction,), {
                    'func':         staticmethod(func),
                    'col':          reduce_cols_ls[0],
                    'call_kwargs':  {'cols': None, 'is_sparse': True, **kwargs},
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
            self,
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
        target: str = self._next_temp_name((self.parent_sdaf or self).name)
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


    def groupby_cols(                                                       # type: ignore[override]
            self,
            colnames:       T_ls,                       # the columns to group by.
            cols:           T_ls | None = None,         # not supported on a SqlDaf yet.
            omit_nulls:     bool = False,               # if True, leave out rows with Python NULL in any of the columns.
            ) -> KeyedList:
        """
        Group the rows by the values of several columns, as `Daf.groupby_cols()` does.

        Each key is a tuple of the values, even for one column. Each group is a selection, as in
        `groupby()`.

        Args:
            colnames: The columns to group by.
            cols: Not supported on a SqlDaf yet.
            omit_nulls: If True, leave out the rows with Python NULL in any of the columns.

        Returns:
            A KeyedList that maps each tuple of values to a selection.

        Raises:
            KeyError: A name in `colnames` is not a column.
            NotImplementedError: `cols` is given.
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


    def _ensure_group_index(self, colnames: T_ls) -> None:
        """
        Make an index on the columns a groupby uses, on the parent table, if none starts with them. Internal use.

        Each group is a selection by its value, so without an index each group reads the whole table.
        """
        root_sdaf: SqlDaf = self.parent_sdaf or self
        if any(index_cols[:len(colnames)] == colnames for index_cols in _indexes_of(self.conn, root_sdaf.name).values()):
            return
        index_name: str = f'{GROUP_INDEX_PREFIX}{root_sdaf.name}_' + '_'.join(colnames)
        with self.transaction():
            self._execute(f'CREATE INDEX IF NOT EXISTS {quote_name(index_name)} ON {quote_name(root_sdaf.name)} '
                          f'({", ".join(quote_name(col) for col in colnames)})')


    def _drop_group_indexes(self, cols: T_ls) -> None:
        """ Drop the indexes groupby() made that use any of these columns, before they are dropped or renamed. Internal use. """
        index_name: str
        index_cols: T_ls
        for index_name, index_cols in _indexes_of(self.conn, self.name).items():
            if index_name.startswith(GROUP_INDEX_PREFIX) and any(col in cols for col in index_cols):
                self._execute(f'DROP INDEX {quote_name(index_name)}')


    def group_where(                                                        # type: ignore[override]
            self,
            where:          Callable[[Any], Any],       # a function of a row: a key, a list of keys, or None.
            *,
            indirect_col:   str | None = None,          # not supported yet.
            cols:           T_ls | None = None,         # the columns the function reads. All if None.
            ) -> KeyedList:
        """
        Group the rows by the keys a Python function returns, as `Daf.group_where()` does.

        The function gets each row as a KeyedList, as in `select_where()`, and returns a key, a
        list of keys, or None. A row goes into the group of each key, so a row can be in any
        number of groups, and None leaves it out. As in `Daf.group_where()`, any iterable other
        than a str is a list of keys, a tuple too. So to use a tuple as one key, return a list
        of tuples.

        The function runs once for each row, in one pass, and each (key, row) pair is written to
        a TEMP table. Each group is then a selection of the rows listed under its key, in the
        order of this SqlDaf. Nothing more is read until a group is read. The result is a
        KeyedList of selections, in the order the keys were first returned.

        Name the columns the function reads in `cols`, as in `select_where()`.

        Args:
            where: A function of a row: a key, a list of keys, or None.
            indirect_col: Not supported on a SqlDaf yet.
            cols: The columns the function reads. All if None.

        Returns:
            A KeyedList that maps each key to a selection.

        Raises:
            KeyError: A name in `cols` is not a column.
            NotImplementedError: `indirect_col` is given.

        Examples:
            >>> import tempfile
            >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
            >>> bif_sdaf = SqlDaf.from_daf(Daf(lol=[['b1', 'P1'], ['b2', 'P2'], ['b3', 'P1']], cols=['ballot_id', 'precinct']), name='bif')
            >>> bif_sdaf.to_daf()
            <BLANKLINE>
            | ballot_id | precinct |
            | --------: | -------: |
            |        b1 |       P1 |
            |        b2 |       P2 |
            |        b3 |       P1 |
            <BLANKLINE>
            %% daf rows=3; cols=2; keyfield=''; name='bif'
            <BLANKLINE>
            >>> groups = bif_sdaf.group_where(lambda row: [row['precinct'], 'all'], cols=['precinct'])
            >>> {key: len(group) for key, group in groups.items()}
            {'P1': 2, 'all': 3, 'P2': 1}
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
        labels: str = self._next_temp_name((self.parent_sdaf or self).name + '_groups')
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


    def __getitem__(self, sel: Any) -> 'SqlDaf':
        """
        Select rows by position or by key, as `daf[n]`, `daf[n:m]` and `daf[key]` do.

        Each gives a selection, not an ordinary Daf, so a large slice stays in the database.
        Read it with `to_daf()`, or `to_dict()` for one row. For one row as a dict, `iloc()` and
        `select_record()` are cheaper: they make no selection.

        - An int is always a position, from 0, even when the keyfield holds ints. Negative counts
          from the end. A position past the end raises IndexError, as in a Daf. For an int key,
          use `select_record()`, or `select_records_daf()` for several.
        - A slice of positions, as in `sdaf[10:20]`. A step is not supported yet. A slice past
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


    def iloc_exists(self, irow: int) -> bool:
        """
        Say whether there is a row at this position, from 0.

        Args:
            irow: The position, from 0.

        Returns:
            True if a row is there.

        Raises:
            ValueError: This is a parent that has had rows deleted. See SqlDaf.
        """
        self._check_positions('iloc_exists')
        rows_sql, rows_params = self._rows_sql()
        if self._is_dense():
            return self._execute(f'SELECT 1 FROM ({rows_sql}) WHERE {ORD_COL} = ?',
                                      rows_params + (irow + 1,)).fetchone() is not None
        cursor: sqlite3.Cursor = self._execute(
                f'SELECT 1 FROM ({rows_sql}) ORDER BY {ORD_COL} LIMIT 1 OFFSET ?', rows_params + (irow,))
        return cursor.fetchone() is not None


    # ---- keys: selecting, deleting ----

    def keys(self) -> T_la:                                                 # type: ignore[override]
        """
        Get the keys, as a list of the keyfield's values in row order, as `Daf.keys()` does.

        Every key is read into memory. To pass many keys to `select_records_daf()` or
        `delete_records()` without reading them, give the SqlDaf itself: its keys are then used
        in SQL. Without a keyfield, as in a Daf, the answer is an empty list.

        Returns:
            The keys, or an empty list if there is no keyfield.
        """
        if not self.keyfield:
            return []
        rows_sql, rows_params = self._rows_sql()
        cursor: sqlite3.Cursor = self._execute(
                f'SELECT {quote_name(self.keyfield)} FROM ({rows_sql}) ORDER BY {ORD_COL}', rows_params)
        return [NULL if row_t[0] is None else row_t[0] for row_t in self._fetch(cursor)]


    def _key_source_sql(self, keys_ls: Any) -> Tuple[str, Tuple[Any, ...]]:
        """
        A query that gives the keys as daf_key and daf_ord, in order, from any form of keys. Internal use.

        A SqlDaf gives the keyfield of its rows. An in-memory Daf gives its keys, and a
        list or other iterable its values: both are written to a TEMP table of keys first, with
        an index, so the keys are read once and joined in SQL.
        """
        if isinstance(keys_ls, SqlDaf):
            self._check_same_db(keys_ls, 'The keys')
            if not keys_ls.keyfield:
                raise ValueError(f"The keys are the SqlDaf '{keys_ls.name}', which has no keyfield.")
            if keys_ls.parent_sdaf is None and keys_ls.keyfield:
                keys_ls.ensure_key_index()
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
        parent_sdaf: SqlDaf = self.parent_sdaf or self
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


    def select_krows(                                                       # type: ignore[override]
            self,
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


    def select_records_daf(
            self,
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


    def delete_records(
            self,
            keys_ls:        Any,                        # the keys of the rows to remove.
            inplace:        bool = False,               # True deletes from this SqlDaf. False makes a selection.
            silent_error:   bool = False,               # if True, keys that are not found are ignored.
            ) -> 'SqlDaf':
        """
        Remove the rows with these keys, as `Daf.delete_records()` does. A Python NULL key gives
        undefined results: see SqlDaf.

        By default nothing is changed: a selection without those rows is returned, as
        `select_records_daf(keys, inverse=True)` gives. That is always safe.

        With `inplace=True`, the rows are removed from this SqlDaf, and it is returned.

        - On a selection, its references to those rows are removed, and the rest renumbered. The
          parent is not changed, so this is safe, as in a Daf.
        - On a parent, the rows are deleted from the database. After that, it cannot be read
          by position: see SqlDaf. A selection made from it earlier still counts a deleted
          row in `len()`, though `to_daf()` leaves it out. So delete in place from a parent
          only when nothing else uses it.

        The keys may be in any form that `select_records_daf()` takes. A key that is not found
        raises KeyError, and nothing is removed, unless `silent_error` is True.

        Args:
            keys_ls: The keys of the rows to remove, in any form `select_records_daf()` takes.
            inplace: If True, remove the rows from this SqlDaf and return it. If False, return a selection.
            silent_error: If True, keys that are not found are ignored.

        Returns:
            This SqlDaf if `inplace`, otherwise a selection without the rows.

        Raises:
            KeyError: A key is not found, and `silent_error` is False. Nothing is removed.
            ValueError: This SqlDaf, or a SqlDaf given as the keys, has no keyfield.
        """
        if not inplace:
            return self.select_records_daf(keys_ls, inverse=True, silent_error=silent_error)

        key_col: str = self._need_keyfield('delete_records')
        keys_sql, keys_params = self._key_source_sql(keys_ls)
        if not silent_error:
            missing_key: Any = self._first_missing_key(keys_sql, keys_params)
            if missing_key is not None:
                raise KeyError(missing_key)

        if self.parent_sdaf is None:
            with self.transaction():
                self._execute(f'DELETE FROM {quote_name(self.name)} WHERE {quote_name(key_col)} IN '
                                   f'(SELECT daf_key FROM ({keys_sql}))', keys_params)
            self._conn_da['no_gaps_ss'].discard(self.name)
            return self

        self._resolve()
        rows_sql, rows_params = self._rows_sql()
        new_name: str = self._next_temp_name(self.parent_sdaf.name)
        with self.transaction():
            self._execute(f'CREATE TEMP TABLE {quote_name(new_name)} (ord INTEGER PRIMARY KEY, ref)')
            self._execute(
                    f'INSERT INTO {quote_name(new_name)} (ref) SELECT r.{REF_COL} FROM ({rows_sql}) AS r '
                    f'WHERE r.{quote_name(key_col)} NOT IN (SELECT daf_key FROM ({keys_sql}) WHERE daf_key IS NOT NULL) '
                    f'OR r.{quote_name(key_col)} IS NULL ORDER BY r.{ORD_COL}',
                    rows_params + keys_params)
            self._execute(f'DROP TABLE {quote_name(self.name)}')
            self._execute(f'ALTER TABLE {quote_name(new_name)} RENAME TO {quote_name(self.name)}')
        return self


    def remove_dups(self, keyfield: str = '') -> Tuple['SqlDaf', 'SqlDaf']:   # type: ignore[override]
        """
        Split the rows into those with a unique key and those with a repeated key, as `Daf.remove_dups()` does.

        Only the key column is compared, not the whole row.
        For each key, the last row is the unique one. The earlier rows with that key are the duplicates.
        The unique rows come in the order in which their keys first appear.
        The duplicates keep this table's order.

        Both are selections of this table's rows, as a Daf shares its rows with both.
        Both have the key column as their keyfield.
        SQL NULL and '' are one key here, as Python NULL is one key in a Daf.
        No key index is made, so the column may hold repeated keys.
        Each side is one GROUP BY, run when it is first read.

        A key of several columns is not supported, since a SqlDaf keyfield is one column.

        Args:
            keyfield: The column that identifies a row. If empty, the keyfield of this SqlDaf.

        Returns:
            A tuple of the selection of unique rows and the selection of duplicate rows.

        Raises:
            KeysDisabledError: There is no `keyfield`, and this SqlDaf has none.
            KeyError: The key column is not found.
            NotImplementedError: The key is several columns.

        Examples:
            >>> import tempfile
            >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
            >>> marks_sdaf = SqlDaf('marks', cols=['id', 'v'], lol=[[1, 'a'], [2, 'b'], [1, 'c']])
            >>> marks_sdaf.to_daf()
            <BLANKLINE>
            | id | v  |
            | -: | -: |
            |  1 |  a |
            |  2 |  b |
            |  1 |  c |
            <BLANKLINE>
            %% daf rows=3; cols=2; keyfield=''; name='marks'
            <BLANKLINE>
            >>> unique_sdaf, dups_sdaf = marks_sdaf.remove_dups('id')
            >>> unique_sdaf.to_daf().lol, dups_sdaf.to_daf().lol
            ([[1, 'c'], [2, 'b']], [[1, 'a']])
        """
        key_col: Any = keyfield or self.keyfield
        if not key_col:
            raise KeysDisabledError("remove_dups: give a keyfield, or set one on the SqlDaf.")
        if not isinstance(key_col, str):
            raise NotImplementedError("remove_dups: a key of several columns is not supported on a SqlDaf.")
        if key_col not in self.hd:
            raise KeyError(key_col)
        self._resolve_parent()
        rows_sql, rows_params = self._rows_sql()

        # The last row of each key is the unique one. It takes the place of the key's first row.
        groups_sql: str = (f"SELECT MAX({ORD_COL}) AS last_ord, MIN({ORD_COL}) AS first_ord FROM ({rows_sql}) "
                           f"GROUP BY COALESCE({quote_name(key_col)}, '')")
        unique_sql: str = (f'SELECT r.{REF_COL} AS {REF_COL}, g.first_ord AS {ORD_COL}{self._cols_sql("r.", lead=", ")} '
                           f'FROM ({rows_sql}) AS r JOIN ({groups_sql}) AS g ON r.{ORD_COL} = g.last_ord')
        dups_sql: str = (f'SELECT * FROM ({rows_sql}) '
                         f'WHERE {ORD_COL} NOT IN (SELECT last_ord FROM ({groups_sql}))')
        return (SqlDaf._new_selection(self, unique_sql, rows_params + rows_params, keyfield=key_col),
                SqlDaf._new_selection(self, dups_sql, rows_params + rows_params, keyfield=key_col))


    # ---- adding rows ----

    def concat(self, other_instance: Daf, respect_kd: bool = False) -> 'SqlDaf':
        """
        Add the rows of another Daf or SqlDaf to the end of this one, as `Daf.concat()` does.

        The other must have the same columns, in the same order, as in a Daf. A SqlDaf of the
        same database file is copied in SQL, in its row order, so no row comes into Python. An in-memory
        Daf is written row by row. All of it is one transaction.

        If this is a child, it first becomes a table of its own, by `copy(inplace=True)`, as
        appending to a selection of a Daf adds only to that selection. The parent is not
        changed.

        A Daf appends a row whose key is already there without a word. A SqlDaf with a keyfield
        has a unique index on it, so a repeated key raises instead, and nothing is added. A
        Python NULL key raises ValueError, and nothing is added.

        Args:
            other_instance: The Daf or SqlDaf whose rows are added. It is not changed.
            respect_kd: If True, a row whose key is there replaces that row, in its place.

        Returns:
            This SqlDaf.

        Raises:
            KeyError: The columns differ.
            ValueError: A key added is a Python NULL. Nothing is added.
            sqlite3.IntegrityError: A key would be repeated. Nothing is added.
            NotImplementedError: The other is a SqlDaf this connection cannot read.
        """
        if other_instance.columns() != self.columns():
            raise KeyError(f"concat: columns mismatch: this one has {self.columns()}, the other {other_instance.columns()}.")
        if isinstance(other_instance, SqlDaf):
            self._check_same_db(other_instance, 'concat')

        if self.parent_sdaf is not None:
            self.copy(inplace=True)
        self._resolve()
        if self.keyfield:
            self.ensure_key_index()

        upsert_sql: str = ''
        if respect_kd and self.keyfield:
            upsert_sql = f' ON CONFLICT ({quote_name(self.keyfield)}) DO UPDATE SET {self._excluded_sql()}'
        with self.transaction():
            if isinstance(other_instance, SqlDaf):
                rows_sql, rows_params = other_instance._rows_sql()
                # WHERE true lets SQLite tell the ON CONFLICT of an upsert from a join's ON.
                self._execute(f'INSERT INTO {quote_name(self.name)} ({self._cols_sql()}) '
                              f'SELECT {self._cols_sql()} FROM ({rows_sql}) WHERE true ORDER BY {ORD_COL}{upsert_sql}', rows_params)
            elif upsert_sql:
                self.conn.executemany(f'INSERT INTO {quote_name(self.name)} ({self._cols_sql()}) '
                                      f'VALUES ({placeholders(len(self.hd))}){upsert_sql}',
                                      (null_to_none(row_la) for row_la in other_instance.lol))
            else:
                self._insert_rows(other_instance.lol)
            if self.keyfield:
                self._check_no_null_key('concat')       # the unique index does not stop SQL NULL keys.
        return self


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


    def _write_rows(self, rows_lol: Iterable[T_la], respect_kd: bool, method: str) -> 'SqlDaf':
        """
        Add rows given in column order, in one transaction. Internal use.

        With `respect_kd` and a keyfield, a row whose key is there replaces that row, in its place.
        A row that fails leaves the table as it was.
        """
        if not self.hd:
            raise NotImplementedError(f"{method}: '{self.name}' has no columns. Make the table with its columns first.")
        if self.parent_sdaf is not None:
            if respect_kd and self.keyfield:
                raise NotImplementedError(f"{method}(respect_kd=True) on a selection is not supported. "
                                          "Write to its parent, or copy() the selection first.")
            self.copy(inplace=True)                     # a selection gets rows of its own, as concat() does.
        self._resolve()
        stmt: str = f'INSERT INTO {quote_name(self.name)} ({self._cols_sql()}) VALUES ({placeholders(len(self.hd))})'
        if self.keyfield:
            self.ensure_key_index()
            if respect_kd:
                stmt += f' ON CONFLICT ({quote_name(self.keyfield)}) DO UPDATE SET {self._excluded_sql()}'
        row_la: T_la
        with self.transaction():
            self.conn.executemany(stmt, (null_to_none(row_la) for row_la in rows_lol))
            if self.keyfield:
                self._check_no_null_key(method)
        return self


    def _excluded_sql(self) -> str:
        """ The SET part of an upsert, which takes every column from the new row. Internal use. """
        return ', '.join(f'{quote_name(col)} = excluded.{quote_name(col)}' for col in self.hd)


    def append(                                                             # type: ignore[override]
            self,
            data_item:  Any = None,
            respect_kd: bool = False,
            *,
            lol:        T_lola | None = None,
            la:         T_la | None = None,
            fast:       bool = False,
            ) -> 'SqlDaf':
        """
        Add one row, or several, to the end, as `Daf.append()` does.

        - A dict or a KeyedList is one row, placed by column name.
        - A list of values is one row, in column order. A short list is padded with Python NULL.
        - A list of dicts is several rows. See `extend()`.
        - A Daf or a SqlDaf is several rows. See `concat()`.
        - `lol=` is several rows, each a list. `la=` is one row.

        Each call is one transaction, and each commit writes to disk. On 1,000,000 rows, one
        call took 9.7 ms on its own, and 55 us inside `transaction()`. So add many rows in one
        call, with `extend()`, or put a loop of single rows inside `transaction()`.

        Keys are unique in a SqlDaf. A key that is already there raises sqlite3.IntegrityError,
        and nothing is added. With `respect_kd=True`, the row with that key is replaced, in its place. A
        Python NULL key raises ValueError, as in `concat()`.

        On a selection, the selection gets rows of its own first, as `concat()` does. Its parent
        is not changed.

        Args:
            data_item: A dict, a KeyedList, a list, a list of dicts, or a Daf.
            respect_kd: If True, replace the row that has the same key.
            lol: Several rows, each a list in column order.
            la: One row, a list in column order.
            fast: Accepted for Daf code. Every row is checked anyway.

        Returns:
            This SqlDaf, which has been changed.

        Raises:
            TypeError: More than one of `data_item`, `lol` and `la`, or a row of an unknown type.
            ValueError: A row has more values than there are columns, or a key is a Python NULL.
            sqlite3.IntegrityError: A key would be repeated. Nothing is added.
        """
        if sum(item is not None for item in (data_item, lol, la)) > 1:
            raise TypeError("append(): give only one of data_item, lol and la.")
        if lol is not None:
            return self.extend(lol=lol, respect_kd=respect_kd)
        if la is not None:
            return self._write_rows([self._row_from_item(list(la), 'append')], respect_kd, 'append')
        if data_item is None:
            return self
        if isinstance(data_item, Daf):
            return self.concat(data_item, respect_kd=respect_kd)
        if isinstance(data_item, list) and data_item and isinstance(data_item[0], (dict, KeyedList)):
            return self.extend(data_item, respect_kd=respect_kd)
        if not isinstance(data_item, (dict, KeyedList, list)):
            raise TypeError(f"append(): data_item must be a dict, KeyedList, list or Daf, not {type(data_item).__name__}.")
        if not data_item:
            return self
        return self._write_rows([self._row_from_item(data_item, 'append')], respect_kd, 'append')


    def record_append(self, record: T_da | KeyedList, respect_kd: bool = True) -> 'SqlDaf':     # type: ignore[override]
        """
        Add one row given as a dict or a KeyedList, as `Daf.record_append()` does.

        By default the row with the same key is replaced, in its place. This is the default
        here, unlike `append()`. An empty record adds nothing.

        Args:
            record: The row, as a dict or a [KeyedList][daffodil.keyedlist.KeyedList].
            respect_kd: If True, the default, replace the row that has the same key.

        Returns:
            This SqlDaf, which has been changed.

        Raises:
            TypeError: The record is not a dict or a KeyedList.
            ValueError: The key is a Python NULL.
            sqlite3.IntegrityError: The key is there, and `respect_kd` is False. Nothing is added.
        """
        if not isinstance(record, (dict, KeyedList)):
            raise TypeError(f"record_append(): the record must be a dict or a KeyedList, not {type(record).__name__}.")
        if not record:
            return self
        return self._write_rows([self._row_from_item(record, 'record_append')], respect_kd, 'record_append')


    def extend(                                                             # type: ignore[override]
            self,
            records_lod:    List[T_da] | None = None,
            respect_kd:     bool = False,
            *,
            lol:            T_lola | None = None,
            ) -> 'SqlDaf':
        """
        Add several rows, as dicts or as lists, as `Daf.extend()` does.

        Each dict is placed by column name. Each list of `lol` is in column order. All rows are
        added in one transaction. If one fails, none is added.

        Args:
            records_lod: The rows, as dicts.
            respect_kd: If True, replace the row that has the same key.
            lol: The rows, as lists in column order.

        Returns:
            This SqlDaf, which has been changed.

        Raises:
            TypeError: Both `records_lod` and `lol` are given, or a row of `lol` is not a list.
            ValueError: A row has more values than there are columns, or a key is a Python NULL.
            sqlite3.IntegrityError: A key would be repeated. Nothing is added.
        """
        if records_lod is not None and lol is not None:
            raise TypeError("extend(): give records_lod or lol, not both.")
        if lol is not None:
            row_la: Any
            for row_la in lol:
                if not isinstance(row_la, list):
                    raise TypeError(f"extend(): each row of lol must be a list, not {type(row_la).__name__}.")
            return self._write_rows((self._row_from_item(row_la, 'extend') for row_la in lol), respect_kd, 'extend')
        records_lod = [record for record in (records_lod or []) if record]
        if not records_lod:
            return self
        return self._write_rows((self._row_from_item(record, 'extend') for record in records_lod), respect_kd, 'extend')


    def assign_record(self, record: T_da) -> 'SqlDaf':                      # type: ignore[override]
        """
        Put one row in by its key, replacing the row that has that key, as `Daf.assign_record()` does.

        The whole row is replaced, in its place. Columns the dict lacks become Python NULL. A new
        key is added at the end. Use `update_record_krow()` to change only some cells.

        Args:
            record: The row, as a dict. It must have the keyfield.

        Returns:
            This SqlDaf, which has been changed.

        Raises:
            KeyError: The record lacks the keyfield.
            ValueError: There is no keyfield, or the key is a Python NULL.
        """
        key_col: str = self._need_keyfield('assign_record')
        if key_col not in record:
            raise KeyError(key_col)
        return self._write_rows([self._row_from_item(record, 'assign_record')], True, 'assign_record')


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
        root_sdaf: SqlDaf = self.parent_sdaf or self
        if root_sdaf.keyfield and root_sdaf.keyfield in cols:
            root_sdaf._check_no_null_key(method)


    def update_by_keylist(self, keylist: T_la | None = None, record: T_da | None = None) -> 'SqlDaf':     # type: ignore[override]
        """
        Change some cells in the rows that have the given keys, as `Daf.update_by_keylist()` does.

        Only the columns named in the dict change. A key that is not found is skipped. It is one
        UPDATE for all the keys. For one row, `update_record_krow()` raises KeyError for a missing key. On a selection, the parent's rows change, as a Daf selection
        shares its rows.

        Args:
            keylist: The keys of the rows to change.
            record: The new values, by column name. Names that are not columns are ignored.

        Returns:
            This SqlDaf, which has been changed. With no keyfield, keys or record, nothing happens.

        Raises:
            ValueError: The change gives a Python NULL key. Nothing changes.
            sqlite3.IntegrityError: The change gives a repeated key. Nothing changes.
        """
        cols_ls: T_ls = [col for col in (record or {}) if col in self.hd]
        if not self.keyfield or not keylist or not cols_ls:
            return self
        key_col: str = self._need_keyfield('update_by_keylist')
        keys_sql, keys_params = self._key_source_sql(keylist)
        table, _ref_col, scope_sql, scope_params = self._update_scope()
        set_sql: str = ', '.join(f'{quote_name(col)} = ?' for col in cols_ls)
        with self.transaction():
            self._execute(f'UPDATE {quote_name(table)} SET {set_sql} WHERE {scope_sql} AND '
                          f'{quote_name(key_col)} IN (SELECT daf_key FROM ({keys_sql}))',
                          tuple(null_to_none([record[col] for col in cols_ls])) + scope_params + keys_params)   # type: ignore[index]
            self._after_key_change(cols_ls, 'update_by_keylist')
        return self


    def update_record_krow(self, key: Any, record: T_da | None = None) -> 'SqlDaf':   # type: ignore[override]
        """
        Change some cells in the row that has this key, as `Daf.update_record_krow()` does.

        Only the columns named in the dict change. An int is always a key here, never a
        position, so it suits a keyfield of ints. It is one UPDATE. On a selection, the
        parent's row changes, and only a row of the selection is found.
        For many rows that take the same values, use `update_by_keylist()`.

        Args:
            key: The key of the row to change.
            record: The new values, by column name. Names that are not columns are ignored.

        Returns:
            This SqlDaf, which has been changed. A record of None changes nothing.

        Raises:
            KeyError: No row has this key.
            ValueError: There is no keyfield, or the change gives a Python NULL key. Nothing changes.
            sqlite3.IntegrityError: The change gives a repeated key. Nothing changes.

        Examples:
            >>> import tempfile
            >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
            >>> ids_sdaf = SqlDaf('ids', cols=['id', 'v'], lol=[[10, 'a'], [20, 'b']], keyfield='id')
            >>> ids_sdaf.to_daf()
            <BLANKLINE>
            | id | v  |
            | -: | -: |
            | 10 |  a |
            | 20 |  b |
            <BLANKLINE>
            %% daf rows=2; cols=2; keyfield='id'; name='ids'
            <BLANKLINE>
            >>> _ = ids_sdaf.update_record_krow(20, {'v': 'q'})
            >>> ids_sdaf.select_record(20)
            {'id': 20, 'v': 'q'}
        """
        if record is None:
            return self
        key_col: str = self._need_keyfield('update_record_krow')
        cols_ls: T_ls = [col for col in record if col in self.hd]
        if not cols_ls:
            self.select_record(key, silent_error=False)         # still raises KeyError for a missing key, as a Daf does.
            return self
        table, _ref_col, scope_sql, scope_params = self._update_scope()
        set_sql: str = ', '.join(f'{quote_name(col)} = ?' for col in cols_ls)
        with self.transaction():
            cursor: sqlite3.Cursor = self._execute(
                    f'UPDATE {quote_name(table)} SET {set_sql} WHERE {scope_sql} AND {quote_name(key_col)} = ?',
                    tuple(null_to_none([record[col] for col in cols_ls])) + scope_params + (key,))
            if cursor.rowcount == 0:
                raise KeyError(key)
            self._after_key_change(cols_ls, 'update_record_krow')
        return self


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


    def assign_col(self, colname: str, la: T_la | None = None, default: Any = NULL) -> 'SqlDaf':   # type: ignore[override]
        """
        Fill a column by name with the values of a list, or add it if it is new, as `Daf.assign_col()` does.

        The values go to the rows in order. A short list is filled out with `default`, and a long
        one is cut. With no list, every row gets `default`. A new column is added at the end, with
        ALTER TABLE, on a parent only.

        Args:
            colname: The column name.
            la: The values, in row order.
            default: The value for rows that the list does not reach.

        Returns:
            This SqlDaf, which has been changed.

        Raises:
            NotImplementedError: The column is new, and this is a selection.
            ValueError: The change gives a Python NULL key. Nothing changes.
            sqlite3.IntegrityError: The change gives a repeated key. Nothing changes.
        """
        if colname not in self.hd:
            if self.parent_sdaf is not None:
                raise NotImplementedError(f"assign_col: '{colname}' is a new column, and '{self.name}' is a selection. "
                                          "Add the column to its parent.")
            self._resolve()
            with self.transaction():
                self._execute(f'ALTER TABLE {quote_name(self.name)} ADD COLUMN {quote_name(colname)}')
                self.hd[colname] = len(self.hd)
            if not la and default is NULL:
                return self                             # ALTER TABLE fills a new column with SQL NULL already.
        self._assign_col_values(colname, None if la is None else list(la), keep_rest=False, default=default)
        return self


    def apply_to_col(self, col: str, func: Callable, **kwargs: Any) -> 'SqlDaf':     # type: ignore[override]
        """
        Replace each value of a column by the result of a function, in place, as `Daf.apply_to_col()` does.

        It is one UPDATE, with the function registered in SQLite. The function gets Python NULL
        for SQL NULL. An error it raises is raised as itself, and nothing changes.

        Args:
            col: The column.
            func: Called as func(value, **kwargs). It returns the new value.
            **kwargs: Passed on to func.

        Returns:
            This SqlDaf, which has been changed.

        Raises:
            ColumnNotFoundError: The name is not a column.
            ValueError: The change gives a Python NULL key. Nothing changes.
            sqlite3.IntegrityError: The change gives a repeated key. Nothing changes.
        """
        if col not in self.hd:
            raise ColumnNotFoundError(col)

        def call(val: Any) -> Any:
            result: Any = func(NULL if val is None else val, **kwargs)
            return None if result is NULL else result

        func_name: str = self._register_function(call, 1)
        table, _ref_col, scope_sql, scope_params = self._update_scope()
        with self.transaction():
            self._execute(f'UPDATE {quote_name(table)} SET {quote_name(col)} = {func_name}({quote_name(col)}) '
                          f'WHERE {scope_sql}', scope_params)
            self._after_key_change([col], 'apply_to_col')
        return self


    def apply_in_place(                                                     # type: ignore[override]
            self,
            func:       Callable[..., Any],         # called as func(row, **kwargs).
            by:         str = 'row',                # 'row' or 'row_klist'.
            rowkeys:    T_la | None = None,         # the keys of the rows to change. All if None.
            **kwargs:   Any,                        # passed on to func.
            ) -> 'SqlDaf':
        """
        Apply a function to each row and store the results, as `Daf.apply_in_place()` does.

        With `by='row'`, the function gets each row as a dict and returns a dict. Its values are
        written back by column name. A column the dict lacks keeps its value. With
        `by='row_klist'`, the function gets a [KeyedList][daffodil.keyedlist.KeyedList], changes
        it, and returns nothing. The whole row is written back.

        Rows are read and written in batches, in one transaction. If the function raises, nothing
        changes. With `rowkeys`, only those rows change, and a key that is not found is skipped,
        as in a Daf. On a selection, the parent's rows change.

        Args:
            func: The function to apply to each row.
            by: 'row' or 'row_klist'.
            rowkeys: The keys of the rows to change. All rows if None.
            **kwargs: Passed on to func.

        Returns:
            This SqlDaf, which has been changed.

        Raises:
            NotImplementedError: `by` is not 'row' or 'row_klist'.
            ValueError: With `by='row'` the function returned None, or a key became a Python NULL.
            sqlite3.IntegrityError: A key became repeated. Nothing changes.
        """
        if by not in ('row', 'row_klist'):
            raise NotImplementedError(f"apply_in_place(by={by!r}) is not supported on a SqlDaf.")
        if rowkeys is not None:
            self.select_records_daf(rowkeys, silent_error=True).apply_in_place(func, by=by, **kwargs)
            return self

        table, ref_col, _scope_sql, _scope_params = self._update_scope()
        rows_sql, rows_params = self._rows_sql()
        cols_ls: T_ls = list(self.hd)
        written_set: set = set()
        last_ord: Any = None
        with self.transaction():
            # Read a batch past the last place seen, change it, write it, and go on.
            while True:
                after_sql: str = '' if last_ord is None else f'WHERE {ORD_COL} > ? '
                batch_lot: List[Tuple[Any, ...]] = self._fetch(self._execute(
                        f'SELECT {REF_COL}, {ORD_COL}, {self._cols_sql()} FROM ({rows_sql}) {after_sql}'
                        f'ORDER BY {ORD_COL} LIMIT 10000', rows_params + (() if last_ord is None else (last_ord,))))
                if not batch_lot:
                    break
                last_ord = batch_lot[-1][1]
                updates_dolt: Dict[Tuple[str, ...], List[Tuple[Any, ...]]] = {}
                for row_t in batch_lot:
                    new_da: T_da = self._apply_one(func, by, cols_ls, none_to_null(row_t[2:]), kwargs)
                    set_cols_t: Tuple[str, ...] = tuple(col for col in new_da if col in self.hd)
                    updates_dolt.setdefault(set_cols_t, []).append(
                            tuple(null_to_none([new_da[col] for col in set_cols_t])) + (row_t[0],))
                set_cols_t2: Tuple[str, ...]
                for set_cols_t2, params_lt in updates_dolt.items():
                    if not set_cols_t2:
                        continue
                    written_set.update(set_cols_t2)
                    set_sql: str = ', '.join(f'{quote_name(col)} = ?' for col in set_cols_t2)
                    self.conn.executemany(f'UPDATE {quote_name(table)} SET {set_sql} WHERE {quote_name(ref_col)} = ?', params_lt)
            self._after_key_change(written_set, 'apply_in_place')
        return self


    @staticmethod
    def _apply_one(func: Callable[..., Any], by: str, cols_ls: T_ls, row_la: T_la, kwargs: T_da) -> T_da:
        """ Run the function of `apply_in_place()` on one row, and give the values to write. Internal use. """
        if by == 'row_klist':
            row_kl: KeyedList = KeyedList(dict(zip(cols_ls, range(len(cols_ls)))), row_la)
            func(row_kl, **kwargs)
            return cast(T_da, row_kl.to_dict())
        new_da: Any = func(dict(zip(cols_ls, row_la)), **kwargs)
        if new_da is None:
            raise ValueError("apply_in_place: func must return a row for by='row'. "
                             "None is only valid for by='row_klist'.")
        return dict(new_da)


    def calc_cols(                                                          # type: ignore[override]
            self,
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


    # ---- column names and layout ----

    def _check_layout_change(self, method: str) -> None:
        """ Raise NotImplementedError on a child, whose columns are its parent's. Resolve a join not read yet. Internal use. """
        if self.parent_sdaf is not None:
            raise NotImplementedError(f"{method}: '{self.name}' is a selection, and its columns are its parent's. "
                                      "Change the parent, or make the selection a table first, with copy(inplace=True).")
        self._resolve()


    def insert_col(                                                         # type: ignore[override]
            self,
            colname:    str,                        # the name of the column.
            col_la:     T_la | None = None,         # the values, in row order.
            icol:       int = -1,                   # the position. Only the end: -1, or the number of columns.
            default:    Any = NULL,                 # the value for rows that the list does not reach.
            ) -> 'SqlDaf':
        """
        Add a named column, or fill it if the name is there, as `Daf.insert_col()` does.

        A new column goes at the end, since ALTER TABLE adds columns only there.
        Any other position raises NotImplementedError.
        To read the columns in another order, use `select_cols()`. It gives new rows, in the order you name.

        A list that is too short is filled out with `default`.
        With no list, every row gets `default`.
        If the name is there, its values are replaced, and `icol` is ignored.
        An empty name does nothing.
        A new column is added on a parent only, as `assign_col()` adds it.

        Args:
            colname: The name of the column.
            col_la: The values, in row order.
            icol: The position. Only the end is supported: -1, or the number of columns.
            default: The value for rows that the list does not reach.

        Returns:
            This SqlDaf, which has been changed.

        Raises:
            NotImplementedError: A new column at a position other than the end, or on a selection.

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
            >>> marks_sdaf.insert_col('w', ['x']).to_daf().lol
            [['b1', 1, 'x'], ['b2', 0, '']]
        """
        if not colname:
            return self
        if colname not in self.hd and icol not in (-1, len(self.hd)):
            raise NotImplementedError(f"insert_col: a new column goes at the end of a SqlDaf, not at position {icol}. "
                                      "Use icol=-1, and select_cols() to read the columns in another order.")
        return self.assign_col(colname, col_la or [], default)


    def drop_cols(self, exclude_cols: T_ls | None = None) -> 'SqlDaf':     # type: ignore[override]
        """
        Remove columns from the table, in place, as `Daf.drop_cols()` does.

        Each column goes with ALTER TABLE DROP COLUMN, which rewrites the table.
        On 1,000,000 rows of 6 columns, one drop took 0.73 s, twice the time of `copy()`.
        A name that is not a column is ignored. With no names, nothing happens.
        The dtypes lose the columns dropped.

        If the keyfield is dropped, it is cleared, as in a Daf, and its index goes with it.
        Set a keyfield again with `set_keyfield()`.

        A selection has its parent's columns, so a drop on one raises NotImplementedError.
        A selection made before the drop still names the dropped column.
        Reading it then raises sqlite3.OperationalError, so make it again.

        Args:
            exclude_cols: The names of the columns to remove.

        Returns:
            This SqlDaf, which has been changed.

        Raises:
            NotImplementedError: This is a selection.
            sqlite3.OperationalError: An index of your own is on a column dropped. Nothing changes.

        Examples:
            >>> import tempfile
            >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
            >>> marks_sdaf = SqlDaf('marks', cols=['ballot_id', 'v', 'n'], lol=[['b1', 1, 10]], keyfield='ballot_id')
            >>> marks_sdaf.to_daf()
            <BLANKLINE>
            | ballot_id | v  | n  |
            | --------: | -: | -: |
            |        b1 |  1 | 10 |
            <BLANKLINE>
            %% daf rows=1; cols=3; keyfield='ballot_id'; name='marks'
            <BLANKLINE>
            >>> marks_sdaf.drop_cols(['v']).columns()
            ['ballot_id', 'n']
        """
        if not exclude_cols:
            return self
        self._check_layout_change('drop_cols')
        drop_ls: T_ls = [col for col in self.hd if col in exclude_cols]
        if not drop_ls:
            return self
        kept_ls: T_ls = [col for col in self.hd if col not in drop_ls]
        col: str
        with self.transaction():
            if self.keyfield in drop_ls:
                self._execute(f'DROP INDEX IF EXISTS {quote_name(self.key_index_name())}')
            self._drop_group_indexes(drop_ls)
            for col in drop_ls:
                self._execute(f'ALTER TABLE {quote_name(self.name)} DROP COLUMN {quote_name(col)}')
            self.hd = {col: idx for idx, col in enumerate(kept_ls)}
            self.dtypes = {col: typ for col, typ in self.dtypes.items() if col in self.hd}
            if self.keyfield in drop_ls:
                self.keyfield = ''
            self._meta_update()
        return self


    def rename_cols(self, from_to_dict: T_ds) -> 'SqlDaf':                 # type: ignore[override]
        """
        Rename columns in place, as `Daf.rename_cols()` does.

        Each is ALTER TABLE RENAME COLUMN, which changes only the schema.
        On 1,000,000 rows it took 15 ms.
        Names that are not in the mapping stay as they are.
        Names in the mapping that are not columns are ignored.
        Names may be swapped, as in `{'a': 'b', 'b': 'a'}`.
        The dtypes are renamed too.

        The keyfield follows the rename, as in a Daf.
        Its index is made again under the new name, the next time a key is used.

        A Daf lets two columns get the same name. A SqlDaf raises then, and nothing changes.
        A selection has its parent's columns, so a rename on one raises NotImplementedError.
        A selection made before the rename still names the old columns.
        Reading it then raises sqlite3.OperationalError, so make it again.

        Args:
            from_to_dict: Maps old names to new names.

        Returns:
            This SqlDaf, which has been changed.

        Raises:
            NotImplementedError: This is a selection.
            ValueError: A new name is reserved by SqlDaf. Nothing changes.
            sqlite3.OperationalError: Two columns would have the same name. Nothing changes.

        Examples:
            >>> import tempfile
            >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
            >>> marks_sdaf = SqlDaf('marks', cols=['id', 'v'], lol=[['b1', 1]], keyfield='id')
            >>> marks_sdaf.to_daf()
            <BLANKLINE>
            | id | v  |
            | -: | -: |
            | b1 |  1 |
            <BLANKLINE>
            %% daf rows=1; cols=2; keyfield='id'; name='marks'
            <BLANKLINE>
            >>> marks_sdaf.rename_cols({'id': 'ballot_id'}).columns(), marks_sdaf.keyfield
            (['ballot_id', 'v'], 'ballot_id')
        """
        self._check_layout_change('rename_cols')
        self._rename_by_map({old: new for old, new in from_to_dict.items() if old in self.hd and new != old})
        return self


    def _rename_by_map(self, old_to_new: T_ds) -> None:
        """
        Rename columns in the table, the dtypes, the keyfield and the record. Internal use.

        Each old name goes to a temporary name first, and then to its new name. So a swap works.
        """
        if not old_to_new:
            return
        reserved_ls: T_ls = [new for new in old_to_new.values() if new in RESERVED_COLS]
        if reserved_ls:
            raise ValueError(f"SqlDaf: {reserved_ls} reserved, and cannot be column names: {list(RESERVED_COLS)}.")
        table_sql: str = quote_name(self.name)
        idx: int
        old: str
        new: str
        with self.transaction():
            if self.keyfield in old_to_new:
                self._execute(f'DROP INDEX IF EXISTS {quote_name(self.key_index_name())}')
            self._drop_group_indexes(list(old_to_new))
            for idx, old in enumerate(old_to_new):
                self._execute(f'ALTER TABLE {table_sql} RENAME COLUMN {quote_name(old)} TO {quote_name(f"daf_tmp_col_{idx}")}')
            for idx, new in enumerate(old_to_new.values()):
                self._execute(f'ALTER TABLE {table_sql} RENAME COLUMN {quote_name(f"daf_tmp_col_{idx}")} TO {quote_name(new)}')
            self.hd = {old_to_new.get(col, col): idx for col, idx in self.hd.items()}
            self.dtypes = {old_to_new.get(col, col): typ for col, typ in self.dtypes.items()}
            self.keyfield = old_to_new.get(self.keyfield, self.keyfield)
            self._meta_update()


    def set_cols(                                                           # type: ignore[override]
            self,
            new_cols:       T_ls | None = None,     # the names, in order. Spreadsheet names if None.
            sanitize_cols:  bool = True,            # if True, make the names unique and not empty.
            unnamed_prefix: str = 'col',            # the start of a name made for an empty one.
            ) -> 'SqlDaf':
        """
        Set the column names by position, in place, as `Daf.set_cols()` does.

        The names are made as in a Daf.
        Without a list, they are A, B, C and so on.
        With `sanitize_cols`, a repeated name gets a suffix, and an empty name becomes the prefix and its position.
        Each name that changes is renamed as `rename_cols()` renames it, so the rows stay as they are.
        The keyfield and the dtypes follow the names.

        Args:
            new_cols: The names, in order. If None, spreadsheet names are made.
            sanitize_cols: If True, make the names unique and not empty.
            unnamed_prefix: The start of a name made for an empty one.

        Returns:
            This SqlDaf, which has been changed.

        Raises:
            AttributeError: The number of names is not the number of columns, as in a Daf.
            NotImplementedError: This is a selection.

        Examples:
            >>> import tempfile
            >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
            >>> SqlDaf('t', cols=['a', 'b', 'c'], lol=[[1, 2, 3]]).set_cols(['x', 'x', '']).columns()
            ['x', 'x_1', 'col2']
        """
        self._check_layout_change('set_cols')
        if new_cols is None:
            new_cols = daf_utils._generate_spreadsheet_column_names_list(len(self.hd))
        elif sanitize_cols:
            new_cols = daf_utils._sanitize_cols(new_cols, unnamed_prefix=unnamed_prefix)
        if len(new_cols) != len(self.hd):
            raise AttributeError("Length of new_cols not the same as existing cols")
        self._rename_by_map({old: new for old, new in zip(self.hd, new_cols) if old != new})
        return self


    def set_dtypes(                                                         # type: ignore[override]
            self,
            default_type:       Type = str,                     # the type of every column not listed.
            typ_to_cols_dict:   Dict[Type, T_ls] | None = None, # maps a type to the names of its columns.
            ) -> 'SqlDaf':
        """
        Set the dtype of every column, from a default and a few exceptions, as `Daf.set_dtypes()` does.

        No value is converted. `apply_dtypes()` does that.
        On a permanent table with a record, the record is changed too, as `set_keyfield()` changes it.
        The record can store these dtypes: int, float, bool, str, list and dict.

        Args:
            default_type: The type of every column that is not listed.
            typ_to_cols_dict: Maps a type to the names of the columns of that type.

        Returns:
            This SqlDaf, which has been changed.

        Raises:
            NotImplementedError: A type the record cannot hold, on a table with a record. Nothing changes.

        Examples:
            >>> import tempfile
            >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
            >>> SqlDaf('t', cols=['a', 'b'], lol=[['1', 'x']]).set_dtypes(str, {int: ['a']}).dtypes
            {'a': <class 'int'>, 'b': <class 'str'>}
        """
        old_dtypes: T_dtype_dict = self.dtypes
        Daf.set_dtypes(self, default_type, typ_to_cols_dict)
        try:
            with self.transaction():
                self._meta_update()
        except NotImplementedError:
            self.dtypes = old_dtypes
            raise
        return self


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
        """
        cols_ls: T_ls = list(self.hd) if not cols else ([cols] if isinstance(cols, str) else list(cols))
        if cols_ls != list(self.hd) and (len(set(cols_ls)) != len(cols_ls) or NULL in cols_ls):
            cols_ls = daf_utils._sanitize_cols(cols=cols_ls)
        root_sdaf: SqlDaf = self.parent_sdaf or self
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


    # ---- sorting ----

    def sort_by_colname(                                                    # type: ignore[override]
            self,
            colname:            str,                # the column to sort by.
            *,
            reverse:            bool = False,       # if True, sort from high to low.
            length_priority:    bool = False,       # if True, sort by length first, then by value.
            as_str:             bool = False,       # if True, sort by the text of each value.
            ) -> 'SqlDaf':
        """
        Sort the rows by one column, in place, as `Daf.sort_by_colname()` does.

        See `sort_by_colnames()` for how the rows are sorted and where they go.

        Args:
            colname: The column to sort by.
            reverse: If True, sort from high to low.
            length_priority: If True, sort by length first, then by value.
            as_str: If True, sort by the text of each value, and Python NULL sorts as ''.

        Returns:
            This SqlDaf, now a sorted TEMP table.

        Raises:
            KeyError: The column name is not found.

        Examples:
            >>> import tempfile
            >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
            >>> SqlDaf('t', cols=['a'], lol=[['10'], ['99'], ['8']]).sort_by_colname('a', length_priority=True).col('a')
            ['8', '10', '99']
        """
        return self.sort_by_colnames([colname], reverse=reverse, length_priority=length_priority, as_str=as_str)


    def sort_by_colnames(                                                   # type: ignore[override]
            self,
            colnames:           T_ls,               # the columns to sort by, main key first.
            reverse:            bool = False,       # if True, sort from high to low.
            length_priority:    bool = False,       # if True, sort by length first, then by value.
            as_str:             bool = False,       # if True, sort by the text of each value.
            ) -> 'SqlDaf':
        """
        Sort the rows by several columns, in place, as `Daf.sort_by_colnames()` does.

        The first column is the main sort key. The others break ties.
        Rows that tie on every column keep their order, as Python's sort keeps them.

        Where the sorted rows go:

        The rows are copied in sorted order into a TEMP table, and this SqlDaf becomes it.
        This is `copy(inplace=True)` with an ORDER BY, in one statement.
        The table it came from is not changed, so its other selections stay as they were.
        A Daf selection sorts its own list of rows, and leaves its parent alone too.
        On a permanent table, the table in the file keeps its order. Use `save()` to keep the sorted copy.
        On 1,000,000 rows of 6 columns, a sort by one text column took 2.4 s, against 1.0 s for a Daf.

        How values compare:

        - Python NULL, SQL NULL or '', sorts first, as an empty cell sorts first in a Daf.
        - Text compares by code point, as Python compares it.
        - `length_priority` sorts by the length of the text first, then by the text. That gives
          numeric order for whole numbers held as text, when none is negative or has a leading zero.
          It is not a numeric sort for negative numbers or decimals.
        - `as_str` sorts by the text of each value. A float such as 1e20 is written differently by SQLite.
        - A column that mixes numbers and text sorts every number before all text.
          A Daf raises TypeError on such a column instead.

        Args:
            colnames: The columns to sort by, main key first.
            reverse: If True, sort from high to low.
            length_priority: If True, sort by length first, then by value.
            as_str: If True, sort by the text of each value, and Python NULL sorts as ''.

        Returns:
            This SqlDaf, now a sorted TEMP table.

        Raises:
            KeyError: A column name is not found.

        Examples:
            >>> import tempfile
            >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
            >>> SqlDaf('t', cols=['p', 'q'], lol=[[2, 'b'], [1, 'z'], [1, 'a']]).sort_by_colnames(['p', 'q']).to_daf().lol
            [[1, 'a'], [1, 'z'], [2, 'b']]
        """
        col: str
        for col in colnames:
            if col not in self.hd:
                raise KeyError(col)
        self._resolve_parent()

        # One ORDER BY term for each sort key, then the row order, to keep ties in place.
        direction: str = ' DESC' if reverse else ''
        order_ls: T_ls = []
        for col in colnames:
            val_sql: str = (f"COALESCE(CAST({quote_name(col)} AS TEXT), '')" if as_str
                            else f"NULLIF({quote_name(col)}, '')")
            if length_priority:
                order_ls.append(f'length({val_sql}){direction}')
            order_ls.append(f'{val_sql}{direction}')
        order_ls.append(ORD_COL)

        root_sdaf: SqlDaf = self.parent_sdaf or self
        target: str = self._next_temp_name(root_sdaf.name)
        table_sql: str = _create_table_sql(self.conn, target, list(self.hd), 'temp')
        rows_sql, rows_params = self._rows_sql()
        with self.transaction():
            self._execute(table_sql)
            self._execute(f'INSERT INTO {quote_name(target)} ({self._cols_sql()}) '
                          f'SELECT {self._cols_sql()} FROM ({rows_sql}) ORDER BY {", ".join(order_ls)}', rows_params)
        self._name          = target
        self.has_rid        = True
        self.is_temp        = True
        self.parent_sdaf    = None
        self.is_resolved    = True
        self.pending_sql    = ''
        self.pending_params = ()
        return self


    # ---- joining ----

    def join(                                                               # type: ignore[override]
            self,
            other_daf:              'SqlDaf',           # the SqlDaf to join with, in the same file.
            how:                    str = 'inner',      # 'inner', 'left', 'right' or 'outer'.
            shared_fields:          T_ls | None = None, # columns that both have and that appear only once.
            tag_other:              bool = False,       # if True, every column of the other gets a suffix.
            custom_translator_daf:  Daf | None = None,  # not supported yet.
            diagnose:               bool = False,       # not used.
            name:                   str = '',           # the name of the result. Generated if empty.
            fill:                   Any = NULL,         # only Python NULL is supported.
            ) -> 'SqlDaf':
        """
        Join two SqlDaf tables on their keyfields, as `Daf.join()` does, in SQL.

        The columns are named exactly as `Daf.join()` names them, by the same translator: a
        column that both have gets the name of its source as a suffix, such as `name_audit`,
        unless it is in `shared_fields`, and `tag_other` suffixes every column of the other. The
        names are those of the parent tables, or `daf1` and `daf2` if they are the same. The rows
        come as in `Daf.join()`: in this table's order, and for `right` and `outer`, the other
        table's unmatched rows after, in its order. A cell with no match is Python NULL. The keyfield of
        the result is this table's.

        The keyfields must be unique, as in `Daf.join()`. A repeated key is not checked, and the
        result is then undefined. Check the keys first if they may repeat. A Python NULL key is
        undefined too: see SqlDaf.

        A row found only in the other table, in a right or outer join, takes its key and its
        shared fields from the other table, as in `Daf.join()`.

        Nothing runs yet. The join is resolved the first time it is read, into a TEMP table of
        new rows, as `Daf.join()` makes new rows. A selection made from it resolves it first.

        Args:
            other_daf: The SqlDaf to join with, in the same database file.
            how: `inner`, `left`, `right` or `outer`.
            shared_fields: Columns that both have and that appear only once.
            tag_other: If True, every column of the other, except shared ones, gets a suffix.
            custom_translator_daf: Not supported on a SqlDaf yet.
            diagnose: Not used.
            name: The name of the result table. Generated if empty.
            fill: The value for cells that have no match. Only Python NULL is supported.

        Returns:
            A SqlDaf of the joined rows, resolved when first read.

        Raises:
            ValueError: `how` is not one of the four, or a table has no keyfield.
            NotImplementedError: `custom_translator_daf` or a `fill` other than Python NULL, or a SqlDaf this connection cannot read.

        Examples:
            >>> import tempfile
            >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
            >>> a_sdaf = SqlDaf.from_daf(Daf(lol=[[1, 'Alice'], [2, 'Bob']], cols=['id', 'name'], keyfield='id'), name='a')
            >>> a_sdaf.to_daf()
            <BLANKLINE>
            | id | name  |
            | -: | ----: |
            |  1 | Alice |
            |  2 |   Bob |
            <BLANKLINE>
            %% daf rows=2; cols=2; keyfield='id'; name='a'
            <BLANKLINE>
            >>> b_sdaf = SqlDaf.from_daf(Daf(lol=[[1, 50], [3, 70]], cols=['id', 'salary'], keyfield='id'), name='b')
            >>> b_sdaf.to_daf()
            <BLANKLINE>
            | id | salary |
            | -: | -----: |
            |  1 |     50 |
            |  3 |     70 |
            <BLANKLINE>
            %% daf rows=2; cols=2; keyfield='id'; name='b'
            <BLANKLINE>
            >>> a_sdaf.join(b_sdaf, how='left').to_daf().lol
            [[1, 'Alice', 50], [2, 'Bob', '']]
        """
        if how not in ('inner', 'left', 'right', 'outer'):
            raise ValueError(f"join: how='{how}' is not inner, left, right or outer.")
        if custom_translator_daf is not None:
            raise NotImplementedError("join(custom_translator_daf=) is not supported on a SqlDaf yet.")
        if fill is not NULL:
            raise NotImplementedError("join(fill=) other than Python NULL is not supported on a SqlDaf yet.")
        if not isinstance(other_daf, SqlDaf):
            raise NotImplementedError("join: the other must be a SqlDaf. Make one with SqlDaf.from_daf().")
        self._check_same_db(other_daf, 'join')
        if not self.keyfield or not other_daf.keyfield:
            raise ValueError("join: both tables need a keyfield.")

        a_name: str = (self.parent_sdaf or self).name
        b_name: str = (other_daf.parent_sdaf or other_daf).name
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
        b_sql, b_params = other_daf._rows_sql()
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
        return join_sdaf


    def select_cols(                                                        # type: ignore[override]
            self,
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
        root_name: str = (self.parent_sdaf or self).name
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
            >>> _ = marks_sdaf.sql("UPDATE marks SET v = 9")
            >>> sel_sdaf.to_daf().lol
            [['b1', 1]]
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

        root_sdaf: SqlDaf = self.parent_sdaf or self
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


    def save(self, name: str = '') -> 'SqlDaf':
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

        A name already in use raises ValueError. To replace a table with a sorted or changed copy,
        delete the old table first, as in `SqlDaf('items').delete()`, then save the copy under its name.

        Args:
            name: The name to keep it under. This SqlDaf's name if empty.

        Returns:
            This SqlDaf, as a permanent table.

        Raises:
            ValueError: This is a child, the name is a generated one, or a table of that name exists.

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
        """
        if self.parent_sdaf is not None:
            raise ValueError(f"SqlDaf.save: '{self.name}' is a child, with no rows of its own. "
                             f"Use copy(inplace=True) first.")
        self._resolve()                                 # a join not read yet.
        target: str = name or self._name
        if not name and self._name.startswith(TEMP_PREFIX):
            raise ValueError(f"SqlDaf.save: '{self._name}' is a generated name. Give the name to save it under.")

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


    def attach_schema(self, schema: type) -> 'SqlDaf':                     # type: ignore[override]
        """
        Attach a `@schemaclass` to this SqlDaf, as `Daf.attach_schema()` does.

        It stores the schema, then fills in the dtypes and the keyfield, each only if there is none.
        No value is converted, and nothing is checked against the rows.
        Call `apply_dtypes()` to convert.
        On a permanent table with a record, the record is changed too, as `set_keyfield()` changes it.

        A Daf with no columns takes them from the schema.
        A SqlDaf gets its columns when its table is made, so one with no columns raises NotImplementedError.

        Args:
            schema: A class decorated with `@schemaclass`.

        Returns:
            This SqlDaf, which has been changed.

        Raises:
            TypeError: The class is not a `@schemaclass`.
            KeyError: The schema's keyfield is not a column.
            NotImplementedError: This SqlDaf has no columns, or a dtype the record cannot hold.

        Examples:
            >>> import tempfile
            >>> from daffodil.lib.schemaclass import schemaclass
            >>> @schemaclass
            ... class Mark:
            ...     __keyfield__ = 'ballot_id'
            ...     ballot_id: str = ''
            ...     v: int = 0
            >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
            >>> marks_sdaf = SqlDaf('marks', cols=['ballot_id', 'v'], lol=[['b1', '1']]).attach_schema(Mark)
            >>> marks_sdaf.to_daf()
            <BLANKLINE>
            | ballot_id | v  |
            | --------: | -: |
            |        b1 |  1 |
            <BLANKLINE>
            %% daf rows=1; cols=2; keyfield='ballot_id'; name='marks'
            <BLANKLINE>
            >>> marks_sdaf.keyfield, marks_sdaf.dtypes
            ('ballot_id', {'ballot_id': <class 'str'>, 'v': <class 'int'>})
        """
        if not getattr(schema, '__is_schemaclass__', False):
            raise TypeError("schema must be a @schemaclass")
        if not self.hd:
            raise NotImplementedError(f"attach_schema: '{self.name}' has no columns. Make the table with its columns first.")
        keyfield: str = self.keyfield or getattr(schema, '__keyfield__', '')
        if keyfield and keyfield not in self.hd:
            raise KeyError(keyfield)
        old_dtypes: T_dtype_dict = self.dtypes
        old_keyfield: str = self.keyfield
        if not self.dtypes:
            self.dtypes = schema.get_dtypes_dict(use_origins=True)                    # type: ignore[attr-defined]
        self.keyfield = keyfield
        try:
            with self.transaction():
                self._meta_update()
        except NotImplementedError:
            self.dtypes, self.keyfield = old_dtypes, old_keyfield
            raise
        self.schema = schema
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
        - A str column is not touched.

        Conversion does not check that every value got the requested type. A converted column
        can still hold some text. `silent_error` does not change that. It is about a column that
        has no dtype, not about a value that does not convert.

        All the columns are converted in one pass over the table, in one transaction.
        On 1,000,000 rows, two int columns took 1.8 s.

        Only int, float, bool and str can be converted yet. A list or a dict column stays as the text
        of the file, as Python writes it. `reduce(by='sparse_row')` reads that text as it goes.

        On a selection, the rows are first copied into a TEMP table of its own, as
        `copy(inplace=True)` does, and the copy is converted. The parent is not changed. A Daf
        selection shares its rows, so there the parent's rows are converted too. Converting is
        costly, so convert a selection rather than a large parent that is not needed otherwise.

        Args:
            dtypes: The types to convert to, by column. It may name columns that are not there. This SqlDaf's dtypes if None.
            cols: Convert only these columns, or this one column. All the columns of the dtypes if None.
            silent_error: If False, a column to convert that has no dtype raises ValueError. If True, it is left as it is.

        Returns:
            This SqlDaf, whose table has been changed.

        Raises:
            KeyError: A name in `cols` is not a column.
            ValueError: A column to convert has no dtype, and `silent_error` is False.
            NotImplementedError: A dtype is not int, float, bool or str.
        """
        if dtypes:
            self.dtypes = {**self.dtypes, **dtypes}
        source_dtypes: T_dtype_dict = dtypes or self.dtypes
        if not source_dtypes:
            return self
        if dtypes:
            with self.transaction():
                self._meta_update()

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
            raise NotImplementedError(f"apply_dtypes() cannot store {[convert_dtypes[col] for col in unstorable_ls]} "
                                       f"for the columns {unstorable_ls} yet. Only int, float, bool and str.")
        if not convert_dtypes:
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
            self._meta_update()
        return self


    def sql(self, stmt: str, params: Iterable[Any] = ()) -> Daf:
        """
        Run a SQL statement of your own, and return its result as a Daf.

        This is the fallback for anything SqlDaf does not do yet. The statement runs on this
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
        cursor: sqlite3.Cursor = self.conn.execute(stmt, null_to_none(params))
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
