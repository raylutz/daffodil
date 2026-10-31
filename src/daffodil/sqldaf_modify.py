# sqldaf_modify.py
# copyright (c) 2026 Ray Lutz
"""
Methods of SqlDaf for modifying.

Each function here takes the table as its first argument, or the class for a classmethod.
It is assigned in the SqlDaf class body under its name, without the underscore.
The docs show it under that name.

Some functions use names defined in sqldaf.py, such as SqlDaf, when they run.
sqldaf.py sets those names here, as listed in _NAMES_FROM_SQLDAF, once they exist.
An import at the top would be circular, since sqldaf.py imports this module.
"""


from typing import Any, Callable, Dict, List, Tuple, Type, cast, TYPE_CHECKING
from collections.abc import Iterable
from daffodil.daf import Daf, NULL, ColumnNotFoundError, KeysDisabledError
from daffodil.keyedlist import KeyedList
from daffodil.lib.daf_types import T_ls, T_la, T_lola, T_da, T_ds, T_dtype_dict
import daffodil.lib.daf_utils as daf_utils
from daffodil.lib.daf_sql import sqlite3

if TYPE_CHECKING:       # for mypy. sqldaf.py sets the names below when the program runs.
    from daffodil.sqldaf import SqlDaf, GROUP_INDEX_PREFIX, ORD_COL, REF_COL, RESERVED_COLS, _create_table_sql, _indexes_of, none_to_null, null_to_none, placeholders, quote_name

# The names of sqldaf.py that the functions here use when they run.
# sqldaf.py sets each one here once it exists. An import at the top would be circular.
_NAMES_FROM_SQLDAF: Tuple[str, ...] = ('SqlDaf', 'GROUP_INDEX_PREFIX', 'ORD_COL', 'REF_COL', 'RESERVED_COLS', '_create_table_sql', '_indexes_of', 'none_to_null', 'null_to_none', 'placeholders', 'quote_name')


def _drop_group_indexes(self: 'SqlDaf', cols: T_ls) -> None:
    """ Drop the indexes groupby() made that use any of these columns, before they are dropped or renamed. Internal use. """
    index_name: str
    index_cols: T_ls
    for index_name, index_cols in _indexes_of(self.conn, self.name).items():
        if index_name.startswith(GROUP_INDEX_PREFIX) and any(col in cols for col in index_cols):
            self._execute(f'DROP INDEX {quote_name(index_name)}')


def _delete_records(
        self: 'SqlDaf',
        keys_ls:        Any,                        # the keys of the rows to remove.
        inplace:        bool = False,               # True deletes from this SqlDaf. False makes a selection.
        silent_error:   bool = False,               # if True, keys that are not found are ignored.
        ) -> 'SqlDaf':
    """
    Remove the rows with these keys, as [`Daf.delete_records()`][daffodil.daf_modify._delete_records] does. A Python NULL key gives
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

    Examples:
        Create a small SqlDaf table of people:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> people_sdaf = SqlDaf('people', cols=['id', 'name', 'age'], keyfield='id', lol=[
        ...     ['p1', 'Ana Lopez', 34], ['p2', 'Ben Okafor', 52], ['p3', 'Chen Wu', 27]])

        By default, get a selection without the rows. The table keeps them:

        >>> people_sdaf.delete_records(['p2']).keys()
        ['p1', 'p3']
        >>> people_sdaf.keys()
        ['p1', 'p2', 'p3']

        A key that is not found raises, unless `silent_error=True`:

        >>> people_sdaf.delete_records(['p9'])
        Traceback (most recent call last):
        KeyError: 'p9'
        >>> people_sdaf.delete_records(['p9', 'p3'], silent_error=True).keys()
        ['p1', 'p2']

        Delete the rows from the table itself, with `inplace=True`:

        >>> people_sdaf.delete_records(['p2'], inplace=True).keys()
        ['p1', 'p3']
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
        self._execute(self._refs_table_sql(new_name))
        self._execute(
                f'INSERT INTO {quote_name(new_name)} (ref) SELECT r.{REF_COL} FROM ({rows_sql}) AS r '
                f'WHERE r.{quote_name(key_col)} NOT IN (SELECT daf_key FROM ({keys_sql}) WHERE daf_key IS NOT NULL) '
                f'OR r.{quote_name(key_col)} IS NULL ORDER BY r.{ORD_COL}',
                rows_params + keys_params)
        self._execute(f'DROP TABLE {quote_name(self.name)}')
        self._execute(f'ALTER TABLE {quote_name(new_name)} RENAME TO {quote_name(self.name)}')
    return self


def _remove_dups(self: 'SqlDaf', keyfield: str = '') -> Tuple['SqlDaf', 'SqlDaf']:
    """
    Split the rows into those with a unique key and those with a repeated key, as [`Daf.remove_dups()`][daffodil.daf_modify._remove_dups] does.

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
        Create a small SqlDaf table of sign-ups, where one email appears twice:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> signups_sdaf = SqlDaf('signups', cols=['email', 'name'], lol=[
        ...     ['ana@example.com', 'Ana Lopez'], ['ben@example.com', 'Ben Okafor'], ['ana@example.com', 'Ana L.']])

        Split it into the first row for each email, and the repeats:

        >>> unique_sdaf, dups_sdaf = signups_sdaf.remove_dups('email')
        >>> unique_sdaf
        <BLANKLINE>
        |      email      |    name    |
        | --------------: | ---------: |
        | ana@example.com |     Ana L. |
        | ben@example.com | Ben Okafor |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield='email'; name='daf_tmp_signups_v1'
        <BLANKLINE>
        >>> dups_sdaf
        <BLANKLINE>
        |      email      |   name    |
        | --------------: | --------: |
        | ana@example.com | Ana Lopez |
        <BLANKLINE>
        %% daf rows=1; cols=2; keyfield='email'; name='daf_tmp_signups_v2'
        <BLANKLINE>
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


def _concat(self: 'SqlDaf', other_instance: Daf, respect_kd: bool = False) -> 'SqlDaf':
    """
    Add the rows of another Daf or SqlDaf to the end of this one, as [`Daf.concat()`][daffodil.daf_modify._concat] does.

    The other must have the same columns, in the same order, as in a Daf. An other with no
    rows adds nothing, whatever its columns, as in a Daf. So the empty result of `apply()`,
    which has no columns, can be added with no error. A SqlDaf of the
    same database file is copied in SQL, in its row order, so no row comes into Python. A SqlDaf of
    another file is first copied into a TEMP table here, which is deleted after. An in-memory
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
        KeyError: The columns differ, and the other has rows.
        ValueError: A key added is a Python NULL. Nothing is added.
        sqlite3.IntegrityError: A key would be repeated. Nothing is added.
        NotImplementedError: The other is a selection or TEMP table of another thread's connection.

    Examples:
        Create two small SqlDaf tables of people, with one key in both:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> people_sdaf = SqlDaf('people', cols=['id', 'name', 'age'], keyfield='id', lol=[['p1', 'Ana Lopez', 34], ['p2', 'Ben Okafor', 52]])
        >>> more_sdaf = SqlDaf('more', cols=['id', 'name', 'age'], keyfield='id', lol=[['p2', 'Ben Okafor', 53], ['p3', 'Chen Wu', 27]])

        Adding a key that is already there raises, and nothing is added:

        >>> people_sdaf.concat(more_sdaf)
        Traceback (most recent call last):
        sqlite3.IntegrityError: UNIQUE constraint failed: people.id

        With `respect_kd=True`, a row whose key is there replaces that row, in its place:

        >>> people_sdaf.concat(more_sdaf, respect_kd=True)
        <BLANKLINE>
        | id |    name    | age |
        | -: | ---------: | --: |
        | p1 |  Ana Lopez |  34 |
        | p2 | Ben Okafor |  53 |
        | p3 |    Chen Wu |  27 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name='people'
        <BLANKLINE>

        The columns must match, in name and order:

        >>> people_sdaf.concat(Daf(cols=['id', 'name', 'city'], lol=[['p4', 'Dana Smith', 'Austin']]))
        Traceback (most recent call last):
        KeyError: "concat() into 'people': the columns differ. Only in this table: ['age']. Only in the other: ['city']."
    """
    # Rows are counted only when the columns differ, so a normal concat costs nothing more.
    if other_instance.columns() != self.columns():
        if not other_instance:
            return self
        raise KeyError(daf_utils.columns_mismatch_msg(self.columns(), other_instance.columns(), self.name))
    # A SqlDaf this connection cannot read, as in another file, is brought in as a TEMP copy, dropped after.
    rows_from: Any = self._readable(other_instance) if isinstance(other_instance, SqlDaf) else other_instance

    if self.parent_sdaf is not None:
        self.copy(inplace=True)
    self._resolve()
    if self.keyfield:
        self.ensure_key_index()

    upsert_sql: str = ''
    if respect_kd and self.keyfield:
        upsert_sql = f' ON CONFLICT ({quote_name(self.keyfield)}) DO UPDATE SET {self._excluded_sql()}'
    with self.transaction():
        if isinstance(rows_from, SqlDaf):
            rows_sql, rows_params = rows_from._rows_sql()
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
    if rows_from is not other_instance:
        self._execute(f'DROP TABLE IF EXISTS temp.{quote_name(rows_from.name)}')
    return self


def _write_rows(self: 'SqlDaf', rows_lol: Iterable[T_la], respect_kd: bool, method: str) -> 'SqlDaf':
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


def _excluded_sql(self: 'SqlDaf') -> str:
    """ The SET part of an upsert, which takes every column from the new row. Internal use. """
    return ', '.join(f'{quote_name(col)} = excluded.{quote_name(col)}' for col in self.hd)


def _append(
        self: 'SqlDaf',
        data_item:  Any = None,
        respect_kd: bool = False,
        *,
        lol:        T_lola | None = None,
        la:         T_la | None = None,
        fast:       bool = False,
        ) -> 'SqlDaf':
    """
    Add one row, or several, to the end, as [`Daf.append()`][daffodil.daf_modify._append] does.

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

    Examples:
        Create a small SqlDaf table of people:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> people_sdaf = SqlDaf('people', cols=['id', 'name', 'age'], keyfield='id', lol=[['p1', 'Ana Lopez', 34], ['p2', 'Ben Okafor', 52]])

        Add one row as a dict, placed by column name:

        >>> people_sdaf.append({'age': 27, 'id': 'p3', 'name': 'Chen Wu'}).select_record('p3')
        {'id': 'p3', 'name': 'Chen Wu', 'age': 27}

        Add one row as a list, in column order. A short list is padded with Python NULL:

        >>> people_sdaf.append(['p4', 'Dana Smith']).select_record('p4')
        {'id': 'p4', 'name': 'Dana Smith', 'age': ''}

        Add several rows as lists, or one row with `la=`:

        >>> people_sdaf.append(lol=[['p5', 'Eva Novak', 45], ['p6', 'Femi Ade', 38]]).keys()
        ['p1', 'p2', 'p3', 'p4', 'p5', 'p6']
        >>> people_sdaf.append(la=['p7', 'Gus Berg', 61]).len()
        7

        A key that is there raises, and nothing is added:

        >>> people_sdaf.append({'id': 'p1', 'name': 'Ana Lopez', 'age': 35})
        Traceback (most recent call last):
        sqlite3.IntegrityError: UNIQUE constraint failed: people.id

        Replace that row instead, in its place, with `respect_kd=True`:

        >>> people_sdaf.append({'id': 'p1', 'name': 'Ana Lopez', 'age': 35}, respect_kd=True).iloc(0)
        {'id': 'p1', 'name': 'Ana Lopez', 'age': 35}

        Add the rows of a Daf with the same columns:

        >>> people_sdaf.append(Daf(cols=['id', 'name', 'age'], lol=[['p8', 'Hana Sato', 29]])).iloc(-1)
        {'id': 'p8', 'name': 'Hana Sato', 'age': 29}

        `fast=True` is accepted, so code written for a Daf runs. Every row is checked anyway:

        >>> people_sdaf.append(['p9', 'Ivan Petrov', 50], fast=True).len()
        9
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


def _record_append(self: 'SqlDaf', record: T_da | KeyedList, respect_kd: bool = True) -> 'SqlDaf':
    """
    Add one row given as a dict or a KeyedList, as [`Daf.record_append()`][daffodil.daf_modify._record_append] does.

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

    Examples:
        Create a small SqlDaf table of people:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> people_sdaf = SqlDaf('people', cols=['id', 'name', 'age'], keyfield='id', lol=[['p1', 'Ana Lopez', 34], ['p2', 'Ben Okafor', 52]])

        Add a row with a new key:

        >>> people_sdaf.record_append({'id': 'p3', 'name': 'Chen Wu', 'age': 27}).len()
        3

        A key that is there replaces its row, by default:

        >>> people_sdaf.record_append({'id': 'p1', 'name': 'Ana Lopez', 'age': 35})
        <BLANKLINE>
        | id |    name    | age |
        | -: | ---------: | --: |
        | p1 |  Ana Lopez |  35 |
        | p2 | Ben Okafor |  52 |
        | p3 |    Chen Wu |  27 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name='people'
        <BLANKLINE>

        With `respect_kd=False`, a key that is there raises instead:

        >>> people_sdaf.record_append({'id': 'p1', 'name': 'Ana Lopez', 'age': 36}, respect_kd=False)
        Traceback (most recent call last):
        sqlite3.IntegrityError: UNIQUE constraint failed: people.id
    """
    if not isinstance(record, (dict, KeyedList)):
        raise TypeError(f"record_append(): the record must be a dict or a KeyedList, not {type(record).__name__}.")
    if not record:
        return self
    return self._write_rows([self._row_from_item(record, 'record_append')], respect_kd, 'record_append')


def _extend(
        self: 'SqlDaf',
        records_lod:    List[T_da] | None = None,
        respect_kd:     bool = False,
        *,
        lol:            T_lola | None = None,
        ) -> 'SqlDaf':
    """
    Add several rows, as dicts or as lists, as [`Daf.extend()`][daffodil.daf_modify._extend] does.

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

    Examples:
        Create a small SqlDaf table of people:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> people_sdaf = SqlDaf('people', cols=['id', 'age'], keyfield='id', lol=[['p1', 34]])

        Add several rows as dicts, each placed by column name:

        >>> people_sdaf.extend([{'id': 'p2', 'age': 52}, {'age': 27, 'id': 'p3'}]).to_lod()
        [{'id': 'p1', 'age': 34}, {'id': 'p2', 'age': 52}, {'id': 'p3', 'age': 27}]

        Add rows as lists. With `respect_kd=True`, a key that is there replaces its row:

        >>> people_sdaf.extend(lol=[['p4', 41], ['p1', 35]], respect_kd=True).to_lod()
        [{'id': 'p1', 'age': 35}, {'id': 'p2', 'age': 52}, {'id': 'p3', 'age': 27}, {'id': 'p4', 'age': 41}]

        If one row fails, none is added:

        >>> people_sdaf.extend(lol=[['p5', 45], ['p2', 53]])
        Traceback (most recent call last):
        sqlite3.IntegrityError: UNIQUE constraint failed: people.id
        >>> people_sdaf.keys()
        ['p1', 'p2', 'p3', 'p4']
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


def _assign_record(self: 'SqlDaf', record: T_da) -> 'SqlDaf':
    """
    Put one row in by its key, replacing the row that has that key, as [`Daf.assign_record()`][daffodil.daf_modify._assign_record] does.

    The whole row is replaced, in its place. Columns the dict lacks become Python NULL. A new
    key is added at the end. Use `update_record_krow()` to change only some cells.

    Args:
        record: The row, as a dict. It must have the keyfield.

    Returns:
        This SqlDaf, which has been changed.

    Raises:
        KeyError: The record lacks the keyfield.
        ValueError: There is no keyfield, or the key is a Python NULL.

    Examples:
        Create a small SqlDaf table of people:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> people_sdaf = SqlDaf('people', cols=['id', 'name', 'age'], keyfield='id', lol=[['p1', 'Ana Lopez', 34], ['p2', 'Ben Okafor', 52]])

        Replace a whole row by its key. Columns the dict lacks become Python NULL:

        >>> people_sdaf.assign_record({'id': 'p1', 'age': 35})
        <BLANKLINE>
        | id |    name    | age |
        | -: | ---------: | --: |
        | p1 |            |  35 |
        | p2 | Ben Okafor |  52 |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield='id'; name='people'
        <BLANKLINE>

        A new key is added at the end:

        >>> people_sdaf.assign_record({'id': 'p3', 'name': 'Chen Wu', 'age': 27}).keys()
        ['p1', 'p2', 'p3']
    """
    key_col: str = self._need_keyfield('assign_record')
    if key_col not in record:
        raise KeyError(key_col)
    return self._write_rows([self._row_from_item(record, 'assign_record')], True, 'assign_record')


def _update_by_keylist(self: 'SqlDaf', keylist: T_la | None = None, record: T_da | None = None) -> 'SqlDaf':
    """
    Change some cells in the rows that have the given keys, as [`Daf.update_by_keylist()`][daffodil.daf_modify._update_by_keylist] does.

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

    Examples:
        Create a small SqlDaf table of people:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> people_sdaf = SqlDaf('people', cols=['id', 'name', 'city'], keyfield='id', lol=[
        ...     ['p1', 'Ana Lopez', 'Denver'], ['p2', 'Ben Okafor', 'Austin'], ['p3', 'Chen Wu', 'Denver']])

        Change the city of two people. A key that is not found, and a name that is not a column, are ignored:

        >>> people_sdaf.update_by_keylist(['p1', 'p3', 'p9'], {'city': 'Boulder', 'email': 'ignored'})
        <BLANKLINE>
        | id |    name    |  city   |
        | -: | ---------: | ------: |
        | p1 |  Ana Lopez | Boulder |
        | p2 | Ben Okafor |  Austin |
        | p3 |    Chen Wu | Boulder |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name='people'
        <BLANKLINE>
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


def _update_record_krow(self: 'SqlDaf', key: Any, record: T_da | None = None) -> 'SqlDaf':
    """
    Change some cells in the row that has this key, as [`Daf.update_record_krow()`][daffodil.daf_modify._update_record_krow] does.

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
        Create a small SqlDaf table whose keys are ints, as employee numbers:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> staff_sdaf = SqlDaf('staff', cols=['emp_no', 'name', 'title'], keyfield='emp_no', lol=[
        ...     [1042, 'Ana Lopez', 'Analyst'], [2071, 'Ben Okafor', 'Engineer']])

        Change some cells of the row with key 2071. An int here is a key, not a position:

        >>> staff_sdaf.update_record_krow(2071, {'title': 'Lead Engineer'})
        <BLANKLINE>
        | emp_no |    name    |     title     |
        | -----: | ---------: | ------------: |
        |   1042 |  Ana Lopez |       Analyst |
        |   2071 | Ben Okafor | Lead Engineer |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield='emp_no'; name='staff'
        <BLANKLINE>
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


def _assign_col(self: 'SqlDaf', colname: str, la: T_la | None = None, default: Any = NULL) -> 'SqlDaf':
    """
    Fill a column by name with the values of a list, or add it if it is new, as [`Daf.assign_col()`][daffodil.daf_modify._assign_col] does.

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

    Examples:
        Create a small SqlDaf table of people:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> people_sdaf = SqlDaf('people', cols=['id', 'age'], keyfield='id', lol=[['p1', 34], ['p2', 52], ['p3', 27]])

        Fill a column from a list, in row order:

        >>> people_sdaf.assign_col('age', [35, 53, 28]).col('age')
        [35, 53, 28]

        A short list is filled out with `default`:

        >>> people_sdaf.assign_col('age', [40], default=0).col('age')
        [40, 0, 0]

        A new column is added at the end, here with the default in every row:

        >>> people_sdaf.assign_col('city', default='unknown')
        <BLANKLINE>
        | id | age |  city   |
        | -: | --: | ------: |
        | p1 |  40 | unknown |
        | p2 |   0 | unknown |
        | p3 |   0 | unknown |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name='people'
        <BLANKLINE>
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


def _apply_to_col(self: 'SqlDaf', col: str, func: Callable, **kwargs: Any) -> 'SqlDaf':
    """
    Replace each value of a column by the result of a function, in place, as [`Daf.apply_to_col()`][daffodil.daf_apply._apply_to_col] does.

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

    Examples:
        Create a small SqlDaf table of people, with heights in centimeters. One is not known:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> people_sdaf = SqlDaf('people', cols=['id', 'height'], keyfield='id', lol=[['p1', 168], ['p2', ''], ['p3', 175]])

        Convert each height to meters. Keyword arguments go to the function:

        >>> def to_meters(height, per_meter=100):
        ...     return height / per_meter if height != '' else ''
        >>> people_sdaf.apply_to_col('height', to_meters, per_meter=100).col('height')
        [1.68, '', 1.75]

        If the function raises, nothing changes:

        >>> people_sdaf.apply_to_col('height', lambda height: 1 / (height - 1.68))
        Traceback (most recent call last):
        ZeroDivisionError: float division by zero
        >>> people_sdaf.col('height')
        [1.68, '', 1.75]
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


def _apply_in_place(
        self: 'SqlDaf',
        func:       Callable[..., Any],         # called as func(row, **kwargs).
        by:         str = 'row',                # 'row' or 'row_klist'.
        rowkeys:    T_la | None = None,         # the keys of the rows to change. All if None.
        **kwargs:   Any,                        # passed on to func.
        ) -> 'SqlDaf':
    """
    Apply a function to each row and store the results, as [`Daf.apply_in_place()`][daffodil.daf_apply._apply_in_place] does.

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

    Examples:
        Create a small SqlDaf table of orders, with a price and a quantity, and a total to fill in:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> orders_sdaf = SqlDaf('orders', cols=['id', 'price', 'qty', 'total'], keyfield='id', lol=[
        ...     ['o1', 12.5, 2, ''], ['o2', 4.0, 6, ''], ['o3', 30.0, 1, '']])

        With `by='row'`, the function returns the cells to change, by name:

        >>> def add_total(row, tax=0.0):
        ...     return {'total': row['price'] * row['qty'] * (1 + tax)}
        >>> orders_sdaf.apply_in_place(add_total)
        <BLANKLINE>
        | id | price | qty | total |
        | -: | ----: | --: | ----: |
        | o1 |  12.5 |   2 |  25.0 |
        | o2 |   4.0 |   6 |  24.0 |
        | o3 |  30.0 |   1 |  30.0 |
        <BLANKLINE>
        %% daf rows=3; cols=4; keyfield='id'; name='orders'
        <BLANKLINE>

        Change only some rows, with `rowkeys`. Other keywords go to the function:

        >>> orders_sdaf.apply_in_place(add_total, rowkeys=['o2'], tax=0.25).col('total')
        [25.0, 30.0, 30.0]

        With `by='row_klist'`, the function changes a KeyedList and returns nothing:

        >>> def double_qty(row):
        ...     row['qty'] = row['qty'] * 2
        >>> orders_sdaf.apply_in_place(double_qty, by='row_klist').col('qty')
        [4, 12, 2]
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


def _check_layout_change(self: 'SqlDaf', method: str) -> None:
    """ Raise NotImplementedError on a child, whose columns are its parent's. Resolve a join not read yet. Internal use. """
    if self.parent_sdaf is not None:
        raise NotImplementedError(f"{method}: '{self.name}' is a selection, and its columns are its parent's. "
                                  "Change the parent, or make the selection a table first, with copy(inplace=True).")
    self._resolve()


def _insert_col(
        self: 'SqlDaf',
        colname:    str,                        # the name of the column.
        col_la:     T_la | None = None,         # the values, in row order.
        icol:       int = -1,                   # the position. Only the end: -1, or the number of columns.
        default:    Any = NULL,                 # the value for rows that the list does not reach.
        ) -> 'SqlDaf':
    """
    Add a named column, or fill it if the name is there, as [`Daf.insert_col()`][daffodil.daf_modify._insert_col] does.

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
        Create a small SqlDaf table of people:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> people_sdaf = SqlDaf('people', cols=['id', 'name'], keyfield='id', lol=[['p1', 'Ana Lopez'], ['p2', 'Ben Okafor']])

        Add a column at the end, from a list. A short list is filled out with `default`:

        >>> people_sdaf.insert_col('city', ['Denver'], default='unknown')
        <BLANKLINE>
        | id |    name    |  city   |
        | -: | ---------: | ------: |
        | p1 |  Ana Lopez |  Denver |
        | p2 | Ben Okafor | unknown |
        <BLANKLINE>
        %% daf rows=2; cols=3; keyfield='id'; name='people'
        <BLANKLINE>

        A column at another position is not supported on a SqlDaf:

        >>> people_sdaf.insert_col('age', icol=1)
        Traceback (most recent call last):
        NotImplementedError: ...
    """
    if not colname:
        return self
    if colname not in self.hd and icol not in (-1, len(self.hd)):
        raise NotImplementedError(f"insert_col: a new column goes at the end of a SqlDaf, not at position {icol}. "
                                  "Use icol=-1, and select_cols() to read the columns in another order.")
    return self.assign_col(colname, col_la or [], default)


def _insert_idx_col(self: 'SqlDaf', colname: str = 'idx', icol: int = 0, startat: int = 0) -> 'SqlDaf':
    """
    Insert a column of row numbers, as [`Daf.insert_idx_col()`][daffodil.daf_modify._insert_idx_col] does.

    The rows are numbered from `startat`, in their order here, so a selection is numbered from
    its own first row. The column goes at `icol`. A negative `icol`, or one past the last column,
    puts it at the end. If the name is a column already, its values are replaced.

    SQLite adds a column only at the end, so this makes a TEMP copy with the column in place,
    numbered in SQL, and this SqlDaf then refers to it, as a sort does. The table it came from is
    not changed. That copies every row, so number a selection rather than a whole large table.

    Args:
        colname: The name of the new column. An empty name does nothing.
        icol: The column position.
        startat: The number of the first row.

    Returns:
        This SqlDaf, which now refers to the TEMP copy.

    Examples:
        Create a small SqlDaf table of people:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> people_sdaf = SqlDaf('people', cols=['id', 'name', 'city'], keyfield='id', lol=[
        ...     ['p1', 'Ana Lopez', 'Denver'], ['p2', 'Ben Okafor', 'Austin'], ['p3', 'Chen Wu', 'Denver']])

        Number the people in Denver, from 1. The selection gets rows of its own first, so the table is not changed:

        >>> people_sdaf.select_by_dict({'city': 'Denver'}).insert_idx_col(startat=1)
        <BLANKLINE>
        | idx | id |   name    |  city  |
        | --: | -: | --------: | -----: |
        |   1 | p1 | Ana Lopez | Denver |
        |   2 | p3 |   Chen Wu | Denver |
        <BLANKLINE>
        %% daf rows=2; cols=4; keyfield='id'; name='daf_tmp_people_v2'
        <BLANKLINE>
        >>> people_sdaf.columns()
        ['id', 'name', 'city']

        Give the column another name, and put it in another place:

        >>> people_sdaf.insert_idx_col('row_no', icol=1)
        <BLANKLINE>
        | id | row_no |    name    |  city  |
        | -: | -----: | ---------: | -----: |
        | p1 |      0 |  Ana Lopez | Denver |
        | p2 |      1 | Ben Okafor | Austin |
        | p3 |      2 |    Chen Wu | Denver |
        <BLANKLINE>
        %% daf rows=3; cols=4; keyfield='id'; name='daf_tmp_people_v3'
        <BLANKLINE>
    """
    if not colname:
        return self
    cols_ls: T_ls = list(self.hd)
    number_sql: str = f'ROW_NUMBER() OVER (ORDER BY {ORD_COL}) - 1 + ?'

    # ---- the columns of the copy, and the SQL that gives each, in order
    exprs_ls: T_ls
    if colname in self.hd:
        exprs_ls = [number_sql if col == colname else quote_name(col) for col in cols_ls]
    else:
        place: int = len(cols_ls) if icol < 0 or icol >= len(cols_ls) else icol
        cols_ls.insert(place, colname)
        exprs_ls = [number_sql if col == colname else quote_name(col) for col in cols_ls]
    cols_sql: str = ', '.join(quote_name(col) for col in cols_ls)

    target: str = self._next_temp_name(self._root().name)
    rows_sql, rows_params = self._rows_sql()
    with self.transaction():
        self._execute(_create_table_sql(self.conn, target, cols_ls, 'temp'))
        self._execute(f'INSERT INTO {quote_name(target)} ({cols_sql}) '
                      f'SELECT {", ".join(exprs_ls)} FROM ({rows_sql}) ORDER BY {ORD_COL}',
                      (startat,) + tuple(rows_params))
    self._name          = target
    self.hd             = {col: idx for idx, col in enumerate(cols_ls)}
    self.has_rid        = True
    self.is_temp        = True
    self.parent_sdaf    = None
    self.is_resolved    = True
    self.pending_sql    = ''
    self.pending_params = ()
    return self


def _drop_cols(self: 'SqlDaf', exclude_cols: T_ls | None = None) -> 'SqlDaf':
    """
    Remove columns from the table, in place, as [`Daf.drop_cols()`][daffodil.daf_modify._drop_cols] does.

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
        Create a small SqlDaf table of people:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> people_sdaf = SqlDaf('people', cols=['id', 'name', 'city', 'age'], keyfield='id', lol=[['p1', 'Ana Lopez', 'Denver', 34]])

        Drop a column from the table:

        >>> people_sdaf.drop_cols(['city'])
        <BLANKLINE>
        | id |   name    | age |
        | -: | --------: | --: |
        | p1 | Ana Lopez |  34 |
        <BLANKLINE>
        %% daf rows=1; cols=3; keyfield='id'; name='people'
        <BLANKLINE>
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


def _rename_cols(self: 'SqlDaf', from_to_dict: T_ds) -> 'SqlDaf':
    """
    Rename columns in place, as [`Daf.rename_cols()`][daffodil.daf_structure._rename_cols] does.

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
        Create a small SqlDaf table of people, keyed by id:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> people_sdaf = SqlDaf('people', cols=['id', 'name'], keyfield='id', lol=[['p1', 'Ana Lopez']])

        Rename the key column. The keyfield follows the new name:

        >>> people_sdaf.rename_cols({'id': 'person_id'})
        <BLANKLINE>
        | person_id |   name    |
        | --------: | --------: |
        |        p1 | Ana Lopez |
        <BLANKLINE>
        %% daf rows=1; cols=2; keyfield='person_id'; name='people'
        <BLANKLINE>
    """
    self._check_layout_change('rename_cols')
    self._rename_by_map({old: new for old, new in from_to_dict.items() if old in self.hd and new != old})
    return self


def _rename_by_map(self: 'SqlDaf', old_to_new: T_ds) -> None:
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


def _set_cols(
        self: 'SqlDaf',
        new_cols:       T_ls | None = None,     # the names, in order. Spreadsheet names if None.
        sanitize_cols:  bool = True,            # if True, make the names unique and not empty.
        unnamed_prefix: str = 'col',            # the start of a name made for an empty one.
        ) -> 'SqlDaf':
    """
    Set the column names by position, in place, as [`Daf.set_cols()`][daffodil.daf_structure._set_cols] does.

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
        Create a small SqlDaf table with plain column names:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'

        Set all the names at once. Repeated and empty names are taken as given:

        >>> SqlDaf('t1', cols=['a', 'b', 'c'], lol=[['Ana Lopez', 'Denver', 34]]).set_cols(['name', 'name', '']).columns()
        ['name', 'name_1', 'col2']

        With `sanitize_cols=True`, an empty name gets one made from `unnamed_prefix`, and a repeated name gets
        a number:

        >>> SqlDaf('t2', cols=['a', 'b', 'c'], lol=[['Ana Lopez', 'Denver', 34]]).set_cols(['name', '', 'name'], sanitize_cols=True).columns()
        ['name', 'col1', 'name_2']
        >>> SqlDaf('t3', cols=['a', 'b', 'c'], lol=[['Ana Lopez', 'Denver', 34]]).set_cols(['name', '', 'age'], sanitize_cols=True, unnamed_prefix='col').columns()
        ['name', 'col1', 'age']
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


def _set_dtypes(
        self: 'SqlDaf',
        default_type:       Type = str,                     # the type of every column not listed.
        typ_to_cols_dict:   Dict[Type, T_ls] | None = None, # maps a type to the names of its columns.
        ) -> 'SqlDaf':
    """
    Set the dtype of every column, from a default and a few exceptions, as [`Daf.set_dtypes()`][daffodil.daf_dtypes._set_dtypes] does.

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
        Create a small SqlDaf table of people, loaded as text:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> people_sdaf = SqlDaf('people', cols=['id', 'name', 'age'], lol=[['p1', 'Ana Lopez', '34']])

        Record str as the type of every column, and int for the age. Nothing is converted:

        >>> people_sdaf.set_dtypes(str, {int: ['age']}).dtypes
        {'id': <class 'str'>, 'name': <class 'str'>, 'age': <class 'int'>}
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


def _sort_by_colname(
        self: 'SqlDaf',
        colname:            str,                # the column to sort by.
        *,
        reverse:            bool = False,       # if True, sort from high to low.
        length_priority:    bool = False,       # if True, sort by length first, then by value.
        as_str:             bool = False,       # if True, sort by the text of each value.
        ) -> 'SqlDaf':
    """
    Sort the rows by one column, in place, as [`Daf.sort_by_colname()`][daffodil.daf_modify._sort_by_colname] does.

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
        Create a small SqlDaf table of people:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> people_sdaf = SqlDaf('people', cols=['id', 'name', 'age'], lol=[
        ...     ['p1', 'Ana Lopez', 34], ['p2', 'Ben Okafor', 52], ['p3', 'Chen Wu', 27]])

        Sort by age, from low to high, and from high to low:

        >>> people_sdaf.sort_by_colname('age')
        <BLANKLINE>
        | id |    name    | age |
        | -: | ---------: | --: |
        | p3 |    Chen Wu |  27 |
        | p1 |  Ana Lopez |  34 |
        | p2 | Ben Okafor |  52 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield=''; name='daf_tmp_people_v1'
        <BLANKLINE>
        >>> people_sdaf.sort_by_colname('age', reverse=True).col('name')
        ['Ben Okafor', 'Ana Lopez', 'Chen Wu']

        Sort text of digits with shorter values first, with `length_priority=True`, so '9' comes before '10':

        >>> SqlDaf('codes', cols=['code'], lol=[['10'], ['9'], ['100']]).sort_by_colname('code', length_priority=True).col('code')
        ['9', '10', '100']

        A column of numbers mixed with text sorts in SQLite's order, numbers first, with no error. With
        `as_str=True` it sorts by text, where '10' comes before '9':

        >>> mixed_sdaf = SqlDaf('mixed', cols=['id', 'v'], lol=[['a', 10], ['b', 9], ['c', 'x']])
        >>> mixed_sdaf.sort_by_colname('v').col('v')
        [9, 10, 'x']
        >>> mixed_sdaf.sort_by_colname('v', as_str=True).col('v')
        [10, 9, 'x']
    """
    return self.sort_by_colnames([colname], reverse=reverse, length_priority=length_priority, as_str=as_str)


def _sort_by_colnames(
        self: 'SqlDaf',
        colnames:           T_ls,               # the columns to sort by, main key first.
        reverse:            bool = False,       # if True, sort from high to low.
        length_priority:    bool = False,       # if True, sort by length first, then by value.
        as_str:             bool = False,       # if True, sort by the text of each value.
        ) -> 'SqlDaf':
    """
    Sort the rows by several columns, in place, as [`Daf.sort_by_colnames()`][daffodil.daf_modify._sort_by_colnames] does.

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
        Create a small SqlDaf table of people:

        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> people_sdaf = SqlDaf('people', cols=['id', 'city', 'name'], lol=[
        ...     ['p1', 'Denver', 'Chen Wu'], ['p2', 'Austin', 'Ben Okafor'], ['p3', 'Denver', 'Ana Lopez']])

        Sort by city, then by name within each city:

        >>> people_sdaf.sort_by_colnames(['city', 'name'])
        <BLANKLINE>
        | id |  city  |    name    |
        | -: | -----: | ---------: |
        | p2 | Austin | Ben Okafor |
        | p3 | Denver |  Ana Lopez |
        | p1 | Denver |    Chen Wu |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield=''; name='daf_tmp_people_v1'
        <BLANKLINE>

        Sort by name from high to low, or with shorter names first:

        >>> people_sdaf.sort_by_colnames(['name'], reverse=True).col('name')
        ['Chen Wu', 'Ben Okafor', 'Ana Lopez']
        >>> people_sdaf.sort_by_colnames(['name'], length_priority=True).col('name')
        ['Chen Wu', 'Ana Lopez', 'Ben Okafor']

        Sort numbers mixed with text by their text, with `as_str=True`:

        >>> SqlDaf('mixed', cols=['id', 'v'], lol=[['a', 10], ['b', 9], ['c', 'x']]).sort_by_colnames(['v'], as_str=True).col('v')
        [10, 9, 'x']
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

    root_sdaf: SqlDaf = self._root()
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


def _attach_schema(self: 'SqlDaf', schema: type) -> 'SqlDaf':
    """
    Attach a `@schemaclass` to this SqlDaf, as [`Daf.attach_schema()`][daffodil.lib.daf_schema._attach_schema] does.

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
        Describe a table of people with a schema class: its columns, their types and its keyfield:

        >>> import tempfile
        >>> from daffodil.lib.schemaclass import schemaclass
        >>> @schemaclass
        ... class Person:
        ...     __keyfield__ = 'id'
        ...     id: str = ''
        ...     name: str = ''
        ...     age: int = 0

        Create a small SqlDaf table, loaded as text, and attach the schema to it:

        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> people_sdaf = SqlDaf('people', cols=['id', 'name', 'age'], lol=[['p1', 'Ana Lopez', '34']]).attach_schema(Person)

        The keyfield and the dtypes now come from the schema:

        >>> people_sdaf.keyfield, people_sdaf.dtypes
        ('id', {'id': <class 'str'>, 'name': <class 'str'>, 'age': <class 'int'>})
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
