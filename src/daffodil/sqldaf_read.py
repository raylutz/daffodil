# sqldaf_read.py
# copyright (c) 2026 Ray Lutz
"""
Methods of SqlDaf for reading.

Each function here takes the table as its first argument, or the class for a classmethod.
It is assigned in the SqlDaf class body under its name, without the underscore.
The docs show it under that name.

Some functions use names defined in sqldaf.py, such as SqlDaf, when they run.
sqldaf.py sets those names here, as listed in _NAMES_FROM_SQLDAF, once they exist.
An import at the top would be circular, since sqldaf.py imports this module.
"""


import io
import csv
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Tuple, overload, TYPE_CHECKING
from daffodil.daf import Daf, NULL
from daffodil.lib.daf_types import T_ls, T_cs, T_la, T_lola, T_da, T_buff
import daffodil.lib.daf_utils as daf_utils
import daffodil.lib.daf_sql as daf_sql
from daffodil.lib.daf_sql import sqlite3

if TYPE_CHECKING:       # for mypy. sqldaf.py sets the names below when the program runs.
    from daffodil.sqldaf import SqlDaf, ORD_COL, none_to_null, quote_name

# The names of sqldaf.py that the functions here use when they run.
# sqldaf.py sets each one here once it exists. An import at the top would be circular.
_NAMES_FROM_SQLDAF: Tuple[str, ...] = ('ORD_COL', 'none_to_null', 'quote_name')


def _md_daf_table_snippet(self: 'SqlDaf') -> str:
    """
    Make a short Markdown table of the rows, with a summary line, as [`Daf.md_daf_table_snippet()`][daffodil.daf_report._md_daf_table_snippet] does.

    It is the same text as for a Daf of the same rows, byte for byte. It reads the rows from the
    table in SQL: a count, then the first and the last rows it shows. So a table of any size
    shows in the same time. On 1,603,908 rows it took 0.64 s.

    Returns:
        The Markdown text.

    Examples:
        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> nums_sdaf = SqlDaf.from_daf(Daf(cols=['id', 'n'], lol=[[f'r{i}', i] for i in range(30)]), name='nums')
        >>> print(nums_sdaf.md_daf_table_snippet())
        | id  |  n  |
        | --: | --: |
        |  r0 |   0 |
        |  r1 |   1 |
        |  r2 |   2 |
        |  r3 |   3 |
        |  r4 |   4 |
        | ... | ... |
        | r25 |  25 |
        | r26 |  26 |
        | r27 |  27 |
        | r28 |  28 |
        | r29 |  29 |
        <BLANKLINE>
        %% daf rows=30; cols=2; keyfield=''; name='nums'
        <BLANKLINE>
    """
    return Daf.md_daf_table_snippet(self)


def _select_rows(self: 'SqlDaf', where_sql: str = '', params: Tuple[Any, ...] = (), limit: int = -1, offset: int = 0,
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


def _to_daf(self: 'SqlDaf', name: str | None = None) -> Daf:
    """
    Pull every row into an ordinary Daf in memory.

    Use it once the rows you need are few enough to hold. The rows come in order. A SQL NULL
    comes back as Python NULL. The new Daf has this SqlDaf's keyfield and dtypes. Nothing
    ties it to the table afterwards: a change to one does not show in the other. A child has
    its rows found first.

    The Daf has the table's name. A selection, or a copy or join made without a name, has a name
    that SqlDaf made up, as `daf_tmp_items_v1`. Give `name` to set another.

    Args:
        name: The name of the new Daf. If None, the table's name.

    Returns:
        A Daf of every row.

    Examples:
        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> items_sdaf = SqlDaf.from_daf(Daf(cols=['id', 'item', 'qty'], keyfield='id', lol=[['a1', 'pen', 3], ['b2', 'ink', ''], ['c3', 'pad', 7]]), name='items')
        >>> items_daf = items_sdaf.to_daf()
        >>> items_daf
        <BLANKLINE>
        | id | item | qty |
        | -: | ---: | --: |
        | a1 |  pen |   3 |
        | b2 |  ink |     |
        | c3 |  pad |   7 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name='items'
        <BLANKLINE>

        The Daf is a copy. A change to it does not reach the table:

        >>> items_daf['a1', 'qty'] = 99
        >>> items_sdaf.select_record('a1')
        {'id': 'a1', 'item': 'pen', 'qty': 3}

        Select first, so only the rows needed come into memory:

        >>> items_sdaf.select_where_sql('qty > ?', [5]).to_daf()
        <BLANKLINE>
        | id | item | qty |
        | -: | ---: | --: |
        | c3 |  pad |   7 |
        <BLANKLINE>
        %% daf rows=1; cols=3; keyfield='id'; name='daf_tmp_items_v1'
        <BLANKLINE>

        A selection has a name that SqlDaf made up. Give another with `name`:

        >>> items_sdaf.select_where_sql('qty > ?', [5]).to_daf(name='big_items').name
        'big_items'
    """
    self._resolve()
    lol: T_lola = [none_to_null(row_t) for row_t in self._select_rows()]
    return Daf(lol=lol, cols=self.columns(), keyfield=self.keyfield, name=self.name if name is None else name,
               dtypes=dict(self.dtypes) if self.dtypes else None)


def _write_csv(self: 'SqlDaf', text_f: Any, line_terminator: str | None, include_header: bool) -> None:
    """ Write the rows as CSV to an open text file, a batch at a time, as [`Daf.to_csv_buff()`][daffodil.daf_convert._to_csv_buff] writes them. Internal use. """
    csv_writer = csv.writer(text_f, lineterminator='\r\n' if line_terminator is None else line_terminator)
    if include_header:
        csv_writer.writerow([daf_sql.sql_unesc_str(col) for col in self.columns()])     # a name stored with the __HH encoding is written decoded.
    batch_lot: List[Tuple[Any, ...]]
    for batch_lot in self._iter_row_batches():
        csv_writer.writerows(batch_lot)                     # SQL NULL is written as an empty cell, as Python NULL is.


def _to_csv_file(
        self: 'SqlDaf',
        file_path:          str | Path = '',
        line_terminator:    str | None = None,
        include_header:     bool = True,
        ) -> str:
    r"""
    Write the rows to a CSV file, as [`Daf.to_csv_file()`][daffodil.daf_convert._to_csv_file] does.

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
        >>> marks_sdaf
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

        Without the header line, and with another line ending:

        >>> import os, tempfile
        >>> csv_sdaf = SqlDaf.from_daf(Daf(cols=['id', 'qty'], lol=[['a1', 3], ['b2', 10]]), name='csv_out')
        >>> csv_path = csv_sdaf.to_csv_file(os.path.join(tempfile.mkdtemp(), 'out.csv'), line_terminator='\n', include_header=False)
        >>> open(csv_path).read()
        'a1,3\nb2,10\n'
    """
    file_path = str(file_path)
    if file_path.startswith('s3'):
        Daf.buff_to_file(self.to_csv_buff(line_terminator=line_terminator, include_header=include_header), file_path)
        return file_path
    with open(daf_utils.path_sep_per_os(file_path), 'w', newline='', encoding='utf-8') as csv_f:
        self._write_csv(csv_f, line_terminator, include_header)
    return file_path


def _to_csv_buff(
        self: 'SqlDaf',
        line_terminator:    str | None = None,
        include_header:     bool = True,
        ) -> T_buff:
    r"""
    Make CSV text from the rows, as [`Daf.to_csv_buff()`][daffodil.daf_convert._to_csv_buff] does.

    The whole text is built in memory, so use it for a selection of modest size. To write a
    large table, use `to_csv_file()`, which writes as it reads.

    Args:
        line_terminator: The line ending. If None, `\r\n` is used.
        include_header: If True, write the column names first.

    Returns:
        The CSV text.

    Examples:
        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> items_sdaf = SqlDaf.from_daf(Daf(cols=['id', 'item', 'qty'], keyfield='id', lol=[['a1', 'pen', 3], ['b2', 'ink', ''], ['c3', 'pad', 7]]), name='items')
        >>> items_sdaf.to_csv_buff()
        'id,item,qty\r\na1,pen,3\r\nb2,ink,\r\nc3,pad,7\r\n'

        Without the header, and with another line ending:

        >>> items_sdaf.to_csv_buff(line_terminator='\n', include_header=False)
        'a1,pen,3\nb2,ink,\nc3,pad,7\n'
    """
    text_f = io.StringIO(newline='')
    self._write_csv(text_f, line_terminator, include_header)
    return text_f.getvalue()


def _to_dict(self: 'SqlDaf') -> T_da:
    """
    Read a table of one row as a dict, as [`Daf.to_dict()`][daffodil.daf_convert._to_dict] does.

    A child has its rows found first. For a row by key or by position, `select_record()` and
    `iloc()` are cheaper: they make no selection.

    Returns:
        The row as a dict. An empty dict if there are no rows.

    Raises:
        ValueError: There is more than one row.

    Examples:
        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> items_sdaf = SqlDaf.from_daf(Daf(cols=['id', 'item', 'qty'], keyfield='id', lol=[['a1', 'pen', 3], ['b2', 'ink', ''], ['c3', 'pad', 7]]), name='items')
        >>> items_sdaf['b2'].to_dict()
        {'id': 'b2', 'item': 'ink', 'qty': ''}
        >>> items_sdaf.select_record('b2')
        {'id': 'b2', 'item': 'ink', 'qty': ''}
    """
    self._resolve()
    rows_lot: List[Tuple[Any, ...]] = self._select_rows(limit=2)
    if len(rows_lot) > 1:
        raise ValueError("Ambiguous 2dim array.")
    if not rows_lot:
        return {}
    return dict(zip(self.hd, none_to_null(rows_lot[0])))


def _to_list(self: 'SqlDaf') -> T_la:
    """
    Read a table of one row, or of one column, as a list, as [`Daf.to_list()`][daffodil.daf_convert._to_list] does.

    One column gives every value of the column, in order, so select first. A child has its
    rows found first.

    Returns:
        The values of the row, or of the column.

    Raises:
        ValueError: There is more than one row and more than one column.

    Examples:
        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> items_sdaf = SqlDaf.from_daf(Daf(cols=['id', 'item', 'qty'], keyfield='id', lol=[['a1', 'pen', 3], ['b2', 'ink', ''], ['c3', 'pad', 7]]), name='items')
        >>> items_sdaf['c3'].to_list()
        ['c3', 'pad', 7]
        >>> items_sdaf[:, 'qty'].to_list()
        [3, '', 7]
    """
    self._resolve()
    if len(self.hd) == 1:
        return [NULL if row_t[0] is None else row_t[0] for row_t in self._select_rows()]
    rows_lot: List[Tuple[Any, ...]] = self._select_rows(limit=2)
    if len(rows_lot) > 1:
        raise ValueError("to_list() requires a 1D Daf (single row or single column)")
    return none_to_null(rows_lot[0]) if rows_lot else []


def _to_value(self: 'SqlDaf') -> Any:
    """
    Read a table of one row and one column as its value, as [`Daf.to_value()`][daffodil.daf_convert._to_value] does.

    A child has its rows found first.

    Returns:
        The value. Python NULL for SQL NULL.

    Raises:
        ValueError: The table is not one row and one column.

    Examples:
        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> items_sdaf = SqlDaf.from_daf(Daf(cols=['id', 'item', 'qty'], keyfield='id', lol=[['a1', 'pen', 3], ['b2', 'ink', ''], ['c3', 'pad', 7]]), name='items')
        >>> items_sdaf['c3', 'qty'].to_value()
        7

        A SQL NULL comes back as Python NULL, the empty string:

        >>> items_sdaf['b2', 'qty'].to_value()
        ''
    """
    self._resolve()
    rows_lot: List[Tuple[Any, ...]] = self._select_rows(limit=2)
    if len(rows_lot) != 1 or len(self.hd) != 1:
        raise ValueError(f"to_value() requires 1×1, got shape ({len(rows_lot) if len(rows_lot) < 2 else '2+'}, {len(self.hd)})")
    return NULL if rows_lot[0][0] is None else rows_lot[0][0]


@overload
def _iloc(self: 'SqlDaf', irow: int = 0, include_cols: T_ls | None = None, rtype: Literal['dict'] = 'dict') -> T_da: ...
@overload
def _iloc(self: 'SqlDaf', irow: int = 0, include_cols: T_ls | None = None, *, rtype: Literal['list']) -> T_la: ...
@overload
def _iloc(self: 'SqlDaf', irow: int = 0, include_cols: T_ls | None = None, rtype: str = 'dict') -> T_da | T_la: ...

def _iloc(
        self: 'SqlDaf',
        irow:           int = 0,                    # the position of the row. Negative counts from the end.
        include_cols:   T_ls | None = None,         # only these columns.
        rtype:          str = 'dict',               # 'dict' or 'list'.
        ) -> T_da | T_la:
    """
    Read one row by position, as [`Daf.iloc()`][daffodil.daf_select._iloc] does, in one query.

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

    Examples:
        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> items_sdaf = SqlDaf.from_daf(Daf(cols=['id', 'item', 'qty'], keyfield='id', lol=[['a1', 'pen', 3], ['b2', 'ink', ''], ['c3', 'pad', 7]]), name='items')
        >>> items_sdaf
        <BLANKLINE>
        | id | item | qty |
        | -: | ---: | --: |
        | a1 |  pen |   3 |
        | b2 |  ink |     |
        | c3 |  pad |   7 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name='items'
        <BLANKLINE>
        >>> items_sdaf.iloc(0)
        {'id': 'a1', 'item': 'pen', 'qty': 3}

        A negative position counts from the end:

        >>> items_sdaf.iloc(-1)
        {'id': 'c3', 'item': 'pad', 'qty': 7}

        Only some columns, in the order given, and as a list:

        >>> items_sdaf.iloc(1, include_cols=['qty', 'id'])
        {'qty': '', 'id': 'b2'}
        >>> items_sdaf.iloc(1, rtype='list')
        ['b2', 'ink', '']

        On a selection, the position is within its own rows:

        >>> items_sdaf.select_where_sql('qty > ?', [5]).iloc(0)
        {'id': 'c3', 'item': 'pad', 'qty': 7}
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


def _select_record(self: 'SqlDaf', key: Any, silent_error: bool = True) -> T_da:
    """
    Read one row by its key, as a dict, as [`Daf.select_record()`][daffodil.daf_select._select_record] does, in one query.

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

    Examples:
        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> items_sdaf = SqlDaf.from_daf(Daf(cols=['id', 'item', 'qty'], keyfield='id', lol=[['a1', 'pen', 3], ['b2', 'ink', ''], ['c3', 'pad', 7]]), name='items')
        >>> items_sdaf
        <BLANKLINE>
        | id | item | qty |
        | -: | ---: | --: |
        | a1 |  pen |   3 |
        | b2 |  ink |     |
        | c3 |  pad |   7 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield='id'; name='items'
        <BLANKLINE>
        >>> items_sdaf.select_record('c3')
        {'id': 'c3', 'item': 'pad', 'qty': 7}

        A missing key gives an empty dict, or raises with `silent_error=False`:

        >>> items_sdaf.select_record('z9')
        {}
        >>> items_sdaf.select_record('z9', silent_error=False)
        Traceback (most recent call last):
        KeyError: 'z9'
    """
    key_col: str = self._need_keyfield('select_record')
    rows_lot: List[Tuple[Any, ...]] = self._select_rows(f'{quote_name(key_col)} IS ?', (key,), limit=1)
    if not rows_lot:
        if silent_error:
            return {}
        raise KeyError(key)
    return dict(zip(self.hd, none_to_null(rows_lot[0])))


def _cols_daf(self: 'SqlDaf', cols_ls: T_ls) -> Daf:
    """ A Daf of only these columns, in row order. Names that are not columns are left out. Internal use. """
    cols_ls = [col for col in dict.fromkeys(cols_ls) if col in self.hd]
    lol: T_lola = []
    batch_lot: List[Tuple[Any, ...]]
    for batch_lot in self._iter_row_batches(cols=cols_ls):
        lol.extend(none_to_null(row_t) for row_t in batch_lot)
    return Daf(lol=lol, cols=cols_ls)


def _col(
        self: 'SqlDaf',
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
    Get one column as a list, by name, as [`Daf.col()`][daffodil.daf_select._col] does.

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

    Examples:
        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> sales_sdaf = SqlDaf.from_daf(Daf(cols=['id', 'g', 'qty', 'extra'], keyfield='id', lol=[
        ...     ['a1', 'x', 3, {'color': 'red'}], ['b2', 'y', '', {}], ['c3', 'x', 7, {'color': 'blue'}]]), name='sales')
        >>> sales_sdaf
        <BLANKLINE>
        | id | g  | qty |       extra       |
        | -: | -: | --: | ----------------: |
        | a1 |  x |   3 |  {'color': 'red'} |
        | b2 |  y |     |                {} |
        | c3 |  x |   7 | {'color': 'blue'} |
        <BLANKLINE>
        %% daf rows=3; cols=4; keyfield='id'; name='sales'
        <BLANKLINE>
        >>> sales_sdaf.col('qty')
        [3, '', 7]

        Each option, as in a Daf:

        >>> sales_sdaf.col('g', unique=True)
        ['x', 'y']
        >>> sales_sdaf.col('qty', omit_nulls=True)
        [3, 7]
        >>> sales_sdaf.col('qty', omit_nulls=True, astype=float)
        [3.0, 7.0]
        >>> sales_sdaf.col('price', silent_error=True)
        []

        A value inside a dict column, with a default for a dict that lacks it:

        >>> sales_sdaf.col('color', indirect_col='extra', default='none')
        ['red', 'none', 'blue']
    """
    return self.col_to_la(colname, unique=unique, omit_nulls=omit_nulls, silent_error=silent_error,
                          astype=astype, indirect_col=indirect_col, default=default)


def _col_to_la(
        self: 'SqlDaf',
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

    Examples:
        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> items_sdaf = SqlDaf.from_daf(Daf(cols=['id', 'item', 'qty'], keyfield='id', lol=[['a1', 'pen', 3], ['b2', 'ink', ''], ['c3', 'pad', 7]]), name='items')
        >>> items_sdaf.col_to_la('qty', omit_nulls=True)
        [3, 7]
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


def _icol(self: 'SqlDaf', icol: int) -> T_la:
    """
    Get one column as a list, by its place among the columns, as [`Daf.icol()`][daffodil.daf_select._icol] does.

    The place is looked up in the column names, so it does not depend on the rows. A
    negative place counts from the end. A place out of range raises IndexError, unless the
    table has no rows, as in a Daf.

    Args:
        icol: The place of the column, from 0.

    Returns:
        The values of the column, in row order.

    Raises:
        IndexError: The place is out of range, and the table has rows.

    Examples:
        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> items_sdaf = SqlDaf.from_daf(Daf(cols=['id', 'item', 'qty'], keyfield='id', lol=[['a1', 'pen', 3], ['b2', 'ink', ''], ['c3', 'pad', 7]]), name='items')
        >>> items_sdaf.icol(1)
        ['pen', 'ink', 'pad']
        >>> items_sdaf.icol(-1)
        [3, '', 7]
    """
    return self.icol_to_la(icol)


def _icol_to_la(self: 'SqlDaf', icol: int, unique: bool = False, omit_nulls: bool = False) -> T_la:
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

    Examples:
        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> items_sdaf = SqlDaf.from_daf(Daf(cols=['id', 'g', 'qty'], keyfield='id', lol=[['a1', 'x', 3], ['b2', 'y', ''], ['c3', 'x', 7]]), name='items')
        >>> items_sdaf.icol_to_la(1, unique=True)
        ['x', 'y']
        >>> items_sdaf.icol_to_la(2, omit_nulls=True)
        [3, 7]
    """
    cols_ls: T_ls = list(self.hd)
    if not -len(cols_ls) <= icol < len(cols_ls):
        if not self:
            return []
        raise IndexError(f"icol: column {icol} is out of range. '{self.name}' has {len(cols_ls)} columns.")
    return self.col_to_la(cols_ls[icol], unique=unique, omit_nulls=omit_nulls)


def _cols_to_dol(self: 'SqlDaf', colname1: str, colname2: str) -> Dict[Any, T_la]:
    """
    Map each value of one column to the values seen with it in another, as [`Daf.cols_to_dol()`][daffodil.daf_apply._cols_to_dol] does.

    Only the two columns are read into memory. If a name is not a column, or there are no
    rows, the result is empty.

    Args:
        colname1: The column of keys.
        colname2: The column of values.

    Returns:
        A dict from each value of the first column to the values of the second, in the order first seen.

    Examples:
        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> items_sdaf = SqlDaf.from_daf(Daf(cols=['id', 'g', 'item'], keyfield='id', lol=[['a1', 'x', 'pen'], ['b2', 'y', 'ink'], ['c3', 'x', 'pad']]), name='items')
        >>> items_sdaf.cols_to_dol('g', 'item')
        {'x': ['pen', 'pad'], 'y': ['ink']}
    """
    if colname1 not in self.hd or colname2 not in self.hd:
        return {}
    return self._cols_daf([colname1, colname2]).cols_to_dol(colname1, colname2)


def _to_lod(self: 'SqlDaf') -> List[T_da]:
    """
    Make a list of dicts, one for each row, as [`Daf.to_lod()`][daffodil.daf_convert._to_lod] does.

    Every row is read into memory. Use `iter_dict()` to go through the rows one at a time.

    Returns:
        The rows as dicts, in order.

    Examples:
        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> items_sdaf = SqlDaf.from_daf(Daf(cols=['id', 'item', 'qty'], keyfield='id', lol=[['a1', 'pen', 3], ['b2', 'ink', '']]), name='items')
        >>> items_sdaf.to_lod()
        [{'id': 'a1', 'item': 'pen', 'qty': 3}, {'id': 'b2', 'item': 'ink', 'qty': ''}]
    """
    return list(self.iter_dict())


def _to_dod(self: 'SqlDaf', remove_keyfield: bool = True) -> Dict[Any, T_da]:
    """
    Make a dict of dicts, named by the keyfield, as [`Daf.to_dod()`][daffodil.daf_convert._to_dod] does.

    Every row is read into memory, through `to_daf()`.

    Args:
        remove_keyfield: If True, leave the keyfield out of the inner dicts.

    Returns:
        Each row as a dict, under its key.

    Raises:
        KeysDisabledError: There is no keyfield, and there are rows.

    Examples:
        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> items_sdaf = SqlDaf.from_daf(Daf(cols=['id', 'item', 'qty'], keyfield='id', lol=[['a1', 'pen', 3], ['b2', 'ink', '']]), name='items')
        >>> items_sdaf.to_dod()
        {'a1': {'item': 'pen', 'qty': 3}, 'b2': {'item': 'ink', 'qty': ''}}
        >>> items_sdaf.to_dod(remove_keyfield=False)
        {'a1': {'id': 'a1', 'item': 'pen', 'qty': 3}, 'b2': {'id': 'b2', 'item': 'ink', 'qty': ''}}
    """
    return self.to_daf().to_dod(remove_keyfield=remove_keyfield)


def _to_json(self: 'SqlDaf', concise: bool = True) -> str:
    """
    Make JSON text that holds the whole table, as [`Daf.to_json()`][daffodil.daf_convert._to_json] does.

    Every row is read into memory, through `to_daf()`. The text holds the rows, the column
    names, the dtypes, the keyfield and the name. [`Daf.from_json()`][daffodil.daf_convert._from_json] reads it back as a Daf.

    Args:
        concise: If True, leave out the parts that are empty.

    Returns:
        The JSON text.

    Examples:
        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> items_sdaf = SqlDaf.from_daf(Daf(cols=['id', 'qty'], keyfield='id', lol=[['a1', 3], ['b2', '']]), name='items')
        >>> items_sdaf.to_json()
        '{"name": "items", "lol": [["a1", 3], ["b2", ""]], "hd": {"id": 0, "qty": 1}, "keyfield": "id"}'
        >>> items_sdaf.to_json(concise=False)
        '{"name": "items", "lol": [["a1", 3], ["b2", ""]], "hd": {"id": 0, "qty": 1}, "dtypes": {}, "keyfield": "id", "attrs": {}, "disp_cols": []}'
    """
    return self.to_daf().to_json(concise=concise)


def _daf_to_lol_summary(
        self: 'SqlDaf',
        max_rows:   int = 10,
        max_cols:   int = 10,
        disp_cols:  T_cs | None = None,
        ) -> T_lola:
    """
    Give the header and the rows to display, as [`Daf.daf_to_lol_summary()`][daffodil.daf_report._daf_to_lol_summary] does.

    With `max_rows`, only the first and last rows are read, around a row of '...'. Without
    it, every row is read.

    Args:
        max_rows: The most rows to show. 0 for all.
        max_cols: The most columns to show. 0 for all.
        disp_cols: Names to show in the header: a dict of column name to name shown, for some columns, or a list for every column.

    Returns:
        The header, then the rows.

    Examples:
        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> nums_sdaf = SqlDaf.from_daf(Daf(cols=['id', 'a', 'b', 'c'], lol=[[f'r{i}', i, i * 10, i * 100] for i in range(6)]), name='nums')
        >>> nums_sdaf.daf_to_lol_summary(max_rows=4, max_cols=0)
        [['id', 'a', 'b', 'c'], ['r0', 0, 0, 0], ['r1', 1, 10, 100], ['...', '...', '...', '...'], ['r4', 4, 40, 400], ['r5', 5, 50, 500]]
        >>> nums_sdaf.daf_to_lol_summary(max_rows=0, max_cols=2)
        [['id', '...', 'c'], ['r0', '...', 0], ['r1', '...', 100], ['r2', '...', 200], ['r3', '...', 300], ['r4', '...', 400], ['r5', '...', 500]]
        >>> nums_sdaf.daf_to_lol_summary(max_rows=2, max_cols=0, disp_cols={'a': 'A'})
        [['id', 'A', 'b', 'c'], ['r0', 0, 0, 0], ['...', '...', '...', '...'], ['r5', 5, 50, 500]]
    """
    colnames_ls: T_ls = daf_utils.display_colnames(list(self.hd), disp_cols)
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


def _to_md(
        self: 'SqlDaf',
        max_rows:           int = 0,
        max_cols:           int = 0,
        just:               str = '',
        shorten_text:       bool = True,
        max_text_len:       int = 80,
        smart_fmt:          bool = False,
        include_summary:    bool = False,
        disp_cols:          T_cs | None = None,
        header:             Any = None,
        ) -> str:
    """
    Make a Markdown table, as [`Daf.to_md()`][daffodil.daf_report._to_md] does.

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
        disp_cols: Names to show in the header: a dict of column name to name shown, for some columns, or a list for every column.
        header: A header to use in place of the column names.

    Returns:
        The Markdown text.

    Examples:
        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> items_sdaf = SqlDaf.from_daf(Daf(cols=['id', 'item', 'price', 'note'], keyfield='id', lol=[
        ...     ['a1', 'pen', 1.5, 'blue, fine tip'], ['b2', 'ink', 4.0, ''], ['c3', 'pad', 2.25, 'lined, 80 sheets of recycled paper']]), name='items')
        >>> print(items_sdaf.to_md())
        | id | item | price |                note                |
        | -: | ---: | ----: | ---------------------------------: |
        | a1 |  pen |   1.5 |                     blue, fine tip |
        | b2 |  ink |   4.0 |                                    |
        | c3 |  pad |  2.25 | lined, 80 sheets of recycled paper |
        <BLANKLINE>

        Only the first and last rows are read with `max_rows`:

        >>> print(items_sdaf.to_md(max_rows=2, max_cols=3))
        | id  | item | ... |                note                |
        | --: | ---: | --: | ---------------------------------: |
        |  a1 |  pen | ... |                     blue, fine tip |
        | ... |  ... | ... |                                ... |
        |  c3 |  pad | ... | lined, 80 sheets of recycled paper |
        <BLANKLINE>

        The other options act as in a Daf:

        >>> print(items_sdaf.to_md(just='<<><', max_text_len=12, smart_fmt=True))
        | id | item | price |     note     |
        | :- | :--- | ----: | :----------- |
        | a1 | pen  |   1.5 | blue,..e tip |
        | b2 | ink  |   4.0 |              |
        | c3 | pad  |   2.2 | lined..paper |
        <BLANKLINE>
        >>> print(items_sdaf.to_md(shorten_text=False, disp_cols={'item': 'Item'}, include_summary=True))
        | id | Item | price |                note                |
        | -: | ---: | ----: | ---------------------------------: |
        | a1 |  pen |   1.5 |                     blue, fine tip |
        | b2 |  ink |   4.0 |                                    |
        | c3 |  pad |  2.25 | lined, 80 sheets of recycled paper |
        <BLANKLINE>
        %% daf rows=3; cols=4; keyfield='id'; name='items'
        <BLANKLINE>
        >>> print(items_sdaf[0:1].to_md(header=['ID', 'Item', 'Price', 'Note']))
        | ID | Item | Price |      Note      |
        | -: | ---: | ----: | -------------: |
        | a1 |  pen |   1.5 | blue, fine tip |
        <BLANKLINE>
    """
    md_str: str = Daf.to_md(self, max_rows=max_rows, max_cols=max_cols, just=just, shorten_text=shorten_text,
                            max_text_len=max_text_len, smart_fmt=smart_fmt, include_summary=False,
                            disp_cols=disp_cols, header=header)
    if include_summary:
        md_str += f"\n%% daf rows={len(self)}; cols={len(self.hd)}; keyfield='{self.keyfield}'; name='{self.name}'\n"
    return md_str


def _to_md_cols(
        self: 'SqlDaf',
        max_rows:           int = 0,
        max_cols:           int = 0,
        just:               str = '',
        shorten_text:       bool = True,
        max_text_len:       int = 80,
        smart_fmt:          bool = False,
        include_summary:    bool = False,
        disp_cols:          T_cs | None = None,
        ) -> str:
    """
    Make a Markdown table with each row as a column, as [`Daf.to_md_cols()`][daffodil.daf_report._to_md_cols] does.

    Set `max_rows` for a large table. Then only the first and last rows are read.

    Args:
        max_rows: The most rows to show, the first and the last. 0 for all.
        max_cols: The most columns to show, the first and the last. 0 for all.
        just: The justification of each column: '<', '^' or '>'.
        shorten_text: If True, shorten long text, keeping the ends.
        max_text_len: The length past which text is shortened.
        smart_fmt: If True, limit the digits after the point in numbers.
        include_summary: Not used, as in a Daf.
        disp_cols: Names to show in the header: a dict of column name to name shown, for some columns, or a list for every column.

    Returns:
        The Markdown text.

    Examples:
        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> items_sdaf = SqlDaf.from_daf(Daf(cols=['id', 'item', 'price', 'note'], keyfield='id', lol=[
        ...     ['a1', 'pen', 1.5, 'blue, fine tip'], ['b2', 'ink', 4.0, ''], ['c3', 'pad', 2.25, 'lined, 80 sheets of recycled paper']]), name='items')
        >>> print(items_sdaf.to_md_cols())
        |  id   |       a1       | b2  |                 c3                 |
        | item  |      pen       | ink |                pad                 |
        | price |      1.5       | 4.0 |                2.25                |
        | note  | blue, fine tip |     | lined, 80 sheets of recycled paper |
        <BLANKLINE>
        >>> print(items_sdaf.to_md_cols(max_rows=2, just='<<<', max_text_len=12, smart_fmt=True, disp_cols={'id': 'ID'}))
        | ID    | a1           | ... | c3           |
        | item  | pen          | ... | pad          |
        | price | 1.5          | ... | 2.2          |
        | note  | blue,..e tip | ... | lined..paper |
        <BLANKLINE>
        >>> print(items_sdaf.to_md_cols(max_cols=2, shorten_text=False))
        |  id  |       a1       | b2  |                 c3                 |
        | ...  |      ...       | ... |                ...                 |
        | note | blue, fine tip |     | lined, 80 sheets of recycled paper |
        <BLANKLINE>
    """
    return Daf.to_md_cols(self, max_rows=max_rows, max_cols=max_cols, just=just, shorten_text=shorten_text,
                          max_text_len=max_text_len, smart_fmt=smart_fmt, include_summary=include_summary,
                          disp_cols=disp_cols)


def _iloc_exists(self: 'SqlDaf', irow: int) -> bool:
    """
    Say whether there is a row at this position, from 0.

    Args:
        irow: The position, from 0.

    Returns:
        True if a row is there.

    Raises:
        ValueError: This is a parent that has had rows deleted. See SqlDaf.

    Examples:
        >>> import tempfile
        >>> SqlDaf.db_path = tempfile.mkdtemp() + '/job.db'
        >>> items_sdaf = SqlDaf.from_daf(Daf(cols=['id', 'qty'], keyfield='id', lol=[['a1', 3], ['b2', '']]), name='items')
        >>> items_sdaf.iloc_exists(1), items_sdaf.iloc_exists(2)
        (True, False)
    """
    self._check_positions('iloc_exists')
    rows_sql, rows_params = self._rows_sql()
    if self._is_dense():
        return self._execute(f'SELECT 1 FROM ({rows_sql}) WHERE {ORD_COL} = ?',
                                  rows_params + (irow + 1,)).fetchone() is not None
    cursor: sqlite3.Cursor = self._execute(
            f'SELECT 1 FROM ({rows_sql}) ORDER BY {ORD_COL} LIMIT 1 OFFSET ?', rows_params + (irow,))
    return cursor.fetchone() is not None
