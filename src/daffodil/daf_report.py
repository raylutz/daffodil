# daf_report.py
# copyright (c) 2026 Ray Lutz
"""
Methods of Daf for statistics and reporting.

Each function here takes the Daf as its first argument, or the class for a classmethod.
It is assigned in the Daf class body under its Daf name, without the underscore.
The docs show it under that name.

Some functions use names defined in daf.py, such as Daf, when they run.
daf.py sets those names here, as listed in _NAMES_FROM_DAF, once they exist.
An import at the top would be circular, since daf.py imports this module.
"""

from __future__ import annotations

from daffodil.lib.daf_types import T_cs, T_da, T_di, T_doda, T_dodi, T_lola, T_lota, T_ls, T_ma
import daffodil.lib.daf_utils    as daf_utils
import daffodil.lib.daf_md       as md

from typing import Tuple, Dict, Any, cast, TYPE_CHECKING

if TYPE_CHECKING:       # for mypy. daf.py sets the names below when the program runs.
    from daffodil.daf import Daf

# The names of daf.py that the functions here use when they run.
# daf.py sets each one here once it exists. An import at the top would be circular.
_NAMES_FROM_DAF: Tuple[str, ...] = ('Daf',)


def _daf_valuecount(
        self: 'Daf',
        by:     str         = 'row',
        cols:   T_cs | None = None,
        ) -> T_ma:
    """
    Count how often each value occurs, in each column, using `reduce()` and `count_values_kl()`.

    Args:
        by: How the function is applied. See `reduce()`.
        cols: The columns to count. All if None.

    Returns:
        A dict that maps each column to a dict of value and count. A column that is
        not in `cols` is empty.

    Examples:
        >>> daf = Daf(lol=[['a', 1, 10], ['b', 2, 20], ['a', 3, 30]], cols=['g', 'x', 'y'])
        >>> daf
        <BLANKLINE>
        | g  | x  | y  |
        | -: | -: | -: |
        |  a |  1 | 10 |
        |  b |  2 | 20 |
        |  a |  3 | 30 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.daf_valuecount(cols=['g'])['g']
        {'a': 2, 'b': 1}
    """

    # by='row' (the default, and the only mode exercised) always reduces to a single record.
    return cast(T_ma, self.reduce(func=type(self).count_values_kl, by=by, cols=cols))


def _valuecounts_for_colname(
        self: 'Daf',
        colname:    str,
        sort:       bool=False,
        reverse:    bool=True,
        omit_nulls: bool=False,
        ) -> Dict[Any, int]:
    """
    Count how often each value occurs in one column.

    With `sort=True` the dict is ordered from the most common value to the least.
    Use `reverse=False` for the other way. A column that does not exist gives an
    empty dict. An empty cell is counted as the empty string, unless
    `omit_nulls` is True.

    Args:
        colname: The column to count.
        sort: If True, order by count.
        reverse: With `sort`, True puts the most common first.
        omit_nulls: If True, leave out the count of empty cells.

    Returns:
        A dict that maps each value to its count.

    Examples:
        >>> daf = Daf(lol=[['a'], ['b'], ['a'], ['']], cols=['g'])
        >>> daf
        <BLANKLINE>
        | g  |
        | -: |
        |  a |
        |  b |
        |  a |
        |    |
        <BLANKLINE>
        %% daf rows=4; cols=1; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.valuecounts_for_colname('g', sort=True, omit_nulls=True)
        {'a': 2, 'b': 1}
    """

    valuecounts_di: T_di = {}

    if colname not in self.hd:
        return {}

    icol = self.hd[colname]

    for irow in range(len(self.lol)):
        val = self.lol[irow][icol]
        if val not in valuecounts_di:
            valuecounts_di[val] = 1
        else:
            valuecounts_di[val] += 1

    if omit_nulls:
        daf_utils.safe_del_key(valuecounts_di, '')

    if sort:
        valuecounts_di = dict(sorted(valuecounts_di.items(), key=lambda x: x[1], reverse=reverse))

    return valuecounts_di


def _valuecounts_for_colnames_ls(
        self: 'Daf',
        colnames_ls:    T_cs | None = None,
        sort:           bool=False,
        reverse:        bool=True,
        omit_nulls:     bool=False,
        ) -> Dict[str, Dict[Any, int]]:
    """
    Count how often each value occurs, in each of several columns.

    Args:
        colnames_ls: The columns to count. All if None.
        sort: If True, order each count by size.
        reverse: With `sort`, True puts the most common first.
        omit_nulls: If True, leave out the count of empty cells.

    Returns:
        A dict that maps each column to a dict of value and count.

    Examples:
        >>> daf = Daf(lol=[['a', 'x'], ['b', 'x']], cols=['g', 'h'])
        >>> daf
        <BLANKLINE>
        | g  | h  |
        | -: | -: |
        |  a |  x |
        |  b |  x |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.valuecounts_for_colnames_ls()
        {'g': {'a': 1, 'b': 1}, 'h': {'x': 2}}
    """

    if not colnames_ls:
        colnames_ls = self.columns()

    colnames_ls = cast(T_ls, colnames_ls)

    valuecounts_dodi: T_dodi = {}

    for colname in colnames_ls:
        valuecounts_dodi[colname] = \
            self.valuecounts_for_colname(colname, sort=sort, reverse=reverse, omit_nulls=omit_nulls)

    return valuecounts_dodi


def _valuecounts_for_colname_selectedby_colname(
        self: 'Daf',
        colname: str,
        selectedby_colname: str,
        selectedby_colvalue: str,
        sort: bool = False,
        reverse: bool = True,
        ) -> Dict[Any, int]:
    """
    Count the values of a column, in the rows where another column has a value.

    Args:
        colname: The column to count.
        selectedby_colname: The column to test.
        selectedby_colvalue: Only rows where that column equals this are counted.
        sort: If True, order the counts by size.
        reverse: With `sort`, True puts the most common first.

    Returns:
        A dict that maps each value to its count. It is empty if a column does not exist.

    Examples:
        >>> daf = Daf(lol=[['a', 'x'], ['b', 'x'], ['a', 'y']], cols=['g', 'h'])
        >>> daf
        <BLANKLINE>
        | g  | h  |
        | -: | -: |
        |  a |  x |
        |  b |  x |
        |  a |  y |
        <BLANKLINE>
        %% daf rows=3; cols=2; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.valuecounts_for_colname_selectedby_colname('g', 'h', 'x')
        {'a': 1, 'b': 1}
    """


    valuecounts_di: T_di = {}

    if colname not in self.hd or selectedby_colname not in self.hd:
        return {}

    icol = self.hd[colname]
    selectedby_colidx = self.hd[selectedby_colname]

    for irow in range(len(self.lol)):
        val = self.lol[irow][selectedby_colidx]
        if val != selectedby_colvalue:
            continue
        val = self.lol[irow][icol]
        if val not in valuecounts_di:
            valuecounts_di[val] = 1
        else:
            valuecounts_di[val] += 1

    if sort:
        valuecounts_di = dict(sorted(valuecounts_di.items(), key=lambda x: x[1], reverse=reverse))

    return valuecounts_di


def _valuecounts_for_colnames_ls_selectedby_colname(
        self: 'Daf',
        colnames_ls: T_cs | None = None,
        selectedby_colname: str = '',
        selectedby_colvalue: str = '',
        sort: bool = False,
        reverse: bool = True,
        ) -> Dict[str, Dict[Any, int]]:

    """
    Count the values of several columns, in the rows where another column has a value.

    Args:
        colnames_ls: The columns to count. All if None.
        selectedby_colname: The column to test.
        selectedby_colvalue: Only rows where that column equals this are counted.
        sort: If True, order the counts by size.
        reverse: With `sort`, True puts the most common first.

    Returns:
        A dict that maps each column to a dict of value and count.

    Examples:
        >>> daf = Daf(lol=[['a', 'x', 1], ['a', 'y', 2], ['b', 'x', 3]], cols=['g', 'k', 'n'])
        >>> daf
        <BLANKLINE>
        | g  | k  | n  |
        | -: | -: | -: |
        |  a |  x |  1 |
        |  a |  y |  2 |
        |  b |  x |  3 |
        <BLANKLINE>
        %% daf rows=3; cols=3; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.valuecounts_for_colnames_ls_selectedby_colname(['k'], 'g', 'a')
        {'k': {'x': 1, 'y': 1}}
        >>> daf.valuecounts_for_colnames_ls_selectedby_colname(['g', 'k'], 'g', 'a')
        {'g': {'a': 2}, 'k': {'x': 1, 'y': 1}}
    """


    if not colnames_ls:
        colnames_ls = self.columns()

    colnames_ls = cast(T_ls, colnames_ls)

    valuecounts_dodi: T_dodi = {}

    for colname in colnames_ls:
        valuecounts_dodi[colname] = \
            self.valuecounts_for_colname_selectedby_colname(
                    colname,
                    selectedby_colname,
                    selectedby_colvalue,
                    sort=sort,
                    reverse=reverse,
                    )

    return valuecounts_dodi


def _valuecounts_for_colname1_groupedby_colname2(
        self: 'Daf',
        colname1: str,
        groupedby_colname2: str,
        sort: bool = False,
        reverse: bool = True,
        ) -> Dict[Any, Dict[Any, int]]:
    """
    Count the values of one column, for each value of another.

    The data is read once. Use it to see whether two columns relate one to one: each
    group should then hold a single value.

    Args:
        colname1: The column whose values are counted.
        groupedby_colname2: The column whose values form the groups.
        sort: If True, order each count by size.
        reverse: With `sort`, True puts the most common first.

    Returns:
        A dict that maps each value of the second column to a dict of value and count.

    Examples:
        >>> daf = Daf(lol=[['a', 'x'], ['b', 'x'], ['a', 'y']], cols=['g', 'h'])
        >>> daf
        <BLANKLINE>
        | g  | h  |
        | -: | -: |
        |  a |  x |
        |  b |  x |
        |  a |  y |
        <BLANKLINE>
        %% daf rows=3; cols=2; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.valuecounts_for_colname1_groupedby_colname2('g', 'h')
        {'x': {'a': 1, 'b': 1}, 'y': {'a': 1}}
    """


    valuecounts_dodi: T_dodi = {}

    if colname1 not in self.hd or groupedby_colname2 not in self.hd:
        return {}

    icol1 = self.hd[colname1]
    groupedby_col2idx = self.hd[groupedby_colname2]

    for row in self.lol:
        groupval = row[groupedby_col2idx]
        val = row[icol1]
        if groupval not in valuecounts_dodi:
            valuecounts_dodi[groupval] = {}
        valuecounts_di: T_di = valuecounts_dodi[groupval]
        if val not in valuecounts_di:
            valuecounts_di[val] = 1
        else:
            valuecounts_di[val] += 1

    if sort:
        # for group, valuecounts_di in valuecounts_dodi.items():
            # valuecounts_dodi[group] = dict(sorted(valuecounts_di.items(), key=lambda x: x[1], reverse=reverse))

        valuecounts_dodi = {group: dict(sorted(valuecounts_di.items(), key=lambda x: x[1], reverse=reverse))
                                for group, valuecounts_di in valuecounts_dodi.items()}


    return valuecounts_dodi


def _gen_stats_daf(self: 'Daf', col_def_lot: T_lota) -> T_doda:

    """
    Work out statistics for columns, given a profile for each.

    `col_def_lot` has a tuple for each column of interest. The tuple is the column
    name, a type, a format, and a profile. The profile is one of `index`,
    `attrib`, `file_paths`, `scalar` or `localidx`, and chooses what is measured.
    An index looks for repeats, an attribute counts the values, and a scalar gives
    the minimum, maximum, mean and standard deviation. The type and format are not
    used.

    Args:
        col_def_lot: A list of tuples of column name, type, format and profile.

    Returns:
        A dict that maps each column name to its statistics, as a dict.

    Raises:
        NotImplementedError: A profile is not one of the five.

    Examples:
        >>> daf = Daf(lol=[[1], [3]], cols=['n'])
        >>> daf
        <BLANKLINE>
        | n  |
        | -: |
        |  1 |
        |  3 |
        <BLANKLINE>
        %% daf rows=2; cols=1; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.gen_stats_daf([('n', int, '', 'scalar')])['n']['mean']
        2
    """

    info_dod = {}

    # from utilities import daf_utils

    for col_def_ta in col_def_lot:
        col_name, col_dtype, col_format, col_profile = col_def_ta

        col_data_la = self.col(col_name)                                       # perflint-reviewed (loop-invariant-statement)

        info_dod[col_name] = daf_utils.list_stats(col_data_la, profile=col_profile)     # perflint-reviewed (loop-invariant-statement)

    return info_dod


def _md_daf_table_snippet(
        self: 'Daf',
        ) -> str:
    """
    Make a short Markdown table of the Daf, with a summary line.

    This is what `str()` shows. It keeps at most `md_max_rows` rows and
    `md_max_cols` columns, 10 by default. Longer text is shortened to 80
    characters.

    If the Daf has no rows, it is empty, and it shows no header. The summary line says
    `rows=0; cols=0`, because the number of columns is counted from the rows.
    `columns()` still gives the names.

    Returns:
        The Markdown text.

    Examples:
        >>> daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['x', 'y'])
        >>> print(daf.md_daf_table_snippet(), end='')
        | x  | y  |
        | -: | -: |
        |  1 |  a |
        |  2 |  b |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield=''; name=''

        An empty Daf has no rows and no columns, so only the summary line shows. This is so even if it has column names:

        >>> Daf(cols=['x', 'y'])
        <BLANKLINE>
        <BLANKLINE>
        %% daf rows=0; cols=0; keyfield=''; name=''
        <BLANKLINE>
    """

    return self.to_md(
            max_rows        = self.md_max_rows,
            max_cols        = self.md_max_cols,
            shorten_text    = True,
            max_text_len    = 80,
            smart_fmt       = False,
            include_summary = True,
            )


def _to_md(
        self: 'Daf',
        max_rows:       int     = 0,         # limit number of rows by keeping leading and trailing rows and omitting middle rows.
        max_cols:       int     = 0,         # limit number of cols by keeping leading and trailing cols and omitting middle cols.
        just:           str     = '',        # provide the justification for each column, using <, ^, > meaning left, center, right justified.
        shorten_text:   bool    = True,      # if the text in any field is more than the max_text_len, then shorten by keeping the ends and redacting the center text.
        max_text_len:   int     = 80,        # see above.
        smart_fmt:      bool    = False,     # if columns are numeric, then limit the number of figures right of the decimal to "smart" numbers.
        include_summary: bool   = False,     # include a one-line summary after the table, describing shape, keyfield, name
        disp_cols:      T_cs | None = None,    # use these column names instead of those defined in daf.
        header:         T_cs | None = None,    # use this header instead.
        ) -> str:
            
    """
    Make a Markdown table of the Daf.

    Use it to put a table into a report or a log, as in `daf[0:50].to_md()`. An f-string of a
    Daf gives `str()`, which keeps only 10 rows and 10 columns and adds a summary line. `to_md()`
    writes every row and column, and leaves the summary line out, unless asked. So it gives the
    rows selected, as most code wants. A SqlDaf gives the same text, reading only those rows.

    Without limits the whole table is written. With `max_rows` or `max_cols`, the
    first and last are kept, and the middle is replaced by `...`. Text longer
    than `max_text_len` is shortened by cutting out its middle. `just` has one
    character for each column: `<` left, `^` center, `>` right. The default is
    right. With no column names, `A`, `B` and so on are used. With `include_summary`,
    the Markdown can be read back with `from_md()`, as text.

    A ragged Daf, with rows of different lengths, is shown with its short rows padded
    with blank cells. The Daf itself is not changed. With `include_summary`, the summary
    line ends with `ragged rows detected`. `from_md()` reads such a table back as
    rectangular.

    Use `max_rows` and `max_cols` together, or neither. With only `max_cols`, a row of
    `...` is added under the header by mistake.

    Args:
        max_rows: The most rows to show. 0 for all.
        max_cols: The most columns to show. 0 for all.
        just: The justification of each column.
        shorten_text: If True, shorten text that is longer than `max_text_len`.
        max_text_len: The longest text to show in full.
        smart_fmt: If True, show numbers with fewer decimal places.
        include_summary: If True, add a line with the size, keyfield and name.
        disp_cols: Column names to show instead of the real ones.
        header: A header to use instead.

    Returns:
        The Markdown text.

    Examples:
        >>> daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'v'])
        >>> print(daf.to_md())
        | id | v  |
        | -: | -: |
        |  1 |  a |
        |  2 |  b |
        <BLANKLINE>
    """

    daf_lol = self.daf_to_lol_summary(max_rows=max_rows, max_cols=max_cols, disp_cols=disp_cols)

    header_exists = bool(self.hd)

    if header_exists and daf_lol:
        # daf_to_lol_summary embeds colnames as the first row; pull it back out and
        # pass it explicitly via `header` rather than relying on the data's first row
        # (daffodil never treats the first row of a lol as an implicit header).
        embedded_header = daf_lol[0]
        data_lol = daf_lol[1:]
    elif daf_lol:
        # Daffodil arrays may have no header at all (a bare lol). A Markdown table needs a
        # header row and a separator row, and Daf.from_md() requires them, so write
        # spreadsheet-style names (A, B, C, ...) in the header. The names are for the
        # text only. The Daf is not given names.
        embedded_header = daf_utils._generate_spreadsheet_column_names_list(len(daf_lol[0]))
        data_lol = daf_lol
        header_exists = True
    else:
        embedded_header = None
        data_lol = daf_lol

    final_header = header if header is not None else embedded_header

    mdstr = md.md_lol_table(
        data_lol,
        header              = final_header,
        just                = just or ('>' * len(data_lol[0])) if data_lol else just,
        omit_header         = not header_exists,
        shorten_text        = shorten_text,
        max_text_len        = max_text_len,
        smart_fmt           = smart_fmt,

        )
    if include_summary:
        parts = []
        parts.append(f"rows={self.num_rows()}")
        parts.append(f"cols={self.num_cols()}")
        parts.append(f"keyfield='{self.keyfield or ''}'")
        parts.append(f"name='{self.name or ''}'")

        schema = self.attrs.get('schema')
        if schema:
            parts.append(f"schema='{schema}'")
        if not self.is_rectangular():
            parts.append("ragged rows detected")

        mdstr += "\n%% daf " + "; ".join(parts) + "\n"

    # if include_summary:
    #     # (compatible with tuple keys.)
    #     mdstr += f"\n\\[{self.num_rows():,} rows x {self.num_cols():,} cols; keyfield='{self.keyfield}'; {len(self._kd):,} keys ] ({self.name or type(self).__name__})\n"
    return mdstr


def _to_md_cols(
        self: 'Daf',
        max_rows:       int     = 0,         # limit the maximum number of row by keeping leading and trailing rows.
        max_cols:       int     = 0,         # limit the maximum number of cols by keeping leading and trailing cols.
        just:           str     = '',        # provide the justification for each column, using <, ^, > meaning left, center, right justified.
        shorten_text:   bool    = True,      # if the text in any field is more than the max_text_len, then shorten by keeping the ends and redacting the center text.
        max_text_len:   int     = 80,        # see above.
        smart_fmt:      bool    = False,     # if columns are numeric, then limit the number of figures right of the decimal to "smart" numbers.
        include_summary: bool   = False,     # include a one-line summary after the table.
        disp_cols:      T_ls | None = None, # use these column names instead of those defined in daf.
        ) -> str:
    """
    Make a Markdown table in which each row of the Daf is a column.

    There is no header row and no separator row. The first column holds the column names of the Daf.
    Use it for a Daf with few rows and many columns. Some Markdown renderers, such as Python-Markdown,
    show this text as plain text, because they need a header and a separator row to make a table.

    Args:
        max_rows: The most rows to show. 0 for all.
        max_cols: The most columns to show. 0 for all.
        just: The justification of each column.
        shorten_text: If True, shorten text that is longer than `max_text_len`.
        max_text_len: The longest text to show in full.
        smart_fmt: If True, show numbers with fewer decimal places.
        include_summary: Not used.
        disp_cols: Column names to show instead of the real ones.

    Returns:
        The Markdown text.

    Examples:
        >>> daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['x', 'y'])
        >>> print(daf.to_md_cols(), end='')
        | x  | 1  | 2  |
        | y  | a  | b  |
    """

    daf_lol = self.daf_to_lol_summary(max_rows=max_rows, max_cols=max_cols, disp_cols=disp_cols)

    #header_exists = bool(self.hd)
    #breakpoint() #temp

    mdstr = md.md_cols_lol_table(
            cols_lol        = daf_lol,
            header          = None,
            just            = just,
            omit_header     = True,
            shorten_text    = shorten_text,
            max_text_len    = max_text_len,
            smart_fmt       = smart_fmt,
            )
    # if include_summary:
        # mdstr += f"\n\[{len(self.lol)} rows x {len(self.hd)} cols; keyfield={self.keyfield}; {len(self._kd)} keys ] ({type(self).__name__})\n"
    return mdstr


def _daf_to_lol_summary(
        self: 'Daf', 
        max_rows: int=10, 
        max_cols: int=10, 
        disp_cols: T_cs | None =None,
        ) -> T_lola:

    # from utilities import daf_utils

    # first build a basic summary by adding colnames, if they exist.
    """
    Make a list of lists for display, with the column names first.

    If there are more rows or columns than the limits, the first and last are
    kept and the middle is replaced by `...`. The limit is the number of data rows
    kept, and an odd limit keeps the extra row at the start. A limit of 0 means no
    limit, so with only `max_cols` all the rows are kept. The rows are not copied.

    Args:
        max_rows: The most rows to keep. 0 for no limit.
        max_cols: The most columns to keep. 0 for no limit.
        disp_cols: Column names to use instead of the real ones.

    Returns:
        The rows, with a header row first if there are column names.

    Examples:
        >>> daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['x', 'y'])
        >>> daf
        <BLANKLINE>
        | x  | y  |
        | -: | -: |
        |  1 |  a |
        |  2 |  b |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.daf_to_lol_summary()
        [['x', 'y'], [1, 'a'], [2, 'b']]
        >>> big = Daf(lol=[[i, i] for i in range(20)], cols=['x', 'y'])
        >>> big
        <BLANKLINE>
        |  x  |  y  |
        | --: | --: |
        |   0 |   0 |
        |   1 |   1 |
        |   2 |   2 |
        |   3 |   3 |
        |   4 |   4 |
        | ... | ... |
        |  15 |  15 |
        |  16 |  16 |
        |  17 |  17 |
        |  18 |  18 |
        |  19 |  19 |
        <BLANKLINE>
        %% daf rows=20; cols=2; keyfield=''; name=''
        <BLANKLINE>
        >>> big.daf_to_lol_summary(max_rows=4)
        [['x', 'y'], [0, 0], [1, 1], ['...', '...'], [18, 18], [19, 19]]
    """

    if disp_cols:
        if isinstance(disp_cols, list):
            colnames_ls = disp_cols
        else:
            colnames_ls = list(disp_cols)
    else:
        colnames_ls = list(self.hd.keys())

    colnames_ls = cast(list, colnames_ls)    

    result_lol = self.lol
    if colnames_ls:
        result_lol = [colnames_ls] + result_lol

    # no limits, return summary. A ragged Daf is shown with its short rows padded, as copies.
    if not max_rows and not max_cols:
        return daf_utils.pad_ragged_lol(result_lol)

    num_rows    = self.num_rows()
    num_cols    = self.num_cols()

    if not max_rows or num_rows <= max_rows:
        # Get all the rows, but potentially limit columns
        result_lol = daf_utils.reduce_lol_cols(result_lol, max_cols=max_cols)

    else:
        # Get the first and last portion of rows. An odd limit gives the extra row to the first part.
        # The last part is cut from the front, because a slice of [-0:] would be every row.

        first_lol   = self.lol[:(max_rows + 1)//2]
        last_lol    = self.lol[num_rows - max_rows//2:]
        divider_lol = [['...'] * num_cols]

        result_lol  = [colnames_ls] + first_lol + divider_lol + last_lol
        result_lol  = daf_utils.reduce_lol_cols(result_lol, max_cols=max_cols)

    return result_lol


def _dict_to_md(da: T_da, cols: T_ls | None = None, just: str='<<') -> str:
    """
    Show a dict as a two column Markdown table, for looking at it.

    This is a static method. Use it as `print(Daf.dict_to_md(my_da))`. The keys are
    in the first column and the values in the second.

    Args:
        da: The dict.
        cols: The two column names. Default `key` and `value`.
        just: The justification of the two columns.

    Returns:
        The Markdown text.

    Examples:
        >>> print(Daf.dict_to_md({'a': 1, 'b': 'two'}))
        | key | value |
        | :-- | :---- |
        | a   | 1     |
        | b   | two   |
        <BLANKLINE>
    """
    if not cols:
        cols = ['key', 'value']

    return Daf.from_lod_to_cols([da], cols=cols).to_md(just=just)


def _value_counts_daf(self: 'Daf',
        colname: str,                       # column name to include in the value_counts table
        sort: bool=False,                   # sort values in the category
        reverse: bool=True,                 # reverse the sort
        include_total: bool=False,          #
        omit_nulls: bool=False,             # set to true if '' should be omitted.
        ) -> 'Daf':
    """
    Make a Daf that lists each value of a column and its count.

    The columns are the name of the column, and `counts`. There is no keyfield.
    With `include_total` a last row holds the total.

    Args:
        colname: The column to count.
        sort: If True, order by count.
        reverse: With `sort`, True puts the most common first.
        include_total: If True, add a row with the total.
        omit_nulls: If True, leave out the count of empty cells.

    Returns:
        The new Daf.

    Examples:
        >>> daf = Daf(lol=[['a'], ['b'], ['a']], cols=['g'])
        >>> daf
        <BLANKLINE>
        | g  |
        | -: |
        |  a |
        |  b |
        |  a |
        <BLANKLINE>
        %% daf rows=3; cols=1; keyfield=''; name=''
        <BLANKLINE>
        >>> daf.value_counts_daf('g', sort=True)
        <BLANKLINE>
        | g  | counts |
        | -: | -----: |
        |  a |      2 |
        |  b |      1 |
        <BLANKLINE>
        %% daf rows=2; cols=2; keyfield=''; name=''
        <BLANKLINE>
    """

    value_counts_di   = self.valuecounts_for_colname(colname=colname, sort=sort, reverse=reverse)

    if omit_nulls:
        daf_utils.safe_del_key(value_counts_di, '')

    value_counts_daf = Daf.from_lod_to_cols([value_counts_di], cols=[colname, 'counts'])

    if include_total:
        value_counts_daf.append({colname: ' **Total** ', 'counts': sum(value_counts_daf.col('counts'))})

    return value_counts_daf
