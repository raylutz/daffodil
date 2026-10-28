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
        The table used in the examples:

        >>> daf = Daf(cols=['precinct', 'party'], lol=[
        ...     ['P1', 'R'], ['P1', 'D'], ['P2', 'D'], ['P2', ''],
        ...     ['P1', 'D'], ['P2', 'R'], ['P2', 'R'], ['P1', 'D']])

        Every column:

        >>> daf.daf_valuecount()
        {'precinct': {'P1': 4, 'P2': 4}, 'party': {'R': 3, 'D': 4, '': 1}}

        One column. The others are in the result, with NULL, the empty string:

        >>> daf.daf_valuecount(cols=['party'])
        {'precinct': '', 'party': {'R': 3, 'D': 4, '': 1}}
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
        The table used in the examples, of ballots by precinct and party. One ballot has no party:

        >>> daf = Daf(cols=['precinct', 'party'], lol=[
        ...     ['P1', 'R'], ['P1', 'D'], ['P2', 'D'], ['P2', ''],
        ...     ['P1', 'D'], ['P2', 'R'], ['P2', 'R'], ['P1', 'D']])
        >>> daf
        <BLANKLINE>
        | precinct | party |
        | -------: | ----: |
        |       P1 |     R |
        |       P1 |     D |
        |       P2 |     D |
        |       P2 |       |
        |       P1 |     D |
        |       P2 |     R |
        |       P2 |     R |
        |       P1 |     D |
        <BLANKLINE>
        %% daf rows=8; cols=2; keyfield=''; name=''
        <BLANKLINE>

        The values come in the order first seen. The empty cell is counted as `''`:

        >>> daf.valuecounts_for_colname('party')
        {'R': 3, 'D': 4, '': 1}

        With `sort=True`, the most common value comes first. With `reverse=False` as well, the least
        common does:

        >>> daf.valuecounts_for_colname('party', sort=True)
        {'D': 4, 'R': 3, '': 1}
        >>> daf.valuecounts_for_colname('party', sort=True, reverse=False)
        {'': 1, 'R': 3, 'D': 4}

        Leave out the empty cells:

        >>> daf.valuecounts_for_colname('party', omit_nulls=True)
        {'R': 3, 'D': 4}

        A column that does not exist gives an empty dict, with no error:

        >>> daf.valuecounts_for_colname('color')
        {}
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
        The table used in the examples:

        >>> daf = Daf(cols=['precinct', 'party'], lol=[
        ...     ['P1', 'R'], ['P1', 'D'], ['P2', 'D'], ['P2', ''],
        ...     ['P1', 'D'], ['P2', 'R'], ['P2', 'R'], ['P1', 'D']])

        Every column, with no list given:

        >>> daf.valuecounts_for_colnames_ls()
        {'precinct': {'P1': 4, 'P2': 4}, 'party': {'R': 3, 'D': 4, '': 1}}

        Some columns, each sorted with the most common first, and without the empty cells:

        >>> daf.valuecounts_for_colnames_ls(['party'], sort=True, omit_nulls=True)
        {'party': {'D': 4, 'R': 3}}
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
        The table used in the examples:

        >>> daf = Daf(cols=['precinct', 'party'], lol=[
        ...     ['P1', 'R'], ['P1', 'D'], ['P2', 'D'], ['P2', ''],
        ...     ['P1', 'D'], ['P2', 'R'], ['P2', 'R'], ['P1', 'D']])

        Count the parties in precinct P1 only:

        >>> daf.valuecounts_for_colname_selectedby_colname('party', 'precinct', 'P1')
        {'R': 1, 'D': 3}

        The same, sorted with the most common first:

        >>> daf.valuecounts_for_colname_selectedby_colname('party', 'precinct', 'P1', sort=True)
        {'D': 3, 'R': 1}
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
        The table used in the examples, with a column for how each ballot was cast:

        >>> daf = Daf(cols=['precinct', 'party', 'how'], lol=[
        ...     ['P1', 'R', 'poll'], ['P1', 'D', 'mail'], ['P2', 'D', 'poll'], ['P2', '', 'poll'],
        ...     ['P1', 'D', 'mail'], ['P2', 'R', 'mail'], ['P2', 'R', 'mail'], ['P1', 'D', 'mail']])

        Count two columns, in precinct P2 only:

        >>> daf.valuecounts_for_colnames_ls_selectedby_colname(['party', 'how'], 'precinct', 'P2')
        {'party': {'D': 1, '': 1, 'R': 2}, 'how': {'poll': 2, 'mail': 2}}

        With each count sorted, the most common first:

        >>> daf.valuecounts_for_colnames_ls_selectedby_colname(['party', 'how'], 'precinct', 'P2', sort=True)
        {'party': {'R': 2, 'D': 1, '': 1}, 'how': {'poll': 2, 'mail': 2}}
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
        The table used in the examples:

        >>> daf = Daf(cols=['precinct', 'party'], lol=[
        ...     ['P1', 'R'], ['P1', 'D'], ['P2', 'D'], ['P2', ''],
        ...     ['P1', 'D'], ['P2', 'R'], ['P2', 'R'], ['P1', 'D']])

        The parties counted in each precinct:

        >>> daf.valuecounts_for_colname1_groupedby_colname2('party', 'precinct')
        {'P1': {'R': 1, 'D': 3}, 'P2': {'D': 1, '': 1, 'R': 2}}

        The same, with each count sorted, the most common first:

        >>> daf.valuecounts_for_colname1_groupedby_colname2('party', 'precinct', sort=True)
        {'P1': {'D': 3, 'R': 1}, 'P2': {'R': 2, 'D': 1, '': 1}}

        To check that one column depends on another, count it by the other. Here each style belongs to
        one precinct, so each precinct shows one style:

        >>> styles = Daf(cols=['precinct', 'style'], lol=[['P1', 's1'], ['P2', 's2'], ['P1', 's1']])
        >>> styles.valuecounts_for_colname1_groupedby_colname2('style', 'precinct')
        {'P1': {'s1': 2}, 'P2': {'s2': 1}}
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
        The table used in the examples, of ballots, their precinct and a count:

        >>> daf = Daf(cols=['ballot', 'precinct', 'n'], lol=[['b1', 'P1', 3], ['b2', 'P1', 5], ['b2', 'P2', 10]])

        Each profile measures something else. The result is a dict for each column, with many entries.
        These examples pick out the main ones.

        An `index` column should not repeat. `num_within_reps` counts the repeats.
        `within_reps_loti` lists each repeat as `(0, row)`, where row is its position. The first
        time a value appears is not listed. Here b2 at row 2 repeats b2 at row 1:

        >>> stats = daf.gen_stats_daf([('ballot', str, '', 'index')])['ballot']
        >>> stats['num_uniques'], stats['num_within_reps'], stats['within_reps_loti']
        (2, 1, [(0, 2)])

        An `attrib` column is counted by value:

        >>> daf.gen_stats_daf([('precinct', str, '', 'attrib')])['precinct']['val_counts']
        {'P1': 2, 'P2': 1}

        A `scalar` column gives its range, mean and standard deviation:

        >>> stats = daf.gen_stats_daf([('n', int, '', 'scalar')])['n']
        >>> stats['min'], stats['max'], stats['mean'], round(stats['stdev'], 2)
        (3, 10, 6, 3.61)

        A `localidx` column holds whole numbers, and says whether they run in sequence:

        >>> stats = daf.gen_stats_daf([('n', int, '', 'localidx')])['n']
        >>> stats['all_ints'], stats['sequential']
        (True, False)

        Several columns at once, each with its own profile:

        >>> sorted(daf.gen_stats_daf([('precinct', str, '', 'attrib'), ('n', int, '', 'scalar')]))
        ['n', 'precinct']
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
    Make a Markdown table of the Daf, as text.

    Use it to put a table into a report or a log, as in `daf[0:50].to_md()`. It writes every row
    and column, with no summary line, unless asked. That is the difference from `str()`, which
    is what `print(daf)` and an f-string give: `str()` keeps 10 rows and 10 columns and adds the
    summary line. A SqlDaf gives the same text, and reads only the rows it shows.

    The parameters shape the table:

    - `max_rows` and `max_cols` keep the first and last rows and columns, and put `...` in place
      of the middle. 0 means no limit.
    - `just` sets the justification of each column, one character for each: `<` left, `^` center,
      `>` right. The default is right for every column.
    - Text longer than `max_text_len` loses its middle, marked `..`, so both ends show.
      `shorten_text=False` writes it in full.
    - `smart_fmt=True` shows numbers for people to read. A whole number gets commas, as 12,345. A
      float above 100 is rounded to a whole number. Above 1 it keeps one decimal, above 0.1 two,
      above 0.01 three, and above 0.001 four. Only the text is changed, not the Daf.
    - `include_summary=True` adds the summary line that `str()` shows, starting with `%%`. With it,
      `from_md()` can read the table back, as text.
    - `disp_cols` names the columns differently in the header. The Daf keeps its own names.
      `header` replaces the header row as it is shown, so it must match the columns shown. With
      `max_cols`, use `disp_cols`, which stays matched to the columns it names.

    A Daf with no column names gets `A`, `B` and so on in the header, as a spreadsheet does. A
    ragged Daf, with rows of different lengths, is shown with blank cells in its short rows, and
    the Daf is not changed. Then the summary line ends with `ragged rows detected`.

    Args:
        max_rows: The most rows to show, the first and the last. 0 for all.
        max_cols: The most columns to show, the first and the last. 0 for all.
        just: The justification of each column, one of `<`, `^` or `>` for each.
        shorten_text: If True, shorten text that is longer than `max_text_len`.
        max_text_len: The longest text to show in full.
        smart_fmt: If True, show floats with fewer digits.
        include_summary: If True, add a line with the size, keyfield and name.
        disp_cols: Names to show in the header in place of the column names.
        header: A header row to show in place of the one made from the column names.

    Returns:
        The Markdown text, ending with a newline.

    Examples:
        The table used in the examples:

        >>> daf = Daf(cols=['id', 'item', 'price', 'note'], keyfield='id', lol=[
        ...     ['a1', 'pen', 1.5, 'blue, fine tip'],
        ...     ['b2', 'ink', 4.0, ''],
        ...     ['c3', 'pad', 2.25, 'lined, 80 sheets of recycled paper'],
        ...     ['d4', 'cap', 0.333333, 'red'],
        ...     ['e5', 'tag', 10.0, 'paper']])
        >>> print(daf.to_md())
        | id | item |  price   |                note                |
        | -: | ---: | -------: | ---------------------------------: |
        | a1 |  pen |      1.5 |                     blue, fine tip |
        | b2 |  ink |      4.0 |                                    |
        | c3 |  pad |     2.25 | lined, 80 sheets of recycled paper |
        | d4 |  cap | 0.333333 |                                red |
        | e5 |  tag |     10.0 |                              paper |
        <BLANKLINE>

        Keep the first and last 2 rows, and the first and last column:

        >>> print(daf.to_md(max_rows=4, max_cols=2))
        | id  | ... |      note      |
        | --: | --: | -------------: |
        |  a1 | ... | blue, fine tip |
        |  b2 | ... |                |
        | ... | ... |            ... |
        |  d4 | ... |            red |
        |  e5 | ... |          paper |
        <BLANKLINE>

        Justify the columns left, center, right and right:

        >>> print(daf[0:2].to_md(just='<^>>'))
        | id | item | price |      note      |
        | :- | :--: | ----: | -------------: |
        | a1 | pen  |   1.5 | blue, fine tip |
        | b2 | ink  |   4.0 |                |
        <BLANKLINE>

        Shorten long text to 20 characters, or write it in full:

        >>> print(daf[:, ['id', 'note']].to_md(max_text_len=20))
        | id |         note         |
        | -: | -------------------: |
        | a1 |       blue, fine tip |
        | b2 |                      |
        | c3 | lined, 80..led paper |
        | d4 |                  red |
        | e5 |                paper |
        <BLANKLINE>
        >>> print(daf[:, ['id', 'note']].to_md(max_text_len=20, shorten_text=False))
        | id |                note                |
        | -: | ---------------------------------: |
        | a1 |                     blue, fine tip |
        | b2 |                                    |
        | c3 | lined, 80 sheets of recycled paper |
        | d4 |                                red |
        | e5 |                              paper |
        <BLANKLINE>

        Show numbers for people to read. The number of decimals depends on the size:

        >>> sizes = Daf(cols=['n'], lol=[[12345], [1234.56], [2.25], [0.333333], [0.0456]])
        >>> print(sizes.to_md(smart_fmt=True))
        |   n    |
        | -----: |
        | 12,345 |
        |  1,235 |
        |    2.2 |
        |   0.33 |
        |  0.046 |
        <BLANKLINE>

        Add the summary line:

        >>> print(daf[0:2].to_md(include_summary=True))
        | id | item | price |      note      |
        | -: | ---: | ----: | -------------: |
        | a1 |  pen |   1.5 | blue, fine tip |
        | b2 |  ink |   4.0 |                |
        <BLANKLINE>
        %% daf rows=2; cols=4; keyfield='id'; name=''
        <BLANKLINE>

        Show other names in the header. `disp_cols` stays matched to its columns when some are
        hidden. `header` is the header row as shown, so with `max_cols` it would need `...` too:

        >>> print(daf[0:2].to_md(disp_cols=['ID', 'Item', 'Price', 'Note']))
        | ID | Item | Price |      Note      |
        | -: | ---: | ----: | -------------: |
        | a1 |  pen |   1.5 | blue, fine tip |
        | b2 |  ink |   4.0 |                |
        <BLANKLINE>
        >>> print(daf[0:2].to_md(disp_cols=['ID', 'Item', 'Price', 'Note'], max_cols=2))
        | ID | ... |      Note      |
        | -: | --: | -------------: |
        | a1 | ... | blue, fine tip |
        | b2 | ... |                |
        <BLANKLINE>
        >>> print(daf[0:2].to_md(header=['ID', '...', 'Note'], max_cols=2))
        | ID | ... |      Note      |
        | -: | --: | -------------: |
        | a1 | ... | blue, fine tip |
        | b2 | ... |                |
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
        The table used in the examples:

        >>> daf = Daf(cols=['id', 'item', 'price', 'note'], lol=[
        ...     ['a1', 'pen', 1.5, 'blue, fine tip'],
        ...     ['b2', 'ink', 4.0, ''],
        ...     ['c3', 'pad', 2.25, 'lined, 80 sheets of recycled paper']])

        Each row of the Daf becomes a column. The first column holds the column names. The cells are
        centered:

        >>> print(daf.to_md_cols())
        |  id   |       a1       | b2  |                 c3                 |
        | item  |      pen       | ink |                pad                 |
        | price |      1.5       | 4.0 |                2.25                |
        | note  | blue, fine tip |     | lined, 80 sheets of recycled paper |
        <BLANKLINE>

        `max_rows` limits the rows of the Daf, which are the columns here, and `max_cols` the columns
        of the Daf, which are the lines. The middle ones become `...`:

        >>> print(daf.to_md_cols(max_rows=2, max_cols=2))
        |  id  |       a1       | ... |                 c3                 |
        | ...  |      ...       | ... |                ...                 |
        | note | blue, fine tip | ... | lined, 80 sheets of recycled paper |
        <BLANKLINE>

        `just` has one character for each column of the text: the names, then one for each row. Here
        text is shortened to 12 characters, a float is shown with fewer digits, and the names are
        replaced:

        >>> print(daf.to_md_cols(just='<<<<', max_text_len=12, smart_fmt=True, disp_cols=['ID', 'Item', 'Price', 'Note']))
        | ID    | a1           | b2  | c3           |
        | Item  | pen          | ink | pad          |
        | Price | 1.5          | 4.0 | 2.2          |
        | Note  | blue,..e tip |     | lined..paper |
        <BLANKLINE>

        `shorten_text=False` writes long text in full, whatever `max_text_len` is:

        >>> notes = Daf(cols=['n'], lol=[['a very long note text']])
        >>> print(notes.to_md_cols(max_text_len=10, shorten_text=False))
        | n  | a very long note text |
        <BLANKLINE>
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
        The table used in the examples:

        >>> daf = Daf(cols=['id', 'item', 'qty'], lol=[['a1', 'pen', 3], ['b2', 'ink', 10], ['c3', 'pad', 7]])

        The column names come first, then the rows:

        >>> daf.daf_to_lol_summary()
        [['id', 'item', 'qty'], ['a1', 'pen', 3], ['b2', 'ink', 10], ['c3', 'pad', 7]]

        Keep 2 rows, the first and the last, with `...` between them:

        >>> daf.daf_to_lol_summary(max_rows=2)
        [['id', 'item', 'qty'], ['a1', 'pen', 3], ['...', '...', '...'], ['c3', 'pad', 7]]

        Keep 2 columns, the first and the last. With only `max_cols`, every row is kept:

        >>> daf.daf_to_lol_summary(max_rows=0, max_cols=2)
        [['id', '...', 'qty'], ['a1', '...', 3], ['b2', '...', 10], ['c3', '...', 7]]

        Other names for the columns:

        >>> daf.daf_to_lol_summary(disp_cols=['ID', 'Item', 'Qty'])
        [['ID', 'Item', 'Qty'], ['a1', 'pen', 3], ['b2', 'ink', 10], ['c3', 'pad', 7]]

        An odd limit keeps the extra row at the start:

        >>> big = Daf(cols=['n'], lol=[[i] for i in range(10)])
        >>> big.daf_to_lol_summary(max_rows=3)
        [['n'], [0], [1], ['...'], [9]]
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
        A dict, as a table of keys and values, left justified:

        >>> print(Daf.dict_to_md({'pen': 3, 'ink': 'out of stock'}))
        | key |    value     |
        | :-- | :----------- |
        | pen | 3            |
        | ink | out of stock |
        <BLANKLINE>

        Other names for the two columns, and right justified:

        >>> print(Daf.dict_to_md({'pen': 3, 'ink': 'out of stock'}, cols=['item', 'qty'], just='>>'))
        | item |     qty      |
        | ---: | -----------: |
        |  pen |            3 |
        |  ink | out of stock |
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
        The table used in the examples:

        >>> daf = Daf(cols=['precinct', 'party'], lol=[
        ...     ['P1', 'R'], ['P1', 'D'], ['P2', 'D'], ['P2', ''],
        ...     ['P1', 'D'], ['P2', 'R'], ['P2', 'R'], ['P1', 'D']])

        Each value and its count, in the order first seen:

        >>> daf.value_counts_daf('party')
        <BLANKLINE>
        | party | counts |
        | ----: | -----: |
        |     R |      3 |
        |     D |      4 |
        |       |      1 |
        <BLANKLINE>
        %% daf rows=3; cols=2; keyfield=''; name=''
        <BLANKLINE>

        Sorted with the most common first, without the empty cells, and with a total. The total's
        label is `**Total**`, which shows in bold in Markdown:

        >>> daf.value_counts_daf('party', sort=True, omit_nulls=True, include_total=True)
        <BLANKLINE>
        |   party   | counts |
        | --------: | -----: |
        |         D |      4 |
        |         R |      3 |
        | **Total** |      7 |
        <BLANKLINE>
        %% daf rows=3; cols=2; keyfield=''; name=''
        <BLANKLINE>

        With `reverse=False`, the least common comes first:

        >>> daf.value_counts_daf('party', sort=True, reverse=False)
        <BLANKLINE>
        | party | counts |
        | ----: | -----: |
        |       |      1 |
        |     R |      3 |
        |     D |      4 |
        <BLANKLINE>
        %% daf rows=3; cols=2; keyfield=''; name=''
        <BLANKLINE>
    """

    value_counts_di   = self.valuecounts_for_colname(colname=colname, sort=sort, reverse=reverse)

    if omit_nulls:
        daf_utils.safe_del_key(value_counts_di, '')

    value_counts_daf = Daf.from_lod_to_cols([value_counts_di], cols=[colname, 'counts'])

    if include_total:
        value_counts_daf.append({colname: ' **Total** ', 'counts': sum(value_counts_daf.col('counts'))})

    return value_counts_daf
