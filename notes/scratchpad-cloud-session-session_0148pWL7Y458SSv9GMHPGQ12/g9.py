import sys
sys.path.insert(0, '/tmp/claude-0/-home-user-daffodil/95e36c09-64f8-59ba-9bd6-c962257ad9e3/scratchpad')
from setdoc import setdoc
M = {}
M['_from_md'] = r'''
Make a Daf from a Markdown table. This reads what `to_md()` writes.

Only the first table in the text is read. Text before it, such as headings and
prose, is skipped. A table row starts and ends with `|`. The table must have
a header row and then a separator row such as `| -: | -: |`. The columns are
the header cells.

A footer line that starts with `%% daf ` and follows the table may give the
keyfield and name, as `to_md(include_summary=True)` writes. Every cell is read as
text. Convert them with `apply_dtypes()`. Text with no table, or empty text,
gives an empty Daf.

Args:
    md_str: The Markdown text.

Returns:
    The new Daf.

Raises:
    RuntimeError: A table is found but it has no header row and separator row.

Examples:
    >>> from daffodil.daf import Daf
    >>> text = "| id | v |\n| -: | -: |\n|  1 | a |\n|  2 | b |\n\n%% daf rows=2; cols=2; keyfield='id'; name='nm'\n"
    >>> d = Daf.from_md(text)
    >>> d.lol, d.keyfield, d.name
    ([['1', 'a'], ['2', 'b']], 'id', 'nm')
'''
M['dodaf_to_md'] = r'''
Make one Markdown report from a dict of Daf instances.

This is a static method. The report has an optional first level heading, then
a second level section for each key, in the order of the dict. Each section
holds that Daf as a table, from its own `to_md()`. A Daf with no rows still gets
its section, with a line `*(no rows)*`, so you can see which sections were empty.
Use `dodaf_from_md()` to read the report back.

Args:
    dodaf: A dict that maps a section name to a Daf.
    report_header: The title of the report. Left out if empty.
    **to_md_kwargs: Passed on to the `to_md()` of each Daf, such as `max_rows`.

Returns:
    The Markdown text.

Examples:
    >>> from daffodil.daf import Daf
    >>> print(Daf.dodaf_to_md({'first': Daf(lol=[[1]], cols=['a'])}, report_header='Report'))
    # Report
    <BLANKLINE>
    <BLANKLINE>
    ## first
    <BLANKLINE>
    <BLANKLINE>
    | a |
    | -: |
    | 1 |
    <BLANKLINE>
    <BLANKLINE>
'''
M['_dodaf_from_md'] = r'''
Read a report made by `dodaf_to_md()` back into a dict of Daf instances.

A section starts at a line that begins with the marker of `header_level`,
which is `## ` by default. Text before the first section, such as the report
title, is skipped. Each section becomes a Daf, read with `from_md()`, so the
cells are text. The section title is the dict key and the `name` of the Daf, unless the table's
footer gives a name. The footer wins for the name, never for the key. Empty
text gives an empty dict.

Args:
    md_str: The Markdown text.
    header_level: The number of `#` that start a section heading.

Returns:
    A dict that maps each section title to a Daf.

Examples:
    >>> from daffodil.daf import Daf
    >>> report = Daf.dodaf_to_md({'first': Daf(lol=[[1]], cols=['a'])})
    >>> back = Daf.dodaf_from_md(report)
    >>> back['first'].lol, back['first'].name
    ([['1']], 'first')
'''
setdoc('src/daffodil/lib/daf_md.py', M)
