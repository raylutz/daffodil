# Display options for to_md(): col_fmt, and the show= ideas

Started 2026-11-02. Ray raised three ideas for clearer tables in the docs and in reports.

## col_fmt: built, Q331

Ray approved it on 2026-11-02.

- `to_md(col_fmt={'share': '.1%', ...})` formats the cells of the named columns, for display only.
- A value is a Python format spec, `'smart'`, `'smart%'` or a function.
- `'smart%'` is Ray's request. A plain spec such as `.0%` shows a tiny share as 0%.
  smart_fmt adapts the decimals to the size, so 10 variants in 100,000 show as 0.0100%.
- NULL and None stay as they are. A value a spec cannot format is shown as it is, with no error.
  A function's own errors are not caught.
- smart_fmt is applied by col_fmt's code to the other columns, and then turned off for the writer.
  Otherwise the writer would reformat a formatted cell, as '.4f' becoming two decimals.
- A name that is not a column raises KeyError, once per call.
- The rows from `daf_to_lol_summary()` are the Daf's own rows. The formatted rows are new lists.
- The column at each shown position comes from `daf_to_lol_summary(max_rows=1, max_cols=...)`,
  so hidden columns and `disp_cols` names do not matter.
- Code: daf_report.py `_format_md_rows()` and `_format_md_cell()`. Tests: test_daf_md.py, test_sqldaf.py.

Before building, AuditEngine was searched for column formatting before `to_md()`. The grep found
no clear pattern of it, so the design rests on Ray's account that AuditEngine needs it often.

## show=: not decided

Ray's idea is an opt-in `show=` for `to_md()`, by name or by bit flags like the COPY flags:

- types: each cell shows its type name only, as str or dict. Then the text of a dict and a dict differ,
  and a column declared int that holds text shows str.
- quotes: text values are quoted, other values are not. Reading such a table back needs an indicator,
  such as `show='quotes'` in the summary line. `_parse_daf_footer()` already keeps keys it does not know.
- ids: the references, to show shared rows. `id()` changes each run, so a doctest cannot check it.
  It would need short labels kept the same across two tables.

A prototype of each, by transforming cells before `to_md()`, is in the chat of 2026-11-01. The
default output must not change.

## Shared example tables: not decided

Ray's idea is to make example tables once and reuse them. A Guide page already runs as one doctest.
Each docstring runs alone, so it would need the tables from a doctest_namespace fixture, made fresh
for each docstring. Then a change in one example cannot reach another. The cost is that an example
can no longer be pasted and run alone.
