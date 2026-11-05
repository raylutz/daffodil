# For the mail to AuditEngine at the next release

Items to include when the next version is released. Clear this list once the mail is sent.
The list was last cleared on 2026-10-26, when 0.9.0 was mailed.

- `cols` of SqlDaf `select_where()`, `split_where()` and `group_where()` is deprecated and undocumented, Q323. It still works, so sdaf_variants.py needs no change. It can be dropped from those four calls at any time, at a small cost on wide tables.
- `select_records()` is the new name of `select_records_daf()`, Q324. The old name still works, so no call needs to change now. AuditEngine has about 65 calls of the old name, to rename at any time.
- A SqlDaf prints sdaf in its summary line, as `%% sdaf rows=3; ...`, Q325. A Daf still prints daf. A test that compares the printout of a SqlDaf with that of a Daf sees the difference.
- `to_md(col_fmt=...)` formats some columns for display, by a spec such as '.1%', 'smart', 'smart%' or a function, Q331. It can replace formatted copies made with f-strings before a report table.
- `smart_fmt` drops trailing zeros, Q332: 0.50 becomes 0.5, 0.0100 becomes 0.01, and 2.0 becomes 2. A report or test that compares smart_fmt text sees the difference.
- `to_md(show=...)` shows the types: 'dtypes' in the header, with a note for columns not yet converted, 'quotes' on text cells, and 'types' as type names in the cells. `from_md()` reads the dtypes and quotes back, Q333 and Q339.
