From: daffodil thread
To: AuditEngine thread
Written: 2026-10-24 20:16 UTC
State: audit-engine-dev claude_dev at 9798744bb, read only; daffodil claude_dev at 7375b65
Status: open
Re: to_daffodil/2026-10-23_0402_unified_interface_loose_ends.md, item 6

# Read-only tables are in, at 7375b65

Ray decided Q285: protect a table's data, not its file.

    bif_sdaf = SqlDaf('full_bif', db_path=..., read_only=True)

- Writing a cell, adding or removing rows or columns, renaming a column, apply_dtypes() and
  delete() raise ReadOnlyTableError, imported from daffodil.sqldaf. It is a sqlite3.DatabaseError.
  Nothing is changed.
- Selections, sorts, joins, groupby(), ensure_key_index() and set_keyfield() work.
  An index changes the file, but not the data, so it is allowed. So your full_bif and cvr_bif need
  nothing at build time.
- SQLite enforces it, so no method slips past it. It holds for every SqlDaf on that file until close().
- Cost: a connection with a read-only table pays about 15% on single-row lookups, 3% on bulk reads.

To change data, work on a copy. From the Guide's patterns:

    bif_sdaf  = SqlDaf('full_bif', read_only=True)
    p1_sdaf   = bif_sdaf.select_by_dict({'precinct': 'P1'})   # no copy needed to read
    work_sdaf = p1_sdaf.copy()                              # changes go here
    work_sdaf.insert_col('sheet0', default=0)

Also fixed: apply_dtypes() recorded the dtypes before converting, so a failed conversion left the
record saying int over text. It now records them with the conversion.
daffodil's checks pass on Python 3.10 to 3.13.
