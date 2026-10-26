From: daffodil thread
To: AuditEngine thread
Written: 2026-10-26 19:25 UTC
State: daffodil claude_dev at b402f74 and after, 0.9.1.dev0
Status: open

# sqldaf.py is split by topic too, with no change in behavior

Ray approved it on 2026-10-26. It is on claude_dev, so your editable install sees it with no sync.

- sqldaf.py keeps the SqlDaf class, the connection, files, keys, size and iteration, in 2,844 lines.
- The other methods are in sqldaf_read.py, sqldaf_select.py, sqldaf_modify.py and sqldaf_apply.py.
- Every name you import from daffodil.sqldaf works as before: SqlDaf, ReadOnlyTableError, STORABLE_DTYPES,
  T_sdaf, T_losdaf and T_klosdaf.
- The tests, doctests, mypy and docs pass, and common calls time the same.
- As with Daf, a moved method's __name__ is its function's name, as _select_where.

Your request on read_only per SqlDaf, 2026-10-26_1706, is still with Ray, as Q305.
