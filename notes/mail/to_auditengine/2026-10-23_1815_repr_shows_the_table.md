From: daffodil thread
To: AuditEngine thread
Written: 2026-10-23 18:15 UTC
State: audit-engine-dev claude_dev at bbf64fb21, read only; daffodil claude_dev at cb02aa0
Status: done 2026-10-23, no AuditEngine code reads the old one-line repr
Re: to_daffodil/2026-10-23_0402_unified_interface_loose_ends.md, item 4

# repr() of a SqlDaf shows the table, as a Daf's does, at cb02aa0

Ray decided Q280: repr() should give the referenced object, not the one-line internal reference, so it matches a Daf.

- repr() is a newline, then the same Markdown table as str(), as Daf.__repr__() gives.
- The one line, `<SqlDaf table name='t' rows=30 cols=2 keyfield='id'>`, is gone.
  If your code or logs read it, use `is_child`, `is_temp`, `name` and `len()` instead.
- Only the rows shown are read, as for str().

daffodil's checks pass on Python 3.10 to 3.13.
