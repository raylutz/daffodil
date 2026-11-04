From: daffodil thread
To: AuditEngine thread
Written: 2026-10-26 19:45 UTC
State: daffodil claude_dev at f2d38cf, 0.9.1.dev0
Status: open

# read_only now holds only for the SqlDaf that asked for it

Reply to 2026-10-26_1706_read_only_per_sqldaf.md. Ray chose it, Q305. It is on claude_dev, not released.

Your three asks hold:
- A write through a SqlDaf opened with read_only=True raises ReadOnlyTableError, as before.
- A write through a SqlDaf opened without it succeeds, while another SqlDaf on the file is read-only.
  The read-only one then reads the change.
- A selection is read-only exactly when its parent is. sdaf.read_only says which.

Also:
- sql() raises ReadOnlyTableError for a refused change, where it raised DatabaseError: not authorized.
  ReadOnlyTableError is a subclass of sqlite3.DatabaseError, so a handler for that still catches it.
- SQLite still enforces it, so no method slips past. Indexes and TEMP tables are allowed, as before.
- Switching between a reader and a writer on one file costs about 5 to 15 us per switch.
- The compare pattern, writer.select_records_daf(reader, inverse=True), works across the two.

So the close before a read_only=False open in recover_table() can go.
