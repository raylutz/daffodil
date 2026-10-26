From: AuditEngine thread
To: daffodil thread
Written: 2026-10-26 17:06 UTC
State: audit-engine-dev claude_dev at e2694df55, daffodil claude_dev at a2e1c76
Status: open

# read_only should hold only for the SqlDaf that asked for it

Ray asks for this change. It replaces 2026-10-26_1658_read_only_per_sqldaf.md, which lacked the second part.

## The ask

Today `read_only=True` holds for every SqlDaf on the same file in the process.
It is kept in the connection's `read_only_ss`, at sqldaf.py:662, and nothing removes a name from it.
Ray's rule: read-only applies only to the one who opened the table that way, not to every user of the file.

Run on daffodil claude_dev at a2e1c76:

    reader = SqlDaf('t', db_path=p, read_only=True)
    writer = SqlDaf('t', db_path=p)
    writer.update_record_krow('b1', {'n': 99})
    -> ReadOnlyTableError: A table opened with read_only=True cannot have its data changed. copy() it, and change the copy.

The writer never asked for read-only, but it is refused.
The only way to write is SqlDaf.close() on the file and an open without read_only.
That ends every other SqlDaf on the file.

Asked:
- A write through a SqlDaf opened with read_only=True raises ReadOnlyTableError, as now.
- A write through a SqlDaf opened without it succeeds, even when another SqlDaf on the file is read-only.
- A selection or other child of a SqlDaf is read-only exactly when its parent is.

## For your information: the rule AuditEngine now follows

Ray set it on 2026-10-26. Nothing is implied. A copy is made only when the caller asks for one.

- recover_table() opens a large table in its own file, read-only by default. It makes no copy.
- A stage that writes into the table's own file opens it with read_only=False.
- A stage that changes a table only for its own work makes the copy itself, with copy() or from_db().
- AuditEngine keeps its chunks in step with a table it changes. Daffodil knows nothing of chunks, and need not.

Before e2694df55, recover_table() copied every large table into a scratch file, about 7 s for 1.6 million rows.
The copy existed so that tables could be changed and compared on one connection.
Most callers only read, so the copy was wasted.

AuditEngine's read_only=False path closes the file first, in recover_table() in utilities/bif_utils.py.
With the change asked above, that close is removed.
