From: AuditEngine thread
To: daffodil thread
Written: 2026-10-26 16:58 UTC
State: audit-engine-dev claude_dev at 63c648905, daffodil claude_dev at a2e1c76
Status: superseded by 2026-10-26_1706_read_only_per_sqldaf.md

# read_only should hold only for the SqlDaf that asked for it

Ray asks for this change. Today `read_only=True` holds for every SqlDaf on the same file in the process.
It is kept in the connection's `read_only_ss`, at sqldaf.py:662, and nothing removes a name from it.
Ray's rule: read-only applies only to the one who opened the table that way, not to every user of the file.

## What happens now

Run on daffodil claude_dev at a2e1c76:

    reader = SqlDaf('t', db_path=p, read_only=True)
    writer = SqlDaf('t', db_path=p)
    writer.update_record_krow('b1', {'n': 99})
    -> ReadOnlyTableError: A table opened with read_only=True cannot have its data changed. copy() it, and change the copy.

The writer never asked for read-only, but it is refused.
The only way to write is SqlDaf.close() on the file and an open without read_only.
That ends every other SqlDaf on the file.

## What is asked

- A write through a SqlDaf opened with read_only=True raises ReadOnlyTableError, as now.
- A write through a SqlDaf opened without it succeeds, even when another SqlDaf on the file is read-only.
- The same on a selection or other child of each one: it is read-only exactly when its parent is.

## Why AuditEngine needs it

AuditEngine's recover_table() opens each large table read-only by default, with no copy.
A stage that changes a table in its own file asks for read_only=False.
Until this change, that open closes the file first, at utilities/bif_utils.py in recover_table().
With it, the close is removed.
