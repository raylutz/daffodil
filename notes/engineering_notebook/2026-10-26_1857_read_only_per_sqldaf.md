# read_only for one SqlDaf, not for the file

Started 2026-10-26. AuditEngine asked, for Ray, in mail 2026-10-26_1706_read_only_per_sqldaf.md, that
`read_only=True` hold only for the SqlDaf that asked for it. Q305. Research and a prototype only.

## What the code does now

- `SqlDaf(name, read_only=True)` adds the table name to the connection's `read_only_ss`, at sqldaf.py
  in the constructor, and sets an authorizer on the connection, `_read_only_authorizer()`.
- Every SqlDaf on the file shares one connection, so the table is read-only for all of them until
  `SqlDaf.close()`. Nothing removes a name from `read_only_ss`.
- Q285 chose the authorizer so that no method slips past it, `sql()` included. Indexes and TEMP
  tables are allowed: the data is protected, not the file.

## The difficulty

The authorizer sees the connection and the statement. It does not see which SqlDaf sent it.

## Prototype: switch the authorizer with the SqlDaf that uses the connection

In a scratch copy, outside the repo:

- `read_only` is an attribute of each SqlDaf. A selection takes its parent's.
- `conn` becomes a property. When a read-only SqlDaf exists on the connection, reading `conn` sets the
  authorizer for the SqlDaf that reads it: one that refuses writes to its own table if it is read-only,
  or none if it is not. It is set again only when that changes.
- First try: one authorizer that looked up the SqlDaf that last read `conn`. It let a read-only write
  through. SQLite calls the authorizer only when it prepares a statement, and Python's sqlite3 keeps
  prepared statements. The writer had prepared the same UPDATE, and the reader reused it unchecked.
- Setting an authorizer makes SQLite prepare its statements again. So switching it is what makes the
  check hold. Checked: after the switch, the reader's UPDATE was refused.
- No method keeps the connection in a local variable, so every statement reads `self.conn` when it runs.

Results, on a file with tables t and u, a reader and a writer on t:

| Call | Prototype | Original |
| :- | :- | :- |
| writer.update_record_krow | ok | ReadOnlyTableError |
| reader.update_record_krow | ReadOnlyTableError | ReadOnlyTableError |
| reader[0:2].update_record_krow | ReadOnlyTableError | ReadOnlyTableError |
| writer[0:2].update_record_krow | ok | ReadOnlyTableError |
| reader.sql('UPDATE t ...') | DatabaseError: not authorized | the same |
| writer.sql('UPDATE t ...') | ok | DatabaseError: not authorized |
| reader.sql('UPDATE u ...') | ok | ok |
| reader.ensure_key_index() | ok | ok |
| reader.copy().update_record_krow | ok | ok |
| writer.select_records_daf(reader, inverse=True) | ok | ok |

Cost, timed with 20,000 rows: with one SqlDaf, nothing measurable. Alternating a reader and a writer,
roughly 5 to 15 us per switch, for SQLite to prepare statements again. A `select_record()` takes about 40 us.

## Rejected

- A check in Python in each method that writes. `sql()` and any method missed would slip past, which
  Q285 ruled out.
- A connection of its own for each read-only SqlDaf. A reader could then not be used in one statement
  with a SqlDaf on the other connection, as in `writer.select_records_daf(reader)`, AuditEngine's compare.

## Found

- `sql()` raises the bare `DatabaseError: not authorized` for a refused write, where other methods raise
  ReadOnlyTableError. This is so in the original too.

## Decided and built

Ray chose the prototype's way, with `sql()` raising ReadOnlyTableError too, on 2026-10-26, Q305.

Built as prototyped, with one change found by the check on Python 3.10: before Python 3.11,
`set_authorizer(None)` does not remove the authorizer, and every statement was refused. A writer
gets `_allow_all`, an authorizer that allows every statement, in its place.

- `read_only` is a class attribute, False, set True on a SqlDaf opened with it. A selection copies its parent's.
- `conn` is a property over `_conn`. The connection's shared state holds `has_read_only` and `guard`, the
  table the authorizer now protects, or None.
- `_read_only_authorizer()` takes the one guarded table. `read_only_ss` is gone.
- Five new tests: a writer and a reader on one table, a statement the writer prepared first, selections
  of each, `sql()`, and the compare pattern. The 14 earlier read-only tests pass unchanged.
- The Guide's SqlDaf page shows a reader and a writer on one table.
- SqlDaf calls with no read-only SqlDaf timed the same as before, within noise.
