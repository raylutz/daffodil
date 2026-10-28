From: AuditEngine thread
To: daffodil thread
Written: 2026-10-28 03:26 UTC
State: audit-engine-dev claude_dev at 01c29c7bb, daffodil claude_dev at 09c22be
Status: open

# A join brings in a table from another file by itself

Ray settled this on 2026-10-28, as Q438.

## The ask

When the other SqlDaf of `join()` is in another database file, the join handles it.
The caller writes the same line as for a Daf:

    r = bia_sdaf.join(cvr_sdaf, how='left')
    r.save('biacvr_bif', db_path=biacvr_path)

There are three tables:
- The primary, whose `join()` runs. It may be opened with read_only=True.
- The other, in its own file. That file is only read.
- The result, kept with `save()` into a file of its own.

## How it works

1. The other table is copied into a TEMP table on the primary's connection.
   Its file is attached read-only for the copy only, as `from_db()` does now.
   A TEMP table is in no database file, so neither file changes.
2. The join is returned unresolved, as now. The copy stays while the join is unresolved.
3. When the join is first read, it resolves into its TEMP result. The copy is deleted then.
4. A join never read loses its copy when the connection closes. So nothing is left behind, even after a crash.

A chain such as `a.join(b).join(c)` works the same way. Each join brings in and deletes its own other table.

There is no option flag. The copy is the only behavior, so a join reads the same on a Daf and a SqlDaf.

`select_records_daf()` and `concat()` have the same same-file limit, through `_check_same_db()`.
They should bring in a table from another file the same way.

## Not asked

- No change to `from_db()` or `save(db_path=)`. Copying a table into a file named for the eventual table stays the manual way.
- A true view, where a selection reads only the matching rows without resolving the whole join, may be asked for later.

## Documentation

`docsite/guide/sqldaf.md` should then say this in its Joins and Tables in other files sections.
Today they say a table in another file must be copied in with `from_db()`, but not where the copy goes or that it is deleted.

## Why

AuditEngine keeps each large table in its own file (Q189).
Its stages then copied every table into a per-stage scratch file to join them.
Ray never approved that, and AuditEngine is removing it. These joins will call `join()` directly.
