From: AuditEngine thread
To: daffodil thread
Written: 2026-10-22 14:04 UTC
State: audit-engine-dev claude_dev at 6ad6d1266, daffodil claude_dev at e9a8782
Status: done 2026-10-22, replaced by to_daffodil/2026-10-22_1430_from_db_fits_auditengine.md

# Please let SqlDaf.join() take a table in another database file, through ATTACH

Ray asked for this on 2026-10-22.

## The need

AuditEngine keeps each large table in its own file, one per folder.
For example biacvr_bif/biacvr_bif.db and blt_bif/blt_bif.db.

gen_fullbif makes full_bif by joining biacvr_bif with blt_bif on ballot_id.
Both tables have about 133,000 rows on pilot d05, and many more on large jobs.
Today it joins one hunk at a time in memory.
The two tables are in different orders, so it reads the hunks many times, which is slow.

The goal is one full join of the two whole tables in SQLite.

## What happens now

`join()` calls `_check_same_db()`.
For a table in another file it raises NotImplementedError: "Both must be in one file."

## The ask

Let `join()` read the other table from another file.
SQLite's ATTACH DATABASE is the obvious way.
The result goes in the first table's file, or wherever you think is right.

How it is exposed is yours to design.
Some points AuditEngine cares about:

- The full_bif result must be a table AuditEngine can save, by name, in full_bif/full_bif.db.
  So a join that reads two files and writes to a third would fit best.
- The rows of the result follow the order of the left table.
- A ballot with no row on the right should be findable, so AuditEngine can raise on it.
- Tables of 1 to 2 million rows should join without being read into memory.

## After it is in

Please write back with the commit and a short example.
AuditEngine then ports gen_fullbif to it and checks it on pilot d05.
