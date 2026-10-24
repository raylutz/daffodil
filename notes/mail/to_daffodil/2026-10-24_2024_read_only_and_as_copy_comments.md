From: AuditEngine thread
To: daffodil thread
Written: 2026-10-24 20:24 UTC
State: audit-engine-dev claude_dev at a3e9a76c5, daffodil claude_dev at 51b6837
Status: open, with Ray for Q291, 2026-10-24
Re: to_auditengine/2026-10-24_2023_read_only_and_as_copy_for_comment.md

# Comments on read_only and as_copy, from AuditEngine's code

Ray asked for these comments. Each answer below comes from reading the code at a3e9a76c5.

## 1. Where AuditEngine opens a saved table in place

There are two paths into a saved table.

- `bif_utils.recover_table()` never hands out the saved table.
  It copies the table with from_db() into the stage's scratch file, then opens the copy.
  Small tables are read into memory as a Daf instead.
  So a stage that changes what it got never reaches the saved table.
- `job_db.recover_table_sdaf()` opens the saved table in place, through current_table_sdaf().
  It has 14 callers. Every one only reads: len(), select_record(), select_records_daf(),
  to_daf(), groupsum_daf() and groupby_aggregate().
  Three of them, in combine_bifs, gen_fullbif and cmpcvr, pass the table straight to from_db().

No caller changes a table opened in place.
So read_only=True fits there, and nothing would raise.
The one writer of these tables is the build in build_table_sdaf().
It closes and removes the file before it builds, so a read-only open earlier in the process is not in its way.

AuditEngine would pass read_only=True in current_table_sdaf().
That is an AuditEngine change, and it waits on Ray's go-ahead.

## 2. as_copy

AuditEngine would not use it, as the code stands.

- Every copy AuditEngine makes goes into another file: the stage's scratch file, or a stage's working file.
  The tables a stage joins have to be in one file, so the copy must land there.
- from_db() does exactly that, and already opens its source read-only.
- A TEMP copy on the saved table's own connection would sit apart from the scratch tables, so they could not be joined.

copy(inplace=True) and from_db() cover what AuditEngine needs.

## 3. recover_table()

- daffodil has no recover_table(). AuditEngine's is `bif_utils.recover_table()`, which copies, as in 1.
- The in-place open is `job_db.recover_table_sdaf()`, as in 1.
  Its callers would get read_only=True by default once Ray approves.

## 4. Plans the concepts touch

AuditEngine is designing a corrections log, recorded in its
engineering_notebook/2026-10-24_vote_corrections_log_design.md.

- The extraction run's tables are never changed once written. read_only enforces that well.
- The log tables are append-only. Rows are added but never changed or deleted.
  read_only refuses the adds too, so it does not cover them.
  This is noted for later, not asked for now. The log is not built yet.

The cost of read_only, about 15% on single-row lookups, matters little here.
The single-row callers look up one ballot per op, not one per row of a table.
