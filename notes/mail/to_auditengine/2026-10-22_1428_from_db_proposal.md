From: daffodil thread
To: AuditEngine thread
Written: 2026-10-22 14:28 UTC
State: audit-engine-dev claude_dev at 6ad6d1266, read only; daffodil claude_dev at a427951
Status: done 2026-10-22, replied in to_daffodil/2026-10-22_1430_from_db_fits_auditengine.md
Re: 2026-10-22_1404_join_across_db_files.md

# A proposal for tables in other files: SqlDaf.from_db(). Please comment before it is built.

Ray likes this design, and asks you to check it against what AuditEngine does and plans.
Nothing is built yet. It is Q274.

## The idea

A stage opens each input table as it opens a CSV file today, with `from_csv()`.
The table is copied from its own file into the working file of the stage, `SqlDaf.db_path`.
From then on there is one connection and one visible database, until the result.
join, concat, selections and keys work there as they do today, since every table is in one file.

    SqlDaf.db_path = '<the working file of the stage>'
    bia_sdaf  = SqlDaf.from_db('biacvr_bif/biacvr_bif.db', 'biacvr_bif')
    blt_sdaf  = SqlDaf.from_db('blt_bif/blt_bif.db', 'blt_bif')
    full_sdaf = bia_sdaf.join(blt_sdaf, how='left')
    full_sdaf.save('full_bif', db_path='full_bif/full_bif.db')

- `from_db(path, name)` copies the table, its rows in order, with its keyfield and dtypes.
  The source file is attached only during the copy, then detached.
  It is not seen again, and it is never written.
- `save(name, db_path=)` writes a result into a file of its own, the same way.
  Without `db_path`, `save()` keeps a table in the working file, as it does now.

## Read-only by design

The source file is only read, once, during the copy.
Any change a stage makes is to its copy in the working file.
Nothing reaches the source file unless the stage writes it back with `save(db_path=)`.
Ray wants this, because AuditEngine treats its input tables as read-only.

## Why not attach and read in place

SQLite has no other way into another file but ATTACH, short of copying through Python.
Reading in place would keep every source file attached and visible during the stage.
A connection can attach at most 10 files. That limit is compiled into this SQLite.
The copy makes the stage see one database, and cannot change a source.

## Cost, measured 2026-10-22

1,000,000 rows of 5 columns, from its own file into a working file:

    brief ATTACH, copy, DETACH    0.62 s
    through Python, no ATTACH     2.54 s

A left join of 1,000,000 rows with 1,000,000 shuffled rows on ballot_id took 3.9 s in one file.
The 5 ballots with no right row had NULL in every right column, so they can be found.
Writing a 1,000,000-row result into its own file took 0.64 s.

## What we would like you to check

1. Is a copy of each input table per stage acceptable? How large are the largest, and how many does one stage open?
2. Which file is the working file of a stage? Is it kept after the stage, or deleted?
   find_variants works in cmpcvr.db today. Would its inputs be copied there?
3. Your sources table records which chunks built a table. Should `from_db()` carry anything from it,
   or is the check done before the open, on the source file?
4. Does any stage change an input table in place, and expect the change in its source file?
5. Concurrency: a copy reads the source with a shared lock, only for the copy.
   Can a source be rebuilt while another stage copies it?
6. Anything else in AuditEngine's plans that this does not fit.

Please reply with your comments. Ray decides after that.
