# Reading a table in another database file

Started 2026-10-22. AuditEngine keeps each large table in its own file, one per folder, such as
biacvr_bif/biacvr_bif.db and blt_bif/blt_bif.db. A file per table tells whether a table is built,
makes it easy to drop, and lets tables be built at the same time.
gen_fullbif joins biacvr_bif with blt_bif on ballot_id, and saves full_bif in a third file.
Mail: to_daffodil/2026-10-22_1404_join_across_db_files.md.

Ray, 2026-10-22: the join should notice that the tables are in different files and attach first.
No separate command. It should be transparent, as if the tables were in one file.

## How SqlDaf opens files

- One connection per thread and real path, in `SqlDaf._open_conns`. Every SqlDaf on a file shares it.
- SQL names a table by its bare name, through `quote_name(self.name)`, in about 25 places.
- A child keeps its rows as a TEMP table of references, which only its own connection sees.
  A join not read yet is a query on its own connection. So is a sort or a copy, as TEMP tables.

## The callers that read another SqlDaf

All three go through `_check_same_db()`, which raises NotImplementedError for another file.

- `join()`: the other table.
- `concat()`: the rows to add.
- `_key_source_sql()`: keys given as a SqlDaf, for `select_records_daf()` and the like.

An in-memory Daf given as keys is already written to a TEMP table on this connection.
That is the same path a selection in another file could take.

## SQLite facts, checked on 2026-10-22

- Every Python here, and AuditEngine's venv, has SQLite 3.53.1, compiled with MAX_ATTACHED=10.
  `setlimit()` cannot raise it. An eleventh ATTACH raises "too many attached databases - max 10".
- ATTACH and DETACH work inside a transaction on this SQLite.
- An attached table is read with its schema name, as in `f1.blt_bif`.

## Timings, 2026-10-22

1,000,000 rows on each side, the right side shuffled, 5 ballots with no right row.
A left join on ballot_id with unique indexes, in the order of the left table.
Script: scratch attach_probe.py, not kept.

    one file, into TEMP              3.89 s
    two files by ATTACH, into TEMP   4.07 s     ATTACH itself: 0.15 ms
    TEMP into a third file           0.64 s
    join straight into a third file  4.23 s
    join through Python into a file  7.24 s

The 5 unmatched rows had NULL in every right column.

## Designs considered

- A: attach the other file when the other is a table there. A selection or TEMP table there raises.
- A+: as A, and a selection or TEMP table in another file is copied through Python into a TEMP table here.
  Every case works. Only that case pays the copy.
- B: one connection per thread, every file attached as a schema, every table name qualified.
  Fully transparent, but it changes every statement, and at most 10 files could be open at once.

## Open

Q274 to Q276, in review.
