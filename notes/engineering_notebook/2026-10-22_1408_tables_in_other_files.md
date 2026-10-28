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

## Ray's question: is there another way than ATTACH?

Ray asked for one connection and one visible database during a stage, with each input read in,
as `from_csv()` reads a file. SQLite reads another file only through ATTACH, or through Python.
1,000,000 rows of 5 columns into a working file, 2026-10-22:

    brief ATTACH, copy, DETACH    0.62 s
    through Python, no ATTACH     2.54 s
    backup API                    0.23 s, but it replaces the whole target file

The backup API removed the working table already there, so it cannot bring in one table.
`deserialize()` loads a whole file into memory, and still needs an attached schema.

## Q274, settled 2026-10-22

Ray: from_db() "is going to be a better design than using attach. Allows us to have any number of
databases involved, and it's faster to boot." He likes that a source is never changed unless a
result is written back. AuditEngine agreed in to_daffodil/2026-10-22_1430_from_db_fits_auditengine.md.
It copies only the named table and its own daf_tables row, as AuditEngine asked, since each file's
table_sources belongs to that file.

- `SqlDaf.from_db(path, name, new_name='', db_path='')` copies a table into the working file.
- `save(name, db_path=)` writes a table into another file. A TEMP table moves, a permanent one is copied.
- `_file_schema()` attaches a file for one block and always detaches it.
  The source of `from_db()` is attached as a URI with mode=ro, so a write to it fails with
  "attempt to write a readonly database". SqlDaf's connections open with `uri=True` for this.
  A plain path, even with a space, # or ?, still opens as that file.
- Both raise ValueError inside an open transaction. Tested: SQLite then fails the DETACH with
  "database f is locked", and the file stays attached, so the next copy would fail too.

## On AuditEngine's pilot d05 files, 2026-10-22

Read-only, from /home/audit-engine-dev/resources/CA_SanFrancisco_20241105_pilot_d05:

    biacvr_bif        133,422 rows, 52 cols   0.37 s
    blt_bif           133,422 rows, 24 cols   0.21 s
    cvr_votes!final 1,603,908 rows, 13 cols   6.75 s, an 804 MB file
    left join of biacvr_bif and blt_bif       3.27 s, 75 cols
    save into full_bif.db                     0.48 s

The copy time follows the bytes more than the rows. For cvr_votes!final, reading every value
took 2.11 s, 680 MB of values, and the whole copy 6.21 s, writing a 760 MB table. So about 4 s is
the write into the working file. A cp of the whole 804 MB file took 0.61 s, but copies every table. The join matched `Daf.join()` of the same
tables in memory. No ballot lacked a blt row. Each source file had the same bytes and mtime after.
Script: scratch from_db_d05.py, not kept.

## A join brings in a table from another file, 2026-10-28, Q312

AuditEngine asked, as their Q438: mail to_daffodil/2026-10-28_0326_join_brings_in_other_file.md, and
their notebook engineering_notebook/2026-10-28_sqldaf_join_across_files.md. Ray: "Yes implement".
Ray's model: the other table is brought in for the join, the join is done, the copy is deleted, and
the result is saved in a file of its own. `from_db()` stays for a file named for the eventual table.

Built:

- `_readable()` in sqldaf.py replaces `_check_same_db()`. It gives the other SqlDaf when this
  connection can read it, and otherwise `_bring_in()`, a TEMP copy here with its key index.
- `_bring_in()` copies a permanent table in SQL through a brief read-only ATTACH, as `from_db()`
  does, and anything else through Python in batches from the other's connection. Inside a transaction
  it also uses Python, since SQLite cannot detach a file used in one.
- A join holds its copies until it is first read. Each copy has a count on the connection, in
  `brought_in_di`, of the unread joins that use it, since a chain `a.join(b).join(c)` reads the copy of
  `a.join(b)` inside its own SQL. `weakref.finalize` releases a join's copies when it is read, or when
  the join object is gone unread, as the inner join of a chain is. A copy at a count of zero is dropped.
- `concat()` drops its copy after the insert. Keys copy only the key column, into the TEMP table of
  keys that a Daf of keys already uses.
- A selection of another thread's connection cannot be read from this thread, as Python's sqlite3
  refuses it. It raises NotImplementedError with a clear message.

Timings, 1,000,000 rows of 5 columns, a left join on unique keys:

    one file                                  3.4 to 3.8 s
    other file, copy without an index         0.30 s, then the join 4.80 s
    other file, copy with its key index       0.7 s, then the join 3.7 to 3.9 s
    a selection in another file, by Python    2.37 s to copy, then 5.1 s

The index on the copy is what makes the join as fast as in one file.
