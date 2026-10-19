# Combining the chunks of a delegation run into a SqlDaf table

Started 2026-10-19 15:51 UTC. Ray asked on 2026-10-19 to look at the concept: instead of combining
the chunks of a Lambda run into one file and loading it into memory, read them into a SqlDaf table
that stays in the database file. Later operations then run on the table through SqlDaf, and it is
never read in again.

Nothing here runs an AuditEngine stage. The checks read the SF pilot d05's chunk files where they
are, and write only to the scratch folder.

## What AuditEngine does now

`DB.combine_dirname_chunks()` in models/DB.py, at AuditEngine's 4fa8df6b1:

1. If the combined file on S3 is newer than the newest chunk, it is used as it is.
2. `DB.download_entire_dirname_v2()` downloads every chunk to the local folder.
3. Without `groupby`, `DB.merge_csv_dirname_local()` appends the text of the chunks into one CSV,
   less their headers, and checks that the headers are the same. This streams on disk. It does
   not hold the chunks in memory. The CSV is uploaded.
4. With `groupby`, the combined CSV is loaded whole as a Daf, converted, grouped and summed, the
   numbers by `groupsum()` and concise_votes by `daf_sum(by='sparse_row')`. This is where memory
   runs out on a large county.

The memory goes later, too. Each stage that uses the combined table loads the CSV again as a Daf:
ACRE, dominion_cvr.py at line 165, the readers of votes.csv in evalvotes, ml_marks,
extraction_coverage_report and ballot_exceptions_report. find_variants imports the CSV into
cmpcvr.db again.

The callers: votes_extractor.py for votes, dominion_cvr.py for the CVR votes, the CVR BIF, the
grouped CVR votes by precinct or batch, the batch manifest, the BIF stats and the stdresults,
hart_cvr.py, and pmvs.py.

## Trial on the SF pilot d05

`notes/sql_backing/chunk_combine_check.py` combines one chunk folder both ways and compares.
The SqlDaf way streams the rows of every chunk into one table with the SqlDaf constructor, in one
transaction, with the same header check. The table is compared with the CSV, row by row.

| Chunks | Rows | Text into one CSV | Into a table | Table file | Peak |
| :----- | ---: | ----------------: | -----------: | ---------: | ---: |
| votes, 1,335 files, 18 MB | 34,534 | 0.1 s | 0.5 s | 18 MB | 60 MB |
| cvr_votes!final, 327 files, 675 MB | 1,603,908 | 7.4 s | 20.7 s | 767 MB | 48 MB |

- The table has the same rows as the CSV, in the same order, for both.
- Opening the table again takes no time. The CSV must be parsed again by each stage.
- The CVR as a Daf would take about 5.5 GB, from the ACRE measure of the same rows. It was not
  loaded, to spare the shared box, which has 15 GB.
- The rows arrive as text. `apply_dtypes()` of four int columns afterwards took 16.0 s.
  Converting them in the stream instead took 24.9 s in all, against 20.7 + 16.0 s, at 39 MB.

Work on the CVR table, 1,603,908 rows:

| Operation | Time | Peak |
| :-------- | ---: | ---: |
| `groupsum_daf('precinct', reduce_cols=['blank', 'v'])`, 511 precincts | 3.0 s | 102 MB |
| `groupby_aggregate('precinct', Daf.sum_da, 'concise_votes')` | 35.8 s | 666 MB |
| the same, on the stdlib driver | 24.0 s with a reduce that keeps nothing | 40 MB |
| `select_where_sql('is_bmd = 1')` | 0.5 s | |
| an index on precinct | 2.8 s | |

The precinct totals are the job of the grouped combine. All 511 precincts gave the same blank,
v and concise_votes as AuditEngine's own `cvr_votes!final!precinct.csv` of the pilot.

The 666 MB is SQLite's, not Python's. pysqlite3-binary keeps every TEMP table and sort in memory.
See the group reductions entry, and Q253.

## The concept, as it would work

A sibling of the combine in AuditEngine, or an option of it:

1. Download the chunks as now.
2. In one pass, stream the rows of every chunk into a table, converting the columns by the
   dirname's dtypes as they go. The same pass may still write the combined CSV for S3, which the
   Lambdas and other machines read. That costs the 7.4 s of today.
3. The table lives in a database file of the job on the local disk, such as
   `<job>/cvr_votes/cvr_votes.db`, named after the combined file, as `cvr_votes!final`.
4. Record the etags of the chunks it was built from in the same file. `DB.gen_etags_summary()`
   makes such a summary already. A later run opens the table if the chunks are the same.
5. A grouped combine is then `groupsum_daf()` and `groupby_aggregate()` on the table, which gave
   AuditEngine's own totals. Nothing is loaded whole.
6. A stage that reads the combined file opens the table instead, one stage at a time. find_variants
   could copy the table into cmpcvr.db by ATTACH, instead of importing the CSV again.

What it does not change: the Lambdas still write CSV chunks, and the CSV on S3 stays the record.
The table is a local copy for the EC2 stages, kept in step by the etags.

## For daffodil

The trial needs nothing new from daffodil. The constructor takes the rows from a generator. A
`SqlDaf.from_csv_files()` would put the parts AuditEngine would otherwise write into one call:
many files into one table, the header check, empty chunks skipped, and an optional dtypes dict
converted in the stream. AuditEngine's merge also realigns headers that differ only in the file
path columns. That rule is AuditEngine's, so it would stay there. Q254.

## Open

- Q253: which SQLite driver SqlDaf prefers, given TEMP_STORE=3 in pysqlite3-binary.
- Q254: `SqlDaf.from_csv_files()`.
- Where AuditEngine keeps the database files of a job, and when they are deleted.

## Pointers

- AuditEngine: models/DB.py `combine_dirname_chunks()`, `merge_csv_dirname_local()`,
  `merge_csv_filepaths_local()`.
- The check: notes/sql_backing/chunk_combine_check.py.
- The driver finding: 2026-10-18_1428_sqldaf_group_reductions.md, pysqlite3-binary keeps TEMP data in memory.
- The full BIF and ACRE: 2026-10-19_1443_auditengine_whole_table_stages_on_sqldaf.md.
