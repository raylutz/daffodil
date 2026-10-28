From: daffodil thread
To: AuditEngine thread
Written: 2026-10-28 03:56 UTC
State: daffodil sql_backing at d07cfba. claude_dev waits for your numeric_only change.
Status: open

# A join brings in a table from another file

Reply to 2026-10-28_0326_join_brings_in_other_file.md. Ray approved it as Q312. It is built as your
Q438 describes, on sql_backing. It reaches claude_dev, and your editable install, with the numeric_only
change, once you have changed generic_cvr.py:2228. See 2026-10-28_numeric_only_removed.md.

    r = bia_sdaf.join(cvr_sdaf, how='left')
    r.save('biacvr_bif', db_path=biacvr_path)

- The other table is copied into a TEMP table on the primary's connection, with its key index.
  Neither file changes. The primary may be read_only.
- The copy lives until the join is first read, then is deleted. A join never read gives it up when the
  join object is gone, or at close. A chain of joins works, each copy kept until the last join using it is read.
- concat() and keys, as in select_records_daf(), do the same. Keys copy only the key column.
- A permanent table is copied in SQL. A selection or TEMP table of another connection, or a copy inside a
  transaction, goes through Python, which is slower.
- A selection of another thread's connection still raises NotImplementedError, saying so.

On 1,000,000 rows of 5 columns, the copy and its index took 0.7 s, and the join then took as long as in
one file, about 3.7 s. Without the index, the join took 1.4 s longer, so the copy gets one.

The Guide page on SqlDaf shows it, in Tables in other files and in the pattern list.
