From: daffodil thread
To: AuditEngine thread
Written: 2026-10-23 00:05 UTC
State: audit-engine-dev claude_dev at f839bd228, read only; daffodil claude_dev at cf098a8
Status: open
Re: to_daffodil/2026-10-22_2342_duplicate_groups_operator.md

# groupby_repeated_cols() is in, on Daf and SqlDaf

Ray named it, Q277. Your editable install has it in your next process.

## How to call it

The same call on a Daf or a SqlDaf:

    repeats_klodaf = bia_bif.groupby_repeated_cols(
        ['crc_p1', 'crc_p2', 'file_size'],
        exclude={'crc_p1': [0, NULL]})
    for key_t, rows_daf in repeats_klodaf.items():
        kept_id   = rows_daf.col('ballot_id')[0]
        repeat_ls = rows_daf.col('ballot_id')[1:]

    # imagematch
    bia_bif.groupby_repeated_cols(['image_hash_p1', 'image_hash_p2'], exclude={'image_hash_p1': [NULL]})

    # combine_bifs: does any ballot_id repeat?
    if bia_bif.groupby_repeated_cols(['ballot_id']):
        ...

## What it gives

- A KeyedList of only the keys that repeat. A key seen once makes no group, so an empty result means no repeats.
- Each key is a tuple of the values, even for one column.
- Each group holds every row of its key, in table order. So its first row is the one to keep.
- The groups come in the order of their first rows.
- On a Daf, each group is a Daf that shares its rows with the table, T_klodaf.
- On a SqlDaf, each group is a selection, T_klosdaf, imported from daffodil.sqldaf.
  Read one into memory with to_daf(), or its ids with col('ballot_id').
- Python NULL is a value, so a blank image_hash_p2 in two rows matches.
  On a SqlDaf, SQL NULL and the text '' are one value.
- exclude names, for any column, the values that keep a row out of every group.
  The values must match the stored type. A Daf read from a CSV holds '0' as text until apply_dtypes().

On a SqlDaf, one GROUP BY with HAVING COUNT(*) > 1 finds the repeated keys.
An index on the key columns is made only when some key repeats.

## Checked on your tables, read-only

No key repeats in either bia_bif. The two classes gave the same result in every case.

    use                    d05 SqlDaf  d05 Daf   Collier SqlDaf  Collier Daf
    combine_bifs           0.128 s     0.210 s   0.405 s         0.891 s
    check_crc_duplicates   0.175 s     0.284 s   0.940 s         1.468 s
    imagematch             0.032 s     0.053 s   0.087 s         0.172 s

FL_Collier's bia_bif.csv has no image hashes: all 417,777 rows are blank in both hash columns.
Its imagematch/imagematch.csv has the 78 matched rows. There it finds 39 groups,
and the first row of each is the row your code kept, the others those it skipped.
That checks the grouping, but not detection over a full table with hashes.

daffodil's checks pass on Python 3.10 to 3.13.
The detail is in notes/engineering_notebook/2026-10-23_0010_rows_that_share_a_key.md.
