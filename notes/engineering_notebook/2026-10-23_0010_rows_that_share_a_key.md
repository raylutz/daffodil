# Rows that share a key, on Daf and SqlDaf

Started 2026-10-23. AuditEngine asked for one operator that finds rows sharing a key, with the
same syntax on a Daf and a SqlDaf and no SQL from the caller.
Mail: to_daffodil/2026-10-22_2342_duplicate_groups_operator.md.

## The uses

| Where | Key | Rows never matched |
| :- | :- | :- |
| combine_bifs | ballot_id | none |
| check_crc_duplicates | crc_p1, crc_p2, file_size | crc_p1 is 0 or blank |
| imagematch | image_hash_p1, image_hash_p2 | image_hash_p1 is blank |

A blank in a later key column is a value. In each group AuditEngine keeps the first row in table
order and marks the others. It coded this twice in Python, in bif_utils.py, and once in SQL.

## What existed

- `remove_dups()` keeps the last row of each key, and on a SqlDaf takes one column.
  AuditEngine chained it with `col(unique=True)`, `select_by_dict()` and `to_daf()`: 0.34 s on d05.
- `groupby_cols()` makes a group for every key, about 133,000 on d05.
- `valuecounts_for_colnames_ls()` counts each column alone, so it misses a repeated pair.

## Q277, settled 2026-10-23

Ray named it `groupby_repeated_cols()`, returning `T_klodaf` on a Daf and `T_klosdaf` on a SqlDaf.
So on a SqlDaf each group is a selection. `T_klosdaf` is new, in sqldaf.py beside the class.
Option A was built, with `exclude`. The separate yes/no method, A+, was not: it saved only
hundredths of a second, since the GROUP BY alone answers when nothing repeats.

- Daf: one dict pass over the rows, then `_new_group_dafs()`, so the groups share rows, as in `groupby_cols()`.
- SqlDaf: one GROUP BY with HAVING COUNT(*) > 1, ordered by the first row of each key.
  `value_sql()` makes the text '' and SQL NULL one value. The group index is made only when a key
  repeats. Each group is `select_by_dict()` of its key.
- `exclude` on a column outside the key is applied to each selection too. A test first missed this:
  the key 7 brought back a row that exclude had left out of the count.

## Measured, 2026-10-23

pilot d05 bia_bif, 133,422 rows, and FL_Collier bia_bif, 417,777 rows. No key repeats in either.

    use                    d05 SqlDaf  d05 Daf   Collier SqlDaf  Collier Daf
    combine_bifs           0.128 s     0.210 s   0.405 s         0.891 s
    check_crc_duplicates   0.175 s     0.284 s   0.940 s         1.468 s
    imagematch             0.032 s     0.053 s   0.087 s         0.172 s

Collier's bia_bif has no image hashes at all. Its imagematch/imagematch.csv holds the 78 matched
rows: 39 groups, and each group's first row is the one AuditEngine kept, the rest those it skipped.
That checks the grouping, not detection over a full table. Script: scratch repeated_check.py, not kept.
