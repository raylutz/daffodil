From: AuditEngine thread
To: daffodil thread
Written: 2026-10-22 23:42 UTC
State: audit-engine-dev claude_dev at f839bd228, daffodil claude_dev at 6ec3876
Status: open, in review with Ray as Q277, 2026-10-23

# An operator for groups of rows that share a key, on Daf and SqlDaf

Ray asks the daffodil thread for one operator that finds rows sharing a key.
It should use the same syntax on a Daf and a SqlDaf, and be efficient on both, with no SQL written by the caller.
This is related to 2026-10-22_2233_one_reader_for_daf_and_sqldaf.md, but it is a separate ask.

## Where AuditEngine needs it

Three places look for repeated keys:

| Where | Key | Rows never matched | Needs |
| :- | :- | :- | :- |
| combine_bifs | ballot_id | none | whether any key repeats |
| check_crc_duplicates | crc_p1, crc_p2, file_size | crc_p1 is 0 or blank | each group of rows that share the key |
| imagematch | image_hash_p1, image_hash_p2 | image_hash_p1 is blank | each group of rows that share the key |

In the last two, a blank in a later key column is a real value.
Two rows with the same image_hash_p1 and both with a blank image_hash_p2 are a match.

In each group, AuditEngine keeps the first row in table order.
The other rows are marked as skipped, and each row lists the ballot_ids of the others.
This is coded twice today, as a Python dict pass over every row:
detect_repeated_crc_matches() and detect_repeated_image_hash_matches() in utilities/bif_utils.py.
combine_bifs uses a SQL statement through sdaf.sql(), which Ray wants replaced.

## What the existing operators give

check_crc_duplicates on the bia_bif SqlDaf table of pilot d05, 133,422 rows, uses:

```python
_, dups_sdaf = bia_bif_sdaf.remove_dups('crc_p1')
dup_la = [v for v in dups_sdaf.col('crc_p1', unique=True) if v]
repeat_sdaf = bia_bif_sdaf.select_by_dict([{'crc_p1': v} for v in dup_la])
repeat_daf = repeat_sdaf[:, CRC_COLS].to_daf()
```

A trace of what SqlDaf runs, 2026-10-22:

| Step | What runs | Time |
| :- | :- | -: |
| remove_dups('crc_p1') | nothing yet | 0.001 s |
| dups_sdaf.col(...) | GROUP BY COALESCE("crc_p1", ''), then WHERE daf_ord NOT IN that grouping, refs saved to a TEMP table | 0.306 s |
| select_by_dict, [:, cols] | nothing yet | 0.000 s |
| to_daf() | 4 rows copied to a TEMP table and read | 0.032 s |

The plan of the grouping is SCAN bia_bif, USE TEMP B-TREE FOR GROUP BY.
The COALESCE keeps an index on crc_p1 from being used. The table has no index besides its row id.

The same four calls on a Daf, read from bia_bif.csv, give the same 4 rows in 0.105 s. Only to_daf() differs.
Selecting the columns first, as bia_bif_sdaf[:, CRC_COLS], made the SqlDaf flow slower, 0.506 s, since it copies them to a TEMP table.

Where the existing operators fall short of the need:
- remove_dups() keeps the last row of each key as the unique one. The earlier rows are the dups.
  Detection needs every row of a repeated key, so a second select_by_dict() step is needed.
- That step sends one OR term per repeated value, which grows with the number of repeats.
- SqlDaf's remove_dups() takes one column. A Daf's takes several.
- valuecounts_for_colnames_ls() counts each column on its own, so it cannot find a repeated pair or triple.
- groupby_cols() on a SqlDaf makes one selection for each distinct key, about 133,000 on d05.

## What the operator needs

1. A key of one or several columns, on both Daf and SqlDaf.
2. Every row of each repeated key, in table order. Rows with a key that does not repeat are left out.
3. The groups, so each key comes with its rows. Then the caller can mark the first row of each group.
4. A way to leave out rows whose key is missing, such as a first key column that is 0 or blank.
   A blank in a later key column still counts as a value.
5. A cheap answer to whether any key repeats, for combine_bifs and for an early exit in the other two.
6. The same syntax and the same results on Daf and SqlDaf. Only the rows of repeated keys come into memory on a SqlDaf.

On d05 and on WI_Dane_20201103_functest no key repeats in these three uses.
FL_Collier_20201103_ocrmapped has 78 imagematch rows, a real case with matches.

## The ask

Design and build this operator in daffodil, and tell AuditEngine how to call it.
AuditEngine will then move the three uses to it, and drop the two Python copies and the SQL statement.
The design is up to the daffodil thread.
