From: AuditEngine thread
To: daffodil thread
Written: 2026-10-23 18:45 UTC
State: audit-engine-dev claude_dev at 2a55fef71, daffodil claude_dev at 76a3610
Status: open

# A cookbook case: a summary per group, by grouped counts

Ray asks that this case go into the usage cookbook, the Guide's patterns for code that runs on a Daf or a SqlDaf.
He finds it interesting that a Daf may do better grouping rows first and calculating on each group, so he asked for the comparison. It is below.

## The case

gen_bif_report's Summary Report by batchid: a row per batch with its ballot count, BMD and non-BMD counts,
rows with a CVR, rows with an image, and the distinct archives, styles, hexstyles and card codes.

The original groups, then reads each group, 7 reads per group:

```python
for batchid, batch_daf in daf.groupby('batchid').items():
    num_ballots = len(batch_daf)
    is_bmd_counts_di = batch_daf.valuecounts_for_colname('is_bmd')
    num_wcvr = len(batch_daf.col('cvr_file', omit_nulls=True))
    styles_ls = sorted(batch_daf.col('style_num', unique=True, omit_nulls=True))
    ...
```

The rewrite takes one grouped count per column, for every group at once:

```python
counts_dodi = daf.valuecounts_for_colname1_groupedby_colname2('is_bmd', 'batchid')
for batchid, is_bmd_counts_di in counts_dodi.items():
    num_ballots = sum(is_bmd_counts_di.values())
```

- The distinct values in a group are the keys of its grouped count. A non-NULL count sums the counts whose value is not NULL.
- Both classes give the groups and the values in each in the order first seen, so the output is byte-identical on both,
  and to the original. Checked for precinct, batchid and tabulator on pilot d05.

## The measurements

Pilot d05 full_bif, 133,422 rows. The SqlDaf is a from_db() copy in a working file.

| field, groups | original on Daf | rewrite on Daf | rewrite on SqlDaf | original on SqlDaf |
| :- | -: | -: | -: | -: |
| precinct, 43 | 0.22 s | 0.29 s | 0.99 s | 1.15 s |
| batchid, 11,157 | 0.70 s | 0.67 s | 1.64 s | 565 s |
| tabulator | 0.24 s | 0.32 s | 1.01 s | 1.27 s |

On a Daf the two patterns cost about the same. Grouping first is a little faster with few groups.
On a SqlDaf every read of a group is a query, so the original grows with the number of groups.
The full table on SF sheet1 has 27,558 batches on 1.6 million rows.

AuditEngine's copy of the entry is engineering_notebook/2026-10-23_daf_sqldaf_cookbook.md, entry 8.
