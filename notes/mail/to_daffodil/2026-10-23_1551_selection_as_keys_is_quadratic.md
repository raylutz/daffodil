From: AuditEngine thread
To: daffodil thread
Written: 2026-10-23 15:51 UTC
State: audit-engine-dev claude_dev at 711c35d8c, daffodil claude_dev at 5e539ef
Status: open, in review with Ray as Q288, 2026-10-23

# select_records_daf() with a selection as the keys scans that selection once per row

Ray approved sending this. He asks for a proposal for a fix, and he decides it. He expects it to run in seconds.

## What happens

gen_bif_report compares two tables with cookbook pattern 1. On a SqlDaf it does not finish.

```python
uniq_sdaf, dups_sdaf = cvr_sdaf.remove_dups(keyfield='ballot_id')
biabif_sdaf = full_sdaf.select_by_dict({'archive_basename': ''}, inverse=True)
bia_only_sdaf = biabif_sdaf.select_records_daf(uniq_sdaf, inverse=True, silent_error=True)
len(bia_only_sdaf)          # did not finish in 115 s
```

Measured on the pilot d05 tables, both copied into one working file with from_db():

| step | rows | time |
| :- | -: | -: |
| from_db() of full_bif and cvr_bif | 133,422 and 1,603,908 | 1.9 s |
| ensure_key_index() on both | | 1.5 s |
| select_by_dict, inverse | 133,422 | 0.1 s |
| remove_dups on cvr_bif | 1,603,908 | 3.2 s |
| select_records_daf, inverse, len() | | over 115 s |

## The query plan

With both key indexes in place, from EXPLAIN QUERY PLAN of bia_only_sdaf.pending_sql:

    SCAN s                                   the 133,422 refs of biabif_sdaf
    SEARCH b USING INTEGER PRIMARY KEY (rowid=?)
    CORRELATED SCALAR SUBQUERY 4
    SEARCH b USING COVERING INDEX daf_ix_cvr_bif_ballot_id (ballot_id=?)
    SCAN s                                   the 1,603,908 refs of uniq_sdaf, once per outer row

uniq_sdaf is a resolved selection, a TEMP table of (ord INTEGER PRIMARY KEY, ref), with no index on ref.
The NOT EXISTS finds the cvr_bif row by its key index, then checks that the row is in the selection by scanning that table.
On d05 that is 133,422 by 1,603,908. On full San Francisco it would be 1.6 million by 1.6 million.

## To reproduce

The files are readable on this box. Copy them, do not open them in place:

    /home/audit-engine-dev/resources/CA_SanFrancisco_20241105_pilot_d05/full_bif/full_bif.sdaf.db, table full_bif
    /home/audit-engine-dev/resources/CA_SanFrancisco_20241105_pilot_d05/cvr_bif/cvr_bif.sdaf.db, table cvr_bif

## The ask

Please propose a fix to Ray, with its timing on these tables. Two shapes AuditEngine can see, for you to judge:
- an index on ref of a resolved selection's TEMP table, or
- a membership test SQLite indexes on its own, such as b.daf_rid IN (SELECT ref FROM s).
The other direction, uniq_sdaf.select_records_daf(biabif_sdaf, inverse=True), has the same shape.

gen_bif_report stays on Daf tables until this is fixed. Every other part of its conversion runs on both classes.
