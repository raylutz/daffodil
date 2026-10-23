From: AuditEngine thread
To: daffodil thread
Written: 2026-10-22 22:33 UTC
State: audit-engine-dev claude_dev at 839fbbb0c, daffodil claude_dev at ae852a0
Status: done 2026-10-23, Daf.to_daf() (Q276)

# A general way to read a table that may be a Daf or a SqlDaf

Ray asks the daffodil thread to work out a general approach for this.
The goal is that AuditEngine code reads a table the same way, whether it is a SqlDaf or an in-memory Daf.
Then AuditEngine can drop the shim it adds at each reader.

## The problem

AuditEngine is moving its large tables to SqlDaf, one stage at a time.
A SqlDaf table exists only on the EC2 box, and only once it has been built.
A Lambda, or a job built before the tables, has only the combined CSV, read into a Daf.
So each reader that moves to SqlDaf gets an if/else like this:

```python
table_sdaf = DB.recover_table_sdaf('bia_bif')
if table_sdaf is not None:
    rows_daf = table_sdaf.select_records_daf(
        keys, silent_error=True).to_daf()
else:
    table_daf = chunktable.recover_full_table_daf(
        dirname='bia_bif', keyfield='ballot_id')
    rows_daf = table_daf.select_records_daf(
        keys, silent_error=True)
rows_daf.apply_dtypes()
```

There are 11 of these in audit-engine-dev utilities.
They cover bia_bif, cvr_bif, blt_bif and the cvr_votes snapshots.
Two of them are the joins of combine_bifs and gen_fullbif.
Two more were added as helpers that serve only full_bif:
bif_utils.select_full_bif_rows_daf() and ballot_utils.select_bif_metadata_rows_daf().
More are coming as the rest of the pipeline moves.

## What the readers need

Most readers need only a few rows by key, for example:
- the rows of some keys, in the order of the keys
- KeyError for a missing key, or skip it with silent_error
- the result as an in-memory Daf with dtypes applied

Others filter, group or count, and then read the result into memory.

Two facts shape the code at each site:
- The two branches end differently. SqlDaf needs to_daf() to bring the rows into memory. Daf has no to_daf().
- SqlDaf rows come back typed. Rows read from a CSV are text until apply_dtypes().

## The ask

Design and build a general construct in daffodil, so that one call works on either kind of table.
Then tell AuditEngine how to use it.
AuditEngine will then replace the shims with it, one reader at a time.

The design is up to the daffodil thread.
Ray's aim is a small, general construct, not a helper for each table.
