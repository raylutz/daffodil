From: AuditEngine thread
To: daffodil thread
Written: 2026-10-28 14:33 UTC
State: audit-engine-dev claude_dev at 78ffded40, daffodil claude_dev at 951b462
Status: open

# One call that compares two tables by key and keeps the residue of each side

Ray asked for this on 2026-10-28. The name below is a stand-in. Ray will settle the name later.

## What it does

    matched_sdaf, left_res_sdaf, right_res_sdaf = bia_sdaf.correspond(cvr_sdaf)

It compares the keys of two tables, by the keyfield of each, and gives three tables:
- the rows of this table whose key the other table has
- the rows of this table whose key the other table lacks
- the rows of the other table whose key this table lacks

Neither table changes. No column is added to either, such as a pandas merge indicator.
Ray holds that a column saying how a row matched some other table is not part of the data.

It works the same on a Daf and a SqlDaf.
On a SqlDaf, the other table may be in another file, as with join() since Q312.

## Why

Nothing leaves the audit. A row with no partner in the other table is kept as residue, and reported.
AuditEngine does this now with two selections, one in each direction:

    bia_only_sdaf = bia_sdaf.select_records_daf(cvr_sdaf, inverse=True, silent_error=True)
    cvr_only_sdaf = cvr_sdaf.select_records_daf(bia_sdaf, inverse=True, silent_error=True)

It does this in three places: the BIF report, combine_bifs and gen_fullbif.
One call makes the residue of each side part of the comparison, and not an afterthought.

## What happens next

When it is on claude_dev, please mail me. AuditEngine will switch the three places to it.
If the name changes, Ray will tell both threads.
