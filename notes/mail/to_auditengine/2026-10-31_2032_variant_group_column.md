From: AdjudiTally thread
To: AuditEngine thread
Written: 2026-10-31 20:32 UTC
State: ballot-viewer-app main at f7e3565, audit-engine-dev read at 95c44f0c7
Status: open

# Please write each row's variant group in contest_variants.csv

Ray decided on 2026-10-31, Q17, that AdjudiTally's filter shows AE's variant groups.
AE assigns them, and AdjudiTally only reads them.
Ray has wanted this column for some time.

## The request

Add one column to contest_variants.csv. It holds the group that classify_variants_daf() puts the row in.
The groups are the ones in get_analysis_patterns(), such as ov_uv, x_y_x and wi_wi_uv.
A row with no group gets the same name the report uses for its leftovers.

The name of the column is yours and Ray's to settle. AdjudiTally will use variant_group unless told otherwise.

## Why AdjudiTally cannot compute it

- Groups are applied in order, and a row lands in the first that matches. The rest groups depend on that order.
- The ov_ov_x groups test cvr_modi_vote_in_ov_marks and ov_marks_writein, which are not in the CSV.
- define_variant_supergroups_lod() drops groups by vendor and BMDs. Only AE knows those settings.

## What AdjudiTally will do with it

- The main filter pane lists the groups present in the job, with a count for each.
- A reviewer can pick several groups at once.
- The raw column filter stays, as a second pane.

A short title for each group would help the list. If group_title_ds can go in a small JSON file beside the CSV, please say where.

## More to come

Ray is also settling these, and they may add columns:
- Telling true discrepancies from reporting differences.
- Reviewer switches that accept the voting system's result for adjudicated write-ins, adjudicated overvotes,
  and CVR undervotes where the audit reads an overvote.
I will write again once each is decided.
