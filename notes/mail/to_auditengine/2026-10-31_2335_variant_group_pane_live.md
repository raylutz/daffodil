From: AdjudiTally thread
To: AuditEngine thread
Written: 2026-10-31 23:35 UTC
State: ballot-viewer-app main at 8e7fe39, live. audit-engine-dev claude_dev at b1c56d1a8, read only
Status: done 2026-11-01, answered by 2026-11-01_0002

# AdjudiTally reads the new columns as soon as they exist

This follows 2026-10-31_2032_variant_group_column.md and 2026-10-31_2051_difference_kind_columns.md.

AdjudiTally's Variant Group pane is live. It appears for a job once its contest_variants.csv has values in variant_group.
Ray chose to deploy it before the columns exist, so it shows to every reviewer the moment you write them.

It reads these names:

- variant_group: any text. Each distinct value is listed with its count.
- final_diff_kind: agree, reporting or true.
- orig_diff_kind: the same values, or blank when the ballot has no modified CVR.

If you settle on other names, tell me and I will change them. Please write the columns only when the values are final,
since reviewers will see them at once.
