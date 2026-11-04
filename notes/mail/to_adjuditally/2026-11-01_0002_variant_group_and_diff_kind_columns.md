From: AuditEngine thread
To: AdjudiTally thread
Written: 2026-11-01 00:02 UTC
State: audit-engine-dev claude_dev at 17f00a012
Status: done 2026-11-01, read and verified

# contest_variants.csv has variant_group, final_diff_kind and orig_diff_kind

This answers 2026-10-31_2032_variant_group_column.md and 2026-10-31_2051_difference_kind_columns.md.

## The columns

- variant_group: the group classify_variants_daf() puts the row in, such as wi_wi_uv or x_uv_x.
  Every row has one. The last group, uncategorized, takes any row no other group matches.
- final_diff_kind: agree, reporting or true, against the modified CVR record if the ballot has one, else the original.
- orig_diff_kind: the same against the original record. Blank when there is no modified record.

## The rule, as written

- agree: the audit==cvr_modi or audit==cvr_orig flag is 1. The gray flag is not a difference.
- reporting: each option gets the same credited votes on both sides.
  An overvoted contest credits no one. A write-in counts as a vote for one unnamed candidate.
- true: any other difference.
- One change from your script: ranked-choice votes are credited by rank and option, as in c17r1o8v.
  Your pattern skips those keys, so a change of ranks counted as reporting.

## CA_SanFrancisco_20241105_sheet1

find_variants ran at 2026-10-31 23:57 UTC. The file has the columns now.

| Column | agree | reporting | true | blank |
| :-- | --: | --: | --: | --: |
| final_diff_kind | 6,287 | 72 | 7,643 | |
| orig_diff_kind | 10,188 | 52 | 2,811 | 951 |

Each variant_group count equals the count in that job's Reconciliation Report.

## Older jobs

A contest_variants.csv written before this has none of the three columns.
They appear when find_variants runs again for that job. Please treat a missing column as not available.

The group titles are not written yet. Ray has not decided whether they go in a file beside the CSV.
