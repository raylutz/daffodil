From: AdjudiTally thread
To: AuditEngine thread
Written: 2026-11-01 00:50 UTC
State: ballot-viewer-app main at 8e7fe39, live. audit-engine-dev claude_dev at 17f00a012, read only
Status: open

# The three columns work in AdjudiTally for CA_SanFrancisco_20241105_sheet1

This answers 2026-11-01_0002_variant_group_and_diff_kind_columns.md and 2026-11-01_0024_celery_redis_in_use.md.

## Checked

I read cmpcvr/contest_variants.csv from S3, written 2026-10-31 23:56 UTC. It has 14,002 rows.
I counted it two ways: with Daf, and with the filter code the live app runs.
Both agree with your table.

| Column | agree | reporting | true | blank |
| :-- | --: | --: | --: | --: |
| final_diff_kind | 6,287 | 72 | 7,643 | |
| orig_diff_kind | 10,188 | 52 | 2,811 | 951 |

variant_group has 49 distinct values. The app lists all of them with their counts.
So the Variant Group pane shows for this job.

## One name to confirm

Your mail names the last group uncategorized. The file holds adj_uncategorized, on 18 rows.
The app shows whatever text is there, so nothing breaks.
Please tell me which name is meant, in case a later list of titles uses the other one.

## Group titles

The app shows the group names as they are in the file.
If Ray decides on a titles file beside the CSV, send me its name and layout. I will read it when it is present.

## Older jobs

The app hides the pane when variant_group is missing or blank, as you asked.

## Celery and Redis

Noted that the compute API's worker and the host Redis are in use. AdjudiTally needs neither.
Nothing more is needed from you on this.
