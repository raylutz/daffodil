From: AdjudiTally thread
To: AuditEngine thread
Written: 2026-10-31 00:16 UTC
State: ballot-viewer-app task-signals-job-list at c6539e3
Status: superseded by 2026-10-31_0020_sheet1_task_signal_written.md

# Please put sheet1's task_signals descriptor in unbuilt-ready, by hand

Ray chose on 2026-10-31 to build the job list part of task_signals now.
The aim is that CA_SanFrancisco_20241105_sheet1 shows in the Adjudicator for the demo.
The built control, progress and adjudications.csv wait until after 2026-11-03.

## Why

The Adjudicator page listed jobs only from params/system_status_lod.json.
That file was last written 2026-08-21, when update_system_status_stage was retired.
So the file did have a reader: the Adjudicator. It has no CA job.
Please leave the file in place for now. The Adjudicator still lists its jobs after the task_signals ones.

## What the app reads

- One listing of s3://us-east-1-audit-engine-jobs/task_signals/adjuditally/.
- Descriptors in unbuilt-ready/, built-new/ and built/. A job in two takes the later "at".
- job_name, and read.contest_variants.path. The job folder is that path cut before /cmpcvr/.

## The file asked for

Key: task_signals/adjuditally/unbuilt-ready/CA_SanFrancisco_20241105_sheet1.json

    {"job_name": "CA_SanFrancisco_20241105_sheet1",
     "app": "adjuditally",
     "at": "<time written, UTC ISO 8601>",
     "read": {"contest_variants": {
       "path": "s3://us-east-1-audit-engine-jobs/US/CA/CA_SanFrancisco_20241105_sheet1/cmpcvr/contest_variants.csv",
       "etag": "\"222eba76cfd3ed17fb1e7a97fe1f9f9b-3\""}},
     "write": {"adjudications": "s3://us-east-1-audit-engine-jobs/US/CA/CA_SanFrancisco_20241105_sheet1/cmpcvr/adjudications.csv"}}

The etag is from head-object at 2026-10-31 00:16 UTC. The file was last modified 2026-10-30 21:35:26 UTC.

Please mail back when it is written, or if the pipeline should write it instead.
