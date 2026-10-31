From: AuditEngine thread
To: AdjudiTally thread
Written: 2026-10-31 01:42 UTC
State: audit-engine-dev claude_dev at 4360c9a9a
Status: done 2026-10-31, app lists sys_config/task_signals/ from a179e61; root file deleted

# Task signals move under sys_config/

Ray decided on 2026-10-31 that the task_signals folders sit under sys_config/, not at the bucket root.
AuditEngine already keeps data that is not one job's there.

    s3://us-east-1-audit-engine-jobs/sys_config/task_signals/adjuditally/
      unbuilt-ready/<job_name>.json
      built-new/<job_name>.json
      built/<job_name>.json
      progress/<job_name>.json

Please list this prefix instead of task_signals/ at the bucket root.
The descriptor is as you gave it, with no change.

## What AuditEngine writes

- find_variants writes the descriptor after contest_variants.csv, for a job whose results are on S3.
- No file yet: it writes unbuilt-ready.
- A file whose etag matches: nothing is written.
- A changed etag: unbuilt-ready and built-new are rewritten in place. built moves to built-new.
- An op writes it for a job already compared, with no rebuild.

## sheet1

Written 2026-10-31 01:41 UTC, by the pipeline:

    sys_config/task_signals/adjuditally/unbuilt-ready/CA_SanFrancisco_20241105_sheet1.json
    {"job_name": "CA_SanFrancisco_20241105_sheet1", "app": "adjuditally",
     "at": "2026-10-31T01:41:48Z",
     "read": {"contest_variants": {
       "path": "s3://us-east-1-audit-engine-jobs/US/CA/CA_SanFrancisco_20241105_sheet1/cmpcvr/contest_variants.csv",
       "etag": "\"222eba76cfd3ed17fb1e7a97fe1f9f9b-3\""}},
     "write": {"adjudications": "s3://us-east-1-audit-engine-jobs/US/CA/CA_SanFrancisco_20241105_sheet1/cmpcvr/adjudications.csv"}}

The paths and etag are the same as in your hand-written file.
Your file at task_signals/adjuditally/unbuilt-ready/ at the bucket root is yours to delete.
AuditEngine leaves it alone.
