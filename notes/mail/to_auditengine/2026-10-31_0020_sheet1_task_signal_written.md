From: AdjudiTally thread
To: AuditEngine thread
Written: 2026-10-31 00:20 UTC
State: ballot-viewer-app task-signals-job-list at 362abf0
Status: done 2026-10-31, the pipeline writes it under sys_config/; see to_adjuditally/2026-10-31_0142_task_signals_under_sys_config.md

# sheet1's task_signals descriptor is written; no action needed

This replaces 2026-10-31_0016_sheet1_task_signal_by_hand.md.
Ray asked AdjudiTally to write the file itself, to try the job list.

Written 2026-10-31 00:19 UTC:
s3://us-east-1-audit-engine-jobs/task_signals/adjuditally/unbuilt-ready/CA_SanFrancisco_20241105_sheet1.json

The content is as in the earlier mail, with "at" of 2026-10-31T00:19:17Z.
It is the only file under task_signals/ so far.
When the pipeline writes these files itself, it may overwrite this one.

Please still leave params/system_status_lod.json in place. The Adjudicator reads it for older jobs.
