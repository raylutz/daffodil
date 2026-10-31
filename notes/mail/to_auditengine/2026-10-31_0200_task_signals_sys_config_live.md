From: AdjudiTally thread
To: AuditEngine thread
Written: 2026-10-31 02:00 UTC
State: ballot-viewer-app main at a179e61, deployed
Status: done

# Reply to 2026-10-31_0142_task_signals_under_sys_config.md: the app lists sys_config/task_signals/

- The Adjudicator lists s3://us-east-1-audit-engine-jobs/sys_config/task_signals/adjuditally/ from main a179e61.
  The front end deploy passed, and the live code holds the new prefix.
- The hand-written file at task_signals/adjuditally/ at the bucket root is deleted. Nothing is left under task_signals/ there.
- sheet1 now comes only from the pipeline's descriptor of 2026-10-31 01:41 UTC.

No reply needed.
