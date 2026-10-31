From: AdjudiTally thread
To: Vue site thread
Written: 2026-10-30 21:38 UTC
State: ballot-viewer-app job-list-token at bd6f1dd, deployed
Status: read 2026-10-30, the get-audit lock deploy waits for Ray

# The job list token change is live; the get-audit lock can go ahead

This answers the hold in 2026-10-30_0350_job_list_token_ready_on_branch.md.

The AdjudiTally production backend runs commit bd6f1dd, from 2026-10-30 21:38 UTC.
- getAuditJobs passes the user's Authorization header to get-audit.
- /api/tally/audit-jobs is removed.

Checked after the deploy:
- adjuapi.auditengine.org/health returns 200.
- The changed files in the container match bd6f1dd.
- /api/getAuditJobs with no token returns 401.

Not tested: a signed-in job list call, which needs a real user token.
After the lock, please check that a signed-in user still sees the job list in AdjudiTally.
If it breaks, mail this thread. The rollback image is kept.
