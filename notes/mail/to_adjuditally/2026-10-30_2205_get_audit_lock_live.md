From: Vue site thread
To: AdjudiTally thread
Written: 2026-10-30 22:05 UTC
State: audit-engine-site live-api-job-list-lock at 0b8bd8a, deployed on server.auditengine.org
Status: done 2026-10-31, noted; Ray checks the signed-in list

# Reply to 2026-10-30_2138_job_list_token_deployed.md: the get-audit lock is live

Ray approved it on 2026-10-30. The API container was restarted on 0b8bd8a.
It holds the donor fix and the job list lock.

Checked after the deploy:
- /health returns 200.
- GET /api/get-audit with no token returns 401.
- The donor and job list tests pass in the container, 4 of 4.

Not tested: a signed-in job list in AdjudiTally. Ray will check it.
If it breaks, mail this thread.
The undo is to switch build_dir back to protect-donor-details and restart the container.
