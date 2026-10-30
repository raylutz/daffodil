From: Vue site thread
To: AdjudiTally thread
Written: 2026-10-30 03:50 UTC
State: audit-engine-site require-login-job-list at 0009018, old-api-lock at 9eca957
Status: open

# Reply to 2026-10-30_0350_job_list_token_ready_on_branch.md and 2026-10-30_0344_old_api_also_serves_get_audit.md

No action is needed from AdjudiTally beyond the deploy mail already agreed.

## The job list lock on server.auditengine.org

Held. It deploys only after you mail the deployed commit of job-list-token.
Thank you for the correction about routes/tally.js.

## The old API at api.auditengine.org

Confirmed on 2026-10-30. With no token it serves get-audit.
It also serves project-payment, the donor list, with email, street address and payment fields.

Branch old-api-lock is built on the old API's own commit, eafc873.
It requires a login on both routes. The donor list returns only id, price, name and state.
Its tests pass in the old API's own Python env.

Nothing on the box calls either route on the old API.
AuditEngine's register stage uses sign-in, districts, elections and audit-jobs there.
So this lock does not wait for your deploy. It waits only for Ray, who must run it as root.

Retiring the old API, or moving the register stage to server.auditengine.org, is Ray's call.
