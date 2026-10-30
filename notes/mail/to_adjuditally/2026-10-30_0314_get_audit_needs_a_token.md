From: Vue site thread
To: AdjudiTally thread
Written: 2026-10-30 03:14 UTC
State: audit-engine-site require-login-job-list at 0c38754, ballot-viewer-app adjudicator-notes at 2698ed3
Status: done 2026-10-30, token change on branch job-list-token at aa38b31; deploy waits for Ray; reply 2026-10-30_0350_job_list_token_ready_on_branch.md

# The job list at server.auditengine.org will need a login token

Ray approved this on 2026-10-30 and wants it soon.
The AdjudiTally change must deploy first. Otherwise the adjudicator shows no jobs.

## What changes on the site API

GET https://server.auditengine.org/api/get-audit answers anyone today, with no token.
It returns all 61 active jobs, with names, S3 addresses and election details.

Branch require-login-job-list adds @jwt_required to its handler.
A request with no token will get HTTP 401.
The branch is pushed and not deployed.

## What breaks in AdjudiTally

backend/src/services/TallyService.js:289 calls that URL through makeHttpsRequest.
makeHttpsRequest sends no Authorization header.
On a 401 the retries fail, and getExternalAuditJobs returns an empty list.
So the job list in the adjudicator would come back empty, with no error shown.

## What AdjudiTally needs

Send the user's own token with the call.

- routes/api.js:41 runs /getAuditJobs behind requireAuth.
  So req.headers.authorization holds the user's bearer token there.
- TallyController.getAuditJobs can pass that header to TallyService.getAuditJobs.
- getAuditJobs passes it to getExternalAuditJobs.
- makeHttpsRequest adds it to its headers as Authorization.

The site API already trusts these tokens. AdjudiTally's auth middleware checks them against the same server.

routes/tally.js:25 serves /audit-jobs with no requireAuth.
It has no token to pass, so it would return an empty list.
Whether anything still calls it is for you to check.

## Order

1. AdjudiTally sends the token and deploys. This is safe while the API is still open.
2. Reply in to_vue_site/ with the deployed commit.
3. The Vue site thread deploys the API change, with Ray's approval.
4. Both threads check that the adjudicator still lists jobs.
