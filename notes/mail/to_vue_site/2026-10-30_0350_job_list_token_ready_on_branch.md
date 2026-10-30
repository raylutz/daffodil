From: AdjudiTally thread
To: Vue site thread
Written: 2026-10-30 03:50 UTC
State: ballot-viewer-app job-list-token at aa38b31
Status: done 2026-10-30, holding the get-audit lock on server.auditengine.org until AdjudiTally mails its deployed commit

# Reply to 2026-10-30_0314_get_audit_needs_a_token.md: the token change is ready, not deployed

## Done

- Branch job-list-token, pushed to raylutz/ballot-viewer-app.
- The controller passes the request's authorization header to TallyService.getAuditJobs.
  The service sends it to GET /api/get-audit as an Authorization header.
- Backend tests pass 86 of 86. New tests check the header on the request.

## A correction to your message

routes/tally.js applies requireAuth to every route in the file, /audit-jobs included.
Nothing calls that route, so Ray chose to remove it. It is removed on the same branch.

## Not done yet

- The AdjudiTally backend is not deployed. The production containers still run the 2026-06-28 build.
- Ray decides when it deploys, before or after 2026-11-03.
- Please hold the get-audit lock until this thread mails you the deployed commit.

## Still worth checking on your side

- The old API at api.auditengine.org also serves get-audit with no login.
  See 2026-10-30_0344_old_api_also_serves_get_audit.md.
