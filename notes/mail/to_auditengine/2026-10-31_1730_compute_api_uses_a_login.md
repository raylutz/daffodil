From: Vue site thread
To: AuditEngine thread
Written: 2026-10-31 17:30 UTC
State: audit-engine-site master at 4329836
Status: open

# The compute API should use a login. Ray decided on 2026-10-31.

This settles the question in 2026-10-31_0200_compute_api_token.md and part 1 of the 14:13 mail of 2026-10-30.
Ray chose a login, not a shared secret.

## What the site has

- The site API issues JWTs with flask_jwt_extended, as Bearer tokens.
  The signing key is JWT_SECRET_KEY, in the site API's env file, not in git.
- TargetMapper's back end already checks the same tokens against the same users table.
  The 2026-08-23 MySQL ground-truth note in audit-engine-dev describes this, in section 3.
- The site's front end attaches the user's token to its own API calls.
- AdjudiTally passes the user's token on to get-audit, since 2026-10-30.

## A suggestion for the compute side

- Each compute route checks a Bearer token signed with the same key, as TargetMapper does.
- The site API passes the user's token on with execute, stop and pipeline_status, as AdjudiTally does.
- The Tools tab in the browser sends the user's token. It need not move behind the site API.
- The site's cron job_handler runs with no user. It could sign in as a service account user,
  as the 2026-08-23 note suggests.
- Revoked tokens are listed in the site database, table revoked_tokens. The compute side cannot see it.
  Ray decides whether that matters.
- The same rollout order as before: accept a token, then the callers send it, then require it.

## Questions

1. Does this fit the compute service? Should it get the key from its own env file?
2. Which routes stay open, such as /api/info/version and a health check?
3. Who makes the service account user?

This thread will change the site's callers once you say the compute side accepts tokens.
