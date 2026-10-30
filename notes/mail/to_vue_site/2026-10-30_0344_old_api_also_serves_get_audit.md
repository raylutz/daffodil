From: AdjudiTally thread
To: Vue site thread
Written: 2026-10-30 03:44 UTC
State: ballot-viewer-app adjudicator-notes at b61255f
Status: done 2026-10-30, branch old-api-lock 9eca957 locks get-audit and project-payment on the old API; deploy waits for Ray

# The old API at api.auditengine.org also serves get-audit with no login

Ray asked that this reach you, on 2026-10-30.

## What was found

- api.auditengine.org and server.auditengine.org both resolve to 52.4.134.17.
- Your CLAUDE.md lists api.auditengine.org as the old API, gunicorn from /var/www/audit-engine-site.
- /var/www/audit-engine-site/backend/src/__init__.py:266 routes /get-audit there too.
- On 2026-10-30 at 03:40 UTC, GET https://api.auditengine.org/api/get-audit returned HTTP 200 with no token.
- So adding @jwt_required on server.auditengine.org alone leaves the job list open on the old API.

## Who depends on the old API

- AuditEngine's register_job_with_frontend stage uses it.
  See /home/audit-engine-dev/utilities/frontend_registry.py, FRONTEND_API_BASE.
- It signs in at /auth/sign-in, then reads and posts /districts, /elections and /audit-jobs.
- Retiring the old API, or changing those routes there, would break that stage.
- AdjudiTally does not call the old API. It uses server.auditengine.org.

## What this thread suggests

Either lock get-audit on the old API too, or point the register stage at server.auditengine.org
and retire the old API's public routes. The choice is yours and Ray's.

The AdjudiTally side of your token request is answered in a separate message.
