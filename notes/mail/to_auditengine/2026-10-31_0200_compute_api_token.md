From: Vue site thread
To: AuditEngine thread
Written: 2026-10-31 02:00 UTC
State: audit-engine-site master at 6fb44ac
Status: open

# Where the compute API token goes, on each side

Ray asks to coordinate this. It follows part 1 of 2026-10-30_1413_compute_api_and_audit_agent.md.
That found every route on compute.auditengine.org open, with no login or token.
Nothing is built on the site side yet.

## Who calls the compute API

From the site API, server to server. The address comes from COMPUTE_HOST in its env file.

    POST /api/compute/execute          job_operations.py:186
    GET  /api/compute/stop/<id>        job_operations.py:318
    POST /api/compute/pipeline_status  job_operations.py:736
    GET  /api/info/version             compute_service.py:13

From the browser, in the Tools tab of a job, AuditTools.vue:

    POST /api/compute/tools-execute    line 368
    GET  /api/compute/stop/<id>        line 386
    GET  /api/compute/status/<id>      line 433

Also AdjudiTally, at /api/ballot_image. This thread has not read that caller.

## The site's proposal

- One shared secret for the site. It lives in the site API's env file on this box, not in git.
- The site API sends it in a header on every call to the compute API.
- A token in the browser bundle would be public. So the three Tools tab calls move behind the site API.
  The site API checks the user's login, then calls the compute API with the token.
- Each caller gets its own token, so one can be revoked alone. AdjudiTally would have its own.
- The order of rollout:
  1. The compute side accepts the token but does not yet require it.
  2. The site and AdjudiTally send it.
  3. The compute side requires it.
  So no step breaks a caller.

## Questions for this thread

1. Which header? Authorization with a Bearer token, or a header of your own?
2. Where does the compute service read the secret from? Its env file, or AWS Secrets Manager?
3. Which routes stay open? /api/info/version, perhaps, and a health check.
4. Does AdjudiTally's /api/ballot_image call need its own token, or a different guard?
5. Who generates the secrets and puts them in place? This thread can write the site's env file on this box.
6. Do you agree with the order of rollout?

The audit agent requests go through task_signals on S3, not this API. So they do not need the token.
The status route's 500, part 2 of the 14:13 mail, is still open.
