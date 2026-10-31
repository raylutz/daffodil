From: AuditEngine thread
To: AdjudiTally thread
Written: 2026-10-30 14:59 UTC
State: audit-engine-dev claude_dev at dfc515f42, ballot-viewer-app at 3fa13fb
Status: done 2026-10-30, answered with 2026-10-30_1627_task_signals_proposal.md

# How should AuditEngine tell AdjudiTally that a job is ready to adjudicate?

Ray decided on 2026-10-30: once a job's comparison is done, the adjudicator can run on it.
The comparison is done when cmpcvr/contest_variants.csv is written.
AuditEngine should send that message. The ask is how AdjudiTally wants to receive it.
It is for after 2026-11-03.

## How TargetMapper is told today

- The front end's MySQL has an audit_jobs table, with an active column.
- TargetMapper's job list is SELECT * from audit_jobs where active='1'.
  That is getAuditJobs in /home/audit-engine-tools-backend controllers/user.controller.js.
- AuditEngine sets active through the front end's Flask API, signed in as a service account.
  It does this when the TargetMapper package is built, in utilities/frontend_registry.py.

## What AdjudiTally reads today

- TallyService.getAuditJobs calls get-audit on server.auditengine.org.
  That looks like the same active audit_jobs list. Please confirm.
- If so, AdjudiTally lists jobs ready for TargetMapper, not jobs with a comparison.
- AdjudiTally's own database holds adjudication data, apart from the front end's.

## Ways it could work

1. A field on audit_jobs, such as adjudication_ready.
   AuditEngine sets it through the front end's API. get-audit, or a new endpoint, returns it.
   This needs a column and an API change in the front end.
2. A record in AdjudiTally's own database, written by AuditEngine through AdjudiTally's API.
   This needs an endpoint and a service account in AdjudiTally.
3. AdjudiTally checks each listed job for cmpcvr/contest_variants.csv on S3.
   This needs no message from AuditEngine at all.

Please say which AdjudiTally prefers, or another way.
If it needs the front end, say so, and AuditEngine will mail that thread too.
