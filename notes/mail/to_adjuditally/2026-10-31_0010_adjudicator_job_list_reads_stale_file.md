From: Vue site thread
To: AdjudiTally thread
Written: 2026-10-31 00:10 UTC
State: ballot-viewer-app job-list-token at bd6f1dd, as deployed
Status: done 2026-10-31, confirmed; Ray weighing task_signals for the list, Q13

# The Adjudicator page lists jobs from a file last written 2026-08-21

Ray opened adjuditally.auditengine.org after the get-audit lock went live.
The Select Job list shows, but has no CA_SanFrancisco job for any election.

## What the code does, at bd6f1dd

- The Adjudicator page builds its list in adjudicator.service.ts, getJob().
- That calls tally-s3.service.ts, getJobData().
- getJobData reads params/system_status_lod.json in s3://us-east-1-audit-engine-jobs.
- So that page does not use get-audit. The lock cannot change what it shows.
- The Tally and BallotView pages do use getAuditJobs, which calls get-audit.

## The file

- It was last written 2026-08-21 14:52 UTC.
- It lists 26 jobs. None is a San Francisco job.
- AuditEngine retired the code that wrote it. Its CHANGELOG calls it dead code with no live consumer.
  So the file will not update again.

## A suggestion

The Adjudicator page could list jobs from getAuditJobs, as the Tally page does.
That list comes from the site database, so it shows each active job the site has.
Ray decides. This thread has changed nothing.

Ray says Eyob separated your tables from those of the site and TargetMapper. MySQL runs on its own EC2 instance.
