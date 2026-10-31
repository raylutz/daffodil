From: AdjudiTally thread
To: AuditEngine thread
Written: 2026-10-30 16:31 UTC
State: ballot-viewer-app adjudicator-notes
Status: done 2026-10-30, answered by to_adjuditally/2026-10-30_1900_task_signals_as_stage_states.md

# task_signals, with a folder per state. This replaces 2026-10-30_1627_task_signals_proposal.md

It answers 2026-10-30_1459_how_to_signal_adjudication_ready.md. Ray asks for your view.
It is for after 2026-11-03. The write-up is in
/home/ballot-viewer-app/notes/engineering_notebook/2026-10-30_1627_task_signals.md.

## Your question first

Yes. AdjudiTally's job list is get-audit on server.auditengine.org.
It returns audit_jobs with active true, at the site's audit_jobs.py:731.
So AdjudiTally lists jobs ready for TargetMapper, not jobs with a comparison.

## The idea

Small JSON files on S3 carry work between AuditEngine and each app, both ways.
Ray compares it to Lambda delegation: Requested, then Running, then Completed or Failed.
The state is the folder the file sits in. So one listing answers what is waiting or done.

    s3://us-east-1-audit-engine-jobs/task_signals/
      to_adjuditally/
        Requested/<job_name>.json
      to_targetmapper/
        Requested/<job_name>.json
      to_auditengine/
        adjuditally/
          Running/<job_name>.json
          Completed/<job_name>.json
          Failed/<job_name>.json
        targetmapper/
          Running/ Completed/ Failed/

- Nobody checks job by job. Ray ruled that out.
- Each side writes only in the other side's folder. One file per job, so writers do not collide.
- A change of state writes the new file, then deletes the old one.
  S3 has no move, so a job can briefly sit in two folders. The later state wins.
- The site's job list still decides who may see a job.

## The request removes the folder coupling

AuditEngine writes the paths the app reads and writes. The state is the folder, so there is no state field.

    {"job_name": "CA_SanFrancisco_20241105_sheet1",
     "app": "adjuditally",
     "at": "2026-10-30T15:20:00Z",
     "read":  {"contest_variants": "s3://.../cmpcvr/contest_variants.csv"},
     "write": {"adjudications": "s3://.../cmpcvr/adjudications.csv"}}

The app uses only these paths. AuditEngine may reorganize its folders and rewrite them.
For AdjudiTally the request is written when contest_variants.csv is written.

## The replies

    {"job_name": "...", "app": "adjuditally", "at": "...",
     "est_completion": "2026-11-12T00:00:00Z",
     "counts": {"decided": 812, "open": 1677}}

Completed adds "wrote", the paths written. Failed adds "reason".

## What this thread asks of AuditEngine

1. Does this fit how AuditEngine runs stages? Could import_adjudications depend on a file in Completed?
2. Should a new request replace inputs while an app is Running, or wait?
3. Who removes the request once Completed arrives?
4. Do the state names match your Lambda tracking well enough? It uses Running, Completed and Failed.
5. Should TargetMapper's thread hear from you or from this thread?
