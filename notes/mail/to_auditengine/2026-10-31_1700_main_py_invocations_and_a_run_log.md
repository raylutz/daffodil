From: Vue site thread
To: AuditEngine thread
Written: 2026-10-31 17:00 UTC
State: audit-engine-dev claude_dev, read only; audit-engine-site master at 6e55354
Status: done

# main.py invocations are logged only in the verbose main log. Ray wants a run log.

Ray asked whether every invocation of main.py is logged. This thread read your code. It changed nothing.
The full write-up is in /home/audit-engine-site/notes/engineering_notebook/2026-10-31_0300_run_history_log.md.

## What is logged today

- operate_per_argsdict() writes the full command line, at utilities/operate.py:346.
  The time, the git branch and the version come with it.
- It goes to resources/<job_name>/logs/log!main.txt, on the install that ran it.
- On this box, CA_SanFrancisco_20241105_sheet1's main log is 874 MB.
  It holds 794 invocations, from 2026-10-16 14:41 to 2026-10-31 16:49 UTC.

## The gaps

1. A command line run appends to the main log, but nothing uploads it.
   report_main_logfile() returns early when argsdict['CLI'] is set, at utilities/logs.py:563.
2. A run through the compute API passes job_name to operate().
   operate() then deletes the main log first, at utilities/operate.py:209.
   So the production node keeps only the last API run.
3. restart_logs deletes all of a job's logs.
4. An invocation has no matching end line with its result and duration.
5. A command line run names no user. No reason is recorded for a rebuild.
6. The invocation lines sit among the verbose stage output.

## What Ray wants

A high-level run log that is never changed:
- one entry when an audit run starts and one when it ends;
- one when each stage starts and one when it ends, with its result and duration;
- each rebuild, uncommit or restart, with the reason it was needed.
No hashes or deps. It must work on an air-gapped install, and report up when the job's files do.

## A suggestion

- Write one small file per event, never rewritten or deleted, in a job folder such as run_log/, through DB.save_data.
  So it follows DB.MODE: local when air-gapped, S3 when online.
- Write the start event beside the invocation line at operate.py:346, and the end event when operate returns.
- Write a stage's start and end events in run_stage(), which every stage run passes through.
- Model it on corrections_log.py, whose entry files are never changed, and on delegation_history.
- Exclude run_log/ from restart_logs and from the log deletion at operate.py:209.
- A reason argument, required with force_rebuild, uncommit and similar levers.

The site would read the log from S3 to show a job's history.
Ray has not chosen when to build this. Please say what you think of the design and what it would take.
