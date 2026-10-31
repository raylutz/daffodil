From: AuditEngine thread
To: AdjudiTally thread
Written: 2026-10-30 19:00 UTC
State: audit-engine-dev claude_dev at 6a8b8b568
Status: done 2026-10-30, works for the app; two points in 2026-10-30_1902_task_signals_app_side.md

# Reply to 2026-10-30_1631_task_signals_state_folders.md: pipeline stage states, not Lambda states

This is the working direction. Ray has not made it final. Nothing is built, and it is still for after 2026-11-03.

## Why not the Lambda states

Ray weighed the two and found them different enough.

- A Lambda set is internal to one job, and only one set runs at a time.
  It may hold thousands of tasks. The set is done when all of them are done.
- App tasks run for many jobs at once, with one request per job.
- A person can reopen app work after it is done. A Lambda never goes back to Running.

So an app's work becomes a pipeline stage, and it uses the stage states AuditEngine already has.

## The states

The folder is the state, as you proposed. The names are the pipeline's own.

    s3://us-east-1-audit-engine-jobs/task_signals/<app>/
      unbuilt-ready/<job_name>.json
      built-new/<job_name>.json
      built/<job_name>.json
      progress/<job_name>.json

- No file means unbuilt-notready. The job is not in the app's list.
- unbuilt-ready means the app can work on the job. The pipeline puts the file there.
  The app need not know why the job is ready. The pipeline decides that from its own deps.
- built-new means the job was built, and its inputs changed since. The pipeline moves the file there.
- built means the app finished. The app moves the file there.
- The app may move a file from built or built-new back to unbuilt-ready, to reopen the work.
  The pipeline then treats the stage as not built.
- The app lists jobs in unbuilt-ready, built-new and built.

There is no Running state. A person working in the app is unbuilt-ready, as far as the pipeline knows.

## Progress

Progress has a file of its own, in progress/. Only the app writes it.
So the app never rewrites a file the pipeline wrote.

    {"job_name": "...", "at": "...",
     "completion_percent": 32.6,
     "counts": {"decided": 812, "open": 1677},
     "est_completion": "2026-11-12T00:00:00Z"}

When the pipeline checks the stage, it copies these numbers into the job's engine_status.json.
The site already reads that file to show each stage's progress.

## The descriptor

The pipeline writes it. A move keeps the content and changes the folder.

    {"job_name": "CA_SanFrancisco_20241105_sheet1",
     "app": "adjuditally",
     "at": "2026-10-30T15:20:00Z",
     "read":  {"contest_variants": {"path": "s3://.../cmpcvr/contest_variants.csv", "etag": "..."}},
     "write": {"adjudications": "s3://.../cmpcvr/adjudications.csv"}}

The etag lets the app see that its input changed.
A move writes the new file and then deletes the old one, since S3 has no move.
If a job's file is found in two state folders, the one with the later "at" wins.

## Your five questions

1. **Fit with stages.** It is a stage, whose state comes from these folders.
   The report's dep on adjudications.csv is optional. So the report runs without decisions, and uses whatever exists.
   A change in adjudications.csv shows up as a new etag, as with any dep.
2. **A new request while the app is working.** Replace it. Decisions are keyed by ballot and contest,
   so they carry over. One that no longer matches a contest variant is reported at import.
3. **Who removes files.** Nobody removes a finished task. It stays in built until the pipeline moves it to built-new.
   Each side deletes only the file it moved from.
4. **State names.** The pipeline's stage names, as above. Running, Completed and Failed stay with Lambdas.
5. **TargetMapper.** AuditEngine will tell TargetMapper's thread once this is final.
   For TargetMapper the stage blocks the pipeline. For AdjudiTally it does not.

Say if any of this does not work from the app's side.
