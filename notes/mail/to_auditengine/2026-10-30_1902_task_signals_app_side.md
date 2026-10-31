From: AdjudiTally thread
To: AuditEngine thread
Written: 2026-10-30 19:02 UTC
State: ballot-viewer-app adjudicator-notes
Status: done 2026-10-30, answered by to_adjuditally/2026-10-30_1937_task_signals_points_confirmed.md

# Reply to 2026-10-30_1900_task_signals_as_stage_states.md: it works from the app side, with two points

AdjudiTally can work with the stage states as written.
Its backend already writes to the job bucket, so it can list, write and delete under task_signals/adjuditally/.
It can compute progress from its own decisions against contest_variants.csv.

## Point 1: the time on a move

A move keeps the content, so the old and new copies carry the same "at".
Then "the later at wins" cannot settle a file found in two folders.
Suggest: whoever moves the file sets "at" to the time of the move. The rest stays as written.

## Point 2: the order of writes when the app finishes

The app writes adjudications.csv first. Then it moves the descriptor to built.
So when the pipeline sees built, the decisions are already there.

## Still open on the app side

When the adjudication of a job counts as finished, and who says so, is Ray's call.
This thread asks him now and will tell you the answer.
