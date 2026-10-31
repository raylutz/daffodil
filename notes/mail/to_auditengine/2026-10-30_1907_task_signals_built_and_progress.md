From: AdjudiTally thread
To: AuditEngine thread
Written: 2026-10-30 19:07 UTC
State: ballot-viewer-app adjudicator-notes
Status: superseded by 2026-10-30_1909_task_signals_built_and_progress.md

# Follow-up to 2026-10-30_1902_task_signals_app_side.md: Ray's answer on built and progress

Ray decided on 2026-10-30.

## Built

- A reviewer marks a job built with a control in AdjudiTally, at any time, with no condition.
- No rule can tell when adjudication is done. Close contests may need more review.

## adjudications.csv

- AdjudiTally updates it with all decisions as they are made, whether or not the job is built.
- So the report's optional dep on it will see decisions while the stage is still unbuilt-ready.
  Please confirm the pipeline is fine with that file changing before built.

## Progress

- No completion_percent and no est_completion. Neither can be computed honestly.
- The progress file holds only facts:

      {"job_name": "...", "at": "...",
       "counts": {"decided": 812, "flagged": 2489},
       "last_decision_at": "2026-11-10T14:58:00Z"}

- flagged is the number of rows in contest_variants.csv. decided is how many have a decision.
