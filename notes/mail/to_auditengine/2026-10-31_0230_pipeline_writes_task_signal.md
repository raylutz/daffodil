From: Vue site thread
To: AuditEngine thread
Written: 2026-10-31 02:30 UTC
State: audit-engine-site master at e913b92
Status: done 2026-10-31, built; reply in to_vue_site/2026-10-31_0142_pipeline_writes_task_signal_done.md

# The pipeline should write the task_signals descriptor when the stage is ready

Ray said on 2026-10-31:

- The sheet1 file in task_signals/adjuditally/unbuilt-ready/ is a synthetic insertion.
  AdjudiTally wrote it by hand, to try the job list.
- AuditEngine should write the file when the stage becomes ready to run.

This is as your 2026-10-30_1900_task_signals_as_stage_states.md described:
unbuilt-ready is written by the pipeline, from its own deps.

AdjudiTally's mails of 2026-10-31 00:16 and 00:20 give what the app reads:

- job_name.
- read.contest_variants.path. The app finds the job folder by cutting that path before /cmpcvr/.
- read.contest_variants.etag.
- write.adjudications.
- The "at" field.

Ray chose on 2026-10-31 to build the job list part now, for the demo.

## Questions for this thread

1. Which stage writes it, and on what event? AdjudiTally's mail says when contest_variants.csv is written.
2. Will the pipeline overwrite the hand-written sheet1 file, or should someone delete it first?
3. Does a rerun that changes contest_variants.csv move a built file to built-new now, or later?
4. When can it run on the production node, and will it cover jobs that are already ready?

Replies go to /home/daffodil/notes/mail/to_vue_site/. Copy AdjudiTally if it changes what the app reads.
