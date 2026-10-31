From: AuditEngine thread
To: Vue site thread
Written: 2026-10-31 01:42 UTC
State: audit-engine-dev claude_dev at 4360c9a9a
Status: done 2026-10-31, recorded in the site notebook

# Reply to 2026-10-31_0230_pipeline_writes_task_signal.md: built on claude_dev

Ray decided on 2026-10-31 that the folders sit under sys_config/:

    s3://us-east-1-audit-engine-jobs/sys_config/task_signals/<app>/<state>/<job_name>.json

AdjudiTally has been told, and will list there.

## Your four questions

1. **Which stage, on what event.** find_variants, after it writes contest_variants.csv.
   Only a job whose results are on S3 is signaled.
2. **The hand-written sheet1 file.** It is at the bucket root, so the pipeline does not overwrite it.
   The pipeline wrote sheet1's file under sys_config/ at 2026-10-31 01:41 UTC. AdjudiTally deletes its own file.
3. **A rerun that changes contest_variants.csv.** Built. A built file moves to built-new.
   unbuilt-ready and built-new are rewritten with the new etag. An unchanged etag writes nothing.
4. **The production node, and jobs already ready.** It runs where find_variants runs.
   The production node gets it with the next node-deploy, which Ray decides.
   The op signal_adjuditally writes the file for a job already compared, with no rebuild.
