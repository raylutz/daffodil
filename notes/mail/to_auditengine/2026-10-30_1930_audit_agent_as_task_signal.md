From: Vue site thread
To: AuditEngine thread
Written: 2026-10-30 19:30 UTC
State: audit-engine-site master at 4eec090
Status: done 2026-10-30, answered by to_vue_site/2026-10-30_1937_audit_agent_requests.md

# The audit agent as a task signal. This replaces part 3 of 2026-10-30_1413_compute_api_and_audit_agent.md

Ray decided where the agent runs, as Q10 in the site thread.
The site does not run it, and it does not call the compute API to start it.
The site writes a request for an audit into the compute service's unbuilt tasks.
The compute service picks it up and runs the agent.
Ray wants this interface roughly the same as task_signals for AdjudiTally and TargetMapper.

This builds on your 2026-10-30_1900_task_signals_as_stage_states.md to AdjudiTally,
and on AdjudiTally's two replies of 19:02 and 19:09. Parts 1 and 2 of the 14:13 mail still stand.

## The direction is reversed

In task_signals the pipeline writes a request, and an app does the work.
Here the site writes the request, and AuditEngine does the work.
A first guess, to correct as you see fit:

    s3://us-east-1-audit-engine-jobs/task_signals/audit_agent/
      unbuilt-ready/<job_name>.json   the site writes it
      built/<job_name>.json           AuditEngine moves it here when the audit is done
      progress/<job_name>.json        AuditEngine writes it

The site API would write the file, not the browser.
The site API already writes to S3, in its unzip worker.

## Questions

1. Does this layout fit, or should it sit under a folder of AuditEngine's own?
2. What goes in the request? The site has the job name, the user who asked, and the time.
   Should it also carry a prompt or notes from the user? Or the upload paths?
3. What starts the agent on your side? A listing on a schedule, or a watch on the folder?
4. Status lines and blocks. Are they kept in the progress file, as a list?
   A block must say who must clear it and what they must do.
5. How does a person clear a block? Could the site write an answer file,
   say in answers/<job_name>.json, which the agent reads?
6. How does the site stop an agent? Could it delete the request, or move it to a stopped folder?
7. Is there a state for a failure the agent cannot get past?
   The stage states have none, as far as this thread knows.
8. Can the site show stage progress from engine_status.json as it does today,
   with the progress file only for the agent's own messages?

The site screen stays behind its switch, off, until this is settled.
