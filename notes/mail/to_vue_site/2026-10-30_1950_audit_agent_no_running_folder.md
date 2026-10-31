From: AuditEngine thread
To: Vue site thread
Written: 2026-10-30 19:50 UTC
State: audit-engine-dev claude_dev at 872739620
Status: done 2026-10-30, recorded in the site notebook

# Correction to 2026-10-30_1937_audit_agent_requests.md: no running or failed folders

Ray decided on 2026-10-30. The audit agent is not a pure code delegation.
Its state is already shown in full in the site, since the site is driven by the pipeline state.
So the request uses the same folders as the app stages, with no running or failed folder.

    s3://us-east-1-audit-engine-jobs/task_signals/audit_agent/
      unbuilt-ready/<job_name>.json   the requester writes it
      built/<job_name>.json           the agent moves it here when the audit is done
      progress/<job_name>.json        the agent writes it
      answers/<job_name>.json         the site writes it
      stop/<job_name>.json            the site writes it

What changes from the 19:37 mail:
- Item 1: the agent does not claim a job by moving it to running.
  While the job is in unbuilt-ready, the pipeline state shows what is happening.
- Item 6: after a stop, the request stays in unbuilt-ready, and the agent adds a status line that it stopped.
  The site deletes the stop file to let it go on.
- Item 7: there is no failed folder. A failure the agent cannot get past is a block,
  with who must clear it and what they must do. The stage state shows where it is.

The rest of the 19:37 mail stands.
