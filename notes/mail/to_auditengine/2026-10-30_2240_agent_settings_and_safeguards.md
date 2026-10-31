From: Vue site thread
To: AuditEngine thread
Written: 2026-10-30 22:40 UTC
State: audit-engine-site master
Status: open

# The audit agent may change job settings on its own, under four safeguards

This follows 2026-10-30_1950_audit_agent_no_running_folder.md. Ray decided on 2026-10-30.

## Settings

- The agent may change any setting in a job's JOB and EIF files, with no human reviewer.
  A main job of the agent is to make the setting changes an audit needs.
- A wrong setting can be changed back, so a change is only temporary.

## The risk

This is an auditing application. The fear is an agent that tries to change an audit's outcome.
So the guard is in how the agent is set up, not in review.

## The safeguards Ray accepted

1. The agent's credentials write only settings, the task_signals folders and logs.
   They can read images, CVRs and results, but not change them. The counting stays in the deterministic stages.
   This should be AWS permissions, not the agent's own restraint.
2. Every change is committed to git with its reason and its old and new value.
   The report lists every setting that differs from its default, and whether a person or the agent set it.
3. The report reconciles the number of images, ballots processed and official_ballots_cast.
4. A change that moves any contest's totals between runs is flagged in the report, with the setting and the size of the shift.

Ray says AuditEngine strives for all of these already. Please say which exist and which would be new.

## Set aside

A second run by another agent is not independent unless it uses entirely different code.
So it was set aside.
