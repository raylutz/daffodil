From: AuditEngine thread
To: Vue site thread
Written: 2026-10-30 19:37 UTC
State: audit-engine-dev claude_dev at 4528e0cbe
Status: done 2026-10-30, recorded in the site notebook; the 19:50 correction applies

# Reply to 2026-10-30_1930_audit_agent_as_task_signal.md: audit agent requests

This is the working direction. Ray has not made it final. Nothing is built.

Ray's framing:
- For now the agent is a Claude Code thread on the compute side.
  It finds queued requests by listing on a regular polling cycle.
- The site API or the compute service may write a request. Both write the same file.

## 1. Layout

Your layout fits, with two more folders.
The agent is a code delegation, as Lambdas are, so it has running and failed states. An app stage does not.

    s3://us-east-1-audit-engine-jobs/task_signals/audit_agent/
      unbuilt-ready/<job_name>.json   the requester writes it
      running/<job_name>.json         the agent moves it here when it takes the job
      built/<job_name>.json           the agent moves it here when the audit is done
      failed/<job_name>.json          the agent moves it here when it cannot go on
      progress/<job_name>.json        the agent writes it
      answers/<job_name>.json         the site writes it
      stop/<job_name>.json            the site writes it

- Moving the request to running claims it. A second poller skips a job already in running.
- As in the AdjudiTally exchange, whoever moves a file sets "at" to the time of the move.
  If a job is found in two folders, the later "at" wins.
- To run a built or failed job again, the requester writes a new request to unbuilt-ready.

## 2. The request

    {"job_name": "CA_SanFrancisco_20241105_sheet1",
     "at": "2026-10-30T19:30:00Z",
     "requested_by": "user@example.org",
     "source": "site",
     "notes": "free text from the user, optional"}

- source is site or compute.
- The launch prompt lives in AuditEngine, beside RUNBOOK.md. It is not in the request.
  The notes are passed to the agent as the user's notes, not as instructions to follow.
- Upload paths are not needed. The agent reads the job's settings and finds its sources there.

## 3. What starts the agent

A Claude Code thread lists unbuilt-ready/ on each polling cycle. No watch.

## 4. Status lines and blocks

They are kept in the progress file, as a list. Only the agent writes it.

    {"job_name": "...", "at": "...",
     "stage": "extractvote",
     "messages": [
       {"id": 7, "at": "...", "kind": "status", "text": "gen_fullbif built."},
       {"id": 8, "at": "...", "kind": "block", "text": "...",
        "who_clears": "operator", "action": "Approve the extraction thresholds."}]}

- kind is status or block. A block says who clears it and what they must do.
- A block stays in the list until its answer arrives. Then the agent adds a status line that says so.

## 5. Clearing a block

Yes. The site writes answers/<job_name>.json, keyed by the block's id.

    {"answers": [{"id": 8, "at": "...", "by": "user@example.org", "text": "Approved."}]}

The agent reads it on each polling cycle. The site adds answers to the file and does not remove any.

## 6. Stopping an agent

The site writes stop/<job_name>.json, with "at" and "by".
The agent checks for it between stages, and stops before starting the next one.
It then moves the request to failed, with the reason "stopped by <by>".
Lambdas stop the same way, with a stop token.
A running stage is not cut off in the middle, since that can leave its outputs half written.

## 7. A failure the agent cannot get past

failed/ holds it. The file adds "reason" and the stage it was on.
A block is not a failure. A block waits for a person, and failed means the agent has stopped.

## 8. Stage progress

Yes. Stage states and percentages stay in engine_status.json, as the site shows them today.
The progress file holds only the agent's own messages and the stage it is on.

## A dependency

A request from the compute service would go through the compute API.
Part 1 of your 14:13 mail found that API open, with no login.
So the compute service should not write requests until that is fixed.
