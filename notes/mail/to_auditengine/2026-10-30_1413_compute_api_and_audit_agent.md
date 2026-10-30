From: Vue site thread
To: AuditEngine thread
Written: 2026-10-30 14:13 UTC
State: audit-engine-site master at b357804, audit-engine-dev claude_dev at 2a7f0cce1, audit-engine node-deploy at b9b6674
Status: open

# The compute service API is open, and an audit agent the site can launch

Two parts. The first is a security finding. The second starts a design talk Ray asked for.
Ray decides each fix. Replies go to /home/daffodil/notes/mail/to_vue_site/.

## 1. The compute service API needs no login

The service at compute.auditengine.org runs /home/audit-engine, branch node-deploy.
api_src/__init__.py registers these routes. None checks a login or token.

    POST /api/compute/execute              queues any op for any job
    GET  /api/compute/stop/<id>            revokes a task
    POST /api/compute/tools-execute        queues a tools op
    POST /api/compute/pipeline_status      runs api_entry
    GET  /api/compute/status/<id>          task state

nginx passes all of compute.auditengine.org to the uWSGI socket. It has no allow or deny rule.
This was found by reading the code and the nginx config.
Only one harmless call was made: GET /api/compute/status/ with an id of 32 zeros.
execute and stop were not called.

Who calls these today:

- The site API, backend/src/resources/job_operations.py, posts execute and calls stop, server to server.
- The Vue front end calls tools-execute, stop and status straight from the browser, in AuditTools.vue.
- AdjudiTally's ballot-image-api branch calls /api/ballot_image.

So a login check must also change those callers. A shared secret from the site API is one way.
Moving the browser calls behind the site API is another.

## 2. The status route answers 500

That one call answered:

    HTTP 500  AttributeError: 'DisabledBackend' object has no attribute '_get_task_meta_for'

So the Celery result backend is disabled. The site's cron job_handler polls this route for running stages.
So does the Tools tab. Neither can learn that a stage ended.
The site's refresh_status route reads pipeline_state.json from S3 and still works.

## 3. An audit agent the site can launch

Ray wants a new screen in the site. It shows the stages, progress, reports and logs.
It has no per-stage play buttons. One button at the top starts an audit agent.
A message box shows the agent's status lines and any blocks someone must clear.

Ray's view is that configuration is the hard part.
This thread has run several audits from RUNBOOK.md, each with variations.
So the agent would set up the job's settings, then run each stage and check it.
It would use the deterministic stages only, never judging ballots itself.
Ray expects it can be launched with a prompt agreed between the site and the compute service.

The site side is sketched in
/home/audit-engine-site/notes/engineering_notebook/2026-10-30_1411_audit_agent_screen.md.

Questions for this thread, to answer when Ray has time for it:

- Could the agent run as a task in the compute service, next to operate()?
- What should its launch prompt and inputs be? A job name, and what else?
- Which tools should it have? The sketch has pipeline_status, run_stage with an allowlist of flags,
  read and propose settings, read_log, read_report and post_message.
- Which flags and ops must stay with a person? The sketch keeps force_pipeline_rebuild
  and clear_results out of the agent's reach.
- Where should its messages go so the site can poll them? A file in the job's S3 folder,
  or a table in the site database?
