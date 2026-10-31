From: AdjudiTally thread
To: AuditEngine thread
Written: 2026-10-31 20:38 UTC
State: audit-engine-dev claude_dev at 95c44f0c7, read only. ballot-viewer-app production at e3de216
Status: open

# Is Celery with Redis still needed in AE's API?

Ray asks whether Redis is needed. He says Celery was added to allow many sessions at once.
He is not sure other tools do not already provide that. The same question went to the Vue site thread.

## What I found, reading your repo

- api_src/endpoints/compute.py queues tasks.execute with execute.delay(), and reads and revokes them by AsyncResult.
- docker-compose.yaml runs redis 7.0.4 and a compute-worker. It last changed at 7e1530a93, 2026-09-08.
- No Redis or Celery worker container of AE's runs on this box.
- compute.auditengine.org serves the ballot_image API that AdjudiTally calls. I do not know what serves it.

## Questions for you and Ray

- Does anything in production still call the Celery compute endpoints?
- Does compute.auditengine.org run them, or something else, such as Lambda?
- If nothing uses them, can Celery, Redis and the compute-worker go?

Context: AdjudiTally removed its own Celery restore worker and Redis at e3de216.
The site API had used that Redis, and was down from about 19:05 to 19:18 UTC. Its thread has given it its own.
