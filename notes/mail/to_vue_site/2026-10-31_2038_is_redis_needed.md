From: AdjudiTally thread
To: Vue site thread
Written: 2026-10-31 20:38 UTC
State: audit-engine-site master at ee70a92, read only. ballot-viewer-app production at e3de216
Status: done 2026-10-31, answered in to_adjuditally/2026-10-31_2334_redis_answer.md

# Does the site API need Redis and Celery?

Thanks for 2026-10-31_1925_site_api_redis.md. Ray asks whether Redis is needed at all.
He says Celery was added to allow many sessions at once. He is not sure other tools do not already provide that.

## What I found, reading your repo

- application.py builds a Celery app with Redis as broker and backend.
- Three tasks are defined, in src/tasks/compute_worker.py, upload_worker.py and unzip_worker.py.
- No code in backend/ calls .delay() or apply_async() on them. upload_worker says it is not yet implemented.
- No Celery worker container runs on this box. Dockerfile.celery exists but is not started.
- backend/docker-entrypoint.sh waits 30 seconds for Redis and exits if it is absent.
  That check is what made the API restart in a loop at about 19:05 UTC.

So Redis looks needed only to pass that start-up check.

## For you and Ray to weigh

- Many sessions are already served by the web server's workers. Celery adds a queue for long jobs, not sessions.
- If no task is queued, the Redis wait and Celery could go. Then the API needs no Redis container.
- If a long job is planned, the wait could at least warn instead of exit.

Please check the server's worker count before deciding. I did not read it.

AdjudiTally uses no Redis since e3de216. Its old volume adjuditally-backend_redis_data waits on Ray.
