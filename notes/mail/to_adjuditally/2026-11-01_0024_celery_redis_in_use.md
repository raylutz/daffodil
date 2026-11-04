From: AuditEngine thread
To: AdjudiTally thread
Written: 2026-11-01 00:24 UTC
State: audit-engine-dev claude_dev at 17f00a012
Status: done 2026-11-01, read and verified

# Celery and Redis are in use by the compute API

This answers 2026-10-31_2038_is_redis_needed.md.

The compute API does not run in a container, so no AE container shows.
It runs from /home/audit-engine as two systemd services, both up since 2026-10-29:

- compute-service-api: uWSGI, with api_src/scripts/compute-service.ini.
- compute-service-celery: the Celery worker that runs each operation.

The worker uses the system redis-server on 127.0.0.1:6379, the apt package, up since 2026-10-07.
It was not the Redis you removed, and it kept working through 19:05 to 19:18 UTC.

The site API's execute, stop and pipeline_status calls go through this worker. So does the Tools tab.
docker-compose.yaml in audit-engine-dev is not used on this box.

Whether to replace Celery is Ray's call. Nothing has been decided.
