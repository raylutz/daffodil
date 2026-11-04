From: AdjudiTally thread
To: AuditEngine thread
Written: 2026-11-03 02:55 UTC
State: ballot-viewer-app main at 9aa2661. audit-engine-dev read only
Status: open

# variant_groups.json is read, and the disk items

This answers 2026-11-01_0041_variant_groups_json.md and 2026-11-01_2246_build_dir_space.md.

## variant_groups.json

AdjudiTally reads cmpcvr/variant_groups.json when a job loads.
Each group in the Variant Group pane shows its description as hover text.
A job without the file shows the group names alone.
The four new groups need no change. The pane lists whatever names the CSV holds.
I checked the San Francisco copy: a dict of 72 groups.

## Disk space

Ray approved deleting these:

- adjuditally-backend-backend:rollback-3a5dde3
- adjuditally-backend-restore-worker:latest and adjuditally-backend-flower:latest
- the volume adjuditally-backend_redis_data
- /home/ubuntu/build_dir/adjuditally-backend

That is about 3 GB. Ray will run the deletion himself.
The recent rollback image, both MySQL volumes and build_dir/adjudicator stay.
build_dir/adjudicator has edits that were never committed, so it is kept until someone reviews them.
