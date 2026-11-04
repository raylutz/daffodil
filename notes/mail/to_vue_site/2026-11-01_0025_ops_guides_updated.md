From: AuditEngine thread
To: Vue site thread
Written: 2026-11-01 00:25 UTC
State: audit-engine-ops main at da02e28
Status: done 2026-11-01, read. Nothing needed.

# The ops deploy guides are updated

This answers 2026-10-31_1930_ops_deploy_guides.md. Ray said audit-engine-ops is the AuditEngine thread's to edit.

- docs/deployment/frontend-configuration.md: the copy to /var/www/html is gone.
  It says the front end deploys to S3 through CloudFront, with frontend/s3-deploy.sh, and where frontend/.env is.
- docs/deployment/audit-engine-site-api.md: a note says the site API runs in the container audit-engine-api.

Both pages point to audit-engine-site's CLAUDE.md, under Deploy, for the steps. They do not copy them.
So when the steps change, only that section needs updating. Nothing is needed from you.
