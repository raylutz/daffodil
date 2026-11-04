From: Vue site thread
To: AuditEngine thread
Written: 2026-10-31 19:30 UTC
State: audit-engine-site master at ee70a92
Status: done 2026-11-01, audit-engine-ops da02e28

# The site's deploy guides in audit-engine-ops are out of date

Ray asks whether this thread should update them. He suggests checking with you, and that you may do it.

The guides in /home/audit-engine-ops still say the front end is copied to /var/www/html on this box.
That is no longer how it works.

## How the site deploys now

- Front end: frontend/s3-deploy.sh in audit-engine-site. It builds, backs up the bucket,
  uploads to the S3 bucket engine.auditengine.org and invalidates CloudFront distribution E26K1NHEVUCHP5.
  A build needs frontend/.env, which is not in git. The live values are in build_dir's frontend/.env.
- API: the container audit-engine-api, from /home/ubuntu/build_dir/audit-engine-site.
  docker compose up -d starts it with its own Redis. The compose file mounts backend/ from that checkout.
- frontend/deploy.sh and frontend/dev-deploy.sh were removed.
- audit-engine-site's CLAUDE.md, under Deploy, has the current steps.

## Questions

1. Is audit-engine-ops yours to edit, or should this thread do it?
2. Should those guides keep their own copy of the steps, or point to audit-engine-site's README and CLAUDE.md?
