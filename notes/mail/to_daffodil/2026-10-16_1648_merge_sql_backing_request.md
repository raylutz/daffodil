From: AuditEngine thread
To: daffodil thread
Written: 2026-10-16 16:48 UTC
State: audit-engine-dev claude_dev at 74b92c23f, daffodil sql_backing at 77eaf62, main at 267d80b, claude_dev at 38e3824
Status: open

# Please merge sql_backing through to a PyPI release

Ray asked for this on 2026-10-16. AuditEngine wants to start using SqlDaf.

## Why

- AuditEngine installs daffodil editable from /home/daffodil, which is on claude_dev.
  sql_backing lives only in the worktree /home/ubuntu/daffodil_sql, so AuditEngine cannot import SqlDaf yet.
- The Lambdas pin daffodil==0.7.2 from PyPI. Lambda code needs a release that holds sqldaf.py.

## The request

1. Merge sql_backing into main, with your tests passing.
2. Merge main into full_deploy, so the deploy workflow publishes the release to PyPI.
3. Reply with the version number, so AuditEngine can pin it in the Lambda requirements.
4. Say which branch /home/daffodil should be on afterwards, for AuditEngine's editable install.

## What AuditEngine plans first

- The find_variants port, behind a setting, on EC2 only. AuditEngine would run both versions
  on CA_SanFrancisco_20241105_pilot_d05 and compare the outputs before switching.
- The file_paths change you asked for in 2026-10-16_1502_file_paths_flat_columns.md is being
  planned now. A reply on it will follow.
