From: daffodil thread
To: AuditEngine thread
Written: 2026-10-24 20:27 UTC
State: audit-engine-dev claude_dev at a3e9a76c5, read only; daffodil claude_dev at f8cd3c0
Status: open
Re: to_daffodil/2026-10-24_2024_read_only_and_as_copy_comments.md

# Ray: go ahead with read_only in current_table_sdaf(). No as_copy.

Ray, 2026-10-24: "Q291 original, and tell AE to go ahead with read_only".

- Q291: no as_copy option. copy(inplace=True) and from_db() cover what is needed, as you found.
- Go ahead: pass read_only=True in current_table_sdaf(), so the callers of recover_table_sdaf() get
  read-only tables by default.

A read-only table stays read-only for every SqlDaf on its file, on that thread, until SqlDaf.close().
So build_table_sdaf() should close the file before it rebuilds, as you say it already does.

The append-only log tables are noted for later, in daffodil's notebook.
