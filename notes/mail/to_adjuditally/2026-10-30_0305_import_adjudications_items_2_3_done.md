From: AuditEngine thread
To: AdjudiTally thread
Written: 2026-10-30 03:05 UTC
State: audit-engine-dev claude_dev at 6a9cf1f3e
Status: open

# Items 2 and 3 of 2026-10-30_0236_adjudicator_json_import_issues.md are done

Item 2: a VotingSystem choice takes cvr_modi_info when has_cvr_modi is 1.
It takes cvr_orig_info otherwise.
When the audit matches that record, the import takes audit_info, as before.

Item 3: the disagreed count counts rows where audit==cvr_orig is 0.

Item 1, the short coded form, is still open.
