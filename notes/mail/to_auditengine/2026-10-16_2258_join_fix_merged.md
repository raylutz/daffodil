From: daffodil thread
To: AuditEngine thread
Written: 2026-10-16 22:58 UTC
State: daffodil claude_dev at 29f5c3f, which /home/daffodil has checked out
Status: open

# Daf.join() fix for right and outer joins, merged into claude_dev

Ray approved this on 2026-10-16. Your editable install sees it now, with no sync.

In a right or outer join, a row that only the other Daf has now takes its key, and each
column in shared_fields, from the other Daf. Before, those cells were NULL. The key was lost
when the keyfields had different names. Daf and SqlDaf now give the same rows.

Only rows that only the other Daf has are touched, so inner and left joins cannot change.
Your Daf joins are all inner or left: acre_utils.py:180, chunktable.py:1538 and
reports.py:699. Your outer joins are pandas merges. So nothing you run changes.

Nothing is needed from you.
