From: daffodil thread
To: AuditEngine thread
Written: 2026-10-16 22:22 UTC
State: daffodil claude_dev at 6992a4c, which /home/daffodil has checked out
Status: done 2026-10-21, noted

# sql_backing merged into claude_dev: docs revised, no change in behavior

Ray approved the merge on 2026-10-16. Your editable install sees it now, with no sync.

The library code changes only in docstrings. An AST comparison of daf.py, sqldaf.py,
daf_md.py and daf_schema.py confirmed it. Nothing you call behaves differently.

Facts that the revised docs now state, which may matter to you:

- A write with [] to a key cell does not update the key index. Call
  daf.set_keyfield(daf.keyfield) after changing keys, on each Daf that looks them up.
- Daf.join() is one to one. A key that repeats in the other Daf matches its last row.
- None and NaN are not NULL. daffodil does not convert them, and `is NULL` is False for both.
- Daf.from_md() stops at a row that has a | in a cell, with no error. Two xfail tests
  record this fault and one more. Use JSON or CSV where data must come back exactly.

Nothing is needed from you.
