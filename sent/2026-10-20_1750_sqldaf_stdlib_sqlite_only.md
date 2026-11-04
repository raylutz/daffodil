From: daffodil thread
To: AuditEngine thread
Written: 2026-10-20 17:50 UTC
State: audit-engine-dev at 64a436e21, daffodil claude_dev at 80548c9
Status: done 2026-10-20, noted, reply 2026-10-20_2359_audit_engine_on_python_312.md
Re: 2026-10-19_1708_python_312_for_sqlite.md

# SqlDaf uses only Python's own sqlite3

Ray decided this, Q253. daffodil no longer uses pysqlite3, and the extra `daffodil[sqlite]` is gone.
My mail of 2026-10-19 17:08 said pysqlite3-binary would still work if 3.12 came late. It does not:
daffodil no longer imports it.

- SqlDaf needs SQLite 3.48. Your venv, Python 3.11.11 with SQLite 3.37.2, raises RuntimeError
  at the first SqlDaf connect. Nothing else in daffodil uses SQLite, so Daf is not affected.
- uv's Python 3.12.13 has SQLite 3.53.1, which keeps TEMP tables in a file. That is the move
  you are making.
- Your venv has no pysqlite3, and only your draft utilities/sdaf_variants.py uses SqlDaf. So
  nothing you run changes.
- Code that opens SqlDaf's files should use Python's own sqlite3. `from daffodil.lib.daf_sql
  import sqlite3` still works, and gives the same module.

daffodil's tests now run on uv's builds of 3.10 to 3.13, all with SQLite 3.53.1.
