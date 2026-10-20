# For the mail to AuditEngine at the next release

Items to include when the next version is released. Clear this list once the mail is sent.
The list was last cleared on 2026-10-16, when 0.8.1 was mailed.

- SqlDaf needs SQLite 3.48 or later, and raises RuntimeError at its first connect on an older one.
  It uses only Python's own sqlite3. AuditEngine's EC2 venv, Python 3.11 with SQLite 3.37.2, needs uv's
  Python 3.12, which has SQLite 3.53.1. pysqlite3 is not used. The Lambdas need nothing, since they do not use SqlDaf.
- Code that opens the same database files as SqlDaf, such as utilities/sql_utils.py, should use Python's own
  sqlite3, as `from daffodil.lib.daf_sql import sqlite3` gives. Two copies of SQLite in one process with the same file open can lose file locks.
