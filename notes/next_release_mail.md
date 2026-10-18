# For the mail to AuditEngine at the next release

Items to include when the next version is released. Clear this list once the mail is sent.
The list was last cleared on 2026-10-16, when 0.8.1 was mailed.

- SqlDaf needs SQLite 3.48 or later, and raises RuntimeError at its first connect on an older one.
  AuditEngine's EC2 venv has SQLite 3.37.2. Install pysqlite3-binary there, as `pip install pysqlite3-binary`,
  before using SqlDaf. Daffodil then uses its SQLite 3.51.1. The Lambdas need nothing, since they do not use SqlDaf.
- Code that opens the same database files as SqlDaf, such as utilities/sql_utils.py, should use
  `from daffodil.lib.daf_sql import sqlite3`. Two copies of SQLite in one process with the same file open can lose file locks.
