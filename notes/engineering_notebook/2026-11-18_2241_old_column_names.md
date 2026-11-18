# Old column names: reading old tables under new names

Started 2026-11-18. On hold, Ray, 2026-11-18. Nothing is built.

## The need

AuditEngine has many tables stored with one set of column names. They are relied upon and hashed,
so they cannot be changed. Code moves to new names. New tables use the new names. An old table must
still be read with no error, with its old name changed to the new one in memory only.

## Settled

- Q346: the old names are declared in the schema, as a `@schemaclass` attribute such as
  `__aliases__ = {'file_path': ['file_paths']}`, or an Aliases column in a schema Daf.
- Q345: old tables are read. Their files never change. Writes to an old SqlDaf table are not required.

## The design so far

- Daf: when a schema is applied to data read from an old file, an old name is renamed in memory.
  Only `hd` and the dtypes change. No row is copied. If both names are present, it raises ValueError.
- Code uses only the new name.

## Open: Q347, how a SqlDaf reads an old table

- a. A TEMP copy with the new names. Every method works. About 0.6 s per million rows, once. Recommended.
- b. A TEMP view with the new names. Tested 2026-11-18: `select_where_sql()` and `select_by_dict()` work
  through `SELECT daf_rid, id, file_paths AS file_path FROM bif`. `groupby()` fails with
  "views may not be indexed", and the keyfield is lost, since the view has no daf_tables row.
- c. Translate each name when SQL is built. 143 calls of `quote_name()` in the sqldaf modules.

## Found on the way: Q344, rename onto a name in use

`rename_cols({'file_paths': 'file_path'})` on a Daf that has both names leaves `hd` as
`{'id': 0, 'file_path': 2}` over rows of three cells. The header then names the wrong cells, and
`daf[:, 'file_path']` raises IndexError. A SqlDaf refuses the same rename, as SQLite rejects a
duplicate name. A strict xfail test records the correct behavior, a ValueError with nothing changed:
`test_rename_cols_onto_a_name_in_use_raises_and_changes_nothing` in tests/test_daf.py.
The fix waits for Ray, with the rest.

## Callers found

AuditEngine renames `file_paths` to `file_path` by hand, at models/job_db.py:281 and utilities/utils.py:2748.
That case also splits the data, so it is more than a rename.
