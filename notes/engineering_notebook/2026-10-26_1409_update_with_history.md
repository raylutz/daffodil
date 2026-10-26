# update_with_history: a rerun merged into a table, with history

Started 2026-10-26. AuditEngine offered its function in mail 2026-10-26_0300_update_with_history.md.
Ray asked the same day to add it as a core method. He also asked for a `from_str` on SqlDaf's
`apply_dtypes()`, so a typed table can be turned back into text and compared with a CSV read.

## Where it came from

- AuditEngine: utilities/daf_update_with_history.py, with 15 tests in tests/test_daf_update_with_history.py.
  Its design notes are engineering_notebook/2026-10-25_selective_rerun_with_history.md in audit-engine-dev.
  Those notes describe a first draft with three results, including a residue. The function in use
  returns two, updated and history.
- One caller: merge_listed_rows() in utilities/votes_extractor.py, with by='ballot_id' and the
  chunk's task list as key_order. It skips an empty rerun before the call.
- On SF sheet1, 2,226 chunks were merged on Lambdas with it, on daffodil 0.8.0.

## Existing methods checked

- `join()` matches rows of two tables by key, one to one. It does not replace groups or keep history.
- `update_record_krow()` and `update_by_keylist()` change cells in place. They keep no history.
- `select_records_daf(keys, inverse=True)` can split a table by key, but does not compare records.
- `Daf.__eq__` compares rows, columns and keyfield. The method uses it to compare two records.
- Nothing did the merge, so it is new.

## What was built

- src/daffodil/daf_join.py, the first module of the split plan's groups. The function is
  `_update_with_history(self, ...)`, assigned in the Daf class as `update_with_history`. It is the
  helper-module pattern of daf_schema.py, way A of the plan. Way A is proven. It needs no change to
  the SqlDaf blocker, the profiler or the docs coverage test. A move to way B later is mechanical.
- The plan put join with report and types in one module. Join is the plan's trial group, so it
  starts as a module of its own.
- Behavior is AuditEngine's, with two changes:
  - A rerun with no rows returns a copy of the table and an empty history. AuditEngine's version
    raised ValueError for Daf() with no columns.
  - A key_order is turned into a set for the stray check, rather than tested with `in` on a list.
- The docs entry is on the Joins page, under Update.
- On a SqlDaf the method is blocked, with NotImplementedError. AuditEngine's Lambdas do not use SqlDaf.
- Tests: 13 in tests/test_daf_transform.py. AuditEngine's own 15 tests also pass against the method,
  run through a shim outside both repos.

## from_str on SqlDaf.apply_dtypes()

- Default True, as on a Daf: a str column is not touched.
- With False, each cell of a str column goes through the converter `Daf.apply_dtypes()` uses, so
  3 becomes '3' and 1.5 becomes '1.5'. SQL NULL stays SQL NULL. The columns have no declared type,
  so SQLite keeps the text as text.
- Checked against a CSV load of the same rows: every kind of cell matched but a bool.

## Open: a bool converted to str (Q299)

- A Daf's str converter turns True into the int 1, so a cell with dtype str is not a str.
- SQLite stores True as 1, so a SqlDaf gives '1'.
- `to_csv_file()` writes True as 'True', and a dict cell holding True keeps True in its text.
- So a typed rerun with a bool never equals its CSV text. AuditEngine stores flags as 0 and 1, so
  it is not affected. Ray to decide.

Decided by Ray on 2026-10-26: option A. A bool is the int 0 or 1 in daffodil, and converted to str it
is '0' or '1'. The rule is in CLAUDE.md under "Booleans are 0 and 1". `_convert_str()` now gives
str(int(val)) for a bool. A Daf and a SqlDaf then agree, since SQLite stores a bool as 1.

Found while checking: `to_csv_file()` writes a stored True as True until the column is converted.
After `apply_dtypes()` with dtype bool it writes 1. A first check declared dtype bool without
calling `apply_dtypes()`, and wrongly reported that the writer ignores the dtype. Ray caught it.
What holds: a dict cell keeps True and None in its PYON text, even after `apply_dtypes()` with
dtype dict, since a dict passes through unchanged. Q300 is that narrower point.

Q300 decided by Ray on 2026-10-26: Original. Values are normalized once, when the row is built, and
never again. No pass in `apply_dtypes()` or in the writers. Checked: a normalized dict cell reads
back as PYON, through SQLite's json_extract, and through a CSV round trip, unchanged. The rule is in
CLAUDE.md, the Booleans page and the cheatsheet.
