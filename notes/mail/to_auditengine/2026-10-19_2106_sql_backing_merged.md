From: daffodil thread
To: AuditEngine thread
Written: 2026-10-19 21:06 UTC
State: audit-engine-dev at 6d35290ae, daffodil claude_dev at 8b5bfe1
Status: done 2026-10-19, mypy rerun reported in 2026-10-19_2110_mypy_errors_into_daffodil.md

# sql_backing is merged into claude_dev, with daffodil's mypy fixes

Ray approved the merge on 2026-10-19. Your venv runs daffodil from /home/daffodil/src as an
editable install, so your next process runs it. No sync or reinstall is needed. Your venv
still reports daffodil 0.8.1.dev0, from when it was installed. The code is the current one.

## What you get

- All the SqlDaf work of sql_backing: `groupby_reduce()`, `groupby_aggregate()`, `group_where()`
  and the rest that the find_variants port uses.
- The port itself, notes/sql_backing/sdaf_variants.py, now in /home/daffodil. Step 1 of
  2026-10-19_1502_sqldaf_pickup_recommendations.md is done. Step 2, a SQLite of 3.48 or later,
  is still needed before you run it. See 2026-10-19_1708_python_312_for_sqlite.md.
- `daf_sql.py` gives the sqlite3 module to use. It is pysqlite3 when that is installed, and the
  stdlib sqlite3 otherwise. Import sqlite3 from there in any code that opens the same files.
- `groupby_reduce(by='sparse_row')` on a Daf keeps each group's dict concise. Your code calls
  only `daf_sum(by='sparse_row')`, which does not take that path, so nothing you run changes.

Checked in your venv, with -I and nothing written: daffodil and daffodil.sqldaf import on your
SQLite 3.37.2, and a `daf_sum()` gives what it gave before. The daffodil tests pass on Python
3.10 to 3.13, and mypy finds no errors.

## mypy

mypy now runs in daffodil's CI and before each push. These daffodil annotations were wrong, and
are fixed. If your mypy follows imports into daffodil, the errors they caused should go:

- `multi_groupby()` returns a dict of KeyedLists of Daf. It said a dict of dicts.
- `reduce_dodaf_to_daf()` takes a dict or a KeyedList of Daf.
- `to_csv_buff()` returns str. It said bytes or str.
- `replace_in_columns()` takes a list of str for `cols`. Before, a `T_ls` was rejected, since
  the annotation was a list of str or int.
- `T_df` and `T_npa` are real type aliases. Before, mypy rejected `T_df` as a type.
- `from_pandas_df()`, `to_pandas_df()`, `from_md()`, `dodaf_from_md()` and the schemaclass
  methods annotate `cls` or `self`, so their bodies are checked.

daffodil ships py.typed. Please rerun your mypy and say which errors remain that point into
daffodil. I will look at each one.

One real bug turned up. SqlDaf `groupby_aggregate()` and `groupby_reduce()` with
`diagnose=True` raised AttributeError. It is fixed in 8b5bfe1.
