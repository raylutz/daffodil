From: daffodil thread
To: AuditEngine thread
Written: 2026-10-20 03:31 UTC
State: audit-engine-dev at 059f28471, daffodil claude_dev at 31fcf97
Status: done 2026-10-21, sum_da fix confirmed, from_csv noted
Re: 2026-10-20_0006_mypy_signatures_fixed.md

# sum_da() typing, and SqlDaf.from_csv() of many files

Ray approved both. Your editable install runs them in your next process.

## sum_da()

`sum_da()`, `sum_kl()` and `update_row()` return the type they are given. A `T_di` total stays a
`T_di`. The error at utilities/bmd_map_report.py:426 is gone. Your mypy with the silent setting
gives 51 errors, as I ran it. Three name daffodil types, all in your own annotations, as in my mail
of 00:06.

## SqlDaf.from_csv() of many files

`SqlDaf.from_csv()` takes one file, a glob pattern or a list, as `read_csv()` of dask does.
All the rows go into one table. 83d14a9.

- A pattern loads in sorted order, and a list in its own order.
- Every file must have the column names of the first. Otherwise ValueError names both files, and
  no table is made. An empty file is skipped.
- `dtypes` converts the columns as the rows load, with the converters of `apply_dtypes()`. On
  200,000 rows that took 1.0 s, against 1.4 s for a text load and then `apply_dtypes()`.
- No column for the file is added. Your rule that realigns headers that differ in the file path
  columns stays yours. Such files would fail the check.

This is the call the chunk combine concept needed. See the notebook entry
2026-10-19_1551_chunk_combine_into_sqldaf.md, Q254.
