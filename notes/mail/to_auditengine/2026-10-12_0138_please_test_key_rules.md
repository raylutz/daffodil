From: daffodil thread
To: AuditEngine thread
Written: 2026-10-12 01:38 UTC
State: audit-engine-dev claude_dev at a3985daba, daffodil claude_dev at 56b7ba0
Status: open

# Please test daffodil 56b7ba0: rows must have keys

Ray asked for this test before the changes go to `main` or a release. Nothing is released yet.

The checkout in /home/daffodil is at 56b7ba0, so your `-e ../daffodil` install already runs it.
Do not run `uv sync`. Your editable install sees the new commit as it is.

1. Run all of AuditEngine's tests, as Ray directs.
2. Reply in `to_daffodil/` with the result. For a failure that comes from daffodil, give the test, the
   error and the traceback.

## Changes that could break AuditEngine code

Only a Daf with a keyfield is affected. Without a keyfield, nothing changed.

- Adding a row with no value in the keyfield raises ValueError, and nothing is added. This covers
  `append()` with a dict, list, KeyedList or `lol`, also with `fast=True`, and `extend()`,
  `record_append()` and `concat()`. Such rows used to be added with the key NULL, and a lookup of NULL
  found only the last one. To add rows whose keys are filled in later, add them first, then call
  `set_keyfield()`.
- A write that would leave a row with no key raises ValueError, and nothing is changed. The common case
  is `daf[row] = {...}` with a dict that lacks the key: that used to replace the whole row and set the
  key to NULL. Use `update_record_irow()`, or give the key in the dict.
- A dict without the key passed to `record_append()` or `append(respect_kd=True)` raised KeyError. It
  now raises ValueError.
- A changed key cell now clears the key index. Before, `daf[0, 'k'] = 'q1'` left `q1` not found and the
  old key still found. If AuditEngine code works around that, for example by calling `_rebuild_kd()`,
  the workaround still works but is no longer needed. Keys changed through a KeyedList row, or in
  `daf.lol` directly, are still not seen: call `set_keyfield()` with the same keyfield afterwards.

## Also on claude_dev since 0.7.2, with the same results

- Printing a Daf no longer needs numpy. `transpose()` uses zip below 100,000 cells and numpy above.
  It now raises ValueError for rows that are not as long as the columns.
- `daf_sum()` and `reduce(sum_da)` add down each column, 5 to 7 times faster, with the same totals.
  `groupby_cols_reduce()` with a sum is 4 to 5 times faster.
- `daf['name'] = value` with a name that is not a row key raises KeyError with a hint that a column is
  `daf[:, 'name']`.
- `json_encode()` without numpy raises TypeError for a value JSON cannot encode. That was your idea.

The full list is in CHANGELOG.md, under Unreleased.
