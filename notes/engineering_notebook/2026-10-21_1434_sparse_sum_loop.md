# How a sparse sum loops

A sparse sum adds up the dict held in one column of each row, such as concise_votes.
`reduce(Daf.sum_da, by='sparse_row', indirect_col=..., cols=...)` makes it, on a Daf or a SqlDaf.
`cols` is the columns to include in the sum. `indirect_col` is the column that holds the dict.

## Ray's rules, 2026-10-21

- A container tested with `in` inside a loop is never a list. Use a KeysView, a dict or a set. Q265.
- The decision on how to loop is made once, before the loop, not inside it. The loop itself is plain.
- When every key is wanted, there is no `cols`, no test and no loop over the columns.
- When `cols` is given and short, loop over `cols`, not over the keys of each row.
- The result should be in the order of `cols`.

## What the code did before

- `reduce()` called `sum_da()` once per row, with `is_sparse=True`.
- `sum_da()` looped over the row's keys. With `cols`, it tested each key with `col not in cols`.
  With no `cols`, it still tested `cols` for truth at each key.
- The result held the keys found, in the order the rows first had them.

## Who calls it

- daffodil: `Daf.reduce()`, `SqlDaf.reduce()`, `Daf.groupby_reduce()` through `reduce()`,
  `SqlDaf.groupby_reduce()` and `SqlDaf.groupby_aggregate()` through the SQLite aggregate CellReduction.
- AuditEngine, read-only grep on 2026-10-21: generic_cvr.py:2188, extract_utils.py:411,
  dominion_cvr.py:1830 and DB.py:3335 call `daf_sum(by='sparse_row')` with no `cols`.
  bmd_map_report.py:426 calls `Daf.sum_da()` directly, with no `cols`.
  Only the port, sdaf_variants.py, passed `cols`: 385 names, nearly every key of the job.

## Measured on pilot d05

audit_nonvariants, 32,253 rows. A concise_votes dict has 29.8 keys on average, 142 at most.
359 keys appear in some row. Decoding the cells with `get_indirect_da()` takes 0.33 s,
which is most of the time of a sum.

The loop alone, in plain Python, on the decoded dicts, with cols the most frequent keys:

| cols | loop over cols | test each key | add every key |
| ---: | ---: | ---: | ---: |
| 1 | 0.021 s | 0.054 s | 0.132 s |
| 5 | 0.048 s | 0.071 s | 0.132 s |
| 10 | 0.060 s | 0.095 s | 0.132 s |
| 20 | 0.112 s | 0.152 s | 0.132 s |
| 30 | 0.163 s | 0.185 s | 0.132 s |
| 100 | 0.284 s | 0.176 s | 0.132 s |
| 359 | 0.742 s | 0.162 s | 0.132 s |

- Testing each key is never the fastest. Adding a key costs about what testing it costs.
- So for long `cols`, add every key and pick the columns once at the end.
- The crossover follows the dict size. On synthetic dicts of 5, 30 and 200 keys, looping over
  `cols` of the dict's size took 0.56, 0.60 and 0.65 of the time of adding every key.
  At 2 times the dict's size it took 0.99 to 1.24 of it.
- The rule: loop over `cols` when it has no more names than the mean keys per row.
  On d05 that is at most 25% slower than the best choice, near the crossover.
  A wrong fixed choice is up to 5 times slower.

## The change, Q266, settled by Ray on 2026-10-21

Merged into claude_dev and sql_backing at 5de5ffc, from branch sparse_sum_plan.

- `Daf._sparse_sum_plan()` makes the choice from the first 100 rows, `Daf._SPARSE_SAMPLE_ROWS`.
- `Daf._sum_sparse_keys()` adds every key, with no test.
- `Daf._sum_sparse_cols()` looks up each column of `cols`. A column the row lacks is skipped, never added as 0.
- `Daf._sum_sparse()` reads the sample, chooses, loops and picks. The sample rows are decoded once.
- `Daf._is_plain_sum()` is true for `sum_da()` or `sum_kl()` with no other options.
  With `initial_da`, `silent_error`, `astype` or a function of the caller's own, the old row loop is used.
- `Daf.reduce()` and `SqlDaf.reduce()` use `_sum_sparse()` for a plain sparse sum.
- `SqlDaf.groupby_reduce()` reads 100 cells of the indirect column first, and gives the aggregate
  the chosen function. CellReduction picks `select_cols` at the end of each group.
- `SqlDaf.groupby_aggregate()` uses `_sum_sparse_keys()` for a plain sum, since it has no `cols`.
- `sum_da()` itself is not changed. A caller who calls it per row still gets the key test.
- The port passes no `cols`. Its totals function picks sumcols_ls from the totals, as before.

What changes for a caller:

- With `cols`, the result is in the order of `cols`. Before, it was in the order the rows first
  had the keys. The keys and the totals are the same.
- `SqlDaf.groupby_reduce()` with `reduce_cols` writes each group's JSON in the order of `reduce_cols`.
- Nothing changes with no `cols`, which is how every AuditEngine caller sums.

Timings on d05, best of 3, Original at 81fce36 against the branch:

| | Original | New |
| :-- | ---: | ---: |
| Daf.reduce, no cols | 0.427 s | 0.441 s |
| Daf.reduce, 5 cols | 0.423 s | 0.313 s |
| Daf.reduce, 385 cols as a list | 1.364 s | 0.425 s |
| Daf.reduce, 385 cols as a KeysView | 0.538 s | 0.425 s |
| SqlDaf.reduce, no cols | 0.546 s | 0.555 s |
| SqlDaf.reduce, 5 cols | 0.551 s | 0.450 s |
| SqlDaf.reduce, 385 cols as a list | 1.441 s | 0.545 s |
| SqlDaf.groupby_aggregate by precinct | 0.528 s | 0.516 s |
| SqlDaf.groupby_reduce, 5 cols | 0.494 s | 0.423 s |
| Daf.groupby_reduce, 5 cols | 0.441 s | 0.314 s |

- With no `cols`, nothing is faster. The truth test of None was cheap, and decoding is most of the time.
- Every total matched the Original. With `cols`, the key order follows `cols`.
- The port's d05 check matched on every table, CSV file and stat, with the port passing no `cols`.

## How the design went wrong first

The first answers treated the port's call as the general case, and only changed the type of `cols`.
That is recorded, with what it changes in how the work is done, in
2026-10-21_1443_design_from_the_general_case.md.

## Pointers

- Code: daf.py, the section "Sparse sums" before `sum_kl()`. sqldaf.py, `reduce()`,
  `groupby_reduce()`, `groupby_aggregate()` and CellReduction.
- Tests: tests/test_daf_reduction.py, the sparse sum tests. tests/test_sqldaf.py,
  `test_groupby_reduce_matches_mdaf`.
- The Q265 entry: 2026-10-19_1443_auditengine_whole_table_stages_on_sqldaf.md.
