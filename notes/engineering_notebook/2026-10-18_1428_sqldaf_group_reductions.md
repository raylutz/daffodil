# SqlDaf group reductions: groupby_aggregate, groupby_reduce and sums in SQL

Started 2026-10-17, written 2026-10-18 14:28 UTC. Branch sql_backing.
This collects the research and trials behind the SqlDaf grouping methods, so the next look at
this area starts from here. Read it before adding or changing a grouping or reducing method.

## Goal

The find_variants port, `notes/sql_backing/sdaf_variants.py`, groups large tables and reduces
each group. Step 15 merges the concise votes of each ballot's contests. The precinct sums total
plain columns and concise votes for each precinct. Step 16 finds the distinct ballots.
The aim is to do these inside SQLite, without reading groups into Python as Dafs.

## What already exists. Check these first

| Method | Where it runs | Notes |
| :-- | :-- | :-- |
| `SqlDaf.groupsum_daf()` | one SQL GROUP BY | Plain totals. Adds only integers and floats. Returns a Daf in memory, rows in first-seen order. Added with Q143. |
| `SqlDaf.sum()` | SQL TOTAL, text cells in Python | Whole-table totals, as `Daf.sum()`. Converts totals to dtypes. |
| `SqlDaf.groupby()` | one query for the keys | Each group is a lazy selection. Reducing every group is a query per group. |
| `SqlDaf.groupby_reduce()` | registered aggregate | See below. Mirrors `Daf.groupby_reduce()`. |
| `SqlDaf.groupby_aggregate()` | registered aggregate | See below. Any Python function, one or more group columns. |
| `SqlDaf.apply_dtypes()` | one pass in SQL | Turns text numbers into numbers. Keeps empty cells empty. |
| `Daf.to_donpa(default=)`, `Daf.to_pandas_df(default=)` | Python | `default` replaces empty, None and NaN cells. For export only. |
| `Daf.replace_in_columns()` | Python | Fills listed values. Blocked on SqlDaf. |
| `Daf.reduce()` | Python | The row reduction. Adds down each column for a plain `sum_da`, about 5 times faster than a call per row. |

`func='sum'` was added to `groupby_aggregate()` in 0b8bd89 without this check, and removed in
291db46, because `groupsum_daf()` already did it (Q245). That is why this table is here.

## What SQL can and cannot do with concise votes

- Concise votes are PYON text, such as `{'c0v': 1}`, with single quotes.
- SQL `SUM()` of the column gives 0.0. `json_extract()` and `json_each()` raise "malformed JSON".
- Swapping the quotes makes it valid JSON, but it breaks on a quote inside a value. Not used.
- pyjson5 reads both quote styles. daffodil reads PYON with it, by `unflatten_val()`.
- So concise dicts are reduced in Python, inside a registered aggregate.

## What SQL SUM does with empty cells

Checked on SQLite 3.37.2 and 3.53.1, values 1, '', SQL NULL, 2.

```
SUM(v)                                    3.0   '' counts as 0, total becomes a float
SUM(NULLIF(v, ''))                        3
SUM(CASE WHEN typeof(v) IN
    ('integer','real') THEN v ELSE 0 END) 3     what groupsum_daf() uses
```

- SQL NULL is always skipped. The text '' never raises.
- With the typeof filter, an int column with empty cells totals as an int, as in a Daf.
  A total is a float only when the group holds a float. No dtype coercion is needed after it.
- Text numbers such as '3' are skipped by the filter, as `Daf.sum_da` skips them.
  `Daf.sum()` converts them instead. After a CSV load, run `apply_dtypes()` first.
- Replacing empty cells with 0 first, as pandas does with a default, changes no total. Not needed.
- SQLite 3.43 and later add floats more exactly than Python, so a float total may differ in the
  last digits. 3.37 agrees with Python. A total of integers over 2**63 raises overflow.

## SQLite version limits

- AuditEngine runs SQLite 3.37.2, from the system Python 3.11 on Ubuntu 22.04.
  daffodil's uv Python has 3.53.1. The Lambda image version is not known yet (Q235).
- 3.37 has no ORDER BY inside an aggregate call. A sorted subquery gives the row order instead.
- A function call takes at most 127 arguments. `json_object()` failed at 64 columns on 3.37.
  So a row is packed as a JSON array of arrays of at most 100 columns, `ARRAY_CHUNK`.
- Test under AuditEngine's Python, read only:
  `PYTHONDONTWRITEBYTECODE=1 /home/audit-engine-dev/.venv/bin/python -I -c "import sys; sys.path.insert(0, 'src'); import pytest; pytest.main([...])"`

## The registered aggregate

- `conn.create_aggregate()` with a class that has `step()` and `finalize()`.
  SQLite groups the rows and calls `step()` for each row of a group, `finalize()` once.
- The result can be a whole JSON record. `json_extract(res, '$[i]')` splits it into columns.
- An error inside the class reaches Python as SQLite's short message. A wrapper saves the real
  exception, and `_execute()` raises it as itself.
- Results are written in daffodil form by `normalize_val()`: True and False as 1 and 0, None as ''.

## Trials and timings, SF sheet1

Step 15 is 80,443 contests into 3,795 ballots. Precinct sums are over 86 precincts.
Times in seconds. The box was shared, so expect about 10% noise.

| Job | First block version | Call per row | `Daf.reduce()` per group | Prototype |
| :-- | --: | --: | --: | --: |
| Step 15, sparse merge | 0.51 | 0.37 | 0.37 | 0.22 |
| Precinct concise sums | 1.52 | 1.11 | 1.09 | |
| Step 16, 73,888 groups | 1.23 | 0.60 | 0.92 | |
| Row sums, 3 plain columns | 0.19 | 0.35 | 0.29 | |
| Row sums, all 13 columns | 0.59 | 1.98, then 1.19 without decoding | 0.64 | |

- **Block version.** The first `groupby_reduce()` read sorted batches and reduced whole groups
  in Python blocks by `Daf.reduce_dodaf_to_daf()`. Replaced in 0b8bd89 (Q243).
- **Call per row.** Decoding every dict cell cost the most, even when `sum_da` then skipped it.
  `Daf.groupby_reduce()` gives those cells as text, so by row they are no longer decoded.
- **Per group.** Keeping each group's rows and reducing them by `Daf.reduce()` at the end got
  back the column-sum fast path and Daf's exact calling form, `silent_error` included.
  It is what `groupby_reduce(by='row')` uses.
- **Prototype.** A bare aggregate with no wrapper and no `normalize_val()`. The gap to 0.36 is the
  error wrapper and the decoding of each cell.
- `normalize_val()` was the hot spot at first, 236,000 recursive calls. A fast path for int and
  str fixed it.
- **Plain totals in SQL.** `groupsum_daf()` took 0.09 s for the three plain precinct columns,
  against 0.40 s for `sum_da` through the aggregate. Same totals.
- What remains in the precinct sums is Python work, about 2.27 million key additions.

## APSW and other SQLite drivers

Looked at on 2026-10-18, because Ray asked whether APSW helps.
Tested in a scratch venv of Python 3.11.11, whose own sqlite3 has SQLite 3.37.2, as AuditEngine's does.

| Problem on 3.37 | stdlib sqlite3, 3.37.2 | pysqlite3-binary 0.5.4, 3.51.1 | apsw 3.54.0.0, 3.54.0 |
| :-- | :-- | :-- | :-- |
| ORDER BY inside an aggregate | syntax error | works | works |
| json_each, json_extract on PYON | malformed JSON | works | works |
| A call with 200 arguments | too many arguments | works | works |
| json_object with 64 pairs | too many arguments | works | works |
| An error inside an aggregate | SQLite's short message | SQLite's short message | the KeyError itself |
| Python aggregate, 1,000,000 rows | 0.78 s | | 0.75 s |

- **APSW** bundles a current SQLite inside its wheel. Its version tracks SQLite, 3.54.0.0 on 2026-10-13.
  It passes Python errors through, with Python frames in the traceback.
  It is not DB-API. There are no implicit transactions, and aggregates are registered by a factory
  that returns the state with step and final functions.
  sqldaf.py has about 140 lines that use the sqlite3 API, and AuditEngine's sql_utils.py about 60.
  Its Linux wheels are manylinux_2_28, which needs glibc 2.28.
- **pysqlite3-binary** is the stdlib sqlite3 module built with a newer SQLite. Its API is the same, so
  `import pysqlite3.dbapi2 as sqlite3` would be the whole change. Its wheels are manylinux2014,
  glibc 2.17. It does not pass errors through.
- **Lambda.** The python3.11 Lambda runtime is Amazon Linux 2, with glibc 2.26, by the Powertools build
  guide. A manylinux_2_28 wheel does not install there. python3.12 and later are Amazon Linux 2023,
  glibc 2.34. What AuditEngine's compute-service-base:3.11 is built on is not known yet (Q235).
- **Speed.** Neither driver makes the Python work faster. The aggregate cost the same in both.
- **JSON5.** SQLite 3.42 and later read JSON5, which allows single quotes. So json_each reads concise
  votes, but not PYON with True, False or None. On SF the precinct concise sums by
  `json_each` GROUP BY in SQL took 2.51 s, against 1.18 s by `groupby_aggregate(sum_da)`, with the same totals.
  So JSON5 is a capability, not a speed gain.
- Daffodil is pure Python. Either driver would be a compiled dependency, best kept optional.

## Rejected or set aside

- **Writers that turn dict cells into JSON.** Too slow, and seldom needed (Q240).
- **A dict subclass whose str() gives JSON.** `dict.__str__` cannot be set. A subclass is lost
  through `dict()`, `{**d}`, `.copy()`, `|` and JSON loads.
- **SQL SUM on concise votes after a quote swap.** Fragile, see above.
- **A null-to-zero default before sums.** Changes no total and costs a pass.

## Decisions

All by Ray. Q217, Q220, Q224 for the first `groupby_reduce()`. Q234, Q237 and Q239 for
`groupby_aggregate()`, with its two forms. Q241 asked for plain sums in SQL. Q243 moved
`groupby_reduce()` onto the aggregate. Q245 removed `func='sum'` in favor of `groupsum_daf()`.

## Still open

- Q242: step 16 needs only the distinct ballot ids. A `SELECT DISTINCT` would do. SqlDaf has no
  method for it yet.
- Q235: the SQLite version in AuditEngine's Lambda image.
- Q236: a CI job on an old SQLite, such as 3.37.
- `groupsum_daf()` returns a Daf in memory. A large result might want a SqlDaf.

## Pointers

- Code: `src/daffodil/sqldaf.py`, the section "Python aggregates for groupby_aggregate()",
  `groupby_reduce()`, `groupby_aggregate()`, `_aggregate_groups()` and `groupsum_daf()`.
- Tests: `tests/test_sqldaf.py`, the groupby sections.
- Port and its check: `notes/sql_backing/sdaf_variants.py`, `notes/sql_backing/sdaf_variants_check.py`.
  The check runs against `/home/audit-engine-dev/resources/CA_SanFrancisco_20241105_sheet1/cmpcvr`.
- Commits: 3afea54, 7e89292, 32bbff0, 45933a3, 7296301, 0b8bd89, 291db46, 3b58b41.
- `notes/sql_backing/groupby_row_sharing.md`: whether groups share rows.
- The timing scripts were in the session scratchpad and are not kept.
