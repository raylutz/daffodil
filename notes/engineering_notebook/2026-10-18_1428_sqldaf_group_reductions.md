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
  daffodil's uv Python has 3.53.1.
- AuditEngine's Lambdas do not use SqlDaf or SQLite, by Ray on 2026-10-18. SqlDaf runs only in local
  stages that work on whole tables, such as find_variants. So only the EC2 SQLite matters, and Q235 is closed.
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
- **Which PYON JSON5 reads.** Run on SQLite 3.51.1 with `json_each`, 2026-10-18.
  - Read: single or double quotes, escaped quotes, '' values, ints, floats, 1e20, inf,
    nested dicts and lists, non-ASCII text and \x escapes.
  - Malformed JSON: True, False, None, tuples, int dict keys and \U escapes.
  - nan reads as SQL NULL, with no error.
  - Python repr writes \U only for an astral character that is not printable, so it is rare.
  - So PYON reads in SQL once True, False and None are written as 1, 0 and ''.
    Tuples and int keys must also be absent. Concise votes have neither.
  - This narrows the Q219 exception. SQLite reads such PYON as it is, so pure JSON is not needed there.
- Daffodil is pure Python. Either driver would be a compiled dependency, best kept optional.

## Step 15 in SQL alone

Tried on 2026-10-18, SF sheet1, SQLite 3.51.1. The box was busy, so these times run higher than the table above.

- The contest votes that step 15 merges are already pure JSON. `split_contests()` writes them so.
  So JSON5 changes nothing here. `json_each` read them on SQLite 3.37 too.
- AuditEngine's sql_variants.py merges them in SQL, by `json_group_object()` over `json_each`.
- The same in one query, with the order inside the call, which needs 3.44:
  ```sql
  SELECT ballot_id,
         json_group_object(j.key, j.value ORDER BY cidx) AS concise_votes,
         SUM(j.key GLOB 'c[0-9]*b') AS num_contests
  FROM nonvariant_contests CROSS JOIN json_each(concise_votes) AS j
  GROUP BY ballot_id
  ```
- Same text for all 3,795 ballots as `groupby_aggregate(update_da)`, key order included.
  The GLOB count matches `num_contests_of()` on every ballot.
- Times, 3 runs each: the query with the count 0.24 to 0.33 s. The aggregate with the num_contests
  pass 0.55 to 0.65 s. The aggregate alone 0.37 to 0.42 s.
- No SqlDaf method merges dicts by group in SQL. The only route is `sql()`, which returns a Daf
  in memory.
- Decided, Q248: keep the aggregate. The saving is about 0.3 s, and the faster routes leave
  SqlDaf's methods or need a new one.

## pysqlite3-binary keeps TEMP data in memory

Found 2026-10-19, while timing the chunk combine. See 2026-10-19_1551_chunk_combine_into_sqldaf.md.

- pysqlite3-binary 0.5.4.post2 is built with SQLITE_TEMP_STORE=3. Every TEMP table and every
  sort is then in memory, and `PRAGMA temp_store` cannot change it.
- The stdlib sqlite3 of uv's Python 3.12 and 3.13, SQLite 3.53.1, and APSW 3.54.0 are built with
  TEMP_STORE=1. That is a file by default, and the pragma can change it.
- `groupby_aggregate('precinct', ...)` on 1,603,908 CVR rows: a peak of 665 MB on pysqlite3, and 40 MB
  on the stdlib driver, 24.0 s against 27.8 s. A reduce function that keeps nothing gave the same
  665 MB, so the memory is SQLite's, not Python's.
- So every SqlDaf TEMP table, such as a selection, `group_where()` groups or a sort, is in RAM
  under pysqlite3-binary. That works against what SqlDaf is for. Q253.

## Moving to a newer SQLite

Checked on 2026-10-18 against the SQLite change log and by running both versions.

- What we use, by the version that brings it: JSON5 input 3.42, ORDER BY inside an aggregate 3.44,
  JSONB 3.45, 1000 function arguments 3.48. So 3.48 is the lowest version that covers all of it.
- The file format did not change. Old and new versions read each other's files, unless JSONB is
  stored or a new schema feature is used.
- Float SUM totals differ: 3.53.1 gave 2.0 where 3.37.2 and Python gave 0.0, on
  ten 0.1s, 1e16, 1.0 and -1e16. Integer totals are the same.
- 3.53 writes a float as text with 17 digits by CAST, where 3.37 wrote 15.
  `CAST(1.0/3 AS TEXT)` is '0.33333333333333332' on 3.53.1 and '0.333333333333333' on 3.37.2.
  Python's str() gives '0.3333333333333333'. This touches `sort_by_colnames(as_str=True)` on a float column.
- `json_array()` still writes 15 digits on 3.53.1.

### Bug: floats lose digits in packed rows

`packed_row_sql()` packs a row by `json_array()`, which writes a float with 15 digits on both versions.
So `groupby_aggregate()` with several columns, and `groupby_reduce(by='row')`, reduce rounded floats.
0.1 + 0.2 arrives as 0.3. The SF check did not show it, since those columns hold integers.
`printf('%!.17g')` inside `json()` is exact on 3.53, but on 3.37 infinity gives 9e999, which its JSON rejects.
Passing the cells as direct arguments, up to 127 of them, avoids JSON for the common case.

## Done: SQLite 3.48 required, cells as arguments

Done on 2026-10-18, Q246 and Q247.

- SqlDaf needs SQLite 3.48 or later. `_connect()` checks `sqlite_version_info` and raises RuntimeError
  with how to get a newer one. Daffodil still imports on an old SQLite, which keeps Lambda working.
- daf_sql.py picks the driver: `pysqlite3.dbapi2` when it is installed, else the standard `sqlite3`.
  SqlDaf and the tests import it from there, so the process loads one SQLite.
  The extra `daffodil[sqlite]` installs pysqlite3-binary, Linux x86-64 only. The dev group includes it.
- Locally uv's Python 3.10 and 3.11 are the system builds, with SQLite 3.37.2.
  With the dev group they run on pysqlite3, SQLite 3.51.1. All four Pythons pass.
- The aggregates take each cell as its own argument, up to 1000. The JSON packing and ARRAY_CHUNK are gone.
  This fixed the float bug above. A test sums 0.1 + 0.2 and 1/3 exactly.
- The sorted subquery is replaced by ORDER BY inside the aggregate call. SQLite guarantees that order.
  It did not formally guarantee that a GROUP BY reads a sorted subquery in order.
- SF results are unchanged. Times now: step 15 0.38 s, precinct concise sums 1.20 s,
  row sums of 3 plain columns 0.16 s, of all 13 columns 0.45 s, step 16 0.89 s.
- The SQLite 3.37 limits above are history now. They stay here to explain the code that came before.

## Table-valued functions: rows from a registered function

Checked on 2026-10-18, since Ray heard APSW can return rows from a function.

- SQLite itself has had table-valued functions since 3.9, through virtual tables. `json_each()` is one.
- The standard `sqlite3` module has no virtual table API, in any version. So it is the driver, not the SQLite version.
- APSW has it: `apsw.ext.make_virtual_module(conn, name, func)`. The function yields rows, and
  `func.columns` names them. A literal call works: `SELECT * FROM concise_items(?)`.
- A trap: in `FROM t, concise_items(t.concise_votes)` the planner first called the function with no
  argument, and the result was empty, with no error. `FROM t CROSS JOIN concise_items(t.concise_votes)`
  makes SQLite read t first, and gave the right totals, with True as 1.
- Use for us: a Python function could unpack a concise dict into rows, so SQL could GROUP BY its keys.
  But json_each already does that for concise votes on 3.42 and later, and it was slower than the aggregate.

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
