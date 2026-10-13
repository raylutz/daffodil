# SQL backing for Daf: design draft

Started 2026-10-13 on the branch sql_backing. This is a draft for Ray. Nothing in it is
approved yet.

## Source

The idea was first scoped in AuditEngine's engineering notebook, in the entry
2026-09-14_sqlite-backed-daf-scoping-for-gen-bif-report.md. That entry is about one
AuditEngine report. This note restates the parts that matter for daffodil, in general terms.

## The problem

A Daf holds the whole table in memory as a list of lists. That is what makes it fast. It also
sets a ceiling. A large enough table, or two of them in one process, runs out of memory.

The notebook found two patterns that hit the ceiling.

- Grouping a large table makes a second full copy of its rows. See the section on groupby
  below.
- A second full table is loaded only to look up a few single values in it.

Neither one needs a full row returned by a Python function. One is GROUP BY with counts and
distinct values. The other is a lookup with WHERE. SQL does both without loading the table.

## What already works

AuditEngine's find_variants stage already does its large-table work in SQLite. Its code is in
utilities/sql_variants.py and utilities/sql_utils.py. The pattern has three steps.

1. Do the joins, filters and set operations in SQL. The table stays on disk.
2. Check that the narrowed set is not too large. A bug that makes it balloon should stop the
   run, not slow it down.
3. Pull the narrowed set into an ordinary Daf, and do the per-row Python work there.

The notebook records timings for 114,143 rows. A Python loop over SQL rows took 10.2 s. A
Python function registered in SQLite took 16.8 s, because such a function returns one value
per call. A join in SQL followed by `apply()` in a Daf took 6.7 s.

## The SQL shapes find_variants uses

locate_variants runs about 30 SQL statements. They fall into these shapes.

1. Anti-join: the rows whose key is not in another table.
2. Semi-join: the rows whose key is in another table.
3. Filter: the rows where any of some flag columns is nonzero.
4. Insert or ignore: append rows, and skip a key already present.
5. Delete the rows whose key is in another table.
6. Compare two tables by key, and split the rows into agreed and disagreed.
7. Explode: one row for each contest, taken from a JSON column.
8. Distinct, and a join that adds suffixes to clashing column names.
9. Narrow, then finish in a Daf.

Most of these are operations on sets of keys. A SQL-backed Daf could offer them as methods.

## Ray's goal

The calling code should look the same. It makes ordinary Daf calls. A setting makes the Daf
SQL-backed underneath. On 2026-10-13, Ray added that find_variants could be written with a
SQL-backed Daf.

## The notebook's advice

- Do not make `groupby()` itself secretly SQL-backed. It has many callers, and some do
  per-group work that is not an aggregate.
- Add a few aggregate methods instead, such as a grouped value count and grouped unique
  values. Each runs in SQL when the Daf is SQL-backed, and in memory otherwise.
- The one helper missing in AuditEngine is a grouped GROUP_CONCAT(DISTINCT col).

## Chunks and hunks

AuditEngine's chunktable.py splits a table into hunks, of up to about 100K rows, and chunks,
of 50 to 100 ballots. A manifest maps each ballot_id to its chunk. That fits handing work to
Lambda functions. It does not help with a set operation across a whole table. Ray, on
2026-10-13: with a SQL-backed Daf, chunktable is less important, but still useful for Lambda
delegation.

## What daffodil has now

- src/daffodil/lib/daf_sql.py holds the encoding of names for SQLite and a few helpers for
  benchmarks.
- Nothing in daf.py uses SQLite.

## Plan

These steps were proposed on 2026-10-13. Ray deferred them while the groupby question is open.

79. Inventory. Map each SQL step in locate_variants to a Daf method, existing or new.
80. Shape. Decide how a Daf gets SQL backing. One option is one Daf class with a SQL store
    underneath. Another is a separate class with the same method names. Much of daf.py reads
    self.lol directly, so this needs a review with options.
81. Prototype the core. Load a CSV into SQLite without building it in memory. Then len,
    columns, lookup by key, select where, anti-join and semi-join, and pulling a narrowed set
    into an ordinary Daf.
82. Aggregates. Grouped value counts, sums and unique values.
83. Benchmark against an in-memory Daf on a synthetic table of 1M rows, in short runs.
84. Trial. Rewrite one stretch of locate_variants with the new methods, as a sketch in these
    notes. AuditEngine is not changed.

## Decisions

88. Shape, decided by Ray on 2026-10-13. A SQL-backed table is a `SqlDaf`, a subclass of Daf,
    in its own module, src/daffodil/sqldaf.py, with the functions that support it.
    - SqlDaf defines only the methods supported under SQL. The first is `sql()`, a general
      method that runs a SQL statement given in the call. It is the fallback for anything
      not yet supported.
    - Selections, groupbys, column sums and the like get SQL versions, added as a need appears.
    - A short safe list names inherited Daf methods that read only the column names or the
      keyfield.
    - A loop at import replaces every other Daf method on SqlDaf with one that raises
      SqlNotSupportedError. No list of unsupported methods is kept, and the Daf class and its
      in-memory path are not changed.
    - Ray chose this over a check at the top of each Daf method.
    - A prototype ran on 2026-10-13: 214 Daf methods were blocked, `print()` needed SqlDaf's
      own `__str__`, and `isinstance(sdaf, Daf)` is True.
89. Engine, decided by Ray on 2026-10-13. SQLite now. Postgres and MySQL may follow, so the
    parts that differ go in a few small helpers in sqldaf.py: connect, placeholder, quoting a
    name, and the few functions that differ, such as JSON and GROUP_CONCAT. The rest is plain
    shared SQL. DuckDB is ruled out: Ray prefers the flexibility of SQLite and Postgres.
    SQLite on this box is 3.37 under Python 3.10 and 3.11, which has no RIGHT or FULL OUTER
    JOIN, and 3.53 under 3.12 and 3.13.
90. Where the data lives, decided by Ray on 2026-10-13. A small SqlStore object owns one
    database. SqlDafs are made from it, and their results go in the same file, so joins need
    no ATTACH. With a path, the file and its named tables are kept. With no path, the file is
    temporary and deleted at `close()`. Unnamed result tables are deleted at `close()`, and
    `save_as(name)` keeps one. `SqlStore(conn=...)` wraps an existing connection, so
    AuditEngine's databases work too. The engine helpers of decision 89 go in SqlStore.
91. Views, decided by Ray on 2026-10-13. Selections and joins return a new SqlDaf on a TEMP
    VIEW over its source, and nothing runs until data is pulled: `to_daf()`, `len()`,
    indexing, an aggregate or `sql()`. `materialize()` makes a TEMP TABLE for a result read
    many times. A condition may be SQL text, a dict, or a Python function registered on the
    store's connection. Writes go to base tables only: append, append skipping existing keys,
    and delete where. Views are read only and see those writes.
    Measured on 1,000,000 rows: a chain of three views ran as one scan, 0.082 s against 0.076 s
    for one query, and each read reruns the chain. A Python function in WHERE was 2 times
    slower than SQL with one column, and 6 times with a row dict of four columns. A registered
    function exists only in its connection, so a view that uses one can only be TEMP.
92. Naming, decided by Ray on 2026-10-13. `name` is the SQL name. A base table is named at load,
    or after its file. A `save_as()` result takes the name given. Generated objects are TEMP and
    named daf_tmp_ plus the parent and a counter, as in daf_tmp_full_bif_daf_v1. TEMP objects
    belong to one connection, so they cannot clash across threads or processes, and they vanish
    when it closes, even after a crash. The prefix keeps a TEMP name from hiding a main table,
    which SQLite would otherwise do. A store owns one connection and belongs to one thread, as
    sqlite3 requires by default. A clash of saved names raises ValueError. `name` is read only,
    and `to_daf()` passes it on. Snowpark was checked: its DataFrames have no name, and its
    temporary objects are named SNOWPARK_TEMP_<type>_<10 random characters>.
93. Row order, decided by Ray on 2026-10-13. Every view keeps its source's row order, by rowid,
    and a join follows the order of `Daf.join()`, so indexing works as for a Daf. Without an
    ORDER BY, adding an index on precinct changed which row came first. `sdaf[n]` and slices
    become LIMIT and OFFSET, and `sdaf[key]` a WHERE on the keyfield. Each index reruns the
    view, so per-row loops should use `to_daf()` on a narrowed view, or batches.
94. Pulling into memory, decided by Ray on 2026-10-13: Original. `to_daf()` pulls whatever the
    view holds, with no limit. A caller who wants a limit checks `len()` first. Ray first said
    A, a `max_rows` limit per call, then corrected it to Original. Measured for the record, on
    1,000,000 rows and a view of 300,495: COUNT took 0.106 s, and the pull then runs the view
    again; LIMIT 50,001 took 0.059 s and stopped at the limit.
95. Types and NULL, decided by Ray on 2026-10-13: A. Daffodil's NULL, '', is stored as SQL NULL,
    and SQL NULL comes back as NULL wherever data reaches Python: rows pulled, and the values
    given to a registered Python function. A test is `val is NULL` in Python and `col IS NULL`
    in SQL, never `== ''`, as Ray said. A dict condition such as {'n': ''} becomes `n IS NULL`.
    The NULL group of an aggregate comes back with the key NULL. Column types are declared
    from dtypes or a schema when given: INTEGER, REAL, TEXT, and JSON text for lists and
    dicts. Without them, TEXT.
    Measured on SQLite 3.37: with '' stored for a missing value, SUM of 1, missing, 2 gave 3.0
    as real, COUNT 3 and AVG 1.0; with SQL NULL, 3 as integer, COUNT 2 and AVG 1.5. Without the
    conversion, a registered function saw None, and `row['n'] is NULL` matched nothing. An ''
    read from sqlite3 is NULL.
97. Selections hold references, decided by Ray on 2026-10-13: A. This replaces the TEMP VIEWs of
    decision 91 for selections. A selection is a TEMP TABLE of references, in order, to the
    rows of its base table, as an in-memory selection holds the same row lists. A join holds
    pairs. Updating through a selection writes the base, as in memory. Appending to a
    selection resolves it into its own table first. The reference table is built when the
    selection is first used, so an unread chain of selections runs as one query.
    It must be VACUUM safe. VACUUM renumbered the rowids of a plain table, 5 to 3, and did not
    change a declared INTEGER PRIMARY KEY. So a reference is an INTEGER PRIMARY KEY, daf_rid in
    every table SqlStore creates, hidden from columns() and to_daf(), or else a unique keyfield.
    SqlStore adds a unique index on the keyfield if there is none, and a duplicate key raises.
    Measured on 1,000,000 rows: building a selection 0.14 s once, against 0.10 s for every read
    of a view; a count of it 0.0001 s; row 1000 0.00041 s, against 0.00107 s for a view.
    References by daf_rid and by a unique ballot_id cost the same, 0.12 against 0.14 s to build
    and 0.49 against 0.50 s to resolve 300,779 rows, and both survived delete plus VACUUM.
98. No usable reference, decided by Ray on 2026-10-13: A. A selection on a table with neither an
    INTEGER PRIMARY KEY nor a unique keyfield raises SqlNotSupportedError, saying to set a
    unique keyfield or copy the table into the store. It is never copied automatically.
99. Deleting, decided by Ray on 2026-10-13, then revised the same day. Records may be deleted
    from base tables, such as input tables, through SqlDaf and through sql(). There is no
    authorizer. A selection whose base rows are deleted does not show them on its next read,
    and the docstrings say so. Rows can also be left out the Daf way, with a new selection
    such as select_records_daf(keys, inverse=True), which leaves older selections whole.
    Ray first chose to refuse record deletes. The authorizer was tested for that: a DROP also
    reports a delete on the dropped table, so it needs a note of the DROP_TABLE action. Kept
    here in case it is wanted later.
    Deferred by Ray, to handle when a real case needs it: what a selection does when its base
    rows are deleted or its base table is dropped, tracking live selections, and dropping the
    TEMP reference table of a collected SqlDaf. Until then TEMP tables go when the store closes.
102. Selection names, decided by Ray on 2026-10-13.
    - `select_where()` takes a function only, in Daf and SqlDaf.
    - `select_by_dict()` on SqlDaf works as on Daf, with the SQL written inside: a dict is an
      AND of equalities, a list of dicts an OR, and `inverse` a NOT.
    - `select_where_sql()` takes SQL text. SqlDaf only.
    - `select_records_daf()` accepts a Daf or a SqlDaf and uses its keys. In memory it uses a
      view of the other's key index, so no list is made. In SQL it is a semi-join, or an
      anti-join with `inverse=True`. Passing `other.keys()` still works. This changes Daf: today
      a Daf passed there raises TypeError, since iterating a Daf gives row dicts. To be built
      with the core.
    - `SqlDaf.keys()` returns a lazy reference to the key column. `list(sdaf.keys())` pulls it.
103. Adding rows without existing keys, decided by Ray on 2026-10-13: Original. There is no skip
    mode for concat(). Rows whose key is already present are left out with
    `concat(new_sdaf.select_records_daf(target_sdaf, inverse=True))`, one INSERT with an
    anti-join. A duplicate key among the rows being added raises, from the unique index. Ray:
    duplicates are not normal, and must never be ignored. AuditEngine finds them, marks both
    copies, and checks whether a duplicate is an exact repeat or a different record with the
    same key.
104. Deleting records, decided by Ray on 2026-10-13: B. `delete_records(keys)` on SqlDaf and on Daf.
    It takes keys as `select_records_daf()` does, a Daf, a SqlDaf or an iterable of keys, and
    returns the number deleted. On SqlDaf it works on base tables only, and raises on a
    selection. On Daf it rebuilds the row list in place and invalidates the key index. An
    in-memory selection made earlier keeps the deleted rows, since it holds the row lists. A
    SqlDaf selection loses them on its next read, as decision 99 says. The docstrings say both.

## groupby and row sharing

See groupby_row_sharing.md in this folder.
