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

Q79. Inventory. Map each SQL step in locate_variants to a Daf method, existing or new.
Q80. Shape. Decide how a Daf gets SQL backing. One option is one Daf class with a SQL store
    underneath. Another is a separate class with the same method names. Much of daf.py reads
    self.lol directly, so this needs a review with options.
Q81. Prototype the core. Load a CSV into SQLite without building it in memory. Then len,
    columns, lookup by key, select where, anti-join and semi-join, and pulling a narrowed set
    into an ordinary Daf.
Q82. Aggregates. Grouped value counts, sums and unique values.
Q83. Benchmark against an in-memory Daf on a synthetic table of 1M rows, in short runs.
Q84. Trial. Rewrite one stretch of locate_variants with the new methods, as a sketch in these
    notes. AuditEngine is not changed.

## Where the work happens

Ray decided on 2026-10-14, question 116, B. The SQL work is in its own git worktree,
/home/ubuntu/daffodil_sql, on the branch sql_backing, and is pushed only to sql_backing.
/home/daffodil stays on claude_dev, because AuditEngine's `-e ../daffodil` install reads that
folder. Until 2026-10-14, sql_backing was checked out in /home/daffodil, and each commit was also
pushed to claude_dev, so claude_dev has sqldaf.py up to d97e6cb. It is not imported by anything
there. Changes to mdaf, the in-memory Daf, are built and tested here first, and go to claude_dev only
once they are settled and tested. Ray, 2026-10-14: "edits ... for mdaf should be first resolved and tested
before they become live for AE."

## Decisions

Numbered as the questions were in the conversation, as Qnnn, since a markdown viewer renumbers a list. A missing number was a question about
what to do next, not a design decision.

Q88. Shape, decided by Ray on 2026-10-13. A SQL-backed table is a `SqlDaf`, a subclass of Daf,
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
Q89. Engine, decided by Ray on 2026-10-13. SQLite now. Postgres and MySQL may follow, so the
    parts that differ go in a few small helpers in sqldaf.py: connect, placeholder, quoting a
    name, and the few functions that differ, such as JSON and GROUP_CONCAT. The rest is plain
    shared SQL. DuckDB is ruled out: Ray prefers the flexibility of SQLite and Postgres.
    SQLite on this box is 3.37 under Python 3.10 and 3.11, which has no RIGHT or FULL OUTER
    JOIN, and 3.53 under 3.12 and 3.13.
Q90. Where the data lives, decided by Ray on 2026-10-13. A small SqlStore object owns one
    database. SqlDafs are made from it, and their results go in the same file, so joins need
    no ATTACH. With a path, the file and its named tables are kept. With no path, the file is
    temporary and deleted at `close()`. Unnamed result tables are deleted at `close()`, and
    `save_as(name)` keeps one. `SqlStore(conn=...)` wraps an existing connection, so
    AuditEngine's databases work too. The engine helpers of decision 89 go in SqlStore.
    Replaced on 2026-10-14 by decisions 122 to 124: there is no SqlStore.
Q91. Views, decided by Ray on 2026-10-13. Selections and joins return a new SqlDaf on a TEMP
    VIEW over its source, and nothing runs until data is pulled: `to_daf()`, `len()`,
    indexing, an aggregate or `sql()`. `materialize()` makes a TEMP TABLE for a result read
    many times. A condition may be SQL text, a dict, or a Python function registered on the
    store's connection. Writes go to base tables only: append, append skipping existing keys,
    and delete where. Views are read only and see those writes.
    Measured on 1,000,000 rows: a chain of three views ran as one scan, 0.082 s against 0.076 s
    for one query, and each read reruns the chain. A Python function in WHERE was 2 times
    slower than SQL with one column, and 6 times with a row dict of four columns. A registered
    function exists only in its connection, so a view that uses one can only be TEMP.
Q92. Naming, decided by Ray on 2026-10-13. `name` is the SQL name. A base table is named at load,
    or after its file. A `save_as()` result takes the name given. Generated objects are TEMP and
    named daf_tmp_ plus the parent and a counter, as in daf_tmp_full_bif_daf_v1. TEMP objects
    belong to one connection, so they cannot clash across threads or processes, and they vanish
    when it closes, even after a crash. The prefix keeps a TEMP name from hiding a main table,
    which SQLite would otherwise do. A store owns one connection and belongs to one thread, as
    sqlite3 requires by default. A clash of saved names raises ValueError. `name` is read only,
    and `to_daf()` passes it on. Snowpark was checked: its DataFrames have no name, and its
    temporary objects are named SNOWPARK_TEMP_<type>_<10 random characters>.
Q93. Row order, decided by Ray on 2026-10-13. Every view keeps its source's row order, by rowid,
    and a join follows the order of `Daf.join()`, so indexing works as for a Daf. Without an
    ORDER BY, adding an index on precinct changed which row came first. `sdaf[n]` and slices
    become LIMIT and OFFSET, and `sdaf[key]` a WHERE on the keyfield. Each index reruns the
    view, so per-row loops should use `to_daf()` on a narrowed view, or batches.
Q94. Pulling into memory, decided by Ray on 2026-10-13: Original. `to_daf()` pulls whatever the
    view holds, with no limit. A caller who wants a limit checks `len()` first. Ray first said
    A, a `max_rows` limit per call, then corrected it to Original. Measured for the record, on
    1,000,000 rows and a view of 300,495: COUNT took 0.106 s, and the pull then runs the view
    again; LIMIT 50,001 took 0.059 s and stopped at the limit.
Q95. Types and NULL, decided by Ray on 2026-10-13: A. Daffodil's NULL, '', is stored as SQL NULL,
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
Q97. Selections hold references, decided by Ray on 2026-10-13: A. This replaces the TEMP VIEWs of
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
Q98. No usable reference, decided by Ray on 2026-10-13: A. A selection on a table with neither an
    INTEGER PRIMARY KEY nor a unique keyfield raises SqlNotSupportedError, saying to set a
    unique keyfield or copy the table into the store. It is never copied automatically.
Q99. Deleting, decided by Ray on 2026-10-13, then revised the same day. Records may be deleted
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
Q102. Selection names, decided by Ray on 2026-10-13.
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
Q103. Adding rows without existing keys, decided by Ray on 2026-10-13: Original. There is no skip
    mode for concat(). Rows whose key is already present are left out with
    `concat(new_sdaf.select_records_daf(target_sdaf, inverse=True))`, one INSERT with an
    anti-join. A duplicate key among the rows being added raises, from the unique index. Ray:
    duplicates are not normal, and must never be ignored. AuditEngine finds them, marks both
    copies, and checks whether a duplicate is an exact repeat or a different record with the
    same key.
Q104. Deleting records, decided by Ray on 2026-10-13: B. `delete_records(keys)` on SqlDaf and on Daf.
    It takes keys as `select_records_daf()` does, a Daf, a SqlDaf or an iterable of keys, and
    returns the number deleted. On SqlDaf it works on base tables only, and raises on a
    selection. On Daf it rebuilds the row list in place and invalidates the key index. An
    in-memory selection made earlier keeps the deleted rows, since it holds the row lists. A
    SqlDaf selection loses them on its next read, as decision 99 says. The docstrings say both.
Q106. Types, keyfield and transactions, decided by Ray on 2026-10-14.
    - Readers only read. `SqlStore.from_csv()` stores the text of the file, in columns with no
      SQL type, and takes no dtypes. `apply_dtypes()` converts in place with Daf's own
      converter, `daf_utils.get_converter()`, registered as a SQL function, so the results
      match `Daf.apply_dtypes()`: '2.5' as int is 2, 'abc' stays, NULL stays NULL, and NULL in
      a bool column is 0. SQLite's own CAST would make 'abc' 0, so it is not used. This
      replaces the declared column types of step 1 and of decision 95.
    - The keyfield is recorded at load. Its unique index is made when a key is first used,
      by `ensure_key_index()`, as a Daf builds its kd lazily. A repeated key raises then. A
      Daf keeps the last copy quietly; a SqlDaf does not, since duplicates must never be missed.
    - `store.transaction()` runs a block as one transaction, nested as a savepoint. It halves
      the cost of runs of small statements.
    Ray: SQL is only for large tables, and the speed tradeoff is not a show stopper.
    Measured on 1,000,000 rows: load as text then convert two columns 5.3 s, convert during
    the load 4.0 s, SQLite INTEGER at load 3.5 s but '2.5' became 2.5. Index at first use 0.5 s.
    A lookup by key 17.9 us, 9.0 us inside a transaction. Peak memory 40 MB throughout.
Q107. Converting during the load, decided by Ray on 2026-10-14: A, reader only, for the best
    compatibility. Ray: CSV is read in a wrapper, not in a testable pure function. The
    conversion that `Daf.from_csv_buff()` does at its end when given dtypes is no better than
    calling `apply_dtypes()` later, and should be deprecated, after checking AuditEngine. On
    2026-10-14 AuditEngine's 6 direct calls to Daf's CSV readers passed no dtypes, and
    DB.load_data() builds `Daf(lol, cols, dtypes=...)` without converting. A separate item.
Q108. Deprecating the dtypes conversion of the CSV readers, decided by Ray on 2026-10-14: in the
    docs. The docstrings of `Daf.from_csv_buff()` and `Daf.from_csv()` say it is deprecated, and
    to call `apply_dtypes()` instead. It still works, and no warning is raised.
    `from_excel_buff()` also takes dtypes; it is not changed.
Q110. NULL in a bool column, decided by Ray on 2026-10-14: Original. `apply_dtypes()` turns it into
    0, as `Daf.apply_dtypes()` does, so the two match. Keeping NULL was weighed: in SQL, NULL is
    not false, so `NOT is_bmd`, `is_bmd = 0` and `is_bmd != 1` all skip it, and AVG leaves it out.
    If "not determined" should be kept apart from 0, it would be a change to both Daf and
    SqlDaf together, as its own item, with a check of AuditEngine's bool columns.
Q111. Names and types, decided by Ray on 2026-10-14. A SqlDaf variable ends in `_sdaf`, and its
    type is `T_sdaf`, with `T_lsdaf` and `T_dosdaf` for lists and dicts. They are real
    TypeAliases of the class, in sqldaf.py, so mypy checks them. Python 3.10 added TypeAlias.
Q112. Indexing, decided by Ray on 2026-10-14. `sdaf[n]`, `sdaf[n:m]` and `sdaf[key]` return a SqlDaf
    selection, as any selection does, not an ordinary Daf, so a large slice is not pulled into
    memory. Reading it resolves the rows: `to_daf()`, `len()`, and one-row readers such as
    `to_dict()`, `to_list()` and `to_value()`. A selection is built lazily, so
    `sdaf[0].to_dict()` is one LIMIT query, and no reference table is written.
Q113. When a selection's rows are fixed, decided by Ray on 2026-10-14: at its first read, as an mdaf
    (in-memory Daf) selection is fixed when made. `select_record()` and `iloc()` read one row
    as a dict in one query, make no selection, and fix nothing. A one-row selection read
    with `to_dict()` cost 134 us against 24 us direct, before positions became lookups.
Q114. len() of a selection, decided by Ray on 2026-10-14: A. It counts the references, as the
    highest order number of the reference table, one lookup. Do not delete rows from a base
    table that has selections, or that is read by position; the docstrings say so. Ray: iloc
    should use daf_rid. So positions and slices of a base table with daf_rid, or of a fixed
    selection, are lookups on daf_ord, and a negative position takes the highest daf_rid as
    the count. Other tables, and selections not fixed yet, step with OFFSET.
    On 1,000,000 rows: iloc(500000) 17 ms before, 0.16 ms after. iloc(-1) 0.28 ms. len() of a
    fixed selection of 499,892 rows 116 ms before, 0.01 ms after. sdaf[900000:900100].to_daf()
    0.61 ms.
Q118. The name persist(), decided by Ray on 2026-10-14. The method that runs a selection's query once
    and keeps which rows it found, or resolves a join into a TEMP table of rows, is `persist()`,
    with the flag `is_persisted`, as Spark and Dask use the word: the result is kept and the
    object stays lazy. It was `fix()` until then. In databases the usual word is materialize,
    but `materialize()` here copies the rows, and a selection keeps only references. The
    decisions above that say "fixed" mean persisted.
    Replaced on 2026-10-14 by decision 126: there is no public persist().
Q122. Making a SqlDaf, decided by Ray on 2026-10-14: A. The bare constructor makes a new table, or
    reopens one by name, as Daf(lol=, cols=) does. There is no class named store: Ray finds the
    word too vague, and wants no new class unless one is needed. Where the connection lives is
    question 124.
Q123. Closing and deleting, decided by Ray on 2026-10-14. `close()` closes the connection. The
    tables stay in the file and can be reopened, and selections that share the connection can no
    longer be read. `delete()` deletes the table from the file, with its metadata. The word drop is
    not used: Ray finds delete more explicit. Without `close()`, Python closes the connection when
    the last instance that holds it goes. Nothing is lost, since every write is committed.
    Changed on 2026-10-14 by Q136 and Q138: `SqlDaf.close()` closes a file, and without it the
    file stays open until the program ends.
Q124. Where the connection lives, decided by Ray on 2026-10-14. There is no SqlStore and no live
    state at class level. Changed by Q136: the class keeps the one connection open on each file. The class holds only settings that every instance follows unless the
    constructor overrides them, as Daf does with `copy_level_default`: the engine, SQLite for now,
    and the default `db_path`, which AuditEngine sets once per job.
    - Each instance opens its own connection from the path, and keeps it in its attributes. The
      caller never passes a connection, since it belongs to how the data is stored.
    - A selection or a join holds its parent's connection.
    - Each instance also keeps what it knows of its table: the name, keyfield and dtypes, and
      whether it is a primary table or depends on another, with its base and its references.
      Unlike a Python object, a SQL table does not describe itself, so the instance may need more
      attributes to track its state. That is expected.
    - A selection can be made independent with a copy in place, question 129.
    - A join is allowed within one file, when neither side depends on TEMP objects of the
      other's connection. Real tables in one file can be joined from any connection on it.
    - What SqlStore held needs no outside state. A connection with an empty file name is a
      private temporary database, which SQLite deletes at close. One savepoint name can be
      nested. TEMP names are checked against the database. Tested on 2026-10-14.
Q125. Metadata, decided by Ray on 2026-10-14: A. A table's metadata is kept in the database file,
    in a table of its own, with one row per daffodil table, written in the same transaction as the
    change it records. While open, the instance holds the metadata in its attributes. Reopening a
    table fills in the instance from that row, as it was before, so the keyfield and dtypes are
    not given again. A sidecar file was turned down because it can drift from the database.
Q126. Persist, decided by Ray on 2026-10-14. There is no public `persist()`. Its job is done by a
    copy in place, decision 129, which makes a selection a table of its own rows. A new
    `save()` writes the table as a permanent table in the database, to be reopened later.
    Before that, a copy is a TEMP table. The automatic step at a selection's first read, which
    keeps its references, stays as an internal detail, as decision 113 requires.
Q127. Streaming the rows of a join, decided by Ray on 2026-10-14: not now. A SqlDaf is backing for
    analysis of static data, not for transactions. A join is usually persisted at once, so reading
    a few rows of a join may build the whole join. This is a possible future enhancement.
Q128. A repeated key in a join, decided by Ray on 2026-10-14: Original. A keyfield must be unique,
    and if it is not, the result is undefined, as in a Daf. Nothing checks it, since a check is too
    much overhead. Finding repeated keys, and telling true duplicates from rows that share a key
    but differ, is the caller's work. In AuditEngine it is a step of its own, which moves such rows
    to a residue table. Seen on 2026-10-14: with a key twice in the other table, a SqlDaf join gives
    both rows, and a Daf join gives the last.
Q129. Copy in place, decided by Ray on 2026-10-14, for both a Daf and a SqlDaf: `copy(inplace=True)`
    copies into the object itself, and returns it, instead of making a new one. After it, a
    selection no longer depends on its base. With inplace, the default level is deep, Claude's
    call, as Ray left it open: it is the safe choice, and it is what a SqlDaf always gives.
    Editable may be asked for, to copy only the rows. A lower level raises ValueError, since it
    would not free the rows. For a Daf, this is a change to mdaf, so it is built and tested on
    sql_backing first.
Q130. The name in save(), decided by Ray on 2026-10-14: A. `save(name='')` uses the name the
    instance already has: the one given to the constructor, or set by `copy(name=)`, or the one
    passed to it. A table with only a generated TEMP name, such as daf_tmp_votes_v1, raises
    ValueError and asks for a name. A name already in use raises ValueError. On 2026-10-14 an mdaf
    selection had the name '', and `copy(name='agreed')` set it to 'agreed'.
Q131. The old copy methods, decided by Ray on 2026-10-14: `materialize()` is replaced by `copy()`,
    which gives a new SqlDaf in a TEMP table, and `save_as(name)` by `save(name)`.
Q132. Parent and child, decided by Ray on 2026-10-14. A primary table is a parent, and a SqlDaf
    that depends on another table, such as a selection, is a child. The terms name the attribute
    that tells them apart.
Q133. keys(), decided by Ray on 2026-10-14. A SqlDaf's `keys()` returns the same type as
    `Daf.keys()`: a list, read into memory. The class SqlKeys goes. To pass many keys without
    reading them, give the SqlDaf itself to `select_records_daf()`, which accepts a Daf.
Q134. SqlNotSupportedError, decided by Ray on 2026-10-14: A. The class goes. A Daf method with no
    SQL version raises a plain NotImplementedError, with the same message, which names the method.
    An earlier record here said Original, from a misread answer.
Q135. A SqlDaf with no database, decided by Ray on 2026-10-14: C. The path is a class attribute,
    `SqlDaf.db_path`, which AuditEngine sets once per job. With no path, the constructor raises
    ValueError. A path given to the constructor overrides it for that instance, as in Q124.
    Examples and tests set it to a file in a temporary folder. Without this, each connection with
    no path would be a private database of its own, and two such tables could not be joined.
Q136. One connection per file, decided by Ray on 2026-10-14: A. Every SqlDaf on one file shares
    one connection, which it holds in its attributes. The class keeps the connection open on each
    file, by thread and real path, since sqlite3 ties a connection to one thread.
    - Why: a child's references are in a TEMP table, which only its own connection sees. With a
      connection per instance, as first built, a selection of one table could not be used with a
      table opened or made separately. The find_variants port does that at every step: step 4 adds
      a selection of the votes to a new variants table, and it raised NotImplementedError. Real
      tables in one file can still be joined from any connection, as tested before.
    - The cost: SqlDafs on a file share its transaction state. A rollback undoes the work of all
      of them. That fits a stage run as one unit. Separate connections are worse here: a write
      through one, while a transaction is open on another, waits for the lock and fails with
      "database is locked", tested on 2026-10-14.
    - Ownership: methods run their own `transaction()`, which becomes a savepoint inside an open
      one, so only the outermost block commits or rolls back. The docstrings say a transaction
      belongs to the file, and not to run BEGIN, COMMIT or ROLLBACK through `sql()`. No check is
      added, under the rule of performance over guard rails.
    - After the change, the port ran on five jobs and gave the results of the first trial:
      Dane, Bartow, Somerset and San Francisco the same as AuditEngine, Rockville the one ballot
      of AuditEngine's stale-table bug.
Q137. Not decided yet: whether to fix on claude_dev the broken docs link in `Daf.delete_records()`,
    from beb5218, which fails the strict docs build.
Q138. How a file is closed, decided by Ray on 2026-10-14: A. `SqlDaf.close(db_path='')` is a
    class method. It closes one file, the class setting's if no path is given, for every SqlDaf
    on it. Called on an instance, it still closes that file, which the docstring warns.

## Possible future enhancements

- Stream the rows of a join, or a selection of one, without building the whole join first.
  Seen on 2026-10-14: reading 2 rows of a 3-row join wrote all 3 rows to its TEMP table.

## groupby and row sharing

See groupby_row_sharing.md in this folder.
