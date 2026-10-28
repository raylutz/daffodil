# Daffodil Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),  
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Adoption of this format started in v0.5.10. Prior notes included in fixed section for
all prior releases. Plans for future moved to ROADMAP.md.

## [Unreleased]

### Changed
- A join, `concat()` and keys with a SqlDaf in another database file bring it in by themselves, where they raised NotImplementedError. The other table is copied into a TEMP table on this table's connection, with its key index, so neither file changes. A join keeps its copy until it is first read, then deletes it. A join never read gives up its copy when it is gone, and a chain of joins keeps each copy until the last join that uses it is read. `concat()` deletes its copy at once, and keys copy only the key column. A permanent table is copied in SQL, with its file attached read-only for the copy, as `from_db()` does. A selection or TEMP table of another connection, or a copy inside a transaction, goes through Python. On 1,000,000 rows of 5 columns, the copy and its index took 0.7 s, and the join then took as long as in one file. A selection of another thread's connection still raises NotImplementedError, with a message that says so. The Guide page on SqlDaf shows it, in Tables in other files. AuditEngine asked for it, as their Q438, and Ray approved it on 2026-10-28, Q312. Tests are in `tests/test_sqldaf.py`.
- `indirect_col` alone means sparse rows in `reduce()`, `daf_sum()` and the grouping reducers, on a Daf and a SqlDaf, as it does in `col()` and `select_where()`. Before, `reduce(func, indirect_col='votes')` ignored it and gave zeros, with no error. `by='sparse_row'` still works and says the same. Every grouping reducer now keeps a sparse result concise, as `groupby_reduce()` did: the group columns, and the indirect column with each group's summed dict. Before, `groupby_cols_reduce()` dropped the dict column, and `multi_groupby_reduce()` and `multi_groupsum()` gave zeros, each with no error. `groupsum_daf()` and `multi_groupsum()` take `indirect_col`. Ray decided it on 2026-10-28, Q311. Tests are in `tests/test_daf_reduction.py` and `tests/test_sqldaf.py`.
- **Breaking.** `sum()` on a Daf and a SqlDaf takes no `numeric_only`. Name the columns to total instead, as `sum(['qty', 'n'])`. It was a pandas habit. With no `cols` it gave the text `'0.0'` for each column it skipped, and a test locked that in. Without it, text that is not a number raises ValueError, as it did without `numeric_only`. AuditEngine calls it once, with the columns named, at generic_cvr.py:2228, and drops the argument. Ray decided it on 2026-10-28, Q310. Tests are in `tests/test_daf.py`, `tests/test_daf_misc.py` and `tests/test_sqldaf.py`.
- The docs examples show the real output of each call, with no `_ =`, and give an example for each parameter of a method, or a few together. The Quick start and Getting started pages are rewritten, and the Quick start runs as a doctest. A moved method's signature in the API pages shows its method name, not the function's. `to_md()`, `col()`, `append()`, the brackets and many more are reworked. Where a docstring said something the code does not do, it now says what the code does, and the bugs found are listed in `notes/engineering_notebook/`. Tests were added for the cases shown, and four xfail tests record two of the bugs. Ray asked for this on 2026-10-28, Q306.

## [0.9.1] - (2026-10-26)

This release splits daf.py and sqldaf.py into modules by topic, with no change to what any method does. Every import works as before. A method's `__name__` is now its function's name, as `_select_where`. `read_only=True` holds only for the SqlDaf that asked for it, where it held for every SqlDaf on the file, and `sql()` raises ReadOnlyTableError for a refused change. The SqlDaf API pages list 11 members they lacked.

### Changed
- `read_only=True` holds only for the SqlDaf that asked for it, and for its selections. Before, it held for every SqlDaf on the file until `SqlDaf.close()`. Now another SqlDaf opened on the same table without `read_only` can change it, and the read-only one reads the change. SQLite still enforces it. While a read-only SqlDaf is open on the file, the connection's authorizer is set for each SqlDaf as it runs a statement, and setting it makes SQLite check its prepared statements again. A first try with one fixed authorizer let a read-only write through, since SQLite checks a statement only when it prepares it, and Python reuses prepared statements. A switch between a reader and a writer costs about 5 to 15 us. With no read-only SqlDaf, calls time as before. `sql()` raises ReadOnlyTableError for a refused change, as every other method does, where it raised the bare `DatabaseError: not authorized`. ReadOnlyTableError is a subclass of DatabaseError, so a handler for that still catches it. AuditEngine asked for this, for Ray, on 2026-10-26. Ray chose it, Q305. Tests are in `tests/test_sqldaf.py`. The research is in `notes/engineering_notebook/2026-10-26_1857_read_only_per_sqldaf.md`.
- sqldaf.py is split the same way as daf.py, with no change to what any method does. It keeps the SqlDaf class, the connection and files, keys, size, iteration and the shared helpers, in 2,844 lines where it had 6,397. The other methods are functions in sqldaf_read.py, sqldaf_select.py, sqldaf_modify.py and sqldaf_apply.py, one for each SqlDaf docs page. Common calls timed the same before and after, within noise. Ray approved it on 2026-10-26, Q304.
- The SqlDaf API pages list 11 members they lacked: `from_db()`, `groupby_reduce()`, `groupby_aggregate()`, `md_daf_table_snippet()`, `str()`, `repr()`, `format()`, and the settings `db_path`, `engine`, `md_max_rows` and `md_max_cols`, which have docstrings now. A test in `tests/test_docsite_coverage.py` checks that each member SqlDaf defines is on one page, as one did for Daf.
- daf.py is split by topic, with no change to what any method does. daf.py keeps the Daf class and its core: construction, copying, the key index, shape, iteration and the dunders, in 3,400 lines where it had 15,600. The other methods are functions in daf_dtypes.py, daf_apply.py, daf_select.py, daf_modify.py, daf_convert.py, daf_structure.py, daf_report.py and daf_join.py, one for each API docs page, assigned in the class by name. Imports, pickling and the docs are unchanged. One difference shows: a moved method's `__name__` is its function's name, as `_select_where`. Calls take the same time: construction, `append()`, `select_record()`, `col()`, `groupby()` and `sum()` timed the same before and after, within noise. Ray chose way A of the plan with refinements on 2026-10-26, Q302 and Q303. The plan is in `notes/engineering_notebook/2026-10-25_1627_addon_hook_and_mixin_split_plan.md`.

## [0.9.0] - (2026-10-26)

This release has three breaking changes, marked **Breaking** under Changed. SqlDaf needs SQLite 3.48 or later. `groupby_reduce(by='sparse_row')` keeps each group's result as one concise dict. A bool converted to str is `'1'` or `'0'`, where it was the int 1 or 0. It adds `update_with_history()`, add-on namespaces for Daf and SqlDaf, read-only SqlDaf tables, `SqlDaf.from_db()` for tables kept in other files, and Guide pages on add-ons and booleans.

### Added
- `Daf.update_with_history(new_daf, by='', key_order=None)` merges a rerun of some records into a table and returns `(updated, history)`. A record is the rows of one value in `by`, the keyfield if empty. The rerun is compared as text, so a typed rerun equal to a table read from CSV is no change. Each changed record takes the rerun's rows, and its prior rows go to history. A new key gets a history row with only its key. So history is empty exactly when nothing changed. A rerun with no rows changes nothing. It comes from AuditEngine, where it merged 2,226 chunks of SF sheet1 on Lambdas, and Ray asked for it as a core method on 2026-10-26. It is the first method in src/daffodil/daf_join.py, a module of the split plan, assigned in the class as `apply_schema()` is. It is blocked on a SqlDaf. Tests are in `tests/test_daf_transform.py`. The notes are in `notes/engineering_notebook/2026-10-26_1409_update_with_history.md`.
- `SqlDaf.apply_dtypes()` takes `from_str`, as `Daf.apply_dtypes()` does. The default, True, leaves a str column as it is. With False, a number in a str column becomes its text, so the table holds what `from_csv()` of the same rows holds. Ray asked for it on 2026-10-26. A test is in `tests/test_sqldaf.py`.
- `register_daf_namespace(name)` in daf.py and `register_sqldaf_namespace(name)` in sqldaf.py let a package add its own methods under one name, as `daf.stats.share('qty')`, with no change to daffodil. Each is a decorator for a class whose constructor takes the table. Reading the name makes a new object for that table each time, so a selection or a copy has the add-on, bound to its own rows. A name already in use, a method, an attribute or another add-on, raises ValueError. A Daf add-on read on a SqlDaf raises NotImplementedError, unless the package registers a SqlDaf version, which may use the same name. Add-ons are active once imported, with no entry points. The Guide has a new page, Add-ons. AuditEngine asked for a plan on 2026-10-25, and Ray decided the five points of the hook the same day. The plan is in `notes/engineering_notebook/2026-10-25_1627_addon_hook_and_mixin_split_plan.md`. Tests are in `tests/test_daf_misc.py` and `tests/test_sqldaf.py`.
- `SqlDaf(name, read_only=True)` opens a table whose data cannot be changed. Writing a cell, adding or removing rows or columns, renaming a column, converting with `apply_dtypes()` and dropping the table raise the new ReadOnlyTableError, a subclass of `sqlite3.DatabaseError`, and change nothing. Selections, sorts, joins, `groupby()`, an index and setting the keyfield work, since the data stays the same. SQLite enforces it through an authorizer on the connection, so no method slips past it, and it holds for every SqlDaf on the file until `close()`. A connection with no read-only table has no authorizer. With one, 500 lookups by key took 0.111 s against 0.096 s, and bulk reads about 3% longer. A Daf changes only memory, but a SqlDaf writes to its file at once, and nothing in the file shows it later. AuditEngine found a stage writing into a saved table by luck of a TEMP copy. Ray decided it, Q285: data is protected, the file is not, so indexes are allowed. Tests are in `tests/test_sqldaf.py`.
- `SqlDaf.insert_idx_col(colname='idx', icol=0, startat=0)` numbers the rows, as `Daf.insert_idx_col()` does: the same columns, positions and numbers, from `startat` in this SqlDaf's row order, so a selection is numbered from its own first row. SQLite adds a column only at the end, so it makes a TEMP copy with the column in place, numbered in SQL with ROW_NUMBER(), and the SqlDaf then refers to it, as a sort does. The table it came from is not changed. Before, it raised NotImplementedError. AuditEngine's gen_bif_report numbers rows before showing them. Ray decided it, Q284. Tests are in `tests/test_sqldaf.py`.
- `Daf.to_daf()` returns the Daf itself, as `SqlDaf.to_daf()` reads a table into memory. So code can read a table the same way, whichever kind it is, as in `table.select_records_daf(keys).to_daf().apply_dtypes()`. Nothing is copied, so a change to the result changes the Daf. `apply_dtypes()` on rows that are already typed changes nothing: on 133,422 rows of AuditEngine's d05 biacvr_bif read from a SqlDaf, a second call gave the same rows. AuditEngine asked for one way to read either kind on 2026-10-22, to drop the branch at 11 readers. Ray decided it, Q276. Tests are in `tests/test_daf_construction.py` and `tests/test_sqldaf.py`.
- `groupby_repeated_cols(colnames, exclude=None)` on Daf and SqlDaf groups the rows whose key, of one or more columns, repeats, and leaves out every key seen once. It returns `T_klodaf` on a Daf and the new `T_klosdaf` on a SqlDaf. Each group holds every row of its key, in table order, and the groups come in the order of their first rows, so an empty result means no key repeats. Python NULL is a value, so blanks in the same key column match. `exclude` names, for any column, the values that keep a row out of every group, such as a first key column of 0 or blank. On a SqlDaf one GROUP BY with HAVING COUNT(*) > 1 finds the repeated keys, an index is made only when a key repeats, and each group is a selection. On AuditEngine's pilot d05 bia_bif, 133,422 rows, the three uses AuditEngine named took 0.03 to 0.18 s on a SqlDaf and 0.05 to 0.28 s on a Daf, with no repeats, the same on both. On FL_Collier's imagematch table, 78 rows, it found 39 groups, and each group's first row is the one AuditEngine kept. AuditEngine asked for this on 2026-10-22, to replace two Python copies and a SQL statement. Ray named it, Q277. Tests are in `tests/test_daf_group_scope.py` and `tests/test_sqldaf.py`.
- `SqlDaf.from_csv()` takes `strict_header`. The default, True, keeps the check that every file has the first file's header, or ValueError. With False, each file's values go into the columns named in its own header, so files whose columns differ in order or in set load into one table. The table has the first file's columns, then each new name in the order first seen, and a file that lacks a column gives Python NULL in it. SQLite puts a row's values into the columns by position, so placing by name is what keeps a value out of the wrong column when headers differ. A misspelled name, or a file of another kind, then becomes a new column with no error, which the strict default stops. The first line of every file is read before any row. Only a file whose columns differ is placed by name, so a file with the table's columns loads as fast as before. On the 20 Dane biacvr_bif hunks, 344,347 rows with one hunk in another order, the load took 5.76 s, and the rows of that hunk matched it by name. A long row is checked against its own file's header. AuditEngine asked for this on 2026-10-21, for that hunk and for chunks with the old file_paths column next to new ones with file_path and file_path2. Ray decided it, Q273, and chose the name and the strict default, Q275. Tests are in `tests/test_sqldaf.py`.
- `SqlDaf.from_db(path, name)` copies a table from another database file into the working file, `SqlDaf.db_path`, and opens the copy. The rows keep their order, with the keyfield and dtypes recorded for the table. Only that table and its own record are copied. The source is attached read-only for the one statement that copies it, then detached, so SQLite refuses any write to it, and any number of files can be copied in. Joins, `concat()` and keys then work across tables kept in separate files. `save()` takes `db_path`, to write a result into a file of its own, and the SqlDaf then refers to it there. Both raise ValueError inside a transaction, where SQLite cannot detach a file it has used. SqlDaf's connections allow URI filenames for the read-only attach. A plain path opens as before. On AuditEngine's pilot d05 files, biacvr_bif, 133,422 rows of 52 columns, copied in 0.37 s, and cvr_votes!final, 1,603,908 rows in an 804 MB file, in 6.75 s. A left join of biacvr_bif and blt_bif matched `Daf.join()`, and its save into its own file took 0.48 s. The source files were unchanged. AuditEngine asked to join tables kept in separate files. Ray chose a copy over reading the files in place through ATTACH, Q274. Tests are in `tests/test_sqldaf.py`. The research is in `notes/engineering_notebook/2026-10-22_1408_tables_in_other_files.md`.
- `SqlDaf.groupby_reduce()` groups the rows by one column and reduces each group to one row, as `Daf.groupby_reduce()` does, with `by='row'` or `by='sparse_row'`. It runs on the registered aggregate of `SqlDaf.groupby_aggregate()`, so the groups are reduced inside SQLite in one query. With `by='row'`, the rows of each group are kept as stored and reduced at the end by `Daf.reduce()`. So the function is called as `Daf.groupby_reduce()` calls it, `silent_error` works, and a plain `Daf.sum_da` adds down each column. With `by='sparse_row'`, the function gets the dict of each `indirect_col` cell. The rows come in the sorted order of the column, and `by='table'` is not supported. `batch_size` is kept but has no effect. A first version, on 2026-10-17, read the rows in sorted batches and reduced whole groups in Python blocks. Ray then asked to move it onto the aggregate, Q243. On SF sheet1, the results of five calls were the same in both versions. The times went from 0.51 s to 0.37 s for step 15, 1.52 s to 1.09 s for the precinct concise sums, and 1.23 s to 0.92 s for 73,888 small groups. Sums by row of plain columns went from 0.19 s to 0.16 s, and of all 13 columns from 0.59 s to 0.45 s, once each cell came as its own argument. Ray decided this on 2026-10-17, Q217, Q220, Q224 and Q243. Tests are in `tests/test_sqldaf.py`.
- `SqlDaf.groupby_aggregate()` groups the rows by one or more columns and reduces each group with a Python function, registered in SQLite as an aggregate. It is one query. SQLite gives the function each row of a group, in the order of the table, and the function returns one result per group, so the rows do not come into Python as Dafs or blocks. With one column, the function gets the cell's dict, as JSON or PYON text, and the result is the reduced dict as JSON. This merges or totals concise dicts, which SQL's own JSON functions cannot read when they are PYON. With several columns, each cell goes to the function as its own argument, up to SQLite's limit of 1000, and the result comes back as one JSON array that SQL splits into the `result_cols`. A first version packed each row as JSON, which SQLite writes with 15 digits, so 0.1 + 0.2 arrived as 0.3. Results are written in daffodil's form by `normalize_val()`. The order of the rows comes from an ORDER BY inside the aggregate call. An error in the function is raised as itself. On the SF sheet1 step 15, 80,443 contests into 3,795 ballots, it took 0.36 s against 0.47 s for `groupby_reduce()`, with the same rows. Ray asked for this on 2026-10-17. Tests are in `tests/test_sqldaf.py`.
- `normalize_val()` in daf_utils.py puts a value into daffodil's form, at any depth in a dict, list or tuple. A bool becomes 1 or 0, and None and NaN become the empty string. It is for a dict or list cell about to be written as JSON or PYON, which would otherwise hold true, null, True or None. It works like `_convert_str()`, which does the same for one scalar. An int or str is kept with no recursive call. Ray decided this on 2026-10-17, Q228. Tests are in `tests/test_daf_utils.py`.

### Changed
- The Guide's Booleans page states that values are put into daffodil's form once, when the row is built: bools as 0 and 1, and None as NULL, with `normalize_val()` inside a dict or list. Such a cell is PYON and JSON at once, and daffodil does not normalize it again. The Types page shows None in cells before and after `apply_dtypes()`. Ray decided it on 2026-10-26, Q300.
- **Breaking.** A bool converted to str is `'1'` or `'0'`, as `apply_dtypes()` with `from_str=False` gives. Before, it became the int 1 or 0, so a cell with dtype str was not a str. In daffodil a boolean is the int 0 or 1, never the word True or False, since each system spells the words its own way. Ray set the rule on 2026-10-26, Q299. A new Guide page, Booleans, states the standard, the reasons and the accepted risk that a type checker sees an int. CLAUDE.md and the cheatsheet state it too. A test is in `tests/test_daf_utils.py`.
- The SqlDaf page of the Guide has a section, How cells are stored. A doctest shows each kind of cell written from a Daf, what SQLite holds by `typeof()`, how it prints and reads back, SQL's sums and counts, SQLite's JSON functions on dict and list text, the same table loaded from CSV and converted by `apply_dtypes()`, and the round trip back to the Daf. Ray asked for these examples on 2026-10-25.
- SqlDaf's `apply_dtypes()` and `from_csv()` take dtype dict and list. They record it with the table and convert nothing, since the stored text is the value. Before, both raised NotImplementedError. `to_daf()` carries the dtype to the Daf, whose `apply_dtypes()` decodes the column. `STORABLE_DTYPES` includes dict and list, and the new `TEXT_DTYPES` names the dtypes stored as text: str, dict and list. Ray decided it, Q295. A test is in `tests/test_sqldaf.py`.
- A dict or list cell written to a SqlDaf is stored as its text, the `str()` that `to_csv_file()` writes for it. Before, SQLite raised ProgrammingError, as "type 'dict' is not supported", in `from_daf()`, `append()`, `extend()`, `update_record_krow()`, `concat()` of a Daf and assignment by brackets. So a table written from a Daf stores the same as the Daf written as CSV and loaded with `from_csv()`, and a test checks that. The text stays text on the SqlDaf. SQLite's JSON functions read daffodil's form of it, with bools as 0 and 1 and missing values as '', as JSON5. `to_daf()`, then the Daf's `apply_dtypes()`, decodes it. Each cell written is tested for being a dict or a list, which adds about 0.14 s per 1,000,000 rows of 5 columns, about 10% of a write. A subclass, such as OrderedDict, and a tuple are still refused by SQLite. AuditEngine asked for this on 2026-10-24, for its corrections log. Ray kept the direct write, with the round-trip test, Q293. Tests are in `tests/test_sqldaf.py`.
- Docstring and Guide examples name each Daf with the type suffix, such as `a_daf`, `k_daf` and `ballot_daf`, where 21 had one letter, as `a`, `e` or `k`. Three more show a new Daf before using it, as Q249 asked: an empty keyed Daf in `set_keyfield()`, and a Daf made from a schema in daf_schema.py and on the Guide page on columns, keys and schemas. The rest were checked by a scan: examples about printing, `from_directory()`, tables built from empty in a loop, and a SqlDaf made from a Daf shown just above are left as they are. Ray asked on 2026-10-24.
- The Guide's patterns for code that runs on a Daf or a SqlDaf have one more: summarize each group with `valuecounts_for_colname1_groupedby_colname2()`, not a loop over `groupby()`. On AuditEngine's pilot d05, 11,157 batches, the loop took 565 s on a SqlDaf and the grouped counts 1.64 s, with about 0.7 s for each on a Daf. Ray asked for it on 2026-10-23.
- `Daf.concat()`, and `append()` of a Daf, take a SqlDaf: its rows are read into memory with `to_daf()`, and the result is a Daf. Before, it raised AttributeError, since a SqlDaf has no `lol`. The rows read are new lists, so they are adopted with no second copy. A Daf's rows are still deep-copied. So `Daf().append(sdaf_selection)` works, as AuditEngine's gen_bif_report did with an empty group. A large SqlDaf then fills memory, so select its rows first. No size check was added. Ray decided it, Q283. Tests are in `tests/test_sqldaf.py`.
- `SqlDaf.keys()` takes `silent_error` and `astype`, as `Daf.keys()` does. `astype='view'` gives a KeysView of a dict built from the keys, in row order, a repeated key once. A Daf's view of its key index is free, but here every key is read. `silent_error=False` with no keyfield raises KeysDisabledError, and another `astype` raises ValueError, as on a Daf. Before, any argument raised TypeError. The docstring and the Guide say it is there for compatibility, and give the better ways, which read no key into Python: give the table itself as keys, test one key with `in`, sample with `table[0:3].keys()`, count with `len()`. Ray decided it, Q282. Tests are in `tests/test_sqldaf.py`.
- The docs recommend `to_md()` over an f-string for a table in a report or a log, as in `daf[0:50].to_md()`. It writes every row selected and leaves the summary line out by default, where an f-string gives `str()`, with 10 rows and the summary line. The text is the same on a Daf and a SqlDaf. It is in the `to_md()` docstring, the SqlDaf page of the Guide and the cheatsheet. Ray asked for it on 2026-10-23, when he kept the TEMP name of a SqlDaf selection, Q281.
- `str()`, `print()` and an f-string of a SqlDaf give the Markdown table a Daf of the same rows gives, byte for byte: at most `md_max_rows` rows and `md_max_cols` columns, 10 each, then the summary line. Before, `str()` gave the one-line `repr()`, and an f-string raised NotImplementedError. The rows are read from the table in SQL, a count and then only the first and last rows shown, so on 1,603,908 rows it took 0.64 s. An f-string with a spec, such as `.2f`, formats one cell, as on a Daf. `repr()` gives the same table, with a newline in front, as for a Daf. Before, it gave one line, such as `<SqlDaf table name='t' rows=30 ...>`. `is_child` and `is_temp` tell the kind. AuditEngine's gen_bif_report writes such a table into a report. Ray decided it, Q279 and Q280. Tests are in `tests/test_sqldaf.py`.
- The SqlDaf page of the Guide has a section, Code that runs on a Daf or a SqlDaf. It shows how to compare two tables, sample, count and split a table so the same code is fast on both, with a doctest that runs each on both. The cheatsheet has two notes from it. AuditEngine settled these patterns while converting gen_bif_report, and Ray asked for them in the docs on 2026-10-23.
- **Breaking, for SqlDaf only.** SqlDaf needs SQLite 3.48 or later, for ORDER BY inside an aggregate, JSON5 input and 1000 function arguments. Opening or making a SqlDaf on an older SQLite raises RuntimeError, which says how to get a newer one. Nothing else changes: importing daffodil, and every Daf method, work on any SQLite, since a Daf never uses it. Checked with SQLite reported as 3.37.2, 2026-10-24. SqlDaf is new since 0.8.0, so this breaks only code that uses it on an old SQLite. The system Python 3.11 of Ubuntu 22.04, which AuditEngine uses on EC2, has SQLite 3.37.2. AuditEngine's Lambdas do not use SqlDaf. Every Python that uv installs has a newer SQLite. Ray decided this on 2026-10-18, Q246. Tests are in `tests/test_sqldaf.py`. The research is in `notes/engineering_notebook/2026-10-18_1428_sqldaf_group_reductions.md`.
- **Breaking.** `Daf.groupby_reduce(by='sparse_row')` keeps each group's result concise. The result has two columns, the group column and the column named by `indirect_col`, whose cell holds the group's dict with only the keys found in it. Before, the keys were spread into columns taken from the first group, and a key that group lacked was dropped with no error, so a sum was lost. On a SqlDaf the cell is compact JSON text. Ray decided this on 2026-10-17, Q222. A test is in `tests/test_daf_reduction.py`.
- SqlDaf's `to_csv_file()` and `to_csv_buff()` write a column name stored with the `__HH` encoding decoded. AuditEngine stores the column group as `__67roup`, and its own exporter writes `group`. So a SqlDaf export of its table now has the same header. The rows are unchanged. The header is decoded with `sql_unesc_str()` of daf_sql.py, as AuditEngine's exporter decodes it with its copy in sql_utils.py. Ray decided this on 2026-10-17. A test is in `tests/test_sqldaf.py`.
- Docstring and Guide examples show each new Daf as a table, with its summary line, before the operation on it. There are 158 in the docstrings and 29 on the Guide pages. Examples about printing, and `from_directory()`, whose paths and times change, are left as they were. A SqlDaf example made with its rows shows them by `.to_daf()`, since a SqlDaf printed as one line then. There are 29 of these in the docstrings and 4 on the Guide page. Each table runs as a doctest. Ray decided this on 2026-10-18, Q249.
- Docstring examples name a Daf `daf`, not `d`, as CLAUDE.md asks. 425 names changed in the examples, and 2 in the text.
- The docs build loads a hook, notes/scripts/mkdocs_hooks.py. It shows the `<BLANKLINE>` of a doctest as a blank line on the Guide pages, as mkdocstrings already does on the API pages.

- `to_md()` with no limits shows a ragged Daf, with its short rows padded as copies. Before, it raised RuntimeError from `transpose_lol()`. The Daf is not changed. With `include_summary`, as in print, the summary line ends with `ragged rows detected`. `from_md()` skips that part, and reads the table back as rectangular. The padding is in `pad_ragged_lol()` in daf_utils.py, which printing also uses. Ray decided this on 2026-10-18, Q251.
- mypy runs in CI, as its own job, and in notes/scripts/check_before_push.sh. Any error fails the check. Nothing ran it before, so its errors grew from 3 to 8 after it was added.
- `multi_groupby()` is annotated and documented as returning a dict of KeyedLists of Daf, which it always returned. It said a dict of dicts.
- `reduce_dodaf_to_daf()` is annotated to take a KeyedList of Daf too, which `multi_groupby_reduce()` passes it.
- `to_csv_buff()` is annotated to return str, which it always returns. It said bytes or str.
- `replace_in_columns()` is annotated to take a list of str, a list of int, or a mix, for `cols`.
- `T_df` and `T_npa` in daf_types.py are declared as type aliases. mypy rejected `T_df` as a type, so `T_dodf`, `T_lodf` and `T_idodf` checked nothing.
- The functions in daf_md.py, daf_pandas.py and schemaclass.py that are attached to a class annotate `cls` or `self`. mypy skipped their bodies before, and checking them found the two annotations above.
- Casts in `groupby_cols_reduce()`, SqlDaf `apply()`, `groupby()` and `apply_in_place()` tell mypy that a group key is a tuple and that a KeyedList row's keys are str. The code is unchanged.
- A row of `for row in daf` is typed as a dict, as in the default itermode. `iter_klist()` is typed for KeyedList rows. `iloc()` on Daf and SqlDaf and `reduce()` have overloads, so a type checker sees a dict, a KeyedList or a list by `rtype` or `by`. Annotations only (Q258).
- KeyedList is generic in its key and value types, as in `KeyedList[str, int]`. A bare KeyedList is `KeyedList[Any, Any]`. `groupby()` and the other group methods are typed to return `KeyedList[Any, Daf]`, so a group key is Any and each group is a Daf. Making a KeyedList took the same time before and after, about 0.5 µs (Q259).
- The value count methods on Daf and SqlDaf are typed to return dicts with keys of any type. The keys are cell values, which are often ints. They were typed as str. `daf_sum()` is typed to return a dict.
- `keys()` is typed by `astype`: a list for `list`, and a KeysView for `view`. Keys are typed Any, in `T_kva` too, which said str, int or tuple. `remove_keylist()` takes a list of any keys, as its example uses ints (Q260).
- `calc_cols()` is typed to return a list of str, which it builds. `include_types` and `exclude_types` take one type or a list, as the code already did (Q261).
- `SqlDaf.from_csv()` loads many files into one table, as `read_csv()` of dask does. It takes one file, a glob pattern or a list, where 0.8.1 took one file. Every file must have the same column names, and an empty file is skipped. A new `dtypes` converts the columns as the rows are loaded. On 200,000 rows that took 1.0 s, against 1.4 s for a text load and `apply_dtypes()` (Q254).
- `sum_da()`, `sum_kl()` and `update_row()` are typed to return the type they are given, with a new type variable `T_rma`. A dict of int total stays a dict of int to mypy (Q263).
- SqlDaf uses Python's own sqlite3 module, as 0.8.1 did. Between the releases it used pysqlite3-binary when installed, with a `sqlite` extra. That was dropped before this release, so neither ships. pysqlite3-binary keeps every TEMP table and sort in memory: one grouped sum on 1.6 million rows peaked at 665 MB with it, and 40 MB with the stdlib driver. For development, pyproject.toml sets `python-preference = "only-managed"` for uv, so the tests and CI run on uv's Python builds, which have SQLite 3.53.1 for 3.10 to 3.13. CI's 3.12 had used Ubuntu's own Python, with SQLite 3.45 (Q253).
- `SqlDaf.reduce()` types `cols` as `T_cs`, as `Daf.reduce()` does, so a KeysView or a dict can be passed. A sparse sum tests each key of each row against `cols`, and on 32,253 rows with 385 columns it took 1.05 s with a list and 0.19 s with a KeysView or a dict. The code does not change (Q265).
- A plain sparse sum, `reduce(Daf.sum_da, by='sparse_row')` on a Daf or a SqlDaf, chooses its loop once, from the first 100 rows, and tests no key inside the loop. With no `cols` it adds every key. With `cols` no longer than the mean keys per row it looks up each column. With longer `cols` it adds every key and picks the columns at the end. With `cols`, the result is in the order of `cols`, where it was in the order the rows first had the keys. `SqlDaf.groupby_reduce()` and `groupby_aggregate()` do the same. On pilot d05, 5 columns took 0.31 s instead of 0.42 s, and 385 columns 0.43 s instead of 1.36 s as a list, with the same totals (Q266).
- `SqlDaf.concat()` adds nothing when the other has no rows, whatever its columns, as `Daf.concat()` does. Before, it raised KeyError when the columns differed. The empty result of `apply()` has no columns, so step 16 of the find_variants port stopped on WI_Dane functest when no variant ballot lacked an agreed contest. The rows are counted only when the columns differ, so a normal concat costs nothing more. AuditEngine asked for this on 2026-10-21. Ray decided it, Q272. A test is in `tests/test_sqldaf.py`.

### Fixed
- A selection used as keys built no key index on its own table. Each key was then looked up in that table by a scan, so the time grew with rows times keys: 2.84 s for 10,000 rows against 10,000 keys. The key index is built lazily, as for any other lookup by key, by the same rule as `_need_keyfield()`: when the selection's keyfield is its table's. Measured on a synthetic table, selections against selections, `inverse=True`, the first call building the indexes: 10,000 by 10,000 keys 0.04 s, 100,000 by 100,000 0.23 s, 133,422 by 1,603,908 1.55 s, and 1,000,000 by 1,000,000 2.21 s. A second call took 0.01 s, 0.08 s, 0.12 s and 0.89 s. `select_record()`, and so `sdaf[key]` and `key in sdaf`, also built the table's key index on a selection keyed by another column, and raised ValueError when the table had no keyfield. They follow the same rule now. `join()` needs no key index: SQLite indexes the join itself, 0.53 s for 200,000 rows each without the indexes and 0.46 s with them. Ray asked how many keys the earlier fix covered, which found it. Ray chose this fix, Q292. Tests are in `tests/test_sqldaf.py`.
- `SqlDaf.apply_dtypes()` recorded the new dtypes in daf_tables, and on the SqlDaf, before it converted the values, in a separate transaction. When the conversion failed, the record said a column was int while its values were still text. The dtypes are now recorded in the same transaction as the conversion, and set on the SqlDaf only after it. When nothing needs converting, they are recorded at once, as before. A value that does not convert is left as text, with no error, in this release as in 0.8.1. So a conversion fails only when it raises, as on a read-only table, which is new here.
- 0.8.1 was not affected by this one. It came in with the fix for a selection as keys below, and is fixed before release. A selection whose parent table had no rows could not be read, sorted or used as keys. SqlDaf found a selection's table with `parent or self`, which tests whether the parent is true, and an empty SqlDaf is false, so the selection was taken for its own table. Since the fix for a selection as keys, this raised ValueError from `ensure_key_index()`, "is a child". AuditEngine's find_variants port stopped on it on pilot d05, where the votes residue table is empty. Now `_root()` tests the parent against None, in all 16 places. A selection is always one level below its table, however it was nested, so this covers any depth. The docs also say that a chain of selections made before any is read nests one query per step, and SQLite stops at 413 levels with OperationalError "Recursion limit". Ray decided it, Q289. Tests are in `tests/test_sqldaf.py`.
- `select_records_daf()`, and every method that takes keys, ran a time that grew with the product of the two sizes when the keys were a SqlDaf selection. Each row scanned every reference of the selection. A selection's table of references now declares ref with the type of the parent column it refers by, INTEGER for daf_rid, and gets an index on ref the first time the selection is used as keys. An index on an untyped ref was not used by SQLite. On AuditEngine's pilot d05, 133,422 rows of full_bif against 1,603,908 of cvr_bif, the inverse select went from past 115 s to 0.92 s, and the other direction took 1.30 s. A selection read only for its rows makes no index. AuditEngine found it with the cookbook's compare pattern on 2026-10-23. Ray chose this fix, Q288. Those timings held only because both tables already had their key indexes. The entry above finishes the fix. In 0.8.1, 10,000 rows against a selection of 10,000 keys took 4.96 s. Now they take 0.04 s. Tests are in `tests/test_sqldaf.py`.
- `annotate_daf()` raised TypeError when the keyfield had more than one column. It looked up the key as a list of values. It now builds the key as other key lookups do, as a tuple (Q262).
- SqlDaf `groupby_aggregate()` and `groupby_reduce()`, both new in this release, with `diagnose=True` raised AttributeError. 0.8.1 was not affected. They called `head()`, which SqlDaf does not have. They print the first three groups now. mypy found it (Q257).
- Printing a ragged Daf no longer pads its short rows. `reduce_lol_cols()` said it did not alter the lol, but it called `equal_cols_lol()`, which pads in place. So printing made a ragged Daf rectangular. It now pads copies of the short rows only. The table looks the same. Ray decided this on 2026-10-18, Q250. The test is in tests/test_daf_md.py.

## [0.8.1] - (2026-10-16)

This release has two breaking changes, marked **Breaking** under Changed. `wide_to_narrow()` names a column `variable` by default, where it was `varname`. `from_md()` raises at a row with the wrong number of cells, where it dropped that row and the rest. It also fixes right and outer joins, and `from_md()` with a `|` in a cell.

### Added
- `update_record_krow(key, record)` on Daf and SqlDaf changes some cells of the row with this key, as `update_record_irow()` does for a position. An int is always a key here, never a position, so it suits a keyfield of ints, where `sdaf[20, 'v'] = 1` raises NotImplementedError. A missing key raises KeyError and changes nothing, and no keyfield raises ValueError. On a Daf, as with `update_record_irow()`, the key index is not updated, so don't change a key with it. On a SqlDaf a key may change, with the usual checks for a repeated or a Python NULL key. `update_by_keylist()` stays as the bulk form. The commented-out `@deprecated` line above it is removed: the bracket form it named, `daf[keylist] = record`, reads int keys as positions and replaces whole rows, so it was no replacement. Ray decided this on 2026-10-16, Q200. Tests are in `tests/test_daf_key_cells.py` and `tests/test_sqldaf.py`.

### Changed
- The error for an int row in SqlDaf assignment, as in `sdaf[20, 'v'] = 1`, says that an int row is a position, which a SqlDaf does not assign by, and names `update_record_krow()` for a row by key. Before, it said to use `sdaf[key, col]`, the form just used, which misled a caller whose keys are ints. Ray decided this on 2026-10-16, Q199.
- README.md and the Home page of the docs get a section, Tables larger than memory. It says what SqlDaf is for, how much slower it is than a Daf, and that it supports the most common Daf methods. The Guide page Tables in SQLite is renamed Tables larger than memory, to match. Its address, guide/sqldaf/, is unchanged. Ray decided these on 2026-10-16, Q194 and Q195.
- The cheatsheet gets a section, Tables larger than memory, with eleven lines of SqlDaf code. Each one runs as a test, as for the other sections. The test restores `SqlDaf.db_path` afterwards.
- The Guide page Tables larger than memory covers more: loading a CSV file and converting its text with `apply_dtypes()`, changing rows and values, `transaction()` for many small writes, why rows should not be deleted from a table that has selections, joins, and the NotImplementedError of an unsupported method. Every example runs as a test.
- Two SqlDaf docstrings: `join()` named a test file that no longer exists, and `apply_dtypes()` said a dict column would need JSON text. A dict or list column stays as the text of the file, which `reduce(by='sparse_row')` reads.
- The SqlDaf docs, revised after an outside review of 2026-10-16. The Guide page gains: the three kinds of result and what a change to each does, when a selection's rows become fixed, a table of what each operation on a selection does to its parent, safe exclusion with `delete_records()` against a delete in place, sorting, saving, closing and reopening in one example, a Differences from Daf section, and a Performance section that holds the timings. The cheatsheet section works on one table throughout, converts the CSV numbers before using them, and lists the rules that differ from a Daf. In the docstrings: `select_record()` and `iloc()` on a selection see only its rows, an int in brackets is always a position, `apply_dtypes()` does not check that every value converted, `length_priority` gives numeric order only for whole numbers without a sign or leading zero, `save()` refuses a name in use and says how to replace a table, `close()` and `delete()` state what they keep and undo, and `sql()` says that a column change made through it is not seen by an open SqlDaf. Only docstrings changed in sqldaf.py, checked by comparing the code with the docstrings removed.
- The whole docs, revised after a second outside review, of 2026-10-16. Every claim was checked by running it. The changes:
  - Key index. The Guide page on selecting gains a section, Changing a key, with an example. A key changed with `[]` after the index is built is found only after `set_keyfield()`, and a selection that shares the row keeps its own index. The Daf class docstring no longer says the index is rebuilt after any change, and `__setitem__()` gains the same example.
  - Shared cells. The Guide, the indexing notes and the docstrings of `__getitem__()` and the select methods say that new row lists still hold the same cell values. A list or dict in a cell is copied only by `copy('deep')`. Wording for copies is the same everywhere: a new outer row list, with the row lists shared, and `copy('editable')` for row lists of your own.
  - Key ranges. A tuple takes the rows between its two keys in row order, and its ends are keys of any type, so `daf[(10, 20), :]` works for int keys. The rule that an int is always a position now says outside a tuple.
  - Repeated keys. `keys()` gives each key once, in the order of its first row, and a lookup finds the last row. The Guide page on keys, `keys()`, `select_records_daf()` and `join()` say this. `join()` is one to one, not many to many, and a composite keyfield raises KeyError.
  - Joins. The Joins page and `join()` say which table a row that only the other table has takes its key and shared fields from. See Fixed for the fault this first documented.
  - Types. The Guide page and `apply_dtypes()` separate declaring, converting and checking types, and say that conversion is permissive. The bool rules are stated: an empty cell becomes 0, and `'False'` gives 0. The save-and-restore example in `apply_dtypes()` is removed, since `cols=` does that. Loading with `dtypes=` is called deprecated in the Guide too, and an example that used it now loads first.
  - Missing values. The Guide no longer says that counts skip NULL and that pandas and NumPy turn it into their missing value. Value counts include it unless `omit_nulls=True`. `to_pandas_df()` keeps it as text, and `default=` writes into the Daf.
  - Apply and reduce. Advice to pass `cols=` to `apply()` and `to_md()`, which have no such argument, is gone. The Guide gains a custom reduction, and shows that `sum()` and `reduce(Daf.sum_da)` total a column with text differently. `reduce()` says that keys of the result that are not columns are dropped, and that `initial_da` is changed in place. The claim that inserting and deleting rows is cheap is narrowed.
  - Formulas. `apply_formulas()` stops after 100 passes. The docs called that cycle detection, and a circle that settles raises nothing. The Guide states the limit, the `eval()` warning and that changed cells stay changed after an error.
  - Markdown. The Guide and `from_md()` list what a round trip loses, and point to JSON for an exact copy.
  - Reshaping. `wide_to_narrow()` names its column `varname` and `narrow_to_wide()` looks for `variable`. Both docstrings say so, with a round trip.
  - Install. Getting started gains a table of the features that need another package, with the install command for each. README.md and Home point to it. The core still installs only Markdown and xlsx2csv.
  - Home. The pandas comparison is shortened to one table, and the detail moves to a new Guide page, Performance. A claim about `df.T` is narrowed to frames whose columns have different dtypes.
  - Coming from pandas and the cheatsheet. The table is of tasks, not equal expressions, with notes on result types and changes in place. `df.loc[k]` maps to `select_record(k)`, `df.sort_values()` to a sort of a copy, and `df.replace()` to `replace_in_columns()`.
  - Cheatsheet. It shows its setup code, and says that each table starts from it and runs in order. Rules are added for copies, loop rows, repeated keys, the key index, permissive conversion and which methods change the Daf. The section Join and out is renamed Joins and export.
  - SqlDaf. A selection fixes its rows when first read whole. `select_record()` and `iloc()` read one row and fix nothing, unless `iloc()` gets a negative position. Some methods run a query at once for their arguments. Indexing with a column part gives an independent SqlDaf, as `select_cols()` does. The table of results is headed Replacing an existing cell.
  - API reference. A new section of the Daf overview explains the T_ names in the signatures, and the `_MISSING` default. `to_lod()`, `to_lota()` and `apply_schema()` gain examples that call them. `from_numpy()` says an object array keeps its values. The Google Sheets placeholders lose their history. `from_pdf()` is described as experimental instead of shown with no docstring.
  - Testing for NULL. The Types guide gains a section. It says to test with `is NULL`, which is about three times faster than `==` on a number cell. It also says that `NULL = ''` in any module is the same object, so no import is needed, that `val is ''` makes Python warn, and that an empty NumPy string or str subclass in a cell is missed. Ray agreed on 2026-10-16, Q206.
  - After an outside review by Gemini, of 2026-10-16. The Types guide says that daffodil does not turn None or NaN into NULL, and that `is NULL` is False for both. `force_rectangular()` says that it pads the rows in place, so a selection sees the new cells, and that the key index stays valid. The cheatsheet says that `daf[('a1', 'c3')]` without `, :` returns one cell, with no error, when a column c3 exists.
  - Wording: Parts copied, The reduction function, is removed, and others the review named.
- `notes/scripts/build_docs_review.py` reads the docs with griffe, as mkdocstrings does. Signatures show `_MISSING`, not the repr of an object, an entry named by a heading option gets that name, and a class lists its members, as KeyedList does on the site.
- **Breaking.** `wide_to_narrow()` names its column of former column names `variable` by default, where it was `varname`. That is the name `narrow_to_wide()` reads by default, and the one pandas `melt()` uses, so the two methods chain with their defaults. Before, the chain raised KeyError. Code that relies on a column named `varname` passes `varname_colname='varname'`. AuditEngine calls neither method. Ray decided this on 2026-10-16, Q205.
- **Breaking.** `from_md()` raises RuntimeError at a row with more or fewer cells than the header. Before, the table ended at such a row, and the rows after it were lost with no error. The check for this existed but could not run, since the table finder ended the table first. The finder now ends a table only at a line that is not a row, such as a blank line. The error gives the row and its line in the text. Ray decided this on 2026-10-16, Q207.

### Fixed
- `Daf.join()`, right and outer joins: a row that only the other Daf has takes its key, and each column in `shared_fields`, from the other Daf. Before, `join_records()` took them from this Daf, which has no such row, so they were NULL. The key was lost when the two keyfields had different names, and a shared field was lost always. `join()` now fills those cells from the other row, after `join_records()`. `join_records()` and the translator are unchanged, so a custom translator works as before. Inner and left joins cannot change. Daf and SqlDaf now give the same rows for every kind of join, with and without `shared_fields`. AuditEngine's joins are all inner or left. Ray decided this on 2026-10-16, Q209.
- `from_md()` reads a `|` in a cell. `to_md()` writes it as `\|`, but `from_md()` split each row on every `|`. That row then seemed to have one more cell, and the table ended at the row before it, with no error. A `|` in the first row gave no rows at all. A new function, `split_md_row()` in daf_md.py, splits only on a `|` with no backslash before it, and turns `\|` back into `|`. It is used in both places that split a row. Ray decided this on 2026-10-16, Q204.
- `from_md()` reads a composite keyfield from the footer. The footer writes it as the text of a tuple or list, and the parser kept that text, so `Daf()` raised KeyError. A keyfield that starts with `(` or `[` is now read with `ast.literal_eval()`. The footer line is unchanged, so files already written read too. Q204.

### Tests
- `tests/test_daf_md.py`: a newline in a cell reads back as `<br>`. A `|` reads back in a cell, in the first row and in a column name, and a keyfield of one name, a tuple or a list reads back from the footer. `split_md_row()` has a test of its own. A row with another cell count raises, and the xfail that recorded the dead check is gone. `wide_to_narrow()` and `narrow_to_wide()` chain with their defaults.
- `tests/test_daf_transform.py`: the two xfails for the outer join pass, and are plain tests. New tests cover a right join, `tag_other`, and inner and left joins. `tests/test_sqldaf.py` compares Daf and SqlDaf joins with `shared_fields` for all four kinds, where it covered inner and left.

## [0.8.0] - (2026-10-16)
### Added
- New class SqlDaf, in `sqldaf.py`: a Daf whose rows live in a SQLite table, for tables too large to hold in memory. It is a Daf, with the methods written for SQL: selections that run no query until read, work by key, joins, groups, counts and sums, `apply()` and `reduce()`, adding and changing rows and values, changing columns, sorting, and loading from and writing to CSV files. Every other Daf method raises NotImplementedError, so nothing quietly loads a large table into memory, and `sql()` runs a statement of your own. The docs have a Guide page, Tables in SQLite, and an API reference of five pages. The entries after this one give the steps of its development, in order. The design and the decisions behind it are in notes/sql_backing/design_draft.md.
- `delete_records(keys, inplace=False, silent_error=False)` removes the rows with these keys. By default it leaves the Daf as it is and returns a new Daf without them, as `select_records_daf(keys, inverse=True)` does, sharing the rows that remain. With `inplace=True` it deletes them from the Daf itself, rebuilding the row list once, and returns it; an earlier selection keeps its rows. The keys may be a list, an iterable or another Daf. A missing key raises KeyError and removes nothing, unless `silent_error=True`. On 1,000,000 rows, deleting 500,000 in place took 0.48 s. SqlDaf will get the same method with the same default, so the same call means the same on both. Ray decided this on 2026-10-14, decision 104 and item 2 of the mdaf review. Tests are in `tests/test_daf_delete_records.py`.
- mypy is in the dev tools, and pyproject.toml has a `[tool.mypy]` section. Run it with `uv run mypy src/daffodil`. boto3 and markdown are optional and not installed, so their imports are not checked. Three errors remain, in `lib/daf_types.py` lines 53 to 55: `T_df` is defined in two branches, so mypy does not accept it as a type. Fixing that is a separate item. Ray asked for mypy on 2026-10-14.
- New module `sqldaf.py`: `SqlStore` owns one SQLite database, and `SqlDaf` is a Daf whose rows live in a table there, for tables too large for memory. This is step 1 of the design in notes/sql_backing/design_draft.md, decisions 88 to 107 of 2026-10-13 and 2026-10-14. `SqlStore` opens a file, a temporary file deleted at `close()`, or a connection you pass in. `from_csv()` only reads, as `Daf.from_csv()` does without dtypes: it stores the text of the file one row at a time, in one transaction, so a failing row leaves no table. `from_daf()` stores each value as it is. `table()`, `create_table()`, `insert_rows()`, `drop_table()` and `transaction()` complete the store. `transaction()` runs a block as one transaction, nested as a savepoint, and halves the cost of runs of small statements. Every table it makes has a hidden `daf_rid` INTEGER PRIMARY KEY, which keeps the row order and survives VACUUM. `SqlDaf` has `len()`, `bool()`, `columns()`, `num_cols()`, `to_daf()`, `sql()`, a fallback that runs any statement, `apply_dtypes()`, which converts in place with Daf's own converter so the results match a Daf, and `set_keyfield()` and `ensure_key_index()`. The unique index on the keyfield is made when a key is first used, as a Daf builds its key index, and a repeated key raises then. Every other Daf method on a SqlDaf raises `SqlNotSupportedError`, which names it, so nothing quietly loads a large table. NULL is stored as SQL NULL and comes back as NULL. On 1,000,000 rows of 6 columns, a 39 MB CSV file: `from_csv()` 4.1 s, `apply_dtypes()` of two int columns 2.1 s, `ensure_key_index()` 0.5 s, and a lookup by key 17.9 us, or 9.0 us inside `transaction()`, all with a peak of 40 MB, the memory of the import alone. `Daf.from_csv()` and `apply_dtypes()` of the same took 3.8 s and 573 MB. Tests are in `tests/test_sqldaf.py`.
- `sqldaf.py` step 2: selections. `select_where()` with a Python function, `select_by_dict()` and `select_where_sql()` select, and so does indexing: `sdaf[n]`, `sdaf[n:m]` and `sdaf[key]`. Each gives a SqlDaf selection that refers to rows of its base table, in order, as a Daf selection holds the same row lists. Nothing runs until the selection is first read; then its rows are found and persisted in a TEMP table of references, decisions 97, 112 and 113. A chain of selections not yet read runs as one query. Values changed in the base show through, and a deleted base row drops out. `select_record()` and `iloc()` read one row as a dict in one query, without making a selection. `to_dict()`, `to_list()` and `to_value()` read a small selection. `materialize()` and `save_as()` copy rows into a new table. A function in `select_where()` gets a KeyedList, as in Daf, and an error it raises is raised as itself. `select_by_dict()` compares with SQL's IS, so NULL matches a missing value and the results match `Daf.select_by_dict()` in the tests. A selection needs a reference that survives VACUUM: `daf_rid`, or a unique keyfield. On 1,000,000 rows: a selection of 499,892 persisted in 0.28 s, a chain of two unread selections counted in 0.10 s, `select_where()` with a Python function on one column in 1.4 s, `select_record()` 0.2 ms, `iloc(500000)` 17 ms, `sdaf[100000:100100].to_daf()` 4.5 ms, all at a peak of 40 MB. The tests compare SqlDaf with Daf on the same rows.
- `sqldaf.py`: positions are lookups. In a base table with `daf_rid`, and in a persisted selection, `iloc()`, `sdaf[n]` and slices look up the row number instead of stepping over the rows before it, and `len()` of a selection reads the highest number of its reference table. On 1,000,000 rows, `iloc(500000)` took 0.16 ms instead of 17 ms, `iloc(-1)` 0.28 ms, and a second `len()` of a selection of 499,892 rows 0.01 ms instead of 116 ms. This relies on not deleting rows from a base table that has selections, or that is read by position, which the docstrings say. After such a delete, `len()` of a selection still counts the row, while `to_daf()` leaves it out. Ray decided this on 2026-10-14.
- `sqldaf.py` step 3: work by key. `select_records_daf()` takes a list, an iterable, an in-memory Daf, another SqlDaf of the store, or `sdaf.keys()`, and is a join in SQL: a semi-join, or with `inverse=True` an anti-join. Rows come in the order of the keys, a missing key raises KeyError unless `silent_error=True`, and the tests match `Daf.select_records_daf()`. `keys()` gives a SqlKeys, read only when iterated. `delete_records(keys, inplace=False)` matches the new `Daf.delete_records()`: by default a selection without the rows; in place on a selection, its references are removed and renumbered; in place on a base table, the rows are deleted, and the docstring warns that earlier selections' `len()` and positions are then stale. `concat()` adds the rows of a Daf or a SqlDaf of the same store, with the same columns, as `Daf.concat()` requires; on a selection it first becomes a TEMP table of its own; a repeated key raises. On 1,000,000 rows against 500,000 keys: a semi-join 0.58 s, an anti-join 0.65 s, `concat()` of 500,000 rows 0.36 s, and `delete_records(inplace=True)` on the base 1.3 s.
- `sqldaf.py`: `join()`, lazy, as `Daf.join()` does it in memory: the same translator names the columns, rows come in this table's order with the other's unmatched rows after for `right` and `outer`, and an unmatched cell is NULL. It is resolved at first read into a TEMP table of new rows, as `Daf.join()` makes new rows. A row found only in the other table takes its key and shared fields from it, which `Daf.join()` does not yet do (see the xfail tests). SQLite 3.37 has no RIGHT or FULL JOIN, so those are a LEFT JOIN and the other's unmatched rows. `fix()` is renamed `persist()`, with `is_persisted`, as Spark and Dask use the word; Ray decided this on 2026-10-14.
- `sqldaf.py`: no more SqlStore, decisions Q122 to Q135 of 2026-10-14. A SqlDaf is made by its constructor, as a Daf is: `SqlDaf(name, cols=, lol=, keyfield=, dtypes=)` makes a table, and `SqlDaf(name)` opens one. The database is the file of the class setting `SqlDaf.db_path`, set once per job, or a `db_path` given to the constructor; with neither, it raises ValueError. Every SqlDaf on one file shares one connection, which the class keeps open by file and thread, Q136; so tables opened separately join and share children, and they share the file's transactions. `from_csv()`, `from_daf()` and `table_exists()` are class methods. The keyfield and dtypes of each table are kept in a table `daf_tables` in the same file, in the same transaction as the change, so a table reopens as it was. A selection is a child and a table a parent: `is_child`, and `parent_sdaf`, which was `base_sdaf`. `copy()` replaces `materialize()`, `copy(inplace=True)` makes a child a table of its own, and `save(name='')` replaces `save_as()`, keeping a TEMP table in the file under its own name or a new one. `persist()` is internal now. `SqlDaf.close(db_path='')`, a class method, closes a file for every SqlDaf on it, Q138, and `delete()` deletes a table and its record. `keys()` returns a list, as `Daf.keys()` does, and SqlKeys is gone. SqlNotSupportedError is gone: a method with no SQL version raises NotImplementedError. Another thread has its own connection, and a child or TEMP table of it cannot be joined, which raises NotImplementedError. The find_variants port in notes/sql_backing is updated, and gives the results of the first trial.
- `Daf.copy(inplace=True)` copies into the Daf itself and returns it, so a selection no longer shares its rows. Its default level is `deep`; `editable` may be asked for, and a level that would share the rows raises ValueError. Ray decided this on 2026-10-14, Q129. Tests are in `tests/test_daf_copy_levels.py`.
- `Daf.from_ods()` reads one sheet of an OpenDocument spreadsheet, an .ods file, as from LibreOffice Calc. The first sheet is read unless `sheet` gives another, by position or by name. Only the content of the cells is read: styles, comments, links and formulas are left out. It needs no package, since `lib/daf_ods.py` reads the file with `zipfile` and `xml.etree.ElementTree`. Values are text, as from a CSV file. A number, date or boolean cell gives its stored value, such as `168.5`, not the text shown, such as `$168.50`. Runs of spaces, line breaks, merged cells and repeated cells and rows are read as LibreOffice means them. Rows are padded to the widest row, and a blank row inside the table is kept as a row of NULL. A PayeeLookup.ods of 870 rows took about 70 ms. Ray decided this on 2026-10-12. Tests are in `tests/test_daf_from_ods.py`, with files made by `tests/make_ods_fixtures.py`.
- `split_where()` is deprecated in its docstring, on Daf and on SqlDaf. Use `group_where()` with a function that returns True or False. Both still work as before, and AuditEngine's calls are not changed. Ray decided this on 2026-10-15, Q161. `split_by_dict()`, added earlier the same day, is removed, since it does not map well onto SqlDaf; it was never released. `SqlDaf.split_where(where, cols=)` is two calls of `select_where()`, one with the test negated, Q145 A. Neither side runs until it is read, and the function runs again for each side read.
- The type alias for a list of SqlDaf is `T_losdaf`, and a variable ends in `_losdaf`. It was `T_lsdaf`. A container of containers puts o between them, as in `T_dosdaf`. Ray decided this on 2026-10-15, Q160.
- SqlDaf gets `len()`, `to_csv_file()` and `to_csv_buff()`, so code written for a Daf runs on it. `len()` is `len(sdaf)`. `to_csv_file()` writes a local file as it reads the rows, 10,000 at a time, so a table larger than memory can be written; an s3 path is built in memory and uploaded, as a Daf does. The text is the same as `Daf.to_csv_buff()` writes for the same rows, as the tests check. Ray, 2026-10-15, Q151 and Q159.
- `SqlDaf.apply(func)` calls a Python function on each row, as `Daf.apply()` does, with its rules: a dict adds a row, a list of dicts adds several, None adds nothing, and the columns come from the first row returned. The rows are read 10,000 at a time and the results written as they come, in one transaction, so memory does not grow; an error in the function is raised as itself and nothing is kept. The result is a TEMP table, kept with `save()`. `by='table'` passes the SqlDaf to the function; `keylist` raises NotImplementedError. On 300,000 rows that each became 4, it took 23.1 s at a flat 122 MB. A SqlDaf with no columns can now be made and read, as an empty table, which `apply()` returns when the function returns nothing. Ray decided this on 2026-10-15, Q150 A. The tests compare it with `Daf.apply()`.
- SqlDaf gets `groupby()`, `groupby_cols()` and `group_where()`, each returning a KeyedList of selections, as Daf now returns a KeyedList of Daf. `groupby()` finds the values in one query, in the order first seen, and each group is `select_by_dict()` of its value, so NULL is a value like any other. `group_where()` calls the function once per row and writes each (key, row) pair to a TEMP table, so a row can be in any number of groups; each group selects the rows under its key. Nothing more is read until a group is read. `cols=` of `groupby()` and `indirect_col=` of `group_where()` raise NotImplementedError. On 100,000 rows: `groupby()` of 50 values took 0.044 s and `len()` of every group 0.36 s; `group_where()` with 40 keys per row, 160 groups, took 8.1 s, against 2.5 s for Daf, and `len()` of every group 2.8 s. The tests compare the groups with Daf's. Ray, 2026-10-15, Q152.
- SqlDaf counts and sums in SQLite, with the results of the Daf methods: `valuecounts_for_colname()`, `valuecounts_for_colnames_ls()`, the two `_selectedby_colname` forms, `valuecounts_for_colname1_groupedby_colname2()`, `value_counts_daf()`, `daf_valuecount()`, `sum()` and `groupsum_daf()`. Each is one GROUP BY or one pass, so no column is read into Python. Values are in the order first seen, as a Daf gives them, and NULL is counted as NULL. `sum()` reads text cells into Python only when a column has them, so text that is not a number raises ValueError, as in Daf. `groupsum_daf()` adds only numbers, as `sum_da()` does. On 1,000,000 rows, against a Daf in memory: value counts of one column 0.48 s and 0.19 s, counts of one column grouped by another 0.77 s and 0.17 s, `sum()` of two columns 0.31 s and 0.18 s, `groupsum_daf()` 0.53 s and 1.37 s. The tests compare each with Daf, on a table and on a selection. Ray, 2026-10-15, Q143.
- SqlDaf gets `select_cols()` and `reduce()`, with the results of the Daf methods, under Q159. `select_cols(cols, exclude_cols)` gives new rows with only those columns, in the order given, as a TEMP table made when first read, as a join is; the keyfield and dtypes carry over for the columns kept. `reduce(func, by=, cols=, initial_da=, indirect_col=, silent_error=)` calls the function in Python on each row, read 10,000 at a time, so memory does not grow. `by='row'` gives each row as a KeyedList, `by='sparse_row'` gives the dict in `indirect_col`, JSON or PYON text decoded as in a Daf, and `by='table'` gives the SqlDaf; `by='col'` raises NotImplementedError. On 300,000 rows: `select_cols()` of 2 of 4 columns 0.36 s, `reduce()` by row of one column 0.80 s, and by sparse row of a dict of 30 keys 7.5 s, most of it decoding the text. The find_variants port uses them for the steps after step 11 and the stats of cmpcvr.py. The tests compare each with Daf.
- SqlDaf treats the text '' in a cell as Python NULL, as SQL NULL already is, wherever it groups, counts or selects by value: the value counts, `groupby()`, `groupsum_daf()` and `select_by_dict()`. SqlDaf writes Python NULL as SQL NULL, but a table written by other code may hold ''. Before, the two were separate values, and in a dict of counts one count replaced the other: in AuditEngine's San Francisco tables, party counted 2,216 empty values, where there are 34,469. A Python NULL in `select_by_dict()` is now `(col IS NULL OR col = '')`, which can use an index. Keys are not changed: the SqlDaf docstring says to avoid a Python NULL key, since SqlDaf's results with one are undefined, Q163. Ray decided both on 2026-10-15, Q162 and Q163. A Python NULL key, SQL NULL or '', now raises ValueError when the key index is made, which every key method does first, and on `concat()` of a row with one; the table is left as it was. The check uses the new index: on 1,000,000 rows the first `ensure_key_index()` took 0.415 s, against 0.399 s before. Ray, 2026-10-15, Q164. The tests store both forms and compare each method with Daf.
- A SqlDaf parent that has had rows deleted can no longer be read by position. `iloc()`, `iloc_exists()`, `sdaf[n]` and `sdaf[n:m]` on it raise ValueError, which says to read positions from `copy()` or to select the rows wanted. Before, a position was looked up as daf_rid n+1, so after a delete it gave a wrong row or a false IndexError. daf_rid stays as it is, as the reference from a selection to its parent's rows. A gap is found by comparing MAX(daf_rid) with COUNT(*), once per table on a connection; a table without one is remembered. `delete_records()` in place on a parent, and a statement through `sql()` that changes rows, make it check again. Selections number their own rows and are not affected, and neither are tables without daf_rid, which count by OFFSET. On 1,000,000 rows the first read by position took 65 ms, against 0.3 ms before, and later reads 29.4 us, as before. `copy()` of that table took about 0.4 s. Ray decided this on 2026-10-16, Q170. Tests are in `tests/test_sqldaf.py`.
- The SqlDaf docstring names one more danger of deleting from a parent that has children: SQLite gives the daf_rid of a deleted last row to the next row added, so a child made before the delete shows that new row. Documented, not fixed, Q169.
- SqlDaf reads rows and columns as a Daf does, step 1 of the port plan, Q173. Looping: `for row in sdaf`, `iter_dict()`, `iter_klist()` and `iter_list()`, read in batches, with rows as copies. Columns: `col()`, `col_to_la()`, `icol()`, `icol_to_la()` and `cols_to_dol()`, which read only their columns. Size: `num_rows()` and `shape()`. Keys: `key in sdaf`, `get_existing_keys()` and `select_krows()`, where a tuple selects a range of keys by their place among the rows, not by row positions, so it works after a delete. A slice raises NotImplementedError, since in a Daf it means positions. `select_first_row_by_dict()`. Exports: `to_lod()`, `to_dod()`, `to_json()`, `to_md()` and `to_md_cols()`. With `max_rows`, `to_md()` reads only the first and last rows. None of them turns a key into a row position. The results match Daf in the tests, also on a table with a deleted row. On 1,000,000 rows of 6 columns: a loop over dicts 2.4 s, against 0.7 s for a Daf and 1.3 s for sqlite3 alone to give the tuples; `col()` 0.5 s; `to_md(max_rows=10)` 0.02 s. The class docstring lists the methods by what they do. Tests are in `tests/test_sqldaf.py`.
- SqlDaf writes rows and values as a Daf does, step 2 of the port plan, Q173. Adding rows: `append()` in all its forms, `record_append()`, `extend()` and `assign_record()`. With `respect_kd=True` a row whose key is there replaces that row, in its place, as an upsert. `concat(respect_kd=True)` now works too. Changing values: `sdaf[key, col] = v`, `sdaf[key] = row`, `sdaf[:, col] = values`, `update_by_keylist()`, `assign_col()`, which adds a new column with ALTER TABLE, `apply_to_col()` and `apply_in_place()`. And `calc_cols()`, which reads only the names. A row is found by its key, never by its position, and a column list goes to the rows by their order. An int row in `sdaf[...] = ` raises NotImplementedError. On a selection, a change goes to the parent's rows, as a Daf selection shares them, and an append gives the selection rows of its own, as `concat()` does. Each call is one transaction, so a failing row or function changes nothing. A repeated key raises sqlite3.IntegrityError, as `concat()` does, and a Python NULL key raises ValueError. The results match Daf in the tests, also after a delete. On 1,000,000 rows: `apply_to_col()` 1.1 s, against 0.24 s for a Daf; `apply_in_place()` changing one column 6.1 s, against 1.6 s; `assign_col()` of a list 2.7 s, against 0.12 s; `sdaf[key, col] = v` 55 us; `extend()` of 10,000 dicts 0.07 s. One `append()` of a dict took 9.7 ms on its own, since each commit writes to disk, and 55 us inside `transaction()`. Tests are in `tests/test_sqldaf.py`.
- SqlDaf indexing reads a list of keys and rows with columns, as a Daf does, under Q159. `sdaf[key_list]` gives the rows in the order of the list, as `select_records_daf()` does. `sdaf[rows, cols]` takes any rows form, or `:`, and a column name, a list of names or `:`, and gives new rows of those columns, as `select_cols()` does. `sdaf[key, col].to_value()` reads one value. These are the forms AuditEngine uses on full_bif, found in the review of 2026-10-16. One value read this way took 1.2 ms on 200,000 rows, against 38 us for `select_record()`, and the docstring says so. A list of positions raises NotImplementedError. Tests are in `tests/test_sqldaf.py`.
- SqlDaf changes its columns as a Daf does, step 3 of the port plan, Q173. `insert_col()` adds a column at the end, since ALTER TABLE adds columns only there, and another position raises NotImplementedError. `drop_cols()` is ALTER TABLE DROP COLUMN, and a dropped keyfield is cleared with its index. `rename_cols()` and `set_cols()` are ALTER TABLE RENAME COLUMN, through temporary names so a swap works, and the keyfield follows. Two columns with one name raise sqlite3.OperationalError, where a Daf keeps one. `set_dtypes()` and `attach_schema()` set the dtypes, and the keyfield for a schema, without converting. `clone_empty()` makes a TEMP table of the same layout. Each change is one transaction and updates the table's record. On a selection, whose columns are its parent's, these raise NotImplementedError, except that a column already there is filled, as in `assign_col()`. A selection made before a drop or rename raises sqlite3.OperationalError when read. `remove_dups()` gives two selections, the last row of each key and the earlier ones, as Daf gives them, with SQL NULL and '' as one key. A new column of Python NULL skips the UPDATE, since ALTER TABLE fills it already: on 1,000,000 rows it took 17 ms, against 1.9 s. On 1,000,000 rows of 6 columns: `rename_cols()` 15 ms, `drop_cols()` of one column 0.73 s, against 0.30 s for a Daf, and `remove_dups()` of 500 keys 1.5 s, against 0.23 s. The sorts wait for a decision, Q178, since a Daf sorts in place. The results match Daf in the tests, also after a delete. Tests are in `tests/test_sqldaf.py`.
- SqlDaf `groupby()` and `groupby_cols()` make an index on the columns they group by, in the parent table, unless an index already starts with those columns. Each group is a selection by its value, so before, each group read the whole table. On AuditEngine's CVR grouped-votes pattern, 100,000 rows in 500 groups, summing every group took 14.8 s before and 2.3 s after, against 0.73 s for a Daf. Making the index took about 0.1 s. The index is named daf_gx_ plus the table and columns, stays with the table, and is dropped by `drop_cols()` or `rename_cols()` of its columns. Ray decided this on 2026-10-16, Q176. Tests are in `tests/test_sqldaf.py`.
- SqlDaf sorts in place, as a Daf does: `sort_by_colname()` and `sort_by_colnames()`, with `reverse`, `length_priority` and `as_str`. The rows are copied in sorted order into a TEMP table, and the SqlDaf becomes it, as `copy(inplace=True)` does, in one statement. So the parent of a selection, and a permanent table in the file, keep their order, and their other selections stay as they were. Ties keep their order. SQL NULL and '' sort first, as Python NULL does in a Daf. A column that mixes numbers and text sorts the numbers first, where a Daf raises TypeError. On 1,000,000 rows of 6 columns, a sort by one text column took 2.4 s, against 1.0 s for a Daf, and a selection of a seventh of them 0.37 s. Also `apply_dtypes()` on a selection copies it in place first and converts the copy, where before it raised NotImplementedError. The parent is not changed, unlike a Daf selection, which shares its rows. Ray decided both on 2026-10-16, Q178 and Q175. The tests compare the sorts with Daf, also after a delete.

### Tests
- 24 small test files are merged into the files for their areas, each under a heading that names the file it came from. tests/ goes from 58 files to 34. The tests are unchanged, and the count is still 3954. Four helpers were renamed where two files used the same name, and a helper defined twice in test_daf_key_cells.py is defined once. CLAUDE.md now says to add tests to the existing file for their area. Ray, 2026-10-15, Q149.
- Two bugs in `join()` are recorded as strict xfail tests in `tests/test_daf_join_other_only_rows.py`, not fixed. In a right or outer join, a row found only in the other Daf gets NULL for its key when the two key columns have different names, since `join_records()` looks the key up by the first Daf's column name. It also loses a shared field, which is taken only from the first Daf. Found on 2026-10-14 while matching `SqlDaf.join()` to `Daf.join()`.

### Fixed
- The docstrings of `SqlDaf.to_csv_file()` and `SqlDaf.to_csv_buff()` held a line break where the line ending `\r\n` was meant. The docs build then showed each one as plain text, with no parameter table. `SqlDaf.iloc_exists()` gets its Args and Returns.
- Selecting by key with `inverse=True` was quadratic: `select_krows()`, `select_records_daf()`, and the default path of `delete_records()`. `gkeys_to_idxs()` tested each row position against a list of the selected positions, which scanned the list each time. It now uses a set. The rows, their order and the errors are the same. With half of the keys selected, 32,000 rows took 3.7 ms instead of 3.6 s, and 1,000,000 rows 1.0 s instead of an estimated hour. AuditEngine has 4 such calls, in cmpcvr.py and gentemplates.py, which can only get faster. Found while timing `delete_records()`. Ray decided this on 2026-10-14, item 3 of the mdaf review. Tests are in `tests/test_daf_inverse_key_selection.py`.
- Printing a Daf needed numpy, though numpy is optional. The display transposes the rows, and `transpose_lol()` did that with numpy. So after a plain `pip install daffodil`, `print(daf)` raised ModuleNotFoundError. `transpose()` needed numpy too.
- A Daf whose cells were all lists of the same length could not be printed. `np.array()` looked inside the lists, so the display raised RuntimeError and `transpose()` moved the items of the lists as well. The numpy path now fills its array with `np.fromiter()`, one reference for each cell, so such a table prints and transposes correctly at any size. An object array from `np.fromiter()` needs numpy 1.23 or later. Ray decided this on 2026-10-12, as item 16 of the review.
- `transpose()` raises ValueError when a row is not as long as the columns, and the message names `is_rectangular()` and `force_rectangular()`. It used to give a wrong result with no error: a short row lost a column or raised ValueError from numpy, and the cells of a long row past the columns were lost. Before 2026-10-12 it returned such rows untransposed. The check takes about 9 ms on 100,000 rows, under 2% of the transpose. Ray decided this on 2026-10-12.
- Without numpy, `json_encode()` of a value that JSON cannot encode, such as a set, raised ModuleNotFoundError instead of TypeError. `NpEncoder` imported numpy for every such value. It now imports numpy only when the type of the value comes from numpy, which means numpy is already loaded. The results with numpy are the same. The idea came from AuditEngine. Ray decided this on 2026-10-12. Tests are in `tests/test_json_encode_numpy.py`.
- `reduce()` with `sum_da()` or `sum_kl()`, and so `daf_sum()`, adds down each column instead of across each row, when no other option is given. A whole column is added in C with `functools.reduce()`. A column with a cell that cannot be added is added again one cell at a time, and text is skipped before trying. The totals are the same, also for floats: the builtin `sum()` is not used, since from Python 3.12 it adds floats with compensated summation. On 100,000 rows, `daf_sum()` of two numeric columns took 0.015 s instead of 0.078 s, of four columns with text mixed in 0.034 s instead of 0.173 s, and of all six columns 0.043 s instead of 0.289 s. With `initial_da`, `astype`, `is_sparse` or `silent_error`, the rows are reduced as before. Ray decided this on 2026-10-12, as item 10 of the review.
- `groupby_cols_reduce()` with `sum_da()` or `sum_kl()` adds up all groups in one pass, with no Daf for each group, when `reduce_cols` are columns that are not grouped. The result is the same Daf. On 100,000 rows in 43,242 groups, summing two columns took 0.174 s instead of 0.767 s, and with two group columns 0.241 s instead of 1.293 s. Other cases take the old path. Tests for both changes are in `tests/test_daf_sum_by_col.py`.
- `daf['name'] = value` with a name that is not a row key raises KeyError with a hint: `daf[name]` selects a row, and a column is `daf[:, 'name']`, or `assign_col('name', values)` to add one. Without a keyfield, the KeysDisabledError gets the same hint. Someone used to pandas often means a column there, and the plain `KeyError: 'name'` did not say that the name was read as a row key. Only the error path is changed. Ray decided this on 2026-10-12, as item 11 of the review. Tests are in `tests/test_daf_setitem_row_key_hint.py`.
- A method that rewrites a keyfield column invalidates the key index (kd), so a lookup finds the new key, and not the old one: `set_icol()`, `set_icol_irows()`, `set_col_irows()`, `assign_icol()`, `assign_col()` also for a composite keyfield, `apply_dtypes()`, `flatten()` and `strip()`. `find_replace()`, `apply_in_place()` and `apply_formulas()` already did. Each tests once per call whether the column is a keyfield column. A write with `__setitem__`, `set_irows_icols()`, `assign_record_irow()` or `update_record_irow()` does not touch the kd, as before, so those writes stay fast. Don't change keys with them, or call `set_keyfield()` with the same keyfield afterwards. The `__setitem__` docstring said the index was rebuilt after a key cell was changed. It now says it is not, and the two record methods point to it. NULL is a valid key, and rows with NULL keys can be written and added as before. Ray decided this on 2026-10-12, narrowing item 19 of the review. Tests are in `tests/test_daf_key_cells.py`.

### Changed
- `reduce(by='sparse_row')` and the other readers of an indirect column, through `get_indirect_da()`, read the cell text with `unflatten_val()`, which tries pyjson5 first. Before, `safe_convert_json_to_obj()` tried json.loads, then swapped every single quote for a double quote and every None for null, and tried again. That swap failed on a key with an apostrophe, and the cell stayed text. It also turned a value such as 'Nonesuch' into 'nullsuch'. A trailing comma also left the cell as text. An empty cell still gives an empty dict. On 100,000 rows of 30 votes, summing the votes took 1.29 s instead of 2.40 s for a Daf, and 1.43 s instead of 2.58 s for a SqlDaf, with the same sums. Ray decided this on 2026-10-16, Q183. Tests are in `tests/test_daf_utils.py`.
- `groupby()`, `groupby_cols()` and `group_where()` return a KeyedList of Daf, a klodaf, type `T_klodaf`, in place of a dict of Daf. It works as the dict did: `groups[key]`, `keys()`, `items()`, `values()`, `get()`, `len()` and `in`. Its values also have positions, as `groups.values()[0]`, and `to_dict()` gives the dict. The groups are the same Daf instances, sharing rows as before. A KeyedList is not a dict, so `isinstance(groups, dict)` is False, and `groups == {}` is False even when empty; use `len()`. No call in AuditEngine checks either, of 43 that keep a grouping. Ray decided this on 2026-10-15, Q154 and Q155.
- `select_records_daf()` accepts another Daf as the keys, and uses its keys through a view of its key index, so no list of them is made. The rows come in that Daf's row order, as with a list of keys. A Daf with no keyfield raises KeysDisabledError. Before, a Daf passed there raised TypeError, since iterating a Daf gives row dicts. On 1,000,000 rows and 500,000 keys it took 350 ms, against 365 ms for `select_records_daf(other.keys())`; the saving is the 20 ms of building the list. The main use is code shared with SqlDaf, where the same call is a semi-join and no keys come into Python. Ray decided this on 2026-10-14, decision 102 and item 1 of the mdaf review. Tests are in `tests/test_daf_select_records_daf_keys.py`.
- The docstrings of `Daf.from_csv_buff()` and `Daf.from_csv()` say that converting with `dtypes` while reading is deprecated. It does no more than `apply_dtypes()` on the result, and a reader should only read. It still works, and nothing warns. The `from_csv_buff()` example now calls `apply_dtypes()`. Ray decided this on 2026-10-14. Docs only.
- `groupby()` and `multi_groupby()` share the rows with the source Daf, as the README has always described. They used to copy every row, while `groupby_cols()` and `group_where()` shared them. A cell changed in a group is now also changed in the source. To keep the source as it is, group a copy made with `copy(level='editable')`. Adding a column to a group, as with `insert_col()`, makes new rows, so the source keeps its shape. With `cols`, or `colnames` for `multi_groupby()`, the rows are new lists of the columns kept, as before. On 100,000 rows of 11 columns, `groupby()` took 0.064 s instead of 0.169 s with 5,000 groups, and 0.309 s instead of 0.459 s with 43,000. The docstrings and the sharing table in the Guide say so. On 2026-10-03 the docstring had been changed to describe the copy; that was the wrong fix. Ray decided this on 2026-10-13. AuditEngine was asked to check its callers, since evalvotes.py may depend on the copy. Tests are in `tests/test_daf_group_scope.py` and `tests/test_daf_misc.py`.
- The new `transpose_rows_lol()` in `lib/daf_utils.py` does every transpose: `transpose_lol()`, `Daf.transpose()` and the display. Below 100,000 cells it uses zip, which is 2 to 4 times faster on a small table. From that size it uses numpy with object references, which is up to 1.7 times faster at a million cells. Without numpy it uses zip at any size. The cells are not copied either way. `transpose_lol()` still raises RuntimeError for ragged rows. Ray decided this on 2026-10-12. Tests are in `tests/test_transpose_zip_numpy.py`.
- The `apply()` docstring says what the row function may return: a dict adds one row, a list of dicts adds several, and None or an empty dict or list adds nothing. It says the source is not changed, and that the columns come from the first row returned, so a later key the first row lacks is dropped without an error. A new doctest shows one row becoming two. Docs only.

## [0.7.2] - (2026-10-11)
### Fixed
- `import daffodil` no longer needs pandas or numpy. Both are optional in pyproject.toml, but `lib/daf_types.py`, `lib/daf_utils.py` and `lib/daf_pandas.py` imported them at the top, so a plain `pip install daffodil` failed with ModuleNotFoundError. They are now imported inside the functions that use them, and the type aliases name them only for type checkers. The import takes 0.09 s instead of 1.08 s, which shortens a Lambda cold start.
- `reduce()` with `count_values_da()` failed with AttributeError on a column of dicts. `reduce()` starts each count at 0, and `count_values_da()` then added a dict to that 0. A count that is 0 or empty now starts the dict.
- In val mode, `daf[row, col]` with an out-of-range row and a column name that does not exist raised IndexError. The general path raises KeyError for the column, and now so does the fast path. Both check the key, then the column, then the row.

### Changed
- Writing one cell, as in `daf[key, col] = v` or `daf[i, col] = v`, skips the general selector parsing when the value is an int, float, str or bool. Any other value takes the general path as before. On a table of 10,000 rows, a write by key went from 4.69 to 0.41 us.
- The single-cell reads and writes look up a key in the key index directly. A key that is not there still goes through `krows_to_irows()`, so the errors are unchanged. `daf[key, col] += x` in val mode went from 6.80 to 0.86 us, and a read by key from 1.88 to 0.43 us.
- `iter_dict()`, `iter_klist()` and `iter_list()`, and so `for row in daf`, return generators instead of a `DafIterator`. On 100,000 rows a loop took 0.049 s instead of 0.093 s with dicts, 0.053 s instead of 0.110 s with KeyedLists, and 0.005 s instead of 0.050 s with lists. The rows and the errors are the same. Two uses differ, and now behave as a loop over a Python list does. An iterator that was used up gives nothing when looped again; `DafIterator` started over. A loop goes on through the rows it started with when a new list of rows is put in place, as by a sort; `DafIterator` jumped into the new list. `DafIterator` stays, but daffodil no longer uses it. Ray decided this on 2026-10-11. Tests are in `tests/test_daf_iter_generators.py`.
- `groupby()`, `groupby_cols()` and `multi_groupby()` make each group as a copy of one template, with its own rows. With all columns the template is the Daf itself, so each group is what `clone_empty()` made before. With `cols`, the template is made once with `clone_empty(cols=...)`, so the cut-down layout is worked out once, not once per group. The groups are the same as before. On 100,000 rows in 43,242 groups, `groupby_cols()` with `cols` took 0.602 s instead of 0.747 s, and `groupby_cols_reduce()` 1.123 s instead of 1.313 s. With all columns, or with 5,000 groups, the change is within the noise. Most of the gain for many groups came earlier, from the faster `clone_empty()` (1b5da8f) and `copy()` (a2b44d5). Ray decided this on 2026-10-11. Tests are in `tests/test_daf_group_template.py`.
- `reduce()` with `by='row'` gives the function a new KeyedList for each row, instead of a dict. The KeyedList points at the row of the table, so no values are copied. It reads like a dict, and `dict(row)` makes one. Writing to it changes the table. A row shorter or longer than the columns gets a KeyedList of a copy, padded with NULL or cut to fit. Ray decided on 2026-10-11 that KeyedList rows are the standard.
- `count_values_da()` and `count_values_kl()` raise TypeError for a cell that holds a list, and the message names the column. A list used to be copied for the first row and appended as one item after that, so `[1, 2]` and `[3]` gave `[1, 2, [3]]`. A list cannot be counted as a value. There is no check on the normal path: the error comes from Python, and daffodil adds the column name. Leave a column of lists out with `cols`. Ray decided this on 2026-10-11.
- The text of a `list` or `dict` cell, as in a CSV file, is read with pyjson5 when it is installed, and with `ast.literal_eval` otherwise. pyjson5 is about 18 times faster per cell. Loading a gradebook of 10,000 rows with a dict column took 0.211 s instead of 1.723 s, with the same result. pyjson5 is an optional extra, `daffodil[pyjson5]`, and is in the dev tools. Text that pyjson5 cannot read, such as `True`, `None`, a tuple or a dict with int keys, falls back to `ast.literal_eval`. Writing is unchanged: the CSV writer writes Python text with spaces, which is the fastest. Ray decided this on 2026-10-11.
- New tests in `tests/test_unflatten_pyjson5.py` check that a cell reads the same with and without pyjson5.
- The docstrings of `sum_kl()`, `sum_da()` and `daf_sum()` say that a value that cannot be added is skipped with no error, so a bad number is left out of a total, and that `check_schema()` finds such cells. The skipping stays, since a sum of all columns relies on it to pass over text columns. Ray decided this on 2026-10-11.
- The `to_csv_buff()` docstring says that a bool column from `apply_dtypes()` is written as 0 and 1, and a column of Python bools as `True` and `False`, and that both read back the same. Writing them all as 0 and 1 would make every save about 47% slower. Ray decided on 2026-10-11 to keep writing the cells as they are held.
- `notes/scripts/check_before_push.sh` now stops at a failing step. It used to print each step's result inside an `echo`, so a failure did not stop it, and it still ended with "All checks passed". Found on 2026-10-11. The pushed commit ef44f07 was checked again with the fixed script and passes on Python 3.10 to 3.13.
- New `Daf.check_schema()`. It raises ValueError for a cell that does not fit its dtype, a NULL cell in a required column, or a missing required column, and lists up to 10 of them with the total. The required columns come from a `Required` column in an attached schema Daf, such as one read from a Markdown table with `from_md()`. A value such as `yes`, `true` or `1` marks a column as required. Loading never calls it, so the fast path is unchanged. Call it when you want the check, for example while developing or in a test. On 100,000 rows of 5 columns it took 0.073 s. Ray decided on 2026-10-11 that the user turns it on by calling it. Tests are in `tests/test_daf_check_schema.py`.
- New `Daf.sum_kl()` and `Daf.count_values_kl()`, the standard reducers. They read a KeyedList row directly and give the same results as `sum_da()` and `count_values_da()`. `reduce()` uses them in place of the `_da` versions, so callers that pass `Daf.sum_da` gain too. `daf_sum()` and `daf_valuecount()` use them. On 100,000 rows, `reduce(Daf.sum_da)` on two columns went from 0.129 to 0.075 s, `daf_valuecount()` from 0.158 to 0.097 s, and `groupby_cols_reduce()` with 40 groups from 0.174 to 0.126 s.
- New tests in `tests/test_daf_reduce_klist.py` compare `reduce()` with the dict rows it used before.
- New tests in `tests/test_daf_setitem_cell_fast.py` compare the write fast path with the general path, for the table, the return value and the error.
- `main` now carries the next dev version, 0.7.2.dev0, between releases. An editable install then shows that it is not the PyPI release.

## [0.7.1] - (2026-10-10)
### Added
- A cheatsheet, `docsite/cheatsheet.md`: one Markdown file for people and for AI tools, with a skill header (`name`, `description`). It has a table for each task, made by `Daf.to_md()`, then rules and common mistakes. It is built from `notes/scripts/cheatsheet_content.py` by `notes/scripts/build_cheatsheet.py`. `tests/test_cheatsheet.py` runs every snippet and checks that the file is current. It is in the docs menu and linked from the README.
- `notes/scripts/transition.sh`, written by the AuditEngine thread: closes this thread's tmux session, updates claude under a lock shared with AuditEngine's transition script, and starts the next thread, which picks up the handoff. CLAUDE.md Part 3 says when to use it.
- An indexing reference: the forms of `daf[rows, cols]`, and what each kind of selector means for rows and for columns. It opens the cheatsheet and the Guide page Selecting and indexing, both generated from one list in `notes/scripts/cheatsheet_content.py`. A test checks both are current.
- The Daffodil logo, `docsite/images/daffodil_logo.png`, as the logo and favicon of the docs, on the Home page, and in the README, from the repository instead of a GitHub upload link.
- A Guide in the docs: Getting started, Building tables, Selecting and indexing, Columns, keys and schemas, Types and conversion, Apply, reduce and formulas, Joins, Markdown reports, and Coming from pandas. It holds what the README explained, checked against the code. Its examples are doctests, run with `pytest --doctest-glob='*.md' docsite/guide` in CI and in `notes/scripts/check_before_push.sh`. `conftest.py` lets those examples sit in code fences.

### Changed
- The README is short: what Daffodil is, how to install it, the data model diagram, a quick start, why to use it, and links to the docs. It was over 1,000 lines, with method names that no longer exist, such as `add_idx()`, `to_csv()` and `group_to_dodaf()`, and size claims that did not hold. The old diagram is replaced by the new one, as a PNG.
- The docs Home page has install, a quick start and links into the Guide.
- With `retmode='val'`, `daf[row]` and `daf[row, col]` read the row or the cell straight from the rows, with no Daf built on the way. This applies when the row is an int or a key, and the column is an int or a name. The results and errors are unchanged. On a 10-row table, `daf[3]` went from 11.9 to 0.5 µs and `daf[3, 'n']` from 26.6 to 0.7 µs. By key, `daf['r3']` went from 14.1 to 3.3 µs and `daf['r3', 'n']` from 28.9 to 3.3 µs. `tests/test_daf_getitem_val_fast.py` checks the fast path against the general path, on 7 tables and 27 selectors.
- Selections in the default obj mode cost about half as much. Three costs are gone. A `cast()` with a type such as `int | slice` built that type on every call, so those types are now quoted. `copy()` no longer deep-copies an empty `attrs`. And `copy()` makes its new object with `__new__` and the attribute dict, which is the shallow copy that `copy.copy()` makes, without its lookup of how to copy. A subclass's own `__copy__` is no longer called. On a 10-row table, `daf[3]` went from 10.5 to 5.1 µs, `daf[3, 'n']` from 25.6 to 14.0 µs, `daf[:, 'n']` from 30.6 to 20.5 µs and `daf.copy()` from 5.7 to 2.6 µs. The results are unchanged.
- CI no longer runs on a push to `main`, to save GitHub Actions minutes. It runs on a pull request, when the full deploy calls it, and by hand with `gh workflow run ci.yml`. Before a push to `main`, `notes/scripts/check_before_push.sh` runs the same checks locally: the tests and the doctests on Python 3.10 to 3.13, and the strict docs build.
- The docs deploy alone on a push from `main` to the new branch `docs_deploy`, as releases do on `full_deploy`. It checks that the commit is on `main`. `gh workflow run docs.yml` still works.

## [0.7.0] - (2026-10-09)
### Added
- `append(row, fast=True)` and `from_lod(lod, fast=True)`, for rows the caller builds in column order. Nothing is copied: with `append()`, a list, or the value list of a KeyedList, becomes the row itself. A KeyedList must share the Daf's `hd`. A dict or a list is checked fully if it is the first row of an empty Daf, and otherwise only for its number of values. A row that fails raises `ValueError`, with a message that says which row and what is wrong. The default stays `fast=False`, which checks every row fully and copies it. At 1,000 columns `append()` of a list went from 15 µs to 1.1 µs a row. `from_lod()` of 1,000 dicts of 1,000 keys went from 121 ms to 27 ms. See the details under Changed.
- A profiling mode, `daffodil/lib/daf_profile.py`. With `DAFFODIL_PROFILE=1`, or `daf_profile.start()` in code, it records for each table the line that created it, its largest size and the methods called on it, and for each method the calls, time and table sizes. Only outer calls are counted. When it is off, nothing is wrapped. The totals are Daf tables with a `stage` column, written at exit to one Markdown file per run. `python -m daffodil.lib.daf_profile combine DIR` adds up the files of the stages of a pipeline with `groupby_cols_reduce()`, and writes a report for all stages and for each stage. Call sites are module and line, so they match across runs. See the Profiling mode page of the docs. Tests in `tests/test_daf_profile.py`, and the whole suite passes with it on.

### Changed
- `default_record()` works out the defaults in column order once, and keeps them until the Daf's columns or schema are replaced. Each call copies them, and copies afresh a default that is a list, dict or set. For a schema of 32 fields, `default_record(astype=KeyedList)` went from 15.5 µs to 1.25 µs, and the dict form from 16.4 µs to 2.4 µs. A default changed on the schema class after the first call is not seen.
- `default_record()` follows the columns of the Daf, in their order, however they were set. A column gets the schema's default, or NULL if the schema has none. Schema fields that are not columns are left out. A Daf with columns but no schema now gets NULL in every column, instead of `AttributeError`. `default_record(astype=KeyedList)` returns a KeyedList that shares the Daf's `hd`, to fill by name in any order. The duplicate assignment of `default_record` in `daf.py` is removed.
- `append()` of a KeyedList that shares the Daf's `hd` skips the check of its columns, as its keys are the columns. Its values are still copied. At 1,000 columns, building and appending such a row took 24 µs, against 92 µs for a KeyedList with its own `hd`, and 1.7 µs with `fast=True`. Other rows are compared with a list of the column names that the Daf keeps until its columns change, built the first time it is needed.
- `append(fast=True)` checks rows this way. A KeyedList must share the Daf's `hd`, and then nothing else is checked. One that does not, even with the same keys, raises `ValueError`. A dict or a list is checked fully if it is the first row of an empty Daf, and otherwise only for its number of values. `from_lod(fast=True)` checks the first dict fully, and every dict for its number of keys. A row that fails raises `ValueError`, whose message names the Daf, if it has a name, and the row, says how its keys or length differ from the columns, and says how to fix it. Without `fast`, every row is checked fully, and a KeyedList with its own `hd` is placed by column name.
- `append()` with a list, on a Daf with columns, adds a copy of the list directly. Its docstring said the list itself was added; that held only for a Daf with no columns, and is corrected. It no longer builds a dict from the list and a list from the dict, and no longer compares the column names on every row. 86 µs a row became 14 µs for 1,000 columns, and 3.1 µs became 1.3 µs for 10 columns. An upsert, `respect_kd=True` with a keyfield, still goes through `record_append()`. Behavior is unchanged. `tests/test_daf_append_list.py` pins it, and passes on the old code too.
- The docs Home page has the data model diagram, an overview, and a comparison with pandas of time and memory to build a table row by row. The scripts are `notes/scripts/bench_build_rows.py` and `notes/scripts/mem_by_string_share.py`.
- Type annotations use `X | Y` in place of `Union[X, Y]`, and `X | None` in place of `Optional[X]`, in src, tests and the README. `Optional` means only that a value may be None, not that the argument is optional. No behavior changes. Python 3.10 is still the minimum.
- The `copy()` arguments `deep` and `for_sorting` are documented as deprecated. Use `level='deep'` and `level='sortable'`. They still work, and give no warning.
- The development status is now Alpha (classifier `Development Status :: 3 - Alpha`), not Pre-Alpha.
- The Daf API docs are split into an overview and 8 section pages, with subheadings, in place of one long page. `tests/test_docsite_coverage.py` checks that every public Daf member is on exactly one page.
- Docs styling in `docsite/stylesheets/extra.css`: compact tables with full cell borders, a tinted header and striped rows; sidebar text the same size as the content; a blue-grey and amber palette with a dark mode switch.
- The full deploy deploys the docs before PyPI, then waits for approval from a required reviewer on the environment `pypi`. PyPI, the tag and the GitHub Release follow the approval. The new workflow `docs.yml` deploys the docs alone, by hand, and is also called by the full deploy.
- The full deploy also creates a GitHub Release for the version, with its CHANGELOG section as the notes. The docs header shows the latest GitHub Release, which was still V0.5.13.

### Removed
- The class `KeyedIndex`. It was added on 2026-04-15 for a possible port to Rust, which is not planned. A KeyedList's `hd` is a plain dict again, of each key and its position. The rows that a Daf hands out, from `iloc()` and `iter_klist()`, share the Daf's own `hd`, so `row.hd is daf.hd`. The Daf no longer keeps a separate copy of its column names for them. A row still copies the shared `hd` before it adds or deletes a key, so the Daf and the other rows never change. A name lookup in a row is faster, as a dict lookup is about 47 ns against 118 ns for a KeyedIndex. `KeyedList(a_dict, values)` now shares the dict as the `hd`, so the dict must map each key to its position, as an `hd` does. Before, only the keys of the dict were used. `KeyedList(keys, values)` with a repeated key still raises `ValueError`.

### Fixed
- `append()` of a KeyedList, without `fast`, now adds a copy of its values. It added the KeyedList's own value list, so changing the KeyedList afterwards changed the table. Without `fast`, `append()` now never keeps the caller's object.
- Selecting rows took time in proportion to the size of the whole table. `clone_empty()`, which every selection uses, copied the full row list and then replaced it with the selected rows. A one-cell read such as `daf[i, 'a']` took 796 µs on 100,000 rows. It now takes about 30 µs at any size. Tests in `tests/test_daf_shared_rows.py`.

## [0.6.0] - (2026-10-06)
### Added
- `apply_dtypes(cols=)` converts only the named columns, at the owner's request on 2026-10-06. `cols` is a list of names, or one name. A column that is not in `cols` is not converted, is not checked, and keeps its entry in the `dtypes`
  of the Daf. A `dtypes` argument then only supplies and updates the entries of the columns in `cols`, where without `cols` it replaces the `dtypes` of the Daf. A name that is not a column raises `KeyError`.
  A column in `cols` with no dtype raises `ValueError`, or takes `default_type` with `silent_error=True`. Conversion costs about 0.04 s for each column of 200,000 rows, so converting only the columns that are used saves that time.
  Without `cols`, a `dtypes` that leaves out a column of the Daf still raises `ValueError` unless `silent_error=True`, and still replaces the `dtypes` of the Daf, which is what made the old pattern of saving the dtypes, keeping the
  columns to convert, converting, and restoring the dtypes necessary. That pattern still works and is shown in the docstring. 13 tests were added.
- `select_by_dict()` takes a list or tuple of dicts, at the owner's request on 2026-10-06. A row matches if it matches any one dict, so a dict is an AND of equalities and a list is an OR of them.
  `select_by_dict([{'name': n} for n in names])` selects the rows whose name is any of the names, and `[{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]` selects by two columns together.
  The dicts are grouped by their keys and the values of each group go in a set, so a row costs about one lookup. For 200,000 rows and 100 names it takes 0.009 s, where `select_where()` takes 0.15 s.
  Fifty dicts of three columns take 0.073 s. The result is what comparing with `==` gives. A selector value or a cell that cannot be hashed is compared with `==`, so it never raises an error, and
  a call with such a cell takes a slower path, 0.096 s in the same test. An empty list matches no row. An empty dict matches every row. A dict that names a missing column matches no row. An item that is not a dict raises `TypeError`.
  A single dict works as before, and a set or list given as a value in a dict is still an equality test. 20 tests were added. Not in 0.5.13.
- GitHub Actions workflows, at the request of the owner on 2026-10-05. `ci.yml` runs the tests, the doctests and the docs build on a push to `main` and on every pull request, and it never deploys.
  `deploy.yml` runs only on a push to the branch `full_deploy`. It checks that the commit is on `main`, that `pyproject.toml` and CHANGELOG.md agree on the version, and that the tag is new.
  It then tests, builds, publishes to PyPI, tags `v` and the version, and deploys the docs to GitHub Pages. The version in `pyproject.toml` is now 0.6.0.
  The workflow needs a secret `PYPI_API_TOKEN` in the environment `pypi`, and Pages set to deploy from Actions. `mkdocs.yml` has `site_url` and `repo_url`.
- `rows_are_shared()` in `daf_utils` tells whether any row is also held by another list, by reading the reference count
  of each row. Scanning 200,000 rows takes about 0.01 s. Copying them takes 0.75 s.
- `select_irows([], inverse=True)` keeps all rows and now shares them, like every other `select_irows()` result. It
  made a deep copy before. For 200,000 rows by 50 columns that took about 3 seconds.
- `sort_by_colname()` and `sort_by_colnames()` take `as_str`. With `True` they sort by the text of each value, and
  None sorts as an empty cell, so a column that mixes numbers with None, blanks or text can be sorted. With
  `length_priority` whole numbers sort in numeric order. Negative and decimal numbers sort in text order. A column
  that cannot be compared now raises `TypeError` that names the method and the column, and points to `as_str`. It
  raised Python's message. `length_priority` on real numbers raised `object of type 'int' has no len()`. The plain
  sort is as fast as before. 9 tests added.
- `Daf.from_directory(include_dirs=)`. With `True`, folders are listed as well as files. Each has an
  `is_dir` of 1, a size of 0 and no extension, and comes before the files in its folder. The default is
  unchanged, so no folder is listed and `is_dir` is 0. The `file_pat` now applies to folder names too.
- `Daf.copy(level=)` with the levels `shallow`, `sortable`, `editable` and `deep`. The default is
  `shallow`, as before. `sortable` gives the copy its own row list, `hd`, `dtypes` and key index,
  so rows can be appended, sorted or dropped, and columns renamed or dropped, without reaching the
  original. `editable` also copies every row, so columns can be added and cells changed. `deep` is
  as before. Text and numbers are not copied by `deep`, only lists, dicts and other containers.
  `deep=True`, `for_sorting=True` and `copy(True)` still work. `for_sorting` now also copies `hd`
  and `dtypes`. Costs for 200,000 rows by 50 columns: shallow 0.0001 s, sortable 0.002 s,
  editable 0.8 s and 93 MB, deep 2.4 s and 107 MB. 107 tests added in
  tests/test_daf_copy_levels.py. Total tests 1828.
- `groupby()` and `groupby_cols()` take `cols`. Each group then holds only those columns,
  and the other columns are never copied. `multi_groupby(colnames=)` was ignored and now
  does the same. A keyfield is kept only if all its columns are kept. The three reduce
  methods group only `reduce_cols` and the `indirect_col` for `by` of `row` or `sparse_row`.
  Results are unchanged on 47 before and after scenarios. A function that reads another
  column of the row will not find it. Timing for 2,000 rows by 1,000 columns, seconds:
  `groupby_reduce()` with 3 reduce columns 0.252 to 0.005, `multi_groupby()` with 3 columns
  0.129 to 0.007. Tests added in tests/test_daf_group_scope.py. Total coverage 96%.
- `Daf.concat()` and `Daf.extend()` take `respect_kd`, as `append()` already did. With
  `respect_kd=True` a row whose key exists is replaced and the other rows are appended.
  This gives a bulk upsert, and composite keys work. The default is unchanged. Tests added
  for overlap, composite keys, empty input and a keyfield mismatch.
- `Daf.is_rectangular()` / `Daf.force_rectangular()` -- verify and, if needed, fix a `Daf`
  whose rows don't all have the same length. `is_rectangular()` checks every row (against
  `len(hd)` if columns are defined, else against each other), unlike `num_cols()`, which only
  samples the first 10 rows and assumes consistency. `force_rectangular()` pads any short row
  with empty strings up to the target width, in a single pass directly over `self.lol` --
  written to replace the CSV-text round trip in `daf_utils.add_trailing_columns_csv()` (its own
  `@@TODO` already flagged this), which only samples the first few rows to guess the target
  width and can under-pad a file whose later rows are genuinely wider. A row that's *longer*
  than the target raises `ValueError` rather than being padded or silently left as-is --
  that's a different problem (e.g. an unquoted comma in a hand-edited CSV field splitting it
  into extra columns), and this method's own name promises a rectangular result, so returning
  a still-non-rectangular `Daf` with no signal at all would be worse than the loud failure.
  Never truncates a long row either, which would hide real data corruption rather than surface
  it. Motivated by a real incident in AuditEngine (`CA_SanFrancisco_20241105_pilot_d05`): a hand-edited
  `LATER_EIF` row's unquoted comma shifted every later column by one, and nothing in the load
  path noticed.
- `js/daffodil-csv.js` -- browser-side, dependency-free (beyond vendored JSON5) reader for
  `Daf.to_csv_buff()`-produced CSV: parses CSV (RFC4180, handles quoted/embedded commas, quotes,
  newlines) into plain JS objects, unflattening PYON-shaped cells (`{...}`/`[...]`/`(...)`) the
  same shape-driven way `daf_utils.unflatten_val()` does on the Python side. `vendor/json5.js` is
  the unmodified upstream `json5` v2.2.3 package (MIT); `pyonKeywordsToJson5()` handles the one gap
  between PYON and JSON5 syntax (Python's capitalized `True`/`False`/`None`), quote-aware so a
  string value that legitimately contains those words is left alone. Verified against a real
  `Daf.to_csv_buff()` round trip (dict/list/tuple columns, nested quotes, empty values). See
  `js/README.md`. Built for AuditEngine's ACRE standalone review app (a fully static, backend-free
  reviewer tool), which needed to consume daffodil-exported ballot data in a browser with no build
  step.
- Daf.dodaf_to_md(dodaf, report_header='') / Daf.dodaf_from_md(md_str, header_level=2) -- render
  a dict-of-Daf ("dodaf") as one combined markdown report (optional top-level report header, one
  `##` section per key using that key's own to_md()), and parse it back. dodaf_from_md() skips any
  report-level material before the first section header, splits on `##` boundaries, and parses
  each section via the existing from_md() -- the section header text becomes both the dict key
  and the resulting Daf's `.name`, unless a `%% daf` footer already supplies a name (footer wins
  for `.name`, never for the dict key). Logic lives in daf_md.py (as with from_md()/_from_md(), to
  avoid a circular import) and is wired onto the Daf class the same way. 12 new test cases in
  test_daf_md.py.
- Tests: coverage rose from 84% to 95% overall.
  - daf.py went from 88% to 99%.
  - daf_sql.py went from 0% to 99%.
  - daf_utils.py went from 86% to 98%.
  - daf_pandas.py went from 95% to 99%.
- New test files: test_daf_coverage_a.py, test_daf_coverage_b.py, test_daf_sql.py,
  test_daf_utils_coverage.py and test_daf_pdf.py. The PDF file is only a smoke test, since
  daf_pdf is experimental.
- The remaining uncovered lines are dead code, debug-only branches, daf_pdf.py and md_demo.py.
- tests/conftest.py makes a test fail if it reaches `breakpoint()`.
- Dev tools: added requests, xlsxwriter and pytest-cov. The tests already needed the first two.
- Added CLAUDE.md, and a startup hook that runs `uv sync`.

- Documentation: the start of a generated API reference. The page sources are in docsite/,
  and mkdocs.yml points to them. The first page covers the Daf class, built from its
  docstrings. mkdocs, Material, mkdocstrings and Ruff were added to the dev tools, with mkdocs
  pinned below version 2. The build still prints 29 warnings about docstrings that don't
  match their signatures.
- Docstrings for `retmode`, `itermode`, `iter_dict`, `iter_klist`, `iter_list` and `__iter__`
  were rewritten to explain what each is for, with examples that are run as doctests.
- Docstrings were rewritten for the class itself and for construction, size, copying,
  columns and keys, dtypes, schemas, `strip`, `clone_empty`, `set_lol` and the conversions
  to and from lists, dicts, CSV, Excel, NumPy, Pandas and JSON, appending, removing,
  indexing, selecting and reading rows, columns and cells, assigning, inserting,
  replacing and sorting, formulas, apply and reduce, grouping, sums and counts, joins, pivots,
  and Markdown. The build of the reference now has no warnings. It had 22 before. Each one says what the
  method is for, what it changes and what it returns. Examples are run as doctests.
  The schema, Pandas and Markdown methods are defined in helper modules. They did not appear in the
  API reference before. The reference now has a section for them. Issues noticed on the way
  are listed in notes/docstring_pass_issues.md. No behavior was changed.
- Type annotations: every parameter and return value is now annotated, in every module except
  daf_pdf.py and md_demo.py. The doc tables read their types from the signatures, so a missing
  annotation left a blank. `Daf.update_row` is now a static method, as it has no self. It was
  only ever called on the class. The annotation of `_attach_schema` said it returns None, but it
  returns the Daf. Modules that cannot import Daf at run time use a TYPE_CHECKING import.
- KeyedList and KeyedIndex have new docstrings, which are the one place that explains how
  KeyedList rows work. The Daf docstrings link to them. They have their own page in the docs.
  The older text stays as the second string. Two errors in the old KeyedIndex text were fixed:
  an example that printed `None` where nothing is printed, and a line that said a KeyedIndex
  built from another one copies it, when it shares it.

### Removed
- `Daf.isin()` is removed. It was a static method that made a list of bools, from an early attempt to match the `isin()` of pandas.
  A list comprehension or `select_where()` does the same. Neither daffodil nor AuditEngine used it. Its 2 tests are removed too.
  It was deprecated and then removed on 2026-10-05 at the request of the owner.
- `unpack_indirect()` is removed from `daf.py`. It was a module function that was not attached to `Daf`, and only its 3 tests called
  it. Its docstring example called `flatten_indirect()`, which does not exist. The accessor that makes a row read as one map
  replaced it. `_IndirectRowView`, `get_indirect_da()` and `get_indirect_val()` are still used and are kept. Removed on 2026-10-04
  at the request of the owner.
- `KeyedListEncoder` is removed from `keyedlist.py`. Nothing in daffodil used it. It was a `json.JSONEncoder` for a KeyedList. `KeyedList.to_json()`
  and `from_json()` are kept. Removed on 2026-10-04 at the request of the owner, who found no use of it. Its 2 tests are removed too.

### Changed
- The minimum Python is now 3.10, at the owner's request on 2026-10-05. It was 3.9, which has not worked for some time: three modules failed at import, and one test failed after that.
  Pip on 3.9 keeps installing 0.5.13. Four small edits were made anyway, and they do not change anything on 3.10 or later. `daf_utils.py` and `daf_md.py` use postponed
  annotations, `daf_types.py` no longer imports the unused `TypeAlias`, and `schemaclass()` reads `__dict__` for the annotations of each class. With them,
  2251 tests pass on 3.9 when run by hand. CI tests 3.10, 3.11, 3.12 and 3.13. The lock file is smaller, because it no longer carries the packages for 3.9.
- `copy()` is redone, at the request of the owner, on 2026-10-05. It takes a sum of the constants `Daf.COPY_ATTRS`, `COPY_OUTER`, `COPY_HD`, `COPY_DTYPES`, `COPY_KD` and `COPY_ROWS` for what the copy owns, and the four names
  `shallow`, `sortable`, `editable` and `deep` stay. The first three are sums of the constants. `deep` is a name only, because it cannot be combined. A copy is always a new object of the same class. Sharing is by choice and
  is what makes daffodil fast. Without a level, the class setting `copy_level_default` is used.
  - The default is now `sortable`, which was `shallow`. A plain `copy()` now has its own row list, `hd`, `dtypes` and key index, and shares the rows. It costs
    0.002 s for 200,000 rows of 50 columns, against 0.00002 s. `append`, `drop_cols` and `insert_icol` on the copy no longer reach the original.
    `copy('shallow')` is still there for a copy that shares the row list. A subclass can set `copy_level_default = 'editable'` to play safe.
  - A copy has no name unless `name=` is given. It used to keep the name of the original.
  - `ATTRS` copies the `disp_cols` list as well as the `attrs`. A `keyfield` that is a list is copied with the key index. An adopted key index
    with no keyfield is copied, not cleared.
  - `copy(False)` now uses the class setting. It used to mean `shallow`.
  - `clone_empty()` is now `copy()` with the layout bits, then no rows. It returns an object of the same class, and not a plain `Daf`. It keeps `md_max_rows`,
    `md_max_cols`, `disp_cols`, `schema`, retmode and itermode, which it dropped. With `cols` given that differ from the columns, it keeps the dtypes for the columns that remain, clears the keyfield if any key column is gone, and
    drops `schema` and `disp_cols`, which cannot be cut to fit. It used to keep the dtypes whole and the keyfield as it was. With the same `cols`, everything is carried over.
    One private method, `_align_with_columns()`, decides this for every Daf made from another one. `groupby()` and its relatives use it through `clone_empty()`, and keep their dtypes.
    Every selector that calls it gets the same change.
  - A keyfield that is not a column now raises `KeyError`, at the owner's request on 2026-10-05. This is in the constructor, in the builders such as
    `from_csv_buff()`, and in `set_keyfield()`. It was stored without an error, and key lookups then found nothing. The old idea was that a keyfield could
    be set before its column exists. A Daf with no column names yet can still be given one, so `Daf(keyfield='id')` followed by `append()` of dicts works.
    `set_keyfield(silent_error=True)` still stores the name, and the default of `silent_error` is now False. After a direct edit of the attribute,
    the lookup still says what is wrong. `join()` now gives a result with no keyfield when the left table is empty, where it gave a keyfield that was not a column.
  - `rename_cols()` and `set_cols()` now keep the keyfield, at the owner's request on 2026-10-05. They cleared it before. The keyfield takes the new name of its column.
    `set_cols()` follows the names by position, as it does for the dtypes. If the Daf has no column names yet, it keeps a keyfield that is one of the new names,
    so `Daf(lol=rows, keyfield='id')` followed by `set_cols(['id', 'v'])` works. A composite keyfield keeps its type. A key column with no new name clears it.
    A new private method, `_remap_keyfield()`, does this. Three tests that expected the old clearing were rewritten, and 12 were added.
  - The constructor now cuts `dtypes` to the names that are columns. Your dict is not changed. A dict that is already right is adopted as before.
    It costs about 1.6 microseconds for 50 columns, on top of 4. The `keyfield` is still stored when it is not a column.
  - The `COPY_` constants are new public names on `Daf`, and `T_copybits` is a new alias in `daf_types.py`. Tests are in tests/test_daf_copy_bits.py. Tests: 2202 before, 2231 after.
- `from_googlesheet()` and `to_googlesheet()` take a required `service_account_file` argument, and raise
  `NotImplementedError`. They had the placeholder path `path/to/your/service_account.json` in the source, so they
  could not be used without editing the library, and they could not be tested here. The draft code is in the
  commit e9e69fd. The two mocked tests are replaced by 4 tests of the new behavior. This also removes two of the
  mypy import errors.
- `from_csv_file()`, which is deprecated, now calls `from_csv()`. It reads the file as UTF-8, where it used the
  encoding of the machine, and a file that cannot be read raises `RuntimeError`, where it printed a message and
  returned None. Code that tests the result for None must change. 2 tests changed or added.
- `update_record_irow()` raises `IndexError` for a position that is out of range, and changes nothing. It did nothing,
  without a message. A Daf with no rows or no columns, and a record of `None`, still change nothing. A position
  found from the key index, as in `update_by_keylist()`, is never out of range. 2 tests changed or added.
- Converting text to `bool`, in `apply_dtypes()` and `convert_type_value()`, recognizes the words `false`, `no`, `n`,
  `f` and `off`, which gave 1, and `true`, `yes`, `y`, `t` and `on`, in lower case, capitalized and upper case, as well
  as `0`, `1`, `0.0`, `1.0` and the empty text. Any other text is kept as it was, as it is for `int` and `float`, so a
  value such as `maybe` is no longer made true without a message. Before, everything except six values gave 1. None and
  NaN still give 0, and other values are true unless they are 0. The result is still the int 0 or 1. A column that held
  1 for text that is not a bool word now holds that text. 7 test groups added.
- `remove_key()` and `remove_keylist()` are marked deprecated in their docstrings and in the README. Their behavior is
  unchanged. They return a shallow new Daf, and `select_krows(..., inverse=True)` does the same.
- A blank column name from parsing a header always becomes `Unnamed` plus its position. The constructor renamed
  blank names only when two names collided, so a single blank stayed `''`. `Unnamed` says that there is no name, is
  unique and is unlikely to be a real name, and `profile_ls_to_lr()` looks for it. `from_md()` already did this.
  `set_cols()` keeps the short prefix `col`, as in `col1`, because its names are printed and are given by you. The
  argument `unnamed_prefix` of `set_cols()` sets another.
- `value_counts_daf()`, `dict_to_md()` and `from_lod_to_cols()` changed a blank key into `Unnamed1`, once blank names
  are renamed, because the keys of the dicts went through the column names. They now keep the key as data. This also
  means that a blank key was never counted wrongly. 9 tests added.
- `from_lot()` without `cols` gives a Daf with no column names, as `Daf(lol=...)` does. It named them `col_0`,
  `col_1`. Use `set_cols()` for names. A Daf that has rows and no column names now raises `KeysDisabledError`,
  with a message that names `set_cols()`, from `to_lod()` and from the dict and KeyedList iterators, so also from
  `to_cols_dol()`, `select_where()` and `select_by_dict()`. These gave `[{}, {}]`, `{}`, an empty result or a
  confusing `ValueError`. `iter_list()` and an empty Daf are not affected. 13 tests added.
- One rule for a Daf with rows and no column names: names exist only if someone gave them. `iloc()` with
  `rtype` of `dict` or `klist`, and so `irow()`, `to_klist()` and `to_dict()`, now raise `KeysDisabledError`.
  They made up the names `A`, `B`, `C`, and `_basic_get_record()` also changed the Daf by calling `set_cols()`.
  `rtype='list'` still works. `to_md()` still writes spreadsheet names in the header of such a Daf, because a
  Markdown table needs a header row and a separator row, and `from_md()` needs them too. A test with the
  Python-Markdown tables extension, which MkDocs uses, showed that a table with no header row is not rendered
  as a table. The names are in the text only, and the Daf is not given names. 5 tests changed or added.
- `append()` and `extend()` say what they add by keyword. `append(lol=[[2, 'b'], [3, 'c']])` and
  `extend(lol=...)` add several rows, each a list in column order. `append(la=[...])` adds one row, even if its
  items are lists. A positional list is one row, as before. A list with more values than there are columns now
  raises `ValueError`. Before, the extra values were dropped without a message, and a list of lists with more
  items than columns lost rows that way. A short list is still padded with NULL, and a Daf with no columns takes
  a list as it is. `extend(lol=...)` checks every row first, so a bad row adds nothing. Giving more than one of
  `data_item`, `lol` and `la` raises `TypeError`, and so does a type that `append()` does not support, which
  raised `RuntimeError` with no message. A branch of `append()` that could not run was removed. The README shows the keywords. 28 tests added.
- The README says that `my_daf[irow] = dict` replaces the whole row, and that columns the dict lacks become NULL. It
  said only that column names are respected. It points to `update_record_irow()` for changing some cells.
- `select_cols()` keeps the columns in the order of `cols`, as `select_kcols()` and `my_daf[:, cols]` do. It
  used the order of the Daf. A name in `cols` that is not a column raises `KeyError`. It was ignored. A name given
  twice is used once. With no `cols`, the order is that of the Daf, and `exclude_cols` ignores names that are not
  columns, as before. It is also much faster for wide tables, because it no longer searches a list for each cell
  of each row. For 2,000 rows and 1,000 columns, keeping 500 of them took 0.068 s, against 4.116 s. 9 tests added.
- `set_col_irows()` raises `KeyError` with the name for a column that is not found. It did nothing. The
  docstring now says the method is deprecated, because `my_daf[irows, colname] = value` does the same. The
  five comments that pointed at GitHub issue 7 were removed. 2 tests changed or added.
- Row positions in `assign_record_irow()`, `update_record_irow()` and `insert_irow()`. The default position of
  `assign_record_irow()` and `insert_irow()` is now `None`, which adds the row at the end. A negative
  position counts from the end, as in a list, in `assign_record_irow()` and `update_record_irow()`. So
  `assign_record_irow(-1, rec)` and `my_daf[-1] = {...}` now replace the last row, and before they added a row.
  `my_daf[-1] = [...]` already replaced the last row, so a dict and a list now agree. A position before the
  first row raises `IndexError`. A position beyond the last row still adds a row. `update_record_irow()` with
  its default `-1` updated nothing, and now updates the last row. `insert_irow(-1, row)` still adds at the end.
  Code that passed `-1` to add a row with `assign_record_irow()` must pass `None` or leave the position out.
  11 tests added.
- `join()` fills the cells of a missing match with NULL, the empty string, as the rest of the library does for
  a missing value. It filled them with `None`. The new argument `fill` sets the value, so `fill=None` gives the
  old result. `join_records()` has the same argument and default. A joined table now prints blanks and not the
  word None. Code that tested `is None` on a join result must change. 12 tests changed, 5 added.
- `iloc()`, `irow()`, `to_klist()`, `icol()` and `icol_to_la()` now treat a position as a list does. A negative
  position counts from the end, so `iloc(-1)` is the last row and `icol(-1)` the last column. A position that is
  out of range raises `IndexError`. Before, both gave an empty dict or an empty list. A Daf with no rows still
  gives an empty result, so `Daf().to_dict()` is still `{}`. Code that tested for the empty answer to detect
  the end of the table must change. Tests changed or added: 6 changed, 6 added.
- The README has a table of which selectors share the rows of the original and which copy them. A test
  checks the table. The docstring of `select_records_daf()` says that it shares the rows. 12 tests added.
- The README said that appending a row whose key exists overwrites it, for all kinds of appending. Only
  `record_append()` does that by default. `append()`, `extend()` and `concat()` add a second row, which is
  faster, and replace the row with `respect_kd=True`. The README and the `respect_kd` lines of the four
  methods now say so. No behavior changed. For 100,000 rows one at a time, `append()` took 0.176 s and 0.295 s with
  `respect_kd=True`. Three tests added.
- `select_irows()` takes `inverse`, the name every other selector uses. The old `invert` is still
  accepted. The README examples for dropping rows and columns used `invert=` with `select_krows()` and
  `select_kcols()`, which raised `TypeError`. They now use `inverse=`. The three examples are checked in a
  test. 5 tests added.
- Ten in place methods that returned `None` now return the Daf, as the other in place methods do, so
  calls can be chained: `assign_record()`, `assign_record_irow()`, `update_record_irow()`,
  `assign_icol()`, `set_icol_irows()`, `find_replace()`, `apply_to_col()`, `apply_in_place()`,
  `apply_formulas()` and `set_col2_from_col1_using_regex_select()`. Code that ignored the result is not
  affected. A Daf with no rows is falsy, so `if d.find_replace(...)` would now read differently. The
  doctests of these methods now assign the result, so the output is not shown. 16 tests added.
- Missing key and column errors are more consistent. `col()` and `to_donpa()` raise the new
  `ColumnNotFoundError` for a column that is not there. It is a `KeyError`, as for any lookup of a name, and
  also a `RuntimeError`, so code that caught the old error still works. `select_record(silent_error=False)`
  raises `KeyError` with the key in it. `select_by_dict(expectmax=)` raises `LookupError` with the number of
  rows and the limit. `to_dod()` on a Daf with rows and no keyfield raises `KeysDisabledError`, as `d[key]`
  does, and not a `KeyError` with an empty name. An empty Daf still gives an empty dict. Tests added.
- Removed `Daf.from_dirlist()`. It was a stub with no code. It returned None. Nothing called
  it. The feature lives in AuditEngine and can be added here later.
- Library code no longer calls `breakpoint()` on error paths. It raises a specific error
  instead. The exceptions are daf_pdf.py and md_demo.py.
  - Before this, code with no breakpoint hook carried on after the breakpoint. It often failed
    later with UnboundLocalError, or returned partial results without any warning.
- New errors for a missing name:
  - `sort_by_colname` and sort_by_colnames raise Python's own KeyError for an unknown column.
  - `alter_daf_per_setting` raises KeyError for a setting missing from the dict. A setting of
    None or empty still does nothing. A new silent_error parameter, False by default, makes a
    missing setting do nothing too.
- Selecting a column slice from rows of uneven length raises IndexError.
- New errors for bad input:
  - `safe_regex_select` raises ValueError when the regex has no capture group.
  - `safe_regex_replace` raises ValueError for a badly formed pattern. That covers too many
    separators, and first and last characters that differ. The second kind used to be skipped
    with a logged message.
  - `convert_type_value` and test_strbool raise TypeError for an unsupported type.
  - `json_encode` raises ValueError for NaN or Infinity. It used to write NaN, which is not
    valid JSON.
- Errors are no longer hidden:
  - `reduce` passes on errors from the reduction function. A new silent_error parameter,
    False by default, skips the failing rows instead.
  - `sum_da` passes on errors other than the non-numeric values it already skips.
  - `create_index_at_cursor` raises on failure. It used to return False.
  - `write_buff_to_fp` passes on the original error.
  - `len_slice` and slice_to_range raise TypeError saying the slice bounds must be integers.
- Assigning a list or a Daf into part of a Daf copies values only where the source and the
  region overlap. There is no size check and no error.
  - A smaller source fills the top-left of the region. The rest is unchanged.
  - A larger source fills the region. The extra values are ignored.
  - Before, a short list hit a breakpoint for each missing value, and a long list was cut
    off.
- Assigning a Daf now copies its values. Before, the cells were filled with Daf objects.
  - The copy starts at the Daf's top-left corner. Its retmode does not matter.
  - A single cell still holds a whole Daf.
- `mypy` (added to `mypy.ini`'s scope, now fully clean: 217 -> 0 errors) run across all of
  `src/daffodil` for the first time. Most fixes were mechanical (missing/narrow annotations,
  `Union[dict, KeyedList]` (`T_ma`) made explicit at the many call sites where a row can
  legitimately be either, `cast()` after a runtime-confirmed branch where `daf_utils.is_list_of_type()`
  -- a runtime check, not a `TypeGuard` -- left mypy unable to narrow). `apply()`/`reduce()`/
  `manifest_*()`/`groupby*()`'s `func` parameters are now typed `Callable[..., Any]`: real
  reduction functions (`sum_da`, `count_values_da`, ...) take further keyword-only params
  (`cols`, `astype`, `omit_nulls`, `is_sparse`, ...) forwarded via `**kwargs`, which no fixed-arity
  `Callable[[X, Y], Z]` form can express. Considered making `KeyedList` a formal
  `collections.abc.MutableMapping` to resolve this more "properly", but decided against it: the
  whole point of `KeyedList` is that the key and the list are two separate, independently
  replaceable parts, not that it's fully a collection -- explicit `Union[dict, KeyedList]` typing
  with `isinstance` narrowing where needed keeps that design intact.
- `daf_pdf.~py` and `daf_indexing.~py` (old editor-backup-looking files, accidentally
  git-tracked) moved out of `src/` to a gitignored, local-only `deprecated/` folder.

### Fixed
- Docs: 127 table examples in the docstrings were regenerated from real output, on 2026-10-06. The output now has the blank lines before the table and before the `%%` footer, written as `<BLANKLINE>`, and the real column spacing.
  The flag `NORMALIZE_WHITESPACE` is removed from `pytest.ini`, so doctests now compare spacing exactly.
  Many old examples had a table that was not aligned, because `NORMALIZE_WHITESPACE` hid it. All 199 doctests now also pass with that flag turned off. The docs show these tables in a code block. A griffe extension that renders them as tables is saved in `notes/` and is not enabled.
- `to_md()`: the ruling row is at least 2 characters wide, as in `-:`. A column whose cells were all one character wide made the ruling 1 character wider than the cells,
  so the vertical bars did not line up in a fixed font. Approved by the owner on 2026-10-06. Tests: 3 new, 5 old expectations updated. The output of `to_md()` and `repr()` changes by one space in such columns.
  The doctests never caught it, because `NORMALIZE_WHITESPACE` ignores spacing.
- Docs: `apply_dtypes()` is described more fully. It said that columns left out of `dtypes` are not touched, which is true only with `silent_error=True`, and it did not say that `dtypes` replaces the `dtypes` of the Daf.
  It now says both, and says that an empty cell, as from two commas together in a CSV file, is NULL, stays NULL in a number column, and is skipped by `sum()`, `sum_np()` and `sum_da()`. A line of its docstring was 146 characters long and is reflowed.
- Docs: a Daf is not limited in what it can contain, and the `dtypes` do not limit it. A cell can hold any Python object. The `dtypes` say how to convert columns when `apply_dtypes()` or a reader is asked to,
  and a Daf built with `dtypes` keeps its cells as they are. This is said in the class docstring, with an example, in the docs of the constructor and of `apply_dtypes()`, and in the README, on 2026-10-06 at the owner's request.
  `apply_dtypes()` keeps text that cannot be converted to a number and makes an empty cell empty. A cell that is not text and not a number, such as a list in an `int` column, raises `TypeError`, and its docstring now says so.
  Three tests pin this down.
- Docs: the `from_directory()` docstring says that the rows are in the order in which the file system lists the files, which is not sorted and differs between machines.
  Its example no longer depends on that order. It did, and the first CI run on GitHub failed because of it, on 2026-10-05.
- `insert_icol()`, `insert_col()` and `insert_idx_col()`, and so `assign_col()` for a new column, changed the rows of
  every Daf that shared them. After `s = d.select_irows([0, 1])`, `s.insert_idx_col()` left `d` with rows of three values
  and a header of two columns. Shared rows are now copied first, so the other Daf is not changed. The method still
  changes this Daf and returns it. Approved on 2026-10-04. Tests: `tests/test_daf_shared_rows.py`. The `sortable` copy
  level is now safe for these methods. A shallow copy shares the row list itself, which is not detected.
- `select_where_idxs()` gives its function a KeyedList for each row, as `select_where()` does. It gave a dict when the
  `itermode` was dict, which is the default, and its docstring said KeyedList. It is about 3 times faster. A function
  that reads `row['name']` works as before. A function that uses a dict-only method on the row, such as `items()`,
  does not. 7 tests added in `tests/test_daf_select_where_idxs.py`.
- Loops over KeyedList rows are about 30% faster, and so is `select_where()`. `KeyedList` tests the case that a Daf loop
  uses first in its constructor, and tries one hashable key first when it reads a value. The results are the same for
  every kind of input. For 200,000 rows by 50 columns `select_where()` went from 0.214 s to 0.148 s. The docstring
  of `select_where()` points to `select_by_dict()` for a test that only compares columns. 16 tests added in
  `tests/test_keyedlist_construction_and_lookup.py`.
- Selectors return their own row list when every row is selected. `select_irows()` with a list of every row in order,
  `d[:]`, `select_krows()` with every key and with nothing and `inverse`, and `select_records_daf([], inverse=True)`
  returned the list object of the original. An append to the result was an append to the original, and the key index
  of the original was left wrong. The rows are still shared, as with every other selector. `copy()` still shares the
  list, as its `shallow` level says. Approved on 2026-10-04. 20 tests added in `tests/test_daf_shared_rows.py`.
- A Daf with a key index (kd) and no keyfield works with every key lookup. `krows_to_irows()` and `select_record()` honored
  it. `select_krows()`, `select_records_daf()`, `remove_key()`, `remove_keylist()` and `keys()` raised `KeysDisabledError`
  or gave an empty list, though the docstring of `select_records_daf()` says that a manual kd is enough. `assign_record()`
  still needs a keyfield, because it reads the key from the record, and its message says so. The messages now name the
  method and what is missing: `select_krows(): key lookups are disabled, as the keyfield is not set and there is no key
  index (kd).` The `(no kd)` message, which was reached only for an empty key index, now says that the index is empty
  and that a Daf with no rows has no keys. Decided on 2026-10-04. 16 tests added in `tests/test_daf_manual_kd.py`.
- `select_krows()` says why key lookups fail when the keyfield is not a column of the Daf. It said `Key lookups are disabled (no kd)`.
  The message names the keyfield and lists the columns. The error class is the same, and a Daf that works is not slowed.
  The constructor and `from_csv_buff()` document that a keyfield that is not a column is stored without an error, as
  `set_keyfield()` does. Decided on 2026-10-04. 7 tests added in `tests/test_daf_keyfield_not_a_column.py`.
- `from_csv_buff()` and `from_csv()` give an empty Daf with no columns for empty text, bytes, an empty iterator or an empty
  file, as `from_md('')` does. They raised `IndexError: pop from empty list`, and `from_csv()` reported that as a failure
  to read the file. Decided on 2026-10-04. 11 tests added in `tests/test_daf_csv_empty.py`.
- Docs: `from_md()` says that a table with no header row is not read, and that a Daf with no column names comes back from
  `to_md()` and `from_md()` with the names `A`, `B` and so on. Decided on 2026-10-04 to keep the behavior.
- Docs: `from_csv_buff()` and `from_csv()` say that the length of each row is not checked, and point to
  `is_rectangular()` and `force_rectangular()`. Decided on 2026-10-04 to keep the readers without a check. The check
  would cost 0.6% to 1.5% of a read.
- `select_by_dict()` is about 40 times faster. It compares the cells by position, and does not turn each row into a dict.
  For 200,000 rows by 50 columns it took 0.53 s and now takes 0.012 s. The rows it returns are the same, and a Daf with
  rows and no column names still raises `KeysDisabledError`. 22 tests added in `tests/test_daf_select_by_dict.py`,
  including a comparison with the row by row rule.
- `select_by_dict()` returns a shallow new Daf, like the other row selectors. Its rows are the rows of the original, not
  copies. It is a live view for values. Decided on 2026-10-04. The deep copy was not needed, because `insert_col()`
  and `insert_idx_col()` now copy shared rows first. The report code that selects and then inserts an index column keeps
  working, and the original is not changed. Tests: 4 in `tests/test_daf_shared_rows.py`.
- Docs: the examples no longer use the `lol` attribute, and a Daf is shown the way daffodil shows it. 126 examples read their result with `.lol`, which is
  not the standard way to read a Daf. A Daf result is now written as the bare expression, so the example shows the Markdown table and the size line
  that `repr()` gives, at most 10 rows. A row is read with `iloc()`. A comparison uses `kept == d`. The examples of `copy()` and
  `select_records_daf()` show sharing by appending a row and by setting a cell, and not by comparing the lists. The expected output was regenerated
  by running each example. `pytest.ini` has `doctest_optionflags = NORMALIZE_WHITESPACE`, so the blank lines of the table need no `<BLANKLINE>`
  markers. No library code changed. Requested on 2026-10-05.
- Docs: `select_where()` shows how to test a value against a set, or against the keys of another Daf, in one pass, with no list of bools. It
  gives the cost: for a test on one column of 200,000 rows and 1,000 values, `select_where()` took 0.11 s and a comprehension over
  `col()` with `select_irows()` took 0.02 s, so it shows both. The README table has a new row for the `isin()` of pandas. Requested on
  2026-10-05.
- Docs: the 47 methods that kept an old second string after the real docstring were read, and the points that the real docstring
  lacked were moved in, after each one was run. `isin()` says that its list of bools must not be used as a column selector,
  because a list of bools is read as positions. `set_keyfield()` says that a Daf with columns and no rows can have a keyfield.
  `to_csv_buff()` says how a dict with keys that are not text is written. `krows_to_irows()` gives the cost of the first lookup,
  which builds the key index. `select_icols()` gives the cost of `flip=True` against select and `transpose()`. `append()` says that
  None and empty values add nothing. The old strings are kept, as the rule says. No code changed.
- Docs: 26 methods of `Daf` got an `Examples:` section, and each example was run: `iter_dict`, `iter_klist`, `iter_list`,
  `num_rows`, `len`, `from_excel_buff`, `from_csv_file`, `buff_to_file`, `from_directory`, `from_googlesheet`,
  `to_googlesheet`, `krows_to_irows`, `kcols_to_icols`, `select_records_daf`, `irow_la`, `col_to_la`, `icol_to_la`,
  `manifest_apply`, `reduce_dodaf_to_daf`, `multi_groupsum`, `valuecounts_for_colnames_ls_selectedby_colname`,
  `derive_join_translator_daf`, `join_records`, `md_daf_table_snippet`, `to_md_cols` and `daf_to_lol_summary`. The docstring of
  `select_records_daf()` said that, with no keys and `inverse`, the result uses the row list of the original. That was
  true before the fix of 2026-10-04 and is no longer. `manifest_apply()` says that `by` must be `table` and that its default
  is `row`, which does not work. `to_md_cols()` says that Python-Markdown shows its text as plain text. 26 doctests added.
- Docs: the 15 public methods of `KeyedList` and `KeyedIndex` that had no docstring now have one, with `Args`, `Returns`,
  `Raises` and examples that were run: `keys`, `values`, `items`, `get`, `update`, `set_values`, `to_dict`, `to_json`,
  `from_json`, and `get`, `index`, `keys`, `append` and `to_dict` of `KeyedIndex`. The dunder methods got short ones too.
  The API page of `KeyedList` showed 4 members, because mkdocstrings hides members without a docstring. It now shows 31.
  The docs say that `set_values()` points at a new list and does not write into the old one, that `del row[key]`
  shortens a Daf row, and that `to_json()` turns int keys into text. 21 doctests added. No code changed.
- `KeyedList.values(astype)` keeps an empty cell, which is NULL, as it is. It raised `ValueError` for `int` and `float`, and
  turned `''` into `False` for `bool`, where `daf_utils.astype_la()` and the rest of daffodil keep NULL. Decided on 2026-10-04.
  `KeyedList` has its own copy of `astype_la()` because it cannot import `daf_utils` without a circular import. It is
  now named `_astype_la()` in `keyedlist.py`, because it is internal. `daf_utils.astype_la()` keeps its name. 24 tests
  added in `tests/test_keyedlist_astype_null.py`.
- A selection is a live view for values. Changing values through it changes the original, by cell, column,
  `apply_in_place()`, `update_record_irow()` or a KeyedList loop. Decided on 2026-10-04. Only changes of shape copy shared
  rows first. `assign_icol()` copies only when it adds a column. The `sortable` copy level is still not safe for
  value changes. Use `editable`. Tests: 20 in `tests/test_daf_shared_rows.py`.
- `apply_formulas()` left the key index as it was after a formula error, though earlier formulas had already changed
  cells. A lookup then found the old key in the changed row, and did not find the new key. The index is now
  rebuilt when it is next needed, on every exit. 2 tests added.
- `from_pandas_df()` lost `name` when `use_csv=True`. The dtypes of a Series were keyed `col` and not by the labels
  that become the column names. The `dtypes` argument does nothing, and is now deprecated. It gives a
  `DeprecationWarning` and will be removed. 4 tests added.
- `insert_irow()` with a row that is not a list or a dict, or with no row, raised `UnboundLocalError`. It
  now raises `TypeError` that names the method and the type it got. 6 tests added.
- An explicit `keyfield` was lost when a schema supplied the cols, as in
  `Daf(schema=B, keyfield='contest')`. The keyfield was cleared, and then the schema's `__keyfield__` was
  adopted, or none. The explicit keyfield now wins, as the README says. This was so for a schemaclass
  and for a schema Daf. Passing `cols` as well already worked. Tests added.
- `apply_formulas()` left `retmode` as `val` after a formula error or after formulas that never settle,
  so later row reads gave lists and not Daf objects. The `retmode` is now restored on every exit.
  Cells that were already changed stay changed. Tests added.
- `sum_np()` raised a NumPy error for a blank cell, and the message named no column. A blank, None or
  NaN cell now counts as 0, as in `sum()`. The columns are read with `to_donpa(default=0)`. A column
  of text raises `TypeError` that names the column. An unknown name still raises `KeyError`. For
  200,000 rows by 10 columns the call took 0.153 s, against 0.112 s before. Tests added.
- `from_directory()` printed its elapsed time on every call. The print is removed. The docstring also
  said `ctime` is the creation time. On Linux and macOS it is the time of the last metadata change.
- `remove_key()` with a composite keyfield read a bare key tuple as a range, and raised a `KeyError` that
  named the first part of the key. A tuple as long as the keyfield, with no tuples inside it, is now
  one key, so `remove_key(('a', 1))` works. A tuple of key tuples is still a range, a list of key
  tuples is a list of keys, and a single column keyfield is unchanged. Tests added.
- `from_json()` read the dtypes `list` and `dict` back as the text `'list'` and `'dict'`, so a saved
  table could not be converted with `apply_dtypes()` after reloading. Both are now read back as types.
  Other names, such as `date`, still come back as text. `to_json()` also set the dtypes of the Daf to an
  empty dict as a side effect when they were None. It now leaves the Daf alone. Tests added.
- `daf_to_lol_summary()` and `to_md()` with only `max_cols` added a row of `...` under the header,
  though no row was hidden. The slice `[-0:]` also made `max_rows=1` show every row. A limit of 0
  now means no row limit. An odd `max_rows` keeps the extra row at the start, so 3 shows 3 data rows
  and not 2. Even limits are unchanged. Tests added.
- `transpose()` without `include_header` named its cols `key`, `A`, `B`, with one name too many. The first
  source row sat under `key`, each later row under the wrong name, and the last name had no data.
  The default cols are now `A`, `B`, `C`, one for each source row. With `include_header=True` they
  still start with `key`. A `new_cols` you pass is unchanged. Tests added.
- `set_cols()` accepted more names than columns. The header was then longer than the rows, and
  `to_lod()` dropped the extra names without a word. It now raises `AttributeError`, as it
  already did for too few names. A Daf with no columns still accepts any list. Tests added.
- `to_donpa(default=...)` had no effect. The value reached `col()`, which uses it only for the
  `indirect_col` case. It now replaces each NULL, None and NaN cell as the column is read, in the
  arrays only, and the Daf is not changed. A column with blanks stays numeric when the default
  is a number. Before, NumPy made it a text array. With 1 in 20 cells blank, it takes 0.155 s,
  not 0.345 s, for 200,000 rows by 10 columns, and 0.216 s, not 0.491 s, for 2,000 rows by 1,000
  columns. With no default, the code is the same as before. `to_pandas_df(use_donpa=True,
  default=...)` now gets the replacement too. Seven tests added.
- `from_cols_dol()` used the length of the first list. A shorter list raised `IndexError`, and a
  longer list lost its extra values without a message. It now raises `ValueError` that names the
  column and both lengths. It builds the rows with `zip`, which is also faster: 0.104 s, not
  0.255 s, for 200,000 rows and 10 columns. Six tests added.
- `include_cols` in `from_csv_buff()`, `from_csv_file()` and `from_csv()` had no effect, so every
  column was read. It now keeps only the listed columns, in the order listed. Each row is cut
  down as it is read, so a stream never holds the other columns. For a 2,000 row by 1,000
  column stream with 3 columns kept, the peak memory was 0.8 MB, not 129.0 MB. Reading from a
  string took 0.177 s, not 0.230 s, and the path without `include_cols` is the same. A name
  that is not in the header raises `KeyError`. With `noheader=True` it raises `ValueError`. A
  short row gives NULL for the cells it lacks. Fourteen tests added.
- Several methods added a value to the rows without adding a column name, so the rows had more
  values than `columns()` had names. The CSV writer, `to_lod()` and `to_pandas_df()` then lost the
  value, and `d[:, 2]` raised `IndexError`. `annotate_daf()` with a field that is not a column,
  and `set_col2_from_col1_using_regex_select()` and `apply_replace_regex()` with a new `col2`,
  now add that column. `assign_icol(-1)` and `insert_icol()` with no `colname` name the new
  column with the next spreadsheet name, such as `C`, made unique, when the Daf has names. After
  the `apply_in_place()` change, the two regex methods had also stopped storing anything for a
  new `col2`. Eleven tests added.
- `apply_dtypes()` and `convert_type_value()` changed a whole number text above 2 to the power
  53 into a different number, because the text went through a float. `'12345678901234567890'`
  became `12345678901234567168`. Whole number text is now converted with `int()`, so every digit
  is kept. Text with a decimal point or an exponent still goes through a float and loses its
  decimal part. The check for a point or an exponent is made first, so no error is raised and
  caught for each cell.
- An `int` or `float` conversion that fails now keeps the original text, as `list` and `dict`
  already did, and no longer turns it into NULL. `'inf'` and a float infinity no longer raise
  `OverflowError`. Empty cells are still empty. A later step such as a sum or a sort may raise an
  error for the text. `reduce()` with `sum_da()` still skips it, as it skips blanks.
- `apply_dtypes()` is faster. It looks up the conversion once for each column and has a fast
  path for plain digit text. For 2,000 rows and 1,000 columns of int text it takes 0.521 s, not
  0.932 s. For 200,000 rows and 3 columns it takes 0.076 s, not 0.175 s. A new function,
  `daf_utils.get_converter()`, gives the conversion for a type. Results are the same as before
  for 521 test values, except for the two changes above. Nine tests added, and one that was marked as a bug now passes.
- `apply_to_col()` passed its keyword arguments to `map()`, so any keyword argument raised
  `TypeError`. They are now passed to the function with each value. Three tests added.
- `from_lod()` lost a value without a message when a later dict had a key that the first dict did not have. It now adds the key as a column at the right,
  at the owner's request on 2026-10-05, and pads the rows before it with NULL. The columns are all the keys of all the dicts, in the order of their first appearance.
  You do not have to look through the data first. A first version of this fix raised `ValueError` instead, and that version was never released. A record that only lacks keys
  still gets NULL.  The first dict may be empty. The keys of each dict are compared
  with a set of the columns in C, and each row is built directly, so a uniform list takes the same time as before: 0.447 s for 200,000 dicts of 10 keys, against 0.449 s.
  Padding is done once at the end, and only if a column was added.
  With `cols` or `dtypes` given, the caller chose the columns, and a dict with another key now raises `ValueError` that names the keys, at the owner's request on 2026-10-05.
  Before, the value was dropped without a message. The new keyword `ignore_extra_keys=True` leaves such keys out on purpose, as when you pick a few columns of wide records.
  This path also builds each row directly, without a dict for each record: 0.39 s with the check for 200,000 dicts of 10 keys, against 0.43 s before.
  `from_lod_to_cols()` and `from_dod()` call `from_lod()` with their `dtypes`. A `dtypes` for `from_lod_to_cols()` that leaves out a key now raises. Before, it dropped the key,
  and when the keys that it left out came first, it put the values under the wrong keys. `from_dod()` with `dtypes` that leave out the keyfield column raised `KeyError`, and now raises `ValueError`.
  `from_dod()` takes `ignore_extra_keys` too, at the owner's request on 2026-10-06, and passes it to `from_lod()`. With `True`, an inner key that `dtypes` does not name is left out, as in 0.5.13.
  The `dtypes` must still name the keyfield column, or `from_dod()` raises `KeyError`. `from_lod_to_cols()` does not take it, because a left out key there puts the values under the wrong keys.
- `drop_cols()` left the keyfield set to a column that was gone, so key lookups returned empty
  results or raised `KeyError`. It missed a composite keyfield too. It now clears the keyfield
  when any column of it is dropped, as `select_cols()`, `rename_cols()` and `set_cols()` do.
  Three tests added.
- `remove_dups()` changed the Daf it was called on. It set the keyfield of the Daf to the
  argument. Called with no argument on a keyed Daf, it cleared the keyfield and returned every
  row as a duplicate. Without a keyfield it did the same, with no error. It now uses the
  keyfield of the Daf when none is given, raises `KeysDisabledError` when there is none, and
  leaves the Daf as it was. The two results have the key columns as their keyfield. When there
  are no repeats, the first result is now a new Daf that shares the rows, not the same Daf.
  Five tests added and one changed.
- `narrow_to_wide()` lost data without a message. Rows of one id that were not next to each
  other gave two rows for the id and dropped columns. A name that first appeared in a later
  id was dropped, even for sorted input. It now groups by id in any order, keeps all names
  in the order first seen, and fills missing combinations with NULL. `wide_cols`, which was
  not used, now chooses and orders the columns. It is slower, 0.60 s against 0.43 s for
  200,000 rows, and holds one dict for each id. Eight tests added.
- `apply_in_place(by='row')` stored the values of the returned dict by position. A dict with
  the keys in another order put values in the wrong columns, a shorter dict made a short row,
  and a longer dict made a long row. The values are now written back by column name. A column
  that the dict lacks keeps its value, and a key that is not a column is ignored. Five tests
  added.
- Assigning a `str` to several rows or to a column corrupted the table. `d[[0, 1]] = 'x'` and
  `d[:] = 'x'` stored the bare string as each row. `d[:, 'v'] = 'x'` set only the first row,
  and `'xyz'` spread its letters over the rows. A `str` or `bytes` is now one value. A list
  assigned to several whole rows was stored as one shared list object, and each row now gets
  its own copy. Seven tests added.
- Reading a column slice with a negative number or a stop of 0 gave wrong data or an error.
  `d[:, -2:]` returned five columns with repeats, `d[:, :-1]` returned none, and `d[:, 1:-1]`
  and `d[:, ::-1]` raised `IndexError`. `select_icols()` now uses `slice.indices()`, the rule
  of a Python list slice. A stop beyond the last column is cut to the end, not an error.
  Eight tests added.
- `apply_schema()` with something that is not a schema stored it in `schema` and returned
  None. It now raises `TypeError` and stores nothing. It matches `attach_schema()`. Test
  added.
- `apply_colwise()` returned None, although its docstring and annotation say it returns the
  Daf. It now returns the Daf. It also passes the default to `insert_col()` by name. The
  results are the same as before. Test added.
- `drop_cols()`, `update_by_keylist()` and `insert_col()` returned None when there was
  nothing to do. They now return the Daf, as the other methods do, so calls can be chained.
  Their return type is now `Daf`, not `Optional[Daf]`. Three tests changed from `is None`
  to `is daf`. One test added.
- `append(Daf, respect_kd=True)` ignored `respect_kd` and left duplicate keys. It now passes
  the flag to `concat()`. `extend()` ignored it too. `concat()` returns the Daf instead of
  None when the other Daf is empty. Its docstring said it kept one record per key, which
  was not true by default. The docstrings now say what happens.
- `manifest_reduce` lost the data columns from each chunk. It collected the chunk results in
  a table that had the manifest's columns. It now keeps the chunks' columns.
- `count_values_da` changed the caller's rows. It stored a row's list or dict, then added to
  it. It now stores a copy.
- `flatten` silently ignored use_pyon=False. A line set use_pyon back to True. Since daffodil
  standardized on PYON, use_pyon=False now raises ValueError, and the JSON code is removed.
  The parameter stays so existing calls with use_pyon=True still work.
- Duplicate column names could lose a column name. With columns a_2, a and a, the second a is
  renamed a_2, which already exists.
  - The header then had 2 names, while each row still had 3 values.
  - The third value stayed in the data but had no name. Code that works by column name
    could not see it.
  - This now raises KeyError.
- `record_append` failed with UnboundLocalError for a mapping that isn't a dict, such as a
  read-only mapping.
- `pandas_dtype_dict_to_python` now uses pandas_dtype_to_python_type(), the same mapping as
  Daf.from_pandas_df(). Before, it had its own copy, which left out category columns with a
  breakpoint. Now a category column maps to str.
- SQL name encoding could not round-trip characters above 0xFF. The euro sign was written as
  __20AC but read back as a space and AC. Encoding now marks the width, like Python's escapes:
  __HH up to 0xFF, __uHHHH up to 0xFFFF and __UHHHHHHHH above. Names up to 0xFF encode as
  before.
- KeyedList rows were slow to create. The iterator and `iloc(rtype='klist')` passed the table's
  hd dict, so every row rebuilt its own index of the column names. The cost grew with the
  number of columns. The rows now share one index, built once per loop, and `iloc` keeps one
  per Daf. The rows are the same. In a test with 50,000 rows and 400 columns, a loop that reads
  one field took 0.047 s, against 0.803 s before and 0.583 s with dict rows. Twenty thousand
  `iloc` calls took 0.009 s, against 0.501 s. A KeyedList that adopts a shared index copies
  it before it adds a key, so a row that adds a key does not change the other rows. The index
  kept by `iloc` is rebuilt when hd is
  replaced or its length changes. Editing the names in hd in place, without changing its
  length, is not detected.
- `join` changed the caller's shared_fields list. It added each table's key to the list. When
  the list was reused for another join, a stale key could make a plain column look shared, and
  the second table's values for that column were dropped. The list is now copied first.
- `join` on a composite key failed with an AssertionError that had no message. It now raises
  KeyError saying join is not supported for complex keys. The check runs once, at the start of
  the join.
- `xlsx_to_csv` failed with StopIteration for a spreadsheet with fewer than 3 rows, such as a
  header only. It called `add_trailing_columns_csv`, which sampled 3 rows. xlsx2csv already pads
  every row to the widest row, so that step did nothing. It is no longer called. The
  add_trailing_blank_cols parameter stays and is ignored. `add_trailing_columns_csv` also no
  longer fails on a short file.
- `is_d1_in_d2` failed with a TypeError when a row was a KeyedList. A KeyedList's `items()` is an
  iterator, and iterators can't be compared with `<=`. So `select_by_dict` and
  `select_first_row_by_dict` failed in keyedlist iteration mode. This was so since 0.5.10. Two
  dicts still use the fast comparison. Anything else is checked key by key.
- `key in klist` on a KeyedList scanned every key. It now checks the index. At 400 columns that
  took 4.8 microseconds before, and about 0.07 now. An unhashable key, as in `['a'] in klist`,
  now raises TypeError, as it does for a dict.
- The list comparison helper now accepts tuples. It used to fail with UnboundLocalError.
- Found and fixed several real bugs during the mypy pass (each verified against the full 1294-test
  suite before/after): `daf_utils.py` used `time.sleep` in `write_buff_to_s3path()`/
  `does_s3path_exist()` without importing the `time` module -- would have raised `NameError` the
  first time that path actually ran. `daf_utils.is_list_of_type()`/`_sanitize_cols()` fell through
  to an implicit `None` instead of `False`/`[]` on certain inputs. `daf_utils.safe_regex_replace()`
  mishandled a `bytes` item during decoding. `daf_utils.len_rowcol_spec()` silently measured a
  `range` input as length 0 (range was already a documented valid input). `daf.py`'s
  `apply_dtypes()` read the `dtypes` parameter instead of `self.dtypes` in its hd-adoption branch,
  after a guard that checked `self.dtypes`. `record_append()`/`_basic_append()` assigned a
  `KeyedIndex` object directly to `self.hd`, which must be a real `dict` everywhere else it's
  used. `krows_to_irows()` constructed a `KeysDisabledError` but never raised it. `apply_in_place()`
  would crash with a bare `AttributeError` if `func` returned `None` for `by='row'` (`None` is
  only valid for `by='row_klist'`) -- now raises a clear `ValueError`. `manifest_reduce()` and
  `multi_groupsum()` silently passed a `None` default into functions that require a real value,
  failing deep inside with `"None" not callable` or a `KeyError` -- now raise a clear `ValueError`
  up front. `groupby_cols_reduce()` crashed (`TypeError: can only concatenate list (not "NoneType")
  to list`) if `reduce_cols` was left at its `None` default.
- Daf.from_lod([], cols=[...])'s empty-records early return dropped the `cols` argument entirely,
  returning a 0-column Daf instead of one with the requested columns (dtypes already survived this
  same early return correctly; cols did not). A 0-column empty Daf serializes via to_csv_buff() to
  a blank, headerless line rather than the expected header-only row (pandas' equivalent,
  `pd.DataFrame(columns=[...]).to_csv()`, correctly emits a header-only row for zero rows). Fixed
  by passing `cols=cols` through on that branch. Found 2026-08-03 via a caller building an empty
  Daf with explicit cols and no dtypes (AuditEngine's mapping_option_names_ocr.py). New regression
  test: test_from_lod_no_records_but_cols.
- `set_keyfield()`'s falsy-Daf guard (`if not self: return self`) fired on any columns-only,
  zero-row Daf (the empty-schema case a caller builds before appending rows), since `bool(daf)`
  follows `num_cols()`'s row-based definition, not `self.hd`. `set_keyfield()` then silently
  no-op'd -- `.keys()` stayed stale even after later appends, invisible until the table happened to
  be reloaded from a file that already had rows. Found via AuditEngine's s3_state work (build a
  columns-only Daf, `set_keyfield()`, then `append()` -- exactly this shape). Fixed by checking
  `self.hd` directly in `set_keyfield()` instead of the Daf's own truthiness; `num_cols()` itself
  is intentionally unchanged (its falsy-means-"no rows" semantics are relied on in 238+ places
  across AuditEngine for `if daf:` checks). New regression test:
  test_set_keyfield_then_append_on_empty_daf.
- `append()` gained a `respect_kd` parameter (default False, preserving existing behavior):
  `append(data_item, respect_kd=True)` now upserts in place on an existing key instead of adding a
  duplicate row, threaded into `record_append()`'s existing implementation. `append()`'s own
  docstring previously claimed upsert behavior the code never actually provided (confirmed
  directly by testing) -- a real, longstanding documentation/behavior mismatch, not a new
  regression: the index used to be actively maintained on every append, which made upsert-by-
  default correct, until that was deliberately changed to a lazily-rebuilt-on-`.keys()` index for
  performance, and the docstring was never updated to match. `respect_kd=False` stays the default
  since eagerly checking for an existing key on every append costs a rebuild of the (lazy) index
  each time, wasteful when the caller already knows the key is new. New tests:
  test_append_default_does_not_upsert_existing_key,
  test_append_respect_kd_true_upserts_existing_key.
- `remove_key()`/`remove_keylist()`'s docstrings claimed in-place mutation of the receiver; both
  actually return a NEW Daf, sharing (not copying) the original's row objects by reference --
  confirmed directly via `is` identity checks, and that mutating a shared row's cell through the
  new Daf is visible through the old one too, unless `.copy()` is called first. Docstrings
  corrected to state this accurately rather than changing the actual (correct, by design)
  behavior. New tests: test_remove_key_does_not_mutate_original,
  test_remove_key_result_shares_row_objects_with_original.
- `_from_md()`/`_dodaf_from_md()` passed `footer_meta.get("keyfield")`/`.get("name")` straight
  into `Daf.__init__`, both of which can be `None` when no footer metadata is present --
  `keyfield`/`name` are typed plain `str` (default `''`), not `Optional`. Worked by accident under
  plain Python (no runtime parameter-type enforcement); surfaced while piloting `mypyc`
  compilation, whose compiled `__init__` does enforce the declared signature and raised a
  `TypeError`. Fixed both call sites with `or ''`.
- `tests/daf_benchmarks.py` couldn't run at all: missing `objsize` dev dependency (added to
  `pyproject.toml`'s `dev` group and `requirements.txt`, replacing the no-longer-used `Pympler`
  pin the script stopped importing a while ago) and two calls to `daf_sum2()`/`daf_sum3()`, long
  commented out of `daf.py` itself as investigatory dead ends. Removed both calls and their
  `report_attrs` row.

### mypyc compilation pilot (not adopted, investigation only)
Explored whether `mypyc` could speed up daffodil, at Ray's request, as an explicit experiment
("not a serious need"). Full writeup, including every source change a real compile needed and why,
lives in AuditEngine's engineering notebook (a sibling repo,
`engineering_notebook/2026-09-08_mypyc-compilation-pilot-daffodil-and-audit-engine.md`) rather than
here, since none of it was applied to real source beyond the two bugfixes above. Numbers:
`docs/daf_benchmarks_mypyc.md` (real 17-48% speedups on operations that stay in daffodil's own
Python loops -- construction, mutation, insert, row-wise sum, keyed lookup; no benefit on
pandas/numpy interop or pure list-reshuffling operations like `transpose`). Verdict: real for
daffodil's own hot paths, but not pursued further given the real ongoing costs (platform-specific
wheel builds instead of one universal pure-Python wheel, a broken local editable-install dev loop,
harder debugging) against no concrete felt need yet.

---

## [0.5.13] - (2026-06-25)

### Added
- Added pytest test coverage for keyedlist.py (62% -> 100%), daf_md.py (13% -> 99%), and a substantial
   portion of daf_utils.py (39% -> 60%), in new test files: test_keyedlist2.py, test_daf_md.py, test_daf_utils.py.

### Changed
- Daf.to_md() reworked to always pass the column header explicitly rather than relying on the first row
   of the rendered table being treated as a header; daffodil never treats the first row of a lol as an
   implicit header. This also fixes a `header=` override being silently ignored whenever the Daf already
   had its own columns.
- Daf.to_md() now generates spreadsheet-style column names (A, B, C, ...) for headerless/bare lol arrays,
   since Daf.from_md() requires a header + separator row.
- Daf.from_md() now requires a header + separator row; headerless markdown tables are no longer supported
   (previously crashed with UnboundLocalError on headerless input).
- transpose_lol() now validates the transposed shape and raises RuntimeError on ragged-right (non-rectangular)
   input, matching its documented contract. Previously, numpy silently returned the original ragged data
   unchanged (no error, no transpose) since np.array(lol, dtype=object) does not raise for ragged lol.
- json_encode()/NpEncoder no longer pass `default=str` to json.dumps, which was silently bypassing
   NpEncoder's numpy handling (np.int64, np.ndarray) for any type that wasn't already a native float
   subclass (np.float64 only "worked" by accident). Added diagnostic breakpoint() calls (marked # temp)
   for NaN/Infinity and for any other unrecognized type reaching the encoder, to surface real cases for
   investigation rather than guessing at handling now.

### Fixed
- Fixed KeyedIndex.__init__() from a KeyedList: was storing a dict_keys view instead of a dict, making the
   resulting KeyedIndex unsubscriptable.
- Fixed KeyedList.to_json()/from_json() and KeyedListEncoder: were not converting the KeyedIndex hd to a
   plain dict before/during JSON serialization; from_json() also called __init__() with nonexistent
   hd=/values= keyword arguments instead of positional args.
- Fixed mdlink(): the s3-path branch called a nonexistent utils.safe_basename() (now uses
   parse_s3path()['basename']), and separately left `url` unassigned (UnboundLocalError) whenever a title
   was already supplied for an s3 path.
- Removed two stray, unmarked debug breakpoint() calls (in KeyedList.__init__() and md_lol_table()'s
   exception handler, the latter now unreachable after the transpose_lol() fix above).

### Removed
- Removed `includes_header` / `sum_col_idxs` parameters from md_lol_table() (legacy from AuditEngine's
   pre-daffodil md.py, unused by daffodil; sum_col_idxs called a nonexistent utils.lol_sum_cols()).
- Removed md_process_template() (unused dead code; was also broken via a Template.safe_substitute() misuse).
- Removed set_type_la() from daf_utils.py: unreachable dead code, only referenced from an
   already-commented-out caller (apply_dtypes_to_hdlol, itself noting "do we need this any more? Use
   my_daf.apply_types()"); its own logic was separately dead due to a `str is not NULL` typo.
- Removed buff_csv_to_lol_old(), profile_ls_to_loti(), dict_with_index()/with_index(), combine_records(),
   safe_max()/safe_min(), notice_beep(), validate_json_with_error_details(), slice_to_list() from
   daf_utils.py -- confirmed unreachable from any production code path via reachability analysis.

### Added (continued)
- Added pytest test coverage for the remainder of daf_utils.py (60% -> 90%), including the CSV/header
   data-munging cluster (profile_ls_to_lr, is_comment_line, make_strbool/test_strbool, unexcelstringify,
   precheck_csv_cols, get_csv_column_names), the dtype-coercion cluster (convert_type_value, astype_value,
   unflatten_val, str2bool, is_numeric, safe_eval, json_decode, safe_convert_json_to_obj, json_encode,
   NpEncoder), lol manipulation helpers, the list_stats family, smart_fmt, logging/misc helpers, slice/
   range/path/s3 helpers, buff_csv_to_lol, xlsx_to_csv, compare_lists, and several smaller helpers.
- Added `_filter_comment_lines()`: a lazy, quote-aware, streaming-compatible comment/blank-line filter
   for CSV input, used by buff_csv_to_lol() when user_format=True (default mode). Tracks open quoted
   fields (counting `"` per line) so an embedded newline inside a quoted field -- which could otherwise
   look like a standalone comment/blank line -- is never misclassified and stripped.
- Added `strict_comment_filter` parameter to buff_csv_to_lol() (default False): opts into the older,
   rigorous-but-non-streaming preprocess_csv_buff() behavior when needed; raises a clear ValueError if
   used with non-str/bytes input, since that path cannot be streaming-compatible.

### Changed (continued)
- buff_csv_to_lol()'s user_format=True comment-filtering is now streaming-compatible by default (see
   _filter_comment_lines() above), rather than requiring the full buffer to be materialized up front.
- beep(): fixed inverted is_linux() condition that caused it to silently no-op on actual Linux; now
   emits the terminal bell character on Linux (no external `beep` package dependency) and continues to
   use winsound/os.system('beep ...') on Windows/other, for this dev-only diagnostic helper.

### Fixed (continued)
- Fixed buff_csv_to_lol()'s Iterator branch (text_line_generator/byte_line_generator): both reused the
   name `buff` for the closure's captured free variable and the reassignment target, which (due to
   Python's late-binding closures) caused the generators to yield from themselves, raising
   'ValueError: generator already executing'. This broke Daf.from_csv() for http(s):// and s3:// sources,
   both of which pass a real generator through this exact path -- not merely a theoretical/unused
   capability. Fixed by capturing the original iterator under a distinct name before reassignment.
- Fixed safe_regex_replace(): its own type hint allows `regex` to be passed as a list of patterns
   directly, but `.strip('"')` was called unconditionally before checking for list input, crashing with
   AttributeError on real list input. Fixed by checking the type first.
- Fixed write_buff_to_fp(): `s3path` was only assigned inside the `if buff:` block, but the `else`
   branch (taken when buff is falsy/empty) referenced it, raising UnboundLocalError.
- Fixed write_buff_to_fp()'s s3 branch: was doing `from . import s3utils`, but daffodil.lib.s3utils does
   not exist anywhere in this package (left over from the AuditEngine migration, where a much larger,
   ~2,600-line, AE-internal-dependency-laden s3utils.py presumably did exist). Replaced with two new
   minimal, self-contained daf_utils.py functions using plain boto3 (no AE-internal, pandas, or
   smart_open dependencies, consistent with daffodil's existing avoidance of pandas/numpy elsewhere):
   - `write_buff_to_s3path(s3path, buff, content_type)`: writes unconditionally, with a small retry
      loop for transient ClientErrors and a post-write existence-polling loop (kept as a safety net for
      occasional propagation delay, even though S3 PUTs are strongly consistent since Dec 2020).
   - `does_s3path_exist(s3path)`: existence check via a HEAD request.
   Also removed the `if_unmodified` parameter and the `write_buff_to_s3path_if_modified()` call it
   gated: the ETag-based "only write if changed" / local-mirror caching behavior is an application-level
   concern (e.g. AuditEngine's `DB.load_data()`/`store_data()` layer), not something daffodil core needs
   to implement itself.

### Known issues found, not yet fixed (flagged for a future round)
- add_trailing_columns_csv() (used by xlsx_to_csv()'s default add_trailing_blank_cols=True path) samples
   the first `num_rows` (default 3) rows via repeated next(reader) calls to estimate max column count --
   raises an uncaught StopIteration if the CSV has fewer than num_rows rows total. The function's own
   docstring already flags it as "@@TODO -- should be DEPRECATED".
- insert_irow(): if `row` is neither a list nor dict (including the default row=None), `row_la` is never
   assigned before use, raising UnboundLocalError. Intended None-row semantics not yet decided.
- append(): the dedicated KeyedList fast-path (`if self.hd == data_item.hd: lol.append(values())`) is
   unreachable -- the preceding `isinstance(data_item, (dict, KeyedList))` check already intercepts every
   KeyedList first, routing it through the slower generic record_append() path instead. Not a correctness
   bug (record_append() handles KeyedList correctly too via its own dedicated branch), just a missed
   optimization. May be related to recently-added dict-like operations on KeyedList; needs investigation
   before deciding whether the dedicated branch is still needed at all.

### Added (daf_pandas.py / daf.py)
- Added pytest test coverage for daf_pandas.py's standalone dtype-mapping helpers (57% -> 95%):
   pandas_dtype_to_python_type, python_dtype_to_pandas, pandas_dtype_dict_to_python,
   dtypes_dict_from_dataframe, and the use_csv=True + default= combination in _to_pandas_df.
- Added pytest test coverage for daf.py's construction/I-O cluster: __init__, copy, from_lot,
   set_dtypes/apply_dtypes, append/_basic_append, insert_dif_row, insert_irow's working paths, and the
   full CSV I/O round trip (from_csv for local/HTTP/S3 sources via mocking, from_csv_file, to_csv_file,
   to_csv_buff, buff_to_file). This batch also serves as an integration-level confirmation of the
   buff_csv_to_lol() streaming fix above, exercised through Daf.from_csv()'s actual HTTP/S3 generator path.

### Fixed (daf_pandas.py)
- Fixed pandas_dtype_to_python_type(): checked np.integer before np.timedelta64, but numpy considers
   timedelta64 a subtype of integer (it's stored internally as an integer count of time units), so every
   timedelta64[ns] column was silently misclassified as int instead of pd.Timedelta. Fixed by checking
   np.timedelta64 first; added a comment explaining why the order matters, to prevent future regression.
- Marked the "Unknown Pandas dtype" fallback breakpoint() in pandas_dtype_dict_to_python() as `#perm`
   (a deliberate diagnostic tripwire that should never fire, but needs visibility if it ever does).

### Changed (daf.py)
- Commented out (not deleted) daf_sum2()/daf_sum3() and the sum_da2()/sum_da3() functions they called:
   investigatory variants used to find out why daf_sum() (sum_da) was slow. The key finding -- comparing
   `value == ''` instead of `value is NULL` made the loop take ~10x longer, since Python implements '' as
   a singleton (like None), so the very fast `is` comparison works -- was already adopted into sum_da()
   and other NULL comparisons throughout daf.py. Preserved as a commented-out record of that
   investigation/finding for future reference, rather than deleted.

### Added (daf.py: __setitem__ / set_irows_icols)
- Added pytest test coverage for set_irows_icols() (the engine behind Daf.__setitem__ /
   `daf[irows, icols] = value`) across all its branches: single row/col, single row/no col, multi-row/no
   col, multi-row/single col, and multi-row/multi-col, each for scalar/list/dict value types.

### Fixed (daf.py: __setitem__ / set_irows_icols)
- Fixed set_irows_icols(): in three of its four `isinstance(value, (list, Iterable))` checks, a `dict`
   value was being silently swallowed by the broader Iterable check (dict is itself Iterable, just not a
   Sequence) before ever reaching its own dedicated `elif isinstance(value, dict):` branch below it. This
   caused two distinct symptoms depending on which branch was hit:
   - Multi-row, no column selection (e.g. `daf[[0,1],:] = {...}`): silent data corruption -- the dict
      object itself was assigned as a row's raw content (`self.lol[irow] = value`) instead of being
      mapped to columns via assign_record_irow().
   - Multi-row, multi-column (e.g. `daf[[0,1],[0,1]] = {...}`): integer-indexed lookup into the dict
      raised KeyError, caught by a bare `except Exception:`, which hit a `breakpoint() #perm ok` tripwire
      -- hanging indefinitely in any non-interactive context (scripts, tests, production).
   A fourth branch (multi-row, single column) had its dict-handling entirely commented out, meaning a
   dict value there hit the exact same hang via the same root cause.
   Fixed by (a) reordering every branch to check `isinstance(value, dict)` first, and (b) replacing the
   overly-broad `Iterable` check with `collections.abc.Sequence`, which correctly excludes dict, set, and
   generators while still covering list/tuple/range. The fourth branch's dict-handling was uncommented
   and given the same fix. The `#perm` breakpoint tripwires were left in place (per project convention,
   they remain valuable for catching genuinely unexpected cases) -- they're simply no longer reachable
   via this particular (now-fixed) misrouting.

### Known issues found, not yet fixed (added to the list above)
- set_irows_icols()'s `elif isinstance(value, type(self)):` branches (assigning a whole Daf as the
   value) appear to have pre-existing correctness problems independent of the dict/Sequence fix above:
   indexing a Daf by a single int (e.g. `value[source_row]`) returns a wrapped single-row Daf object, not
   a plain list of values, so `self.lol[irow] = value[source_row]` stores a Daf object as row content
   rather than its values. In the multi-row/multi-col branch the indexing also appears semantically
   inverted: `value[source_col]` uses a column-position index to select a *row* out of the Daf value
   (since indexing by a single int selects a row), which doesn't correspond to "the value for this
   column" at all. Not yet fixed -- needs a decision on what assigning a whole Daf as a __setitem__
   value is actually supposed to mean in each branch before attempting a fix.

### Added (daf.py: select_irows / select_icols / gkeys_to_idxs / col_to_la)
- Added pytest test coverage for select_irows(), select_icols(), gkeys_to_idxs() (and its
   krows_to_irows()/kcols_to_icols()/select_krows()/select_kcols() callers), and col_to_la().

### Fixed (daf.py: select_irows)
- Fixed select_irows(int, invert=True): `row_sliced_lol = self.lol` (not a copy) followed by
   `row_sliced_lol.pop(irows)` was destructively mutating the *original* Daf's lol in place --
   contradicting the function's own documented contract of being a non-mutating selection that returns
   a new instance. Fixed to build a fresh filtered list instead, correctly handling negative indices the
   way `list.pop()` does.
- Fixed select_irows(slice(...), invert=True): raised `TypeError: argument of type 'slice' is not
   iterable`, since a raw slice object doesn't support membership testing (`in`). Fixed by converting the
   slice to a range via daf_utils.slice_to_range() first.
- Fixed select_irows()'s list-of-ints fast path: `if len(irows) == len(self.lol): row_sliced_lol =
   self.lol` silently ignored reordering or repeated-index selections that happened to be the same
   length as the original (e.g. `select_irows([2,0,1])` returned rows unchanged instead of reordered).
   Fixed with a short-circuiting natural-order check (`all(irow == i for i, irow in enumerate(irows))`)
   that bails out on the first mismatch rather than doing full-length work regardless -- chosen
   specifically to stay cheap for large arrays (e.g. 500K+ rows) in the common case where the fast path
   doesn't apply.

### Fixed (daf.py: select_icols)
- Fixed select_icols(): `self.num_cols` was missing `()` (comparing a bound method object to an int,
   always False), making the analogous "selecting all columns, reuse self.lol directly" fast path
   permanently dead. Naively adding `()` alone would have introduced a *new* correctness bug (confirmed
   via direct testing): a same-length but reordered or repeated-index `icols` selection would then
   silently return data unchanged while still correctly reordering the column *labels* -- a label/data
   mismatch worse than the original dead code. Fixed with the same short-circuiting natural-order check
   as select_irows() above, which is correct for both possible cases.

### Fixed (daf.py: gkeys_to_idxs)
- Fixed gkeys_to_idxs()'s slice branch: previously built `range(gkeys.start or 0, gkeys.stop or
   len(keydict), ...)` and then looked up each resulting ordinal position *as if it were itself a key* in
   keydict (`keydict[gkey]`) -- which only coincidentally worked if keydict's keys happened to be the
   identity mapping `{0:0, 1:1, ...}`; for any realistic keydict (string keys, or non-sequential int
   keys) this raised KeyError immediately, including for plain numeric slices like `slice(0, 2)`.
   Clarified semantics: slice.start/.stop/.step are ordinal positions into keydict's iteration order (the
   same convention integer indices use elsewhere), never keys to look up -- for arbitrary keys there is
   no well-defined "next key" to support exclusive-stop slicing the way integer positions do, which is
   exactly why the existing `(start_key, stop_key)` tuple form (inclusive of stop_key) exists as the
   correct mechanism for genuine key-range selection. Fixed by treating slice bounds as plain ordinal
   positions (filling in 0/len(keydict)/1 defaults) and returning a `slice` object directly, mirroring
   how the Tuple branch already builds and returns a `slice`.
   Also documented (via comment) an architectural finding from tracing this function's callers: this
   slice branch is only reachable via the explicit select_krows()/select_kcols() method calls -- the
   `daf[...]` / `__getitem__`/`__setitem__` `[]` syntax (_parse_selectors) routes ANY slice object
   straight to the plain-index path without ever calling krows_to_irows()/kcols_to_icols()/
   gkeys_to_idxs() at all, regardless of whether the slice's bounds are ints or strings. So
   `daf['a':'c']` does not perform key-based slicing via `[]` (it errors trying to use the strings as
   literal list-slice bounds) -- only `daf.select_krows(slice(...))`/`daf.select_kcols(slice(...))`
   reach this code path.


### Added (daf.py: reduction operations)
- Added pytest test coverage for the reduction operations cluster: reduce(), sum_da(),
   count_values_da(), groupby(), groupby_cols(), groupby_reduce(), multi_groupby(),
   multi_groupby_reduce().

### Fixed (daf.py: reduce)
- Fixed reduce(by='row')'s default accumulator initialization: `if initial_da:` treated an
   explicitly-passed empty dict (`initial_da={}`) as falsy, silently overriding it with
   `dict.fromkeys(cols_iter, 0)` regardless. That default is itself sum_da-specific (0 is the correct
   additive identity for summing) and is incompatible with reduction functions needing a dict-based
   accumulator, such as count_values_da() -- whose own docstring example demonstrates exactly this usage
   (`my_daf.reduce(count_values_da, value_counts_dodi, cols_of_interest, ...)`, though that particular
   example is itself stale relative to reduce()'s current positional signature). The combination meant
   even the documented workaround (passing an explicit empty dict) silently failed, raising
   `AttributeError: 'int' object has no attribute 'get'` from inside sum_da() when count_values_da()
   tried to merge a dict-valued column against the leftover `0`. Fixed by checking
   `if initial_da is not None:` instead, so an explicitly-passed empty dict is honored.

### Added (daf.py: sorting, joining, transforming)
- Added pytest test coverage for calc_cols(), sort_by_colname()/sort_by_colnames(), split_where(),
   remove_dups(), apply_in_place(), annotate_daf(), update_by_keylist(), replace_in_columns(),
   join()/join_records().

### Fixed (daf.py: calc_cols)
- Fixed calc_cols(): for both include_cols and exclude_cols, `if isinstance(include_cols, str):
   include_cols = [include_cols]` was followed by an `else:` clause containing the actual filtering
   logic -- but that `else` paired with the `isinstance` check, not the outer `if include_cols:`, so
   passing a plain string (e.g. `include_cols='id'`) converted it to a list but then never applied any
   filter at all, silently returning all columns unchanged. Fixed by separating the str-to-list
   normalization from the filtering, which now always runs.

### Fixed (daf.py: split_where)
- Fixed split_where(): `true_daf`/`false_daf` were only constructed inside the non-`indirect_col`
   branch, so calling `split_where(..., indirect_col=...)` raised UnboundLocalError on the final
   `return true_daf, false_daf` -- the rows were correctly collected into true_lol/false_lol in both
   branches, just never wrapped into Daf instances when indirect_col was used. Fixed by moving the Daf
   construction after both branches, so it always runs.

### Fixed (daf.py: replace_in_columns)
- Fixed replace_in_columns(cols=None, ...): the keyfield-invalidation check at the end did
   `self.keyfield in cols`, but `cols` is the original parameter (which is None in the documented
   "process the entire array" case) -- not the already-resolved `target_indices` list -- raising
   `TypeError: argument of type 'NoneType' is not iterable`. Separately, even when cols was provided,
   this comparison checked a column *name* string against a list of column *indices*, which could never
   match. Fixed by checking the keyfield's own resolved column index against target_indices instead
   (preserving the original conservative "always invalidate" behavior for composite/tuple keyfields).
- Fixed replace_in_columns(): `replacement: Any = _MISSING` is a sentinel meant to detect whether the
   caller passed an explicit value (the same pattern already used correctly by `_to_pandas_df`'s
   `default: Any = _MISSING` parameter), but nothing in this function ever checked for it -- so omitting
   `replacement` (which the signature makes look optional) silently wrote the literal `_MISSING` sentinel
   object into the caller's data wherever a match was found, rather than raising a clear error. Fixed by
   raising ValueError if `replacement is _MISSING`.

### Added (daf.py: from_directory)
- Added pytest test coverage for from_directory() using pytest's tmp_path fixture (real files/
   directories on disk): recursive harvesting, field population, file_pat filtering, empty
   directories, and custom schema support. As a side effect, this also substantially increased
   coverage of daf_schema.py (12% -> 23%) and schemaclass.py (0% -> 70%), since from_directory()
   exercises the @schemaclass machinery directly.

### Fixed (daf.py: from_directory)
- Removed a stray `, 3` argument from a `print()` call (leftover from a logs.sts(msg, 3)-style call
   that was swapped to plain print() without removing the verbosity-level argument); it was being
   printed as a literal trailing "3" on every call.
- Fixed from_directory(recursive=False): os.listdir() returns subdirectory names alongside file
   names, and the non-recursive code path didn't distinguish them (unlike os.walk(), used in the
   recursive=True path, which already separates dirnames from filenames) -- so a subdirectory could
   show up as a row as if it were a file. Fixed by filtering out directory entries via
   os.path.isdir() before processing, matching os.walk()'s existing behavior of only ever yielding
   files (never bare directory entries) as rows in either mode. This also resolves the related
   confusion around `is_dir` being hardcoded to `0`: since directories are now never turned into rows
   in either mode, that hardcoded value is now actually consistent in both.

### Added (daf.py: long tail of smaller methods)
- Added pytest test coverage for the remaining smaller methods/properties: retmode/itermode,
   __contains__/__str__/__repr__/__format__, isin, to_value, to_klist, to_json, extend, drop_cols,
   set_cols, flatten, keys, select_where, dict_to_md, set_keyfield, _rebuild_kd/_build_kd/
   _get_keyval, update_row, diff_da, sum, the valuecounts_for_* family, set_icol/set_icol_irows/
   set_col_irows, apply_to_col, iloc, and the DafIterator/_IndirectRowView helper classes
   (daf.py: 83% -> 87%).

### Fixed (daf.py: long tail of smaller methods)
- Fixed keys(astype='view') with no keyfield set: `().keys()` doesn't exist (`()` is an empty
   tuple, not a dict), raising AttributeError. Fixed to use `{}.keys()`.
- Fixed drop_cols(): crashed with AttributeError when self.dtypes was None (the normal default for
   a Daf with no dtypes specified), since it unconditionally called `self.dtypes.items()`.
   Separately, even with dtypes populated, the filter condition was inverted and assumed
   self.dtypes's iteration order exactly matched self.hd's column order -- confirmed via direct
   testing that this kept the dtype of the column being *dropped* while discarding dtypes for the
   columns actually being kept. Fixed by filtering self.dtypes by column *name* against the
   retained columns, which is both correct and avoids the fragile positional-ordering assumption.
- Fixed _IndirectRowView.values()/items(): using `yield` later in the function body makes the
   whole function a generator (regardless of any earlier `return`), so `return self.row.values()`
   in the no-indirect-col branch was silently discarded -- calling a generator function always
   returns a generator object immediately, and `return <value>` inside one just ends iteration
   without producing a usable value, rather than evaluating that branch's intended return value.
   Fixed by using generator expressions for the indirect-col case instead of `yield` statements,
   so the early `return` actually works.
- Fixed set_keyfield(''): the `if not keyfield:` reset branch set `self.keyfield = ''` but didn't
   return immediately, falling through to `_is_keyfield_valid('')`, which evaluates `'' in self.hd`
   -- always False, treated as an invalid keyfield. So resetting the keyfield with
   silent_error=False raised KeyError unexpectedly, even though resetting to empty is the
   documented, valid "disable key lookups" case. Fixed by returning immediately after the reset.
- Fixed apply_to_col(): `self[:, col]` returns a Daf (a single-column sub-table), and iterating a
   Daf yields row dicts, not raw cell values -- so `map(func, self[:, col])` was calling `func`
   with a dict (e.g. `{'a': 1}`) instead of the actual value, raising TypeError for any numeric
   function. Fixed to use `self.col(col)` instead, which correctly returns a flat list of values.
- Fixed iloc(rtype='list'): called `self.to_list(irow=..., icol=...)`, but `to_list()`'s irow/icol
   parameters were removed in a prior refactor (left as "(removed)" in to_list()'s own docstring,
   with no commensurate update to this call site), raising TypeError. Fixed to return
   `list(self.lol[irow])` directly, consistent with how the dict/klist rtype branches each return
   the row in a different shape without needing to round-trip through a separate method.

### Added (schemaclass.py / daf_schema.py)
- Added pytest test coverage for schemaclass.py (96% -> 100%) and daf_schema.py (23% -> 100%):
   the @schemaclass decorator and SchemaBase (default_record, record_from, get_dtypes_dict,
   get_columns, get_pandas_dtypes_from_schema, validate_keys_debug, inheritance, instantiation
   prevention, validation errors), and daf_schema.py's _apply_schema/_attach_schema/
   _default_record (both schemaclass and schema_daf code paths, including all six dtype-name
   aliases, the keyfield-from-attrs and keyfield-from-__keyfield__ paths, and the various
   error/edge cases for each).

### Changed (daf_schema.py)
- Removed ~210 lines of duplicate code from daf_schema.py: the entire SchemaBase class and
   schemaclass() decorator from schemaclass.py had been pasted in verbatim after a `#====`
   separator comment, immediately following daf_schema.py's own (unrelated) _apply_schema/
   _attach_schema/_default_record functions -- an apparent accidental copy-paste left over from
   development. Confirmed via a repo-wide search that nothing imports schemaclass/SchemaBase from
   daf_schema specifically (daf.py and everything else import them from the canonical
   daffodil.lib.schemaclass module instead), and that the duck-typed `__is_schemaclass__` marker
   check in _attach_schema works correctly regardless of which copy of the decorator created a
   given schema, so this duplicate block was provably dead code, not a behavior change to remove.
   Also removed the KeyedList and daf_utils imports that were only used by the removed block.

### Fixed (daf_schema.py)
- Fixed _apply_schema()'s schema_daf branch: setting `cols` (from the schema's 'Name' column)
   and setting `dtypes` (from its 'dtype' column) are independent operations that should both
   happen when a schema_daf provides both columns (the normal, expected case) -- but they were
   structured as `if`/`elif`, so the dtypes block was silently skipped entirely whenever the cols
   block ran. Confirmed via direct testing that `Daf(schema=schema_daf)` left `dtypes` as None
   even when the schema_daf had a complete 'dtype' column. Fixed by changing `elif` to a second,
   independent `if`.

### Added (daf.py: final coverage pass)
- Added pytest test coverage for the remaining small/easy methods identified in a final sweep of
   daf.py (87% -> 88%): keys()'s astype error path, set_keyfield()'s force_kd_rebuild branch,
   flatten()'s bool-to-int and no-op branches, from_lod()/from_lot()'s edge cases, find_replace(),
   apply() (all by= modes and keylist filtering), kcols_to_icols(), _basic_get_record(),
   sum_dodis(), groupsum_daf(), and multi_groupsum().

### Known issues found, not yet fixed -- flagged for review (no supervision available to confirm
intent, so left as-is rather than guessing)
- flatten(): `use_pyon = True` is hardcoded immediately after the `use_pyon` parameter,
   unconditionally overriding whatever the caller passes -- so `use_pyon=False` (the JSON-encoding
   branch) is currently unreachable. Per clarification: flatten() is largely deprecated now that
   to_csv_buff()/to_csv_file() automatically flatten complex cell types via PYON encoding as part
   of CSV writing (a significant performance win over a separate flatten-then-write pass), so this
   is not worth fixing -- left as-is.

### Changed (daf.py: set_cols)
- Simplified set_cols(): previously attempted to "repair" the keyfield after a column rename via
   `self.hd.get[self.keyfield]` (a syntax bug -- subscripting a bound method object instead of
   calling it -- that always raised TypeError, silently caught, always resetting the keyfield to
   '' regardless). Per explicit decision: column renaming is rare, and the keyfield should always
   be reset to '' on rename rather than auto-remapped, requiring the caller to explicitly call
   set_keyfield() afterward with the correct new name. Removed the broken repair attempt entirely
   and documented this behavior clearly in the docstring. Updated the existing
   test_set_cols_repair_keyfield's comment (in test_daf.py) to match, since it had asserted the
   same `keyfield == ''` outcome all along but with a misleading "repairing" comment.

---

## [0.5.12] - (not formally released)
### Added
- .from_md() method added to allow round-trip of daffodil arrays through markdown representation.
   - changed footer syntax so it does not need markdown escaping, uses standard syntax for easy parsing.
   - footer contains, rows, cols, keyfield, name, schema
   - will detect abbreviated markdown table output and raise an error.
- added 'astype' parameter to .col_to_la() operator.
- Added schemaclass record_from() method, used like: da = schema.BifSchema.record_from(other_da)
- Added indirect_col to .col() and .col_to_la()
- Added indirect_col to select_where() split_where()
- Added group_where() method to allow arbitrary grouping including fanout grouping.
   - Added tests for group_where()
- Introduced KeyedIndex class for future porting to rust. Also added tests.
- Added \_basic_append() to streamline appends when it is known that the hd is compatible.

### Changed
- Changed some uses of [:, col_name].to_list() to .col(col_name)
- Implemented lazy kd generation rather than respecting it. {} indicates invalidated.
- .kd no longer should be accessed by user code. Changed variable to .\_kd.
- .keys() now has astype parameter which can be 'list' or 'view' to give access to keys and 
    allowing fast containment tests.
- modified README to reflect dynamic building of kd. This should ease use while still being backward compatible.
    however, any direct use of .kd is now banned. Use .keys() to access the keys either as list or view.


### Fixed
- Corrected .apply_dtypes() so it would allow that not all columns defined by the schema exist, if columns are already defined.
    However, dtypes must include all columns that do exist.
- Added removal of empty list items when read from csv if there are blank lines at the end of the file. Edge case.
- Fix edge case selecting rows by empty list or iterator.
- Fixed edge case of single integer row specified as 0 due to using 'if not' construction.
- Fixed pyproject.toml to include dependencies required for testing.
   - to test, use `uv run --group dev pytest -s`
- Fixed a number of datatype conversion issues in to_pandas() and from_pandas()
- Went back to desired_type == int instead of is int, because == will include int, int64 etc.
   - Included a comment but use an noqa override for ruff linter.
- Fixed error in indexing that required rebuilding kd and it was not done.
- changed == '' to is NULL, with NULL defined as '' (performance improvement)
- Use npao for transpose operation. (performance improvement)

- created a large set of tests for indexing modes, which resulted in improvements for edge cases and some api/ui changes.
   
   1. Changed the parameters for .to_list() and .to_value() to use indexing and then chaining to those functions. not as performant but simpler interface. 

   2. Fixed the README about using tuples of strings as inclusive ranges, as None is required as the first parameter to start at the beginning through the named item. 

   3. Kept shape() as a method and not a property as it is somewhat costly to calculate. 

   4. Improved parsing of indexing with a common section for both getitem and setitem. 

   5. Fixed a number of edge cases when selections were no columns, no rows, etc. with rules for how these would be handled without needing to make every nonsensical change supported. 

   6. disabled using kd and set keyfield='' if the keyfield column was dropped. 

   7. narrowed the application of .to_value() to handle a single-value only, from location 0,0 in a rudimentary daf array. 

   8. Return [] if .to_list() is used on an array with no columns or rows. 

   9. Added SchemaBase in schemaclass.py to allow mypy to understand dynamically added methods when using the @schemaclass decorator. 

   10. removed tests for using indices inside .to_list and .to_value. 

   11. Fixed improper tests for tuple string ranges. 

   12. Added test file 'test_daf_indexing2.py' with about 100 tests for indexing.

- Introduce T_cs, T_ca, T_ci, T_ma. Start phasing them in to avoid mypy errors.
- use Iterable, Iterator, keysView from collections.abc
- deprecate 'align' in md table functions.
- Fixed edge case in keyedlist.py, when instanciating with KeyedList().

- Add option of using daffodil tables as schema instead of schemaclass.
- Moved .attach_schema from daf to daf_schema.py
- Added .apply_schema which is like apply dtypes.

- .default_record now properly a method of daf rather than schema.
- Added option to .sort() to do a shallow copy of lol and kd in so a copy can be independently sorted.
- Added .from_directory to get a daf table of a directory of file system
- Improved .from_md() with improved scanning for the first table in the passed block.
- Added cleanup of colnames by removing md or wiki markup.
- Added crud in schemaclass.py to allow inheritance.

---

## [0.5.11] - 2026-02-02

### Added
- Added optional schema class support for declarative column definitions using Python type annotations.
- Added default_record() method to create new records initialized from schema defaults.
- Added text to README.md to explain the new schema approach.
- Added decorator approach to schemaclass and placed in a separate file schemaclass.py with schemaclass_README.md in `/lib`
- Added method .attach_schema(self, schema: type) to allow schema to be attached after reading the file. 

### Changed
- daf.keys() now has optional parameter 'silent_error' which defaults to True.
   - If True and there is no keyfield set, do not produce KeysDisabledError and instead return []
   - If False, raise KeysDisabledError
- added compatibility for astype_la() to use type values like int,str,float,bool as well as `int`,`str`, etc.
- apply_dtypes() now enforces exact match by default
   - Use silent_error=True to allow mismatches.
   
### Fixed
- (add entries here)

---
## [0.5.10] - 2025-09-28

### Added
- `CHANGELOG.md` added to repo; moved log from `daf.py` to this file and reformatted to md.
- `ROADMAP.md` added to repo; moved TODO plans from `daf.py` to that file and reformatted to md. 
- `manifest.in` added to repo to force inclusion of these documents into the distribution.

### Fixed
- Neglected to bump `pyproject.toml` version in 0.5.9.
- Removed change log and roadmap todos from `daf.py`

---

## [0.5.9] - 2025-09-22
### Added
- `attrs` argument on instance creation (parity with pandas’ `.attrs`).
- `.to_donpa()` to convert selected columns to a dict of NumPy arrays (columnar ops).
- Tests for JSON round-trip including `disp_cols` and `attrs`.

### Changed
- Initialize `disp_cols` robustly: accepts `None`/`list`/`tuple`; emits `[]` if not provided.
- Improved `.to_json()` / `.from_json()` to handle `disp_cols` and `attrs`.

### Fixed
- Introduce `KeysDisabledError` when key lookups occur without `keyfield` set.
  - If `keyfield` **is** set but key missing: raise `KeyError` (when `silent_error=False`).

<pre>
    v0.5.9  (2025-09-22)
            Add 'attrs' as argument to creation of the array instance. This mimics Pandas usage.
            Improve initialization of disp_cols so it can be None, List or tuple and emitted as [] if not provided.
            Added .to_donpa() which would convert specified columns to dict of numpy arrays of each col, where dict keys are col names.
            This creates a simple pandas-like object where NumPy column operators can be easily applied.
            Improved .to_json() and .from_json() to handle disp_cols and attrs. Fixed tests.
            Added use of KeysDisabledError whenever a function attempts to lookup a row when keyfield is not set.
                In contrast, if keyfield is set and key not found, then KeyError if silent_error False.
                Fixed unit tests.
</pre>
---

## [0.5.8] - 2025-07-17
### Fixed
- Flattening now converts textual containers to actual objects (`'[]' → []`, dict, tuple).

### Changed
- Clean up imports to satisfy `pyflakes`.
- Correct SQL index-name creation in `sql_utils` / `daf_sql` (after quoting change).
- Handle empty `self.lol` edge case in `select_irows()`.

<pre>
    v0.5.8  (2025-07-17)
            Correct flattening operation so that strings like '[]' are converted to []. Same for dict and tuple.
            Cleaned up a number of imports due to pyflakes linting
            Fixed creation of index name in sql_utils and daf_sql due to change in name escaping.
            Fixed edge case in select_irows() if self.lol is empty.
            RELEASED AS v0.5.8
            
</pre>

---

## [0.5.7] - 2025-05-16
### Added
- `.replace_in_columns()` to replace values across specified columns.
- `.apply_colwise()` (apply per row, store in `target_col`).
- Options in `.to_pandas_df()`:
  - `use_donpa` path for faster conversion via NumPy vectors.
  - Default/NA handling (''/None → chosen default). Tests added.
- Expanded tests for `.to_list()` default handling.

### Changed
- SQL identifier handling: switch from escaping to quoting. **Breaking** for old SQL tables.
- Adopt `pytest`; remove ad-hoc `sys.path.append()` uses.
- Use `_MISSING` sentinel internally for default handling in `.to_donpa()`, `.to_list()`, `.to_pandas_df()`.

### Fixed
- Various improvements in `from_pdf()` for complex layouts.

<pre>
    v0.5.7  (2025-05-16)
            Change SQL identifier escaping from character escaping to quoting.
                Note, this is a breaking change if prior SQL tables are encountered.
            Move to using pytest. Removed some sys.path.append() statements as a result.
            Improve default handling in to_donpa() to allow None to be the default by using a sentinel _MISSING
            Improved .to_list() default handling to allow None as the default, by using sentinel _MISSING.
            Improved .to_pandas_df() default handling
            Added option use_donpa to .to_pandas_df() to improve convertion to pandas df by first converting columns to numpy vectors.
            Added tests for .to_list() including various options.
            Added .replace_in_columns() to Replace all values in `find_values` with `replacement` for the specified columns.
            Added .apply_colwise() Apply a function to each row and store the result in target_col
            Improved .to_pandas_df() 
                to provide an option to convert first to donpa (dict of numpy array)
                to handle conversion of '' or None to a default such as NA or 0.
                improved tests.
            Improved from_pdf() to support complex conversion scenarios.
            RELEASED AS v0.5.7
            

</pre>
---

## [0.5.6] - 2025-03-04
### Added
- `from_lot()` classmethod.
- `join()` with tests; `tag_other` option to tag other columns (supports chained joins).
- `from_pdf()` to parse multi-page tables.
- `name` parameter for `from_lod()`, `from_csv_buff()`, `join()`, `clone_empty()`.
- `omit_other_cols` in `derive_join_translator()`; simplified translator table.
- `.attrs` dict on the core instance for descriptors/metadata handoff.
- `from_csv()` (local, S3, http/https) with streaming improvements.

### Changed
- Indexing: 25 tests; correct parsing of tuple-of-strings for `krows`/`kcols`.
- `concat()` uses deep copy to avoid shared-frame mutation issues.
- Use context manager for local file closure in `from_csv()`.
- Prefer `daf_utils` (deprecate bare `utils` naming).
- Raw docstring strings in `derive_join_translator` to avoid escape complaints.


<pre>
    v0.5.6  (2025-03-04)
            Added from_lot() class method. Perhaps these can be unified in main init function by examining type of the data passed.
            Added join() method, including unit tests.
            Added from_pdf() class method, used to parse PDF files with table structure across multiple pages.
            Added name argument to from_lod()
            Added name argument to from_csv_buff()
            using raw docstring format to avoid complaints of escape characters in derive_join_translator.
            Added 'tag_other' boolean parameter to tag all other column names during join, to support chained joins.
            Simplified translator_daf table so it is easier to produce by hand and use across many tables being joined.
            Added name argument for join() method, to provide the name of the resulting joined instance.
            Improve unit tests for derive_join_translator
            Added 25 tests for various indexing modes. 
                corrected parsing of tuple of strings for krows and kcols.
            Added name argument for clone_empty() method.
            Added omit_other_cols parameter for 'derive_join_translator' method. 
                this can probably displace the "shared_fields" parameter.
                Fixed omit_other_cols so it could be properly omitted.
            concat (which is called by append if a daf array is appended) was not using deep copy when 
                copying in the frame, and this became a real mess. Added copy with deep to concat.
            Added from_csv() which will load csv to Daf from local files, s3 path or http/s path.
                improved operation of streaming from file to avoid buffer recopying.
            from_csv_buff() still exists for those times when a buffer or file-like object already exists.
            Added .attrs dictionary to core dataframe instance definition to allow for descriptors to be 
                provided to users of the dataframe, esp. between when the daf array is defined and built 
                and when it is used and modified.
            Improve file closure by using context manager in from_csv() for local file usage.
            Deprecated use of utils instead of daf_utils.
            
</pre>

---

## [0.5.5] - 2024-12-27
### Added
- Demo `daf_crm.py`; `op_import()`.
- `get_csv_column_names()`, `precheck_csv_cols()`, `compare_lists()` (refactor from `daf_utils`).
- `ops_daf` for operation runners (docstring-driven).
- `default_type` in `apply_dtypes` for unspecified columns.
- Better CSV preprocessing (commented lines with embedded newlines).
- `__contains__` for key existence (`if key in my_daf:`; requires `kd`).
- Indirect-column support and `sparse_rows` in reductions; revised `daf_sum()`.
- `astype` argument in `.to_list()` / `.to_value()`.
- Standardize on **PYON** instead of JSON for cell serialization.
- SQL benchmarks scaffolding (`daf_sql.py`, `create_index_at_cursor()` in `daf_benchmarks.py`).

### Changed
- Improve README examples and memory benchmarking (`objsize`).
- `set_keyfield` is a no-op on empty frames.
- Revised `.sum_da()` per user feedback.
- `apply_in_place(by="row_klist")` mutates row in place; rename kwarg `keylist` → `rowkeys`.

<pre>
    v0.5.5  (2024-12-27)
            add daf_crm.py as demonstration.
                add op_import() 
            add get_csv_column_names() as refactoring in daf_utils for reading csv.
            precheck_csv_cols()
            compare_lists() -- imported from daf_utils
            Introduce ops_daf for running operations, can also use in audit-engine.
                operation descriptions taken from docstring.
            added 'default_type' to apply_dtypes for any cols not specified in passed dtypes.
            Improved preprocessing of csv file when line is commented out and embedded newlines exist in the line.
            Improved Daf.from_lod() by using columns in dtypes dict if provided instead of relying only on first record of lod.
            
            Added indexing with range and T_lor (list of range) types, for both column and row indexing.
            Added __contains__ method to allow " if key in my_daf: " to test if a given key exists. Requires kd exists.
            revised .sum_da() based on feedback from user group.
            Improve formatting of README.md to include tables of examples.
            improve daf_benchmarks.py to use objsize instead of pympler to evaluate memory use.
            Corrected set_keyfield in daffodil to do nothing if daf is empty.
            Added 'sparse_rows' to reduction 'by' type using an indirect_col.
            Improve daf_sum() to support indirect_col.
            Revised apply_in_place to support by='row_klist'. Func will modify row_klist and that will modify the array.
                Changed name of keyword parameter in apply_in_place() from keylist to rowkeys to avoid confusion.
            added astype parameter for to_list() and to_value()
            Introduced standardization around PYON instead of JSON:
                - Easier to convert esp. during serialization using csv.writer().
                - Compatible with more Python data types.
                - Still easy to convert to JSON.
            Copied function create_index_at_cursor() for sql tables in daf_benchmarks.py
            Added daf_sql.py mainly to support benchmarks at this point.
            This will be the last release before sql enhancements.
            
</pre>

---

## [0.5.4] - 2024-07-02
### Added
- `sort_by_colnames()` (+ `daf_utils.sort_lol_by_cols()`).
- `omit_nulls` in `.to_list()`.
- `annotate_daf()` to join columns from another table via mapping.
- `value_counts_daf()` totals via `.to_list()`.

### Changed
- Internals: `klist.values` → `._values` to avoid property ambiguity.

<pre>
    v0.5.4  (2024-07-02)
            Add sort_by_colnames(self, colnames:T_ls, reverse: bool=False, length_priority: bool=False)
                Add daf_utils.sort_lol_by_cols()
            Add argument 'omit_nulls' to .to_list() method.
            Change references to klist.values to ._values to avoid amiguity with property getter and setters.
            Add annotate_daf(self, other_daf: 'Daf', my_to_other_dict: T_ds) to effectively join two tables.
            Fix value_counts_daf() by adding .to_list for total.

</pre>
---

## [0.5.3] - 2024-06-28
### Added
- `__format__` to enable `{:,...}` etc. (delegates to `.to_value()`).
- Alias `value_counts()` for `valuecounts_for_colname()` (pandas parity).
- `.to_klist()`, `.iter_list()`; extend `.iloc` to `klist` and list rtypes.
- `.assign_col()` can insert new columns.

### Fixed
- `insert_col_in_lol_at_icol()` off-by-one at “append” boundary.
- Create single-col `lol` when needed for empty input.
- `num_cols()` more robust (samples first rows).
- KeyedList supports empty init; rename `values` to `._values`.

### Changed
- More iterable support in row/col selectors.
- `remove_dups()` returns unique and duplicated sets (by `keyfield`).
- Formatting: bare `{daf}` prints summary (requires more than `{daf:}` to trigger formatting).


<pre>
    v0.5.3  (2024-06-28)            
            added tests:
                flatten()
                to_json()       not completely working.
                from_json()     not completely working.
            added __format__ to allow use of {:,} and other f-string formatting. Invokes .to_value()    
            added alias for valuecounts_for_colname() to value_counts() to match Pandas syntax.
            
            extend .iloc to support klist and list rtypes.
            Added .to_klist() to return a record as KeyedList type.
            extended .assign_col() to insert a column if the colname does not exist.
            Enhanced KeyedList() to allow both args to be None, and thus initialize to empty KeyedList.
            insert_col_in_lol_at_icol(): 
                fix bug if icol resolves to add a column. --> '>' changed to '>='
                allow empty lol and create a lol with one column if col_la exists.
            Add .iter_list() to allow iteration over just lol without cols defined.
            fixed __format__ so unadorned daf name prints summary. It takes more than {daf:} in fstring to cause formatting.
            Improved robustness of num_cols() to check first few rows.
                TODO: It will probably be better to keep a value of the num cols and not calculate evertime.
            changed name of values in KeyedList to _values and created accessors.
            added support for Iterables passed for row and col selection.
            Added method "remove_dups()" which returns unique records and duplicated records based on keyfield.
            Changed operation of assign_col to append col to right if colname not exist.
            
            worked around error in Pympler.asizeof.asizeof() function, used in daf_benchmarks.
                this appears to be resolved in future updates of pympler.
                

</pre>
---

## [0.5.2] - 2024-05-30
### Added
- `.iter_dict()` and `.iter_klist()`. Mutating `KeyedList` mutates underlying `lol`.

### Fixed
- Correct `slice_len` for column assignment from another column (clarifies nested list vs scalar intent).

<pre>
    v0.5.2  (2024-05-30)
            Added .iter_dict() and .iter_klist() to force iteration to produce either dicts or KeyedLists.
                Producing KeyedLists means the list is not copied into a dict but can be mutated and the lol will be mutated.
            Correct calculation of slice_len to correct column assignment from another column
                This may still have some ambiguity if a nested list structure is meant to be assigned to an array cell.
                    collist = my_daf[:, 'colname'].to_list()    # this will return a list, but sometimes of only one value.
                    my_daf[:, 'colname2'] = collist             # there is ambiguity here as to whether the list with one
                                                                # item should be placed in the cell or if just the value.
                
</pre>

---

## [0.5.1] - 2024-05-25
### Changed
- Dependency ranges relaxed in `pyproject.toml`.
- Upgrade to Python 3.11 and latest libs; using `venv311`.

<pre>
    v0.5.1  (2024-05-25)
            changed dependencies in pyproject.toml so they would allow newer versions.
            Upgraded to Python 3.11 and upgraded all libraries to the latest.
            Using venv311
    
</pre>
---

## [0.5.0] - 2024-05-23
### Added
- `split_where()` returns `(true_daf, false_daf)`.
- `multi_groupby()`, `reduce_dodaf_to_daf()`, `multi_groupby_reduce()`.
- `KeyedList` data type (dict-like API backed by list).
- `_build_hd()`, `to_json()`, `from_json()`, and same for `KeyedList`.
- `.strip()`, `.num_rows()`, `.to_value(flatten=…)`.

### Changed
- Remove `_da` suffix across many methods (prep for KeyedList).
- Behavior: `.select_record()` returns `{}` when frame is empty.
- Join/tagging, index handling, negative index tests, indirect-col handling in reductions.

<pre>
    v0.5.0  (2024-05-23)
            Added split_where(self, where: Callable) which makes a single pass and splits the daf array in two
                true_daf, false_daf.
            Added to Daffodil multi_groupby(), reduce_dodaf_to_daf() and multi_groupby_reduce()
            Added class KeyedList() to provide a new data item that functions like a dict but is a dex plus list.
                can result in much better performance by not redistributing values in the dict structure.
                This is not yet integrated into daffodil, but should be.
                
            Removed '_da' from many Daffodil methods and for keyword parameters, to allow future upgrade to KeyList.
                select_record_da()      -> select_record()
                record_append()
                _basic_get_record_da    -> _basic_get_record
                assign_record_da()      -> assign_record()
                assign_record_da_irow   -> assign_record_irow
                update_by_keylist()
                update_record_da_irow   -> update_record_irow
            changed test_daf accordingly.
                
            Added _build_hd() to consistently build header dict structure.
            Added to_json() and from_json() methods to allow generation of custom JSONEncoder.
            Changed nomenclature in KeyedList class from dex to hd.
            Added from_json and to_json to KeyedList class to allow custom JSONEncoder to be developed.
            
            select_record() silently returns {} if self is empty.
            mark chunks saved to local_mirror if not use_lambdas
            
            fixed _itermode vs. itermode.
            Added .strip() method.
            correct icols when providing a single str column name, and when column names have more than one character each.
            Added 'flatten' in '.to_list' method which will combine lol to a single list.
            Added .num_rows() which will more robustly calculate the number of rows in edge cases.
            Fix unflattening issue discovered when running edge_test_utils.py.
            Updated documentation to reflect new approach to dtypes and flattening.
             

</pre>

---

## [0.4.4] - 2024-05-02
### Added
- `daf_utils.safe_eval()`.

### Fixed
- Unflattening for f-string-flattened values (tuples & friends).
- `set_dtypes()` bug.

<pre>
    v0.4.4  (2024-05-02)
            Improved unflattening to handle f-string type flattening, including tuples.
            fixed bug in set_dtypes()
            added daf_utils.safe_eval()
            
</pre>
---

## [0.4.3] - 2024-05-02
### Changed
- Back to `pyproject.toml`; drop `setup.py`. Lambdas now import Daffodil.

<pre>
    v0.4.3  (2024-05-02)
            Went back to pyproject.toml approach. Eliminated setup.py.
            Lambdas now importing daffodil

</pre>

---

## [0.4.2] - 2024-05-01
### Packaging
- Attempted `flit` (TOML parse issues), reverted to `setuptools`, reorganized to `daffodil/src`.
- For editable installs, set `PYTHONPATH` to `daffodil/src`.
- Import modules as `daffodil.lib.daf_utils`, etc.

<pre>
    v0.4.2  (2024-05-01)
            Modified packaging for package distribution on PyPI to hopefully make it compatible with installing into AWS Lambdas.
                Tried to use pyproject.toml and flit, but flit has poor toml parsing it seems, and could not find a suitable toml file.
                Went back to setup.py and setuptools, but reorganized files into daffodil/src folder which will be included in the distro.
                To use --editable mode for local development, must set PYTHONPATH to refer to the daffodil/src folder.
                In that folder is daffodil/daf.py and daffodil/lib/daf_(name).py 
                To import supporting py files from lib, must use import daffodil.lib.daf_utils as daf_utils, for example.
                
</pre>

---

## [0.4.1] - 2024-04-30
### Changed
- `apply_dtypes(from_str=…)` (rename from `initially_all_str`).
- `set_dict_dtypes(dtypes=…)` now modifies in place; handles `dtypes={}`.

<pre>
    v0.4.1  (2024-04-30)
            fixed tests to reflect changes to type conversion paradigm.
            Changed apply_dtypes parameter 'initially_all_str' to 'from_str'
            fixed set_dict_dtypes() in the case of dtypes = {}; Changed parameter to 'dtypes' for uniformity.
            set_dict_dtypes() now also modifies types in-place.
            
</pre>

---

## [0.4.0] - 2024-04-30
### Added
- Better dtype support: `apply_dtypes()`, `flatten()`, `.copy(deep)`.
- `convert_type_value()`, `unflatten_val()`, `json_decode()`, `validate_json_with_error_details`, `safe_convert_json_to_obj`.
- Benchmarks: disable GC during timing; add references; switch to `pyproject`.

### Deprecated
- `unflatten_cols()` → `apply_dtypes()`
- `unflatten_by_dtypes()` → `apply_dtypes()`
- `flatten_cols()` → `flatten()`
- `flatten_by_dtypes()` → `flatten()`
- Functions for `hdlol` type (precursor to Daffodil).

<pre>
    v0.4.0  (2024-04-30)
            v0.4.0 Better dtypes support; apply_dtypes(), flatten(), copy()
            added disabling of garbage collection during timing, getting more consistent results, but does not explain anomaly.
            Improved philosophy of apply_dtypes() and flatten()
                Upon loading of csv file, set dtypes and then use my_daf.apply_dtypes()
                Before writing, use my_daf.flatten() to flatten any list or dict types, if applicable.
                
            apply_dtypes() now handles the entire array, and will skip str entries if initially_all_str is True.
                
                unflatten_cols()        DEPRECATED. use apply_dtypes()
                unflatten_by_dtypes()   DEPRECATED. use apply_dtypes()
                flatten_cols()          DEPRECATED. use flatten()
                flatten_by_dtypes()     Renamed:    use flatten()
                
            added optional dtypes parameter in apply_dtypes() which will be used to initialize dtypes in daf object and
                use it to convert types within the array.
            Changed from la type to interable in reduce()
            added disabling of garbage collection in daf_benchmarks.py
            deprecated functions dealing with hdlol type which was a precursor to daf.
            added convert_type_value() to convert a single value to a desired type and unflatten if enabled.
            removed use of set_dict_dtypes from apply_dtypes() and instead it is done on the entire daf array for efficiency.
            added in daf_utils.py unflatten_val(), json_decode(), validate_json_with_error_details, and safe_convert_json_to_obj.
            Added .copy(deep:bool) method to match pandas syntax.
            Added reference to 1994 workshop in flatten() method docstr.
            Changed packaging code from setup.py approach to pyproject, but still not able to correctly import in Lambdas container.

</pre>

---

## [0.3.0] - 2024-04-14
### Added
- `fmt` parameter (Content-Type metadata when saving).
- `CODE_OF_CONDUCT.md`.

### Changed
- `__init__.py` import helpers (later removed).
- Performance: `daf_sum()`, `reduce()`, `sum_da()` via try/except.

<pre>
    v0.3.0  (2024-04-14) 
            Added fmt parameter so file are saved with proper Content-Type metadata.
            Added 'from .Daf import Daf' to __init__.py to reduce level.  Eventually removed this.
            Added CODE_OF_CONDUCT.md
            Improved performance of daf_sum(), reduce() and sum_da() by avoiding explicit comparisons and leveraging try/except.
            Improved daf_benchmarks.py to help to diagnose nonperformant design of sum_da().
            Added basic indexed manipulation to daf_demo.py
            Changes due to the suggestions by Trey Hunner.
</pre>

---

## [0.2.2] - 2024-03-?? 
### Changed
- Repo layout aligned with Python norms:
  - `from daffodil.daf import Daf`; `daf.py` at top-level; helpers in `lib`.
- Added: `narrow_to_wide()` and `wide_to_narrow()`.

<pre>
    v0.2.2  Changed the file structure to be incompliance with Python traditions.
                user can use 'from daffodil.daf import Daf' and then Daf()
            Moved daf.py containing class Daf to the top level.
            put supporting functions in lib.
            added narrow_to_wide and wide_to_narrow methods.
</pre>

---

## [0.2.0] - 2024-02-28
### Added
- Multi-column `groupby`.
- `omit_nulls` in `col()/col_to_la()/icol_to_la()/valuecounts_for_colname()`.
- `groupby_cols_reduce()`, `sum_np()`, `to_cols_dol()`, `buff_to_file()`.
- Numerous tests (init, `set_cols()`, `set_keyfield()`, selection, calc, pandas I/O, Excel, etc.).

### Changed
- Indexing/refactors for `__getitem__/__setitem__`; bug fixes; Windows/mac/Linux demo; `num_cols()`; name Pydf → Daffodil.
- Removed several legacy helpers and key-mutating variants (see commits).

<pre>
    v0.2.0  (2024-02-28) 
            Copied related code to Pydf repo and resolved all imports. All tests running.
            Added option of appending a plain list to daf instance using .append()
            Added 'omit_nulls' option to col(), col_to_la(), icol_to_la(), valuecounts_for_colname()
            Added ability to groupby multiple cols
            in select_records_daf(self, keys_ls: T_ls, inverse:bool=False), added inverse boolean.
            Started to add selcols for zero-copy support.
            Added _num_cols() 
            Added unit tests.
            Add groupby_cols_reduce() and sum_np()
            Fixed bug in item setter for str row.
            Added demo of making list of file in a folder.
            groupby_cols_reduce() added, unit tests added. Demo added.
            Fix demo to run on windows, mac or linux.
            Add produced test files to gitignore.
            changed _num_cols() to num_cols()
            removed selcols_ls from class. 
            Pulled in from_csv_file()
            Added buff_to_file()
            improved is_d1_in_d2() by using idiom in Python 3.
            moved sanitize_cols to set_cols() method.
                1. read with no headers.
                2. pull off first row using indexing (could add pop_row())
                3. set set_cols with sanitize_cols=True.
            Removed:
                unflatten_dirname() <-- remove? (YES)
                flatten_dirname() <-- remove?
                cols_to_strbool() <-- remove?
                insert_icol()
                    keyfield <-- remove!  use set_keyfield instead
                insert_col() (removed keyfield)
                from_selected_cols <-- remove! but check for usage! (use my_daf[:, colnames_ls]
           Refactor get_item and set_item
                started this but complexity not that much better.
                Redid it again after discussion with Jeremy which was helpful.
            
            
            tests added
                initialization from dtypes and no cols.
                set_cols()
                set_keyfield()
                daf_utils.is_d1_in_d2()
                improved set_cols() to test sanitize_cols flag.
                .len(), num_cols(), shape(), len()
                row_idx_of()   (REMOVE!)
                remove_key -- keyfield not set. (remove)
                get_existing_keys
                select_record -- row_idx >= len(self.lol)
                _basic_get_record -- no hd, include_cols
                select_first_row_by_dict    
                select_where_idxs
                select_cols
                calc_cols
                    no hd defined.
                    include_cols > 10
                    exclude_cols > 10
                    exclude_types
                insert_col()
                    colname already exists.
                from_lod() with empty records_lod
                to_cols_dol()
                set_lol()
                from_pandas_df()
                to_pandas_df()
                from_dod()
                to_dod()
                from_excel_buff()
                
            Added:
                to_cols_dol()
            Moved code for __getitem__ and __setitem__ to 'indexing' and 
                used method equating to introduce them in the class.
            Moved to_pandas and from_pandas to file daf_pandas.py
            Changed constructors from_... from staticmethods to classmethods
            move toward deprecating remove_key() remove_keys()
            add silent_error in gkeys_to_idxs and trigger error if not selected.
            Handle missing keyfield, hd, kd, in inverse mode.
            Added to_value(), to_list(), to_dict() and tests.
            Tested negative indexes in []
            retmode attribute and constructor parameter to set 'val' vs. 'obj' return value.
            moved __getitem__ and __setitem__ back to main class now that they are vastely reduced in complexity.
            Name change from Pydf to Daffodil and resolve issues.

</pre>

---

## [0.1.x] - 2024-02-??
### Added
- Begin separate package; move comment text to `README.md`.
- `apply_formulas()` adds `$r`, `$c`, `$d` references; row access `$d[$r, :$c]` returns list (use `select_irow()` for dict).

<pre>
    v0.1.X 
            Started creating separate package, moved comment text to README.md
            For apply_formulas(), added relative row and column references $r and $c plus $d to reference the daf object.
            Changed the result of a row access, such as $d[$r, :$c] to return a list so it could be compatible with sum.
                use daf.select_irow() to select a row with dict as the result.
</pre>