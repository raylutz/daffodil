# Prompt for the AuditEngine thread: test daffodil f40d286, and move hot append loops to KeyedList rows and fast=True

Written on 2026-10-08 at 16:49 UTC by the daffodil thread, when daffodil main was at f40d286. This whole file is the
prompt for the AuditEngine thread. It follows that thread's report of the same day,
/home/audit-engine-dev/engineering_notebook/2026-10-08_daffodil_profile_functests.md.

Changed in daffodil since that report, all in CHANGELOG.md under Unreleased:

- The class KeyedIndex is gone. A KeyedList's `hd` is a plain dict. The rows a Daf hands out share the Daf's own
  `hd`, so `row.hd is daf.hd`.
- `default_record()` follows the Daf's columns, in their order. `default_record(astype=KeyedList)` returns a row
  that shares the Daf's `hd`, with the schema defaults, to fill by name in any order.
- `append()` of a KeyedList that shares the Daf's `hd` skips the column check. With `fast=True` it also skips the
  copy. At 1,000 columns, building and appending a row took about 90 µs as a dict, 24 µs as a KeyedList sharing
  the `hd`, and 1.7 µs with `fast=True`.
- `fast=True` checks only the first row added to an empty Daf, and raises ValueError if it does not fit.

## The task

This task has two parts: check that daffodil f40d286 breaks nothing in AuditEngine, then move the hot append loops to KeyedList rows with fast=True, where it is safe. Run the code to check a claim. Do not guess.

Background. daffodil is developed in /home/daffodil by another Claude thread. The local AuditEngine environments use an editable install of it, so they run whatever is in /home/daffodil. The Lambdas install daffodil from requirements-lambda.in, which pins git commit 6fdf366. So nothing here reaches the Lambdas until that pin moves, and a change that uses the new features breaks the Lambdas if it is deployed before the pin moves. Read /home/daffodil/CHANGELOG.md, the Unreleased section, and the docstrings of Daf.append(), Daf.from_lod() and default_record() (in /home/daffodil/src/daffodil/lib/daf_schema.py).

Limits.
- Do not edit, commit, pull or switch branches in /home/daffodil. Read it only. Report daffodil bugs with a small example, and do not work around them.
- Do not change requirements-lambda.in or any other pin. Ask me first.
- Work on a new local branch, daffodil-klist-fast, started from the current branch. One commit for each call site. Do not push, merge or tag.
- Do not run a stage on a real job in a way that rewrites its files. Use the functional tests, or copies in a scratch folder.
- The CPU is shared with the daffodil thread. Run one test or stage at a time.
- If a step says to ask, stop and ask me.

### Step 1

Record the daffodil commit, with git -C /home/daffodil rev-parse --short HEAD. It must be f40d286 or later.

### Step 2

Look for code that these changes could break. Search AuditEngine, not daffodil, and report each hit with its file and line:
- KeyedIndex, or a method that only KeyedIndex had, used on the hd of a KeyedList: .hd.to_dict(), .hd.index(, .hd.append(.
- default_record( on a Daf whose columns differ from its schema, in order or in which fields. Its dict now follows the Daf's columns, not the schema.
- default_record( inside try/except AttributeError. A Daf with columns but no schema now returns NULL in every column instead of raising.
- KeyedList(a_dict, a_list). The dict is now used as the hd, so it must map each key to its position.
- append( of a KeyedList that the code changes after the append, expecting the table to change. Without fast, the values are now copied.
- from_lod(..., fast=True). None is expected, as it is new.

### Step 3

Check for regressions. Run the unit tests, then the same two functional jobs as in the 2026-10-08 report, WI_Dane_20201103_functest and GA_Bartow_20201103_functest, the same way. Compare with that report: the same stages should build, with the same outputs. Report every difference, and whether it comes from daffodil.

### Step 4

Move the hot append loops. Start with the two sites the report found, analysis_utils:3108 and analysis_utils:3231, in extractvote, which build each row from page_marks_daf.default_record(). Then look at the other append loops in the profile with the most calls, and judge each the same way. The change at a site is:
    row = some_daf.default_record(astype=KeyedList)     # was: some_daf.default_record()
    row['col'] = value                                   # unchanged, in any order
    some_daf.append(row, fast=True)                      # was: some_daf.append(row)
The KeyedList shares the hd of some_daf, so its keys are the columns, in order. That is why it is safe to skip the check. A site qualifies only if all of these hold:
- The row comes from default_record(astype=KeyedList) of the same Daf it is appended to.
- The row is new for each append. It is not appended twice, and not changed after the append. With fast=True the table keeps the row's own list, so a later change to the row changes the table.
- Every key set on the row is a column. A key that is not a column gives the row its own hd, and it is then checked by name, and the extra key is dropped, as before with a dict.
- The row is not used as a plain dict before the append. A KeyedList is not a dict. isinstance(row, dict) is False. json.dumps(row) fails. It has no copy(), setdefault() or pop(). row == a_dict is False. {**row}, dict(row), row.to_dict(), row.get(), row.update(), 'k' in row, iteration and row.items() work. If the row is passed to a helper, read the helper.
For each site, before you change it:
- Measure the time of the stage, or of the loop, on the functional job, with the current code.
- Make the change, and run the job again. The outputs must be identical. Compare the files the stage writes.
- Measure again, and report both times.
Commit each site separately, with a message that starts with "requires daffodil f40d286 or later" and names the call site and the measured gain.

### Step 5

Report only, do not change: the profile showed that most daffodil time is in select_where() and in one-row slices such as daf[i] and daf[i, 'col'], on tables of 10 rows or fewer. For the ten busiest of those call sites, say what each reads, and whether one of these would do it without building a table: daf.select_record(key) for one row by key as a dict, daf.irow_la(i) for one row as a list, daf.lol[i][daf.hd['col']] for one cell, daf.col_to_la('col') for one column, or select_by_dict({'col': value}) for a test of equality. Do not change these yet.

### Step 6

Write your report to a file in the AuditEngine notes folder, named with the date and time and what it is about, and give me its path. Include the daffodil commit you ran against, the results of steps 2 to 5, and the branch and commits. Remind me that the Lambda pin must move before the branch is deployed.
