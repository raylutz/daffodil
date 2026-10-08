# Briefing for the AuditEngine thread: daffodil 360c892, append rules, KeyedList rows and fast=True

Written on 2026-10-08 at 23:08 UTC by the daffodil thread. daffodil main is at 360c892. This whole file is the
briefing and the task. It replaces the prompt /home/daffodil/notes/2026-10-08_1649_auditengine_keyedlist_fast_prompt.md,
which was wrong about fast=True: it said that a KeyedList with its own hd is placed by column name. With fast=True it
is not. Read this file instead. It follows your report
/home/audit-engine-dev/engineering_notebook/2026-10-08_daffodil_profile_functests.md.

## 1. What changed in daffodil since the 0.6.0 release

All of it is on main, and listed in /home/daffodil/CHANGELOG.md under Unreleased.

- KeyedIndex is gone. A KeyedList's hd is a plain dict, of each key and its position. The rows a Daf hands out, from
  iloc() and iter_klist(), share the Daf's own hd, so row.hd is daf.hd. A row that adds or deletes a key first gets
  its own copy of the hd, so the Daf and the other rows never change.
- default_record() follows the Daf's columns, in their order, however they were set. A column gets the schema's
  default, or NULL if the schema has none for it. Schema fields that are not columns are left out. A Daf with
  columns but no schema now gets NULL in every column, instead of AttributeError. A Daf with no columns and no
  schema still raises AttributeError.
- default_record(astype=KeyedList) returns a KeyedList that shares the Daf's hd, with the defaults. Fill it by
  name, in any order: assigning to a key writes to that column's position.
- append() and from_lod() take fast=True. The rules are in section 2.
- append() of a KeyedList, without fast, now copies its values. Before, the table kept the KeyedList's own list.
- append() of a list adds a copy directly, without a dict round trip: 86 µs to 14 µs a row at 1,000 columns.
- Selecting rows no longer copies the whole row list of the table: a one-cell read on 100,000 rows went from
  796 µs to 30 µs.
- KeyedList(a_dict, a_list) shares the dict as the hd, so the dict must map each key to its position.
  KeyedList(keys, values) with a list of names is unchanged.
- The profiling mode, DAFFODIL_PROFILE=1, that you used on 2026-10-08.
- Type annotations use X | None. No behavior change.

## 2. The append rules

A Daf keeps its rows in lol and its column names in hd. A KeyedList made by the Daf shares that hd. The one test
row.hd is daf.hd proves that the row's keys are the Daf's columns, in order.

Without fast, the default: every row is checked fully, and the table always gets its own copy.

| Row | What happens |
|---|---|
| KeyedList that shares the Daf's hd | Not checked, since its keys are the columns. Values copied. |
| Other KeyedList, or a dict | Keys compared with the columns. In order: copied. Otherwise placed by name: a missing key gets NULL, a key that is not a column is dropped. |
| list, the right length | Copied. |
| list, too short | Padded with NULL, and copied. |
| list, too long | ValueError. |

With fast=True: the caller promises that the row is complete and in column order, and gives it to the table.
Nothing is copied. The checks are:

| Row | Check | If it fails |
|---|---|---|
| KeyedList | Must share the Daf's hd. Nothing else is checked. A KeyedList with its own hd fails even if its keys match. | ValueError |
| dict, the first row of an empty Daf | Its keys are the columns, in order. | ValueError |
| dict, any later row | Its number of keys is the number of columns. | ValueError |
| list, any row | Its number of values is the number of columns. | ValueError |

What fast=True cannot catch: a later dict with the right number of keys in another order, and a list in the wrong
order. Their values land in the wrong columns. And since nothing is copied, the caller must not change a list or a
KeyedList after appending it. Giving a variable a new list each time is fine. Filling one list in place and
appending it again is not: every row is then that same list.

fast=True makes no difference to the first row of a Daf with no columns, to a Daf or a list of dicts, or with
respect_kd=True and a keyfield. Those take the usual path.

from_lod(lod, fast=True): the columns are cols, or the keys of dtypes, or the keys of the first dict. The first dict
must match the columns exactly, and every dict must have the right number of keys, or it raises ValueError.

A ValueError from fast=True means the program is wrong. Do not catch it. The message names the Daf, if it has a
name, and the row, says how its keys or length differ from the columns, and says how to fix it. For example:

    append(fast=True) to Daf 'marks': the KeyedList for row 1 does not share the hd of this Daf, so its keys are not
    known to be the columns: it has keys that are not columns, ['note']. Make the row with
    default_record(astype=KeyedList) of this Daf, and set only keys that are columns, or leave out fast to place its
    values by column name.

    append(fast=True) to Daf 'marks': row 1 has 2 keys for 3 columns: it lacks the columns ['y']. With fast=True a
    dict must have the columns as its keys, in order. Make it with default_record(), or leave out fast to give a
    missing key NULL.

## 3. The loop to use

    d = Daf(schema=MarksSchema)
    for mark in marks:
        row = d.default_record(astype=KeyedList)   # the defaults, sharing d.hd
        row['y'] = mark.y                          # by name, in any order
        row['x'] = mark.x
        d.append(row, fast=True)                   # verified by the shared hd

Measured at 1,000 columns, per row, including building it: about 90 µs as a dict, or as a KeyedList with its own
hd; 24 µs as a KeyedList that shares the hd, without fast; 1.7 µs with fast=True.

A KeyedList is not a dict. These work on it: row['k'], row['k'] = v, row.get(), row.update(), 'k' in row, len(row),
iteration over the keys, row.items(), row.to_dict(), dict(row) and {**row}. These do not: isinstance(row, dict) is
False, json.dumps(row) fails, it has no copy(), setdefault() or pop(), and row == a_dict is False. Use
row.to_dict() where a dict is needed.

## 4. The task

Limits.
- Do not edit, commit, pull or switch branches in /home/daffodil. Read it only. Report a daffodil bug with a small
  example, and do not work around it.
- Do not change requirements-lambda.in or any other pin. The Lambdas install daffodil from git commit 6fdf366, which
  has none of this. A change that uses default_record(astype=KeyedList) or fast=True breaks the Lambdas if it is
  deployed before that pin moves. Ask me before touching the pin.
- Work on the local branch daffodil-klist-fast, started from the current branch, or keep using it if you have
  started it. One commit for each call site. Do not push, merge or tag.
- Do not run a stage on a real job in a way that rewrites its files. Use the functional tests, or copies in a
  scratch folder.
- The CPU is shared with the daffodil thread. Run one test or stage at a time.
- If a step says to ask, stop and ask me.

Step 1. Record the daffodil commit, with git -C /home/daffodil rev-parse --short HEAD. It must be 360c892 or later.

Step 2. If you already changed any call site under the earlier prompt, check each one against section 2. A
KeyedList appended with fast=True must come from default_record(astype=KeyedList) of the same Daf, and must have
only column keys set. Otherwise it now raises ValueError. Fix the call site, not daffodil.

Step 3. Search AuditEngine, not daffodil, for code that these changes could break, and report each hit with its file
and line:
- .hd.to_dict(), .hd.index( or .hd.append( on the hd of a KeyedList. Those were KeyedIndex methods.
- default_record( on a Daf whose columns differ from its schema, in order or in which fields.
- default_record( inside try/except AttributeError.
- KeyedList(a_dict, a_list).
- append( of a KeyedList that the code changes after the append, expecting the table to change.
- append( of a list longer than the columns, which raises ValueError as before.

Step 4. Check for regressions. Run the unit tests, then WI_Dane_20201103_functest and GA_Bartow_20201103_functest,
the same way as in your 2026-10-08 report. Compare: the same stages should build, with the same outputs. Report every
difference, and whether it comes from daffodil.

Step 5. Move the hot append loops to the loop in section 3. Start with analysis_utils:3108 and analysis_utils:3231 in
extractvote, which build each row from page_marks_daf.default_record(). Then the other append loops with the most
calls in the profile. A call site qualifies only if all of these hold:
- The row comes from default_record(astype=KeyedList) of the same Daf it is appended to.
- The row is new for each append. It is not appended twice, and not changed after the append.
- Every key set on the row is a column.
- The row is not used as a dict before the append. See the list in section 3. If it is passed to a helper, read the
  helper.
For each site:
- Time the stage, or the loop, on the functional job, with the current code.
- Make the change, run the job again, and compare the files the stage writes. They must be identical.
- Time it again, and report both times.
Commit each site on its own, with a message that starts with "requires daffodil 360c892 or later" and names the call
site and the measured gain.

Step 6. Report only, do not change. Most daffodil time in your profile was in select_where() and in one-row slices
such as daf[i] and daf[i, 'col'], on tables of 10 rows or fewer. For the ten busiest of those call sites, say what
each reads, and whether one of these would do it without building a table: daf.select_record(key) for one row by key
as a dict, daf.irow_la(i) for one row as a list, daf.lol[i][daf.hd['col']] for one cell, daf.col_to_la('col') for one
column, or select_by_dict({'col': value}) for a test of equality.

Step 7. Write your report to a file in the AuditEngine notes folder, named with the date and time and what it is
about, and give me its path. Include the daffodil commit you ran against, the results of steps 2 to 6, and the branch
and its commits. Say anything in daffodil that did not behave as this file says. Remind me that the Lambda pin must
move before the branch is deployed.
