# Prompt for the AuditEngine thread: profile daffodil use, check for regressions, find fast=True candidates

Written on 2026-10-07 at 18:12 UTC, when daffodil main was at d876a31. Relay the text inside the fence to the Claude
session that works in /home/audit-engine-dev on the EC2 box.

Since the 0.6.0 release, daffodil main has these changes. None is on PyPI yet.

- A profiling mode, daffodil/lib/daf_profile.py, turned on with DAFFODIL_PROFILE=1. It records table sizes, the
  operations on each table and the busiest call sites, one file per run, and combines the runs of several stages.
- `append(row, fast=True)` and `from_lod(lod, fast=True)` skip all checks and copies, for rows built in column order.
- Selecting rows no longer copies the whole row list. A one-cell read on 100,000 rows went from 796 µs to 30 µs.
- `append()` of a list adds a copy without a dict round trip: 86 µs to 14 µs a row at 1,000 columns.
- `append()` of a KeyedList, without `fast`, now copies its values. Before, the table kept the KeyedList's own list.
- Type annotations use `X | None`. No behavior change.

Local AuditEngine uses an editable install, so it may run this code already. The Lambdas load daffodil 0.6.0 from
PyPI, so none of this reaches them until a release.

````
This is a measurement and review task. Do not change AuditEngine code unless a step says so, and do not change daffodil at all.

Background. daffodil is developed in /home/daffodil, by another Claude thread on this machine. Its main branch has changes that are not on PyPI. The local AuditEngine environments use an editable install of daffodil, so they read the files in /home/daffodil as they are at the moment of the run. The Lambdas load daffodil 0.6.0 from PyPI. Read these pages for the details: /home/daffodil/docsite/profiling.md for the profiling mode, and the docstrings of Daf.append() and Daf.from_lod() in /home/daffodil/src/daffodil/daf.py for fast=True. The changes are listed in /home/daffodil/CHANGELOG.md under Unreleased.

Limits.
- Do not edit, commit, pull or switch branches in /home/daffodil. Read it only.
- Do not push, merge or tag anything in AuditEngine.
- Do not run a stage on a real job in a way that rewrites its files. Use the functional tests, or copies of the files in a scratch folder.
- The machine's CPU is shared with the daffodil thread. Run one test or stage at a time, and keep the runs to what is needed.
- If a step says to ask, stop and ask me.

Step 1. Find out which daffodil the local environments use. For each virtual environment that the functional tests use, report the daffodil location from pip show daffodil, whether it is an editable install, and the commit, from git -C /home/daffodil rev-parse --short HEAD. If an environment does not point at /home/daffodil, say so and use only the ones that do for steps 2 and 3.

Step 2. Check for regressions. Run the functional tests that run locally, in one go, as they normally run. Record the daffodil commit at the start of the run. Report every failure with its traceback, and say whether it looks related to daffodil. These daffodil changes could change behavior:
- append() of a KeyedList now adds a copy. Code that appended a KeyedList and then changed it, expecting the table to change too, would now see no change.
- A selection no longer copies the row list of the table it came from. The rows were shared before too, so results should be the same.
- Text from repr() and to_md() changed by one space in one-character columns in 0.6.0. A test that compares that text exactly may need an update.
If a failure comes from daffodil, do not work around it. Report it, with the smallest example you can make, so it can be fixed in daffodil.

Step 3. Profile the stages. For each stage that the functional tests can run locally, run it with these environment variables, one stage at a time:
    DAFFODIL_PROFILE=1
    DAFFODIL_PROFILE_STAGE=<a short stage name, such as tabulate>
    DAFFODIL_PROFILE_DIR=<one scratch folder for this whole run>
Each run writes daffodil_profile_<stage>_<host>_<pid>.md to that folder, and prints a report to stderr. Then combine them:
    python -m daffodil.lib.daf_profile combine <the folder> -o <the folder>/report.md --data <the folder>/combined.md
Notes.
- Counts and sizes are exact. Times are inflated by about 3 µs for each counted call, and about 15 µs for each call that makes a table, so compare times only between methods, not with runs that were not profiled.
- If a stage uses multiprocessing, the workers end without writing a file, and only the parent process is counted. Say which stages do this. Do not add daf_profile.dump() calls to the workers without asking me.
- Call sites are named by module and line, such as auditengine.tabulate:212.
Report the path of report.md, and summarize it:
- The table size distribution, by rows and by columns, for all stages. How many tables have more than 256 rows, and how many more than 10,000.
- The ten creation lines with the most tables or the most operations, with their stage, how the tables were made, their sizes and their main operations.
- The ten methods and the ten call sites with the most time.
- Where append() is called most, and with what kind of row: list, dict or KeyedList.

Step 4. Find candidates for fast=True. Make no changes in this step. From the profile and the code, list the call sites where append() or from_lod() could take fast=True. fast=True skips all checks, and with append() the table keeps the caller's list itself, with no copy. So a call site qualifies only if all of these hold:
- Each row is complete and in column order. For a dict, its keys are in the same order as the columns, every time. For from_lod(), every dict has the same keys in the same order as the first.
- For append() with a list or a KeyedList, the list is a new one for each row, and is not changed after the append. A list that is filled in place and appended again, such as a buffer reused in a loop, does NOT qualify: every row of the table would be that same list. A variable that is given a new list on each pass is fine.
- The call is hot enough to matter in the profile.
For each candidate, give the call site, the kind of row, the number of calls in the profile, how you know the rows are complete and in order, how you know the list is new each time, and the expected gain. The measured costs at 1,000 columns are about 15 µs a row for append() of a list without fast, and 1.1 µs with fast=True. For a KeyedList, 19 µs and 0.9 µs. For a dict, 40 µs and 21 µs. from_lod() of 1,000 dicts of 1,000 keys took 121 ms, and 27 ms with fast=True. At 10 columns the gains are smaller: 1.3 µs to 0.7 µs a row for a list.

Step 5. Write your report to a file in the AuditEngine notes folder, named with the date and time and what it is about, and give me its path. Include the daffodil commit you ran against. Any change that uses fast=True must wait for a daffodil release, because the Lambdas run 0.6.0, which does not accept fast. Do not make those changes yet. Ask me first.
````
