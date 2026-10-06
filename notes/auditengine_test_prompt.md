# Prompt for the AuditEngine thread on the EC2 machine

Paste the text between the lines into a Claude session on the EC2 machine, in the AuditEngine repository.
It tests daffodil 0.6.0 against AuditEngine without changing anything that runs. Written on 2026-10-05.

---

You are testing a new version of the library daffodil, 0.6.0, against AuditEngine. The new version has behavior changes,
and I need to know what they do to AuditEngine before the machine takes the new code. Report only. Do not change AuditEngine,
do not change the installed daffodil, and do not touch anything that a running job uses.

Work in a single thread. Describe what you see, and run the code to check a claim. Do not guess.

## Part 1. Set up a separate environment

The daffodil repository is a sibling folder of this one. The machine may run its code through an editable install, so do not switch its branch,
do not pull into it, and do not change any file in it. Only read from it, and fetch into it.

1. Find how daffodil is installed now (`pip show -f daffodil`, or `pip list -e`). Say which folder it points to, and which branch or commit that folder has.
2. In the sibling daffodil folder, run `git fetch origin`. This changes only the stored references. Then check that `origin/main` has version 0.6.0 in `pyproject.toml`
   with `git show origin/main:pyproject.toml`. If it says something else, stop and tell me.
3. Export `origin/main` to a scratch folder, outside both repositories: `mkdir -p /tmp/daf_new && git -C <sibling> archive origin/main | tar -x -C /tmp/daf_new`.
4. Make two Python 3.11 virtual environments in the scratch folder, `venv_old` and `venv_new`. Install AuditEngine's requirements in both.
   - In `venv_old`, install daffodil from the sibling folder as it is now, as a normal install and not an editable one. This is the version the machine runs.
   - In `venv_new`, install daffodil from `/tmp/daf_new`, as a normal install.
   Install daffodil last, so that your choice wins over AuditEngine's own requirements. Check with `pip show daffodil` in each one.

## Part 2. Run the AuditEngine tests in both

Run the full AuditEngine test suite in `venv_old`, and then in `venv_new`. Use the same command and the same data for both. Save each output to a file in the scratch folder.
Report the number of tests that pass, fail, skip and error in each. Then list every test whose result differs between the two.
For each difference, give the test, the error message, and the line of AuditEngine or daffodil where it starts. Say whether the cause is a change in daffodil.
Tests that fail in both are not caused by the new version. List them, but do not look into them.

## Part 3. Check the call sites

Search the AuditEngine source, not the virtual environments, for each pattern below. For each hit, read the code and say whether the change can affect it.
Keep a hit only if it is on a Daf, and not on a pandas object or a plain list. Give the file and line, and one sentence on why.

1. `.copy(` on a Daf. A plain `copy()` now has its own row list, `hd`, `dtypes` and key index, and shares the rows. A copy has no name unless `name=` is given.
   Look for code that appends to a copy and expects the original to grow. Look for code that reads the `name` of a copy, and for `join(` calls on a copy with
   a custom translator that names its sources. Look for `copy(False)` and `copy(True)`.
2. `keyfield *=` and `set_keyfield(`. A keyfield that is not a column now raises `KeyError`. Confirm that the named column is always there, also when `cols` or
   `include_cols` is given, and for files whose header may lack it. Look at calls of `from_csv_buff(`, `from_csv(`, `from_lod(`, `from_lot(` and `from_cols_dol(`.
3. `class .*Daf` and `clone_empty(`. Selections now return the class of the original, and keep `md_max_rows`, `md_max_cols`, `disp_cols`, `schema`, `retmode` and `itermode`.
4. `dtypes *=` and `clone_empty(` with `cols`. The constructor cuts the dtypes to the names that are columns. Look for code that reads `dtypes` and expects names that are not columns.
5. `set_cols(` and `rename_cols(`. They now keep the keyfield, and it follows the new names. They cleared it before. Look for code that relies on the keyfield being empty afterward.
6. `isin(`, `KeyedListEncoder`, `unpack_indirect` and `from_dirlist`. These were removed from daffodil. Keep only the calls on a Daf, not pandas.
7. Read the sections Removed, Deprecated and Changed under Unreleased in `CHANGELOG.md` in `/tmp/daf_new`. Name any other change that AuditEngine uses.
8. Read `notes/auditengine_action_items.md` in `/tmp/daf_new`. It has 16 checks. For items 1 to 8, say what you find. They were written without seeing the AuditEngine code.

## Part 4. Report

Give, in this order:
- A table of the test results from Part 2.
- A numbered list of every call site from Part 3 that the change can affect, most serious first. For each, give the file and line, what happens now, what would
  happen with 0.6.0, and the smallest change that would keep it working. Describe the change. Do not make it.
- A short line saying whether you believe the machine can take the new version, and what you are unsure about.

Delete nothing. Leave the scratch folder in place, so that I can look at it.

---

## Second review prompt, 2026-10-05

This was used after from_lod() began to raise for a key outside cols or dtypes. It tested daffodil origin/main at aabed5a. Its result is in notes/auditengine_action_items.md,
in the section "Second review on the EC2 machine". It reuses the scratch folder, venv_old and the saved test output of the first review. The steps are these.

1. Fetch into the sibling daffodil folder only. Check that origin/main is aabed5a or later and that the version is 0.6.0. Export it with git archive to /tmp/daf_new2.
2. Make venv_new2 with Python 3.11 and AuditEngine's requirements, and install daffodil from /tmp/daf_new2 last, as a normal install. Reuse venv_old (0.5.13).
3. Run the full suite in venv_new2, and compare test by test with the saved output of venv_old.
4. Search the AuditEngine source: every from_lod( with cols= or dtypes=; every from_lod_to_cols( and from_dod( with dtypes=; every from_lod( without cols and dtypes,
   to see whether its dicts can have different keys; and any call of manifest_apply( or manifest_reduce(.
5. Report, as in the first review: a table of the test results, a numbered list of every call site that the change can affect with the smallest change, and one line on whether the machine can take the version.
   Report only. Change nothing.
