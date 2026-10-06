# Items to check in AuditEngine

Collected on 2026-10-04. Items 9 to 16 were added on 2026-10-05, for the release 0.6.0. These are checks, not directives. They come from the AuditEngine impact review and from
changes made in daffodil. The daffodil side has not seen the AuditEngine code for most of them. Each one is for the
AuditEngine owner to confirm and decide.

1. Check the count columns in `BIF.py`. The review reported that one variant of the cmpcvr report has count columns that
   are typed as bool. Daffodil now converts the words true and false to bool, so a column that holds only those values
   may come out as bool. If these columns are meant to be counts, int may be what is wanted.
2. Check the `from_lod()` call at `pdf_image_indexer2.py:74`. If its dicts always have the same keys, nothing needs to change.
   If a later dict can have a key that the first dict lacks, `from_lod()` now raises `ValueError` and says to pass `cols=`.
   Before, that key was dropped without any error.
3. Check `.index` at `args.py:1007`. The review reported that it may be reversed. This is not caused by a change in daffodil.
4. Check the JOB CSV for GA_Dekalb. The review reported malformed JSON in it. This is not caused by a change in daffodil.
5. Check the `select_where()` calls that test equality on one or more columns. `select_by_dict()` gives the same rows
   and is about 12 times faster. Look at three things before a call is changed:
   - A call that converts before it compares, such as `int(row['n']) == 5`, may need to stay. `select_by_dict()` compares
     the cell as stored, so text `'5'` does not match 5.
   - `inverse=True` is the negation of the whole match, not `!=` on each field.
   - `select_where()` raises `KeyError` for a column that does not exist. `select_by_dict()` returns no rows.
6. Check for a call that selects every row and then appends to, sorts or pops the result, and expects the original to
   change too. Daffodil now returns its own row list from `select_irows()` with every row in order, `d[:]`,
   `select_krows()` with every key, and `select_records_daf([], inverse=True)`. Such a caller would now change only the result.
7. In `chunktable.py`, `create_chunks()` appears not to have been migrated from pandas. It calls the deprecated `read_hunk_df()`,
   `utils.split_df_into_chunks_lodf(hunki_df, max_chunk_size)`, `DB.save_data(..., rtype='df')` and the pandas `is_chunk_all_bmd()`.
   The daffodil forms are already in the same file: `read_hunk_daf()`, `Daf.split_daf_into_chunks_lodaf()` and `is_daf_chunk_all_bmd()`.
   `read_hunk_df()` is also still called near `cvr_votes_hunk_df`. Before switching, compare the chunk sizes of the two splits. The daffodil
   split gives near-equal chunks, for example 250 rows with a maximum of 100 give sizes 84, 83 and 83.
8. In `chunktable.py`, no code calls `manifest_apply()`, `manifest_reduce()` or `manifest_process()`. Check the rest of AuditEngine. If nothing
   calls them, daffodil could remove them. They were written for an earlier chunk manifest idea. `split_daf_into_chunks_lodaf()` is
   used twice in `chunktable.py` and stays.

Items 9 to 16 come from changes made on 2026-10-05, after the first eight were written. The EC2 machine runs Python 3.11, which
version 0.6.0 accepts, because it needs 3.10 or later. Each item says how to look. The searches are a start. Judge each hit by reading the code.

9. Check the calls of `copy()` on a Daf. A plain `copy()` now has its own row list, `hd`, `dtypes` and key index, and it shares the rows.
   A copy also has no name unless `name=` is given. It kept the name of the original before. Look at three things:
   - Code that appends to a copy and expects the original to grow, or sorts the row list of a copy and expects the original to change.
     This was a side effect, and it is now gone. Changing a cell through a copy still changes the original.
   - Code that reads `name` of a copy. `join()` uses the name of each Daf to find its rows in the translator Daf, and uses `daf1` and `daf2`
     for a name that is empty. A `join()` of a copy with a custom translator that names the source may no longer match.
   - Code that calls `copy(False)`, `copy(True)` or `copy(deep=True)`. `copy(True)` still means deep. `copy(False)` now uses the class default.
   How to look: search for `.copy(` and keep the calls on a Daf. Search for `join(` and look at the names of both tables.
10. Check every `keyfield` given to `Daf()`, to the builders and to `set_keyfield()`. A keyfield that is not a column now raises `KeyError`.
   It was stored without an error, and key lookups then found nothing. A Daf with no column names yet still accepts a keyfield.
   A builder reads the columns from the data first, so a key that is in the header of the file is fine. Look at files whose header may lack the key column,
   and at calls that give `cols` or `include_cols` without the key. Look at `from_csv_buff()`, `from_csv()`, `from_lod()`, `from_lot()` and `from_cols_dol()`.
   How to look: search for `keyfield *=` and for `set_keyfield(`. For each, confirm that the named column is always there.
11. Check the subclasses of `Daf`, and the calls of `clone_empty()`. Every selector goes through `clone_empty()`, and it now returns an object of the same
   class as the original, and keeps `md_max_rows`, `md_max_cols`, `disp_cols`, `schema`, `retmode` and `itermode`. It returned a plain `Daf` with the defaults before.
   A subclass with its own `__init__` signature is the case to look at. Also look at code that sets `disp_cols`, `retmode` or `itermode` on a table, and then
   expects a selection of it to show or return the defaults.
   How to look: search for `class .*Daf` and for `clone_empty(`.
12. Check the Dafs that are built with both `cols` and `dtypes`. The constructor now cuts the dtypes to the names that are columns, and your dict is not changed.
   Look for code that reads `dtypes` later and expects names that are not columns, such as dtypes for columns that come from another file.
   A `clone_empty()` with new `cols` keeps the dtypes only for the columns that remain, drops the `schema` and the `disp_cols`, and clears the keyfield if a
   key column is gone. `groupby()` keeps its dtypes.
   How to look: search for `dtypes *=` and for `clone_empty(` with `cols`.
13. Check the calls of `set_cols()` and `rename_cols()`. They now keep the keyfield. The keyfield takes the new name of its column. They cleared it before.
   A later `set_keyfield()` still works. Look for code that relies on key lookups being off after a rename, or that tests `keyfield` for the empty string.
   With no column names yet, `set_cols()` keeps a keyfield that is one of the new names.
   How to look: search for `set_cols(` and `rename_cols(`.
14. Check for the removed names: `Daf.isin()`, `KeyedListEncoder`, `unpack_indirect()` and `from_dirlist()`. Nothing in daffodil used them, and the owner said that AuditEngine does not.
   A search is a cheap way to be sure. Many `.isin(` calls will be pandas, which is not changed. Keep only the calls on a Daf.
   How to look: search for `isin(`, `KeyedListEncoder`, `unpack_indirect` and `from_dirlist`.
15. Check the AuditEngine test suite on the new version. Run it first with the daffodil that is installed now, and then with the new one, in a separate environment, so that
   nothing that runs is changed. The two results should be the same. The scripts for a test of this kind are in notes/auditengine_test_prompt.md.
16. The release notes in CHANGELOG.md, under Unreleased, list 129 entries. Many are fixes. The first eight items above and items 9 to 14 are the ones that
   the daffodil side expects to matter. Read the sections Removed, Deprecated and Changed once more with AuditEngine in mind. Add to this file any item that matters.


## Results of the review on the EC2 machine, 2026-10-05

The AuditEngine thread on the EC2 machine tested daffodil 0.6.0 against AuditEngine on Python 3.11. It changed nothing there. It read `origin/main`
before the checks 9 to 16 were merged, so it saw only the first eight. Its result for each is below. The first paragraph is the summary.

No test result changed. The AuditEngine suite gave 1060 passed, 2 failed and 1 skipped on both versions, compared one test at a time over 1063 tests.
The two failures are on both versions. One is a stale fixture. The other fails only in the scratch environments, and points to the package pins and not to daffodil.
The thread believes that the machine can take 0.6.0.

Results for the first eight items:
1. Confirmed. In the real file `contest_variants.csv` the columns hold counts from 0 to 6. Version 0.5.13 turns every count of 1 or more into 1. Version 0.6.0 gives
   int 0 and 1 and the text of the larger counts, so a later sort or sum could raise `TypeError`. The call sites are `cmpcvr_report.py:93` and line 927.
   The smallest change is to type those four columns int in `BIF.py`, as `schema.py:333` already says.
2. Confirmed at `pdf_image_indexer2.py:74`, and not run on real data. It raises `ValueError` for a key that only a later page has. Passing `cols=` with the union of the keys avoids it.
3. The `.index` at `args.py:1007` is reversed, but cannot be reached, because the keys of the dict are unique. Not a daffodil change.
4. The malformed JSON is real, in `params/GA_Dekalb_20220524_clone/JOB_GA_Dekalb_20220524_clone.csv` at line 13. It fails the same way on both versions.
5. There are 19 `select_where()` calls that test only equality. This is a speed opportunity and not a fault.
6. Only `select_by_dict()` and `select_irows([], inverse=True)` changed how they share rows. Every in place edit on their results is safe. A cell edit through a
   `select_by_dict()` result does reach the original, as designed. `parse_utils.py:2382` is such a write, and the result is the same.
7. Confirmed. `create_chunks()` is still pandas. It is reached only from `operate.py:1537`, and 0.6.0 does not affect it.
8. Wrong, and daffodil must not remove the methods. `manifest_process()` is used at `ess_cvr.py:257`, on the live ES&S preparse path. The thread did not say whether
   `manifest_apply()` and `manifest_reduce()` are used.

For items 9 to 14 the thread checked the call sites by reading the code and probing with real data, and found none that breaks:
- `copy()`: the only plain copy is `evalvotes.py:267`, and it only reads its copy. The others are `copy(deep=True)`. The two reads of `name` are a log message and a join
  that builds its translator from the same two tables.
- Keyfields: 157 sites were classified. Each file that is loaded with a keyfield has that column in its real header, or is empty, which 0.6.0 allows.
- AuditEngine has no subclass of `Daf`. `eif.py` sets `retmode='val'` six times and never selects from those tables.
- The one column that is inserted after a load, at `archives.py:1040`, gives the same CSV on both versions.
- None of the three `set_cols()` and `rename_cols()` sites has a keyfield to keep. A comment at `map_targets_ai.py:331` still says that `set_cols()` clears the keyfield. It does not now.
- Every `isin(` is pandas. The other removed names are not used.

Two findings that are in AuditEngine and not in daffodil:
- `cmpcvr.py:652` calls `select_krows(krows=keys_ls, inverse=True)` under the comment "remove these rows" and discards the result. It never removed anything, on either version.
- `col()` now raises `KeyError`, which makes two existing `except KeyError` handlers work as written. `select_cols()` reorders three columns at `acre_utils.py:102`.

Not checked: `pdf_image_indexer2.py:74` on real PDF archives, and the paths that the test suite does not reach.

Update on 2026-10-05, after the review: `from_lod()` no longer raises for a key that first appears in a later dict. It adds a column. So the check in item 2, at
`pdf_image_indexer2.py:74`, needs no change in AuditEngine. With `cols` or `dtypes` given, keys that are not columns are still left out.
17. Check every `from_lod(` call that gives `cols=` or `dtypes=`. A dict with a key that is not one of those columns now raises `ValueError`, and names the keys. It dropped the
    value without a message before. If a call picks a few columns of wide records on purpose, pass `ignore_extra_keys=True`. The first review found no such call, and did not look for one.
    How to look: search for `from_lod(` with `cols=` or `dtypes=`. Also search for `from_lod_to_cols(` with `dtypes=`, and `from_dod(` with `dtypes=`, which call `from_lod()`.


## Second review on the EC2 machine, 2026-10-05

The thread tested daffodil `origin/main` at aabed5a, version 0.6.0, after `from_lod()` began to raise for a key that is not in `cols` or `dtypes`.
The result is that the machine should not take this version yet. The test suite gave the same result as before, 1060 passed, 2 failed and 1 skipped on both
versions. The two failures are on both. The suite does not reach the four sites below.

Sites that break, most serious first. Each stopped silently losing a value in 0.5.13, and now raises `ValueError` that names the keys:
1. `dominion_cvr.py:3446`, `from_dod(dod, keyfield='contest_name', dtypes=BIF.contestinfo_dtypes)`, in the pipeline stage `cvr_to_eif`, for every Dominion JSON-CVR job.
   Each contest dict has the key `id`, and `BIF.contestinfo_dtypes` has `contest_id`. In 0.5.13 every contest id is dropped and the `contest_id` column is empty.
   In 0.6.0 the stage stops. The suggested change in the report, `ignore_extra_keys=True`, does not work here, because `from_dod()` does not take it yet.
   The change that fixes the data is to name the key `contest_id` in `parse_contest_manifest_core()`. That changes the EIF.
2. `mapping_option_names_ocr.py:692`, `from_lod(updated_rows_lod, cols=rows_daf.columns())`. Seven of 11 local jobs have a file without `ballot_option`. The rewritten values are lost in 0.5.13.
   Add the missing columns to `cols`.
3. `mapping_option_names_ocr.py:2485` and `:2726`. The summaries pick four columns of rows that have `expected`, `detected` and `diff` as well. `ignore_extra_keys=True` is right for both,
   because they pick columns on purpose.
4. `profiled_bif.py:175` is safe with the local data, and a chunk without `is_bmd` or `is_nonbmd` would raise. Adding those two names to `cols` makes it safe.

Not a break: `pdf_image_indexer2.py:74` adds columns. The code after it reads columns by name. Not checked on a real single-file PDF archive.
All other calls, 27 with `cols` or `dtypes`, are safe by an AST check and by real files. Nothing calls `manifest_apply()` or `manifest_reduce()`.

Update on 2026-10-06: `from_dod()` now takes `ignore_extra_keys`, so `dominion_cvr.py:3446` can pass `ignore_extra_keys=True` and keep today's output. The better change is
still to name the key `contest_id` in `parse_contest_manifest_core()`, because it keeps the contest ids, which are lost now. `from_lod_to_cols()` does not take the parameter.


## Result of the change prompt, 2026-10-06

The thread took notes/auditengine_changes_prompt.md as suggestions and made five commits on a LOCAL branch of AuditEngine named daffodil-0.6.0-prep. It was not pushed, merged or tagged.
It lives on the EC2 machine only. The working tree there is back on claude_dev. Every change was run on daffodil 0.5.13 and on 0.6.0 at 002f7cb, and ignore_extra_keys is used nowhere.

Full suite, same command and data as before: 1074 passed, 2 failed, 1 skipped on both versions. The 14 new tests are the only difference from the last review. Both failures are the old ones.
ruff and mypy are clean. The branch can be deployed while the machine runs 0.5.13.

Commits:
1. 61be9bf16, dominion_cvr.py. A helper, contestinfo_daf_from_dod(), keeps in each contest dict only the keys named by BIF.contestinfo_dtypes, then calls from_dod() as before.
   Paulding: the Daf is identical on both versions, 31 rows, 8 columns, 248 cells. contest_id is still empty. 4 tests.
2. ee5a86643, mapping_option_names_ocr.py. rescore_rows() adds the missing ballot_option, ocr_match and ocr_metric to cols. The values now land for jobs whose file lacks those columns.
   This changes what the tool writes. Passaic, 8,115 rows: 277 rows matched, no mismatch against an independent recompute, the same hash on both versions. FL_Collier is byte-identical. 4 tests.
3. c390c8c8f, mapping_option_names_ocr.py. build_fill_summary_daf() keeps only the keys in cols. Both fill summaries are identical on both versions. 3 tests.
4. 754e76a17, profiled_bif.py. selected_chunk_daf() adds is_bmd and is_nonbmd to cols when missing. All 39,850 local chunks have both, and the output hashes are identical. 3 tests.
   An earlier report said 1,440 chunks. That counted two jobs only.
5. ced51cb4c, map_targets_ai.py. Docstring and comment only. No code changed.

Decisions still open for the AuditEngine owner. The thread changed none of them:
a. BIF.py lines 412 to 425: nine count columns are typed bool and should be int, as schema.py:333 says. They are audit_writeins, audit_undervotes, audit_overvotes, cvr_orig_writeins,
   cvr_orig_undervotes, cvr_orig_overvotes, cvr_modi_writeins, cvr_modi_undervotes and cvr_modi_overvotes. On Rockville's contest_variants.csv, audit_writeins is {0: 135, 1: 229} under bool in 0.5.13,
   mixed ints and text under bool in 0.6.0, and {0: 135, 1: 184, 2: 24, 3: 8, 4: 4, 5: 2, 6: 7} under int on both. Changing the type changes the Variant-list counts from 1 to the true count.
   The thread recommends int.
b. cmpcvr.py:652 discards the result of select_krows(inverse=True). It never removed the ballots found on only one side. It has no effect today, because all 10 local jobs have the same ballots on both sides.
   The smallest fix is to drop the line, and after the loop to assign audit_variants_daf and cvr_variants_daf from select_krows(keys_only_in_audit_ls, inverse=True). Assigning through the loop variable cannot work.
c. pdf_image_indexer2.py:74. Passing cols with the union of the keys would make 0.5.13 behave as 0.6.0. It costs one cheap pass. The thread advises not doing it now.
d. Renaming id to contest_id, in the helper of commit 1 and after add_optioninfo_to_contestinfo(). Done before that call, it lost all of Paulding's options. Done after it, only the 31 cells of contest_id
   change, and Paulding's EIF is byte-identical. The only consumer, cvr_to_eif, only logs it. Harmless, and of no use until a later step wants the Dominion contest id.

Not known: whether the values that commit 2 now saves are what the owner wants for the 7 jobs. Items 1 to 3 were checked through their helpers and not by running the tool ops end to end, because that would rewrite job files.
