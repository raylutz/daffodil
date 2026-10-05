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
16. The release notes in CHANGELOG.md, under Unreleased, list about two hundred changes. Many are fixes. The first eight items above and items 9 to 14 are the ones that
   the daffodil side expects to matter. Read the sections Removed, Deprecated and Changed once more with AuditEngine in mind. Add to this file any item that matters.

