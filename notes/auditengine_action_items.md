# Items to check in AuditEngine

Collected on 2026-10-04. These are checks, not directives. They come from the AuditEngine impact review and from
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
