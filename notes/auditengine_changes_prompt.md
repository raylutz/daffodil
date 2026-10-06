# Prompt for the AuditEngine thread: make the changes for daffodil 0.6.0

Written on 2026-10-06. Paste the text inside the fence into a Claude session on the EC2 machine, in the AuditEngine repository.
Unlike the earlier prompts, this one authorizes edits, on a local branch only. The thread took it as suggestions, and made the changes described
in the section "Result" of notes/auditengine_action_items.md. The advice in it holds on both daffodil 0.5.13 and 0.6.0, because it does not use
`ignore_extra_keys`, which 0.5.13 lacks.

````
You are making changes in AuditEngine to prepare for daffodil 0.6.0. Earlier reviews were report-only. This one authorizes edits, with the limits below. Work in a single thread. Run the code to check a claim. Do not guess.

Background. The machine runs daffodil 0.5.13 now. The new version, 0.6.0, is on origin/main of the sibling daffodil repository and is not yet released. In 0.6.0, from_lod() with cols or dtypes raises ValueError for a dict key that is not one of those columns, and so does from_dod() with dtypes. In 0.5.13 such a value was dropped without a message. The earlier reviews found four sites that would raise, and some related items. 0.6.0 adds a parameter ignore_extra_keys to from_lod() and from_dod(). 0.5.13 does not have it, so a call that passes it raises TypeError on 0.5.13. So make every change work on BOTH versions. Do not use ignore_extra_keys anywhere. Restrict the keys or the columns in AuditEngine code instead.

Limits.
- Work on a new branch of AuditEngine, one commit for each item below, with a message that names the item. Do not push, merge or tag anything.
- Do not switch, pull or change the sibling daffodil repository, and do not change the installed daffodil. Use the scratch folder and the virtual environments of the last review: venv_old with 0.5.13, and venv_new2 with origin/main. Rebuild venv_new2 from a fresh export of origin/main at 002f7cb or later if it is older.
- Do not run a pipeline stage on a real job in a way that rewrites its files. Reproduce with copies of the files, in the scratch folder.
- If an item says to ask, stop and ask me. Do not decide.

Change these (do each, in this order).

1. dominion_cvr.py:3446, from_dod(dod=contestinfo_dod, keyfield='contest_name', dtypes=BIF.contestinfo_dtypes). Each contest dict has the key id, and BIF.contestinfo_dtypes has contest_id. In 0.5.13 the id is dropped and the contest_id column is empty. Advice: before the call, make each inner dict hold only the keys that are in BIF.contestinfo_dtypes, which is what 0.5.13 did silently. That keeps today's output exactly, on both versions. Do NOT rename the key to contest_id in this commit. That would fill the contest_id column, change the EIF output, and is a decision for me. Instead, in your report, say what would change if the key were renamed. Check on the real Paulding ContestManifest.json that the Daf built is identical, cell by cell, on venv_old before the change and on both venvs after it.

2. mapping_option_names_ocr.py:692, rescore_ocr_targetmap_rows, from_lod(updated_rows_lod, cols=rows_daf.columns()). Seven local jobs have a ocr_targetmap_rows.csv without ballot_option, and some without ocr_match and ocr_metric. The rewritten values are lost in 0.5.13, so the tool's update never lands. Advice: add the missing names of those three columns to cols, in that order, after the columns that the file has. The values then land, which is what the tool is for, on both versions. This changes what the tool writes for those jobs, so say so in the commit message and the report. Test on a copy of the Passaic file, which has 8,115 rows, on both venvs, and show that the output has the three columns and the right values.

3. mapping_option_names_ocr.py:2485 and :2726, the summaries of fill_missing_ovals_from_hexstyle_siblings and fill_missing_ovals_from_gap_analysis. The rows added for unresolved styles have expected, detected and diff, which the cols lists do not name, and that is intended. Advice: build each summary row with only the keys in the cols list, for example {k: row.get(k) for k in cols}, or drop the extra keys from the dict that is added. That keeps today's summary exactly, on both versions. Test on a copy of Passaic's real oval_count_mismatches.csv, which has 13 short styles, and show the same summary on venv_old before and after, and no error on venv_new2.

4. profiled_bif.py:175, from_lod(selected_lod, cols=chunk_daf.columns()) in gen_profiled_bif. It is safe with the local data. A chunk without is_bmd or is_nonbmd would raise in 0.6.0. Advice: add is_bmd and is_nonbmd to cols when the chunk lacks them. Show that the output on the local chunks is identical to the output before.

5. The comment at map_targets_ai.py:331 says that set_cols() clears the keyfield. In 0.6.0 it keeps the keyfield, and it follows the new names, and in 0.5.13 it clears it. Rewrite the comment so that it is true for both: the keyfield may or may not be kept, so the code sets it again, which is why line 352 does that. Change no code.

Do NOT change these. Report each one, with the file, the line and your advice, and ask me:
a. The count columns in BIF.py that are typed bool and hold counts from 0 to 6, used by cmpcvr_report.py:93 and line 927. The right type is int, as schema.py:333 says. Changing it changes the values in the reports, so ask me first. Say which columns, and show the value counts on the real contest_variants.csv under both types.
b. cmpcvr.py:652 calls select_krows(krows=keys_ls, inverse=True) under the comment "remove these rows" and discards the result. It never removed anything. Say what the line is meant to do and what the smallest fix would be. Do not fix it.
c. pdf_image_indexer2.py:74, from_lod() with no cols. In 0.6.0 a key that appears only on a later page adds a column, and in 0.5.13 it is dropped. Nothing after it breaks. Do not change it. Say whether passing cols with the union of the keys is worth doing, and what it would cost.
d. The rename of id to contest_id in item 1, as above.

Check your work. After the five commits, run the full AuditEngine test suite in venv_old and in venv_new2, with the same command and data as before, and compare with the saved results of the last review, test by test. Nothing may differ. If something differs, find out why before you go on. Search again for every from_lod( call with cols= or dtypes=, and for every from_dod( call with dtypes=, and confirm that each is now safe on both versions. List any that is not.

Report, in this order: a table of the test results; one section for each of the five commits, with the file and lines changed, the diff, and the check you ran with its output; the four items that you did not change, each with your advice and your question for me; and one line on whether the branch can be deployed to the machine while it still runs 0.5.13, and what you are unsure about. Delete nothing. Leave the scratch folder in place.
````
