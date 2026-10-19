# AuditEngine's whole-table stages on SqlDaf

Started 2026-10-19 14:43 UTC. Ray asked for this on 2026-10-19:

> go back to the find variants and the cmpcvr work to make sure that's all done ... look at also the
> stages with anything that uses a full BIF. See if those can be ported over to SDaf. I think also the
> ACRE report uses full BIF. It also uses a group daf ... evaluate the completion status of all this
> for AuditEngine and make code segment recommendations for AuditEngine pickup. ... Don't you try to
> run [the state audits].

So nothing here runs an AuditEngine stage. Every check reads AuditEngine's files, copies what it uses
to the scratch folder, and calls AuditEngine functions only where they are pure. AuditEngine's
`logs.sts()` is replaced by a no-op in those checks, so nothing is written under its folder.

## Names

The result of `group_where()` or `groupby()` is called a group daf here. It maps each key to a
Daf, or on a SqlDaf to a selection. On a Daf it is a dict. On a SqlDaf it is a KeyedList. Ray
wants a shorter name than "keyed list of Daf", and will settle the naming later.

## find_variants and cmpcvr

The port is `notes/sql_backing/sdaf_variants.py`, with its check `sdaf_variants_check.py`. It is
the deliverable of AuditEngine's mail 2026-10-17_0019_find_variants_port_task.md, which is still open.

- 2026-10-18: SF sheet1, on AuditEngine's cmpcvr.db of 2026-10-16. Every table, CSV file and stat matched.
  But sheet1 is not mapped and has no extraction, Ray, 2026-10-19. Its audit votes are not real, so
  that run shows only that the port does what AuditEngine's code does. It is no test on real data.
- 2026-10-19: SF pilot d05, on AuditEngine's cmpcvr.db rebuilt 2026-10-18 23:58. This is the real-data check. That build has
  the concise_extras column of AuditEngine's 898cdf628 and 4464e7444. Every table, CSV file and
  stat matched again. The port needs no change for concise_extras. Step 18 joins every votes
  column, and AuditEngine's own `create_contest_variant_record()` picks concise_extras up.
- AuditEngine's changes since the draft, 73dfbd3c4, touch how concise votes are made and how
  variants are classified later. None touches find_variants.
- One difference stays, Q230. contest_variants.csv from the port writes dict cells as JSON, with
  True, False and None as 1, 0 and ''. AuditEngine's file writes PYON. A grep of AuditEngine
  found no `is False` or `is None` test on agreed, note or flags. Only the readers can say if
  the change matters, so it goes to AuditEngine.

What AuditEngine needs before it can run the port:

1. sql_backing merged into claude_dev. AuditEngine's venv runs daffodil from /home/daffodil/src.
   The port uses `groupby_aggregate()`, `groupby_reduce()` and `groupsum_daf()` of sql_backing. Q252.
2. pysqlite3-binary in AuditEngine's venv. It has SQLite 3.37.2, and SqlDaf needs 3.48.
   AuditEngine uses SqlDaf nowhere else, so the merge breaks nothing before that install.
3. AuditEngine's own draft, utilities/sdaf_variants.py of 73dfbd3c4, is replaced by the port.

## The full BIF

AuditEngine loads it through one function, `bif_utils.simple_recover_full_bif_daf()`. It loads
full_bif.csv as a Daf, attaches BifSchema, converts the columns, and caches the Daf for the process.
ACRE loads it through `ballot_utils.recover_bif_metadata_daf()`, which falls back to biacvr_bif.

Sizes, measured 2026-10-16: the SF pilot d05 is 133,422 rows and 241 MB as a Daf. A full county
of 1,603,862 rows is 2,675 MB. SF's full_bif.csv for sheet1 is 567 MB on disk.

### Where it is used

At AuditEngine's 4fa8df6b1. "Rows" means the stage picks a few rows; "table" means it reads the whole table.

| Site | Kind | Change for SqlDaf |
| :--- | :--- | :---------------- |
| acre_utils.acre_build_cvr_daf | table | the join, see ACRE below |
| ballot_exceptions_report.analyze_ballot_exceptions | table | `id in x.keys()` per ballot |
| ballot_utils.get_image_links_snippet | rows | none |
| ballot_utils.process_images_to_upload_by_ballot_id | rows | none |
| ballot_utils.predict_png_fns_of_ballot_ids | rows | none |
| bif_report.gen_bif_report_per_sourcelist | table | `groupby('corrupted')` then `append()` |
| bif_utils.build_style_to_contests_dol_daf, full_bif_daf_to_style_to_contests_dol | table | none |
| bif_utils.recompute_is_bmd_from_captured_data | table | none |
| bif_utils.fix_full_bif_is_bmd | table, writes | saves full_bif.csv by DB.save_data() |
| bif_utils.get_ballot_full_bif_dict | rows | none |
| bif_utils.imagematch, imagematch_core | table | none |
| bmd_map_report.create_draft_bmdif_file | table | none |
| bmd_map_report.derive_bmd_contest_name_proposals, derive_bmd_option_proposals | table | `id in x.keys()` and `x[id, col]` per ballot |
| cmpcvr_report.contest_discrepancy_report_daf | rows | none |
| evalvotes.apply_strikeouts | rows | none |
| extraction_coverage_report, four compute_ functions | table | `x[id, col]` per correction |
| gentemplates.gen_template_tasklists_randomized | table | none |
| mapper.import_targetmap, only_bmd_styles | table | none |
| narrative_report.gen_ballot_figure_html | table | none |
| parse_utils.get_bmd_styles_to_contests_cached | table | `{'is_bmd': '1'}` |
| votes_extractor.extract_multiple | rows | none |
| votes_extractor.gather_bmd_contest_ocr_samples, gather_bmd_option_ocr_samples | table | none |
| writein_inspection_report.create_writein_inspection_report | table | none |

Not counted: dvs_order.py, self_test.py and the `full_bif_df` functions of bif_utils.py use pandas.
The Lambdas stay on Daf. See the memory note on AuditEngine's SqlDaf use.

### The call forms, run on both

`notes/sql_backing/full_bif_forms_check.py`, on a copy of the SF pilot d05 full_bif.csv, 2026-10-19.
Each form runs on a Daf and on a SqlDaf of the same rows, and the results are compared. Every form
gave the same result, with these sizes:

- `x[ballot_id_ls]` 3. `x[id].apply_dtypes().to_dict()` 51 cells, so Q175 is settled.
- `select_where` on is_bmd 1,402, on not corrupted and not BMD 132,020, on archive_basename and on
  card codes 133,422.
- `select_by_dict({'is_bmd': 1})` 1,402. `col(unique=True)`, `columns()`, `daf_valuecount()`,
  `value_counts_daf()`, `valuecounts_for_colname1_groupedby_colname2()` 440 styles.
- `groupby('corrupted')` keys and sizes. `select_krows()`, `select_records_daf()`, `to_md()` of a slice.
- `for row in x`, `x[id, col] = v`, `set(x.keys())`, `id in x`.

Load and convert: Daf 2.1 s, SqlDaf 5.4 s, for 133,422 rows. The SqlDaf load is done once per change
of the file, if the database is kept.

Findings:

- **`id in x.keys()` in a loop.** On a SqlDaf one call took 0.23 s, 200 calls 47 s. On a Daf one call
  took 7 ms, since `keys()` builds a list. `id in x` took 0.24 ms on the SqlDaf and gives the same
  answer on both. Three functions run it once per ballot.
- **`x[id, col].to_value()` in a loop.** Each call makes a TEMP table, about 1 ms. `x.select_record(id)[col]`
  is about 30 times faster, as the `__getitem__()` docstring says.
- **`groupby().get(0).append(...)`.** bif_report.py appends the group of '' to the group of 0. On a
  SqlDaf a group is a selection, and appending a Daf to it raises AttributeError. A `select_where`
  on both values gives the same rows.
- **`select_by_dict({'is_bmd': '1'})` finds nothing once the dtypes are applied.** That is so on a
  Daf too. parse_utils.py asks for the raw table, but `simple_recover_full_bif_daf()` returns the
  converted one whenever it is cached. So it depends on what ran before, in the same process. On
  a SqlDaf there is one table, converted, so it would always find nothing. `{'is_bmd': 1}` found 1,402.
- **file_paths is a list column.** SqlDaf's `apply_dtypes()` raises NotImplementedError for list.
  The check converts it as str, and AuditEngine's `utils.file_paths_of_row()` reads the text.
  Phase 2 of the flat file_path columns removes the list.

### Loading it into SQLite

A sibling of the loader, `simple_recover_full_bif_sdaf()`, in AuditEngine:

- Bring the local full_bif.csv up to date by the local-mirror rules of `DB.load_data()`, without
  parsing it. DB has no such call yet, so AuditEngine would add one.
- Keep full_bif/full_bif.db next to it. Record the etag of the CSV it was built from.
- If the etag is the same, open the table. Otherwise build it with `SqlDaf.from_csv()`,
  `attach_schema()` and `apply_dtypes()`, with list columns as str.
- Writers stay on the Daf for now. fix_full_bif_is_bmd is the only one. Its new CSV has a new etag,
  so the next load rebuilds the table.

## ACRE

`acre_build_cvr_daf()` joins the CVR votes with the BIF, one row per ballot with its concise votes.
`acre_stratify()` makes the strata with `group_where()`, a group daf in which a ballot is in about
one stratum per contest. `acre_select_from_strata()` walks the strata smallest first.

The checks use the SF pilot d05. Its BIF has 133,422 sheets. Its `cvr_votes!final.csv` is the CVR
of the whole county, 1,603,908 rows, 704 MB. ACRE's inner join keeps the rows of the BIF's sheets,
so the checks first keep those rows, by streaming the CVR. That keeps 133,422 rows, one for each sheet.
ACRE needs only the CVR and the BIF, so the pilot is a valid test. SF sheet1 is not, see above.

The ACRE input is the output of AuditEngine's own `_join_cvr_votes_with_bif()` on those rows and
the pilot's BIF. `notes/sql_backing/acre_strata_check.py` runs AuditEngine's own `acre_stratify()`
and `acre_select_from_strata()` on it as a Daf, and the SqlDaf `group_where()` with the same
`where()`, decoding the votes text there. The SqlDaf group daf goes into `acre_select_from_strata()` as it is.

| 133,422 sheets | Load | Stratify | Select | Peak memory |
| :------------- | ---: | -------: | -----: | ----------: |
| Daf | 5.2 s | 4.7 s | 0.3 s | 630 MB |
| SqlDaf | 3.6 s | 11.5 s | 4.6 s | 284 MB |

- The imports alone take 171 MB. So the Daf holds about 460 MB, and the SqlDaf about 113 MB.
  At that rate the whole county, 1.6 million rows, would take about 5.5 GB as a Daf.
- 791 strata and 119 ballots on both. The same strata keys, the same ballots in the same order,
  and the same reason for each.
- An earlier run on the first 100,000 rows of sheet1's CVR gave the same agreement, with 2,920
  strata. There, the first `len()` of each stratum took 3.4 s of the walk's 3.7 s, by cProfile,
  since it resolves the selection. `group_where()` could keep the size of each group while it
  builds them. Not proposed yet.

`notes/sql_backing/acre_join_check.py` checks the join. `_join_cvr_votes_with_bif()` uses
`custom_translator_daf`, which SqlDaf's `join()` does not support. Renaming the BIF's shared
columns first, a plain join, then filling each shared column by the side that wins, gave the
same columns and the same rows:

- On the pilot's CVR rows and BIF: 133,422 rows. No shared value is empty there, but is_bmd differs
  on 1,402 sheets and precinct on all of them, so the rule of which side wins is used. Daf 3.7 s,
  SqlDaf 5.2 s.
- On a BIF made from 100,000 CVR rows, with every 7th value of the CVR and every 11th of the BIF
  blanked, so every fallback is used. Daf 2.3 s, SqlDaf 3.2 s.

## Decided

- Q248, 2026-10-18: step 15 keeps the aggregate. See the group reductions entry.

## Open

- Q252: merge sql_backing into claude_dev, so AuditEngine can run the port.
- Q230: the JSON dict cells of contest_variants.csv.
- The name for a group daf.
- `group_where()` keeping the size of each group.

## Pointers

- The port and its checks: notes/sql_backing/sdaf_variants.py, sdaf_variants_check.py,
  full_bif_forms_check.py, acre_strata_check.py, acre_join_check.py.
- The review of 2026-10-16: notes/sql_backing/ae_large_tables_review.md.
- The group reductions: 2026-10-18_1428_sqldaf_group_reductions.md.
- AuditEngine's request: notes/mail/to_daffodil/2026-10-17_0019_find_variants_port_task.md.
- The recommendations sent to AuditEngine: the mail of 2026-10-19 on SqlDaf pickup.
