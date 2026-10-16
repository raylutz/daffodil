# AuditEngine's large tables and SqlDaf

Review of 2026-10-16, for Ray. It covers full_bif and the CVR votes table, the two that reach
millions of rows: 4.5 million in Los Angeles, 2.1 million in Maricopa.

## find_variants and the merge

- find_variants runs on SqlDaf in the trial port, find_variants_trial.py.
- The merge of cvr_bif and bia_bif into full_bif joins hunk by hunk, so no step holds the whole table.

## full_bif

It is loaded at about 25 places, all through bif_utils.simple_recover_full_bif_daf().
Each call form was run on a SqlDaf and on a Daf, and the results compared.

Works and matches Daf: select_where with a lambda, select_by_dict, select_krows,
select_records_daf, col(unique=True), cols_to_dol, the value counts, groupby with the per-group
calls of the BIF summary report, groupsum_daf, `sdaf[id, col] = v`, len and bool.

Added at 2463d18: `sdaf[id, col].to_value()`, `sdaf[id_list]` and `sdaf[0, col]`.
Added at 307848b: attach_schema().

Still open:

- apply_dtypes() on a small selection, at 6 places, as in `sdaf[id].apply_dtypes()`. Q175.
- apply_dtypes() of a list column. BifSchema's file_paths is a list. Q177.
- Sorting in place, as in votes_extractor.py, on a selection of full_bif. Q178.

A pattern that is slow on a Daf too: `ballot_id in full_bif_daf.keys()`, once per ballot, in
bmd_map_report.py and ballot_exceptions_report.py. keys() builds a new list each call.
`ballot_id in full_bif_daf` is a lookup on both.

## The CVR votes table

The grouped totals, in dominion_cvr.py and DB.combine_dirname_chunks(), load the whole table,
group it, and sum each group's numbers and its concise_votes dict. This is where AuditEngine
reached 13.2 GB. On 100,000 rows:

    Daf groupby + daf_sum x2            0.73 s
    SqlDaf groupby + reduce, no index  14.8 s
    same, with an index on precinct     2.2 s

- groupby makes no index, so each group is a full scan. Q176.
- concise_votes can be stored as JSON text, and reduce(by='sparse_row') reads it.
  apply_dtypes() cannot make it a dict. Q177.

### concise_votes stays as Pyon text

Ray decided on 2026-10-16, Q177. The column stays as the Pyon text that csv.writer() writes,
such as {'c17b': 21}. Nothing converts it when it is stored. It is decoded only when a reduction
uses it. Any conversion code goes in a Pyon utilities module.

pyjson5 reads this text as it is. On 100,000 dicts of 30 keys:

    pyjson5.loads on the Pyon text             0.81 s
    unflatten_val, which tries pyjson5 first   0.86 s
    json.loads on the same dicts as JSON       0.91 s
    safe_convert_json_to_obj, used by reduce   2.05 s
    ast.literal_eval                          11.88 s

pyjson5 rejects None, True, a tuple and a dict with int keys. unflatten_val falls back to
ast.literal_eval for those.

## When AuditEngine moves to SqlDaf

Ray decided on 2026-10-16, Q187. AuditEngine manages its own runs. The SF sheet 1 run grows
in steps: the D05 pilot first takes in District 01, and once that works, all 11 districts.
That is about 400,000 sheet 1 images. The runs stay on Daf until the step to all districts.
SqlDaf becomes useful at that step, and that is when AuditEngine starts to use it.
Meanwhile AuditEngine is phasing in the flat file_path and file_path2 columns.

Measured on 2026-10-16, as a Daf of text, before apply_dtypes():

    D05 pilot full_bif, all 4 cards        133,422 rows    241 MB
    older full-county full_bif           1,603,862 rows   2675 MB
