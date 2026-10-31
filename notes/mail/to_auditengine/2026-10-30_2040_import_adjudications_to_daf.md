From: AdjudiTally thread
To: AuditEngine thread
Written: 2026-10-30 20:40 UTC
State: ballot-viewer-app adjudicator-notes at 6f1cefb
Status: open

# import_adjudications still uses pandas

Ray asks that this move to Daf. It is a cleanup for after 2026-11-03.

## Ray's decision on the import fixes

Both changes in 6a9cf1f3e stand, as made.
- The disagreed count counts rows where audit==cvr_orig is not 1.
- A VotingSystem choice takes the adjudicated CVR record when one exists.

## What still uses pandas

In utilities/generic_cvr.py, import_adjudications:
- DB.load_data loads each cmpcvr group as a DataFrame, into cmpcvr_dodf.
- utils.group_df_to_dodf splits contest_variants by contest.
- The per-contest counts use df[...] with astype(int).
  These are detected, audit_gray_eval, audit_writeins and audit_overvotes.
- disagreed_count takes df['audit==cvr_orig'].tolist().
- The module imports lazy_pandas as pd.

voting_system_info and disagreed_count already take plain dicts and lists.

## Checked on sheet1 with Daf

The same numbers come out with Daf.from_csv on
resources/CA_SanFrancisco_20241105_sheet1/cmpcvr/contest_variants.csv:

    rows 14005
    audit==cvr_orig is 1: 10960
    disagreed: 3045
    VotingSystem source: cvr_modi_info 7536,
      audit_info 6287, cvr_orig_info 182

No reply is needed unless the plan changes how adjudications.csv is read.
