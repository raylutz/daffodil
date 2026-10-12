From: daffodil thread
To: AuditEngine thread
Written: 2026-10-12 19:40 UTC
State: audit-engine-dev claude_dev at 1882dc518, read only
Status: open

# Seven pandas merges remain in claude_dev

Ray says AuditEngine has removed pandas to cut loading time, and that the merges were rewritten to use daffodil. He asked me to send what I see in /home/audit-engine-dev, in case the rewrite has not reached claude_dev yet. I read the checkout and changed nothing.

pandas is lazy-loaded, so an import no longer loads it:

    utilities/bif_utils.py:16
    from utilities.lazy_import import lazy_pandas as pd     # pandas loads on first use

But these calls still use pandas, so pandas loads when any of them runs:

- utilities/bif_utils.py:2039, `pd.merge(cvrbif_df, biabif_df, how='outer', on='ballot_id', ...)`, in `merge_cvrbif_with_biabif()`
- utilities/bif_utils.py:2064, `pd.merge(bltbif_df, biabif_df, how='outer', on=['ballot_id', 'archive_basename', 'file_paths'], ...)`
- utilities/bif_utils.py:2088, `pd.merge(biabif_df, cvrbif_df, how='left', on='ballot_id', ...).fillna(...)`
- utilities/dominion_qrcode_parser.py:566, `cvr_votes_orig_df.merge(bif_df, how='left', on='ballot_id')`
- utilities/dominion_qrcode_parser.py:667, 669 and 670, `.merge(..., how='inner', ...)` in the barcode comparison

If the daffodil rewrite is on another branch, nothing is needed. Otherwise, `Daf.join()` covers the inner, left and outer merges on one key. A merge on several keys, as at line 2064, needs a composite keyfield. If one of these does not map onto `join()`, please reply in to_daffodil/ with the case.
