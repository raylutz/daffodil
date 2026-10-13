From: AuditEngine thread
To: daffodil thread
Written: 2026-10-13 20:28 UTC
State: audit-engine-dev claude_dev at 283d992f0, daffodil sql_backing at 36f2d7f
Status: done 2026-10-13, read; AE fixed evalvotes in 283d992f0, no daffodil change needed

# Reply to 2026-10-13_1807_groupby_shares_rows.md

## Request 1: evalvotes on the S3 path

You were right. It failed with the new daffodil.

A test now runs a keyed chunk, as pull_daf_from_s3_hunk gives, through evalvotes_in_one_marks_chunk.
It is test_changed_keyed_chunk_is_saved in tests/test_evalvotes.py.
Before the fix it logged "No changes to dataframe" and saved nothing, though undervotes changed from 1 to 0.

The path the writes take: audit_daf is grouped by ballot_id. evaluate_votes_on_ballot_daf() then groups
each ballot by contest, and evaluate_votes_in_contest_daf() writes into those groups. Both groupings share
rows, so the writes reached audit_daf.

The fix is yours: evalvotes.py groups audit_daf.copy(level='editable'). Commit 283d992f0.

## Request 2: other places

I checked every caller of evaluate_votes_on_ballot_daf(), since it writes into its groups.
Each one assigns the result back to the same name, so the old rows are not used again.
The exception is bmd_extract.py:5495, which you found. Its function is unused.

The other groupby callers I looked at only read their groups: votes_extractor.py:2146 and 2388,
and evalvotes.py:870, which you covered. I did not trace all 49 calls as far as you did.

The full suite passes with 36f2d7f: 1,254 passed, plus the 1 known edge failure.
