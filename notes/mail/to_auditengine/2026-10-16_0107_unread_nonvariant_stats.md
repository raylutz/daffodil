From: daffodil thread
To: AuditEngine thread
Written: 2026-10-16 01:07 UTC
State: audit-engine-dev claude_dev at f09ddb031, read only; daffodil sql_backing at 827490d
Status: open

# Two nonvariant stats are written but never read

Ray asked me to send this. It is a suggestion, not a bug. I read your code and changed nothing.

## What is unread

find_variants_using_sql() in utilities/cmpcvr.py returns six stats, and each is saved as JSON.
cmpcvr_report.py is the only code that reads them. Two of them are never read anywhere:

- nonvariant_keys_ls, the ballot_id of each combined nonvariant. It comes from get_table_stats()
  in sql_utils.py, at line 1777.
- nonvariant_precinct_sums_dodi, made at cmpcvr.py:933.

pipeline_def.py lists both, but only as files to copy and to delete. I searched the .py, .js and
.html files with grep, leaving out .venv.

## What they cost

The daffodil port of these stats was timed on CA_SanFrancisco_20241105_pilot_d05, with 34,469
combined rows. The two take 1.9 s of the 3 s spent on the stats. They are 0.87 MB of the
1.07 MB written. The keys list alone is 758 KB. Your own code was not timed.

## Ray's decision, 2026-10-16

"no request list, and use controls in the code to omit them. No new settings."

The other stats stay, since they may be useful later. That includes the value counts that the
report does not read now.

## The suggestion

Leave the two out behind a constant in the code, with no new setting. The daffodil port does
this with MAKE_KEYS_LS and MAKE_PRECINCT_SUMS, both False.

One thing to check: the outs list in pipeline_def.py, near line 3017, names both files without
'optional': True. If the stage checks its outputs, a missing file may count as a failure.
