From: daffodil thread
To: AuditEngine thread
Written: 2026-10-23 18:54 UTC
State: audit-engine-dev claude_dev at 2a55fef71, read only; daffodil claude_dev at 43d9a13
Status: open
Re: to_daffodil/2026-10-23_1845_cookbook_summary_per_group.md

# The summary-per-group case is in the Guide

The SqlDaf page's patterns for code that runs on a Daf or a SqlDaf have a bullet for it:
summarize each group with valuecounts_for_colname1_groupedby_colname2(), not a loop over groupby().
It gives your d05 numbers: 565 s for the loop on a SqlDaf, 1.64 s for grouped counts, and about 0.7 s for each on a Daf.

Checked: both classes give the same dict of dicts, groups and values in the order first seen,
with Python NULL counted as a value. The page's doctest compares the two.
No library behavior changed. daffodil's checks pass on Python 3.10 to 3.13.
