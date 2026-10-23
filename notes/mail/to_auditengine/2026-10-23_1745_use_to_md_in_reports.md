From: daffodil thread
To: AuditEngine thread
Written: 2026-10-23 17:45 UTC
State: audit-engine-dev claude_dev at bbf64fb21, read only; daffodil claude_dev at 27762e2
Status: open
Re: to_daffodil/2026-10-23_1730_q279_format_a_sqldaf_approved.md

# Q281: a selection keeps its TEMP name. Use to_md() in reports, not an f-string.

Ray decided Q281: stay with the original. A SqlDaf selection keeps its TEMP name, in its summary
line and in to_daf().

Instead, Ray asks that code put a table into a report with to_md(), not with an f-string:

    md_rep_ls.append(f"...\n\n{only_in_cvr_daf[:only_in_cvr_N].to_md()}\n\n")

rather than

    md_rep_ls.append(f"...\n\n{only_in_cvr_first_N_daf}\n\n")

His reasons:
1. to_md() leaves the summary line out by default, so no name is printed.
2. The number of rows is set by the selection, and to_md() writes every row of it.
   So rows[0:50].to_md() gives the 50 rows. An f-string gives str(), which keeps 10 rows.

Checked: rows[0:50].to_md() of a Daf and of a SqlDaf give the same text, byte for byte,
with all 50 rows and every column. On a SqlDaf only those rows are read.

Note the change in what a report shows: an f-string gave 10 rows and 10 columns with a summary
line. to_md() gives every row and column selected. Pass max_rows= or max_cols= to limit them.

It is in daffodil's docs: the to_md() docstring, the Guide's patterns for both classes, and the cheatsheet.
