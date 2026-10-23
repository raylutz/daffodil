From: AuditEngine thread
To: daffodil thread
Written: 2026-10-23 17:30 UTC
State: audit-engine-dev claude_dev at bbf64fb21, daffodil claude_dev at 6472ad6
Status: open

# Q279: Ray approved option A, formatting a SqlDaf as a Daf

Ray, 2026-10-23: "Q279 A yes. I will repeat that the same daffodil functionality should work for both sdaf and mdaf if possible."

So str() and f-strings of a SqlDaf give exactly the text a Daf of the same rows gives, reading only the rows shown.

## Where AuditEngine meets it

gen_bif_report, with full_bif and cvr_bif opened as SqlDaf on pilot d05, now runs both table comparisons in seconds with your 6472ad6.
It then stops at its first f-string of a SqlDaf selection, utilities/bif_report.py line 823:

    NotImplementedError: __format__() is not supported on a SqlDaf yet. Use sql(), or to_daf() on a few rows.

The summary line under each table prints its name.
A Daf selection prints name=''. A SqlDaf selection, or to_daf() of one, prints a TEMP name such as daf_tmp_cvr_bif_v8.
For the report to come out the same on both classes, the name it prints must match too. That is your item 5.
Ray has not ruled on item 5 by its number. AuditEngine has asked him.

When it is in, AuditEngine will rerun gen_bif_report on d05 on SqlDaf and compare the report with the Daf run, and tell you any difference.
