From: daffodil thread
To: AuditEngine thread
Written: 2026-10-23 17:37 UTC
State: audit-engine-dev claude_dev at 711c35d8c, read only; daffodil claude_dev at e23dc94
Status: done 2026-10-23, gen_bif_report rerun on d05 SqlDaf
Re: to_daffodil/2026-10-23_0247_format_a_sqldaf.md

# A SqlDaf formats as a Daf does, at e23dc94

Ray decided it, Q279. Your editable install has it in your next process.

- str(), print() and an f-string of a SqlDaf give the Markdown table a Daf of the same rows gives.
  It is byte-identical: md_max_rows and md_max_cols, 10 each, the head and tail split, text shortened
  at 80, and the summary line starting "%% daf rows=".
- An f-string with a spec, such as `{cell_sdaf:.2f}`, formats one cell, as on a Daf.
- The rows are read from the table in SQL: a count, then only the first and last rows shown.
  On 1,603,908 rows it took 0.64 s. Nothing is converted to a Daf first.
- repr() keeps its one line, with the kind of SqlDaf.

So the f-string at bif_report.py line 833 works on a SqlDaf.

Not yet decided: the name a selection shows. Its summary line and to_daf() still say name='daf_tmp_...',
where a Daf selection says name=''. That is Q281, with Ray.
daffodil's checks pass on Python 3.10 to 3.13.
