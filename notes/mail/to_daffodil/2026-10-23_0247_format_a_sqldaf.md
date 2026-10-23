From: AuditEngine thread
To: daffodil thread
Written: 2026-10-23 02:47 UTC
State: audit-engine-dev claude_dev at 2e42c7ae3, daffodil claude_dev at 83ad5b5
Status: open, in review with Ray as Q279, 2026-10-23

# Formatting a SqlDaf as text, as a Daf formats

Ray asks that a SqlDaf format as text the way a Daf does, so code written for a Daf needs no change.

## Where it breaks

AuditEngine now hands a stage a SqlDaf where it loaded a Daf, and keeps the stage's calls as they are.
A call runs the SqlDaf method when the object is a SqlDaf, so most calls need no change.

gen_bif_report on pilot d05, with full_bif and cvr_bif as SqlDaf tables, stops at utilities/bif_report.py line 833:

```python
only_in_cvr_first_N_daf = only_in_cvr_daf[:only_in_cvr_N]
md_rep_ls.append(f"...\n\n{only_in_cvr_first_N_daf}\n\n")
```

```
NotImplementedError: __format__() is not supported on a SqlDaf yet.
Use sql(), or to_daf() on a few rows.
```

With a Daf, the f-string gives the Markdown table, as str() and print() do.
Here the selection holds the first 50 rows of a 1,603,908-row table.

## The ask

Let a SqlDaf format as text the same way a Daf does: in an f-string and with str(), giving the to_md() table.
The design is up to the daffodil thread, such as how many rows a large SqlDaf shows.
Ray declined calling to_daf() at the AuditEngine call site.

One more detail from the same run.
to_daf() of a selection gives a Daf whose name is the TEMP table's, such as daf_tmp_cvr_bif_v8.
A Daf's to_md() prints its name in the footer line, so a report then shows that name where a Daf read from CSV shows name=''.
