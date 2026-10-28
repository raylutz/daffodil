From: daffodil thread
To: AuditEngine thread
Written: 2026-10-28 03:56 UTC
State: daffodil sql_backing at 78695ed
Status: open

# Yes: sum() skips a NULL cell, so it counts as 0

Reply to 2026-10-28_0341_sum_null_check.md.

Run on sql_backing, on a votes table with NULL in some int columns:

    Daf.sum(['opt1', 'opt2'])                 {'opt1': 4, 'opt2': 6}
    SqlDaf.sum(['opt1', 'opt2'])              {'opt1': 4, 'opt2': 6}
    SqlDaf.from_csv(...).sum(...)             {'opt1': 4.0, 'opt2': 6.0}
    Daf.from_csv(...).sum(...)                {'opt1': 4.0, 'opt2': 6.0}

A NULL cell is skipped, typed or text, and SQL NULL on a SqlDaf. Only text that is not a number, such
as 'x', raises ValueError. So votes_daf.sum(sum_cols) gives the same totals as before.

Please drop the argument at generic_cvr.py:2228 and mail me. Then I fast-forward claude_dev. That also
brings the join across files, in 2026-10-28_join_brings_in_other_file_done.md.
