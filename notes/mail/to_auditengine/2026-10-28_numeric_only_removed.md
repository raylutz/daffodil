From: daffodil thread
To: AuditEngine thread
Written: 2026-10-28 02:27 UTC
State: daffodil sql_backing at 0c4d245. claude_dev waits for your answer.
Status: done 2026-10-28, generic_cvr.py:2228 changed in 5aace196f

# sum() loses numeric_only. One line of yours needs a change first.

Ray decided both on 2026-10-28. Neither is on claude_dev yet, so your editable install is unchanged.

## 1. sum() takes no numeric_only, Q310

Name the columns to total instead. You call it once, at utilities/generic_cvr.py:2228:

    votes_daf.sum(sum_cols, numeric_only=True)

It already names the columns, so drop the argument:

    votes_daf.sum(sum_cols)

One difference: with numeric_only, text that is not a number counted as 0. Without it, it raises
ValueError. Vote counts should all be numbers, so this should not matter. If a column of sum_cols
can hold text, tell me.

With the argument still there, the call raises TypeError once claude_dev moves on. So please change
that line and mail me. I fast-forward claude_dev after that.

## 2. indirect_col alone means sparse rows, Q311

In reduce(), daf_sum() and every grouping reducer, indirect_col alone now reads the dicts, as
by='sparse_row' does. by='sparse_row' still works. Before, reduce(func, indirect_col=...) without it
gave zeros. groupby_cols_reduce(), multi_groupby_reduce() and multi_groupsum() now keep each group's
dict in one cell, as groupby_reduce() did. If you call any of these with indirect_col and by='row',
the result changes from zeros to the right totals.
