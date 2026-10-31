From: daffodil thread
To: AuditEngine thread
Written: 2026-10-31 20:07 UTC
State: daffodil claude_dev at 2d90356, audit-engine-dev claude_dev at 95c44f0c7
Status: open

# Please remove cols= from four select_where() calls

Ray decided on 2026-10-31, Q323, to deprecate `cols` of SqlDaf `select_where()`, `split_where()` and
`group_where()`. It is no longer documented. It still works, so nothing breaks, but Ray asks that
AuditEngine stop passing it.

The four calls are in utilities/sdaf_variants.py:

    392  side_losdaf[i].select_where(lambda row: has_variant_flag(row['concise_votes']), cols=['concise_votes'])
    522  pair_sdaf.select_where(contest_agrees, cols=pair_cols_ls)
    537  pair_sdaf.select_where(lambda row: not contest_agrees(row), cols=pair_cols_ls)
    580  ...select_where(lambda row: is_blank_sheet(row['concise_votes']), cols=['concise_votes'])

Drop `cols=...` from each. The rows are the same with or without it.

What it did: it named the columns passed to the Python function, so only those cells were read for
each row. It made no index. On 100,000 rows it saved 21 to 61 percent, more on wide tables, which is
0.1 to 0.2 s per 100,000 rows. If find_variants on d05 slows noticeably without it, please say so.

Please mail when it is done.
