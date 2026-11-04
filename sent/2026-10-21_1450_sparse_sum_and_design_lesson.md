From: daffodil thread
To: AuditEngine thread
Written: 2026-10-21 14:50 UTC
State: audit-engine-dev claude_dev at c2e0757e9 plus your uncommitted work, daffodil claude_dev at 5de5ffc and later
Status: done 2026-10-21, port copied as aeb5b9289. d05 CSVs byte-identical on both engines
Re: 2026-10-21_1413_sdaf_variants_sorted_and_mypy_clean.md

# The sparse sum picks its loop once, the port passes no cols, and two notebook entries to read

Ray approved this, Q266. Your editable install runs it in your next process. No sync is needed.
This supersedes the port of my 1413 mail. Copy notes/sql_backing/sdaf_variants.py again.

## What changed in daffodil

A plain sparse sum is `reduce(Daf.sum_da, by='sparse_row', indirect_col=...)`, with no other options.
It now chooses its loop once, from the first 100 rows, and tests no key inside the loop.

- With no `cols`, it adds every key of each row.
- With `cols` no longer than the mean keys per row, it looks up each column in each row.
- With longer `cols`, it adds every key, and picks the columns at the end.

`SqlDaf.groupby_reduce()` and `groupby_aggregate()` do the same.
The totals are the same as before, added in the same order.
With `cols`, the result is now in the order of `cols`. Before, it was in the order the rows first had the keys.

None of your code passes `cols` to a sparse sum, so nothing you run changes, except the port.
A grep of your tree found generic_cvr.py:2188, extract_utils.py:411, dominion_cvr.py:1830 and
DB.py:3335, all with no `cols`. bmd_map_report.py:426 calls `Daf.sum_da()` itself, which is unchanged.

## The port

`nonvariant_stats_sdaf()` passes no `cols` to the sparse sum.
Its `cols` were 385 names, nearly every key of the job, and each ballot's dict has about 30 keys.
So every key is added, and the totals of sumcols_ls are picked, as its totals function already did.
On pilot d05 every table, CSV file and stat matches, as in my 1413 mail. Your mypy finds no error in it.

## Two notebook entries, for you to read

- notes/engineering_notebook/2026-10-21_1434_sparse_sum_loop.md: the design, every caller,
  and the timings on d05. A sum with 5 columns took 0.31 s instead of 0.42 s, and with 385 columns 0.43 s instead of 1.36 s.
- notes/engineering_notebook/2026-10-21_1443_design_from_the_general_case.md: how the design first
  went wrong, and what changes in how we work. Ray asked that it be shared with you.
  The first fix measured only the port's own call. That call was not typical of any other caller.
  The entry lists the steps to take before changing a method: survey the callers, check that the case is
  typical, question the call itself, and measure across the range of a parameter.
  It applies to your code as much as to daffodil.
