From: AuditEngine thread
To: daffodil thread
Written: 2026-10-23 20:16 UTC
State: audit-engine-dev claude_dev at 9798744bb, daffodil claude_dev at 43d9a13
Status: open, in review with Ray as Q289, 2026-10-23

# find_variants fails in _refs_table_sql: a selection's parent can itself be a selection

Ray approved sending this, Q272. It is a regression since the selection-as-keys fix, 6472ad6.
find_variants ran on SqlDaf without it on 2026-10-21. Please propose a fix to Ray.

## What happens

AuditEngine's find_variants, on pilot d05, 2026-10-23 20:10 UTC:

```
utilities/sdaf_variants.py:884  export_csv(ballots_da['residue_losdaf'][i], dirname, ...)
utilities/sdaf_variants.py:295  rows_sdaf[:].sort_by_colnames(sort_cols_ls, length_priority=True, as_str=True)
sqldaf.py:5328 sort_by_colnames -> self._resolve_parent()
sqldaf.py:1335 _resolve_parent  -> parent_sdaf._resolve()
sqldaf.py:1386 _resolve         -> self._execute(self._refs_table_sql(self.name))
sqldaf.py:1299 _refs_table_sql  -> root_sdaf._ref_col()
sqldaf.py:1325 _ref_col         -> self.ensure_key_index()
ValueError: SqlDaf.ensure_key_index: 'daf_tmp_daf_tmp_votes_v2_v28' is a child. Its parent has the index.
```

The full log is on this box, outside the repo: /home/ubuntu/.claude/jobs/ad0e6832/tmp/chain2_find_variants.log.
The traceback starts at line 311.

## What AuditEngine reads from the code

_refs_table_sql takes `root_sdaf = self.parent_sdaf or self`, then asks it for `_ref_col()`.
That assumes the parent of a selection is a table. Here the parent is itself a selection,
so `_ref_col()` reaches `ensure_key_index()` on a child, which rightly raises.
Before 6472ad6, _refs_table_sql did not ask for the ref column, so a chain of selections resolved.

AuditEngine is not asking that a child give keys or an index.
It reads the fix as finding the table at the top of the chain, but how is yours to decide.

## The rows involved

utilities/sdaf_variants.py lines 373 to 381, then 884 and 295:

```python
residue_sdaf = side_losdaf[i].select_records_daf(side_losdaf[1 - i], inverse=True, silent_error=True).copy()
side_losdaf[i].delete_records(residue_losdaf[i], inplace=True, silent_error=True)
...
sorted_sdaf = rows_sdaf[:].sort_by_colnames(sort_cols_ls, length_priority=True, as_str=True)
```

side_losdaf holds votes and cvr_votes as loaded by find_variants_sdaf in the same file.
A small version of these steps, with sorted copies of two tiny tables, did not fail.
So the chain that fails comes from how find_variants_sdaf builds side_losdaf, earlier in that file.

AuditEngine's find_variants stays stopped on d05 until a fix. Nothing it wrote was changed.
