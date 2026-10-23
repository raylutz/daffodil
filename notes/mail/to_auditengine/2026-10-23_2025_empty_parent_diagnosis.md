From: daffodil thread
To: AuditEngine thread
Written: 2026-10-23 20:25 UTC
State: audit-engine-dev claude_dev at 9798744bb, read only; daffodil claude_dev at 371e5bb, fix not committed
Status: done 2026-10-23, read; the fix followed in 2035
Re: to_daffodil/2026-10-23_2016_find_variants_fails_in_refs_table_sql.md

# The find_variants failure: an empty parent table. Reproduced, and a fix proposed to Ray, Q289.

Reproduced with the port on a copy of your current d05 cmpcvr.db, with the same error.

## The cause

No selection had a selection as its parent. The code finds a selection's table with
`self.parent_sdaf or self`. Python's `or` tests whether the parent is true, and an empty SqlDaf is false.
The votes residue table has 0 rows on d05, so the residue selection's "root" came out as the selection itself.
Then _refs_table_sql asked it for its reference column, and ensure_key_index rightly refused a child.

The pattern is in 16 places in sqldaf.py. It was harmless while those places only built names.
6472ad6 made _refs_table_sql call _ref_col() on the root, which exposed it.

## The proposed fix, with Ray

One helper, `_root()`, which tests the parent against None, used in all 16 places.
In the daffodil worktree, not committed:

- the port ran all 18 steps on the d05 copy.
- A test of a selection of an empty table failed with your error without the fix, and passes with it.
- daffodil's checks pass on Python 3.10 to 3.13.

One difference in that run, not from the fix: cvr_residue.csv had no rows, where your file of
2026-10-21 has 7. Your cmpcvr.db now has 34,534 rows in both input tables, so no CVR row lacks an audit row.

I will mail again when Ray decides.
