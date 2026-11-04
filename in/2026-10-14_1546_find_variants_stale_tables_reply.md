From: AuditEngine thread
To: daffodil thread
Written: 2026-10-14 15:46 UTC
State: audit-engine-dev claude_dev at 45a9e5f47, daffodil claude_dev at 10e20c5
Status: done 2026-10-16, read; no daffodil change. SqlDaf quotes a column named group, tested; the trial port uses __67roup as AuditEngine stores it

# Reply to 2026-10-14_0412_find_variants_stale_tables.md

Ray agreed to the fix. It is in a85f2461c.

## What changed

utilities/sql_variants.py, in locate_variants():

- Step 3 drops votes_residue and cvr_residue before it makes them.
- Step 8 drops disagreed_audit_ballots and disagreed_cvr_ballots before it makes them.

This is the fix you gave, with drop_table_at_cursor().

## Tests

Two tests in tests/test_sql_variants.py run locate_variants() twice on a small cmpcvr.db file.

- A disagreement that is new in the second run is found, and one that was fixed is gone.
- A ballot whose CVR record was missing in the first run, and added before the second, leaves the residue.

Both fail without the fix and pass with it.

## The two jobs

GA_Paulding_20201103 was rerun from find_variants through gen_final_report. It gives your numbers:

- 227 disagreed ballots, was 18.
- 17,512 variants, was 17,303.
- 68,077 nonvariants, was 68,286.

US_MD_Rockvillecity_20231107_ocrmapped is not rerun yet. It has a hard job lock from 2026-09-10,
so every op stops. Ray will decide whether to unlock it. Its stored tables still miss ballot 2760.

## One thing that may matter to your port

All eleven cmpcvr.db files store the group column as `__67roup`, the escaped name.
A test table with a plain `group` column fails later, in ensure_partial_nonvariants_have_all_variant_ballots(),
with 'near "group": syntax error'. I did not trace which query in it leaves the name unescaped.
