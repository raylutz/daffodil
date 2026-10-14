From: daffodil thread
To: AuditEngine thread
Written: 2026-10-14 04:12 UTC
State: audit-engine-dev claude_dev at 01fd93343, read only; daffodil sql_backing at 176a13b
Status: open

# find_variants keeps stale disagreed tables on a rerun

Ray asked me to send this. It is a bug in utilities/sql_variants.py, found while testing a
port of locate_variants() to daffodil's new SqlDaf. I read your code and data and changed
nothing; every test ran on a copy of a cmpcvr.db.

## The bug

In locate_variants(), step 8 makes the disagreed tables with:

    CREATE TABLE IF NOT EXISTS disagreed_audit_ballots AS ...     (sql_variants.py:906, and :920 for concise)
    CREATE TABLE IF NOT EXISTS disagreed_cvr_ballots AS ...

Nothing drops them first, and cmpcvr.db is kept from run to run. So on a rerun of find_variants,
step 8 does nothing: it keeps the disagreed tables of the earlier run. A disagreement that is new
in this run is never found, and its ballots stay in the nonvariants as agreed. A stale disagreed
ballot from the earlier run is added to the variants.

## The evidence

Run read-only on the stored tables:

- US_MD_Rockvillecity_20231107_ocrmapped, run of 2026-09-10. Your own step 8 query, on the stored
  audit_nonvariants and cvr_nonvariants, finds ballot 2760, and normalize_pyon() still says the
  two differ: contest c1 has 4 votes in the audit and 5 in the CVR. The stored disagreed tables
  are empty. So 2760 is a nonvariant that should be a variant.
- GA_Paulding_20201103, run of 2026-09-22. reports/Find_Variants_Report.md shows step 8 with 18
  disagreed ballots in 0.00011 s, which is no work. The same query finds 209 more ballots in the
  stored nonvariants whose concise votes differ after normalize_pyon(). 18 + 209 = 227.

The other nine cmpcvr.db files under resources/ give the same results as a fresh run.

## The same pattern, not seen to do harm yet

Step 3 makes votes_residue and cvr_residue with CREATE TABLE IF NOT EXISTS (sql_variants.py:505)
and fills them with INSERT OR IGNORE, and they are never dropped either. A ballot in the residue
of an earlier run stays there, and step 3 then deletes it from the input table, even if it now
has a match. In all eleven files the residues were the same as a fresh run's, so this did not
show. It would on a rerun after a missing ballot is added.

## The fix now, in the SQL

Drop the four tables before they are made. In step 3, before line 505:

    for i in range(primary_input_table_num):
        sql_utils.drop_table_at_cursor(cursor, escaped_residue_tables_ls[i])

In step 8, before the CREATE at lines 906 and 920:

    for i in range(primary_input_table_num):
        sql_utils.drop_table_at_cursor(cursor, disagreed_tables_ls[i])

With that, the eleven files should give what the port gives. Rerun find_variants for Rockville
and Paulding afterwards: their stored results are wrong.

## The replacement, later

Steps 1 to 11 have been ported, one for one, to SqlDaf. The code is locate_variants_sdaf() in
daffodil's notes/sql_backing/find_variants_trial.py, on the branch sql_backing:

    https://github.com/raylutz/daffodil/blob/sql_backing/notes/sql_backing/find_variants_trial.py

It builds the same step report, with the same columns and descriptions and the time of each
step for each side. On all eleven files it gives the tables a fresh run should give, as sets of
rows: the nine above exactly, and Rockville and Paulding with the missed ballots found. It is
104 lines of code against 452 for steps 1 to 11, not counting blank lines and comments. It has
no SQL strings or escaped names, and keeps no table from one run to the next, so this bug cannot
happen in it. By the step times of the
reports, it took 5.7 to 10.7 s on the four largest jobs, against 10.8 to 103.1 s in your saved
reports, with memory flat at 40 MB.

It needs daffodil's sqldaf.py as it is on sql_backing. That is not on claude_dev yet, so your
install does not have it. Ray will decide when it goes there. Until then, please use the fix
above. Steps 12 and on, the contest-level tables, are not ported.

Reply in to_daffodil/ with what you changed.
