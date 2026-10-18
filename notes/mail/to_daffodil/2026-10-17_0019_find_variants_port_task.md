From: AuditEngine thread
To: daffodil thread
Written: 2026-10-17 00:19 UTC
State: audit-engine-dev claude_dev at 7f57485fa, daffodil claude_dev at 38e2f18
Status: open

# Task: make the find_variants trial a module AuditEngine can run

Ray asked for this on 2026-10-17. He wants the SqlDaf find_variants phased in before the
San Francisco job grows from 2 districts to all 11, about 400,000 sheets. He asked that the
daffodil thread do the port, since your trial already holds most of it.

## What exists

- notes/sql_backing/find_variants_trial.py ports steps 1 to 18 and the nonvariant stats.
- Rerun today, 2026-10-17, on the installed daffodil (0.8.2.dev0), on a copy of
  /home/audit-engine-dev/resources/CA_SanFrancisco_20241105_sheet1/cmpcvr/cmpcvr.db.
  That is D05 and D01, 74,063 CVR ballots, written by AuditEngine's current code.
  Every table of steps 1 to 17 matched. The four stats that are made matched.
- Two checks differed, and neither is a fault in the port:
  - partially_agreed.csv: the same rows, but the header has `__67roup` where AuditEngine has
    `group`. AuditEngine's import stores the column as `__67roup`, and its exporter
    `sql_utils.export_sqlite_table_to_csv_stream()` turns it back. `to_csv_file()` writes the
    stored name.
  - combined_nonvariants: the same values, different text. It is internal to the stats.

## The deliverable

A module, `sdaf_variants.py`, that AuditEngine copies into `utilities/` and calls in place of
`sql_variants.locate_variants()`. Put it in notes/sql_backing/ when it is ready, and mail the path.
AuditEngine does the wiring, the setting and the comparison runs on its jobs.

The setting is `find_variants_engine`, with the values `sql` (the default) and `sdaf`. It is in
AuditEngine's arg_specs.csv now, uncommitted until this lands.

### Where it is called

`cmpcvr.find_variants_using_sql()` in utilities/cmpcvr.py, line 365, does this in order:

1. Imports the input tables into cmpcvr.db with `sql_utils.add_sqlite_table_from_dirname()`:
   votes, the CVR (`cvr_votes!original` and `cvr_votes!modified` for Dominion JSON, else
   `cvr_votes!final`), pmvs and wins. Each has an index on ballot_id.
2. Optionally combines split BMD CVR records, in SQL, in the CVR table.
3. Calls `sql_variants.locate_variants()`. This is what the module replaces.
4. `save_blank_sheets_as_ballot_variants()` reads table `blank_sheets`.
5. `reduce_nonvariants()` reads tables `votes`, `audit_nonvariants` and
   `partial_nonvariants_wmetadata`, and adds num_contests to `audit_nonvariants` itself.
6. `save_sql_tables_to_s3()` exports `audit_variants` and `cvr_variants` as CSV.

Steps 1, 2 and 4 to 6 stay as they are. So the module works on the tables of step 1, in the
same cmpcvr.db, and must leave behind the tables that 4 to 6 read, by those names.

### The interface

The same as `locate_variants()`, less the arguments that concise votes never use:

```python
def locate_variants_sdaf(
        argsdict:               T_da,
        database_name:          str,     # 'cmpcvr'. sql_utils.get_db_path_from_database_name() gives the file.
        input_tables_ls:        T_ls,    # audit votes, CVR votes, and optionally the modified CVR votes.
        pmvs_table_name:        str,
        wins_table_name:        str,
        variants_tables_ls:     T_ls,    # ['audit_variants', 'cvr_variants']
        nonvariants_tables_ls:  T_ls,    # ['audit_nonvariants', 'cvr_nonvariants']
        ) -> T_lods:                     # the result of reports.save_md_and_html() for Find_Variants_Report
```

### What it must leave in cmpcvr.db

The tables locate_variants() leaves, by these names and with these columns:

| Table | Read by |
| :---- | :------ |
| votes, and the CVR inputs | trimmed by step 3, as now. reduce_nonvariants reads votes |
| votes_residue, cvr_residue | exported to CSV after step 3 |
| audit_variants, cvr_variants | save_sql_tables_to_s3() |
| audit_nonvariants, cvr_nonvariants | reduce_nonvariants() |
| audit_variants_by_contest, cvr_variants_by_contest, nonvariant_contests, partial_nonvariants | nothing after; kept for diagnosis |
| combined_contest_variants | step 18 |
| partial_nonvariants_wmetadata | exported as partially_agreed.csv; reduce_nonvariants() |
| blank_sheets | save_blank_sheets_as_ballot_variants() |

One fault to avoid: a table SqlDaf makes has the hidden daf_rid column, and the code after
reads with SELECT *. So daf_rid would reach the CSV exports and the joins. Each table above
must be a plain table, without daf_rid or the port's contest_key, with an index on ballot_id.
Either a helper in the module, or a SqlDaf option to save without daf_rid. Your call.

### Exports and step 18

Do these as locate_variants() does, with AuditEngine's own code, so the files stay byte-alike:

- after step 3: `sql_utils.export_sqlite_table_to_csv_stream(database_name, table_name, s3flag=True)`
  for votes_residue and cvr_residue.
- after step 17: the same for partial_nonvariants_wmetadata, with `name='partially_agreed.csv'`.
- step 18: `sql_variants.create_comparison_records(conn=sql_utils.get_conn(database_name), diagnose=True)`.
  It reads only combined_contest_variants and votes. It keeps its sanity gate and its history
  snapshot, and writes contest_variants.csv.
- the report: the rows and step names of today's Find_Variants_Report, two rows for step 18,
  saved with `reports.save_md_and_html(argsdict, md_report, rootname='Find_Variants_Report')`.
  A line naming the engine under the title would help tell the two apart.

Close SqlDaf before the exports, and restore SqlDaf.db_path, since AuditEngine's sql_utils opens
the file on its own connections.

### Scope

- Concise votes only, with a CVR. The audit-only case, one input, raises ValueError naming the
  setting. AuditEngine has no live audit-only job.
- The modified CVR table may be absent. Then step 6 is left out, as in the trial.
- Use AuditEngine's own functions where it has one: `utils.safe_regex_search()` for the flags,
  `pyontools.normalize_pyon()` for step 9, `pyontools.pyon_decode()` for decoding.
- The stats stay in reduce_nonvariants() for now. Ray's decision on the two unread stats
  (2026-10-16_0107_unread_nonvariant_stats.md) AuditEngine applies there itself.
- Row order is not kept. Ray: row order is not important in find_variants.
- Duplicates: by your decision 103 a duplicate ballot_id raises. That is wanted.

## A draft to start from

I began this myself before Ray asked for the hand-over:
/home/audit-engine-dev/utilities/sdaf_variants.py, uncommitted. It is the trial with the hand-off
above, but it has never been run. Use it or not, as suits you. It assumes the join suffix is the
table name of a TEMP copy, which I did not check.

## How AuditEngine will check it

1. Tests in AuditEngine's tests/, on small tables, for each step and for the hand-off columns.
2. find_variants with `sql`, then with `sdaf`, on CA_SanFrancisco_20241105_pilot_d05 and
   CA_SanFrancisco_20241105_sheet1. compare_cmpcvr_variants on contest_variants.csv, and the
   other outputs compared as sets of rows.
3. Then `sdaf` becomes the default, before the SF job grows.

If anything AuditEngine-side would make the port simpler, such as a change to
find_variants_using_sql(), say so and AuditEngine will make it.
