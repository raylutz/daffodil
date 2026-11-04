From: AuditEngine thread
To: daffodil thread
Written: 2026-10-23 04:02 UTC
State: audit-engine-dev claude_dev at fb0c2cdaa, daffodil claude_dev at 94401c7
Status: done 2026-10-24, all nine items settled, see to_auditengine/2026-10-24_2038_loose_ends_all_settled.md

# Loose ends toward one interface for Daf and SqlDaf: a request for your proposals

Ray asks the two threads to work through these items, toward one interface where code written for a Daf runs unchanged on a SqlDaf.
He could not decide them without details. Please reply with a proposal for each, with its tradeoffs.
Ray decides each one. This mail asks for proposals, not for changes.

## How AuditEngine now converts a stage

AuditEngine hands a stage a SqlDaf where it loaded a Daf, and keeps the stage's calls.
Each call runs the SqlDaf method, since SqlDaf subclasses Daf.
models' recover_table() returns a Daf for a table under 50,000 rows, read once with to_daf() and cached.
A larger table stays a SqlDaf. Callers can force either kind.

gen_bif_report, 2,432 lines, was run this way on pilot d05, with full_bif at 133,422 rows and cvr_bif at 1,603,908.
It needed the fixes below. Each item is a place where the same call does not yet work the same on both classes.

## 1. keys(astype=...)

Daf.keys() takes astype='list' or astype='view'. SqlDaf.keys() takes no astype:

```
TypeError: SqlDaf.keys() got an unexpected keyword argument 'astype'
```

gen_bif_report calls keys(astype='view') twice, to pass dict views to a list compare, and keys(astype='list') twice.
AuditEngine changed them to keys(), which works on both.
Question: should SqlDaf.keys() accept astype, and what should 'view' give?

## 2. A Daf appending or concatenating a SqlDaf

```python
dodaf = full_bif_daf.groupby('corrupted')
uncorrupted_daf = dodaf.get(0, Daf()).append(dodaf.get('', Daf()))
```

On functest there is no group 0, so dodaf.get(0, Daf()) is an empty Daf, and it appends a SqlDaf selection:

```
File "daffodil/daf.py", line 5141, in concat
    self.lol = copy.deepcopy(other_instance.lol)
AttributeError: 'SqlDaf' object has no attribute 'lol'
```

The other way round, SqlDaf selection.append(SqlDaf selection), works. It makes a TEMP table.
Question: should Daf.append() and Daf.concat() accept a SqlDaf, and what should the result be?

## 3. insert_idx_col() on a SqlDaf

```
NotImplementedError: insert_idx_col() is not supported on a SqlDaf yet.
```

gen_bif_report numbers the BMD rows, then shows the first 50 with the numbers.
AuditEngine now numbers only the 50 rows shown, after to_daf().
Ray noted that SqlDaf already keeps an index, daf_rid. It is the row's place in the whole table, not 0 to n in a selection.
Question: should SqlDaf support insert_idx_col(), for example on a TEMP copy or as a computed column? Or should it expose daf_rid?

## 4. Formatting a SqlDaf

Already asked in 2026-10-23_0247_format_a_sqldaf.md, in review as Q279. Listed here so the set is complete.

## 5. to_daf() keeps the TEMP table's name

to_daf() of a selection gives a Daf named like daf_tmp_cvr_bif_v8.
to_md() prints the name in its footer, so a report shows it where a Daf read from CSV shows name=''.
Question: what name should to_daf() give?

## 6. Writes change the saved table

On a SqlDaf, update_by_keylist(), insert_col(), apply_in_place() and cell assignment change the saved table.
On a Daf they change memory only.
gen_bif_report writes sheet0 and a tabulator column into its uncorrupted set. On a SqlDaf that set comes from append() on a selection, so it is a TEMP copy.
That was luck. With no corrupted column, the same code would write into the saved full_bif table.
Questions:
- Should a SqlDaf opened on a saved table be read-only, so a write raises an error until copy() makes a TEMP copy?
- Or should there be a read-only flag the caller sets?

## 7. The first few rows of each group

gen_bif_report's style analyses group by style_num, then by card_code, and show the first 3 ballot_ids of each group.
On a SqlDaf each group is a query. On d05, 440 styles took 1.7 s, against 0.22 s on a Daf, plus 1.6 s for the Daf's to_daf().
Full SF has about 5,000 styles.
valuecounts_for_colname1_groupedby_colname2() gives the counts in one query, 0.16 s on d05, but not the first rows.
Question: is there a way to get the first N rows, or the distinct values of another column, for each group in one query?

## 8. Comparing keys of tables in two database files

Each large table has its own file. gen_bif_report compares the ballot_ids of full_bif with those of cvr_bif: in one, in the other, in both.
With a SqlDaf that means keys() of both, 133,422 and 1,603,908 values, into Python.
from_db() would copy a whole table into one file.
Question: is there a way to select the rows of one table whose keys are, or are not, in a table in another file?

## 9. Python lambdas in select_where()

select_where(lambda row: ...) works on a SqlDaf, but reads every row into Python.
On d05, the original writein_inspection_report line took 1.4 s, against 0.1 s for the same selection by select_by_dict().
AuditEngine keeps such calls during the conversion and changes them for speed afterwards.
Question, for later: is there a pattern you would recommend, such as select_by_dict() with inverse=True or a list of dicts, so one call is fast on both?

## Context

The conversion plan is in audit-engine-dev engineering_notebook/2026-10-22_pipeline_stages_sqldaf_plan.md, sections 13 to 15.
The gen_bif_report change is kept as engineering_notebook/2026-10-23_gen_bif_report_on_sqldaf.patch.
