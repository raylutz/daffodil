# groupby and row sharing

Found on 2026-10-13. Review item 87, still open.

## The design

The README has described row sharing since 2024. The version of 2026-04-02 says rows are
"re-used" by any selection, and only references are copied. Ray confirmed on 2026-10-13 that
groupby was meant to share rows too.

## What the code does

This was checked with `is` on each row.

```python
data_daf = Daf(
    lol=[['a', 1], ['b', 2], ['a', 3]],
    cols=['g', 'x'])
groups_dodaf = data_daf.groupby('g')
groups_dodaf['a'].lol[0] is data_daf.lol[0]   # False
```

| Method          | Shares rows |
| :-------------- | :---------- |
| `groupby()`       | no          |
| `multi_groupby()` | no          |
| `groupby_cols()`  | yes         |
| `group_where()`   | yes         |
| `select_where()`  | yes         |

groupby has always copied. The old loop iterated the Daf, which gives a new dict for each row
by default, and appended that. The code from 9e091b2, 2026-10-02, copies too.

The docstring said the rows are shared. On 2026-10-03, d50afd9 changed the docstring to match
the code. The code should have been changed instead.

## What sharing changes

With shared rows, a cell written in a group is written in the source. Adding a column to a
group does not reach the source, because `insert_col()` builds new rows. Both checked with
groupby_cols:

```python
group_daf = data_daf.groupby_cols(['g'])[('a',)]
group_daf[0, 'x'] = 99
group_daf.insert_col('y', default=0)
data_daf.lol   # [['a', 99], ['b', 2], ['a', 3]]
```

## AuditEngine callers

Read in /home/audit-engine-dev on 2026-10-13. There are 35 Daf calls to groupby. No caller
uses multi_groupby.

All 35 were read, with the functions their groups are passed to. No caller uses a group or
its source after the loop in a way that needs a copy, with one possible exception.

evalvotes.py, in the function that starts at line 178:

- Line 215 groups audit_daf by ballot_id.
- Each group goes to evaluate_votes_on_ballot_daf() or evalvotes_of_ballot_marks_daf().
  Those write cells: num_votes, overvotes, undervotes, gray_eval and the thresholds. The first
  groups the ballot again by contest, at analysis_utils.py:3637, and writes into those groups.
- Line 289 skips the save if `new_marks_daf == audit_daf`.

Daf equality compares rows, columns and keyfield. new_marks_daf is built from MarksSchema,
so its keyfield is ballot_id.

- A local chunk is loaded by DB.load_data with no keyfield. The test is never true, so the
  marks are always saved, with copies or with sharing.
- A chunk from S3 is loaded by pull_daf_from_s3_hunk with keyfield ballot_id, and with
  unflatten=False its values stay strings. With copies, the evaluated ints differ from the
  strings, so the marks are saved. With sharing, the writes would reach audit_daf, and the
  two could be equal, which would skip the save.
- Sharing ends for a group when insert_col() adds a new column, as evaluate_thresholds_daf()
  does for ml_pct when the chunk lacks it. So the S3 case depends on the chunk's columns.
  Only a run of evalvotes with a sharing build would settle it.

Correction, 2026-10-13: the first version of this note said the save would always be skipped.
That was wrong for the local path. Ray pointed it out.

Other notes:

- evalvotes.py:870 groups modified_marks_daf and evaluates the groups the same way. It never
  reads modified_marks_daf again, so sharing is safe there.
- bmd_extract.py:5495 calls evaluate_votes_on_ballot_daf() and discards the result. With
  copies, that call leaves marks_daf unchanged. With sharing, it would change it. The function
  it is in looks unused: convert_ballot_clips_to_marks() is only `pass`.
