# One interface for Daf and SqlDaf: the loose ends

Started 2026-10-23. AuditEngine converts a stage by handing it a SqlDaf where it loaded a Daf, and
keeping the stage's calls. gen_bif_report on pilot d05 found nine places where a call does not work
the same on both. Ray asked for a proposal for each, with its tradeoffs, and decides each one.
Mail: to_daffodil/2026-10-23_0402_unified_interface_loose_ends.md. Questions Q279 to Q287.

## Facts checked, 2026-10-23

- SqlDaf's `str()` is its one-line `repr()`. An f-string raises NotImplementedError, as does
  `md_daf_table_snippet()`. `md_max_rows` is set in `Daf.__init__()`, which a SqlDaf never runs.
- `to_md(max_rows=10, max_cols=10, include_summary=True)` on a SqlDaf gives the same text as the
  Daf of the same rows. On cvr_votes!final, 1,603,908 rows, it took 0.64 s. On 50 rows, 0.001 s.
- A Daf selection has the name ''. A SqlDaf selection's `to_daf()` has its TEMP name, such as daf_tmp_t_v1.
- `Daf.keys()` takes `silent_error` and `astype`, 'list' or 'view', a view of the key index.
  `SqlDaf.keys()` takes neither.
- `Daf.concat()` reads `other_instance.lol`, which a SqlDaf does not have.
- `Daf.insert_idx_col()` puts the column first by default. `SqlDaf.insert_col()` adds only at the end.
- A connection opened with mode=ro makes TEMP tables and TEMP indexes. SQLite refuses UPDATE, ALTER
  and CREATE INDEX on the file: "attempt to write a readonly database". It also refuses a TEMP index
  on a table of the file: "cannot create a TEMP index on non-TEMP table".
  So on a read-only file, the key and group indexes SqlDaf makes when first needed cannot be made.
- On d05 full_bif, the first 3 ballot_ids of each of 440 styles came in one window query,
  ROW_NUMBER() OVER (PARTITION BY style_num ORDER BY daf_rid), in 0.19 s. AuditEngine measured
  1.7 s for one query per group.
- full_bif against cvr_bif on d05, the second attached read-only, with IN and NOT IN subqueries:
  not in cvr_bif 0 rows in 1.72 s, cvr_bif not in full_bif 1,470,486 in 1.33 s, in both 133,422 in 1.70 s.
  Both key lists read into Python and compared as sets: all three in 1.70 s, holding 1.6M keys.
  A first try with NOT EXISTS and no index on cvr_bif.ballot_id ran past 10 minutes and was stopped.
- Neither saved file has an index on ballot_id. Each has only the indexes of daf_tables and table_sources.

## The proposals

1. keys(astype). `SqlDaf.keys(silent_error=True, astype='list')`. 'list' as now. 'view' gives the
   KeysView of a dict built from the keys, in row order, a repeated key once, as a Daf's view of its
   key index. On a Daf the view is free. On a SqlDaf it builds a dict of every key.
2. A Daf appending a SqlDaf. `Daf.concat()` calls `other_instance.to_daf()` first, which is free on a Daf.
   The result is a Daf in memory. A large SqlDaf then fills memory. The other choice is a TypeError.
3. insert_idx_col(). On a SqlDaf it makes a TEMP copy with the column at `icol`, numbered from
   `startat` in row order, and the SqlDaf then refers to it, as a sort does. The saved table is not
   changed. It copies the rows. The other choice is to show daf_rid, the place in the whole table.
4. Formatting, Q279. `str()` and an f-string give `md_daf_table_snippet()`, 10 rows and 10 columns.
5. The name of `to_daf()`. A generated TEMP name gives '', as a Daf selection has. Q281.
6. Writes to a saved table. `read_only=True` opens the file's connection with mode=ro, so SQLite
   refuses any write to the file. Selections, copies, sorts and joins still work, in TEMP.
   The connection is shared by every SqlDaf on the file, so the mode is the file's.
   The key and group indexes must then be in the file already, made when the table is built.
   The other choice, every saved table read-only by default, would break code that updates tables.
7. The first rows of each group. A method on both that gives the first n rows of each group, in
   one window query on a SqlDaf, as a KeyedList of Daf in memory. The distinct values of another
   column per group are the keys of `valuecounts_for_colname1_groupedby_colname2()` already.
8. Keys of a table in another file. `select_records_daf(other)` with `other` a table in another file
   attaches that file read-only for the one statement, resolves the selection, and detaches.
   No copy, and no keys in memory. The other choice is no change: Python sets are as fast on d05,
   but hold every key, about 10 million on a county.
9. Lambdas. No change. `select_by_dict()` is fast on both, with a list of dicts for OR and
   `inverse=True` for NOT. `select_where_sql()` is SqlDaf only.

## A selection as the keys, 2026-10-23

Mail: to_daffodil/2026-10-23_1551_selection_as_keys_is_quadratic.md. Q288.
`select_records_daf(keys)` with `keys` a resolved selection ran past 115 s on d05, both with and
without `inverse`. `_key_source_sql()` gives the selection's own query. That joins its TEMP table of
references to the parent by ref, and SQLite finds each key's row by the key index, then scans the
references for it, once per row: 133,422 by 1,603,908.

Measured on copies of the d05 full_bif and cvr_bif, keys from cvr_bif after remove_dups():

    1. an index on ref of the references table      not done in 75 s
    2. NOT IN (SELECT daf_key ...), inverse only     1.96 s, and the plain select still crawled
    3. the keys copied to an indexed TEMP table      1.52 s to copy and index, then 0.18 s

With 3 patched in from scratch, AuditEngine's three calls took 1.78 s, 1.21 s and 1.66 s, with
0, 1,470,486 and 133,422 rows, the same counts as Python sets. A list or a Daf of keys already takes
that path. A parent table as the keys keeps its own key index, with no copy.
Scripts: scratch quad_probe.py and quad_fix.py, not kept.

### Q288, settled 2026-10-23

Ray asked whether the selection's table could be indexed. It can. An index on ref had measured
slow, and was left out of the first options without a look at the plan. EXPLAIN QUERY PLAN showed
SQLite never used it: ref had no declared type, and was compared with the INTEGER daf_rid.
With ref declared INTEGER and indexed, the plan ends SEARCH s USING COVERING INDEX, and the
inverse select took 0.31 s. Ray: "A, go ahead (crazy that you did not include this as a possible
solution, why not)??" Lesson: when an approach that ought to work measures slow, read the plan first.

Built, with the library code on the d05 copies:

    biabif.select_records_daf(uniq, inverse)        0 rows        0.92 s, with the index build
    uniq.select_records_daf(biabif, inverse)        1,470,486     1.30 s
    biabif.select_records_daf(uniq)                 133,422       0.83 s
    a join not read yet, of 1,000 rows, as keys     132,422       0.13 s

- `_refs_table_sql()` declares ref with the type of the parent's ref column. A table keyed by a
  TEXT keyfield, with no daf_rid, gets ref TEXT.
- `_ensure_ref_index()` runs in `_key_source_sql()`, so only a selection used as keys pays for it.
- Option B, copying the keys to an indexed TEMP table, was not needed. A join not read yet was fast as keys.

## Q279, settled 2026-10-23

Ray: yes, and "the sql table should not do a lot of conversion, but rather look directly at the sql table."
He pointed to AuditEngine's sql_utils.sqltable_to_md_summary(), which counts the rows and reads the
first and last with LIMIT and OFFSET. `SqlDaf.daf_to_lol_summary()` already does the same: a COUNT,
the first rows with LIMIT, and the last with ORDER BY DESC LIMIT, which does not step over the middle.
So only the rows shown come into Python.

- `md_daf_table_snippet()`, `__str__()` and `__format__()` are SqlDaf methods that call Daf's code.
- `md_max_rows` and `md_max_cols` are class settings of SqlDaf, 10 each, since `Daf.__init__()` never runs.
- `repr()` keeps its one line. That is Q280.
- The test that str() equals repr() was changed: it held the old behavior.

## Q281, settled 2026-10-23

Ray: "Q281 stay with original." A SqlDaf selection keeps its TEMP name, in its summary line and in `to_daf()`.
Instead, the docs recommend `to_md()` in code, rather than an f-string of a table, because
"the summary line is suppressed by default" and "the number of rows can be explicitly set, and it
defaults to all, so my_daf[0:50].to_md() will work the way most people want it to work."

Checked: `daf[0:50].to_md()` gives 50 rows, every column, no summary line, the same text on a Daf
and a SqlDaf. An f-string of the same selection gives 10 rows and the summary line.
Added to the `to_md()` docstring, the Guide's patterns for both classes, and the cheatsheet.
This also reads as keeping the `to_md()` default of every row on a SqlDaf, Q278.

## Q280, settled 2026-10-23

Ray: repr "should NOT give the one line internal reference but rather the referenced object, so it
matches mdaf." `repr()` is a newline and then `md_daf_table_snippet()`, as `Daf.__repr__()` is.
Two tests that held the one line were changed. `is_child` and `is_temp` tell the kind of SqlDaf.

## Q282, settled 2026-10-23

Ray: yes, "but also add in the docs that although this exists for compatibility, that there are
better ways to handle this, and give the examples." `SqlDaf.keys(silent_error=True, astype='list')`.
'view' is `dict.fromkeys(keys).keys()`. The docstring lists the better ways, with a doctest, and the
Guide's patterns for both classes have a bullet on it. A parity test checks list, view, a bad
astype and no keyfield against a Daf.

## Q283, settled 2026-10-23

Ray chose A, with no size guard. `Daf.concat()` calls `other_instance.to_daf()` first: free on a Daf,
a read into memory on a SqlDaf. Rows read from a SqlDaf are adopted as they are, since they are new,
so a large group is not copied twice. A Daf's rows are deep-copied as before. `append()` of a Daf goes
through `concat()`, so `Daf().append(sdaf)` works. AuditEngine moved off the pattern anyway.

## Q289, settled 2026-10-23

Mail: to_daffodil/2026-10-23_2016_find_variants_fails_in_refs_table_sql.md. A regression from Q288.
`self.parent_sdaf or self` tested the parent's truth, and an empty SqlDaf is false. On d05 the votes
residue is empty, so `_refs_table_sql()` took the selection for its table and called `_ref_col()` on
it. AuditEngine read it as a selection whose parent is a selection. A hook on `_new_selection()`
showed no such selection was ever made. `_root()` replaces the pattern in all 16 places.

Ray asked whether this covers deep nesting, and whether it hangs. A selection's parent is always its
table: `_new_selection()` takes the root of the selection it is made from. Four nested selections, on
an empty and a full table, all had the table as parent. A chain of selections not yet read nests its
queries: 413 levels work with SQLite 3.53.1, and the next raises OperationalError "Recursion limit",
the same for `select_where_sql()`, `[:]` and `select_by_dict(inverse=True)`. Read each as made, a
chain of 2,000 took 1.7 s. Ray: "document the recursion limit error." It is in the class docstring and
the Guide's Selections section.

With the fix, the port ran all 18 steps on a copy of AuditEngine's current d05 cmpcvr.db.
cvr_residue.csv had no rows, where AuditEngine's file of 2026-10-21 has 7: its inputs now have
34,534 rows in both tables. Scratch check copy: rp/sdaf_variants_check.py, not kept.

## Q284, settled 2026-10-23

Ray chose A. `SqlDaf.insert_idx_col()` makes a TEMP copy with the column at `icol`, numbered by
ROW_NUMBER() OVER (ORDER BY daf_ord) - 1 + startat, and the SqlDaf refers to it, as after a sort.
It follows Daf's rules, checked by running them: a negative `icol`, or one at or past the number of
columns, puts the column at the end. An existing name has its values replaced. An empty name does
nothing. A parametrized test compares eight cases with a Daf.
