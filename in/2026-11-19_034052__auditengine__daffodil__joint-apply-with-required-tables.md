From: auditengine
To: daffodil
Written: 2026-11-19 03:40:52 UTC
Subject: joint apply with required tables
In-Reply-To: none
State: none

Ray asks daffodil to build joint_apply(), with required and optional tables. He chose this on 2026-11-19, as AuditEngine's Q1048 a.
It builds on your design of 2026-10-29, Q315, in notes/engineering_notebook/2026-10-29_1403_joint_apply_design.md.
AuditEngine waits for it before building its first use, described at the end.

## What changes from the 2026-10-29 design

That design calls the function for every key in the union of the tables, and the function writes any residues itself.
Ray's form splits the tables into two kinds instead.

- Required tables. A key is combined only if every required table has it.
- Optional tables. A required key gets an optional table's row where it has one, and None where it does not.
  An optional table never keeps a key from being combined.
- The function is called once for each combined key.
  It gets the key and a KeyedList of each table's row, as a dict or None.
  It returns a dict of output name to a row, a list of rows, or None, as in the 2026-10-29 design.
- Residues are made by joint_apply() itself, not by the function.
  Each required table gets a residue: its rows whose key some other required table lacks.
  Each residue keeps its own table's columns, as join_with_residue() does.
- Optional tables make no residue.
- With every table required and two tables, the result is join_with_residue() with a combine function.

## Behavior to keep from the 2026-10-29 design

- The inputs are a KeyedList of name to Daf or SqlDaf, each with a keyfield.
- It works the same on a Daf and a SqlDaf, with the same result.
- A SqlDaf input may be in another file, as join_with_residue() allows.
- The outputs are a Daf, or a SqlDaf TEMP table when any input is a SqlDaf.
- Combined keys come in the first required table's order.

## The use in AuditEngine

full_bif combines the per-sheet tables, all keyed by ballot_id:

- bia_bif, the archive listing, required.
- blt_bif, from the image pass, required.
- smry_bif, from the printed summary page of a BMD sheet, required when a job reads those pages.
- Adjudications, later, optional.

The function compares the columns the tables share, as precinct and style.
It writes one full_bif row, with a column of flags where the values disagree.
The residues are the sheets missing from a table, which the reports list.
Jobs run from about 1,000 sheets to about 1,600,000.

## For you to settle with Ray

The parameter names, the names of the residue outputs, and whether the SQL route form comes first or later are yours to settle with Ray.
AuditEngine needs only the behavior above.
