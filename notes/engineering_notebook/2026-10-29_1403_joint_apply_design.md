# joint_apply(): one function over several tables matched by key

Started 2026-10-29. Ray proposed it as a general form of `join()` with its translator, of
`join_with_residue()`, Q314, and of the comparison stages of find_variants in AuditEngine.
This entry is the design, before any code. Q315.

## The idea, in Ray's words, shortened

It takes a keyed list of tables, or names of tables in the open database, and a function. The
tables are associated, as by their keyfields. It goes over them and makes a list of output tables.

## Research

- `join()` on a Daf loops over this table's rows, finds each key in the other, and builds the row
  with `join_records()` and a translator Daf of column names. On a SqlDaf it is one SQL join, set-based.
- `join_with_residue()` is `join(how='inner')` plus two inverse `select_records_daf()` calls.
- AuditEngine `utilities/sdaf_variants.py`, steps 3 to 11, splits the ballots of the audit and the
  CVR into residue, variants and nonvariants for each side, in nine set operations. The decision per
  ballot is one rule: residue if one side lacks it, a variant if flagged on either side, in the
  modified CVR, or its votes differ, else a nonvariant. Step 5 exists only to make the two sides agree.
- Steps 13 and 14 do the same at the contest level, keyed by contest_key, with a combined row for
  each contest that disagrees. Step 12 is an `apply()` from one row to many. Step 15 is
  `groupby_aggregate()`. Steps 16 and 17 are selections.

## Prototype timings, scratch code

On three SQLite tables, 200,000 keys: the union of the keys, a left join of each table by key,
read in batches, a Python function per key, and the output keys written: 1.08 s. The same two
counts by set-based SQL: 0.17 s. So about 5 us a key in Python. At 1,600,000 keys, about 9 s.
find_variants on pilot d05 takes 31 s in all today.

## Design

    outputs_klist = Daf.joint_apply(tables_klist, func)

- tables: a KeyedList of name to Daf or SqlDaf. A str in place of a table is the SqlDaf of that
  name in the open file.
- The keys are the union of the keys of all the tables, by each table's keyfield. Order: the first
  table's order, then the keys new in each later table, in its order.
- func(key, rows_klist) is called once for each key. rows_klist maps each table's name to its row as
  a dict, or None where that table lacks the key.
- func returns a dict of output name to a row, a list of rows, or None. None means no output.
- The outputs are a KeyedList of name to table, in the order first written. Each takes its columns
  from its first row, as `apply()` does.
- Outputs are SqlDaf TEMP tables on the first SqlDaf's connection if any input is a SqlDaf,
  otherwise Daf objects. Tables in other files are read in place, as Q312 brings them in.
- `cols=` limits the columns read from each table.

## How the cases map onto it

- join_with_residue: both rows give joined, one row gives that side's residue.
- join with a translator: the function builds the joined row and names its columns itself. The
  translator can stay as a helper that gives the renaming.
- find_variants steps 3 to 11: one function over audit, cvr and modi gives six outputs.
- Steps 13 and 14: the same over the contest tables, with the combined row built in the function.

## Recommendation

- Add `joint_apply()` as the general form, on Daf and SqlDaf, with this one behavior.
- Keep `join()` and `join_with_residue()`. They are set-based in SQL, about 5 times faster per key.
- AuditEngine could replace steps 3 to 11 with one call, and decide by measuring on d05.

## Open

- Q315: build it as above?
- Several rows for one key in a table, as for one-to-many matching, are not covered.
  A SqlDaf keeps keys unique, so a row per key is the rule there.
