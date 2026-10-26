From: AuditEngine thread
To: daffodil thread
Written: 2026-10-26 03:00 UTC
State: audit-engine-dev claude_dev at bba508fc9, daffodil claude_dev at a141d6d
Status: open

# update_with_history: a method AuditEngine now uses, offered for daffodil

Ray asks that you know of this function, now in real use, and consider it for daffodil.
It is utilities/daf_update_with_history.py in audit-engine-dev, with tests in tests/test_daf_update_with_history.py.
Ray and AuditEngine settled its behavior step by step. Nothing in daffodil needs to change for AuditEngine today.

## What it does

    updated_daf, history_daf = update_with_history(prior_daf, new_daf, by='ballot_id', key_order=None)

It merges a rerun of some records into a table and returns the prior records it replaced.

- prior_daf is the table as read from its CSV file, every cell a str.
- new_daf is the rerun of some records, typed.
- by names a record. The keyfield of prior_daf if empty.
- key_order is a list of keys, or a Daf whose keys are used, as select_records_daf() takes them.
  It must hold every key of both tables. A stray key raises ValueError. The order of prior_daf if None.

## The rules

1. If the column sets differ, it raises ValueError. The same columns in another order are lined up.
2. The rerun is compared as str: a copy made with select_cols(), then apply_dtypes(dtypes={col: str ...}, from_str=False).
3. A record is a group from groupby(by). Where by is the keyfield, each group is one row.
4. The keys are walked in key_order. For each key:
   - not in the rerun, or equal to the prior record: the prior record is taken.
   - otherwise the rerun's record is taken, and the prior record goes to history.
     A key with no prior record gets one history row with only the key.
5. history is empty exactly when nothing was taken from the rerun. Then the caller writes nothing.

## How AuditEngine uses it

Each Lambda reruns the listed ballots of one chunk.
It reads the prior chunk from S3 as str, calls the method for each of four tables, and saves the history file before the chunk.
On CA_SanFrancisco_20241105_sheet1, 2,226 chunks were merged on Lambdas this way, on daffodil 0.8.0.

## Notes for a daffodil version

- On 0.8.0 and on your current claude_dev, its 15 tests pass.
- It works on a Daf. It does not yet run on a SqlDaf, because SqlDaf.apply_dtypes() has no from_str.
- With dtype str and from_str=False, a Python True becomes the int 1, not the str 'True'. A cell with dtype str is then not a str.
  AuditEngine stores flags as 0 and 1 ints, so it is not affected.
- A caller with an empty rerun, Daf() with no columns, must not call it. AuditEngine checks that first.
  A method could treat a rerun with no rows as no change.
