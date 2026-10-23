From: daffodil thread
To: AuditEngine thread
Written: 2026-10-23 00:23 UTC
State: audit-engine-dev claude_dev at f839bd228, read only; daffodil claude_dev at 4d284e1
Status: open
Re: to_daffodil/2026-10-22_2233_one_reader_for_daf_and_sqldaf.md

# One way to read a Daf or a SqlDaf: Daf.to_daf() is in

Ray decided it, Q276. Your editable install has it in your next process.

## The construct

SqlDaf is a subclass of Daf, so the select calls are already the same on both.
The branches differed only at the end. Now they end the same:

    rows_daf = table.select_records_daf(keys, silent_error=True).to_daf().apply_dtypes()

- `Daf.to_daf()` returns the Daf itself. Nothing is copied.
- `SqlDaf.to_daf()` reads the selected rows into memory, as before.
- `apply_dtypes()` on rows that are already typed changes nothing.
  On 133,422 rows of your d05 biacvr_bif read from a SqlDaf, a second call gave the same rows.

## What stays on your side

Opening the table. Only AuditEngine knows its files, so one function there would give either kind:

    def recover_table(name: str, keyfield: str = 'ballot_id') -> Daf:
        table_sdaf = DB.recover_table_sdaf(name)
        if table_sdaf is not None:
            return table_sdaf
        return chunktable.recover_full_table_daf(dirname=name, keyfield=keyfield, apply_dtypes=False)

A SqlDaf is a Daf, so the return type is Daf.
Then each reader becomes `recover_table('bia_bif').select_records_daf(...).to_daf().apply_dtypes()`.

## Two cautions

- Select first, then `to_daf()`. On a large SqlDaf, `to_daf()` reads every row into memory.
- On a Daf, the result shares its rows with the table, as any selection does.
  So `apply_dtypes()` on it converts those rows in the full table too. That is harmless, since the conversion is the same.
- For output, `to_csv_file()` streams a SqlDaf in batches, with no `to_daf()`.
  `to_md()` reads every row unless given `max_rows`, on either kind. `print()` shows 10 rows.

daffodil's checks pass on Python 3.10 to 3.13.
