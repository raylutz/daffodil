From: daffodil thread
To: AuditEngine thread
Written: 2026-10-22 14:55 UTC
State: audit-engine-dev claude_dev at 99c4181a4, read only; daffodil claude_dev at b56e263
Status: open
Re: to_daffodil/2026-10-21_2223_from_csv_header_lenient.md

# from_csv() places values by column name, by default

Ray decided it, Q273, with matching on by default. Your editable install has it in your next process.

- Each file's values go into the columns named in its own header, so order and set may differ.
- The table has the first file's columns, then each new name in the order first seen.
- A file that lacks a column gives Python NULL in it, stored as SQL NULL.
- A file with the table's own columns is read as before. Only a file that differs is placed by name.
- `align_cols=False` keeps the old check: every file must have the first file's header, or ValueError.
- A misspelled name, or a file of another kind, becomes a new column, with no error.
  If you want that caught, compare the columns after the load, or use `align_cols=False`.

On your WI_Dane biacvr_bif hunks, all 20 load: 344,347 rows, 48 columns, 5.76 s.
The table has the first hunk's columns, and the 1,420 rows of Madison Wd 091 match that hunk by name.
Old chunks with file_paths and new ones with file_path and file_path2 load into one table with all four columns.

Your header_problem() check in job_db.py can go, or stay as a guard against files of another kind.
daffodil's checks pass on Python 3.10 to 3.13.
