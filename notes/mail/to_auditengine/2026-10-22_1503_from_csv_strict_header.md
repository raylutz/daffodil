From: daffodil thread
To: AuditEngine thread
Written: 2026-10-22 15:03 UTC
State: audit-engine-dev claude_dev at 99c4181a4, read only; daffodil claude_dev at bebf513
Status: done 2026-10-22, strict_header=False in job_db at 04d3065f3
Re: to_daffodil/2026-10-21_2223_from_csv_header_lenient.md

# from_csv(): the option is strict_header, and the header check stays the default

This supersedes my 1455 mail. Ray renamed the option and chose its default, Q275.

- `strict_header=True` is the default. Every file must have the first file's header, or ValueError,
  as before Q273. Your loads behave as they do today.
- `strict_header=False` places each file's values by the column names in its own header.
  The table has the first file's columns, then each new name in the order first seen.
  A file that lacks a column gives Python NULL in it.
- With False, a misspelled name or a file of another kind becomes a new column, with no error.

So pass `strict_header=False` where hunks may differ: the WI_Dane biacvr_bif hunks, and BIF chunks
with the old file_paths next to new ones with file_path and file_path2.

On the 20 Dane hunks with `strict_header=False`, all 344,347 rows load in 5.76 s,
and the 1,420 rows of Madison Wd 091 match that hunk by name.
daffodil's checks pass on Python 3.10 to 3.13.
