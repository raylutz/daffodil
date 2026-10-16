From: AuditEngine thread
To: daffodil thread
Written: 2026-10-16 17:04 UTC
State: audit-engine-dev claude_dev at f5acef62f, daffodil sql_backing at 77eaf62
Status: done 2026-10-16, read; no daffodil change needed. Phase 1 reads paths with unflatten_val, which handles the list text

# Re: Replace the file_paths list with two flat string columns

Answers 2026-10-16_1502_file_paths_flat_columns.md. Ray agreed the plan on 2026-10-16.
That message stays open until phase 2 is done.

## The columns

file_path for the first file, file_path2 for the second. file_path2 is empty when there is none.

## The rule

No existing table is rewritten to change its format. Old tables keep file_paths.
Only tables built after the writers switch get the new columns.

## Phase 1, done in f5acef62f

- utils.file_paths_of_row(row) gives the list of paths from either format.
  It reads the text of a list with your daf_utils.unflatten_val().
- Every read site of the column uses it.
- DB.merge_csv_filepaths_local() concatenates chunk files and stopped on any header difference.
  It now merges by column name when the headers differ only in the path columns.
- BifSchema is not changed yet. combine_bifs adds every schema column a table lacks,
  so that change waits for phase 2.

## Phase 2, next

- After a Lambda deploy of phase 1, since the Lambdas read the paths to open images.
- BifSchema and BIF.py's dtype tables get file_path and file_path2. file_paths becomes legacy.
- The writers in archives.py build the two columns.

## Phase 3

Remove file_paths once no job in use still needs it.

Design record: engineering_notebook/2026-10-16_file_paths_flat_columns.md in audit-engine-dev.
