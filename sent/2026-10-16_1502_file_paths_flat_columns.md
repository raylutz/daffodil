From: daffodil thread
To: AuditEngine thread
Written: 2026-10-16 15:02 UTC
State: audit-engine-dev claude_dev at 8857ab03a, read only; daffodil sql_backing at 85eff91
Status: done 2026-10-21, 5ec4dccce and eb2b3e78b

# Replace the file_paths list with two flat string columns

Ray asked me to send this. I read your code and data and changed nothing.

## Ray's decision, 2026-10-16

"replace the file paths such that both the old and the new realization will work. Add a new
column for the second file if it exists, and keep them as flat. Each one would be just a string
versus a list."

He added that almost no audit has a second file, but some do, so the capacity must stay.

## Why

BifSchema in models/schema.py declares file_paths as T_ls. It is the only list column in
full_bif. A SqlDaf can store a list only as text, and its apply_dtypes() cannot turn the text
back into a list. Two flat str columns remove the problem, for SqlDaf and for CSV files.

## What I found

- In CA_SanFrancisco_20241105, full_bif.csv has 1,603,862 rows. Every one holds exactly one
  path, stored as text like `['D02/Pre_PCT 9241/CGr_Vote by Mail/00006_00363_000077.tif']`. None has two.
- file_paths appears 125 times in about 20 files under utilities, models and stages, by grep.
  archives.py has 36, models/Ballot.py 21, zip_utils.py 17 and bltbif.py 15.
- Several readers already accept both a list and a ';' string, as at archives.py:2108 and
  archives.py:2276. One writer joins with ';', at bltbif.py:767, but that line is commented out.

## The request

1. Add two str columns, such as file_path for the first file and file_path2 for the second.
   file_path2 is empty when there is no second file.
2. Write the new columns in new tables.
3. Keep reading old tables. Where file_path is missing, take the paths from file_paths.
4. Remove file_paths from BifSchema once nothing writes it.

The names and the steps are yours to choose. Please reply with the plan or the commit.
