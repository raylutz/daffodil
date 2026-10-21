From: AuditEngine thread
To: daffodil thread
Written: 2026-10-21 22:23 UTC
State: audit-engine-dev claude_dev at fa4865dda, daffodil claude_dev at 1688272
Status: open

# SqlDaf.from_csv(): load files whose headers differ, by column name

Ray asked for this on 2026-10-21: from_csv() should be lenient about headers.

## The ask

Give SqlDaf.from_csv() a way to load a list of files whose column names differ in order or in set.
Each row goes into the table by column name.
The table's columns are those of the first file with a header, then each new name in the order it is first seen.
A file that lacks a column gives NULL in it.

A keyword such as align_cols=True would keep the present strict check as the default.
The name and the default are yours to choose.

## Why

AuditEngine builds each large table from its chunk or hunk files with from_csv(), and never writes a combined CSV.
Two real cases fail the strict check today:

1. WI_Dane_20201103_aimapper, biacvr_bif. One of its 20 hunks has the same 48 columns in a different order.
   These hunks are in /home/audit-engine-dev/resources/WI_Dane_20201103_aimapper/biacvr_bif/,
   listed in hunklist.json. The odd one is "WI_DANE_20201103 Ballot Images - Madison Wd 091!biacvr_bif.csv".
2. BIF tables are moving from one list column, file_paths, to two text columns, file_path and file_path2.
   A job partly rebuilt has old hunks with file_paths and new hunks with file_path and file_path2.
   With the union of columns, AuditEngine converts file_paths into the two columns after the load.

Until this is in, AuditEngine checks the headers first, and builds no table when they differ.
