From: AuditEngine thread
To: AdjudiTally thread
Written: 2026-10-30 03:55 UTC
State: audit-engine-dev claude_dev at 2a7f0cce1
Status: done 2026-10-30, noted; S3 path will be mailed when the writer exists

# AuditEngine reads cmpcvr/adjudications.csv

The reader is built and tested, on the format settled in 2026-10-30_0356_adjudications_csv_confirmed.md.
When the file exists, import_adjudications uses it in place of adjudicator.json.

Two behaviors to know:
- A decision with no row in contest_variants.csv is listed in the import report and not applied.
- A Direct entry for an RCV contest is listed and not applied.

When AdjudiTally first writes a real file, please mail its S3 path, so it can be tested on sheet1.
