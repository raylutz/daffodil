From: AuditEngine thread
To: AdjudiTally thread
Written: 2026-10-30 03:44 UTC
State: audit-engine-dev claude_dev at 6a9cf1f3e
Status: done 2026-10-30, confirmed with notes in 2026-10-30_0356_adjudications_csv_confirmed.md

# Proposal: AdjudiTally writes only its decisions

Ray chose this design on 2026-10-30. It answers item 1 of 2026-10-30_0236_adjudicator_json_import_issues.md.
The aim is the least distance between what AuditEngine holds and what AdjudiTally sends back.
AdjudiTally reads cmpcvr/contest_variants.csv, as it does today.
It sends back only what the adjudicator decided.
AuditEngine looks up the audit and CVR results in its own contest_variants table.
Nothing AuditEngine owns is copied out and read back.

The ask: please confirm this format, or say what AdjudiTally needs changed.
It is for after 2026-11-03.

## The file

Name: cmpcvr/adjudications.csv, in the job's folder on S3, next to adjudicator.json.
It is a CSV with a header row, as Daf writes it.
js/daffodil-csv.js in /home/daffodil reads this format, dict cells included.

One row per decision. The file is append-only.
For the same ballot_id and contest, the row with the latest adjudicated_at counts.
Earlier rows stay as the record of what was done.

| Column | Value |
| :-- | :-- |
| ballot_id | as in contest_variants.csv |
| contest | as in contest_variants.csv |
| choice | AuditEngine, VotingSystem or Direct |
| adj_info | Direct only, else empty. See below |
| adj_annotation | free text, may be empty |
| tags | a JSON list of strings, may be empty |
| adjudicator | who decided |
| adjudicated_at | UTC time, ISO 8601 |

## adj_info for a Direct choice

It uses the same short keys as audit_info, for that contest only.
A JSON object is fine, since AuditEngine reads it as a Python literal too.
For contest number 13, list each option voted and the write-in count:

    {"c13o0v": 1, "c13o2v": 1, "c13w": 1}

For an overvoted contest, give the overvote flag and no options:

    {"c13ov": 1}

An empty object means no votes in the contest.
AuditEngine derives the ballot count, total votes and undervotes from vote_for.

## What each choice means in AuditEngine

- AuditEngine: keep the audit's reading.
- VotingSystem: take the adjudicated CVR record if one exists, else the original record.
- Direct: take adj_info.

AuditEngine keeps reading the old adjudicator.json when adjudications.csv is missing.
