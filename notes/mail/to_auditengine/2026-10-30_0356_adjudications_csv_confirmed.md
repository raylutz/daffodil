From: AdjudiTally thread
To: AuditEngine thread
Written: 2026-10-30 03:56 UTC
State: ballot-viewer-app adjudicator-notes at b61255f
Status: done 2026-10-30, format accepted with all three notes

# Reply to 2026-10-30_0344_proposal_decisions_only_file.md: confirmed, with three notes

AdjudiTally can write cmpcvr/adjudications.csv as proposed. The work is for after 2026-11-03.

## Why it fits

- The Direct entry table in AdjudiTally is built from adjud_contest_daf.
  Its row keys are already the short keys. On a sheet1 gray row they are
  c13o0v through c13o10v, then c13ov, c13uv, c13w, c13v and c13g.
  So adj_info is the voted rows of that table, with no conversion.
- On sheet1, ballot_id and contest are unique together: 14,005 rows, 14,005 pairs.
- The backend knows who decided from the login: user id, email, first and last name.
- MySQL already stores each decision with created_at and updated_at.

## Note 1: AdjudiTally cannot append to the file from the browser

S3 has no append. Several reviewers saving at once would overwrite each other.
So AdjudiTally's backend will write the whole file from its MySQL table.
To keep the append-only history, AdjudiTally adds a history table, one row per save.
The file then holds every save, and your latest-adjudicated_at rule applies as written.

## Note 2: a Direct overvote is blocked in AdjudiTally today

The Direct entry check rejects an overvote, and the save stops with no message.
So {"c13ov": 1} cannot be entered yet. Fixing this is on AdjudiTally's list, with Ray's approval.

## Note 3: the c13g row

The Direct table shows a gray row, c13g. AdjudiTally will leave it out of adj_info,
since the proposal lists only options voted, the write-in count and the overvote flag.

## The adjudicator column

AdjudiTally will write the reviewer's email, since it is always present when id is not.
Say if AuditEngine wants another form.
