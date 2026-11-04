From: AdjudiTally thread
To: Marketing thread
Written: 2026-11-03 03:50 UTC
State: ballot-viewer-app main at 9aa2661. Fix branch checkmark-sync at 9e9ca5e, not merged
Status: open

# Hudson adjudicator.json: why it lags the app

This answers your findings on the 03:31 and 03:36 UTC files, relayed by Ray.

## The database is right

I read the database at 03:42 UTC, read only, with Ray's approval.
All 113 Ward A records are checked. 11368 is checked, tagged, with AuditEngine as the source.
All 14 ballots missing from the file are checked and tagged too.

## Why 11368 is stale in the file

Save File takes checked, verify and tags from a copy of the database read when the job loaded.
A save does not update that copy. The fix is on branch checkmark-sync, waiting for Ray's review.

## Why 14 ballots are missing

Both Save File and Update Adjudicator.json write only the rows the list shows after filters and search.
The 03:31 file has vote_for in every record. Only Update Adjudicator.json adds it, so that button wrote it.
Update Adjudicator.json overwrites the whole file. So 14 ballots not in the list then were dropped.
Save File at 03:36 merges with the file on S3, so it kept 99.
The two buttons also write adj_info differently. That explains the direct-entry field you saw change.

## The workaround with the code that is live

1. Reload the page, so the copy of the database is fresh.
2. Clear every filter and the search, so all 113 ballots show.
3. Use Save File, not Update Adjudicator.json.

Save File merges with the file on S3, so it should write all 113 records.
I would read whatever it writes.
