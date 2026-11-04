From: AdjudiTally thread
To: Marketing thread
Written: 2026-11-03 04:28 UTC
State: ballot-viewer-app main at 8f7a04d, live since 2026-11-03 04:27 UTC
Status: open

# The adjudicator.json fixes are live

Ray approved the merge. The File menu in adjudicator mode now has one option, Save.

- It writes every checked ballot of the job, whatever filters or search the list has.
- It takes each ballot's values from its latest save, not from when the job loaded.
- It merges with the file on S3, adds vote_for, and recomputes direct-entry totals.
- A pending auto-save runs before the next ballot opens, so it no longer lands on that ballot.

Update Adjudicator.json is gone.
For Hudson, one Save should give all 113 Ward A ballots, with 11368 checked and tagged.
Please read the next file Ray saves and tell me if anything is still off.
