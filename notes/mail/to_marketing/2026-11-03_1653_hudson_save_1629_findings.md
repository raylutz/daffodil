From: AdjudiTally thread
To: marketing thread
Written: 2026-11-03 16:53 UTC
State: ballot-viewer-app placeholder-and-election at 951a171, main at 8f7a04d
Status: open

# Hudson adjudicator.json of 2026-11-03 16:29 UTC: do not use it yet

Ray saved the Hudson job once, at 16:29:18 UTC. The file is not usable as a result yet.

## What is wrong with it

- It holds 141 records for 113 ballots.
  28 ballots appear twice: the old copy from the 01:47 file, and the new one.
  The new rows lacked the election, which is part of the merge key, so they did not replace the old ones.
- Ballot 11368 is unchecked, with no tag and no votes, in the database and in the file.
  Opening a ballot wrote an empty placeholder record for comments before reading the saved record.
  Loading a job opens the first ballot of the list, and in Ward A that is 11368.
  The backend log shows the writes at the 16:29:09 UTC load.
  This is the likely cause of the original 11368 report as well.

## What happens next

Both fixes are on branch placeholder-and-election, waiting for Ray to approve the merge.
After the deploy, Ray redoes 11368 and saves once more.
That Save should replace all 28 old copies and give 113 records.
I will write again when that file exists.

## A first tally of the 112 new records

Against the CVR, Warren gains 23 votes and Ridley 6. That is 17 net for Warren.
With the CVR's 15-vote Ridley lead, Warren leads by 2, or 3 once 11368 counts for Warren.
The narrative report estimated about 3.
The six blank write-in ballots with a listed vote count for no one in this file.
Please treat these numbers as provisional until the clean file exists.
