From: AdjudiTally thread
To: AuditEngine thread
Written: 2026-10-31 23:00 UTC
State: audit-engine-dev claude_dev at 95c44f0c7, read only. ballot-viewer-app main at f7e3565
Status: open

# What adjudications.csv will hold

This follows 2026-10-31_2051_difference_kind_columns.md. Ray decided Q20 and Q21 on 2026-10-31.

- adjudications.csv holds only results a person reviewed by eye and changed.
- A contest with no difference can never change the result, so it is never written.
- AdjudiTally offers two switches, for adjudicated write-ins and adjudicated overvotes.
  They only hide rows from the review list. They write nothing.
- No switch accepts CVR undervotes where the audit reads an overvote.
  Those are reporting differences under Q18, and the difference kind filter hides them.

So a contest absent from adjudications.csv was not changed by review.
AE's accepted views stay AE's own. AdjudiTally does not record acceptance.
