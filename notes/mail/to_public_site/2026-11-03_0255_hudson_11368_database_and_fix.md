From: AdjudiTally thread
To: Public site thread
Written: 2026-11-03 02:55 UTC
State: ballot-viewer-app branch checkmark-sync at 9e9ca5e, not merged
Status: done 2026-11-03, relayed to Ray

# Hudson 11368: the database and the fix

This follows 2026-11-03_0243_hudson_11368_findings.md.

## The database

Ray approved a read-only query.
11368 holds checked false, verify with AuditEngine 1, and tags as an empty list.
The other 112 Ward A records are checked.
So the database itself holds the unchecked state. The stale copy in Save File does not explain it alone.

updated_at cannot say when it was written.
Opening a ballot locks it, and the lock rewrites updated_at on every other record of that reviewer.
Ray's 23 records all show 2026-11-03 02:35:32 UTC, the moment 11368 was opened.

The stored state matches the pending auto-save defect field for field.
That is the likely cause. The time of the write is lost, so it cannot be proven.

## The fix

Ray approved it. It is on branch checkmark-sync for his review.

1. One rule computes the checkmark, in the list, the detail and the filter panel.
2. Each save updates the in-memory copy that Save File reads.
3. A pending auto-save runs at once when another ballot opens, so it saves the ballot it belongs to.

The stored record for 11368 is unchanged. Ray can check it again in the detail view and save it.
