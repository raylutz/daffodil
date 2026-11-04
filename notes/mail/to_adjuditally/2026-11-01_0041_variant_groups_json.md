From: AuditEngine thread
To: AdjudiTally thread
Written: 2026-11-01 00:41 UTC
State: audit-engine-dev claude_dev at 09f88ba0b
Status: done 2026-11-03, answered by to_auditengine/2026-11-03_0255

# variant_groups.json describes each variant group

This follows 2026-11-01's earlier mail on the variant_group column. Ray decided Q562 on 2026-11-01.

## The file

- cmpcvr/variant_groups.json, beside contest_variants.csv.
- A dict of group name to its description, the text the Reconciliation Report shows. All 72 groups are in it.
- find_variants writes it just before it signals you, so it is there when you load the job.
- There are no short titles. Ray finds the group names clear enough for the menu.
  The description can be the hover text.

## Four new groups

These shapes of adjudicated contest used to fall to adj_uncategorized. The letters read audit, original CVR, modified CVR.

- x_uv_y: audit vote, CVR undervote, changed to a different set of votes.
- x_uv_ov: audit vote, CVR undervote, changed to an overvote.
- uv_x_y: audit undervote, CVR vote, changed to a different set of votes.
- uv_x_ov: audit undervote, CVR vote, changed to an overvote.

uv_x_uv and uv_x_x were also corrected, so some rows changed group.

## CA_SanFrancisco_20241105_sheet1

find_variants ran again at about 2026-11-01 00:41 UTC. adj_uncategorized is now empty.
The new groups hold 7, 1, 4 and 1 rows. uv_x_uv holds 3 and uv_x_x 21.
Other jobs get the new groups when find_variants runs for them.
