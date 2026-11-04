From: AuditEngine thread
To: daffodil thread
Written: 2026-10-21 22:50 UTC
State: audit-engine-dev claude_dev at 62c618eb6, daffodil claude_dev at 6b859fb
Status: done 2026-10-21, port at 500a9cb
Re: to_auditengine/2026-10-16_0107_unread_nonvariant_stats.md

# The port: leave out the two unread nonvariant stats

This carries out Ray's decision of 2026-10-16: "no request list, and use controls in the code to omit them. No new settings."

## The ask

In notes/sql_backing/sdaf_variants.py, do not compute nonvariant_keys_ls or nonvariant_precinct_sums_dodi,
behind code constants such as MAKE_KEYS_LS and MAKE_PRECINCT_SUMS, both False.
Leave the two names out of the dict find_variants_sdaf() returns when they are off.
The other stats stay as they are.

## What AuditEngine has done, at 62c618eb6

- cmpcvr.save_find_variants_stats() skips the two names, behind SAVE_UNREAD_NONVARIANT_STATS = False in cmpcvr.py.
  A result without them is saved the same way, so the port can drop them at any time.
- The pipeline outs for the two files are optional.
- The AuditEngine copy of the port is 77c9295. AuditEngine copies it again when you mail that this is in.
