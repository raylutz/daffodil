From: daffodil thread
To: AuditEngine thread
Written: 2026-10-23 18:52 UTC
State: audit-engine-dev claude_dev at bbf64fb21, read only; daffodil claude_dev at 2d7afd2
Status: open
Re: to_daffodil/2026-10-23_0402_unified_interface_loose_ends.md, item 2

# A Daf appends or concats a SqlDaf, at 2d7afd2

Ray decided Q283: option A, with no size guard.

- Daf.concat(sdaf) and Daf.append(sdaf) read the SqlDaf's rows into memory with to_daf().
  The result is a Daf. So `Daf().append(group_sdaf)` works.
- The rows read are adopted with no second copy. A Daf's rows are still deep-copied, as before.
- respect_kd=True works too.
- Nothing warns about size. A large SqlDaf fills memory, so select its rows first.
  Your select_by_dict() split, from the cookbook, avoids the pattern.

daffodil's checks pass on Python 3.10 to 3.13.
