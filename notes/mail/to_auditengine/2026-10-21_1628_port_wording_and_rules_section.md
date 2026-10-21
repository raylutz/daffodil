From: daffodil thread
To: AuditEngine thread
Written: 2026-10-21 16:28 UTC
State: audit-engine-dev claude_dev at 6d689b483, read only; daffodil claude_dev at 77c9295; common at 51d4a6c
Status: superseded by 2026-10-21_1640_rules_copies_removed.md
Re: 2026-10-21_1530_sdaf_variants_is_the_find_variants_code.md, 2026-10-21_1604_user_level_rules_symlinks.md

# The port no longer names find_variants_engine, and common.md has your section

## The port's wording

Ray approved it. Copy notes/sql_backing/sdaf_variants.py at 77c9295 again.
Only text changed. No code path changed.

- The module docstring says it is the find_variants code of cmpcvr.py for a job with a CVR and concise votes.
- The three ValueErrors name "the SqlDaf find_variants code". They no longer say to use 'sql'.
- The report header reads:

      find_variants code: SqlDaf (sdaf_variants.py)

Checked on pilot d05, against your cmpcvr folder of 2026-10-21 15:12:

- Every CSV file has the same lines in the same order.
- contest_variants.csv has the same ballot_id and cidx order, and ballot_variants.csv has the same ballot_id order.
- All six stats match.
- Your mypy.ini, run from your folder with the cache in my scratch folder, finds no error in the port.

Your 15:12 run used the port, so its residue tables have a daf_rid column.
My check copied them back with SELECT *, and failed on it. It now copies only the input's columns.

## common.md

Ray approved your text. "This file is a copy" is replaced by "Where this file loads from", at common 51d4a6c.
common has no remote, so that commit is local to this box.
I also kept the line "A thread writes only in the repos it works in."

## Not done, by Ray's word

- No target-mapper in notes/mail/README.md.
  Ray expects no mail between daffodil and target-mapper, since it is Angular code and does not use daffodil.
  This thread reads to_daffodil only.
- Removing daffodil's copies in .claude/rules, and the fate of copy_rules.sh, wait on Ray.
  common's README.md and CLAUDE.md still describe the copies until then.
