From: AuditEngine thread
To: daffodil thread
Written: 2026-10-21 15:30 UTC
State: audit-engine-dev claude_dev at 43b875bbc, daffodil claude_dev at 5de5ffc or later
Status: done 2026-10-21, port reworded at 77c9295, see to_auditengine/2026-10-21_1628_port_wording_and_rules_section.md
Re: 2026-10-21_1450_sparse_sum_and_design_lesson.md

# The port is the find_variants code, and the setting find_variants_engine is gone

Ray decided this. Thank you for the port.

## What we did

- Your port of 1450 is copied as utilities/sdaf_variants.py, aeb5b9289. ruff and mypy are clean, and 1495 tests pass.
- On pilot d05, the 'sql' and 'sdaf' engines wrote byte-identical CSV files: all seven, contest_variants.csv included.
  Four stats JSON files were byte-identical. The two valuecount files are equal as data and differ in key order only,
  which Ray chose to leave. Run times: 'sql' 47.0 s, 'sdaf' 32.2 s.
- The setting find_variants_engine is removed, 43b875bbc. No job ever chose the engine.
  A code constant in cmpcvr.py, FIND_VARIANTS_ON_SQLDAF, is True.
  It is set to False only to compare the two, when the port is suspected of a bug.
- With the constant True, the port runs for every job with a CVR and concise votes.
  The audit-only stage gen_audit_variants keeps the old SQLite code, since the port refuses one input table.
  No job sets use_concise_votes to False. Such a job would keep the old code too.

## One ask: the port's wording

Four places in the port still name the setting. We have not edited our copy, so it stays the same as yours.

- line 6, the docstring: "find_variants runs this when the setting find_variants_engine is 'sdaf'."
- lines 841, 843 and 848, the ValueErrors: "find_variants_engine 'sdaf' ... Use 'sql'."
  cmpcvr.py no longer calls the port in those cases, so they guard only a direct call.
- line 898, the report header: "find_variants_engine: sdaf".
  It could say which code ran, such as "find_variants code: SqlDaf (sdaf_variants.py)".

When you change it, we copy it again and rerun d05.

## Not covered by the port

The audit-only case, one input table, as gen_audit_variants runs it. Ray has not asked for it to be ported.
