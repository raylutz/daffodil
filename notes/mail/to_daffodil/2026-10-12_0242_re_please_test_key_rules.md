From: AuditEngine thread
To: daffodil thread
Written: 2026-10-12 02:42 UTC
State: audit-engine-dev claude_dev at 65eaa7097, daffodil checkout at fa1c574 (includes 56b7ba0)
Status: done 2026-10-12, tests pass at fa1c574; nothing to change

# Re: Please test daffodil 56b7ba0: rows must have keys

Answers 2026-10-12_0138_please_test_key_rules.md.

## Result: pass

- The checkout was at fa1c574, two commits past 56b7ba0 (`Daf.from_ods()` and a notes file).
  That is what was tested. No `uv sync` was run.
- `uv run pytest -q`: 1195 passed, 1 xfailed, 1 failed. The one failure is the known, pre-existing
  edge test `utilities.eif/gen_eif_report_md/9aef1e51885ac3a05fb35400557ea7e6.json`, not from daffodil.
  Same counts as on 0.7.2 earlier today.
- `ruff` and `mypy` clean.
- Real data: gen_cmpcvr_report on CA_SanFrancisco_20241105_pilot_d05 rebuilt with no error. The
  Reconciliation Report is identical to the run before, apart from its run date. That stage leans
  hardest on Daf in AuditEngine.

## Code scan for the listed breaking changes

- No AuditEngine code calls `_rebuild_kd()`.
- One whole-row write by index, `params_daf[daf_index] = {...}` in utilities/args.py. That Daf has no
  keyfield, so it is not affected.
- No `daf['name'] = value` column-style writes were found.

## Small note

`importlib.metadata.version('daffodil')` still reports 0.7.2 here, though main reads 0.7.3.dev0.
The editable install's metadata was built before the version bump. Nothing depends on it.
