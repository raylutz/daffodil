# Daffodil — notes for Claude

Daffodil is a lightweight, pure-Python 2-D dataframe library (`Daf` class). Source is in
`src/daffodil/` (src layout); tests are in `tests/`.

## Setup and tests

- Use `uv`. `uv sync` installs the package plus the dev group (pytest, pytest-cov, numpy,
  pandas, pdfplumber, requests, xlsxwriter). A SessionStart hook in `.claude/settings.json`
  runs it automatically.
- Run the suite: `uv run pytest -q -p no:cacheprovider` (about 2–5 s).
- Coverage: `uv run pytest -q -p no:cacheprovider --cov=daffodil --cov-report=term-missing`.
- The full suite must stay green before any commit.

## Layout

- `src/daffodil/daf.py`: the `Daf` class (most of the code).
- `src/daffodil/keyedlist.py`: `KeyedList`.
- `src/daffodil/lib/`: helpers. Some `Daf` methods are module-level functions wired onto the
  class (e.g. `from_md = daf_md._from_md`) to avoid circular imports; this is deliberate.
- `tests/test_*.py` are collected by pytest. Other files in `tests/` (`*_demo.py`,
  `npao_*.py`, `daf_benchmarks.py`, ...) are scripts, not tests.

## Conventions

- `daf_pdf.py` is experimental and its API isn't settled. Don't write tests that lock in its
  current behavior, and don't refactor it unless asked.
- Tests are plain pytest functions, grouped by the method under test, asserting actual output
  values (not just that lines execute). No network access in tests; mock it.
- When a test exposes a real bug and the fix isn't in scope, write the test for the *correct*
  behavior and mark it `@pytest.mark.xfail(strict=True, reason="BUG: <what> (file:line)")`.
  Never encode buggy behavior as expected.
- Record notable changes in `CHANGELOG.md` under `## [Unreleased]` (Keep a Changelog format),
  including test-coverage additions with before -> after percentages.

## Errors, not breakpoint()

- Daffodil library code raises a specific exception on error paths (e.g. `KeyError` for a
  column or setting that doesn't exist, `ValueError` for a value of the wrong shape,
  `TypeError` for an unsupported type) instead of calling `breakpoint()`. Use
  `raise ... from exc_info` when re-raising from an `except` clause.
- `tests/conftest.py` makes any `breakpoint()` reached during a test fail, so new ones are
  caught. `daf_pdf.py` (experimental) and `md_demo.py` still contain some.
- Background: in AuditEngine, `utilities/breakpoint_hook.py` replaces pdb so AI assistants can
  run code non-interactively; it captures a report and raises `BreakpointCaptured`. That hook
  looks up the exception in a local named `exc_info`, so name captured exceptions `exc_info`
  where an `except` clause is followed by a `breakpoint()`. The default hook
  (`pdb.set_trace(*, header=None)`) rejects unknown keywords, so never use
  `breakpoint(extra=...)` in daffodil.
- Commit and push current work before starting any sweep across many call sites.
