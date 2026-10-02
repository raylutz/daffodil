# Daffodil: notes for Claude

Daffodil is a small, fast, pure-Python library for 2-D data tables. The main class is Daf.
Source code is in src/daffodil and tests are in tests.

## How to work

- Don't use sub-agents. Work in a single thread. Speed doesn't matter. Cost and accuracy do.
- Get approval before any change to library behavior. This includes fixes, new errors and
  removed checks. Adding tests or docs needs no approval, but say what you added.
- Check every claim by running the code. Describe what actually happens. For example, don't
  say a column was dropped when only its name was lost and the data is still there.

## Reviewing changes with the owner

- The owner often reads on a phone and can't open files in the cloud session. Present
  reviews in the chat.
- Start with a short numbered list, one line per item.
- Then go through one item at a time. Wait for a decision before moving on. Some items need
  a night to think over, and some are easy.
- For each item, give:
  - what happens now, from actually running the code;
  - the relevant code, with the file and line number;
  - why it matters and what it could affect;
  - the options, and which one you recommend;
  - the exact change you propose.

## Writing style

This applies to code comments, the changelog, docs and chat.

- Write short sentences, with one idea per sentence.
- Don't use em dashes, or double hyphens as dashes.
- Avoid parenthetical asides. If it matters, give it its own sentence.
- Use at most one or two backticked names per line.
- Don't fill the text with quoted values. Put code in a code block when it's needed.
- Never use relative times such as "yesterday", "earlier today" or "this morning". Give the
  actual date, and the time if it matters.
- Put a date or a name once, at the top of an entry. Don't repeat them through the text.
- Revise before presenting. Don't hand over a first draft.

## Setup and tests

- Use uv. Running `uv sync` installs the package and the dev tools. A startup hook in
  .claude/settings.json runs it at the start of each session.
- Run the tests with `uv run pytest -q -p no:cacheprovider`. It takes a few seconds.
- For coverage, add `--cov=daffodil --cov-report=term-missing`.
- All tests must pass before any commit.

## Layout

- src/daffodil/daf.py holds the Daf class, which is most of the code.
- src/daffodil/keyedlist.py holds KeyedList.
- src/daffodil/lib holds helper modules. Some Daf methods are defined there and attached to
  the class. This avoids circular imports and is deliberate.
- pytest only collects files named test_*.py. The other files in tests are scripts.

## Conventions

- daf_pdf.py is experimental and its design isn't settled. Don't write tests that lock in its
  current behavior. Don't refactor it unless asked.
- Tests are plain pytest functions, grouped by the method they test. They check actual
  output values. Tests must not use the network, so mock any network calls.
- When a test finds a real bug that you aren't fixing, write the test for the correct
  behavior. Mark it with `xfail(strict=True)` and a reason that starts with "BUG:" and names
  the file and line. Never write a test that expects the buggy behavior.
- Record notable changes in CHANGELOG.md under Unreleased. Include coverage changes with
  before and after percentages.

## Errors, not breakpoint()

- Library code raises a specific error instead of calling `breakpoint()`. For example, use
  KeyError for a missing column, ValueError for a value of the wrong shape and TypeError for
  an unsupported type.
- When re-raising inside an except clause, use `raise ... from exc_info`.
- tests/conftest.py makes any test fail if it reaches `breakpoint()`. This catches new ones.
  daf_pdf.py and md_demo.py still contain some.
- AuditEngine installs its own breakpoint hook, so AI assistants can run code without pdb.
  The hook saves a report and raises an error instead of stopping.
- That hook looks for a local variable named exc_info. So if an except clause is followed by
  `breakpoint()`, name the caught exception exc_info.
- Never pass extra keywords to `breakpoint()` in daffodil. Python's default hook rejects them
  with a TypeError.
- Commit and push before starting a change that touches many places.
