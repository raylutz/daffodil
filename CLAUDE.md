# Notes for Claude

This file has two parts. The global rules apply to all of the owner's projects. The
Daffodil part applies only to this repo.

# Part 1: Global rules

## How to work

- Don't use sub-agents. Work in a single thread. Speed doesn't matter. Cost and accuracy do.
- Get approval before any change to library behavior. This includes fixes, new errors and
  removed checks. Adding tests or docs needs no approval, but say what you added.
- Check every claim by running the code. Describe what actually happens. For example, don't
  say a column was dropped when only its name was lost and the data is still there.
- Commit and push before starting a change that touches many places.

## Reviewing changes with the owner

- The owner often reads on a phone and can't open files in the cloud session. Present
  reviews in the chat.
- Start with a short numbered list, one line per item.
- Then go through one item at a time. Wait for a decision before moving on. Some items need
  a night to think over, and some are easy.
- Present each item as a list of options only. Don't describe it in terms of branches or of
  what is there now. One option is always "Original": how the code behaved before any changes.
- For each option, give its code, the file and line, and real output from running it.
- Then say why it matters, what it could affect and which option you recommend.

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

## Code style

- Use type annotations throughout.
- Use the project's type aliases instead of spelling out types. In daffodil they are in
  src/daffodil/lib/daf_types.py. Other projects may have their own types module.
- An alias name is T_ followed by letters for the type:
  - d is a dict with str keys, and id is a dict with int keys.
  - l is a list, s a str, i an int, f a float, b a bool, t a tuple and a is Any.
  - o means "of".
  - For example, T_da is a dict of Any and T_ls is a list of str.
  - T_dols is a dict of lists of str, and T_dodi is a dict of dicts of int.
- End each identifier with the same letters, so its type shows in its name. For example:
  row_da, colnames_ls, result_dodi. A list of lists is lol, as in data_lol.
- Follow PEP 8 in general, but not its line-length limit.
- Don't use Black or any other auto-formatter. Don't reflow existing code.
- Lining things up in columns is good. This includes assignments and trailing comments.
- A short, commonly used function can keep its signature on one line.
- Otherwise, put one parameter per line, with names, types, defaults and comments lined up
  in columns. derive_join_translator_daf in daffodil's daf.py is a good example.
- When showing code in the chat, keep it narrow enough to read on a phone. Show only the
  relevant lines, and trim long comments.

## breakpoint() in the owner's projects

- AuditEngine uses `breakpoint()` at places where reaching it means the code is probably
  written wrong. An example is asking for a setting that isn't in the settings dict. There
  are about 400 of these.
- A breakpoint hook replaces pdb. It writes a report with the stack trace and locals to the
  file system, where an AI assistant can inspect it.
- The hook can be set to exit the program with code 42 at a breakpoint.
- Or it can be set to continue, for example when running in AWS Lambda. Then the code after
  the breakpoint runs. So that code must do something reasonable, never something
  catastrophic.
- A breakpoint that stays should be followed by a raise. Otherwise, continuing often fails a
  few lines later with UnboundLocalError, which hides the real problem.
  - Inside an except clause, a bare `raise` re-raises the original error, and its traceback
    still points at the line that failed.
  - Outside an except clause, a bare `raise` gives RuntimeError. Raise a named error there,
    such as `raise ValueError(...)`.
- Daffodil itself now raises errors instead. See the daffodil part below.

## Commits

- Commits use the owner's identity: Raymond Lutz <raylutz@cognisys.com>.
- A cloud container starts with a Claude identity, so set the owner's identity first.
- Never add Co-Authored-By, Claude-Session or any other line crediting Claude. Claude is a
  tool, not an author. This also applies to pull request descriptions.
- Before committing, check the author with `git log -1 --format='%an <%ae>'`.

# Part 2: Daffodil

Daffodil is a small, fast, pure-Python library for 2-D data tables. The main class is Daf.
Source code is in src/daffodil and tests are in tests.

## Setup and tests

- Use uv. Running `uv sync` installs the package and the dev tools.
- A startup hook in .claude/settings.json runs `uv sync` and sets the git identity at the
  start of each session.
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

## Documentation

- The API reference is built from docstrings with mkdocs, Material and mkdocstrings. The
  settings are in mkdocs.yml, and the page sources are in docsite/. The folder docs/ is
  not used yet. The plan is to rename docsite to docs later.
- Build it with `uv run mkdocs build -d <folder>`. Build into a scratch folder, not the
  repo. The folder site/ is ignored by git.
- mkdocs is pinned below version 2. Version 2 drops plugins, so the setup would stop working.
- Ruff is only a dev tool here. mkdocstrings uses it to wrap long signatures in the docs. It
  is not used to reformat code.
- Docstrings use the Google style, with Args, Returns and Raises sections. Only the first
  string in a function is a docstring. A second string after it is ignored by the tools.
- The build prints warnings for docstrings that don't match their signatures. Fix them
  when you edit that function.

### Parameters and types in the docs

- The parameter table in the docs comes from the signature. The type and the default are read
  from it, so don't repeat them in the docstring. That removes a source of drift.
- Every parameter and every return value must be annotated. A missing annotation leaves a
  blank in the table. Use the T_ aliases.
- A parameter shows in the table only if the docstring has an Args entry for it. Write one
  short line for each: `name: what it does`. Put anything longer in the main description, or
  link to the page that explains it.
- Put the return type in the signature too. In Returns, say what the value means, not its
  type.

### Writing a docstring

- A docstring must teach. Say what the thing is, what it is for, when to use it and when
  not to. Never just restate the name. "Get the current mode" is not a description.
- Check every claim by running the code. Don't describe what you have not seen happen.
- Name the behavior a caller could trip over. For example, say if a result is a copy or a
  view, and if the original is changed.
- Add a short example when it helps. Write it in the doctest style, with `>>>`, and run it.
- The heading must be `Examples:`, in the plural. The singular `Example:` is shown as a
  note box, and the code is not highlighted.
- Put names that contain an underscore in backticks, such as `iter_dict()`. Otherwise the
  underscores turn text into italics.
- Keep a second string after the docstring as it is. If it holds useful information, move
  that into the real docstring.
- Explain each concept once, on its primary page, and link to it from everywhere else. For
  example, KeyedList is explained in the KeyedList docstring. Other docstrings say one
  sentence and link with `[KeyedList][daffodil.keyedlist.KeyedList]`. Don't repeat the
  explanation, because the copies drift apart.
- Docstrings use the Google style. A NumPy style heading such as `Examples` over a line of
  dashes is not parsed, and its examples show as plain text.

## Names

- Column and table names must not contain a double underscore. It is reserved for the
  encoding used to store any name in SQLite: __HH up to 0xFF, __uHHHH up to 0xFFFF and
  __UHHHHHHHH above. README.md states this rule.

## Missing values

- A missing or unknown value in daffodil is NULL, which is the empty string ''. It is defined
  in daf.py and daf_utils.py.
- Test for it with `val is NULL`. Python keeps one shared empty string, so `is` works and is
  faster than `==`.
- Printability comes first. An empty cell prints as nothing, which is what we want to see.
- Use NULL for missing values, not None, NaN or a typed placeholder.

## Performance over guard rails

- Daffodil is for people who want speed. Don't add checks to the normal path to protect
  against user mistakes. Users who need a check can do it themselves first.
- For rare cases, pick a cheap and reasonable behavior instead of adding a check. For
  example, assigning a list or Daf copies only where source and target overlap.
- Let Python raise its own errors where it already would. Add an explicit raise only where a
  mistake would otherwise pass silently and corrupt data, and keep it off the normal path.

## Errors, not breakpoint()

- Library code raises a specific error instead of calling `breakpoint()`. For example, use
  KeyError for a missing column, ValueError for a value of the wrong shape and TypeError for
  an unsupported type.
- Many of these errors are mistakes in the calling code, such as a wrong column name. No one
  will write a handler for them. The goal is to stop at the mistake while developing.
- Use Python's own error for a direct lookup of a name the caller gave, such as a column
  name. A plain KeyError already names it, so don't wrap it just to reword the message.
- Use a custom error when the failure is deep inside daffodil. Python's message there, such
  as "unsupported operand type(s)", doesn't tell the caller what they did wrong. Say what
  was wrong in daffodil terms, and name the function.
- To stop at the failing line with all locals, use post-mortem debugging. For example, run
  `python -m pdb -c continue script.py`, or `pytest --pdb`.
- When re-raising inside an except clause, use `raise ... from exc_info`.
- tests/conftest.py makes any test fail if it reaches `breakpoint()`. This catches new ones.
  daf_pdf.py and md_demo.py still contain some.
- AuditEngine installs its own breakpoint hook, so AI assistants can run code without pdb.
  The hook saves a report and raises an error instead of stopping.
- That hook looks for a local variable named exc_info. So if an except clause is followed by
  `breakpoint()`, name the caught exception exc_info.
- Never pass extra keywords to `breakpoint()` in daffodil. Python's default hook rejects them
  with a TypeError.

# Part 3: Handoff protocol (cloud sandbox)

This thread runs in a cloud sandbox. Its files do not survive the session, and the next thread
starts from a fresh clone. So a handoff exists only once it is committed and pushed. Ray starts
the next thread by hand. There is no transition script.

Two files, both tracked in git:

- `notes/handoffs/<YYYY-MM-DD_HHMM>.md`: the permanent archive. One file for each handoff, never
  edited after it is written. History for later review.
- `notes/handoffs/PENDING.md`: the pointer. It exists only while a handoff waits to be picked up.
  It holds one line: the file name of that handoff.

Both go on the branch that the next session will start from. That is `main`, unless Ray names
another branch. If the work is on another branch, commit the handoff files to `main` too, or tell
Ray which branch the next session must start from.

## Winding down: writing the handoff

When Ray says something like "wind down" or "hand off", do these steps in order.

1. Close out the work:
   - The tests pass: `uv run pytest -q`, the doctests and the docs build, as in CI. Report any that
     fail, with the output. Do not hide them.
   - Every change is committed and pushed. The working tree is clean.
   - CHANGELOG.md is current, with one line for each change.
   - Anything decided but not yet recorded goes into `notes/` or CLAUDE.md, wherever it belongs.
2. Write the handoff to `notes/handoffs/<YYYY-MM-DD_HHMM>.md`. Use the current UTC time, from
   `date -u +%F_%H%M`. Write it for a new session that has no other context. Keep it short. Cover:
   - **State:** the branch and commit, the version in pyproject.toml, whether it is released, and
     the test results.
   - **Decided and done:** what was settled this round, with the commits.
   - **In progress:** anything started but not finished, and exactly where it stopped.
   - **Next:** the next task, with Ray's own words if he gave it. Note anything that waits on
     AuditEngine or on a release.
   - **Open questions:** decisions that belong to Ray and have not been made.
   - **Pointers:** the notes files that hold the detail. Do not repeat what CLAUDE.md or the notes
     already say.
3. Write `notes/handoffs/PENDING.md` with the file name of that handoff. If a PENDING.md is already
   there, it points to an older handoff that was never picked up. Name it in your reply to Ray,
   then replace it.
4. Commit both files with the message "Handoff <YYYY-MM-DD_HHMM>", and push.
5. Check that the push reached the remote: `git log origin/<branch> -1` shows the handoff commit.
6. Tell Ray the handoff file name and the branch it is on, then end. Nothing else runs after that.

## Starting up: picking up a handoff

Do this first in a new thread, before any other work.

1. Read `notes/handoffs/PENDING.md`.
   - If it does not exist, nothing is waiting. Tell Ray so, and ask whether to read the newest file
     in `notes/handoffs/` instead. Then wait for his task.
   - If it exists, read the handoff file that it names.
2. Check that the handoff file is in `notes/handoffs/` and committed. It should be, since the last
   thread committed it. If it is missing, stop and tell Ray.
3. Delete PENDING.md, commit with the message "Pick up handoff <name>", and push. This marks the
   handoff as taken, so no later thread picks it up again. The handoff file itself stays, as history.
4. Tell Ray in a few lines what you picked up: the state, the next task and the open questions. If
   the handoff says something that the repository does not match, for example a test it says passes
   now fails, or a commit it names is missing, say so. Do not act on the handoff until Ray confirms
   the next step.

## Handoff notes

- Never edit or delete an archived handoff. A correction goes in the next handoff.
- Keep a handoff free of secrets, tokens and credentials.
- A handoff is a summary for the next thread, not the record. Decisions belong in CHANGELOG.md,
  `notes/` and CLAUDE.md, where they stay.
