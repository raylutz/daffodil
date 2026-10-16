# Common rules for all projects

These rules apply to all of the owner's projects. Each project keeps its own rules in its
own CLAUDE.md.

## This file is a copy

- The master copy is in the repo named common, in its rules folder. On the EC2 box that is
  /home/ubuntu/common/rules.
- Each project keeps a copy of that folder in .claude/rules. Claude Code loads it at the
  start of every session, as it loads CLAUDE.md.
- The copy is committed in the project. So it loads on any machine and in a cloud session.
- Never edit the copy. Make the change in common and commit it there. Then copy the folder
  to each project with copy_rules.sh in common, and commit it in the project.
- A thread writes only in the repos it works in. A project that another thread owns takes
  the new copy through that thread or through the owner.

## How to work

- Don't use sub-agents. Work in a single thread. Speed doesn't matter. Cost and accuracy do.
- Get approval before any change to library behavior. This includes fixes, new errors and
  removed checks. Adding tests or docs needs no approval, but say what you added.
- Check every claim by running the code. Describe what actually happens. For example, don't
  say a column was dropped when only its name was lost and the data is still there.
- Commit and push before starting a change that touches many places.

## Reviewing changes with the owner

- The owner often reads on a phone and can't open files on the EC2 box. Present
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

## Code comments

These rules apply to code comments, docstrings and test docstrings.

- Write for a human reader, not only for a parser. Use short, plain sentences.
- Use few backticked names. Leave space between ideas.
- Put one idea on each line. If a sentence needs "and" to join two claims, split it.
- Keep each line to about 20 words.
- Use the active voice and the present tense.
- Don't use em dashes, double hyphens, semicolons or parenthetical asides. Write a second line
  instead.
- Don't name the function or file being commented. The code is right there.
- Don't use words tied to a session or a time, such as "now", "this round", "recently" or
  "the fix". State facts, not how they were learned.
- Don't quote a person, or give the name, date or number of a decision. State the rule or
  fact that came out of it.
- Say why the code exists once, in the docstring. A comment at a line points back to it in
  one line. It doesn't repeat the story.
- Use one term for each concept throughout a file.
- Break a long comment block into sections with headers.
- Comment on blocks of code, to say what each block does at a higher level. Don't comment
  only on single lines.

Where history goes:

- The story of a change goes in CHANGELOG.md: what an earlier version did, what broke and
  what changed. The code says what it does and why, once.
- Who decided, when, and decision numbers such as Q168 go in CHANGELOG.md or notes/.
- A test docstring follows the same rules. "Regression (name, date): ..." breaks them.
- Where docs are built from docstrings, as in daffodil, a docstring is published text. It may
  say why the code works as it does, as a fact, but it holds no history. AuditEngine builds
  no docs.

Check each new docstring and comment right after writing it. Scan for "--", ";" and a name
with a date. Slips are most common when writing many new functions quickly.

## Code style

- The full coding standards are in coding_standards.md, next to this file. It loads when a
  Python file is read or edited. The points below are a summary.
- Use type annotations throughout, on local and loop variables too.
- Write unions with `|`, as in `int | str`, not `Union[int, str]`. Write a type that may be None
  as `T_ls | None`, not `Optional[T_ls]`. `Optional` only means the value may be None; whether an
  argument is optional is a separate thing, shown by its default.
- A class name in quotes cannot be joined with `|` at run time: `'Daf' | None` raises TypeError
  on Python 3.10 to 3.13. Quote the whole annotation instead, as in `'Daf | None'`. Do not add
  `from __future__ import annotations` to fix it in a module with doctests that define schema
  classes: their annotations become strings.
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
- Daffodil itself raises errors instead. See its CLAUDE.md.

## Commits

- Commits use the owner's identity: Raymond Lutz <raylutz@cognisys.com>.
- The global git config on the EC2 box has the name `raylutz`, so the repo sets the full name in
  its local config. The startup hook does that, but check: it did not run on 2026-10-06.
- Never add Co-Authored-By, Claude-Session or any other line crediting Claude. Claude is a
  tool, not an author. This also applies to pull request descriptions.
- Before committing, check the author with `git log -1 --format='%an <%ae>'`.

