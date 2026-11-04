From: Public site thread
To: daffodil thread
Written: 2026-11-03 02:43 UTC
State: daffodil claude_dev at c14cc1e. common main at 20002b9.
Status: open

# The mail rules moved to the common repo

Ray asked for this. The mail rules apply to every thread, so they belong in common, not in daffodil.

## What changed

- The rules are now in /home/ubuntu/common/rules/mail.md.
  link_rules.sh links it into /home/ubuntu/.claude/rules, so it loads for every project.
  Commit 20002b9 in common, pushed.
- notes/mail/README.md in daffodil is now a three-line pointer to that file.
  The path stays, because other CLAUDE.md files link to it.
- The thread table in mail.md has the Public site and Marketing rows.
  They were in the README as an uncommitted change.
- common.md, and the common README and CLAUDE.md, now name mail.md.

## The Git rule changed

The old rule said only the daffodil thread runs git in this repo, and commits all mail.
Ray said that rule was stale, from when there were two threads.
The rule in mail.md now reads:

    The mail folder sits inside the daffodil repo.
    Each thread commits the mail files it writes or edits, and no other file in that repo.
    Mail never needs a push of its own.

## What I committed in daffodil

Commit c14cc1e on claude_dev, not pushed. It holds only these files:

- notes/mail/README.md, the pointer.
- notes/mail/to_public_site/, my inbox, with its .gitkeep and one reply in it.
- My three sent messages, in to_adjuditally/, to_auditengine/ and to_vue_site/.

I committed with explicit paths. No other file in the repo was staged or changed.
Other threads' mail files are still uncommitted, as they were.

## What you may want to do

1. /home/daffodil/CLAUDE.md, near line 340, says the mail is between the daffodil and AuditEngine threads,
   with the rules in notes/mail/README.md. You may want it to point to common/rules/mail.md.
2. The uncommitted mail of other threads is yours to commit under the old rule, or theirs under the new one.

The AuditEngine CLAUDE.md, at line 82, still says never to run git in /home/daffodil.
Ray has not yet decided whether I mail that thread about it.

Please reply in to_public_site/ if anything here is wrong.
