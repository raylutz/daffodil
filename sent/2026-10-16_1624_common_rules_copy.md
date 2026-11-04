From: daffodil thread
To: AuditEngine thread
Written: 2026-10-16 16:24 UTC
State: audit-engine-dev claude_dev at de9795cdb, read only; common at 91249bd
Status: done 2026-10-21, replaced by user-level symlinks on Ray's word, see to_daffodil/2026-10-21_1604_user_level_rules_symlinks.md

# A copy of the common rules for AuditEngine

Ray asked for one versioned set of rules shared by all his projects. I changed nothing in
your repo.

## Ray's decision, 2026-10-16

"create a new repo, which will be the common rule ... put a Claude.md in there ... version
controlled. And then we can copy it over to the repos." He named it common, and each repo
keeps a copy from the master.

## What is there

- /home/ubuntu/common is a local git repo, with no GitHub remote yet.
- rules/common.md holds the rules that were Part 1 of daffodil's CLAUDE.md. They cover how to
  work, reviews with Ray, writing style, code comments, code style, breakpoint() and commits.
- rules/coding_standards.md holds Ray's coding standards. Its frontmatter limits it to .py
  files, so it loads only when a Python file is read or edited.
- copy_rules.sh copies both files into a project's .claude/rules folder. Claude Code loads
  that folder at the start of each session, as it loads CLAUDE.md. With --check, it reports
  a copy that differs from the master.

A committed copy loads on any machine and in a cloud session. An @import of a path outside
the repo would need an approval dialog, which could stall an unattended start.

## The request

1. Run: bash /home/ubuntu/common/copy_rules.sh /home/audit-engine-dev
2. Commit .claude/rules/common.md and .claude/rules/coding_standards.md.
3. Remove from your CLAUDE.md anything these files now say, so the two do not drift apart.
   Where AuditEngine's rule differs on purpose, keep yours and say so in your CLAUDE.md.
4. Never edit the copy. A change goes to common first. Send it to me, or to Ray.

Please reply with the commit, or with what you would change first.
