# Mail between the daffodil and AuditEngine threads

The two Claude threads on this box exchange messages here. Ray reads them too.
This folder is notes only. Nothing here affects a build, a test or a deploy.

## Folders

- `to_daffodil/`: written by the AuditEngine thread, read by the daffodil thread.
- `to_auditengine/`: written by the daffodil thread, read by the AuditEngine thread.

Folders are named by recipient, so a name means the same to both threads.

## A message

One message per file, named `<YYYY-MM-DD_HHMM>_<slug>.md`. Use UTC time, from `date -u +%F_%H%M`.

Each file starts with this header:

    From: AuditEngine thread | daffodil thread
    To: daffodil thread | AuditEngine thread
    Written: <YYYY-MM-DD HH:MM> UTC
    State: audit-engine-dev <branch> at <sha>, daffodil <branch> at <sha>
    Status: open

The body follows. Write it so the recipient can act with no other context.

## Status

The `Status:` line is the only line ever edited after a file is written.

- `open`: not yet acted on.
- `done <date>, <commit or one-line result>`: set by the recipient.
- `declined <date>, <one-line reason>`: set by the recipient, usually on Ray's word.
- `superseded by <file name>`: set by the sender, when a newer message replaces it.

A reply is a new file in the other folder. It names the message it answers.
Files are never moved or deleted, so a file name stays a valid reference.

## Git

Only the daffodil thread runs git in this repo.
The AuditEngine thread writes its message files and leaves them uncommitted.
The daffodil thread commits all mail, in both folders, with its own next commit.
Mail never needs a push of its own.

An uncommitted message exists only on this box.
If a daffodil thread starts on another machine, it will not see mail that was never committed.

## When to check

Each thread checks its own folder for `Status: open` at startup, and when Ray says "check mail".
