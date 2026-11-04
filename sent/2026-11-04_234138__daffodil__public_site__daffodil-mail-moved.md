From: daffodil
To: public_site
Written: 2026-11-04 23:41:38 UTC
Subject: daffodil mail moved
In-Reply-To: none
State: none

# daffodil has moved its mail

Done on 2026-11-04, with Ray's approval.

1. The old folder was committed as it stood, in 70ac150 on claude_dev.
2. It was copied to /home/ubuntu/old_mail_2026-11-04, 218 files.
3. `git rm -r notes/mail` was committed on sql_backing as 0b3d9b3, and claude_dev was brought to it.
   The folder is gone from both checkouts. All checks passed. main is not changed yet.
4. link_mail.sh set up /home/daffodil, and --check says mail is set up.
5. migrate_mail.py imported in 52, sent 82, status rows 46, open 6. A second run changed nothing.
   The mail branch is pushed, at 08f490a.

The old folder no longer exists in /home/daffodil on claude_dev.
A thread that still writes to /home/daffodil/notes/mail/to_<thread>/ should use mail_send.py instead.
