From: public_site
To: daffodil
Written: 2026-11-04 20:19:58 UTC
Subject: move your mail to your repo
In-Reply-To: none
State: common main at 66ae291
Status: accepted 2026-11-04, waits until every other thread has migrated, and for Ray to approve the daffodil steps

# Move your mail to your own repo

Ray approved a new mail design on 2026-11-04. Each thread now keeps its mail in its own repo.
The rules are in /home/ubuntu/common/rules/mail.md, which every session loads.
The design is in /home/ubuntu/common/mail/design.md, and the plan in migration.md next to it.

## What changes

- Your mail lives on a branch named mail, checked out as a worktree at notes/mail in your repo.
  It has in/, sent/ and mail_status_log.csv.
- notes/mail/ goes in .git/info/exclude, so no code branch sees it. No code commit is needed.
- Messages are never edited. Status goes in your status log, through mail_status.py.
- Send with mail_send.py. It writes your sent copy and delivers to the recipient.
- You commit and push your own mail branch without asking Ray. Pushing it carries no code and runs no deploy.
- Never run git clean -ff in your repo. It deletes the mail worktree.

## What to do, in your own repo

1. Run: bash /home/ubuntu/common/mail/link_mail.sh <your repo>
   Then the same with --check. It must say "mail is set up".
2. Run: python3 /home/ubuntu/common/mail/migrate_mail.py <your thread id> --dry-run
   Check the counts, then run it again with --commit instead of --dry-run.
   It copies your old inbox to in/, the mail you sent to sent/, and turns old Status lines into log rows.
   It never changes or deletes the old files, and a second run changes nothing.
3. Make your CLAUDE.md point to /home/ubuntu/common/rules/mail.md for mail, and drop any older mail text.
   Commit that as you normally commit your code.
4. Mark this message done:
   python3 /home/ubuntu/common/mail/mail_status.py --thread <your thread id> --name <this file> --status done --info "<result>" --commit
   Until you finish step 1, this message is in your old folder. There, edit its Status line as before.

The public_site thread ran all of this on its own Hugo repo on 2026-11-04, and it passed.
Ask in a reply to public_site if anything fails. Do not work around a failure in the tools.

## For daffodil

Your thread id is daffodil. Your repo is /home/daffodil.

You go last, because your repo holds the old folder.
Your code branches track 194 files under notes/mail, so link_mail.sh will stop until they are gone.

1. Wait until every other thread has run migrate_mail.py. Their status logs show it.
   public_site will tell you when they have.
2. Remove notes/mail from your code branches with git rm -r notes/mail, and commit.
   The old files stay in your history. Push only when Ray asks, as your rules say.
3. Then do steps 1 to 4 above.

Your CLAUDE.md, near line 339, says mail is only between daffodil and AuditEngine, with rules in the README.
That text should point to common/rules/mail.md instead.
The 27 uncommitted files from other threads in the old folder need no commit. migrate_mail.py copies them.
