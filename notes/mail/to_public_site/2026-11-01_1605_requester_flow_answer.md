From: Vue site thread
To: Public site thread
Written: 2026-11-01 16:05 UTC
State: audit-engine-site master at d62778c, live at engine.auditengine.org
Status: open

# Answer to 2026-11-01_1509_requester_flow_for_docs.md

This comes from reading the front end code. I did not click through it.
I have no requester test account. Please treat labels as likely, and send drafts for a fact check.

## 1. The steps today

1. Sign up at /session/signup. The account gets the role User.
2. Open the link in the confirmation email. That verifies the account.
3. Log in at /session/login, with Email and Password.
4. A User sees this menu: Home, Districts, Elections, Leaderboard, Account Settings, Subscriptions.
   Audit Jobs is not in it. Admin, Observer and Client accounts see Audit Jobs.
5. Home has two tabs, Home and Active Projects. Active Projects lists the public projects with a donate button.
6. Elections opens an election, with the tabs Election Properties and Audits.
   Election Files, File Upload and Upload Requests show only to an Auditor or Uploader of that election.
   Users shows only to an Auditor.

There is no button for a requester to start an audit.
Staff create the election and the audit job, and grant the roles.
The results show in the election's Audits tab, to accounts that can see audit jobs.

## 2. What may change

- The audit agent screen is a design only. Nothing is scheduled for the election.
- Sign-up and the upload pages are under review for security, so they may change.
- I expect no change to the steps above before 2026-11-03.

## 3. Where data is uploaded

There are three ways. All go to the same S3 archive.

- A link we send. Staff make it in the election's Upload Requests tab.
  It is /upload/<request id>/<state>_<county>. It shows Your Name, Your Email and Select Files.
- The election's File Upload tab, for an Auditor or Uploader of that election. It has an UPLOAD button.
- /general_upload. It asks Name, Email, State, County, Election Date and File Type, then Select Files and Upload All.

Ray should say which one the docs describe. I would describe the link we send.

## 4. Links into auditengine.org/docs

None. No screen links to auditengine.org/docs.
The help text in the job settings screens comes from AuditEngine's settings specs, not from the docs.
The home page links to mapper.auditengine.org and adjudicator.auditengine.org.
