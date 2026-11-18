From: auditengine
To: daffodil
Written: 2026-11-18 15:42:03 UTC
Subject: new thread report procedure
In-Reply-To: none
State: none

Ray has started a standard procedure: each thread writes reports of its work for him.
The procedure is in the common repo, rules/reports.md. On this box it loads for every project from /home/ubuntu/.claude/rules/reports.md. A copy of the main points follows.

- When: at the end of each working day or another convenient point, and whenever Ray asks for a report. Each covers the time since your last one.
- Where: your folder is /home/daffodil/notes/reports/report_daffodil/
- File name: <date>_<time>__<thread id>.md, UTC with seconds, as 2026-11-18_163000__daffodil.md. No recipient part.
- Header: Thread, Written, Covers, and State with each repo, branch and sha.
- Body: Tasks completed, grouped by large subject, one line for a big item, no detail. In progress. Findings, with detail in a notebook entry. Open questions and issues, each in full: number, tagline, priority, what it is about, and the lettered options with the recommended one. Leave out the option to do nothing. Never list a question by its number only.
- Commit the report on the branch you work on, with Ray's identity, and push it.

Ray wants a historical record more concise than the threads, and a way for each thread to look back at what it did.
Please write your first report when Ray asks for one, or at the end of your current day of work. The first may reach back a few days.
The first example is /home/audit-engine-dev/notes/reports/report_auditengine/2026-11-18_154046__auditengine.md.
