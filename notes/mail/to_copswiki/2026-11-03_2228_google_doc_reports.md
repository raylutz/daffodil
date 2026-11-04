From: Marketing thread
To: copswiki thread
Written: 2026-11-03 22:28 UTC
State: copswiki content master at 772477c, code editor-prototype at 3133075
Status: superseded by 2026-11-04_0512_google_doc_to_markdown.md

The Marketing thread works in /home/ubuntu/audit-engine-mktg. Its inbox is to_marketing/.
Ray asked me to hand this task to you.
This folder is new. Please add a row for this thread to the table in /home/ubuntu/common/rules/mail.md.

# Publish Google Doc reports as copswiki topics

## Why

Ray writes his reports in Google Docs.
It is the only format where readers reliably comment.
So the Google Doc stays the master.

Readers also need a page that opens in one tap, and a PDF to pass around.
Each published version is a fresh export of the doc.
So the copies never drift from the master.

## What is needed

A repeatable way to turn a Google Doc into a copswiki media topic:

1. Export the doc as Markdown. This works by link:
   docs.google.com/document/d/<id>/export?format=md
   Tables come out as Markdown tables.
   Images come out as base64 data in reference links, such as `[image1]: <data:image/png;base64,...>`.
2. Save each image as a file attachment, and point the page at it.
3. Export the PDF the same way, with format=pdf, and attach it.
4. Add a short header: version, date, a PDF link, and a link to the Google Doc for comments.
5. Make it a new topic with the next M number, like the other reports in AuditReports and ElectionAudits.
6. Running it again for a new version updates the same topic.

I tested steps 1 and 3 on the first doc. The export held 13 PNG images and the variant table.
Nothing else was tried.

## The first doc

"Critical Ballot Design Issues in New Jersey", Ray Lutz, draft 0.3 of 2026-06-19.
docs.google.com/document/d/18AdLqCV9DCWg-Q5Y__Ag6Ca5pLMQKywjKkcMNij1Cm8

It covers the Jersey City Ward A contest in the Hudson County 2025 audit.
Its companion is the narrative report, M2052.

Ray is fixing these in the doc before it is published:

- The Background says the November 2024 election. It was the 2025 general election.
- The variant table sums to 86, not 113. One sentence should count the rest.
- Typos: "invalided" and "Not List Candidate Specific".
- The date line shows two dates.

## Limits

- Build and test freely. Deploy only after Ray approves this topic.
- Please reply in to_marketing/ with the topic name and link when it is ready.
  The auditengine.org projects page will link to it.
