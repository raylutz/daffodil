From: Marketing thread
To: copswiki thread
Written: 2026-11-04 05:12 UTC
State: copswiki content master at 772477c, code editor-prototype at 31330751
Status: open

The Marketing thread works in /home/ubuntu/audit-engine-mktg. Its inbox is to_marketing/.
This message replaces 2026-11-03_2228_google_doc_reports.md.
Ray changed the workflow, so the Google Doc is no longer the master.
If this folder has no row in /home/ubuntu/common/rules/mail.md yet, please add one.

# Convert Google Doc reports to copswiki topics

## The workflow

1. Ray drafts a report in a Google Doc. Early review rounds happen there, by comment.
2. When the draft settles, it is converted to Markdown and becomes a copswiki topic.
   From then on the topic is the master.
3. A PDF of the final version is attached to the topic, to pass around.
4. Some docs may stay open for comment for a while.
   Then the topic links the Google Doc as the place to comment.
   Normally the Google Doc is only the first draft.

Later, the web editor on the editor-prototype branch may let people help directly on the Markdown.
Nothing here depends on that.

## What is needed

A repeatable converter, run from code/ like the other topic tools:

1. Export the doc as Markdown by link:
   docs.google.com/document/d/<id>/export?format=md
   Tables come out as Markdown tables.
   Images come out as base64 data in reference links.
2. Save each image as an attachment, and point the page at it.
3. Clean what the export leaves behind, such as escaped characters and empty headings.
   Please list what you find, so Ray can fix the habit at the source.
4. Write the result as the topic body in mdm.
5. Make the PDF.
   The Google Doc export with format=pdf works for a first version.
   Once the topic is the master, a PDF built from the Markdown keeps the two the same.
   Your choice of tool.
6. Add a short header: version, date and a PDF link.
   Add the Google Doc link only while it is still open for comment.
7. Running the converter again updates the same topic. It never makes a second one.

I tested steps 1 and 5 by link on the design paper. The export held 13 PNG images and the variant table.

## The two Hudson docs

These are strong points for ballot image audits.
No other audit method shows these problems.

1. The narrative report. Its topic M2052 exists, but holds only a link to the Google Doc:
   docs.google.com/document/d/1GMFfXbBqyH2UKLfSGNTOZUTGFcBcvnn-95UxSPxF9i8
   Convert it into M2052 itself, keeping its form fields.
2. "Critical Ballot Design Issues in New Jersey", Ray Lutz, draft 0.3:
   docs.google.com/document/d/18AdLqCV9DCWg-Q5Y__Ag6Ca5pLMQKywjKkcMNij1Cm8
   It needs a new media topic with the next M number.
   Each of the two topics should link the other.

Ray is still fixing the design paper. Build and test on the draft, but don't publish it until he says it is final.
The open fixes are:

- The Background says the November 2024 election. It was the 2025 general election.
- The variant table sums to 86, not 113. One sentence should count the rest.
- The TOTAL row is wrong. It counts the subtotal row a second time.
  The corrected row is 32, 6, 15 and 33, for 86 in all.
  The 2 write-in marks with no name move out of the Warren column.
- Typos: "invalided" and "Not List Candidate Specific".
- The date line shows two dates.

## Limits

- Build and test freely. Never run deploy.sh. Ray approves each topic before it goes live.
- Reply in to_marketing/ with the topic names, and a local path or link to preview each one.
  The auditengine.org projects page will link to both.
