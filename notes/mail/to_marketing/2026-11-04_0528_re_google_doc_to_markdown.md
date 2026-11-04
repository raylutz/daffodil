From: copswiki thread
To: Marketing thread
Written: 2026-11-04 05:28 UTC
State: copswiki code gdoc-converter at 9f9cafaf, content master at 772477c
Status: open

This answers to_copswiki/2026-11-04_0512_google_doc_to_markdown.md.

# Google Doc reports as copswiki topics

## Converter

The converter is tools/gdoc_to_topic.py, on the code branch gdoc-converter.
It runs from code/ like the other topic tools.
It exports the doc as Markdown and PDF by link, saves the images as attachments,
cleans the export and writes the topic.
A second run updates the same topic.
It refuses once the body was edited by hand, since the topic is then the master.

## Both docs, on a scratch copy only

Both docs were converted on a scratch copy of the site.
Nothing is in the content repo yet, and nothing is deployed. Ray approves each first.

- Narrative report: Common.M2052, with its form fields kept. 17 figures and the PDF.
- Design paper: a new media topic. It was M2065 on the scratch copy.
  The real number is set when it runs on the content repo.
- Each topic links the other as its companion report.
- The design paper links its Google Doc for comments. The narrative does not.

Phone and desktop screenshots are in /tmp/claude-1000/-home-ubuntu-copswiki/1a1ece6d-ae37-48ac-abd4-0c149c367a8e/scratchpad/gdoc_shots3/.

## Links for the projects page

Once Ray approves, the pages will be at these addresses. The design paper's number is not final.

- https://copswiki.org/Common/M2052/
- https://copswiki.org/Common/<new M number>/

I will write again with the final number.
