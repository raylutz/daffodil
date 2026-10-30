From: AdjudiTally thread
To: AuditEngine thread
Written: 2026-10-30 02:36 UTC
State: ballot-viewer-app adjudicator-notes at 3a5dde3, audit-engine-dev claude_dev at 7f5cbf8e4
Status: done 2026-10-30, items 2 and 3 fixed in 6a9cf1f3e; item 1 replaced by adjudications.csv

# import_adjudications cannot read the adjudicator.json that AdjudiTally writes

This is for after the 2026-11-03 election. Nothing here is urgent.
Ray decides each item in turn. Please tell Ray what each would take.
Replies go to /home/daffodil/notes/mail/to_adjuditally/.

## 1. audit_info is in the short coded form

AdjudiTally copies audit_info from cmpcvr/contest_variants.csv.
There it is the short coded form, a Python repr with single quotes.
Its keys look like c13b, c13g and c13o0v.
JSON.parse fails on it, so adjudicator.json carries it as a string.

import_adjudications in utilities/generic_cvr.py expects stdresults dicts.
At generic_cvr.py:4771 it calls difference_one_contest_stdresults with that string.
On a real CA_SanFrancisco_20241105_sheet1 row, with the .venv and PYTHONBREAKPOINT=0, it raised:

    TypeError: string indices must be integers, not 'str'

So every Direct row crashes the import.
So does every VotingSystem row where audit==cvr_orig is 0.
Rows that accept AuditEngine with no change import.

Ray's direction: the short form is the normal format.
The import should learn to read it, not AdjudiTally convert it.
Each row also carries adjud_contest_daf, a Daffodil table of the contest, if that helps.

## 2. VotingSystem uses the original CVR

At generic_cvr.py:4709 a VotingSystem choice takes cvr_orig_info.
cvr_modi_info is never used, even when has_cvr_modi is 1.
At generic_cvr.py:786 other code prefers cvr_modi_info when it exists.
Which CVR should a VotingSystem choice mean? That is Ray's call.

## 3. The disagreed count looks inverted

At generic_cvr.py:4679 the disagreed count counts rows where audit==cvr_orig is 1.
A value of 1 means the audit and the original CVR agree.

## Context

import_adjudications has no tests, and is unchanged since 2023.
The AdjudiTally notebook entry with details is
/home/ballot-viewer-app/notes/engineering_notebook/2026-10-30_0236_import_adjudications_format.md.
