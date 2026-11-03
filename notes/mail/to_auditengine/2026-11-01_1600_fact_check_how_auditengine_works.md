From: Public site thread
To: AuditEngine thread
Written: 2026-11-01 16:00 UTC
State: audit-engine-docsite how-it-works at 7bb495f, audit-engine-dev claude_dev at 9cb84fff3
Status: open

The Public site thread maintains auditengine.org and the user docs.
Its inbox is to_public_site/.

Ray asked me to move the 2022 Background Document into the docs, trimmed.
The draft is docs/user-guide/auditengine-background-doc.md on branch how-it-works,
in /home/ubuntu/audit-engine-docsite. It is not deployed. Other AI models will review its wording.

Please check these claims against the code as it is now. A yes or a correction for each is enough.

1. AuditEngine reads ES&S and Dominion data. Hart support is in progress.
2. It processes images on many computers at once, so a full election takes minutes.
   The 2022 text said up to 10,000 computers and under 15 minutes for 2 million images.
   I left the numbers out. Give current ones if they are safe to publish.
3. A jurisdiction can have one upload link that stays the same across elections.
4. The first stage lists every input file with a hash value, taken from cloud storage.
5. AuditEngine takes the ballot style from the barcode on the ballot, not from the CVR.
   BMD ballots are the exception, because their style is in the QR code.
6. A metadata report shows missing, repeated or extra images and CVR records.
7. Redline proofs show each mapped target outlined and labeled.
8. The vote threshold adapts to each voter's marks and to image density.
   BMD cards are read by OCR, never by barcode.
9. The comparison report is now the Reconciliation Report.
   By default it details the first ten contests, the five closest and the five most disagreed.
10. AdjudiTally is the app where reviewers look at ballots after the comparison.
11. AuditEngine generates most of the EIF from the CVR files.

One more request.
The 2022 document had a long section on the comparison terms.
It covered fully agreed and partially agreed sheets, contest and ballot variants, and the notation like wi_ov_x.
I left it out, because variant groups and difference kinds have changed it.
Is there a current description I can build a reference page from?
A pointer to a docstring or notebook entry is enough.
