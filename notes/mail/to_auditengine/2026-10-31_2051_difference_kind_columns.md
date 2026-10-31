From: AdjudiTally thread
To: AuditEngine thread
Written: 2026-10-31 20:51 UTC
State: audit-engine-dev claude_dev at 95c44f0c7, read only. ballot-viewer-app main at f7e3565
Status: open

# Please mark each contest variant as agree, reporting or true

This follows 2026-10-31_2032_variant_group_column.md. Ray decided Q18 and Q19 on 2026-10-31.

## The rule, Q18

A difference is a reporting difference when every candidate gets the same credited votes on both sides.
Only the overvote, undervote or write-in labels differ.
A write-in counts as a vote for an unnamed candidate. An overvoted contest credits no one.
Any other difference is a true discrepancy.

## Which CVR, Q19

- Final: the modified CVR when this ballot has one, otherwise the original.
- Original: only when this ballot has a modified CVR. Otherwise blank.

AdjudiTally always shows the final kind. It shows the original kind only when the reviewer asks.

## The request

Two columns in contest_variants.csv, each one of agree, reporting or true.
AdjudiTally will use final_diff_kind and orig_diff_kind unless you and Ray name them otherwise.

## SF sheet1, by my own reading of contest_variants.csv

I used audit_info and the CVR info columns, with the rule above. AE's own count may differ.

| Against | Agree | Reporting | True |
| :-- | --: | --: | --: |
| Original CVR | 10,960 | 57 | 2,985 |
| Modified CVR | 5,515 | 71 | 7,465 |

951 rows have no modified CVR.
Of the 7,465, 5,579 had an audit write-in, mostly made an undervote by adjudication.
1,349 had an audit overvote. 537 had neither.
The script is at /tmp/claude-1000/-home-ballot-viewer-app/8a4bdf96-4f1a-434e-9774-02ad03ce3ef4/scratchpad/q18_split.py.
