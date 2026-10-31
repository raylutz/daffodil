# For the mail to AuditEngine at the next release

Items to include when the next version is released. Clear this list once the mail is sent.
The list was last cleared on 2026-10-26, when 0.9.0 was mailed.

- `cols` of SqlDaf `select_where()`, `split_where()` and `group_where()` is deprecated and undocumented, Q323. It still works, so sdaf_variants.py needs no change. It can be dropped from those four calls at any time, at a small cost on wide tables.
