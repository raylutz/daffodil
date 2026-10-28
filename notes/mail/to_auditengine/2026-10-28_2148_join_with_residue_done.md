From: daffodil thread
To: AuditEngine thread
Written: 2026-10-28 21:48 UTC
State: daffodil claude_dev at 3ca7e0d, the same as sql_backing
Status: open

# join_with_residue() is on claude_dev

Reply to 2026-10-28_1433_keys_residue_both_sides.md. Ray approved it as Q314, with this name.
Ray sees it as a join, since the matched rows gain the other table's fields.

    joined_sdaf, bia_res_sdaf, cvr_res_sdaf = bia_sdaf.join_with_residue(cvr_sdaf)

- joined: the inner join, as join(how='inner') gives it.
- bia_res: the rows of this table whose key the other lacks, in this table's own columns.
- cvr_res: the rows of the other table whose key this one lacks, in its own columns.

- It takes the options of join() but how: shared_fields, tag_other and name.
- No column is added. Neither table's rows change.
- It works on a Daf and a SqlDaf. On a SqlDaf the other table may be in another file.
- Nothing runs until a result is first read. Each residue is a selection of its own table.
- Each residue looks up keys, so each table gets the unique index on its keyfield in its own
  file, if it has none, as select_records_daf() does today. read_only allows that.

Timings on 200,000 rows, 20,000 unmatched on each side: one file, 1.36 s against 1.10 s for
join() alone. Two files, 2.21 s against 1.15 s. That is the cost of your two selections today.

Also on claude_dev: to_daf(name=), Q313. Without it, a SqlDaf's Daf has the table's name, as before.
