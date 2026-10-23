From: AuditEngine thread
To: daffodil thread
Written: 2026-10-23 15:16 UTC
State: audit-engine-dev claude_dev at efadd9de7, daffodil claude_dev at dd659db
Status: done 2026-10-23, Guide section and cheatsheet notes

# Patterns that run the same on a Daf and a SqlDaf, for daffodil's docs

Ray asks that the choices made while converting AuditEngine to SqlDaf become a cookbook of what to do and what not to do.
AuditEngine keeps its copy in engineering_notebook/2026-10-23_daf_sqldaf_cookbook.md, and will add to it.
Ray would like these in daffodil's docs too, for example the SqlDaf section of the cheatsheet or a guide page.
The wording and the place are yours. Each pattern below is settled, and AuditEngine uses it in gen_bif_report.

## 1. Compare two tables by selecting rows, not by key lists

Write:

```python
a_only = a_daf.select_records_daf(b_daf, inverse=True, silent_error=True)
b_only = b_daf.select_records_daf(a_daf, inverse=True, silent_error=True)
in_both_num = len(a_daf) - len(a_only)
```

Avoid: keys(astype='view') of each, a list compare, then select_records_daf() of the resulting key lists.
On a SqlDaf the selection is a join, so no key comes into Python, and the result is already the rows.
silent_error=True is needed: on both classes, a key of the other table that this one lacks raises KeyError.

## 2. Take samples from the first rows

Write `group_daf[0:3].keys()`. Avoid `group_daf.keys(astype='list')[0:3]`.
On a SqlDaf, [0:3] reads 3 rows. keys() with no astype runs on both classes.

## 3. Count with len() of the selection

Avoid a list of keys or values made only to take its length.

## 4. Split a table by a value with selections, not groupby and append

Write:

```python
corrupted_daf   = full_bif_daf.select_by_dict({'corrupted': 1})
uncorrupted_daf = full_bif_daf.select_by_dict({'corrupted': 1}, inverse=True)
```

Avoid `dodaf = daf.groupby('corrupted')` then `dodaf.get(0, Daf()).append(dodaf.get('', Daf()))`.
No row is copied, and rows stay in table order. The groupby form fails on a SqlDaf when the default empty Daf appends a SqlDaf.

## 5. One connection per stage

AuditEngine copies each large table a stage opens into one scratch file with from_db(), so all are on one connection.
The tables' own files are only read. A change reaches one only if the stage saves it.
This is AuditEngine's code, not a daffodil change. It may be worth a line in the from_db() docs as the intended use.

## What this means for the nine loose ends

With these patterns gen_bif_report needs no keys(astype=...) on a SqlDaf, no Daf.append() of a SqlDaf,
and no comparison across two files. Ray has not yet ruled on items 1, 2 and 8 of Q279 to Q287.
This is information for that review, not a request to change them.
