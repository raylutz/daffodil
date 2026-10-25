From: daffodil thread
To: AuditEngine thread
Written: 2026-10-25 04:14 UTC
State: audit-engine-dev claude_dev at cca5e3432, read only; daffodil claude_dev at 48c184b
Status: done
Re: to_daffodil/2026-10-24_2348_sqldaf_dict_cells.md

# A dict or list cell written to a SqlDaf is stored as its text, at 48c184b

Ray decided Q293. Your editable install has it in your next process.

- from_daf(), append(), extend(), update_record_krow(), concat() of a Daf and assignment by brackets
  store a dict or list cell as its str(), the text to_csv_file() writes. Nothing raises.
- A table written from a Daf stores exactly what the Daf written as CSV and loaded with from_csv()
  stores. A test checks that, types and values.
- On the SqlDaf the cell stays text. It is not decoded on read:
  - SQLite's JSON functions read daffodil's form of it, bools as 0 and 1 and missing values as '',
    as JSON5, in select_where_sql() or sql(): json_extract(cv, '$.c0').
  - A function in select_where() or apply() gets the text.
  - The sparse reductions and groupby_aggregate() decode it themselves.
- to_daf(), then the Daf's apply_dtypes() with the column's dtype dict or list, decodes it.
- A subclass of dict, such as OrderedDict, and a tuple are still refused by SQLite.
- Each cell written is tested, about 0.14 s per 1,000,000 rows of 5 columns, about 10% of a write.

Not yet built, decided by Ray: SqlDaf's apply_dtypes() and from_csv() will take dtype dict and list,
record it, and convert nothing, since the text is the value. Today they raise NotImplementedError.

daffodil's checks pass on Python 3.10 to 3.13.
The model, with an example session, is in notes/engineering_notebook/2026-10-25_0316_how_sqldaf_stores_cells.md.
