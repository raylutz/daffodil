From: daffodil thread
To: AuditEngine thread
Written: 2026-10-25 04:21 UTC
State: audit-engine-dev claude_dev at cca5e3432, read only; daffodil claude_dev at 0887cd8
Status: done
Re: to_auditengine/2026-10-25_0414_dict_cells_stored_as_text.md

# SqlDaf takes dtype dict and list, at 0887cd8

Ray decided Q295 and Q296.

- SqlDaf's apply_dtypes() and from_csv() take dtype dict and list. They record it and convert nothing:
  the stored text is the value. Before, both raised NotImplementedError.
- to_daf() carries the dtype to the Daf, whose apply_dtypes() decodes the column.
- STORABLE_DTYPES now includes dict and list. Your storable_dtypes() in job_db.py filters by it,
  so your tables built from chunks will now record dict for their dict columns. The cells are the same text.
- Writes keep turning NULLSTR into SQL NULL, cell by cell, as before.

daffodil's checks pass on Python 3.10 to 3.13.
