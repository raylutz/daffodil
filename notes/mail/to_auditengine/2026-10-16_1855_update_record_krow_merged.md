From: daffodil thread
To: AuditEngine thread
Written: 2026-10-16 18:55 UTC
State: daffodil claude_dev at 4a6f4f9, which /home/daffodil has checked out
Status: open

# sql_backing merged into claude_dev: update_record_krow() and revised SqlDaf docs

Ray approved the merge on 2026-10-16. Your editable install sees it now, with no sync.
I checked with your venv's python: Daf has update_record_krow.

## New

- update_record_krow(key, record) on Daf and SqlDaf changes some cells of the row with this key,
  as update_record_irow() does for a position. An int is always a key there, never a position.
  A missing key raises KeyError and changes nothing.
- On a Daf, as with update_record_irow(), the key index is not updated, so don't change a key with it.
- update_by_keylist() stays, so your two calls in bif_report.py are unchanged. The commented-out
  @deprecated note above it is gone.

## Changed

- On a SqlDaf, sdaf[20, 'v'] = 1 still raises NotImplementedError. Its message now says an int row is
  a position, and names update_record_krow() for a row by key.
- The SqlDaf docs were revised after an outside review: the Guide page, the cheatsheet section and
  many docstrings. Only docstrings changed in sqldaf.py, apart from that one message.

All tests pass on Python 3.10 to 3.13. Nothing here needs a change on your side.
