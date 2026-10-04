# Action items for AuditEngine

Collected on 2026-10-04. These are changes in AuditEngine, not in daffodil.

1. Change the bool count columns in `BIF.py` to int.
2. Pass `cols=` at `pdf_image_indexer2.py:74`, because `from_lod()` raises `ValueError` for a key that only a later dict has.
3. Fix the reversed `.index` at `args.py:1007`.
4. Fix the malformed JSON in the GA_Dekalb JOB CSV.
5. Use `select_by_dict()` in place of `select_where()` where the test is equality on one or more columns. It is about 12 times
   faster. Check three things first:
   - A call that converts before it compares, such as `int(row['n']) == 5`, must stay. `select_by_dict()` compares the
     cell as stored, so text `'5'` does not match 5.
   - `inverse=True` is the negation of the whole match, not `!=` on each field.
   - `select_where()` raises `KeyError` for a column that does not exist. `select_by_dict()` returns no rows.
6. Review the calls that select rows and then append to, sort or pop the result, and the calls that select every row.
   Daffodil may change `select_irows()` with every row in order, `d[:]`, `select_krows()` with every key and
   `select_records_daf([], inverse=True)` to return their own row list.
