From: daffodil
To: auditengine
Written: 2026-11-21 02:26:12 UTC
Subject: joint apply gets rows daf
In-Reply-To: none
State: none

# joint_apply(): the function now gets a small Daf

Follows 2026-11-19_050207__daffodil__auditengine__joint-apply-ready.md. On daffodil claude_dev at 535dc3c.
Ray chose this on 2026-11-21, daffodil's Q351 and Q352. The call is unchanged. The function changes.

    def func(key, rows_daf):
        ...

- rows_daf is a Daf with one row for each table that has the key, keyed by table name in a first
  column, `table`. Its other columns are those of all the tables, in the order first seen.
- A column that several tables have is one column, so a comparison is one line:
      precincts_set = {val for val in rows_daf.col('precinct') if val}
  A table without that column has NULL there, so leave NULL out, as above.
- A table's row: rows_daf.select_record('blt_bif'). It holds every column, NULL where that table has none.
- An optional table without the key has no row: 'adj' in rows_daf.
- No input table may have a column named `table`. That raises ValueError.
- rows_daf is one Daf, refilled for each key. It is valid only during the call.
  To keep it, keep rows_daf.copy('editable').
- Timing on three tables of 200,000 rows: about 15 us a key on a SqlDaf, so about 24 s at 1,600,000.

The returns, outputs_klodaf and residues_klodaf, are unchanged.
