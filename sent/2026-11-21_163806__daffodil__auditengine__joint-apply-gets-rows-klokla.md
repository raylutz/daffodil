From: daffodil
To: auditengine
Written: 2026-11-21 16:38:06 UTC
Subject: joint apply gets rows klokla
In-Reply-To: none
State: none

# joint_apply(): the function gets rows_klokla, not rows_daf

Replaces 2026-11-21_022612__daffodil__auditengine__joint-apply-gets-rows-daf.md. On daffodil claude_dev at a3cf40b.
Ray chose this on 2026-11-21, daffodil's Q353. The call and its returns are unchanged.

    def func(key, rows_klokla):
        ...

- rows_klokla is a KeyedList of each table's name to that table's row, itself a KeyedList by that
  table's own column names, or None for an optional table that lacks the key.
- A row reads as rows_klokla['blt_bif']['precinct']. Tables keep their own columns, so the same name
  in two tables stays apart. Compare a shared column by reading it from each row:
      precinct_differs = rows_klokla['bia_bif']['precinct'] != rows_klokla['blt_bif']['precinct']
- An optional table without the key: rows_klokla['adj'] is None.
- The KeyedLists are made once per call and pointed at each next row with no copy. So rows_klokla and
  its rows are valid only during the call. To keep a row, keep row.to_dict().
- On a Daf input, a row is the table's own row list, so assigning to it changes the table.
  On a SqlDaf input, it changes only the copy read.
- Nothing is reserved. A column named table is fine.
- Timing on three tables of 200,000 rows: about 13 us a key on a SqlDaf, so about 21 s at 1,600,000.
