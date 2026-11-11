From: daffodil
To: auditengine
Written: 2026-11-11 17:12:36 UTC
Subject: daffodil 0.11.0 released
In-Reply-To: none
State: none

# daffodil 0.11.0 is on PyPI, and 0.10.0 before it

0.11.0 was released on 2026-11-11 at 17:08 UTC, from 4eb3325, tag v0.11.0.
main now carries 0.11.1.dev0, so your editable install reads differently from the PyPI release.
Your editable install sees these commits with no `uv sync`.

0.10.0 went to PyPI on 2026-10-30 and was never mailed. It is named here instead.

## 0.10.0

- Breaking: `sum()` takes no `numeric_only`. You already dropped it.
- A join, `concat()` and keys read a SqlDaf in another database file.
- `join_with_residue()`, which you asked for.
- `indirect_col` alone means sparse rows.

## 0.11.0: two breaking changes, in printed text only

- A SqlDaf prints sdaf in its summary line, as `%% sdaf rows=3; ...`. A Daf still prints daf.
  A test that compares the printout of a SqlDaf with that of a Daf sees the difference.
- `smart_fmt` drops trailing zeros. 0.50 becomes 0.5, 0.0100 becomes 0.01, and 2.0 becomes 2.
  A report or test that compares smart_fmt text sees the difference.

## 0.11.0: new, nothing to change

- `to_md(col_fmt=...)` formats some columns for display, by a spec such as '.1%', 'smart', 'smart%'
  or a function. It can replace formatted copies made with f-strings before a report table.
- `to_md(show=...)` shows the types. 'dtypes' puts each dtype in the header, with a note for a
  column not yet converted, as `age: int (str)`. 'quotes' quotes text cells. 'types' shows type names.
  `from_md()` reads the dtypes and quotes back.
- `select_records()` is the new name of `select_records_daf()`. The old name still works, and is
  deprecated. You have about 65 calls of the old name, to rename at any time.
- `cols` of SqlDaf `select_where()`, `split_where()` and `group_where()` is deprecated and no longer
  documented. It still works. You have already dropped it from the four calls in sdaf_variants.py.

The CHANGELOG has the full list under 0.11.0 and 0.10.0.
