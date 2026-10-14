# Coding standards

Ray Lutz, dictated on 2026-10-12. These apply to daffodil, AuditEngine, the ledger app and Ray's other Python
projects. The Code style section of daffodil's CLAUDE.md says much of the same, in shorter form.

## Layout

- Follow PEP 8 for spacing and the like, but not its line-length limit. Lines may be somewhat longer than
  PEP 8 allows, though not very long.
- Don't use Black or another auto-formatter, and don't reflow existing code.
- A function signature is a block, with one parameter per line, in three columns lined up vertically:
  the names, the type annotations with their defaults, and a comment for each parameter.

  ```python
  def convert_export(
          export_daf:     Daf,                    # the file as read.
          source_file:    str,                    # its name, kept in each row.
          keep_cols_ls:   T_ls | None = None,     # columns to keep. All if None.
          reverse:        bool = False,           # if True, the file is newest first.
          ) -> Daf:
  ```

  `derive_join_translator_daf()` in daffodil's daf.py is a real example.
- A short function that is used often can keep its signature on one line.
- A call follows the same rule. A call to a common function, such as a search with a few flags, goes on one
  line. A call with many arguments puts one argument per line, lined up like a signature.
- Lining up assignments and trailing comments in columns is good.

## Types

- Annotate everything: every parameter, every return value and every local variable, including loop
  variables. It costs nothing at run time.
- The annotations keep track of what each variable holds. Iterating over a `T_dols`, for example, gives a
  key that is a str and a value that is a `T_ls`, and the annotations say so where the reader needs it.
- A variable never holds two types, and annotating every variable enforces that.
- It also lets mypy check the code, and could let it be compiled with mypyc later.
- Use the type aliases from daffodil's `src/daffodil/lib/daf_types.py`. AuditEngine has its own,
  `my_types.py`, with the same scheme. Each alias is `T_` followed by letters for the type:
  - `s` str, `i` int, `f` float, `b` bool, `t` tuple, `a` Any.
  - `l` list, and `d` a dict with str keys. `id` is a dict with int keys.
  - `o` means "of".
  - So `T_ls` is a list of str, `T_la` a list of anything, `T_li` a list of int, `T_dols` a dict of lists of
    str, and `T_lola` a list of lists of anything.
  - daf_types.py explains the rest.
- Write unions with `|`, as in `int | str`, and a value that may be None as `T_ls | None`. Not `Union` or
  `Optional`.

## Names

- A name ends with the letters of its type, as in `names_ls`, `row_da` or `counts_di`. So a name never holds
  two types, and a name need not spell its type out, as `list_of_names` would.
- A Daf is a type too, so `daf` is a suffix, not a name: `export_daf`, `common_daf`. In daffodil's own
  examples and docs, a Daf is called `daf`, as pandas uses `df`.
- A SqlDaf, a Daf whose rows are in SQLite, ends in `_sdaf`, as in `bif_sdaf`, and its type is `T_sdaf`.
  A list of them is `_lsdaf`, `T_lsdaf`, and a dict of them `_dosdaf`, `T_dosdaf`. The aliases are in
  daffodil's sqldaf.py, not daf_types.py, which sqldaf.py imports. Ray, 2026-10-14.
- Common names such as `name` or `count` may leave the suffix off, but it does no harm.

## Functions

- Make each function pure where possible. It takes its data as arguments and returns its result. It does
  not read or write files, and it does not change global state.
- When files must be read or written, do it in a thin wrapper outside the pure function. The wrapper reads
  the files, calls the function and writes the result.
- This is what makes a function testable with the edge-test fixture. The fixture captures a function's
  inputs and outputs and compares a later run with that gold reference. It cannot do that for a function
  that loads its own files. AuditEngine's `edgetest/edge_test_utils.py` has the fixture.
- A script has a `main()` function. `if __name__ == '__main__':` only calls `main()`.

## Classes

- Use few classes. Most data is a table in a Daf.
- The Daf class does both jobs: it manipulates table data, and it marshals data in and out of files. For
  the mixed row data these projects use, it is competitive with other table structures.

## Documentation

- Write docstrings in the Google style, with Args, Returns and Raises sections, as daffodil does for its
  mkdocs reference. Daffodil's CLAUDE.md, under Documentation, has the rules.
