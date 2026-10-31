# Docs examples rework: one example per parameter, real output, no `_ =`

Started 2026-10-28. Ray asked for docs that are easy to read: examples at the `>>>` prompt with their
real output, crafted so the output stays short, and an example for each parameter of a method, or for
a few together. The examples show behavior. Catching errors is the unit tests' job. Q306.

## How the work is done

- Tools outside the repo, in the job's scratch folder: a filler that runs examples and writes their
  real output where an example says `<OUT>`, a helper that replaces or extends a function's Examples
  section, and an audit that lists the parameters no example passes.
- A call that returns the Daf is chained to a narrow read, as `daf.append(row).select_record('a1')`,
  so the output is one line, not the whole table.
- A loop goes in a small function, so nothing prints inside it, as in a program.
- Text keys in example tables, so a key is not confused with a position.
- Every claim was checked by running the code. Where the code does not do what a docstring said,
  the docstring now says what the code does, and the bug is listed below.

## Bugs found, for Ray, not fixed

1. `to_md(disp_cols={...})`: the annotation allows a dict, but a dict gives a wrong header with no
   error. `{'item': 'Item'}` puts `item` over the first column and blanks over the rest.
2. `append()` of a Daf with other columns raises KeyError with a message hard to read, starting
   `columns mismatch: this_instance ():`.
3. `sum(numeric_only=True)` gives the text `'0.0'` for the columns it skips, as
   `{'id': '0.0', 'qty': 7, 'note': '0.0'}`.
4. `sum_da(astype=...)` converts only with `cols` and without `is_sparse`, at daf_apply.py line 2409.
   `sum_kl()` passes no `cols`, so its `astype` never converts. xfail tests in test_daf_reduction.py.
5. `by='sparse_row'` gives a wrong result with no error in `groupby_cols_reduce()` (drops the dict
   column) and in `multi_groupby_reduce()` and `multi_groupsum()` (zeros). `groupsum_daf()` cannot pass
   `indirect_col`, so it raises. Only `groupby_reduce()` handles it. Their `by` entries now say to
   leave it as `row`. xfail tests in test_daf_reduction.py.

6. A SqlDaf's `sum(numeric_only=True)` gives 0.0 for the columns it skips, as the Daf gives `'0.0'`.
7. Unsupported on a SqlDaf, each raising NotImplementedError, and documented as such:
   `indirect_col` in `select_where()`, `split_where()` and `group_where()`, `apply(keylist=)`,
   `groupby(cols=)`, `insert_col()` at a position other than the end, `join(fill=)` other than NULL.
   Not bugs, but gaps a user meets.
8. A SqlDaf's sort of a column of numbers mixed with text follows SQLite's order, with no error,
   where a Daf raises TypeError. Documented.
9. `buff_to_file(fmt=)` is not used. Its Args entry now says so.

Tests for the bugs, as xfail with a reason that starts with BUG:

- Bug 1: tests/test_daf_md.py, assuming a dict maps column names to display names. Ray to confirm.
- Bug 3: tests/test_daf.py `test_sum_numeric_only` expected `'0.0'`, which locked the bug in. It now
  expects the skipped column left out, as `sum(cols, numeric_only=True)` does, and is xfail.
- Bugs 4 and 5: tests/test_daf_reduction.py.
- Bug 2 is the wording of a message, so its test waits for Ray's wording.

## Decided and fixed, 2026-10-28

- Bug 3: Ray chose to remove `numeric_only`, Q310. `sum()` takes the columns to total instead.
- Bug 5: Ray chose that `indirect_col` alone means sparse rows, Q311, in `reduce()` and every grouping
  reducer. Each grouping reducer keeps the concise form of `groupby_reduce()`. The two xfail tests
  pass, and their markers are gone.
- Bug 1: Ray chose that a dict as `disp_cols` maps column names to names shown, and the others keep
  their names, Q307. `display_colnames()` in daf_utils.py serves the Daf and the SqlDaf.
- Bug 2: Ray approved a shorter message, Q308. `columns_mismatch_msg()` in daf_utils.py names the
  columns found in only one table, or shows both orders. The Daf and the SqlDaf give the same text.
- Bug 4: fixed with Ray's approval. `_astype_val()` converts in both loops of `sum_da()`.
- All three at 6030424. No xfail tests remain.

## Done so far

- Quick start and Getting started rewritten. The Quick start runs as a doctest, locally and in CI.
- Building tables: `append()` one example per kind of row.
- A moved method's signature shows its method name, by a template override. Build tools in
  docsite/ are not published.
- daf_report.py: `to_md()` with an example for each parameter, the value counts, `to_md_cols()`,
  `daf_to_lol_summary()`, `dict_to_md()`, `gen_stats_daf()`.
- daf_select.py and the brackets: an example per kind of selector, `col()` rewritten.
- daf_modify.py: `append()` rewritten, the options of the rest.
- daf_apply.py: `apply()`, `reduce()` and the grouping, sum and count options.
- daf_convert.py, daf_structure.py, daf_dtypes.py, daf_join.py and the core: every option, with an
  .ods file made in memory for `from_ods()`, which had no example.
- The SqlDaf modules: every option, and each place where a SqlDaf differs from a Daf.
- A second audit leaves only options that are deprecated, not used, not supported on a SqlDaf,
  shown on a linked method, or waiting on Ray (`numeric_only`). `diagnose` prints timestamps, so it
  is described in words, not shown.

## Second round, 2026-10-28, overnight

Ray asked whether the full review, as done for `to_md()`, was finished. The audit of the first round
counted an option as shown when the SqlDaf docstring said "as Daf.x() does". So about 35 SqlDaf
methods had no example at all. This round:

- Every SqlDaf method on the API pages has examples, one for each option, or for a few together.
  Commits d051ac7, 96ef308, 77e8534, 3b85f30.
- Each mention of a Daf method in a SqlDaf docstring links to its entry. A script found the
  entry by the `heading` of each `:::` line in docsite/api/daf.
- Thin Daf descriptions were filled out, each claim checked by running. 91d91a2.
- Two audits run over every documented function: parameters with no Args entry (only `from_pdf()`,
  which is experimental and left alone), and errors raised by name with no Raises entry (four, now
  listed). c98a636.
- Stale text fixed: SqlDaf `join()` said the other table must be in the same file, which Q312
  changed. SqlDaf `groupsum_daf()` said only `by='row'`, which Q311 changed. SqlDaf `apply_dtypes()`
  said `cols=None` meant the columns of the dtypes. It means every column, as in a Daf.

Tools in the job's scratch folder: triage.py scores descriptions and counts examples for each API
entry. args_audit.py and raises_audit.py are the two audits. desc_only.py replaces a description
and keeps the examples. add_raises.py adds a Raises entry. link_daf.py adds the links.

### Seen, not changed, for Ray

10. `to_daf()` of a SqlDaf selection gives a Daf named for the internal TEMP table, as
    `daf_tmp_items_v1`. A Daf selection has an empty name. Shown in the `to_daf()` example.
11. The class attributes on the API pages, such as `RETMODE_VAL` and `SqlDaf.engine`, have no
    examples. They are constants, described in words.

## Third round, 2026-10-31: headers and general data

Ray asked that every example have a header, a plain line before it that says what it does, and that
the examples move away from election data to general data, with distinctive numbers. The rules are in
CLAUDE.md, under Writing a docstring.

- Every docstring with examples, 320 of them, and the Guide pages. A header-gap tool finds an example
  group with no prose line before it. It reports 0 in every module.
- Data: people with ids, names, cities and ages. Orders with a city, a quantity and an amount. Carts,
  a dict of item to count, for sparse rows. Customers with a plan. Election words are gone from the
  examples. The scope script still matches "selection" and "sheet", which are false positives.
- SqlDaf examples echo the SqlDaf at the prompt. A selection's last line shows its made-up name.
- Headers that state behavior were checked against the outputs. Six were wrong when first written and
  were fixed. Five docstrings were wrong about the code: `set_icol_irows()` skips a negative position,
  `to_donpa()` gives a text array for a column with an empty cell, `sum()` converts text of digits,
  `keys()` lists a repeated key once, `groupby_cols_reduce()` keeps only the grouped column and totals.
- Tools in the job folder: retheme.py swaps text inside chosen docstrings and marks their outputs for
  refilling. gaps.py lists or inserts headers. blank_before.py puts a blank line before an inserted
  header. md_mark.py marks outputs on a Guide page. fill_prop.py fills a property docstring, whose line
  numbers confuse the filler.
