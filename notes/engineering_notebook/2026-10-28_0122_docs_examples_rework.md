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
