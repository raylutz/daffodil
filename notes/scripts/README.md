# Scratch scripts worth keeping

Saved on 2026-10-06 from a cloud session whose files do not survive. They are one-off tools, not part of the library. Paths inside them point at the old session and may need editing.

- regen_doctest_tables.py: reruns every doctest, takes the real output of table examples and rewrites the docstring, with `<BLANKLINE>` for blank lines. Run it without arguments for a dry run and with `apply` to edit files. It finds each example by its text, because the line numbers from doctest are unreliable for some methods.
- survey_doctests.py: counts doctests and how many print tables.
- mem_by_string_share.py: kept memory of a 1000 x 1000 table in Daffodil and pandas, for 1 to 1000 string columns, with small and large ints.
- bench_build_rows.py: builds a table of 1 str column and some int columns, the pandas way and the Daffodil way, and prints time and memory. The default is 200,000 rows of 10 columns; the Home page uses BENCH_ROWS=1000 BENCH_INTS=999. Its results are on the docs Home page.
- perf_select_by_dict.py and loda_probe.py: timing probes for `select_by_dict()` with a list of dicts.
- mkdocs_rendered_tables.yml: a copy of mkdocs.yml that loads the griffe extension in notes/griffe_blankline_extension.py, so the docs render tables. The extension is not enabled in the real mkdocs.yml.

About 230 other scratch files from the session were not kept. They were probes for single questions, and their results are in the CHANGELOG and in notes/docstring_pass_issues.md.
