# Open items, 2026-10-06

State: branch claude/daffodil-test-coverage-jkhn2m is pushed. It is ahead of main. CI has not run on it. Version 0.6.0 is untagged and not deployed.

1. Release. Set the date in the CHANGELOG heading, push to main, check all 5 CI jobs, then `git push origin main:full_deploy`. The owner sets PYPI_API_TOKEN in the pypi environment.
2. Push the branch to main. It carries the to_md alignment fix, 127 regenerated doctest examples and the removal of NORMALIZE_WHITESPACE. Check CI after the push.
3. Docs rendering of tables. Decide whether to enable the griffe extension in notes/griffe_blankline_extension.py. It needs one `extensions:` line in mkdocs.yml.
4. Doctest tooling. Stock doctest stays. Sybil was not evaluated properly. xdoctest ignores output after a blank line, so it is not a fix.
5. AuditEngine. Send notes/auditengine_performance_prompt.md, if wanted. Decisions a to d are in notes/auditengine_action_items.md. Exact repr() and to_md() text changes by a space in one-character columns.
6. mypy baseline is 11 errors. 3.10 and 3.13 doctests were last run before NORMALIZE_WHITESPACE was removed.
