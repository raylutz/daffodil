# For the mail to AuditEngine at the next release

Items to include when the next version is released. Started 2026-10-11. Clear this list once the
mail is sent.

- Install `daffodil[pyjson5]` in the Lambdas, or add `pyjson5` to the requirements. Daffodil then
  reads the text of list and dict cells, such as indirect columns in CSV files, with pyjson5. An
  indirect-column load of 10,000 rows took 0.211 s instead of 1.723 s. AuditEngine's
  `rays_requirements.txt` lists `json5==0.9.5`. That is a different package, pure Python, and
  daffodil does not use it. Without pyjson5, daffodil works as before.
- `import daffodil` no longer needs pandas or numpy, and takes 0.09 s instead of 1.08 s. That
  shortens a Lambda cold start.
- `reduce()` gives each row as a KeyedList, not a dict. `sum_kl()` and `count_values_kl()` are the
  standard reducers, and `reduce()` uses them in place of `sum_da()` and `count_values_da()`. A
  function of AuditEngine's own that is passed to `reduce()` and needs a real dict should call
  `dict(row)`.
- `count_values_da()` raises TypeError on a list cell, naming the column. Leave such a column out
  with `cols`.
- One-cell reads and writes by key are much faster: `daf[key, col] += x` went from 6.8 to 0.86 us.

The CHANGELOG section for the release has the full list.
