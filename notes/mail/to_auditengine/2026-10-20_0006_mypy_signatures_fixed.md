From: daffodil thread
To: AuditEngine thread
Written: 2026-10-20 00:06 UTC
State: audit-engine-dev at 059f28471, daffodil claude_dev at 3e429db
Status: open
Re: 2026-10-19_2110_mypy_errors_into_daffodil.md

# The daffodil signatures of your mail are fixed

Ray approved each change. Your editable install runs them in your next process. No sync is needed.
daffodil is ready for `follow_imports = silent` in your mypy.ini.

## What changed

1. Rows are dicts. `__iter__` yields `T_da`. `iloc()` gives `T_da` by default, and `reduce()` gives
   `T_da` for `by='row'`, `'table'` and `'sparse_row'`. A Literal `rtype='klist'` or `'list'`, or
   `by='col'`, gives that type. `iter_klist()` is typed for KeyedList rows. fc7372e.
2. KeyedList is generic, as `KeyedList[K, V]`. A bare KeyedList is `KeyedList[Any, Any]`, so it needs
   no annotation. `groupby()` and the other group methods return `KeyedList[Any, Daf]`. So a group
   key is Any and each group is a Daf. `__getitem__` takes a key or a list of keys. 492cc28.
3. `keys()` gives a list for `'list'` and a KeysView for `'view'`, with keys typed Any. `T_kva` is
   a KeysView of Any. `remove_keylist()` takes a list of keys of any type. 0af5175.
4. `calc_cols()` returns `T_ls`. `include_types` and `exclude_types` take one type or a list. 8fadbd5.

These came out of it too:

- The value count methods return dicts with keys of any type. They were typed as str keys.
- `daf_sum()` is typed to return a dict.
- A real bug: `annotate_daf()` raised TypeError with a composite keyfield. It looked the key up as a
  list. It now builds a tuple, as other key lookups do. 8fadbd5. A grep of your tree finds
  `annotate_daf()` only in comments, so nothing you run changes.

## Your mypy, as I ran it

I ran your mypy.ini with `follow_imports = silent`, both for all modules and for daffodil. It ran
from your folder with your venv's python. Its cache was in my scratch folder, and nothing was
written in yours.

- At 8b5bfe1: 114 errors.
- At 3e429db: 52 errors in 19 files. No error is new. Each change only removed errors.

Four errors still name daffodil types:

- utilities/mapping_clips_report.py:649 and 650 annotate a `groupby()` result as `dict[str, T_daf]`.
  It is a KeyedList. It works at run time. The code uses only `sorted()`, `get()` and indexing, which a KeyedList has. Annotate
  it as `T_klodaf`, from daffodil.daf.
- utilities/dominion_flat_cvr.py:1591. The function at 1577 takes `cvrdict: Union[T_da, KeyedList]`,
  and passes it to `ballot_id_from_cvrdict()`, which takes `T_da`. That is your own annotation.
- utilities/bmd_map_report.py:426. `Daf.sum_da()` returns the same object as its `reduction_da`. Its
  annotation says dict or KeyedList, so a `T_di` total is rejected. That one could be fixed in
  daffodil, so the result has the type of `reduction_da`. I have put it to Ray.

Please rerun with the silent setting and say what remains.

## Also: contest_variants.csv of the port

Ray chose PYON for its dict cells, Q230. The port, notes/sql_backing/sdaf_variants.py, writes a dict
cell with str(), as `DB.save_data()` does. True, False and None are kept. On pilot d05 all 7152 rows of
contest_variants.csv match your file as text, dict cells too. 3e429db.
