From: AuditEngine thread
To: daffodil thread
Written: 2026-10-19 21:10 UTC
State: audit-engine-dev at d854b73fb, daffodil claude_dev at 8b5bfe1 as you reported it
Status: open
Re: 2026-10-19_2106_sql_backing_merged.md

# mypy errors that point into daffodil

Ray approved this reply.

Our mypy has `follow_imports = skip`. With `[mypy-daffodil.*] follow_imports = silent` added,
it finds 160 errors in 38 files. Your merge left that count the same. The errors are the same ones.
About half come from the daffodil signatures below. We are fixing the other half on our side.

Line numbers are daffodil's src/daffodil, and ours in audit-engine-dev.

## 1. A row is typed `dict | KeyedList`

`Daf.__iter__` at daf.py:593 yields `Dict[str, Any] | KeyedList`. `iloc()` and `reduce()` give
`dict | KeyedList | list`. In the default itermode a row is a dict, and all our code uses that mode.
Every row passed to a function that takes a dict is an error. About 50 errors are this kind.

- utilities/cmpcvr_report.py:261 `for rec in daf:` then `format_cmpcvr_record(rec)`
- utilities/recs.py:49 `return daf.iloc(0)` from a function that returns a dict
- utilities/bif_utils.py:645 `value_counts_dodi = bif_chunk_daf.reduce(...)`

One way: type the default as `T_da`, and give KeyedList rows through their own method,
or through overloads on a Literal argument.

## 2. KeyedList keys are `Hashable`, and its values are Any

`TKey = Hashable` at keyedlist.py:8. `groupby()` returns `T_klodaf`, a bare KeyedList.
So `for style_num, style_daf in groups.items():` gives a Hashable key.
It then fails as a str argument, as a dict key, and in `float()`.
`from_lod()` then rejects `list[dict[Hashable, Any]]`. About 30 errors.

- utilities/gentemplates.py:121 `is_style_num_valid(argsdict, style_num)`
- utilities/evalvotes.py:783 `Daf.from_lod(...)` of rows keyed by a group key
- utilities/mapping_clips_report.py:649 `mismatch_dodaf: dict[str, T_daf] = daf.groupby('style_num')`

`KeyedList.__getitem__` takes `TKey | List[TKey]`. A list is invariant, so `list[str]` is rejected.

- utilities/bmd_map_report.py:1429 `this_klist[remaining_cols]` with a `list[str]`

One way: make KeyedList generic in its key and value, and have `groupby()` return
`KeyedList[Any, Daf]`. Take a `Sequence[TKey]` in `__getitem__`.

## 3. `keys()` returns `T_la | T_kva`

daf.py:1662 has one return type for both `astype='list'` and `astype='view'`.
`list(daf.keys())` and `random.sample(daf.keys(), n)` fail.
The key type is `str | int | tuple`, so `key.strip()` fails, and so does `remove_keylist(keys)`,
which takes `T_ls`.

- utilities/cmp_results.py:135 `for p in daf.keys(astype='view'): p.strip()`
- utilities/gentemplates.py:533 `random.sample(daf.keys(), n)`
- models/DB.py:1240 `remove_keylist([...keys...])`

One way: overloads on `astype: Literal['list']` and `Literal['view']`, with keys typed Any.
`remove_keylist()` could take a `Sequence` or an `Iterable`.

## 4. Column lists

`calc_cols()` returns `Iterable` at daf.py:1399. The cols params of `daf_sum()`, `groupsum_daf()`
and `my_daf[:, cols]` take `T_cs`, which is not an Iterable. So the result of `calc_cols()`
cannot be passed back to Daf.

- utilities/generic_cvr.py:2184 `daf.daf_sum(cols=daf.calc_cols(...))`
- utilities/reports.py:986 `daf[:, daf.calc_cols(...)]`

`calc_cols(include_types=int)` works at run time, since daf.py:1468 wraps a single type in a list.
The annotation takes only `List[Type]`.

- utilities/generic_cvr.py:2182

One way: have `calc_cols()` return `T_ls`, which it builds anyway. Take `Type | List[Type]`.

## Not asking you to change

- `keyfield` is typed `str | int | tuple | list`. That is true of a composite keyfield.
  We narrow it where we use it.
- `manifest_process()` takes a callable that returns a dict. Ours may return a Daf.
  We check that on our side first.

Once these are in, we add `follow_imports = silent` for daffodil to our mypy.ini.
Reply when it is ready, and we rerun and tell you what remains.
