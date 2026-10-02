# Issues found during the docstring pass

Collected while rewriting docstrings. No behavior was changed for any of these.
Each item says what runs today. Line numbers are in `src/daffodil/daf.py`.
The date of the first entry is 2026-10-02.

## Group 1: construction, size, copy, columns, keys

1. `copy()` default is shallow (line 821). It shares `lol`, the rows and `hd`.
   Appending a row to the copy changes the original. Adding a name to the
   copy's `hd` changes the original. Output: original had 2 rows, 3 after
   `c.lol.append(...)` on the copy.
2. `set_cols()` accepts more names than columns (line 1121). A 2 column Daf
   given 3 names gets a 3 entry `hd` while the rows still hold 2 values.
3. `set_cols()` raises `AttributeError` for too few names. `ValueError` fits
   the other errors in the class.
4. `set_keyfield()` stores a name that is not a column (line 1313) unless
   `silent_error=False`. The default stores it. With `silent_error=False` it
   raises a bare `KeyError()` with no message.
5. `__contains__` raises `KeyError` for a Daf that has rows but no keyfield
   (line 629). `key in d` is usually expected to return False. The class has
   `KeysDisabledError` for this case.
6. `num_cols()` answers 3 for rows of length 3 with 2 column names (line 667).
   `columns()` answers 2. `shape()` uses `num_cols()`.
7. `keys()` ignores a `kd` passed to the constructor when no keyfield is set
   (line 1186). Its old docstring said it used a separately provided `kd`.
   It returns `[]`.
8. `Daf(hd=..., dtypes=...)` replaces `hd` by the keys of `dtypes` when `cols`
   is not given (line 172). The `hd` argument is lost without a message.
9. The constructor accepts any `retmode` text, such as `'zzz'`, without a
   check (line 172). The `retmode` setter checks its value.
10. The second string of `isin()` (line 925) shows `my_daf.columns().isin(...)`
    and `~` on a list. `isin()` is a static method and `columns()` returns a
    list, so that example does not run. The second string is left as it was.
11. `set_keyfield()` does not check that keys are unique. Duplicate keys give
    a `keys()` list without the repeat and a lookup that finds the last row.

## Group 2: dtypes, schemas, strip, clone_empty, set_lol

12. An explicit `keyfield` passed with `schema=` is lost when the column names
    come from the schema. `Daf(schema=B, keyfield='contest')` ends with the
    schema's `__keyfield__`. The cause is that `attach_schema()` calls
    `set_cols()`, and `set_cols()` clears the keyfield. The README says the
    schema keyfield is used only if none was given. Passing `cols` too keeps
    the explicit keyfield.
13. The README schema example uses a plain class. Since the apply_schema change
    on 2026-10-02, `Daf(schema=PlainClass)` raises `TypeError`. Before, it was
    silently ignored and no columns were defined. The example needs
    `@schemaclass`. The README is not changed yet.
14. `default_record = daf_schema._default_record` appears twice in the class
    body, at `daf.py` lines 1496 and 1881. It is harmless.
15. `apply_dtypes()` turns a value that cannot be converted into NULL without
    any message. `'x'` as an int becomes `''`. This is by design in
    `convert_type_value()`, but a bad value is lost.
16. `set_dtypes()` raises `NotImplementedError` when the Daf has no column
    names. `ValueError` would fit better.
17. `clone_empty()` keeps the keyfield even when `cols` is given and does not
    contain it. It does not keep the name. It returns a `Daf` even from a
    subclass. It has a dead test, `if self is None`.
18. `_safe_tofloat()` has no `@staticmethod` and no `self`. Its docstring says
    it returns the original value on failure, but it returns 0.0.
19. Methods attached from helper modules were missing from the API reference.
    The schema ones are added. The `from_md`, `dodaf_to_md`, `dodaf_from_md`,
    `from_pdf`, `from_pandas_df` and `to_pandas_df` entries come with their groups.

## Group 3: conversions to and from other forms

20. `from_csv_buff(include_cols=...)` and `from_csv_file(include_cols=...)` have
    no effect. The argument reaches `buff_csv_to_lol()` in `daf_utils.py`, which
    never uses it. All columns are read. The old docstrings said it includes
    only the columns given.
21. `from_csv()` docstring said it does not set the keyfield. It does, when
    `keyfield=` is passed, because the keyword arguments go to `from_csv_buff()`.
    The docstring is fixed.
22. `from_csv()` reports any error while parsing a local file as
    `RuntimeError: Failed to read local file`. A column mismatch in the CSV is
    labelled that way too.
23. `from_csv_file()` prints a message and returns None when the file cannot be
    read. It is marked deprecated. It also reads with the default encoding,
    while `from_csv()` uses UTF-8.
24. `from_lod()` takes the columns from the first dict only. A later dict with
    an extra key loses that value without a message. Empty dicts and non dict
    items are skipped without a message, so rows can be lost.
25. `from_cols_dol()` uses the length of the first list. A shorter list raises
    `IndexError`. A longer list loses its extra values without a message.
26. `to_dod()` on a Daf with no keyfield raises a bare `KeyError('')`. The
    message does not say the keyfield is missing.
27. `from_lot()` names columns `col_0`, `col_1`. `set_cols()` and
    `from_googlesheet()` name them `A`, `B`. The two defaults differ.
28. `from_directory()` prints its elapsed time to standard output. It never
    lists folders, so the `is_dir` column is always 0. A schema that leaves out
    standard fields drops those columns.
29. `to_donpa(default=...)` has no effect on NULL cells. In `col_to_la()` the
    `default` is used only with `indirect_col`. `to_pandas_df(use_donpa=True,
    default=...)` passes it on, so it has no effect there either.
30. `from_googlesheet()` had its imports before the docstring, so Python did not
    treat the text as a docstring. It was None. The docstring is now first.
    Both Google Sheet methods use the placeholder path
    `path/to/your/service_account.json`, so they cannot work as shipped.
31. `to_json()` writes dtypes by name, but `from_json()` knows only `int`,
    `float`, `str` and `bool`. A `list` or `dict` dtype comes back as the text
    `'list'`. The round trip loses it.
32. `to_json()` sets `self.dtypes = {}` when dtypes is None. That is a side
    effect of a method that should only read. A NaN is written as `NaN`, which
    is not valid JSON. A tuple cell comes back as a list.
33. `from_pandas_df()` ignores its `dtypes` argument. With `use_csv=True` it also
    loses `name`. For a Series the dtypes dict has the key `col`, not the
    index labels used as column names.
34. `buff_to_file()` and several `from_*` methods had no `Returns` text.
