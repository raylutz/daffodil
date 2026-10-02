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
