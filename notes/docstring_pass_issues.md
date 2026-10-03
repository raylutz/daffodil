# Issues found during the docstring pass

Collected while rewriting docstrings. No behavior was changed for any of these.
Each item says what runs today. Line numbers are in `src/daffodil/daf.py`.
The date of the first entry is 2026-10-02.

## Group 1: construction, size, copy, columns, keys

1. `copy()` default is shallow (line 821). It shares `lol`, the rows and `hd`.
   Appending a row to the copy changes the original. Adding a name to the
   copy's `hd` changes the original. Output: original had 2 rows, 3 after
   `c.lol.append(...)` on the copy.
   Resolved with approval on 2026-10-03. The default stays shallow. `copy(level=)` adds
   `sortable` and `editable`, and the docstring has the table of safe actions.
2. `set_cols()` accepts more names than columns (line 1121). A 2 column Daf
   given 3 names gets a 3 entry `hd` while the rows still hold 2 values.
   Fixed with approval on 2026-10-03. It now raises `AttributeError`.
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
    Fixed with approval on 2026-10-03, in both the schemaclass path and the schema Daf path.
13. The README schema example uses a plain class. Since the apply_schema change
    on 2026-10-02, `Daf(schema=PlainClass)` raises `TypeError`. Before, it was
    silently ignored and no columns were defined. The example needs
    `@schemaclass`. The README is not changed yet.
14. `default_record = daf_schema._default_record` appears twice in the class
    body, at `daf.py` lines 1496 and 1881. It is harmless.
15. Changed with approval on 2026-10-02, the text is kept. `apply_dtypes()` turned a value that cannot be converted into NULL without
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

20. Fixed with approval on 2026-10-03. `from_csv_buff(include_cols=...)` and `from_csv_file(include_cols=...)` had
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
24. Fixed with approval on 2026-10-02 for the extra keys. `from_lod()` takes the columns from the first dict only. A later dict with
    an extra key loses that value without a message. Empty dicts and non dict
    items are skipped without a message, so rows can be lost.
25. Fixed with approval on 2026-10-03. `from_cols_dol()` used the length of the first list. A shorter list raises
    `IndexError`. A longer list loses its extra values without a message.
26. `to_dod()` on a Daf with no keyfield raises a bare `KeyError('')`. The
    message does not say the keyfield is missing.
    Fixed with approval on 2026-10-03. It raises `KeysDisabledError` for a Daf with rows and no keyfield.
27. `from_lot()` names columns `col_0`, `col_1`. `set_cols()` and
    `from_googlesheet()` name them `A`, `B`. The two defaults differ.
28. `from_directory()` prints its elapsed time to standard output. It never
    lists folders, so the `is_dir` column is always 0. A schema that leaves out
    standard fields drops those columns.
    Fixed with approval on 2026-10-03. The print is removed. `include_dirs=True` lists
    folders with `is_dir` of 1. The `ctime` text in the docstring was corrected.
29. Fixed with approval on 2026-10-03. `to_donpa(default=...)` had no effect on NULL cells. In `col_to_la()` the
    `default` is used only with `indirect_col`. `to_pandas_df(use_donpa=True,
    default=...)` passes it on, so it has no effect there either.
30. `from_googlesheet()` had its imports before the docstring, so Python did not
    treat the text as a docstring. It was None. The docstring is now first.
    Both Google Sheet methods use the placeholder path
    `path/to/your/service_account.json`, so they cannot work as shipped.
31. `to_json()` writes dtypes by name, but `from_json()` knows only `int`,
    `float`, `str` and `bool`. A `list` or `dict` dtype comes back as the text
    `'list'`. The round trip loses it.
    Fixed with approval on 2026-10-03. `list` and `dict` are now read back as types.
32. `to_json()` sets `self.dtypes = {}` when dtypes is None. That is a side
    effect of a method that should only read. A NaN is written as `NaN`, which
    is not valid JSON. A tuple cell comes back as a list.
    Fixed with approval on 2026-10-03. The table is no longer changed.
33. `from_pandas_df()` ignores its `dtypes` argument. With `use_csv=True` it also
    loses `name`. For a Series the dtypes dict has the key `col`, not the
    index labels used as column names.
34. `buff_to_file()` and several `from_*` methods had no `Returns` text.

## Group 4: appending and removing rows

35. `append()` takes a list of lists as one row whose cells are lists. It does
    not take it as several rows. `append([[3,'c'],[4,'d']])` adds the row
    `[[3,'c'],[4,'d']]`. Only a list of dicts is read as several rows.
36. `append(list)` drops values beyond the columns and pads short lists with
    NULL, with no message.
37. `append()` defaults to `respect_kd=False`, and `record_append()` defaults to
    `respect_kd=True`. The same word has the opposite default in the two methods.
    Resolved with approval on 2026-10-03, by documentation only. The defaults stay as they are.
    The Args lines say which default each method has.
38. `append()` has a branch for a `KeyedList` that is never reached, because an
    earlier branch already takes `dict` and `KeyedList` together.
39. `remove_key()` and `remove_keylist()` do not remove anything. They return a
    new Daf, and the names suggest an in place change. The new Daf shares the
    surviving rows with the original.
40. `remove_key((1, 'a'))` with a composite key raises `KeyError: 1`, because a
    bare tuple is read as a range. Only `remove_key([(1, 'a')])` works. The
    annotation of `keyval` allows a tuple.
    Fixed with approval on 2026-10-03. A tuple as long as a composite keyfield, with no tuples
    inside it, is now one key.
41. `remove_key(None, silent_error=True)` raises `TypeError`. The flag does not
    cover it.

## Group 5: indexing (serious, found 2026-10-02, fixed with approval the same day)

42. SERIOUS. Column slices with a negative number, or a stop of 0, give wrong
    data or an error. The cause is `select_icols()` at `daf.py` line 4720:
    `range(slice.start or 0, slice.stop or num_cols, slice.step or 1)`.
    Real output for a table with columns id, v and n:

        d[:, -2:]     columns v, n, id, v_3, n_4 (five columns, data repeated)
        d[:, :-1]     no columns
        d[:, 1:-1]    IndexError
        d[:, ::-1]    IndexError
        d[:, 0:0]     all three columns
        d[:, -3:-1]   id, v (correct)

    Row slices are correct, and so is assigning to a column slice. Only reading
    a column slice is wrong. Python's `slice.indices()` gives the right answer
    for every one of these. Eight tests are in `tests/test_daf_select_icols.py`.
    Five of them are marked `xfail(strict=True)` with a reason that starts
    with BUG.

## Group 5 continued: selecting and reading rows, columns and cells

43. SERIOUS, fixed with approval on 2026-10-02. Assigning one value or one list to several whole rows corrupts
    the table. `set_irows_icols()` tests `isinstance(value, (list, Sequence))`,
    and a `str` is a Sequence. Real output for three rows:

        d[[0, 1]] = 'x'     rows 0 and 1 become the bare string 'x'
        d[:] = 'x'          every row becomes the bare string 'x'
        d[[0, 1]] = [7,8,9] rows 0 and 1 are the same list object, so
                            a change to one shows in the other
        d[:, 'v'] = 'x'     only the first row of v is set
        d[:, 'v'] = 'xyz'   the letters x, y, z go to rows 0, 1, 2
        d[[0, 1]] = 5       correct, [[5,5,5],[5,5,5],...]
        d[2] = 'x'          correct, one row

    Tests are in `tests/test_daf_setitem_rows.py`. They passed after the fix, and the markers are gone.
    The `__setitem__` docstring describes the fixed behavior.
44. `d[0] = {'v': 'q'}` sets the other cells of the row to NULL. It does not
    update only `v`. `update_record_irow()` is the method that merges. The
    README says the column names are respected, which does not say this.
45. Fixed with approval on 2026-10-02. `remove_dups()` with no argument clears the keyfield, and then returns every
    row as a duplicate. With an argument it sets the keyfield of the Daf as a
    side effect. It keeps the last row of each key.
46. Fixed with approval on 2026-10-02. `drop_cols()` of the keyfield column left `keyfield` set to a column that
    no longer exists.
47. `select_cols()` returns the columns in the order of the Daf, not in the order
    asked for, unlike `select_kcols()` and `d[:, [...]]`. Names that are not
    columns give rows with no columns.
48. Row sharing differs between selectors. `select_where()`, `split_where()` and
    `select_irows()` share the rows. `select_by_dict()` copies them.
    `select_irows([], invert=True)` makes a deep copy.
    Documented with approval on 2026-10-03: a table in the README, and a test of it. Changes to
    `select_irows([], inverse=True)` and `select_by_dict()` wait for a review of AuditEngine. Also
    found: `select_records_daf([], inverse=True)` returns a Daf that uses this Daf's row list itself.
49. Missing keys raise different errors. `col()` raises `RuntimeError` for a
    missing column. `select_record(silent_error=False)` raises `KeyError()` with
    no key in it. `select_by_dict(expectmax=)` raises `LookupError()` with no
    message. Elsewhere a missing key gives `KeyError` with the key.
    Fixed with approval on 2026-10-03. `col()` raises `ColumnNotFoundError`, a `KeyError` and also a
    `RuntimeError`. `select_record()` and `select_by_dict()` now say what failed.
50. `iloc(-1)` returns `{}` and `icol(-1)` returns `[]`, while `d[-1]` and
    `d[:, -1]` count from the end.
    Fixed with approval on 2026-10-03, option C. A negative position counts from the end. An out of
    range position raises `IndexError`. A Daf with no rows still gives an empty result. Affects
    `iloc()`, `irow()`, `to_klist()`, `icol()` and `icol_to_la()`.
51. The old second string of `to_list()` said that a table with several rows
    and columns gives an empty list. The code raises `ValueError`. The old
    `to_dict()` text named an `include_cols` argument that it does not have.
52. The README says appending a row whose key exists overwrites it. The default
    is `respect_kd=False`, which adds a second row.
    Fixed with approval on 2026-10-03. The README now says `append()` adds a second row, and that
    `record_append()` is the one that replaces by default.
53. The names of the flags differ: `inverse` in `select_krows()` and
    `select_kcols()`, `invert` in `select_irows()`, and `flip` in
    `select_icols()` and `select_kcols()`.
    Corrected on 2026-10-03: `flip` is not an exclusion flag. It turns the selected columns
    into rows. Only `inverse` and `invert` mean the same. Fixed with approval on 2026-10-03.
    `select_irows()` takes `inverse` and still accepts `invert`. The README used `invert`
    for `select_krows()` and `select_kcols()`, which raised `TypeError`. It now uses `inverse`.

## Group 6: assigning, inserting, replacing and sorting

54. Fixed with approval on 2026-10-02. `assign_icol(-1, ...)` added a column of data but not a column name. Each
    row then has one more value than `columns()` has names. `insert_icol()`
    without `colname` does the same. The README and docstrings call these
    ways to add a column.
55. `assign_record_irow()` appends the row when the position is negative or
    beyond the end, and its default position is -1. A caller who passes a bad
    position adds a row without a message. `update_record_irow()` ignores a
    bad position.
    Changed with approval on 2026-10-03, option C. `None` is the default position and adds a row at the
    end. A negative position counts from the end, so `-1` replaces the last row, as `d[-1] = [...]` did.
    `update_record_irow(-1)` now reaches the last row. `insert_irow()` takes `None`, and `-1` still adds at the
    end. A position beyond the last row still adds a row. A position that is out of range for
    `update_record_irow()` still does nothing.
56. `insert_irow(0, 'zz')` raises `UnboundLocalError`, because a row that is
    neither a list nor a dict leaves `row_la` unset.
    Fixed with approval on 2026-10-03. It raises `TypeError` that names the method.
57. `find_replace()` replaces the whole cell when the pattern matches anywhere
    in it. The name suggests a substitution inside the text. It also returns
    None, while the other mutating methods return the Daf.
58. `set_col_irows()` ignores a column name that is not found. `set_icol()`
    raises `IndexError` for a bad column. Several of these methods are marked
    `DEPRECATE?` in their old text.
    Fixed with approval on 2026-10-03. `set_col_irows()` raises `KeyError` for a name that is not a
    column, and is marked deprecated in its docstring. `my_daf[irows, colname] = value` does the same.
    The five comments that pointed at GitHub issue 7 were removed. `update_record_irow()` still ignores names that
    are not columns, and a position out of range.
59. `assign_record()`, `assign_record_irow()`, `update_record_irow()`,
    `assign_icol()` and `set_icol_irows()` return None. The methods that
    do the same kind of change elsewhere return the Daf.
    Fixed with approval on 2026-10-03. All ten in place methods that returned None now return
    the Daf. This includes `find_replace()` from item 57, and `apply_formulas()`,
    `apply_in_place()`, `apply_to_col()` and the regex method.
60. `sort_by_colname()` raises `TypeError` for a column that mixes None and
    numbers. NULL, which is the empty string, mixed with numbers does the same. A
    column of text with NULL sorts the NULL first.

## Group 7: formulas, apply, reduce, grouping, sums, counts, joins, pivots, Markdown

61. Fixed with approval on 2026-10-02. `apply_in_place(by='row')` stored the values of the returned dict by
    position. A dict with the keys in another order puts values in the wrong
    columns. A shorter dict makes a shorter row. A longer dict makes a longer
    row. Real output for columns g, x, y and row `['a', 1, 10]`:

        returns {'y':..,'g':..,'x':..}  row becomes [10, 'a', 1]
        returns {'y': 10}               row becomes [10]
        returns an extra key 'new'      row becomes ['a', 1, 10, 5]

    `apply()` builds its new Daf from the first row returned, so it follows the
    returned dict. `by='row_klist'` avoids the problem.
62. Fixed with approval on 2026-10-02. Three more methods added a value to the rows but not a name to the columns,
    like item 54: `annotate_daf()` with a field that is not a column, and
    `set_col2_from_col1_using_regex_select()` and `apply_replace_regex()` with
    a new `col2`.
63. `apply_formulas()` runs `eval()` on the formula text. The docstring now
    warns about this. After a formula error, it prints the error, raises it,
    and leaves `retmode` as `val`.
    Fixed with approval on 2026-10-03. `retmode` is restored after an error. The cells
    already changed stay changed. The stale key index after an error is not handled.
64. `sum_np()` is described as accepting blanks. A blank, which is `''`, makes
    NumPy raise `TypeError`. `sum()` raises `ValueError` for a text column unless
    the column is left out with `colnames_ls`.
    Fixed with approval on 2026-10-03. Blank, None and NaN cells now count as 0.
65. `join()` fills a missing match with `None`. The README says the same. The
    rule for the rest of the library is NULL, the empty string.
    Fixed with approval on 2026-10-03, option D. A missing match is NULL. `fill=None` gives the old
    result. `join_records()` has the same `fill`.
66. `transpose()` without `include_header` names the columns `key`, `A`, `B`.
    The data has no key column, so there is one name too many, and the names
    are shifted from the data. Passing `new_cols` avoids it.
    Fixed with approval on 2026-10-03. The default cols are now `A`, `B`, `C`.
67. Fixed with approval on 2026-10-02. `narrow_to_wide()` assumed that rows of one id are next to each other. If
    they are not, it loses columns without a message. Its `wide_cols` argument is
    not used. The old `wide_to_narrow()` docstring named `value_cols` and
    `varval_cols`, which are not arguments.
68. `to_md(max_cols=2)` with no `max_rows` adds a row of `...` under the header.
    The cause is in `daf_to_lol_summary()`, which treats `max_rows=0` as a
    limit of zero.
    Fixed with approval on 2026-10-03. A limit of 0 now means no limit. The slice
    `[-0:]` also gave every row for `max_rows=1`. An odd limit now keeps the extra
    row at the start.
69. `from_md()` raises `RuntimeError` for a table with no header and separator
    rows. Its old text said the header is optional. The values come back as text.
70. Code that cannot run: `apply(by='col')` raises before the code under it. The
    argument `colnames` of `multi_groupby()` was not used. Fixed with approval on
    2026-10-03, along with a `cols` argument for `groupby()` and `groupby_cols()`.
    `multi_groupby_reduce()` ignores `by`. `manifest_apply()`
    works only with `by='table'`, because `apply()` returns a Daf for other
    values and the method expects a tuple.
71. The `dtype` and `format` items of `gen_stats_daf()` are not used.
    `valuecounts_for_colname_selectedby_colname()` has no `omit_nulls`, unlike
    `valuecounts_for_colname()`.

## Found while discussing item 6

72. Fixed with approval on 2026-10-02. `apply_to_col(col, func, **kwargs)` passed the keyword arguments to `map()`,
    which takes none. Any keyword argument raises
    `TypeError: map() takes no keyword arguments`. The docstring now says so.
    The likely intent is to pass them to `func`.
73. `convert_type_value()`: `'false'` and `'no'` convert to the `bool` value 1,
    `'inf'` to `int` raises `OverflowError`, and `'1.9'` to `int` gives 1.
74. Fixed with approval on 2026-10-02. SERIOUS for ids. `convert_type_value()` turns a text number into an `int`
    with `int(float(val))`. Above 2 to the power 53 a float cannot hold every
    digit, so the number is silently changed. Real output:

        '9007199254740993'      becomes 9007199254740992
        '12345678901234567890'  becomes 12345678901234567168
        Daf.apply_dtypes() gives the same changed value.

    Python's own `int()` on the text keeps every digit. A test is in
    `tests/test_daf_utils_coverage.py`, marked `xfail(strict=True)`.
75. Sped up with approval on 2026-10-02. `apply_dtypes()` was the slowest way to convert many columns. For 2,000 rows
    and 1,000 columns, all converted to int: `apply_dtypes()` 0.900 s, `apply_to_col()`
    for each column 0.632 s, `apply_in_place()` by row 0.476 s, a plain loop with a
    minimal int conversion 0.198 s. The cause is a call of the general conversion
    function for every cell, with several checks each.

## Side effect of item 61, found on 2026-10-02

76. Fixed with approval on 2026-10-02. After `apply_in_place(by='row')` began writing back by column name, two
    methods that relied on it to add a value stopped storing anything when
    `col2` is a new name. `set_col2_from_col1_using_regex_select('s', 'n')` and
    `apply_replace_regex('s', 't')` now leave the table unchanged. Before, they
    added a value to each row and no column name. Both docstrings now say that
    `col2` must be a column. Item 7 covers what they should do.

## Found on 2026-10-03

77. Row sharing differs between the grouping methods, which extends item 48.
    `groupby()` and `multi_groupby()` copy the rows. `groupby_cols()` and
    `group_where()` share them. The `groupby()` docstring said it shares them. It
    now says it copies.
