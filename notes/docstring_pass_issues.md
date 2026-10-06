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
    Fixed with approval on 2026-10-04, option B. It calls `from_csv()`. It reads UTF-8, and a file that cannot be
    read raises `RuntimeError`. It is still deprecated.
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
    Fixed with approval on 2026-10-03, option D. `from_lot()` makes no names without `cols`, as
    `Daf(lol=...)` does. A Daf with rows and no names raises `KeysDisabledError` from `to_lod()` and from the
    dict and KeyedList iterators, and so from `to_cols_dol()`, `select_where()` and `select_by_dict()`.
    Then the same rule on 2026-10-03: `iloc()` with `rtype` of `dict` or `klist`, and so `irow()`, `to_klist()`
    and `to_dict()`, raise for a Daf with rows and no names. `to_md()` still writes spreadsheet names in the header, which
    Markdown needs, and `from_md()` needs it too. The names are not stored in the Daf.
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
    Resolved with approval on 2026-10-04. Both methods take a required `service_account_file` argument and raise
    `NotImplementedError`. The draft code, with the placeholder path, is in the commit e9e69fd. The script
    tests/daf_googlesheets_demo.py is not a test and uses the old `Pydf` name.
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
    Fixed with approval on 2026-10-04, option B. `name` is kept with `use_csv=True`, and the dtypes of a Series are
    keyed by its labels. The `dtypes` argument is deprecated and gives a `DeprecationWarning`. It will be removed.
34. `buff_to_file()` and several `from_*` methods had no `Returns` text.

## Group 4: appending and removing rows

35. `append()` takes a list of lists as one row whose cells are lists. It does
    not take it as several rows. `append([[3,'c'],[4,'d']])` adds the row
    `[[3,'c'],[4,'d']]`. Only a list of dicts is read as several rows.
    Fixed with approval on 2026-10-03. `append(lol=...)` adds several rows. `append(la=...)` adds one row
    and does not read its items as rows. `extend(lol=...)` adds several rows.
36. `append(list)` drops values beyond the columns and pads short lists with
    NULL, with no message.
    Fixed with approval on 2026-10-03. A list with more values than the columns raises `ValueError`.
    A short list is still padded with NULL.
37. `append()` defaults to `respect_kd=False`, and `record_append()` defaults to
    `respect_kd=True`. The same word has the opposite default in the two methods.
    Resolved with approval on 2026-10-03, by documentation only. The defaults stay as they are.
    The Args lines say which default each method has.
38. `append()` has a branch for a `KeyedList` that is never reached, because an
    earlier branch already takes `dict` and `KeyedList` together.
    Fixed on 2026-10-03. The branch that could not run was deleted. An unsupported type raises `TypeError`.
39. `remove_key()` and `remove_keylist()` do not remove anything. They return a
    new Daf, and the names suggest an in place change. The new Daf shares the
    surviving rows with the original.
    Resolved on 2026-10-04, option C. Both are marked deprecated in their docstrings, and their behavior is
    unchanged. They are `select_krows(..., inverse=True)`. No method that removes rows by key in place was added.
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
    Documented on 2026-10-03. The README now says assignment replaces the whole row, and points to
    `update_record_irow()` and `my_daf[irow, colname] = value` for changing some cells.
45. Fixed with approval on 2026-10-02. `remove_dups()` with no argument clears the keyfield, and then returns every
    row as a duplicate. With an argument it sets the keyfield of the Daf as a
    side effect. It keeps the last row of each key.
46. Fixed with approval on 2026-10-02. `drop_cols()` of the keyfield column left `keyfield` set to a column that
    no longer exists.
47. `select_cols()` returns the columns in the order of the Daf, not in the order
    asked for, unlike `select_kcols()` and `d[:, [...]]`. Names that are not
    columns give rows with no columns.
    Fixed with approval on 2026-10-03, option D. It keeps the order given, and a name that is not a
    column raises `KeyError`. It was also very slow for wide tables, and is now much faster.
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
    Then, on 2026-10-04, `update_record_irow()` raises `IndexError` for a position that is out of range, with approval.
    An update is a mutation, so a bad position should be seen.
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
    Fixed with approval on 2026-10-04, option C. A column that cannot be compared raises `TypeError` that names
    the method and the column. `as_str=True` sorts by the text of the values, and None sorts as an empty cell. With
    `length_priority` it sorts whole numbers in numeric order. `length_priority` on real numbers failed with `object
    of type 'int' has no len()`, and now the error says to use `as_str`.

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
    already changed stay changed. The key index is now also invalidated after an error, with approval on 2026-10-04.
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
    Fixed with approval on 2026-10-04, option C, for `bool`. The words `false`, `no`, `n`, `f` and `off`, and
    the true words, in lower case, capitalized and upper case, are recognized. Other text is kept. `'inf'` to `int`
    was fixed on 2026-10-02, and keeps the text.
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

## Found on 2026-10-04

78. The constructor renamed a blank column name only when two names collided. A single blank stayed `''`, and
    `set_cols()` used `col1` where the constructor used `Unnamed1`. Fixed with approval on 2026-10-04. Names from
    parsing a header always use `Unnamed` plus the position. `set_cols()` keeps `col`, which is short for printing. `Unnamed` is also the marker that `profile_ls_to_lr()` looks for in a merged heading.
79. `from_lod_to_cols()`, and so `value_counts_daf()` and `dict_to_md()`, took the keys of the dicts from the column
    names of an intermediate Daf. With blank names renamed, a blank key became `Unnamed1`. It is data, so the keys
    are now taken from the dicts. Other places where a value becomes a column name, such as `narrow_to_wide()`, were
    not checked with a blank value.
80. `select_irows([], inverse=True)` made a deep copy of every row, which took about 3 seconds for 200,000 rows by 50
    columns. It now returns a new Daf with a new row list and shared rows, like the other selectors. Approved on
    2026-10-04 after the AuditEngine impact review found no caller that depends on the copy. The review kept `select_by_dict` as it is, because AuditEngine edits those rows in place.
81. `insert_idx_col()` and `insert_col()` wrote into rows that other Dafs shared. After a selection, the insert left the original
    with rows of three values and a header of two columns. AuditEngine calls `insert_idx_col()` on a selection and ignores
    the return value, so returning a new Daf would have broken it silently. Fixed on 2026-10-04 with approval: a Daf
    whose rows have a reference count above its own copies the rows first. Not yet checked: other methods that write into every row, and cell assignment.
82. Decision on 2026-10-04. A selection is a live view for values. Changing values through it, by cell, column,
    `apply_in_place()`, `update_record_irow()` or a KeyedList loop, changes the original on purpose. Dict iteration builds
    new data and cannot. Only changes of shape copy shared rows first: the inserts, and `assign_icol()` when it adds a column.
    A first version of this item copied rows in the value writers too. It was reverted the same day. A prototype of
    a KeyedList that copies a shared row on its first write was tried and dropped for the same reason.
    The aliased row lists were fixed the same day, see item 87.

83. `select_by_dict()` copied its rows. The reporting code in AuditEngine selects with it and then calls `insert_idx_col()`.
    The roles were reversed: the select copied and the insert wrote into shared rows. With item 81 the insert copies
    when it must, so the select can share. Decided by the owner on 2026-10-04. This resolves A2 of the impact review. The README table
    and the `select_by_dict()` docstring now say shared.
84. `select_by_dict()` built a dict or KeyedList for every row to compare one or two fields. It now compares the cells
    by position. Same results on 10,800 comparisons, about 40 times faster. Approved by the owner on 2026-10-04.
    A first draft of the faster loop failed on an empty selector. The comparison found it before it went into the library.
85. KeyedList loops were slow because `KeyedList.__init__` tried the hd plus row case after six other cases, and
    `__getitem__` ran two type tests before the lookup. Both now try the usual case first. The results are
    identical on a probe of 21 inputs. `select_where()` is 30% faster. Approved by the owner on 2026-10-04. A new
    method that tests one column was not added, because `select_by_dict()` is about 40 times faster for that.
86. `select_where_idxs()` iterated the Daf, so its function got a dict in the default itermode. Its docstring said
    KeyedList, as for `select_where()`. It now uses `iter_klist()`. About 3 times faster. Approved by the owner on 2026-10-04.
    AuditEngine uses it once, with a test of `row['so_ind']`, which works with both row types.
87. `select_irows()` with a list of every row in order, `d[:]`, `select_krows()` with every key and with nothing and
    `inverse`, and `select_records_daf([], inverse=True)` returned the same row list object as the original. An append to
    one was an append to the other. Each now returns its own row list, with the rows shared. Two places in `daf.py` changed.
    Approved by the owner on 2026-10-04. No test relied on the alias. AuditEngine should check for a caller that did, see
    `auditengine_action_items.md`.
88. `from_csv_buff()` and `from_csv()` do not check the length of rows. A short row fails later with `IndexError`, and a
    long row loses its extra cell in `iter_dict()` without any sign. The owner decided on 2026-10-04 to keep the original
    behavior and to document it. The docstrings now say so, and point to `is_rectangular()` and `force_rectangular()`. The
    cost of a check was measured at 0.6% to 1.5% of a read. An opt-in `check_rectangular` and a `NotRectangularError` that
    subclasses `ValueError` were considered and not made.
89. `from_md()` raises `RuntimeError` for a table with no header row, which is also not a table that Python-Markdown renders. A
    Daf with no column names is written by `to_md()` with the names `A`, `B` and read back with those names. The owner decided on
    2026-10-04 to keep the behavior and to document it. A `noheader` keyword was considered and not made.
90. `from_csv_buff('')` raised `IndexError: pop from empty list`, and `from_csv()` of an empty file raised `RuntimeError`
    that said the file could not be read. An empty source now gives an empty Daf with no columns, as `from_md('')` and
    `from_csv_buff('', noheader=True)` did. Approved by the owner on 2026-10-04. A caller that wants an error for an empty
    file can test `len()` of the result.
91. A `keyfield` that is not a column is stored without an error by the constructor, the builders and `set_keyfield()`
    unless `silent_error=False` is given. A lookup then said `Key lookups are disabled (no kd)`, and `keys()` and
    `select_record()` gave empty results. The owner decided on 2026-10-04 to keep the rule and to document it, and to make
    the error of `select_krows()` say what is wrong. Still unclear: a Daf with no rows and a valid keyfield also gives
    `Key lookups are disabled (no kd)`. `keys()` and `select_record()` still give empty results for an unknown keyfield.
92. Item 91 left one inconsistency. A Daf with a key index (kd) and no keyfield is supported by the constructor, and
    `krows_to_irows()` and `select_record()` honored it. `select_krows()`, `select_records_daf()`, `remove_key()`,
    `remove_keylist()` and `keys()` did not. The owner decided on 2026-10-04 that they all honor it, and that the
    messages name the method and say what is missing. `assign_record()` needs a keyfield and says so. The `(no kd)` message
    now says that the key index is empty, which happens for a Daf with no rows. Transpose does not carry a kd in this
    version. `transpose()` gives an empty kd and no keyfield. The design intent for that is not recorded here.
93. The docs audit on 2026-10-04 found 15 public methods of `KeyedList` and `KeyedIndex` with no docstring. They were hidden from
    the API page. Each now has a docstring and tested examples. Still in `keyedlist.py`: `astype_la()` has an old style
    docstring and is not on the API page, and the second strings after the docstrings of `KeyedList` and `KeyedIndex` are
    still there. `KeyedListEncoder.default()` and `to_json()` appear unused in daffodil.
94. `KeyedListEncoder` was removed on 2026-10-04. Nothing in daffodil used it, and the owner found no use of it elsewhere. Only its
    2 tests used it. Open: `astype_la()` in `keyedlist.py` duplicates `daf_utils.astype_la()`, but does not keep an empty (NULL)
    cell. `KeyedList.values(int)` raises `ValueError` for an empty cell. The `daf_utils` version keeps it. `daf_utils` cannot be
    imported from `keyedlist.py` without a circular import, because `daf_types` imports `KeyedList`.
95. `astype_la()` in `keyedlist.py` now keeps an empty (NULL) cell, as `daf_utils.astype_la()` does. Approved by the owner on
    2026-10-04. `KeyedList.values(int)` on a row with a missing cell returned an error before. It is still a copy of the `daf_utils`
    function, because of the circular import. A test compares the two. It was renamed to `_astype_la` the same day, with the owner's approval.
96. Docs audit, item 2, on 2026-10-04: 26 `Daf` methods got examples. Not done: `retmode` and `itermode` are flagged only
    because of their setters, whose getters have examples. `from_googlesheet()` and `to_googlesheet()` always raise, so they have no
    Returns. Open for the owner: (a) `manifest_apply()` has the default `by='row'`, but its docstring says `by` must be `table`. With
    the default, a function written for a table fails with `AttributeError: 'dict' object has no attribute 'lol'`. (b) The
    module function `unpack_indirect()` at the end of `daf.py` is not attached to `Daf`. Only a test calls it, as
    `daf_module.unpack_indirect()`. Its docstring example calls `daf.flatten_indirect()`, which does not exist. It is not on the
    API page.
97. `unpack_indirect()` was removed on 2026-10-04, with its 3 tests, because the owner said that the accessor for indirect columns made it
    obsolete. It closes finding (b) of item 96. `daf_doc.txt` still lists it. That file is an old generated listing of `daf.py`.
98. Docs audit, item 3, on 2026-10-04. 47 methods keep a second string after the docstring, not 43. Most repeat the real docstring
    and were left alone. Moved in after a run: `isin`, `set_keyfield`, `to_csv_buff`, `krows_to_irows`, `select_icols`, `append`.
    Found while checking, and not changed:
    (a) `my_daf[:, mask]` with a list of bools reads the bools as positions. `Daf(lol=[[1, 2, 3]], cols=['a', 'b', 'c'])[:, [False, True, False]]`
        gives the columns `['a', 'b', 'a_2']`. The old text of `isin()` showed masks as a way to select columns, with `columns().isin(...)`
        and `~`. Neither works, because `columns()` is a list and has no `isin`, and `~` fails for a list.
    (b) The old strings of `to_list()` and `to_numpy()` describe the parameters `irow`, `icol` and `dtype`, which these methods no longer have.
        The one of `to_list()` says that a table with several rows and columns gives an empty list. It raises `ValueError`.
    (c) The old string of `clone_empty()` says that `attrs` are not carried over. They are deep copied.
    (d) The old string of `from_dod()` says that a Daf is 1/3 the size of a dod. For 5,000 rows of 5 int columns I measured a Daf
        at 0.9 of the dod, with `objsize`. The claim was not moved in.
    (e) The old string of `from_csv_buff()` says that it streams. A test with 300,000 rows gave the same peak memory from text and from
        an iterator of lines, because the table dominates. The claim was not moved in.
99. `Daf.isin()` was deprecated on 2026-10-05, at the request of the owner. It was an early attempt to match the `isin()` of pandas, and
    `select_where()` with a function replaced that use. Daffodil uses it nowhere, only its 2 tests, and the owner says AuditEngine does not use it.
    The deprecation is in the docstring only, as for `remove_key()`. The method works as before. The old second string of `isin()` still shows
    `columns().isin(...)`, which never worked, and is kept. Removal is for later.
100. Requested on 2026-10-05: show a one pass selection against a set in place of a list of bools. `select_where()` now has examples with `in`,
    `not in`, `and`, `or`, a set built from another Daf, and the faster form with `col()` and `select_irows()`. I measured the one pass
    form as slower on a large Daf: 0.109 s against 0.020 s for 200,000 rows and 1,000 values. The docstring says so.
101. Requested on 2026-10-05: the examples must not read the `lol` attribute. All 126 examples that did were changed. First they used `to_lod()`. The owner then
    said that the default `repr()` of a Daf is `to_md()`, limited to 10 rows, and that all example output should show it directly. A Daf result is now the bare
    expression. A flipped result has no column names, and its repr shows the `A`, `B` header, so no special case is needed. A row is read with `iloc()`.
    The examples of `copy()` and `select_records_daf()` showed sharing by comparing the lists. They now append a row to a copy and set a cell. The repr
    starts with a blank line and has a blank line before the size line, so `pytest.ini` got `doctest_optionflags = NORMALIZE_WHITESPACE`. The README had no `.lol`.
    The tests still use it. The rule is not written in CLAUDE.md.
102. WITHDRAWN on 2026-10-05, see item 103. Found on 2026-10-05: a Daf with column names and no rows shows no header. `Daf(cols=['a', 'b'])` prints only
    `%% daf rows=0; cols=0; keyfield=''; name=''`, though `columns()` returns both names. `num_cols()` reads the first row, so it gives 0. An empty result of
    `select_records_daf([])` shows the same. The size line says `cols=0`.
103. The owner said on 2026-10-05 that an empty array has no columns and no rows, so item 102 is the design. `num_cols()` already said that it counts from the rows. The
    docstring of `md_daf_table_snippet()`, which gives `repr()` and `str()`, now says it too, with an example. `columns()` still returns the names.
104. Decided on 2026-10-05: no bool masks. Item 98(a) is closed with no code change. Use `select_where()` or `select_by_dict()` instead.
    `isin()` stays deprecated, and its docstring already warns about bool masks.
105. Decided on 2026-10-05: the three `manifest_*` methods are marked legacy in their docstrings and in the README. No code change.
    The README example of `manifest_apply()` is left as it was, and the note above it says the methods are legacy.
    The example still lacks `load_func` and `save_func`. The AuditEngine check in notes/auditengine_action_items.md decides what happens to them.
106. Decided on 2026-10-05: the `daf_utils` helpers are not part of the public docs. The site keeps showing `Daf` and `KeyedList` only.
    No page is added and no helper docstrings are written for the docs. 34 functions without an underscore have no docstring.
    Open: `get_indirect_da()` and `get_indirect_val()` are still in the source. The owner said on 2026-10-05 that the indirect functions are obsolete.
    The owner said on 2026-10-05 to ignore the open parts: the `get_indirect_*` check, the site structure and the CLAUDE.md line. They are not planned.
107. Removed on 2026-10-05 at the request of the owner: `Daf.isin()` and its 2 tests. The `select_where()` docstring no longer mentions it.
    The README rows that map the `isin()` of pandas to `select_krows()` and `select_where()` stay. The file `src/daffodil/daf_doc.txt` is
    a generated listing and still shows `isin()`. I did not edit it.
108. Design decided on 2026-10-05 for `copy()` and `clone_empty()`. Nothing is implemented yet. The owner chose option 1 in each item.
    (a) `copy()` takes bit flags for what is not shared, and the four level names stay as presets. A copy is always a new object of the same class.
    (b) Without an argument, `copy()` uses a default set on the class. A subclass can set it to play safe. `clone_empty()` is a thin wrapper
        over `copy()` and passes explicit bits, so the class default does not change it. It then keeps the class, the display settings and the schema.
    (c) A copy has no name unless `name=` is given. The deep path follows the same rule.
    (d) The method keeps the name `copy()`. The docs say that it shares by choice. Sharing is a design feature of daffodil, and not a leak.
    Still to define: the name of the class default, the shipped default, and the names of the flags and presets.
109. Implemented on 2026-10-05, the design of item 108. The bits are the constants `COPY_ATTRS`, `COPY_OUTER`, `COPY_HD`, `COPY_DTYPES`, `COPY_KD` and `COPY_ROWS` on `Daf`.
    The owner chose to put `disp_cols` under `COPY_ATTRS`, and a list `keyfield` under `COPY_KD`. The class setting is `copy_level_default`, and its value is `sortable`.
    The owner chose plain constants over an enum, as OpenCV does, and names for the common sums, as pandas does. I first made an `IntFlag` class named `CopyBits`. It was removed
    the same day. `deep` is a name only, because a deep copy cannot be combined with any bit. Details are in CHANGELOG.md.
    The owner said on 2026-10-05 that `dtypes`, `schema` and `disp_cols` are linked to the columns. So `clone_empty(cols=...)` drops all three, and keeps them when `cols` is not given.
    The first version of this change dropped the dtypes of every `groupby()` result, because `_new_group_daf()` passed `cols` even when all the columns were kept.
    No test caught it. It is fixed, and `test_groupby_results_keep_the_dtypes` covers it.
110. Decided on 2026-10-05: the attributes tied to the columns are aligned in one private method, `_align_with_columns()`. The owner chose to keep the dtypes that survive.
    The dtypes keep the entries for columns that exist. With changed columns the keyfield is cleared if a key column is gone, and `schema` and `disp_cols` are dropped.
    The constructor cuts the dtypes to the columns, and does not touch the keyfield. A keyfield that is not a column is still stored, as item 91 says.
    `_new_group_daf()` lost its own cutting code. Same columns means everything is carried over. `set_cols()` is unchanged, because it renames the dtypes by position.
    Open for the owner: the constructor with `hd` and `dtypes` but no `cols` still replaces the given `hd` with the names of the dtypes.
    One choice of mine, for the owner to check. `copy(False)` uses the class setting.
111. Decided on 2026-10-05, replacing the design of item 91. A `keyfield` that is not a column raises `KeyError` in the constructor, the builders and `set_keyfield()`.
    The owner dropped the idea of setting a keyfield before its column exists. A Daf with no column names can still have a keyfield, for the columns that come later.
    The full suite found two places that depended on the old behavior. `join()` built a result whose keyfield was not one of its columns when the left table was empty.
    It now gives that result no keyfield. `test_from_lod_to_cols_empty_lod` passed a keyfield that was not a column, and it now passes one that is.
    Four tests of item 91 were rewritten, and five were added. The lookup message `_keyfield_not_a_column_message()` stays for a direct edit of the attribute, and for a Daf with no names.
112. Decided on 2026-10-05, with item 111. `rename_cols()` and `set_cols()` keep the keyfield. `set_cols()` was cleared on purpose before, and the old tests called that a design decision.
    The owner reversed it. With names already there, `set_cols()` follows the names by position, which is a little more than the option the owner approved.
    That option kept the keyfield only if its name was among the new names. For a table with names `['a', 'b']` and `set_cols(['b', 'a'])` it would have kept a key on the other column.
    The position rule keeps the key on the same data. The two rules agree when the Daf had no names yet, and when the names do not change.
113. Deployment set up on 2026-10-05. The owner chose 0.6.0, a deploy by pushing to the branch `full_deploy`, and a PyPI token as a secret. Nothing is deployed from a push to `main`.
    Checking Python 3.9 found that it has not worked on `main` for some time. Three modules fail at import, and one test fails after that. Four small edits fix it, and with them 2251 tests and 195 doctests pass.
    The owner decided on 2026-10-05 to raise `requires-python` to 3.10 and to apply the four edits as well. 3.9 still passes by hand, with 2251 tests, and is not tested in CI.
114. Decided on 2026-10-05: the legacy wording on `manifest_apply()`, `manifest_reduce()` and `manifest_process()` is removed from their docstrings, the README and the changelog. Item 105 is replaced.
    The review on the EC2 machine found that `manifest_process()` is used on the live ES&S path in AuditEngine, at `ess_cvr.py:257`. The thread did not say whether the other two are used.
    The README example of `manifest_apply()` still lacks `load_func` and `save_func`.
115. Decided on 2026-10-05: `from_lod()` adds a column for a key that first appears in a later dict, where it raised `ValueError`. The error was never released. In 0.5.13 the value was dropped.
    Option 1 of three was chosen: add the columns as they appear and pad the earlier rows once at the end. A prescan was the other way, and it costs an extra pass.
    The time for 200,000 dicts of 10 keys is 0.447 s, the same as the 0.449 s of the version that raised. Eight tests were rewritten or added.
    The owner then chose, also on 2026-10-05, that with `cols` or `dtypes` given a dict with another key raises `ValueError` unless `ignore_extra_keys=True`. The reason: daffodil shares data and is fast,
    which makes it somewhat unsafe, and the owner wants these areas tightened. That path was also made faster, by building each row directly. 0.39 s with the check, against 0.43 s.
    The change closed a data bug in `from_lod_to_cols()`: with `dtypes` that named only later keys, the values were put under the wrong keys.
    `append()` and `extend()` also drop a key that is not a column, and that is documented.
116. Added to the checks for AuditEngine on 2026-10-05: item 17 of notes/auditengine_action_items.md, for the calls of `from_lod()` that give `cols=` or `dtypes=`.
117. On 2026-10-05 the changelog heading `Unreleased` became `[0.6.0] - (not yet released)`, with an empty `Unreleased` above it. The version is not tagged and not deployed.
    Set the release date in the heading when the branch `full_deploy` is pushed. The deploy workflow needs a heading that starts with `## [0.6.0]`.
    The owner pushed the work to `main` on the same day. Pushing to `main` runs only the tests and the docs build, and never deploys.
118. The first CI run on `main`, on 2026-10-05, failed in the doctests of 3.10 and 3.11. The example of `from_directory()` listed two files in the order of the machine where it was written.
    The CI machine listed them the other way. The example now sorts. The docstring says that the order is the order of the file system. The jobs for 3.12 and 3.13 were cancelled
    after about 15 minutes of waiting for a runner, so they had not run. Doctests passed on 3.10, 3.11 and 3.13 on the machine of the session, and the file system order was the only difference.

119. The second review on the EC2 machine, on 2026-10-05, found that daffodil `aabed5a` should not be taken yet, because of four call sites in AuditEngine. The details are in notes/auditengine_action_items.md.
    One of them exposed a gap in daffodil. `from_dod()` and `from_lod_to_cols()` call `from_lod()` with their `dtypes`, so they raise for a key that `dtypes` does not name, and the message says
    to pass `ignore_extra_keys=True`. Neither takes that parameter. The thread suggested it for `dominion_cvr.py:3446`, which calls `from_dod()`, and it would raise `TypeError`.
    The owner chose on 2026-10-06 to add `ignore_extra_keys` to `from_dod()`, and it was added. It is not wanted for `from_lod_to_cols()`, because a dropped key there puts the values under the wrong keys.
120. Saved on 2026-10-06, so that it survives the end of the cloud session: notes/auditengine_changes_prompt.md, the second review prompt in notes/auditengine_test_prompt.md, and the result of the change prompt
    in notes/auditengine_action_items.md. The AuditEngine branch `daffodil-0.6.0-prep` is local on the EC2 machine and is not pushed. Four decisions, a to d, are open for the AuditEngine owner.
    State of daffodil at this point: `main` is at 002f7cb, CI green on 3.10 to 3.13 and the strict docs build. Nothing is tagged or deployed. The changelog heading is `[0.6.0] - (not yet released)`.
    To release: set the date in that heading, push to `main`, check CI, then `git push origin main:full_deploy`. The secret `PYPI_API_TOKEN` is in the environment `pypi`, and `full_deploy` is allowed in the
    deployment branches of `pypi` and `github-pages`.
