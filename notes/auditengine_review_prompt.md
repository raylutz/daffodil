# Prompt for the AuditEngine thread: impact of pending Daffodil changes

Task: a read only impact review. Do not edit any AuditEngine code. Report findings only.

Daffodil is the table library that AuditEngine uses. Many behaviors changed on the branch
claude/daffodil-test-coverage-jkhn2m of raylutz/daffodil, and a few more wait for your answer.
Search all AuditEngine source, tests, scripts and notebooks. For each item below, find the calls
that depend on the old behavior. Quote file:line and the call. Say "no impact" when the result is
only read, or when nothing matches. Do not guess. Run the code or search the source.

## Part A. Changes that are NOT made yet. They wait for your review.

A1. select_irows([], inverse=True), also spelled invert=True, would share the row lists of the original.
    Today it makes a deep copy, which takes 3 s for 200,000 rows by 50 columns.
    Look for select_irows( with an empty selection and inverse or invert, and for remove_dups( and any
    code that drops rows by position. Report code that later edits cells of the result and then uses the original.
A2. select_by_dict() would share the rows of the original. Today it builds new row lists.
    Look for select_by_dict( and select_first_row_by_dict(. Report code that edits cells, appends rows or
    sorts the result in place, and then uses the original table.
A3. select_records_daf([], inverse=True) would get its own row list. Today it returns a Daf that uses the row
    list of the original itself, so appending to one appends to the other.
    Look for select_records_daf( with a list of keys that can be empty.

## Part B. Changes that ARE made. Look for code that depends on the old behavior.

### Most likely to break calls

B1. Positions. iloc(), irow(), to_klist(), icol(), icol_to_la(): a negative position now counts from the end,
    and a position out of range raises IndexError. Before, both gave {} or []. A Daf with no rows still gives
    an empty result. Look for iloc( irow( to_klist( icol( with a negative or computed position, and for code
    that tests for the empty answer to find the end of a table.
B2. Row positions. assign_record_irow() and insert_irow() now take irow=None, the default, which adds at the
    end. In assign_record_irow() a negative position counts from the end, so assign_record_irow(-1, rec) and
    my_daf[-1] = {...} now REPLACE the last row. Before they added a row. This is a silent change.
    update_record_irow(-1) now updates the last row, and a position out of range raises IndexError.
    Look for assign_record_irow( with -1, my_daf[-1] = with a dict, and update_record_irow(.
B3. join() fills a missing match with NULL, the empty string, and not None. Pass fill=None for the old result.
    join_records() has the same fill. Look for join( with how left, right or outer, and for code that tests
    "is None" on a joined cell.
B4. A Daf with rows and no column names. from_lot() without cols now gives no names. It named them col_0,
    col_1. to_lod(), iter_dict(), iter_klist(), for row in daf, to_cols_dol(), select_where(),
    select_by_dict(), iloc(), irow(), to_klist() and to_dict() now raise KeysDisabledError for such a Daf.
    Before, some gave [{}, {}] or made up the names A, B, C. iter_list() and to_md() still work. Look for
    from_lot(, Daf(lol=...) with no cols, from_numpy( with no cols, and the dict views of those tables.
B5. select_cols() keeps the order of cols, and a name that is not a column raises KeyError. It used the
    order of the Daf and ignored unknown names. exclude_cols is unchanged. Look for select_cols(.
B6. append() and extend(). A positional list longer than the number of columns raises ValueError. It was cut
    without a message. A type that is not a dict, KeyedList, list or Daf raises TypeError, not RuntimeError.
    New keywords: append(lol=...) several rows, append(la=...) one row, extend(lol=...). Look for append( with
    a list, a tuple or a list of lists, and for "except RuntimeError" around append.
B7. Converting text. For bool, the words false, no, n, f, off, true, yes, y, t, on, in lower case,
    capitalized and upper case, are recognized. Other text is KEPT as text. Before, only six values gave 0
    and everything else gave 1, so "no" was 1. For int and float, text that cannot be converted is kept, and
    a whole number text above 2 to the power 53 keeps every digit. Look for apply_dtypes(, dtypes= with bool,
    and code that works around any of this.
B8. Errors. col(), col_to_la() and to_donpa() raise ColumnNotFoundError for a missing column. It is both a
    KeyError and a RuntimeError, so old handlers still work. select_record(key, silent_error=False) raises
    KeyError(key). select_by_dict(expectmax=) raises LookupError with a message. to_dod() on a Daf with rows
    and no keyfield raises KeysDisabledError, not KeyError. Look for "except RuntimeError", "except KeyError"
    and "except LookupError" around those calls, and for code that reads the old message text.
B9. Column names. set_cols() raises AttributeError for MORE names than columns, as it did for fewer. A blank
    column name from parsing a header, in the constructor and in from_md(), is always Unnamed plus its
    position, and a lone blank no longer stays an empty string. set_cols() keeps the prefix col for a blank
    name, as in col1. Look for set_cols( with a list that may be longer or have blanks, for reading names
    that start with Unnamed or col, and for profile_ls_to_lr(. It is unchanged.
B10. In place methods now return the Daf and not None: assign_record, assign_record_irow, update_record_irow,
    assign_icol, set_icol_irows, find_replace, apply_to_col, apply_in_place, apply_formulas and
    set_col2_from_col1_using_regex_select. A Daf with no rows is falsy. Look for code that uses the result in
    a condition, such as "if d.find_replace(".
B11. transpose() without include_header now has the cols A, B, C, one for each source row. Before they were
    key, A, B, with one name too many. With include_header=True they are still key, A, B. Look for
    transpose( and for code that reads the old names.
B12. sum_np() counts a blank, None or NaN cell as 0. It raised. A text column raises TypeError that names the
    column. Look for sum_np( and code that catches the old error to skip a column.
B13. sort_by_colname() and sort_by_colnames() raise TypeError that names the column when values cannot be
    compared. New option as_str=True. Look for sort_by_colname( and sort_by_colnames(.
B14. Reading files. from_csv_file() now calls from_csv(). It reads UTF-8, and a file that cannot be read
    raises RuntimeError. It printed a message and returned None, and used the encoding of the machine.
    from_googlesheet() and to_googlesheet() now need service_account_file and raise NotImplementedError.
    from_directory() no longer prints, and has include_dirs. Look for from_csv_file(, from_googlesheet(,
    to_googlesheet(, from_directory(, and code that tests the result for None.
B15. Copying. copy() has levels: shallow (the default, unchanged), sortable, editable, deep. copy(for_sorting=True)
    also copies hd and dtypes now. Look for .copy( calls whose copy is edited while the original is used.
B16. remove_key() and remove_keylist() are only marked deprecated, and behave as before. A tuple as long as
    a composite keyfield is now one key in remove_key().

### Less likely to break calls

B17. to_json() no longer changes dtypes to {}, and from_json() reads the dtypes list and dict as types.
B18. to_md() and daf_to_lol_summary(): a limit of 0 means no limit. An odd max_rows shows one more row. A Daf
    with no names has spreadsheet names in the to_md() header, as before. Look for text that compares output.
B19. An explicit keyfield now wins over the schema keyfield, as in Daf(schema=..., keyfield=...).
B20. from_pandas_df() keeps name with use_csv=True, and the dtypes of a Series are keyed by its labels. The
    dtypes argument is deprecated and gives a DeprecationWarning.
B21. apply_formulas() restores retmode and invalidates the key index on every exit, also after an error.
B22. groupby() and groupby_cols() take cols. multi_groupby(colnames=) is used now. groupby_reduce(),
    groupby_cols_reduce() and multi_groupby_reduce() group only the reduce_cols, and the indirect_col, for by
    row and sparse_row. A function that reads another column will not find it.
B23. value_counts_daf(), dict_to_md() and from_lod_to_cols() keep a blank key as data.
B24. select_irows() takes inverse. The old name invert still works.

### Changes of 2026-10-02, in case they were not reviewed

B25. to_donpa(default=...) replaces blank, None and NaN cells as it reads a column. include_cols works in
    from_csv_buff(), from_csv_file() and from_csv(). apply_in_place(by='row') stores a returned dict by column
    name, and no longer by position. apply_dtypes() is faster. apply_to_col() passes keyword arguments to the
    function. drop_cols() clears a keyfield that was dropped. remove_dups() no longer changes the keyfield of
    the Daf. narrow_to_wide() handles ids that are not next to each other. Assigning a str to several rows
    or to a column no longer corrupts the table. Column slices with negative numbers work. from_lod() raises
    ValueError for a key that only a later dict has, unless cols= is given. from_cols_dol() uses the longest list. Methods that added a value without
    a column name now add the name.

## Output format

Start with a short numbered list of findings, one line each. Then give the detail for one finding at a
time. For each finding give a list of options, with "Original" as one option, and say which you recommend.
Use markdown tables for Daf output. Write short sentences. Do not use dashes as punctuation. Keep each
finding short enough to read on a phone. If nothing matches an item, say so in one line.
