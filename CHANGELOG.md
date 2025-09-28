# Daffodil Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),  
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Adoption of this format started in v0.5.10. Prior notes included in fixed section for
all prior releases. Plans for future moved to ROADMAP.md.

## [Unreleased]
### Added
- (add entries here)

### Changed
- (add entries here)

### Fixed
- (add entries here)

---

## [0.5.11] - pending
_No published tag yet._

### Added
- (pending notes)

---
## [0.5.10] - 2025-09-28

### Added
- `CHANGELOG.md` added to repo; moved log from `daf.py` to this file and reformatted to md.
- `ROADMAP.md` added to repo; moved TODO plans from `daf.py` to that file and reformatted to md. 
- `manifest.in` added to repo to force inclusion of these documents into the distribution.

### Fixed
- Neglected to bump `pyproject.toml` version in 0.5.9.
- Removed change log and roadmap todos from `daf.py`

---

## [0.5.9] - 2025-09-22
### Added
- `attrs` argument on instance creation (parity with pandas’ `.attrs`).
- `.to_donpa()` to convert selected columns to a dict of NumPy arrays (columnar ops).
- Tests for JSON round-trip including `disp_cols` and `attrs`.

### Changed
- Initialize `disp_cols` robustly: accepts `None`/`list`/`tuple`; emits `[]` if not provided.
- Improved `.to_json()` / `.from_json()` to handle `disp_cols` and `attrs`.

### Fixed
- Introduce `KeysDisabledError` when key lookups occur without `keyfield` set.
  - If `keyfield` **is** set but key missing: raise `KeyError` (when `silent_error=False`).

<pre>
    v0.5.9  (2025-09-22)
            Add 'attrs' as argument to creation of the array instance. This mimics Pandas usage.
            Improve initialization of disp_cols so it can be None, List or tuple and emitted as [] if not provided.
            Added .to_donpa() which would convert specified columns to dict of numpy arrays of each col, where dict keys are col names.
            This creates a simple pandas-like object where NumPy column operators can be easily applied.
            Improved .to_json() and .from_json() to handle disp_cols and attrs. Fixed tests.
            Added use of KeysDisabledError whenever a function attempts to lookup a row when keyfield is not set.
                In contrast, if keyfield is set and key not found, then KeyError if silent_error False.
                Fixed unit tests.
</pre>
---

## [0.5.8] - 2025-07-17
### Fixed
- Flattening now converts textual containers to actual objects (`'[]' → []`, dict, tuple).

### Changed
- Clean up imports to satisfy `pyflakes`.
- Correct SQL index-name creation in `sql_utils` / `daf_sql` (after quoting change).
- Handle empty `self.lol` edge case in `select_irows()`.

<pre>
    v0.5.8  (2025-07-17)
            Correct flattening operation so that strings like '[]' are converted to []. Same for dict and tuple.
            Cleaned up a number of imports due to pyflakes linting
            Fixed creation of index name in sql_utils and daf_sql due to change in name escaping.
            Fixed edge case in select_irows() if self.lol is empty.
            RELEASED AS v0.5.8
            
</pre>

---

## [0.5.7] - 2025-05-16
### Added
- `.replace_in_columns()` to replace values across specified columns.
- `.apply_colwise()` (apply per row, store in `target_col`).
- Options in `.to_pandas_df()`:
  - `use_donpa` path for faster conversion via NumPy vectors.
  - Default/NA handling (''/None → chosen default). Tests added.
- Expanded tests for `.to_list()` default handling.

### Changed
- SQL identifier handling: switch from escaping to quoting. **Breaking** for old SQL tables.
- Adopt `pytest`; remove ad-hoc `sys.path.append()` uses.
- Use `_MISSING` sentinel internally for default handling in `.to_donpa()`, `.to_list()`, `.to_pandas_df()`.

### Fixed
- Various improvements in `from_pdf()` for complex layouts.

<pre>
    v0.5.7  (2025-05-16)
            Change SQL identifier escaping from character escaping to quoting.
                Note, this is a breaking change if prior SQL tables are encountered.
            Move to using pytest. Removed some sys.path.append() statements as a result.
            Improve default handling in to_donpa() to allow None to be the default by using a sentinel _MISSING
            Improved .to_list() default handling to allow None as the default, by using sentinel _MISSING.
            Improved .to_pandas_df() default handling
            Added option use_donpa to .to_pandas_df() to improve convertion to pandas df by first converting columns to numpy vectors.
            Added tests for .to_list() including various options.
            Added .replace_in_columns() to Replace all values in `find_values` with `replacement` for the specified columns.
            Added .apply_colwise() Apply a function to each row and store the result in target_col
            Improved .to_pandas_df() 
                to provide an option to convert first to donpa (dict of numpy array)
                to handle conversion of '' or None to a default such as NA or 0.
                improved tests.
            Improved from_pdf() to support complex conversion scenarios.
            RELEASED AS v0.5.7
            

</pre>
---

## [0.5.6] - 2025-03-04
### Added
- `from_lot()` classmethod.
- `join()` with tests; `tag_other` option to tag other columns (supports chained joins).
- `from_pdf()` to parse multi-page tables.
- `name` parameter for `from_lod()`, `from_csv_buff()`, `join()`, `clone_empty()`.
- `omit_other_cols` in `derive_join_translator()`; simplified translator table.
- `.attrs` dict on the core instance for descriptors/metadata handoff.
- `from_csv()` (local, S3, http/https) with streaming improvements.

### Changed
- Indexing: 25 tests; correct parsing of tuple-of-strings for `krows`/`kcols`.
- `concat()` uses deep copy to avoid shared-frame mutation issues.
- Use context manager for local file closure in `from_csv()`.
- Prefer `daf_utils` (deprecate bare `utils` naming).
- Raw docstring strings in `derive_join_translator` to avoid escape complaints.


<pre>
    v0.5.6  (2025-03-04)
            Added from_lot() class method. Perhaps these can be unified in main init function by examining type of the data passed.
            Added join() method, including unit tests.
            Added from_pdf() class method, used to parse PDF files with table structure across multiple pages.
            Added name argument to from_lod()
            Added name argument to from_csv_buff()
            using raw docstring format to avoid complaints of escape characters in derive_join_translator.
            Added 'tag_other' boolean parameter to tag all other column names during join, to support chained joins.
            Simplified translator_daf table so it is easier to produce by hand and use across many tables being joined.
            Added name argument for join() method, to provide the name of the resulting joined instance.
            Improve unit tests for derive_join_translator
            Added 25 tests for various indexing modes. 
                corrected parsing of tuple of strings for krows and kcols.
            Added name argument for clone_empty() method.
            Added omit_other_cols parameter for 'derive_join_translator' method. 
                this can probably displace the "shared_fields" parameter.
                Fixed omit_other_cols so it could be properly omitted.
            concat (which is called by append if a daf array is appended) was not using deep copy when 
                copying in the frame, and this became a real mess. Added copy with deep to concat.
            Added from_csv() which will load csv to Daf from local files, s3 path or http/s path.
                improved operation of streaming from file to avoid buffer recopying.
            from_csv_buff() still exists for those times when a buffer or file-like object already exists.
            Added .attrs dictionary to core dataframe instance definition to allow for descriptors to be 
                provided to users of the dataframe, esp. between when the daf array is defined and built 
                and when it is used and modified.
            Improve file closure by using context manager in from_csv() for local file usage.
            Deprecated use of utils instead of daf_utils.
            
</pre>

---

## [0.5.5] - 2024-12-27
### Added
- Demo `daf_crm.py`; `op_import()`.
- `get_csv_column_names()`, `precheck_csv_cols()`, `compare_lists()` (refactor from `daf_utils`).
- `ops_daf` for operation runners (docstring-driven).
- `default_type` in `apply_dtypes` for unspecified columns.
- Better CSV preprocessing (commented lines with embedded newlines).
- `__contains__` for key existence (`if key in my_daf:`; requires `kd`).
- Indirect-column support and `sparse_rows` in reductions; revised `daf_sum()`.
- `astype` argument in `.to_list()` / `.to_value()`.
- Standardize on **PYON** instead of JSON for cell serialization.
- SQL benchmarks scaffolding (`daf_sql.py`, `create_index_at_cursor()` in `daf_benchmarks.py`).

### Changed
- Improve README examples and memory benchmarking (`objsize`).
- `set_keyfield` is a no-op on empty frames.
- Revised `.sum_da()` per user feedback.
- `apply_in_place(by="row_klist")` mutates row in place; rename kwarg `keylist` → `rowkeys`.

<pre>
    v0.5.5  (2024-12-27)
            add daf_crm.py as demonstration.
                add op_import() 
            add get_csv_column_names() as refactoring in daf_utils for reading csv.
            precheck_csv_cols()
            compare_lists() -- imported from daf_utils
            Introduce ops_daf for running operations, can also use in audit-engine.
                operation descriptions taken from docstring.
            added 'default_type' to apply_dtypes for any cols not specified in passed dtypes.
            Improved preprocessing of csv file when line is commented out and embedded newlines exist in the line.
            Improved Daf.from_lod() by using columns in dtypes dict if provided instead of relying only on first record of lod.
            
            Added indexing with range and T_lor (list of range) types, for both column and row indexing.
            Added __contains__ method to allow " if key in my_daf: " to test if a given key exists. Requires kd exists.
            revised .sum_da() based on feedback from user group.
            Improve formatting of README.md to include tables of examples.
            improve daf_benchmarks.py to use objsize instead of pympler to evaluate memory use.
            Corrected set_keyfield in daffodil to do nothing if daf is empty.
            Added 'sparse_rows' to reduction 'by' type using an indirect_col.
            Improve daf_sum() to support indirect_col.
            Revised apply_in_place to support by='row_klist'. Func will modify row_klist and that will modify the array.
                Changed name of keyword parameter in apply_in_place() from keylist to rowkeys to avoid confusion.
            added astype parameter for to_list() and to_value()
            Introduced standardization around PYON instead of JSON:
                - Easier to convert esp. during serialization using csv.writer().
                - Compatible with more Python data types.
                - Still easy to convert to JSON.
            Copied function create_index_at_cursor() for sql tables in daf_benchmarks.py
            Added daf_sql.py mainly to support benchmarks at this point.
            This will be the last release before sql enhancements.
            
</pre>

---

## [0.5.4] - 2024-07-02
### Added
- `sort_by_colnames()` (+ `daf_utils.sort_lol_by_cols()`).
- `omit_nulls` in `.to_list()`.
- `annotate_daf()` to join columns from another table via mapping.
- `value_counts_daf()` totals via `.to_list()`.

### Changed
- Internals: `klist.values` → `._values` to avoid property ambiguity.

<pre>
    v0.5.4  (2024-07-02)
            Add sort_by_colnames(self, colnames:T_ls, reverse: bool=False, length_priority: bool=False)
                Add daf_utils.sort_lol_by_cols()
            Add argument 'omit_nulls' to .to_list() method.
            Change references to klist.values to ._values to avoid amiguity with property getter and setters.
            Add annotate_daf(self, other_daf: 'Daf', my_to_other_dict: T_ds) to effectively join two tables.
            Fix value_counts_daf() by adding .to_list for total.

</pre>
---

## [0.5.3] - 2024-06-28
### Added
- `__format__` to enable `{:,...}` etc. (delegates to `.to_value()`).
- Alias `value_counts()` for `valuecounts_for_colname()` (pandas parity).
- `.to_klist()`, `.iter_list()`; extend `.iloc` to `klist` and list rtypes.
- `.assign_col()` can insert new columns.

### Fixed
- `insert_col_in_lol_at_icol()` off-by-one at “append” boundary.
- Create single-col `lol` when needed for empty input.
- `num_cols()` more robust (samples first rows).
- KeyedList supports empty init; rename `values` to `._values`.

### Changed
- More iterable support in row/col selectors.
- `remove_dups()` returns unique and duplicated sets (by `keyfield`).
- Formatting: bare `{daf}` prints summary (requires more than `{daf:}` to trigger formatting).


<pre>
    v0.5.3  (2024-06-28)            
            added tests:
                flatten()
                to_json()       not completely working.
                from_json()     not completely working.
            added __format__ to allow use of {:,} and other f-string formatting. Invokes .to_value()    
            added alias for valuecounts_for_colname() to value_counts() to match Pandas syntax.
            
            extend .iloc to support klist and list rtypes.
            Added .to_klist() to return a record as KeyedList type.
            extended .assign_col() to insert a column if the colname does not exist.
            Enhanced KeyedList() to allow both args to be None, and thus initialize to empty KeyedList.
            insert_col_in_lol_at_icol(): 
                fix bug if icol resolves to add a column. --> '>' changed to '>='
                allow empty lol and create a lol with one column if col_la exists.
            Add .iter_list() to allow iteration over just lol without cols defined.
            fixed __format__ so unadorned daf name prints summary. It takes more than {daf:} in fstring to cause formatting.
            Improved robustness of num_cols() to check first few rows.
                TODO: It will probably be better to keep a value of the num cols and not calculate evertime.
            changed name of values in KeyedList to _values and created accessors.
            added support for Iterables passed for row and col selection.
            Added method "remove_dups()" which returns unique records and duplicated records based on keyfield.
            Changed operation of assign_col to append col to right if colname not exist.
            
            worked around error in Pympler.asizeof.asizeof() function, used in daf_benchmarks.
                this appears to be resolved in future updates of pympler.
                

</pre>
---

## [0.5.2] - 2024-05-30
### Added
- `.iter_dict()` and `.iter_klist()`. Mutating `KeyedList` mutates underlying `lol`.

### Fixed
- Correct `slice_len` for column assignment from another column (clarifies nested list vs scalar intent).

<pre>
    v0.5.2  (2024-05-30)
            Added .iter_dict() and .iter_klist() to force iteration to produce either dicts or KeyedLists.
                Producing KeyedLists means the list is not copied into a dict but can be mutated and the lol will be mutated.
            Correct calculation of slice_len to correct column assignment from another column
                This may still have some ambiguity if a nested list structure is meant to be assigned to an array cell.
                    collist = my_daf[:, 'colname'].to_list()    # this will return a list, but sometimes of only one value.
                    my_daf[:, 'colname2'] = collist             # there is ambiguity here as to whether the list with one
                                                                # item should be placed in the cell or if just the value.
                
</pre>

---

## [0.5.1] - 2024-05-25
### Changed
- Dependency ranges relaxed in `pyproject.toml`.
- Upgrade to Python 3.11 and latest libs; using `venv311`.

<pre>
    v0.5.1  (2024-05-25)
            changed dependencies in pyproject.toml so they would allow newer versions.
            Upgraded to Python 3.11 and upgraded all libraries to the latest.
            Using venv311
    
</pre>
---

## [0.5.0] - 2024-05-23
### Added
- `split_where()` returns `(true_daf, false_daf)`.
- `multi_groupby()`, `reduce_dodaf_to_daf()`, `multi_groupby_reduce()`.
- `KeyedList` data type (dict-like API backed by list).
- `_build_hd()`, `to_json()`, `from_json()`, and same for `KeyedList`.
- `.strip()`, `.num_rows()`, `.to_value(flatten=…)`.

### Changed
- Remove `_da` suffix across many methods (prep for KeyedList).
- Behavior: `.select_record()` returns `{}` when frame is empty.
- Join/tagging, index handling, negative index tests, indirect-col handling in reductions.

<pre>
    v0.5.0  (2024-05-23)
            Added split_where(self, where: Callable) which makes a single pass and splits the daf array in two
                true_daf, false_daf.
            Added to Daffodil multi_groupby(), reduce_dodaf_to_daf() and multi_groupby_reduce()
            Added class KeyedList() to provide a new data item that functions like a dict but is a dex plus list.
                can result in much better performance by not redistributing values in the dict structure.
                This is not yet integrated into daffodil, but should be.
                
            Removed '_da' from many Daffodil methods and for keyword parameters, to allow future upgrade to KeyList.
                select_record_da()      -> select_record()
                record_append()
                _basic_get_record_da    -> _basic_get_record
                assign_record_da()      -> assign_record()
                assign_record_da_irow   -> assign_record_irow
                update_by_keylist()
                update_record_da_irow   -> update_record_irow
            changed test_daf accordingly.
                
            Added _build_hd() to consistently build header dict structure.
            Added to_json() and from_json() methods to allow generation of custom JSONEncoder.
            Changed nomenclature in KeyedList class from dex to hd.
            Added from_json and to_json to KeyedList class to allow custom JSONEncoder to be developed.
            
            select_record() silently returns {} if self is empty.
            mark chunks saved to local_mirror if not use_lambdas
            
            fixed _itermode vs. itermode.
            Added .strip() method.
            correct icols when providing a single str column name, and when column names have more than one character each.
            Added 'flatten' in '.to_list' method which will combine lol to a single list.
            Added .num_rows() which will more robustly calculate the number of rows in edge cases.
            Fix unflattening issue discovered when running edge_test_utils.py.
            Updated documentation to reflect new approach to dtypes and flattening.
             

</pre>

---

## [0.4.4] - 2024-05-02
### Added
- `daf_utils.safe_eval()`.

### Fixed
- Unflattening for f-string-flattened values (tuples & friends).
- `set_dtypes()` bug.

<pre>
    v0.4.4  (2024-05-02)
            Improved unflattening to handle f-string type flattening, including tuples.
            fixed bug in set_dtypes()
            added daf_utils.safe_eval()
            
</pre>
---

## [0.4.3] - 2024-05-02
### Changed
- Back to `pyproject.toml`; drop `setup.py`. Lambdas now import Daffodil.

<pre>
    v0.4.3  (2024-05-02)
            Went back to pyproject.toml approach. Eliminated setup.py.
            Lambdas now importing daffodil

</pre>

---

## [0.4.2] - 2024-05-01
### Packaging
- Attempted `flit` (TOML parse issues), reverted to `setuptools`, reorganized to `daffodil/src`.
- For editable installs, set `PYTHONPATH` to `daffodil/src`.
- Import modules as `daffodil.lib.daf_utils`, etc.

<pre>
    v0.4.2  (2024-05-01)
            Modified packaging for package distribution on PyPI to hopefully make it compatible with installing into AWS Lambdas.
                Tried to use pyproject.toml and flit, but flit has poor toml parsing it seems, and could not find a suitable toml file.
                Went back to setup.py and setuptools, but reorganized files into daffodil/src folder which will be included in the distro.
                To use --editable mode for local development, must set PYTHONPATH to refer to the daffodil/src folder.
                In that folder is daffodil/daf.py and daffodil/lib/daf_(name).py 
                To import supporting py files from lib, must use import daffodil.lib.daf_utils as daf_utils, for example.
                
</pre>

---

## [0.4.1] - 2024-04-30
### Changed
- `apply_dtypes(from_str=…)` (rename from `initially_all_str`).
- `set_dict_dtypes(dtypes=…)` now modifies in place; handles `dtypes={}`.

<pre>
    v0.4.1  (2024-04-30)
            fixed tests to reflect changes to type conversion paradigm.
            Changed apply_dtypes parameter 'initially_all_str' to 'from_str'
            fixed set_dict_dtypes() in the case of dtypes = {}; Changed parameter to 'dtypes' for uniformity.
            set_dict_dtypes() now also modifies types in-place.
            
</pre>

---

## [0.4.0] - 2024-04-30
### Added
- Better dtype support: `apply_dtypes()`, `flatten()`, `.copy(deep)`.
- `convert_type_value()`, `unflatten_val()`, `json_decode()`, `validate_json_with_error_details`, `safe_convert_json_to_obj`.
- Benchmarks: disable GC during timing; add references; switch to `pyproject`.

### Deprecated
- `unflatten_cols()` → `apply_dtypes()`
- `unflatten_by_dtypes()` → `apply_dtypes()`
- `flatten_cols()` → `flatten()`
- `flatten_by_dtypes()` → `flatten()`
- Functions for `hdlol` type (precursor to Daffodil).

<pre>
    v0.4.0  (2024-04-30)
            v0.4.0 Better dtypes support; apply_dtypes(), flatten(), copy()
            added disabling of garbage collection during timing, getting more consistent results, but does not explain anomaly.
            Improved philosophy of apply_dtypes() and flatten()
                Upon loading of csv file, set dtypes and then use my_daf.apply_dtypes()
                Before writing, use my_daf.flatten() to flatten any list or dict types, if applicable.
                
            apply_dtypes() now handles the entire array, and will skip str entries if initially_all_str is True.
                
                unflatten_cols()        DEPRECATED. use apply_dtypes()
                unflatten_by_dtypes()   DEPRECATED. use apply_dtypes()
                flatten_cols()          DEPRECATED. use flatten()
                flatten_by_dtypes()     Renamed:    use flatten()
                
            added optional dtypes parameter in apply_dtypes() which will be used to initialize dtypes in daf object and
                use it to convert types within the array.
            Changed from la type to interable in reduce()
            added disabling of garbage collection in daf_benchmarks.py
            deprecated functions dealing with hdlol type which was a precursor to daf.
            added convert_type_value() to convert a single value to a desired type and unflatten if enabled.
            removed use of set_dict_dtypes from apply_dtypes() and instead it is done on the entire daf array for efficiency.
            added in daf_utils.py unflatten_val(), json_decode(), validate_json_with_error_details, and safe_convert_json_to_obj.
            Added .copy(deep:bool) method to match pandas syntax.
            Added reference to 1994 workshop in flatten() method docstr.
            Changed packaging code from setup.py approach to pyproject, but still not able to correctly import in Lambdas container.

</pre>

---

## [0.3.0] - 2024-04-14
### Added
- `fmt` parameter (Content-Type metadata when saving).
- `CODE_OF_CONDUCT.md`.

### Changed
- `__init__.py` import helpers (later removed).
- Performance: `daf_sum()`, `reduce()`, `sum_da()` via try/except.

<pre>
    v0.3.0  (2024-04-14) 
            Added fmt parameter so file are saved with proper Content-Type metadata.
            Added 'from .Daf import Daf' to __init__.py to reduce level.  Eventually removed this.
            Added CODE_OF_CONDUCT.md
            Improved performance of daf_sum(), reduce() and sum_da() by avoiding explicit comparisons and leveraging try/except.
            Improved daf_benchmarks.py to help to diagnose nonperformant design of sum_da().
            Added basic indexed manipulation to daf_demo.py
            Changes due to the suggestions by Trey Hunner.
</pre>

---

## [0.2.2] - 2024-03-?? 
### Changed
- Repo layout aligned with Python norms:
  - `from daffodil.daf import Daf`; `daf.py` at top-level; helpers in `lib`.
- Added: `narrow_to_wide()` and `wide_to_narrow()`.

<pre>
    v0.2.2  Changed the file structure to be incompliance with Python traditions.
                user can use 'from daffodil.daf import Daf' and then Daf()
            Moved daf.py containing class Daf to the top level.
            put supporting functions in lib.
            added narrow_to_wide and wide_to_narrow methods.
</pre>

---

## [0.2.0] - 2024-02-28
### Added
- Multi-column `groupby`.
- `omit_nulls` in `col()/col_to_la()/icol_to_la()/valuecounts_for_colname()`.
- `groupby_cols_reduce()`, `sum_np()`, `to_cols_dol()`, `buff_to_file()`.
- Numerous tests (init, `set_cols()`, `set_keyfield()`, selection, calc, pandas I/O, Excel, etc.).

### Changed
- Indexing/refactors for `__getitem__/__setitem__`; bug fixes; Windows/mac/Linux demo; `num_cols()`; name Pydf → Daffodil.
- Removed several legacy helpers and key-mutating variants (see commits).

<pre>
    v0.2.0  (2024-02-28) 
            Copied related code to Pydf repo and resolved all imports. All tests running.
            Added option of appending a plain list to daf instance using .append()
            Added 'omit_nulls' option to col(), col_to_la(), icol_to_la(), valuecounts_for_colname()
            Added ability to groupby multiple cols
            in select_records_daf(self, keys_ls: T_ls, inverse:bool=False), added inverse boolean.
            Started to add selcols for zero-copy support.
            Added _num_cols() 
            Added unit tests.
            Add groupby_cols_reduce() and sum_np()
            Fixed bug in item setter for str row.
            Added demo of making list of file in a folder.
            groupby_cols_reduce() added, unit tests added. Demo added.
            Fix demo to run on windows, mac or linux.
            Add produced test files to gitignore.
            changed _num_cols() to num_cols()
            removed selcols_ls from class. 
            Pulled in from_csv_file()
            Added buff_to_file()
            improved is_d1_in_d2() by using idiom in Python 3.
            moved sanitize_cols to set_cols() method.
                1. read with no headers.
                2. pull off first row using indexing (could add pop_row())
                3. set set_cols with sanitize_cols=True.
            Removed:
                unflatten_dirname() <-- remove? (YES)
                flatten_dirname() <-- remove?
                cols_to_strbool() <-- remove?
                insert_icol()
                    keyfield <-- remove!  use set_keyfield instead
                insert_col() (removed keyfield)
                from_selected_cols <-- remove! but check for usage! (use my_daf[:, colnames_ls]
           Refactor get_item and set_item
                started this but complexity not that much better.
                Redid it again after discussion with Jeremy which was helpful.
            
            
            tests added
                initialization from dtypes and no cols.
                set_cols()
                set_keyfield()
                daf_utils.is_d1_in_d2()
                improved set_cols() to test sanitize_cols flag.
                .len(), num_cols(), shape(), len()
                row_idx_of()   (REMOVE!)
                remove_key -- keyfield not set. (remove)
                get_existing_keys
                select_record -- row_idx >= len(self.lol)
                _basic_get_record -- no hd, include_cols
                select_first_row_by_dict    
                select_where_idxs
                select_cols
                calc_cols
                    no hd defined.
                    include_cols > 10
                    exclude_cols > 10
                    exclude_types
                insert_col()
                    colname already exists.
                from_lod() with empty records_lod
                to_cols_dol()
                set_lol()
                from_pandas_df()
                to_pandas_df()
                from_dod()
                to_dod()
                from_excel_buff()
                
            Added:
                to_cols_dol()
            Moved code for __getitem__ and __setitem__ to 'indexing' and 
                used method equating to introduce them in the class.
            Moved to_pandas and from_pandas to file daf_pandas.py
            Changed constructors from_... from staticmethods to classmethods
            move toward deprecating remove_key() remove_keys()
            add silent_error in gkeys_to_idxs and trigger error if not selected.
            Handle missing keyfield, hd, kd, in inverse mode.
            Added to_value(), to_list(), to_dict() and tests.
            Tested negative indexes in []
            retmode attribute and constructor parameter to set 'val' vs. 'obj' return value.
            moved __getitem__ and __setitem__ back to main class now that they are vastely reduced in complexity.
            Name change from Pydf to Daffodil and resolve issues.

</pre>

---

## [0.1.x] - 2024-02-??
### Added
- Begin separate package; move comment text to `README.md`.
- `apply_formulas()` adds `$r`, `$c`, `$d` references; row access `$d[$r, :$c]` returns list (use `select_irow()` for dict).

<pre>
    v0.1.X 
            Started creating separate package, moved comment text to README.md
            For apply_formulas(), added relative row and column references $r and $c plus $d to reference the daf object.
            Changed the result of a row access, such as $d[$r, :$c] to return a list so it could be compatible with sum.
                use daf.select_irow() to select a row with dict as the result.
</pre>