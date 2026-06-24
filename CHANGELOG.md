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

## [0.5.13] - (pending)
### Added
- Added pytest test coverage for keyedlist.py (62% -> 100%), daf_md.py (13% -> 99%), and a substantial
   portion of daf_utils.py (39% -> 60%), in new test files: test_keyedlist2.py, test_daf_md.py, test_daf_utils.py.

### Changed
- Daf.to_md() reworked to always pass the column header explicitly rather than relying on the first row
   of the rendered table being treated as a header; daffodil never treats the first row of a lol as an
   implicit header. This also fixes a `header=` override being silently ignored whenever the Daf already
   had its own columns.
- Daf.to_md() now generates spreadsheet-style column names (A, B, C, ...) for headerless/bare lol arrays,
   since Daf.from_md() requires a header + separator row.
- Daf.from_md() now requires a header + separator row; headerless markdown tables are no longer supported
   (previously crashed with UnboundLocalError on headerless input).
- transpose_lol() now validates the transposed shape and raises RuntimeError on ragged-right (non-rectangular)
   input, matching its documented contract. Previously, numpy silently returned the original ragged data
   unchanged (no error, no transpose) since np.array(lol, dtype=object) does not raise for ragged lol.
- json_encode()/NpEncoder no longer pass `default=str` to json.dumps, which was silently bypassing
   NpEncoder's numpy handling (np.int64, np.ndarray) for any type that wasn't already a native float
   subclass (np.float64 only "worked" by accident). Added diagnostic breakpoint() calls (marked # temp)
   for NaN/Infinity and for any other unrecognized type reaching the encoder, to surface real cases for
   investigation rather than guessing at handling now.

### Fixed
- Fixed KeyedIndex.__init__() from a KeyedList: was storing a dict_keys view instead of a dict, making the
   resulting KeyedIndex unsubscriptable.
- Fixed KeyedList.to_json()/from_json() and KeyedListEncoder: were not converting the KeyedIndex hd to a
   plain dict before/during JSON serialization; from_json() also called __init__() with nonexistent
   hd=/values= keyword arguments instead of positional args.
- Fixed mdlink(): the s3-path branch called a nonexistent utils.safe_basename() (now uses
   parse_s3path()['basename']), and separately left `url` unassigned (UnboundLocalError) whenever a title
   was already supplied for an s3 path.
- Removed two stray, unmarked debug breakpoint() calls (in KeyedList.__init__() and md_lol_table()'s
   exception handler, the latter now unreachable after the transpose_lol() fix above).

### Removed
- Removed `includes_header` / `sum_col_idxs` parameters from md_lol_table() (legacy from AuditEngine's
   pre-daffodil md.py, unused by daffodil; sum_col_idxs called a nonexistent utils.lol_sum_cols()).
- Removed md_process_template() (unused dead code; was also broken via a Template.safe_substitute() misuse).
- Removed set_type_la() from daf_utils.py: unreachable dead code, only referenced from an
   already-commented-out caller (apply_dtypes_to_hdlol, itself noting "do we need this any more? Use
   my_daf.apply_types()"); its own logic was separately dead due to a `str is not NULL` typo.
- Removed buff_csv_to_lol_old(), profile_ls_to_loti(), dict_with_index()/with_index(), combine_records(),
   safe_max()/safe_min(), notice_beep(), validate_json_with_error_details(), slice_to_list() from
   daf_utils.py -- confirmed unreachable from any production code path via reachability analysis.

### Added (continued)
- Added pytest test coverage for the remainder of daf_utils.py (60% -> 90%), including the CSV/header
   data-munging cluster (profile_ls_to_lr, is_comment_line, make_strbool/test_strbool, unexcelstringify,
   precheck_csv_cols, get_csv_column_names), the dtype-coercion cluster (convert_type_value, astype_value,
   unflatten_val, str2bool, is_numeric, safe_eval, json_decode, safe_convert_json_to_obj, json_encode,
   NpEncoder), lol manipulation helpers, the list_stats family, smart_fmt, logging/misc helpers, slice/
   range/path/s3 helpers, buff_csv_to_lol, xlsx_to_csv, compare_lists, and several smaller helpers.
- Added `_filter_comment_lines()`: a lazy, quote-aware, streaming-compatible comment/blank-line filter
   for CSV input, used by buff_csv_to_lol() when user_format=True (default mode). Tracks open quoted
   fields (counting `"` per line) so an embedded newline inside a quoted field -- which could otherwise
   look like a standalone comment/blank line -- is never misclassified and stripped.
- Added `strict_comment_filter` parameter to buff_csv_to_lol() (default False): opts into the older,
   rigorous-but-non-streaming preprocess_csv_buff() behavior when needed; raises a clear ValueError if
   used with non-str/bytes input, since that path cannot be streaming-compatible.

### Changed (continued)
- buff_csv_to_lol()'s user_format=True comment-filtering is now streaming-compatible by default (see
   _filter_comment_lines() above), rather than requiring the full buffer to be materialized up front.
- beep(): fixed inverted is_linux() condition that caused it to silently no-op on actual Linux; now
   emits the terminal bell character on Linux (no external `beep` package dependency) and continues to
   use winsound/os.system('beep ...') on Windows/other, for this dev-only diagnostic helper.

### Fixed (continued)
- Fixed buff_csv_to_lol()'s Iterator branch (text_line_generator/byte_line_generator): both reused the
   name `buff` for the closure's captured free variable and the reassignment target, which (due to
   Python's late-binding closures) caused the generators to yield from themselves, raising
   'ValueError: generator already executing'. This broke Daf.from_csv() for http(s):// and s3:// sources,
   both of which pass a real generator through this exact path -- not merely a theoretical/unused
   capability. Fixed by capturing the original iterator under a distinct name before reassignment.
- Fixed safe_regex_replace(): its own type hint allows `regex` to be passed as a list of patterns
   directly, but `.strip('"')` was called unconditionally before checking for list input, crashing with
   AttributeError on real list input. Fixed by checking the type first.
- Fixed write_buff_to_fp(): `s3path` was only assigned inside the `if buff:` block, but the `else`
   branch (taken when buff is falsy/empty) referenced it, raising UnboundLocalError.
- Fixed write_buff_to_fp()'s s3 branch: was doing `from . import s3utils`, but daffodil.lib.s3utils does
   not exist anywhere in this package (left over from the AuditEngine migration, where a much larger,
   ~2,600-line, AE-internal-dependency-laden s3utils.py presumably did exist). Replaced with two new
   minimal, self-contained daf_utils.py functions using plain boto3 (no AE-internal, pandas, or
   smart_open dependencies, consistent with daffodil's existing avoidance of pandas/numpy elsewhere):
   - `write_buff_to_s3path(s3path, buff, content_type)`: writes unconditionally, with a small retry
      loop for transient ClientErrors and a post-write existence-polling loop (kept as a safety net for
      occasional propagation delay, even though S3 PUTs are strongly consistent since Dec 2020).
   - `does_s3path_exist(s3path)`: existence check via a HEAD request.
   Also removed the `if_unmodified` parameter and the `write_buff_to_s3path_if_modified()` call it
   gated: the ETag-based "only write if changed" / local-mirror caching behavior is an application-level
   concern (e.g. AuditEngine's `DB.load_data()`/`store_data()` layer), not something daffodil core needs
   to implement itself.

### Known issues found, not yet fixed (flagged for a future round)
- add_trailing_columns_csv() (used by xlsx_to_csv()'s default add_trailing_blank_cols=True path) samples
   the first `num_rows` (default 3) rows via repeated next(reader) calls to estimate max column count --
   raises an uncaught StopIteration if the CSV has fewer than num_rows rows total. The function's own
   docstring already flags it as "@@TODO -- should be DEPRECATED".
- insert_irow(): if `row` is neither a list nor dict (including the default row=None), `row_la` is never
   assigned before use, raising UnboundLocalError. Intended None-row semantics not yet decided.
- append(): the dedicated KeyedList fast-path (`if self.hd == data_item.hd: lol.append(values())`) is
   unreachable -- the preceding `isinstance(data_item, (dict, KeyedList))` check already intercepts every
   KeyedList first, routing it through the slower generic record_append() path instead. Not a correctness
   bug (record_append() handles KeyedList correctly too via its own dedicated branch), just a missed
   optimization. May be related to recently-added dict-like operations on KeyedList; needs investigation
   before deciding whether the dedicated branch is still needed at all.

### Added (daf_pandas.py / daf.py)
- Added pytest test coverage for daf_pandas.py's standalone dtype-mapping helpers (57% -> 95%):
   pandas_dtype_to_python_type, python_dtype_to_pandas, pandas_dtype_dict_to_python,
   dtypes_dict_from_dataframe, and the use_csv=True + default= combination in _to_pandas_df.
- Added pytest test coverage for daf.py's construction/I-O cluster: __init__, copy, from_lot,
   set_dtypes/apply_dtypes, append/_basic_append, insert_dif_row, insert_irow's working paths, and the
   full CSV I/O round trip (from_csv for local/HTTP/S3 sources via mocking, from_csv_file, to_csv_file,
   to_csv_buff, buff_to_file). This batch also serves as an integration-level confirmation of the
   buff_csv_to_lol() streaming fix above, exercised through Daf.from_csv()'s actual HTTP/S3 generator path.

### Fixed (daf_pandas.py)
- Fixed pandas_dtype_to_python_type(): checked np.integer before np.timedelta64, but numpy considers
   timedelta64 a subtype of integer (it's stored internally as an integer count of time units), so every
   timedelta64[ns] column was silently misclassified as int instead of pd.Timedelta. Fixed by checking
   np.timedelta64 first; added a comment explaining why the order matters, to prevent future regression.
- Marked the "Unknown Pandas dtype" fallback breakpoint() in pandas_dtype_dict_to_python() as `#perm`
   (a deliberate diagnostic tripwire that should never fire, but needs visibility if it ever does).

### Changed (daf.py)
- Commented out (not deleted) daf_sum2()/daf_sum3() and the sum_da2()/sum_da3() functions they called:
   investigatory variants used to find out why daf_sum() (sum_da) was slow. The key finding -- comparing
   `value == ''` instead of `value is NULL` made the loop take ~10x longer, since Python implements '' as
   a singleton (like None), so the very fast `is` comparison works -- was already adopted into sum_da()
   and other NULL comparisons throughout daf.py. Preserved as a commented-out record of that
   investigation/finding for future reference, rather than deleted.

### Added (daf.py: __setitem__ / set_irows_icols)
- Added pytest test coverage for set_irows_icols() (the engine behind Daf.__setitem__ /
   `daf[irows, icols] = value`) across all its branches: single row/col, single row/no col, multi-row/no
   col, multi-row/single col, and multi-row/multi-col, each for scalar/list/dict value types.

### Fixed (daf.py: __setitem__ / set_irows_icols)
- Fixed set_irows_icols(): in three of its four `isinstance(value, (list, Iterable))` checks, a `dict`
   value was being silently swallowed by the broader Iterable check (dict is itself Iterable, just not a
   Sequence) before ever reaching its own dedicated `elif isinstance(value, dict):` branch below it. This
   caused two distinct symptoms depending on which branch was hit:
   - Multi-row, no column selection (e.g. `daf[[0,1],:] = {...}`): silent data corruption -- the dict
      object itself was assigned as a row's raw content (`self.lol[irow] = value`) instead of being
      mapped to columns via assign_record_irow().
   - Multi-row, multi-column (e.g. `daf[[0,1],[0,1]] = {...}`): integer-indexed lookup into the dict
      raised KeyError, caught by a bare `except Exception:`, which hit a `breakpoint() #perm ok` tripwire
      -- hanging indefinitely in any non-interactive context (scripts, tests, production).
   A fourth branch (multi-row, single column) had its dict-handling entirely commented out, meaning a
   dict value there hit the exact same hang via the same root cause.
   Fixed by (a) reordering every branch to check `isinstance(value, dict)` first, and (b) replacing the
   overly-broad `Iterable` check with `collections.abc.Sequence`, which correctly excludes dict, set, and
   generators while still covering list/tuple/range. The fourth branch's dict-handling was uncommented
   and given the same fix. The `#perm` breakpoint tripwires were left in place (per project convention,
   they remain valuable for catching genuinely unexpected cases) -- they're simply no longer reachable
   via this particular (now-fixed) misrouting.

### Known issues found, not yet fixed (added to the list above)
- set_irows_icols()'s `elif isinstance(value, type(self)):` branches (assigning a whole Daf as the
   value) appear to have pre-existing correctness problems independent of the dict/Sequence fix above:
   indexing a Daf by a single int (e.g. `value[source_row]`) returns a wrapped single-row Daf object, not
   a plain list of values, so `self.lol[irow] = value[source_row]` stores a Daf object as row content
   rather than its values. In the multi-row/multi-col branch the indexing also appears semantically
   inverted: `value[source_col]` uses a column-position index to select a *row* out of the Daf value
   (since indexing by a single int selects a row), which doesn't correspond to "the value for this
   column" at all. Not yet fixed -- needs a decision on what assigning a whole Daf as a __setitem__
   value is actually supposed to mean in each branch before attempting a fix.

### Added (daf.py: select_irows / select_icols / gkeys_to_idxs / col_to_la)
- Added pytest test coverage for select_irows(), select_icols(), gkeys_to_idxs() (and its
   krows_to_irows()/kcols_to_icols()/select_krows()/select_kcols() callers), and col_to_la().

### Fixed (daf.py: select_irows)
- Fixed select_irows(int, invert=True): `row_sliced_lol = self.lol` (not a copy) followed by
   `row_sliced_lol.pop(irows)` was destructively mutating the *original* Daf's lol in place --
   contradicting the function's own documented contract of being a non-mutating selection that returns
   a new instance. Fixed to build a fresh filtered list instead, correctly handling negative indices the
   way `list.pop()` does.
- Fixed select_irows(slice(...), invert=True): raised `TypeError: argument of type 'slice' is not
   iterable`, since a raw slice object doesn't support membership testing (`in`). Fixed by converting the
   slice to a range via daf_utils.slice_to_range() first.
- Fixed select_irows()'s list-of-ints fast path: `if len(irows) == len(self.lol): row_sliced_lol =
   self.lol` silently ignored reordering or repeated-index selections that happened to be the same
   length as the original (e.g. `select_irows([2,0,1])` returned rows unchanged instead of reordered).
   Fixed with a short-circuiting natural-order check (`all(irow == i for i, irow in enumerate(irows))`)
   that bails out on the first mismatch rather than doing full-length work regardless -- chosen
   specifically to stay cheap for large arrays (e.g. 500K+ rows) in the common case where the fast path
   doesn't apply.

### Fixed (daf.py: select_icols)
- Fixed select_icols(): `self.num_cols` was missing `()` (comparing a bound method object to an int,
   always False), making the analogous "selecting all columns, reuse self.lol directly" fast path
   permanently dead. Naively adding `()` alone would have introduced a *new* correctness bug (confirmed
   via direct testing): a same-length but reordered or repeated-index `icols` selection would then
   silently return data unchanged while still correctly reordering the column *labels* -- a label/data
   mismatch worse than the original dead code. Fixed with the same short-circuiting natural-order check
   as select_irows() above, which is correct for both possible cases.

### Fixed (daf.py: gkeys_to_idxs)
- Fixed gkeys_to_idxs()'s slice branch: previously built `range(gkeys.start or 0, gkeys.stop or
   len(keydict), ...)` and then looked up each resulting ordinal position *as if it were itself a key* in
   keydict (`keydict[gkey]`) -- which only coincidentally worked if keydict's keys happened to be the
   identity mapping `{0:0, 1:1, ...}`; for any realistic keydict (string keys, or non-sequential int
   keys) this raised KeyError immediately, including for plain numeric slices like `slice(0, 2)`.
   Clarified semantics: slice.start/.stop/.step are ordinal positions into keydict's iteration order (the
   same convention integer indices use elsewhere), never keys to look up -- for arbitrary keys there is
   no well-defined "next key" to support exclusive-stop slicing the way integer positions do, which is
   exactly why the existing `(start_key, stop_key)` tuple form (inclusive of stop_key) exists as the
   correct mechanism for genuine key-range selection. Fixed by treating slice bounds as plain ordinal
   positions (filling in 0/len(keydict)/1 defaults) and returning a `slice` object directly, mirroring
   how the Tuple branch already builds and returns a `slice`.
   Also documented (via comment) an architectural finding from tracing this function's callers: this
   slice branch is only reachable via the explicit select_krows()/select_kcols() method calls -- the
   `daf[...]` / `__getitem__`/`__setitem__` `[]` syntax (_parse_selectors) routes ANY slice object
   straight to the plain-index path without ever calling krows_to_irows()/kcols_to_icols()/
   gkeys_to_idxs() at all, regardless of whether the slice's bounds are ints or strings. So
   `daf['a':'c']` does not perform key-based slicing via `[]` (it errors trying to use the strings as
   literal list-slice bounds) -- only `daf.select_krows(slice(...))`/`daf.select_kcols(slice(...))`
   reach this code path.

---

## [0.5.12] - (pending)
### Added
- .from_md() method added to allow round-trip of daffodil arrays through markdown representation.
   - changed footer syntax so it does not need markdown escaping, uses standard syntax for easy parsing.
   - footer contains, rows, cols, keyfield, name, schema
   - will detect abbreviated markdown table output and raise an error.
- added 'astype' parameter to .col_to_la() operator.
- Added schemaclass record_from() method, used like: da = schema.BifSchema.record_from(other_da)
- Added indirect_col to .col() and .col_to_la()
- Added indirect_col to select_where() split_where()
- Added group_where() method to allow arbitrary grouping including fanout grouping.
   - Added tests for group_where()
- Introduced KeyedIndex class for future porting to rust. Also added tests.
- Added \_basic_append() to streamline appends when it is known that the hd is compatible.

### Changed
- Changed some uses of [:, col_name].to_list() to .col(col_name)
- Implemented lazy kd generation rather than respecting it. {} indicates invalidated.
- .kd no longer should be accessed by user code. Changed variable to .\_kd.
- .keys() now has astype parameter which can be 'list' or 'view' to give access to keys and 
    allowing fast containment tests.
- modified README to reflect dynamic building of kd. This should ease use while still being backward compatible.
    however, any direct use of .kd is now banned. Use .keys() to access the keys either as list or view.


### Fixed
- Corrected .apply_dtypes() so it would allow that not all columns defined by the schema exist, if columns are already defined.
    However, dtypes must include all columns that do exist.
- Added removal of empty list items when read from csv if there are blank lines at the end of the file. Edge case.
- Fix edge case selecting rows by empty list or iterator.
- Fixed edge case of single integer row specified as 0 due to using 'if not' construction.
- Fixed pyproject.toml to include dependencies required for testing.
   - to test, use `uv run --group dev pytest -s`
- Fixed a number of datatype conversion issues in to_pandas() and from_pandas()
- Went back to desired_type == int instead of is int, because == will include int, int64 etc.
   - Included a comment but use an noqa override for ruff linter.
- Fixed error in indexing that required rebuilding kd and it was not done.
- changed == '' to is NULL, with NULL defined as '' (performance improvement)
- Use npao for transpose operation. (performance improvement)

- created a large set of tests for indexing modes, which resulted in improvements for edge cases and some api/ui changes.
   
   1. Changed the parameters for .to_list() and .to_value() to use indexing and then chaining to those functions. not as performant but simpler interface. 

   2. Fixed the README about using tuples of strings as inclusive ranges, as None is required as the first parameter to start at the beginning through the named item. 

   3. Kept shape() as a method and not a property as it is somewhat costly to calculate. 

   4. Improved parsing of indexing with a common section for both getitem and setitem. 

   5. Fixed a number of edge cases when selections were no columns, no rows, etc. with rules for how these would be handled without needing to make every nonsensical change supported. 

   6. disabled using kd and set keyfield='' if the keyfield column was dropped. 

   7. narrowed the application of .to_value() to handle a single-value only, from location 0,0 in a rudimentary daf array. 

   8. Return [] if .to_list() is used on an array with no columns or rows. 

   9. Added SchemaBase in schemaclass.py to allow mypy to understand dynamically added methods when using the @schemaclass decorator. 

   10. removed tests for using indices inside .to_list and .to_value. 

   11. Fixed improper tests for tuple string ranges. 

   12. Added test file 'test_daf_indexing2.py' with about 100 tests for indexing.

- Introduce T_cs, T_ca, T_ci, T_ma. Start phasing them in to avoid mypy errors.
- use Iterable, Iterator, keysView from collections.abc
- deprecate 'align' in md table functions.
- Fixed edge case in keyedlist.py, when instanciating with KeyedList().

- Add option of using daffodil tables as schema instead of schemaclass.
- Moved .attach_schema from daf to daf_schema.py
- Added .apply_schema which is like apply dtypes.

- .default_record now properly a method of daf rather than schema.
- Added option to .sort() to do a shallow copy of lol and kd in so a copy can be independently sorted.
- Added .from_directory to get a daf table of a directory of file system
- Improved .from_md() with improved scanning for the first table in the passed block.
- Added cleanup of colnames by removing md or wiki markup.
- Added crud in schemaclass.py to allow inheritance.

---

## [0.5.11] - 2026-02-02

### Added
- Added optional schema class support for declarative column definitions using Python type annotations.
- Added default_record() method to create new records initialized from schema defaults.
- Added text to README.md to explain the new schema approach.
- Added decorator approach to schemaclass and placed in a separate file schemaclass.py with schemaclass_README.md in `/lib`
- Added method .attach_schema(self, schema: type) to allow schema to be attached after reading the file. 

### Changed
- daf.keys() now has optional parameter 'silent_error' which defaults to True.
   - If True and there is no keyfield set, do not produce KeysDisabledError and instead return []
   - If False, raise KeysDisabledError
- added compatibility for astype_la() to use type values like int,str,float,bool as well as `int`,`str`, etc.
- apply_dtypes() now enforces exact match by default
   - Use silent_error=True to allow mismatches.
   
### Fixed
- (add entries here)

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