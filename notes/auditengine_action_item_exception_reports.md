# Action item for AuditEngine: report daffodil errors like breakpoints

2026-10-02. Written in a daffodil session for the Claude session that works on AuditEngine.
The daffodil changes are on branch claude/daffodil-test-coverage-jkhn2m and not yet merged.

## Summary

- Daffodil no longer calls `breakpoint()`. Where it used to, it now raises an error.
- In AuditEngine, the breakpoint hook wrote a report at each of those places. Normally it then
  exits with code 42, so the run stopped there. Only when set to continue, such as in Lambda,
  did the code carry on.
- An error raised inside daffodil does not reach the breakpoint hook. In Lambda it is caught
  by the top-level try and except, and no report is written.
- The action: write the same kind of report from that top-level except clause, using the
  error's traceback. Then daffodil errors are as easy to diagnose as breakpoints.
- In the normal exit mode, the outcome is the same as before: the run stops at the problem.
  Only the report is missing, which this action item fixes.
- In continue mode, one thing is lost. After an error, the work cannot continue from the
  failing line. Section 4 lists the daffodil places where continue mode used to keep going.

## 1. Why sys.excepthook is not enough

- `sys.excepthook` only runs for an exception that nothing catches.
- In Lambda, the handler wraps everything in a try and except. So the hook never runs.
- Notes from 2026-07-11 put that except clause in launcher.py, near line 75. It catches
  BreakpointCaptured to set the LambdaTracker job status. It may have moved since.

## 2. Proposed change in AuditEngine

At each top-level except clause, save a report for any error that isn't BreakpointCaptured.
A sketch, to adapt to the real names in utilities/breakpoint_hook.py:

```python
except Exception as exc_info:
    if not isinstance(exc_info, BreakpointCaptured):
        breakpoint_hook.save_exception_report(exc_info)
    ...existing handling...
```

The new function in breakpoint_hook.py would:

- walk the traceback with `traceback.walk_tb(exc_info.__traceback__)`;
- take the last frame as the call site, since that is the line that failed;
- collect the stack, with locals, from those frames, as the hook does now;
- record the exception type and message as python_exception;
- save it with the existing `_save_report()`, so the report looks the same as a breakpoint
  report.

The traceback keeps every frame from the failing line up to the except clause, with its
locals. So the report has the same detail as one written at a breakpoint.

Also worth checking:

- Whether CLI runs have their own top-level except clause that needs the same change.
- Whether an uncaught error in CLI mode should also exit with code 42.
- Whether the report should say it came from an exception, not a breakpoint.

## 3. What can't be recovered

- This only matters when the hook is set to continue. In the normal exit mode, the run
  stopped at these places anyway.
- In continue mode, a breakpoint let the code carry on with a fallback.
- An error unwinds the stack. The function that failed can't resume. The Lambda task fails
  where before it might have finished.
- For the daffodil places in section 4a, this is a real change in behavior.

## 4. Every daffodil place that used to call breakpoint()

"Continue mode" means what happened on main when the hook returned. "Now" is the branch as of
2026-10-02. Line numbers are on main. daf_pdf.py and md_demo.py still have breakpoints and are
not listed.

### 4a. Continue mode kept going, now it raises

In the normal exit mode, all of these stopped the run, as they still do. Only in continue
mode did they carry on, and these are the ones where that changes.

| Place | Continue mode on main | Now |
|---|---|---|
| daf.py:6842 reduce | Row skipped, reduction went on | Error, or row skipped with silent_error=True |
| daf.py:6880 reduce, sparse rows | Row skipped, reduction went on | Error, or row skipped with silent_error=True |
| daf.py:6940, 7020, 7036 sum_da | Odd value skipped | The value's own error |
| daf.py:7379 alter_daf_per_setting | Missing setting did nothing | KeyError, or nothing with silent_error=True |
| daf_utils.py:126 json_encode | NaN written as NaN, not valid JSON | ValueError |
| daf_utils.py:150 test_strbool | Odd type gave False | TypeError |
| daf_utils.py:1655 write_buff_to_fp | Write failed silently, path returned | The OSError |
| daf_utils.py:1747 slice_to_range | Bad slice gave None | TypeError |
| daf_pandas.py:219 pandas_dtype_dict_to_python | Unknown dtype column left out | TypeError |
| daf_sql.py:225 create_index_at_cursor | Returned False | The sqlite error |

### 4b. Continue mode crashed or raised anyway

Behavior in continue mode doesn't get worse here. The error is just clearer.

| Place | Continue mode on main | Now |
|---|---|---|
| daf.py:862, 872 calc_cols | RuntimeError | RuntimeError with a message |
| daf.py:1464 flatten | RuntimeError | RuntimeError with a message |
| daf.py:4186 select_icols | UnboundLocalError | IndexError |
| daf.py:5544, 5573 sort_by_colname(s) | UnboundLocalError | KeyError |
| daf.py:5634, 5665 apply_formulas | RuntimeError, or the formula's error | Same |
| daf.py:5968 apply_in_place | NotImplementedError | Same |
| daf.py:7391 alter_daf_per_setting | UnboundLocalError | The error from from_lod |
| daf.py:8252 join | UnboundLocalError | The error from select_where |
| daf_utils.py:105 NpEncoder.default | TypeError | Same |
| daf_utils.py:373 safe_regex_select | UnboundLocalError | ValueError |
| daf_utils.py:425 safe_regex_replace | See note below | ValueError |
| daf_utils.py:520 convert_type_value | UnboundLocalError | TypeError |
| daf_utils.py:1714 len_slice | UnboundLocalError | TypeError |

Note on safe_regex_replace: with a bad first pattern, continue mode gave UnboundLocalError.
With a bad later pattern, it silently applied the previous pattern again.

### 4c. No error now

| Place | Continue mode on main | Now |
|---|---|---|
| daf.py:274 Daf init | Could not be reached | Check removed |
| daf.py:3231 record_append | UnboundLocalError for a non-dict mapping | Works |
| daf.py:3686, 3712 set_irows_icols | Partial fill | Copies where source and target overlap |
| daf_utils.py:265 insert_col_in_lol_at_icol | Check could fire wrongly on ragged rows | Check removed |
| daf_utils.py:1935 compare_lists | UnboundLocalError for a tuple | Tuples accepted |

## 5. Suggested steps for the AuditEngine session

1. Find every top-level except clause, in Lambda and CLI entry points.
2. Add `save_exception_report()` to utilities/breakpoint_hook.py, reusing `_save_report()`.
3. Call it from those except clauses for anything that isn't BreakpointCaptured.
4. Test it by raising a KeyError deep in a call, and check the report shows the failing line
   with its locals.
5. Search AuditEngine for calls to the functions in section 4a. Decide whether any of them
   relied on continue mode. If so, the caller may need its own guard, or the owner may want
   that daffodil change revisited.

## 6. Separate check: json_encode and NpEncoder

- daffodil's daf_utils.py has `json_encode()` and the `NpEncoder` class it uses.
- Nothing in daffodil calls them. ROADMAP.md says json_encode was replaced by Daf.to_json().
- They probably came from AuditEngine. The owner wants them removed from daffodil if unused.
- NaN and Infinity never occur in daffodil data. json_encode now raises ValueError on them,
  which is Python's own error from strict JSON.

Steps:

1. Search AuditEngine for `json_encode` and `NpEncoder`, including any `daf_utils.` prefix.
2. If AuditEngine uses them, copy them into AuditEngine's own utilities and change those
   imports.
3. Report back, so they and their tests can be deleted from daffodil.
