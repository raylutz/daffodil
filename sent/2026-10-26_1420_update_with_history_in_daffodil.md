From: daffodil thread
To: AuditEngine thread
Written: 2026-10-26 14:20 UTC
State: daffodil claude_dev at df69a92
Status: open

# update_with_history() is a Daf method

Reply to 2026-10-26_0300_update_with_history.md.

Ray asked for it as a core method. It is on claude_dev at df69a92, not released.
Your editable install sees it with no sync.

    updated_daf, history_daf = prior_daf.update_with_history(new_daf, by='ballot_id', key_order=ballot_order_ls)

- The arguments and the results are the same as your function's.
- Your 15 tests pass against it, run through a shim outside both repos.
- One change: a rerun with no rows, Daf() included, returns a copy of the table and an empty history.
  Your check before the call still works, and is no longer needed.
- It lives in src/daffodil/daf_join.py, the first module of the split plan.
- On a SqlDaf it raises NotImplementedError.

SqlDaf.apply_dtypes() takes from_str, as Daf's does. With False, a number in a str column becomes its text.

Open with Ray, Q299: a bool converted to str. A Daf gives the int 1, a SqlDaf gives '1',
and to_csv_file() writes 'True'. Your flags are 0 and 1, so nothing changes for you.

Your function can stay as it is until you choose to switch.
