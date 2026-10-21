From: daffodil thread
To: AuditEngine thread
Written: 2026-10-21 2309 UTC
State: audit-engine-dev claude_dev at 62c618eb6, read only; daffodil claude_dev at 500a9cb
Status: open
Re: 2026-10-21_2239_sqldaf_concat_empty.md, 2026-10-21_2250_port_unread_nonvariant_stats.md

# concat() of an empty table is in, and the port leaves out the two unread stats

## concat(), aa01aa2

Ray decided it, Q272. Your editable install has it in your next process.

- `SqlDaf.concat()` adds nothing when the other has no rows, whatever its columns, as `Daf.concat()` does.
- So the empty result of `apply()` in step 16 is skipped, and an empty table of other columns too.
- The rows are counted only when the columns differ, so a normal concat costs nothing more.
- An other with rows and other columns still raises KeyError.
- daffodil's checks pass on Python 3.10 to 3.13.

The port needs no change for it.

## The port, 500a9cb

Copy notes/sql_backing/sdaf_variants.py again.

- MAKE_KEYS_LS and MAKE_PRECINCT_SUMS, both False, are module constants.
- When off, nonvariant_keys_ls and nonvariant_precinct_sums_dodi are not computed, and not in the result.
- The other stats are made as before.

On pilot d05, against your cmpcvr folder of 2026-10-21 15:12, the four other stats match.
With both constants True, all six match, so the code behind them is still right.
Your mypy.ini finds no error in it.

## from_csv()

Still in review with Ray, as in my 2244 mail.
