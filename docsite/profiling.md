# Profiling mode

The profiling mode shows how a program uses Daf: how big its tables get, what is done
with them, and which lines of the program do it. Use it to decide where Daffodil needs to be
faster, or where a program uses it in a costly way.

## Turning it on

Set an environment variable. The program needs no change.

```bash
DAFFODIL_PROFILE=1 python my_program.py
```

When the program ends, the report is printed to stderr and written to a file. The file is
`daffodil_profile_<pid>.md` in the current directory. Set `DAFFODIL_PROFILE_FILE` for another
name. `{pid}` in the name is replaced with the process id.

Or turn it on in code:

```python
from daffodil.lib import daf_profile

daf_profile.start()
run_the_work()
print(daf_profile.report())
daf_profile.stop()
```

`report(path)` also writes the report to a file. `reset()` clears the totals.

## What it records

It keeps running totals, not a log of every call, so memory stays small.

- **Each table.** The line that created it, and how: `Daf()`, `from_csv`, `select_where` and
  so on. The most rows and columns it had, whether it had a keyfield, and a count of each
  method called on it. When the table is freed, this summary is added to the totals for its
  creation line.
- **Each method.** Calls, total time, and the rows of the table at each call, in size bands.
- **Each call site.** Calls and total time, by method.

Only outer calls are counted. When a Daf method calls another one inside, the inner call is
not counted. A table that is built by 10,000 calls to `append()` is one table, with
`append 10,000` in its list of operations.

## The report

The report has four Markdown tables.

1. **Tables by the line that created them.** Tables made at the same line are grouped, so a
   loop that makes 3,000 small tables is one line, with the median and the largest size.
2. **Table sizes.** How many tables reached each size, by rows and by columns.
3. **Methods.** The methods by total time, with the rows at each call in size bands.
4. **Busiest call sites.** The lines of the program that spent the most time in Daf.

## Costs and limits

- When it is off, nothing is wrapped, and Daf runs at full speed.
- Each counted call costs about 3 µs, and each call that makes a new table about 15 µs more.
  A call that Daf makes inside another one costs about 0.4 µs. So the times of cheap calls
  are inflated. Counts and sizes are exact.
- Only calls to Daf methods are seen. A loop that reads `daf.lol` directly is not.
- A child process that ends with `os._exit()`, as multiprocessing workers do, writes no
  report. Call `daf_profile.report(path)` at the end of the worker instead.

## API

::: daffodil.lib.daf_profile.start

::: daffodil.lib.daf_profile.stop

::: daffodil.lib.daf_profile.report

::: daffodil.lib.daf_profile.reset

::: daffodil.lib.daf_profile.is_active
