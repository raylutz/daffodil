# Profiling mode

The profiling mode shows how a program uses Daf: how big its tables get, what is done
with them, and which lines of the program do it. Use it to decide where Daffodil needs to be
faster, or where a program uses it in a costly way.

## Turning it on

Set an environment variable. The program needs no change.

```bash
DAFFODIL_PROFILE=1 DAFFODIL_PROFILE_STAGE=tabulate python my_program.py
```

When the program ends, the profile is written to `daffodil_profile_<stage>_<host>_<pid>.md`
in `DAFFODIL_PROFILE_DIR`, the current directory by default. The report is printed to
stderr. `DAFFODIL_PROFILE_FILE` also writes the report to a file.

Or turn it on in code:

```python
from daffodil.lib import daf_profile

daf_profile.start(stage='tabulate')
run_the_work()
print(daf_profile.report())
daf_profile.stop()
```

## A pipeline of stages

A pipeline whose stages run as separate programs gets one profile file per stage. Give each
stage its name, and write the files to one directory:

```bash
export DAFFODIL_PROFILE=1 DAFFODIL_PROFILE_DIR=profile_run
DAFFODIL_PROFILE_STAGE=load     python stage_load.py
DAFFODIL_PROFILE_STAGE=tabulate python stage_tabulate.py
python -m daffodil.lib.daf_profile combine profile_run -o report.md
```

The report has a section for all the stages together, and one for each stage. `--data
combined.md` also writes the combined tables.

## The profile is Daf tables

The totals are kept as Daf tables, and a profile file is those tables in Markdown, written
with `Daf.dodaf_to_md()`. Every table has a `stage` column.

| Table | One row for each | Columns |
|---|---|---|
| `info` | process | host, pid, Python and daffodil versions, start time, seconds |
| `tables` | stage, creation line and way of making a table | tables, keyed, most rows and columns, tables in each size band |
| `ops` | stage, creation line, way of making, and method | calls on those tables |
| `methods` | stage and method | calls, seconds, rows at each call in size bands, most rows |
| `sites` | stage, call site and method | calls, seconds |

`daf_profile.tables()` returns them for the current process, `load(path)` reads a file, and
`combine(runs)` adds the tables of several runs. Counts and times are added. The most rows
and columns take the largest. `combine()` uses `groupby_cols_reduce()` to do it.

A call site is the module and line, as in `auditengine.tabulate:212`, so the same line has the
same name in every run. A script run directly is named by its file name.

## What it records

It keeps running totals, not a log of every call, so memory stays small.

- **Each table.** The line that created it, and how: `Daf()`, `from_csv`, `select_where` and
  so on. The most rows and columns it had, whether it had a keyfield, and a count of each
  method called on it. When the table is freed, this summary is added to the totals for its
  creation line. The median size is reported as the size band that holds it, since bands
  can be combined across runs and a running sample cannot.
- **Each method.** Calls, total time, and the rows of the table at each call, in size bands.
- **Each call site.** Calls and total time, by method.

Only outer calls are counted. When a Daf method calls another one inside, the inner call is
not counted. A table that is built by 10,000 calls to `append()` is one table, with
`append 10,000` in its list of operations.

## The report

The report lists the runs, then has four Markdown tables, for all stages and for each stage.

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
- A child process that ends with `os._exit()`, as multiprocessing workers do, writes nothing
  at exit. Call `daf_profile.dump()` at the end of the worker instead.
- Lambda and other hosts that freeze or kill the process are not handled yet.

## API

::: daffodil.lib.daf_profile.start

::: daffodil.lib.daf_profile.stop

::: daffodil.lib.daf_profile.tables

::: daffodil.lib.daf_profile.dump

::: daffodil.lib.daf_profile.load

::: daffodil.lib.daf_profile.combine

::: daffodil.lib.daf_profile.report

::: daffodil.lib.daf_profile.reset

::: daffodil.lib.daf_profile.is_active
