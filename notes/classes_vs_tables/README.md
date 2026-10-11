# Classes and marshalling, or one table

Started 2026-10-11 by the daffodil thread, at Ray's request. A Guide page may come from this later.

## The question

A common way to handle records in Python is a class for each kind of record. A library such as
marshmallow or pydantic loads the records from JSON into objects and dumps them back. The code
then works on a list of objects, one at a time.

Daffodil keeps the records as rows of one table, stored as CSV. The code works on the whole table
or on its columns, and the rows stay close to the stored form.

This note compares the two on the same job. It looks at code size, speed, memory, and which
mistakes each one catches, and where.

## The job

There are 100,000 ballot batches. Each has five fields: batch_id, precinct, ballot_type,
num_ballots and is_duplicate. The program does four steps:

1. Load the file.
2. Sum num_ballots for each precinct, leaving out duplicates.
3. Set a new field, flagged, where num_ballots is over 500.
4. Write the file back out.

There are three versions, and all three give the same totals and the same records:

| Version | File | On disk | Code lines |
| :-- | :-- | :-- | --: |
| marshmallow | impl_marshmallow.py | JSON | 36 |
| pydantic | impl_pydantic.py | JSON | 23 |
| daffodil | impl_daffodil.py | CSV | 16 |

The marshmallow version names every field twice: once in the dataclass and once in the schema.
The pydantic version names each field once, in the model. The daffodil version names each field
once, in a dtypes dict.

## How to run it

The scripts need marshmallow and pydantic, which are not daffodil dependencies. `run.sh` adds them
for the run only, with Python 3.13 and this repo's daffodil.

```
notes/classes_vs_tables/run.sh check_same.py   # the three agree
notes/classes_vs_tables/run.sh timing.py       # speed and memory
notes/classes_vs_tables/run.sh faults.py       # one fault at a time
```

`typo_check.py` is for mypy. Run it with mypy, marshmallow and pydantic installed.

## Speed and memory

Python 3.13.1 on the EC2 box, 2026-10-11. Times are in seconds, the best of three.

| | load | totals | flag | dump | total | memory |
| :-- | --: | --: | --: | --: | --: | --: |
| marshmallow | 1.684 | 0.013 | 0.006 | 0.729 | 2.432 | 28.5 MB |
| pydantic | 0.298 | 0.021 | 0.050 | 0.090 | 0.460 | 114.6 MB |
| daffodil | 0.163 | 0.046 | 0.036 | 0.125 | 0.370 | 28.5 MB |

The JSON file is 11.4 MB. The CSV file is 3.1 MB.

- Daffodil is the fastest overall. Loading is where it gains, since CSV text is short and a table
  is cheap to build.
- Marshmallow is the slowest, mostly in load and dump, which run in Python.
- Pydantic does its checks in Rust, so it is fast. But each object costs about four times the
  memory.
- On Python 3.10, every version is slower. Daffodil's total there was 0.55 s.

The daffodil totals step is a plain loop over three columns. The built-in way,
`groupby_cols_reduce()` with `Daf.sum_da`, took 0.211 s. See the section on reduce below.

## Which mistakes are caught, and where

`faults.py` puts one fault into record 5, then runs the job.

| Fault | marshmallow | pydantic | daffodil |
| :-- | :-- | :-- | :-- |
| a number is "abc" | at load | at load | later: TypeError in the sum |
| a number is empty | at load | at load | later: TypeError in the sum |
| a bool is "no" | read as False | read as False | read as 0, which is right |
| a field is missing | at load | at load | silent: precinct becomes '' |
| an extra field | at load | silent: dropped | at load: ValueError from dtypes |
| a typo in the code | at run time | at run time | at run time: ColumnNotFoundError |

The class libraries check every value at load. The error names the record and the field. That is
their main strength.

Daffodil does not check values at load. `apply_dtypes()` leaves a cell that does not convert as
text, by design, so that loading stays fast. The fault shows up later, far from its cause, or not
at all.

- With the plain loop, a bad number raises TypeError in the sum.
- With `Daf.sum_da`, a bad number is skipped with no error. Its docstring says so. In a test, the
  sum of 10, "abc", empty and 5 came out 15. A total can then be wrong with no sign of it.
- A missing value becomes NULL, which is ''. Here it made a precinct named ''.

mypy catches a misspelled attribute in both class versions before the code runs:

```
typo_check.py:4: error: "Batch" has no attribute "num_balots";
  maybe "num_ballots"?  [attr-defined]
```

A daffodil column name is a string, so mypy cannot check it. The misspelling is found only when
that line runs.

## Writing and keeping the classes

The class versions need code that the table does not: a class for each kind of record, and with
marshmallow a schema for each class as well. Each field is named in one or two extra places. When
a field is added or renamed, each place must change, and a place that is missed is a bug. Records
inside records need a nested class and a nested schema for each level.

The daffodil version names a field only where it is used, plus once in the dtypes dict. Code that
works on the whole table, such as a group or a join, is a single call. The cost is that the
checks the classes give for free must be done some other way.

This prototype does not yet measure the effort of a change. See Next steps.

## A textbook example: the gradebook

Ray asked for an example from the books, not from AuditEngine. *Effective Python* by Brett Slatkin
uses a gradebook to argue for classes over nested dicts. Students have subjects, and subjects
have grades with a score and a weight. The code here is written for this note, in that style. It
is not copied from the book.

| Version | File | On disk | Code lines |
| :-- | :-- | :-- | --: |
| classes, with pydantic | gradebook/gb_classes.py | JSON | 44 |
| daffodil | gradebook/gb_daffodil.py | CSV | 27 |

The class version nests four classes: Gradebook, Student, Subject and Grade. The daffodil version
is one table with one row per grade: student, subject, score and weight. Both give the same
averages. `gb_check.py` checks that.

### Speed and memory

200,000 grades: 10,000 students, 5 subjects, 4 grades each. Python 3.13.1, 2026-10-11, seconds.

| | load | student averages | math average | dump | memory |
| :-- | --: | --: | --: | --: | --: |
| classes | 0.649 | 0.049 | 0.020 | 0.130 | 134.0 MB |
| daffodil | 0.229 | 0.245 | 0.081 | 0.195 | 42.6 MB |

The JSON file is 6.5 MB, and the CSV file is 4.2 MB.

- The table loads three times faster and uses a third of the memory.
- The classes answer the questions faster. The nesting groups the grades by student and subject
  once, at load. The flat table groups them again on each question. The load plus one question
  still favors the table, 0.47 s against 0.70 s.
- How the table is grouped matters. One loop over four columns took 0.14 to 0.17 s.
  `groupby_cols()` on student and subject took 0.563 s, because it builds a Daf for each of the
  50,000 groups. Selecting the subject first with `select_by_dict()` took 0.006 s.
- A question that cuts across the nesting is where the table is simplest. The average for one
  subject over all students is a selection, then a loop. The classes must walk every student.

### The change test: add a date to each grade

The change: store a date with each grade, and average only the grades since a given date.
`gb_classes_v2.py` and `gb_daffodil_v2.py` hold the changed code. `gb_change_check.py` checks them.

| | lines changed | places |
| :-- | --: | :-- |
| classes | 12 | Grade, Subject, Student, build, student_averages |
| daffodil | 6 | dtypes, report_grade, student_averages |

- In the classes, the new argument has to be passed down through each level of the nesting.
- The table filters the rows once, at the top, with `select_where()`.
- The change brought a bug into the class version. A subject with no grades since the date
  divided by zero. The fix took 3 more lines, included in the 12. The table version cannot hit
  this, because a subject with no rows is not a group.
- Both versions still load a file saved before the change. Pydantic uses the default for the
  missing date. Daffodil accepts dtypes for a column the file lacks.

## The cost of reduce

`groupby_cols_reduce()` took 0.211 s for the totals. A plain loop over the columns took 0.031 s.
The time splits like this:

| Part | Seconds |
| :-- | --: |
| `groupby_cols()`, making the groups | 0.025 |
| `reduce()` on each group | 0.172 |
| of which, building a dict for each row | 0.139 |
| a column loop that skips non-numbers, for comparison | 0.031 |

Most of the time goes to the row iterator, which builds a dict for each row before `sum_da` is
called. `reduce()` alone, on all 100,000 rows, took 0.127 s. Other ways to give it rows:

| Rows given to the function | Seconds |
| :-- | --: |
| a dict for each row, as now | 0.127 |
| the same, with `itermode='keyedlist'` | 0.106 |
| a new KeyedList for each row, made in the loop | 0.073 |
| one KeyedList, pointed at each row in turn | 0.041 |

A KeyedList reads like a dict, and `dict(row)` makes a real one. But `json.dumps(row)` and
`row | other` raise TypeError, and `isinstance(row, dict)` is False. Writing to the row changes the
table. Ray asked on 2026-10-11 to look at KeyedList rows for reduce. `for row in daf` as a generator, instead of a class with `__next__`, was about 30%
faster: 0.064 s against 0.093 s. A fast path in `reduce()` for the common sums would save more.
Both would be library changes.

## Found along the way

- `import daffodil` needed pandas and numpy, though both are optional. Fixed in ef44f07. The
  import now takes 0.09 s instead of 1.08 s.
- `daf['flagged'] = ...` selects a row with the key 'flagged', not a column, and raises KeyError.
  A new column needs `insert_col()`. This is easy to get wrong for anyone who knows pandas.
- `apply_dtypes()` makes a bool column 0 and 1, and writes it to CSV that way. A new column of
  Python bools is written as True and False. Both read back the same.

## Next steps

1. A strict load for daffodil. It would check each converted cell against the dtypes and raise on
   one that did not convert. It is off by default, for tests and development, so the fast path
   does not change. Ray agreed to this on 2026-10-11, and set the design aside for later.
   - Ray's direction: the rules come from the schema. A schema marks a column as required, so
     the check needs no separate list of columns that must not be NULL.
   - `@schemaclass` has no way to mark a column required yet. Every field must have a default.
   - A prototype in the scratchpad found all three planted faults, and took 0.055 s for 100,000
     rows of 5 columns.
2. A textbook example. Take a well-known example where classes are used, and write it again with
   daffodil.
3. A change test. Add a field and rename one, then count the edits each version needs and what
   breaks.
4. A larger example with nested records, where classes need nested models and daffodil would use
   two tables and a join.
