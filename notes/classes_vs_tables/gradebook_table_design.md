# The gradebook designed for tables

Written 2026-10-11 by the daffodil thread. A draft for Ray to check.

The code is `gradebook/gb_daffodil_summary.py`. The other versions are described in comparison_draft.md.

Ray asked on 2026-10-11 how the gradebook would be designed if daffodil were the plan from the
start. The daffodil table has no nesting. But the nesting is what made the classes fast at answering:
it groups the grades once, at load. A table design can do the same with a second table.

`gb_daffodil_summary.py` keeps two tables:

- the grades, one row per grade, which is what is saved as CSV
- a summary, one row per student, keyed by student, with two columns for each subject: the
  weighted total and the total weight

The summary is built once at load, in one pass over four columns. Adding a grade appends a row to
the grades and adds to two cells of the summary. A question then reads the summary, with no
grouping.

| | load | student averages | math average | dump |
| :-- | --: | --: | --: | --: |
| classes | 0.649 | 0.049 | 0.020 | 0.130 |
| daffodil table | 0.229 | 0.245 | 0.081 | 0.195 |
| grades and summary | 0.375 | 0.013 | 0.002 | 0.201 |

Building the summary is 0.129 s of the 0.375 s load. After that, the table design is faster than
the classes at every step, and the averages are the same.

Adding one grade was slower at first: 17.7 us, against 3.6 us for the classes. Appending the row
took 1.5 us. The rest was the two updates written as `summary[student, col] += x`, each a keyed
read and a keyed write. Commit 476dd81 gave one-cell reads and writes a fast path. Adding a grade
now takes 4.8 us.

## A second design: an indirect column

Ray asked on 2026-10-11 whether daffodil could hold the nested part in an indirect column: a
column whose cells are dicts. `gradebook/gb_daffodil_indirect.py` keeps one row per student. The
grades column holds a dict that maps each subject to a list of [score, weight] pairs. The file
then names each student once, and each subject once for each student, as the nested JSON does.

Daffodil saves a dict cell as its Python text, and reads it back with `ast.literal_eval`. A second
pair of functions, `dump_json()` and `load_json()`, keeps compact JSON in the cell instead. The
two cells look like this:

```
s00000,"{'math': [[95, 0.1], [45, 0.1], ...], 'gym': [...]}"
s00000,"{""math"":[[95,0.1],[45,0.1],...],""gym"":[...]}"
```

200,000 grades, Python 3.13.1, 2026-10-11. Sizes gzipped at level 6, times in seconds:

| | bytes | gzipped | load | student averages | math average | dump |
| :-- | --: | --: | --: | --: | --: | --: |
| pydantic, nested JSON | 6,483,290 | 518,667 | 0.651 | 0.048 | 0.020 | 0.129 |
| daffodil table | 4,163,306 | 656,411 | 0.239 | 0.238 | 0.074 | 0.195 |
| indirect column, as daffodil saves it | 2,853,292 | 447,794 | 1.504 | 0.073 | 0.021 | 0.212 |
| indirect column, JSON in the cell | 2,513,292 | 438,656 | 0.157 | 0.068 | 0.018 | 0.219 |

All four give the same averages, also after a grade is added.

- With JSON in the cell, the indirect column is the smallest file, raw and gzipped, and the
  fastest load. Its answers are almost as fast as the nested classes.
- As daffodil saves it now, the load takes 1.5 s. Almost all of that is `ast.literal_eval`,
  which reads each cell safely but slowly. Python text is also bigger than compact JSON.
- Reading JSON cells takes custom code: a `json.loads` on the column after the load, and a
  `json.dumps` before the save. That is most of the 140 load/save tokens this version needs.
  If daffodil wrote and read dict cells as JSON itself, that code would go away.
