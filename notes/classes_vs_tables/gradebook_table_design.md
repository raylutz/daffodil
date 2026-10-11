# The gradebook designed for tables

Written 2026-10-11 by the daffodil thread. A draft for Ray to check.

The code is `gradebook/gb_daffodil_summary.py`. The other versions are described in comparison_draft.md.

Ray asked on 2026-10-11 how the gradebook would be designed if daffodil were the plan from the
start. The flat table has no nesting. But the nesting is what made the classes fast at answering:
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
| flat table | 0.229 | 0.245 | 0.081 | 0.195 |
| grades and summary | 0.375 | 0.013 | 0.002 | 0.201 |

Building the summary is 0.129 s of the 0.375 s load. After that, the table design is faster than
the classes at every step, and the averages are the same.

Adding one grade is slower: 17.7 us, against 3.6 us for the classes. Appending the row takes 1.5
us. The rest is the two updates written as `summary[student, col] += x`, each a keyed read and a
keyed write. That is a place to look for speed in daffodil itself.
