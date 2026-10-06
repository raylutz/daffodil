# Prompt for the AuditEngine thread: optional performance changes with daffodil

Written on 2026-10-06. Paste the text inside the fence into a Claude session on the EC2 machine, in the AuditEngine repository.
The changes are optional. They are worth making only where a measurement on real AuditEngine work shows a gain. The numbers below were measured on
2026-10-06 with daffodil at 002f7cb, on a sandbox machine, for 200,000 rows of 5 columns, and they will differ on the EC2 machine.

Measured, best of three:
- select_where(lambda row: row['grp'] == 7): 0.149 s. select_by_dict({'grp': 7}): 0.010 s. About 15 times faster.
- Two equalities: 0.152 s with select_where, and 0.011 s with select_by_dict.
- select_where(lambda row: row['name'] in a_set): 0.156 s. A comprehension over col() followed by select_irows(): 0.043 s.
- A copy of 200,000 rows of 50 columns: copy('deep') 4.7 s, copy('editable') 0.42 s, copy('sortable') 0.003 s.

````
You are looking for optional performance gains in AuditEngine's use of daffodil. Nothing here is required. Make a change only where a measurement on real AuditEngine work shows a gain that matters, and the output is identical. Work in a single thread. Run the code to check a claim. Do not guess. Do not make a change for style.

Background. Daffodil shares data between tables, which makes it fast, so the way a call is written can change its time a great deal. The machine runs daffodil 0.5.13 today, and 0.6.0 is coming. Every change must work on BOTH versions and give the same output on both. Do not use any parameter that only 0.6.0 has, such as ignore_extra_keys, and do not use the copy levels shallow, sortable, editable or the COPY_ constants, which only 0.6.0 has. Use copy(deep=True) or a plain Python copy where a copy is needed on 0.5.13. If a gain needs a 0.6.0 feature, do not make the change. Describe it in the report as a later option.

Limits.
- Work on a new branch of AuditEngine, started from the branch daffodil-0.6.0-prep if it exists, and otherwise from the current branch. One commit for each change, with a message that names the call site and the measured gain. Do not push, merge or tag anything.
- Do not switch, pull or change the sibling daffodil repository, and do not change the installed daffodil. Use the scratch folder and the virtual environments of the earlier reviews: venv_old with 0.5.13, and venv_new2 with origin/main.
- Do not run a pipeline stage on a real job in a way that rewrites its files. Reproduce on copies of the files, in the scratch folder.
- If an item says to ask, stop and ask me.

Step 1. Find where the time goes. Pick the pipeline stages and tool ops that take longest on the largest local jobs, and profile them with cProfile, or with timing around the stages if cProfile is too heavy. Report the ten functions or lines with the most time, and say which of them are daffodil calls or loops over a Daf. Only work on those. Do not try to speed up anything that is not in that list. If nothing in the list is a daffodil call, say so and stop, and report what is slow.

Step 2. For each hot spot that is one of the cases below, make the change, and check it.

A. Equality tests in select_where(). A call such as select_where(lambda row: row['col'] == value), with one or more equalities joined by and, can be select_by_dict({'col': value}). It compares the cells by position, without a row object for each row, and measured about 15 times faster on 200,000 rows. Check four things before you change a call.
   - select_by_dict() compares the cell as stored. A lambda that converts first, such as int(row['n']) == 5, must stay as it is, because the text '5' does not match the number 5.
   - inverse=True is the negation of the whole match, not != on each field. A lambda with != or with or must stay, or must be rewritten with care.
   - select_where() raises KeyError for a column that does not exist. select_by_dict() returns no rows. If the column can be missing, keep the lambda or check the column first.
   - The result shares its rows with the table it came from, as select_where() does. A change of a cell in the result changes the original in both.
   The earlier review found 19 select_where() calls that test only equality. Start there. Show the same rows from both forms on real data, then the time of each.

B. Membership tests. select_where(lambda row: row['col'] in values) builds a row object for each row. In 0.6.0 select_by_dict([{'col': v} for v in values]) does this in about 0.009 s for 200,000 rows and 100 values, but 0.5.13 does not accept a list, so it can only be a later option. Report each such call, with its file and line, as a later option, and do not use it now. When values is a list, build a set once before the call, so that each test is fast. For a large table, a comprehension over the column followed by select_irows(), as in d.select_irows([i for i, v in enumerate(d.col('col')) if v in values_set]), measured about 3.5 times faster than the lambda. It is worth it only for a large table. Do not use a list of bools as a mask.

C. Deep copies. Many AuditEngine calls use copy(deep=True), which measured 4.7 s for 200,000 rows of 50 columns, against 0.42 s for a copy that makes a new list for each row. Where a deep copy is made only so that rows can be added, sorted or dropped, or so that columns can be added, a cheaper copy is enough. On 0.5.13 and on 0.6.0 you can get one by building a new Daf from copied rows, for example Daf(lol=[list(row) for row in d.lol], cols=d.columns(), keyfield=d.keyfield, dtypes=d.dtypes). Do this only where no cell holds a list or a dict that is changed in place, because those cells would then be shared. Check the cells of the real data. Show that the output is identical, and the time of each.

D. Conversions that make a dict for each row. Code that turns every row into a dict, with to_lod(), iterating with a dict for each row, or building a dict only to read two values, can often read the row as a list, or use iter_klist(). Look for loops over a Daf that touch only a few columns, and for to_lod() calls whose result is only looped over. Change one only if the loop is a hot spot from step 1.

E. Appends in a loop. A loop that appends one row at a time and then looks up by key can rebuild the key index after every append. Look for loops of append() on a Daf that has a keyfield, with a lookup by key inside the loop. Building the rows as a list first and making the Daf once is often faster. Change one only if it is a hot spot from step 1.

F. Anything else in the list from step 1 that is a daffodil call and is not above. Describe it, with the measurement and a suggested change, and ask me before you change it.

Check each change. After every commit, run the full AuditEngine test suite in venv_old and in venv_new2, with the same command and data as before, and compare with the saved results of the last review, test by test. Nothing may differ, apart from tests that you add. For each change, show that the output is identical on real data from the largest jobs, with a hash or a cell-by-cell comparison, on both versions, and give the time before and after on both versions, best of three. Drop any change whose gain is less than 10 percent of its stage, or less than a quarter of a second, and say that you dropped it.

Add a test with each change where a test is practical, for the helper or the call, with data that includes the edge cases named above.

Report, in this order: the profile of step 1; a table of the test results; one section for each change that you kept, with the file and lines, the diff, the check and its output, and the times before and after; the changes that you dropped or did not make, each with the reason; and a short list of later options that need daffodil 0.6.0, with your estimate of the gain. End with one line on whether the branch can be deployed to the machine while it still runs 0.5.13, and what you are unsure about. Delete nothing. Leave the scratch folder in place.
````
