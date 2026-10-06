# Prompt for the AuditEngine thread: optional performance changes with daffodil

Written on 2026-10-06, and updated the same day for the list form of select_by_dict(). Paste the text inside the fence into a Claude session on
the EC2 machine, in the AuditEngine repository. The changes are optional. They are worth making only where a measurement on real AuditEngine work
shows a gain. The numbers below were measured on 2026-10-06 with daffodil at 95f5716 or later, on a sandbox machine, for 200,000 rows of 5 columns.
They will differ on the EC2 machine.

Measured, best of three:
- select_where(lambda row: row['name'] in a_set): 0.149 s.
- select_by_dict([{'name': n} for n in names]), 100 names: 0.009 s. About 16 times faster. Only in 0.6.0.
- select_by_dict, 50 dicts of 3 columns each: 0.073 s. With one cell that cannot be hashed in the column: 0.096 s.
- select_where(lambda row: row['grp'] == 7): 0.149 s. select_by_dict({'grp': 7}): 0.009 s. Both 0.5.13 and 0.6.0.
- A comprehension over col() followed by select_irows(): 0.025 s to 0.043 s over several runs. Both versions.
- A copy of 200,000 rows of 50 columns: copy('deep') 4.7 s, a copy with a new list for each row 0.42 s, copy('sortable') 0.003 s. The last is 0.6.0 only.

````
You are looking for optional performance gains in AuditEngine's use of daffodil. Nothing here is required. Make a change only where a measurement on real AuditEngine work shows a gain that matters, and the output is identical. Work in a single thread. Run the code to check a claim. Do not guess. Do not make a change for style.

Background. Daffodil shares data between tables, which makes it fast, so the way a call is written can change its time a great deal. A Daf cell can hold any Python object, and the dtypes do not limit it, so do not assume that a column holds only one type. The machine runs daffodil 0.5.13 today, and 0.6.0 is coming but is not released. So there are TWO tiers of change, and they must not be mixed.

Tier 1 works on BOTH 0.5.13 and 0.6.0 and gives the same output on both. These go on the branch daffodil-perf, which you start from the branch daffodil-0.6.0-prep if it exists, and otherwise from the current branch. They may be deployed while the machine runs 0.5.13. Do not use in tier 1 any parameter or behavior that only 0.6.0 has: not ignore_extra_keys, not the copy levels shallow, sortable, editable or the COPY_ constants, and not a list of dicts in select_by_dict().

Tier 2 needs 0.6.0 and cannot run on 0.5.13, which does not accept a list in select_by_dict(). These go on a second branch, daffodil-perf-0.6.0, which you start from daffodil-perf AFTER the tier 1 commits are done. Start the message of each tier 2 commit with "requires daffodil 0.6.0". Test tier 2 only in venv_new2, and say clearly in the report that this branch must not be deployed to the machine until it runs 0.6.0.

Limits.
- Work only on those two local branches, one commit for each change, with a message that names the call site and the measured gain. Do not push, merge or tag anything.
- Do not switch, pull or change the sibling daffodil repository, and do not change the installed daffodil. Use the scratch folder and the virtual environments of the earlier reviews: venv_old with 0.5.13, and venv_new2 with origin/main. Rebuild venv_new2 from a fresh export of origin/main, at 95f5716 or later, if it is older, because select_by_dict() takes a list only from that commit.
- Do not run a pipeline stage on a real job in a way that rewrites its files. Reproduce on copies of the files, in the scratch folder.
- If an item says to ask, stop and ask me.

Step 1. Find where the time goes. Pick the pipeline stages and tool ops that take longest on the largest local jobs, and profile them with cProfile, or with timing around the stages if cProfile is too heavy. Report the ten functions or lines with the most time, and say which of them are daffodil calls or loops over a Daf. Only work on those. If nothing in the list is a daffodil call, say so and stop, and report what is slow.

Step 2. For each hot spot that is one of the cases below, make the change, and check it.

A. Equality tests in select_where(), tier 1. A call such as select_where(lambda row: row['col'] == value), with one or more equalities joined by and, can be select_by_dict({'col': value}). It compares the cells by position, without a row object for each row, and measured about 16 times faster on 200,000 rows. Check these before you change a call.
   - select_by_dict() compares the cell as stored. A lambda that converts first, such as int(row['n']) == 5, must stay as it is, because the text '5' does not match the number 5.
   - inverse=True is the negation of the whole match, not != on each field. A lambda with != or with or must stay, or be rewritten with care.
   - select_where() raises KeyError for a column that does not exist. select_by_dict() returns no rows. If the column can be missing, keep the lambda or check the column first.
   - The result shares its rows with the table it came from, as select_where() does. A change of a cell in the result changes the original in both.
   - A value in the dict is only compared for equality. Never give a list or a set as a value in order to say "any of these". It matches only a cell that equals that list or set, and finds nothing otherwise, with no error. Use B for that.
   The earlier review found 19 select_where() calls that test only equality. Start there. Show the same rows from both forms on real data, then the time of each.

B. Membership tests.
   Tier 1: select_where(lambda row: row['col'] in values) builds a row object for each row. Build a set once, before the call, and for a large table use d.select_irows([i for i, v in enumerate(d.col('col')) if v in values_set]). It measured 0.025 s to 0.043 s against 0.149 s. Do not use a list of bools as a mask.
   Tier 2: in 0.6.0, select_by_dict() takes a list of dicts, and a row matches if it matches any one of them. select_by_dict([{'col': v} for v in values]) selects the rows whose cell is any of the values, and it measured 0.009 s. It can also select by several columns together, for example select_by_dict([{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]), which is a composite membership that tier 1 does with a comprehension over two columns. Rules: an empty list matches no row, and inverse=True then matches all rows; an empty dict inside the list matches every row; a dict that names a column that does not exist matches no row, and the others still apply; an item that is not a dict raises TypeError; the comparison is == , so 1, True and 1.0 are equal, and a cell that cannot be hashed, such as a list, is compared with == too and never raises. Use it for each hot membership test that tier 1 would have changed, and show that the rows are identical to the tier 1 result and to the original, in the same order.

C. Deep copies, tier 1. Many AuditEngine calls use copy(deep=True), which measured 4.7 s for 200,000 rows of 50 columns, against 0.42 s for a copy that makes a new list for each row. Where a deep copy is made only so that rows can be added, sorted or dropped, or so that columns can be added, a cheaper copy is enough. On 0.5.13 and on 0.6.0 you can build one with Daf(lol=[list(row) for row in d.lol], cols=d.columns(), keyfield=d.keyfield, dtypes=d.dtypes). Do this only where no cell holds a list, a dict or a set that is changed in place, because those cells would then be shared. Check the cells of the real data. Show that the output is identical, and the time of each.

D. Conversions that make a dict for each row, tier 1. Code that turns every row into a dict, with to_lod(), iterating with a dict for each row, or building a dict only to read two values, can often read the row as a list, or use iter_klist(). Look for loops over a Daf that touch only a few columns, and for to_lod() calls whose result is only looped over. Change one only if the loop is a hot spot from step 1.

E. Appends in a loop, tier 1. A loop that appends one row at a time and then looks up by key can rebuild the key index after every append. Look for loops of append() on a Daf that has a keyfield, with a lookup by key inside the loop. Building the rows as a list first and making the Daf once is often faster. Change one only if it is a hot spot from step 1.

F. Anything else in the list from step 1 that is a daffodil call and is not above. Describe it, with the measurement and a suggested change, and ask me before you change it.

Check each change. After every commit, run the full AuditEngine test suite, with the same command and data as before, and compare with the saved results of the last review, test by test. Run tier 1 commits in venv_old and in venv_new2. Run tier 2 commits in venv_new2 only. Nothing may differ, apart from tests that you add. For each change, show that the output is identical on real data from the largest jobs, with a hash or a cell-by-cell comparison, and give the time before and after, best of three, on each version that it runs on. Drop any change whose gain is less than 10 percent of its stage, or less than a quarter of a second, and say that you dropped it.

Add a test with each change where a test is practical, with data that includes the edge cases named above. A tier 2 test must be marked to skip on a daffodil without the list form, so that the suite still runs on 0.5.13.

Report, in this order: the profile of step 1; a table of the test results for each branch and each version; one section for each change that you kept, with the tier, the file and lines, the diff, the check and its output, and the times before and after; the changes that you dropped or did not make, each with the reason; and the later options that you did not try, with an estimate of the gain. End with two lines: whether the branch daffodil-perf can be deployed to the machine while it still runs 0.5.13, and that daffodil-perf-0.6.0 must wait until the machine runs 0.6.0. Say what you are unsure about. Delete nothing. Leave the scratch folder in place.
````
