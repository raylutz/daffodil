# Design from the general case, not from the case at hand

On 2026-10-21 a small mypy fix turned into a redesign of the sparse sum, Q265 and Q266.
The first three answers given to Ray were wrong in their framing, not in their numbers.
Every measurement was correct. Each was taken on one call that was not representative.
This entry records how that happened, so the same mistake is not made again.
The technical result is in 2026-10-21_1434_sparse_sum_loop.md.

## What happened

1. AuditEngine's mypy flagged one line of the port, notes/sql_backing/sdaf_variants.py:776.
   It passed a set as `cols` to `SqlDaf.reduce()`, which was typed for a list.
2. The question was taken as given: which type should `cols` be?
   The port's own call was measured with a list, a set and a dict, on its 385 columns.
   The answer was to widen the type, Q265. The design of the call was not questioned.
3. The 385 names were called "the indirect columns". `indirect_col` is the column that holds the dict.
   `cols` is the columns to include in the sum. Mixing the terms confused the review.
4. Ray asked why the sum loops over every key of each row, and not over `cols`.
   The port's call was measured again. With 385 names in `cols` and about 30 keys in a row,
   the row loop won. So the answer was to keep it, the Original. That generalized from the one case.
5. Ray pointed out that `cols` is almost always shorter than the dicts.
   Only then were the callers surveyed. Every AuditEngine caller passes no `cols` at all.
   The port's `cols` was nearly every key of the job, so it should not have passed `cols`.
6. The design that came out has three cases, chosen once, before the loop.
   No key is tested in the loop. The keyed test, the subject of Q265, turned out never to be the fastest choice.

## Why it went wrong

- **The case at hand was taken as the workload.** The port is one caller. Its parameters were
  measured as if they described every use of the method. They described the least typical use.
- **The question was answered as asked.** "Which type for `cols`" assumed that passing `cols` was right.
  A narrow question invites a narrow fix: list to set. The right fix was to drop the argument.
- **Measuring came before understanding.** The timings were precise. They answered which of
  three containers is fastest, not what `cols` is for or how the loop should be built.
- **One point was measured, not a range.** A sweep of `cols` from 1 to 359 names shows the
  crossover at once. One point at 385 hides it.
- **The function's own terms were not used.** Naming the parameters wrongly hid what each one does.
- **Reviews came before the work was done.** Each review offered options around the narrow framing.
  Ray had to supply the design.

## How the work is done from now on

Before changing or tuning a method, even for a one-line fix:

1. **Say what each parameter is for, in the method's own terms.** Use its names: `cols`, `indirect_col`.
2. **Survey the callers.** Grep daffodil and AuditEngine, read-only, and list how each one calls it.
   Note the common case and the rare ones. Record the survey in the notebook entry.
3. **Ask whether the case at hand is typical.** Compare its arguments with the other callers.
   A port, a test or a benchmark is one caller, not the workload.
4. **Question the call, not only the line.** Ask whether the argument should be passed at all,
   and whether the method is the right one. A type error may point at a wrong call.
5. **Measure across the range of the parameter.** Sweep sizes on real data, and on synthetic data
   when the real data covers only part of the range. Find where the choice changes.
6. **Make decisions once, outside the loop.** A loop over rows should be plain. Choose its form before it starts.
7. **Bring a design to review, not a question.** It should cover the general case and every
   caller, with measured results and tests. Present options only after that work is done.

These rules are provisional. Ray, 2026-10-21: keep them local to daffodil for now, not in the common rules.
He is still thinking them over, and is not sure they capture the issues well. Expect them to be revised.

The research-first rule in CLAUDE.md points here.
The research-first rule exists because of a sum that `groupsum_daf()` already did, 2026-10-17.
This entry adds the other half: research the callers and the general case, not only the existing methods.
