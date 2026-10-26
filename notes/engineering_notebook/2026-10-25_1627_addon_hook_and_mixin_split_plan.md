# A plan for add-ons and for splitting daf.py: a draft for review

Started 2026-10-25. AuditEngine asked, for Ray, for a plan only: no code changes until Ray approves.
Mail: to_daffodil/2026-10-25_1604_plan_namespace_hook_and_mixins.md.
Facts below were checked at daffodil 9caa806.

## What the code is today

- daf.py has 15,643 lines. The Daf class body is 15,350 of them: 222 methods, 14,091 lines in methods.
- sqldaf.py has 6,329 lines. The SqlDaf class body is 5,819: 151 methods, 5,317 lines in methods.
- Nine Daf methods already live in helper modules and are assigned in the class body, as
  `apply_schema = daf_schema._apply_schema` and `from_md = md._from_md`. Each is documented by an
  entry that names its module path with a heading, as `::: daffodil.lib.daf_schema._apply_schema`
  with `heading: "apply_schema"`. The docs build renders them.
- Three places find Daf's methods by walking `vars(Daf)`, the class's own dict:
  - SqlDaf's blocker, sqldaf.py:6321. Every Daf method SqlDaf does not define is replaced by one that
    raises NotImplementedError.
  - The profiler, daf_profile.py:194, which wraps each method to count its calls.
  - The docs coverage test, tests/test_docsite_coverage.py:31, which checks each public method is on one page.
- daf.py makes the Dafs it returns as `Daf(...)` 183 times and as `type(self)(...)` once.
  So a subclass's result is a plain Daf, and its added methods are lost after a select or a copy.

The API docs already group the methods by topic, one page each. Lines of method code per page:

| Daf page | Lines | Methods | SqlDaf page | Lines | Methods |
| :- | -: | -: | :- | -: | -: |
| apply: apply, reduce, group | 2,897 | 37 | apply | 964 | 16 |
| select | 2,423 | 25 | select | 710 | 13 |
| modify | 2,299 | 32 | modify | 1,181 | 23 |
| convert: I/O and formats | 2,029 | 31 | read | 635 | 30 |
| structure: columns, shape | 1,291 | 26 | index: the class, file, connection | 751 | 16 |
| report | 825 | 16 | | | |
| join | 569 | 5 | | | |
| types | 485 | 4 | | | |
| index: the class itself | 185 | 1 | | | |
| private helpers | 1,088 | 45 | private helpers | 733 | 49 |

## 1. The add-on hook

### The design

One decorator registers a class of methods under a name, for Daf, SqlDaf or both:

    from daffodil.daf import register_namespace

    @register_namespace('hist')                     # Daf only, the default
    class HistNamespace:
        def __init__(self, daf):
            self._daf = daf
        def update_with_history(self, new_daf, by=''):
            ...

    daf.hist.update_with_history(new_daf)

- The name becomes a descriptor on the class. Reading `daf.hist` makes a new HistNamespace for that
  Daf each time, which took 0.31 µs. It is not cached on the Daf: a Daf copies its attributes to a
  selection, daf.py:1315, so a cached object would follow the selection, still bound to the parent.
  A prototype found that: `daf[0:1].hist` counted the parent's 2 rows, not its own 1.
- The namespace sits on the class, so every Daf has it, a selection or a copy included. The 183
  `Daf(...)` calls need no change. That is why a namespace serves here and a subclass does not.

### The points to settle, with a proposal for each

1. **Registered separately for each class.** `register_namespace(name, cls=Daf)`, and again with
   `cls=SqlDaf` for a SqlDaf version, which may use the same name. Tried in a prototype on the real
   classes, 2026-10-25:
   - With Daf only, `daf.hist` and a Daf selection's work, and `sdaf.hist` raises NotImplementedError,
     since a Daf add-on reads rows in memory, which a SqlDaf does not hold.
   - With both, a Daf gets the Daf version, and a SqlDaf and its selections get the SqlDaf version,
     since a class attribute of SqlDaf takes precedence over one it inherits from Daf.
2. **A name in use raises.** If the name is a method or an attribute of the class, or a namespace
   already registered, `register_namespace()` raises ValueError. A silent override would change
   what a core method does. Registering the same class again under the same name is allowed, so a
   module reload works.
3. **Import only, no entry points.** An add-on is active once its package is imported. No packaging
   entry point loads add-ons by themselves, so importing daffodil never runs a third party's code.
   pandas, xarray and polars work this way.
4. **Type checkers.** mypy and editors do not see a namespace registered at run time. An add-on
   that wants typed use offers its class directly, as `HistNamespace(daf).update_with_history(...)`,
   which mypy checks fully. The docs say so.
5. **Selections keep their namespaces**, since a namespace is on the class, as above.

### What it needs

- `register_namespace()` and its descriptor, about 60 lines, in a new src/daffodil/lib/daf_namespace.py.
- Tests: a registered namespace on a Daf, on a selection and a copy; blocked on a SqlDaf unless
  named; a clash with a method raises; a second registration raises.
- A Guide page, Add-ons, and an API entry for `register_namespace()`.

## 2. The split

### Two ways to split

**A. Helper modules assigned in the class body**, as the nine methods are now. Each group's methods
become functions in a module, such as src/daffodil/daf_select.py, and the class assigns them:
`select_where = daf_select._select_where`.

- Proven here: the docs render such methods, the blocker and the profiler see them, and pickling is
  unchanged, since `vars(Daf)` still holds every method.
- Costs: about 200 docs entries change to the module path with a heading. Each function annotates
  `self`. Tracebacks name the helper module.

**B. Mixin classes.** Each group becomes a class, and Daf inherits them:
`class Daf(DafSelect, DafModify, ...)`.

- Plain methods, with their own names in tracebacks, and the docs entries keep their paths, if the
  docs tool finds inherited members.
- Costs: the methods leave `vars(Daf)`. So the blocker, the profiler and the coverage test must
  walk the class's whole inheritance instead, three changes. If they were missed, a SqlDaf would
  silently inherit Daf methods that read rows it does not hold. Not proven: whether mkdocstrings
  renders `daffodil.daf.Daf.x` for a method inherited from a mixin. That must be tried first.

### The modules, the same for A or B

| Module | From the page | Lines, about |
| :- | :- | -: |
| daf.py, the core | index, structure's shape and keys, `__init__`, the dunders, the shared private helpers | 2,500 |
| daf_select.py | select | 2,400 |
| daf_modify.py | modify | 2,300 |
| daf_apply.py | apply: apply, reduce, group | 2,900 |
| daf_convert.py | convert: I/O and formats | 2,000 |
| daf_structure.py | structure: columns and reshaping | 1,300 |
| daf_report.py | report, join and types | 1,900 |

SqlDaf lines up with the same groups: sqldaf.py keeps its core, the file and the connection, about
1,500 lines, and sqldaf_select.py, sqldaf_modify.py, sqldaf_apply.py and sqldaf_read.py take about 700
to 1,200 lines each. A docs page stays one page per class and topic, as now.

### The order of the work

1. Commit and push before starting, as the rules ask for a change that touches many places.
2. A trial with one small group, join, 5 methods and 569 lines, by way B. Build the docs, run the
   coverage test, the blocker and the profiler. If mkdocstrings cannot render the inherited methods,
   use way A for the rest.
3. One group per commit, largest first, each with the full checks passing: the tests and the doctests
   on Python 3.10 to 3.13, mypy, and the strict docs build.
4. Then SqlDaf, the same way.

### What could break, and how each is checked

- **Imports.** A group module needs Daf, NULL and the types, and daf.py imports the group modules.
  The group modules import nothing from daf.py at load time, as daf_md.py and daf_schema.py do now.
  Checked by the test run.
- **SqlDaf's blocker.** Way B must walk the inheritance. Checked by `test_blocker_covers_every_daf_method_not_defined` in
  tests/test_sqldaf.py, which already checks that every Daf method is defined, inherited safely or blocked.
- **The profiler and the coverage test.** The same change in way B. Checked by their own tests.
- **Pickling.** A Daf is pickled by its class path, daffodil.daf.Daf, which does not change.
- **The doctests.** They run from each module, so each group module joins the doctest run in
  check_before_push.sh and CI. Checked by the doctest count, 238 now, which must stay the same.
- **AuditEngine.** At cca5e3432 it imports these names from daffodil.daf and daffodil.sqldaf: Daf,
  SqlDaf, NULL, KeyedList, T_klodaf, T_klosdaf, T_sdaf, T_losdaf, STORABLE_DTYPES and ReadOnlyTableError.
  Each stays importable from the same module.

## Open, for Ray

- The hook: are the five proposals right?
- The split: way A or way B, after the trial?
- Whether the split waits until after a release, so 0.9.0 ships with daf.py as it is.

## Decided and built: the hook

Ray decided the five points of the hook on 2026-10-25, as proposed:

1. Separate registration for each class. Built as two functions, `register_daf_namespace()` in
   daf.py and `register_sqldaf_namespace()` in sqldaf.py, rather than one with `cls=`.
2. A name in use raises ValueError.
3. Import only, no entry points.
4. Type checkers use the add-on class directly. Checked: mypy reports `"Daf" has no attribute "stats"`
   for `daf.stats`, and checks `Stats(daf).share(...)` fully.
5. Selections keep their namespaces.

The split waits. Ray: "wait on code org split."

What was built:

- src/daffodil/lib/daf_namespace.py holds the descriptor Namespace and `register_namespace()`, shared by both.
- The class flag `_rows_in_memory`, True on Daf and False on SqlDaf, tells the descriptor to refuse a Daf
  add-on on a SqlDaf. A SqlDaf add-on is on SqlDaf only, so a Daf never sees it.
- The names a table sets on itself count as in use. For a Daf they come from `vars(Daf())`. For a SqlDaf
  they are listed in `_SQLDAF_INSTANCE_ATTRS`, since a SqlDaf needs a file to make. A SqlDaf has no `lol`,
  so `lol` is free for a SqlDaf add-on.
- The Guide page docsite/guide/addons.md, and an entry on each API index page.
- Tests in tests/test_daf_misc.py and tests/test_sqldaf.py remove each add-on after the test. The docs
  coverage test and the blocker test walk `vars(Daf)`, so a leftover add-on would fail them.

## The first module: daf_join.py

On 2026-10-26 Ray asked that `update_with_history()` go into the file the plan would give it.
It went into src/daffodil/daf_join.py, by way A, as the first method there. Join keeps a module of
its own, rather than sharing daf_report.py with report and types. The other methods do not move
until Ray decides the split. See 2026-10-26_1409_update_with_history.md.

## Way A, with refinements (2026-10-26)

Ray, 2026-10-26, assuming way A, refined the plan:

1. Join, report and types stay in separate modules. A combined daf_report.py would be a miscellaneous
   module. daf_join.py is the model.
2. Core state stays together in daf.py where practical: construction, the row-sharing rules, the
   copying of metadata, and the invalidation of the key index. Other modules must respect these.
3. Each method is assigned in the class body by name, as `join = daf_join._join`. Nothing attaches a
   module's functions by a loop. The list of assignments documents the class.
4. Move code first. Signatures and behavior stay the same during the split. Better algorithms and a
   policy for result types come in later changes.
5. Verify the tooling after each group: the docs, mypy, the profiler, doctest discovery, and that every
   moved method is implemented on SqlDaf, safely inherited, or blocked.

On result types: replacing `Daf(...)` with `type(self)(...)` is not the fix by itself. A subclass may
need other constructor arguments or state, as SqlDaf does. If keeping the concrete type becomes a goal,
a deliberate method that builds a result is safer than a blanket replacement.

Checked on daf_join.py, the one module built by way A so far: the docs render `update_with_history`
under its Daf name, the strict build passes, mypy passes, its doctest runs in the doctest pass, the
profiler wraps it and restores it on stop, and the SqlDaf blocker raises NotImplementedError for it.

The modules, by the docs pages: daf_select.py, daf_modify.py, daf_apply.py, daf_convert.py,
daf_structure.py, daf_report.py, daf_join.py and daf_types.py, with daf.py keeping the core.
