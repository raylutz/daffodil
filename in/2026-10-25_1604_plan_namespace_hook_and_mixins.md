From: AuditEngine thread
To: daffodil thread
Written: 2026-10-25 16:04 UTC
State: audit-engine-dev claude_dev at b40a8dc59, daffodil claude_dev at 9caa806
Status: done 2026-10-25, the hook is built at 3f990d2. The split waits for Ray.

# A plan for add-ons and for splitting daf.py, planning only

Ray asks for a planning cycle. No code is to change. The result is a plan for Ray to review.

## The ask

Write a plan that covers two things.

1. A hook that lets a third party add methods to Daf and SqlDaf without a change to the core package.
2. A split of the Daf and SqlDaf classes into mixin modules, so each source file is smaller.

## 1. The add-on hook

The plan should say how a separate package adds methods, and how others can use them as an add-on.
The usual way in pandas, xarray and polars is a registered namespace:

    @register_daf_namespace('hist')
    class HistNamespace:
        def __init__(self, daf):
            self._daf = daf
        def update_with_history(self, new_daf, by=''):
            ...

    daf.hist.update_with_history(new_daf)

Points for the plan to settle:
- One hook for Daf and SqlDaf, or one each.
- What happens when a name is already a method or a registered namespace.
- Whether installed add-ons are found by a packaging entry point, or only by import.
- How mypy and editors see a namespace.
- Whether a selection of a Daf keeps its namespaces.

Subclassing does not serve here today. daf.py builds the Dafs it returns as Daf(...) 183 times, and as type(self)(...) once.
So a result of a subclass is a plain Daf, and its added methods are gone after the first select or copy.

## 2. The mixin split

daf.py has 15,643 lines and sqldaf.py 6,329, at 9caa806.
The plan should propose:
- Which groups of methods become mixin modules, such as selection, setting values, I/O, types, reductions and joins.
- The rough size of each module after the split.
- What stays in the core class.
- How SqlDaf's overrides line up with the same groups.
- How the published docs, built from docstrings, keep one page per class.
- The order of the work, and what could break, such as imports, pickling or the doctest run.

## Not part of this

No method moves, and no hook is built, until Ray approves the plan.
