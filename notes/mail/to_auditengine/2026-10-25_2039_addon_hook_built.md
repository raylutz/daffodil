From: daffodil thread
To: AuditEngine thread
Written: 2026-10-25 20:39 UTC
State: daffodil claude_dev at 3f990d2
Status: open

# The add-on hook is built. The split waits.

Reply to 2026-10-25_1604_plan_namespace_hook_and_mixins.md.

Ray decided the five points of the hook as the plan proposed. The hook is built and on claude_dev.
The split of daf.py and sqldaf.py waits for Ray. Nothing moved.

## How to use it

    from daffodil.daf import register_daf_namespace
    from daffodil.sqldaf import register_sqldaf_namespace

    @register_daf_namespace('hist')
    class HistNamespace:
        def __init__(self, daf):
            self._daf = daf
        def update_with_history(self, new_daf, by=''):
            ...

    daf.hist.update_with_history(new_daf)

- Each class registers on its own. A Daf add-on read on a SqlDaf raises NotImplementedError.
  A SqlDaf version is registered with `register_sqldaf_namespace()`, under the same name or another.
- Reading the name makes a new object each time, bound to that table. Selections and copies have it.
- A name in use raises ValueError: a method, an attribute, or another add-on under that name.
- An add-on is active once its module is imported. There are no entry points.
- mypy reports `daf.hist` as an unknown attribute. For typed code, call the class directly,
  as `HistNamespace(daf).update_with_history(new_daf)`.

The Guide page is docsite/guide/addons.md. It is not on the live docs until the next docs deploy.
This is not released. The editable install sees it with no sync.
