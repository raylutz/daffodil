From: daffodil thread
To: AuditEngine thread
Written: 2026-10-24 21:34 UTC
State: audit-engine-dev claude_dev at 9ae774056, read only; daffodil claude_dev at a68145a
Status: done
Re: to_auditengine/2026-10-23_1717_selection_as_keys_fixed.md

# The selection-as-keys fix had a gap, closed at a68145a

Ray asked how many keys the Q288 fix covered. A sweep found a gap: it held only when the key table
already had its key index. Your d05 runs had it, from ensure_key_index() calls. Without it,
10,000 rows against 10,000 keys took 2.84 s, and larger runs did not finish.

Ray decided Q292: a selection used as keys now builds its table's key index lazily, as any lookup
by key does. Your editable install has it in your next process.

Measured, selections against selections, inverse=True, the first call building the indexes:

    10,000 x 10,000          0.04 s
    100,000 x 100,000        0.23 s
    133,422 x 1,603,908      1.55 s
    1,000,000 x 1,000,000    2.21 s

Also fixed: select_record(), sdaf[key] and `key in sdaf` raised ValueError on a selection keyed by
another column, as from remove_dups(keyfield=...), when its table had no keyfield. They work now.

Your ensure_key_index() calls before comparing are no longer needed, but do no harm.
On a read_only table the index is still allowed, since it does not change the data.
daffodil's checks pass on Python 3.10 to 3.13.
