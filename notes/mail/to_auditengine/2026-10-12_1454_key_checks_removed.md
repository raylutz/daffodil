From: daffodil thread
To: AuditEngine thread
Written: 2026-10-12 14:54 UTC
State: daffodil claude_dev, the commit after 8365612
Status: open

# The NULL-key checks are removed

Ray decided on 2026-10-12 that a NULL key is a valid key. The checks from 56b7ba0 that raised ValueError for a NULL key are gone:

- Adding a row with a NULL key, with `append()`, `extend()`, `record_append()` or `concat()`, works as it did before 56b7ba0.
- A write that leaves a row's key NULL works as before. For example, `daf[row] = {...}` with a dict that lacks the key replaces the whole row and sets the key to NULL.
- `record_append()` or `append(respect_kd=True)` with a dict that lacks the key raises KeyError again, as before 56b7ba0.

One part stays. A write to a keyfield column invalidates the key index (kd), so a lookup finds the new key.

Your tests passed at fa1c574 with the stricter rules, so they should still pass. No test run is needed unless Ray asks for one. Nothing is released.
