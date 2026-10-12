From: daffodil thread
To: AuditEngine thread
Written: 2026-10-12 15:23 UTC
State: daffodil claude_dev, the commit after 57145f8
Status: done 2026-10-12, noted, no test run needed

# Keys: writes are plain again, and NULL keys are allowed

This replaces 2026-10-12_1454_key_checks_removed.md. Ray decided on 2026-10-12:

- A NULL key is a valid key. The ValueError checks from 56b7ba0 are gone. Adding or writing rows with NULL keys works as it did before 56b7ba0.
- A write with `daf[...] = ...`, `set_irows_icols()`, `assign_record_irow()` or `update_record_irow()` does not touch the key index (kd), as before 56b7ba0. If you change a key that way, the old key still finds the row. Call `set_keyfield()` with the same keyfield afterwards.
- A method that rewrites a whole keyfield column invalidates the kd, so a lookup finds the new key. These are `set_icol()`, `set_icol_irows()`, `set_col_irows()`, `assign_icol()`, `assign_col()`, `apply_dtypes()`, `flatten()` and `strip()`.

All of this is as before 56b7ba0 or looser than it, so your tests should still pass. No test run is needed unless Ray asks for one. Nothing is released.
