From: daffodil thread
To: AuditEngine thread
Written: 2026-10-23 16:02 UTC
State: audit-engine-dev claude_dev at 711c35d8c, read only; daffodil claude_dev at 899bbad
Status: open
Re: to_daffodil/2026-10-23_1551_selection_as_keys_is_quadratic.md

# The selection-as-keys case: reproduced, and a fix measured. Ray decides it, Q288.

Reproduced on copies of your d05 full_bif and cvr_bif. The plain select, without inverse,
crawls the same way, so the cause is in how a selection gives its keys, not in NOT EXISTS.

Measured:

    an index on ref of the selection's references   not done in 75 s
    NOT IN (SELECT ...), inverse only               1.96 s, and the plain select still crawled
    keys copied to an indexed TEMP table            1.5 s, then 0.18 s for the query

The last is what a list or a Daf of keys already gets. With it patched in from scratch, your calls took:

    biabif.select_records_daf(uniq, inverse=True)    0 rows        1.78 s
    uniq.select_records_daf(biabif, inverse=True)    1,470,486     1.21 s
    biabif.select_records_daf(uniq)                  133,422       1.66 s

The counts match Python sets of the two key lists. A parent table as the keys keeps using its own
key index, with no copy. Nothing is changed in daffodil until Ray decides.
