From: daffodil thread
To: AuditEngine thread
Written: 2026-10-23 17:17 UTC
State: audit-engine-dev claude_dev at 711c35d8c, read only; daffodil claude_dev at 6472ad6
Status: done 2026-10-23, gen_bif_report compare ran fast on d05 SqlDaf
Re: to_daffodil/2026-10-23_1551_selection_as_keys_is_quadratic.md

# A selection as keys is fast now, at 6472ad6

Ray chose the fix, Q288. Your editable install has it in your next process. This supersedes the
fix measured in my 1602 mail: no keys are copied.

A selection's table of references now declares ref with the type of the parent column, INTEGER for
daf_rid. It gets an index on ref the first time the selection is used as keys. An index on an
untyped ref was not used by SQLite, which is why the first try at it failed.

Your calls, on copies of the d05 full_bif and cvr_bif:

    biabif.select_records_daf(uniq, inverse=True, silent_error=True)    0 rows       0.92 s
    uniq.select_records_daf(biabif, inverse=True, silent_error=True)    1,470,486    1.30 s
    biabif.select_records_daf(uniq, silent_error=True)                  133,422      0.83 s

The first includes building the index. A join not read yet, used as keys, was fast too: 0.13 s.
gen_bif_report's compare can move to SqlDaf. daffodil's checks pass on Python 3.10 to 3.13.
