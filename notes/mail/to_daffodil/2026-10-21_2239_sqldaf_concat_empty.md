From: AuditEngine thread
To: daffodil thread
Written: 2026-10-21 22:39 UTC
State: audit-engine-dev claude_dev at dcfe453cc, daffodil claude_dev at 1688272
Status: done 2026-10-21, aa01aa2 (Q272)

# SqlDaf.concat(): accept an empty table with no columns, as Daf.concat() does

Ray approved this on 2026-10-21.

## The ask

SqlDaf.concat(other) should add nothing, and raise nothing, when other has no rows and no columns.
Daf.concat() already does that.

## How it shows

sdaf_variants.py, step 16, in locate_contest_variants_sdaf():

    missing_sdaf = variant_ids_sdaf.select_records_daf(partial_sdaf, inverse=True, silent_error=True)
    shells_sdaf  = missing_sdaf.apply(lambda row_da: {...})
    partial_sdaf.concat(shells_sdaf)

When every variant ballot has a contest that agrees, missing_sdaf is empty.
apply() on it gives a SqlDaf with no columns, and concat() raises:

    KeyError: "concat: columns mismatch: this one has ['ballot_id', 'concise_votes', 'num_contests'], the other []."

It stops find_variants on WI_Dane_20201103_functest, run by AuditEngine's --op find_variants.

## A short reproduction

    a = Daf(cols=['ballot_id', 'v'], lol=[['b1', 1]])
    e = Daf(cols=['ballot_id'], lol=[]).apply(lambda r: {'ballot_id': r['ballot_id'], 'v': 0})
    a.concat(e)                     # works, a keeps 1 row

    s  = SqlDaf('t', cols=['ballot_id', 'v'], lol=[['b1', 1]])
    m  = SqlDaf('m', cols=['ballot_id'], lol=[])
    sh = m.apply(lambda r: {'ballot_id': r['ballot_id'], 'v': 0})
    s.concat(sh)                    # KeyError
