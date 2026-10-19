# acre_strata_check.py: ACRE's stratify and select on a Daf and on a SqlDaf of the same CVR rows.
#
# Runs AuditEngine's own acre_stratify() and acre_select_from_strata(), read-only.
# logs.sts() is replaced first, so nothing is written under AuditEngine.
#
# Usage, with AuditEngine's venv python -I:
#   python acre_strata_check.py <cvr_votes csv> <contests_dod.json> <scratch dir>
# A fourth argument runs only the SqlDaf side, so its peak memory is its own.

import os
import sys
import json
import time
import resource

sys.path.insert(0, '/home/ubuntu/daffodil_sql/src')
sys.path.insert(1, '/home/audit-engine-dev')
# AuditEngine's venv needs pysqlite3-binary. Until it has it, PYSQLITE3_SITE names a site-packages that does.
if os.environ.get('PYSQLITE3_SITE'):
    sys.path.append(os.environ['PYSQLITE3_SITE'])

import pyjson5                                          # noqa: E402
from daffodil.daf import Daf                            # noqa: E402
from daffodil.sqldaf import SqlDaf                      # noqa: E402
from utilities import logs                              # noqa: E402
logs.sts = logs.stsloc = lambda *args, **kwargs: None
from utilities import acre_utils                        # noqa: E402

csv_path, contests_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
contests_dod = json.load(open(contests_path))
argsdict = {'acre_mode': 'hexstyle', 'include_writeins': False, 'include_uncontested_contests': False, 'random_seed': 1}


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def cvr_daf_of(path: str) -> Daf:
    """ The columns ACRE's where() reads, with the votes decoded, as acre_build_cvr_daf() leaves them. """
    daf = Daf.from_csv(path, keyfield='ballot_id')
    daf = daf.select_cols(['ballot_id', 'batchid', 'is_bmd', 'hexstyle', 'style', 'concise_votes'])
    daf.rename_cols({'style': 'style_num'})
    daf.apply_in_place(lambda row: {'concise_votes': pyjson5.loads(row['concise_votes']), 'is_bmd': int(row['is_bmd'] or 0)})
    return daf


SQL_ONLY = len(sys.argv) > 4
if not SQL_ONLY:
    # The Daf, as ACRE runs today.
    start = time.perf_counter()
    cvr_daf = cvr_daf_of(csv_path)
    load_secs = time.perf_counter() - start
    start = time.perf_counter()
    strat_daf = acre_utils.acre_stratify(argsdict, cvr_daf, contests_dod)
    strat_secs = time.perf_counter() - start
    start = time.perf_counter()
    sel_daf = acre_utils.acre_select_from_strata(argsdict, strat_daf)
    sel_secs = time.perf_counter() - start
    print(f'Daf:    load {load_secs:.1f} s, stratify {strat_secs:.1f} s, select {sel_secs:.1f} s, '
          f'{len(strat_daf["strata_dodaf"])} strata, {len(sel_daf["selected_ballot_ids_ls"])} ballots, peak {rss_mb():.0f} MB')

# The SqlDaf. The votes stay text, and where() decodes them.
SqlDaf.db_path = os.path.join(out_dir, 'acre.db')
if os.path.exists(SqlDaf.db_path):
    os.remove(SqlDaf.db_path)
start = time.perf_counter()
cvr_sdaf = SqlDaf.from_csv(csv_path, name='cvr_votes', keyfield='ballot_id')
cvr_sdaf.rename_cols({'style': 'style_num'})
cvr_sdaf.apply_dtypes(dtypes={'is_bmd': int}, cols=['is_bmd'])
load_secs = time.perf_counter() - start

lookup = acre_utils._build_contest_lookup(contests_dod)


def where(row):
    return acre_utils._ballot_strata_keys(
            concise_votes                   = pyjson5.loads(row['concise_votes']),
            is_bmd                          = int(row['is_bmd']),
            hexstyle                        = row['hexstyle'],
            style_num                       = row['style_num'],
            acre_mode                       = 'hexstyle',
            lookup                          = lookup,
            include_writein_strata          = False,
            include_uncontested_contests    = False,
            ) or None


start = time.perf_counter()
groups_kl = cvr_sdaf.group_where(where, cols=['concise_votes', 'is_bmd', 'hexstyle', 'style_num'])
strat_secs = time.perf_counter() - start
strat_sdaf = dict(acre_mode='hexstyle', lookup=lookup, strata_dodaf=groups_kl, where=where)
start = time.perf_counter()
sel_sdaf = acre_utils.acre_select_from_strata(argsdict, strat_sdaf)
sel_secs = time.perf_counter() - start
print(f'SqlDaf: load {load_secs:.1f} s, stratify {strat_secs:.1f} s, select {sel_secs:.1f} s, '
      f'{len(groups_kl)} strata, {len(sel_sdaf["selected_ballot_ids_ls"])} ballots, peak {rss_mb():.0f} MB')

if not SQL_ONLY:
    print('same strata keys:', list(strat_daf['strata_dodaf']) == list(groups_kl.keys()))
    print('same selected ballots, in order:', sel_daf['selected_ballot_ids_ls'] == sel_sdaf['selected_ballot_ids_ls'])
    print('same reasons:', sel_daf['selection_reason_by_ballot_id'] == sel_sdaf['selection_reason_by_ballot_id'])
