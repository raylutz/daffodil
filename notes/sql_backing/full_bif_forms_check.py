# full_bif_forms_check.py: run AuditEngine's full_bif call forms on a Daf and on a SqlDaf, and compare.
#
# Usage, with the daffodil_sql venv: python full_bif_forms_check.py <full_bif.csv> <scratch dir>

import os
import sys
import time
import traceback

sys.path.insert(0, '/home/audit-engine-dev')
from models import schema                           # noqa: E402

from daffodil.daf import Daf                        # noqa: E402
from daffodil.sqldaf import SqlDaf                  # noqa: E402

csv_path, out_dir = sys.argv[1], sys.argv[2]
SqlDaf.db_path = os.path.join(out_dir, 'full_bif.db')
if os.path.exists(SqlDaf.db_path):
    os.remove(SqlDaf.db_path)

start = time.perf_counter()
daf = Daf.from_csv(csv_path, keyfield='ballot_id')
daf.attach_schema(schema.BifSchema)
# The list columns stay text, as AuditEngine's utils.file_paths_of_row() reads them.
flat_dtypes = {col: (str if dt in (list, dict) else dt) for col, dt in daf.dtypes.items()}
print('left as text:', [c for c, dt in daf.dtypes.items() if dt in (list, dict)])
daf.apply_dtypes(dtypes=flat_dtypes, silent_error=False)
print(f'Daf load and apply_dtypes: {time.perf_counter() - start:.1f} s, {len(daf)} rows')

start = time.perf_counter()
sdaf = SqlDaf.from_csv(csv_path, name='full_bif', keyfield='ballot_id')
sdaf.attach_schema(schema.BifSchema)
sdaf.apply_dtypes(dtypes=flat_dtypes, silent_error=False)
print(f'SqlDaf load and apply_dtypes: {time.perf_counter() - start:.1f} s, {len(sdaf)} rows')

ids = daf.col('ballot_id')
some_ids = [ids[5], ids[500], ids[50000]]


def rows(x):
    """ The rows of a result as a list of dicts, for either kind. """
    if isinstance(x, SqlDaf):
        x = x.to_daf()
    return x.to_lod()


def same(label, func):
    try:
        a = func(daf)
    except Exception as exc_info:
        print(f'DAF FAILS  {label}: {type(exc_info).__name__}: {exc_info}')
        return
    try:
        b = func(sdaf)
    except Exception as exc_info:
        print(f'SQL FAILS  {label}: {type(exc_info).__name__}: {str(exc_info)[:160]}')
        return
    if isinstance(a, Daf) or isinstance(b, SqlDaf):
        a, b = rows(a), rows(b)
    ok = a == b
    if not ok and isinstance(a, list) and isinstance(b, list):
        ok = sorted(map(repr, a)) == sorted(map(repr, b))
        label += ' (as sets)' if ok else ''
    size = len(a) if hasattr(a, '__len__') else a
    print(f'{"same" if ok else "DIFFERS"}       {label}   [{size if isinstance(size, int) else type(a).__name__}]')
    if not ok:
        print('   daf:', repr(a)[:200]); print('   sql:', repr(b)[:200])


# ballot_utils: a list of keys
same('x[ballot_id_ls]',                     lambda x: x[some_ids])
# bif_utils.get_ballot_full_bif_dict: Q175
same('x[id].apply_dtypes().to_dict()',      lambda x: x[some_ids[0]].apply_dtypes().to_dict())
# bmd_map_report, votes_extractor, gentemplates, writein_inspection, bif_report
same('select_where is_bmd',                 lambda x: x.select_where(lambda row: bool(int(row['is_bmd'] or 0))))
same('select_where not corrupted, not bmd', lambda x: x.select_where(lambda row: not row.get('corrupted', 0) and not row.get('is_bmd', 0)))
same('select_where archive_basename',       lambda x: x.select_where(lambda row: bool(row['archive_basename'])))
same('select_where card codes',             lambda x: x.select_where(lambda row: bool(row['card_code'] or row.get('cvr_card_code'))))
# parse_utils
same("select_by_dict({'is_bmd': '1'})",     lambda x: x.select_by_dict({'is_bmd': '1'}))
same("select_by_dict({'is_bmd': 1})",       lambda x: x.select_by_dict({'is_bmd': 1}))
# bif_report
same("col('archive_basename', unique=True)", lambda x: x.col('archive_basename', unique=True))
same('columns()',                           lambda x: x.columns())
same('daf_valuecount(cols)',                lambda x: x.daf_valuecount(cols=['precinct', 'is_bmd', 'corrupted']))
same("groupby('corrupted') keys and sizes", lambda x: {k: len(v) for k, v in x.groupby('corrupted').items()})
same("groupby: get(0).append(get(''))",     lambda x: (lambda g: g.get(0, Daf()).append(g.get('', Daf())))(x.groupby('corrupted')))
# bmd_map_report
same('id in x.keys(); x[id, col].to_value()', lambda x: [x[i, 'hexstyle'].to_value() for i in some_ids if i in x.keys()])
same('id in x; select_record(id)[col]',     lambda x: [x.select_record(i)['hexstyle'] for i in some_ids if i in x])
# cmpcvr_report
same('x[ids][0:2].to_md()',                 lambda x: x[some_ids][0:2].to_md())
same('select_krows(ids)',                   lambda x: x.select_krows(krows=some_ids, inverse=False, silent_error=True))
# evalvotes, votes_extractor
same('select_records_daf(ids)',             lambda x: x.select_records_daf(some_ids))
# recompute_is_bmd, imagematch: read every row
same('for row in x: count bmd',             lambda x: sum(1 for row in x if row['is_bmd']))
# bif_utils.fix_full_bif_is_bmd: change cells by key, then save
def set_cells(x):
    x[some_ids[0], 'is_bmd'] = 1
    x[some_ids[0], 'is_nonbmd'] = 0
    return x.select_record(some_ids[0])
same('x[id, col] = v',                      set_cells)
# extraction_coverage_report, bif_utils.full_bif_daf_to_style_to_contests_dol, mapper, narrative_report
same("value_counts_daf('style_num', sort=True)", lambda x: x.value_counts_daf('style_num', sort=True))
same('valuecounts_for_colname1_groupedby_colname2', lambda x: x.select_where(lambda row: not row.get('corrupted', 0) and not row.get('is_bmd', 0))
        .valuecounts_for_colname1_groupedby_colname2(colname1='hexstyle', groupedby_colname2='style_num', sort=True, reverse=True))
same('set(x.keys())',                       lambda x: len(set(x.keys())))
def keys_loop(x):
    start = time.perf_counter()
    found = sum(1 for i in ids[:200] if i in x.keys())
    return found, f'{time.perf_counter() - start:.2f} s for 200'
def in_loop(x):
    start = time.perf_counter()
    found = sum(1 for i in ids[:200] if i in x)
    return found, f'{time.perf_counter() - start:.4f} s for 200'
print('id in x.keys(), 200 times: daf', keys_loop(daf), ' sdaf', keys_loop(sdaf))
print('id in x, 200 times:        daf', in_loop(daf), ' sdaf', in_loop(sdaf))
