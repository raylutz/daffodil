# acre_join_check.py: ACRE's CVR and BIF join, by AuditEngine's function on Daf, and by a SqlDaf form.
#
# The BIF here is made from the CVR rows, with every 7th value of the overlap columns blanked
# on one side and every 11th on the other, so each fallback is used.
# Usage, with AuditEngine's venv python -I: python acre_join_check.py <cvr_votes csv> <scratch dir>

import os
import sys
import time

sys.path.insert(0, '/home/ubuntu/daffodil_sql/src')
sys.path.insert(1, '/home/audit-engine-dev')
# AuditEngine's venv needs pysqlite3-binary. Until it has it, PYSQLITE3_SITE names a site-packages that does.
if os.environ.get('PYSQLITE3_SITE'):
    sys.path.append(os.environ['PYSQLITE3_SITE'])

from daffodil.daf import Daf                            # noqa: E402
from daffodil.sqldaf import SqlDaf                      # noqa: E402
from utilities import logs                              # noqa: E402
logs.sts = logs.stsloc = lambda *args, **kwargs: None
from utilities import acre_utils                        # noqa: E402

csv_path, out_dir = sys.argv[1], sys.argv[2]
overlap_ls = ['is_bmd', 'batchid', 'precinct', 'hexstyle']

cvr_daf = Daf.from_csv(csv_path, keyfield='ballot_id').select_cols(['ballot_id', 'batchid', 'precinct', 'hexstyle', 'is_bmd', 'concise_votes'])
bif_daf = cvr_daf.select_cols(['ballot_id', 'batchid', 'is_bmd', 'hexstyle', 'precinct']).copy()
for irow in range(len(cvr_daf)):
    for icol, col in enumerate(overlap_ls):
        if (irow + icol) % 7 == 0:
            cvr_daf[irow, col] = ''
        if (irow + icol) % 11 == 0:
            bif_daf[irow, col] = ''
bif_daf.insert_col('style_num', default='s1')
bif_daf.set_keyfield('ballot_id')

start = time.perf_counter()
ref_daf = acre_utils._join_cvr_votes_with_bif(cvr_daf, bif_daf)
print(f'Daf, AuditEngine function: {time.perf_counter() - start:.2f} s, {len(ref_daf)} rows, cols {ref_daf.columns()}')

SqlDaf.db_path = os.path.join(out_dir, 'acre_join.db')
if os.path.exists(SqlDaf.db_path):
    os.remove(SqlDaf.db_path)
cvr_sdaf = SqlDaf.from_daf(cvr_daf, name='cvr_votes')
bif_sdaf = SqlDaf.from_daf(bif_daf, name='bif')

# --- The SqlDaf form, for AuditEngine ---
start = time.perf_counter()
BIF_WINS = ('is_bmd', 'batchid')
CVR_WINS = ('precinct', 'hexstyle')
overlap_fields = [col for col in BIF_WINS + CVR_WINS if col in bif_sdaf.columns()]

# The BIF copies of the shared columns get their own names before the join.
renamed_sdaf = bif_sdaf.select_cols(bif_sdaf.columns()).copy()
renamed_sdaf.rename_cols({col: f'{col}_bifdup' for col in overlap_fields})
joined_sdaf = cvr_sdaf.join(renamed_sdaf, how='inner').copy(name='cvr_bif_joined')


def resolved(row):
    """ Each shared column takes the value of the side that wins, or else the other. """
    out = {}
    for col in overlap_fields:
        bif_val, cvr_val = row[f'{col}_bifdup'], row[col]
        first, second = (bif_val, cvr_val) if col in BIF_WINS else (cvr_val, bif_val)
        out[col] = first if first not in ('', None, 'nan') else (second if second not in ('', None, 'nan') else '')
    return out


joined_sdaf.apply_in_place(resolved)
joined_sdaf.drop_cols([f'{col}_bifdup' for col in overlap_fields])
# --- end ---
print(f'SqlDaf form: {time.perf_counter() - start:.2f} s, {len(joined_sdaf)} rows, cols {joined_sdaf.columns()}')

got_daf = joined_sdaf.to_daf()
print('same columns:', got_daf.columns() == ref_daf.columns())
ref_lod = sorted(ref_daf.to_lod(), key=lambda r: r['ballot_id'])
got_lod = sorted(got_daf.to_lod(), key=lambda r: r['ballot_id'])
print('same rows:', ref_lod == got_lod)
if ref_lod != got_lod:
    for a, b in zip(ref_lod, got_lod):
        if a != b:
            print('  ref', a); print('  got', b); break
