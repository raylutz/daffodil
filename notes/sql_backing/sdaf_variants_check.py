# sdaf_variants_check.py
#
# Run sdaf_variants.py on a copy of a cmpcvr.db that AuditEngine's 'sql' engine has run in,
# and compare what it makes with what AuditEngine made: the tables, the CSV files and the stats.
#
# AuditEngine's modules are not imported, since they read job settings and write to S3.
# Each is replaced by a stand-in that writes to the scratch folder.
# pyontools is AuditEngine's own, read from its folder.
# Step 18's create_contest_variant_record() needs the job's contests, so its stand-in returns the
# joined row. The joined rows are compared with AuditEngine's view joined_contests instead.
#
# AuditEngine's inputs were trimmed by step 3, so the residue tables are added back first.
#
# Usage, with the daffodil venv:
#   PYTHONDONTWRITEBYTECODE=1 uv run python notes/sql_backing/sdaf_variants_check.py <cmpcvr dir> <scratch dir>

import os
import sys
import ast
import csv
import json
import shutil
import sqlite3
import types

from typing import Any, Dict, List, Tuple

from daffodil.daf import Daf
from daffodil.sqldaf import SqlDaf
from daffodil.lib.daf_types import T_ls, T_da
from daffodil.lib import daf_types

AE_ROOT: str = '/home/audit-engine-dev'


#===================================
# Stand-ins for AuditEngine's modules

def install_stand_ins(work_db_path: str, out_dir: str, sumcols_ls: T_ls) -> T_da:
    """ Put stand-in modules in sys.modules, for the imports of sdaf_variants.py. Returns what they record. """
    record_da: T_da = {'saved_dafs': {}, 'uploads': [], 'reports': []}

    sys.path.insert(0, AE_ROOT)
    import pyontools.pyontools                                     # noqa: F401  AuditEngine's own, pure Python.

    my_types = types.ModuleType('my_types')
    for name in dir(daf_types):
        if name.startswith('T_'):
            setattr(my_types, name, getattr(daf_types, name))
    my_types.T_doda = Dict[str, Dict[str, Any]]

    class DB:
        @staticmethod
        def build_filepath(dirname: str, name: str, s3flag: bool = False) -> str:
            return os.path.join(out_dir, name)

        @staticmethod
        def upload_file_dirname(dirname: str, filename: str, local_mirror: bool = True) -> str:
            record_da['uploads'].append(filename)
            return os.path.join(out_dir, filename)

        @staticmethod
        def save_data(data_daf: Daf, dirname: str, name: str, rtype: str = 'daf', local_mirror: bool = True) -> str:
            record_da['saved_dafs'][name] = data_daf
            return os.path.join(out_dir, name)

        @staticmethod
        def file_exists(name: str, dirname: str) -> bool:
            return False

        @staticmethod
        def load_data(**kwargs: Any) -> Any:
            return None

    models = types.ModuleType('models')
    models_db = types.ModuleType('models.DB')
    models_db.DB = DB
    models.DB = models_db

    utilities = types.ModuleType('utilities')

    def module(name: str, **funcs: Any) -> types.ModuleType:
        mod = types.ModuleType(f'utilities.{name}')
        for func_name, func in funcs.items():
            setattr(mod, func_name, func)
        setattr(utilities, name, mod)
        sys.modules[f'utilities.{name}'] = mod
        return mod

    import re
    module('utils',
           safe_regex_search    = lambda regex, string, flags=0: bool(string) and bool(re.search(regex, string, flags=flags)),
           history_tag          = lambda: 'check')
    module('logs',
           stsloc               = lambda *args, **kwargs: None,
           prog_loc             = lambda: 'sdaf_variants',
           exception_report     = lambda text: print(text))
    module('reports', save_md_and_html=lambda argsdict, md_report, rootname: record_da['reports'].append(md_report) or [])
    module('sql_utils', get_db_path_from_database_name=lambda name: work_db_path)
    module('sql_variants', create_contest_variant_record=lambda row_da: row_da)
    module('cmpcvr', blank_sheets_to_ballot_variants_daf=lambda blank_sheets_daf: blank_sheets_daf)
    module('split_bmd', combine_cvr_records=lambda kept_da, siblings_lda: kept_da)
    indirect_ls: T_ls = [col for col in sumcols_ls if col not in ('blank', 'v', 'is_bmd')]
    module('generic_cvr', gen_flat_votes_cols=lambda **kwargs: indirect_ls)

    sys.modules['my_types'] = my_types
    sys.modules['models'] = models
    sys.modules['models.DB'] = models_db
    sys.modules['utilities'] = utilities
    return record_da


#===================================
# Comparisons

def decoded_cell(val: Any) -> Any:
    """ A dict in JSON or PYON text as sorted pairs, so that two texts of the same dict compare equal. """
    if isinstance(val, str) and val.startswith('{'):
        try:
            decoded: Any = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            try:
                decoded = json.loads(val)
            except ValueError:
                return val
        if isinstance(decoded, dict):
            return tuple(sorted(decoded.items()))
    return val


def table_rows(conn: sqlite3.Connection, name: str) -> Tuple[T_ls, List[tuple]]:
    """ The columns and rows of a table or view, without daf_rid. """
    cursor = conn.execute(f'SELECT * FROM "{name}"')
    cols_ls: T_ls = [desc[0] for desc in cursor.description]
    keep_li: List[int] = [idx for idx, col in enumerate(cols_ls) if col != 'daf_rid']
    return [cols_ls[idx] for idx in keep_li], [tuple(row[idx] for idx in keep_li) for row in cursor]


def compare_tables(work_conn: sqlite3.Connection, ae_conn: sqlite3.Connection, names_ls: T_ls) -> None:
    """ Print, for each table, the rows of each, and if the rows are the same, as text and as values. """
    print(f'\n{"table":32} {"sdaf":>7} {"sql":>7}  same columns, same text, same values')
    for name in names_ls:
        ours_cols_ls, ours_lt = table_rows(work_conn, name)
        ae_cols_ls, ae_lt = table_rows(ae_conn, name)
        same_cols: bool = sorted(ours_cols_ls) == sorted(ae_cols_ls)
        if same_cols:
            order_li: List[int] = [ours_cols_ls.index(col) for col in ae_cols_ls]
            ours_lt = [tuple(row[idx] for idx in order_li) for row in ours_lt]
        same_text: bool = same_cols and set(ours_lt) == set(ae_lt)
        same_vals: bool = same_cols and ({tuple(map(decoded_cell, row)) for row in ours_lt} ==
                                         {tuple(map(decoded_cell, row)) for row in ae_lt})
        print(f'{name:32} {len(ours_lt):7} {len(ae_lt):7}  {same_cols}, {same_text}, {same_vals}')
        if not same_cols:
            print(f'    sdaf {ours_cols_ls}\n    sql  {ae_cols_ls}')


def compare_csv_files(out_dir: str, ae_dir: str, names_ls: T_ls) -> None:
    """ Print, for each CSV file, the lines of each, and if the header and the set of rows are the same. """
    print(f'\n{"csv file":32} {"sdaf":>7} {"sql":>7}  same header, same rows')
    for name in names_ls:
        with open(os.path.join(out_dir, name), newline='', encoding='utf-8') as ours_f, \
             open(os.path.join(ae_dir, name), newline='', encoding='utf-8') as ae_f:
            ours_ls: T_ls = ours_f.read().splitlines()
            ae_ls: T_ls = ae_f.read().splitlines()
        print(f'{name:32} {len(ours_ls):7} {len(ae_ls):7}  {ours_ls[0] == ae_ls[0]}, {set(ours_ls[1:]) == set(ae_ls[1:])}')


def compare_stats(stats_da: T_da, ae_dir: str) -> None:
    """ Print, for each stat, if it is the same as AuditEngine's JSON file. The keys list is compared as a set. """
    print(f'\n{"stat":45} {"sdaf":>7} {"sql":>7}  same')
    for name, ours_a in stats_da.items():
        with open(os.path.join(ae_dir, f'{name}.json')) as json_f:
            ae_a: Any = json.load(json_f)
        ours_a = json.loads(json.dumps(ours_a))
        same: bool = set(ours_a) == set(ae_a) if name == 'nonvariant_keys_ls' else ours_a == ae_a
        print(f'{name:45} {len(ours_a):7} {len(ae_a):7}  {same}')


#===================================
# Split BMD CVR records

def check_combine_split_bmd(sdaf_variants: Any) -> None:
    """ combine_split_bmd_sdaf() on a small table, with a stand-in that adds the votes of the siblings. """
    sdaf_variants.split_bmd.combine_cvr_records = lambda kept_da, siblings_lda: kept_da | {
            'v': kept_da['v'] + sum(sib['v'] for sib in siblings_lda), 'concise_votes': {'n': len(siblings_lda)}}
    SqlDaf.db_path = os.path.join(os.path.dirname(SqlDaf.db_path), 'bmd_check.db')
    cvr_sdaf = SqlDaf('cvr', cols=['ballot_id', 'v', 'concise_votes', 'style'], keyfield='ballot_id',
                      lol=[['b1', 1, '{}', 's'], ['b2', 2, '{}', 's'], ['b3', 4, '{}', 's'], ['b4', 8, '{}', 's']])
    num_combined: int = sdaf_variants.combine_split_bmd_sdaf(cvr_sdaf, {
            'b1': {'combined_ids': ['b2', 'b3']},       # combined.
            'b4': {'combined_ids': ['b9']},             # no sibling is there: skipped.
            'b8': {'combined_ids': ['b2']},             # the kept row is not there: skipped.
            })
    print(f'\nsplit BMD: {num_combined} combined; rows {cvr_sdaf.to_daf().lol}')
    assert num_combined == 1
    assert cvr_sdaf.to_daf().lol == [['b1', 7, "{'n': 2}", 's'], ['b4', 8, '{}', 's']]
    SqlDaf.close()


#===================================

def main() -> None:
    ae_dir, scratch_dir = sys.argv[1], sys.argv[2]
    ae_db_path: str = os.path.join(ae_dir, 'cmpcvr.db')
    work_db_path: str = os.path.join(scratch_dir, 'cmpcvr_sdaf.db')
    out_dir: str = os.path.join(scratch_dir, 'out')
    os.makedirs(out_dir, exist_ok=True)
    shutil.copyfile(ae_db_path, work_db_path)

    with open(os.path.join(ae_dir, 'nonvariant_sums_di.json')) as json_f:
        sumcols_ls: T_ls = list(json.load(json_f))
    record_da: T_da = install_stand_ins(work_db_path, out_dir, sumcols_ls)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import sdaf_variants

    # The inputs as they were before step 3.
    with sqlite3.connect(work_db_path) as conn:
        conn.execute('INSERT INTO votes SELECT * FROM votes_residue')
        conn.execute('INSERT INTO "cvr_votes!original" SELECT * FROM cvr_residue')

    argsdict: T_da = {'use_concise_votes': True, 'combine_split_bmd_cvrs': False}
    stats_da: T_da = sdaf_variants.find_variants_sdaf(
            argsdict,
            dirname                 = 'cmpcvr',
            input_tables_ls         = ['votes', 'cvr_votes!original', 'cvr_votes!modified'],
            other_input_tables_ls   = ['pmvs', 'wins'],
            variants_tables_ls      = ['audit_variants', 'cvr_variants'],
            nonvariants_tables_ls   = ['audit_nonvariants', 'cvr_nonvariants'],
            )

    print(record_da['reports'][0])
    work_conn = sqlite3.connect(work_db_path)
    ae_conn = sqlite3.connect(f'file:{ae_db_path}?mode=ro', uri=True)
    compare_tables(work_conn, ae_conn, [name for name in sdaf_variants.RESULT_TABLES_LS if name != 'combined_nonvariants'])
    compare_csv_files(out_dir, ae_dir, ['votes_residue.csv', 'cvr_residue.csv', 'partially_agreed.csv',
                                        'audit_variants.csv', 'cvr_variants.csv'])
    compare_stats(stats_da, ae_dir)

    with open(os.path.join(ae_dir, 'ballot_variants.csv'), newline='') as csv_f:
        ae_ballot_variants_lol: list = list(csv.reader(csv_f))
    blank_daf: Daf = record_da['saved_dafs']['ballot_variants.csv']
    print(f"\nballot_variants: {len(blank_daf)} blank sheets given, AE file has {len(ae_ballot_variants_lol) - 1} rows. "
          f"group is a column: {'group' in blank_daf.columns()}")
    joined_daf: Daf = record_da['saved_dafs']['contest_variants.csv']
    print(f"contest_variants: {len(joined_daf)} rows given to the record function. group is a column: {'group' in joined_daf.columns()}")
    print(f"uploads: {record_da['uploads']}")
    tables_ls: T_ls = sorted(row[0] for row in work_conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'"))
    print(f"tables left: {tables_ls}")
    work_conn.close()
    ae_conn.close()

    check_combine_split_bmd(sdaf_variants)


if __name__ == '__main__':
    main()
