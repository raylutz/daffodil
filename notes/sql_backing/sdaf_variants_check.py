# sdaf_variants_check.py
#
# Run sdaf_variants.py on a copy of a cmpcvr.db that AuditEngine's 'sql' engine has run in,
# and compare what it makes with what AuditEngine made: the tables, the CSV files and the stats.
#
# AuditEngine's modules are not imported, since they read job settings and write to S3.
# Each is replaced by a stand-in that writes to the scratch folder.
# pyontools is AuditEngine's own, read from its folder.
# Step 18's create_contest_variant_record() needs the job's contests, so its stand-in returns the
# joined row, with a dict cell added. The joined rows are compared with AuditEngine's view joined_contests.
# contest_variants.csv is compared with what DB.save_data() writes of the same records, as a Daf.
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
from daffodil.lib import daf_types, daf_sql

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
    module('sql_utils', get_db_path_from_database_name=lambda name: work_db_path, sql_unesc_str=daf_sql.sql_unesc_str)
    module('sql_variants', create_contest_variant_record=stand_in_contest_variant_record)
    module('cmpcvr', blank_sheets_to_ballot_variants_daf=lambda blank_sheets_daf: blank_sheets_daf)
    module('split_bmd', combine_cvr_records=lambda kept_da, siblings_lda: kept_da)
    indirect_ls: T_ls = [col for col in sumcols_ls if col not in ('blank', 'v', 'is_bmd')]
    module('generic_cvr', gen_flat_votes_cols=lambda **kwargs: indirect_ls)

    sys.modules['my_types'] = my_types
    sys.modules['models'] = models
    sys.modules['models.DB'] = models_db
    sys.modules['utilities'] = utilities
    return record_da


def stand_in_contest_variant_record(row_da: T_da) -> T_da:
    """ The joined row, with a dict cell that holds True, False and None, as vote_diff_orig may. """
    record_da: T_da = dict(row_da)
    record_da['vote_diff_orig'] = {'cidx': row_da['cidx'], 'agreed': False, 'flags': [True, None], 'note': None}
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
    """ Print, for each CSV file, the lines of each, if the header and the set of rows are the same, and if the files are the same. """
    print(f'\n{"csv file":32} {"sdaf":>7} {"sql":>7}  same header, same rows, same lines in order')
    for name in names_ls:
        with open(os.path.join(out_dir, name), newline='', encoding='utf-8') as ours_f, \
             open(os.path.join(ae_dir, name), newline='', encoding='utf-8') as ae_f:
            ours_ls: T_ls = ours_f.read().splitlines()
            ae_ls: T_ls = ae_f.read().splitlines()
        print(f'{name:32} {len(ours_ls):7} {len(ae_ls):7}  {ours_ls[0] == ae_ls[0]}, {set(ours_ls[1:]) == set(ae_ls[1:])}, {ours_ls == ae_ls}')


def compare_stats(stats_da: T_da, ae_dir: str) -> None:
    """ Print, for each stat, if it is the same as AuditEngine's JSON file. The keys list is compared as a set. """
    print(f'\n{"stat":45} {"sdaf":>7} {"sql":>7}  same')
    for name, ours_a in stats_da.items():
        with open(os.path.join(ae_dir, f'{name}.json')) as json_f:
            ae_a: Any = json.load(json_f)
        ours_a = json.loads(json.dumps(ours_a))
        same: bool = set(ours_a) == set(ae_a) if name == 'nonvariant_keys_ls' else ours_a == ae_a
        print(f'{name:45} {len(ours_a):7} {len(ae_a):7}  {same}')


def check_contest_variants(work_db_path: str, out_dir: str, dict_cols_ls: T_ls, sort_cols_ls: T_ls) -> None:
    """
    Compare contest_variants.csv with the file DB.save_data() writes of the same records.

    The reference converts the rows of joined_contests as one Daf, sorts them as the module does,
    and writes it with CRLF line ends.
    Every cell is compared as text. A dict cell is PYON text in both.
    """
    with sqlite3.connect(f'file:{work_db_path}?mode=ro', uri=True) as conn:
        cursor = conn.execute('SELECT * FROM joined_contests ORDER BY daf_rid')
        cols_ls: T_ls = [daf_sql.sql_unesc_str(desc[0]) for desc in cursor.description]
        rows_loda: List[T_da] = [{col: '' if val is None else val for col, val in zip(cols_ls, row) if col != 'daf_rid'}
                                 for row in cursor]
    ref_daf: Daf = Daf.from_lod(rows_loda).apply(func=stand_in_contest_variant_record)
    ref_daf.sort_by_colnames(sort_cols_ls, length_priority=True, as_str=True)
    ref_ls: List[T_ls] = list(csv.reader(ref_daf.to_csv_buff(line_terminator='\r\n', include_header=True).splitlines()))

    with open(os.path.join(out_dir, 'contest_variants.csv'), newline='', encoding='utf-8') as ours_f:
        ours_text: str = ours_f.read()
    ours_ls: List[T_ls] = list(csv.reader(ours_text.splitlines()))
    crlf_ends: bool = ours_text.count('\r\n') == len(ours_ls)

    header_ls: T_ls = ref_ls[0]
    dict_idx_ls: List[int] = [header_ls.index(col) for col in dict_cols_ls if col in header_ls]
    num_text_diff: int = 0
    num_dict_diff: int = 0
    first_diff: str = ''
    for ref_row_ls, ours_row_ls in zip(ref_ls[1:], ours_ls[1:]):
        for idx, (ref_cell, ours_cell) in enumerate(zip(ref_row_ls, ours_row_ls)):
            same: bool = ref_cell == ours_cell
            if idx in dict_idx_ls:
                num_dict_diff += not same
            else:
                num_text_diff += not same
            if not same and not first_diff:
                first_diff = f'{header_ls[idx]}: reference {ref_cell!r}, module {ours_cell!r}'
    print(f"\ncontest_variants.csv: {len(ours_ls) - 1} rows, reference {len(ref_ls) - 1}. "
          f"Same header: {ours_ls[0] == header_ls}. CRLF line ends: {crlf_ends}.")
    print(f"    cells that differ as text: {num_text_diff}. dict cells that differ as text: {num_dict_diff}.")
    if first_diff:
        print(f"    first difference: {first_diff}")
    if len(ours_ls) > 1:
        print(f"    first dict cell: reference {ref_ls[1][dict_idx_ls[0]]!r}, module {ours_ls[1][dict_idx_ls[0]]!r}")


#===================================
# Split BMD CVR records

def check_combine_split_bmd(sdaf_variants: Any, scratch_dir: str) -> None:
    """ combine_split_bmd_sdaf() on a small table, with a stand-in that adds the votes of the siblings. """
    sdaf_variants.split_bmd.combine_cvr_records = lambda kept_da, siblings_lda: kept_da | {
            'v': kept_da['v'] + sum(sib['v'] for sib in siblings_lda), 'concise_votes': {'n': len(siblings_lda)}}
    SqlDaf.db_path = os.path.join(scratch_dir, 'bmd_check.db')
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
# Step 15, piece by piece

def check_step_15(sdaf_variants: Any, work_db_path: str, ae_db_path: str) -> None:
    """
    Show each piece of step 15 on the run's nonvariant_contests, against AuditEngine's partial_nonvariants.

    AuditEngine's table is taken before step 16 adds the empty shells and step 17 moves the blank sheets.
    So it is compared only on the ballots step 15 makes.
    """
    SqlDaf.db_path = work_db_path
    contests_sdaf: SqlDaf = SqlDaf('nonvariant_contests')

    # Piece 1: the merged dict of each ballot.
    merged_sdaf: SqlDaf = contests_sdaf.groupby_aggregate('ballot_id', sdaf_variants.update_da, 'concise_votes')
    merged_ds: Dict[Any, str] = {row_da['ballot_id']: row_da['concise_votes'] for row_da in merged_sdaf.iter_dict()}

    # Piece 2: num_contests of each.
    merged_sdaf.insert_col('num_contests', default=0)
    merged_sdaf.apply_in_place(lambda row_da: {'num_contests': sdaf_variants.num_contests_of(row_da['concise_votes'])})
    counts_di: Dict[Any, int] = {row_da['ballot_id']: row_da['num_contests'] for row_da in merged_sdaf.iter_dict()}
    SqlDaf.close()

    with sqlite3.connect(f'file:{ae_db_path}?mode=ro', uri=True) as ae_conn:
        ae_lt: List[tuple] = list(ae_conn.execute('SELECT ballot_id, concise_votes, num_contests FROM partial_nonvariants'))
    # Step 17 moved the blank sheets out of AuditEngine's table, so only the ballots in both are compared.
    ae_text_ds: Dict[Any, str] = {row[0]: row[1] for row in ae_lt if row[0] in merged_ds}
    ae_counts_di: Dict[Any, int] = {row[0]: row[2] for row in ae_lt if row[0] in merged_ds}
    both_text_ds: Dict[Any, str] = {key: merged_ds[key] for key in ae_text_ds}
    both_counts_di: Dict[Any, int] = {key: counts_di[key] for key in ae_counts_di}
    print(f'\nstep 15: {len(merged_ds)} ballots merged, {len(ae_text_ds)} of them in AE partial_nonvariants. '
          f'Same text: {both_text_ds == ae_text_ds}. Same num_contests: {both_counts_di == ae_counts_di}.')
    first_key: Any = next(iter(ae_text_ds))
    print(f'  first ballot {first_key}: sdaf {merged_ds[first_key][:70]}\n'
          f'  {"":{len(str(first_key)) + 15}}sql  {ae_text_ds.get(first_key, "")[:70]}')


def check_contest_variants_order(out_dir: str, ae_dir: str, sort_cols_ls: T_ls) -> None:
    """
    Compare the order of the rows of contest_variants.csv with AuditEngine's file, by the sort columns.

    The stand-in for create_contest_variant_record() makes other cells, so only the sort columns are compared.
    """
    keys_dolt: Dict[str, List[tuple]] = {}
    for label, dir_path in (('sdaf', out_dir), ('sql', ae_dir)):
        with open(os.path.join(dir_path, 'contest_variants.csv'), newline='', encoding='utf-8') as csv_f:
            rows_lol: List[T_ls] = list(csv.reader(csv_f))
        idx_li: List[int] = [rows_lol[0].index(col) for col in sort_cols_ls]
        keys_dolt[label] = [tuple(row[idx] for idx in idx_li) for row in rows_lol[1:]]
    print(f"contest_variants.csv: the same {', '.join(sort_cols_ls)} in the same order as AE's file: "
          f"{keys_dolt['sdaf'] == keys_dolt['sql']}")


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
    # A residue table written by the port also has a daf_rid column, so only the input's columns are copied.
    with sqlite3.connect(work_db_path) as conn:
        for input_table, residue_table in (('votes', 'votes_residue'), ('cvr_votes!original', 'cvr_residue')):
            cols_ls: T_ls = [row[1] for row in conn.execute(f'PRAGMA table_info("{input_table}")')]
            cols_sql: str = ', '.join(f'"{col}"' for col in cols_ls)
            conn.execute(f'INSERT INTO "{input_table}" ({cols_sql}) SELECT {cols_sql} FROM {residue_table}')

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
    ae_tables_ls: T_ls = [name for name in sdaf_variants.RESULT_TABLES_LS
                          if name not in ('combined_nonvariants', sdaf_variants.CONTEST_VARIANTS_TABLE)]
    compare_tables(work_conn, ae_conn, ae_tables_ls)
    compare_csv_files(out_dir, ae_dir, ['votes_residue.csv', 'cvr_residue.csv', 'partially_agreed.csv',
                                        'audit_variants.csv', 'cvr_variants.csv'])
    compare_stats(stats_da, ae_dir)

    with open(os.path.join(ae_dir, 'ballot_variants.csv'), newline='') as csv_f:
        ae_ballot_variants_lol: list = list(csv.reader(csv_f))
    blank_daf: Daf = record_da['saved_dafs']['ballot_variants.csv']
    print(f"\nballot_variants: {len(blank_daf)} blank sheets given, AE file has {len(ae_ballot_variants_lol) - 1} rows. "
          f"group is a column: {'group' in blank_daf.columns()}. "
          f"Same ballot_id order: {blank_daf.col('ballot_id') == [row[ae_ballot_variants_lol[0].index('ballot_id')] for row in ae_ballot_variants_lol[1:]]}")
    print(f"uploads: {record_da['uploads']}")
    tables_ls: T_ls = sorted(row[0] for row in work_conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'"))
    print(f"tables left: {tables_ls}")
    work_conn.close()
    ae_conn.close()

    check_contest_variants(work_db_path, out_dir, ['vote_diff_orig'], sdaf_variants.CONTEST_VARIANTS_SORT_LS)
    check_contest_variants_order(out_dir, ae_dir, sdaf_variants.CONTEST_VARIANTS_SORT_LS)
    check_step_15(sdaf_variants, work_db_path, ae_db_path)
    check_combine_split_bmd(sdaf_variants, scratch_dir)


if __name__ == '__main__':
    main()
