# find_variants_trial.py
#
# A trial port of AuditEngine's locate_variants(), steps 1 to 11, to SqlDaf. Question 117 C,
# 2026-10-14. It follows today's steps one for one, as find_variants_sketch.md describes, and
# compares each result table with the one AuditEngine wrote in a real run. AuditEngine is not
# changed: its database is copied, and the copy is used.
#
# Usage:
#   uv run python notes/sql_backing/find_variants_trial.py <cmpcvr.db> <scratch dir>
#
# The database is one that locate_variants() has already run in, with use_concise. Its input
# tables were trimmed by step 3, so the originals are rebuilt by adding back the residue tables.

import os
import re
import ast
import sys
import time
import shutil

from typing import Any, Dict, List, Tuple

from daffodil.daf import Daf, NULL
from daffodil.lib.daf_types import T_ls
from daffodil.sqldaf import SqlDaf, T_sdaf, T_lsdaf


VARIANT_FLAG_RE = r"'c\d+(w|ov|g)'"     # as in sql_variants.py step 4: a writein, overvote or gray contest.


def has_variant_flag(concise_votes: Any) -> bool:
    """ utils.safe_regex_search() of AuditEngine, on the concise votes of one row. """
    return bool(concise_votes) and bool(re.search(VARIANT_FLAG_RE, concise_votes))


def decoded_pyon(pyon_str: Any) -> Any:
    """
    The value of a PYON string, for comparing two of them.

    pyontools.normalize_pyon() decodes the text, sorts the dict keys and encodes it again. Two
    normalized strings are equal exactly when the decoded values are equal, so this compares the
    values. Text that does not decode is compared as it is, as normalize_pyon() returns it.
    """
    try:
        return ast.literal_eval(pyon_str)
    except (ValueError, SyntaxError):
        return pyon_str


REPORT_COLS_LS: T_ls = ['Step', 'Description', 'Audit recs', 'CVR Recs', 'Audit(s)', 'CVR(s)']    # as in sql_variants.py


def timed(func: Any) -> Tuple[Any, float]:
    """ Call a function of no arguments, and give its result and the seconds it took. """
    start_f: float = time.perf_counter()
    result: Any = func()
    return result, time.perf_counter() - start_f


def locate_variants_sdaf(
        side_lsdaf:         T_lsdaf,            # the audit votes and the CVR votes.
        modi_sdaf:          T_sdaf | None,      # the modified CVR votes, if any.
        names_ls:           T_ls,               # the names for the results: audit and cvr.
        prefix:             str,                # put before each result table name.
        ) -> Dict[str, Any]:                    # the result tables by name, the disagreed keys and the report.
    """
    Find the variant ballots, as locate_variants() does in steps 1 to 11, with concise votes.

    Each step is the same as in sql_variants.py, written with SqlDaf. The inputs are changed in
    step 3, as today, where the residue is deleted from them. A report Daf with the columns and
    step descriptions of sql_variants.py gives the rows and the seconds of each step, for each
    side. A selection runs when it is first read, so the time of a step is measured around the
    call that reads it, such as the concat() that adds its rows, or a len().
    """
    keyfield: str = 'ballot_id'
    variants_names_ls: T_ls = [f'{prefix}{name}_variants' for name in names_ls]
    nonvariants_names_ls: T_ls = [f'{prefix}{name}_nonvariants' for name in names_ls]
    residue_names_ls: T_ls = [f'{prefix}{name}_residue' for name in names_ls]
    report_daf: Daf = Daf(cols=REPORT_COLS_LS)
    secs_lf: List[float]
    i: int

    def report(step: int, descr: str, counts_li: List[Any], secs_lf: List[Any]) -> None:
        report_daf.append({'Step': step, 'Description': descr, 'Audit recs': counts_li[0], 'CVR Recs': counts_li[1],
                           'Audit(s)': secs_lf[0], 'CVR(s)': secs_lf[1]})

    def add_new_rows(target_sdaf: SqlDaf, rows_sdaf: SqlDaf) -> None:
        """ INSERT OR IGNORE of today: add the rows whose key is not in the target yet. """
        target_sdaf.concat(rows_sdaf.select_records_daf(target_sdaf, inverse=True, silent_error=True))

    report(0, 'Input tables.', [len(side_sdaf) for side_sdaf in side_lsdaf], [NULL, NULL])

    # step 1: drop any tables of an earlier run.
    for name in variants_names_ls + nonvariants_names_ls + residue_names_ls:
        if SqlDaf.table_exists(name):
            SqlDaf(name).delete()
    report(1, 'Dropping any existing variant tables.', [0, 0], [NULL, NULL])

    # step 2: empty variants tables, with a unique keyfield.
    variants_lsdaf: T_lsdaf = []
    secs_lf = []
    for i in range(2):
        variants_sdaf, secs_f = timed(lambda: SqlDaf(variants_names_ls[i], side_lsdaf[i].columns(),
                                                     keyfield=keyfield).ensure_key_index())
        variants_lsdaf.append(variants_sdaf)                # each with its own input's columns, as AuditEngine does.
        secs_lf.append(secs_f)
    report(2, 'Create Variant tables with indexes.', [len(sdaf) for sdaf in variants_lsdaf], secs_lf)

    # step 3: the residue of each side, then deleted from it.
    residue_lsdaf: T_lsdaf = []
    secs_lf = []
    for i in range(2):
        residue_sdaf, secs_f = timed(lambda: side_lsdaf[i].select_records_daf(
                side_lsdaf[1 - i], inverse=True, silent_error=True).copy(name=residue_names_ls[i]).save())
        residue_lsdaf.append(residue_sdaf)
        secs_lf.append(secs_f)
    for i in range(2):
        _, secs_f = timed(lambda: side_lsdaf[i].delete_records(residue_lsdaf[i], inplace=True, silent_error=True))
        secs_lf[i] += secs_f
    report(3, 'Remove Missing Records to Residue Tables', [len(sdaf) for sdaf in residue_lsdaf], secs_lf)
    report(3, 'Remaining ballots in adjusted input tables', [len(sdaf) for sdaf in side_lsdaf], [NULL, NULL])

    # step 4: the flagged rows.
    secs_lf = []
    for i in range(2):
        flagged_sdaf: SqlDaf = side_lsdaf[i].select_where(lambda row: has_variant_flag(row['concise_votes']),
                                                          cols=['concise_votes'])
        secs_lf.append(timed(lambda: add_new_rows(variants_lsdaf[i], flagged_sdaf))[1])
    report(4, 'Add records with variant flags', [len(sdaf) for sdaf in variants_lsdaf], secs_lf)

    # step 5: harmonize, so each side has the variants of the other.
    secs_lf = []
    for i in range(2):
        secs_lf.append(timed(lambda: add_new_rows(
                variants_lsdaf[i], side_lsdaf[i].select_records_daf(variants_lsdaf[1 - i], silent_error=True)))[1])
    report(5, 'Harmonize variant tables', [len(sdaf) for sdaf in variants_lsdaf], secs_lf)

    # step 6: the ballots of the modified CVR.
    if modi_sdaf is not None:
        secs_lf = []
        for i in range(2):
            secs_lf.append(timed(lambda: add_new_rows(
                    variants_lsdaf[i], side_lsdaf[i].select_records_daf(modi_sdaf, silent_error=True)))[1])
        report(6, 'Add records from cvr_modified', [len(sdaf) for sdaf in variants_lsdaf], secs_lf)

    # step 7: the rest are nonvariants for now.
    nonvariants_lsdaf: T_lsdaf = []
    secs_lf = []
    for i in range(2):
        nonvariants_sdaf, secs_f = timed(lambda: side_lsdaf[i].select_records_daf(
                variants_lsdaf[i], inverse=True, silent_error=True).copy(name=nonvariants_names_ls[i]).save())
        nonvariants_lsdaf.append(nonvariants_sdaf)
        secs_lf.append(secs_f)
    report(7, 'Create nonflagged nonvariants', [len(sdaf) for sdaf in nonvariants_lsdaf], secs_lf)

    # step 8: the pairs whose concise votes differ as text.
    pair_sdaf: SqlDaf = nonvariants_lsdaf[0].join(nonvariants_lsdaf[1], tag_other=True)
    cv_cols_ls: T_ls = [col for col in pair_sdaf.columns() if col.startswith('concise_votes_')]
    a_cv, b_cv = cv_cols_ls
    text_differ_sdaf: SqlDaf = pair_sdaf.select_where_sql(f'"{a_cv}" IS NOT "{b_cv}"')
    num_text_differ, secs_f = timed(lambda: len(text_differ_sdaf))
    report(8, 'Create disagreed_ballots from nonflagged nonvariants', [num_text_differ, num_text_differ], [secs_f, NULL])

    # step 9: of those, the ones that still differ when normalized.
    disagreed_sdaf: SqlDaf = text_differ_sdaf.select_where(
            lambda row: decoded_pyon(row[a_cv]) != decoded_pyon(row[b_cv]), cols=[a_cv, b_cv])
    num_disagreed, secs_f = timed(lambda: len(disagreed_sdaf))
    report(9, 'Double check disagreed_ballots using normalized concise_votes', [num_disagreed, num_disagreed], [secs_f, NULL])

    # step 10: the disagreed ballots are variants after all.
    secs_lf = []
    for i in range(2):
        secs_lf.append(timed(lambda: add_new_rows(
                variants_lsdaf[i], nonvariants_lsdaf[i].select_records_daf(disagreed_sdaf, silent_error=True)))[1])
    report(10, 'Add disagreed ballots to variants', [len(sdaf) for sdaf in variants_lsdaf], secs_lf)

    # step 11: and are no longer nonvariants.
    secs_lf = []
    for i in range(2):
        secs_lf.append(timed(lambda: nonvariants_lsdaf[i].delete_records(variants_lsdaf[i], inplace=True, silent_error=True))[1])
    report(11, 'Remove variant ballot records from nonvariant ballots table.', [len(sdaf) for sdaf in nonvariants_lsdaf], secs_lf)

    result_da: Dict[str, Any] = {}
    for i, name in enumerate(names_ls):
        result_da[f'{name}_variants'] = variants_lsdaf[i]
        result_da[f'{name}_nonvariants'] = nonvariants_lsdaf[i]
        result_da[f'{name}_residue'] = residue_lsdaf[i]
    result_da['disagreed_keys'] = disagreed_sdaf.keys()
    result_da['report_daf'] = report_daf
    return result_da


# Columns that AuditEngine fills after steps 1 to 11, so they are left out of the comparison.
# cmpcvr.py adds num_contests to audit_nonvariants with add_computed_column(), after locate_variants().
LATER_COLS_LS: T_ls = ['num_contests']


def rows_set(sdaf: SqlDaf) -> set:
    """ The rows of a table, as a set of tuples, without LATER_COLS_LS, for tables whose row order differs. """
    keep_ls: list = [idx for idx, col in enumerate(sdaf.columns()) if col not in LATER_COLS_LS]
    return {tuple(row_la[idx] for idx in keep_ls) for row_la in sdaf.to_daf().lol}


def compare(result_da: Dict[str, Any], ae_names_da: Dict[str, str]) -> List[Tuple[str, int, int, bool]]:
    """ Compare each result with the table AuditEngine wrote. A list of (name, ours, theirs, same). """
    report_lt: List[Tuple[str, int, int, bool]] = []
    ours_name: str
    for ours_name, ae_name in ae_names_da.items():
        ours_sdaf: SqlDaf = result_da[ours_name]
        ae_sdaf: SqlDaf = SqlDaf(ae_name)
        report_lt.append((ours_name, len(ours_sdaf), len(ae_sdaf), rows_set(ours_sdaf) == rows_set(ae_sdaf)))
    disagreed_ae_ls: list = sorted(SqlDaf('disagreed_audit_ballots').sql(
            'SELECT ballot_id FROM disagreed_audit_ballots').to_list(), key=str)
    disagreed_ours_ls: list = sorted(result_da['disagreed_keys'], key=str)
    report_lt.append(('disagreed keys', len(disagreed_ours_ls), len(disagreed_ae_ls), disagreed_ours_ls == disagreed_ae_ls))
    return report_lt


def main() -> None:
    db_path, scratch_dir = sys.argv[1], sys.argv[2]
    work_path: str = os.path.join(scratch_dir, 'cmpcvr_trial.db')
    shutil.copyfile(db_path, work_path)

    SqlDaf.db_path = work_path
    cvr_name: str = 'cvr_votes!original' if SqlDaf.table_exists('cvr_votes!original') else 'cvr_votes!final'
    modi_name: str = 'cvr_votes!modified'

    # rebuild the inputs as they were before step 3 trimmed them.
    audit_sdaf: SqlDaf = SqlDaf('votes', keyfield='ballot_id')
    cvr_sdaf: SqlDaf = SqlDaf(cvr_name, keyfield='ballot_id')
    audit_sdaf.concat(SqlDaf('votes_residue'))
    cvr_sdaf.concat(SqlDaf('cvr_residue'))
    modi_sdaf: SqlDaf | None = SqlDaf(modi_name, keyfield='ballot_id') if SqlDaf.table_exists(modi_name) else None

    result_da: Dict[str, Any] = locate_variants_sdaf([audit_sdaf, cvr_sdaf], modi_sdaf, ['audit', 'cvr'], prefix='sd_')
    report_lt = compare(result_da, {
            'audit_variants': 'audit_variants', 'cvr_variants': 'cvr_variants',
            'audit_nonvariants': 'audit_nonvariants', 'cvr_nonvariants': 'cvr_nonvariants',
            'audit_residue': 'votes_residue', 'cvr_residue': 'cvr_residue'})
    SqlDaf.close()

    print('# Find Variants Report, SqlDaf trial\n')
    print(result_da['report_daf'].to_md(smart_fmt=True, shorten_text=False))
    print(f'\n{"table":20} {"SqlDaf":>7} {"AE":>7}  same, without {LATER_COLS_LS}')
    for name, ours, theirs, same in report_lt:
        print(f'{name:20} {ours:7} {theirs:7}  {same}')


if __name__ == '__main__':
    main()
