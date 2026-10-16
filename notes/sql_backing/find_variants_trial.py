# find_variants_trial.py
#
# A trial port of AuditEngine's locate_variants() to SqlDaf.
# It follows AuditEngine's steps one for one, as find_variants_sketch.md describes.
# It compares each result table with the one AuditEngine wrote in a real run.
#
# AuditEngine is not changed. Its database is copied, and the copy is used.
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
import json
import time
import shutil

from typing import Any, Dict, List, Tuple

from daffodil.daf import Daf, NULL
from daffodil.lib.daf_types import T_ls, T_da, T_loda
from daffodil.sqldaf import SqlDaf, T_sdaf, T_losdaf
from daffodil.lib.daf_sql import sql_unesc_str


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
        side_losdaf:         T_losdaf,            # the audit votes and the CVR votes.
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

    report(0, 'Input tables.', [len(side_sdaf) for side_sdaf in side_losdaf], [NULL, NULL])

    # step 1: drop any tables of an earlier run.
    for name in variants_names_ls + nonvariants_names_ls + residue_names_ls:
        if SqlDaf.table_exists(name):
            SqlDaf(name).delete()
    report(1, 'Dropping any existing variant tables.', [0, 0], [NULL, NULL])

    # step 2: empty variants tables, with a unique keyfield.
    variants_losdaf: T_losdaf = []
    secs_lf = []
    for i in range(2):
        variants_sdaf, secs_f = timed(lambda: SqlDaf(variants_names_ls[i], side_losdaf[i].columns(),
                                                     keyfield=keyfield).ensure_key_index())
        variants_losdaf.append(variants_sdaf)                # each with its own input's columns, as AuditEngine does.
        secs_lf.append(secs_f)
    report(2, 'Create Variant tables with indexes.', [len(sdaf) for sdaf in variants_losdaf], secs_lf)

    # step 3: the residue of each side, then deleted from it.
    residue_losdaf: T_losdaf = []
    secs_lf = []
    for i in range(2):
        residue_sdaf, secs_f = timed(lambda: side_losdaf[i].select_records_daf(
                side_losdaf[1 - i], inverse=True, silent_error=True).copy(name=residue_names_ls[i]).save())
        residue_losdaf.append(residue_sdaf)
        secs_lf.append(secs_f)
    for i in range(2):
        _, secs_f = timed(lambda: side_losdaf[i].delete_records(residue_losdaf[i], inplace=True, silent_error=True))
        secs_lf[i] += secs_f
    report(3, 'Remove Missing Records to Residue Tables', [len(sdaf) for sdaf in residue_losdaf], secs_lf)
    report(3, 'Remaining ballots in adjusted input tables', [len(sdaf) for sdaf in side_losdaf], [NULL, NULL])

    # step 4: the flagged rows.
    secs_lf = []
    for i in range(2):
        flagged_sdaf: SqlDaf = side_losdaf[i].select_where(lambda row: has_variant_flag(row['concise_votes']),
                                                          cols=['concise_votes'])
        secs_lf.append(timed(lambda: add_new_rows(variants_losdaf[i], flagged_sdaf))[1])
    report(4, 'Add records with variant flags', [len(sdaf) for sdaf in variants_losdaf], secs_lf)

    # step 5: harmonize, so each side has the variants of the other.
    secs_lf = []
    for i in range(2):
        secs_lf.append(timed(lambda: add_new_rows(
                variants_losdaf[i], side_losdaf[i].select_records_daf(variants_losdaf[1 - i], silent_error=True)))[1])
    report(5, 'Harmonize variant tables', [len(sdaf) for sdaf in variants_losdaf], secs_lf)

    # step 6: the ballots of the modified CVR.
    if modi_sdaf is not None:
        secs_lf = []
        for i in range(2):
            secs_lf.append(timed(lambda: add_new_rows(
                    variants_losdaf[i], side_losdaf[i].select_records_daf(modi_sdaf, silent_error=True)))[1])
        report(6, 'Add records from cvr_modified', [len(sdaf) for sdaf in variants_losdaf], secs_lf)

    # step 7: the rest are nonvariants for now.
    nonvariants_losdaf: T_losdaf = []
    secs_lf = []
    for i in range(2):
        nonvariants_sdaf, secs_f = timed(lambda: side_losdaf[i].select_records_daf(
                variants_losdaf[i], inverse=True, silent_error=True).copy(name=nonvariants_names_ls[i]).save())
        nonvariants_losdaf.append(nonvariants_sdaf)
        secs_lf.append(secs_f)
    report(7, 'Create nonflagged nonvariants', [len(sdaf) for sdaf in nonvariants_losdaf], secs_lf)

    # step 8: the pairs whose concise votes differ as text.
    pair_sdaf: SqlDaf = nonvariants_losdaf[0].join(nonvariants_losdaf[1], tag_other=True)
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
                variants_losdaf[i], nonvariants_losdaf[i].select_records_daf(disagreed_sdaf, silent_error=True)))[1])
    report(10, 'Add disagreed ballots to variants', [len(sdaf) for sdaf in variants_losdaf], secs_lf)

    # step 11: and are no longer nonvariants.
    secs_lf = []
    for i in range(2):
        secs_lf.append(timed(lambda: nonvariants_losdaf[i].delete_records(variants_losdaf[i], inplace=True, silent_error=True))[1])
    report(11, 'Remove variant ballot records from nonvariant ballots table.', [len(sdaf) for sdaf in nonvariants_losdaf], secs_lf)

    result_da: Dict[str, Any] = {}
    for i, name in enumerate(names_ls):
        result_da[f'{name}_variants'] = variants_losdaf[i]
        result_da[f'{name}_nonvariants'] = nonvariants_losdaf[i]
        result_da[f'{name}_residue'] = residue_losdaf[i]
    result_da['disagreed_keys'] = disagreed_sdaf.keys()
    result_da['report_daf'] = report_daf
    return result_da


CONTEST_KEY_RE = re.compile(r'^c(\d+)(b|ov|uv|g|w|o|v|r)')         # as in extract_contests() of sql_variants.py.
CONTEST_FLAG_RE = re.compile(r"""['"]c\d+(?:ov|w|g)['"](?=\s*:)""")  # a writein, overvote or gray contest, as in steps 13 and 14.
BLANK_CONCISE_KEY_RE = re.compile(r'^c\d+(b|uv)$')                  # as in move_blank_sheets_to_table().


def split_contests(row_da: T_da) -> T_loda:
    """
    One row per contest of a ballot, for apply(), as extract_contests() of AuditEngine gives.

    The votes of each contest are JSON with sorted keys and no spaces, as json_extract() of
    SQLite writes them there, so the text of two contests can be compared. contest_key joins the
    ballot and the contest index, so a join on it is a join on both.
    """
    by_cidx_doda: Dict[int, T_da] = {}
    key: str
    for key, val in (decoded_pyon(row_da['concise_votes']) or {}).items():
        match: re.Match | None = CONTEST_KEY_RE.match(key)
        if match:
            by_cidx_doda.setdefault(int(match.group(1)), {})[key] = val
    return [{'contest_key':   f"{row_da['ballot_id']}/{cidx}",
             'ballot_id':     row_da['ballot_id'],
             'cidx':          cidx,
             'concise_votes': json.dumps(dict(sorted(votes_da.items())), separators=(',', ':'))}
            for cidx, votes_da in by_cidx_doda.items()]


def has_contest_flag(concise_votes: Any) -> bool:
    """ utils.safe_regex_search() of AuditEngine, with the flag pattern of steps 13 and 14. """
    return bool(concise_votes) and bool(CONTEST_FLAG_RE.search(concise_votes))


def is_blank_sheet(concise_votes: Any) -> bool:
    """ concise_votes_are_blank() of AuditEngine: every contest on the sheet is fully undervoted. """
    votes_da: Any = decoded_pyon(concise_votes) or {}
    return any(key.endswith('b') for key in votes_da) and all(BLANK_CONCISE_KEY_RE.match(key) for key in votes_da)


def num_contests_of(concise_votes: Any) -> int:
    """ calc_num_contests_from_concise_votes() of AuditEngine: the keys that start a contest. """
    return len([key for key in decoded_pyon(concise_votes) if re.search(r'^c\d+b', key)])


def locate_contest_variants_sdaf(
        variants_losdaf:    T_losdaf,           # the audit and CVR variants, from steps 1 to 11.
        modi_sdaf:          T_sdaf | None,      # the modified CVR votes, if any.
        votes_sdaf:         T_sdaf,             # the audit votes, trimmed by step 3, for the metadata.
        pmvs_sdaf:          T_sdaf,             # the concise_pmv of each ballot.
        wins_sdaf:          T_sdaf,             # the concise_wins of each ballot.
        report_daf:         Daf,                # the step report, added to.
        ) -> Dict[str, Any]:                    # the result tables by name.
    """
    Split the variant ballots into contests, and find the contests that agree, as locate_variants()
    does after step 11.

    Each step is the same as in sql_variants.py, written with SqlDaf methods. A contest agrees when
    its audit and CVR votes are the same text, neither has a flag, and the modified CVR, if it has
    the contest, did not change it. The contests that agree are put back together, one row per
    ballot, as partial_nonvariants. The rest are combined_contest_variants, one row per contest,
    with the votes of each side.
    """
    result_da: Dict[str, Any] = {}
    names_ls: T_ls = ['audit_contests', 'cvr_contests', 'modi_contests']
    a_cv, c_cv, m_cv = [f'concise_votes_{name}' for name in names_ls]
    i: int

    def report(step: int, descr: str, counts_la: List[Any], secs_lf: List[Any]) -> None:
        report_daf.append({'Step': step, 'Description': descr, 'Audit recs': counts_la[0], 'CVR Recs': counts_la[1],
                           'Audit(s)': secs_lf[0], 'CVR(s)': secs_lf[1]})

    # step 12: one row per contest, for each side, and for the whole modified CVR.
    contests_losdaf: T_losdaf = []
    secs_lf: List[Any] = []
    for i in range(2):
        contests_sdaf, secs_f = timed(lambda: variants_losdaf[i].apply(split_contests).copy(name=names_ls[i]))
        contests_losdaf.append(contests_sdaf.set_keyfield('contest_key'))
        secs_lf.append(secs_f)
    report(12, 'Split ballot variants into contest-level tables', [len(sdaf) for sdaf in contests_losdaf], secs_lf)
    modi_contests_sdaf: SqlDaf | None = None
    if modi_sdaf is not None:
        modi_contests_sdaf = modi_sdaf.apply(split_contests).copy(name=names_ls[2]).set_keyfield('contest_key')

    # steps 13 and 14: pair the contests of the two sides, and split them into those that agree and the rest.
    start_f: float = time.perf_counter()
    pair_sdaf: SqlDaf = contests_losdaf[1].join(contests_losdaf[0], shared_fields=['ballot_id', 'cidx'], tag_other=True)
    if modi_contests_sdaf is not None:
        pair_sdaf = pair_sdaf.join(modi_contests_sdaf, how='left', shared_fields=['ballot_id', 'cidx'], tag_other=True)
    pair_cols_ls: T_ls = [a_cv, c_cv, m_cv] if modi_contests_sdaf is not None else [a_cv, c_cv]

    def contest_agrees(row: Any) -> bool:
        modi_cv: Any = row[m_cv] if modi_contests_sdaf is not None else NULL
        return (row[a_cv] == row[c_cv] and not has_contest_flag(row[a_cv]) and not has_contest_flag(row[c_cv])
                and (modi_cv is NULL or modi_cv == row[c_cv]))

    agreed_sdaf: SqlDaf = pair_sdaf.select_where(contest_agrees, cols=pair_cols_ls)
    nonvariant_contests_sdaf: SqlDaf = contests_losdaf[1].select_records_daf(agreed_sdaf)
    num_rows, secs_f = timed(lambda: len(nonvariant_contests_sdaf))
    report(13, "Compare '_by_contest' tables to produce 'nonvariant_contests' table.", [num_rows, NULL],
           [time.perf_counter() - start_f, NULL])

    def combined_row(row_da: T_da) -> T_da:
        return {'ballot_id':      row_da['ballot_id'],
                'cidx':           row_da['cidx'],
                'concise_votes1': row_da[a_cv],
                'concise_votes2': row_da[c_cv],
                'concise_votes3': (row_da[m_cv] or '{}') if modi_contests_sdaf is not None else '{}'}

    start_f = time.perf_counter()
    disagreed_sdaf: SqlDaf = pair_sdaf.select_where(lambda row: not contest_agrees(row), cols=pair_cols_ls)
    combined_sdaf: SqlDaf = (disagreed_sdaf.apply(combined_row).set_keyfield('ballot_id')
                             .join(pmvs_sdaf.select_cols(['ballot_id', 'concise_pmv']))
                             .join(wins_sdaf, how='left'))
    num_rows = len(combined_sdaf)
    report(14, "Join '_by_contest' tables to produce 'combined_contest_variants'.", [num_rows, NULL],
           [time.perf_counter() - start_f, NULL])

    # step 15: the contests that agree, put back together, one row per ballot.
    start_f = time.perf_counter()
    contests_sdaf = nonvariant_contests_sdaf.copy()                 # a table, so that SQL can name it.
    contests_sdaf.sql(                                              # SqlDaf has no reduce by group yet. This is AuditEngine's query.
            f'CREATE TEMP TABLE recombined AS SELECT ballot_id, json_group_object(key, value) AS concise_votes '
            f'FROM (SELECT ballot_id, json_each.key AS key, json_each.value AS value '
            f'FROM "{contests_sdaf.name}", json_each(concise_votes)) GROUP BY ballot_id')
    partial_sdaf: SqlDaf = SqlDaf('recombined').apply(
            lambda row_da: row_da | {'num_contests': num_contests_of(row_da['concise_votes'])}).set_keyfield('ballot_id')
    num_rows = len(partial_sdaf)
    report(15, "Recombine 'nonvariant_contests' into 'partial_nonvariants'", [num_rows, NULL],
           [time.perf_counter() - start_f, NULL])

    # step 16: an empty row for each variant ballot that has no contest that agrees.
    start_f = time.perf_counter()
    combined_ids_ls: list = list(combined_sdaf.valuecounts_for_colname('ballot_id'))    # a ballot has a row per contest here.
    partial_ids_set: set = set(partial_sdaf.keys())
    missing_ls: list = sorted(ballot_id for ballot_id in combined_ids_ls if ballot_id not in partial_ids_set)   # in key order, as AuditEngine's index gives them.
    partial_sdaf.concat(Daf(lol=[[ballot_id, '{}', 0] for ballot_id in missing_ls], cols=['ballot_id', 'concise_votes', 'num_contests']))
    num_rows = len(partial_sdaf)
    report(16, 'Ensure_partial_nonvariants_have_all_variant_ballots (even empty shells)', [num_rows, NULL],
           [time.perf_counter() - start_f, NULL])

    # step 17: the sheets blank on both sides are moved to blank_sheets.
    start_f = time.perf_counter()
    metadata_sdaf: SqlDaf = votes_sdaf.select_cols(exclude_cols=['concise_votes', 'num_contests'])
    blank_sdaf: SqlDaf = (partial_sdaf.select_records_daf(combined_ids_ls, inverse=True, silent_error=True)
                          .select_where(lambda row: is_blank_sheet(row['concise_votes']), cols=['concise_votes'])
                          .copy())
    blank_sheets_sdaf: SqlDaf = blank_sdaf.join(metadata_sdaf)
    partial_sdaf.delete_records(blank_sdaf, inplace=True)
    partial_wmetadata_sdaf: SqlDaf = partial_sdaf.join(metadata_sdaf)
    num_rows = len(blank_sheets_sdaf)
    report(17, 'Move sheets blank in audit and CVR to blank_sheets (ballot variants)', [num_rows, NULL],
           [time.perf_counter() - start_f, NULL])

    # step 18, first part: the variant contests with the metadata of the ballot, for create_comparison_records().
    # The second part is AuditEngine's own function on each row, which this trial cannot import:
    #   joined_sdaf.apply(create_contest_variant_record).to_csv_file(contest_variants_path)
    # apply() streams, where AuditEngine today reads the whole table into a Daf first.
    start_f = time.perf_counter()
    joined_sdaf: SqlDaf = combined_sdaf.join(votes_sdaf)
    num_rows = len(joined_sdaf)
    report(18, "Create 'contest_variants' compatible with cmpcvr and AdjudiTally app SQL Join", [num_rows, NULL],
           [time.perf_counter() - start_f, NULL])

    result_da['audit_variants_by_contest'] = contests_losdaf[0]
    result_da['cvr_variants_by_contest'] = contests_losdaf[1]
    result_da['nonvariant_contests'] = nonvariant_contests_sdaf
    result_da['combined_contest_variants'] = combined_sdaf
    result_da['partial_nonvariants'] = partial_sdaf
    result_da['partial_nonvariants_wmetadata'] = partial_wmetadata_sdaf
    result_da['blank_sheets'] = blank_sheets_sdaf
    result_da['joined_contests'] = joined_sdaf
    return result_da


VALUECOUNTS_COLS_LS: T_ls = ['batchid', 'style', 'precinct', '__67roup', 'party']   # as in cmpcvr.py. AuditEngine stores group as __67roup.
DIRECT_SUM_COLS_LS: T_ls = ['blank', 'v', 'is_bmd']                                   # the columns of sumcols_ls in cmpcvr.py. The rest are in concise_votes.

# Stats that cmpcvr.py writes but nothing reads, so they are not made.
# Set one to True to make it again.
MAKE_KEYS_LS:       bool = False    # the ballot_id of each row: most of the bytes of the stats.
MAKE_PRECINCT_SUMS: bool = False    # the sums for each precinct: most of the time of the stats.


def nonvariant_stats_sdaf(
        nonvariants_sdaf:       T_sdaf,         # the audit nonvariants, from steps 1 to 11.
        partial_wmetadata_sdaf: T_sdaf,         # the partly agreed ballots, from step 17.
        sumcols_ls:             T_ls,           # the columns to total: DIRECT_SUM_COLS_LS, then the keys of concise_votes.
        ) -> Dict[str, Any]:                    # the stats, by the names of cmpcvr.py.
    """
    The stats of the ballots that agree, as the code after locate_variants() in cmpcvr.py gives them.

    num_contests is counted from concise_votes. The nonvariants and the partly agreed ballots are
    put together, as combined_nonvariants, and totaled. A total of a key of concise_votes reads
    the dict in each row, which is JSON or PYON text. AuditEngine gets sumcols_ls from the
    contests of the job, so here it is given.

    The keys list and the precinct sums are left out unless MAKE_KEYS_LS or MAKE_PRECINCT_SUMS
    is True.
    """
    indirect_cols_set: set = {col for col in sumcols_ls if col not in DIRECT_SUM_COLS_LS}     # a set: sum_da() tests each key in it.

    def totals_di(rows_sdaf: SqlDaf, absent: Any) -> Dict[str, Any]:
        """ The totals of sumcols_ls in these rows. A key that no row has is absent. """
        direct_da: Dict[str, Any] = rows_sdaf.sum(DIRECT_SUM_COLS_LS)
        indirect_da: Dict[str, Any] = rows_sdaf.reduce(Daf.sum_da, by='sparse_row', indirect_col='concise_votes', cols=indirect_cols_set)
        return {col: int(direct_da[col]) if col in direct_da else indirect_da.get(col, absent) for col in sumcols_ls}

    counted_sdaf: SqlDaf = nonvariants_sdaf.apply(
            lambda row_da: row_da | {'num_contests': num_contests_of(row_da['concise_votes'])}).set_keyfield('ballot_id')
    combined_sdaf: SqlDaf = counted_sdaf.copy(name='combined_nonvariants_t')
    combined_sdaf.concat(partial_wmetadata_sdaf.select_cols(counted_sdaf.columns()))

    valuecounts_dodi: Dict[str, Any] = combined_sdaf.valuecounts_for_colnames_ls(VALUECOUNTS_COLS_LS)
    stats_da: Dict[str, Any] = {
        'combined_nonvariants':                         combined_sdaf,
        'nonvariant_sums_di':                           totals_di(combined_sdaf, 0),
        'nonvariant_valuecounts_dodi':                  {sql_unesc_str(col): vc_di for col, vc_di in valuecounts_dodi.items()},
        'nonvariant_precinct_styles_valuecounts_dodi':  counted_sdaf.valuecounts_for_colname1_groupedby_colname2('style', 'precinct'),
        'nonvariant_sheets_contests_di':                {'sheets': len(counted_sdaf), 'contests': int(counted_sdaf.sum(['num_contests'])['num_contests'])},
        }
    if MAKE_PRECINCT_SUMS:
        stats_da['nonvariant_precinct_sums_dodi'] = {precinct: totals_di(group_sdaf, None)     # SQL SUM() of no value is SQL NULL.
                                                     for precinct, group_sdaf in combined_sdaf.groupby('precinct').items()}
    if MAKE_KEYS_LS:
        stats_da['nonvariant_keys_ls'] = [str(key) for key in combined_sdaf.keys()]
    return stats_da


def compare_stats(stats_da: Dict[str, Any], cmpcvr_dir: str) -> List[Tuple[str, int, int, bool]]:
    """
    Compare each stat with the JSON file AuditEngine wrote, after a JSON round trip of ours.
    Only the stats that were made are compared.

    A list of (name, ours, theirs, same), where ours and theirs are the lengths. A file that
    is not there is shown with -1.
    """
    report_lt: List[Tuple[str, int, int, bool]] = []
    name: str
    for name, ours_a in stats_da.items():
        if name == 'combined_nonvariants':
            continue
        path: str = os.path.join(cmpcvr_dir, f'{name}.json')
        ours_a = json.loads(json.dumps(ours_a))
        if not os.path.exists(path):
            report_lt.append((name, len(ours_a), -1, False))
            continue
        with open(path) as json_fh:
            ae_a: Any = json.load(json_fh)
        report_lt.append((name, len(ours_a), len(ae_a), ours_a == ae_a))
    return report_lt


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


def decoded_cell(val: Any) -> Any:
    """ A dict in JSON or PYON text as sorted pairs, so that two texts of the same dict compare equal. """
    if isinstance(val, str) and val.startswith('{'):
        decoded: Any = decoded_pyon(val)
        if isinstance(decoded, dict):
            return tuple(sorted(decoded.items()))
    return val


def compare_contest_tables(result_da: Dict[str, Any]) -> List[Tuple[str, int, int, bool, bool]]:
    """
    Compare each table of steps 12 to 17 with the one AuditEngine wrote, as sets of rows.

    A list of (name, ours, theirs, same text, same values). Same values decodes each dict in
    JSON or PYON first, so it ignores the order of the keys. The column contest_key of the port
    is left out. A table of AuditEngine that is not in the file is shown with -1 rows.
    """
    report_lt: List[Tuple[str, int, int, bool, bool]] = []
    name: str
    ours_sdaf: SqlDaf
    for name, ours_sdaf in result_da.items():
        if not SqlDaf.table_exists(name) and not len(ours_sdaf.sql(f"SELECT 1 FROM sqlite_master WHERE name = ?", [name])):
            report_lt.append((name, len(ours_sdaf), -1, False, False))
            continue
        ae_daf: Daf = ours_sdaf.sql(f'SELECT * FROM "{name}"')       # some are views, which SqlDaf cannot open in order.
        ae_cols_ls: T_ls = ae_daf.columns()
        ours_daf: Daf = ours_sdaf.to_daf()
        if 'concise_votes3' in ours_daf.hd and 'concise_votes3' not in ae_cols_ls:     # AuditEngine before the modified CVR was added.
            if set(ours_daf[:, 'concise_votes3'].to_list()) <= {'{}'}:
                ours_daf = ours_daf.select_cols(exclude_cols=['concise_votes3'])
        if sorted(ours_daf.columns()) != sorted(ae_cols_ls + (['contest_key'] if 'contest_key' in ours_daf.hd else [])):
            print(f'{name}: columns differ\n  ours {ours_daf.columns()}\n  AE   {ae_cols_ls}')
            report_lt.append((name, len(ours_daf), len(ae_daf), False, False))
            continue
        ours_lol: list = ours_daf.select_cols(ae_cols_ls).lol
        ae_lol: list = ae_daf.lol
        same_text: bool = {tuple(row_la) for row_la in ours_lol} == {tuple(row_la) for row_la in ae_lol}
        same_vals: bool = ({tuple(decoded_cell(val) for val in row_la) for row_la in ours_lol} ==
                           {tuple(decoded_cell(val) for val in row_la) for row_la in ae_lol})
        report_lt.append((name, len(ours_lol), len(ae_lol), same_text, same_vals))
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

    contest_da: Dict[str, Any] = locate_contest_variants_sdaf(
            variants_losdaf = [result_da['audit_variants'], result_da['cvr_variants']],
            modi_sdaf       = modi_sdaf,
            votes_sdaf      = audit_sdaf,
            pmvs_sdaf       = SqlDaf('pmvs', keyfield='ballot_id'),
            wins_sdaf       = SqlDaf('wins', keyfield='ballot_id'),
            report_daf      = result_da['report_daf'])
    contest_report_lt = compare_contest_tables(contest_da)

    # step 17, the export: partially_agreed.csv, compared with AuditEngine's as sets of lines.
    csv_path: str = os.path.join(scratch_dir, 'partially_agreed.csv')
    contest_da['partial_nonvariants_wmetadata'].to_csv_file(csv_path)
    ae_csv_path: str = os.path.join(os.path.dirname(db_path), 'partially_agreed.csv')
    if os.path.exists(ae_csv_path):
        with open(csv_path) as ours_fh, open(ae_csv_path) as ae_fh:
            ours_lines_ls: T_ls = ours_fh.read().splitlines()
            ae_lines_ls: T_ls = ae_fh.read().splitlines()
        contest_report_lt.append(('partially_agreed.csv', len(ours_lines_ls), len(ae_lines_ls),
                                  set(ours_lines_ls[1:]) == set(ae_lines_ls[1:]), ours_lines_ls[0] == ae_lines_ls[0]))
        if ours_lines_ls[0] != ae_lines_ls[0]:
            print(f'partially_agreed.csv header\n  ours {ours_lines_ls[0]}\n  AE   {ae_lines_ls[0]}')

    # the stats of cmpcvr.py. sumcols_ls comes from the contests of the job there, so it is read from AuditEngine's file.
    cmpcvr_dir: str = os.path.dirname(db_path)
    sums_path: str = os.path.join(cmpcvr_dir, 'nonvariant_sums_di.json')
    stats_report_lt: List[Tuple[str, int, int, bool]] = []
    if os.path.exists(sums_path):
        with open(sums_path) as json_fh:
            sumcols_ls: T_ls = list(json.load(json_fh))
        start_f: float = time.perf_counter()
        stats_da: Dict[str, Any] = nonvariant_stats_sdaf(result_da['audit_nonvariants'], contest_da['partial_nonvariants_wmetadata'], sumcols_ls)
        stats_secs_f: float = time.perf_counter() - start_f
        stats_report_lt = compare_stats(stats_da, cmpcvr_dir)
        contest_report_lt += compare_contest_tables({'combined_nonvariants': stats_da['combined_nonvariants']})
    SqlDaf.close()

    print('# Find Variants Report, SqlDaf trial\n')
    print(result_da['report_daf'].to_md(smart_fmt=True, shorten_text=False))
    print(f'\n{"table":20} {"SqlDaf":>7} {"AE":>7}  same, without {LATER_COLS_LS}')
    for name, ours, theirs, same in report_lt:
        print(f'{name:20} {ours:7} {theirs:7}  {same}')
    print(f'\n{"table":30} {"SqlDaf":>7} {"AE":>7}  same text, same values (for the csv: same rows, same header)')
    for name, ours, theirs, same_text, same_vals in contest_report_lt:
        print(f'{name:30} {ours:7} {theirs:7}  {same_text}, {same_vals}')
    if stats_report_lt:
        print(f'\n{"stats, in {stats_secs_f:.2f} s":45} {"SqlDaf":>7} {"AE":>7}  same'.replace('{stats_secs_f:.2f}', f'{stats_secs_f:.2f}'))
        for name, ours, theirs, same in stats_report_lt:
            print(f'{name:45} {ours:7} {theirs:7}  {same}')


if __name__ == '__main__':
    main()
