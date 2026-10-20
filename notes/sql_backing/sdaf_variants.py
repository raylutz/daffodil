# sdaf_variants.py

"""
Find the variant ballots with SqlDaf, as find_variants_using_sql() of cmpcvr.py does.

find_variants runs this when the setting find_variants_engine is 'sdaf'.
It takes over after the input tables are imported into cmpcvr.db.
From there it does all that find_variants_using_sql() does:

- the split BMD CVR records are combined, if combine_split_bmd_cvrs is set.
- steps 1 to 18 of locate_variants() in sql_variants.py, and its report.
- the blank sheets are saved as ballot variants.
- the nonvariant stats of reduce_nonvariants() are made.
- the CSV files are written and uploaded.

The import stays in cmpcvr.py, since it is file input from S3.
The work on the tables is done with SqlDaf methods.

Names
-----
AuditEngine's import stores some column names encoded, such as group as __67roup.
SqlDaf shows a name as it is stored.
So each row given to an AuditEngine function has its names decoded first.
AuditEngine's own reader decodes them with sql_unesc_str() of sql_utils.py, and so does this module.
SqlDaf decodes them in the header of a CSV file.

Tables left in cmpcvr.db
------------------------
The result tables are kept under the names sql_variants.py gives them, for diagnosis.
Nothing after this reads them.
They are SqlDaf tables, so each has the hidden daf_rid column.

Scope
-----
Only concise votes are handled, with a CVR and an optional modified CVR.
Row order in the tables and the CSV files is not kept. Nothing reads it.
A repeated ballot_id in an input raises ValueError.
"""

import re
import json
import time
import itertools

from typing import Any, Callable, Dict, Iterator, List, Tuple

from daffodil.daf import Daf
from daffodil.sqldaf import SqlDaf, T_sdaf, T_losdaf
import daffodil.lib.daf_utils as daf_utils

from models.DB import DB
from my_types import T_da, T_di, T_doda, T_dodi, T_la, T_ls, T_loda
from utilities import cmpcvr, generic_cvr, logs, reports, split_bmd, sql_utils, sql_variants, utils


KEYFIELD:           str     = 'ballot_id'
REPORT_COLS_LS:     T_ls    = ['Step', 'Description', 'Audit recs', 'CVR Recs', 'Audit(s)', 'CVR(s)']
WORK_PREFIX:        str     = 'sdaf_'       # the TEMP work tables of a run.
APPLY_BATCH_ROWS:   int     = 10000         # the rows held in memory at a time in step 18.
CONTEST_VARIANTS_TABLE: str = 'contest_variants'    # the table of step 18, exported as contest_variants.csv.

# Patterns of sql_variants.py.
VARIANT_FLAG_PAT:       str         = r"'c\d+(w|ov|g)'"                             # step 4: a writein, overvote or gray contest.
CONTEST_FLAG_PAT:       str         = r"""['"]c\d+(?:ov|w|g)['"](?=\s*:)"""        # steps 13 and 14.
CONTEST_KEY_RE:         re.Pattern  = re.compile(r'^c(\d+)(b|ov|uv|g|w|o|v|r)')   # extract_contests().
BLANK_CONCISE_KEY_RE:   re.Pattern  = re.compile(r'^c\d+(b|uv)$')                  # move_blank_sheets_to_table().
CONTEST_START_RE:       re.Pattern  = re.compile(r'^c\d+b')                       # the key that starts a contest, for num_contests.

# Columns of reduce_nonvariants() in cmpcvr.py, by their decoded names.
DIRECT_SUM_COLS_LS:     T_ls    = ['blank', 'v', 'is_bmd']
VALUECOUNTS_COLS_LS:    T_ls    = ['batchid', 'style', 'precinct', 'group', 'party']

# The tables locate_variants() leaves, by the names it gives them.
RESULT_TABLES_LS: T_ls = [
    'audit_variants', 'cvr_variants', 'audit_nonvariants', 'cvr_nonvariants', 'votes_residue', 'cvr_residue',
    'audit_variants_by_contest', 'cvr_variants_by_contest', 'nonvariant_contests', 'combined_contest_variants',
    'partial_nonvariants', 'partial_nonvariants_wmetadata', 'blank_sheets', 'joined_contests', 'combined_nonvariants',
    CONTEST_VARIANTS_TABLE,
    ]


#===================================
# Names

def decoded_row(row_da: T_da) -> T_da:
    """ The row with each stored name decoded, as AuditEngine's own reader gives it. """
    decoded_da: T_da = {sql_utils.sql_unesc_str(col): val for col, val in row_da.items()}
    return decoded_da


def decoded_daf(rows_daf: Daf) -> Daf:
    """ The Daf with each stored column name decoded. """
    renames_ds: Dict[str, str] = {col: sql_utils.sql_unesc_str(col) for col in rows_daf.columns() if sql_utils.sql_unesc_str(col) != col}
    if renames_ds:
        rows_daf.rename_cols(renames_ds)
    return rows_daf


def stored_cols(rows_sdaf: SqlDaf, decoded_cols_ls: T_ls) -> T_ls:
    """ The stored names of these columns, given by their decoded names. """
    stored_by_decoded_ds: Dict[str, str] = {sql_utils.sql_unesc_str(col): col for col in rows_sdaf.columns()}
    stored_ls: T_ls = [stored_by_decoded_ds[col] for col in decoded_cols_ls]
    return stored_ls


#===================================
# Row functions

def decoded_votes(concise_votes: Any) -> T_da:
    """
    The dict of a concise_votes cell, PYON or JSON text, read with pyjson5 by `unflatten_val()`.

    An empty cell gives an empty dict.
    Text that is not a dict raises ValueError, so bad data stops the run.
    """
    if not concise_votes:
        return {}
    votes_a: Any = daf_utils.unflatten_val(concise_votes)
    if not isinstance(votes_a, dict):
        raise ValueError(f"sdaf_variants: a concise_votes cell is not a dict: {str(concise_votes)[:80]!r}")
    return votes_a


def has_variant_flag(concise_votes: Any) -> bool:
    """ The test of step 4: the ballot has a writein, an overvote or a gray contest. """
    flagged: bool = utils.safe_regex_search(VARIANT_FLAG_PAT, concise_votes)
    return flagged


def has_contest_flag(concise_votes: Any) -> bool:
    """ The test of steps 13 and 14, on the votes of one contest. """
    flagged: bool = utils.safe_regex_search(CONTEST_FLAG_PAT, concise_votes)
    return flagged


def concise_votes_differ(concise_votes1: Any, concise_votes2: Any) -> bool:
    """
    The test of step 9: the votes still differ once each is decoded.

    normalize_pyon() of sql_variants.py sorts the keys of each and compares the text.
    Two dicts are equal whatever the order of their keys, so comparing the dicts is the same test.
    """
    differ: bool = decoded_votes(concise_votes1) != decoded_votes(concise_votes2)
    return differ


def split_contests(row_da: T_da) -> T_loda:
    """
    One row per contest of a ballot, as extract_contests() gives them in step 12.

    The votes of a contest are JSON with sorted keys and no spaces.
    That is the text json_extract() writes, so two contests compare as text.
    contest_key holds the ballot and the contest index.
    A one-key join on it is a join on both.
    """
    by_cidx_doda: Dict[int, T_da] = {}
    key: str
    val: Any
    for key, val in decoded_votes(row_da['concise_votes']).items():
        match: re.Match | None = CONTEST_KEY_RE.match(key)
        if match:
            by_cidx_doda.setdefault(int(match.group(1)), {})[key] = val

    contests_loda: T_loda = []
    cidx: int
    votes_da: T_da
    for cidx, votes_da in by_cidx_doda.items():
        contests_loda.append({'contest_key':     f"{row_da['ballot_id']}/{cidx}",
                              'ballot_id':       row_da['ballot_id'],
                              'cidx':            cidx,
                              'concise_votes':   contest_votes_json(votes_da)})
    return contests_loda


def contest_votes_json(votes_da: T_da) -> str:
    """ The votes of one contest as JSON, with sorted keys and no spaces. """
    votes_json: str = json.dumps(dict(sorted(votes_da.items())), separators=(',', ':'))
    return votes_json


def is_blank_sheet(concise_votes: Any) -> bool:
    """ The test of step 17: every contest on the sheet is fully undervoted. """
    votes_da: T_da = decoded_votes(concise_votes)
    has_contest: bool = any(key.endswith('b') for key in votes_da)
    all_blank: bool = all(BLANK_CONCISE_KEY_RE.match(key) for key in votes_da)
    blank: bool = has_contest and all_blank
    return blank


def num_contests_of(concise_votes: Any) -> int:
    """ The contests of a ballot: the keys that start a contest, as calc_num_contests_from_concise_votes() counts them. """
    num_contests: int = len([key for key in decoded_votes(concise_votes) if CONTEST_START_RE.match(key)])
    return num_contests


def update_da(
        row_da:         T_da,                   # the dict of one row.
        reduction_da:   T_da,                   # the result so far. It is changed.
        *,
        cols:           Any = None,             # the keys to take. All if None.
        is_sparse:      bool = False,           # given by Daf.reduce(). Not used.
        diagnose:       bool = False,           # given by Daf.reduce(). Not used.
        ) -> T_da:                              # the result, with the row merged in.
    """
    Merge the dict of a row into the result, for `groupby_aggregate()` of a dict column.

    Each key takes the row's value, as `dict.update()` does.
    The contests of a ballot have their own keys, so nothing is overwritten.
    """
    key: str
    val: Any
    for key, val in row_da.items():
        if cols and key not in cols:
            continue
        reduction_da[key] = val
    return reduction_da


def keep_no_value(
        row_da:         T_da,                   # the dict of one row. Not used.
        reduction_da:   T_da,                   # the result so far, returned as it is.
        **kwargs:       Any,                    # given by Daf.reduce(). Not used.
        ) -> T_da:                              # the result, unchanged.
    """
    Keep nothing of the rows, for `groupby_reduce()` with `by='row'`.

    The result has one row for each value of the group column, with the other cells empty.
    So it gives the distinct values of the column, as a table keyed by it.
    """
    return reduction_da


#===================================
# Tables

def timed(func: Callable[[], Any]) -> Tuple[Any, float]:
    """ Call a function of no arguments. Give its result and the seconds it took. """
    start_f: float = time.perf_counter()
    result: Any = func()
    return result, time.perf_counter() - start_f


def add_new_rows(target_sdaf: SqlDaf, rows_sdaf: SqlDaf) -> None:
    """ INSERT OR IGNORE of sql_variants.py: add the rows whose key is not in the target yet. """
    target_sdaf.concat(rows_sdaf.select_records_daf(target_sdaf, inverse=True, silent_error=True))


def drop_table(any_sdaf: SqlDaf, name: str) -> None:
    """ Remove the table or view of this name, if there is one. any_sdaf is any SqlDaf of the same file. """
    kinds_ls: T_ls = any_sdaf.sql('SELECT type FROM sqlite_master WHERE name = ?', [name]).to_list()
    if not kinds_ls:
        return
    if kinds_ls[0] == 'table' and SqlDaf.table_exists(name):
        SqlDaf(name).delete()                               # removes what SqlDaf recorded about it too.
        return
    any_sdaf.sql(f'DROP {kinds_ls[0].upper()} IF EXISTS "{name}"')


def keep_table(rows_sdaf: SqlDaf, name: str, exclude_cols_ls: T_ls | None = None) -> SqlDaf:
    """
    Keep the rows in the file as a table of this name, without the columns of exclude_cols_ls.

    A table made under this name is kept as it is.
    Any other is copied, and the copy is saved under the name.
    """
    if rows_sdaf.name == name and not exclude_cols_ls:
        return rows_sdaf
    kept_sdaf: SqlDaf = rows_sdaf.select_cols(exclude_cols=exclude_cols_ls) if exclude_cols_ls else rows_sdaf
    return kept_sdaf.copy(name=f'{WORK_PREFIX}keep_{name}').save(name)


def export_csv(
        rows_sdaf:      SqlDaf,         # the rows to write.
        dirname:        str,            # the AuditEngine dirname of the file.
        name:           str,            # the file name, such as partially_agreed.csv.
        ) -> str:                       # the S3 path.
    """ Write the rows to a local CSV file and upload it, as export_sqlite_table_to_csv_stream() of sql_utils.py does. """
    logs.stsloc(f"Writing '{name}' to s3 (+ mirror)", 3)
    rows_sdaf.to_csv_file(DB.build_filepath(dirname=dirname, name=name, s3flag=False))
    return DB.upload_file_dirname(dirname=dirname, filename=name, local_mirror=True)


#===================================
# Split BMD CVR records

def combine_split_bmd_sdaf(
        cvr_sdaf:       SqlDaf,         # the CVR votes, keyed by ballot_id. Changed in place.
        groups_dod:     T_doda,         # the groups of split_bmd_combined.json, by the ballot_id kept.
        ) -> int:                       # the number of groups combined.
    """
    Combine the split CVR records of BMD ballots, as combine_split_bmd_cvrs_in_sqlite() of split_bmd.py does.

    The kept row of each group gets the combined votes.
    The other rows of the group are deleted.
    A group is skipped if its kept row or all of its other rows are missing.
    """
    setcols_ls: T_ls = [col for col in ('hexstyle', 'v', 'num_contests', 'blank', 'concise_votes') if col in cvr_sdaf.columns()]
    num_combined: int = 0
    kept_id: str
    group_da: T_da
    with cvr_sdaf.transaction():
        for kept_id, group_da in groups_dod.items():
            kept_rec_da: T_da = cvr_sdaf.select_record(kept_id)
            sibling_ids_ls: T_ls = [bid for bid in group_da['combined_ids'] if cvr_sdaf.select_record(bid)]
            if not kept_rec_da or not sibling_ids_ls:
                continue

            combined_da: T_da = split_bmd.combine_cvr_records(kept_rec_da, [cvr_sdaf.select_record(bid) for bid in sibling_ids_ls])
            combined_da['concise_votes'] = str(combined_da['concise_votes'])
            cvr_sdaf.update_record_krow(kept_id, {col: combined_da[col] for col in setcols_ls})
            cvr_sdaf.delete_records(sibling_ids_ls, inplace=True)
            num_combined += 1
    return num_combined


#===================================
# Steps 1 to 11

def locate_ballot_variants_sdaf(
        side_losdaf:        T_losdaf,           # the audit votes and the CVR votes. Step 3 trims them.
        modi_sdaf:          T_sdaf | None,      # the modified CVR votes, if any.
        variants_names_ls:  T_ls,               # the names of the variants tables, audit then CVR.
        report_daf:         Daf,                # the step report, added to.
        ) -> T_da:                              # the result tables, as lists of two SqlDaf, audit then CVR.
    """
    Steps 1 to 11: split the ballots into variants and nonvariants, for each side.

    A ballot is a variant when it has a flag, when the other side has it as a variant,
    when the modified CVR has it, or when its votes differ from the other side's.
    The ballots that only one side has go to that side's residue, and leave the input.
    """
    def report(step: int, descr: str, counts_la: List[Any], secs_la: List[Any]) -> None:
        report_daf.append({'Step': step, 'Description': descr, 'Audit recs': counts_la[0], 'CVR Recs': counts_la[1],
                           'Audit(s)': secs_la[0], 'CVR(s)': secs_la[1]})

    i: int
    secs_f: float
    report(0, 'Input tables.', [len(side_sdaf) for side_sdaf in side_losdaf], ['', ''])

    # Step 1: the result tables of an earlier run go.
    for name in RESULT_TABLES_LS:
        drop_table(side_losdaf[0], name)
    report(1, 'Dropping any existing variant tables.', [0, 0], ['', ''])

    # Step 2: empty variants tables, each with its own input's columns and a unique key.
    # They are made under their names in the file, as sql_variants.py makes them.
    variants_losdaf: T_losdaf = []
    secs_la: List[Any] = []
    for i in range(2):
        variants_sdaf, secs_f = timed(lambda: SqlDaf(variants_names_ls[i], cols=side_losdaf[i].columns(), keyfield=KEYFIELD).ensure_key_index())
        variants_losdaf.append(variants_sdaf)
        secs_la.append(secs_f)
    report(2, 'Create Variant tables with indexes.', [len(sdaf) for sdaf in variants_losdaf], secs_la)

    # Step 3: the ballots only one side has are its residue, and leave its input.
    residue_losdaf: T_losdaf = []
    secs_la = []
    for i in range(2):
        residue_sdaf, secs_f = timed(lambda: side_losdaf[i].select_records_daf(
                side_losdaf[1 - i], inverse=True, silent_error=True).copy())
        residue_losdaf.append(residue_sdaf)
        secs_la.append(secs_f)
    for i in range(2):
        secs_la[i] += timed(lambda: side_losdaf[i].delete_records(residue_losdaf[i], inplace=True, silent_error=True))[1]
    report(3, 'Remove Missing Records to Residue Tables', [len(sdaf) for sdaf in residue_losdaf], secs_la)
    report(3, 'Remaining ballots in adjusted input tables', [len(sdaf) for sdaf in side_losdaf], ['', ''])

    # Step 4: the flagged ballots.
    secs_la = []
    for i in range(2):
        flagged_sdaf: SqlDaf = side_losdaf[i].select_where(lambda row: has_variant_flag(row['concise_votes']), cols=['concise_votes'])
        secs_la.append(timed(lambda: add_new_rows(variants_losdaf[i], flagged_sdaf))[1])
    report(4, 'Add records with variant flags', [len(sdaf) for sdaf in variants_losdaf], secs_la)

    # Step 5: each side takes the variants of the other.
    secs_la = []
    for i in range(2):
        secs_la.append(timed(lambda: add_new_rows(
                variants_losdaf[i], side_losdaf[i].select_records_daf(variants_losdaf[1 - i], silent_error=True)))[1])
    report(5, 'Harmonize variant tables', [len(sdaf) for sdaf in variants_losdaf], secs_la)

    # Step 6: the ballots of the modified CVR.
    if modi_sdaf is not None:
        secs_la = []
        for i in range(2):
            secs_la.append(timed(lambda: add_new_rows(
                    variants_losdaf[i], side_losdaf[i].select_records_daf(modi_sdaf, silent_error=True)))[1])
        report(6, 'Add records from cvr_modified', [len(sdaf) for sdaf in variants_losdaf], secs_la)

    # Step 7: the rest are nonvariants for now.
    nonvariants_losdaf: T_losdaf = []
    secs_la = []
    for i in range(2):
        nonvariants_sdaf, secs_f = timed(lambda: side_losdaf[i].select_records_daf(
                variants_losdaf[i], inverse=True, silent_error=True).copy())
        nonvariants_losdaf.append(nonvariants_sdaf.set_keyfield(KEYFIELD))
        secs_la.append(secs_f)
    report(7, 'Create nonflagged nonvariants', [len(sdaf) for sdaf in nonvariants_losdaf], secs_la)

    # Step 8: the pairs whose concise votes differ as text.
    pair_sdaf: SqlDaf = nonvariants_losdaf[0].join(nonvariants_losdaf[1], tag_other=True)
    cv0_col, cv1_col = [col for col in pair_sdaf.columns() if col.startswith('concise_votes_')]
    text_differ_sdaf: SqlDaf = pair_sdaf.select_where_sql(f'"{cv0_col}" IS NOT "{cv1_col}"')
    num_text_differ, secs_f = timed(lambda: len(text_differ_sdaf))
    report(8, 'Create disagreed_ballots from nonflagged nonvariants', [num_text_differ, num_text_differ], [secs_f, ''])

    # Step 9: of those, the ones that still differ once normalized.
    disagreed_sdaf: SqlDaf = text_differ_sdaf.select_where(
            lambda row: concise_votes_differ(row[cv0_col], row[cv1_col]), cols=[cv0_col, cv1_col])
    num_disagreed, secs_f = timed(lambda: len(disagreed_sdaf))
    report(9, 'Double check disagreed_ballots using normalized concise_votes', [num_disagreed, num_disagreed], [secs_f, ''])

    # Step 10: the disagreed ballots are variants after all.
    secs_la = []
    for i in range(2):
        secs_la.append(timed(lambda: add_new_rows(
                variants_losdaf[i], nonvariants_losdaf[i].select_records_daf(disagreed_sdaf, silent_error=True)))[1])
    report(10, 'Add disagreed ballots to variants', [len(sdaf) for sdaf in variants_losdaf], secs_la)

    # Step 11: and are no longer nonvariants.
    secs_la = []
    for i in range(2):
        secs_la.append(timed(lambda: nonvariants_losdaf[i].delete_records(variants_losdaf[i], inplace=True, silent_error=True))[1])
    report(11, 'Remove variant ballot records from nonvariant ballots table.', [len(sdaf) for sdaf in nonvariants_losdaf], secs_la)

    return {'variants_losdaf': variants_losdaf, 'nonvariants_losdaf': nonvariants_losdaf, 'residue_losdaf': residue_losdaf}


#===================================
# Steps 12 to 17

def locate_contest_variants_sdaf(
        variants_losdaf:    T_losdaf,           # the audit and CVR variants, from step 11.
        modi_sdaf:          T_sdaf | None,      # the modified CVR votes, if any.
        votes_sdaf:         SqlDaf,             # the audit votes, trimmed by step 3, for the metadata.
        pmvs_sdaf:          SqlDaf,             # the concise_pmv of each ballot.
        wins_sdaf:          SqlDaf,             # the concise_wins of each ballot.
        report_daf:         Daf,                # the step report, added to.
        ) -> Dict[str, SqlDaf]:                 # the result tables, by the names of sql_variants.py.
    """
    Steps 12 to 17: split the variant ballots into contests, and find the contests that agree.

    A contest agrees when its audit and CVR votes are the same text and neither has a flag.
    If the modified CVR has the contest, it must also have the same votes.
    The contests that agree are put back together, one row per ballot, as partial_nonvariants.
    The rest are combined_contest_variants, one row per contest, with the votes of each side.
    Sheets blank on both sides go to blank_sheets.
    """
    def report(step: int, descr: str, num_rows: int, secs_f: float) -> None:
        report_daf.append({'Step': step, 'Description': descr, 'Audit recs': num_rows, 'CVR Recs': '',
                           'Audit(s)': secs_f, 'CVR(s)': ''})

    # The contest tables are TEMP tables with these names.
    # A join with tag_other names the other's concise_votes with the other's table name as a suffix.
    names_ls: T_ls = [f'{WORK_PREFIX}audit_contests', f'{WORK_PREFIX}cvr_contests', f'{WORK_PREFIX}modi_contests']
    a_cv, c_cv, m_cv = [f'concise_votes_{name}' for name in names_ls]
    i: int
    secs_f: float

    # Step 12: one row per contest, for each side and for the modified CVR.
    contests_losdaf: T_losdaf = []
    secs_la: List[Any] = []
    for i in range(2):
        contests_sdaf, secs_f = timed(lambda: variants_losdaf[i].apply(split_contests).copy(name=names_ls[i]))
        contests_losdaf.append(contests_sdaf.set_keyfield('contest_key'))
        secs_la.append(secs_f)
    report_daf.append({'Step': 12, 'Description': 'Split ballot variants into contest-level tables',
                       'Audit recs': len(contests_losdaf[0]), 'CVR Recs': len(contests_losdaf[1]),
                       'Audit(s)': secs_la[0], 'CVR(s)': secs_la[1]})
    modi_contests_sdaf: SqlDaf | None = None
    if modi_sdaf is not None:
        modi_contests_sdaf = modi_sdaf.apply(split_contests).copy(name=names_ls[2]).set_keyfield('contest_key')

    # Step 13: pair the contests of the two sides, and keep those that agree.
    start_f: float = time.perf_counter()
    pair_sdaf: SqlDaf = contests_losdaf[1].join(contests_losdaf[0], shared_fields=['ballot_id', 'cidx'], tag_other=True)
    pair_cols_ls: T_ls = [a_cv, c_cv]
    if modi_contests_sdaf is not None:
        pair_sdaf = pair_sdaf.join(modi_contests_sdaf, how='left', shared_fields=['ballot_id', 'cidx'], tag_other=True)
        pair_cols_ls += [m_cv]

    def contest_agrees(row: Any) -> bool:
        modi_cv: Any = row[m_cv] if modi_contests_sdaf is not None else ''
        return (row[a_cv] == row[c_cv] and not has_contest_flag(row[a_cv]) and not has_contest_flag(row[c_cv])
                and (not modi_cv or modi_cv == row[c_cv]))

    agreed_sdaf: SqlDaf = pair_sdaf.select_where(contest_agrees, cols=pair_cols_ls)
    nonvariant_contests_sdaf: SqlDaf = contests_losdaf[1].select_records_daf(agreed_sdaf).copy()
    report(13, "Compare '_by_contest' tables to produce 'nonvariant_contests' table.",
           len(nonvariant_contests_sdaf), time.perf_counter() - start_f)

    # Step 14: the contests that do not agree, with the pmvs and wins of the ballot.
    def combined_row(row_da: T_da) -> T_da:
        return {'ballot_id':        row_da['ballot_id'],
                'cidx':             row_da['cidx'],
                'concise_votes1':   row_da[a_cv],
                'concise_votes2':   row_da[c_cv],
                'concise_votes3':   (row_da[m_cv] or '{}') if modi_contests_sdaf is not None else '{}',
                }

    start_f = time.perf_counter()
    disagreed_sdaf: SqlDaf = pair_sdaf.select_where(lambda row: not contest_agrees(row), cols=pair_cols_ls)
    combined_sdaf: SqlDaf = (disagreed_sdaf.apply(combined_row).set_keyfield('ballot_id')
                             .join(pmvs_sdaf.select_cols(['ballot_id', 'concise_pmv']))
                             .join(wins_sdaf, how='left')
                             .copy())
    report(14, "Join '_by_contest' tables to produce 'combined_contest_variants'.",
           len(combined_sdaf), time.perf_counter() - start_f)

    # Step 15: the contests that agree, put back together, one row per ballot.
    # The contest dicts of each ballot are merged in row order, as json_group_object() merges them.
    start_f = time.perf_counter()
    # A registered aggregate merges them inside SQLite, one ballot at a time.
    partial_sdaf: SqlDaf = nonvariant_contests_sdaf.groupby_aggregate('ballot_id', update_da, 'concise_votes')
    logs.stsloc(f"Step 15: {len(partial_sdaf)} ballots recombined from {len(nonvariant_contests_sdaf)} contests. "
                f"First: {partial_sdaf.iloc(0) if len(partial_sdaf) else ''}", 3)

    # num_contests, counted from each merged dict.
    partial_sdaf.insert_col('num_contests', default=0)
    partial_sdaf.apply_in_place(lambda row_da: {'num_contests': num_contests_of(row_da['concise_votes'])})
    report(15, "Recombine 'nonvariant_contests' into 'partial_nonvariants'", len(partial_sdaf), time.perf_counter() - start_f)

    # Step 16: an empty row for each variant ballot that has no contest that agrees.
    # They are added in ballot_id order, as the index of sql_variants.py gives them.
    start_f = time.perf_counter()

    # The variant ballots, one row each, keyed and sorted by ballot_id.
    # The reduction keeps no value, so only ballot_id is read.
    variant_ids_sdaf: SqlDaf = combined_sdaf.groupby_reduce('ballot_id', keep_no_value, by='row', reduce_cols=['ballot_id'])
    logs.stsloc(f"Step 16: {len(variant_ids_sdaf)} variant ballots in {len(combined_sdaf)} contest rows", 3)

    # The variant ballots with no row in partial_nonvariants.
    missing_sdaf: SqlDaf = variant_ids_sdaf.select_records_daf(partial_sdaf, inverse=True, silent_error=True)
    logs.stsloc(f"Step 16: {len(missing_sdaf)} variant ballots have no contest that agrees", 3)

    # An empty shell for each of them.
    shells_sdaf: SqlDaf = missing_sdaf.apply(lambda row_da: {'ballot_id': row_da['ballot_id'], 'concise_votes': '{}', 'num_contests': 0})
    partial_sdaf.concat(shells_sdaf)
    report(16, 'Ensure_partial_nonvariants_have_all_variant_ballots (even empty shells)', len(partial_sdaf), time.perf_counter() - start_f)

    # Step 17: the sheets blank on both sides leave partial_nonvariants for blank_sheets.
    start_f = time.perf_counter()
    metadata_sdaf: SqlDaf = votes_sdaf.select_cols(exclude_cols=['concise_votes', 'num_contests'])
    blank_sdaf: SqlDaf = (partial_sdaf.select_records_daf(variant_ids_sdaf, inverse=True, silent_error=True)
                          .select_where(lambda row: is_blank_sheet(row['concise_votes']), cols=['concise_votes'])
                          .copy())
    blank_sheets_sdaf: SqlDaf = blank_sdaf.join(metadata_sdaf).copy()
    partial_sdaf.delete_records(blank_sdaf, inplace=True)
    partial_wmetadata_sdaf: SqlDaf = partial_sdaf.join(metadata_sdaf).copy()
    report(17, 'Move sheets blank in audit and CVR to blank_sheets (ballot variants)', len(blank_sheets_sdaf), time.perf_counter() - start_f)

    return {
        'audit_variants_by_contest':        contests_losdaf[0],
        'cvr_variants_by_contest':          contests_losdaf[1],
        'nonvariant_contests':              nonvariant_contests_sdaf,
        'combined_contest_variants':        combined_sdaf,
        'partial_nonvariants':              partial_sdaf,
        'partial_nonvariants_wmetadata':    partial_wmetadata_sdaf,
        'blank_sheets':                     blank_sheets_sdaf,
        }


#===================================
# Step 18

def check_variants_per_ballot(
        num_rows:           int,            # the rows of joined_contests.
        num_ballots:        int,            # the rows of votes.
        max_per_ballot:     float,          # find_variants_max_variants_per_ballot.
        ) -> None:
    """
    The sanity gate of create_comparison_records(), before the rows are converted.

    A real ballot has about 20 contests, and even 10% of them flagged is 2 rows per ballot.
    More than max_per_ballot rows per ballot means the data is wrong upstream.
    """
    max_rows: float = num_ballots * max_per_ballot
    if num_rows <= max_rows:
        return
    logs.exception_report(
            f"{logs.prog_loc()} 'joined_contests' has {num_rows:,} rows for {num_ballots:,} "
            f"ballots ({num_rows / num_ballots:.1f}/ballot), more than the sanity gate of "
            f"{max_rows:,} ({num_ballots:,} * {max_per_ballot}). "
            "This almost certainly means upstream data is wrong, not that real variants are "
            "this common. Investigate before proceeding.")
    breakpoint()    # perm
    raise ValueError(f"find_variants: 'joined_contests' has {num_rows:,} rows for {num_ballots:,} ballots, "
                     f"more than find_variants_max_variants_per_ballot = {max_per_ballot} per ballot.")


def cell_text(val: Any) -> Any:
    """
    A dict, list or tuple cell as PYON text, for a table cell. Other values are kept.

    SQLite cannot hold a dict.
    PYON is the text that str() gives, which DB.save_data() writes for a dict cell.
    So contest_variants.csv has the same text as the file of the 'sql' engine.
    True, False and None are kept as they are.
    """
    if isinstance(val, (dict, list, tuple)):
        text_a: Any = str(val)
    else:
        text_a = val
    return text_a


def converted_batch_daf(
        batch_loda:     T_loda,                         # the joined rows of one batch, with names decoded.
        record_func:    Callable[[T_da], T_da],         # the function that converts a row.
        ) -> Daf:                                       # the converted rows, with each dict cell as text.
    """
    Convert a batch of joined rows to contest_variants rows, with `Daf.apply()`.

    The result has no keyfield, since a ballot has a row for each of its variant contests.
    """
    batch_daf: Daf = Daf.from_lod(batch_loda)
    converted_daf: Daf = batch_daf.apply(func=record_func)

    # Each dict cell written as text, so the block can go into a table.
    row_la: T_la
    for row_la in converted_daf.lol:
        idx: int
        val: Any
        for idx, val in enumerate(row_la):
            if isinstance(val, (dict, list, tuple)):
                row_la[idx] = cell_text(val)
    return converted_daf


def contest_variants_sdaf_of(
        joined_sdaf:    SqlDaf,                         # the variant contests with the metadata of the ballot.
        record_func:    Callable[[T_da], T_da],         # create_contest_variant_record() of sql_variants.py.
        name:           str,                            # the name of the result table.
        ) -> SqlDaf | None:                             # the contest_variants rows, or None if there are none.
    """
    Convert each variant contest to a contest_variants row, as create_comparison_records() does.

    The rows are read APPLY_BATCH_ROWS at a time, with their names decoded.
    Each batch is converted in Python, and the block is added to the result table.
    So memory holds one batch, not the whole result.
    The first block makes the table, with the columns the record function gives.
    """
    # A table of an earlier run goes first, before the rows are read.
    drop_table(joined_sdaf, name)
    result_sdaf: SqlDaf | None = None
    num_block: int = 0
    rows_iter: Iterator[T_da] = joined_sdaf.iter_dict()
    while True:
        # One batch of joined rows.
        batch_loda: T_loda = [decoded_row(row_da) for row_da in itertools.islice(rows_iter, APPLY_BATCH_ROWS)]
        if not batch_loda:
            break

        # The batch converted and added to the table.
        block_daf: Daf = converted_batch_daf(batch_loda, record_func)
        if result_sdaf is None:
            result_sdaf = SqlDaf.from_daf(block_daf, name)
        else:
            result_sdaf.concat(block_daf)
        num_block += 1
        logs.stsloc(f"Step 18: block {num_block}, {len(block_daf)} rows. Table {name} has {len(result_sdaf)} rows.", 3)
    return result_sdaf


def create_comparison_records_sdaf(
        argsdict:           T_da,
        combined_sdaf:      SqlDaf,         # combined_contest_variants, from step 14.
        votes_sdaf:         SqlDaf,         # the audit votes, trimmed by step 3.
        dirname:            str,            # cmpcvr.
        ) -> Tuple[SqlDaf, int, float, int, float]:     # joined_contests, num_rows1, elapsed_secs1, num_rows2, elapsed_secs2
    """
    Step 18: make contest_variants.csv, as create_comparison_records() of sql_variants.py does.

    The variant contests are joined with the votes, for the metadata of each ballot.
    The sanity gate is checked, and the contest_variants.csv of the last run is kept in history.
    Each row is then converted by create_contest_variant_record() of sql_variants.py.
    """
    start_f: float = time.perf_counter()
    joined_sdaf: SqlDaf = combined_sdaf.join(votes_sdaf)
    num_rows1: int = len(joined_sdaf)
    elapsed_secs1: float = time.perf_counter() - start_f

    check_variants_per_ballot(num_rows1, len(votes_sdaf), argsdict.get('find_variants_max_variants_per_ballot', 2))

    start_f = time.perf_counter()

    # The file of the last run is kept, so two runs can be compared.
    if DB.file_exists(name='contest_variants.csv', dirname=dirname):
        DB.s3copy_file_dirnames(
            src_dirname     = dirname,
            src_name        = 'contest_variants.csv',
            dest_dirname    = dirname,
            dest_subdir     = f'history/{utils.history_tag()}',
            silent_error    = True,
            )

    contest_variants_sdaf: SqlDaf | None = contest_variants_sdaf_of(
            joined_sdaf, sql_variants.create_contest_variant_record, CONTEST_VARIANTS_TABLE)
    num_rows2: int = len(contest_variants_sdaf) if contest_variants_sdaf is not None else 0
    logs.stsloc(f"Applied create_contest_variant_record() to create {CONTEST_VARIANTS_TABLE}: {num_rows2} rows", 3)
    elapsed_secs2: float = time.perf_counter() - start_f

    # The save is not timed, as in create_comparison_records(), since it also uploads the file.
    # With no rows, the empty Daf is saved, as before.
    if contest_variants_sdaf is not None:
        export_csv(contest_variants_sdaf, dirname, 'contest_variants.csv')
    else:
        DB.save_data(Daf(), dirname=dirname, name='contest_variants.csv', rtype='daf', local_mirror=True)
    return joined_sdaf, num_rows1, elapsed_secs1, num_rows2, elapsed_secs2


#===================================
# Nonvariant stats

def precinct_sums_sdaf(
        rows_sdaf:          SqlDaf,         # the combined nonvariants.
        precinct_col:       str,            # the precinct column, as stored.
        direct_cols_ls:     T_ls,           # the columns of sumcols_ls that the table has.
        sumcols_ls:         T_ls,           # the columns to total, as reduce_nonvariants() makes the list.
        diagnose:           bool = False,   # if True, show each query and its first results.
        ) -> T_dodi:                        # the totals of each precinct, by precinct.
    """
    The totals of sumcols_ls for each precinct, as nonvariant_precinct_sums_dodi.

    The rows are grouped by precinct and reduced inside SQLite, by groupby_aggregate().
    The direct columns hold numbers, so SQL totals them, by groupsum_daf().
    The other columns are keys of concise_votes, totaled from the dict in each row.
    A key that no row of a precinct has is None, as SQL SUM() gives it.
    """
    # The two reductions, each with a row for each precinct.
    direct_daf: Daf = rows_sdaf.groupsum_daf(precinct_col, reduce_cols=direct_cols_ls)
    logs.stsloc(f"Precinct sums: {len(direct_daf)} precincts of direct sums", 3)
    indirect_sdaf: SqlDaf = rows_sdaf.groupby_aggregate(precinct_col, Daf.sum_da, 'concise_votes', diagnose=diagnose)
    logs.stsloc(f"Precinct sums: {len(indirect_sdaf)} precincts of concise_votes sums", 3)

    # The totals of each precinct, in the form of reduce_nonvariants().
    # The precincts come in sorted order, as the concise sums give them.
    direct_doda: Dict[str, T_da] = {row_da[precinct_col]: row_da for row_da in direct_daf.iter_dict()}
    sums_dodi: T_dodi = {}
    row_da: T_da
    for votes_row_da in indirect_sdaf.iter_dict():
        precinct: str = votes_row_da[precinct_col]
        votes_di: T_di = decoded_votes(votes_row_da['concise_votes'])
        row_da = direct_doda[precinct]
        sums_di: T_di = {}
        col: str
        for col in sumcols_ls:
            if col in direct_cols_ls:
                sums_di[col] = int(row_da[col])
            else:
                sums_di[col] = votes_di.get(col)
        sums_dodi[precinct] = sums_di
    return sums_dodi


def nonvariant_stats_sdaf(
        nonvariants_sdaf:       SqlDaf,         # the audit nonvariants, from step 11. num_contests is set in it.
        partial_wmetadata_sdaf: SqlDaf,         # the partly agreed ballots, from step 17.
        sumcols_ls:             T_ls,           # the columns to total, as reduce_nonvariants() makes the list.
        ) -> T_da:                              # the stats, by the names of reduce_nonvariants(), and combined_nonvariants.
    """
    The stats of the ballots that agree, as reduce_nonvariants() of cmpcvr.py makes them.

    num_contests is counted from concise_votes.
    The nonvariants and the partly agreed ballots are put together, as combined_nonvariants, and totaled.
    A column of sumcols_ls that the table does not have is a key of concise_votes.
    Its total reads the dict in each row.
    A key that no row has totals to 0, and to None in the sums of a precinct, as SQL SUM() gives them.
    """
    direct_cols_ls: T_ls = [col for col in sumcols_ls if col in nonvariants_sdaf.columns()]
    indirect_cols_set: set = {col for col in sumcols_ls if col not in direct_cols_ls}     # a set: sum_da() tests each key in it.

    def totals_da(rows_sdaf: SqlDaf, absent: Any) -> T_da:
        """ The totals of sumcols_ls in these rows. """
        direct_da: T_da = rows_sdaf.sum(direct_cols_ls) if direct_cols_ls else {}
        indirect_da: T_da = rows_sdaf.reduce(Daf.sum_da, by='sparse_row', indirect_col='concise_votes', cols=indirect_cols_set)
        return {col: int(direct_da[col]) if col in direct_da else indirect_da.get(col, absent) for col in sumcols_ls}

    # num_contests is written in place, so the other cells are not rewritten.
    # A rewritten row would store each Python NULL as SQL NULL, where the text '' was.
    counted_sdaf: SqlDaf = nonvariants_sdaf.apply_in_place(lambda row_da: {'num_contests': num_contests_of(row_da['concise_votes'])})
    combined_sdaf: SqlDaf = counted_sdaf.copy()
    combined_sdaf.concat(partial_wmetadata_sdaf.select_cols(counted_sdaf.columns()))

    valuecounts_dodi: T_da = combined_sdaf.valuecounts_for_colnames_ls(stored_cols(combined_sdaf, VALUECOUNTS_COLS_LS))
    precinct_col, style_col = stored_cols(counted_sdaf, ['precinct', 'style'])
    precinct_sums_dodi: T_dodi = precinct_sums_sdaf(combined_sdaf, precinct_col, direct_cols_ls, sumcols_ls)
    return {
        'nonvariant_sums_di':                           totals_da(combined_sdaf, 0),
        'nonvariant_valuecounts_dodi':                  {sql_utils.sql_unesc_str(col): vc_di for col, vc_di in valuecounts_dodi.items()},
        'nonvariant_precinct_sums_dodi':                precinct_sums_dodi,
        'nonvariant_precinct_styles_valuecounts_dodi':  counted_sdaf.valuecounts_for_colname1_groupedby_colname2(style_col, precinct_col),
        'nonvariant_keys_ls':                           [str(key) for key in combined_sdaf.keys()],
        'nonvariant_sheets_contests_di':                {'sheets': len(counted_sdaf),
                                                         'contests': int(counted_sdaf.sum(['num_contests'])['num_contests'])},
        'combined_nonvariants':                         combined_sdaf,
        }


def nonvariant_sumcols_ls(votes_cols_ls: T_ls, use_concise: bool) -> T_ls:
    """ The columns reduce_nonvariants() totals: the direct ones, the contest columns, and the keys of concise_votes. """
    cx_ls: T_ls = [col for col in votes_cols_ls if re.search(r'^c\.?\d+', col)]
    indirect_cx_ls: T_ls = generic_cvr.gen_flat_votes_cols(contests_dod=None, include_gray=False, use_concise_cols=True) if use_concise else []
    return DIRECT_SUM_COLS_LS + cx_ls + indirect_cx_ls


#===================================
# Entry point

def find_variants_sdaf(
        argsdict:               T_da,
        dirname:                str,            # cmpcvr, the database the inputs were imported into.
        input_tables_ls:        T_ls,           # audit votes, CVR votes, and optionally the modified CVR votes.
        other_input_tables_ls:  T_ls,           # the pmvs table, then the wins table.
        variants_tables_ls:     T_ls,           # ['audit_variants', 'cvr_variants']
        nonvariants_tables_ls:  T_ls,           # ['audit_nonvariants', 'cvr_nonvariants']
        ) -> T_da:                              # the nonvariant stats, as reduce_nonvariants() returns them.
    """
    Find the variants, as find_variants_using_sql() of cmpcvr.py does after its import.

    It writes the same CSV files and the same report, and returns the same stats.
    See the module docstring for what it covers.
    """
    if len(input_tables_ls) not in (2, 3):
        raise ValueError(f"find_variants_engine 'sdaf' needs a CVR, but the inputs are {input_tables_ls}. Use 'sql'.")
    if variants_tables_ls != ['audit_variants', 'cvr_variants'] or nonvariants_tables_ls != ['audit_nonvariants', 'cvr_nonvariants']:
        raise ValueError(f"find_variants_engine 'sdaf' uses the table names of sql_variants.py, "
                         f"not {variants_tables_ls} and {nonvariants_tables_ls}.")

    use_concise: bool = argsdict['use_concise_votes']
    if not use_concise:
        raise ValueError("find_variants_engine 'sdaf' handles concise votes only. Set use_concise_votes, or use 'sql'.")

    report_daf: Daf = Daf(cols=REPORT_COLS_LS)
    prior_db_path: str = SqlDaf.db_path
    SqlDaf.db_path = sql_utils.get_db_path_from_database_name(dirname)

    try:
        side_losdaf: T_losdaf = [SqlDaf(name, keyfield=KEYFIELD) for name in input_tables_ls[:2]]
        modi_sdaf: SqlDaf | None = None
        if len(input_tables_ls) == 3 and SqlDaf.table_exists(input_tables_ls[2]):
            modi_sdaf = SqlDaf(input_tables_ls[2], keyfield=KEYFIELD)

        # The split CVR records of a BMD ballot become one record.
        if argsdict['combine_split_bmd_cvrs']:
            groups_dod: T_doda = DB.load_data(dirname='cache', name='split_bmd_combined.json', silent_error=True) or {}
            num_combined: int = combine_split_bmd_sdaf(side_losdaf[1], groups_dod)
            logs.stsloc(f"Combined the split CVR records of {num_combined} BMD ballots in '{input_tables_ls[1]}'.", 3)

        # Steps 1 to 11, and the residue files.
        ballots_da: T_da = locate_ballot_variants_sdaf(side_losdaf, modi_sdaf, variants_tables_ls, report_daf)
        residue_names_ls: T_ls = ['votes_residue', 'cvr_residue']
        tables_dsdaf: Dict[str, SqlDaf] = {}
        i: int
        for i in range(2):
            tables_dsdaf[residue_names_ls[i]] = ballots_da['residue_losdaf'][i]
            tables_dsdaf[variants_tables_ls[i]] = ballots_da['variants_losdaf'][i]
            tables_dsdaf[nonvariants_tables_ls[i]] = ballots_da['nonvariants_losdaf'][i]
            export_csv(ballots_da['residue_losdaf'][i], dirname, f'{residue_names_ls[i]}.csv')

        # Steps 12 to 17, and partially_agreed.csv.
        tables_dsdaf |= locate_contest_variants_sdaf(
                variants_losdaf = ballots_da['variants_losdaf'],
                modi_sdaf       = modi_sdaf,
                votes_sdaf      = side_losdaf[0],
                pmvs_sdaf       = SqlDaf(other_input_tables_ls[0], keyfield=KEYFIELD),
                wins_sdaf       = SqlDaf(other_input_tables_ls[1], keyfield=KEYFIELD),
                report_daf      = report_daf,
                )
        export_csv(tables_dsdaf['partial_nonvariants_wmetadata'], dirname, 'partially_agreed.csv')

        # Step 18, and contest_variants.csv.
        joined_sdaf, num_rows1, elapsed_secs1, num_rows2, elapsed_secs2 = create_comparison_records_sdaf(
                argsdict, tables_dsdaf['combined_contest_variants'], side_losdaf[0], dirname)
        tables_dsdaf['joined_contests'] = joined_sdaf
        step_descr: str = "Create 'contest_variants' compatible with cmpcvr and AdjudiTally app"
        report_daf.append({'Step': 18, 'Description': step_descr + ' SQL Join',  'Audit recs': num_rows1, 'CVR Recs': '',
                           'Audit(s)': elapsed_secs1, 'CVR(s)': ''})
        report_daf.append({'Step': 18, 'Description': step_descr + ' Daf Apply', 'Audit recs': num_rows2, 'CVR Recs': '',
                           'Audit(s)': elapsed_secs2, 'CVR(s)': ''})

        md_report: str = (f"# Find Variants Report:\n\nfind_variants_engine: sdaf\n\n"
                          f"{report_daf.to_md(smart_fmt=True, shorten_text=False)}")
        logs.stsloc(md_report, 3)
        reports.save_md_and_html(argsdict, md_report, rootname='Find_Variants_Report')

        # The blank sheets, as ballot variants.
        blank_sheets_daf: Daf = decoded_daf(tables_dsdaf['blank_sheets'].to_daf())
        if blank_sheets_daf:
            ballot_variants_daf: Daf = cmpcvr.blank_sheets_to_ballot_variants_daf(blank_sheets_daf)
            logs.stsloc(f"{len(ballot_variants_daf)} blank sheets saved as ballot variants", 3)
            DB.save_data(ballot_variants_daf, dirname=dirname, name='ballot_variants.csv', rtype='daf', local_mirror=True)

        # The nonvariant stats. audit_nonvariants gets num_contests, as in reduce_nonvariants().
        stats_da: T_da = nonvariant_stats_sdaf(
                nonvariants_sdaf        = ballots_da['nonvariants_losdaf'][0],
                partial_wmetadata_sdaf  = tables_dsdaf['partial_nonvariants_wmetadata'],
                sumcols_ls              = nonvariant_sumcols_ls([sql_utils.sql_unesc_str(col) for col in side_losdaf[0].columns()], use_concise),
                )
        tables_dsdaf['combined_nonvariants'] = stats_da.pop('combined_nonvariants')

        # The variants files, as save_sql_tables_to_s3() writes them.
        for name in variants_tables_ls:
            export_csv(tables_dsdaf[name], dirname, f'{name}.csv')

        # The result tables stay in the file, for diagnosis.
        name: str
        rows_sdaf: SqlDaf
        for name, rows_sdaf in tables_dsdaf.items():
            keep_table(rows_sdaf, name, exclude_cols_ls=['contest_key'] if 'contest_key' in rows_sdaf.columns() else None)
    finally:
        SqlDaf.close()
        SqlDaf.db_path = prior_db_path

    return stats_da
