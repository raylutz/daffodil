# test_sqldaf.py
# Tests for SqlDaf in sqldaf.py: making and opening tables, loading, reading, selecting, keys, joins and the blocker.

import os
import json
from daffodil.lib.daf_sql import sqlite3
import threading

from typing import Any

import pytest

from daffodil.daf import Daf, NULL, KeysDisabledError
from daffodil.keyedlist import KeyedList
from daffodil.lib.daf_types import T_ls, T_dtype_dict
from daffodil.sqldaf import SqlDaf, BLOCKED_LS, SAFE_INHERITED_SET, KEEP_SET, RID_COL, META_TABLE, quote_name, ReadOnlyTableError
import daffodil.sqldaf as sqldaf_module


@pytest.fixture(autouse=True)
def job_db(tmp_path, monkeypatch):
    """ Every test gets its own database file, as the class setting SqlDaf.db_path. """
    db_path: str = str(tmp_path / 'job.db')
    monkeypatch.setattr(SqlDaf, 'db_path', db_path)
    yield db_path
    SqlDaf.close(db_path)


def make_marks_daf() -> Daf:
    return Daf(
        lol=[['b1', 'c0', 1], ['b2', 'c1', NULL], ['b3', 'c0', 0]],
        cols=['ballot_id', 'contest', 'num_votes'],
        keyfield='ballot_id',
    )


def write_csv(path: str, text: str) -> str:
    with open(path, 'w', newline='', encoding='utf-8') as csv_f:
        csv_f.write(text)
    return path


def key_indexes(conn: sqlite3.Connection) -> T_ls:
    """ The indexes SqlDaf made on keyfields, without SQLite's own. """
    return [row_t[0] for row_t in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index' AND name LIKE 'daf_ix_%'")]


def temp_tables(conn: sqlite3.Connection) -> T_ls:
    return [row_t[0] for row_t in conn.execute("SELECT name FROM sqlite_temp_master WHERE type = 'table'")]


# the database file and the connection

def test_sqldaf_and_daf_sql_share_one_driver():
    """ One SQLite in the process: SqlDaf and daf_sql use the same module, the newer one when installed. """
    import daffodil.sqldaf as sqldaf_module
    import daffodil.lib.daf_sql as daf_sql_module
    assert sqldaf_module.sqlite3 is daf_sql_module.sqlite3 is sqlite3
    assert sqlite3.sqlite_version_info >= sqldaf_module.MIN_SQLITE_VERSION


def test_an_old_sqlite_raises_at_connect(tmp_path, monkeypatch):
    """ The version is checked when a connection opens, with how to get a newer SQLite. """
    monkeypatch.setattr(sqlite3, 'sqlite_version_info', (3, 37, 2))
    monkeypatch.setattr(sqlite3, 'sqlite_version', '3.37.2')
    with pytest.raises(RuntimeError, match=r'needs SQLite 3\.48\.0 or later.*has SQLite 3\.37\.2.*Every Python that uv installs'):
        SqlDaf('marks', ['a'], db_path=str(tmp_path / 'old.db'))


def test_no_db_path_raises(monkeypatch):
    monkeypatch.setattr(SqlDaf, 'db_path', '')
    with pytest.raises(ValueError, match='no database'):
        SqlDaf('marks', ['a'])
    with pytest.raises(ValueError):
        SqlDaf.table_exists('marks')


def test_db_path_in_the_constructor_overrides_the_class_setting(tmp_path, job_db):
    other_path: str = str(tmp_path / 'other.db')
    other_sdaf: SqlDaf = SqlDaf('marks', ['a'], lol=[[1]], db_path=other_path)
    assert other_sdaf.db_path == other_path
    assert SqlDaf.table_exists('marks', db_path=other_path)
    assert not os.path.exists(job_db)


def test_open_in_a_missing_file_raises_and_makes_no_file(job_db):
    with pytest.raises(FileNotFoundError):
        SqlDaf('marks')
    assert not os.path.exists(job_db)
    assert SqlDaf.table_exists('marks') is False


def test_open_a_missing_table_raises_keyerror():
    SqlDaf('marks', ['a'])
    with pytest.raises(KeyError):
        SqlDaf('nothing')


def test_lol_without_cols_raises():
    with pytest.raises(ValueError):
        SqlDaf('marks', lol=[[1]])


def test_engine_other_than_sqlite_raises(monkeypatch):
    monkeypatch.setattr(SqlDaf, 'engine', 'postgres')
    with pytest.raises(NotImplementedError):
        SqlDaf('marks', ['a'])


def test_every_sqldaf_on_a_file_shares_one_connection(tmp_path):
    marks_sdaf: SqlDaf = SqlDaf.from_daf(make_marks_daf(), name='marks')
    again_sdaf: SqlDaf = SqlDaf('marks')
    assert again_sdaf.conn is marks_sdaf.conn
    other_sdaf: SqlDaf = SqlDaf('marks', ['a'], db_path=str(tmp_path / 'other.db'))
    assert other_sdaf.conn is not marks_sdaf.conn
    SqlDaf.close(str(tmp_path / 'other.db'))
    sel_sdaf: SqlDaf = marks_sdaf.select_by_dict({'contest': 'c0'})
    assert sel_sdaf.conn is marks_sdaf.conn
    assert sel_sdaf.is_child and not marks_sdaf.is_child
    assert sel_sdaf.parent_sdaf is marks_sdaf


def test_reopen_restores_keyfield_and_dtypes():
    marks_daf: Daf = make_marks_daf()
    marks_daf.dtypes = {'ballot_id': str, 'contest': str, 'num_votes': int}
    SqlDaf.from_daf(marks_daf, name='marks')
    SqlDaf.close()
    again_sdaf: SqlDaf = SqlDaf('marks')
    assert again_sdaf.keyfield == 'ballot_id'
    assert again_sdaf.dtypes == {'ballot_id': str, 'contest': str, 'num_votes': int}
    assert SqlDaf('marks', keyfield='contest').keyfield == 'contest'


def test_metadata_table_holds_one_row_per_table():
    marks_sdaf: SqlDaf = SqlDaf('marks', ['ballot_id', 'n'], keyfield='ballot_id', dtypes={'n': int})
    meta_lot = marks_sdaf.conn.execute(f'SELECT name, keyfield, dtypes FROM {META_TABLE}').fetchall()
    assert meta_lot == [('marks', 'ballot_id', '{"n": "int"}')]
    marks_sdaf.set_keyfield('n')
    marks_sdaf.apply_dtypes(dtypes={'ballot_id': str}, cols='ballot_id')
    assert SqlDaf('marks').keyfield == 'n'
    assert SqlDaf('marks').dtypes == {'n': int, 'ballot_id': str}


def test_metadata_cannot_record_an_unknown_dtype():
    class Odd:
        pass
    with pytest.raises(NotImplementedError):
        SqlDaf('marks', ['a'], dtypes={'a': Odd})
    assert not SqlDaf.table_exists('marks')


def test_a_table_made_by_other_code_has_no_record(job_db):
    conn: sqlite3.Connection = sqlite3.connect(job_db)
    conn.execute('CREATE TABLE plain (a TEXT)')
    conn.commit()
    conn.close()
    plain_sdaf: SqlDaf = SqlDaf('plain', keyfield='a')
    assert plain_sdaf.keyfield == 'a'
    plain_sdaf.set_keyfield('')
    assert not SqlDaf.table_exists(META_TABLE)
    assert SqlDaf('plain').keyfield == ''


def test_close_closes_the_file_for_every_sqldaf_on_it():
    marks_sdaf: SqlDaf = SqlDaf.from_daf(make_marks_daf(), name='marks')
    sel_sdaf: SqlDaf = marks_sdaf.select_by_dict({'contest': 'c0'})
    SqlDaf.close()
    SqlDaf.close()
    with pytest.raises(sqlite3.ProgrammingError):
        len(sel_sdaf)
    with pytest.raises(sqlite3.ProgrammingError):
        len(marks_sdaf)
    again_sdaf: SqlDaf = SqlDaf('marks')
    assert len(again_sdaf) == 3 and again_sdaf.conn is not marks_sdaf.conn


def test_close_without_a_path_raises(monkeypatch):
    monkeypatch.setattr(SqlDaf, 'db_path', '')
    with pytest.raises(ValueError):
        SqlDaf.close()


def test_a_transaction_belongs_to_the_file():
    marks_sdaf: SqlDaf = SqlDaf.from_daf(make_marks_daf(), name='marks')
    names_sdaf: SqlDaf = SqlDaf('names', ['ballot_id', 'who'], lol=[['b1', 'Ann']])
    with pytest.raises(RuntimeError):
        with marks_sdaf.transaction():
            marks_sdaf.sql("UPDATE marks SET num_votes = 9")
            names_sdaf.sql("UPDATE names SET who = 'Bob'")
            raise RuntimeError('stop')
    assert names_sdaf.to_daf().lol == [['b1', 'Ann']]
    assert marks_sdaf.select_record('b1')['num_votes'] == 1


def test_temp_tables_go_when_the_connection_closes():
    marks_sdaf: SqlDaf = SqlDaf.from_daf(make_marks_daf(), name='marks')
    copy_sdaf: SqlDaf = marks_sdaf.copy(name='work')
    assert copy_sdaf.is_temp
    SqlDaf.close()
    assert not SqlDaf.table_exists('work')


def test_delete_removes_the_table_and_its_record():
    marks_sdaf: SqlDaf = SqlDaf.from_daf(make_marks_daf(), name='marks')
    sel_sdaf: SqlDaf = marks_sdaf.select_by_dict({'contest': 'c0'})
    with pytest.raises(ValueError):
        sel_sdaf.delete()
    marks_sdaf.delete()
    assert not SqlDaf.table_exists('marks')
    assert marks_sdaf.conn.execute(f'SELECT COUNT(*) FROM {META_TABLE}').fetchone()[0] == 0
    work_sdaf: SqlDaf = SqlDaf.from_daf(make_marks_daf(), name='marks2').copy(name='work')
    work_sdaf.delete()
    assert temp_tables(work_sdaf.conn) == []


def test_next_temp_name_counts_up_and_skips_names_in_use():
    marks_sdaf: SqlDaf = SqlDaf('marks', ['a'])
    marks_sdaf.sql('CREATE TEMP TABLE daf_tmp_marks_v1 (x)')
    assert marks_sdaf._next_temp_name('marks') == 'daf_tmp_marks_v2'
    assert marks_sdaf._next_temp_name('marks') == 'daf_tmp_marks_v3'


# making a table

def test_create_table_has_hidden_rid_and_untyped_columns():
    marks_sdaf: SqlDaf = SqlDaf('marks', ['ballot_id', 'n'], keyfield='ballot_id', dtypes={'n': int})
    info_lot = marks_sdaf.conn.execute('PRAGMA table_info(marks)').fetchall()
    assert [(info_t[1], info_t[2], info_t[5]) for info_t in info_lot] == [
        (RID_COL, 'INTEGER', 1), ('ballot_id', '', 0), ('n', '', 0)]
    assert marks_sdaf.columns() == ['ballot_id', 'n']
    assert marks_sdaf.dtypes == {'n': int}
    assert key_indexes(marks_sdaf.conn) == []


def test_create_table_rejects_an_existing_name():
    SqlDaf('marks', ['a'])
    with pytest.raises(ValueError):
        SqlDaf('marks', ['a'])


def test_create_table_rejects_a_column_named_rid():
    with pytest.raises(ValueError):
        SqlDaf('marks', [RID_COL])


def test_create_table_keyfield_must_be_a_column():
    with pytest.raises(KeyError):
        SqlDaf('marks', ['a'], keyfield='b')


# SqlDaf.from_csv

def test_from_csv_loads_rows_in_order_with_null(tmp_path):
    csv_path: str = write_csv(str(tmp_path / 'marks.csv'), 'ballot_id,contest,num_votes\nb1,c0,1\nb2,c1,\nb3,c0,0\n')
    marks_sdaf: SqlDaf = SqlDaf.from_csv(csv_path, keyfield='ballot_id')
    assert marks_sdaf.name == 'marks'
    marks_daf: Daf = marks_sdaf.to_daf()
    assert marks_daf.lol == [['b1', 'c0', '1'], ['b2', 'c1', NULL], ['b3', 'c0', '0']]
    assert marks_daf.lol[1][2] is NULL
    assert marks_sdaf.conn.execute('SELECT COUNT(*) FROM marks WHERE num_votes IS NULL').fetchone()[0] == 1


def test_from_csv_reads_text_only(tmp_path):
    csv_path: str = write_csv(str(tmp_path / 'm.csv'), 'k,n,f\na,1,1.5\nb,,\nc,abc,2\n')
    m_sdaf: SqlDaf = SqlDaf.from_csv(csv_path)
    assert m_sdaf.to_daf().lol == [['a', '1', '1.5'], ['b', NULL, NULL], ['c', 'abc', '2']]
    assert m_sdaf.sql('SELECT typeof(n) FROM m').to_list() == ['text', 'null', 'text']


def test_from_csv_text_compares_as_text_in_sql(tmp_path):
    csv_path: str = write_csv(str(tmp_path / 't.csv'), 'k,n\na,1\nb,10\nc,9\n')
    t_sdaf: SqlDaf = SqlDaf.from_csv(csv_path)
    assert t_sdaf.sql('SELECT k FROM t WHERE n < 5').lol == []
    assert t_sdaf.sql('SELECT MAX(n) FROM t').lol == [['9']]
    t_sdaf.apply_dtypes(dtypes={'k': str, 'n': int})
    assert t_sdaf.sql('SELECT k FROM t WHERE n < 5').to_list() == ['a']
    assert t_sdaf.sql('SELECT MAX(n) FROM t').lol == [[10]]


def test_from_csv_pads_short_rows_and_rejects_long_ones(tmp_path):
    short_path: str = write_csv(str(tmp_path / 'short.csv'), 'a,b,c\n1\n')
    long_path: str = write_csv(str(tmp_path / 'long.csv'), 'a,b\n1,2,3\n')
    assert SqlDaf.from_csv(short_path).to_daf().lol == [['1', NULL, NULL]]
    with pytest.raises(ValueError):
        SqlDaf.from_csv(long_path)
    assert not SqlDaf.table_exists('long')
    assert SqlDaf('long', ['a']).conn.in_transaction is False


def test_from_csv_duplicate_key_raises_at_first_key_use(tmp_path):
    csv_path: str = write_csv(str(tmp_path / 'dup.csv'), 'ballot_id,v\nb1,1\nb1,2\n')
    dup_sdaf: SqlDaf = SqlDaf.from_csv(csv_path, keyfield='ballot_id')
    assert len(dup_sdaf) == 2
    with pytest.raises(sqlite3.IntegrityError):
        dup_sdaf.ensure_key_index()
    assert key_indexes(dup_sdaf.conn) == []


def test_from_csv_keyfield_must_be_a_column(tmp_path):
    csv_path: str = write_csv(str(tmp_path / 'k.csv'), 'a,b\n1,2\n')
    with pytest.raises(KeyError):
        SqlDaf.from_csv(csv_path, keyfield='c')
    assert not SqlDaf.table_exists('k')


def test_from_csv_glob_loads_files_in_sorted_order_and_skips_empty_ones(tmp_path):
    write_csv(str(tmp_path / 'votes_2.csv'), 'b,v\nb3,3\n')
    write_csv(str(tmp_path / 'votes_0.csv'), 'b,v\nb1,1\nb2,2\n')
    write_csv(str(tmp_path / 'votes_1.csv'), '')
    write_csv(str(tmp_path / 'votes_3.csv'), 'b,v\n')
    votes_sdaf: SqlDaf = SqlDaf.from_csv(str(tmp_path / 'votes_*.csv'), name='votes', keyfield='b')
    assert votes_sdaf.to_daf().lol == [['b1', '1'], ['b2', '2'], ['b3', '3']]
    assert votes_sdaf.keyfield == 'b'


def test_from_csv_list_loads_files_in_list_order_and_names_table_after_first(tmp_path):
    a_path: str = write_csv(str(tmp_path / 'a.csv'), 'k\na1\n')
    b_path: str = write_csv(str(tmp_path / 'b.csv'), 'k\nb1\n')
    ba_sdaf: SqlDaf = SqlDaf.from_csv([b_path, tmp_path / 'a.csv'])
    assert ba_sdaf.name == 'b'
    assert ba_sdaf.to_list() == ['b1', 'a1']
    assert SqlDaf.from_csv(a_path).to_list() == ['a1']


def test_from_csv_rejects_a_file_whose_column_names_differ(tmp_path):
    write_csv(str(tmp_path / 'c_0.csv'), 'b,v\nb1,1\n')
    write_csv(str(tmp_path / 'c_1.csv'), 'v,b\n2,b2\n')
    with pytest.raises(ValueError, match='c_1.csv differ from those of .*c_0.csv'):
        SqlDaf.from_csv(str(tmp_path / 'c_*.csv'), name='c')
    assert not SqlDaf.table_exists('c')


def test_from_csv_places_values_by_column_name(tmp_path):
    """
    A file in another order loads by name, not by position.
    New names follow the first file's, in the order first seen, and a missing column is NULL.
    """
    write_csv(str(tmp_path / 'c_0.csv'), 'b,v\nb1,1\n')
    write_csv(str(tmp_path / 'c_1.csv'), 'v,b\n2,b2\n')
    write_csv(str(tmp_path / 'c_2.csv'), '')
    write_csv(str(tmp_path / 'c_3.csv'), 'b,w,v\nb3,x,3\n')
    write_csv(str(tmp_path / 'c_4.csv'), 'u,b\ny,b4\n')
    c_sdaf: SqlDaf = SqlDaf.from_csv(str(tmp_path / 'c_*.csv'), name='c', keyfield='b', dtypes={'v': int}, strict_header=False)
    assert c_sdaf.to_daf().lol == [['b1', 1, NULL, NULL], ['b2', 2, NULL, NULL], ['b3', 3, 'x', NULL], ['b4', NULL, NULL, 'y']]
    assert c_sdaf.columns() == ['b', 'v', 'w', 'u']


def test_from_csv_old_and_new_path_columns_load_into_one_table(tmp_path):
    """ Old chunks have file_paths, and new ones file_path and file_path2. Each keeps its own columns. """
    write_csv(str(tmp_path / 'old.csv'), 'ballot_id,file_paths,v\nb1,"[\'a\', \'b\']",5\n')
    write_csv(str(tmp_path / 'new.csv'), 'ballot_id,file_path,file_path2,v\nb2,c,d,6\n')
    bif_sdaf: SqlDaf = SqlDaf.from_csv([str(tmp_path / 'old.csv'), str(tmp_path / 'new.csv')], name='bif', strict_header=False)
    assert bif_sdaf.columns() == ['ballot_id', 'file_paths', 'v', 'file_path', 'file_path2']
    assert bif_sdaf.to_daf().lol == [['b1', "['a', 'b']", '5', NULL, NULL], ['b2', NULL, '6', 'c', 'd']]


def test_from_csv_a_long_row_raises_against_its_own_header(tmp_path):
    write_csv(str(tmp_path / 'c_0.csv'), 'b,v,w\nb1,1,2\n')
    write_csv(str(tmp_path / 'c_1.csv'), 'b,v\nb2,2,9\n')
    with pytest.raises(ValueError, match='c_1.csv has 3 values, more than the 2 columns'):
        SqlDaf.from_csv(str(tmp_path / 'c_*.csv'), name='c', strict_header=False)
    assert not SqlDaf.table_exists('c')


def test_from_csv_pattern_with_no_match_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        SqlDaf.from_csv(str(tmp_path / 'none_*.csv'))
    with pytest.raises(FileNotFoundError):
        SqlDaf.from_csv(str(tmp_path / 'none.csv'))


def test_from_csv_dtypes_convert_as_apply_dtypes_does(tmp_path):
    text: str = 'k,n,f,b\na,1,1.5,1\nb,,,\nc,abc,2,0\n'
    write_csv(str(tmp_path / 'm.csv'), text)
    write_csv(str(tmp_path / 't.csv'), text)
    dtypes: T_dtype_dict = {'k': str, 'n': int, 'f': float, 'b': bool, 'gone': int}
    m_sdaf: SqlDaf = SqlDaf.from_csv(str(tmp_path / 'm.csv'), dtypes=dtypes)
    t_sdaf: SqlDaf = SqlDaf.from_csv(str(tmp_path / 't.csv')).apply_dtypes(dtypes=dtypes)
    assert m_sdaf.to_daf().lol == t_sdaf.to_daf().lol == [['a', 1, 1.5, True], ['b', NULL, NULL, False], ['c', 'abc', 2.0, False]]
    assert m_sdaf.sql('SELECT typeof(n) FROM m').to_list() == t_sdaf.sql('SELECT typeof(n) FROM t').to_list() == ['integer', 'null', 'text']
    assert m_sdaf.dtypes == dtypes
    assert SqlDaf('m').dtypes == dtypes


def test_from_csv_dtype_that_cannot_be_stored_raises(tmp_path):
    csv_path: str = write_csv(str(tmp_path / 'd.csv'), 'k\n(1,)\n')
    with pytest.raises(NotImplementedError):
        SqlDaf.from_csv(csv_path, dtypes={'k': tuple})
    assert not SqlDaf.table_exists('d')


def test_from_csv_makes_repeated_names_unique(tmp_path):
    csv_path: str = write_csv(str(tmp_path / 'rep.csv'), 'a,a,\n1,2,3\n')
    assert SqlDaf.from_csv(csv_path).columns() == Daf(cols=['a', 'a', '']).columns()


def test_from_csv_name_with_quote_and_space(tmp_path):
    csv_path: str = write_csv(str(tmp_path / 'q.csv'), 'my "col",b\n1,2\n')
    q_sdaf: SqlDaf = SqlDaf.from_csv(csv_path, name='odd name')
    assert q_sdaf.columns() == ['my "col"', 'b']
    col_sql: str = quote_name('my "col"')
    assert col_sql == '"my ""col"""'
    assert q_sdaf.sql(f'SELECT {col_sql} FROM {quote_name(q_sdaf.name)}').lol == [['1']]


# SqlDaf.from_daf and opening

def test_from_daf_keeps_value_types_without_dtypes():
    mixed_daf: Daf = Daf(lol=[['a', 1, 1.5, True], ['b', '2', NULL, False]], cols=['k', 'n', 'f', 'b'], dtypes={'n': int})
    mixed_sdaf: SqlDaf = SqlDaf.from_daf(mixed_daf, name='mixed')
    assert mixed_sdaf.to_daf().lol == [['a', 1, 1.5, 1], ['b', '2', NULL, 0]]
    assert mixed_sdaf.dtypes == {'n': int}
    assert mixed_sdaf.to_daf().dtypes == {'n': int}


def test_from_daf_list_cell_is_stored_as_text():
    lists_sdaf: SqlDaf = SqlDaf.from_daf(Daf(lol=[['a', [1, 2]]], cols=['k', 'x']), name='lists')
    assert lists_sdaf.to_daf().lol == [['a', '[1, 2]']]


def test_from_daf_round_trips_with_keyfield():
    marks_sdaf: SqlDaf = SqlDaf.from_daf(make_marks_daf(), name='marks')
    assert marks_sdaf.keyfield == 'ballot_id'
    marks_daf: Daf = marks_sdaf.to_daf()
    assert marks_daf.lol == make_marks_daf().lol
    assert marks_daf.keyfield == 'ballot_id'
    assert marks_daf['b2', 'contest'].to_value() == 'c1'


def make_plain_table(db_path: str, rows: list, cols_sql: str = 'a TEXT, b INTEGER') -> None:
    """ A table made by other code, with no daf_rid and no metadata record. """
    conn: sqlite3.Connection = sqlite3.connect(db_path)
    conn.execute(f'CREATE TABLE plain ({cols_sql})')
    conn.executemany(f'INSERT INTO plain VALUES ({", ".join(["?"] * len(rows[0]))})', rows)
    conn.commit()
    conn.close()


def test_open_a_table_without_rid(job_db):
    make_plain_table(job_db, [('x', 1), ('y', None)])
    plain_sdaf: SqlDaf = SqlDaf('plain')
    assert plain_sdaf.has_rid is False
    assert plain_sdaf.to_daf().lol == [['x', 1], ['y', NULL]]


# SqlDaf: reading

def test_len_bool_columns_num_cols():
    marks_sdaf: SqlDaf = SqlDaf.from_daf(make_marks_daf(), name='marks')
    empty_sdaf: SqlDaf = SqlDaf('empty', ['a'])
    assert len(marks_sdaf) == 3
    assert bool(marks_sdaf) is True
    assert bool(empty_sdaf) is False
    assert marks_sdaf.columns() == ['ballot_id', 'contest', 'num_votes']
    assert marks_sdaf.num_cols() == 3


@pytest.mark.parametrize('label, write', [
    ('cell by key',         lambda sdaf: sdaf.__setitem__(('b1', 'style_num'), 9)),
    ('update_record_krow',  lambda sdaf: sdaf.update_record_krow('b1', {'style_num': 9})),
    ('column',              lambda sdaf: sdaf.__setitem__((slice(None), 'style_num'), 0)),
    ('insert_col',          lambda sdaf: sdaf.insert_col('w', [1, 2, 3, 4, 5])),
    ('append',              lambda sdaf: sdaf.append({'ballot_id': 'b9', 'precinct': 'P9', 'style_num': 1, 'is_bmd': 0})),
    ('delete_records',      lambda sdaf: sdaf.delete_records(['b2'], inplace=True)),
    ('rename_cols',         lambda sdaf: sdaf.rename_cols({'style_num': 'style'})),
    ('apply_in_place',      lambda sdaf: sdaf.apply_in_place(lambda row_da: {'style_num': 0})),
    ('apply_dtypes',        lambda sdaf: sdaf.apply_dtypes(dtypes={'precinct': int}, cols='precinct')),
    ('delete',              lambda sdaf: sdaf.delete()),
    ('through a selection', lambda sdaf: sdaf.select_by_dict({'precinct': 'P1'}).update_record_krow('b1', {'style_num': 7})),
])
def test_read_only_refuses_any_change_to_the_data(label, write):
    """ A change to a read-only table's data raises, and leaves the data, its record and the SqlDaf as they were. """
    SqlDaf.from_daf(make_bif_daf(), name='bif')
    SqlDaf.close()
    bif_sdaf: SqlDaf = SqlDaf('bif', read_only=True)
    with pytest.raises(ReadOnlyTableError):
        write(bif_sdaf)
    SqlDaf.close()
    again_sdaf: SqlDaf = SqlDaf('bif')
    assert again_sdaf.to_daf().lol == make_bif_daf().lol
    assert again_sdaf.columns() == make_bif_daf().columns()
    assert again_sdaf.dtypes == {}


def test_read_only_allows_indexes_the_keyfield_and_work_on_copies():
    """ An index or a keyfield leaves the data as it is. Selections, sorts, joins and copies are TEMP tables. """
    SqlDaf.from_daf(make_bif_daf(), name='bif')
    SqlDaf.close()
    bif_sdaf: SqlDaf = SqlDaf('bif', read_only=True)
    bif_sdaf.ensure_key_index()
    bif_sdaf.set_keyfield('ballot_id')
    assert len(bif_sdaf.groupby('precinct')) == 3
    assert bif_sdaf.select_by_dict({'precinct': 'P1'}).keys() == ['b1', 'b3', 'b5']
    assert bif_sdaf[:].sort_by_colname('style_num', reverse=True).col('ballot_id')[0] == 'b3'
    copy_sdaf: SqlDaf = bif_sdaf.copy()
    copy_sdaf.update_record_krow('b1', {'style_num': 9})
    assert copy_sdaf.select_record('b1')['style_num'] == 9
    assert bif_sdaf.select_record('b1')['style_num'] == 1
    other_sdaf: SqlDaf = SqlDaf('other', cols=['k', 'a'], lol=[['k1', 1]], keyfield='k')
    other_sdaf.update_record_krow('k1', {'a': 2})
    assert other_sdaf.col('a') == [2]


def test_read_only_holds_only_for_the_sqldaf_that_asked_for_it():
    """ A writer on the same table changes it while a reader is open. The reader reads the change. """
    SqlDaf.from_daf(make_bif_daf(), name='bif')
    SqlDaf.close()
    reader_sdaf: SqlDaf = SqlDaf('bif', read_only=True)
    writer_sdaf: SqlDaf = SqlDaf('bif')
    assert (reader_sdaf.read_only, writer_sdaf.read_only) == (True, False)
    writer_sdaf.update_record_krow('b1', {'style_num': 9})
    with pytest.raises(ReadOnlyTableError):
        reader_sdaf.update_record_krow('b1', {'style_num': 8})
    writer_sdaf.update_record_krow('b2', {'style_num': 7})
    assert reader_sdaf.select_record('b1')['style_num'] == 9
    assert reader_sdaf.select_record('b2')['style_num'] == 7


def test_read_only_refuses_a_statement_the_writer_prepared_first():
    """ SQLite checks a statement when it prepares it, and sqlite3 reuses prepared ones. The reader's same UPDATE is still refused. """
    SqlDaf.from_daf(make_bif_daf(), name='bif')
    SqlDaf.close()
    reader_sdaf: SqlDaf = SqlDaf('bif', read_only=True)
    writer_sdaf: SqlDaf = SqlDaf('bif')
    for value in (5, 6):
        writer_sdaf.update_record_krow('b1', {'style_num': value})
        with pytest.raises(ReadOnlyTableError):
            reader_sdaf.update_record_krow('b1', {'style_num': value + 10})
    assert writer_sdaf.select_record('b1')['style_num'] == 6


def test_read_only_selections_follow_their_parent():
    """ A selection of the reader is read-only. A selection of the writer is not. """
    SqlDaf.from_daf(make_bif_daf(), name='bif')
    SqlDaf.close()
    reader_sdaf: SqlDaf = SqlDaf('bif', read_only=True)
    writer_sdaf: SqlDaf = SqlDaf('bif')
    reader_sel: SqlDaf = reader_sdaf.select_by_dict({'precinct': 'P1'})
    writer_sel: SqlDaf = writer_sdaf.select_by_dict({'precinct': 'P1'})
    assert (reader_sel.read_only, writer_sel.read_only) == (True, False)
    with pytest.raises(ReadOnlyTableError):
        reader_sel.update_record_krow('b1', {'style_num': 3})
    writer_sel.update_record_krow('b1', {'style_num': 4})
    assert reader_sdaf.select_record('b1')['style_num'] == 4


def test_read_only_sql_raises_read_only_table_error():
    """ sql() raises ReadOnlyTableError for a refused change, as every other method does. """
    SqlDaf.from_daf(make_bif_daf(), name='bif')
    SqlDaf.close()
    reader_sdaf: SqlDaf = SqlDaf('bif', read_only=True)
    writer_sdaf: SqlDaf = SqlDaf('bif')
    with pytest.raises(ReadOnlyTableError):
        reader_sdaf.sql("UPDATE bif SET style_num = 0")
    writer_sdaf.sql("UPDATE bif SET style_num = 0")
    assert set(reader_sdaf.col('style_num')) == {0}


def test_read_only_writer_and_reader_compare_on_one_connection():
    """ The compare pattern runs with a reader and a writer on one connection. """
    SqlDaf.from_daf(make_bif_daf(), name='bif')
    SqlDaf.close()
    reader_sdaf: SqlDaf = SqlDaf('bif', read_only=True)
    writer_sdaf: SqlDaf = SqlDaf('bif')
    writer_sdaf.delete_records(['b2'], inplace=True)
    assert reader_sdaf.select_records_daf(writer_sdaf, inverse=True, silent_error=True).keys() == []
    assert len(writer_sdaf) == len(make_bif_daf()) - 1


def make_cells_daf() -> Daf:
    return Daf(cols=['id', 'cv', 'ls'], lol=[['a1', {'x': 1, 'n': None}, [1, 'two']]], keyfield='id',
               dtypes={'id': str, 'cv': dict, 'ls': list})


def test_dict_and_list_cells_are_stored_as_their_csv_text():
    """ Every write stores a dict or list as the text to_csv_file() writes. to_daf().apply_dtypes() decodes it. """
    cells_daf: Daf = make_cells_daf()
    cells_sdaf: SqlDaf = SqlDaf.from_daf(cells_daf, name='cells')
    assert cells_sdaf.select_record('a1') == {'id': 'a1', 'cv': "{'x': 1, 'n': None}", 'ls': "[1, 'two']"}
    assert cells_daf.to_csv_buff().splitlines()[1] == 'a1,"{\'x\': 1, \'n\': None}","[1, \'two\']"'
    cells_sdaf.update_record_krow('a1', {'cv': {'x': 2}})
    cells_sdaf.append({'id': 'b2', 'cv': {'y': 1}, 'ls': []})
    cells_sdaf.extend([{'id': 'c3', 'cv': {'b': True}, 'ls': ['q']}])
    cells_sdaf['b2', 'cv'] = {'y': 9}
    cells_sdaf.concat(Daf(cols=['id', 'cv', 'ls'], lol=[['d4', {'w': 1}, [0]]]))
    assert cells_sdaf.to_daf().apply_dtypes().lol == [
            ['a1', {'x': 2}, [1, 'two']], ['b2', {'y': 9}, []], ['c3', {'b': True}, ['q']], ['d4', {'w': 1}, [0]]]
    assert cells_sdaf.reduce(Daf.sum_da, by='sparse_row', indirect_col='cv') == \
           cells_sdaf.to_daf().apply_dtypes().reduce(Daf.sum_da, by='sparse_row', indirect_col='cv')


def test_a_direct_write_stores_what_a_csv_round_trip_stores(tmp_path):
    """
    from_daf() writes the cells directly. It must store the same as writing the Daf as CSV, loading
    that with from_csv() and converting its number columns: the same types, and the same values.
    A dict or list becomes its text, and a missing value SQL NULL, either way.
    """
    dtypes: T_dtype_dict = {'id': str, 'n': int, 'x': float, 'ok': bool, 'p': str, 'cv': dict, 'ls': list}
    cells_daf: Daf = Daf(cols=list(dtypes), keyfield='id', dtypes=dtypes, lol=[
        ['b1', 3, 2.5, 1, 'P1', {'c0': 1, 'c1': ''}, [1, 'two']],
        ['b2', NULL, NULL, 0, NULL, {}, []],
        ['b3', -7, 0.1 + 0.2, 1, "it's", {'a': {'b': [1, '']}}, ['']],
    ])
    direct_sdaf: SqlDaf = SqlDaf.from_daf(cells_daf, name='direct')
    csv_path: str = str(tmp_path / 'cells.csv')
    cells_daf.to_csv_file(csv_path)
    via_sdaf: SqlDaf = SqlDaf.from_csv(csv_path, name='via', keyfield='id', dtypes={'n': int, 'x': float, 'ok': bool})

    def stored(sdaf: SqlDaf) -> list:
        return sdaf.conn.execute(f'SELECT {", ".join(f"typeof({quote_name(col)}), {quote_name(col)}" for col in dtypes)} '
                                 f'FROM {quote_name(sdaf.name)} ORDER BY daf_rid').fetchall()
    assert stored(direct_sdaf) == stored(via_sdaf)
    assert direct_sdaf.to_daf().lol == via_sdaf.to_daf().lol
    decoded_daf: Daf = direct_sdaf.to_daf()
    decoded_daf.dtypes = dict(dtypes)
    assert decoded_daf.apply_dtypes().lol == cells_daf.lol


def test_dict_and_list_dtypes_are_recorded_and_keep_the_text(tmp_path):
    """ On a SqlDaf a dict or list dtype means the cell's text. apply_dtypes() and from_csv() record it and convert nothing. """
    cells_daf: Daf = make_cells_daf()
    csv_path: str = str(tmp_path / 'cells.csv')
    cells_daf.to_csv_file(csv_path)
    loaded_sdaf: SqlDaf = SqlDaf.from_csv(csv_path, name='loaded', keyfield='id', dtypes={'cv': dict, 'ls': list})
    assert loaded_sdaf.select_record('a1') == {'id': 'a1', 'cv': "{'x': 1, 'n': None}", 'ls': "[1, 'two']"}
    assert SqlDaf('loaded').dtypes == {'cv': dict, 'ls': list}

    text_sdaf: SqlDaf = SqlDaf.from_csv(csv_path, name='text', keyfield='id')
    assert text_sdaf.apply_dtypes(dtypes={'cv': dict, 'ls': list}, cols=['cv', 'ls']) is text_sdaf
    assert text_sdaf.select_record('a1') == loaded_sdaf.select_record('a1')
    assert SqlDaf('text').dtypes == {'cv': dict, 'ls': list}
    decoded_daf: Daf = text_sdaf.to_daf()
    assert decoded_daf.dtypes == {'cv': dict, 'ls': list}
    assert decoded_daf.apply_dtypes(cols=['cv', 'ls']).lol == cells_daf.lol


def test_a_subclass_of_dict_or_a_tuple_cell_is_refused_by_sqlite():
    """ Only dict and list themselves are turned into text, so the test of each cell stays cheap. """
    import collections
    cells_sdaf: SqlDaf = SqlDaf.from_daf(make_cells_daf(), name='cells')
    with pytest.raises(sqlite3.Error):
        cells_sdaf.append({'id': 'e5', 'cv': collections.OrderedDict(a=1), 'ls': []})
    with pytest.raises(sqlite3.Error):
        cells_sdaf.append({'id': 'e6', 'cv': {}, 'ls': (1, 2)})


def test_isinstance_daf_and_repr():
    marks_sdaf: SqlDaf = SqlDaf.from_daf(make_marks_daf(), name='marks')
    assert isinstance(marks_sdaf, Daf)
    assert repr(marks_sdaf) == '\n' + str(marks_sdaf)
    assert repr(marks_sdaf) == repr(SqlDaf('marks').to_daf())


def test_str_and_format_give_the_text_of_a_daf():
    """ The same text as a Daf of the same rows, byte for byte, with only the rows shown read. """
    cols_ls: T_ls = ['ballot_id'] + [f'c{idx}' for idx in range(14)]
    rows_lol: list = [[f'b{row}'] + [('x' * (90 if (row + idx) % 7 == 0 else 3)) if idx % 3 == 0 else row * idx
                                     for idx in range(14)] for row in range(60)]
    wide_daf: Daf = Daf(cols=cols_ls, lol=rows_lol, keyfield='ballot_id', name='wide')
    wide_sdaf: SqlDaf = SqlDaf.from_daf(wide_daf, name='wide')
    assert str(wide_sdaf) == str(wide_daf)
    assert f'{wide_sdaf}' == str(wide_daf)
    assert f"{wide_sdaf['b7', 'c1']:.2f}" == '7.00'
    with pytest.raises(ValueError):
        f'{wide_sdaf:.2f}'

    marks_sdaf: SqlDaf = SqlDaf.from_daf(make_marks_daf(), name='marks')
    marks_daf: Daf = make_marks_daf()
    marks_daf.name = 'marks'
    assert str(marks_sdaf) == str(marks_daf)
    marks_sdaf.md_max_rows = 2
    assert '...' in str(marks_sdaf)


def test_name_is_read_only():
    marks_sdaf: SqlDaf = SqlDaf.from_daf(make_marks_daf(), name='marks')
    with pytest.raises(AttributeError):
        marks_sdaf.name = 'other'                       # type: ignore[misc]


def test_to_daf_is_not_tied_to_the_table():
    marks_sdaf: SqlDaf = SqlDaf.from_daf(make_marks_daf(), name='marks')
    marks_daf: Daf = marks_sdaf.to_daf()
    marks_daf['b1', 'num_votes'] = 99
    assert marks_sdaf.to_daf()['b1', 'num_votes'].to_value() == 1


def test_len_method_matches_len():
    marks_sdaf: SqlDaf = SqlDaf.from_daf(make_marks_daf(), name='marks')
    assert marks_sdaf.len() == len(marks_sdaf) == 3
    assert marks_sdaf.select_by_dict({'contest': 'c0'}).len() == 2


def test_to_csv_matches_mdaf(tmp_path):
    marks_daf: Daf = Daf(lol=[['b1', 1, 'a,b'], ['b2', NULL, 'x'], ['b3', 2.5, NULL]], cols=['ballot_id', 'n', 'text'])
    marks_sdaf: SqlDaf = SqlDaf.from_daf(marks_daf, name='marks')
    assert marks_sdaf.to_csv_buff() == marks_daf.to_csv_buff()
    assert marks_sdaf.to_csv_buff(line_terminator='\n', include_header=False) == marks_daf.to_csv_buff(line_terminator='\n', include_header=False)
    path: str = marks_sdaf.to_csv_file(tmp_path / 'marks.csv')
    assert path == str(tmp_path / 'marks.csv')
    with open(path, newline='', encoding='utf-8') as csv_f:
        assert csv_f.read() == marks_daf.to_csv_buff()
    sel_sdaf: SqlDaf = marks_sdaf.select_where_sql('"n" IS NOT NULL')
    assert sel_sdaf.to_csv_buff() == marks_daf.select_where(lambda row: row['n'] is not NULL).to_csv_buff()


def test_to_csv_file_streams_more_than_one_batch(tmp_path):
    big_daf: Daf = Daf(lol=[[f'b{i}', i] for i in range(25000)], cols=['ballot_id', 'n'])
    big_sdaf: SqlDaf = SqlDaf.from_daf(big_daf, name='big')
    with open(big_sdaf.to_csv_file(tmp_path / 'big.csv'), newline='', encoding='utf-8') as csv_f:
        assert csv_f.read() == big_daf.to_csv_buff()


def test_to_csv_decodes_a_stored_name_in_the_header():
    """ A table another program stored with encoded names writes them decoded. The rows are unchanged. """
    conn = sqlite3.connect(SqlDaf.db_path)
    conn.execute('CREATE TABLE votes (ballot_id TEXT, __67roup TEXT, caf__E9 INTEGER)')
    conn.execute("INSERT INTO votes VALUES ('b1', 'g1', 3)")
    conn.commit()
    conn.close()
    votes_sdaf: SqlDaf = SqlDaf('votes')
    assert votes_sdaf.columns() == ['ballot_id', '__67roup', 'caf__E9']
    assert votes_sdaf.to_csv_buff() == 'ballot_id,group,caf\u00e9\r\nb1,g1,3\r\n'


def test_a_table_with_no_columns_reads_as_empty():
    empty_sdaf: SqlDaf = SqlDaf('empty', [])
    assert len(empty_sdaf) == 0 and not empty_sdaf
    assert empty_sdaf.columns() == []
    assert empty_sdaf.to_daf().lol == []
    assert empty_sdaf.to_csv_buff() == Daf().to_csv_buff()


# SqlDaf.apply

APPLY_FUNCS = [
    ('one dict', lambda row: {'id': row['ballot_id'], 'n2': (row['n'] or 0) * 2}),
    ('the row itself or none', lambda row: row if row['n'] is not NULL else None),
    ('several rows', lambda row: [{'id': row['ballot_id'], 'part': i} for i in range(2)]),
    ('missing and extra names', lambda row: {'id': row['ballot_id'], 'a': 1} if row['ballot_id'] == 'b1' else {'id': row['ballot_id'], 'zz': 9}),
    ('nothing at all', lambda row: None),
    ]


@pytest.mark.parametrize('func', [f for _, f in APPLY_FUNCS], ids=[name for name, _ in APPLY_FUNCS])
def test_apply_matches_mdaf(func):
    marks_daf: Daf = Daf(lol=[['b1', 1], ['b2', NULL], ['b3', 3]], cols=['ballot_id', 'n'])
    marks_sdaf: SqlDaf = SqlDaf.from_daf(marks_daf, name='marks')
    result_sdaf: SqlDaf = marks_sdaf.apply(func)
    result_daf: Daf = marks_daf.apply(func)
    assert result_sdaf.columns() == result_daf.columns()
    assert result_sdaf.to_daf().lol == result_daf.lol
    assert result_sdaf.is_temp and result_sdaf.keyfield == ''


def test_apply_on_a_selection_with_kwargs_and_a_keyedlist():
    marks_sdaf: SqlDaf = SqlDaf.from_daf(make_marks_daf(), name='marks')
    def tag(row, label):
        return KeyedList({'ballot_id': 0, 'label': 1}, [row['ballot_id'], label])
    result_sdaf: SqlDaf = marks_sdaf.select_by_dict({'contest': 'c0'}).apply(tag, label='x')
    assert result_sdaf.to_daf().lol == [['b1', 'x'], ['b3', 'x']]


def test_apply_over_several_batches_and_save():
    big_sdaf: SqlDaf = SqlDaf.from_daf(Daf(lol=[[i] for i in range(25000)], cols=['n']), name='big')
    result_sdaf: SqlDaf = big_sdaf.apply(lambda row: {'n': row['n'], 'odd': row['n'] % 2})
    assert len(result_sdaf) == 25000
    assert result_sdaf.iloc(24999) == {'n': 24999, 'odd': 1}
    result_sdaf.save('odds')
    assert SqlDaf.table_exists('odds')


def test_apply_error_is_raised_as_itself_and_nothing_is_kept():
    marks_sdaf: SqlDaf = SqlDaf.from_daf(make_marks_daf(), name='marks')
    before_ls: T_ls = temp_tables(marks_sdaf.conn)
    with pytest.raises(ZeroDivisionError):
        marks_sdaf.apply(lambda row: {'x': 1} if row['ballot_id'] == 'b1' else 1 / 0)
    assert temp_tables(marks_sdaf.conn) == before_ls
    with pytest.raises(TypeError):
        marks_sdaf.apply(lambda row: ['not', 'a', 'dict'])


def test_apply_by_table_and_unsupported_arguments():
    marks_sdaf: SqlDaf = SqlDaf.from_daf(make_marks_daf(), name='marks')
    assert marks_sdaf.apply(len, by='table') == 3
    with pytest.raises(NotImplementedError):
        marks_sdaf.apply(lambda row: row, keylist=['b1'])
    with pytest.raises(NotImplementedError):
        marks_sdaf.apply(lambda row: row, by='col')


# SqlDaf.reduce

def make_reduce_daf() -> Daf:
    """ Ints, a float, text that is not a number, NULL, and dicts as PYON, JSON and NULL. """
    return Daf(lol=[['b1', 1, 2.5, "{'c0b': 1, 'c0v': 1}"],
                    ['b2', NULL, 'x', '{"c0b": 1, "c1b": 1}'],
                    ['b3', 3, 1, NULL]],
               cols=['ballot_id', 'v', 'f', 'concise_votes'])


REDUCE_ARGS = [
    ('sum of all', dict(func=Daf.sum_da)),
    ('sum of some', dict(func=Daf.sum_da, cols=['v', 'f'])),
    ('count values', dict(func=Daf.count_values_da, cols=['v'])),
    ('sparse row', dict(func=Daf.sum_da, by='sparse_row', indirect_col='concise_votes')),
    ('sparse row, some keys', dict(func=Daf.sum_da, by='sparse_row', indirect_col='concise_votes', cols=['c0b'])),
    ('some as a KeysView', dict(func=Daf.sum_da, cols=dict.fromkeys(['v', 'f']).keys())),
    ('sparse row, keys as a KeysView', dict(func=Daf.sum_da, by='sparse_row', indirect_col='concise_votes',
                                            cols=dict.fromkeys(['c0b', 'c1b']).keys())),
    ('sparse row, keys as a dict', dict(func=Daf.sum_da, by='sparse_row', indirect_col='concise_votes',
                                        cols=dict.fromkeys(['c0b']))),
    ]


@pytest.mark.parametrize('args', [a for _, a in REDUCE_ARGS], ids=[name for name, _ in REDUCE_ARGS])
def test_reduce_matches_mdaf(args):
    marks_daf: Daf = make_reduce_daf()
    marks_sdaf: SqlDaf = SqlDaf.from_daf(marks_daf, name='marks')
    assert marks_sdaf.reduce(**args) == marks_daf.reduce(**args)
    assert marks_sdaf.select_where_sql('"ballot_id" != ?', ['b1']).reduce(**args) == marks_daf[1:].reduce(**args)


def test_reduce_initial_da_and_silent_error_match_mdaf():
    marks_daf: Daf = make_reduce_daf()
    marks_sdaf: SqlDaf = SqlDaf.from_daf(marks_daf, name='marks')
    assert marks_sdaf.reduce(Daf.sum_da, initial_da={'v': 100}) == marks_daf.reduce(Daf.sum_da, initial_da={'v': 100}) == \
        {'ballot_id': '', 'v': 104, 'f': 3.5, 'concise_votes': ''}
    def count_inverse(row, acc, cols):
        acc['v'] += 1 / row['v']
        return acc
    assert marks_sdaf.reduce(count_inverse, silent_error=True) == marks_daf.reduce(count_inverse, silent_error=True)
    with pytest.raises(TypeError):
        marks_sdaf.reduce(count_inverse)


def test_reduce_over_several_batches():
    big_sdaf: SqlDaf = SqlDaf.from_daf(Daf(lol=[[i, f"{{'c0v': {i % 3}}}"] for i in range(25000)], cols=['n', 'cv']), name='big')
    assert big_sdaf.reduce(Daf.sum_da, cols=['n']) == {'n': 25000 * 24999 // 2, 'cv': ''}
    assert big_sdaf.reduce(Daf.sum_da, by='sparse_row', indirect_col='cv') == {'c0v': 24999}


def test_reduce_empty_by_table_and_unsupported():
    empty_sdaf: SqlDaf = SqlDaf('empty', cols=['a'])
    assert empty_sdaf.reduce(Daf.sum_da) == Daf(cols=['a']).reduce(Daf.sum_da) == {}
    assert empty_sdaf.reduce(Daf.sum_da, by='sparse_row', indirect_col='a') == {}
    marks_sdaf: SqlDaf = SqlDaf.from_daf(make_marks_daf(), name='marks')
    assert marks_sdaf.reduce(lambda sdaf, cols: len(sdaf), by='table') == 3
    with pytest.raises(ValueError):
        marks_sdaf.reduce(Daf.sum_da, by='sparse_row')
    with pytest.raises(NotImplementedError):
        marks_sdaf.reduce(Daf.sum_da, by='col')


# SqlDaf.sql

def test_sql_select_gives_a_daf_with_null():
    marks_sdaf: SqlDaf = SqlDaf.from_daf(make_marks_daf(), name='marks')
    result_daf: Daf = marks_sdaf.sql('SELECT contest, SUM(num_votes) AS v, COUNT(num_votes) AS c '
                                     'FROM marks GROUP BY contest ORDER BY contest')
    assert result_daf.columns() == ['contest', 'v', 'c']
    assert result_daf.lol == [['c0', 1, 2], ['c1', NULL, 0]]
    assert result_daf.lol[1][1] is NULL


def test_sql_params_turn_null_into_sql_null():
    marks_sdaf: SqlDaf = SqlDaf.from_daf(make_marks_daf(), name='marks')
    assert marks_sdaf.sql('SELECT ballot_id FROM marks WHERE contest = ?', ['c0']).to_list() == ['b1', 'b3']
    assert marks_sdaf.sql('SELECT ? IS NULL', [NULL]).lol == [[1]]


def test_sql_write_is_committed_and_gives_an_empty_daf():
    marks_sdaf: SqlDaf = SqlDaf.from_daf(make_marks_daf(), name='marks')
    result_daf: Daf = marks_sdaf.sql("UPDATE marks SET num_votes = 5 WHERE ballot_id = 'b2'")
    assert result_daf.lol == [] and result_daf.columns() == []
    assert marks_sdaf.to_daf()['b2', 'num_votes'].to_value() == 5
    assert marks_sdaf.conn.in_transaction is False


# the blocker

@pytest.mark.parametrize('call', [
    lambda sdaf: sdaf[[0, 1]],
    lambda sdaf: sdaf.irow(0),
    lambda sdaf: sdaf.select_irows([0]),
    lambda sdaf: sdaf.multi_groupby(['contest']),
    lambda sdaf: sdaf.transpose(),
    lambda sdaf: sdaf == sdaf,
    lambda sdaf: sdaf.retmode,
    lambda sdaf: sdaf.from_lod([]),
    ])
def test_unsupported_daf_methods_raise(call):
    marks_sdaf: SqlDaf = SqlDaf.from_daf(make_marks_daf(), name='marks')
    with pytest.raises(NotImplementedError):
        call(marks_sdaf)


def test_blocker_message_names_the_method():
    marks_sdaf: SqlDaf = SqlDaf.from_daf(make_marks_daf(), name='marks')
    with pytest.raises(NotImplementedError, match=r'^select_irows\(\) is not supported on a SqlDaf'):
        marks_sdaf.select_irows([0])


def test_blocker_covers_every_daf_method_not_defined():
    defined_ls = [name for name in vars(SqlDaf) if name not in BLOCKED_LS]
    for name in ('to_daf', 'sql', '__len__', '__bool__', 'num_cols', '__repr__'):
        assert name in defined_ls
    assert 'columns' not in vars(SqlDaf)
    own_set: set = {name for name in vars(SqlDaf) if name not in BLOCKED_LS}
    for name, attr in vars(Daf).items():                    # every Daf method SqlDaf does not define is blocked.
        if name in own_set or name in SAFE_INHERITED_SET or name in KEEP_SET:
            continue
        if callable(attr) or isinstance(attr, (property, staticmethod, classmethod)):
            assert name in BLOCKED_LS, name


def test_plain_daf_is_unchanged_by_the_blocker():
    marks_daf: Daf = make_marks_daf()
    assert marks_daf.select_irows([0]).lol == [['b1', 'c0', 1]]
    assert marks_daf.retmode == 'obj'


def test_constructor_takes_an_iterator_of_rows_with_null():
    rows_iter = iter([['b1', 1], ['b2', NULL]])
    marks_sdaf: SqlDaf = SqlDaf('marks', ['ballot_id', 'v'], lol=rows_iter, keyfield='ballot_id')
    assert marks_sdaf.to_daf().lol == [['b1', 1], ['b2', NULL]]
    assert marks_sdaf.sql('SELECT COUNT(*) FROM marks WHERE v IS NULL').to_value() == 1


def test_constructor_with_a_bad_row_makes_nothing():
    with pytest.raises(sqlite3.Error):
        SqlDaf('marks', ['ballot_id', 'v'], lol=[['b1', 1], ['b2', (2,)]])
    assert not SqlDaf.table_exists('marks')


# SqlDaf.set_keyfield and ensure_key_index

def test_ensure_key_index_makes_one_unique_index_and_is_used():
    marks_sdaf: SqlDaf = SqlDaf.from_daf(make_marks_daf(), name='marks')
    assert key_indexes(marks_sdaf.conn) == []
    marks_sdaf.ensure_key_index()
    marks_sdaf.ensure_key_index()
    assert key_indexes(marks_sdaf.conn) == ['daf_ix_marks_ballot_id']
    plan_ls = [r[-1] for r in marks_sdaf.conn.execute("EXPLAIN QUERY PLAN SELECT * FROM marks WHERE ballot_id = 'b2'")]
    assert plan_ls == ['SEARCH marks USING INDEX daf_ix_marks_ballot_id (ballot_id=?)']


def test_ensure_key_index_without_keyfield_raises():
    marks_sdaf: SqlDaf = SqlDaf.from_daf(make_marks_daf(), name='marks', keyfield='')
    marks_sdaf.set_keyfield('')
    with pytest.raises(ValueError):
        marks_sdaf.ensure_key_index()


def test_set_keyfield():
    marks_sdaf: SqlDaf = SqlDaf.from_daf(make_marks_daf(), name='marks')
    assert marks_sdaf.set_keyfield('contest') is marks_sdaf
    assert marks_sdaf.keyfield == 'contest'
    with pytest.raises(KeyError):
        marks_sdaf.set_keyfield('nothing')


# A Python NULL key raises when the key index is made, and on concat().

@pytest.mark.parametrize('stored_sql', ["NULL", "''"], ids=['SQL NULL', 'empty text'])
def test_null_key_raises_when_the_index_is_made_and_leaves_no_index(stored_sql):
    marks_sdaf: SqlDaf = SqlDaf.from_daf(make_marks_daf(), name='marks')
    marks_sdaf.sql(f"INSERT INTO marks (ballot_id, contest, num_votes) VALUES ({stored_sql}, 'c0', 1)")
    with pytest.raises(ValueError, match='Python NULL key'):
        marks_sdaf.ensure_key_index()
    assert key_indexes(marks_sdaf.conn) == []
    with pytest.raises(ValueError, match='Python NULL key'):
        marks_sdaf.select_records_daf(['b1'])          # every key method makes the index first.


def test_null_key_from_the_constructor_raises_at_first_key_use():
    marks_sdaf: SqlDaf = SqlDaf('marks', cols=['ballot_id', 'v'], lol=[['b1', 1], [NULL, 2]], keyfield='ballot_id')
    assert len(marks_sdaf) == 2                         # nothing is checked until a key is used.
    with pytest.raises(ValueError, match='Python NULL key'):
        marks_sdaf['b1']


@pytest.mark.parametrize('from_sql', [False, True], ids=['a Daf', 'a SqlDaf'])
def test_concat_of_a_null_key_raises_and_adds_nothing(from_sql):
    marks_sdaf: SqlDaf = SqlDaf.from_daf(make_marks_daf(), name='marks')
    more_daf: Daf = Daf(lol=[['b9', 'c0', 1], [NULL, 'c1', 2]], cols=['ballot_id', 'contest', 'num_votes'], keyfield='ballot_id')
    more = SqlDaf.from_daf(more_daf, name='more') if from_sql else more_daf
    with pytest.raises(ValueError, match='Python NULL key'):
        marks_sdaf.concat(more)
    assert marks_sdaf.keys() == ['b1', 'b2', 'b3']


def test_no_keyfield_allows_null_values():
    marks_sdaf: SqlDaf = SqlDaf('marks', cols=['ballot_id', 'v'], lol=[[NULL, 1]])
    marks_sdaf.concat(Daf(lol=[[NULL, 2]], cols=['ballot_id', 'v']))
    assert marks_sdaf.to_daf().lol == [[NULL, 1], [NULL, 2]]


# SqlDaf.apply_dtypes

def test_apply_dtypes_matches_daf():
    lol = [['1', '2.5', '', 'x'], ['', 'abc', 'True', 'y'], [' 7', '1e3', '0', 'z']]
    dtypes = {'i': int, 'f': float, 'b': bool, 's': str}
    mem_daf: Daf = Daf(lol=[list(row) for row in lol], cols=['i', 'f', 'b', 's'], dtypes=dict(dtypes))
    mem_daf.apply_dtypes()
    sql_sdaf: SqlDaf = SqlDaf.from_daf(Daf(lol=lol, cols=['i', 'f', 'b', 's']), name='t')
    sql_sdaf.apply_dtypes(dtypes=dtypes)
    assert sql_sdaf.to_daf().lol == mem_daf.lol
    assert sql_sdaf.to_daf().lol == [[1, 2.5, 0, 'x'], [NULL, 'abc', 1, 'y'], [7, 1000.0, 0, 'z']]
    assert sql_sdaf.dtypes == dtypes


def test_apply_dtypes_null_stays_sql_null_for_int():
    t_sdaf: SqlDaf = SqlDaf.from_daf(Daf(lol=[['1'], ['']], cols=['n']), name='t')
    t_sdaf.apply_dtypes(dtypes={'n': int})
    assert t_sdaf.sql('SELECT typeof(n) FROM t').to_list() == ['integer', 'null']


def test_apply_dtypes_cols_converts_only_those():
    t_sdaf: SqlDaf = SqlDaf.from_daf(Daf(lol=[['1', '2']], cols=['a', 'b']), name='t')
    t_sdaf.apply_dtypes(dtypes={'a': int, 'b': int}, cols=['a'])
    assert t_sdaf.to_daf().lol == [[1, '2']]


def test_apply_dtypes_missing_dtype_raises_unless_silent():
    t_sdaf: SqlDaf = SqlDaf.from_daf(Daf(lol=[['1', '2']], cols=['a', 'b']), name='t')
    with pytest.raises(ValueError):
        t_sdaf.apply_dtypes(dtypes={'a': int})
    t_sdaf.apply_dtypes(dtypes={'a': int}, silent_error=True)
    assert t_sdaf.to_daf().lol == [[1, '2']]


def test_apply_dtypes_unknown_col_raises_keyerror():
    t_sdaf: SqlDaf = SqlDaf.from_daf(Daf(lol=[['1']], cols=['a']), name='t')
    with pytest.raises(KeyError):
        t_sdaf.apply_dtypes(dtypes={'a': int}, cols=['z'])


def test_apply_dtypes_tuple_is_not_supported_yet():
    t_sdaf: SqlDaf = SqlDaf.from_daf(Daf(lol=[['(1,)']], cols=['a']), name='t')
    with pytest.raises(NotImplementedError):
        t_sdaf.apply_dtypes(dtypes={'a': tuple})


def test_apply_dtypes_uses_stored_dtypes():
    t_sdaf: SqlDaf = SqlDaf.from_daf(Daf(lol=[['1']], cols=['a'], dtypes={'a': int}), name='t')
    t_sdaf.apply_dtypes()
    assert t_sdaf.to_daf().lol == [[1]]


def test_apply_dtypes_str_leaves_numbers_by_default_and_converts_them_with_from_str_false(job_db):
    typed_daf: Daf = Daf(cols=['id', 'n', 'f', 'cv', 'e'], lol=[['a', 3, 1.5, {'c0': 1}, NULL], ['b', 10**15, 1e-7, [1, 2.0], 'x']])
    t_sdaf: SqlDaf = SqlDaf.from_daf(typed_daf, name='t')
    str_dtypes: T_dtype_dict = {col: str for col in typed_daf.columns()}
    t_sdaf.apply_dtypes(dtypes=str_dtypes)
    assert t_sdaf.to_daf().lol == [['a', 3, 1.5, "{'c0': 1}", ''], ['b', 10**15, 1e-7, '[1, 2.0]', 'x']]
    t_sdaf.apply_dtypes(dtypes=str_dtypes, from_str=False)
    assert t_sdaf.to_daf().lol == [['a', '3', '1.5', "{'c0': 1}", ''], ['b', '1000000000000000', '1e-07', '[1, 2.0]', 'x']]
    assert t_sdaf.sql('SELECT typeof(n), typeof(f), typeof(e) FROM t').lol == [['text', 'text', 'null'], ['text', 'text', 'text']]

    # The table then holds what from_csv() of the same rows holds.
    csv_path: str = os.path.join(os.path.dirname(job_db), 'typed.csv')
    typed_daf.to_csv_file(csv_path)
    assert SqlDaf.from_csv(csv_path).to_daf().lol == t_sdaf.to_daf().lol


# SqlDaf.transaction

def test_transaction_commits_together():
    marks_sdaf: SqlDaf = SqlDaf.from_daf(make_marks_daf(), name='marks')
    with marks_sdaf.transaction():
        marks_sdaf.sql("UPDATE marks SET num_votes = 7 WHERE ballot_id = 'b1'")
        assert marks_sdaf.conn.in_transaction is True
    assert marks_sdaf.conn.in_transaction is False
    assert marks_sdaf.to_daf()['b1', 'num_votes'].to_value() == 7


def test_transaction_rolls_back_on_error():
    marks_sdaf: SqlDaf = SqlDaf.from_daf(make_marks_daf(), name='marks')
    with pytest.raises(RuntimeError):
        with marks_sdaf.transaction():
            marks_sdaf.sql("UPDATE marks SET num_votes = 7 WHERE ballot_id = 'b1'")
            raise RuntimeError('stop')
    assert marks_sdaf.conn.in_transaction is False
    assert marks_sdaf.to_daf()['b1', 'num_votes'].to_value() == 1


def test_transaction_nested_undoes_only_the_inner_block():
    marks_sdaf: SqlDaf = SqlDaf.from_daf(make_marks_daf(), name='marks')
    with marks_sdaf.transaction():
        marks_sdaf.sql("UPDATE marks SET num_votes = 7 WHERE ballot_id = 'b1'")
        with pytest.raises(RuntimeError):
            with marks_sdaf.transaction():
                marks_sdaf.sql("UPDATE marks SET num_votes = 8 WHERE ballot_id = 'b3'")
                raise RuntimeError('stop')
    assert marks_sdaf.to_daf().to_lod() == [
        {'ballot_id': 'b1', 'contest': 'c0', 'num_votes': 7},
        {'ballot_id': 'b2', 'contest': 'c1', 'num_votes': NULL},
        {'ballot_id': 'b3', 'contest': 'c0', 'num_votes': 0}]


def test_a_failed_copy_inside_a_transaction_is_undone_alone():
    marks_sdaf: SqlDaf = SqlDaf.from_daf(make_marks_daf(), name='marks')
    with marks_sdaf.transaction():
        marks_sdaf.sql("UPDATE marks SET num_votes = 5 WHERE ballot_id = 'b1'")
        bad_sdaf: SqlDaf = marks_sdaf.select_where(lambda row: 1 / 0)
        with pytest.raises(ZeroDivisionError):
            bad_sdaf.copy(name='work')
        assert 'work' not in temp_tables(marks_sdaf.conn)
    assert marks_sdaf.select_record('b1')['num_votes'] == 5


# ---- step 2: selections ----

def make_bif_daf() -> Daf:
    return Daf(
        lol=[['b1', 'P1', 1, 0], ['b2', 'P2', 2, 1], ['b3', 'P1', 3, NULL], ['b4', 'P3', 1, 1], ['b5', 'P1', 2, 0]],
        cols=['ballot_id', 'precinct', 'style_num', 'is_bmd'],
        keyfield='ballot_id',
    )


def test_select_where_matches_mdaf():
    bif_daf: Daf = make_bif_daf()
    bif_sdaf: SqlDaf = SqlDaf.from_daf(bif_daf, name='bif')
    def where(row) -> bool:
        return row['precinct'] == 'P1' and row['is_bmd'] is not NULL
    assert bif_sdaf.select_where(where).to_daf().lol == bif_daf.select_where(where).lol
    assert bif_sdaf.select_where(where, cols=['precinct', 'is_bmd']).to_daf().lol == [['b1', 'P1', 1, 0], ['b5', 'P1', 2, 0]]


def test_select_where_gives_a_keyedlist_with_null():
    seen_la: list = []
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    def where(row) -> bool:
        seen_la.append((type(row).__name__, row['is_bmd']))
        return True
    bif_sdaf.select_where(where, cols=['is_bmd']).to_daf()
    assert seen_la[2] == ('KeyedList', NULL)
    assert seen_la[2][1] is NULL


def test_select_where_error_is_raised_as_itself():
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    sel_sdaf: SqlDaf = bif_sdaf.select_where(lambda row: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        sel_sdaf.to_daf()


def test_select_where_bad_col_and_indirect_col():
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    with pytest.raises(KeyError):
        bif_sdaf.select_where(lambda row: True, cols=['nothing'])
    with pytest.raises(NotImplementedError):
        bif_sdaf.select_where(lambda row: True, indirect_col='x')


@pytest.mark.parametrize('selector, inverse', [
    ({'precinct': 'P1'}, False),
    ({'precinct': 'P1', 'style_num': 2}, False),
    ([{'precinct': 'P2'}, {'style_num': 1}], False),
    ({'precinct': 'P1'}, True),
    ({'is_bmd': NULL}, False),
    ({'is_bmd': 0}, True),
    ({}, False),
    ([], False),
    ({'nothing': 1}, False),
    ([{'nothing': 1}, {'precinct': 'P3'}], False),
    ({'style_num': 1.0}, False),
    ({'style_num': '1'}, False),
    ])
def test_select_by_dict_matches_mdaf(selector, inverse):
    bif_daf: Daf = make_bif_daf()
    bif_sdaf: SqlDaf = SqlDaf.from_daf(bif_daf, name='bif')
    assert bif_sdaf.select_by_dict(selector, inverse=inverse).to_daf().lol == bif_daf.select_by_dict(selector, inverse=inverse).lol


def test_select_by_dict_expectmax_and_keyfield():
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    with pytest.raises(LookupError):
        bif_sdaf.select_by_dict({'precinct': 'P1'}, expectmax=2)
    assert len(bif_sdaf.select_by_dict({'precinct': 'P1'}, expectmax=3)) == 3
    assert bif_sdaf.select_by_dict({'precinct': 'P1'}, keyfield='style_num').keyfield == 'style_num'
    with pytest.raises(TypeError):
        bif_sdaf.select_by_dict('P1')                       # type: ignore[arg-type]


def test_split_where_matches_mdaf():
    bif_daf: Daf = make_bif_daf()
    bif_sdaf: SqlDaf = SqlDaf.from_daf(bif_daf, name='bif')
    def where(row) -> bool:
        return row['style_num'] > 1
    yes_sdaf, no_sdaf = bif_sdaf.split_where(where, cols=['style_num'])
    yes_daf, no_daf = bif_daf.split_where(where)
    assert not yes_sdaf.is_resolved and not no_sdaf.is_resolved
    assert yes_sdaf.to_daf().lol == yes_daf.lol
    assert no_sdaf.to_daf().lol == no_daf.lol


def test_split_where_bad_col_and_indirect_col():
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    with pytest.raises(KeyError):
        bif_sdaf.split_where(lambda row: True, cols=['nothing'])
    with pytest.raises(NotImplementedError):
        bif_sdaf.split_where(lambda row: True, indirect_col='x')


# groupby, groupby_cols and group_where

def groups_as_lol(groups) -> dict:
    """ A grouping of Daf or of SqlDaf, as a dict of each key and its rows. """
    return {key: (group.to_daf() if isinstance(group, SqlDaf) else group).lol for key, group in groups.items()}


@pytest.mark.parametrize('args', [
    {'colname': 'precinct'},
    {'colname': 'is_bmd'},
    {'colname': 'is_bmd', 'omit_nulls': True},
    {'colnames': ['style_num']},
    {'colnames': ['precinct', 'style_num']},
    {'colname': ['precinct', 'is_bmd']},
    ], ids=repr)
def test_groupby_matches_mdaf(args):
    bif_daf: Daf = make_bif_daf()
    bif_sdaf: SqlDaf = SqlDaf.from_daf(bif_daf, name='bif')
    groups_kl: KeyedList = bif_sdaf.groupby(**args)
    assert isinstance(groups_kl, KeyedList)
    assert all(not group.is_resolved for group in groups_kl.values())
    assert groups_as_lol(groups_kl) == groups_as_lol(bif_daf.groupby(**args))
    assert list(groups_kl.keys()) == list(bif_daf.groupby(**args).keys())


# SqlDaf.groupby_reduce

def _update_da(row_da: dict, reduction_da: dict, *, cols: Any = None, is_sparse: bool = False, diagnose: bool = False) -> dict:
    """ A reducer that merges each row's dict into the result. """
    for key, val in row_da.items():
        if cols and key not in cols:
            continue
        reduction_da[key] = val
    return reduction_da


GROUPBY_REDUCE_DAF = Daf(lol=[['b2', 1, "{'c0b': 1, 'c0v': 1}"], ['b1', 2, "{'c0b': 1}"], ['b2', 3, "{'c1b': 1}"],
                              ['', 4, "{'c0b': 1}"], ['b2', 5, "{'c2b': 1, 'c2v': 2}"], ['b1', 7, "{'c1v': 1}"]],
                         cols=['g', 'x', 'cv'])


@pytest.mark.parametrize('batch_size', [1, 2, 3, 10000])
@pytest.mark.parametrize('args', [
    dict(func=Daf.sum_da, reduce_cols=['x']),
    dict(func=Daf.sum_da),
    dict(func=Daf.sum_da, by='sparse_row', indirect_col='cv'),
    dict(func=_update_da, by='sparse_row', indirect_col='cv'),
    dict(func=Daf.sum_da, by='sparse_row', indirect_col='cv', reduce_cols=['c2v', 'c0b']),
    dict(func=Daf.sum_da, by='sparse_row', indirect_col='cv', reduce_cols=['c2v', 'c0b', 'c1v', 'c0v', 'zz']),
    ])
def test_groupby_reduce_matches_mdaf(args, batch_size):
    """ The rows are those of Daf.groupby_reduce(), in key order, with a dict cell as JSON text. """
    rows_sdaf: SqlDaf = SqlDaf.from_daf(GROUPBY_REDUCE_DAF, name='rows')
    result_sdaf: SqlDaf = rows_sdaf.groupby_reduce('g', batch_size=batch_size, **args)
    mdaf_lol: list = sorted(GROUPBY_REDUCE_DAF.groupby_reduce('g', **args).lol)
    if args.get('by') == 'sparse_row':
        mdaf_lol = [[key, json.dumps(votes_da, separators=(',', ':'))] for key, votes_da in mdaf_lol]
    assert result_sdaf.to_daf().lol == mdaf_lol
    assert result_sdaf.keyfield == 'g'


def test_groupby_reduce_merges_concise_votes_in_row_order():
    rows_sdaf: SqlDaf = SqlDaf.from_daf(GROUPBY_REDUCE_DAF, name='rows')
    result_sdaf: SqlDaf = rows_sdaf.groupby_reduce('g', _update_da, by='sparse_row', indirect_col='cv', batch_size=2)
    assert result_sdaf.columns() == ['g', 'cv']
    assert result_sdaf.to_daf().lol == [
            ['', '{"c0b":1}'],
            ['b1', '{"c0b":1,"c1v":1}'],
            ['b2', '{"c0b":1,"c0v":1,"c1b":1,"c2b":1,"c2v":2}']]


def _keep_none(row_da: dict, reduction_da: dict, **kwargs: Any) -> dict:
    """ A reducer that keeps nothing of the rows. """
    return reduction_da


def test_groupby_reduce_of_the_group_column_only():
    """ Nothing left to reduce: one row for each key, with the other columns empty, as in Daf.groupby_reduce(). """
    rows_sdaf: SqlDaf = SqlDaf.from_daf(GROUPBY_REDUCE_DAF, name='rows')
    result_sdaf: SqlDaf = rows_sdaf.groupby_reduce('g', _keep_none, reduce_cols=['g'])
    assert result_sdaf.columns() == ['g', 'x', 'cv']
    assert result_sdaf.to_daf().lol == sorted(GROUPBY_REDUCE_DAF.groupby_reduce('g', _keep_none, reduce_cols=['g']).lol)
    assert result_sdaf.to_daf().lol == [['', '', ''], ['b1', '', ''], ['b2', '', '']]


def test_groupby_reduce_by_row_passes_the_options_of_daf_reduce():
    """ Each group is reduced by Daf.reduce(), so its own options, such as silent_error, work. """
    def fail_on_three(row_da: Any, reduction_da: dict, cols: Any = None, **kwargs: Any) -> dict:
        if row_da['x'] == 3:
            raise ValueError('three')
        reduction_da['x'] += row_da['x']
        return reduction_da
    rows_sdaf: SqlDaf = SqlDaf.from_daf(GROUPBY_REDUCE_DAF, name='rows')
    result_sdaf: SqlDaf = rows_sdaf.groupby_reduce('g', fail_on_three, reduce_cols=['x'], silent_error=True)
    assert result_sdaf.to_daf().lol == [['', 4, ''], ['b1', 9, ''], ['b2', 6, '']]
    with pytest.raises(ValueError, match='three'):
        rows_sdaf.groupby_reduce('g', fail_on_three, reduce_cols=['x'])


def test_groupby_reduce_of_no_rows_and_bad_args():
    empty_sdaf: SqlDaf = SqlDaf('empty', cols=['g', 'cv'])
    assert empty_sdaf.groupby_reduce('g', Daf.sum_da, by='sparse_row', indirect_col='cv').to_daf().lol == []
    with pytest.raises(KeyError):
        empty_sdaf.groupby_reduce('nope', Daf.sum_da)
    with pytest.raises(NotImplementedError):
        empty_sdaf.groupby_reduce('g', Daf.sum_da, by='table')


@pytest.mark.parametrize('func', [Daf.sum_da, _update_da])
def test_groupby_aggregate_of_a_dict_column_matches_groupby_reduce(func):
    """ One column of dict cells: the same rows as groupby_reduce() by sparse row. """
    rows_sdaf: SqlDaf = SqlDaf.from_daf(GROUPBY_REDUCE_DAF, name='rows')
    expected_lol: list = rows_sdaf.groupby_reduce('g', func, by='sparse_row', indirect_col='cv').to_daf().lol
    result_sdaf: SqlDaf = rows_sdaf.groupby_aggregate('g', func, 'cv')
    assert result_sdaf.to_daf().lol == expected_lol
    assert result_sdaf.columns() == ['g', 'cv']
    assert result_sdaf.keyfield == 'g'


def test_groupby_aggregate_merges_in_row_order():
    """ A later row of a group overwrites an earlier one, so the rows come in the order of the table. """
    rows_sdaf: SqlDaf = SqlDaf('rows', cols=['g', 'cv'], lol=[['b', "{'k': 1}"], ['a', "{'k': 5}"], ['b', '{"k": 2}'], ['b', "{'k': 3}"]])
    assert rows_sdaf.groupby_aggregate('g', _update_da, 'cv').to_daf().lol == [['a', '{"k":5}'], ['b', '{"k":3}']]


def test_groupby_aggregate_of_several_columns_matches_mdaf():
    """ Several columns: the same totals as Daf.groupby_reduce(), and only the result columns. """
    rows_sdaf: SqlDaf = SqlDaf.from_daf(GROUPBY_REDUCE_DAF, name='rows')
    mdaf_lol: list = sorted(GROUPBY_REDUCE_DAF.groupby_reduce('g', Daf.sum_da, reduce_cols=['x']).lol)
    result_sdaf: SqlDaf = rows_sdaf.groupby_aggregate('g', Daf.sum_da, ['x', 'cv'], result_cols=['x'])
    assert result_sdaf.to_daf().lol == [[g, x] for g, x, _cv in mdaf_lol]


def test_groupby_aggregate_of_a_plain_column_gives_its_value():
    rows_sdaf: SqlDaf = SqlDaf.from_daf(GROUPBY_REDUCE_DAF, name='rows')
    assert rows_sdaf.groupby_aggregate('g', Daf.sum_da, 'x').to_daf().lol == [['', 4], ['b1', 9], ['b2', 9]]


def test_groupby_aggregate_by_two_columns_with_null_and_empty_text_together():
    """ SQL NULL and the text '' are one group, first, as in groupby_reduce(). """
    conn = sqlite3.connect(SqlDaf.db_path)
    conn.execute('CREATE TABLE other (p, s, n)')
    conn.executemany('INSERT INTO other VALUES (?, ?, ?)',
                     [('P1', 'x', 1), (None, 'x', 2), ('', 'x', 3), ('P1', 'y', 4), ('P1', 'x', 5)])
    conn.commit()
    conn.close()
    result_sdaf: SqlDaf = SqlDaf('other').groupby_aggregate(['p', 's'], Daf.sum_da, ['n', 's'], result_cols=['n'])
    assert result_sdaf.to_daf().lol == [['', 'x', 5], ['P1', 'x', 6], ['P1', 'y', 4]]
    assert result_sdaf.keyfield == ''


def test_groupby_aggregate_of_more_columns_than_one_call_takes():
    """ 150 columns go to the function, more than the 127 arguments of one call on SQLite 3.37. """
    cols_ls: list = [f'c{idx}' for idx in range(150)]
    rows_sdaf: SqlDaf = SqlDaf('wide', cols=['g'] + cols_ls, lol=[['a'] + [1] * 150, ['a'] + [2] * 150, ['b'] + [5] * 150])
    result_daf: Daf = rows_sdaf.groupby_aggregate('g', Daf.sum_da, cols_ls).to_daf()
    assert result_daf.lol == [['a'] + [3] * 150, ['b'] + [5] * 150]


def test_groupby_aggregate_writes_results_in_daffodil_form():
    """ A dict result is JSON text, True is 1, and None is Python NULL, stored as SQL NULL. """
    def set_flags(row_da: dict, reduction_da: dict, *, cols: Any = None) -> dict:
        reduction_da['flags'] = {'seen': True, 'none': None}
        reduction_da['ok'] = True
        reduction_da['gone'] = None
        return reduction_da
    rows_sdaf: SqlDaf = SqlDaf('rows', cols=['g', 'flags', 'ok', 'gone'], lol=[['a', '', '', '']])
    result_sdaf: SqlDaf = rows_sdaf.groupby_aggregate('g', set_flags, ['flags', 'ok', 'gone'])
    assert result_sdaf.to_daf().lol == [['a', '{"seen":1,"none":""}', 1, '']]
    stored_lt: list = result_sdaf.conn.execute(f'SELECT "gone" FROM temp."{result_sdaf.name}"').fetchall()
    assert stored_lt == [(None,)]


def test_groupby_aggregate_raises_the_function_error_as_itself():
    def broken(row_da: dict, reduction_da: dict, **kwargs: Any) -> dict:
        raise ValueError('a bad row')
    rows_sdaf: SqlDaf = SqlDaf.from_daf(GROUPBY_REDUCE_DAF, name='rows')
    with pytest.raises(ValueError, match='a bad row'):
        rows_sdaf.groupby_aggregate('g', broken, 'cv')
    with pytest.raises(ValueError, match='a bad row'):
        rows_sdaf.groupby_aggregate('g', broken, ['x', 'cv'])


def test_groupby_aggregate_of_no_rows_and_bad_names():
    empty_sdaf: SqlDaf = SqlDaf('empty', cols=['g', 'cv'])
    assert empty_sdaf.groupby_aggregate('g', Daf.sum_da, 'cv').to_daf().lol == []
    with pytest.raises(KeyError):
        empty_sdaf.groupby_aggregate('nope', Daf.sum_da, 'cv')
    with pytest.raises(KeyError):
        empty_sdaf.groupby_aggregate('g', Daf.sum_da, ['nope'])


def test_groupby_aggregate_of_a_selection():
    """ Only the selected rows are grouped. """
    rows_sdaf: SqlDaf = SqlDaf.from_daf(GROUPBY_REDUCE_DAF, name='rows')
    selected_sdaf: SqlDaf = rows_sdaf.select_where(lambda row: row['x'] > 2, cols=['x'])
    assert selected_sdaf.groupby_aggregate('g', Daf.sum_da, 'x').to_daf().lol == [['', 4], ['b1', 7], ['b2', 8]]


def test_groupby_aggregate_and_reduce_keep_every_digit_of_a_float():
    """ Each cell reaches the function as itself, so floats are not rounded on the way. """
    rows_sdaf: SqlDaf = SqlDaf.from_daf(Daf(lol=[['a', 0.1 + 0.2, 1], ['a', 1 / 3, 2]], cols=['g', 'f', 'n']), name='rows')
    expected: float = (0.1 + 0.2) + 1 / 3
    assert rows_sdaf.groupby_aggregate('g', Daf.sum_da, ['f', 'n']).to_daf().lol == [['a', expected, 3]]
    assert rows_sdaf.groupby_reduce('g', Daf.sum_da, reduce_cols=['f']).to_daf().lol == [['a', expected, '']]
    assert rows_sdaf.groupby_aggregate('g', Daf.sum_da, 'f').to_daf().lol == [['a', expected]]



def test_groupby_aggregate_and_reduce_with_diagnose_give_the_groups():
    """ With diagnose, the first groups are printed, and the result is the same. """
    rows_sdaf: SqlDaf = SqlDaf.from_daf(GROUPBY_REDUCE_DAF, name='rows')
    plain_lol: list = rows_sdaf.groupby_aggregate('g', Daf.sum_da, 'x').to_daf().lol
    assert rows_sdaf.groupby_aggregate('g', Daf.sum_da, 'x', diagnose=True).to_daf().lol == plain_lol
    assert rows_sdaf.groupby_reduce('g', Daf.sum_da, diagnose=True).to_daf().lol == rows_sdaf.groupby_reduce('g', Daf.sum_da).to_daf().lol

def make_images_daf() -> Daf:
    """ Rows that share a crc or a hash, with 0 and blank values that must never match. """
    return Daf(
        cols=['ballot_id', 'crc_p1', 'crc_p2', 'hash_p1', 'hash_p2'],
        lol=[['b1', 7,    1, 'h1', NULL],
             ['b2', 0,    1, NULL, NULL],
             ['b3', 7,    1, 'h1', NULL],
             ['b4', 0,    1, NULL, NULL],
             ['b5', NULL, 1, 'h2', 'x'],
             ['b6', NULL, 1, 'h2', 'y'],
             ['b7', 7,    2, 'h1', NULL]],
    )


@pytest.mark.parametrize('colnames, exclude', [
    (['hash_p1', 'hash_p2'], {'hash_p1': [NULL]}),
    (['hash_p1', 'hash_p2'], None),
    (['crc_p1', 'crc_p2'], None),
    (['crc_p1', 'crc_p2'], {'crc_p1': [0, NULL]}),
    (['crc_p1', 'crc_p2'], {'crc_p1': [0]}),
    (['crc_p1'], {'crc_p2': [2]}),
    (['ballot_id'], None),
])
def test_groupby_repeated_cols_matches_mdaf(colnames, exclude):
    """ The same keys, in the same order, with the same rows in each group, as a Daf gives. """
    images_daf: Daf = make_images_daf()
    images_sdaf: SqlDaf = SqlDaf.from_daf(images_daf, name='images')
    expect_klodaf = images_daf.groupby_repeated_cols(colnames, exclude=exclude)
    repeats_klosdaf = images_sdaf.groupby_repeated_cols(colnames, exclude=exclude)
    assert list(repeats_klosdaf.keys()) == list(expect_klodaf.keys())
    key_t: tuple
    for key_t in expect_klodaf.keys():
        assert isinstance(repeats_klosdaf[key_t], SqlDaf)
        assert repeats_klosdaf[key_t].to_daf().lol == expect_klodaf[key_t].lol


def test_groupby_repeated_cols_reads_blank_text_as_python_null(job_db):
    """ A table written by other code may hold the text '' where SqlDaf writes SQL NULL. Both are one value. """
    make_plain_table(job_db, [('b1', 'h1', ''), ('b2', 'h1', None), ('b3', 'h2', '')], 'ballot_id TEXT, hash_p1 TEXT, hash_p2 TEXT')
    plain_sdaf: SqlDaf = SqlDaf('plain', keyfield='ballot_id')
    repeats_klosdaf = plain_sdaf.groupby_repeated_cols(['hash_p1', 'hash_p2'])
    assert list(repeats_klosdaf.keys()) == [('h1', NULL)]
    assert repeats_klosdaf[('h1', NULL)].keys() == ['b1', 'b2']


def test_groupby_repeated_cols_makes_an_index_only_when_a_key_repeats():
    images_sdaf: SqlDaf = SqlDaf.from_daf(make_images_daf(), name='images')
    assert not images_sdaf.groupby_repeated_cols(['ballot_id'])
    assert key_indexes(images_sdaf.conn) == []
    with pytest.raises(KeyError):
        images_sdaf.groupby_repeated_cols(['crc_p1'], exclude={'nothing': [0]})
    images_sdaf.groupby_repeated_cols(['crc_p1', 'crc_p2'])
    assert any('crc_p1' in name for name in sqldaf_module._indexes_of(images_sdaf.conn, 'images'))


def test_one_line_reads_rows_from_a_daf_or_a_sqldaf():
    """ The same chain ends in the same in-memory Daf, typed, from a table of either kind. """
    text_daf: Daf = Daf(cols=['ballot_id', 'style_num'], lol=[['b1', '3'], ['b2', '5'], ['b3', '7']],
                        keyfield='ballot_id', dtypes={'ballot_id': str, 'style_num': int})
    typed_daf: Daf = text_daf.copy('editable').apply_dtypes()
    table_sdaf: SqlDaf = SqlDaf.from_daf(typed_daf, name='bif')
    table_obj: Any
    for table_obj in (text_daf, table_sdaf):
        rows_daf: Daf = table_obj.select_records_daf(['b3', 'b9', 'b1'], silent_error=True).to_daf().apply_dtypes()
        assert type(rows_daf) is Daf
        assert rows_daf.lol == [['b3', 7], ['b1', 3]]


def test_groupby_cols_keys_are_tuples_and_bad_names_raise():
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    assert list(bif_sdaf.groupby_cols(['precinct']).keys()) == [('P1',), ('P2',), ('P3',)]
    with pytest.raises(KeyError):
        bif_sdaf.groupby('nothing')
    with pytest.raises(NotImplementedError):
        bif_sdaf.groupby('precinct', cols=['ballot_id'])


GROUP_WHERE_FUNCS = [
    ('one key', lambda row: row['precinct']),
    ('overlapping keys', lambda row: [row['precinct'], 'all'] if row['style_num'] > 1 else ['all']),
    ('none leaves a row out', lambda row: row['precinct'] if row['is_bmd'] is not NULL else None),
    ('tuple keys in a list', lambda row: [(row['precinct'], row['style_num']), ('any', 0)]),
    ('a tuple is a list of keys', lambda row: (row['precinct'], 'x')),
    ('true and false', lambda row: row['style_num'] > 1),
    ('no row in any group', lambda row: []),
    ]


@pytest.mark.parametrize('func', [f for _, f in GROUP_WHERE_FUNCS], ids=[name for name, _ in GROUP_WHERE_FUNCS])
def test_group_where_matches_mdaf(func):
    bif_daf: Daf = make_bif_daf()
    bif_sdaf: SqlDaf = SqlDaf.from_daf(bif_daf, name='bif')
    groups_kl: KeyedList = bif_sdaf.group_where(func)
    expect_kl = bif_daf.group_where(func)
    assert list(groups_kl.keys()) == list(expect_kl.keys())
    assert groups_as_lol(groups_kl) == groups_as_lol(expect_kl)


def test_group_where_on_a_selection_with_cols_sorted_by_len():
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    sel_sdaf: SqlDaf = bif_sdaf.select_where_sql('"ballot_id" != ?', ['b4'])
    groups_kl: KeyedList = sel_sdaf.group_where(lambda row: [row['precinct'], 'all'], cols=['precinct'])
    sizes_lt = sorted(((key, group.len()) for key, group in groups_kl.items()), key=lambda kv: kv[1])
    assert sizes_lt == [('P2', 1), ('P1', 3), ('all', 4)]
    assert groups_kl['all'].keyfield == 'ballot_id'


def test_group_where_raises_errors_as_themselves():
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    with pytest.raises(ZeroDivisionError):
        bif_sdaf.group_where(lambda row: 1 / 0)
    with pytest.raises(KeyError):
        bif_sdaf.group_where(lambda row: 1, cols=['nothing'])
    with pytest.raises(NotImplementedError):
        bif_sdaf.group_where(lambda row: 1, indirect_col='x')


def test_group_where_with_many_overlapping_groups_matches_mdaf():
    many_daf: Daf = Daf(lol=[[f'b{i}', i] for i in range(3000)], cols=['ballot_id', 'n'], keyfield='ballot_id')
    many_sdaf: SqlDaf = SqlDaf.from_daf(many_daf, name='many')
    def strata(row):
        return [(c, row['n'] % (c + 2)) for c in range(20)]
    groups_kl: KeyedList = many_sdaf.group_where(strata, cols=['n'])
    expect_kl = many_daf.group_where(strata)
    assert len(groups_kl) == len(expect_kl) == sum(c + 2 for c in range(20))
    assert {key: group.len() for key, group in groups_kl.items()} == {key: len(group) for key, group in expect_kl.items()}
    assert groups_kl[(7, 3)].keys() == expect_kl[(7, 3)].keys()


# value counts and sums

def make_counts_daf() -> Daf:
    return Daf(
        lol=[['b1', 'P1', 1, 10, 'x'], ['b2', 'P2', 2, NULL, '5'], ['b3', 'P1', 3, 2.5, NULL],
             ['b4', NULL, NULL, 'abc', 'y'], ['b5', 'P2', 1, 4, 'x'], ['b6', 'P1', 2, 1, 'x']],
        cols=['ballot_id', 'precinct', 'style', 'n', 't'],
        keyfield='ballot_id',
    )


COUNT_CALLS = [
    ('valuecounts_for_colname', lambda d: d.valuecounts_for_colname('precinct')),
    ('valuecounts_for_colname sorted', lambda d: d.valuecounts_for_colname('style', sort=True)),
    ('valuecounts_for_colname ascending', lambda d: d.valuecounts_for_colname('style', sort=True, reverse=False)),
    ('valuecounts_for_colname omit_nulls', lambda d: d.valuecounts_for_colname('precinct', omit_nulls=True)),
    ('valuecounts_for_colname missing column', lambda d: d.valuecounts_for_colname('nothing')),
    ('valuecounts_for_colnames_ls', lambda d: d.valuecounts_for_colnames_ls(['precinct', 't'])),
    ('valuecounts_for_colnames_ls all', lambda d: d.valuecounts_for_colnames_ls()),
    ('selectedby', lambda d: d.valuecounts_for_colname_selectedby_colname('style', 'precinct', 'P1')),
    ('selectedby NULL', lambda d: d.valuecounts_for_colname_selectedby_colname('t', 'precinct', NULL)),
    ('selectedby ls', lambda d: d.valuecounts_for_colnames_ls_selectedby_colname(['style', 't'], 'precinct', 'P2', sort=True)),
    ('groupedby', lambda d: d.valuecounts_for_colname1_groupedby_colname2('style', 'precinct')),
    ('groupedby sorted', lambda d: d.valuecounts_for_colname1_groupedby_colname2('t', 'precinct', sort=True)),
    ('daf_valuecount', lambda d: d.daf_valuecount(cols=['precinct', 't'])),
    ('daf_valuecount all', lambda d: d.daf_valuecount()),
    ('value_counts_daf', lambda d: d.value_counts_daf('precinct', sort=True, include_total=True).lol),
    ('sum', lambda d: d.sum(['style'])),
    ('groupsum_daf', lambda d: (d.groupsum_daf('precinct', reduce_cols=['style', 'n', 't']).lol, d.groupsum_daf('precinct').keyfield)),
    ('groupsum_daf all', lambda d: d.groupsum_daf('precinct').lol),
    ]


@pytest.mark.parametrize('call', [c for _, c in COUNT_CALLS], ids=[name for name, _ in COUNT_CALLS])
def test_counts_and_sums_match_mdaf(call):
    counts_daf: Daf = make_counts_daf()
    counts_sdaf: SqlDaf = SqlDaf.from_daf(counts_daf, name='counts')
    assert call(counts_sdaf) == call(counts_daf)
    sel_sdaf: SqlDaf = counts_sdaf.select_where_sql('"ballot_id" != ?', ['b1'])
    assert call(sel_sdaf) == call(counts_daf.select_where(lambda row: row['ballot_id'] != 'b1'))


# Python NULL is stored two ways.
# SqlDaf writes SQL NULL. Other code may write the text ''.

def null_form_rows(d: Any) -> list:
    """ The rows of a selection, of a Daf or of a SqlDaf. """
    return d.to_daf().lol if isinstance(d, SqlDaf) else d.lol


NULL_FORM_CALLS = COUNT_CALLS + [
    ('select_by_dict NULL', lambda d: null_form_rows(d.select_by_dict({'precinct': NULL}))),
    ('select_by_dict NULL inverse', lambda d: null_form_rows(d.select_by_dict({'t': NULL}, inverse=True))),
    ('groupby', lambda d: {key: len(group) for key, group in d.groupby('precinct').items()}),
    ('groupby omit_nulls', lambda d: list(d.groupby('t', omit_nulls=True).keys())),
    ]


@pytest.mark.parametrize('call', [c for _, c in NULL_FORM_CALLS], ids=[name for name, _ in NULL_FORM_CALLS])
def test_sql_null_and_empty_text_are_one_python_null(call):
    counts_daf: Daf = make_counts_daf()
    counts_daf.append({'ballot_id': 'b7', 'precinct': NULL, 'style': 2, 'n': 1, 't': NULL})
    counts_sdaf: SqlDaf = SqlDaf.from_daf(counts_daf, name='counts')
    counts_sdaf.sql("UPDATE counts SET precinct = '', t = '' WHERE ballot_id = 'b7'")       # as other code writes it.
    assert counts_sdaf.sql("SELECT typeof(precinct) FROM counts WHERE ballot_id IN ('b4', 'b7')").to_list() == ['null', 'text']
    assert call(counts_sdaf) == call(counts_daf)


def test_counts_keep_the_order_first_seen():
    counts_sdaf: SqlDaf = SqlDaf.from_daf(make_counts_daf(), name='counts')
    assert list(counts_sdaf.valuecounts_for_colname('t')) == ['x', '5', NULL, 'y']


def test_sum_raises_on_text_and_uses_dtypes():
    counts_daf: Daf = make_counts_daf()
    counts_sdaf: SqlDaf = SqlDaf.from_daf(counts_daf, name='counts')
    with pytest.raises(ValueError):
        counts_sdaf.sum(['n'])
    with pytest.raises(ValueError):
        counts_daf.sum(['n'])
    nums_daf: Daf = Daf(lol=[[1, 2.5], [2, 1]], cols=['i', 'f'], dtypes={'i': int, 'f': float})
    nums_sdaf: SqlDaf = SqlDaf.from_daf(nums_daf, name='nums')
    assert nums_sdaf.sum() == nums_daf.sum() == {'i': 3, 'f': 3.5}
    assert nums_sdaf.sum(['i', 'nothing']) == nums_daf.sum(['i', 'nothing']) == {'i': 3}


def test_counts_on_an_empty_table():
    empty_daf: Daf = Daf(cols=['a', 'b'])
    empty_sdaf: SqlDaf = SqlDaf('empty', ['a', 'b'])
    for call in (lambda d: d.valuecounts_for_colname('a'), lambda d: d.daf_valuecount(), lambda d: d.sum(),
                 lambda d: d.groupsum_daf('a').lol):
        assert call(empty_sdaf) == call(empty_daf)


def test_select_where_sql_with_params_and_null():
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    assert bif_sdaf.select_where_sql('"style_num" >= ? AND "precinct" != ?', [2, 'P2']).to_daf().lol == [
        ['b3', 'P1', 3, NULL], ['b5', 'P1', 2, 0]]
    assert bif_sdaf.select_where_sql('"is_bmd" IS ?', [NULL]).to_list() == ['b3', 'P1', 3, NULL]


def test_selection_is_lazy_and_persisted_at_first_read():
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    sel_sdaf: SqlDaf = bif_sdaf.select_by_dict({'precinct': 'P1'})
    assert temp_tables(bif_sdaf.conn) == []
    bif_sdaf.sql("UPDATE bif SET precinct = 'P1' WHERE ballot_id = 'b4'")      # before the first read: counts
    assert len(sel_sdaf) == 4
    assert temp_tables(bif_sdaf.conn) == [sel_sdaf.name]
    bif_sdaf.sql("UPDATE bif SET precinct = 'P1' WHERE ballot_id = 'b2'")      # after: the rows are persisted
    bif_sdaf.sql("UPDATE bif SET style_num = 9 WHERE ballot_id = 'b1'")        # but values show through
    assert sel_sdaf.to_daf().lol == [['b1', 'P1', 9, 0], ['b3', 'P1', 3, NULL], ['b4', 'P1', 1, 1], ['b5', 'P1', 2, 0]]


def test_deleted_base_row_drops_out_of_to_daf_but_len_counts_references():
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    sel_sdaf: SqlDaf = bif_sdaf.select_by_dict({'precinct': 'P1'})._resolve()
    bif_sdaf.sql("DELETE FROM bif WHERE ballot_id = 'b3'")
    assert sel_sdaf.to_daf()[:, 'ballot_id'].to_list() == ['b1', 'b5']
    assert len(sel_sdaf) == 3                      # len() counts references: do not delete base rows.


def test_chain_of_selections_runs_as_one_query_until_read():
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    p1_sdaf: SqlDaf = bif_sdaf.select_by_dict({'precinct': 'P1'})
    p1_style2_sdaf: SqlDaf = p1_sdaf.select_where_sql('"style_num" = 2')
    assert p1_style2_sdaf.parent_sdaf is bif_sdaf
    assert p1_style2_sdaf.to_daf().lol == [['b5', 'P1', 2, 0]]
    assert temp_tables(bif_sdaf.conn) == [p1_style2_sdaf.name]


def test_selection_of_a_persisted_selection_uses_its_rows():
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    p1_sdaf: SqlDaf = bif_sdaf.select_by_dict({'precinct': 'P1'})._resolve()
    bif_sdaf.sql("UPDATE bif SET precinct = 'P1' WHERE ballot_id = 'b2'")
    assert p1_sdaf.select_where_sql('"style_num" = 2').to_daf()[:, 'ballot_id'].to_list() == ['b5']


def test_selection_keeps_keyfield_dtypes_and_columns():
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    sel_sdaf: SqlDaf = bif_sdaf.select_by_dict({'precinct': 'P1'})
    assert sel_sdaf.columns() == bif_sdaf.columns()
    assert sel_sdaf.keyfield == 'ballot_id'
    assert sel_sdaf.name.startswith('daf_tmp_bif_v')
    assert sel_sdaf.is_child


# indexing

@pytest.mark.parametrize('sel', [0, 2, -1, -5, slice(1, 3), slice(None, 2), slice(3, None), slice(-2, None),
                                 slice(None, -3), slice(4, 9), slice(7, 9), slice(3, 1)])
def test_getitem_by_position_matches_mdaf(sel):
    bif_daf: Daf = make_bif_daf()
    bif_sdaf: SqlDaf = SqlDaf.from_daf(bif_daf, name='bif')
    assert bif_sdaf[sel].to_daf().lol == bif_daf[sel].lol


def test_getitem_on_a_selection():
    bif_daf: Daf = make_bif_daf()
    bif_sdaf: SqlDaf = SqlDaf.from_daf(bif_daf, name='bif')
    p1_sdaf: SqlDaf = bif_sdaf.select_by_dict({'precinct': 'P1'})
    assert p1_sdaf[1].to_dict() == {'ballot_id': 'b3', 'precinct': 'P1', 'style_num': 3, 'is_bmd': NULL}
    assert p1_sdaf[1:].to_daf().lol == bif_daf.select_by_dict({'precinct': 'P1'})[1:].lol


def test_getitem_errors():
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    with pytest.raises(IndexError):
        bif_sdaf[5]
    with pytest.raises(IndexError):
        bif_sdaf[-6]
    with pytest.raises(KeyError):
        bif_sdaf['nothing']
    with pytest.raises(NotImplementedError):
        bif_sdaf[0:4:2]
    with pytest.raises(NotImplementedError):
        bif_sdaf[[0, 2]]
    with pytest.raises(NotImplementedError):
        bif_sdaf[0, 'precinct', 'style_num']
    nokey_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='nokey', keyfield='').set_keyfield('')
    with pytest.raises(NotImplementedError):
        nokey_sdaf['b1']


def test_getitem_by_key():
    bif_daf: Daf = make_bif_daf()
    bif_sdaf: SqlDaf = SqlDaf.from_daf(bif_daf, name='bif')
    assert bif_sdaf['b3'].to_daf().lol == bif_daf['b3'].lol
    assert bif_sdaf['b3'].to_dict() == bif_daf['b3'].to_dict()
    assert 'daf_ix_bif_ballot_id' in key_indexes(bif_sdaf.conn)


def test_getitem_by_key_list_matches_mdaf():
    bif_daf: Daf = make_bif_daf()
    bif_sdaf: SqlDaf = SqlDaf.from_daf(bif_daf, name='bif')
    assert bif_sdaf[['b4', 'b1']].to_daf().lol == bif_daf[['b4', 'b1']].lol
    with pytest.raises(KeyError):
        bif_sdaf[['b1', 'nothing']]


@pytest.mark.parametrize('sel', [
    ('b3', 'precinct'), ('b3', ['style_num', 'ballot_id']), (0, 'precinct'), (-1, 'style_num'),
    (slice(1, 3), 'style_num'), (['b4', 'b2'], 'precinct'), (slice(None), 'precinct'),
    (slice(None), ['is_bmd', 'precinct']), ('b2', slice(None))])
def test_getitem_rows_and_cols_matches_mdaf(sel):
    bif_daf: Daf = make_bif_daf()
    bif_sdaf: SqlDaf = SqlDaf.from_daf(bif_daf, name='bif')
    assert bif_sdaf[sel].to_daf().lol == bif_daf[sel].lol
    assert bif_sdaf[sel].columns() == bif_daf[sel].columns()


def test_getitem_one_value():
    bif_daf: Daf = make_bif_daf()
    bif_sdaf: SqlDaf = SqlDaf.from_daf(bif_daf, name='bif')
    assert bif_sdaf['b3', 'style_num'].to_value() == bif_daf['b3', 'style_num'].to_value() == 3
    assert bif_sdaf.select_by_dict({'precinct': 'P1'})[0, 'ballot_id'].to_value() == 'b1'


def test_getitem_rows_and_cols_after_a_delete():
    bif_daf: Daf = make_bif_daf()
    bif_sdaf: SqlDaf = SqlDaf.from_daf(bif_daf, name='bif')
    bif_sdaf.delete_records(['b2'], inplace=True)
    assert bif_sdaf['b3', 'precinct'].to_value() == 'P1'
    assert bif_sdaf[:, 'ballot_id'].to_daf().lol == [['b1'], ['b3'], ['b4'], ['b5']]
    with pytest.raises(ValueError):
        bif_sdaf[0, 'precinct']


def test_getitem_rows_and_cols_errors():
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    with pytest.raises(KeyError):
        bif_sdaf['b1', 'nothing']
    with pytest.raises(KeyError):
        bif_sdaf['nothing', 'precinct']
    with pytest.raises(NotImplementedError):
        bif_sdaf['b1', 0]


# reading one row

@pytest.mark.parametrize('irow', [0, 3, -1, -5])
def test_iloc_matches_mdaf(irow):
    bif_daf: Daf = make_bif_daf()
    bif_sdaf: SqlDaf = SqlDaf.from_daf(bif_daf, name='bif')
    assert bif_sdaf.iloc(irow) == bif_daf.iloc(irow)
    assert bif_sdaf.iloc(irow, rtype='list') == bif_daf.iloc(irow, rtype='list')
    assert bif_sdaf.iloc(irow, include_cols=['style_num', 'ballot_id']) == bif_daf.iloc(irow, include_cols=['style_num', 'ballot_id'])


def test_iloc_errors_and_does_not_fix():
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    with pytest.raises(IndexError):
        bif_sdaf.iloc(5)
    with pytest.raises(IndexError):
        bif_sdaf.iloc(-6)
    with pytest.raises(KeyError):
        bif_sdaf.iloc(0, include_cols=['nothing'])
    with pytest.raises(NotImplementedError):
        bif_sdaf.iloc(0, rtype='klist')
    sel_sdaf: SqlDaf = bif_sdaf.select_by_dict({'precinct': 'P1'})
    assert sel_sdaf.iloc(1)['ballot_id'] == 'b3'
    assert sel_sdaf.is_resolved is False
    assert temp_tables(bif_sdaf.conn) == []


def test_select_record_matches_mdaf():
    bif_daf: Daf = make_bif_daf()
    bif_sdaf: SqlDaf = SqlDaf.from_daf(bif_daf, name='bif')
    assert bif_sdaf.select_record('b3') == bif_daf.select_record('b3')
    assert bif_sdaf.select_record('nothing') == bif_daf.select_record('nothing') == {}
    with pytest.raises(KeyError):
        bif_sdaf.select_record('nothing', silent_error=False)
    sel_sdaf: SqlDaf = bif_sdaf.select_by_dict({'precinct': 'P1'})
    assert sel_sdaf.select_record('b2') == {}
    assert sel_sdaf.select_record('b5')['style_num'] == 2
    assert sel_sdaf.is_resolved is False


def test_select_record_without_keyfield_raises():
    t_sdaf: SqlDaf = SqlDaf.from_daf(Daf(lol=[['a']], cols=['x']), name='t')
    with pytest.raises(ValueError):
        t_sdaf.select_record('a')


# to_dict, to_list, to_value

def test_to_dict_to_list_to_value_match_mdaf():
    bif_daf: Daf = make_bif_daf()
    bif_sdaf: SqlDaf = SqlDaf.from_daf(bif_daf, name='bif')
    assert bif_sdaf[2].to_dict() == bif_daf[2].to_dict()
    assert bif_sdaf[2].to_list() == bif_daf[2].to_list()
    assert bif_sdaf.select_where_sql('"style_num" = 9').to_dict() == {}
    with pytest.raises(ValueError):
        bif_sdaf.to_dict()
    with pytest.raises(ValueError):
        bif_sdaf.to_list()
    with pytest.raises(ValueError):
        bif_sdaf[0].to_value()
    one_col_sdaf: SqlDaf = SqlDaf.from_daf(Daf(lol=[['a'], [NULL], ['c']], cols=['x']), name='one')
    assert one_col_sdaf.to_list() == ['a', NULL, 'c']
    assert one_col_sdaf[1].to_value() is NULL
    assert one_col_sdaf[2].to_value() == 'c'


# copy and save

def test_copy_makes_a_temp_table_of_its_own_rows():
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    copy_sdaf: SqlDaf = bif_sdaf.select_by_dict({'precinct': 'P1'}).copy()
    assert copy_sdaf.is_child is False and copy_sdaf.is_temp is True
    assert copy_sdaf.name in temp_tables(bif_sdaf.conn)
    assert copy_sdaf.name.startswith('daf_tmp_bif_v')
    copy_sdaf.sql(f'UPDATE "{copy_sdaf.name}" SET style_num = 99')
    assert bif_sdaf.select_record('b1')['style_num'] == 1
    assert copy_sdaf.to_daf()[:, 'style_num'].to_list() == [99, 99, 99]
    assert copy_sdaf.keyfield == 'ballot_id'
    whole_sdaf: SqlDaf = bif_sdaf.copy(name='whole')
    assert whole_sdaf is not bif_sdaf and len(whole_sdaf) == 5


def test_copy_inplace_frees_a_child_from_its_parent_as_in_mdaf():
    bif_daf: Daf = make_bif_daf()
    bif_sdaf: SqlDaf = SqlDaf.from_daf(bif_daf, name='bif')
    sel_sdaf: SqlDaf = bif_sdaf.select_by_dict({'precinct': 'P1'})
    sel_daf: Daf = bif_daf.select_by_dict({'precinct': 'P1'})
    assert sel_sdaf.copy(inplace=True) is sel_sdaf
    assert sel_daf.copy(inplace=True) is sel_daf
    assert sel_sdaf.is_child is False and sel_sdaf.parent_sdaf is None
    bif_sdaf.sql("UPDATE bif SET style_num = 99 WHERE ballot_id = 'b1'")
    bif_daf[0, 'style_num'] = 99
    assert sel_sdaf.to_daf().lol == sel_daf.lol == [['b1', 'P1', 1, 0], ['b3', 'P1', 3, NULL], ['b5', 'P1', 2, 0]]
    assert sel_sdaf.select_record('b5') == sel_daf.select_record('b5')


@pytest.mark.parametrize('level', ['shallow', 'sortable', 2])
def test_copy_inplace_with_a_level_that_keeps_rows_shared_raises(level):
    bif_daf: Daf = make_bif_daf()
    bif_sdaf: SqlDaf = SqlDaf.from_daf(bif_daf, name='bif')
    with pytest.raises(ValueError, match='does not copy the rows'):
        bif_sdaf.select_by_dict({'precinct': 'P1'}).copy(level, inplace=True)
    with pytest.raises(ValueError, match='does not copy the rows'):
        bif_daf.select_by_dict({'precinct': 'P1'}).copy(level, inplace=True)
    with pytest.raises(ValueError):
        bif_sdaf.copy('bogus')


def test_copy_name_in_use_raises():
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    with pytest.raises(ValueError):
        bif_sdaf.copy(name='bif')


def test_save_a_copy_and_reopen_it(job_db):
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    middle_sdaf: SqlDaf = bif_sdaf[1:4].copy(name='middle')
    assert middle_sdaf.save() is middle_sdaf
    assert middle_sdaf.is_temp is False
    assert 'middle' not in temp_tables(bif_sdaf.conn)
    SqlDaf.close()
    again_sdaf: SqlDaf = SqlDaf('middle')
    assert again_sdaf.to_daf()[:, 'ballot_id'].to_list() == ['b2', 'b3', 'b4']
    assert again_sdaf.keyfield == 'ballot_id'


def test_save_needs_a_name_for_a_generated_one_and_no_child():
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    sel_sdaf: SqlDaf = bif_sdaf.select_by_dict({'precinct': 'P1'})
    with pytest.raises(ValueError, match='is a child'):
        sel_sdaf.save('p1')
    sel_sdaf.copy(inplace=True)
    with pytest.raises(ValueError, match='generated name'):
        sel_sdaf.save()
    with pytest.raises(ValueError, match='already exists'):
        sel_sdaf.save('bif')
    sel_sdaf.save('p1')
    assert SqlDaf('p1').keys() == ['b1', 'b3', 'b5']


def test_save_a_permanent_table_under_its_own_name_or_a_new_one():
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    assert bif_sdaf.save() is bif_sdaf
    assert bif_sdaf.name == 'bif'
    bif_sdaf.save('bif2')
    assert bif_sdaf.name == 'bif2'
    assert SqlDaf.table_exists('bif') and SqlDaf.table_exists('bif2')
    assert SqlDaf('bif2').keyfield == 'ballot_id'


def test_save_records_a_table_made_by_other_code(job_db):
    make_plain_table(job_db, [('b1', 1), ('b2', 2)], 'ballot_id TEXT, v INTEGER')
    plain_sdaf: SqlDaf = SqlDaf('plain', keyfield='ballot_id')
    plain_sdaf.save()
    assert SqlDaf('plain').keyfield == 'ballot_id'


def test_save_a_join_under_its_name():
    audit_daf, cvr_daf = make_join_pair()
    audit_sdaf: SqlDaf = SqlDaf.from_daf(audit_daf, name='audit')
    cvr_sdaf: SqlDaf = SqlDaf.from_daf(cvr_daf, name='cvr')
    audit_sdaf.join(cvr_sdaf, name='both').save()
    assert SqlDaf('both').to_daf().lol == audit_daf.join(cvr_daf).lol


# tables copied in from other files, and results written out to them

def file_tables(db_path: str) -> T_ls:
    """ The tables in a database file, read on a connection of its own. """
    conn: sqlite3.Connection = sqlite3.connect(db_path)
    names_ls: T_ls = [row_t[0] for row_t in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]
    conn.close()
    return names_ls


def file_bytes(db_path: str) -> bytes:
    with open(db_path, 'rb') as db_f:
        return db_f.read()


def schemas_of(conn: sqlite3.Connection) -> T_ls:
    return [row_t[1] for row_t in conn.execute('PRAGMA database_list')]


def test_from_db_copies_a_table_in_order_with_its_keyfield_and_dtypes(tmp_path, job_db):
    """
    The rows come in their order, with Python NULL kept, and are numbered again from 1.
    The source file is only read, so its bytes do not change, and nothing stays attached.
    """
    source_path: str = str(tmp_path / 'bif.db')
    bif_daf: Daf = make_bif_daf()
    bif_daf.dtypes = {'ballot_id': str, 'precinct': str, 'style_num': int, 'is_bmd': int}
    source_sdaf: SqlDaf = SqlDaf.from_daf(bif_daf, name='bif', db_path=source_path)
    source_sdaf.delete_records(['b2'], inplace=True)
    SqlDaf.close(source_path)
    before_b: bytes = file_bytes(source_path)

    work_sdaf: SqlDaf = SqlDaf.from_db(source_path, 'bif')
    assert work_sdaf.db_path == job_db
    assert work_sdaf.to_daf().lol == bif_daf.select_records_daf(['b2'], inverse=True).lol
    assert work_sdaf.keyfield == 'ballot_id'
    assert work_sdaf.dtypes == bif_daf.dtypes
    assert work_sdaf[2].to_daf().lol == [['b4', 'P3', 1, 1]]            # a position, so daf_rid has no gap.
    assert schemas_of(work_sdaf.conn) == ['main', 'temp']

    work_sdaf.update_record_krow('b1', {'precinct': 'P9'})
    assert file_bytes(source_path) == before_b
    assert SqlDaf('bif', db_path=source_path).select_record('b1')['precinct'] == 'P1'


def test_from_db_copies_only_the_named_table_and_its_own_metadata(tmp_path, job_db):
    source_path: str = str(tmp_path / 'votes.db')
    SqlDaf.from_daf(make_bif_daf(), name='cvr_votes!final', db_path=source_path)
    SqlDaf.from_daf(Daf(lol=[['cvr_votes!final', 'abc']], cols=['table_name', 'chunks_hash'], keyfield='table_name'),
                    name='table_sources', db_path=source_path)

    votes_sdaf: SqlDaf = SqlDaf.from_db(source_path, 'cvr_votes!final')
    assert votes_sdaf.name == 'cvr_votes!final'
    assert file_tables(job_db) == ['cvr_votes!final', 'daf_tables']
    assert SqlDaf('daf_tables').col('name') == ['cvr_votes!final']


def test_from_db_under_a_new_name_and_within_the_working_file(tmp_path, job_db):
    source_path: str = str(tmp_path / 'bif.db')
    SqlDaf.from_daf(make_bif_daf(), name='bif', db_path=source_path)
    assert SqlDaf.from_db(source_path, 'bif', new_name='bif_in').name == 'bif_in'

    again_sdaf: SqlDaf = SqlDaf.from_db(job_db, 'bif_in', new_name='bif_again')
    assert again_sdaf.to_daf().lol == make_bif_daf().lol
    with pytest.raises(ValueError, match='already exists'):
        SqlDaf.from_db(job_db, 'bif_in')


def test_from_db_attaches_the_source_read_only(tmp_path, job_db):
    """ SQLite refuses a write to the source. A path with a space, # or ? is still the plain file. """
    source_path: str = str(tmp_path / 'odd name #1?.db')
    SqlDaf.from_daf(make_bif_daf(), name='bif', db_path=source_path)
    work_sdaf: SqlDaf = SqlDaf.from_db(source_path, 'bif')
    assert work_sdaf.keys() == make_bif_daf().keys()
    with sqldaf_module._file_schema(work_sdaf.conn, job_db, source_path, read_only=True) as schema:
        with pytest.raises(sqlite3.OperationalError, match='readonly'):
            work_sdaf.conn.execute(f'DELETE FROM {schema}.bif')
    assert len(SqlDaf('bif', db_path=source_path)) == 5


def test_from_db_of_a_table_made_by_other_code(tmp_path, job_db):
    """ A table with no daf_rid comes in rowid order, and with no record it has no keyfield. """
    source_path: str = str(tmp_path / 'plain.db')
    make_plain_table(source_path, [('x', 1), ('y', None)])
    plain_sdaf: SqlDaf = SqlDaf.from_db(source_path, 'plain')
    assert plain_sdaf.to_daf().lol == [['x', 1], ['y', NULL]]
    assert plain_sdaf.keyfield == ''


def test_from_db_errors_leave_nothing_attached(tmp_path, job_db):
    """ A transaction would keep the file attached, since SQLite cannot detach a file it has used in one. """
    source_path: str = str(tmp_path / 'bif.db')
    SqlDaf.from_daf(make_bif_daf(), name='bif', db_path=source_path)
    work_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')

    with pytest.raises(FileNotFoundError):
        SqlDaf.from_db(str(tmp_path / 'missing.db'), 'bif', new_name='other')
    assert not os.path.exists(tmp_path / 'missing.db')
    with pytest.raises(KeyError):
        SqlDaf.from_db(source_path, 'nothing')
    with pytest.raises(ValueError, match='already exists'):
        SqlDaf.from_db(source_path, 'bif')
    with pytest.raises(ValueError, match='transaction'):
        with work_sdaf.transaction():
            SqlDaf.from_db(source_path, 'bif', new_name='bif_in')
    assert schemas_of(work_sdaf.conn) == ['main', 'temp']
    assert file_tables(job_db) == ['bif', 'daf_tables']


def test_join_of_tables_from_two_files_saved_into_a_third(tmp_path, job_db):
    """ The join reads both copies in the working file. The result moves into its own file, with its record. """
    audit_daf, cvr_daf = make_join_pair()
    SqlDaf.from_daf(audit_daf, name='audit', db_path=str(tmp_path / 'audit.db'))
    SqlDaf.from_daf(cvr_daf, name='cvr', db_path=str(tmp_path / 'cvr.db'))
    full_path: str = str(tmp_path / 'full' / 'full.db')
    os.makedirs(os.path.dirname(full_path))

    audit_sdaf: SqlDaf = SqlDaf.from_db(str(tmp_path / 'audit.db'), 'audit')
    cvr_sdaf: SqlDaf = SqlDaf.from_db(str(tmp_path / 'cvr.db'), 'cvr')
    join_sdaf: SqlDaf = audit_sdaf.join(cvr_sdaf, how='left')
    temp_name: str = join_sdaf.name
    assert join_sdaf.save('full', db_path=full_path) is join_sdaf

    assert join_sdaf.db_path == full_path
    assert join_sdaf.name == 'full' and join_sdaf.is_temp is False
    assert join_sdaf.to_daf().lol == audit_daf.join(cvr_daf, how='left').lol
    assert temp_name not in temp_tables(audit_sdaf.conn)
    assert schemas_of(audit_sdaf.conn) == ['main', 'temp']
    assert file_tables(full_path) == ['daf_tables', 'full']
    assert file_tables(job_db) == ['audit', 'cvr', 'daf_tables']
    SqlDaf.close(full_path)
    assert SqlDaf('full', db_path=full_path).keyfield == 'id'
    assert audit_sdaf.select_records_daf(cvr_sdaf, inverse=True, silent_error=True).keys() == ['k1']


def test_save_a_permanent_table_into_another_file_copies_it(tmp_path, job_db):
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    out_path: str = str(tmp_path / 'out.db')
    bif_sdaf.save(db_path=out_path)
    assert bif_sdaf.db_path == out_path and bif_sdaf.name == 'bif'
    assert SqlDaf('bif').to_daf().lol == SqlDaf('bif', db_path=out_path).to_daf().lol == make_bif_daf().lol
    with pytest.raises(ValueError, match='already exists'):
        SqlDaf('bif').save(db_path=out_path)
    with pytest.raises(ValueError, match='transaction'):
        with SqlDaf('bif').transaction():
            SqlDaf('bif').save('bif2', db_path=out_path)
    assert schemas_of(SqlDaf('bif').conn) == ['main', 'temp']
    assert file_tables(out_path) == ['bif', 'daf_tables']


# references that survive VACUUM

def test_selection_needs_a_vacuum_safe_reference(job_db):
    make_plain_table(job_db, [('b1', 1), ('b2', 2)], 'ballot_id TEXT, v INTEGER')
    plain_sdaf: SqlDaf = SqlDaf('plain')
    assert plain_sdaf.to_daf().lol == [['b1', 1], ['b2', 2]]
    with pytest.raises(NotImplementedError):
        plain_sdaf.select_where_sql('v = 2')
    keyed_sdaf: SqlDaf = SqlDaf('plain', keyfield='ballot_id')
    assert keyed_sdaf.select_where_sql('v = 2').to_daf().lol == [['b2', 2]]


def test_key_references_survive_delete_and_vacuum(job_db):
    make_plain_table(job_db, [(f'b{i}', i % 3) for i in range(10)], 'ballot_id TEXT, v INTEGER')
    plain_sdaf: SqlDaf = SqlDaf('plain', keyfield='ballot_id')
    sel_sdaf: SqlDaf = plain_sdaf.select_where_sql('v = 0')._resolve()
    plain_sdaf.sql("DELETE FROM plain WHERE ballot_id IN ('b0', 'b1', 'b2')")
    plain_sdaf.sql('VACUUM')
    assert sel_sdaf.to_daf()[:, 'ballot_id'].to_list() == ['b3', 'b6', 'b9']


def test_apply_dtypes_on_a_selection_converts_a_copy():
    text_daf: Daf = Daf(cols=['ballot_id', 'precinct', 'n'], lol=[['b1', 'P1', '1'], ['b2', 'P2', '2'], ['b3', 'P1', '3']],
                        keyfield='ballot_id')
    text_sdaf: SqlDaf = SqlDaf.from_daf(text_daf, name='bif')
    p1_sdaf: SqlDaf = text_sdaf.select_by_dict({'precinct': 'P1'})
    assert p1_sdaf.apply_dtypes(dtypes={'n': int}, cols='n') is p1_sdaf
    assert p1_sdaf.to_daf().lol == [['b1', 'P1', 1], ['b3', 'P1', 3]]
    assert not p1_sdaf.is_child and p1_sdaf.is_temp
    assert text_sdaf.col('n') == ['1', '2', '3']
    assert p1_sdaf['b3', 'n'].to_value() == 3


def test_ensure_key_index_on_a_selection_raises():
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    with pytest.raises(ValueError):
        bif_sdaf.select_by_dict({'precinct': 'P1'}).ensure_key_index()


# positions use daf_rid and the order of a persisted selection

def plan_of(conn: sqlite3.Connection, stmt: str, params: tuple = ()) -> str:
    return ' | '.join(row_t[-1] for row_t in conn.execute('EXPLAIN QUERY PLAN ' + stmt, params))


def test_iloc_and_slices_look_up_dense_positions():
    bif_daf: Daf = make_bif_daf()
    bif_sdaf: SqlDaf = SqlDaf.from_daf(bif_daf, name='bif')
    sel_sdaf: SqlDaf = bif_sdaf.select_by_dict({'precinct': 'P1'})
    sel_daf: Daf = bif_daf.select_by_dict({'precinct': 'P1'})
    assert sel_sdaf.iloc(1) == sel_daf.iloc(1)                 # not persisted: steps with OFFSET
    sel_sdaf._resolve()
    assert sel_sdaf._is_dense() and bif_sdaf._is_dense()
    assert sel_sdaf.iloc(2) == sel_daf.iloc(2)
    assert sel_sdaf[1:3].to_daf().lol == sel_daf[1:3].lol
    assert sel_sdaf[-1].to_dict() == sel_daf[-1].to_dict()
    assert len(sel_sdaf) == 3
    rows_sql, _ = bif_sdaf._rows_sql()
    assert 'USING INTEGER PRIMARY KEY' in plan_of(bif_sdaf.conn, f'SELECT * FROM ({rows_sql}) WHERE daf_ord = ?', (3,))


def test_len_and_bool_of_an_empty_selection():
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    none_sdaf: SqlDaf = bif_sdaf.select_by_dict({'precinct': 'P9'})
    assert len(none_sdaf) == 0
    assert bool(none_sdaf) is False
    assert bool(bif_sdaf.select_by_dict({'precinct': 'P1'})) is True


# ---- step 3: keys, delete_records, concat ----

@pytest.mark.parametrize('keys_ls, inverse', [
    (['b3', 'b1'], False),
    (['b3', 'b1'], True),
    ([], False),
    ([], True),
    (['b2', 'b2'], False),
    ])
def test_select_records_daf_with_a_list_matches_mdaf(keys_ls, inverse):
    bif_daf: Daf = make_bif_daf()
    bif_sdaf: SqlDaf = SqlDaf.from_daf(bif_daf, name='bif')
    assert bif_sdaf.select_records_daf(keys_ls, inverse=inverse).to_daf().lol == \
           bif_daf.select_records_daf(keys_ls, inverse=inverse).lol


def test_select_records_daf_missing_key_matches_mdaf():
    bif_daf: Daf = make_bif_daf()
    bif_sdaf: SqlDaf = SqlDaf.from_daf(bif_daf, name='bif')
    with pytest.raises(KeyError):
        bif_daf.select_records_daf(['b1', 'b9'])
    with pytest.raises(KeyError):
        bif_sdaf.select_records_daf(['b1', 'b9'])
    assert bif_sdaf.select_records_daf(['b9', 'b4', 'b1'], silent_error=True).to_daf().lol == \
           bif_daf.select_records_daf(['b9', 'b4', 'b1'], silent_error=True).lol
    assert bif_sdaf.select_records_daf(['b9', 'b4'], inverse=True, silent_error=True).to_daf().lol == \
           bif_daf.select_records_daf(['b9', 'b4'], inverse=True, silent_error=True).lol


def test_select_records_daf_with_another_sqldaf_and_its_keys():
    bif_daf: Daf = make_bif_daf()
    other_daf: Daf = Daf(lol=[['b4', 0], ['b2', 0], ['b9', 0]], cols=['ballot_id', 'x'], keyfield='ballot_id')
    bif_sdaf: SqlDaf = SqlDaf.from_daf(bif_daf, name='bif')
    other_sdaf: SqlDaf = SqlDaf.from_daf(other_daf, name='other')
    expect_ls = bif_daf.select_records_daf(other_daf, silent_error=True).lol
    assert bif_sdaf.select_records_daf(other_sdaf, silent_error=True).to_daf().lol == expect_ls
    assert bif_sdaf.select_records_daf(other_sdaf.keys(), silent_error=True).to_daf().lol == expect_ls
    assert bif_sdaf.select_records_daf(other_daf, silent_error=True).to_daf().lol == expect_ls
    assert bif_sdaf.select_records_daf(other_sdaf, inverse=True, silent_error=True).to_daf().lol == \
           bif_daf.select_records_daf(other_daf, inverse=True, silent_error=True).lol
    with pytest.raises(KeyError):
        bif_sdaf.select_records_daf(other_sdaf)


def test_a_selection_as_keys_is_found_in_by_an_index_on_its_references():
    """
    A selection's references are declared INTEGER, as daf_rid is, and indexed when first used as keys.
    Then SQLite looks a key up in the selection, instead of scanning every reference for each row.
    """
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    p1_sdaf: SqlDaf = bif_sdaf.select_by_dict({'precinct': 'P1'})
    len(p1_sdaf)
    ref_info_t = next(info_t for info_t in bif_sdaf.conn.execute(f'PRAGMA table_info({quote_name(p1_sdaf.name)})') if info_t[1] == 'ref')
    assert ref_info_t[2] == 'INTEGER'
    assert not any(row_t[1] == p1_sdaf.name + '_ref' for row_t in bif_sdaf.conn.execute('PRAGMA temp.index_list(' + quote_name(p1_sdaf.name) + ')'))

    other_sdaf: SqlDaf = bif_sdaf.select_by_dict({'precinct': 'P2'})
    rest_sdaf: SqlDaf = other_sdaf.select_records_daf(p1_sdaf, inverse=True, silent_error=True)
    plan_ls: T_ls = [row_t[3] for row_t in bif_sdaf.conn.execute('EXPLAIN QUERY PLAN ' + rest_sdaf.pending_sql, rest_sdaf.pending_params)]
    assert any(p1_sdaf.name + '_ref' in step for step in plan_ls)
    assert rest_sdaf.keys() == ['b2']
    assert bif_sdaf.select_records_daf(p1_sdaf).keys() == ['b1', 'b3', 'b5']


def test_a_selection_as_keys_builds_the_key_index_of_its_table():
    """
    Each key is looked up in the selection's table. Without its key index, every row scanned that table.
    The key index is built lazily, as for any lookup by key.
    """
    rows_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='rows')
    keys_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='keys')
    keys_sel_sdaf: SqlDaf = keys_sdaf.select_by_dict({'precinct': 'P1'})
    assert keys_sdaf.key_index_name() not in key_indexes(keys_sdaf.conn)
    assert rows_sdaf.select_records_daf(keys_sel_sdaf, inverse=True).keys() == ['b2', 'b4']
    assert keys_sdaf.key_index_name() in key_indexes(keys_sdaf.conn)


def test_select_record_on_a_selection_keyed_by_another_column():
    """ A selection keyed by a column that is not its table's keyfield needs no key index of the table. """
    plain_sdaf: SqlDaf = SqlDaf.from_daf(Daf(cols=['k', 'g', 'v'], lol=[['a', 'x', 1], ['b', 'y', 2], ['c', 'x', 3]]), name='plain')
    unique_sdaf, _dups_sdaf = plain_sdaf.remove_dups(keyfield='g')
    assert unique_sdaf.select_record('y') == {'k': 'b', 'g': 'y', 'v': 2}
    assert 'x' in unique_sdaf
    assert unique_sdaf['y'].to_daf().lol == [['b', 'y', 2]]


def test_a_selection_of_a_table_keyed_by_text_refers_by_the_key(job_db):
    """ A table made by other code has no daf_rid, so its selections refer to rows by key, with the key's type. """
    make_plain_table(job_db, [('b1', 1), ('b2', 2), ('b3', 1)], 'ballot_id TEXT, v INTEGER')
    plain_sdaf: SqlDaf = SqlDaf('plain', keyfield='ballot_id')
    v1_sdaf: SqlDaf = plain_sdaf.select_where_sql('v = 1')
    len(v1_sdaf)
    ref_info_t = next(info_t for info_t in plain_sdaf.conn.execute(f'PRAGMA table_info({quote_name(v1_sdaf.name)})') if info_t[1] == 'ref')
    assert ref_info_t[2] == 'TEXT'
    assert plain_sdaf.select_records_daf(v1_sdaf, inverse=True).keys() == ['b2']


def test_a_selection_of_an_empty_table_resolves_sorts_and_gives_keys():
    """
    An empty SqlDaf is false, so the parent of a selection is tested against None, not for truth.
    A residue table with no rows is a real case, in the find_variants port.
    """
    empty_sdaf: SqlDaf = SqlDaf.from_daf(Daf(cols=make_bif_daf().columns(), keyfield='ballot_id'), name='empty')
    assert not empty_sdaf
    all_sdaf: SqlDaf = empty_sdaf[:]
    assert all_sdaf.parent_sdaf is empty_sdaf
    assert len(all_sdaf.sort_by_colnames(['ballot_id'], length_priority=True, as_str=True)) == 0
    assert empty_sdaf[:][:].parent_sdaf is empty_sdaf
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    assert bif_sdaf.select_records_daf(empty_sdaf[:], inverse=True, silent_error=True).keys() == make_bif_daf().keys()
    assert len(empty_sdaf[:].copy()) == 0


def test_a_deep_chain_of_selections_raises_unless_read_on_the_way():
    """
    A selection of a selection not read yet nests its query. SQLite stops at about 400 levels, at once.
    A selection that has been read starts a fresh query, so a chain read on the way has no limit.
    """
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    sel_sdaf: SqlDaf = bif_sdaf
    with pytest.raises(sqlite3.OperationalError, match='Recursion limit'):
        for _ in range(1000):
            sel_sdaf = sel_sdaf.select_where_sql('style_num >= ?', [0])
        len(sel_sdaf)
    sel_sdaf = bif_sdaf
    for _ in range(500):
        sel_sdaf = sel_sdaf.select_where_sql('style_num >= ?', [0])
        len(sel_sdaf)
    assert sel_sdaf.parent_sdaf is bif_sdaf
    assert sel_sdaf.keys() == make_bif_daf().keys()


def test_select_records_daf_with_a_selection_as_keys():
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    p1_sdaf: SqlDaf = bif_sdaf.select_by_dict({'precinct': 'P1'})
    assert bif_sdaf.select_records_daf(p1_sdaf, inverse=True).to_daf()[:, 'ballot_id'].to_list() == ['b2', 'b4']


def test_select_records_daf_needs_keyfields():
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    nokey_sdaf: SqlDaf = SqlDaf.from_daf(Daf(lol=[['b1']], cols=['ballot_id']), name='nokey')
    with pytest.raises(ValueError):
        nokey_sdaf.select_records_daf(['b1'])
    with pytest.raises(ValueError):
        bif_sdaf.select_records_daf(nokey_sdaf)


def test_keys_takes_the_arguments_of_mdaf():
    """ astype and silent_error work as on a Daf, so code written for a Daf runs unchanged. """
    bif_daf: Daf = make_bif_daf()
    bif_sdaf: SqlDaf = SqlDaf.from_daf(bif_daf, name='bif')
    assert bif_sdaf.keys(astype='list') == bif_daf.keys(astype='list')
    view_kva = bif_sdaf.keys(astype='view')
    assert type(view_kva) is type(bif_daf.keys(astype='view'))
    assert list(view_kva) == list(bif_daf.keys(astype='view'))
    assert 'b3' in view_kva
    with pytest.raises(ValueError):
        bif_sdaf.keys(astype='set')
    with pytest.raises(ValueError):
        bif_daf.keys(astype='set')

    nokey_daf: Daf = Daf(lol=[['b1']], cols=['ballot_id'])
    nokey_sdaf: SqlDaf = SqlDaf.from_daf(nokey_daf, name='nokey')
    assert nokey_sdaf.keys() == nokey_daf.keys() == []
    with pytest.raises(KeysDisabledError):
        nokey_sdaf.keys(silent_error=False)
    with pytest.raises(KeysDisabledError):
        nokey_daf.keys(silent_error=False)


def test_keys_is_a_list_in_order_as_in_mdaf():
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    assert bif_sdaf.keys() == make_bif_daf().keys()
    assert type(bif_sdaf.keys()) is type(make_bif_daf().keys()) is list
    assert list(bif_sdaf.select_by_dict({'precinct': 'P1'}).keys()) == ['b1', 'b3', 'b5']
    assert SqlDaf.from_daf(Daf(lol=[['a']], cols=['x']), name='nk').keys() == []


@pytest.mark.parametrize('inplace', [False, True])
def test_delete_records_on_mdaf_and_sdaf_agree(inplace):
    bif_daf: Daf = make_bif_daf()
    bif_sdaf: SqlDaf = SqlDaf.from_daf(bif_daf, name='bif')
    result_sdaf: SqlDaf = bif_sdaf.delete_records(['b2', 'b4'], inplace=inplace)
    result_daf: Daf = bif_daf.delete_records(['b2', 'b4'], inplace=inplace)
    assert result_sdaf.to_daf().lol == result_daf.lol
    assert (result_sdaf is bif_sdaf) is inplace
    assert bif_sdaf.to_daf().lol == bif_daf.lol


def test_delete_records_default_changes_nothing():
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    kept_sdaf: SqlDaf = bif_sdaf.delete_records(['b2'])
    assert len(bif_sdaf) == 5
    assert kept_sdaf.parent_sdaf is bif_sdaf
    assert len(kept_sdaf) == 4


def test_delete_records_inplace_on_a_selection_leaves_the_base():
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    p1_sdaf: SqlDaf = bif_sdaf.select_by_dict({'precinct': 'P1'})
    assert p1_sdaf.delete_records(['b3'], inplace=True) is p1_sdaf
    assert p1_sdaf.to_daf()[:, 'ballot_id'].to_list() == ['b1', 'b5']
    assert len(p1_sdaf) == 2
    assert p1_sdaf.iloc(1)['ballot_id'] == 'b5'                 # renumbered, so positions stay lookups
    assert len(bif_sdaf) == 5


def test_delete_records_inplace_on_a_base_makes_selections_stale():
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    p1_sdaf: SqlDaf = bif_sdaf.select_by_dict({'precinct': 'P1'})._resolve()
    bif_sdaf.delete_records(['b3'], inplace=True)
    assert len(bif_sdaf) == 4
    assert p1_sdaf.to_daf()[:, 'ballot_id'].to_list() == ['b1', 'b5']
    assert len(p1_sdaf) == 3                                    # stale, as the docstring warns


def test_delete_records_inplace_on_a_base_stops_positions():
    bif_daf: Daf = make_bif_daf()
    bif_sdaf: SqlDaf = SqlDaf.from_daf(bif_daf, name='bif')
    bif_sdaf.delete_records(['b2'], inplace=True)
    bif_daf.delete_records(['b2'], inplace=True)
    for read in (lambda: bif_sdaf.iloc(1), lambda: bif_sdaf.iloc(-1), lambda: bif_sdaf.iloc_exists(1),
                 lambda: bif_sdaf[1], lambda: bif_sdaf[1:3]):
        with pytest.raises(ValueError, match="'bif' has had rows deleted"):
            read()
    assert bif_sdaf['b3'].to_daf().lol == [bif_daf['b3'].lol[0]]       # by key still works
    copy_sdaf: SqlDaf = bif_sdaf.copy()
    assert copy_sdaf.iloc(1) == bif_daf.iloc(1)                         # a copy numbers the rows again
    assert copy_sdaf[1:3].to_daf().lol == bif_daf[1:3].lol
    p1_sdaf: SqlDaf = bif_sdaf.select_by_dict({'precinct': 'P1'})
    assert p1_sdaf.iloc(1) == bif_daf.select_by_dict({'precinct': 'P1'}).iloc(1)   # a selection numbers its own rows


def test_delete_records_inplace_then_concat_reuses_the_last_rid():
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    tail_sdaf: SqlDaf = bif_sdaf[3:]._resolve()
    assert tail_sdaf.keys() == ['b4', 'b5']
    bif_sdaf.delete_records(['b5'], inplace=True)
    bif_sdaf.concat(Daf(cols=bif_sdaf.columns(), lol=[['b9'] + [NULL] * (bif_sdaf.num_cols() - 1)]))
    assert tail_sdaf.keys() == ['b4', 'b9']                     # the documented risk of deleting from a parent with children


def test_positions_after_deleting_only_the_last_row():
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    bif_sdaf.delete_records(['b5'], inplace=True)
    assert bif_sdaf.iloc(-1)['ballot_id'] == 'b4'                       # no gap, so positions still work
    assert bif_sdaf[1:3].to_daf()[:, 'ballot_id'].to_list() == ['b2', 'b3']


def test_positions_check_a_reopened_table_and_sql_deletes(job_db):
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    assert bif_sdaf.iloc(1)['ballot_id'] == 'b2'
    bif_sdaf.sql('SELECT * FROM bif')
    assert 'bif' in bif_sdaf._conn_da['no_gaps_ss']                     # a SELECT changes no rows, so no check again
    assert bif_sdaf.sql('DELETE FROM bif WHERE ballot_id = ? RETURNING ballot_id', ['b5']).to_list() == ['b5']
    assert 'bif' not in bif_sdaf._conn_da['no_gaps_ss']
    assert bif_sdaf.iloc(-1)['ballot_id'] == 'b4'                       # the last row leaves no gap
    bif_sdaf.sql('DELETE FROM bif WHERE ballot_id = ?', ['b1'])
    with pytest.raises(ValueError):
        bif_sdaf.iloc(1)
    SqlDaf.close(job_db)
    with pytest.raises(ValueError):
        SqlDaf('bif').iloc(1)                                           # found again on a new connection


def test_positions_after_delete_and_remake(job_db):
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    bif_sdaf.delete_records(['b2'], inplace=True)
    with pytest.raises(ValueError):
        bif_sdaf.iloc(1)
    bif_sdaf.delete()
    again_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    assert again_sdaf.iloc(1)['ballot_id'] == 'b2'


def test_positions_in_a_table_without_rid_after_a_delete(job_db):
    make_plain_table(job_db, [['x', 1], ['y', 2], ['z', 3]])
    plain_sdaf: SqlDaf = SqlDaf('plain', keyfield='a')
    plain_sdaf.sql("DELETE FROM plain WHERE a = 'x'")
    assert plain_sdaf.iloc(0) == {'a': 'y', 'b': 2}                      # counted by OFFSET, so still right
    assert plain_sdaf[1:2].to_daf().lol == [['z', 3]]


def test_delete_records_missing_key_removes_nothing():
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    with pytest.raises(KeyError):
        bif_sdaf.delete_records(['b1', 'b9'], inplace=True)
    assert len(bif_sdaf) == 5
    bif_sdaf.delete_records(['b1', 'b9'], inplace=True, silent_error=True)
    assert list(bif_sdaf.keys()) == ['b2', 'b3', 'b4', 'b5']


def test_delete_records_with_another_sqldaf():
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    p1_sdaf: SqlDaf = bif_sdaf.select_by_dict({'precinct': 'P1'})
    rest_sdaf: SqlDaf = bif_sdaf.delete_records(p1_sdaf)
    assert list(rest_sdaf.keys()) == ['b2', 'b4']


@pytest.mark.parametrize('colname, icol, startat', [
    ('idx', 0, 0), ('idx', 2, 1), ('idx', 4, 0), ('idx', 9, 5), ('idx', -1, 0), ('idx', -2, 0),
    ('style_num', 0, 10), ('', 0, 0),
])
def test_insert_idx_col_matches_mdaf(colname, icol, startat):
    """ The same columns and numbers as a Daf, on a TEMP copy. The table it came from is not changed. """
    bif_daf: Daf = make_bif_daf()
    bif_sdaf: SqlDaf = SqlDaf.from_daf(bif_daf, name='bif')
    numbered_sdaf: SqlDaf = bif_sdaf.select_by_dict({'precinct': 'P1'}).insert_idx_col(colname, icol, startat)
    expect_daf: Daf = bif_daf.select_by_dict({'precinct': 'P1'}).copy('editable').insert_idx_col(colname, icol, startat)
    assert numbered_sdaf.columns() == expect_daf.columns()
    assert numbered_sdaf.to_daf().lol == expect_daf.lol
    assert numbered_sdaf.keyfield == 'ballot_id'
    assert SqlDaf('bif').to_daf().lol == bif_daf.lol


def test_insert_idx_col_on_a_table_refers_to_a_temp_copy():
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    assert bif_sdaf.insert_idx_col() is bif_sdaf
    assert bif_sdaf.is_temp and bif_sdaf.name != 'bif'
    assert bif_sdaf.col('idx') == [0, 1, 2, 3, 4]
    assert bif_sdaf.select_record('b3')['idx'] == 2
    assert SqlDaf('bif').columns() == make_bif_daf().columns()


def test_concat_matches_mdaf_and_returns_self():
    bif_daf: Daf = make_bif_daf()
    add_daf: Daf = Daf(lol=[['b6', 'P2', 1, 1], ['b7', 'P1', 2, NULL]], cols=bif_daf.columns(), keyfield='ballot_id')
    bif_sdaf: SqlDaf = SqlDaf.from_daf(bif_daf, name='bif')
    assert bif_sdaf.concat(add_daf) is bif_sdaf
    assert bif_sdaf.to_daf().lol == bif_daf.copy('editable').concat(add_daf).lol
    add_sdaf: SqlDaf = SqlDaf.from_daf(Daf(lol=[['b8', 'P3', 3, 0]], cols=bif_daf.columns()), name='add')
    bif_sdaf.concat(add_sdaf)
    assert list(bif_sdaf.keys())[-1] == 'b8'


def test_concat_columns_must_match():
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    with pytest.raises(KeyError, match=r"concat\(\) into 'bif': the columns differ\. Only in this table: \[.*\]\. Only in the other: \[\]\."):
        bif_sdaf.concat(Daf(lol=[['b9', 'P1']], cols=['ballot_id', 'precinct']))
    bif_sdaf.concat(make_bif_daf(), respect_kd=True)                   # every key is there, so every row is replaced
    assert bif_sdaf.to_daf().lol == make_bif_daf().lol


def test_concat_of_an_empty_other_adds_nothing_whatever_its_columns():
    """
    An other with no rows is skipped, as in a Daf.
    The empty result of apply() has no columns, and a SqlDaf of other columns is empty too.
    """
    bif_daf: Daf = make_bif_daf()
    bif_sdaf: SqlDaf = SqlDaf.from_daf(bif_daf, name='bif')
    none_sdaf: SqlDaf = SqlDaf.from_daf(Daf(cols=['ballot_id']), name='none')
    shells_sdaf: SqlDaf = none_sdaf.apply(lambda row_da: {'ballot_id': row_da['ballot_id'], 'v': 0})
    assert shells_sdaf.columns() == []
    other_sdaf: SqlDaf = SqlDaf.from_daf(Daf(cols=['z']), name='other')
    empty_obj: Any
    for empty_obj in (shells_sdaf, other_sdaf, Daf(cols=['z'])):
        assert bif_sdaf.concat(empty_obj) is bif_sdaf
        assert bif_daf.copy('editable').concat(empty_obj).lol == bif_daf.lol
    assert bif_sdaf.to_daf().lol == bif_daf.lol


def test_a_daf_appends_or_concats_a_sqldaf_by_reading_it():
    """ A Daf reads the SqlDaf's rows with to_daf(), so the result is a Daf in memory, as if it were given a Daf. """
    bif_daf: Daf = make_bif_daf()
    bif_sdaf: SqlDaf = SqlDaf.from_daf(bif_daf, name='bif')
    p1_sdaf: SqlDaf = bif_sdaf.select_by_dict({'precinct': 'P1'})
    p1_daf: Daf = bif_daf.select_by_dict({'precinct': 'P1'})

    empty_daf: Daf = Daf().append(p1_sdaf)
    assert type(empty_daf) is Daf
    assert empty_daf.lol == Daf().append(p1_daf).lol
    assert empty_daf.keyfield == 'ballot_id'

    rest_daf: Daf = bif_daf.select_by_dict({'precinct': 'P1'}, inverse=True).copy('editable')
    assert rest_daf.concat(p1_sdaf).lol == bif_daf.select_by_dict({'precinct': 'P1'}, inverse=True).copy('editable').concat(p1_daf).lol
    upsert_daf: Daf = bif_daf.copy('editable')
    changed_sdaf: SqlDaf = SqlDaf.from_daf(Daf(lol=[['b1', 'P9', 9, 1]], cols=bif_daf.columns(), keyfield='ballot_id'), name='changed')
    assert upsert_daf.concat(changed_sdaf, respect_kd=True).select_record('b1')['precinct'] == 'P9'
    assert bif_sdaf.select_record('b1')['precinct'] == 'P1'


def test_concat_duplicate_key_raises_and_adds_nothing():
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    dup_daf: Daf = Daf(lol=[['b6', 'P2', 1, 1], ['b1', 'P1', 9, 9]], cols=make_bif_daf().columns())
    with pytest.raises(sqlite3.IntegrityError):
        bif_sdaf.concat(dup_daf)
    assert len(bif_sdaf) == 5


def test_concat_on_a_selection_makes_it_a_table_of_its_own():
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    p1_sdaf: SqlDaf = bif_sdaf.select_by_dict({'precinct': 'P1'})
    p1_sdaf.concat(Daf(lol=[['b9', 'P1', 5, 0]], cols=make_bif_daf().columns()))
    assert p1_sdaf.parent_sdaf is None
    assert list(p1_sdaf.keys()) == ['b1', 'b3', 'b5', 'b9']
    assert len(bif_sdaf) == 5


# ---- join ----

def make_join_pair() -> tuple:
    audit_daf: Daf = Daf(lol=[['k3', 'A3', 1], ['k1', 'A1', 2], ['k2', 'A2', NULL]], cols=['id', 'name', 'v'], keyfield='id', name='audit')
    cvr_daf: Daf = Daf(lol=[['k2', 'B2', 9], ['k4', 'B4', 8], ['k3', NULL, 7]], cols=['id', 'name', 'w'], keyfield='id', name='cvr')
    return audit_daf, cvr_daf


@pytest.mark.parametrize('how', ['inner', 'left', 'right', 'outer'])
@pytest.mark.parametrize('tag_other', [False, True])
def test_join_matches_mdaf(how, tag_other):
    audit_daf, cvr_daf = make_join_pair()
    audit_sdaf: SqlDaf = SqlDaf.from_daf(audit_daf, name='audit')
    cvr_sdaf: SqlDaf = SqlDaf.from_daf(cvr_daf, name='cvr')
    join_sdaf: SqlDaf = audit_sdaf.join(cvr_sdaf, how=how, tag_other=tag_other)
    join_daf: Daf = audit_daf.join(cvr_daf, how=how, tag_other=tag_other)
    assert join_sdaf.columns() == join_daf.columns()
    assert join_sdaf.to_daf().lol == join_daf.lol
    assert join_sdaf.keyfield == join_daf.keyfield == 'id'


@pytest.mark.parametrize('how', ['inner', 'left', 'right', 'outer'])
def test_join_shared_fields_match_mdaf(how):
    audit_daf, cvr_daf = make_join_pair()
    audit_sdaf: SqlDaf = SqlDaf.from_daf(audit_daf, name='audit')
    cvr_sdaf: SqlDaf = SqlDaf.from_daf(cvr_daf, name='cvr')
    assert audit_sdaf.join(cvr_sdaf, how=how, shared_fields=['name']).to_daf().lol == \
           audit_daf.join(cvr_daf, how=how, shared_fields=['name']).lol


def test_join_other_only_row_takes_key_and_shared_field_from_the_other():
    audit_daf: Daf = Daf(lol=[['k3', 'A3', 1]], cols=['id', 'name', 'v'], keyfield='id')
    cvr_daf: Daf = Daf(lol=[['k3', 'B3', 7], ['k4', 'B4', 8]], cols=['ballot', 'name', 'w'], keyfield='ballot')
    audit_sdaf: SqlDaf = SqlDaf.from_daf(audit_daf, name='audit')
    cvr_sdaf: SqlDaf = SqlDaf.from_daf(cvr_daf, name='cvr')
    assert audit_sdaf.join(cvr_sdaf, how='outer', shared_fields=['name']).to_daf().lol == [
        ['k3', 'A3', 1, 7], ['k4', 'B4', NULL, 8]]
    assert audit_daf.join(cvr_daf, how='outer', shared_fields=['name']).lol == [
        ['k3', 'A3', 1, 7], ['k4', 'B4', NULL, 8]]


def test_join_is_lazy_and_resolves_once():
    audit_daf, cvr_daf = make_join_pair()
    audit_sdaf: SqlDaf = SqlDaf.from_daf(audit_daf, name='audit')
    cvr_sdaf: SqlDaf = SqlDaf.from_daf(cvr_daf, name='cvr')
    join_sdaf: SqlDaf = audit_sdaf.join(cvr_sdaf, how='outer')
    assert join_sdaf.is_resolved is False and temp_tables(audit_sdaf.conn) == []
    assert join_sdaf.iloc(3)['id'] == 'k4'                      # a one-row read does not resolve it
    assert join_sdaf.is_resolved is False
    assert len(join_sdaf) == 4
    assert join_sdaf.is_resolved is True and temp_tables(audit_sdaf.conn) == [join_sdaf.name]
    assert join_sdaf.name.startswith('daf_tmp_audit_join_cvr_v')


def test_join_result_is_new_rows_not_tied_to_its_sources():
    audit_daf, cvr_daf = make_join_pair()
    audit_sdaf: SqlDaf = SqlDaf.from_daf(audit_daf, name='audit')
    cvr_sdaf: SqlDaf = SqlDaf.from_daf(cvr_daf, name='cvr')
    join_sdaf: SqlDaf = audit_sdaf.join(cvr_sdaf)._resolve()
    audit_sdaf.sql("UPDATE audit SET v = 99 WHERE id = 'k3'")
    assert join_sdaf.select_record('k3')['v'] == 1


def test_selection_of_an_unread_join_resolves_it_first():
    audit_daf, cvr_daf = make_join_pair()
    audit_sdaf: SqlDaf = SqlDaf.from_daf(audit_daf, name='audit')
    cvr_sdaf: SqlDaf = SqlDaf.from_daf(cvr_daf, name='cvr')
    join_sdaf: SqlDaf = audit_sdaf.join(cvr_sdaf, tag_other=True)
    differ_sdaf: SqlDaf = join_sdaf.select_where_sql('"name_audit" IS NOT "name_cvr"')
    assert join_sdaf.is_resolved is True
    assert differ_sdaf.parent_sdaf is join_sdaf
    assert list(differ_sdaf.keys()) == ['k3', 'k2']
    assert join_sdaf[1:].to_daf()[:, 'id'].to_list() == ['k2']
    assert list(audit_sdaf.select_records_daf(differ_sdaf, inverse=True).keys()) == ['k1']


def test_join_of_selections():
    audit_daf, cvr_daf = make_join_pair()
    audit_sdaf: SqlDaf = SqlDaf.from_daf(audit_daf, name='audit')
    cvr_sdaf: SqlDaf = SqlDaf.from_daf(cvr_daf, name='cvr')
    some_sdaf: SqlDaf = audit_sdaf.select_records_daf(['k2', 'k3'])
    assert some_sdaf.join(cvr_sdaf).to_daf().lol == audit_daf.select_records_daf(['k2', 'k3']).join(cvr_daf).lol


def test_join_errors():
    audit_daf, cvr_daf = make_join_pair()
    audit_sdaf: SqlDaf = SqlDaf.from_daf(audit_daf, name='audit')
    cvr_sdaf: SqlDaf = SqlDaf.from_daf(cvr_daf, name='cvr')
    with pytest.raises(ValueError):
        audit_sdaf.join(cvr_sdaf, how='cross')
    with pytest.raises(NotImplementedError):
        audit_sdaf.join(cvr_daf)                               # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        audit_sdaf.join(cvr_sdaf, fill=None)
    nokey_sdaf: SqlDaf = SqlDaf.from_daf(Daf(lol=[['k1']], cols=['id']), name='nokey')
    with pytest.raises(ValueError):
        audit_sdaf.join(nokey_sdaf)


def test_tables_opened_separately_join_and_share_children():
    audit_daf, cvr_daf = make_join_pair()
    SqlDaf.from_daf(audit_daf, name='audit')
    SqlDaf.from_daf(cvr_daf, name='cvr')
    audit_sdaf: SqlDaf = SqlDaf('audit')
    cvr_sdaf: SqlDaf = SqlDaf('cvr')
    assert audit_sdaf.join(cvr_sdaf, how='outer').to_daf().lol == audit_daf.join(cvr_daf, how='outer').lol
    assert audit_sdaf.select_records_daf(cvr_sdaf, silent_error=True).keys() == ['k2', 'k3']
    some_sdaf: SqlDaf = cvr_sdaf.select_records_daf(['k2'])
    assert audit_sdaf.join(some_sdaf).keys() == ['k2']
    assert audit_sdaf.join(cvr_sdaf.copy()).keys() == ['k3', 'k2']


def test_another_thread_has_its_own_connection_and_its_children_cannot_be_joined():
    audit_daf, cvr_daf = make_join_pair()
    audit_sdaf: SqlDaf = SqlDaf.from_daf(audit_daf, name='audit')
    SqlDaf.from_daf(cvr_daf, name='cvr')
    made_ls: list = []

    def open_in_thread() -> None:
        cvr_sdaf: SqlDaf = SqlDaf('cvr')
        made_ls.extend([cvr_sdaf, cvr_sdaf.select_records_daf(['k2'])])
        SqlDaf.close()                                  # the thread's own connection.

    thread = threading.Thread(target=open_in_thread)
    thread.start()
    thread.join()
    cvr_sdaf, some_sdaf = made_ls
    assert cvr_sdaf.conn is not audit_sdaf.conn
    with pytest.raises(NotImplementedError, match="another thread's connection"):
        audit_sdaf.join(some_sdaf)


def _temp_tables(sdaf: SqlDaf) -> list:
    return [row_t[0] for row_t in sdaf.conn.execute("SELECT name FROM sqlite_temp_master WHERE type = 'table'")]


def test_join_with_a_table_in_another_file_brings_it_in_and_matches_one_file(tmp_path):
    """ The other table is copied into TEMP for the join, and the result is as for two tables in one file. """
    audit_daf, cvr_daf = make_join_pair()
    audit_sdaf: SqlDaf = SqlDaf.from_daf(audit_daf, name='audit')
    near_sdaf: SqlDaf = SqlDaf.from_daf(cvr_daf, name='cvr_here')
    other_path: str = str(tmp_path / 'other.db')
    far_sdaf: SqlDaf = SqlDaf.from_daf(cvr_daf, name='cvr', db_path=other_path)
    for how in ('inner', 'left', 'right', 'outer'):
        far_lol = audit_sdaf.join(far_sdaf, how=how).to_daf().lol
        near_lol = audit_sdaf.join(near_sdaf, how=how).to_daf().lol
        assert far_lol == near_lol, how


def test_join_copy_lives_until_the_join_is_read(tmp_path):
    audit_daf, cvr_daf = make_join_pair()
    audit_sdaf: SqlDaf = SqlDaf.from_daf(audit_daf, name='audit')
    far_sdaf: SqlDaf = SqlDaf.from_daf(cvr_daf, name='cvr', db_path=str(tmp_path / 'other.db'))
    join_sdaf: SqlDaf = audit_sdaf.join(far_sdaf, how='left')
    assert [name for name in _temp_tables(audit_sdaf) if 'cvr' in name and 'join' not in name]
    join_sdaf.to_daf()
    assert not [name for name in _temp_tables(audit_sdaf) if 'cvr' in name and 'join' not in name]


def test_join_never_read_drops_its_copy_when_it_is_gone(tmp_path):
    audit_daf, cvr_daf = make_join_pair()
    audit_sdaf: SqlDaf = SqlDaf.from_daf(audit_daf, name='audit')
    far_sdaf: SqlDaf = SqlDaf.from_daf(cvr_daf, name='cvr', db_path=str(tmp_path / 'other.db'))
    join_sdaf: SqlDaf | None = audit_sdaf.join(far_sdaf, how='left')
    assert [name for name in _temp_tables(audit_sdaf) if 'cvr' in name]
    join_sdaf = None
    assert not [name for name in _temp_tables(audit_sdaf) if 'cvr' in name]


def test_chain_of_joins_across_three_files_leaves_no_copy(tmp_path):
    audit_daf, cvr_daf = make_join_pair()
    audit_sdaf: SqlDaf = SqlDaf.from_daf(audit_daf, name='audit')
    cvr_sdaf: SqlDaf = SqlDaf.from_daf(cvr_daf, name='cvr', db_path=str(tmp_path / 'cvr.db'))
    notes_daf: Daf = Daf(cols=[audit_daf.keyfield, 'note'], lol=[[key, f'n{i}'] for i, key in enumerate(audit_daf.keys())],
                         keyfield=audit_daf.keyfield)
    notes_sdaf: SqlDaf = SqlDaf.from_daf(notes_daf, name='notes', db_path=str(tmp_path / 'notes.db'))
    chain_sdaf: SqlDaf = audit_sdaf.join(cvr_sdaf, how='left').join(notes_sdaf, how='left')
    assert chain_sdaf.col('note') == [f'n{i}' for i in range(len(audit_daf))]
    assert all('join' in name for name in _temp_tables(audit_sdaf))


def test_join_leaves_both_files_as_they_were(tmp_path):
    audit_daf, cvr_daf = make_join_pair()
    audit_path: str = str(tmp_path / 'audit.db')
    other_path: str = str(tmp_path / 'other.db')
    SqlDaf.from_daf(audit_daf, name='audit', db_path=audit_path)
    SqlDaf.from_daf(cvr_daf, name='cvr', db_path=other_path)
    SqlDaf.close(audit_path)
    SqlDaf.close(other_path)
    other_stat = os.stat(other_path)
    audit_sdaf: SqlDaf = SqlDaf('audit', db_path=audit_path, read_only=True)
    audit_sdaf.join(SqlDaf('cvr', db_path=other_path), how='left').to_daf()
    assert (os.stat(other_path).st_size, os.stat(other_path).st_mtime) == (other_stat.st_size, other_stat.st_mtime)
    assert [row_t[0] for row_t in audit_sdaf.conn.execute("SELECT name FROM main.sqlite_master WHERE type = 'table'")] == ['audit', 'daf_tables']


def test_join_with_a_selection_in_another_file_copies_it_through_python(tmp_path):
    audit_daf, cvr_daf = make_join_pair()
    audit_sdaf: SqlDaf = SqlDaf.from_daf(audit_daf, name='audit')
    far_sdaf: SqlDaf = SqlDaf.from_daf(cvr_daf, name='cvr', db_path=str(tmp_path / 'other.db'))
    some_sdaf: SqlDaf = far_sdaf[0:2]
    near_sdaf: SqlDaf = SqlDaf.from_daf(cvr_daf, name='cvr_here')[0:2]
    assert audit_sdaf.join(some_sdaf, how='inner').to_daf().lol == audit_sdaf.join(near_sdaf, how='inner').to_daf().lol


def test_join_inside_a_transaction_copies_through_python(tmp_path):
    audit_daf, cvr_daf = make_join_pair()
    audit_sdaf: SqlDaf = SqlDaf.from_daf(audit_daf, name='audit')
    far_sdaf: SqlDaf = SqlDaf.from_daf(cvr_daf, name='cvr', db_path=str(tmp_path / 'other.db'))
    near_sdaf: SqlDaf = SqlDaf.from_daf(cvr_daf, name='cvr_here')
    with audit_sdaf.transaction():
        far_lol = audit_sdaf.join(far_sdaf, how='left').to_daf().lol
    assert far_lol == audit_sdaf.join(near_sdaf, how='left').to_daf().lol


def test_concat_and_keys_from_another_file(tmp_path):
    audit_daf, cvr_daf = make_join_pair()
    audit_sdaf: SqlDaf = SqlDaf.from_daf(audit_daf, name='audit')
    far_sdaf: SqlDaf = SqlDaf.from_daf(audit_daf.clone_empty(), name='more', db_path=str(tmp_path / 'other.db'))
    far_sdaf.append(['z9'] + [''] * (len(audit_daf.columns()) - 1))
    audit_sdaf.concat(far_sdaf)
    assert audit_sdaf.keys()[-1] == 'z9'
    assert not [name for name in _temp_tables(audit_sdaf) if 'more' in name]
    keys_sdaf: SqlDaf = SqlDaf.from_daf(Daf(cols=[audit_daf.keyfield], lol=[[audit_daf.keys()[0]], ['z9']], keyfield=audit_daf.keyfield),
                                        name='keys', db_path=str(tmp_path / 'keys.db'))
    assert audit_sdaf.select_records_daf(keys_sdaf).keys() == [audit_daf.keys()[0], 'z9']


# ---- select_cols ----

SELECT_COLS_ARGS = [
    ('some, reordered', dict(cols=['num_votes', 'ballot_id'])),
    ('one, as a str', dict(cols='contest')),
    ('a name twice', dict(cols=['contest', 'contest', 'ballot_id'])),
    ('exclude', dict(exclude_cols=['contest', 'nope'])),
    ('both', dict(cols=['contest', 'ballot_id', 'num_votes'], exclude_cols=['num_votes'])),
    ('all', dict()),
    ]


@pytest.mark.parametrize('args', [a for _, a in SELECT_COLS_ARGS], ids=[name for name, _ in SELECT_COLS_ARGS])
def test_select_cols_matches_mdaf(args):
    marks_daf: Daf = make_marks_daf()
    marks_sdaf: SqlDaf = SqlDaf.from_daf(marks_daf, name='marks')
    cols_sdaf: SqlDaf = marks_sdaf.select_cols(**args)
    cols_daf: Daf = marks_daf.select_cols(**args)
    assert cols_sdaf.columns() == cols_daf.columns()
    assert cols_sdaf.to_daf().lol == cols_daf.lol
    assert cols_sdaf.keyfield == cols_daf.keyfield


def test_select_cols_of_a_selection_and_of_a_join_keeps_the_order():
    audit_daf, cvr_daf = make_join_pair()
    audit_sdaf: SqlDaf = SqlDaf.from_daf(audit_daf, name='audit')
    cvr_sdaf: SqlDaf = SqlDaf.from_daf(cvr_daf, name='cvr')
    some_sdaf: SqlDaf = audit_sdaf.select_records_daf(['k2', 'k3']).select_cols(['v', 'id'])
    assert some_sdaf.to_daf().lol == [[NULL, 'k2'], [1, 'k3']]
    assert some_sdaf['k3'].to_dict() == {'v': 1, 'id': 'k3'}
    join_sdaf: SqlDaf = audit_sdaf.join(cvr_sdaf, how='left')
    assert join_sdaf.select_cols(['id', 'w']).to_daf().lol == audit_daf.join(cvr_daf, how='left').select_cols(['id', 'w']).lol


def test_select_cols_is_new_rows_and_keeps_dtypes():
    marks_daf: Daf = make_marks_daf()
    marks_sdaf: SqlDaf = SqlDaf('marks', cols=marks_daf.columns(), lol=marks_daf.lol, dtypes={'ballot_id': str, 'num_votes': int})
    cols_sdaf: SqlDaf = marks_sdaf.select_cols(['ballot_id', 'num_votes'])
    assert cols_sdaf.dtypes == {'ballot_id': str, 'num_votes': int}
    assert len(cols_sdaf) == 3
    marks_sdaf.sql('UPDATE marks SET num_votes = 9')
    assert cols_sdaf.to_daf()[:, 'num_votes'].to_list() == [1, NULL, 0]


def test_select_cols_unknown_name_raises_keyerror():
    marks_sdaf: SqlDaf = SqlDaf.from_daf(make_marks_daf(), name='marks')
    with pytest.raises(KeyError):
        marks_sdaf.select_cols(['ballot_id', 'nope'])


# ---- reading: iteration, columns, keys and exports ----

def make_read_daf() -> Daf:
    """ Rows with a Python NULL value, a repeated value and a dict column. """
    return Daf(
        cols=['k', 'p', 'v', 'j'],
        lol=[['b1', 'P1', 1, '{"a": 1}'], ['b2', 'P2', NULL, ''], ['b3', 'P1', 3, '{"a": 3}'],
             ['b4', 'P2', 4, '{}'], ['b5', 'P1', 1, '{"a": 5}']],
        keyfield='k',
    )


def test_iteration_matches_mdaf():
    read_daf: Daf = make_read_daf()
    read_sdaf: SqlDaf = SqlDaf.from_daf(read_daf, name='t')
    assert list(read_sdaf) == list(read_daf)
    assert list(read_sdaf.iter_dict()) == list(read_daf.iter_dict())
    assert [dict(row) for row in read_sdaf.iter_klist()] == [dict(row) for row in read_daf.iter_klist()]
    assert list(read_sdaf.iter_list()) == list(read_daf.iter_list())


def test_iteration_rows_are_copies():
    read_sdaf: SqlDaf = SqlDaf.from_daf(make_read_daf(), name='t')
    for row_kl in read_sdaf.iter_klist():
        row_kl['v'] = 99
    for row_la in read_sdaf.iter_list():
        row_la[2] = 99
    assert 99 not in read_sdaf.col('v')


def test_iteration_streams_more_rows_than_a_batch():
    big_sdaf: SqlDaf = SqlDaf('big', cols=['n'], lol=[[i] for i in range(25000)])
    assert sum(row['n'] for row in big_sdaf) == sum(range(25000))


@pytest.mark.parametrize('kwargs', [
    {}, {'unique': True}, {'omit_nulls': True}, {'astype': str}, {'unique': True, 'omit_nulls': True}])
def test_col_matches_mdaf(kwargs):
    read_daf: Daf = make_read_daf()
    read_sdaf: SqlDaf = SqlDaf.from_daf(read_daf, name='t')
    for colname in ('p', 'v'):
        assert read_sdaf.col(colname, **kwargs) == read_daf.col(colname, **kwargs)
        assert read_sdaf.col_to_la(colname, **kwargs) == read_daf.col_to_la(colname, **kwargs)


def test_col_indirect_and_missing():
    read_daf: Daf = make_read_daf()
    read_sdaf: SqlDaf = SqlDaf.from_daf(read_daf, name='t')
    assert read_sdaf.col('a', indirect_col='j') == read_daf.col('a', indirect_col='j')
    assert read_sdaf.col('zz', silent_error=True) == []
    with pytest.raises(Exception) as sdaf_exc:
        read_sdaf.col('zz')
    with pytest.raises(Exception) as daf_exc:
        read_daf.col('zz')
    assert type(sdaf_exc.value) is type(daf_exc.value)


def test_col_omit_nulls_counts_text_empty_as_python_null(job_db):
    make_plain_table(job_db, [['x', ''], ['y', None], ['z', 'v']])
    assert SqlDaf('plain').col('b', omit_nulls=True) == ['v']


def test_icol_matches_mdaf():
    read_daf: Daf = make_read_daf()
    read_sdaf: SqlDaf = SqlDaf.from_daf(read_daf, name='t')
    assert read_sdaf.icol(1) == read_daf.icol(1)
    assert read_sdaf.icol(-2) == read_daf.icol(-2)
    assert read_sdaf.icol_to_la(1, unique=True) == read_daf.icol_to_la(1, unique=True)
    with pytest.raises(IndexError):
        read_sdaf.icol(9)
    assert SqlDaf('empty', cols=['a']).icol(9) == []


def test_cols_to_dol_shape_and_num_rows_match_mdaf():
    read_daf: Daf = make_read_daf()
    read_sdaf: SqlDaf = SqlDaf.from_daf(read_daf, name='t')
    assert read_sdaf.cols_to_dol('p', 'v') == read_daf.cols_to_dol('p', 'v')
    assert read_sdaf.cols_to_dol('p', 'zz') == {}
    assert read_sdaf.shape() == read_daf.shape() == (5, 4)
    assert read_sdaf.num_rows() == 5
    assert SqlDaf('empty', cols=['a']).shape() == (0, 0)


def test_contains_and_get_existing_keys_match_mdaf():
    read_daf: Daf = make_read_daf()
    read_sdaf: SqlDaf = SqlDaf.from_daf(read_daf, name='t')
    assert ('b3' in read_sdaf, 'zz' in read_sdaf) == ('b3' in read_daf, 'zz' in read_daf) == (True, False)
    assert read_sdaf.get_existing_keys(['b4', 'zz', 'b1']) == read_daf.get_existing_keys(['b4', 'zz', 'b1'])
    p1_sdaf: SqlDaf = read_sdaf.select_by_dict({'p': 'P1'})
    assert 'b2' not in p1_sdaf
    assert p1_sdaf.get_existing_keys(['b2', 'b3']) == ['b3']
    nokey_sdaf: SqlDaf = SqlDaf('nokey', cols=['a'], lol=[[1]])
    assert nokey_sdaf.get_existing_keys([1]) == []
    with pytest.raises(ValueError):
        1 in nokey_sdaf
    assert 1 not in SqlDaf('nokey_empty', cols=['a'])


@pytest.mark.parametrize('krows', [['b3', 'b1'], 'b2', ('b2', 'b4'), ('b3',), (None, 'b2'), ('b2', None)])
@pytest.mark.parametrize('inverse', [False, True])
def test_select_krows_matches_mdaf(krows, inverse):
    read_daf: Daf = make_read_daf()
    read_sdaf: SqlDaf = SqlDaf.from_daf(read_daf, name='t')
    assert read_sdaf.select_krows(krows, inverse=inverse).to_daf().lol == read_daf.select_krows(krows, inverse=inverse).lol


def test_select_krows_errors():
    read_sdaf: SqlDaf = SqlDaf.from_daf(make_read_daf(), name='t')
    with pytest.raises(KeyError):
        read_sdaf.select_krows(['b1', 'zz'])
    assert read_sdaf.select_krows(['b1', 'zz'], silent_error=True).to_daf().lol == [make_read_daf().lol[0]]
    with pytest.raises(KeyError):
        read_sdaf.select_krows(('b1', 'zz'))
    with pytest.raises(NotImplementedError):
        read_sdaf.select_krows(slice(0, 2))
    with pytest.raises(TypeError):
        read_sdaf.select_krows(None)
    with pytest.raises(TypeError):
        read_sdaf.select_krows(('b1', 'b2', 'b3'))


def test_select_first_row_by_dict_matches_mdaf():
    read_daf: Daf = make_read_daf()
    read_sdaf: SqlDaf = SqlDaf.from_daf(read_daf, name='t')
    for selector_da, inverse in (({'p': 'P2'}, False), ({'p': 'P1'}, True), ({'p': 'P9'}, False)):
        assert read_sdaf.select_first_row_by_dict(selector_da, inverse=inverse) == \
            dict(read_daf.select_first_row_by_dict(selector_da, inverse=inverse))


def test_to_daf_name_is_the_table_name_or_none():
    """ A permanent table gives its name. A selection, a copy or a join gives none. A name given is used. """
    read_daf: Daf = make_read_daf()
    read_sdaf: SqlDaf = SqlDaf.from_daf(read_daf, name='t')
    assert read_sdaf.to_daf().name == 't'
    assert read_sdaf.select_krows(['b1']).to_daf().name == ''
    assert read_sdaf.copy().to_daf().name == ''
    assert read_sdaf.join(read_sdaf.copy()).to_daf().name == ''
    assert read_sdaf.select_krows(['b1']).to_daf(name='picked').name == 'picked'
    assert read_sdaf.to_daf(name='').name == ''
    assert read_daf.to_daf(name='mine') is read_daf and read_daf.name == 'mine'


def test_exports_match_mdaf():
    read_daf: Daf = make_read_daf()
    read_sdaf: SqlDaf = SqlDaf.from_daf(read_daf, name='t')
    assert read_sdaf.to_lod() == read_daf.to_lod()
    assert read_sdaf.to_dod() == read_daf.to_dod()
    assert read_sdaf.to_dod(remove_keyfield=False) == read_daf.to_dod(remove_keyfield=False)
    assert Daf.from_json(read_sdaf.to_json()).lol == read_daf.lol
    assert read_sdaf.to_md() == read_daf.to_md()
    assert read_sdaf.to_md_cols() == read_daf.to_md_cols()
    read_daf.name = 't'
    assert read_sdaf.to_md(include_summary=True) == read_daf.to_md(include_summary=True)


@pytest.mark.parametrize('max_rows, max_cols', [(3, 0), (4, 2), (2, 3), (9, 0), (0, 2)])
def test_to_md_with_limits_matches_mdaf(max_rows, max_cols):
    read_daf: Daf = make_read_daf()
    read_sdaf: SqlDaf = SqlDaf.from_daf(read_daf, name='t')
    assert read_sdaf.to_md(max_rows=max_rows, max_cols=max_cols) == read_daf.to_md(max_rows=max_rows, max_cols=max_cols)
    assert read_sdaf.to_md_cols(max_rows=max_rows, max_cols=max_cols) == read_daf.to_md_cols(max_rows=max_rows, max_cols=max_cols)



def test_to_md_disp_cols_dict_matches_mdaf():
    """ A dict renames only the columns it names, as on a Daf. Limits keep it matched. """
    read_daf: Daf = make_read_daf()
    read_sdaf: SqlDaf = SqlDaf.from_daf(read_daf, name='t')
    rename_da: dict = {'v': 'Value'}
    assert read_sdaf.to_md(disp_cols=rename_da) == read_daf.to_md(disp_cols=rename_da)
    assert read_sdaf.to_md(disp_cols=rename_da, max_cols=2) == read_daf.to_md(disp_cols=rename_da, max_cols=2)
    assert read_sdaf.to_md_cols(disp_cols=rename_da) == read_daf.to_md_cols(disp_cols=rename_da)
    assert 'Value' in read_sdaf.to_md(disp_cols=rename_da).splitlines()[0]

def test_reading_by_key_and_name_works_after_a_delete():
    read_daf: Daf = make_read_daf()
    read_sdaf: SqlDaf = SqlDaf.from_daf(read_daf, name='t')
    read_sdaf.delete_records(['b2'], inplace=True)
    read_daf.delete_records(['b2'], inplace=True)
    assert list(read_sdaf) == list(read_daf)
    assert read_sdaf.col('v') == read_daf.col('v')
    assert read_sdaf.icol(0) == read_daf.icol(0)
    assert 'b3' in read_sdaf and 'b2' not in read_sdaf
    assert read_sdaf.select_krows(('b1', 'b4')).to_daf().lol == read_daf.select_krows(('b1', 'b4')).lol
    assert read_sdaf.select_first_row_by_dict({'p': 'P2'}) == dict(read_daf.select_first_row_by_dict({'p': 'P2'}))
    assert read_sdaf.to_md(max_rows=2) == read_daf.to_md(max_rows=2)


# ---- writing: adding rows and changing values ----

def make_write_daf() -> Daf:
    return Daf(cols=['id', 'v', 'n'], lol=[['a', 'x', 1], ['b', 'y', 2], ['c', 'z', 3]], keyfield='id')


WRITE_OPS: list = [
    lambda daf: daf.append({'id': 'd', 'n': 4}),
    lambda daf: daf.append(KeyedList({'id': 0, 'v': 1}, ['d', 'w'])),
    lambda daf: daf.append(['d', 'w']),
    lambda daf: daf.append([{'id': 'd'}, {'id': 'e', 'v': 'q'}]),
    lambda daf: daf.append(lol=[['d', 'w', 4], ['e']]),
    lambda daf: daf.append(la=['d', 'w', 4]),
    lambda daf: daf.append({'id': 'b', 'v': 'B'}, respect_kd=True),
    lambda daf: daf.append(Daf(cols=['id', 'v', 'n'], lol=[['b', 'w', 4], ['d', 'q', 5]], keyfield='id'), respect_kd=True),
    lambda daf: daf.append({}),
    lambda daf: daf.record_append({'id': 'a', 'v': 'A'}),
    lambda daf: daf.record_append({'id': 'd', 'v': 'D'}),
    lambda daf: daf.extend([{'id': 'd'}, {}]),
    lambda daf: daf.extend(lol=[['d', 'w'], ['e', 'q', 9]]),
    lambda daf: daf.assign_record({'id': 'b', 'n': 9}),
    lambda daf: daf.assign_record({'id': 'z', 'n': 9}),
    lambda daf: daf.update_by_keylist(['a', 'zz', 'c'], {'v': 'U', 'bogus': 1}),
    lambda daf: daf.update_record_krow('b', {'v': 'U', 'bogus': 1}),
    lambda daf: daf.update_record_krow('c', {'n': ''}),
    lambda daf: daf.update_record_krow('a', None),
    lambda daf: daf.__setitem__(('b', 'v'), 'Q'),
    lambda daf: daf.__setitem__(('b', 'v'), ['L']),
    lambda daf: daf.__setitem__((slice(None), 'n'), [7, 8, 9]),
    lambda daf: daf.__setitem__((slice(None), 'n'), [7]),
    lambda daf: daf.__setitem__((slice(None), 'n'), [7, 8, 9, 10]),
    lambda daf: daf.__setitem__((slice(None), 'v'), 'xyz'),
    lambda daf: daf.__setitem__('b', {'id': 'b', 'v': 'q'}),
    lambda daf: daf.assign_col('n', [4, 5, 6]),
    lambda daf: daf.assign_col('n', [4]),
    lambda daf: daf.assign_col('n', default=0),
    lambda daf: daf.assign_col('w', [1, 2]),
    lambda daf: daf.apply_to_col('v', str.upper),
    lambda daf: daf.apply_to_col('n', lambda val, k: val * k, k=10),
    lambda daf: daf.apply_in_place(lambda row: {'n': row['n'] + 1, 'bogus': 0}),
    lambda daf: daf.apply_in_place(lambda row: row.__setitem__('v', row['v'] * 2), by='row_klist'),
    lambda daf: daf.apply_in_place(lambda row: {'n': 0}, rowkeys=['a', 'zz']),
]


@pytest.mark.parametrize('op', WRITE_OPS)
def test_writes_match_mdaf(op):
    write_daf: Daf = make_write_daf()
    write_sdaf: SqlDaf = SqlDaf.from_daf(write_daf, name='w')
    op(write_daf)
    op(write_sdaf)
    assert write_sdaf.to_daf().lol == write_daf.lol
    assert write_sdaf.columns() == write_daf.columns()


@pytest.mark.parametrize('op', WRITE_OPS)
def test_writes_match_mdaf_after_a_delete(op):
    write_daf: Daf = make_write_daf()
    write_sdaf: SqlDaf = SqlDaf.from_daf(write_daf, name='w')
    write_daf.append({'id': 'q', 'v': 'gone', 'n': 0})
    write_sdaf.append({'id': 'q', 'v': 'gone', 'n': 0})
    write_daf.delete_records(['a'], inplace=True)
    write_sdaf.delete_records(['a'], inplace=True)
    op(write_daf)
    op(write_sdaf)
    assert write_sdaf.to_daf().lol == write_daf.lol


def test_calc_cols_matches_mdaf():
    write_daf: Daf = make_write_daf()
    write_sdaf: SqlDaf = SqlDaf.from_daf(write_daf, name='w')
    assert write_sdaf.calc_cols(exclude_cols=['v']) == write_daf.calc_cols(exclude_cols=['v']) == ['id', 'n']


def test_writes_reject_repeated_and_python_null_keys():
    write_sdaf: SqlDaf = SqlDaf.from_daf(make_write_daf(), name='w')
    with pytest.raises(sqlite3.IntegrityError):
        write_sdaf.append({'id': 'a'})
    with pytest.raises(sqlite3.IntegrityError):
        write_sdaf.update_by_keylist(['a'], {'id': 'b'})
    with pytest.raises(ValueError, match='Python NULL key'):
        write_sdaf.append({'v': 'no key'})
    with pytest.raises(ValueError, match='Python NULL key'):
        write_sdaf['b'] = {'v': 'q'}                                    # a whole row without its key
    assert write_sdaf.to_daf().lol == make_write_daf().lol


def test_writes_are_all_or_nothing():
    write_sdaf: SqlDaf = SqlDaf.from_daf(make_write_daf(), name='w')
    with pytest.raises(ValueError):
        write_sdaf.extend(lol=[['d'], ['e', 1, 2, 3]])
    with pytest.raises(ZeroDivisionError):
        write_sdaf.apply_in_place(lambda row: {'n': 1 / (row['n'] - 3)})
    with pytest.raises(ZeroDivisionError):
        write_sdaf.apply_to_col('n', lambda val: 1 / (val - 3))
    assert write_sdaf.to_daf().lol == make_write_daf().lol


def test_write_errors_match_mdaf():
    write_sdaf: SqlDaf = SqlDaf.from_daf(make_write_daf(), name='w')
    with pytest.raises(TypeError):
        write_sdaf.append(['d'], la=[1])
    with pytest.raises(TypeError):
        write_sdaf.append(('d', 'w'))
    with pytest.raises(TypeError):
        write_sdaf.extend(lol=[('d',)])
    with pytest.raises(TypeError):
        write_sdaf.record_append(['d'])
    with pytest.raises(KeyError):
        write_sdaf['zz', 'v'] = 1
    with pytest.raises(KeyError):
        write_sdaf['a', 'zz'] = 1
    with pytest.raises(KeyError):
        write_sdaf[:, 'zz'] = 1
    with pytest.raises(KeyError):
        write_sdaf.assign_record({'v': 1})
    with pytest.raises(KeyError):
        write_sdaf.apply_to_col('zz', str)
    with pytest.raises(ValueError):
        write_sdaf.apply_in_place(lambda row: None)
    with pytest.raises(NotImplementedError):
        write_sdaf.apply_in_place(lambda row: row, by='col')
    with pytest.raises(NotImplementedError, match='an int row is a position.*update_record_krow'):
        write_sdaf[0, 'v'] = 1                                          # a position, not a key
    with pytest.raises(NotImplementedError, match='an int row is a position'):
        write_sdaf[2] = {'id': 'c', 'v': 'q'}


def test_update_record_krow_errors_match_mdaf():
    """ A missing key raises KeyError and changes nothing, on both. So does a missing key with no known column. """
    write_daf: Daf = make_write_daf()
    write_sdaf: SqlDaf = SqlDaf.from_daf(write_daf, name='w')
    for record in ({'v': 'q'}, {'bogus': 1}):
        with pytest.raises(KeyError):
            write_daf.update_record_krow('zz', record)
        with pytest.raises(KeyError):
            write_sdaf.update_record_krow('zz', record)
    assert write_sdaf.to_daf().lol == write_daf.lol == make_write_daf().lol
    with pytest.raises(ValueError, match='no keyfield'):
        SqlDaf('nokey', cols=['a'], lol=[[1]]).update_record_krow(1, {'a': 2})


def test_update_record_krow_int_key_and_key_checks():
    ids_sdaf: SqlDaf = SqlDaf('ids', cols=['id', 'v'], lol=[[10, 'a'], [20, 'b'], [0, 'c']], keyfield='id')
    ids_sdaf.update_record_krow(0, {'v': 'q'})                          # 0 is a key, not the first row
    ids_sdaf.update_record_krow(20, {'id': 21})                         # a key may change, and is found by its new value
    assert ids_sdaf.to_daf().lol == [[10, 'a'], [21, 'b'], [0, 'q']]
    assert ids_sdaf.select_record(21) == {'id': 21, 'v': 'b'}
    with pytest.raises(sqlite3.IntegrityError):
        ids_sdaf.update_record_krow(10, {'id': 0})
    with pytest.raises(ValueError, match='Python NULL key'):
        ids_sdaf.update_record_krow(10, {'id': ''})
    assert ids_sdaf.to_daf().lol == [[10, 'a'], [21, 'b'], [0, 'q']]


def test_update_record_krow_through_a_selection():
    write_sdaf: SqlDaf = SqlDaf.from_daf(make_write_daf(), name='w')
    tail_sdaf: SqlDaf = write_sdaf.select_where_sql("id <> 'a'")
    tail_sdaf.update_record_krow('b', {'v': 'B'})
    with pytest.raises(KeyError):
        tail_sdaf.update_record_krow('a', {'v': 'no'})                 # 'a' is in the parent, not the selection
    assert write_sdaf.to_daf().lol == [['a', 'x', 1], ['b', 'B', 2], ['c', 'z', 3]]


def test_writes_through_a_selection_change_the_parent():
    write_sdaf: SqlDaf = SqlDaf.from_daf(make_write_daf(), name='w')
    tail_sdaf: SqlDaf = write_sdaf.select_where_sql("id <> 'a'")
    tail_sdaf['b', 'v'] = 'B'
    tail_sdaf[:, 'n'] = [20, 30]
    tail_sdaf.apply_to_col('v', str.upper)
    tail_sdaf.apply_in_place(lambda row: {'n': row['n'] + 1})
    tail_sdaf.update_by_keylist(['a', 'c'], {'v': 'C'})                # 'a' is not in the selection
    assert write_sdaf.to_daf().lol == [['a', 'x', 1], ['b', 'B', 21], ['c', 'C', 31]]
    with pytest.raises(KeyError):
        tail_sdaf['a', 'v'] = 'no'
    with pytest.raises(NotImplementedError):
        tail_sdaf.assign_col('new', [1])
    with pytest.raises(NotImplementedError):
        tail_sdaf.record_append({'id': 'd'})


def test_append_to_a_selection_leaves_the_parent():
    write_sdaf: SqlDaf = SqlDaf.from_daf(make_write_daf(), name='w')
    tail_sdaf: SqlDaf = write_sdaf.select_where_sql("id <> 'a'")
    tail_sdaf.append({'id': 'd'})
    assert tail_sdaf.keys() == ['b', 'c', 'd']
    assert write_sdaf.keys() == ['a', 'b', 'c']


def test_concat_respect_kd_from_a_sqldaf():
    write_sdaf: SqlDaf = SqlDaf.from_daf(make_write_daf(), name='w')
    other_sdaf: SqlDaf = SqlDaf('other', cols=['id', 'v', 'n'], lol=[['b', 'B', 0], ['d', 'D', 4]], keyfield='id')
    write_sdaf.concat(other_sdaf, respect_kd=True)
    assert write_sdaf.to_daf().lol == [['a', 'x', 1], ['b', 'B', 0], ['c', 'z', 3], ['d', 'D', 4]]


# ---- column names and layout ----

LAYOUT_OPS: list = [
    lambda daf: daf.insert_col('w', ['p', 'q']),
    lambda daf: daf.insert_col('w'),
    lambda daf: daf.insert_col('w', default=5),
    lambda daf: daf.insert_col('w', ['p'], icol=3),
    lambda daf: daf.insert_col('v', ['p'], icol=0),
    lambda daf: daf.insert_col(''),
    lambda daf: daf.drop_cols(['v']),
    lambda daf: daf.drop_cols(['v', 'nothing']),
    lambda daf: daf.drop_cols([]),
    lambda daf: daf.drop_cols(['id']),
    lambda daf: daf.rename_cols({'v': 'vv'}),
    lambda daf: daf.rename_cols({'v': 'n', 'n': 'v'}),
    lambda daf: daf.rename_cols({'id': 'key', 'nothing': 'x'}),
    lambda daf: daf.set_cols(['a', 'b', 'c']),
    lambda daf: daf.set_cols(),
    lambda daf: daf.set_cols(['a', 'a', '']),
    lambda daf: daf.set_dtypes(str, {int: ['n']}),
    ]


@pytest.mark.parametrize('op', LAYOUT_OPS)
@pytest.mark.parametrize('deleted', [False, True])
def test_layout_ops_match_mdaf(op, deleted):
    write_daf: Daf = make_write_daf()
    write_sdaf: SqlDaf = SqlDaf.from_daf(write_daf, name='w')
    if deleted:
        write_daf = write_daf.delete_records(['b'])
        write_sdaf.delete_records(['b'], inplace=True)
    op(write_daf)
    op(write_sdaf)
    assert write_sdaf.columns() == write_daf.columns()
    assert write_sdaf.to_daf().lol == write_daf.lol
    assert write_sdaf.keyfield == write_daf.keyfield
    assert write_sdaf.dtypes == (write_daf.dtypes or {})


def test_layout_changes_are_recorded():
    write_daf: Daf = make_write_daf()
    write_daf.dtypes = {'id': str, 'v': str, 'n': int}
    SqlDaf.from_daf(write_daf, name='w').rename_cols({'id': 'key'}).drop_cols(['v']).set_dtypes(str, {float: ['n']})
    SqlDaf.close()
    again_sdaf: SqlDaf = SqlDaf('w')
    assert again_sdaf.columns() == ['key', 'n']
    assert again_sdaf.keyfield == 'key'
    assert again_sdaf.dtypes == {'key': str, 'n': float}


def test_rename_and_drop_of_the_keyfield_keep_key_lookups_right():
    write_sdaf: SqlDaf = SqlDaf.from_daf(make_write_daf(), name='w')
    write_sdaf.ensure_key_index()
    write_sdaf.rename_cols({'id': 'key'})
    assert key_indexes(write_sdaf.conn) == []
    assert write_sdaf['b', 'v'].to_value() == 'y'
    assert key_indexes(write_sdaf.conn) == ['daf_ix_w_key']
    write_sdaf.drop_cols(['key'])
    assert write_sdaf.keyfield == ''
    assert key_indexes(write_sdaf.conn) == []


def test_layout_errors_change_nothing():
    write_sdaf: SqlDaf = SqlDaf.from_daf(make_write_daf(), name='w')
    with pytest.raises(sqlite3.OperationalError):
        write_sdaf.rename_cols({'v': 'n'})
    with pytest.raises(ValueError):
        write_sdaf.rename_cols({'v': RID_COL})
    with pytest.raises(AttributeError):
        write_sdaf.set_cols(['a'])
    with pytest.raises(NotImplementedError):
        write_sdaf.insert_col('w', icol=0)
    write_sdaf.sql('CREATE INDEX my_ix ON w (v)')
    with pytest.raises(sqlite3.OperationalError):
        write_sdaf.drop_cols(['n', 'v'])
    with pytest.raises(NotImplementedError):
        write_sdaf.set_dtypes(bytes)
    assert write_sdaf.columns() == ['id', 'v', 'n']
    assert write_sdaf.dtypes == {}
    SqlDaf.close()
    assert SqlDaf('w').columns() == ['id', 'v', 'n']


def test_layout_changes_on_a_selection_raise():
    write_sdaf: SqlDaf = SqlDaf.from_daf(make_write_daf(), name='w')
    tail_sdaf: SqlDaf = write_sdaf.select_where_sql("id <> 'a'")
    for op in (lambda sdaf: sdaf.drop_cols(['v']), lambda sdaf: sdaf.rename_cols({'v': 'vv'}),
               lambda sdaf: sdaf.set_cols(['a', 'b', 'c']), lambda sdaf: sdaf.insert_col('w')):
        with pytest.raises(NotImplementedError):
            op(tail_sdaf)
    tail_sdaf.insert_col('v', ['B'])
    assert write_sdaf.col('v') == ['x', 'B', NULL]


def test_a_selection_made_before_a_drop_raises_when_read():
    write_sdaf: SqlDaf = SqlDaf.from_daf(make_write_daf(), name='w')
    tail_sdaf: SqlDaf = write_sdaf.select_where_sql("id <> 'a'")
    len(tail_sdaf)
    write_sdaf.drop_cols(['v'])
    with pytest.raises(sqlite3.OperationalError):
        tail_sdaf.to_daf()


def test_clone_empty_matches_mdaf():
    write_daf: Daf = make_write_daf()
    write_daf.dtypes = {'id': str, 'v': str, 'n': int}
    write_sdaf: SqlDaf = SqlDaf.from_daf(write_daf, name='w')
    for kwargs in ({}, {'lol': [['d', 'w', 4]]}, {'cols': ['v', 'n']}, {'cols': ['id', 'id']}):
        clone_daf: Daf = write_daf.clone_empty(**kwargs)
        clone_sdaf: SqlDaf = write_sdaf.clone_empty(**kwargs)
        assert clone_sdaf.columns() == clone_daf.columns()
        assert clone_sdaf.to_daf().lol == clone_daf.lol
        assert clone_sdaf.keyfield == clone_daf.keyfield
        assert clone_sdaf.dtypes == clone_daf.dtypes
        assert clone_sdaf.is_temp and clone_sdaf.name.startswith('daf_tmp_w_v')
    assert write_sdaf.select_where_sql("id <> 'a'").clone_empty(name='blank').name == 'blank'
    assert len(write_sdaf) == 3


def make_dups_daf() -> Daf:
    """ Repeated keys, with SQL NULL and '' as one Python NULL key. """
    return Daf(cols=['id', 'v'], lol=[[1, 'a'], [2, 'b'], [1, 'c'], [NULL, 'd'], [3, 'e'], [NULL, 'f'], [2, 'g']])


@pytest.mark.parametrize('deleted', [False, True])
def test_remove_dups_matches_mdaf(deleted):
    dups_daf: Daf = make_dups_daf()
    dups_sdaf: SqlDaf = SqlDaf.from_daf(dups_daf, name='d')
    dups_sdaf.sql("UPDATE d SET id = '' WHERE v = 'f'")
    if deleted:
        dups_daf = dups_daf.select_where(lambda row: row['v'] != 'e')
        dups_sdaf.sql("DELETE FROM d WHERE v = 'e'")
    unique_daf, repeat_daf = dups_daf.remove_dups('id')
    unique_sdaf, repeat_sdaf = dups_sdaf.remove_dups('id')
    assert unique_sdaf.to_daf().lol == unique_daf.lol
    assert repeat_sdaf.to_daf().lol == repeat_daf.lol
    assert unique_sdaf.keyfield == repeat_sdaf.keyfield == 'id'
    assert key_indexes(dups_sdaf.conn) == []


def test_remove_dups_on_a_selection_and_by_keyfield():
    dups_daf: Daf = make_dups_daf()
    dups_sdaf: SqlDaf = SqlDaf.from_daf(dups_daf, name='d')
    sel_sdaf: SqlDaf = dups_sdaf.select_where_sql("v <> 'a'")
    unique_daf, repeat_daf = dups_daf.select_where(lambda row: row['v'] != 'a').remove_dups('id')
    unique_sdaf, repeat_sdaf = sel_sdaf.remove_dups('id')
    assert unique_sdaf.to_daf().lol == unique_daf.lol
    assert repeat_sdaf.to_daf().lol == repeat_daf.lol
    write_sdaf: SqlDaf = SqlDaf.from_daf(make_write_daf(), name='w')
    assert write_sdaf.remove_dups()[0].to_daf().lol == make_write_daf().lol
    assert len(write_sdaf.remove_dups()[1]) == 0


def test_remove_dups_errors():
    dups_sdaf: SqlDaf = SqlDaf.from_daf(make_dups_daf(), name='d')
    with pytest.raises(LookupError):
        dups_sdaf.remove_dups()
    with pytest.raises(KeyError):
        dups_sdaf.remove_dups('nothing')
    with pytest.raises(NotImplementedError):
        dups_sdaf.remove_dups(('id', 'v'))


def test_attach_schema_matches_mdaf():
    from daffodil.lib.schemaclass import schemaclass

    @schemaclass
    class Row:
        __keyfield__ = 'id'
        id: str = ''
        v: list = []
        n: int = 0

    write_daf: Daf = Daf(cols=['id', 'v', 'n'], lol=[['a', 'x', 1]])
    write_sdaf: SqlDaf = SqlDaf.from_daf(write_daf, name='w')
    write_daf.attach_schema(Row)
    write_sdaf.attach_schema(Row)
    assert write_sdaf.dtypes == write_daf.dtypes == {'id': str, 'v': list, 'n': int}
    assert write_sdaf.keyfield == write_daf.keyfield == 'id'
    assert write_sdaf.schema is Row
    SqlDaf.close()
    assert SqlDaf('w').dtypes == {'id': str, 'v': list, 'n': int}


def test_attach_schema_keeps_what_is_set():
    from daffodil.lib.schemaclass import schemaclass

    @schemaclass
    class Row:
        __keyfield__ = 'nothing'
        id: str = ''

    write_sdaf: SqlDaf = SqlDaf('w', cols=['id', 'v'], lol=[['a', 'x']], keyfield='id', dtypes={'id': str, 'v': str})
    assert write_sdaf.attach_schema(Row).keyfield == 'id'
    assert write_sdaf.dtypes == {'id': str, 'v': str}
    with pytest.raises(KeyError):
        SqlDaf('u', cols=['id'], lol=[['a']]).attach_schema(Row)
    with pytest.raises(TypeError):
        write_sdaf.attach_schema(dict)



def group_indexes(conn: sqlite3.Connection) -> T_ls:
    """ The indexes groupby() made, in main and TEMP. """
    return [row_t[0] for row_t in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND name LIKE 'daf_gx_%' "
            "UNION ALL SELECT name FROM sqlite_temp_master WHERE type = 'index' AND name LIKE 'daf_gx_%'")]


def test_groupby_makes_an_index_on_the_parent_once():
    bif_daf: Daf = make_bif_daf()
    bif_sdaf: SqlDaf = SqlDaf.from_daf(bif_daf, name='bif')
    assert {key: g.to_daf().lol for key, g in bif_sdaf.groupby('precinct').items()} == \
           {key: g.lol for key, g in bif_daf.groupby('precinct').items()}
    assert group_indexes(bif_sdaf.conn) == ['daf_gx_bif_precinct']
    bif_sdaf.select_where_sql('style_num > 1').groupby('precinct')
    bif_sdaf.groupby_cols(['precinct', 'style_num'])
    assert group_indexes(bif_sdaf.conn) == ['daf_gx_bif_precinct', 'daf_gx_bif_precinct_style_num']
    bif_sdaf.groupby('ballot_id')
    bif_sdaf.ensure_key_index()
    bif_sdaf.groupby('ballot_id')
    assert 'daf_gx_bif_ballot_id' in group_indexes(bif_sdaf.conn)


def test_groupby_uses_an_index_already_there():
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    bif_sdaf.ensure_key_index()
    bif_sdaf.groupby('ballot_id')
    bif_sdaf.sql('CREATE INDEX my_ix ON bif (style_num, precinct)')
    bif_sdaf.groupby('style_num')
    assert group_indexes(bif_sdaf.conn) == []


def test_groupby_index_on_a_temp_table():
    copy_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif').copy()
    copy_sdaf.groupby('precinct')
    assert group_indexes(copy_sdaf.conn) == [f'daf_gx_{copy_sdaf.name}_precinct']


def test_drop_and_rename_of_a_grouped_column():
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    bif_sdaf.groupby('precinct')
    bif_sdaf.groupby('style_num')
    bif_sdaf.rename_cols({'precinct': 'pct'})
    assert group_indexes(bif_sdaf.conn) == ['daf_gx_bif_style_num']
    bif_sdaf.drop_cols(['style_num'])
    assert group_indexes(bif_sdaf.conn) == []
    assert list(bif_sdaf.groupby('pct').keys()) == ['P1', 'P2', 'P3']



def make_sort_daf() -> Daf:
    """ Text that sorts differently by length, a Python NULL, and ties to keep in order. """
    return Daf(cols=['k', 't', 'n'],
               lol=[['a', '10', 3], ['b', '8', 1], ['c', NULL, 2], ['d', '100', 1], ['e', '8', 2], ['f', 'x', 3]],
               keyfield='k')


SORT_OPS: list = [
    lambda daf: daf.sort_by_colname('t'),
    lambda daf: daf.sort_by_colname('t', reverse=True),
    lambda daf: daf.sort_by_colname('t', length_priority=True),
    lambda daf: daf.sort_by_colname('t', length_priority=True, reverse=True),
    lambda daf: daf.sort_by_colname('n'),
    lambda daf: daf.sort_by_colname('n', reverse=True),
    lambda daf: daf.sort_by_colname('n', as_str=True),
    lambda daf: daf.sort_by_colname('n', as_str=True, length_priority=True),
    lambda daf: daf.sort_by_colnames(['n', 't']),
    lambda daf: daf.sort_by_colnames(['n', 't'], reverse=True),
    lambda daf: daf.sort_by_colnames(['n', 'k'], length_priority=True, as_str=True),
    ]


@pytest.mark.parametrize('op', SORT_OPS)
@pytest.mark.parametrize('deleted', [False, True])
def test_sorts_match_mdaf(op, deleted):
    sort_daf: Daf = make_sort_daf()
    sort_sdaf: SqlDaf = SqlDaf.from_daf(sort_daf, name='s')
    if deleted:
        sort_daf = sort_daf.delete_records(['b'])
        sort_sdaf.delete_records(['b'], inplace=True)
    assert op(sort_sdaf) is sort_sdaf
    op(sort_daf)
    assert sort_sdaf.to_daf().lol == sort_daf.lol
    assert sort_sdaf.keyfield == 'k'


def test_sort_of_a_selection_leaves_the_parent():
    sort_daf: Daf = make_sort_daf()
    sort_sdaf: SqlDaf = SqlDaf.from_daf(sort_daf, name='s')
    sel_sdaf: SqlDaf = sort_sdaf.select_where_sql("k <> 'a'")
    sel_sdaf.sort_by_colname('n')
    assert sel_sdaf.keys() == ['b', 'd', 'c', 'e', 'f']
    assert sel_sdaf[0, 'k'].to_value() == 'b'
    assert sort_sdaf.keys() == ['a', 'b', 'c', 'd', 'e', 'f']
    sel_sdaf['b', 'n'] = 9
    assert sort_sdaf['b', 'n'].to_value() == 1


def test_sort_of_a_permanent_table_keeps_the_file_table():
    sort_sdaf: SqlDaf = SqlDaf.from_daf(make_sort_daf(), name='s')
    sort_sdaf.sort_by_colname('n')
    assert sort_sdaf.is_temp and sort_sdaf.name != 's'
    assert SqlDaf('s').keys() == ['a', 'b', 'c', 'd', 'e', 'f']


def test_sort_treats_sql_null_and_empty_text_alike():
    sort_sdaf: SqlDaf = SqlDaf('s', cols=['k', 'n'], lol=[['a', 2], ['b', NULL], ['c', 1]], keyfield='k')
    sort_sdaf.sql("INSERT INTO s (k, n) VALUES ('d', '')")
    assert sort_sdaf.sort_by_colname('n').keys() == ['b', 'd', 'c', 'a']


def test_sort_errors_and_mixed_types():
    sort_sdaf: SqlDaf = SqlDaf('s', cols=['k', 'n'], lol=[['a', 'x'], ['b', 2], ['c', 1]], keyfield='k')
    with pytest.raises(KeyError):
        sort_sdaf.sort_by_colname('nothing')
    assert sort_sdaf.sort_by_colname('n').col('n') == [1, 2, 'x']


# ---- add-on namespaces: register_sqldaf_namespace() ----

from daffodil.daf import register_daf_namespace
from daffodil.sqldaf import register_sqldaf_namespace


class _DafTotals:
    """ A Daf add-on, which reads the rows in memory. """
    def __init__(self, daf):
        self._daf = daf
    def of(self, col):
        return sum(self._daf.col(col))


class _SqlTotals:
    """ The SqlDaf version of the same add-on, which sums in SQLite. """
    def __init__(self, sdaf):
        self._sdaf = sdaf
    def of(self, col):
        return self._sdaf.sum([col])[col]


@pytest.fixture
def daf_totals():
    register_daf_namespace('totals')(_DafTotals)
    yield
    del Daf.totals


@pytest.fixture
def sql_totals():
    register_sqldaf_namespace('totals')(_SqlTotals)
    yield
    del SqlDaf.totals


def make_qty_sdaf() -> SqlDaf:
    return SqlDaf('items', cols=['id', 'qty'], lol=[['a1', 3], ['b2', 10], ['c3', 7]],
                  keyfield='id', dtypes={'id': str, 'qty': int})


def test_daf_addon_raises_not_implemented_on_a_sqldaf(daf_totals):
    sdaf = make_qty_sdaf()
    with pytest.raises(NotImplementedError, match="registered for Daf, not for SqlDaf"):
        sdaf.totals
    assert Daf(cols=['qty'], lol=[[3]]).totals.of('qty') == 3


def test_sqldaf_addon_is_bound_to_each_sqldaf_and_its_selections(sql_totals):
    sdaf = make_qty_sdaf()
    assert type(sdaf.totals) is _SqlTotals
    assert sdaf.totals.of('qty') == 20
    assert sdaf[0:2].totals.of('qty') == 13
    assert sdaf.select_where_sql('qty > ?', [5]).totals.of('qty') == 17
    assert 'totals' not in vars(sdaf)


def test_sqldaf_addon_is_not_seen_on_a_daf(sql_totals):
    assert not hasattr(Daf(cols=['qty'], lol=[[3]]), 'totals')


def test_daf_and_sqldaf_versions_under_one_name_each_reach_their_own_class(daf_totals, sql_totals):
    daf = Daf(cols=['id', 'qty'], lol=[['a1', 3], ['b2', 10]], keyfield='id')
    sdaf = make_qty_sdaf()
    assert type(daf.totals) is _DafTotals and type(daf[0:1].totals) is _DafTotals
    assert type(sdaf.totals) is _SqlTotals and type(sdaf[0:1].totals) is _SqlTotals
    assert (daf.totals.of('qty'), sdaf.totals.of('qty')) == (13, 20)


def test_register_sqldaf_namespace_raises_for_a_second_class_under_the_same_name(sql_totals):
    with pytest.raises(ValueError, match='already registered'):
        register_sqldaf_namespace('totals')(_DafTotals)


@pytest.mark.parametrize('name', ['select_where_sql', 'conn', 'pending_sql', 'is_resolved', 'db_path', 'to_daf'])
def test_register_sqldaf_namespace_raises_for_a_name_in_use(name):
    with pytest.raises(ValueError):
        register_sqldaf_namespace(name)(_SqlTotals)


def test_sqldaf_indirect_col_alone_means_sparse_rows():
    ballots_daf = Daf(cols=['id', 'precinct', 'votes'], lol=[
        ['b1', 'P1', {'mayor': 1}], ['b2', 'P1', {'council': 1}], ['b3', 'P2', {'mayor': 1, 'council': 1}]])
    ballots_sdaf = SqlDaf.from_daf(ballots_daf, name='ballots')
    assert ballots_sdaf.reduce(Daf.sum_da, indirect_col='votes') == {'mayor': 2, 'council': 2}
    want = [['P1', '{"mayor":1,"council":1}'], ['P2', '{"mayor":1,"council":1}']]
    assert ballots_sdaf.groupby_reduce('precinct', Daf.sum_da, indirect_col='votes').to_daf().lol == want
    assert ballots_sdaf.groupsum_daf('precinct', indirect_col='votes').lol == want
    with pytest.raises(ValueError, match='needs indirect_col'):
        ballots_sdaf.groupsum_daf('precinct', by='sparse_row')
    with pytest.raises(NotImplementedError):
        ballots_sdaf.groupsum_daf('precinct', by='table')
