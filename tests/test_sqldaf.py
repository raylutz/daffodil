# test_sqldaf.py
# Tests for SqlDaf in sqldaf.py: making and opening tables, loading, reading, selecting, keys, joins and the blocker.

import os
import sqlite3
import threading

import pytest

from daffodil.daf import Daf, NULL
from daffodil.keyedlist import KeyedList
from daffodil.lib.daf_types import T_ls
from daffodil.sqldaf import SqlDaf, BLOCKED_LS, RID_COL, META_TABLE, quote_name


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


def test_from_daf_list_cell_raises():
    with pytest.raises(sqlite3.Error):
        SqlDaf.from_daf(Daf(lol=[['a', [1, 2]]], cols=['k', 'x']), name='lists')


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


def test_isinstance_daf_and_repr():
    marks_sdaf: SqlDaf = SqlDaf.from_daf(make_marks_daf(), name='marks')
    assert isinstance(marks_sdaf, Daf)
    assert repr(marks_sdaf) == "<SqlDaf table name='marks' rows=3 cols=3 keyfield='ballot_id'>"
    assert str(marks_sdaf) == repr(marks_sdaf)


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
    lambda sdaf: sdaf[0, 'contest'],
    lambda sdaf: list(sdaf),
    lambda sdaf: sdaf.select_irows([0]),
    lambda sdaf: sdaf.multi_groupby(['contest']),
    lambda sdaf: sdaf.to_lod(),
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
    assert len(BLOCKED_LS) > 190


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
        SqlDaf('marks', ['ballot_id', 'v'], lol=[['b1', 1], ['b2', [2]]])
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


def test_apply_dtypes_list_is_not_supported_yet():
    t_sdaf: SqlDaf = SqlDaf.from_daf(Daf(lol=[['[1]']], cols=['a']), name='t')
    with pytest.raises(NotImplementedError):
        t_sdaf.apply_dtypes(dtypes={'a': list})


def test_apply_dtypes_uses_stored_dtypes():
    t_sdaf: SqlDaf = SqlDaf.from_daf(Daf(lol=[['1']], cols=['a'], dtypes={'a': int}), name='t')
    t_sdaf.apply_dtypes()
    assert t_sdaf.to_daf().lol == [[1]]


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
    assert repr(sel_sdaf).startswith("<SqlDaf child name='daf_tmp_bif_v")


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
        bif_sdaf[0, 'precinct']
    nokey_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='nokey', keyfield='').set_keyfield('')
    with pytest.raises(NotImplementedError):
        nokey_sdaf['b1']


def test_getitem_by_key():
    bif_daf: Daf = make_bif_daf()
    bif_sdaf: SqlDaf = SqlDaf.from_daf(bif_daf, name='bif')
    assert bif_sdaf['b3'].to_daf().lol == bif_daf['b3'].lol
    assert bif_sdaf['b3'].to_dict() == bif_daf['b3'].to_dict()
    assert 'daf_ix_bif_ballot_id' in key_indexes(bif_sdaf.conn)


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


def test_apply_dtypes_on_a_selection_raises():
    bif_sdaf: SqlDaf = SqlDaf.from_daf(make_bif_daf(), name='bif')
    with pytest.raises(NotImplementedError):
        bif_sdaf.select_by_dict({'precinct': 'P1'}).apply_dtypes(dtypes={'style_num': int})


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
    with pytest.raises(KeyError):
        bif_sdaf.concat(Daf(lol=[['b9', 'P1']], cols=['ballot_id', 'precinct']))
    with pytest.raises(NotImplementedError):
        bif_sdaf.concat(make_bif_daf(), respect_kd=True)


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


@pytest.mark.parametrize('how', ['inner', 'left'])
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
    with pytest.raises(NotImplementedError, match='child or a TEMP table'):
        audit_sdaf.join(some_sdaf)


def test_join_with_a_table_in_another_file_raises(tmp_path):
    audit_daf, cvr_daf = make_join_pair()
    audit_sdaf: SqlDaf = SqlDaf.from_daf(audit_daf, name='audit')
    cvr_sdaf: SqlDaf = SqlDaf.from_daf(cvr_daf, name='cvr', db_path=str(tmp_path / 'other.db'))
    with pytest.raises(NotImplementedError, match='another database file'):
        audit_sdaf.join(cvr_sdaf)
