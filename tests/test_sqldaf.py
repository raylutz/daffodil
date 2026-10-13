# test_sqldaf.py
# Tests for SqlStore and SqlDaf in sqldaf.py, step 1: the store, loading, reading and the blocker.

import os
import sqlite3

import pytest

from daffodil.daf import Daf, NULL
from daffodil.sqldaf import SqlStore, SqlDaf, SqlNotSupportedError, BLOCKED_LS, RID_COL, quote_name


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


# SqlStore: opening and closing

def test_store_without_path_uses_a_temp_file_and_deletes_it():
    store: SqlStore = SqlStore()
    temp_path: str = store.path
    assert os.path.exists(temp_path)
    store.close()
    assert not os.path.exists(temp_path)


def test_store_with_path_keeps_the_file_and_its_tables(tmp_path):
    db_path: str = str(tmp_path / 'kept.db')
    with SqlStore(db_path) as store:
        store.from_daf(make_marks_daf(), name='marks')
    assert os.path.exists(db_path)
    with SqlStore(db_path) as store:
        assert len(store.table('marks')) == 3


def test_store_with_conn_leaves_it_open():
    conn: sqlite3.Connection = sqlite3.connect(':memory:')
    with SqlStore(conn=conn) as store:
        store.from_daf(make_marks_daf(), name='marks')
    assert conn.execute('SELECT COUNT(*) FROM marks').fetchone()[0] == 3


def test_store_rejects_path_and_conn_together():
    with pytest.raises(ValueError):
        SqlStore('x.db', conn=sqlite3.connect(':memory:'))


def test_store_close_twice_is_harmless():
    store: SqlStore = SqlStore()
    store.close()
    store.close()


def test_store_next_temp_name_counts_up():
    with SqlStore() as store:
        assert store.next_temp_name('marks') == 'daf_tmp_marks_v1'
        assert store.next_temp_name('marks') == 'daf_tmp_marks_v2'


# SqlStore.create_table

def test_create_table_has_hidden_rid_and_declared_types():
    with SqlStore() as store:
        marks_sdaf: SqlDaf = store.create_table('marks', ['ballot_id', 'n', 'f', 'x'], dtypes={'n': int, 'f': float, 'x': list})
        info_lot = store.conn.execute('PRAGMA table_info(marks)').fetchall()
        assert [(info_t[1], info_t[2], info_t[5]) for info_t in info_lot] == [
            (RID_COL, 'INTEGER', 1), ('ballot_id', '', 0), ('n', 'INTEGER', 0), ('f', 'REAL', 0), ('x', '', 0)]
        assert marks_sdaf.columns() == ['ballot_id', 'n', 'f', 'x']


def test_create_table_rejects_an_existing_name():
    with SqlStore() as store:
        store.create_table('marks', ['a'])
        with pytest.raises(ValueError):
            store.create_table('marks', ['a'])


def test_create_table_rejects_a_column_named_rid():
    with SqlStore() as store:
        with pytest.raises(ValueError):
            store.create_table('marks', [RID_COL])


def test_create_table_keyfield_must_be_a_column():
    with SqlStore() as store:
        with pytest.raises(KeyError):
            store.create_table('marks', ['a'], keyfield='b')


# SqlStore.from_csv

def test_from_csv_loads_rows_in_order_with_null(tmp_path):
    csv_path: str = write_csv(str(tmp_path / 'marks.csv'), 'ballot_id,contest,num_votes\nb1,c0,1\nb2,c1,\nb3,c0,0\n')
    with SqlStore() as store:
        marks_sdaf: SqlDaf = store.from_csv(csv_path, keyfield='ballot_id')
        assert marks_sdaf.name == 'marks'
        marks_daf: Daf = marks_sdaf.to_daf()
        assert marks_daf.lol == [['b1', 'c0', '1'], ['b2', 'c1', NULL], ['b3', 'c0', '0']]
        assert marks_daf.lol[1][2] is NULL
        assert store.conn.execute('SELECT COUNT(*) FROM marks WHERE num_votes IS NULL').fetchone()[0] == 1


def test_from_csv_with_dtypes_stores_numbers(tmp_path):
    csv_path: str = write_csv(str(tmp_path / 'm.csv'), 'k,n,f\na,1,1.5\nb,,\nc,abc,2\n')
    with SqlStore() as store:
        m_sdaf: SqlDaf = store.from_csv(csv_path, dtypes={'n': int, 'f': float})
        assert m_sdaf.to_daf().lol == [['a', 1, 1.5], ['b', NULL, NULL], ['c', 'abc', 2.0]]
        # 'abc' does not fit INTEGER, so it is kept as text: COUNT counts it, and SUM turns real.
        assert m_sdaf.sql('SELECT SUM(n) AS s, COUNT(n) AS c FROM m').lol == [[1.0, 2]]


def test_from_csv_pads_short_rows_and_rejects_long_ones(tmp_path):
    short_path: str = write_csv(str(tmp_path / 'short.csv'), 'a,b,c\n1\n')
    long_path: str = write_csv(str(tmp_path / 'long.csv'), 'a,b\n1,2,3\n')
    with SqlStore() as store:
        assert store.from_csv(short_path).to_daf().lol == [['1', NULL, NULL]]
        with pytest.raises(ValueError):
            store.from_csv(long_path)
        assert not store.table_exists('long')


def test_from_csv_duplicate_key_raises_and_makes_no_table(tmp_path):
    csv_path: str = write_csv(str(tmp_path / 'dup.csv'), 'ballot_id,v\nb1,1\nb1,2\n')
    with SqlStore() as store:
        with pytest.raises(sqlite3.IntegrityError):
            store.from_csv(csv_path, keyfield='ballot_id')
        assert not store.table_exists('dup')
        assert store.conn.in_transaction is False


def test_from_csv_makes_repeated_names_unique(tmp_path):
    csv_path: str = write_csv(str(tmp_path / 'rep.csv'), 'a,a,\n1,2,3\n')
    with SqlStore() as store:
        assert store.from_csv(csv_path).columns() == Daf(cols=['a', 'a', '']).columns()


def test_from_csv_name_with_quote_and_space(tmp_path):
    csv_path: str = write_csv(str(tmp_path / 'q.csv'), 'my "col",b\n1,2\n')
    with SqlStore() as store:
        q_sdaf: SqlDaf = store.from_csv(csv_path, name='odd name')
        assert q_sdaf.columns() == ['my "col"', 'b']
        col_sql: str = quote_name('my "col"')
        assert col_sql == '"my ""col"""'
        assert q_sdaf.sql(f'SELECT {col_sql} FROM {quote_name(q_sdaf.name)}').lol == [['1']]


# SqlStore.from_daf, table, drop_table

def test_from_daf_keeps_value_types_without_dtypes():
    with SqlStore() as store:
        mixed_sdaf: SqlDaf = store.from_daf(Daf(lol=[['a', 1, 1.5, True], ['b', '2', NULL, False]], cols=['k', 'n', 'f', 'b']), name='mixed')
        assert mixed_sdaf.to_daf().lol == [['a', 1, 1.5, 1], ['b', '2', NULL, 0]]


def test_from_daf_list_cell_raises():
    with SqlStore() as store:
        with pytest.raises(sqlite3.Error):
            store.from_daf(Daf(lol=[['a', [1, 2]]], cols=['k', 'x']), name='lists')


def test_from_daf_round_trips_with_keyfield():
    with SqlStore() as store:
        marks_sdaf: SqlDaf = store.from_daf(make_marks_daf(), name='marks')
        assert marks_sdaf.keyfield == 'ballot_id'
        marks_daf: Daf = marks_sdaf.to_daf()
        assert marks_daf.lol == make_marks_daf().lol
        assert marks_daf.keyfield == 'ballot_id'
        assert marks_daf['b2', 'contest'].to_value() == 'c1'


def test_table_missing_raises_keyerror():
    with SqlStore() as store:
        with pytest.raises(KeyError):
            store.table('nothing')


def test_table_on_a_table_without_rid_works():
    conn: sqlite3.Connection = sqlite3.connect(':memory:')
    conn.execute('CREATE TABLE plain (a TEXT, b INTEGER)')
    conn.executemany('INSERT INTO plain VALUES (?, ?)', [('x', 1), ('y', None)])
    with SqlStore(conn=conn) as store:
        plain_sdaf: SqlDaf = store.table('plain')
        assert plain_sdaf.has_rid is False
        assert plain_sdaf.to_daf().lol == [['x', 1], ['y', NULL]]


def test_drop_table():
    with SqlStore() as store:
        store.from_daf(make_marks_daf(), name='marks')
        store.drop_table('marks')
        assert not store.table_exists('marks')
        store.drop_table('marks', missing_ok=True)
        with pytest.raises(KeyError):
            store.drop_table('marks')


# SqlDaf: reading

def test_len_bool_columns_num_cols():
    with SqlStore() as store:
        marks_sdaf: SqlDaf = store.from_daf(make_marks_daf(), name='marks')
        empty_sdaf: SqlDaf = store.create_table('empty', ['a'])
        assert len(marks_sdaf) == 3
        assert bool(marks_sdaf) is True
        assert bool(empty_sdaf) is False
        assert marks_sdaf.columns() == ['ballot_id', 'contest', 'num_votes']
        assert marks_sdaf.num_cols() == 3


def test_isinstance_daf_and_repr():
    with SqlStore() as store:
        marks_sdaf: SqlDaf = store.from_daf(make_marks_daf(), name='marks')
        assert isinstance(marks_sdaf, Daf)
        assert repr(marks_sdaf) == "<SqlDaf name='marks' rows=3 cols=3 keyfield='ballot_id'>"
        assert str(marks_sdaf) == repr(marks_sdaf)


def test_name_is_read_only():
    with SqlStore() as store:
        marks_sdaf: SqlDaf = store.from_daf(make_marks_daf(), name='marks')
        with pytest.raises(AttributeError):
            marks_sdaf.name = 'other'                       # type: ignore[misc]


def test_to_daf_is_not_tied_to_the_table():
    with SqlStore() as store:
        marks_sdaf: SqlDaf = store.from_daf(make_marks_daf(), name='marks')
        marks_daf: Daf = marks_sdaf.to_daf()
        marks_daf['b1', 'num_votes'] = 99
        assert marks_sdaf.to_daf()['b1', 'num_votes'].to_value() == 1


# SqlDaf.sql

def test_sql_select_gives_a_daf_with_null():
    with SqlStore() as store:
        marks_sdaf: SqlDaf = store.from_daf(make_marks_daf(), name='marks')
        result_daf: Daf = marks_sdaf.sql('SELECT contest, SUM(num_votes) AS v, COUNT(num_votes) AS c '
                                         'FROM marks GROUP BY contest ORDER BY contest')
        assert result_daf.columns() == ['contest', 'v', 'c']
        assert result_daf.lol == [['c0', 1, 2], ['c1', NULL, 0]]
        assert result_daf.lol[1][1] is NULL


def test_sql_params_turn_null_into_sql_null():
    with SqlStore() as store:
        marks_sdaf: SqlDaf = store.from_daf(make_marks_daf(), name='marks')
        assert marks_sdaf.sql('SELECT ballot_id FROM marks WHERE contest = ?', ['c0']).to_list() == ['b1', 'b3']
        assert marks_sdaf.sql('SELECT ? IS NULL', [NULL]).lol == [[1]]


def test_sql_write_is_committed_and_gives_an_empty_daf():
    with SqlStore() as store:
        marks_sdaf: SqlDaf = store.from_daf(make_marks_daf(), name='marks')
        result_daf: Daf = marks_sdaf.sql("UPDATE marks SET num_votes = 5 WHERE ballot_id = 'b2'")
        assert result_daf.lol == [] and result_daf.columns() == []
        assert marks_sdaf.to_daf()['b2', 'num_votes'].to_value() == 5
        assert store.conn.in_transaction is False


# the blocker

@pytest.mark.parametrize('call', [
    lambda sdaf: sdaf[0],
    lambda sdaf: list(sdaf),
    lambda sdaf: sdaf.select_irows([0]),
    lambda sdaf: sdaf.groupby('contest'),
    lambda sdaf: sdaf.to_lod(),
    lambda sdaf: sdaf == sdaf,
    lambda sdaf: sdaf.retmode,
    lambda sdaf: sdaf.from_lod([]),
    ])
def test_unsupported_daf_methods_raise(call):
    with SqlStore() as store:
        marks_sdaf: SqlDaf = store.from_daf(make_marks_daf(), name='marks')
        with pytest.raises(SqlNotSupportedError):
            call(marks_sdaf)


def test_blocker_message_names_the_method():
    with SqlStore() as store:
        marks_sdaf: SqlDaf = store.from_daf(make_marks_daf(), name='marks')
        with pytest.raises(SqlNotSupportedError, match=r'^select_irows\(\) is not supported on a SqlDaf'):
            marks_sdaf.select_irows([0])


def test_blocker_covers_every_daf_method_not_defined():
    defined_ls = [name for name in vars(SqlDaf) if name not in BLOCKED_LS]
    for name in ('to_daf', 'sql', '__len__', '__bool__', 'num_cols', '__repr__'):
        assert name in defined_ls
    assert 'columns' not in vars(SqlDaf)
    assert len(BLOCKED_LS) > 200


def test_plain_daf_is_unchanged_by_the_blocker():
    marks_daf: Daf = make_marks_daf()
    assert marks_daf.select_irows([0]).lol == [['b1', 'c0', 1]]
    assert marks_daf.retmode == 'obj'


def test_insert_rows_into_keyed_table_duplicate_adds_none():
    with SqlStore() as store:
        marks_sdaf: SqlDaf = store.create_table('marks', ['ballot_id', 'v'], keyfield='ballot_id')
        assert store.insert_rows('marks', ['ballot_id', 'v'], [['b1', 1], ['b2', NULL]]) == 2
        with pytest.raises(sqlite3.IntegrityError):
            store.insert_rows('marks', ['ballot_id', 'v'], [['b3', 3], ['b1', 4]])
        assert marks_sdaf.to_daf().lol == [['b1', 1], ['b2', NULL]]
