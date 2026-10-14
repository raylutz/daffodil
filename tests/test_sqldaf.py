# test_sqldaf.py
# Tests for SqlStore and SqlDaf in sqldaf.py, step 1: the store, loading, reading and the blocker.

import os
import sqlite3

import pytest

from daffodil.daf import Daf, NULL
from daffodil.lib.daf_types import T_ls
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

def test_create_table_has_hidden_rid_and_untyped_columns():
    with SqlStore() as store:
        marks_sdaf: SqlDaf = store.create_table('marks', ['ballot_id', 'n'], keyfield='ballot_id', dtypes={'n': int})
        info_lot = store.conn.execute('PRAGMA table_info(marks)').fetchall()
        assert [(info_t[1], info_t[2], info_t[5]) for info_t in info_lot] == [
            (RID_COL, 'INTEGER', 1), ('ballot_id', '', 0), ('n', '', 0)]
        assert marks_sdaf.columns() == ['ballot_id', 'n']
        assert marks_sdaf.dtypes == {'n': int}
        assert store.conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type = 'index'").fetchone()[0] == 0


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


def test_from_csv_reads_text_only(tmp_path):
    csv_path: str = write_csv(str(tmp_path / 'm.csv'), 'k,n,f\na,1,1.5\nb,,\nc,abc,2\n')
    with SqlStore() as store:
        m_sdaf: SqlDaf = store.from_csv(csv_path)
        assert m_sdaf.to_daf().lol == [['a', '1', '1.5'], ['b', NULL, NULL], ['c', 'abc', '2']]
        assert m_sdaf.sql('SELECT typeof(n) FROM m').to_list() == ['text', 'null', 'text']


def test_from_csv_text_compares_as_text_in_sql(tmp_path):
    csv_path: str = write_csv(str(tmp_path / 't.csv'), 'k,n\na,1\nb,10\nc,9\n')
    with SqlStore() as store:
        t_sdaf: SqlDaf = store.from_csv(csv_path)
        assert t_sdaf.sql('SELECT k FROM t WHERE n < 5').lol == []
        assert t_sdaf.sql('SELECT MAX(n) FROM t').lol == [['9']]
        t_sdaf.apply_dtypes(dtypes={'k': str, 'n': int})
        assert t_sdaf.sql('SELECT k FROM t WHERE n < 5').to_list() == ['a']
        assert t_sdaf.sql('SELECT MAX(n) FROM t').lol == [[10]]


def test_from_csv_pads_short_rows_and_rejects_long_ones(tmp_path):
    short_path: str = write_csv(str(tmp_path / 'short.csv'), 'a,b,c\n1\n')
    long_path: str = write_csv(str(tmp_path / 'long.csv'), 'a,b\n1,2,3\n')
    with SqlStore() as store:
        assert store.from_csv(short_path).to_daf().lol == [['1', NULL, NULL]]
        with pytest.raises(ValueError):
            store.from_csv(long_path)
        assert not store.table_exists('long')
        assert store.conn.in_transaction is False


def test_from_csv_duplicate_key_raises_at_first_key_use(tmp_path):
    csv_path: str = write_csv(str(tmp_path / 'dup.csv'), 'ballot_id,v\nb1,1\nb1,2\n')
    with SqlStore() as store:
        dup_sdaf: SqlDaf = store.from_csv(csv_path, keyfield='ballot_id')
        assert len(dup_sdaf) == 2
        with pytest.raises(sqlite3.IntegrityError):
            dup_sdaf.ensure_key_index()
        assert store.conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type = 'index'").fetchone()[0] == 0


def test_from_csv_keyfield_must_be_a_column(tmp_path):
    csv_path: str = write_csv(str(tmp_path / 'k.csv'), 'a,b\n1,2\n')
    with SqlStore() as store:
        with pytest.raises(KeyError):
            store.from_csv(csv_path, keyfield='c')
        assert not store.table_exists('k')


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
        mixed_daf: Daf = Daf(lol=[['a', 1, 1.5, True], ['b', '2', NULL, False]], cols=['k', 'n', 'f', 'b'], dtypes={'n': int})
        mixed_sdaf: SqlDaf = store.from_daf(mixed_daf, name='mixed')
        assert mixed_sdaf.to_daf().lol == [['a', 1, 1.5, 1], ['b', '2', NULL, 0]]
        assert mixed_sdaf.dtypes == {'n': int}
        assert mixed_sdaf.to_daf().dtypes == {'n': int}


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
        assert repr(marks_sdaf) == "<SqlDaf table name='marks' rows=3 cols=3 keyfield='ballot_id'>"
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
    lambda sdaf: sdaf[0, 'contest'],
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
        marks_sdaf.ensure_key_index()
        with pytest.raises(sqlite3.IntegrityError):
            store.insert_rows('marks', ['ballot_id', 'v'], [['b3', 3], ['b1', 4]])
        assert marks_sdaf.to_daf().lol == [['b1', 1], ['b2', NULL]]


# SqlDaf.set_keyfield and ensure_key_index

def test_ensure_key_index_makes_one_unique_index_and_is_used():
    with SqlStore() as store:
        marks_sdaf: SqlDaf = store.from_daf(make_marks_daf(), name='marks')
        assert store.conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type = 'index'").fetchone()[0] == 0
        marks_sdaf.ensure_key_index()
        marks_sdaf.ensure_key_index()
        index_lot = store.conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'").fetchall()
        assert index_lot == [('daf_ix_marks_ballot_id',)]
        plan_ls = [r[-1] for r in store.conn.execute("EXPLAIN QUERY PLAN SELECT * FROM marks WHERE ballot_id = 'b2'")]
        assert plan_ls == ['SEARCH marks USING INDEX daf_ix_marks_ballot_id (ballot_id=?)']


def test_ensure_key_index_without_keyfield_raises():
    with SqlStore() as store:
        marks_sdaf: SqlDaf = store.from_daf(make_marks_daf(), name='marks', keyfield='')
        marks_sdaf.set_keyfield('')
        with pytest.raises(ValueError):
            marks_sdaf.ensure_key_index()


def test_set_keyfield():
    with SqlStore() as store:
        marks_sdaf: SqlDaf = store.from_daf(make_marks_daf(), name='marks')
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
    with SqlStore() as store:
        sql_sdaf: SqlDaf = store.from_daf(Daf(lol=lol, cols=['i', 'f', 'b', 's']), name='t')
        sql_sdaf.apply_dtypes(dtypes=dtypes)
        assert sql_sdaf.to_daf().lol == mem_daf.lol
        assert sql_sdaf.to_daf().lol == [[1, 2.5, 0, 'x'], [NULL, 'abc', 1, 'y'], [7, 1000.0, 0, 'z']]
        assert sql_sdaf.dtypes == dtypes


def test_apply_dtypes_null_stays_sql_null_for_int():
    with SqlStore() as store:
        t_sdaf: SqlDaf = store.from_daf(Daf(lol=[['1'], ['']], cols=['n']), name='t')
        t_sdaf.apply_dtypes(dtypes={'n': int})
        assert t_sdaf.sql('SELECT typeof(n) FROM t').to_list() == ['integer', 'null']


def test_apply_dtypes_cols_converts_only_those():
    with SqlStore() as store:
        t_sdaf: SqlDaf = store.from_daf(Daf(lol=[['1', '2']], cols=['a', 'b']), name='t')
        t_sdaf.apply_dtypes(dtypes={'a': int, 'b': int}, cols=['a'])
        assert t_sdaf.to_daf().lol == [[1, '2']]


def test_apply_dtypes_missing_dtype_raises_unless_silent():
    with SqlStore() as store:
        t_sdaf: SqlDaf = store.from_daf(Daf(lol=[['1', '2']], cols=['a', 'b']), name='t')
        with pytest.raises(ValueError):
            t_sdaf.apply_dtypes(dtypes={'a': int})
        t_sdaf.apply_dtypes(dtypes={'a': int}, silent_error=True)
        assert t_sdaf.to_daf().lol == [[1, '2']]


def test_apply_dtypes_unknown_col_raises_keyerror():
    with SqlStore() as store:
        t_sdaf: SqlDaf = store.from_daf(Daf(lol=[['1']], cols=['a']), name='t')
        with pytest.raises(KeyError):
            t_sdaf.apply_dtypes(dtypes={'a': int}, cols=['z'])


def test_apply_dtypes_list_is_not_supported_yet():
    with SqlStore() as store:
        t_sdaf: SqlDaf = store.from_daf(Daf(lol=[['[1]']], cols=['a']), name='t')
        with pytest.raises(SqlNotSupportedError):
            t_sdaf.apply_dtypes(dtypes={'a': list})


def test_apply_dtypes_uses_stored_dtypes():
    with SqlStore() as store:
        t_sdaf: SqlDaf = store.from_daf(Daf(lol=[['1']], cols=['a'], dtypes={'a': int}), name='t')
        t_sdaf.apply_dtypes()
        assert t_sdaf.to_daf().lol == [[1]]


# SqlStore.transaction

def test_transaction_commits_together():
    with SqlStore() as store:
        marks_sdaf: SqlDaf = store.from_daf(make_marks_daf(), name='marks')
        with store.transaction():
            marks_sdaf.sql("UPDATE marks SET num_votes = 7 WHERE ballot_id = 'b1'")
            assert store.conn.in_transaction is True
        assert store.conn.in_transaction is False
        assert marks_sdaf.to_daf()['b1', 'num_votes'].to_value() == 7


def test_transaction_rolls_back_on_error():
    with SqlStore() as store:
        marks_sdaf: SqlDaf = store.from_daf(make_marks_daf(), name='marks')
        with pytest.raises(RuntimeError):
            with store.transaction():
                marks_sdaf.sql("UPDATE marks SET num_votes = 7 WHERE ballot_id = 'b1'")
                raise RuntimeError('stop')
        assert store.conn.in_transaction is False
        assert marks_sdaf.to_daf()['b1', 'num_votes'].to_value() == 1


def test_transaction_nested_undoes_only_the_inner_block():
    with SqlStore() as store:
        marks_sdaf: SqlDaf = store.from_daf(make_marks_daf(), name='marks')
        with store.transaction():
            marks_sdaf.sql("UPDATE marks SET num_votes = 7 WHERE ballot_id = 'b1'")
            with pytest.raises(RuntimeError):
                with store.transaction():
                    marks_sdaf.sql("UPDATE marks SET num_votes = 8 WHERE ballot_id = 'b3'")
                    raise RuntimeError('stop')
        assert marks_sdaf.to_daf().to_lod() == [
            {'ballot_id': 'b1', 'contest': 'c0', 'num_votes': 7},
            {'ballot_id': 'b2', 'contest': 'c1', 'num_votes': NULL},
            {'ballot_id': 'b3', 'contest': 'c0', 'num_votes': 0}]


def test_load_inside_a_transaction_is_a_savepoint(tmp_path):
    csv_path: str = write_csv(str(tmp_path / 'bad.csv'), 'a,b\n1,2\n1,2,3\n')
    with SqlStore() as store:
        store.from_daf(make_marks_daf(), name='marks')
        with store.transaction():
            with pytest.raises(ValueError):
                store.from_csv(csv_path)
            assert not store.table_exists('bad')
        assert store.table_exists('marks')


# ---- step 2: selections ----

def make_bif_daf() -> Daf:
    return Daf(
        lol=[['b1', 'P1', 1, 0], ['b2', 'P2', 2, 1], ['b3', 'P1', 3, NULL], ['b4', 'P3', 1, 1], ['b5', 'P1', 2, 0]],
        cols=['ballot_id', 'precinct', 'style_num', 'is_bmd'],
        keyfield='ballot_id',
    )


def temp_tables(store: SqlStore) -> T_ls:
    return [row_t[0] for row_t in store.conn.execute("SELECT name FROM sqlite_temp_master WHERE type = 'table'")]



def test_select_where_matches_mdaf():
    bif_daf: Daf = make_bif_daf()
    with SqlStore() as store:
        bif_sdaf: SqlDaf = store.from_daf(bif_daf, name='bif')
        def where(row) -> bool:
            return row['precinct'] == 'P1' and row['is_bmd'] is not NULL
        assert bif_sdaf.select_where(where).to_daf().lol == bif_daf.select_where(where).lol
        assert bif_sdaf.select_where(where, cols=['precinct', 'is_bmd']).to_daf().lol == [['b1', 'P1', 1, 0], ['b5', 'P1', 2, 0]]


def test_select_where_gives_a_keyedlist_with_null():
    seen_la: list = []
    with SqlStore() as store:
        bif_sdaf: SqlDaf = store.from_daf(make_bif_daf(), name='bif')
        def where(row) -> bool:
            seen_la.append((type(row).__name__, row['is_bmd']))
            return True
        bif_sdaf.select_where(where, cols=['is_bmd']).to_daf()
        assert seen_la[2] == ('KeyedList', NULL)
        assert seen_la[2][1] is NULL


def test_select_where_error_is_raised_as_itself():
    with SqlStore() as store:
        bif_sdaf: SqlDaf = store.from_daf(make_bif_daf(), name='bif')
        sel_sdaf: SqlDaf = bif_sdaf.select_where(lambda row: 1 / 0)
        with pytest.raises(ZeroDivisionError):
            sel_sdaf.to_daf()


def test_select_where_bad_col_and_indirect_col():
    with SqlStore() as store:
        bif_sdaf: SqlDaf = store.from_daf(make_bif_daf(), name='bif')
        with pytest.raises(KeyError):
            bif_sdaf.select_where(lambda row: True, cols=['nothing'])
        with pytest.raises(SqlNotSupportedError):
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
    with SqlStore() as store:
        bif_sdaf: SqlDaf = store.from_daf(bif_daf, name='bif')
        assert bif_sdaf.select_by_dict(selector, inverse=inverse).to_daf().lol == bif_daf.select_by_dict(selector, inverse=inverse).lol


def test_select_by_dict_expectmax_and_keyfield():
    with SqlStore() as store:
        bif_sdaf: SqlDaf = store.from_daf(make_bif_daf(), name='bif')
        with pytest.raises(LookupError):
            bif_sdaf.select_by_dict({'precinct': 'P1'}, expectmax=2)
        assert len(bif_sdaf.select_by_dict({'precinct': 'P1'}, expectmax=3)) == 3
        assert bif_sdaf.select_by_dict({'precinct': 'P1'}, keyfield='style_num').keyfield == 'style_num'
        with pytest.raises(TypeError):
            bif_sdaf.select_by_dict('P1')                       # type: ignore[arg-type]


def test_select_where_sql_with_params_and_null():
    with SqlStore() as store:
        bif_sdaf: SqlDaf = store.from_daf(make_bif_daf(), name='bif')
        assert bif_sdaf.select_where_sql('"style_num" >= ? AND "precinct" != ?', [2, 'P2']).to_daf().lol == [
            ['b3', 'P1', 3, NULL], ['b5', 'P1', 2, 0]]
        assert bif_sdaf.select_where_sql('"is_bmd" IS ?', [NULL]).to_list() == ['b3', 'P1', 3, NULL]


def test_selection_is_lazy_and_fixed_at_first_read():
    with SqlStore() as store:
        bif_sdaf: SqlDaf = store.from_daf(make_bif_daf(), name='bif')
        sel_sdaf: SqlDaf = bif_sdaf.select_by_dict({'precinct': 'P1'})
        assert temp_tables(store) == []
        bif_sdaf.sql("UPDATE bif SET precinct = 'P1' WHERE ballot_id = 'b4'")      # before the first read: counts
        assert len(sel_sdaf) == 4
        assert temp_tables(store) == [sel_sdaf.name]
        bif_sdaf.sql("UPDATE bif SET precinct = 'P1' WHERE ballot_id = 'b2'")      # after: the rows are fixed
        bif_sdaf.sql("UPDATE bif SET style_num = 9 WHERE ballot_id = 'b1'")        # but values show through
        assert sel_sdaf.to_daf().lol == [['b1', 'P1', 9, 0], ['b3', 'P1', 3, NULL], ['b4', 'P1', 1, 1], ['b5', 'P1', 2, 0]]


def test_deleted_base_row_drops_out_of_a_fixed_selection():
    with SqlStore() as store:
        bif_sdaf: SqlDaf = store.from_daf(make_bif_daf(), name='bif')
        sel_sdaf: SqlDaf = bif_sdaf.select_by_dict({'precinct': 'P1'}).fix()
        bif_sdaf.sql("DELETE FROM bif WHERE ballot_id = 'b3'")
        assert sel_sdaf.to_daf()[:, 'ballot_id'].to_list() == ['b1', 'b5']
        assert len(sel_sdaf) == 2


def test_chain_of_selections_runs_as_one_query_until_read():
    with SqlStore() as store:
        bif_sdaf: SqlDaf = store.from_daf(make_bif_daf(), name='bif')
        p1_sdaf: SqlDaf = bif_sdaf.select_by_dict({'precinct': 'P1'})
        p1_style2_sdaf: SqlDaf = p1_sdaf.select_where_sql('"style_num" = 2')
        assert p1_style2_sdaf.base_sdaf is bif_sdaf
        assert p1_style2_sdaf.to_daf().lol == [['b5', 'P1', 2, 0]]
        assert temp_tables(store) == [p1_style2_sdaf.name]


def test_selection_of_a_fixed_selection_uses_its_rows():
    with SqlStore() as store:
        bif_sdaf: SqlDaf = store.from_daf(make_bif_daf(), name='bif')
        p1_sdaf: SqlDaf = bif_sdaf.select_by_dict({'precinct': 'P1'}).fix()
        bif_sdaf.sql("UPDATE bif SET precinct = 'P1' WHERE ballot_id = 'b2'")
        assert p1_sdaf.select_where_sql('"style_num" = 2').to_daf()[:, 'ballot_id'].to_list() == ['b5']


def test_selection_keeps_keyfield_dtypes_and_columns():
    with SqlStore() as store:
        bif_sdaf: SqlDaf = store.from_daf(make_bif_daf(), name='bif')
        sel_sdaf: SqlDaf = bif_sdaf.select_by_dict({'precinct': 'P1'})
        assert sel_sdaf.columns() == bif_sdaf.columns()
        assert sel_sdaf.keyfield == 'ballot_id'
        assert sel_sdaf.name.startswith('daf_tmp_bif_v')
        assert repr(sel_sdaf).startswith("<SqlDaf selection name='daf_tmp_bif_v")


# indexing

@pytest.mark.parametrize('sel', [0, 2, -1, -5, slice(1, 3), slice(None, 2), slice(3, None), slice(-2, None),
                                 slice(None, -3), slice(4, 9), slice(7, 9), slice(3, 1)])
def test_getitem_by_position_matches_mdaf(sel):
    bif_daf: Daf = make_bif_daf()
    with SqlStore() as store:
        bif_sdaf: SqlDaf = store.from_daf(bif_daf, name='bif')
        assert bif_sdaf[sel].to_daf().lol == bif_daf[sel].lol


def test_getitem_on_a_selection():
    bif_daf: Daf = make_bif_daf()
    with SqlStore() as store:
        bif_sdaf: SqlDaf = store.from_daf(bif_daf, name='bif')
        p1_sdaf: SqlDaf = bif_sdaf.select_by_dict({'precinct': 'P1'})
        assert p1_sdaf[1].to_dict() == {'ballot_id': 'b3', 'precinct': 'P1', 'style_num': 3, 'is_bmd': NULL}
        assert p1_sdaf[1:].to_daf().lol == bif_daf.select_by_dict({'precinct': 'P1'})[1:].lol


def test_getitem_errors():
    with SqlStore() as store:
        bif_sdaf: SqlDaf = store.from_daf(make_bif_daf(), name='bif')
        with pytest.raises(IndexError):
            bif_sdaf[5]
        with pytest.raises(IndexError):
            bif_sdaf[-6]
        with pytest.raises(KeyError):
            bif_sdaf['nothing']
        with pytest.raises(SqlNotSupportedError):
            bif_sdaf[0:4:2]
        with pytest.raises(SqlNotSupportedError):
            bif_sdaf[0, 'precinct']
        nokey_sdaf: SqlDaf = store.from_daf(make_bif_daf(), name='nokey', keyfield='').set_keyfield('')
        with pytest.raises(SqlNotSupportedError):
            nokey_sdaf['b1']


def test_getitem_by_key():
    bif_daf: Daf = make_bif_daf()
    with SqlStore() as store:
        bif_sdaf: SqlDaf = store.from_daf(bif_daf, name='bif')
        assert bif_sdaf['b3'].to_daf().lol == bif_daf['b3'].lol
        assert bif_sdaf['b3'].to_dict() == bif_daf['b3'].to_dict()
        assert 'daf_ix_bif_ballot_id' in [r[0] for r in store.conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")]


# reading one row

@pytest.mark.parametrize('irow', [0, 3, -1, -5])
def test_iloc_matches_mdaf(irow):
    bif_daf: Daf = make_bif_daf()
    with SqlStore() as store:
        bif_sdaf: SqlDaf = store.from_daf(bif_daf, name='bif')
        assert bif_sdaf.iloc(irow) == bif_daf.iloc(irow)
        assert bif_sdaf.iloc(irow, rtype='list') == bif_daf.iloc(irow, rtype='list')
        assert bif_sdaf.iloc(irow, include_cols=['style_num', 'ballot_id']) == bif_daf.iloc(irow, include_cols=['style_num', 'ballot_id'])


def test_iloc_errors_and_does_not_fix():
    with SqlStore() as store:
        bif_sdaf: SqlDaf = store.from_daf(make_bif_daf(), name='bif')
        with pytest.raises(IndexError):
            bif_sdaf.iloc(5)
        with pytest.raises(IndexError):
            bif_sdaf.iloc(-6)
        with pytest.raises(KeyError):
            bif_sdaf.iloc(0, include_cols=['nothing'])
        with pytest.raises(SqlNotSupportedError):
            bif_sdaf.iloc(0, rtype='klist')
        sel_sdaf: SqlDaf = bif_sdaf.select_by_dict({'precinct': 'P1'})
        assert sel_sdaf.iloc(1)['ballot_id'] == 'b3'
        assert sel_sdaf.is_fixed is False
        assert temp_tables(store) == []


def test_select_record_matches_mdaf():
    bif_daf: Daf = make_bif_daf()
    with SqlStore() as store:
        bif_sdaf: SqlDaf = store.from_daf(bif_daf, name='bif')
        assert bif_sdaf.select_record('b3') == bif_daf.select_record('b3')
        assert bif_sdaf.select_record('nothing') == bif_daf.select_record('nothing') == {}
        with pytest.raises(KeyError):
            bif_sdaf.select_record('nothing', silent_error=False)
        sel_sdaf: SqlDaf = bif_sdaf.select_by_dict({'precinct': 'P1'})
        assert sel_sdaf.select_record('b2') == {}
        assert sel_sdaf.select_record('b5')['style_num'] == 2
        assert sel_sdaf.is_fixed is False


def test_select_record_without_keyfield_raises():
    with SqlStore() as store:
        t_sdaf: SqlDaf = store.from_daf(Daf(lol=[['a']], cols=['x']), name='t')
        with pytest.raises(ValueError):
            t_sdaf.select_record('a')


# to_dict, to_list, to_value

def test_to_dict_to_list_to_value_match_mdaf():
    bif_daf: Daf = make_bif_daf()
    with SqlStore() as store:
        bif_sdaf: SqlDaf = store.from_daf(bif_daf, name='bif')
        assert bif_sdaf[2].to_dict() == bif_daf[2].to_dict()
        assert bif_sdaf[2].to_list() == bif_daf[2].to_list()
        assert bif_sdaf.select_where_sql('"style_num" = 9').to_dict() == {}
        with pytest.raises(ValueError):
            bif_sdaf.to_dict()
        with pytest.raises(ValueError):
            bif_sdaf.to_list()
        with pytest.raises(ValueError):
            bif_sdaf[0].to_value()
        one_col_sdaf: SqlDaf = store.from_daf(Daf(lol=[['a'], [NULL], ['c']], cols=['x']), name='one')
        assert one_col_sdaf.to_list() == ['a', NULL, 'c']
        assert one_col_sdaf[1].to_value() is NULL
        assert one_col_sdaf[2].to_value() == 'c'


# materialize and save_as

def test_materialize_copies_into_a_temp_table():
    with SqlStore() as store:
        bif_sdaf: SqlDaf = store.from_daf(make_bif_daf(), name='bif')
        assert bif_sdaf.materialize() is bif_sdaf
        copy_sdaf: SqlDaf = bif_sdaf.select_by_dict({'precinct': 'P1'}).materialize()
        assert copy_sdaf.base_sdaf is None
        assert copy_sdaf.name in temp_tables(store)
        copy_sdaf.sql(f'UPDATE "{copy_sdaf.name}" SET style_num = 99')
        assert bif_sdaf.select_record('b1')['style_num'] == 1
        assert copy_sdaf.to_daf()[:, 'style_num'].to_list() == [99, 99, 99]
        assert copy_sdaf.keyfield == 'ballot_id'


def test_save_as_makes_a_kept_table(tmp_path):
    db_path: str = str(tmp_path / 'kept.db')
    with SqlStore(db_path) as store:
        bif_sdaf: SqlDaf = store.from_daf(make_bif_daf(), name='bif')
        p1_sdaf: SqlDaf = bif_sdaf[1:4].save_as('middle')
        assert p1_sdaf.to_daf()[:, 'ballot_id'].to_list() == ['b2', 'b3', 'b4']
        with pytest.raises(ValueError):
            bif_sdaf.save_as('middle')
    with SqlStore(db_path) as store:
        assert store.table('middle').to_daf()[:, 'ballot_id'].to_list() == ['b2', 'b3', 'b4']


# references that survive VACUUM

def test_selection_needs_a_vacuum_safe_reference():
    conn: sqlite3.Connection = sqlite3.connect(':memory:')
    conn.execute('CREATE TABLE plain (ballot_id TEXT, v INTEGER)')
    conn.executemany('INSERT INTO plain VALUES (?, ?)', [('b1', 1), ('b2', 2)])
    with SqlStore(conn=conn) as store:
        plain_sdaf: SqlDaf = store.table('plain')
        assert plain_sdaf.to_daf().lol == [['b1', 1], ['b2', 2]]
        with pytest.raises(SqlNotSupportedError):
            plain_sdaf.select_where_sql('v = 2')
        keyed_sdaf: SqlDaf = store.table('plain', keyfield='ballot_id')
        assert keyed_sdaf.select_where_sql('v = 2').to_daf().lol == [['b2', 2]]


def test_key_references_survive_delete_and_vacuum(tmp_path):
    db_path: str = str(tmp_path / 'v.db')
    conn: sqlite3.Connection = sqlite3.connect(db_path, isolation_level=None)
    conn.execute('CREATE TABLE plain (ballot_id TEXT, v INTEGER)')
    conn.executemany('INSERT INTO plain VALUES (?, ?)', [(f'b{i}', i % 3) for i in range(10)])
    with SqlStore(conn=conn) as store:
        plain_sdaf: SqlDaf = store.table('plain', keyfield='ballot_id')
        sel_sdaf: SqlDaf = plain_sdaf.select_where_sql('v = 0').fix()
        conn.execute("DELETE FROM plain WHERE ballot_id IN ('b0', 'b1', 'b2')")
        conn.execute('VACUUM')
        assert sel_sdaf.to_daf()[:, 'ballot_id'].to_list() == ['b3', 'b6', 'b9']


def test_apply_dtypes_on_a_selection_raises():
    with SqlStore() as store:
        bif_sdaf: SqlDaf = store.from_daf(make_bif_daf(), name='bif')
        with pytest.raises(SqlNotSupportedError):
            bif_sdaf.select_by_dict({'precinct': 'P1'}).apply_dtypes(dtypes={'style_num': int})


def test_ensure_key_index_on_a_selection_raises():
    with SqlStore() as store:
        bif_sdaf: SqlDaf = store.from_daf(make_bif_daf(), name='bif')
        with pytest.raises(ValueError):
            bif_sdaf.select_by_dict({'precinct': 'P1'}).ensure_key_index()
