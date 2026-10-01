# test_daf_sql.py
#
# Tests for daffodil/lib/daf_sql.py: SQL identifier escaping/unescaping and the small
# sqlite helper functions (lod -> table, index creation, column sums, row selection).

import sqlite3
import sys
from unittest import mock

import pytest

from daffodil.lib import daf_sql


# --- sql_unesc_str ---

def test_sql_unesc_str_plain_name_unchanged():
    assert daf_sql.sql_unesc_str('abc') == 'abc'


def test_sql_unesc_str_strips_whitespace():
    assert daf_sql.sql_unesc_str('  abc  ') == 'abc'


def test_sql_unesc_str_removes_quotes_and_unescapes_embedded_quotes():
    assert daf_sql.sql_unesc_str('"My Col"') == 'My Col'
    assert daf_sql.sql_unesc_str('"a""b"') == 'a"b'


def test_sql_unesc_str_decodes_hex_escapes():
    assert daf_sql.sql_unesc_str('a__20b') == 'a b'
    assert daf_sql.sql_unesc_str('__73elect') == 'select'
    assert daf_sql.sql_unesc_str('__31abc') == '1abc'


def test_decode_colname_from_sqlcol_is_alias():
    assert daf_sql.decode_colname_from_sqlcol is daf_sql.sql_unesc_str


# --- sql_escape_str (quoting_ok=True) ---

@pytest.mark.parametrize("name", ['abc', 'col_1', '_private', 'rowkey'])
def test_sql_escape_str_safe_names_unquoted(name):
    assert daf_sql.sql_escape_str(name) == name


@pytest.mark.parametrize("name, expected", [
    ('Abc',     '"Abc"'),           # uppercase
    ('1abc',    '"1abc"'),          # leading digit
    ('a b',     '"a b"'),           # space
    ('select',  '"select"'),        # reserved word
    ('Name',    '"Name"'),          # reserved word (case-insensitive) + uppercase
    ('a"b',     '"a""b"'),          # embedded quote doubled
])
def test_sql_escape_str_quotes_when_needed(name, expected):
    assert daf_sql.sql_escape_str(name) == expected


def test_sql_escape_str_is_idempotent():
    once = daf_sql.sql_escape_str('My Col')
    assert daf_sql.sql_escape_str(once) == once == '"My Col"'


@pytest.mark.parametrize("name", ['abc', 'Abc', 'a b', 'a"b', 'select', '1abc'])
def test_sql_escape_str_roundtrip_quoted(name):
    assert daf_sql.sql_unesc_str(daf_sql.sql_escape_str(name)) == name


# --- sql_escape_str (quoting_ok=False) ---

@pytest.mark.parametrize("name, expected", [
    ('abc',     'abc'),
    ('Abc',     'Abc'),             # uppercase alone is legal when not quoting
    ('a b',     'a__20b'),
    ('a"b',     'a__22b'),
    ('a-b.c',   'a__2Db__2Ec'),
    ('1abc',    '__31abc'),         # leading digit escaped
    ('select',  '__73elect'),       # reserved word: first char escaped
    ('Name',    '__4Eame'),         # reserved check is case-insensitive
])
def test_sql_escape_str_no_quoting_encodes(name, expected):
    assert daf_sql.sql_escape_str(name, quoting_ok=False) == expected


@pytest.mark.parametrize("name", ['a b', 'a"b', 'a-b.c', '1abc', 'select', 'Name', 'café'])
def test_sql_escape_str_no_quoting_roundtrip_and_idempotent(name):
    enc = daf_sql.sql_escape_str(name, quoting_ok=False)
    assert daf_sql.sql_unesc_str(enc) == name
    assert daf_sql.sql_escape_str(enc, quoting_ok=False) == enc


def test_sql_escape_str_no_quoting_result_is_usable_as_bare_identifier():
    enc = daf_sql.sql_escape_str('my table', quoting_ok=False)
    conn = sqlite3.connect(':memory:')
    conn.execute(f"CREATE TABLE {enc} (x)")
    conn.execute(f"INSERT INTO {enc} VALUES (1)")
    assert conn.execute(f"SELECT x FROM {enc}").fetchall() == [(1,)]
    conn.close()


@pytest.mark.xfail(strict=True, reason="BUG: sql_unesc_str decodes '__HH' inside an already-safe "
                   "identifier, so a legal name like 'data__ab' is mangled to '\"data\\xab\"'")
def test_sql_escape_str_safe_name_containing_double_underscore_hex():
    assert daf_sql.sql_escape_str('data__ab') == 'data__ab'


@pytest.mark.xfail(strict=True, reason="BUG: quoting_ok=False encodes chars > 0xFF with >2 hex "
                   "digits (e.g. '__20AC') but sql_unesc_str only decodes exactly 2, so not reversible")
def test_sql_escape_str_no_quoting_roundtrip_non_latin1():
    enc = daf_sql.sql_escape_str('€x', quoting_ok=False)
    assert daf_sql.sql_unesc_str(enc) == '€x'


# --- create_index_at_cursor ---

def _mem_table(table_name='t', cols='rowkey, val'):
    conn = sqlite3.connect(':memory:')
    conn.execute(f"CREATE TABLE {daf_sql.sql_escape_str(table_name)} ({cols})")
    return conn


def _index_names(conn):
    return sorted(r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'").fetchall())


def test_create_index_at_cursor_creates_named_index():
    conn = _mem_table()
    assert daf_sql.create_index_at_cursor(conn.cursor(), 'rowkey', 't') is True
    assert _index_names(conn) == ['idx_t_rowkey']


def test_create_index_at_cursor_is_repeatable():
    conn = _mem_table()
    cur = conn.cursor()
    assert daf_sql.create_index_at_cursor(cur, 'rowkey', 't') is True
    assert daf_sql.create_index_at_cursor(cur, 'rowkey', 't') is True
    assert _index_names(conn) == ['idx_t_rowkey']


def test_create_index_at_cursor_escapes_unusual_names():
    conn = _mem_table(table_name='My Table', cols='"Key Col", val')
    assert daf_sql.create_index_at_cursor(conn.cursor(), 'Key Col', 'My Table') is True
    assert _index_names(conn) == ['idx_My__20Table_Key__20Col']


def test_create_index_at_cursor_unique_enforced():
    conn = _mem_table()
    assert daf_sql.create_index_at_cursor(conn.cursor(), 'rowkey', 't', unique=True) is True
    conn.execute("INSERT INTO t VALUES ('a', 1)")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO t VALUES ('a', 2)")


def test_create_index_at_cursor_drop_first_replaces_nonunique_with_unique():
    conn = _mem_table()
    cur = conn.cursor()
    daf_sql.create_index_at_cursor(cur, 'rowkey', 't')
    # without drop_first, an existing non-unique index is kept as-is
    daf_sql.create_index_at_cursor(cur, 'rowkey', 't', unique=True)
    conn.execute("INSERT INTO t VALUES ('a', 1)")
    conn.execute("INSERT INTO t VALUES ('a', 2)")
    conn.execute("DELETE FROM t")
    # with drop_first, it becomes unique
    assert daf_sql.create_index_at_cursor(cur, 'rowkey', 't', unique=True, drop_first=True, diagnose=True)
    conn.execute("INSERT INTO t VALUES ('a', 1)")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO t VALUES ('a', 2)")


def test_create_index_at_cursor_already_exists_error_treated_as_success():
    cursor = mock.MagicMock()
    cursor.execute.side_effect = sqlite3.OperationalError("index idx_t_rowkey already exists")
    assert daf_sql.create_index_at_cursor(cursor, 'rowkey', 't') is True


@pytest.mark.parametrize("cursor_factory", [
    lambda: _mem_table().cursor(),                       # OperationalError: no such column
    lambda: mock.MagicMock(**{'execute.side_effect': RuntimeError('boom')}),
])
def test_create_index_at_cursor_failure_returns_false(monkeypatch, cursor_factory):
    beeps, breaks = [], []
    monkeypatch.setattr(daf_sql.logs, 'error_beep', lambda: beeps.append(1))
    monkeypatch.setattr(sys, 'breakpointhook', lambda *a, **k: breaks.append(1))
    assert daf_sql.create_index_at_cursor(cursor_factory(), 'missing_col', 't') is False
    assert beeps == [1] and breaks == [1]


# --- lod_to_sqlite_table ---

SAMPLE_LOD = [
    {'rowkey': 'a', 'x': 1, 'y': 10},
    {'rowkey': 'b', 'x': 2, 'y': 20},
    {'rowkey': 'c', 'x': 3, 'y': 30},
]


def test_lod_to_sqlite_table_writes_rows_and_index(tmp_path):
    db = str(tmp_path / 'data.db')
    daf_sql.lod_to_sqlite_table(SAMPLE_LOD, table_name='tbl', db_file_path=db)
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT rowkey, x, y FROM tbl ORDER BY rowkey").fetchall() == \
        [('a', 1, 10), ('b', 2, 20), ('c', 3, 30)]
    assert _index_names(conn) == ['idx_tbl_rowkey']
    conn.close()


def test_lod_to_sqlite_table_replaces_existing_table(tmp_path):
    db = str(tmp_path / 'data.db')
    daf_sql.lod_to_sqlite_table(SAMPLE_LOD, table_name='tbl', db_file_path=db)
    daf_sql.lod_to_sqlite_table([{'rowkey': 'z', 'x': 9, 'y': 99}], table_name='tbl', db_file_path=db)
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT * FROM tbl").fetchall() == [('z', 9, 99)]
    conn.close()


def test_lod_to_sqlite_table_no_key_col_creates_no_index(tmp_path):
    db = str(tmp_path / 'data.db')
    daf_sql.lod_to_sqlite_table(SAMPLE_LOD, table_name='tbl', db_file_path=db, key_col=None)
    conn = sqlite3.connect(db)
    assert _index_names(conn) == []
    assert conn.execute("SELECT COUNT(*) FROM tbl").fetchone() == (3,)
    conn.close()


def test_lod_to_sqlite_table_default_path_uses_table_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    daf_sql.lod_to_sqlite_table(SAMPLE_LOD, table_name='mytab')
    assert (tmp_path / 'mytab.db').exists()


def test_lod_to_sqlite_table_empty_raises(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        daf_sql.lod_to_sqlite_table([], table_name='tbl', db_file_path=str(tmp_path / 'e.db'))


# --- sum_columns_in_sqlite_table ---

def test_sum_columns_in_sqlite_table(tmp_path):
    db = str(tmp_path / 'data.db')
    daf_sql.lod_to_sqlite_table(SAMPLE_LOD, table_name='tbl', db_file_path=db)
    result = daf_sql.sum_columns_in_sqlite_table(table_name='tbl', db_file_path=db)
    # non-numeric text sums to 0 in sqlite
    assert result == {'rowkey': 0.0, 'x': 6, 'y': 60}


def test_sum_columns_in_sqlite_table_empty_table_gives_nones(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    conn = sqlite3.connect('empty.db')
    conn.execute("CREATE TABLE empty (a, b)")
    conn.commit()
    conn.close()
    assert daf_sql.sum_columns_in_sqlite_table(table_name='empty') == {'a': None, 'b': None}


# --- get_memory_usage_of_table_in_memory ---

def test_get_memory_usage_in_memory_db_is_zero():
    assert daf_sql.get_memory_usage_of_table_in_memory(':memory:') == 0


def test_get_memory_usage_matches_db_file_size(tmp_path):
    # note: despite its name, the argument is opened as a database path.
    db = tmp_path / 'data.db'
    daf_sql.lod_to_sqlite_table(SAMPLE_LOD, table_name='tbl', db_file_path=str(db))
    size = daf_sql.get_memory_usage_of_table_in_memory(str(db))
    assert size > 0
    assert size == db.stat().st_size


# --- print_table_summary ---

def test_print_table_summary_found(tmp_path, capsys):
    db = str(tmp_path / 'data.db')
    daf_sql.lod_to_sqlite_table(SAMPLE_LOD, table_name='tbl', db_file_path=db, key_col=None)
    daf_sql.print_table_summary(table_name='tbl', db_file_path=db)
    out = capsys.readouterr().out.splitlines()
    assert out == ["Table 'tbl' summary:", "CREATE TABLE tbl (rowkey, x, y)"]


def test_print_table_summary_not_found(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    daf_sql.print_table_summary(table_name='nosuch')
    assert capsys.readouterr().out == "Table 'nosuch' not found.\n"


# --- sqlite_selectrow ---

def test_sqlite_selectrow_found_and_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    daf_sql.lod_to_sqlite_table(SAMPLE_LOD, table_name='tbl')
    assert daf_sql.sqlite_selectrow('tbl', key_col='rowkey', value='b') == {'rowkey': 'b', 'x': 2, 'y': 20}
    assert daf_sql.sqlite_selectrow('tbl', key_col='x', value=3) == {'rowkey': 'c', 'x': 3, 'y': 30}
    assert daf_sql.sqlite_selectrow('tbl', key_col='rowkey', value='nope') is None
