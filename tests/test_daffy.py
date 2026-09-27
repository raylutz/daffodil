# test_daffy.py
#
# Tests for daffodil.daffy: the CLI tool for inspecting/filtering CSV files, built on Daf.
# v1 scope: inspect and select only (Ray, 2026-09-27: "get our feet wet").

import json

import pytest

from daffodil.daffy import cli, profile, sniff


# =====================================================================
# sniff_csv -- line ending / dialect detection from raw bytes
# =====================================================================

def test_sniff_detects_crlf(tmp_path):
    path = tmp_path / "crlf.csv"
    path.write_bytes(b"id,name\r\n001,Alice\r\n002,Bob\r\n")

    result = sniff.sniff_csv(path)

    assert result.line_terminator == '\r\n'
    assert sniff.line_terminator_label(result.line_terminator) == 'CRLF'
    assert not result.mixed_line_endings


def test_sniff_detects_lf(tmp_path):
    path = tmp_path / "lf.csv"
    path.write_bytes(b"id,name\n001,Alice\n002,Bob\n")

    result = sniff.sniff_csv(path)

    assert result.line_terminator == '\n'
    assert sniff.line_terminator_label(result.line_terminator) == 'LF'
    assert not result.mixed_line_endings


def test_sniff_detects_mixed_line_endings(tmp_path):
    path = tmp_path / "mixed.csv"
    path.write_bytes(b"id,name\r\n001,Alice\n002,Bob\r\n")

    result = sniff.sniff_csv(path)

    assert result.mixed_line_endings


def test_sniff_detects_semicolon_delimiter(tmp_path):
    path = tmp_path / "semi.csv"
    path.write_bytes(b"id;name\n001;Alice\n002;Bob\n")

    result = sniff.sniff_csv(path)

    assert result.delimiter == ';'


# =====================================================================
# profile.py -- sidecar discovery, load, save
# =====================================================================

def test_profile_path_uses_full_csv_name(tmp_path):
    csv_path = tmp_path / "foo.csv"

    assert profile.profile_path_for(csv_path) == tmp_path / "foo.csv.profile.json"


def test_missing_profile_returns_empty_dict(tmp_path):
    csv_path = tmp_path / "foo.csv"
    csv_path.write_text("a,b\n1,2\n")

    assert profile.load_profile(csv_path) == {}


def test_save_then_load_profile_round_trips(tmp_path):
    csv_path = tmp_path / "foo.csv"
    csv_path.write_text("a,b\n1,2\n")

    saved_path = profile.save_profile(csv_path, {'keyfield': 'a', 'widths': {'b': 10}, 'dtypes': {}})

    assert saved_path == tmp_path / "foo.csv.profile.json"
    assert profile.load_profile(csv_path) == {'keyfield': 'a', 'widths': {'b': 10}, 'dtypes': {}}


def test_profile_survives_unknown_fields(tmp_path):
    # a hand-added or future-version field must not be dropped on save/load.
    csv_path = tmp_path / "foo.csv"
    profile.save_profile(csv_path, {'keyfield': 'a', 'future_field': 'kept'})

    loaded = profile.load_profile(csv_path)

    assert loaded['future_field'] == 'kept'


def test_explicit_profile_path_overrides_discovery(tmp_path):
    csv_path = tmp_path / "foo.csv"
    csv_path.write_text("a,b\n1,2\n")
    explicit_path = tmp_path / "elsewhere.json"
    explicit_path.write_text(json.dumps({'keyfield': 'elsewhere'}))

    assert profile.load_profile(csv_path, explicit_path=explicit_path) == {'keyfield': 'elsewhere'}


def test_malformed_profile_raises(tmp_path):
    csv_path = tmp_path / "foo.csv"
    (tmp_path / "foo.csv.profile.json").write_text("[1, 2, 3]")  # a list, not an object

    with pytest.raises(ValueError):
        profile.load_profile(csv_path)


# =====================================================================
# cli.py -- inspect / select, schema-free operation, string preservation
# =====================================================================

@pytest.fixture
def sample_csv(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text("id,name,city\r\n001,Alice,San Diego\r\n002,Bob,Reno\r\n003,Carl,San Diego\r\n", newline='')
    return path


def test_inspect_reports_columns_and_row_count_schema_free(sample_csv, capsys):
    exit_code = cli.main(['inspect', str(sample_csv), '--format', 'json'])
    out = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert out['columns'] == ['id', 'name', 'city']
    assert out['num_rows'] == 3
    assert out['line_terminator'] == 'CRLF'
    assert out['profile_path'] is None  # no sidecar required for basic inspection


def test_inspect_missing_file_errors_cleanly(tmp_path, capsys):
    exit_code = cli.main(['inspect', str(tmp_path / "nope.csv")])

    assert exit_code == 2
    assert "does not exist" in capsys.readouterr().err


def test_select_preserves_leading_zero_strings(sample_csv, capsys):
    exit_code = cli.main(['select', str(sample_csv), '--format', 'json'])
    rows = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert rows[0]['id'] == '001'  # not int 1 -- no numeric type inference by default


def test_select_filters_by_equality(sample_csv, capsys):
    exit_code = cli.main(['select', str(sample_csv), '--filter', 'city=San Diego', '--format', 'json'])
    rows = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert {row['name'] for row in rows} == {'Alice', 'Carl'}


def test_select_cols_keeps_source_column_order_not_requested_order(sample_csv, capsys):
    # Daf.select_cols() is a subset operation, not a reorder -- its own docstring warns
    # reordering isn't its job ("provide cols parameter in .apply/.reduce/.from_xxx/.to_xxx
    # instead" for efficiency). --cols "city,id" still comes back as id,city (source order).
    exit_code = cli.main(['select', str(sample_csv), '--cols', 'city,id', '--format', 'json'])
    rows = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert list(rows[0].keys()) == ['id', 'city']


def test_select_limit_bounds_row_count(sample_csv, capsys):
    exit_code = cli.main(['select', str(sample_csv), '--limit', '1', '--format', 'json'])
    rows = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert len(rows) == 1


def test_select_unknown_filter_column_errors_cleanly(sample_csv, capsys):
    exit_code = cli.main(['select', str(sample_csv), '--filter', 'nosuchcol=x'])

    assert exit_code == 2
    assert "nosuchcol" in capsys.readouterr().err


def test_select_unknown_cols_column_errors_cleanly(sample_csv, capsys):
    exit_code = cli.main(['select', str(sample_csv), '--cols', 'nosuchcol'])

    assert exit_code == 2
    assert "nosuchcol" in capsys.readouterr().err


def test_select_malformed_filter_errors_cleanly(sample_csv, capsys):
    exit_code = cli.main(['select', str(sample_csv), '--filter', 'not-a-valid-term'])

    assert exit_code == 2
    assert "not of the form col=value" in capsys.readouterr().err


def test_select_pyon_format_is_valid_python_literal(sample_csv, capsys):
    import ast

    exit_code = cli.main(['select', str(sample_csv), '--format', 'pyon'])
    out = capsys.readouterr().out

    assert exit_code == 0
    parsed = ast.literal_eval(out)
    assert parsed[0]['id'] == '001'
