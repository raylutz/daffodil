# test_daf_md.py
#
# Tests for daffodil/lib/daf_md.py: markdown link/escape helpers, table rendering
# (md_lol_table / md_cols_lol_table, reached via Daf.to_md() / Daf.to_md_cols()), and
# markdown-table parsing (Daf.from_md(), find_first_markdown_table, footer metadata).

import pytest

from daffodil.daf import Daf
from daffodil.lib import daf_md as md


# =====================================================================
# Group 1 -- pure helper functions (no Daf needed)
# =====================================================================

# --- mdlink / mdlink_s3path / new_window_link ---

def test_mdlink_no_url_returns_not_available():
    assert md.mdlink(url_or_s3path='', title='My Title') == "My Title (Not Available)"


def test_mdlink_plain_url():
    assert md.mdlink(url_or_s3path='https://example.com', title='Example') == "[Example](https://example.com)"


def test_mdlink_new_window():
    result = md.mdlink(url_or_s3path='https://example.com', title='Example', new_window=True)
    assert result == '<a href="https://example.com" target="_blank">Example</a>'


def test_mdlink_s3path_no_title_generates_from_basename():
    result = md.mdlink(url_or_s3path='s3://my-bucket/some/path/file.txt', title='')
    assert '[file.txt]' in result
    assert 'my-bucket.s3.amazonaws.com' in result


def test_mdlink_s3path_with_title_given():
    result = md.mdlink(url_or_s3path='s3://my-bucket/some/path/file.txt', title='My Title')
    assert result == '[My Title](https://my-bucket.s3.amazonaws.com/some/path/file.txt)'
    assert md.mdlink_s3path(s3path='', title='X') == "X (Not Available)"


def test_new_window_link():
    assert md.new_window_link('http://a.com', 'A') == '<a href="http://a.com" target="_blank">A</a>'


# --- escape_internal_link ---

def test_escape_internal_link_basic():
    assert md.escape_internal_link("Precinct V Windsor Wds 1-2") == "precinct-v-windsor-wds-1-2"


def test_escape_internal_link_removes_special_chars():
    assert md.escape_internal_link("Hello, World! (Test)") == "hello-world-test"


def test_escape_internal_link_collapses_repeated_hyphens():
    assert md.escape_internal_link("A   B") == "a-b"


# --- md_parse_link ---

def test_md_parse_link_plain_text():
    text, link = md.md_parse_link("just plain text")
    assert text == "just plain text"
    assert link == ''


def test_md_parse_link_markdown_link():
    text, link = md.md_parse_link("[My Title](https://example.com)")
    assert text == "My Title"
    assert link == "https://example.com"


# --- md_toc ---

def test_md_toc_basic():
    result = md.md_toc(['Intro', 'Methods'])
    assert result.startswith("# Table of Contents")
    assert "[Intro](#intro)" in result
    assert "[Methods](#methods)" in result


def test_md_toc_empty_list():
    result = md.md_toc([])
    assert result.startswith("# Table of Contents")


# --- clean_header_cell / normalize_colnames ---

def test_clean_header_cell_bold_stars():
    assert md.clean_header_cell("**Name**") == "Name"


def test_clean_header_cell_bold_underscores():
    assert md.clean_header_cell("__Name__") == "Name"


def test_clean_header_cell_italic_star():
    assert md.clean_header_cell("*Name*") == "Name"


def test_clean_header_cell_italic_underscore():
    assert md.clean_header_cell("_Name_") == "Name"


def test_clean_header_cell_plain():
    assert md.clean_header_cell("  Name  ") == "Name"


def test_normalize_colnames():
    assert md.normalize_colnames(["**A**", "_B_", "C"]) == ["A", "B", "C"]


# --- escape_md_table_text / escape_raw_text ---

def test_escape_md_table_text_pipe_and_newline():
    assert md.escape_md_table_text("a|b\nc") == r"a\|b<br>c"


def test_escape_raw_text_empty():
    assert md.escape_raw_text('') == ''


def test_escape_raw_text_special_chars():
    result = md.escape_raw_text(r'(test).path')
    # every char in spec_chars (includes ( ) . ) should now be backslash-escaped
    assert result == r'\(test\)\.path'


# --- md_2_html_snippet / md_2_html ---

def test_md_2_html_snippet_basic():
    result = md.md_2_html_snippet("# Heading\n\nSome text.")
    assert "<h1" in result
    assert "Some text." in result


def test_md_2_html_snippet_strip_newlines():
    result = md.md_2_html_snippet("# Heading\n\nSome text.", strip_newlines=True)
    assert '\n' not in result


def test_md_2_html_wraps_in_full_document():
    result = md.md_2_html("My Title", "# Heading")
    assert "<!DOCTYPE html>" in result
    assert "<title>My Title</title>" in result
    assert "<h1" in result


# --- find_first_markdown_table ---

def test_find_first_markdown_table_none_found():
    lines = ["just some text", "no tables here"]
    table_lines, start_idx, end_idx = md.find_first_markdown_table(lines)
    assert table_lines is None
    assert start_idx is None
    assert end_idx is None


def test_find_first_markdown_table_simple():
    lines = ["| a | b |", "| 1 | 2 |"]
    table_lines, start_idx, end_idx = md.find_first_markdown_table(lines)
    assert table_lines == lines
    assert start_idx == 0
    assert end_idx == 1


def test_find_first_markdown_table_with_leading_prose():
    lines = ["Some prose here.", "", "| a | b |", "| 1 | 2 |", "more prose"]
    table_lines, start_idx, end_idx = md.find_first_markdown_table(lines)
    assert table_lines == ["| a | b |", "| 1 | 2 |"]
    assert start_idx == 2
    assert end_idx == 3


def test_find_first_markdown_table_stops_on_inconsistent_columns():
    # second row has a different column count, so the table block ends after the first row
    lines = ["| a | b |", "| 1 | 2 | 3 |"]
    table_lines, start_idx, end_idx = md.find_first_markdown_table(lines)
    assert table_lines == ["| a | b |"]
    assert start_idx == 0
    assert end_idx == 0


# --- _parse_daf_footer ---

def test_parse_daf_footer_basic():
    meta = md._parse_daf_footer("%% daf name=mytable; keyfield=id")
    assert meta == {'name': 'mytable', 'keyfield': 'id'}


def test_parse_daf_footer_quoted_values():
    meta = md._parse_daf_footer("%% daf name='my table'")
    assert meta == {'name': 'my table'}


def test_parse_daf_footer_skips_malformed_parts():
    meta = md._parse_daf_footer("%% daf name=mytable; no_equals_sign; keyfield=id")
    assert meta == {'name': 'mytable', 'keyfield': 'id'}


# =====================================================================
# Group 2 -- table rendering via Daf.to_md() / Daf.to_md_cols()
# =====================================================================

def test_to_md_basic_table():
    daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'letter'])
    result = daf.to_md()
    assert '| id | letter |' in result
    assert '1' in result and 'a' in result
    assert '2' in result and 'b' in result


def test_to_md_empty_daf_returns_empty_string():
    daf = Daf(lol=[], cols=['id', 'letter'])
    result = daf.to_md()
    assert result == ''


def test_to_md_with_justification():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'letter'])
    result = daf.to_md(just='><')
    assert '| id | letter |' in result


def test_to_md_shorten_text():
    long_text = 'x' * 200
    daf = Daf(lol=[[1, long_text]], cols=['id', 'text'])
    result = daf.to_md(shorten_text=True, max_text_len=20)
    # the full long string should not appear unshortened
    assert long_text not in result


def test_to_md_shorten_text_disabled():
    long_text = 'x' * 200
    daf = Daf(lol=[[1, long_text]], cols=['id', 'text'])
    result = daf.to_md(shorten_text=False)
    assert long_text in result


def test_to_md_smart_fmt_numeric():
    daf = Daf(lol=[[1.23456789], [2.1]], cols=['val'])
    result = daf.to_md(smart_fmt=True)
    assert '| val |' in result


def test_to_md_include_summary():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'letter'], keyfield='id', name='mytable')
    result = daf.to_md(include_summary=True)
    assert '%% daf' in result
    assert "name='mytable'" in result
    assert "keyfield='id'" in result


def test_to_md_disp_cols_override():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'letter'])
    result = daf.to_md(disp_cols=['letter'])
    assert '| letter |' in result


def test_to_md_header_override():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'letter'])
    result = daf.to_md(header=['ID', 'Letter'])
    assert '| ID | Letter |' in result


def test_to_md_cols_basic():
    daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'letter'])
    result = daf.to_md_cols()
    # rows-as-columns: each original row becomes a column of the rendered table
    assert '1' in result and 'a' in result


def test_to_md_bare_lol_generates_spreadsheet_header():
    daf = Daf(lol=[[1, 'a'], [2, 'b']])  # no cols at all
    result = daf.to_md()
    assert '| A | B |' in result


def test_to_md_bare_lol_empty_returns_empty_string():
    daf = Daf(lol=[])
    assert daf.to_md() == ''


# =====================================================================
# Group 3 -- Daf.from_md() parsing
# =====================================================================

def test_from_md_round_trip():
    original = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'letter'])
    md_str = original.to_md()
    restored = Daf.from_md(md_str)
    assert restored.lol == [['1', 'a'], ['2', 'b']]
    assert list(restored.hd.keys()) == ['id', 'letter']


def test_from_md_round_trip_bare_lol_no_header():
    original = Daf(lol=[[1, 'a'], [2, 'b']])  # no cols
    md_str = original.to_md()
    restored = Daf.from_md(md_str)
    assert restored.lol == [['1', 'a'], ['2', 'b']]
    assert list(restored.hd.keys()) == ['A', 'B']


def test_from_md_with_header_and_separator():
    md_str = (
        "| id | letter |\n"
        "| --- | --- |\n"
        "| 1 | a |\n"
        "| 2 | b |\n"
    )
    daf = Daf.from_md(md_str)
    assert list(daf.hd.keys()) == ['id', 'letter']
    assert daf.lol == [['1', 'a'], ['2', 'b']]


def test_from_md_no_header_raises():
    md_str = (
        "| 1 | a |\n"
        "| 2 | b |\n"
    )
    with pytest.raises(RuntimeError):
        Daf.from_md(md_str)


def test_from_md_no_table_found_returns_empty_daf():
    daf = Daf.from_md("just some prose, no table here")
    assert daf.lol == []


def test_from_md_empty_string_returns_empty_daf():
    daf = Daf.from_md('')
    assert daf.lol == []


def test_from_md_footer_metadata():
    md_str = (
        "| id | letter |\n"
        "| --- | --- |\n"
        "| 1 | a |\n"
        "\n"
        "%% daf name=mytable; keyfield=id\n"
    )
    daf = Daf.from_md(md_str)
    assert daf.name == 'mytable'
    assert daf.keyfield == 'id'


def test_from_md_footer_metadata_with_no_table():
    md_str = (
        "no table here\n"
        "%% daf name=mytable; keyfield=id\n"
    )
    daf = Daf.from_md(md_str)
    assert daf.lol == []
    assert daf.name == 'mytable'
    assert daf.keyfield == 'id'


def test_from_md_abbreviated_table_raises():
    original = Daf(lol=[[i] for i in range(10)], cols=['n'])
    md_str = original.to_md(max_rows=4, include_summary=True)
    with pytest.raises(RuntimeError):
        Daf.from_md(md_str)


@pytest.mark.xfail(
    reason=(
        "The 'Inconsistent column count' RuntimeError in _from_md (after header/separator "
        "detection) appears unreachable via the public API: find_first_markdown_table already "
        "truncates a table block at the first row whose column count differs from the block's "
        "first row, so by the time _from_md validates row lengths, all rows are already uniform. "
        "Left in place as defensive code; this test documents that it's currently dead."
    ),
    strict=True,
)
def test_from_md_inconsistent_column_count_raises():
    md_str = (
        "| id | letter |\n"
        "| --- | --- |\n"
        "| 1 | a | extra |\n"
    )
    with pytest.raises(RuntimeError):
        Daf.from_md(md_str)


def test_from_md_cleans_bold_header_cells():
    md_str = (
        "| **id** | _letter_ |\n"
        "| --- | --- |\n"
        "| 1 | a |\n"
    )
    daf = Daf.from_md(md_str)
    assert list(daf.hd.keys()) == ['id', 'letter']


def test_from_md_leading_prose_before_table():
    md_str = (
        "# Some Heading\n"
        "\n"
        "Some explanatory prose.\n"
        "\n"
        "| id | letter |\n"
        "| --- | --- |\n"
        "| 1 | a |\n"
    )
    daf = Daf.from_md(md_str)
    assert daf.lol == [['1', 'a']]


# =====================================================================
# Group 4 -- md_lol_table / md_cols_lol_table direct branch coverage
# =====================================================================

def test_md_lol_table_empty_records_returns_empty_string():
    assert md.md_lol_table([]) == ''


def test_md_lol_table_explicit_header():
    result = md.md_lol_table([[1, 'a']], header=['id', 'letter'])
    assert '| id | letter |' in result


def test_md_lol_table_no_header_omitted():
    result = md.md_lol_table([[1, 'a']], omit_header=True)
    assert result != ''
    assert '1' in result and 'a' in result


def test_md_lol_table_include_idx():
    result = md.md_lol_table([[1, 'a'], [2, 'b']], header=['id', 'letter'], include_idx=True)
    assert '| idx | id | letter |' in result
    lines = result.splitlines()
    assert any('0' in line and '1' in line for line in lines[2:])


def test_md_lol_table_ragged_raises_runtime_error():
    with pytest.raises(RuntimeError):
        md.md_lol_table([[1, 2, 3], [4, 5]], header=['a', 'b', 'c'])


def test_md_cols_lol_table_empty_returns_empty_string():
    assert md.md_cols_lol_table([]) == ''


def test_md_cols_lol_table_omit_header_true():
    result = md.md_cols_lol_table([[1, 2], ['a', 'b']], omit_header=True)
    assert result != ''
    assert '1' in result


def test_md_cols_lol_table_missing_header_raises():
    with pytest.raises(RuntimeError):
        md.md_cols_lol_table([[1, 2], ['a', 'b']])


def test_md_cols_lol_table_with_header():
    result = md.md_cols_lol_table([[1, 2], ['a', 'b']], header=['nums', 'letters'])
    assert '| nums | letters |' in result


def test_md_cols_lol_table_ragged_columns_padded():
    # second column is shorter than the first; missing cells should be padded with ''
    result = md.md_cols_lol_table([[1, 2, 3], ['a']], header=['nums', 'letters'])
    lines = result.splitlines()
    assert len(lines) == 5  # header + ruling + 3 data rows
