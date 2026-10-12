# test_daf_from_ods.py
#
# Daf.from_ods() reads one sheet of an .ods file with the standard library. The files in
# tests/ods_fixtures are made by tests/make_ods_fixtures.py. patterns.ods holds the markup
# LibreOffice can write, and patterns_expected.csv the text expected from each cell.

import csv
from pathlib import Path

import pytest
from daffodil.daf import Daf
from daffodil.lib.daf_ods import ods_sheet_names, ods_to_lol

FIXTURES = Path(__file__).parent / 'ods_fixtures'
PATTERNS = FIXTURES / 'patterns.ods'
LAYOUT   = FIXTURES / 'layout.ods'


def _expected_patterns() -> list:
    with open(FIXTURES / 'patterns_expected.csv', newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    return [(row[:5], row[5]) for row in rows[1:]]


@pytest.mark.parametrize('irow', range(len(_expected_patterns())), ids=[what for _, what in _expected_patterns()])
def test_each_pattern_reads_as_its_content(irow: int) -> None:
    daf = Daf.from_ods(PATTERNS)
    expected, _what = _expected_patterns()[irow]
    assert daf.lol[irow] == expected


def test_patterns_header() -> None:
    assert Daf.from_ods(PATTERNS).columns() == ['PartialDescription', 'Amount', 'Name', 'Account', 'LookupMemo']


def test_layout_first_sheet_by_default() -> None:
    assert ods_to_lol(LAYOUT) == [
        ['id',   'when',       'ok',   '',   'rate', ''],
        ['a1',   '2026-09-01', 'true', '',   '0.05', ''],
        ['same', 'x',          '',     '',   '',     ''],
        ['same', 'x',          '',     '',   '',     ''],
        ['',     '',           '',     '',   '',     ''],
        ['',     '',           '',     '',   '',     ''],
        ['rep',  'rep',        'rep',  '',   '',     ''],
        ['wide', '',           '',     '',   '',     'past the header'],
    ]


def test_repeated_rows_are_separate_lists() -> None:
    rows = ods_to_lol(LAYOUT)
    rows[2][0] = 'changed'
    assert rows[3][0] == 'same'
    assert rows[4] is not rows[5]


def test_blank_header_names_are_filled_in() -> None:
    daf = Daf.from_ods(LAYOUT)
    assert daf.columns() == ['id', 'when', 'ok', 'Unnamed3', 'rate', 'Unnamed5']
    assert daf.num_rows() == 7


@pytest.mark.parametrize('sheet', ['Second', 1, -1])
def test_other_sheet(sheet: int | str) -> None:
    daf = Daf.from_ods(LAYOUT, sheet=sheet, keyfield='k', dtypes={'k': str, 'v': int})
    daf.retmode = 'val'
    assert daf.lol == [['b1', 2]]
    assert daf['b1', 'v'] == 2


def test_sheet_names() -> None:
    assert ods_sheet_names(LAYOUT) == ['First', 'Second']


def test_bad_sheet() -> None:
    with pytest.raises(IndexError, match='there is no sheet 2. The file has 2 sheets'):
        Daf.from_ods(LAYOUT, sheet=2)
    with pytest.raises(KeyError, match=r"there is no sheet named 'Third'. The sheets are \['First', 'Second'\]"):
        Daf.from_ods(LAYOUT, sheet='Third')


def test_dtypes_convert_stored_values() -> None:
    dtypes = {'id': str, 'when': str, 'ok': bool, 'Unnamed3': str, 'rate': float, 'Unnamed5': str}
    daf = Daf.from_ods(LAYOUT, dtypes=dtypes)
    assert daf.lol[0][:5] == ['a1', '2026-09-01', 1, '', 0.05]


def test_source_as_bytes_or_str() -> None:
    from_path = Daf.from_ods(PATTERNS).lol
    assert Daf.from_ods(PATTERNS.read_bytes()).lol == from_path
    assert Daf.from_ods(str(PATTERNS)).lol == from_path


def test_noheader_and_name() -> None:
    daf = Daf.from_ods(LAYOUT, sheet='Second', noheader=True, name='second')
    assert daf.lol == [['k', 'v'], ['b1', '2']]
    assert daf.name == 'second'
