# test_daf_construction.py
#
# Tests for Daf.__init__, copy, from_lot, set_dtypes/apply_dtypes, append/_basic_append,
# insert_dif_row, and the CSV I/O round trip (from_csv, from_csv_file, to_csv_file, to_csv_buff,
# buff_to_file). These are the entry points most heavily exercised via load_data()/save_data()
# style usage in downstream projects.

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import requests

from daffodil.daf import Daf
from daffodil.keyedlist import KeyedList
import csv
from daffodil.lib.daf_ods import ods_sheet_names, ods_to_lol


# =====================================================================
# __init__
# =====================================================================

def test_init_cols_as_single_string():
    daf = Daf(cols='single_col_as_str')
    assert list(daf.hd.keys()) == ['single_col_as_str']


def test_init_disp_cols_invalid_type_raises():
    with pytest.raises(TypeError):
        Daf(disp_cols=123)


def test_init_hd_and_kd_adopted_directly():
    daf = Daf(lol=[[1, 2]], hd={'a': 0, 'b': 1}, kd={'x': 0})
    assert daf.hd == {'a': 0, 'b': 1}
    assert daf._kd == {'x': 0}


def test_init_cols_from_dtypes_when_no_cols_given():
    daf = Daf(dtypes={'a': int, 'b': str})
    assert list(daf.hd.keys()) == ['a', 'b']


def test_init_use_copy_deep_copies_lol():
    original_lol = [[1, 2]]
    daf = Daf(lol=original_lol, use_copy=True)
    daf.lol[0][0] = 99
    assert original_lol[0][0] == 1


# =====================================================================
# copy
# =====================================================================

def test_copy_shallow_independent_attrs():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    daf.attrs = {'meta': 'x'}
    copied = daf.copy()
    copied.attrs['meta'] = 'y'
    assert daf.attrs == {'meta': 'x'}
    assert copied.attrs == {'meta': 'y'}


def test_copy_deep_creates_independent_lol():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    copied = daf.copy(deep=True)
    assert copied.lol is not daf.lol
    copied.lol[0][0] = 99
    assert daf.lol[0][0] == 1


def test_copy_for_sorting_independent_lol_reference():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    copied = daf.copy(for_sorting=True)
    assert copied.lol is not daf.lol


# =====================================================================
# from_lot
# =====================================================================

def test_from_lot_basic():
    lot = [(1, 'Alice', 30), (2, 'Bob', 25)]
    daf = Daf.from_lot(lot)
    assert daf.columns() == []
    assert daf.lol == [[1, 'Alice', 30], [2, 'Bob', 25]]


def test_from_lot_with_cols():
    lot = [(1, 'Alice'), (2, 'Bob')]
    daf = Daf.from_lot(lot, cols=['id', 'name'])
    assert list(daf.hd.keys()) == ['id', 'name']


def test_from_lot_empty_returns_empty_daf():
    daf = Daf.from_lot([])
    assert daf.lol == []


# =====================================================================
# set_dtypes / apply_dtypes
# =====================================================================

def test_set_dtypes_with_typ_to_cols_dict():
    daf = Daf(lol=[[1, 'a', 2.5]], cols=['id', 'name', 'val'])
    daf.set_dtypes(default_type=str, typ_to_cols_dict={int: ['id'], float: ['val']})
    assert daf.dtypes == {'id': int, 'name': str, 'val': float}


def test_set_dtypes_no_hd_raises():
    with pytest.raises(NotImplementedError):
        Daf().set_dtypes()


def test_apply_dtypes_mismatch_raises():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    with pytest.raises(ValueError):
        daf.apply_dtypes(dtypes={'id': int}, silent_error=False)


def test_apply_dtypes_converts_columns():
    daf = Daf(lol=[['1', '2.5']], cols=['id', 'val'])
    daf.apply_dtypes(dtypes={'id': int, 'val': float})
    assert daf.lol == [[1, 2.5]]


def test_apply_dtypes_no_dtypes_is_noop():
    daf = Daf(lol=[['1', 'a']], cols=['id', 'name'])
    daf.apply_dtypes()
    assert daf.lol == [['1', 'a']]


def test_apply_dtypes_empty_lol_is_noop():
    daf = Daf(lol=[], cols=['id', 'name'])
    daf.apply_dtypes(dtypes={'id': int, 'name': str})
    assert daf.lol == []


# =====================================================================
# append
# =====================================================================

def test_append_falsy_is_noop():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    daf.append(None)
    daf.append([])
    assert daf.lol == [[1, 'a']]


def test_append_dict():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    daf.append({'id': 2, 'name': 'b'})
    assert daf.lol == [[1, 'a'], [2, 'b']]


def test_append_list_of_dicts():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    daf.append([{'id': 2, 'name': 'b'}, {'id': 3, 'name': 'c'}])
    assert daf.lol == [[1, 'a'], [2, 'b'], [3, 'c']]


def test_append_simple_list_no_hd():
    daf = Daf()
    daf.append([1, 'x'])
    assert daf.lol == [[1, 'x']]
    assert daf.hd == {}


def test_append_simple_list_with_hd():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    daf.append([5, 'e'])
    assert daf.lol == [[1, 'a'], [5, 'e']]


def test_append_keyedlist_matching_hd():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    kl = KeyedList(['id', 'name'], [6, 'f'])
    daf.append(kl)
    assert daf.lol == [[1, 'a'], [6, 'f']]


def test_append_keyedlist_non_matching_hd():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    kl = KeyedList(['name', 'id'], ['g', 7])
    daf.append(kl)
    assert daf.lol == [[1, 'a'], [7, 'g']]


def test_append_daf_concat():
    daf1 = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    daf2 = Daf(lol=[[2, 'b']], cols=['id', 'name'])
    daf1.append(daf2)
    assert daf1.lol == [[1, 'a'], [2, 'b']]


def test_append_unsupported_type_raises():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    with pytest.raises(TypeError, match='append'):
        daf.append(42)


# =====================================================================
# _basic_append
# =====================================================================

def test_basic_append_list_no_hd():
    daf = Daf()
    daf._basic_append([1, 'x'])
    assert daf.lol == [[1, 'x']]


def test_basic_append_dict_no_hd_adopts_structure():
    daf = Daf()
    daf._basic_append({'a': 1, 'b': 2})
    assert daf.hd == {'a': 0, 'b': 1}
    assert daf.lol == [[1, 2]]


def test_basic_append_dict_with_hd_aligns_order():
    daf = Daf(lol=[[1, 2]], cols=['a', 'b'])
    daf._basic_append({'b': 5, 'a': 3})
    assert daf.lol == [[1, 2], [3, 5]]


def test_basic_append_unsupported_type_raises():
    with pytest.raises(TypeError):
        Daf()._basic_append(42)


# =====================================================================
# insert_dif_row / insert_irow (working list/dict paths)
# =====================================================================

def test_insert_dif_row_default_irow2():
    daf = Daf(lol=[[1, 10], [2, 15], [3, 30]], cols=['id', 'val'])
    daf.insert_dif_row(irow1=0)
    assert daf.lol == [[1, 10], [-1, -5], [2, 15], [3, 30]]


def test_insert_dif_row_explicit_cols():
    daf = Daf(lol=[[1, 10], [2, 15], [3, 30]], cols=['id', 'val'])
    daf.insert_dif_row(irow1=0, irow2=2, irow_insert=1, cols=['val'])
    assert daf.lol == [[1, 10], ['', -20], [2, 15], [3, 30]]


def test_insert_irow_with_list():
    daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'name'])
    daf.insert_irow(irow=1, row=[99, 'z'])
    assert daf.lol == [[1, 'a'], [99, 'z'], [2, 'b']]


def test_insert_irow_with_dict():
    daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'name'])
    daf.insert_irow(irow=0, row={'id': 100, 'name': 'y'})
    assert daf.lol == [[100, 'y'], [1, 'a'], [2, 'b']]


# =====================================================================
# CSV I/O: from_csv, from_csv_file, to_csv_file, to_csv_buff, buff_to_file
# =====================================================================

def test_to_csv_buff_basic():
    daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'name'])
    buff = daf.to_csv_buff()
    assert buff == 'id,name\r\n1,a\r\n2,b\r\n'


def test_to_csv_buff_no_header():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    buff = daf.to_csv_buff(include_header=False)
    assert buff == '1,a\r\n'


def test_round_trip_local_file_basic():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'out.csv')
        daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'name'])
        result_path = daf.to_csv_file(p)
        assert result_path == p
        restored = Daf.from_csv(p)
        assert restored.lol == [['1', 'a'], ['2', 'b']]
        assert list(restored.hd.keys()) == ['id', 'name']


def test_round_trip_local_file_with_dtypes():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'out.csv')
        daf = Daf(lol=[[1, 2.5], [3, 4.5]], cols=['a', 'b'], dtypes={'a': int, 'b': float})
        daf.to_csv_file(p)
        restored = Daf.from_csv_buff(open(p).read(), dtypes={'a': int, 'b': float})
        assert restored.lol == [[1, 2.5], [3, 4.5]]


def test_round_trip_complex_cell_via_pyon():
    daf = Daf(lol=[[1, [1, 2, 3]]], cols=['id', 'items'])
    buff = daf.to_csv_buff()
    restored = Daf.from_csv_buff(buff, dtypes={'id': int, 'items': list})
    assert restored.lol == [[1, [1, 2, 3]]]


def test_from_csv_accepts_path_object():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'test.csv'
        p.write_text('a,b\n1,2\n')
        daf = Daf.from_csv(p)
        assert daf.lol == [['1', '2']]


def test_from_csv_local_file_not_found_raises_runtimeerror():
    with pytest.raises(RuntimeError):
        Daf.from_csv('/tmp/this_does_not_exist_xyz_daffodil_test.csv')


def test_from_csv_http_source(monkeypatch):
    mock_response = MagicMock()
    mock_response.iter_lines.return_value = [b'a,b', b'1,2', b'3,4']
    mock_response.raise_for_status.return_value = None

    with patch('requests.get', return_value=mock_response) as mock_get:
        daf = Daf.from_csv('https://example.com/data.csv')
        assert daf.lol == [['1', '2'], ['3', '4']]
        assert list(daf.hd.keys()) == ['a', 'b']
        mock_get.assert_called_once_with('https://example.com/data.csv', stream=True)


def test_from_csv_http_request_exception_raises_runtimeerror():
    with patch('requests.get', side_effect=requests.RequestException('connection failed')):
        with pytest.raises(RuntimeError):
            Daf.from_csv('https://example.com/data.csv')


def test_from_csv_s3_source_without_boto3_raises_runtimeerror():
    # boto3 is not installed in this environment; confirms the ImportError->RuntimeError path.
    with pytest.raises(RuntimeError):
        Daf.from_csv('s3://bucket/key.csv')


def test_buff_to_file_writes_local_file():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'out.csv')
        result = Daf.buff_to_file('a,b\n1,2\n', file_path=p)
        assert result == p
        assert open(p).read() == 'a,b\n1,2\n'


# =====================================================================
# from_csv_file (deprecated; it calls from_csv)
# =====================================================================

def test_from_csv_file_basic():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'out.csv')
        with open(p, 'w') as f:
            f.write('id,name\n1,a\n')
        daf = Daf.from_csv_file(p)
        assert daf.lol == [['1', 'a']]


def test_from_csv_file_missing_file_raises_runtimeerror():
    with pytest.raises(RuntimeError, match='Failed to read local file'):
        Daf.from_csv_file('/tmp/this_does_not_exist_xyz_daffodil_test.csv')


def test_from_csv_file_reads_utf8_and_gives_the_same_result_as_from_csv():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'u.csv')
        with open(p, 'w', encoding='utf-8') as f:
            f.write('id,name\n1,Zo\u00eb\n2,\u00c5ngstr\u00f6m\n')
        assert Daf.from_csv_file(p).lol == [['1', 'Zo\u00eb'], ['2', '\u00c5ngstr\u00f6m']]
        assert Daf.from_csv_file(p, keyfield='id', name='n').lol == Daf.from_csv(p, keyfield='id', name='n').lol
        assert Daf.from_csv_file(p, keyfield='id', name='n').keyfield == 'id'


def test_from_csv_file_passes_its_arguments_on():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 's.csv')
        with open(p, 'w', encoding='utf-8') as f:
            f.write('id;name;n\n1;a;5\n')
        daf = Daf.from_csv_file(p, sep=';', include_cols=['n', 'id'], dtypes={'n': int, 'id': int})
        assert daf.columns() == ['n', 'id'] and daf.lol == [[5, 1]]


# ---- from test_daf_csv_empty.py ----
# An empty CSV source gives an empty Daf with no columns, as from_md('') does.

@pytest.mark.parametrize('source', ['', b'', '\n', '\n\n'])
def test_from_csv_buff_of_empty_text_gives_an_empty_daf(source):
    d = Daf.from_csv_buff(source)
    assert d.lol == []
    assert d.columns() == []
    assert len(d) == 0


def test_from_csv_buff_of_an_empty_iterator_gives_an_empty_daf():
    d = Daf.from_csv_buff(iter([]))
    assert d.lol == [] and d.columns() == []


def test_from_csv_buff_noheader_of_empty_text_is_still_an_empty_daf():
    assert Daf.from_csv_buff('', noheader=True).lol == []


def test_from_csv_buff_keeps_the_name_for_empty_text():
    assert Daf.from_csv_buff('', name='nm').name == 'nm'


def test_from_csv_buff_accepts_keyfield_and_include_cols_for_empty_text():
    assert len(Daf.from_csv_buff('', keyfield='id')) == 0
    assert len(Daf.from_csv_buff('', include_cols=['id'])) == 0


def test_from_csv_of_an_empty_file_gives_an_empty_daf(tmp_path):
    path = tmp_path / 'empty.csv'
    path.write_text('')
    d = Daf.from_csv(path)
    assert d.lol == [] and d.columns() == []


def test_a_header_with_no_rows_keeps_its_columns():
    d = Daf.from_csv_buff('id,v\n')
    assert d.columns() == ['id', 'v'] and d.lol == []


def test_a_normal_file_is_read_as_before():
    d = Daf.from_csv_buff('id,v\n1,a\n2,b\n')
    assert d.columns() == ['id', 'v']
    assert d.lol == [['1', 'a'], ['2', 'b']]


# ---- from test_daf_from_ods.py ----
# Daf.from_ods() reads one sheet of an .ods file with the standard library. The files in
# tests/ods_fixtures are made by tests/make_ods_fixtures.py. patterns.ods holds the markup
# LibreOffice can write, and patterns_expected.csv the text expected from each cell.

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


# ---- from test_daf_from_directory.py ----
# Tests for Daf.from_directory(): harvesting filesystem metadata into a Daf, using pytest's
# built-in tmp_path fixture to create real files/directories.

def test_from_directory_recursive_default(tmp_path):
    (tmp_path / 'a.txt').write_text('hello')
    (tmp_path / 'b.csv').write_text('x,y')
    sub = tmp_path / 'sub'
    sub.mkdir()
    (sub / 'c.txt').write_text('world!')

    result = Daf.from_directory(tmp_path)

    assert list(result.hd.keys()) == [
        'filepath', 'dirpath', 'basename', 'rootname', 'extension',
        'size', 'mtime', 'ctime', 'is_dir',
    ]
    assert result.num_rows() == 3
    basenames = sorted(row['basename'] for row in result)
    assert basenames == ['a.txt', 'b.csv', 'c.txt']


def test_from_directory_populates_fields_correctly(tmp_path):
    (tmp_path / 'a.txt').write_text('hello')

    result = Daf.from_directory(tmp_path)
    row = next(iter(result))

    assert row['basename'] == 'a.txt'
    assert row['rootname'] == 'a'
    assert row['extension'] == '.txt'
    assert row['size'] == 5
    assert row['is_dir'] == 0
    assert row['dirpath'] == str(tmp_path).replace('\\', '/')


def test_from_directory_non_recursive_excludes_subdirectories():
    # this is the bug we found and fixed: os.listdir() returns subdirectory names alongside
    # file names, and previously the non-recursive path didn't distinguish them, so a
    # subdirectory could show up as if it were a file.
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, 'a.txt'), 'w') as f:
            f.write('hello')
        with open(os.path.join(d, 'b.csv'), 'w') as f:
            f.write('x,y')
        os.mkdir(os.path.join(d, 'sub'))

        result = Daf.from_directory(d, recursive=False)
        basenames = sorted(row['basename'] for row in result)
        assert basenames == ['a.txt', 'b.csv']
        assert result.num_rows() == 2


def test_from_directory_recursive_finds_nested_files_not_found_non_recursively(tmp_path):
    (tmp_path / 'a.txt').write_text('hello')
    sub = tmp_path / 'sub'
    sub.mkdir()
    (sub / 'c.txt').write_text('world!')

    recursive_result = Daf.from_directory(tmp_path, recursive=True)
    assert recursive_result.num_rows() == 2

    non_recursive_result = Daf.from_directory(tmp_path, recursive=False)
    assert non_recursive_result.num_rows() == 1
    assert non_recursive_result.col('basename') == ['a.txt']


def test_from_directory_file_pat_filters():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        import os
        with open(os.path.join(d, 'a.txt'), 'w') as f:
            f.write('hello')
        with open(os.path.join(d, 'b.csv'), 'w') as f:
            f.write('x,y')

        result = Daf.from_directory(d, file_pat=r'\.txt$')
        assert result.num_rows() == 1
        assert result.col('basename') == ['a.txt']


def test_from_directory_empty_directory(tmp_path):
    result = Daf.from_directory(tmp_path)
    assert result.num_rows() == 0


def test_from_directory_custom_schema(tmp_path):
    from daffodil.lib.schemaclass import schemaclass, SchemaBase

    @schemaclass
    class MySchema(SchemaBase):
        filepath: str = ''
        basename: str = ''
        extension: str = ''
        custom_field: str = 'default_val'

    (tmp_path / 'a.txt').write_text('hi')

    result = Daf.from_directory(tmp_path, schema=MySchema)
    assert list(result.hd.keys()) == ['filepath', 'basename', 'extension', 'custom_field']
    row = next(iter(result))
    assert row['custom_field'] == 'default_val'
    assert row['basename'] == 'a.txt'


def test_from_directory_accepts_path_object(tmp_path):
    (tmp_path / 'a.txt').write_text('hi')
    result = Daf.from_directory(tmp_path)  # tmp_path is already a Path object
    assert result.num_rows() == 1


# include_dirs, and no printing

def _make_tree(tmp_path):
    (tmp_path / 'a.txt').write_text('hi')
    (tmp_path / 'v1.2').mkdir()
    (tmp_path / 'v1.2' / 'b.csv').write_text('yo!')
    return tmp_path


def test_from_directory_does_not_print(tmp_path, capsys):
    Daf.from_directory(_make_tree(tmp_path))
    assert capsys.readouterr().out == ''


def test_from_directory_default_lists_only_files_with_is_dir_zero(tmp_path):
    result = Daf.from_directory(_make_tree(tmp_path))
    assert sorted(result.col('basename')) == ['a.txt', 'b.csv']
    assert set(result.col('is_dir')) == {0}


def test_from_directory_include_dirs_recursive(tmp_path):
    result = Daf.from_directory(_make_tree(tmp_path), include_dirs=True)
    rows = {rec['basename']: rec for rec in result.to_lod()}
    assert sorted(rows) == ['a.txt', 'b.csv', 'v1.2']
    assert rows['v1.2']['is_dir'] == 1
    assert rows['v1.2']['size'] == 0
    assert rows['v1.2']['extension'] == ''
    assert rows['v1.2']['rootname'] == 'v1.2'
    assert rows['b.csv']['is_dir'] == 0 and rows['b.csv']['size'] == 3


def test_from_directory_include_dirs_lists_folders_before_files_in_same_folder(tmp_path):
    result = Daf.from_directory(_make_tree(tmp_path), include_dirs=True)
    top = [rec['basename'] for rec in result.to_lod() if rec['dirpath'] == tmp_path.as_posix()]
    assert top == ['v1.2', 'a.txt']


def test_from_directory_include_dirs_not_recursive(tmp_path):
    result = Daf.from_directory(_make_tree(tmp_path), recursive=False, include_dirs=True)
    assert sorted(result.col('basename')) == ['a.txt', 'v1.2']


def test_from_directory_not_recursive_without_include_dirs_skips_folders(tmp_path):
    result = Daf.from_directory(_make_tree(tmp_path), recursive=False)
    assert result.col('basename') == ['a.txt']


def test_from_directory_file_pat_applies_to_folder_names(tmp_path):
    result = Daf.from_directory(_make_tree(tmp_path), include_dirs=True, file_pat=r'^v1')
    assert result.col('basename') == ['v1.2']
