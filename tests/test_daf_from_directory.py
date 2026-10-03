# test_daf_from_directory.py
#
# Tests for Daf.from_directory(): harvesting filesystem metadata into a Daf, using pytest's
# built-in tmp_path fixture to create real files/directories.

from daffodil.daf import Daf


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
