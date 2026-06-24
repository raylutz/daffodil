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
