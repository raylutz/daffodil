# test_daf_csv_empty.py
#
# An empty CSV source gives an empty Daf with no columns, as from_md('') does.

import pytest

from daffodil.daf import Daf


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
