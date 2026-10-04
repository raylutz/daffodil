# test_daf_keyfield_not_a_column.py
#
# A keyfield that is not a column is stored without an error, as set_keyfield() documents. The failure is reported when a
# lookup by key is made, and the message says what is wrong.

import pytest

from daffodil.daf import Daf, KeysDisabledError


def test_the_builders_store_a_keyfield_that_is_not_a_column():
    assert Daf.from_csv_buff('a,b\n1,2\n', keyfield='zz').keyfield == 'zz'
    assert Daf(lol=[[1, 2]], cols=['a', 'b'], keyfield='zz').keyfield == 'zz'


def test_select_krows_names_a_keyfield_that_is_not_a_column():
    d = Daf.from_csv_buff('a,b\n1,2\n3,4\n', keyfield='zz')
    with pytest.raises(KeysDisabledError, match=r"keyfield 'zz' is not a column .*\['a', 'b'\].*set_keyfield"):
        d.select_krows(['1'])


def test_select_krows_names_a_composite_keyfield_with_a_bad_column():
    d = Daf(lol=[[1, 2]], cols=['a', 'b'], keyfield=('a', 'zz'))
    with pytest.raises(KeysDisabledError, match=r"keyfield \('a', 'zz'\) is not a column"):
        d.select_krows([(1, 2)])


def test_select_krows_names_a_keyfield_on_a_daf_with_no_column_names():
    d = Daf(lol=[[1, 2]], keyfield='a')
    with pytest.raises(KeysDisabledError, match="has no column names.*set_cols"):
        d.select_krows([1])


def test_a_valid_keyfield_still_works_and_still_raises_keyerror_for_a_missing_key():
    d = Daf(lol=[[1, 2], [3, 4]], cols=['a', 'b'], keyfield='a')
    assert d.select_krows([3]).lol == [[3, 4]]
    with pytest.raises(KeyError):
        d.select_krows([9])


def test_no_keyfield_still_gives_the_old_message():
    with pytest.raises(KeysDisabledError, match='requires keyfield is set'):
        Daf(lol=[[1, 2]], cols=['a', 'b']).select_krows([1])


def test_set_keyfield_with_silent_error_false_raises_keyerror_for_a_bad_column():
    d = Daf(lol=[[1, 2]], cols=['a', 'b'])
    with pytest.raises(KeyError):
        d.set_keyfield('zz', silent_error=False)
