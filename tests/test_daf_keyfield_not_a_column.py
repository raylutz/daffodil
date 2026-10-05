# test_daf_keyfield_not_a_column.py
#
# A keyfield that is not a column raises KeyError in the constructor, the builders and set_keyfield(). A Daf with no column
# names can still have one, for the columns that come later. If the attribute is edited directly, the failure is reported
# when a lookup by key is made, and the message says what is wrong.

import pytest

from daffodil.daf import Daf, KeysDisabledError


def test_the_constructor_and_the_builders_raise_keyerror_for_a_keyfield_that_is_not_a_column():
    with pytest.raises(KeyError, match=r"keyfield 'zz' is not a column. The columns are \['a', 'b'\]"):
        Daf(lol=[[1, 2]], cols=['a', 'b'], keyfield='zz')
    with pytest.raises(KeyError, match="keyfield 'zz'"):
        Daf.from_csv_buff('a,b\n1,2\n', keyfield='zz')
    with pytest.raises(KeyError, match="is not a column"):
        Daf(lol=[[1, 2]], cols=['a', 'b'], keyfield=('a', 'zz'))


def test_set_keyfield_raises_keyerror_by_default_and_stores_only_if_asked():
    d = Daf(lol=[[1, 2]], cols=['a', 'b'])
    with pytest.raises(KeyError, match="set_keyfield.*keyfield 'zz'"):
        d.set_keyfield('zz')
    assert d.keyfield == ''
    d.set_keyfield('zz', silent_error=True)
    assert d.keyfield == 'zz'


def test_an_empty_daf_with_no_columns_can_have_a_keyfield_for_the_columns_that_come_later():
    d = Daf(keyfield='id')
    d.append({'id': 1, 'v': 'a'})
    assert d.columns() == ['id', 'v'] and d.keyfield == 'id' and d.keys() == [1]


def test_select_krows_names_a_keyfield_that_is_not_a_column_after_a_direct_edit():
    d = Daf(lol=[[1, 2], [3, 4]], cols=['a', 'b'])
    d.keyfield = 'zz'
    with pytest.raises(KeysDisabledError, match=r"keyfield 'zz' is not a column .*\['a', 'b'\].*set_keyfield"):
        d.select_krows([1])


def test_select_krows_names_a_composite_keyfield_with_a_bad_column_after_a_direct_edit():
    d = Daf(lol=[[1, 2]], cols=['a', 'b'])
    d.keyfield = ('a', 'zz')
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


def test_no_keyfield_and_no_kd_says_so():
    with pytest.raises(KeysDisabledError, match=r"select_krows\(\): key lookups are disabled, as the keyfield is not set and there is no key index"):
        Daf(lol=[[1, 2]], cols=['a', 'b']).select_krows([1])


def test_set_keyfield_with_silent_error_false_raises_keyerror_for_a_bad_column():
    d = Daf(lol=[[1, 2]], cols=['a', 'b'])
    with pytest.raises(KeyError):
        d.set_keyfield('zz', silent_error=False)
