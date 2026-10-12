# test_daf_setitem_row_key_hint.py
#
# daf['name'] = ... selects a row by key. When the name is not a row key, the error says
# so, and points to daf[:, 'name'] and assign_col() for a column. Only the error path is
# changed. These tests check the messages, and that other errors are as before.

import pytest
from daffodil.daf import Daf, KeysDisabledError


def _keyed() -> Daf:
    return Daf(lol=[['r1', 1], ['r2', 2]], cols=['k', 'x'], keyfield='k')


HINT = r"daf\[name\] selects a row\. For a column use daf\[:, 'flagged'\], or assign_col\('flagged', values\) to add one\."


def test_not_a_row_key() -> None:
    daf = _keyed()
    with pytest.raises(KeyError, match=r"'flagged' is not a row key\. " + HINT) as exc_info:
        daf['flagged'] = [True, False]
    assert type(exc_info.value.__cause__) is KeyError
    assert daf.lol == [['r1', 1], ['r2', 2]]


def test_column_name_that_is_not_a_row_key() -> None:
    daf = Daf(lol=[['r1', 1], ['r2', 2]], cols=['k', 'flagged'], keyfield='k')
    with pytest.raises(KeyError, match=HINT):
        daf['flagged'] = True


def test_no_keyfield() -> None:
    daf = Daf(lol=[['r1', 1], ['r2', 2]], cols=['k', 'x'])
    with pytest.raises(KeysDisabledError, match=r'key lookups are disabled.*' + HINT) as exc_info:
        daf['flagged'] = [True, False]
    assert isinstance(exc_info.value.__cause__, KeysDisabledError)


def test_row_key_still_works() -> None:
    daf = _keyed()
    daf['r2'] = ['r2', 20]
    assert daf.lol == [['r1', 1], ['r2', 20]]


def test_what_the_hint_suggests_works() -> None:
    daf = _keyed()
    daf[:, 'x'] = [5, 6]
    daf.assign_col('flagged', [True, False])
    assert daf.lol == [['r1', 5, True], ['r2', 6, False]]


def test_other_selectors_unchanged() -> None:
    """ A list of keys, a key with a column, or a read, gives the error it gave before. """
    daf = _keyed()
    with pytest.raises(KeyError) as exc_info:
        daf[['zz']] = [1, 2]
    assert 'selects a row' not in str(exc_info.value)
    with pytest.raises(KeyError) as exc_info:
        daf['zz', 'x'] = 1
    assert 'selects a row' not in str(exc_info.value)
    with pytest.raises(KeyError) as exc_info:
        daf['zz']
    assert 'selects a row' not in str(exc_info.value)
