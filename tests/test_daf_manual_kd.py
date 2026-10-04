# test_daf_manual_kd.py
#
# A Daf may have a key index (kd) and no keyfield. The key lookups all honor it, and the messages say what is missing.

import pytest

from daffodil.daf import Daf, KeysDisabledError


def _daf() -> Daf:
    return Daf(lol=[['x', 1], ['y', 2], ['z', 3]], cols=['n', 'v'], kd={'x': 0, 'y': 1, 'z': 2})


def test_the_daf_has_a_kd_and_no_keyfield():
    d = _daf()
    assert d.keyfield == '' and d._kd == {'x': 0, 'y': 1, 'z': 2}


def test_keys_lists_the_kd():
    assert _daf().keys() == ['x', 'y', 'z']
    assert list(_daf().keys(astype='view')) == ['x', 'y', 'z']


def test_select_krows_uses_the_kd():
    assert _daf().select_krows(['y']).lol == [['y', 2]]
    assert _daf().select_krows(['y'], inverse=True).lol == [['x', 1], ['z', 3]]


def test_select_records_daf_uses_the_kd():
    assert _daf().select_records_daf(['z', 'x']).lol == [['z', 3], ['x', 1]]


def test_remove_key_and_remove_keylist_use_the_kd():
    assert _daf().remove_key('y').lol == [['x', 1], ['z', 3]]
    assert _daf().remove_keylist(['x', 'z']).lol == [['y', 2]]


def test_krows_to_irows_and_select_record_use_the_kd():
    assert _daf().krows_to_irows(['y']) == [1]
    assert _daf().select_record('y') == {'n': 'y', 'v': 2}


def test_a_missing_key_still_raises_keyerror():
    with pytest.raises(KeyError):
        _daf().select_krows(['nope'])


NO_KEYS = Daf(lol=[['x', 1]], cols=['n', 'v'])


@pytest.mark.parametrize('method, arg', [
    ('select_krows', ['x']), ('select_records_daf', ['x']), ('remove_key', 'x'), ('remove_keylist', ['x']),
    ('krows_to_irows', ['x']), ('select_record', 'x'),
    ])
def test_no_keyfield_and_no_kd_names_the_method_and_what_is_missing(method, arg):
    with pytest.raises(KeysDisabledError, match=r"\(\): key lookups are disabled, as the keyfield is not set and there is no key index"):
        getattr(NO_KEYS, method)(arg)


def test_keys_with_silent_error_false_names_what_is_missing():
    assert NO_KEYS.keys() == []
    with pytest.raises(KeysDisabledError, match=r"keys\(\): key lookups are disabled"):
        NO_KEYS.keys(silent_error=False)


def test_assign_record_needs_a_keyfield_not_just_a_kd():
    with pytest.raises(KeysDisabledError, match="assign_record\\(\\): the Daf needs a keyfield"):
        _daf().assign_record({'n': 'x', 'v': 9})


def test_a_daf_with_no_rows_and_a_valid_keyfield_says_the_index_is_empty():
    with pytest.raises(KeysDisabledError, match="the rowkeys index is empty.*no rows has no keys"):
        Daf(cols=['a', 'b'], keyfield='a').select_krows([1])
