# test_keyedlist_astype_null.py
#
# KeyedList.values(astype) and astype_la() keep an empty cell, which is NULL, as it is. This is the rule of daf_utils.astype_la().

import pytest

from daffodil.keyedlist import KeyedList, astype_la
from daffodil.lib import daf_utils


@pytest.mark.parametrize('astype', [int, float, str, bool, 'int', 'float', 'str', 'bool', lambda v: v + v])
def test_an_empty_cell_is_kept_for_every_kind_of_astype(astype):
    assert astype_la(['1', '', '2'], astype)[1] == ''


def test_astype_la_converts_the_other_cells():
    assert astype_la(['1', '', '3'], int) == [1, '', 3]
    assert astype_la(['1.5', ''], 'float') == [1.5, '']
    assert astype_la([0, '', 1], 'bool') == [False, '', True]


def test_values_keeps_an_empty_cell():
    assert KeyedList(['a', 'b'], ['1', '']).values(int) == [1, '']
    assert KeyedList(['a', 'b'], ['', '2.5']).values('float') == ['', 2.5]


def test_values_of_a_daf_row_with_a_missing_cell():
    from daffodil.daf import Daf
    d = Daf(lol=[['1', ''], ['3', '4']], cols=['a', 'b'])
    assert [row.values(int) for row in d.iter_klist()] == [[1, ''], [3, 4]]


def test_a_cell_that_is_not_empty_and_cannot_be_converted_still_raises():
    with pytest.raises(ValueError):
        KeyedList(['a'], ['x']).values(int)


def test_none_is_not_an_empty_cell():
    with pytest.raises(TypeError):
        KeyedList(['a'], [None]).values(int)


def test_unsupported_astype_still_raises_even_for_an_empty_list():
    with pytest.raises(ValueError, match='astype not supported'):
        astype_la([], 'date')
    with pytest.raises(ValueError, match='astype not supported'):
        astype_la([], 42)


def test_astype_la_none_returns_the_same_list():
    la = ['1', '']
    assert astype_la(la, None) is la


@pytest.mark.parametrize('astype', [int, float, str, bool, 'int', 'float', 'str', 'bool'])
def test_the_keyedlist_version_agrees_with_the_daf_utils_version(astype):
    la = ['1', '', '0', '2']
    assert astype_la(la, astype) == daf_utils.astype_la(la, astype)
