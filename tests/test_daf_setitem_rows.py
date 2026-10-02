# test_daf_setitem_rows.py
#
# Assigning one value or one list to several whole rows: my_daf[[0, 1]] = value.

import pytest

from daffodil.daf import Daf


def _daf() -> Daf:
    return Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30]], cols=['id', 'v', 'n'])


def test_setitem_scalar_number_to_several_rows():
    d = _daf()
    d[[0, 1]] = 5
    assert d.lol == [[5, 5, 5], [5, 5, 5], [3, 'c', 30]]


def test_setitem_scalar_str_to_one_row():
    d = _daf()
    d[2] = 'x'
    assert d.lol == [[1, 'a', 10], [2, 'b', 20], ['x', 'x', 'x']]


@pytest.mark.xfail(strict=True, reason="BUG: daf.py set_irows_icols() tests isinstance(value, (list, Sequence)) "
                   "for several rows, and a str is a Sequence. d[[0, 1]] = 'x' stores the bare "
                   "string 'x' as each row instead of ['x', 'x', 'x'].")
def test_setitem_scalar_str_to_several_rows():
    d = _daf()
    d[[0, 1]] = 'x'
    assert d.lol == [['x', 'x', 'x'], ['x', 'x', 'x'], [3, 'c', 30]]


@pytest.mark.xfail(strict=True, reason="BUG: daf.py set_irows_icols() same cause. d[:] = 'x' turns every "
                   "row into the bare string 'x'.")
def test_setitem_scalar_str_to_all_rows():
    d = _daf()
    d[:] = 'x'
    assert d.lol == [['x', 'x', 'x']] * 3


@pytest.mark.xfail(strict=True, reason="BUG: daf.py set_irows_icols() stores the same list object in every "
                   "selected row (self.lol[irow] = value). After d[[0, 1]] = [7, 8, 9], a change "
                   "to one row changes the other.")
def test_setitem_list_to_several_rows_makes_independent_rows():
    d = _daf()
    d[[0, 1]] = [7, 8, 9]
    assert d.lol[0] is not d.lol[1]
    d.lol[0][0] = 'X'
    assert d.lol[1][0] == 7


@pytest.mark.xfail(strict=True, reason="BUG: daf.py set_irows_icols() same cause. d[:, 'v'] = 'x' sets only the "
                   "first row, because the str is zipped with the rows as a list of characters.")
def test_setitem_scalar_str_to_a_column():
    d = _daf()
    d[:, 'v'] = 'x'
    assert d.col('v') == ['x', 'x', 'x']


def test_setitem_list_to_a_column():
    d = _daf()
    d[:, 'v'] = ['x', 'y', 'z']
    assert d.col('v') == ['x', 'y', 'z']
