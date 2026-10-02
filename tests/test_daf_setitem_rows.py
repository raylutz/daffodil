# test_daf_setitem_rows.py
#
# Assigning one value or one list to several whole rows: my_daf[[0, 1]] = value.

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


def test_setitem_scalar_str_to_several_rows():
    d = _daf()
    d[[0, 1]] = 'x'
    assert d.lol == [['x', 'x', 'x'], ['x', 'x', 'x'], [3, 'c', 30]]


def test_setitem_scalar_str_to_all_rows():
    d = _daf()
    d[:] = 'x'
    assert d.lol == [['x', 'x', 'x']] * 3


def test_setitem_list_to_several_rows_makes_independent_rows():
    d = _daf()
    d[[0, 1]] = [7, 8, 9]
    assert d.lol[0] is not d.lol[1]
    d.lol[0][0] = 'X'
    assert d.lol[1][0] == 7


def test_setitem_scalar_str_to_a_column():
    d = _daf()
    d[:, 'v'] = 'x'
    assert d.col('v') == ['x', 'x', 'x']


def test_setitem_list_to_a_column():
    d = _daf()
    d[:, 'v'] = ['x', 'y', 'z']
    assert d.col('v') == ['x', 'y', 'z']


def test_setitem_str_to_block():
    d = _daf()
    d[[0, 1], ['v', 'n']] = 'ab'
    assert d.lol == [[1, 'ab', 'ab'], [2, 'ab', 'ab'], [3, 'c', 30]]


def test_setitem_str_to_a_column_whole_word():
    d = _daf()
    d[:, 'v'] = 'xyz'
    assert d.col('v') == ['xyz', 'xyz', 'xyz']


def test_setitem_tuple_to_several_rows_is_a_list_per_row():
    d = _daf()
    d[[0, 1]] = (7, 8, 9)
    assert d.lol[:2] == [[7, 8, 9], [7, 8, 9]]
    assert d.lol[0] is not d.lol[1]
