# test_daf_select_icols.py
#
# Column slices through select_icols() and through my_daf[:, slice].
# Python's own slicing is the model: my_daf[:, a:b:c] should return the columns
# that list(range(n))[a:b:c] gives.

from daffodil.daf import Daf


def _daf() -> Daf:
    return Daf(lol=[[1, 'a', 10], [2, 'b', 20]], cols=['id', 'v', 'n'])


def test_select_icols_positive_slice():
    result = _daf()[:, 1:3]
    assert result.columns() == ['v', 'n']
    assert result.lol == [['a', 10], ['b', 20]]


def test_select_icols_negative_start_and_stop():
    result = _daf()[:, -3:-1]
    assert result.columns() == ['id', 'v']
    assert result.lol == [[1, 'a'], [2, 'b']]


def test_select_icols_step():
    result = _daf()[:, ::2]
    assert result.columns() == ['id', 'n']
    assert result.lol == [[1, 10], [2, 20]]


def test_select_icols_negative_start_no_stop():
    result = _daf()[:, -2:]
    assert result.columns() == ['v', 'n']
    assert result.lol == [['a', 10], ['b', 20]]


def test_select_icols_negative_stop_no_start():
    result = _daf()[:, :-1]
    assert result.columns() == ['id', 'v']
    assert result.lol == [[1, 'a'], [2, 'b']]


def test_select_icols_positive_start_negative_stop():
    result = _daf()[:, 1:-1]
    assert result.columns() == ['v']
    assert result.lol == [['a'], ['b']]


def test_select_icols_negative_step():
    result = _daf()[:, ::-1]
    assert result.columns() == ['n', 'v', 'id']


def test_select_icols_stop_zero_is_empty():
    result = _daf()[:, 0:0]
    assert result.columns() == []


def test_select_icols_stop_beyond_end_is_clamped():
    result = _daf()[:, 1:10]
    assert result.columns() == ['v', 'n']


def test_select_icols_negative_slice_with_flip():
    result = _daf().select_icols(slice(-2, None), flip=True)
    assert result.lol == [['a', 'b'], [10, 20]]
