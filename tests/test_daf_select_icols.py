# test_daf_select_icols.py
#
# Column slices through select_icols() and through my_daf[:, slice].
# Python's own slicing is the model: my_daf[:, a:b:c] should return the columns
# that list(range(n))[a:b:c] gives.

import pytest

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


@pytest.mark.xfail(strict=True, reason="BUG: daf.py select_icols() builds range(start or 0, stop or num_cols, step or 1), "
                   "so a negative start with no stop gives range(-2, 3). d[:, -2:] returns "
                   "five columns, repeating id, v and n.")
def test_select_icols_negative_start_no_stop():
    result = _daf()[:, -2:]
    assert result.columns() == ['v', 'n']
    assert result.lol == [['a', 10], ['b', 20]]


@pytest.mark.xfail(strict=True, reason="BUG: daf.py select_icols() uses the negative stop as a range stop, "
                   "so d[:, :-1] gives range(0, -1), which is empty, and the result has no columns.")
def test_select_icols_negative_stop_no_start():
    result = _daf()[:, :-1]
    assert result.columns() == ['id', 'v']
    assert result.lol == [[1, 'a'], [2, 'b']]


@pytest.mark.xfail(strict=True, reason="BUG: daf.py select_icols() same cause as above. d[:, 1:-1] raises IndexError.")
def test_select_icols_positive_start_negative_stop():
    result = _daf()[:, 1:-1]
    assert result.columns() == ['v']
    assert result.lol == [['a'], ['b']]


@pytest.mark.xfail(strict=True, reason="BUG: daf.py select_icols() range(..., step or 1) cannot go backwards from the "
                   "default start. d[:, ::-1] raises IndexError.")
def test_select_icols_negative_step():
    result = _daf()[:, ::-1]
    assert result.columns() == ['n', 'v', 'id']


@pytest.mark.xfail(strict=True, reason="BUG: daf.py select_icols() uses 'slice.stop or num_cols', so a stop of 0 "
                   "counts as no stop. d[:, 0:0] returns all three columns instead of none.")
def test_select_icols_stop_zero_is_empty():
    result = _daf()[:, 0:0]
    assert result.columns() == []
