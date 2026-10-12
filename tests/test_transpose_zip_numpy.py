# test_transpose_zip_numpy.py
#
# transpose_rows_lol() transposes with zip below TRANSPOSE_NUMPY_MIN_CELLS cells, and with
# numpy from that size. Without numpy it uses zip at any size. transpose_lol(),
# Daf.transpose() and the markdown display all use it. These tests check that both ways
# give the same result, that nothing needs numpy, and that the cells are not copied.

import sys
from typing import Any

import pytest
from daffodil.daf import Daf
from daffodil.lib import daf_md as md
from daffodil.lib import daf_utils
from daffodil.lib.daf_utils import transpose_lol, transpose_rows_lol


def _lol(num_rows: int, num_cols: int) -> list:
    return [[f'{irow}-{icol}' if icol % 2 else irow * icol for icol in range(num_cols)] for irow in range(num_rows)]


def _by_zip(lol: list) -> list:
    return [list(col_t) for col_t in zip(*lol)]


def _no_numpy(monkeypatch: Any) -> None:
    """ Make `import numpy` fail, as if it were not installed. """
    monkeypatch.setitem(sys.modules, 'numpy', None)


SHAPES: list = [(0, 0), (1, 1), (1, 5), (5, 1), (3, 4), (40, 7)]


@pytest.mark.parametrize('shape', SHAPES, ids=repr)
@pytest.mark.parametrize('min_cells', [1, 100_000], ids=['numpy', 'zip'])
def test_same_result_by_zip_and_numpy(monkeypatch: Any, min_cells: int, shape: tuple) -> None:
    monkeypatch.setattr(daf_utils, 'TRANSPOSE_NUMPY_MIN_CELLS', min_cells)
    lol = _lol(*shape)
    assert transpose_rows_lol(lol, shape[1]) == _by_zip(lol)
    assert transpose_lol(lol) == _by_zip(lol)


def test_threshold_is_100000_cells() -> None:
    assert daf_utils.TRANSPOSE_NUMPY_MIN_CELLS == 100_000
    lol = _lol(10_000, 10)
    assert transpose_lol(lol) == _by_zip(lol)


@pytest.mark.parametrize('min_cells', [1, 100_000], ids=['numpy', 'zip'])
def test_cells_are_not_copied(monkeypatch: Any, min_cells: int) -> None:
    monkeypatch.setattr(daf_utils, 'TRANSPOSE_NUMPY_MIN_CELLS', min_cells)
    text = 'abc' * 20
    cell = {'k': [1]}
    result = transpose_lol([[text, cell], ['x', 'y']])
    assert result[0][0] is text
    assert result[1][0] is cell


def test_large_table_without_numpy(monkeypatch: Any) -> None:
    _no_numpy(monkeypatch)
    monkeypatch.setattr(daf_utils, 'TRANSPOSE_NUMPY_MIN_CELLS', 1)
    lol = _lol(30, 4)
    assert transpose_lol(lol) == _by_zip(lol)
    daf = Daf(lol=lol, cols=['a', 'b', 'c', 'd'])
    assert daf.transpose().lol == _by_zip(lol)


def test_print_without_numpy(monkeypatch: Any) -> None:
    """ Printing a Daf transposes its rows. It used to need numpy for that. """
    _no_numpy(monkeypatch)
    daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['n', 's'])
    assert '|  2 |  b |' in repr(daf)
    assert '|  2 |  b |' in daf.to_md()


@pytest.mark.parametrize('min_cells', [1, 100_000], ids=['numpy', 'zip'])
def test_transpose_lol_ragged_raises(monkeypatch: Any, min_cells: int) -> None:
    monkeypatch.setattr(daf_utils, 'TRANSPOSE_NUMPY_MIN_CELLS', min_cells)
    with pytest.raises(RuntimeError, match='ragged list-of-lists'):
        transpose_lol([[1, 2, 3], [4, 5]])
    with pytest.raises(RuntimeError, match='ragged list-of-lists'):
        transpose_lol([[1, 2], [3, 4, 5]])
    with pytest.raises(RuntimeError):
        md.md_lol_table([[1, 2, 3], [4, 5]], header=['a', 'b', 'c'])


def test_daf_transpose_small_table() -> None:
    daf = Daf(lol=[[1, 'a'], [2, 'b'], [3, 'c']], cols=['id', 'v'])
    assert daf.transpose().lol == [[1, 2, 3], ['a', 'b', 'c']]
    assert daf.transpose(include_header=True).lol == [['id', 1, 2, 3], ['v', 'a', 'b', 'c']]
    assert Daf(cols=['id', 'v']).transpose().lol == []


def test_list_cells_kept_whole_by_zip() -> None:
    """ Below the threshold, cells that are lists of the same length stay whole. """
    daf = Daf(lol=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], cols=['a', 'b'])
    assert daf.transpose().lol == [[[1, 2], [5, 6]], [[3, 4], [7, 8]]]
    assert '[1, 2]' in repr(daf)


@pytest.mark.xfail(strict=True, reason="BUG: daf_utils.py transpose_rows_lol(): np.array() looks inside cells that "
                                       "are lists of the same length, and transposes their items too. Item 16.")
def test_list_cells_kept_whole_by_numpy(monkeypatch: Any) -> None:
    monkeypatch.setattr(daf_utils, 'TRANSPOSE_NUMPY_MIN_CELLS', 1)
    daf = Daf(lol=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], cols=['a', 'b'])
    assert daf.transpose().lol == [[[1, 2], [5, 6]], [[3, 4], [7, 8]]]
