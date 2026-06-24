# test_daf_setitem.py
#
# Tests for Daf.set_irows_icols() -- the implementation behind Daf.__setitem__ (daf[irows, icols] =
# value). Covers all branches: single row/single col, single row/no col, multi-row/no col,
# multi-row/single col, and multi-row/multi-col, each across scalar/list/dict/Daf value types.
#
# Of particular note: dict values must be checked for *before* the broader Sequence/list check in
# every branch, since dict is itself Iterable (though not a Sequence). Getting this ordering wrong
# previously caused either silent data corruption (dict assigned as raw row content) or a hang via
# the deliberate breakpoint() tripwire in the except-block fallback. These tests guard against
# regressing that ordering.

import pytest

from daffodil.daf import Daf


# --- single row, no column selection (whole-row assignment) ---

def test_setitem_single_row_list_value():
    daf = Daf(lol=[[1, 2], [3, 4]], cols=['a', 'b'])
    daf[0, :] = [9, 9]
    assert daf.lol == [[9, 9], [3, 4]]


def test_setitem_single_row_dict_value():
    daf = Daf(lol=[[1, 2], [3, 4]], cols=['a', 'b'])
    daf[0, :] = {'a': 10, 'b': 20}
    assert daf.lol == [[10, 20], [3, 4]]


def test_setitem_single_row_scalar_broadcasts():
    daf = Daf(lol=[[1, 2], [3, 4]], cols=['a', 'b'])
    daf[0, :] = 0
    assert daf.lol == [[0, 0], [3, 4]]


# --- single row, single column (single cell) ---

def test_setitem_single_cell_scalar():
    daf = Daf(lol=[[1, 2], [3, 4]], cols=['a', 'b'])
    daf[0, 0] = 99
    assert daf.lol == [[99, 2], [3, 4]]


def test_setitem_single_cell_dict_value():
    daf = Daf(lol=[[1, 2], [3, 4]], cols=['a', 'b'])
    daf[0, 0] = {'a': 7, 'b': 8}
    assert daf.lol == [[7, 8], [3, 4]]


def test_setitem_single_cell_single_item_list_unwraps():
    daf = Daf(lol=[[1, 2], [3, 4]], cols=['a', 'b'])
    daf[0, 0] = [55]
    assert daf.lol == [[55, 2], [3, 4]]


# --- multiple rows, no column selection ---

def test_setitem_multi_row_list_value_broadcasts_whole_row():
    daf = Daf(lol=[[1, 2], [3, 4], [5, 6]], cols=['a', 'b'])
    daf[[0, 1], :] = [0, 0]
    assert daf.lol == [[0, 0], [0, 0], [5, 6]]


def test_setitem_multi_row_scalar_broadcasts():
    daf = Daf(lol=[[1, 2], [3, 4], [5, 6]], cols=['a', 'b'])
    daf[[0, 1], :] = 7
    assert daf.lol == [[7, 7], [7, 7], [5, 6]]


def test_setitem_multi_row_dict_value_maps_columns():
    # this is the bug we found: dict must not be swallowed by the broader Sequence check,
    # or it gets assigned as the raw row content instead of being mapped to columns.
    daf = Daf(lol=[[1, 2], [3, 4], [5, 6]], cols=['a', 'b'])
    daf[[0, 1], :] = {'a': 9, 'b': 8}
    assert daf.lol == [[9, 8], [9, 8], [5, 6]]


# --- all rows or multiple rows, single column ---

def test_setitem_single_col_list_value_per_row():
    daf = Daf(lol=[[1, 2], [3, 4], [5, 6]], cols=['a', 'b'])
    daf[:, 0] = [100, 200, 300]
    assert daf.lol == [[100, 2], [200, 4], [300, 6]]


def test_setitem_single_col_scalar_broadcasts():
    daf = Daf(lol=[[1, 2], [3, 4], [5, 6]], cols=['a', 'b'])
    daf[:, 0] = 0
    assert daf.lol == [[0, 2], [0, 4], [0, 6]]


def test_setitem_single_col_dict_value_maps_all_matching_cols():
    # dict ignores the narrower icols selection and updates whichever of its own keys match --
    # documented in the source as "this is the same as cols=0 bc dict updates the corresponding cols."
    daf = Daf(lol=[[1, 2], [3, 4], [5, 6]], cols=['a', 'b'])
    daf[:, 0] = {'a': 100, 'b': 200}
    assert daf.lol == [[100, 200], [100, 200], [100, 200]]


# --- multiple rows, multiple columns ---

def test_setitem_multi_multi_scalar_broadcasts():
    daf = Daf(lol=[[1, 2], [3, 4]], cols=['a', 'b'])
    daf[:, :] = 0
    assert daf.lol == [[0, 0], [0, 0]]


def test_setitem_multi_multi_list_value_same_for_each_row():
    daf = Daf(lol=[[1, 2], [3, 4]], cols=['a', 'b'])
    daf[[0, 1], [0, 1]] = [10, 20]
    assert daf.lol == [[10, 20], [10, 20]]


def test_setitem_multi_multi_dict_value_maps_columns():
    # this is the bug we found: previously hung via breakpoint() since dict was swallowed by
    # the broader Iterable check, then tried integer-indexed lookup into the dict and raised
    # KeyError inside a bare except that hit the tripwire.
    daf = Daf(lol=[[1, 2], [3, 4]], cols=['a', 'b'])
    daf[[0, 1], [0, 1]] = {'a': 5, 'b': 6}
    assert daf.lol == [[5, 6], [5, 6]]
