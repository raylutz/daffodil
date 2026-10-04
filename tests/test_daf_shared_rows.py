# test_daf_shared_rows.py
#
# Column-wide inserts must not change the rows of another Daf that shares them.

from daffodil.daf import Daf
from daffodil.lib import daf_utils


def _daf() -> Daf:
    return Daf(lol=[[1, 'a'], [2, 'b'], [3, 'c']], cols=['id', 'v'], keyfield='id')


# rows_are_shared

def test_rows_are_shared_false_for_own_rows():
    assert daf_utils.rows_are_shared([[1], [2]]) is False
    assert daf_utils.rows_are_shared([]) is False
    assert daf_utils.rows_are_shared(_daf().lol) is False


def test_rows_are_shared_true_after_a_selection_for_both_dafs():
    d = _daf()
    s = d.select_irows([0, 1])
    assert daf_utils.rows_are_shared(d.lol) is True
    assert daf_utils.rows_are_shared(s.lol) is True


def test_rows_are_shared_false_after_the_other_daf_is_gone():
    d = _daf()
    s = d.select_irows([0, 1])
    del s
    assert daf_utils.rows_are_shared(d.lol) is False


def test_rows_are_shared_false_with_a_built_key_index():
    d = _daf()
    d.select_krows([1])
    assert daf_utils.rows_are_shared(_daf().lol) is False


# insert_idx_col and insert_col

def test_insert_idx_col_on_a_selection_leaves_the_original_alone():
    d = _daf()
    s = d.select_irows([0, 1])
    result = s.insert_idx_col()
    assert result is s
    assert s.lol == [[0, 1, 'a'], [1, 2, 'b']]
    assert list(s.hd) == ['idx', 'id', 'v']
    assert d.lol == [[1, 'a'], [2, 'b'], [3, 'c']]
    assert list(d.hd) == ['id', 'v']


def test_insert_idx_col_on_the_original_leaves_the_selection_alone():
    d = _daf()
    s = d.select_irows([0, 1])
    d.insert_idx_col()
    assert d.lol == [[0, 1, 'a'], [1, 2, 'b'], [2, 3, 'c']]
    assert s.lol == [[1, 'a'], [2, 'b']]
    assert list(s.hd) == ['id', 'v']


def test_insert_col_on_a_selection_leaves_the_original_alone():
    d = _daf()
    s = d.select_irows([0, 1])
    s.insert_col('w', ['x', 'y'], icol=0)
    assert s.lol == [['x', 1, 'a'], ['y', 2, 'b']]
    assert d.lol == [[1, 'a'], [2, 'b'], [3, 'c']]


def test_insert_icol_keeps_keyfield_and_key_index_on_a_selection():
    d = _daf()
    s = d.select_irows([0, 1])
    s.insert_idx_col()
    assert s.keyfield == 'id'
    assert s.col('id') == [1, 2]


def test_insert_after_a_copy_does_not_copy_again():
    d = _daf()
    s = d.select_irows([0, 1])
    s.insert_idx_col()
    assert daf_utils.rows_are_shared(s.lol) is False
    row_ids = [id(row) for row in s.lol]
    s.insert_col('z', ['p', 'q'])
    assert [id(row) for row in s.lol] == row_ids


def test_insert_on_unshared_daf_changes_the_rows_in_place():
    d = _daf()
    row_ids = [id(row) for row in d.lol]
    d.insert_idx_col()
    assert [id(row) for row in d.lol] == row_ids
    assert d.lol[0] == [0, 1, 'a']


def test_assign_col_of_a_new_column_on_a_selection_leaves_the_original_alone():
    d = _daf()
    s = d.select_irows([0, 1])
    s.assign_col('w', ['x', 'y'])
    assert s.lol == [[1, 'a', 'x'], [2, 'b', 'y']]
    assert d.lol == [[1, 'a'], [2, 'b'], [3, 'c']]


# the column-wide writers and update_record_irow

def _three() -> Daf:
    return Daf(lol=[[1, ' a ', 10], [2, ' b ', 20], [3, ' c ', 30]], cols=['id', 'v', 'n'], keyfield='id')


WRITERS = {
    'assign_col':         lambda s: s.assign_col('v', ['X', 'Y']),
    'assign_icol':        lambda s: s.assign_icol(1, ['X', 'Y']),
    'set_icol':           lambda s: s.set_icol(1, 'X'),
    'replace_in_columns': lambda s: s.replace_in_columns(['v'], [' a '], 'Q'),
    'apply_row':          lambda s: s.apply_in_place(lambda r: {**r, 'v': 'Z'}, by='row'),
    'apply_row_klist':    lambda s: s.apply_in_place(lambda r: r.__setitem__('v', 'Z'), by='row_klist'),
    'strip':              lambda s: s.strip(),
    'update_record_irow': lambda s: s.update_record_irow(0, {'v': 'U'}),
}


import pytest


@pytest.mark.parametrize('name', list(WRITERS))
def test_writer_on_a_selection_leaves_the_original_alone(name):
    d = _three()
    s = d.select_irows([0, 1])
    result = WRITERS[name](s)
    assert result is s
    assert d.lol == [[1, ' a ', 10], [2, ' b ', 20], [3, ' c ', 30]]
    assert s.lol != [[1, ' a ', 10], [2, ' b ', 20]]


@pytest.mark.parametrize('name', list(WRITERS))
def test_writer_on_the_original_leaves_the_selection_alone(name):
    d = _three()
    s = d.select_irows([0, 1])
    WRITERS[name](d)
    assert s.lol == [[1, ' a ', 10], [2, ' b ', 20]]


@pytest.mark.parametrize('name', list(WRITERS))
def test_writer_on_unshared_rows_changes_them_in_place(name):
    d = _three()
    row_ids = [id(row) for row in d.lol]
    WRITERS[name](d)
    assert [id(row) for row in d.lol] == row_ids


def test_update_record_irow_copies_only_the_row_it_changes():
    d = _three()
    s = d.select_irows(slice(0, 3))
    d.update_record_irow(1, {'v': 'U'})
    assert d.lol[1] == [2, 'U', 20]
    assert s.lol[1] == [2, ' b ', 20]
    assert d.lol[0] is s.lol[0]
    assert d.lol[2] is s.lol[2]


def test_row_is_shared_reads_one_row():
    lol = [[1], [2]]
    assert daf_utils.row_is_shared(lol, 0) is False
    other = [lol[0]]
    assert daf_utils.row_is_shared(lol, 0) is True
    assert daf_utils.row_is_shared(lol, 1) is False
