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


# adding a column through assign_icol and assign_col

def test_assign_icol_that_adds_a_column_leaves_the_original_alone():
    d = _daf()
    s = d.select_irows([0, 1])
    s.assign_icol(-1, ['x', 'y'])
    assert s.lol == [[1, 'a', 'x'], [2, 'b', 'y']]
    assert d.lol == [[1, 'a'], [2, 'b'], [3, 'c']]


# a selection is a live view for values: value writers reach the original

import pytest


def _three() -> Daf:
    return Daf(lol=[[1, ' a ', 10], [2, ' b ', 20], [3, ' c ', 30]], cols=['id', 'v', 'n'], keyfield='id')


VALUE_WRITERS = {
    'assign_col':         lambda s: s.assign_col('v', ['X', 'Y']),
    'assign_icol':        lambda s: s.assign_icol(1, ['X', 'Y']),
    'set_icol':           lambda s: s.set_icol(1, 'X'),
    'replace_in_columns': lambda s: s.replace_in_columns(['v'], [' a ', ' b '], 'Q'),
    'apply_row':          lambda s: s.apply_in_place(lambda r: {**r, 'v': 'Z'}, by='row'),
    'apply_row_klist':    lambda s: s.apply_in_place(lambda r: r.__setitem__('v', 'Z'), by='row_klist'),
    'strip':              lambda s: s.strip(),
    'update_record_irow': lambda s: s.update_record_irow(0, {'v': 'U'}),
    'klist_loop':         lambda s: [kl.__setitem__('v', 'K') for kl in s.iter_klist()],
    'cell':               lambda s: s.__setitem__((0, 'v'), 'C'),
}


@pytest.mark.parametrize('name', list(VALUE_WRITERS))
def test_value_writer_on_a_selection_reaches_the_original(name):
    d = _three()
    s = d.select_irows([0, 1])
    VALUE_WRITERS[name](s)
    assert d.lol[0][1] == s.lol[0][1]
    assert d.lol[0][1] != ' a '
    assert d.lol[2] == [3, ' c ', 30]
    assert all(len(row) == 3 for row in d.lol)


@pytest.mark.parametrize('name', list(VALUE_WRITERS))
def test_value_writer_on_unshared_rows_changes_them_in_place(name):
    d = _three()
    row_ids = [id(row) for row in d.lol]
    VALUE_WRITERS[name](d)
    assert [id(row) for row in d.lol] == row_ids


# select_by_dict returns a shallow new Daf, and the report scenario

def test_select_by_dict_shares_the_rows():
    d = _daf()
    s = d.select_by_dict({'v': 'b'})
    assert s.lol == [[2, 'b']]
    assert s.lol is not d.lol
    assert s.lol[0] is d.lol[1]
    assert s.keyfield == 'id'


def test_select_by_dict_inverse_and_expectmax_still_work():
    d = _daf()
    assert d.select_by_dict({'v': 'b'}, inverse=True).lol == [[1, 'a'], [3, 'c']]
    assert d.select_by_dict({'v': 'zz'}).lol == []
    with pytest.raises(LookupError):
        d.select_by_dict({'id': 1}, expectmax=0)


def test_report_scenario_select_by_dict_then_insert_idx_col_leaves_the_original_alone():
    d = _daf()
    report_daf = d.select_by_dict({'v': 'b'}, inverse=True)
    report_daf.insert_idx_col(colname='idx', icol=0)
    assert report_daf.lol == [[0, 1, 'a'], [1, 3, 'c']]
    assert list(report_daf.hd) == ['idx', 'id', 'v']
    assert d.lol == [[1, 'a'], [2, 'b'], [3, 'c']]
    assert list(d.hd) == ['id', 'v']


def test_select_by_dict_is_a_live_view_for_values():
    d = _daf()
    s = d.select_by_dict({'v': 'b'})
    s.set_icol(1, 'Z')
    assert d.lol == [[1, 'a'], [2, 'Z'], [3, 'c']]


# selectors return their own row list, also when every row is selected

def _four() -> Daf:
    return Daf(lol=[[1, 'a'], [2, 'b'], [3, 'c'], [4, 'd']], cols=['id', 'v'], keyfield='id')


ALL_ROWS_CALLS = {
    'select_irows list of all':     lambda d: d.select_irows([0, 1, 2, 3]),
    'select_irows range':           lambda d: d.select_irows(range(4)),
    'select_irows slice':           lambda d: d.select_irows(slice(None)),
    'select_irows nothing inverse': lambda d: d.select_irows([], inverse=True),
    'getitem list of all':          lambda d: d[[0, 1, 2, 3]],
    'getitem slice':                lambda d: d[:],
    'select_krows all keys':        lambda d: d.select_krows([1, 2, 3, 4]),
    'select_krows nothing inverse': lambda d: d.select_krows([], inverse=True),
    'select_records_daf inverse':   lambda d: d.select_records_daf([], inverse=True),
    'select_where all':             lambda d: d.select_where(lambda r: True),
}


@pytest.mark.parametrize('name', list(ALL_ROWS_CALLS))
def test_a_selection_of_every_row_has_its_own_row_list_and_shares_the_rows(name):
    d = _four()
    s = ALL_ROWS_CALLS[name](d)
    assert s.lol == d.lol
    assert s.lol is not d.lol
    assert all(a is b for a, b in zip(s.lol, d.lol))


@pytest.mark.parametrize('name', list(ALL_ROWS_CALLS))
def test_appending_to_a_selection_of_every_row_leaves_the_original_alone(name):
    d = _four()
    d.keys()
    s = ALL_ROWS_CALLS[name](d)
    s.append([5, 'e'])
    assert len(d) == 4 and len(s) == 5
    assert 5 not in d.keys()
    assert d.select_record(4)['v'] == 'd'


# clone_empty

class _CountingList(list):
    """ A list that counts how often it is walked through. """
    walks = 0

    def __iter__(self):
        _CountingList.walks += 1
        return super().__iter__()


def test_selecting_one_row_does_not_walk_the_whole_row_list():
    d = _daf()
    d.lol = _CountingList(d.lol)
    _CountingList.walks = 0
    one = d.select_irows([1])
    assert one.lol == [[2, 'b']]
    assert d[1, 'v'].to_value() == 'b'
    assert _CountingList.walks == 0


def test_clone_empty_adopts_the_given_rows_and_keeps_the_columns():
    d = _daf()
    rows = [[9, 'z']]
    c = d.clone_empty(lol=rows)
    assert c.lol is rows
    assert c.columns() == ['id', 'v'] and c.keyfield == 'id'
    assert c.select_record(9) == {'id': 9, 'v': 'z'}
    assert d.num_rows() == 3 and d.select_record(1) == {'id': 1, 'v': 'a'}
