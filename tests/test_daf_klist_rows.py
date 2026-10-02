# test_daf_klist_rows.py
#
# KeyedList rows share one index of the column names. They do not build a new one per row.

from daffodil.daf import Daf


def _daf():
    return Daf(cols=['a', 'b', 'c'], lol=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])


def test_iter_klist_rows_share_one_index():
    rows = list(_daf().iter_klist())
    assert rows[0].hd is rows[1].hd is rows[2].hd


def test_iter_klist_rows_point_at_the_rows_in_the_daf():
    daf = _daf()
    for row in daf.iter_klist():
        row['b'] = 0
    assert daf.lol == [[1, 0, 3], [4, 0, 6], [7, 0, 9]]


def test_iter_klist_values_match_the_dict_rows():
    daf = _daf()
    as_klist = [dict(zip(row.keys(), row.values())) for row in daf.iter_klist()]
    assert as_klist == list(daf.iter_dict())


def test_iter_klist_on_an_empty_daf():
    assert list(Daf(cols=['a', 'b']).iter_klist()) == []


def test_iloc_klist_calls_share_one_index():
    daf = _daf()
    assert daf.iloc(0, rtype='klist').hd is daf.iloc(2, rtype='klist').hd


def test_iloc_klist_index_follows_a_replaced_hd():
    daf = _daf()
    daf.iloc(0, rtype='klist')
    daf.hd = {'x': 0, 'y': 1, 'z': 2}
    assert list(daf.iloc(0, rtype='klist').keys()) == ['x', 'y', 'z']


def test_iloc_klist_index_follows_a_new_column():
    daf = _daf()
    daf.iloc(0, rtype='klist')
    daf.hd['d'] = 3
    for row in daf.lol:
        row.append(0)
    assert list(daf.iloc(1, rtype='klist').keys()) == ['a', 'b', 'c', 'd']
    assert daf.iloc(1, rtype='klist')['d'] == 0


def test_iloc_klist_values_are_the_rows_own_list():
    daf = _daf()
    daf.iloc(1, rtype='klist')['c'] = 99
    assert daf.lol[1] == [4, 5, 99]
