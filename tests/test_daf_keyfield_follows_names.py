# test_daf_keyfield_follows_names.py
#
# rename_cols() and set_cols() keep the keyfield: it takes the new name of its column.
# With no column names yet, set_cols() keeps a keyfield that is one of the new names.

import pytest

from daffodil.daf import Daf


def make_daf(keyfield='id') -> Daf:
    return Daf(lol=[[1, 'a', 5], [2, 'b', 6]], cols=['id', 'v', 'n'], keyfield=keyfield)


# =====================================================================
# rename_cols
# =====================================================================

def test_rename_cols_keyfield_follows_the_rename_and_lookups_work():
    d = make_daf().rename_cols({'id': 'ident'})
    assert d.keyfield == 'ident' and d.keys() == [1, 2]
    assert d.select_krows([2]).lol == [[2, 'b', 6]]


def test_rename_cols_keeps_the_keyfield_when_another_column_is_renamed():
    d = make_daf().rename_cols({'v': 'w'})
    assert d.keyfield == 'id' and d.keys() == [1, 2]


def test_rename_cols_composite_keyfield_keeps_its_type():
    d = make_daf(('id', 'v')).rename_cols({'id': 'ident'})
    assert d.keyfield == ('ident', 'v') and d.keys() == [(1, 'a'), (2, 'b')]
    d = make_daf(['id', 'v']).rename_cols({'v': 'w'})
    assert d.keyfield == ['id', 'w']


def test_rename_cols_with_no_keyfield_leaves_it_empty():
    assert make_daf('').rename_cols({'id': 'ident'}).keyfield == ''


# =====================================================================
# set_cols
# =====================================================================

def test_set_cols_keyfield_follows_by_position_not_by_name():
    d = make_daf().set_cols(['b', 'id', 'n'])
    assert d.keyfield == 'b' and d.keys() == [1, 2]


def test_set_cols_composite_keyfield_follows():
    d = make_daf(('id', 'n')).set_cols(['x', 'y', 'z'])
    assert d.keyfield == ('x', 'z') and d.keys() == [(1, 5), (2, 6)]


def test_set_cols_with_no_names_yet_keeps_a_keyfield_that_is_a_new_name():
    d = Daf(lol=[[1, 'a'], [2, 'b']], keyfield='id')
    assert d.columns() == [] and d.keyfield == 'id'
    d.set_cols(['id', 'v'])
    assert d.keyfield == 'id' and d.keys() == [1, 2]


def test_set_cols_with_no_names_yet_clears_a_keyfield_that_is_not_a_new_name():
    d = Daf(lol=[[1, 'a']], keyfield='id').set_cols(['p', 'q'])
    assert d.keyfield == ''


def test_set_cols_with_no_names_yet_composite_keyfield():
    assert Daf(lol=[[1, 'a']], keyfield=('id', 'v')).set_cols(['id', 'v']).keyfield == ('id', 'v')
    assert Daf(lol=[[1, 'a']], keyfield=('id', 'zz')).set_cols(['id', 'v']).keyfield == ''


def test_set_cols_default_names_keep_a_keyfield_that_matches_them():
    d = Daf(lol=[[1, 2]], keyfield='A').set_cols()
    assert d.columns() == ['A', 'B'] and d.keyfield == 'A'


def test_set_cols_renames_the_dtypes_and_the_keyfield_together():
    d = Daf(lol=[[1, 'a']], cols=['id', 'v'], dtypes={'id': int, 'v': str}, keyfield='id').set_cols(['k', 'w'])
    assert d.dtypes == {'k': int, 'w': str} and d.keyfield == 'k'
