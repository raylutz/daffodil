# test_daf_append_list.py
#
# append() with a list, on a Daf that has columns. The list is added in column order
# without building a dict. These tests pin the behavior that path must keep.

import pytest

from daffodil.daf import Daf


def _daf(keyfield='') -> Daf:
    return Daf(lol=[['a', 1], ['b', 2]], cols=['k', 'v'], keyfield=keyfield)


def test_append_list_adds_a_copy():
    d = _daf()
    row = ['c', 3]
    d.append(row)
    row[1] = 99
    assert d.lol[-1] == ['c', 3]
    assert d.lol[-1] is not row


def test_append_la_adds_a_copy():
    d = _daf()
    row = ['c', 3]
    d.append(la=row)
    assert d.lol[-1] == ['c', 3]
    assert d.lol[-1] is not row


def test_append_short_list_is_padded_with_null():
    d = _daf()
    d.append(['c'])
    assert d.lol[-1] == ['c', '']


def test_append_long_list_raises():
    d = _daf()
    with pytest.raises(ValueError):
        d.append(['c', 3, 'extra'])
    assert d.num_rows() == 2


def test_append_list_with_keyfield_updates_key_lookup():
    d = _daf(keyfield='k')
    assert d.select_record('b') == {'k': 'b', 'v': 2}     # builds the key index
    d.append(['c', 3])
    assert d.select_record('c') == {'k': 'c', 'v': 3}


def test_append_list_with_existing_key_adds_a_second_row():
    d = _daf(keyfield='k')
    d.append(['a', 10])
    assert d.num_rows() == 3
    assert d.lol[-1] == ['a', 10]


def test_append_list_respect_kd_replaces_the_row_with_the_same_key():
    d = _daf(keyfield='k')
    d.append(['a', 10], respect_kd=True)
    assert d.num_rows() == 2
    assert d.lol[0] == ['a', 10]
    d.append(['c', 3], respect_kd=True)
    assert d.num_rows() == 3
    assert d.select_record('c') == {'k': 'c', 'v': 3}


def test_append_list_with_tuple_keyfield():
    d = Daf(lol=[['a', 1, 'x']], cols=['k1', 'k2', 'v'], keyfield=('k1', 'k2'))
    d.append(['b', 2, 'y'])
    assert d.select_record(('b', 2))['v'] == 'y'


def test_append_list_to_daf_without_columns_adds_the_list_itself():
    d = Daf()
    row = [1, 2]
    d.append(row)
    assert d.lol == [[1, 2]]


def test_append_list_matches_append_dict():
    a = _daf(keyfield='k')
    b = _daf(keyfield='k')
    for row in (['c', 3], ['d'], ['a', 7]):
        a.append(row)
        b.append(dict(zip(['k', 'v'], row)))
    assert a.lol == b.lol
    assert a.keys() == b.keys()


# append(fast=True): no checks and no copies.

def test_fast_list_is_kept_as_the_row():
    d = _daf()
    row = ['c', 3]
    d.append(row, fast=True)
    assert d.lol[-1] is row


def test_fast_keyedlist_keeps_its_value_list():
    from daffodil.keyedlist import KeyedList
    d = _daf()
    kl = KeyedList(['k', 'v'], ['c', 3])
    d.append(kl, fast=True)
    assert d.lol[-1] == ['c', 3]
    assert d.lol[-1] is kl.values()


def test_fast_dict_takes_its_values_in_order():
    d = _daf()
    d.append({'k': 'c', 'v': 3}, fast=True)
    assert d.lol[-1] == ['c', 3]


def test_fast_does_not_check_the_order_of_a_dict():
    d = _daf()
    d.append({'v': 3, 'k': 'c'}, fast=True)        # keys out of order: the caller broke the promise
    assert d.lol[-1] == [3, 'c']


def test_fast_with_keyfield_updates_key_lookup():
    d = _daf(keyfield='k')
    assert d.select_record('b') == {'k': 'b', 'v': 2}
    d.append(['c', 3], fast=True)
    assert d.select_record('c') == {'k': 'c', 'v': 3}


def test_fast_with_respect_kd_still_replaces_the_row_with_the_same_key():
    d = _daf(keyfield='k')
    d.append(['a', 10], respect_kd=True, fast=True)
    assert d.num_rows() == 2 and d.lol[0] == ['a', 10]


def test_fast_empty_row_adds_nothing():
    d = _daf()
    d.append([], fast=True)
    d.append({}, fast=True)
    assert d.num_rows() == 2


def test_fast_first_row_of_a_daf_without_columns_sets_the_columns():
    d = Daf()
    d.append({'k': 'a', 'v': 1}, fast=True)
    assert d.columns() == ['k', 'v'] and d.lol == [['a', 1]]


def test_fast_list_of_dicts_is_several_rows():
    d = _daf()
    d.append([{'k': 'c', 'v': 3}, {'k': 'd', 'v': 4}], fast=True)
    assert d.lol[-2:] == [['c', 3], ['d', 4]]


def test_fast_lol_and_la():
    d = _daf()
    rows = [['c', 3], ['d', 4]]
    d.append(lol=rows, fast=True)
    assert d.lol[-1] is rows[1]
    d.append(la=[{'x': 1}, 'e'], fast=True)          # la is one row, even if its first item is a dict
    assert d.lol[-1] == [{'x': 1}, 'e']


def test_fast_matches_the_checked_append_for_good_rows():
    from daffodil.keyedlist import KeyedList
    rows = [['c', 3], {'k': 'd', 'v': 4}, KeyedList(['k', 'v'], ['e', 5])]
    checked, fast = _daf(keyfield='k'), _daf(keyfield='k')
    for row in rows:
        checked.append(row)
        fast.append(row, fast=True)
    assert checked.lol == fast.lol
    assert checked.keys() == fast.keys()


# from_lod(fast=True)

def test_from_lod_fast_matches_from_lod_for_good_records():
    lod = [{'k': 'a', 'v': 1}, {'k': 'b', 'v': 2}]
    assert Daf.from_lod(lod, fast=True).lol == Daf.from_lod(lod).lol
    assert Daf.from_lod(lod, fast=True).columns() == ['k', 'v']
    d = Daf.from_lod(lod, keyfield='k', fast=True)
    assert d.select_record('b') == {'k': 'b', 'v': 2}


def test_from_lod_fast_uses_cols_or_dtypes_as_the_columns():
    lod = [{'k': 'a', 'v': 1}]
    assert Daf.from_lod(lod, cols=['k', 'v'], fast=True).columns() == ['k', 'v']
    assert Daf.from_lod(lod, dtypes={'k': str, 'v': int}, fast=True).columns() == ['k', 'v']
    with pytest.raises(ValueError, match='first dict'):
        Daf.from_lod(lod, cols=['key', 'val'], fast=True)       # cols does not rename


def test_from_lod_fast_does_not_check_the_keys():
    lod = [{'k': 'a', 'v': 1}, {'v': 2, 'k': 'b'}]  # second dict out of order: the caller broke the promise
    assert Daf.from_lod(lod, fast=True).lol == [['a', 1], [2, 'b']]


def test_from_lod_fast_empty_list():
    assert Daf.from_lod([], cols=['a'], fast=True).columns() == ['a']


def test_append_keyedlist_adds_a_copy():
    from daffodil.keyedlist import KeyedList
    d = _daf()
    kl = KeyedList(['k', 'v'], ['c', 3])
    d.append(kl)
    kl['v'] = 99
    assert d.lol[-1] == ['c', 3]
    first = Daf()
    first.append(kl)
    kl['v'] = 100
    assert first.lol == [['c', 99]]


def test_fast_list_filled_in_place_gives_the_same_row_each_time():
    # The documented trap: with fast=True the Daf keeps the list itself.
    d = Daf(cols=['k', 'v'])
    buf = ['', 0]
    for k, v in (('a', 1), ('b', 2)):
        buf[0], buf[1] = k, v
        d.append(buf, fast=True)
    assert d.lol == [['b', 2], ['b', 2]]


# fast=True checks only the first row of an empty Daf

def test_fast_first_row_of_an_empty_daf_is_checked():
    for bad in ({'v': 1, 'k': 'a'}, ['a'], ['a', 1, 'x']):
        d = Daf(cols=['k', 'v'])
        with pytest.raises(ValueError, match='first row'):
            d.append(bad, fast=True)
        assert d.num_rows() == 0


def test_fast_first_row_that_fits_is_added():
    from daffodil.keyedlist import KeyedList
    for good in ({'k': 'a', 'v': 1}, ['a', 1], KeyedList(['k', 'v'], ['a', 1])):
        d = Daf(cols=['k', 'v'])
        d.append(good, fast=True)
        assert d.lol == [['a', 1]]


def test_fast_first_row_keyedlist_with_its_own_hd_in_another_order_raises():
    from daffodil.keyedlist import KeyedList
    d = Daf(cols=['k', 'v'])
    with pytest.raises(ValueError, match='first row'):
        d.append(KeyedList(['v', 'k'], [1, 'a']), fast=True)


def test_fast_lol_into_an_empty_daf_checks_its_first_row():
    d = Daf(cols=['k', 'v'])
    with pytest.raises(ValueError, match='first row'):
        d.append(lol=[['a'], ['b', 2]], fast=True)


def test_from_lod_fast_checks_the_first_dict_against_cols():
    with pytest.raises(ValueError, match='first dict'):
        Daf.from_lod([{'v': 1, 'k': 'a'}], cols=['k', 'v'], fast=True)
    assert Daf.from_lod([{'k': 'a', 'v': 1}], cols=['k', 'v'], fast=True).lol == [['a', 1]]


# a KeyedList made by the Daf skips the column check

def test_append_keyedlist_from_default_record_is_copied_and_placed_by_column():
    d = Daf(cols=['k', 'v'])
    row = d.default_record(astype=__import__('daffodil.keyedlist', fromlist=['KeyedList']).KeyedList)
    assert row.hd is d.hd
    row['v'] = 2
    row['k'] = 'a'                                  # filled in any order
    d.append(row)
    assert d.lol == [['a', 2]] and d.lol[0] is not row.values()
    d.append(row, fast=True)
    assert d.lol[-1] is row.values()


def test_append_keyedlist_that_added_a_key_is_checked_by_name():
    from daffodil.keyedlist import KeyedList
    d = Daf(lol=[['a', 1]], cols=['k', 'v'])
    row = d.default_record(astype=KeyedList)
    row['extra'] = 'x'                              # now has its own hd
    row['k'] = 'b'
    d.append(row)
    assert d.lol[-1] == ['b', ''] and d.columns() == ['k', 'v']


def test_the_kept_column_list_follows_a_change_of_columns():
    d = Daf(lol=[['a', 1]], cols=['k', 'v'])
    d.append({'k': 'b', 'v': 2})
    d.insert_col('n', [0, 0])
    d.append({'k': 'c', 'v': 3, 'n': 9})
    assert d.lol[-1] == ['c', 3, 9]
    assert d._col_names() == ['k', 'v', 'n']


# fast=True checks the length of every row

def test_fast_later_row_of_the_wrong_length_raises():
    from daffodil.keyedlist import KeyedList
    for bad in (['a'], ['a', 1, 'x'], {'k': 'a'}, KeyedList(['k'], ['a'])):
        d = _daf()
        with pytest.raises(ValueError, match='values for 2 columns'):
            d.append(bad, fast=True)
        assert d.num_rows() == 2


def test_fast_lol_checks_the_length_of_every_row():
    d = _daf()
    with pytest.raises(ValueError, match='row 1 of lol'):
        d.append(lol=[['c', 3], ['d']], fast=True)
    assert d.num_rows() == 2


def test_from_lod_fast_checks_the_length_of_every_dict():
    with pytest.raises(ValueError, match='dict 1 has 1 keys'):
        Daf.from_lod([{'k': 'a', 'v': 1}, {'k': 'b'}], fast=True)
