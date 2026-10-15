# test_daf_key_cells.py
#
# Keys and the key index (kd).
#   - NULL is a valid key. Rows with NULL keys can be written and added, with or without a keyfield.
#   - Adding rows invalidates the kd.
#   - A write to a cell or a row does not touch the kd, so a changed key is not seen until
#     set_keyfield() is called again. That keeps writes fast. The __setitem__ docstring says so.
#   - A method that rewrites a whole column, or a column of chosen rows, invalidates the kd
#     if that column is a keyfield column.

from typing import Any, Callable

import pytest
from daffodil.daf import Daf
from daffodil.keyedlist import KeyedList
from daffodil.daf import Daf, KeysDisabledError


def _keyed() -> Daf:
    daf = Daf(lol=[['r1', 1, 'a'], ['r2', 2, 'b']], cols=['k', 'x', 'y'], keyfield='k')
    daf.retmode = 'val'
    assert daf['r1', 'x'] == 1                          # build the key index
    return daf


def _composite() -> Daf:
    daf = Daf(lol=[['a', 1, 10], ['a', 2, 20]], cols=['g', 'n', 'v'], keyfield=['g', 'n'])
    daf.retmode = 'val'
    return daf


def _kd_is_current(daf: Daf) -> bool:
    """ True if the key index, once rebuilt as needed, is what a fresh index of the same rows gives. """
    daf._rebuild_kd_if_invalidated()
    fresh = Daf(lol=[list(row_la) for row_la in daf.lol], cols=daf.columns(), keyfield=daf.keyfield)
    fresh._rebuild_kd_if_invalidated()
    return dict(daf._kd) == dict(fresh._kd)


# --- NULL is a valid key ---

WRITES: list = [
    ('cell by name',      lambda d: d.__setitem__((0, 'k'), '')),
    ('cell by position',  lambda d: d.__setitem__((0, 0), '')),
    ('cell by -3',        lambda d: d.__setitem__((0, -3), '')),
    ('cell by key',       lambda d: d.__setitem__(('r1', 'k'), '')),
    ('row list',          lambda d: d.__setitem__(0, ['', 9, 'z'])),
    ('row by key, dict',  lambda d: d.__setitem__('r1', {'x': 9})),
    ('row dict, key ""',  lambda d: d.__setitem__(0, {'k': '', 'x': 9})),
    ('rows, dict',        lambda d: d.__setitem__(slice(None), {'x': 9})),
    ('rows, list',        lambda d: d.__setitem__(slice(None), ['', 9, 'z'])),
    ('column, value',     lambda d: d.__setitem__((slice(None), 'k'), '')),
    ('column, list',      lambda d: d.__setitem__((slice(None), 'k'), ['q1', ''])),
    ('two cells',         lambda d: d.__setitem__((0, ['k', 'x']), ['', 5])),
    ('rows from a Daf',   lambda d: d.__setitem__(slice(None), Daf(lol=[['q1', 5, 'p'], ['', 6, 'q']], cols=['k', 'x', 'y']))),
    ('assign_record_irow', lambda d: d.assign_record_irow(1, {'x': 9})),
    ('update_record_irow', lambda d: d.update_record_irow(1, {'k': '', 'x': 9})),
]


@pytest.mark.parametrize('write', WRITES, ids=[w[0] for w in WRITES])
def test_write_of_null_key_is_allowed(write: tuple) -> None:
    daf = _keyed()
    write[1](daf)
    assert any(row_la[0] == '' for row_la in daf.lol)
    daf.set_keyfield('k')
    assert _kd_is_current(daf)


@pytest.mark.parametrize('write', WRITES, ids=[w[0] for w in WRITES])
def test_without_keyfield_null_is_allowed(write: tuple) -> None:
    daf = Daf(lol=[['r1', 1, 'a'], ['r2', 2, 'b']], cols=['k', 'x', 'y'])
    if write[0] in ('cell by key', 'row by key, dict'):
        pytest.skip('selects a row by key')
    write[1](daf)
    assert any(row_la[0] == '' for row_la in daf.lol)


def test_null_key_is_found() -> None:
    daf = _keyed()
    daf[0, 'k'] = ''
    daf.set_keyfield('k')
    assert daf['', 'x'] == 1


def test_composite_key_part_null_is_allowed() -> None:
    daf = _composite()
    daf[0, 'n'] = ''
    assert daf.lol[0] == ['a', '', 10]


def test_update_record_irow_keeps_the_key() -> None:
    daf = _keyed()
    daf.update_record_irow(0, {'x': 9})
    assert daf.lol[0] == ['r1', 9, 'a']
    assert daf['r1', 'x'] == 9


# --- adding rows ---

ADDS: list = [
    ('dict',              lambda d: d.append({'x': 3})),
    ('dict, key ""',      lambda d: d.append({'k': '', 'x': 3})),
    ('list',              lambda d: d.append(['', 3, 'c'])),
    ('short list',        lambda d: d.append(la=[])),
    ('KeyedList',         lambda d: d.append(KeyedList(['k', 'x', 'y'], ['', 3, 'c']))),
    ('lol',               lambda d: d.append(lol=[['r3', 3, 'c'], ['', 4, 'd']])),
    ('list of dicts',     lambda d: d.append([{'k': 'r3', 'x': 3}, {'x': 4}])),
    ('extend lol',        lambda d: d.extend(lol=[['r3', 3, 'c'], ['', 4, 'd']])),
    ('extend dicts',      lambda d: d.extend([{'k': 'r3', 'x': 3}, {'k': '', 'x': 4}])),
    ('respect_kd',        lambda d: d.append({'k': '', 'x': 3}, respect_kd=True)),
    ('record_append',     lambda d: d.record_append({'k': '', 'x': 3})),
    ('concat',            lambda d: d.concat(Daf(lol=[['r3', 3, 'c'], ['', 4, 'd']], cols=['k', 'x', 'y']))),
    ('fast list',         lambda d: d.append(['', 3, 'c'], fast=True)),
    ('fast dict',         lambda d: d.append({'k': '', 'x': 3, 'y': 'c'}, fast=True)),
    ('fast lol',          lambda d: d.append(lol=[['r3', 3, 'c'], ['', 4, 'd']], fast=True)),
    ('fast KeyedList',    lambda d: d.append(_blank_row(d), fast=True)),
]


def _blank_row(daf: Daf) -> Any:
    row = daf.default_record(astype=KeyedList)
    row['x'] = 3
    return row


@pytest.mark.parametrize('add', ADDS, ids=[a[0] for a in ADDS])
def test_add_with_null_key_is_allowed(add: tuple) -> None:
    daf = _keyed()
    if add[0] == 'short list':
        daf.append(la=['r3'])                           # a short list is padded with NULL.
        assert daf.lol[-1] == ['r3', '', '']
        return
    add[1](daf)
    assert len(daf) > 2
    assert any(row_la[0] == '' for row_la in daf.lol)
    assert _kd_is_current(daf)


def test_respect_kd_dict_without_key_raises_keyerror() -> None:
    """ As before the key checks: the key is read from the dict to look it up. """
    daf = _keyed()
    with pytest.raises(KeyError):
        daf.record_append({'x': 3})


@pytest.mark.parametrize('add', ADDS, ids=[a[0] for a in ADDS])
def test_add_without_keyfield_allows_null(add: tuple) -> None:
    daf = Daf(lol=[['r1', 1, 'a'], ['r2', 2, 'b']], cols=['k', 'x', 'y'])
    if add[0] == 'short list':
        pytest.skip('an empty list adds nothing')
    add[1](daf)
    assert len(daf) > 2


def test_rows_added_then_keyfield_set() -> None:
    """ The way to add rows whose keys are filled in later. """
    daf = Daf(cols=['k', 'x'])
    daf.append({'x': 1})
    daf[0, 'k'] = 'r1'
    daf.set_keyfield('k')
    daf.retmode = 'val'
    assert daf['r1', 'x'] == 1


# --- a write to a key cell or a row leaves the kd as it was ---

WRITE_RENAMES: list = [
    ('cell fast path',    lambda d: d.__setitem__((0, 'k'), 'q1')),
    ('cell by key',       lambda d: d.__setitem__(('r1', 'k'), 'q1')),
    ('row list',          lambda d: d.__setitem__(0, ['q1', 1, 'a'])),
    ('row dict',          lambda d: d.__setitem__(0, {'k': 'q1', 'x': 1, 'y': 'a'})),
    ('column list',       lambda d: d.__setitem__((slice(None), 'k'), ['q1', 'r2'])),
    ('assign_record_irow', lambda d: d.assign_record_irow(0, {'k': 'q1', 'x': 1, 'y': 'a'})),
    ('update_record_irow', lambda d: d.update_record_irow(0, {'k': 'q1'})),
]


@pytest.mark.parametrize('rename', WRITE_RENAMES, ids=[r[0] for r in WRITE_RENAMES])
def test_written_key_found_after_set_keyfield(rename: tuple) -> None:
    daf = _keyed()
    kd = daf._kd
    rename[1](daf)
    assert daf.lol[0][0] == 'q1'
    assert daf._kd is kd and kd == {'r1': 0, 'r2': 1}   # not touched, as documented.
    daf.set_keyfield('k')
    assert daf['q1', 'x'] == 1
    assert daf['r2', 'x'] == 2
    with pytest.raises(KeyError):
        daf['r1', 'x']


# --- a method that rewrites a key column invalidates the kd ---

RENAMES: list = [
    ('set_icol_irows',    lambda d: d.set_icol_irows(0, [0], 'q1')),
    ('set_col_irows',     lambda d: d.set_col_irows('k', [0], 'q1')),
    ('assign_col',        lambda d: d.assign_col('k', ['q1', 'r2'])),
    ('assign_icol',       lambda d: d.assign_icol(0, ['q1', 'r2'])),
    ('find_replace',      lambda d: d.find_replace('^r1$', 'q1')),
    ('apply_in_place',    lambda d: d.apply_in_place(lambda row: {**row, 'k': 'q1' if row['k'] == 'r1' else row['k']})),
]


@pytest.mark.parametrize('rename', RENAMES, ids=[r[0] for r in RENAMES])
def test_changed_key_is_found(rename: tuple) -> None:
    daf = _keyed()
    rename[1](daf)
    assert daf.lol[0][0] == 'q1'
    assert daf['q1', 'x'] == 1
    assert daf['r2', 'x'] == 2
    with pytest.raises(KeyError):
        daf['r1', 'x']


def test_set_icol_changes_every_key() -> None:
    daf = _keyed()
    daf.set_icol(0, 'same')
    assert daf.col('k') == ['same', 'same']
    assert daf._kd == {}
    assert daf.keys() == ['same']


def test_strip_and_apply_dtypes_change_keys() -> None:
    daf = Daf(lol=[[' r1 ', 1], [' r2 ', 2]], cols=['k', 'x'], keyfield='k')
    daf.retmode = 'val'
    assert daf[' r1 ', 'x'] == 1
    daf.strip()
    assert daf['r1', 'x'] == 1

    daf = Daf(lol=[['1', 'a'], ['2', 'b']], cols=['n', 'v'], keyfield='n')
    daf.retmode = 'val'
    assert daf['1', 'v'] == 'a'
    daf.apply_dtypes(dtypes={'n': int, 'v': str})
    assert daf[1, 'v'] == 'b'                           # by position, then by key below
    assert daf.krows_to_irows(krows=1) == [0]


def test_composite_key_change_found_after_set_keyfield() -> None:
    daf = _composite()
    assert daf.krows_to_irows(krows=[('a', 2)]) == [1]
    daf[1, 'n'] = 3
    daf.set_keyfield(['g', 'n'])
    assert daf.krows_to_irows(krows=[('a', 3)]) == [1]


Check = Callable[[Daf], Any]


@pytest.mark.parametrize('write', [
    lambda d: d.__setitem__((0, 'x'), 9),
    lambda d: d.__setitem__((0, 'k'), 'r1'),
    lambda d: d.__setitem__((slice(None), 'x'), 0),
    lambda d: d.update_record_irow(0, {'x': 9}),
    lambda d: d.assign_record_irow(0, {'k': 'r1', 'x': 9}),
    lambda d: d.set_col_irows('x', [0], 9),
], ids=['other cell', 'same key', 'other column', 'update other cell', 'same key, new row', 'set other column'])
def test_index_kept_when_no_key_changes(write: Check) -> None:
    daf = _keyed()
    kd = daf._kd
    write(daf)
    assert daf._kd is kd and kd == {'r1': 0, 'r2': 1}


def test_keyedlist_writes_need_set_keyfield() -> None:
    """ A key changed through a KeyedList row is not seen until set_keyfield() rebuilds the index. """
    daf = _keyed()
    for row in daf.iter_klist():
        row['k'] = row['k'].upper()
    daf.set_keyfield('k')
    assert daf['R1', 'x'] == 1


# ---- from test_daf_keyfield_not_a_column.py ----
# A keyfield that is not a column raises KeyError in the constructor, the builders and set_keyfield(). A Daf with no column
# names can still have one, for the columns that come later. If the attribute is edited directly, the failure is reported
# when a lookup by key is made, and the message says what is wrong.

def test_the_constructor_and_the_builders_raise_keyerror_for_a_keyfield_that_is_not_a_column():
    with pytest.raises(KeyError, match=r"keyfield 'zz' is not a column. The columns are \['a', 'b'\]"):
        Daf(lol=[[1, 2]], cols=['a', 'b'], keyfield='zz')
    with pytest.raises(KeyError, match="keyfield 'zz'"):
        Daf.from_csv_buff('a,b\n1,2\n', keyfield='zz')
    with pytest.raises(KeyError, match="is not a column"):
        Daf(lol=[[1, 2]], cols=['a', 'b'], keyfield=('a', 'zz'))


def test_set_keyfield_raises_keyerror_by_default_and_stores_only_if_asked():
    d = Daf(lol=[[1, 2]], cols=['a', 'b'])
    with pytest.raises(KeyError, match="set_keyfield.*keyfield 'zz'"):
        d.set_keyfield('zz')
    assert d.keyfield == ''
    d.set_keyfield('zz', silent_error=True)
    assert d.keyfield == 'zz'


def test_an_empty_daf_with_no_columns_can_have_a_keyfield_for_the_columns_that_come_later():
    d = Daf(keyfield='id')
    d.append({'id': 1, 'v': 'a'})
    assert d.columns() == ['id', 'v'] and d.keyfield == 'id' and d.keys() == [1]


def test_select_krows_names_a_keyfield_that_is_not_a_column_after_a_direct_edit():
    d = Daf(lol=[[1, 2], [3, 4]], cols=['a', 'b'])
    d.keyfield = 'zz'
    with pytest.raises(KeysDisabledError, match=r"keyfield 'zz' is not a column .*\['a', 'b'\].*set_keyfield"):
        d.select_krows([1])


def test_select_krows_names_a_composite_keyfield_with_a_bad_column_after_a_direct_edit():
    d = Daf(lol=[[1, 2]], cols=['a', 'b'])
    d.keyfield = ('a', 'zz')
    with pytest.raises(KeysDisabledError, match=r"keyfield \('a', 'zz'\) is not a column"):
        d.select_krows([(1, 2)])


def test_select_krows_names_a_keyfield_on_a_daf_with_no_column_names():
    d = Daf(lol=[[1, 2]], keyfield='a')
    with pytest.raises(KeysDisabledError, match="has no column names.*set_cols"):
        d.select_krows([1])


def test_a_valid_keyfield_still_works_and_still_raises_keyerror_for_a_missing_key():
    d = Daf(lol=[[1, 2], [3, 4]], cols=['a', 'b'], keyfield='a')
    assert d.select_krows([3]).lol == [[3, 4]]
    with pytest.raises(KeyError):
        d.select_krows([9])


def test_no_keyfield_and_no_kd_says_so():
    with pytest.raises(KeysDisabledError, match=r"select_krows\(\): key lookups are disabled, as the keyfield is not set and there is no key index"):
        Daf(lol=[[1, 2]], cols=['a', 'b']).select_krows([1])


def test_set_keyfield_with_silent_error_false_raises_keyerror_for_a_bad_column():
    d = Daf(lol=[[1, 2]], cols=['a', 'b'])
    with pytest.raises(KeyError):
        d.set_keyfield('zz', silent_error=False)


# ---- from test_daf_keyfield_follows_names.py ----
# rename_cols() and set_cols() keep the keyfield: it takes the new name of its column.
# With no column names yet, set_cols() keeps a keyfield that is one of the new names.

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


# ---- from test_daf_manual_kd.py ----
# A Daf may have a key index (kd) and no keyfield. The key lookups all honor it, and the messages say what is missing.

def _daf() -> Daf:
    return Daf(lol=[['x', 1], ['y', 2], ['z', 3]], cols=['n', 'v'], kd={'x': 0, 'y': 1, 'z': 2})


def test_the_daf_has_a_kd_and_no_keyfield():
    d = _daf()
    assert d.keyfield == '' and d._kd == {'x': 0, 'y': 1, 'z': 2}


def test_keys_lists_the_kd():
    assert _daf().keys() == ['x', 'y', 'z']
    assert list(_daf().keys(astype='view')) == ['x', 'y', 'z']


def test_select_krows_uses_the_kd():
    assert _daf().select_krows(['y']).lol == [['y', 2]]
    assert _daf().select_krows(['y'], inverse=True).lol == [['x', 1], ['z', 3]]


def test_select_records_daf_uses_the_kd():
    assert _daf().select_records_daf(['z', 'x']).lol == [['z', 3], ['x', 1]]


def test_remove_key_and_remove_keylist_use_the_kd():
    assert _daf().remove_key('y').lol == [['x', 1], ['z', 3]]
    assert _daf().remove_keylist(['x', 'z']).lol == [['y', 2]]


def test_krows_to_irows_and_select_record_use_the_kd():
    assert _daf().krows_to_irows(['y']) == [1]
    assert _daf().select_record('y') == {'n': 'y', 'v': 2}


def test_a_missing_key_still_raises_keyerror():
    with pytest.raises(KeyError):
        _daf().select_krows(['nope'])


NO_KEYS = Daf(lol=[['x', 1]], cols=['n', 'v'])


@pytest.mark.parametrize('method, arg', [
    ('select_krows', ['x']), ('select_records_daf', ['x']), ('remove_key', 'x'), ('remove_keylist', ['x']),
    ('krows_to_irows', ['x']), ('select_record', 'x'),
    ])
def test_no_keyfield_and_no_kd_names_the_method_and_what_is_missing(method, arg):
    with pytest.raises(KeysDisabledError, match=r"\(\): key lookups are disabled, as the keyfield is not set and there is no key index"):
        getattr(NO_KEYS, method)(arg)


def test_keys_with_silent_error_false_names_what_is_missing():
    assert NO_KEYS.keys() == []
    with pytest.raises(KeysDisabledError, match=r"keys\(\): key lookups are disabled"):
        NO_KEYS.keys(silent_error=False)


def test_assign_record_needs_a_keyfield_not_just_a_kd():
    with pytest.raises(KeysDisabledError, match="assign_record\\(\\): the Daf needs a keyfield"):
        _daf().assign_record({'n': 'x', 'v': 9})


def test_a_daf_with_no_rows_and_a_valid_keyfield_says_the_index_is_empty():
    with pytest.raises(KeysDisabledError, match="the rowkeys index is empty.*no rows has no keys"):
        Daf(cols=['a', 'b'], keyfield='a').select_krows([1])
