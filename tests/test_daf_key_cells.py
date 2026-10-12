# test_daf_key_cells.py
#
# The key index (kd) must follow the keys.
#   - A method that changes a keyfield cell invalidates the kd, so a lookup finds the new key.
#   - NULL is a valid key. Rows with NULL keys can be written and added, with or without a keyfield.

from typing import Any, Callable

import pytest
from daffodil.daf import Daf
from daffodil.keyedlist import KeyedList


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
    assert daf['', 'x'] == 1
    assert _kd_is_current(daf)


def test_composite_key_part_null_is_allowed() -> None:
    daf = _composite()
    daf[0, 'n'] = ''
    assert daf.lol[0] == ['a', '', 10]
    assert _kd_is_current(daf)


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


# --- the key index follows changed keys ---

RENAMES: list = [
    ('cell fast path',    lambda d: d.__setitem__((0, 'k'), 'q1')),
    ('cell by key',       lambda d: d.__setitem__(('r1', 'k'), 'q1')),
    ('row list',          lambda d: d.__setitem__(0, ['q1', 1, 'a'])),
    ('row dict',          lambda d: d.__setitem__(0, {'k': 'q1', 'x': 1, 'y': 'a'})),
    ('column list',       lambda d: d.__setitem__((slice(None), 'k'), ['q1', 'r2'])),
    ('assign_record_irow', lambda d: d.assign_record_irow(0, {'k': 'q1', 'x': 1, 'y': 'a'})),
    ('update_record_irow', lambda d: d.update_record_irow(0, {'k': 'q1'})),
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


def test_composite_key_change_is_found() -> None:
    daf = _composite()
    assert daf.krows_to_irows(krows=[('a', 2)]) == [1]
    daf[1, 'n'] = 3
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
