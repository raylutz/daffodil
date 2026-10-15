# test_daf_selection.py
#
# Tests for select_irows, select_icols, gkeys_to_idxs (and its krows_to_irows/kcols_to_icols/
# select_krows/select_kcols callers), and col_to_la.
#
# This batch surfaced several real bugs, now fixed:
#   - select_irows(int, invert=True) was destructively mutating the original Daf's lol.
#   - select_irows(slice(...), invert=True) raised TypeError (slice doesn't support `in`).
#   - select_irows/select_icols had a length-only "fast path" check (missing parens on
#     self.num_cols in one case) that, even once corrected naively, would have silently
#     ignored reordering/repeated-index selections of the same length as the original. Fixed
#     with a short-circuiting natural-order check instead.
#   - gkeys_to_idxs's slice branch was completely non-functional for any realistic keydict
#     (tried using ordinal positions as literal dict keys). Fixed to treat slice bounds as
#     pure ordinal positions, matching how [] indexing treats integer slices, while leaving
#     genuine key-range selection to the (start_key, stop_key) tuple form.

import pytest

from daffodil.daf import Daf
from daffodil.daf import Daf, KeysDisabledError
from daffodil.keyedlist import KeyedList


# =====================================================================
# select_irows
# =====================================================================

def test_select_irows_single_int():
    daf = Daf(lol=[[1, 'a'], [2, 'b'], [3, 'c']], cols=['id', 'name'])
    result = daf.select_irows(1)
    assert result.lol == [[2, 'b']]


def test_select_irows_int_invert_does_not_mutate_original():
    daf = Daf(lol=[[1, 'a'], [2, 'b'], [3, 'c']], cols=['id', 'name'])
    original_lol_id = id(daf.lol)
    result = daf.select_irows(1, invert=True)
    assert result.lol == [[1, 'a'], [3, 'c']]
    assert daf.lol == [[1, 'a'], [2, 'b'], [3, 'c']]  # original untouched
    assert id(daf.lol) == original_lol_id


def test_select_irows_negative_int_invert():
    daf = Daf(lol=[[1, 'a'], [2, 'b'], [3, 'c']], cols=['id', 'name'])
    result = daf.select_irows(-1, invert=True)
    assert result.lol == [[1, 'a'], [2, 'b']]
    assert daf.lol == [[1, 'a'], [2, 'b'], [3, 'c']]  # original untouched


def test_select_irows_list_of_ints():
    daf = Daf(lol=[[1, 'a'], [2, 'b'], [3, 'c']], cols=['id', 'name'])
    result = daf.select_irows([0, 2])
    assert result.lol == [[1, 'a'], [3, 'c']]


def test_select_irows_list_reorders_correctly_even_at_full_length():
    # this is the bug we found: a same-length-as-original but reordered selection must
    # actually reorder, not silently fall into a "reuse self.lol unchanged" fast path.
    daf = Daf(lol=[[1, 2], [3, 4], [5, 6]], cols=['a', 'b'])
    result = daf.select_irows([2, 0, 1])
    assert result.lol == [[5, 6], [1, 2], [3, 4]]


def test_select_irows_list_natural_order_still_correct():
    daf = Daf(lol=[[1, 2], [3, 4], [5, 6]], cols=['a', 'b'])
    result = daf.select_irows([0, 1, 2])
    assert result.lol == [[1, 2], [3, 4], [5, 6]]


def test_select_irows_list_of_ints_invert():
    daf = Daf(lol=[[1, 'a'], [2, 'b'], [3, 'c']], cols=['id', 'name'])
    result = daf.select_irows([0, 2], invert=True)
    assert result.lol == [[2, 'b']]


def test_select_irows_list_of_ranges():
    daf = Daf(lol=[[1, 'a'], [2, 'b'], [3, 'c'], [4, 'd']], cols=['id', 'name'])
    result = daf.select_irows([range(0, 2)])
    assert result.lol == [[1, 'a'], [2, 'b']]


def test_select_irows_generic_iterable():
    daf = Daf(lol=[[1, 'a'], [2, 'b'], [3, 'c']], cols=['id', 'name'])
    result = daf.select_irows(iter([0, 2]))
    assert result.lol == [[1, 'a'], [3, 'c']]


def test_select_irows_slice():
    daf = Daf(lol=[[1, 'a'], [2, 'b'], [3, 'c'], [4, 'd']], cols=['id', 'name'])
    result = daf.select_irows(slice(0, 2))
    assert result.lol == [[1, 'a'], [2, 'b']]


def test_select_irows_slice_invert_no_longer_raises():
    daf = Daf(lol=[[1, 'a'], [2, 'b'], [3, 'c'], [4, 'd']], cols=['id', 'name'])
    result = daf.select_irows(slice(0, 2), invert=True)
    assert result.lol == [[3, 'c'], [4, 'd']]


def test_select_irows_empty_selection_no_invert():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    result = daf.select_irows([])
    assert result.lol == []


def test_select_irows_empty_selection_invert():
    daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'name'])
    result = daf.select_irows([], invert=True)
    assert result.lol == [[1, 'a'], [2, 'b']]


def test_select_irows_empty_daf():
    daf = Daf(lol=[], cols=['id', 'name'])
    result = daf.select_irows(0)
    assert result.lol == []


# =====================================================================
# select_icols
# =====================================================================

def test_select_icols_single_int():
    daf = Daf(lol=[[1, 'a', True], [2, 'b', False]], cols=['id', 'name', 'flag'])
    result = daf.select_icols(0)
    assert result.lol == [[1], [2]]
    assert list(result.hd.keys()) == ['id']


def test_select_icols_single_int_flip():
    daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'name'])
    result = daf.select_icols(0, flip=True)
    assert result.lol == [1, 2]


def test_select_icols_slice():
    daf = Daf(lol=[[1, 'a', True], [2, 'b', False]], cols=['id', 'name', 'flag'])
    result = daf.select_icols(slice(0, 2))
    assert result.lol == [[1, 'a'], [2, 'b']]
    assert list(result.hd.keys()) == ['id', 'name']


def test_select_icols_slice_flip():
    daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'name'])
    result = daf.select_icols(slice(0, 2), flip=True)
    assert result.lol == [[1, 2], ['a', 'b']]


def test_select_icols_empty_slice_returns_empty_daf():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    result = daf.select_icols(slice(5, 5))
    assert result.lol == []


def test_select_icols_list_of_ints():
    daf = Daf(lol=[[1, 'a', True], [2, 'b', False]], cols=['id', 'name', 'flag'])
    result = daf.select_icols([0, 2])
    assert result.lol == [[1, True], [2, False]]
    assert list(result.hd.keys()) == ['id', 'flag']


def test_select_icols_reorders_correctly_even_at_full_length():
    # this is the bug we found: a same-length-as-original but reordered/repeated selection
    # must actually reorder/repeat, not silently fall into a "reuse self.lol unchanged" path
    # (which previously also left column labels mismatched with the actual data).
    daf = Daf(lol=[[1, 2, 3], [4, 5, 6]], cols=['a', 'b', 'c'])
    result = daf.select_icols([2, 0, 1])
    assert result.lol == [[3, 1, 2], [6, 4, 5]]
    assert list(result.hd.keys()) == ['c', 'a', 'b']


def test_select_icols_repeated_index_same_length():
    daf = Daf(lol=[[1, 2, 3], [4, 5, 6]], cols=['a', 'b', 'c'])
    result = daf.select_icols([0, 0, 0])
    assert result.lol == [[1, 1, 1], [4, 4, 4]]


def test_select_icols_natural_order_still_correct():
    daf = Daf(lol=[[1, 2, 3], [4, 5, 6]], cols=['a', 'b', 'c'])
    result = daf.select_icols([0, 1, 2])
    assert result.lol == [[1, 2, 3], [4, 5, 6]]


def test_select_icols_list_flip():
    daf = Daf(lol=[[1, 'a', True], [2, 'b', False]], cols=['id', 'name', 'flag'])
    result = daf.select_icols([0, 2], flip=True)
    assert result.lol == [[1, 2], [True, False]]


def test_select_icols_empty_list():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    result = daf.select_icols([])
    assert result.lol == []


def test_select_icols_range():
    daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'name'])
    result = daf.select_icols(range(0, 2))
    assert result.lol == [[1, 'a'], [2, 'b']]


def test_select_icols_list_of_ranges():
    daf = Daf(lol=[[1, 'a', True], [2, 'b', False]], cols=['id', 'name', 'flag'])
    result = daf.select_icols([range(0, 2)])
    assert result.lol == [[1, 'a'], [2, 'b']]
    assert list(result.hd.keys()) == ['id', 'name']


def test_select_icols_none_no_flip_returns_self():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    assert daf.select_icols(None) is daf


def test_select_icols_none_flip_transposes_all():
    daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'name'])
    result = daf.select_icols(None, flip=True)
    assert result.lol == [[1, 2], ['a', 'b']]


def test_select_icols_drops_keyfield_dtype_when_col_removed():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'], keyfield='id', dtypes={'id': int, 'name': str})
    result = daf.select_icols([1])  # drop the keyfield column
    assert result.keyfield == ''
    assert result.dtypes == {'name': str}


# =====================================================================
# gkeys_to_idxs (and its krows_to_irows/kcols_to_icols/select_krows/select_kcols callers)
# =====================================================================

def test_gkeys_to_idxs_single_string_key():
    kd = {'a': 0, 'b': 1, 'c': 2}
    assert Daf.gkeys_to_idxs(kd, 'b') == [1]


def test_gkeys_to_idxs_single_int_key():
    kd = {10: 0, 20: 1, 30: 2}
    assert Daf.gkeys_to_idxs(kd, 20) == [1]


def test_gkeys_to_idxs_list_of_keys():
    kd = {'a': 0, 'b': 1, 'c': 2}
    assert Daf.gkeys_to_idxs(kd, ['a', 'c']) == [0, 2]


def test_gkeys_to_idxs_empty_list():
    kd = {'a': 0, 'b': 1}
    assert Daf.gkeys_to_idxs(kd, []) == []


def test_gkeys_to_idxs_tuple_single_start():
    kd = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    assert Daf.gkeys_to_idxs(kd, ('b',)) == slice(1, 4, 1)


def test_gkeys_to_idxs_tuple_start_stop_inclusive():
    kd = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    assert Daf.gkeys_to_idxs(kd, ('b', 'c')) == slice(1, 3, 1)


def test_gkeys_to_idxs_tuple_none_start():
    kd = {'a': 0, 'b': 1, 'c': 2}
    assert Daf.gkeys_to_idxs(kd, (None, 'b')) == slice(0, 2, 1)


def test_gkeys_to_idxs_tuple_none_stop():
    kd = {'a': 0, 'b': 1, 'c': 2}
    assert Daf.gkeys_to_idxs(kd, ('b', None)) == slice(1, 3, 1)


def test_gkeys_to_idxs_empty_tuple_is_empty_selection():
    # an empty tuple is caught by the earlier "empty selection" check (same as empty list),
    # before ever reaching the tuple-specific n==0 TypeError branch.
    assert Daf.gkeys_to_idxs({'a': 0}, ()) == []


def test_gkeys_to_idxs_tuple_too_long_raises():
    with pytest.raises(TypeError):
        Daf.gkeys_to_idxs({'a': 0}, ('a', 'b', 'c'))


def test_gkeys_to_idxs_tuple_none_alone_raises():
    with pytest.raises(TypeError):
        Daf.gkeys_to_idxs({'a': 0}, (None,))


def test_gkeys_to_idxs_slice_ordinal_positions():
    # slice bounds are ordinal positions, NOT keys to look up -- this was the bug.
    kd = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    assert Daf.gkeys_to_idxs(kd, slice(0, 2)) == slice(0, 2, 1)


def test_gkeys_to_idxs_slice_open_ended():
    kd = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    assert Daf.gkeys_to_idxs(kd, slice(None, None)) == slice(0, 4, 1)
    assert Daf.gkeys_to_idxs(kd, slice(1, None)) == slice(1, 4, 1)


def test_gkeys_to_idxs_slice_inverse():
    kd = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    assert Daf.gkeys_to_idxs(kd, slice(0, 2), inverse=True) == [2, 3]


def test_gkeys_to_idxs_missing_key_raises():
    kd = {'a': 0}
    with pytest.raises(KeyError):
        Daf.gkeys_to_idxs(kd, 'z')


def test_gkeys_to_idxs_missing_key_silent_error():
    kd = {'a': 0}
    assert Daf.gkeys_to_idxs(kd, 'z', silent_error=True) == []


def test_gkeys_to_idxs_no_keydict_raises():
    from daffodil.daf import KeysDisabledError
    with pytest.raises(KeysDisabledError):
        Daf.gkeys_to_idxs({}, 'a')


def test_gkeys_to_idxs_none_gkeys_raises():
    with pytest.raises(TypeError):
        Daf.gkeys_to_idxs({'a': 0}, None)


def test_gkeys_to_idxs_list_inverse():
    kd = {'a': 0, 'b': 1, 'c': 2}
    assert Daf.gkeys_to_idxs(kd, ['a', 'c'], inverse=True) == [1]


# --- end-to-end via select_krows / select_kcols ---

def test_select_krows_with_numeric_slice():
    daf = Daf(lol=[[1, 'a'], [2, 'b'], [3, 'c'], [4, 'd']], cols=['id', 'name'], keyfield='id')
    result = daf.select_krows(slice(0, 2))
    assert result.lol == [[1, 'a'], [2, 'b']]


def test_select_krows_with_key_tuple():
    daf = Daf(lol=[[1, 'a'], [2, 'b'], [3, 'c'], [4, 'd']], cols=['id', 'name'], keyfield='id')
    result = daf.select_krows((2, 3))
    assert result.lol == [[2, 'b'], [3, 'c']]


def test_select_kcols_with_numeric_slice():
    daf = Daf(lol=[[1, 'a', True], [2, 'b', False]], cols=['id', 'name', 'flag'])
    result = daf.select_kcols(slice(0, 2))
    assert result.lol == [[1, 'a'], [2, 'b']]


def test_bracket_syntax_with_string_slice_bypasses_key_translation():
    # documents the architectural finding: [] syntax routes ANY slice (including string-bounded)
    # straight to the plain-index path, never through krows_to_irows/gkeys_to_idxs. Only the
    # explicit select_krows()/select_kcols() methods do key-based slicing.
    daf = Daf(lol=[[1, 'a'], [2, 'b'], [3, 'c']], cols=['id', 'name'], keyfield='id')
    with pytest.raises(TypeError):
        daf['1':'2']


# =====================================================================
# col_to_la
# =====================================================================

def test_col_to_la_basic():
    daf = Daf(lol=[[1, 'a'], [2, 'b'], [1, 'c']], cols=['id', 'name'])
    assert daf.col_to_la('name') == ['a', 'b', 'c']


def test_col_to_la_unique():
    daf = Daf(lol=[[1, 'a'], [2, 'b'], [1, 'c']], cols=['id', 'name'])
    assert daf.col_to_la('id', unique=True) == [1, 2]


def test_col_to_la_astype():
    daf = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'name'])
    assert daf.col_to_la('id', astype=str) == ['1', '2']


def test_col_to_la_empty_colname_raises():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    with pytest.raises(RuntimeError):
        daf.col_to_la('')


def test_col_to_la_missing_col_raises():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    with pytest.raises(RuntimeError):
        daf.col_to_la('missing')


def test_col_to_la_missing_col_silent_error():
    daf = Daf(lol=[[1, 'a']], cols=['id', 'name'])
    assert daf.col_to_la('missing', silent_error=True) == []


def test_col_to_la_indirect_col():
    daf = Daf(lol=[[1, {'x': 10}], [2, {'x': 20}]], cols=['id', 'meta'])
    assert daf.col_to_la('x', indirect_col='meta') == [10, 20]


def test_col_to_la_indirect_col_missing_key_uses_default():
    daf = Daf(lol=[[1, {'x': 10}], [2, {'x': 20}]], cols=['id', 'meta'])
    assert daf.col_to_la('y', indirect_col='meta', default=-1) == [-1, -1]


# ---- from test_daf_select_records_daf_keys.py ----
# Tests for select_records_daf() given another Daf as the keys (Ray, 2026-10-14).

def make_daf() -> Daf:
    return Daf(lol=[['r1', 'a'], ['r2', 'b'], ['r3', 'c']], cols=['id', 'v'], keyfield='id')


def make_other_daf() -> Daf:
    return Daf(lol=[['r3', 1], ['r9', 2], ['r1', 3]], cols=['id', 'w'], keyfield='id')


def test_a_daf_as_keys_gives_the_same_as_its_keys():
    data_daf: Daf = make_daf()
    other_daf: Daf = make_other_daf()
    assert data_daf.select_records_daf(other_daf, silent_error=True).lol == \
           data_daf.select_records_daf(other_daf.keys(), silent_error=True).lol == [['r3', 'c'], ['r1', 'a']]


def test_a_daf_as_keys_with_inverse():
    data_daf: Daf = make_daf()
    assert data_daf.select_records_daf(make_other_daf(), inverse=True, silent_error=True).lol == [['r2', 'b']]


def test_a_daf_as_keys_missing_key_raises_unless_silent():
    with pytest.raises(KeyError):
        make_daf().select_records_daf(make_other_daf())


def test_a_daf_as_keys_shares_rows_and_leaves_the_other_unchanged():
    data_daf: Daf = make_daf()
    other_daf: Daf = make_other_daf()
    selected_daf: Daf = data_daf.select_records_daf(other_daf, silent_error=True)
    assert selected_daf.lol[0] is data_daf.lol[2]
    assert other_daf.lol == [['r3', 1], ['r9', 2], ['r1', 3]]


def test_a_daf_without_keyfield_as_keys_raises():
    no_key_daf: Daf = Daf(lol=[['r1']], cols=['id'])
    with pytest.raises(KeysDisabledError):
        make_daf().select_records_daf(no_key_daf)


def test_an_empty_daf_as_keys():
    empty_daf: Daf = Daf(cols=['id'], keyfield='id')
    assert make_daf().select_records_daf(empty_daf).lol == []
    assert make_daf().select_records_daf(empty_daf, inverse=True).lol == make_daf().lol


# ---- from test_daf_inverse_key_selection.py ----
# Inverse key selection gives the same rows after the fix in gkeys_to_idxs(), which made it linear (2026-10-14).

def make_numbered_daf(n: int) -> Daf:
    return Daf(lol=[[f'k{i:06d}', i] for i in range(n)], cols=['id', 'n'], keyfield='id')


def test_inverse_keeps_the_others_in_order():
    data_daf: Daf = make_numbered_daf(10)
    assert data_daf.select_krows(['k000007', 'k000002', 'k000005'], inverse=True)[:, 'n'].to_list() == [0, 1, 3, 4, 6, 8, 9]


def test_inverse_with_repeated_keys():
    data_daf: Daf = make_numbered_daf(5)
    assert data_daf.select_records_daf(['k000001', 'k000001', 'k000003'], inverse=True)[:, 'n'].to_list() == [0, 2, 4]


def test_inverse_of_half_of_a_large_table():
    data_daf: Daf = make_numbered_daf(200_000)
    keys_ls: list = [f'k{i:06d}' for i in range(0, 200_000, 2)]
    kept_daf: Daf = data_daf.select_records_daf(keys_ls, inverse=True)
    assert len(kept_daf) == 100_000
    assert kept_daf[:3, 'n'].to_list() == [1, 3, 5]
    assert kept_daf.lol[0] is data_daf.lol[1]


# ---- from test_daf_select_where_idxs.py ----
# select_where_idxs() gives its function a KeyedList for each row, as select_where() does.

def _daf() -> Daf:
    return Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'a', 30]], cols=['id', 'v', 'n'], keyfield='id')


def test_the_function_gets_a_keyedlist_for_every_row():
    seen = []
    _daf().select_where_idxs(lambda row: seen.append(type(row)) or True)
    assert seen == [KeyedList] * 3


def test_the_function_gets_a_keyedlist_in_every_itermode():
    d = _daf()
    d._itermode = Daf.ITERMODE_DICT
    seen = []
    d.select_where_idxs(lambda row: seen.append(type(row)) or True)
    assert seen == [KeyedList] * 3


def test_positions_of_the_matching_rows():
    d = _daf()
    assert d.select_where_idxs(lambda row: row['v'] == 'a') == [0, 2]
    assert d.select_where_idxs(lambda row: row['n'] > 10) == [1, 2]
    assert d.select_where_idxs(lambda row: row['n'] > 99) == []
    assert d.select_where_idxs(lambda row: True) == [0, 1, 2]


def test_positions_agree_with_select_where():
    d = _daf()
    test = lambda row: row['n'] >= 20 and row['v'] == 'a'
    assert [d.lol[i] for i in d.select_where_idxs(test)] == d.select_where(test).lol


def test_on_a_slice_the_positions_are_those_of_the_slice():
    d = _daf()
    assert d[1:].select_where_idxs(lambda row: row['v'] == 'a') == [1]


def test_empty_daf_gives_no_positions():
    assert Daf(cols=['id', 'v']).select_where_idxs(lambda row: True) == []


def test_daf_with_rows_and_no_column_names_raises():
    with pytest.raises(KeysDisabledError):
        Daf(lol=[[1, 2]]).select_where_idxs(lambda row: True)


# ---- from test_daf_select_icols.py ----
# Column slices through select_icols() and through my_daf[:, slice].
# Python's own slicing is the model: my_daf[:, a:b:c] should return the columns
# that list(range(n))[a:b:c] gives.

def _icols_daf() -> Daf:
    return Daf(lol=[[1, 'a', 10], [2, 'b', 20]], cols=['id', 'v', 'n'])


def test_select_icols_positive_slice():
    result = _icols_daf()[:, 1:3]
    assert result.columns() == ['v', 'n']
    assert result.lol == [['a', 10], ['b', 20]]


def test_select_icols_negative_start_and_stop():
    result = _icols_daf()[:, -3:-1]
    assert result.columns() == ['id', 'v']
    assert result.lol == [[1, 'a'], [2, 'b']]


def test_select_icols_step():
    result = _icols_daf()[:, ::2]
    assert result.columns() == ['id', 'n']
    assert result.lol == [[1, 10], [2, 20]]


def test_select_icols_negative_start_no_stop():
    result = _icols_daf()[:, -2:]
    assert result.columns() == ['v', 'n']
    assert result.lol == [['a', 10], ['b', 20]]


def test_select_icols_negative_stop_no_start():
    result = _icols_daf()[:, :-1]
    assert result.columns() == ['id', 'v']
    assert result.lol == [[1, 'a'], [2, 'b']]


def test_select_icols_positive_start_negative_stop():
    result = _icols_daf()[:, 1:-1]
    assert result.columns() == ['v']
    assert result.lol == [['a'], ['b']]


def test_select_icols_negative_step():
    result = _icols_daf()[:, ::-1]
    assert result.columns() == ['n', 'v', 'id']


def test_select_icols_stop_zero_is_empty():
    result = _icols_daf()[:, 0:0]
    assert result.columns() == []


def test_select_icols_stop_beyond_end_is_clamped():
    result = _icols_daf()[:, 1:10]
    assert result.columns() == ['v', 'n']


def test_select_icols_negative_slice_with_flip():
    result = _icols_daf().select_icols(slice(-2, None), flip=True)
    assert result.lol == [['a', 'b'], [10, 20]]


# ---- from test_daf_delete_records.py ----
# Tests for delete_records() (Ray, 2026-10-14, decision 104 and item 2 of the mdaf review: inplace=False by default).

def make_delete_daf() -> Daf:
    return Daf(lol=[['r1', 'a'], ['r2', 'b'], ['r3', 'c'], ['r4', 'd']], cols=['id', 'v'], keyfield='id')


def test_default_returns_a_new_daf_and_leaves_this_one():
    data_daf: Daf = make_delete_daf()
    result_daf: Daf = data_daf.delete_records(['r2', 'r4'])
    assert result_daf.lol == [['r1', 'a'], ['r3', 'c']]
    assert data_daf.lol == make_delete_daf().lol
    assert result_daf.lol[0] is data_daf.lol[0]
    assert result_daf == data_daf.select_records_daf(['r2', 'r4'], inverse=True)


def test_inplace_deletes_from_this_daf_and_returns_it():
    data_daf: Daf = make_delete_daf()
    row_list_la: list = data_daf.lol
    result_daf: Daf = data_daf.delete_records(['r4', 'r2'], inplace=True)
    assert result_daf is data_daf
    assert data_daf.lol == [['r1', 'a'], ['r3', 'c']]
    assert data_daf.lol is row_list_la
    assert data_daf['r3', 'v'].to_value() == 'c'
    assert data_daf.keys() == ['r1', 'r3']


def test_inplace_leaves_an_earlier_selection_whole():
    data_daf: Daf = make_delete_daf()
    sel_daf: Daf = data_daf.select_where(lambda row: True)
    data_daf.delete_records(['r2'], inplace=True)
    assert sel_daf.lol == make_delete_daf().lol


def test_missing_key_raises_and_removes_nothing():
    data_daf: Daf = make_delete_daf()
    with pytest.raises(KeyError):
        data_daf.delete_records(['r2', 'r9'], inplace=True)
    assert data_daf.lol == make_delete_daf().lol
    with pytest.raises(KeyError):
        data_daf.delete_records(['r9'])


def test_silent_error_ignores_missing_keys():
    data_daf: Daf = make_delete_daf()
    assert data_daf.delete_records(['r2', 'r9'], silent_error=True).lol == [['r1', 'a'], ['r3', 'c'], ['r4', 'd']]
    data_daf.delete_records(['r9'], inplace=True, silent_error=True)
    assert data_daf.lol == make_delete_daf().lol


def test_a_daf_as_keys():
    data_daf: Daf = make_delete_daf()
    other_daf: Daf = Daf(lol=[['r3', 0], ['r1', 0]], cols=['id', 'x'], keyfield='id')
    assert data_daf.delete_records(other_daf).lol == [['r2', 'b'], ['r4', 'd']]
    data_daf.delete_records(other_daf, inplace=True)
    assert data_daf.lol == [['r2', 'b'], ['r4', 'd']]
    with pytest.raises(KeysDisabledError):
        data_daf.delete_records(Daf(lol=[['r2']], cols=['id']))


def test_no_keys():
    data_daf: Daf = make_delete_daf()
    assert data_daf.delete_records([]).lol == make_delete_daf().lol
    assert data_daf.delete_records([], inplace=True) is data_daf
    assert data_daf.lol == make_delete_daf().lol
