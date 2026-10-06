# test_daf_select_by_dict.py
#
# select_by_dict() compares the cells by position. It must give the same rows as matching each row as a dict.

import random

import pytest

from daffodil.daf import Daf, KeysDisabledError


def _reference(daf: Daf, selector: dict, inverse: bool) -> list:
    return [list(row.values()) for row in daf.iter_dict() if inverse ^ all(row.get(k, object()) == v for k, v in selector.items())]


def _daf(rows=None) -> Daf:
    rows = rows if rows is not None else [[1, 'a', 10], [2, 'b', 20], [3, 'a', 30], [4, 'b', 20]]
    return Daf(lol=rows, cols=['id', 'v', 'n'], keyfield='id')


@pytest.mark.parametrize('selector', [{}, {'v': 'a'}, {'v': 'b', 'n': 20}, {'v': 'a', 'n': 20}, {'zz': 1}, {'v': 'a', 'zz': 1}, {'n': [1]}, {'v': None}])
@pytest.mark.parametrize('inverse', [False, True])
def test_select_by_dict_matches_the_row_by_row_rule(selector, inverse):
    d = _daf()
    assert d.select_by_dict(selector, inverse=inverse).lol == _reference(d, selector, inverse)


def test_select_by_dict_matches_the_rule_on_mixed_random_cells():
    random.seed(3)
    cells = [0, 1, 2, 'a', '', None, 1.0, True, (1, 2), [1, 2]]
    for _ in range(100):
        rows = [[random.choice(cells) for _ in range(3)] for _ in range(random.randint(0, 10))]
        d = _daf(rows)
        for selector in ({}, {'id': 1}, {'v': 'a', 'n': 1}, {'n': [1, 2]}, {'v': None}):
            for inverse in (False, True):
                assert d.select_by_dict(selector, inverse=inverse).lol == _reference(d, selector, inverse)


def test_select_by_dict_empty_selector_and_unknown_column():
    d = _daf()
    assert d.select_by_dict({}).lol == d.lol
    assert d.select_by_dict({}, inverse=True).lol == []
    assert d.select_by_dict({'zz': 1}).lol == []
    assert d.select_by_dict({'zz': 1}, inverse=True).lol == d.lol


def test_select_by_dict_expectmax_and_keyfield():
    d = _daf()
    assert len(d.select_by_dict({'v': 'a'}, expectmax=2)) == 2
    with pytest.raises(LookupError, match='select_by_dict'):
        d.select_by_dict({'v': 'a'}, expectmax=1)
    assert d.select_by_dict({'v': 'a'}).keyfield == 'id'
    assert d.select_by_dict({'v': 'a'}, keyfield='n').keyfield == 'n'


def test_select_by_dict_on_a_daf_with_no_column_names_raises():
    d = Daf(lol=[[1, 2]])
    with pytest.raises(KeysDisabledError):
        d.select_by_dict({'a': 1})


def test_select_by_dict_on_an_empty_daf_returns_an_empty_daf():
    assert Daf(cols=['id', 'v']).select_by_dict({'v': 'a'}).lol == []


def test_select_by_dict_result_shares_the_rows():
    d = _daf()
    result = d.select_by_dict({'v': 'a'})
    assert result.lol[0] is d.lol[0] and result.lol is not d.lol


# =====================================================================
# a list of dicts: a row matches if it matches any one of them
# =====================================================================


def _reference_ids(daf: Daf, loda: list, inverse: bool = False) -> list:
    """The meaning of a list of dicts: any dict, all of whose fields are == to the cells."""
    hd = daf.hd
    ids = []
    for row in daf.lol:
        hit = any(all(c in hd and row[hd[c]] == v for c, v in da.items()) for da in loda)
        if hit is not inverse:
            ids.append(row[0])
    return ids


def _members() -> Daf:
    return Daf(lol=[[1, 'a', 10], [2, 'b', 20], [3, 'c', 30], [4, 'a', 20]], cols=['id', 'v', 'n'], keyfield='id')


def test_list_of_dicts_selects_rows_matching_any_one_value_of_a_column():
    assert _members().select_by_dict([{'v': 'a'}, {'v': 'c'}]).col('id') == [1, 3, 4]


def test_list_of_dicts_selects_by_several_columns_together():
    d = _members()
    assert d.select_by_dict([{'v': 'a', 'n': 20}, {'v': 'c', 'n': 30}]).col('id') == [3, 4]
    assert d.select_by_dict([{'v': 'a', 'n': 30}]).col('id') == []


def test_list_of_dicts_with_different_keys_in_each_dict():
    d = _members()
    assert d.select_by_dict([{'v': 'b'}, {'n': 30}, {'id': 1, 'v': 'a'}]).col('id') == [1, 2, 3]


def test_list_of_dicts_inverse_negates_the_whole_match():
    d = _members()
    assert d.select_by_dict([{'v': 'a'}, {'n': 30}], inverse=True).col('id') == [2]


def test_an_empty_list_matches_no_row_and_inverse_matches_every_row():
    d = _members()
    assert d.select_by_dict([]).num_rows() == 0
    assert d.select_by_dict([], inverse=True).col('id') == [1, 2, 3, 4]


def test_an_empty_dict_in_the_list_matches_every_row():
    d = _members()
    assert d.select_by_dict([{'v': 'zz'}, {}]).col('id') == [1, 2, 3, 4]
    assert d.select_by_dict([{'v': 'zz'}, {}], inverse=True).num_rows() == 0


def test_a_dict_with_an_unknown_column_matches_nothing_and_the_others_still_apply():
    d = _members()
    assert d.select_by_dict([{'nope': 1}, {'v': 'b'}]).col('id') == [2]
    assert d.select_by_dict([{'nope': 1}]).num_rows() == 0
    assert d.select_by_dict([{'nope': 1}], inverse=True).col('id') == [1, 2, 3, 4]


def test_an_item_that_is_not_a_dict_raises_typeerror():
    with pytest.raises(TypeError, match='must be a dict'):
        _members().select_by_dict([{'v': 'a'}, 'b'])


def test_a_selector_that_is_not_a_dict_or_a_list_raises_typeerror():
    with pytest.raises(TypeError, match='selector_da'):
        _members().select_by_dict('v')       # type: ignore[arg-type]


def test_a_tuple_of_dicts_is_accepted_and_the_list_is_not_changed():
    d = _members()
    loda = [{'v': 'a'}, {'v': 'c'}]
    before = [dict(da) for da in loda]
    assert d.select_by_dict(tuple(loda)).col('id') == [1, 3, 4]
    assert d.select_by_dict(loda).col('id') == [1, 3, 4]
    assert loda == before


def test_list_of_dicts_with_expectmax_and_keyfield_and_shared_rows():
    d = _members()
    with pytest.raises(LookupError):
        d.select_by_dict([{'v': 'a'}, {'v': 'c'}], expectmax=2)
    assert d.select_by_dict([{'v': 'a'}], expectmax=2).num_rows() == 2
    result = d.select_by_dict([{'v': 'a'}], keyfield='v')
    assert result.keyfield == 'v' and d.keyfield == 'id'
    assert result.lol[0] is d.lol[0] and result.lol is not d.lol
    assert result.dtypes == d.dtypes


def test_list_of_dicts_on_a_daf_with_rows_and_no_column_names_raises():
    with pytest.raises(KeysDisabledError):
        Daf(lol=[[1, 2]]).select_by_dict([{'a': 1}])


def test_list_of_dicts_on_a_daf_with_no_rows():
    d = Daf(cols=['a', 'b'])
    assert d.select_by_dict([{'a': 1}]).num_rows() == 0
    assert d.select_by_dict([{'a': 1}], inverse=True).num_rows() == 0


def test_a_dict_still_compares_a_collection_value_by_equality():
    d = Daf(lol=[[1, {'a', 'b'}], [2, ['x']], [3, 'a'], [4, ('x',)]], cols=['id', 'v'])
    assert d.select_by_dict({'v': {'a', 'b'}}).col('id') == [1]
    assert d.select_by_dict({'v': ['x']}).col('id') == [2]
    assert d.select_by_dict({'v': 'a'}).col('id') == [3]


def test_cells_that_cannot_be_hashed_never_raise_and_match_as_equals_does():
    d = Daf(lol=[[1, {'a', 'b'}], [2, ['x']], [3, 'a'], [4, {'c'}], [5, ['x']], [6, True], [7, 1.0], [8, 1]],
            cols=['id', 'v'])
    for loda in ([{'v': 'a'}, {'v': 1}],
                 [{'v': {'a', 'b'}}],
                 [{'v': ['x']}],
                 [{'v': 'a'}, {'v': {'c'}}, {'v': ['x']}],
                 [{'v': 1}],
                 [{'v': 'zz'}]):
        assert d.select_by_dict(loda).col('id') == _reference_ids(d, loda)
        assert d.select_by_dict(loda, inverse=True).col('id') == _reference_ids(d, loda, inverse=True)


def test_one_equals_true_equals_one_point_zero_as_a_dict_does():
    d = Daf(lol=[[1, True], [2, 1.0], [3, 1], [4, 2], [5, False], [6, 0]], cols=['id', 'v'])
    assert d.select_by_dict([{'v': 1}]).col('id') == [1, 2, 3]
    assert d.select_by_dict([{'v': 0}, {'v': 2}]).col('id') == [4, 5, 6]


def test_unhashable_selector_values_in_a_group_with_hashable_ones():
    d = Daf(lol=[[1, 'a', ['x']], [2, 'b', ['y']], [3, 'a', ['y']], [4, 'c', ['x']]], cols=['id', 'v', 'w'])
    loda = [{'v': 'a', 'w': ['y']}, {'v': 'c', 'w': ['x']}, {'v': 'b'}]
    assert d.select_by_dict(loda).col('id') == _reference_ids(d, loda) == [2, 3, 4]


def test_the_same_dict_twice_and_dicts_with_the_keys_in_a_different_order():
    d = _members()
    assert d.select_by_dict([{'v': 'a', 'n': 20}, {'n': 20, 'v': 'a'}, {'v': 'a', 'n': 20}]).col('id') == [4]


def test_list_of_dicts_agrees_with_the_meaning_for_random_cases():
    rng = random.Random(7)
    values = [0, 1, 2, 'a', 'b', 1.0, True, None, ('t',), ['l'], {'s'}]
    for _ in range(300):
        rows = [[i] + [rng.choice(values) for _ in range(3)] for i in range(rng.randint(0, 12))]
        d = Daf(lol=rows, cols=['id', 'c1', 'c2', 'c3'])
        loda = []
        for _ in range(rng.randint(0, 5)):
            cols = rng.sample(['c1', 'c2', 'c3', 'nope'], rng.randint(0, 3))
            loda.append({c: rng.choice(values) for c in cols})
        for inverse in (False, True):
            assert d.select_by_dict(loda, inverse=inverse).col('id') == _reference_ids(d, loda, inverse), (rows, loda, inverse)


def test_list_of_dicts_gives_the_same_rows_as_select_where_on_many_rows():
    rows = [[i, i % 50, f's{i % 100}'] for i in range(5000)]
    d = Daf(lol=rows, cols=['id', 'grp', 'name'], keyfield='id')
    names = {f's{i}' for i in range(0, 100, 7)}
    got = d.select_by_dict([{'name': n} for n in names])
    assert got.lol == d.select_where(lambda row: row['name'] in names).lol
    pairs = [{'grp': g, 'name': f's{g}'} for g in range(50)]
    assert d.select_by_dict(pairs).lol == [r for r in rows if r[1] < 50 and r[2] == f's{r[1]}']
