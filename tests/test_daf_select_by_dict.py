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
