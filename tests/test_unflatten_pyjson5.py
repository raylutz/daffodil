# test_unflatten_pyjson5.py
#
# unflatten_val() reads the Pyon text of a dict or list cell with pyjson5 when it is
# installed, and falls back to ast.literal_eval. These tests check that the result is
# the same with pyjson5 as without it, for any value that str() writes.

from typing import Any

import pytest
from daffodil.daf import Daf
from daffodil.lib import daf_utils
from daffodil.lib.daf_utils import unflatten_val


VALUES: list = [
    {'math': [[95, 0.1], [45, 0.1]], 'gym': [[94, 0.2]]},
    [1, 2, 3],
    [],
    {},
    (1, 2),
    (),
    [(1, 2), (3,)],
    {1: 'x', 2: 'y'},
    {'b': True, 'n': None, 'f': False},
    {'s': "it's", 't': 'say "hi"', 'u': 'a, b: c'},
    {'n': '\n\t\\'},
    ['A', 'é', '日本'],
    {'big': 123456789012345678901234567890},
    [1.5e+16, -0.0, 0.1, -3],
    {'k': [1, {'z': []}]},
    {1, 2},
    [float('inf')],
]


def _without_pyjson5(monkeypatch: Any, text: str) -> Any:
    with monkeypatch.context() as m:
        m.setattr(daf_utils, '_pyjson5_loads', None)
        return unflatten_val(text)


def test_pyjson5_is_installed_for_the_tests() -> None:
    assert daf_utils._pyjson5_loads is not None


@pytest.mark.parametrize('value', VALUES, ids=repr)
def test_same_result_with_and_without_pyjson5(monkeypatch: Any, value: Any) -> None:
    text = str(value)
    assert unflatten_val(text) == _without_pyjson5(monkeypatch, text)


@pytest.mark.parametrize('value', [v for v in VALUES if v not in ({1, 2}, [float('inf')])], ids=repr)
def test_reads_back_the_value(value: Any) -> None:
    assert unflatten_val(str(value)) == value


@pytest.mark.parametrize('text', ['abc', '', '[not pyon', '{broken: ', "'quoted'"], ids=repr)
def test_text_that_is_not_pyon_stays_text(monkeypatch: Any, text: str) -> None:
    assert unflatten_val(text) == _without_pyjson5(monkeypatch, text)


def test_json_text_is_read(monkeypatch: Any) -> None:
    text = '{"done": true, "note": null, "n": [1, 2]}'
    assert unflatten_val(text) == {'done': True, 'note': None, 'n': [1, 2]}
    assert _without_pyjson5(monkeypatch, text) == {'done': True, 'note': None, 'n': [1, 2]}


def test_text_only_pyjson5_reads(monkeypatch: Any) -> None:
    """ Unquoted keys are JSON5, not Pyon. pyjson5 reads them; without it the text stays. """
    assert unflatten_val('{a: 1}') == {'a': 1}
    assert _without_pyjson5(monkeypatch, '{a: 1}') == '{a: 1}'


def test_dict_column_round_trip_through_csv() -> None:
    daf = Daf(lol=[['r1', {'a': [1, 2], 'b': True}], ['r2', {'a': [], 'c': None}]], cols=['k', 'd'], keyfield='k')
    back = Daf.from_csv_buff(daf.to_csv_buff(), keyfield='k', dtypes={'k': str, 'd': dict})
    assert back.lol == daf.lol
