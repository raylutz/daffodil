# test_json_encode_numpy.py
#
# NpEncoder converts numpy values for json_encode(). It imports numpy only for a value
# whose type comes from numpy. These tests check the results with numpy, and that a value
# JSON cannot encode raises TypeError, with or without numpy.

import sys
from typing import Any

import numpy as np
import pytest
from daffodil.lib.daf_utils import json_encode


def test_numpy_values_are_converted() -> None:
    assert json_encode({'i': np.int64(3), 'f': np.float32(1.5), 'a': np.array([1, 2])}) == '{"i": 3, "f": 1.5, "a": [1, 2]}'


def test_plain_values_unchanged() -> None:
    assert json_encode({'a': [1, 'x', None, True]}) == '{"a": [1, "x", null, true]}'
    assert json_encode(None) == ''


@pytest.mark.parametrize('value', [{1, 2}, object(), np.bool_(True)], ids=['set', 'object', 'numpy bool'])
def test_value_json_cannot_encode_raises_type_error(value: Any) -> None:
    with pytest.raises(TypeError, match='is not JSON serializable'):
        json_encode(value)


@pytest.mark.parametrize('value', [{1, 2}, object()], ids=['set', 'object'])
def test_without_numpy_raises_type_error(monkeypatch: Any, value: Any) -> None:
    monkeypatch.setitem(sys.modules, 'numpy', None)
    with pytest.raises(TypeError, match='is not JSON serializable'):
        json_encode(value)
    assert json_encode({'a': 1}) == '{"a": 1}'
