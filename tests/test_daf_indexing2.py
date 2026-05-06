# test_daf_indexing.py

import pytest
from daffodil.daf import Daf, KeysDisabledError


# -----------------------------
# Test Set 1: irow (positional row indexing)
# -----------------------------


@pytest.fixture
def daf():
    return Daf(lol=
        [
            ["r1", 10, 100],
            ["r2", 20, 200],
            ["r3", 30, 300],
            ["r4", 40, 400],
            ["r5", 50, 500],
        ],
        cols=["key", "A", "B"],
        keyfield="key"
    )


# --- Basic irow selection ---

def test_irow_single(daf):
    result = daf[2]
    assert result[:, "key"].to_list() == ["r3"]


def test_irow_slice(daf):
    result = daf[2:5]
    assert result[:, "key"].to_list() == ["r3", "r4", "r5"]


def test_irow_list(daf):
    result = daf[[2, 3, 4]]
    assert result[:, "key"].to_list() == ["r3", "r4", "r5"]


def test_irow_range(daf):
    result = daf[range(2, 5)]
    assert result[:, "key"].to_list() == ["r3", "r4", "r5"]


def test_irow_list_of_ranges(daf):
    result = daf[[range(1, 3), range(3, 5)]]
    assert result[:, "key"].to_list() == ["r2", "r3", "r4", "r5"]


# --- Order and duplication ---

def test_irow_order_preserved(daf):
    result = daf[[4, 2]]
    assert result[:, "key"].to_list() == ["r5", "r3"]


def test_irow_duplicates_preserved(daf):
    result = daf[[2, 2, 3]]
    assert result[:, "key"].to_list() == ["r3", "r3", "r4"]


# --- Empty selectors ---

def test_irow_empty_list(daf):
    result = daf[[]]
    assert result.num_rows() == 0


def test_irow_empty_range(daf):
    result = daf[range(0)]
    assert result.num_rows() == 0


def test_irow_empty_list_of_ranges(daf):
    result = daf[[]]
    assert result.num_rows() == 0


# --- None and invalid selectors ---

def test_irow_none_selector(daf):
    with pytest.raises(TypeError):
        _ = daf[None]


def test_irow_invalid_type(daf):
    with pytest.raises(TypeError):
        _ = daf[3.14]


def test_krow_none_key_present():
    d = Daf(lol=
        [
            [None, 10],
            ["r2", 20],
        ],
        cols=["key", "A"],
        keyfield="key"
    )
    with pytest.raises(TypeError): 
        d[None].to_list()


def test_krow_none_in_list_present():
    d = Daf(
        lol=[
            [None, 10],
            ["r2", 20],
        ],
        cols=["key", "A"],
        keyfield="key"
    )
    with pytest.raises(TypeError): 
        result = d[[None, "r2"]]

def test_krow_none_in_list_missing(daf):
    with pytest.raises(TypeError):
        _ = daf[[None, "r2"]]


def test_krow_none_without_keyfield():
    d = Daf(
        lol=[
            [None, 10],
            ["r2", 20],
        ],
        cols=["key", "A"],
        keyfield=None
    )
    # No keyfield → cannot interpret None as krow
    with pytest.raises(TypeError):
        _ = d[None]

# --- Out of range ---

def test_irow_out_of_range_single(daf):
    with pytest.raises(IndexError):
        _ = daf[100]


def test_irow_out_of_range_list(daf):
    with pytest.raises(IndexError):
        _ = daf[[1, 100]]


# --- Mixed selector types (should fail) ---

def test_irow_mixed_types_list(daf):
    with pytest.raises(TypeError):
        _ = daf[[1, "r2"]]


def test_irow_mixed_types_list_of_ranges(daf):
    with pytest.raises(TypeError):
        _ = daf[[range(1, 3), "r2"]]


# --- Slice edge cases ---

def test_irow_full_slice(daf):
    result = daf[:]
    assert result.num_rows() == 5


def test_irow_reverse_slice(daf):
    result = daf[::-1]
    assert result[:, "key"].to_list() == ["r5", "r4", "r3", "r2", "r1"]


def test_irow_slice_step(daf):
    result = daf[::2]
    assert result[:, "key"].to_list() == ["r1", "r3", "r5"]


# --- Basic krow selection ---

def test_krow_single(daf):
    result = daf["r2"]
    assert result[:, "A"].to_list() == [20]


def test_krow_list(daf):
    result = daf[["r2", "r5"]]
    assert result[:, "A"].to_list() == [20, 50]


def test_krow_tuple_range(daf):
    result = daf[("r2", "r5"), :]
    assert result[:, "key"].to_list() == ["r2", "r3", "r4", "r5"]


# --- Tuple range edge cases ---

def test_krow_tuple_single_start_to_end(daf):
    result = daf[("r3",), :]
    assert result[:, "key"].to_list() == ["r3", "r4", "r5"]


def test_krow_tuple_start_to_end_explicit(daf):
    result = daf[("r1", "r3"), :]
    assert result[:, "key"].to_list() == ["r1", "r2", "r3"]


def test_krow_tuple_invalid_order(daf):
    result = daf[("r5", "r2"), :]
    assert result.num_rows() == 0


# --- Order and duplication ---

def test_krow_order_preserved(daf):
    result = daf[["r5", "r2"]]
    assert result[:, "key"].to_list() == ["r5", "r2"]


def test_krow_duplicates_preserved(daf):
    result = daf[["r2", "r2"]]
    assert result[:, "key"].to_list() == ["r2", "r2"]


# --- Empty selectors ---

def test_krow_empty_list(daf):
    result = daf[[]]
    assert result.num_rows() == 0


def test_krow_empty_tuple(daf):
    result = daf[()]
    assert result.num_rows() == 0


# --- Missing keys ---

def test_krow_missing_single(daf):
    with pytest.raises(KeyError):
        _ = daf["r999"]


def test_krow_missing_in_list(daf):
    with pytest.raises(KeyError):
        _ = daf[["r2", "r999"]]


# --- None and invalid selectors ---

def test_krow_none_selector(daf):
    with pytest.raises(TypeError):
        _ = daf[None]


def test_krow_invalid_type_float(daf):
    with pytest.raises(TypeError):
        _ = daf[3.14]


# --- Mixed selector types (should fail) ---

def test_krow_mixed_types_list(daf):
    with pytest.raises(KeyError):
        _ = daf[["r2", 3]]


def test_krow_mixed_tuple_types(daf):
    with pytest.raises(IndexError):
        _ = daf[("r2", 3)]


# -----------------------------
# Test Set 3: Column indexing (icol and kcol)
# -----------------------------


# --- Basic icol selection (integer-based) ---

def test_icol_single(daf):
    result = daf[:, 1]
    assert result.to_list() == [10, 20, 30, 40, 50]


def test_icol_slice(daf):
    result = daf[:, 1:3]
    assert result.shape() == (5, 2)


def test_icol_list(daf):
    result = daf[:, [1, 2]]
    assert result.shape() == (5, 2)


def test_icol_range(daf):
    result = daf[:, range(1, 3)]
    assert result.shape() == (5, 2)


def test_icol_list_of_ranges(daf):
    result = daf[:, [range(1, 2), range(2, 3)]]
    assert result.shape() == (5, 2)


# --- Basic kcol selection (string-based) ---

def test_kcol_single(daf):
    result = daf[:, "A"]
    assert result.to_list() == [10, 20, 30, 40, 50]


def test_kcol_list(daf):
    result = daf[:, ["A", "B"]]
    assert result.shape() == (5, 2)


def test_kcol_tuple_range(daf):
    result = daf[:, ("A", "B")]
    assert result.shape() == (5, 2)


# --- Tuple range edge cases ---

def test_kcol_tuple_single_start_to_end(daf):
    result = daf[:, ("A",)]
    assert result.shape() == (5, 2)  # from A to end


def test_kcol_tuple_invalid_order(daf):
    result = daf[:, ("B", "A")]
    assert result.num_rows() == 0

# --- Order and duplication ---

def test_icol_order_preserved(daf):

    result = daf[:, [2, 1]]
    assert result.lol == [
        [100, 10],
        [200, 20],
        [300, 30],
        [400, 40],
        [500, 50],
    ]


def test_kcol_order_preserved(daf):
    result = daf[:, ["B", "A"]]
    assert result.lol == [
        [100, 10],
        [200, 20],
        [300, 30],
        [400, 40],
        [500, 50],
    ]


def test_icol_duplicates_preserved(daf):
    result = daf[:, [1, 1]]
    assert result.lol == [
        [10, 10],
        [20, 20],
        [30, 30],
        [40, 40],
        [50, 50],
    ]


def test_kcol_duplicates_preserved(daf):
    result = daf[:, ["A", "A"]]
    assert result.lol == [
        [10, 10],
        [20, 20],
        [30, 30],
        [40, 40],
        [50, 50],
    ]


# --- Empty selectors ---

def test_icol_empty_list(daf):

    result = daf[:, []]
    assert result.num_cols() == 0


def test_icol_empty_range(daf):
    result = daf[:, range(0)]
    assert result.num_cols() == 0


def test_kcol_empty_list(daf):
    result = daf[:, []]
    assert result.num_cols() == 0


# --- Missing columns ---

def test_kcol_missing_single(daf):
    with pytest.raises(KeyError):
        _ = daf[:, "Z"]


def test_kcol_missing_in_list(daf):
    with pytest.raises(KeyError):
        _ = daf[:, ["A", "Z"]]


# --- None and invalid selectors ---

def test_col_none_selector(daf):
    with pytest.raises(TypeError):
        _ = daf[:, None]


def test_col_invalid_type(daf):
    with pytest.raises(TypeError):
        _ = daf[:, 3.14]


# --- Mixed selector types (should fail) ---

def test_col_mixed_types_list(daf):
    with pytest.raises(TypeError):
        _ = daf[:, [1, "A"]]


def test_col_mixed_tuple_types(daf):
    with pytest.raises(KeyError):
        _ = daf[:, ("A", 1)]


# -----------------------------
# Test Set 4: 2D indexing (row, col combinations)
# -----------------------------


# --- Basic cell selection ---

def test_2d_single_cell(daf):
    val = daf[2, 1].to_value()
    assert val == 30


def test_2d_single_cell_krow_kcol(daf):
    val = daf["r3", "A"].to_value()
    assert val == 30


# --- Row subset + column subset ---

def test_2d_irow_icol(daf):
    result = daf[[2, 3], [1, 2]]
    assert result.shape() == (2, 2)
    assert result.lol == [
        [30, 300],
        [40, 400],
    ]


def test_2d_krow_kcol(daf):
    result = daf[["r2", "r5"], ["A", "B"]]
    assert result.shape() == (2, 2)
    assert result.lol == [
        [20, 200],
        [50, 500],
    ]


def test_2d_irow_kcol(daf):
    result = daf[[1, 4], ["A", "B"]]
    assert result.lol == [
        [20, 200],
        [50, 500],
    ]


def test_2d_krow_icol(daf):
    result = daf[["r2", "r5"], [1, 2]]
    assert result.lol == [
        [20, 200],
        [50, 500],
    ]


# --- Slice combinations ---

def test_2d_slice_rows_slice_cols(daf):
    result = daf[1:4, 1:3]
    assert result.lol == [
        [20, 200],
        [30, 300],
        [40, 400],
    ]


def test_2d_slice_rows_kcol(daf):
    result = daf[1:4, "A"]
    assert result.to_list() == [20, 30, 40]


def test_2d_krow_slice_cols(daf):
    result = daf[["r2", "r3"], 1:3]
    assert result.lol == [
        [20, 200],
        [30, 300],
    ]


# --- Tuple range usage ---

def test_2d_krow_range_kcol_range(daf):
    result = daf[("r2", "r4"), ("A", "B")]
    assert result.lol == [
        [20, 200],
        [30, 300],
        [40, 400],
    ]


# --- Order preservation ---

def test_2d_order_preserved_rows(daf):
    result = daf[["r5", "r2"], ["A", "B"]]
    assert result.lol == [
        [50, 500],
        [20, 200],
    ]


def test_2d_order_preserved_cols(daf):
    result = daf[["r2", "r3"], ["B", "A"]]
    assert result.lol == [
        [200, 20],
        [300, 30],
    ]


# --- Duplicate preservation ---

def test_2d_duplicate_rows(daf):
    result = daf[["r2", "r2"], ["A", "B"]]
    assert result.lol == [
        [20, 200],
        [20, 200],
    ]


def test_2d_duplicate_cols(daf):
    result = daf[["r2", "r3"], ["A", "A"]]
    assert result.lol == [
        [20, 20],
        [30, 30],
    ]


# --- Empty selectors ---

def test_2d_empty_rows(daf):
    result = daf[[], ["A", "B"]]
    assert result.num_rows() == 0


def test_2d_empty_cols(daf):
    result = daf[["r2", "r3"], []]
    assert result.num_cols() == 0


def test_2d_empty_both(daf):
    result = daf[[], []]
    assert result.shape() == (0, 0)


# --- None and invalid selectors ---

def test_2d_none_row(daf):
    with pytest.raises(TypeError):
        _ = daf[None, ["A"]]


def test_2d_none_col(daf):
    with pytest.raises(TypeError):
        _ = daf[["r2"], None]


def test_2d_invalid_types(daf):
    with pytest.raises(TypeError):
        _ = daf[3.14, ["A"]]


# --- Mixed selector types (should fail) ---

def test_2d_mixed_row_types(daf):
    with pytest.raises(TypeError):
        _ = daf[[1, "r2"], ["A"]]


def test_2d_mixed_col_types(daf):
    with pytest.raises(TypeError):
        _ = daf[["r2"], [1, "A"]]


# -----------------------------
# Test Set 5: Extraction helpers and keyfield propagation
# -----------------------------


# --- to_value() ---

def test_to_value_cell_2d(daf):
    result = daf[2, 1].to_value()
    assert result == 30

def test_to_value_cell_krow_kcol(daf):
    assert daf["r3", "A"].to_value() == 30


def test_to_value_larger_array(daf):
    
    with pytest.raises(ValueError):    
        result = daf.to_value()


# --- to_list() for rows ---

def test_to_list_row_irow(daf):
    assert daf[2].to_list() == ["r3", 30, 300]


def test_to_list_row_krow(daf):
    assert daf["r2"].to_list() == ["r2", 20, 200]


def test_to_list_multiple_rows(daf):
    with pytest.raises(ValueError):    
        result = daf[["r2", "r3"]].to_list()


# --- to_list() for columns ---

def test_to_list_col_kcol(daf):
    assert daf[:, "A"].to_list() == [10, 20, 30, 40, 50]


def test_to_list_col_icol(daf):
    assert daf[:, 1].to_list() == [10, 20, 30, 40, 50]


def test_to_list_multiple_cols(daf):
    with pytest.raises(ValueError):    
        result = daf[:, ["A", "B"]].to_list()


# --- to_dict() ---

def test_to_dict_single_row(daf):
    d = daf["r2"].to_dict()
    assert d == {"key": "r2", "A": 20, "B": 200}


def test_to_dict_multiple_rows(daf):
    with pytest.raises(ValueError):    
        d = daf[["r2", "r3"]].to_dict()


# --- to_klist() ---

# def test_to_klist_basic(daf):
#     klist = daf[0].to_klist()
#     assert klist.values == ["r1", 10, 100]


# def test_to_klist_empty(daf):
#     klist = daf[[]].to_klist()
#     assert klist.values() == []


# --- keyfield propagation ---

def test_keyfield_preserved_when_present(daf):
    result = daf[:, ["key", "A"]]
    assert result.keyfield == "key"


def test_keyfield_removed_when_missing(daf):
    result = daf[:, ["A", "B"]]
    assert result.keyfield == ''


# --- extraction with empty selections ---

def test_to_list_empty_rows(daf):
    assert daf[[]].to_list() == []


def test_to_list_empty_cols(daf):

    result = daf[:, []].lol
    assert len(result) == 0


# --- invalid extraction usage ---

def test_to_dict_empty(daf):
    result = daf[[]].to_dict()
    assert result == {}


def test_to_value_empty(daf):
    with pytest.raises(ValueError):
        _ = daf[[]].to_value()


# -----------------------------
# Test Set 6: Selector normalization and edge-case semantics
# -----------------------------


# --- Full selection / default slice behavior ---

def test_full_row_slice(daf):
    result = daf[:]
    assert result.shape() == (5, 3)


def test_full_2d_slice(daf):
    result = daf[:, :]
    assert result.shape() == (5, 3)


# --- Negative indexing ---

def test_negative_irow_single(daf):
    result = daf[-1]
    assert result[:, "key"].to_list() == ["r5"]


def test_negative_irow_list(daf):
    result = daf[[-1, -2]]
    assert result[:, "key"].to_list() == ["r5", "r4"]


def test_negative_icol_single(daf):
    result = daf[:, -1]
    assert result.to_list() == [100, 200, 300, 400, 500]


def test_negative_icol_list(daf):
    result = daf[:, [-1, -2]]
    assert result.lol == [
        [100, 10],
        [200, 20],
        [300, 30],
        [400, 40],
        [500, 50],
    ]


# --- Slice edge behavior ---

def test_slice_start_none(daf):
    result = daf[:3]
    assert result[:, "key"].to_list() == ["r1", "r2", "r3"]


def test_slice_end_none(daf):
    result = daf[3:]
    assert result[:, "key"].to_list() == ["r4", "r5"]


def test_slice_negative_bounds(daf):
    result = daf[-3:-1]
    assert result[:, "key"].to_list() == ["r3", "r4"]


def test_slice_step_zero_invalid(daf):
    with pytest.raises(ValueError):
        _ = daf[::0]


# --- Range normalization ---

def test_range_basic(daf):
    result = daf[range(1, 4)]
    assert result[:, "key"].to_list() == ["r2", "r3", "r4"]


def test_range_with_step(daf):
    result = daf[range(0, 5, 2)]
    assert result[:, "key"].to_list() == ["r1", "r3", "r5"]


def test_empty_range(daf):
    result = daf[range(2, 2)]
    assert result.num_rows() == 0


# --- List of ranges normalization ---

def test_list_of_ranges_flatten(daf):
    result = daf[[range(0, 2), range(2, 4)]]
    assert result[:, "key"].to_list() == ["r1", "r2", "r3", "r4"]


def test_list_of_ranges_with_overlap(daf):
    result = daf[[range(0, 3), range(2, 5)]]
    assert result[:, "key"].to_list() == ["r1", "r2", "r3", "r3", "r4", "r5"]


def test_list_of_ranges_empty_components(daf):
    result = daf[[range(1, 1), range(2, 3)]]
    assert result[:, "key"].to_list() == ["r3"]


# --- Tuple normalization edge cases ---

def test_tuple_single_element(daf):
    result = daf[("r3",)]
    assert result[:, "key"].to_list() == ["r3", "r4", "r5"]


def test_tuple_empty_invalid(daf):
    result = daf[()]
    result.num_rows() == 0


def test_tuple_non_string_invalid(daf):
    with pytest.raises(IndexError):
        result = daf[(1, 3)]


# --- Selector dimensionality errors ---

def test_3d_selector_invalid(daf):
    with pytest.raises(TypeError):
        result = daf[1, 2, 3]


def test_nested_list_invalid(daf):
    with pytest.raises(TypeError):
        _ = daf[[[1, 2], [3, 4]]]


# --- Consistency checks across equivalent selectors ---

def test_equivalent_slice_and_range(daf):
    result_slice = daf[1:4]
    result_range = daf[range(1, 4)]
    assert result_slice.lol == result_range.lol


def test_equivalent_list_and_range(daf):
    result_list = daf[[1, 2, 3]]
    result_range = daf[range(1, 4)]
    assert result_list.lol == result_range.lol


# -----------------------------
# Test Set 7: Invariants and cross-behavior consistency
# -----------------------------


# --- Idempotence (selecting twice should be stable) ---

def test_idempotent_row_selection(daf):
    result1 = daf[["r2", "r3"]]
    result2 = result1[:]
    assert result1.lol == result2.lol


def test_idempotent_col_selection(daf):
    result1 = daf[:, ["A", "B"]]
    result2 = result1[:, :]
    assert result1.lol == result2.lol


# --- Axis-aware composability (valid invariants) ---

def test_row_then_col_equals_2d(daf):
    result1 = daf[["r2", "r3"]][:, ["A", "B"]]
    result2 = daf[["r2", "r3"], ["A", "B"]]
    assert result1.lol == result2.lol


# def test_col_then_row_equals_2d(daf):
#     result1 = daf[:, ["A", "B"]][["r2", "r3"]]
#     result2 = daf[["r2", "r3"], ["A", "B"]]
#     assert result1.to_list() == result2.to_list()


# --- Explicit rejection of invalid chained semantics ---

def test_invalid_chained_row_selection_not_equal(daf):
    result1 = daf[["r2", "r3"]]["r2"]   # second is krow again
    result2 = daf[["r2", "r3"], "A"]    # different meaning entirely
    assert result1 != result2


# --- Shape invariants ---

def test_shape_matches_selection(daf):
    result = daf[["r2", "r3"], ["A", "B"]]
    assert result.shape() == (2, 2)


def test_shape_after_empty_row(daf):
    result = daf[[], ["A", "B"]]
    assert result.shape() == (0, 0)


def test_shape_after_empty_col(daf):
    result = daf[["r2", "r3"], []]
    assert result.shape() == (0, 0)


# --- Keyfield invariants ---

def test_keyfield_survives_row_filter(daf):
    result = daf[["r2", "r3"]]
    assert result.keyfield == daf.keyfield


def test_keyfield_survives_col_subset_if_present(daf):
    result = daf[:, ["key", "A"]]
    assert result.keyfield == "key"


def test_keyfield_removed_if_not_present(daf):
    result = daf[:, ["A", "B"]]
    assert result.keyfield == ''


# --- Data integrity (no unintended mutation) ---

def test_original_unchanged_after_selection(daf):
    original = daf.lol.copy()
    _ = daf[["r2", "r3"]]
    assert daf.lol == original


def test_selection_is_view(daf):
    result = daf[["r2", "r3"]]
    result[0, 1] = 999
    assert daf["r2", "A"].to_value() == 999


# --- Equality of equivalent selectors ---

def test_equivalent_krow_and_tuple_range_single(daf):
    result1 = daf["r3"]
    with pytest.raises(KeyError):    
        result2 = daf[("r3", "r3")]
    # assert result1.to_list() == result2.to_list()


def test_equivalent_single_list_and_scalar(daf):
    result1 = daf["r3"]
    result2 = daf[["r3"]]
    assert result1.to_list() == result2.to_list()


# --- Column equivalence ---

def test_equivalent_kcol_and_tuple_range_single(daf):
    result1 = daf[:, "A"]
    # with pytest.raises(KeyError):    
    result2 = daf[:, ("A", "A")]
    assert result1 == result2


# --- Empty propagation consistency ---

def test_empty_then_slice(daf):
    result = daf[[]][:, :]
    assert result.shape() == (0, 0)


def test_empty_then_col_select(daf):
    result = daf[[]][:, ["A"]]
    assert result.shape() == (0, 0)


def test_empty_col_then_row_select(daf):
    with pytest.raises(KeysDisabledError):    
        result = daf[:, []][["r2", "r3"]]
