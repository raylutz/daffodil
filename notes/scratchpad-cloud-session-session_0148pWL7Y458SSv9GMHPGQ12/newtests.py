NEW = {
'tests/test_daf_coverage_a.py': {
'test_flatten_use_pyon_false_json_encodes': '''def test_flatten_use_pyon_false_json_encodes():
    daf = Daf(lol=[[{'x': 1}, True]], cols=['a', 'b'], dtypes={'a': dict, 'b': bool})
    daf.flatten(use_pyon=False)
    assert daf.lol == [['{"x": 1}', 1]]
''',
'test_record_append_mapping_same_order': '''def test_record_append_mapping_same_order():
    daf = Daf(lol=[[1, 2]], cols=['a', 'b'])
    daf.record_append(MappingProxyType({'a': 10, 'b': 20}))
    assert daf.lol == [[1, 2], [10, 20]]
''',
'test_set_irows_icols_short_column_list_partial': '''def test_set_irows_icols_column_list_wrong_length_raises():
    daf = _daf3()
    with pytest.raises(ValueError, match='1 values given for 3 rows'):
        daf.set_irows_icols([0, 1, 2], 1, [100])
    assert daf.lol == _daf3().lol
''',
'test_set_irows_icols_short_row_list_partial': '''def test_set_irows_icols_row_list_wrong_length_raises():
    daf = _daf3()
    with pytest.raises(ValueError, match='1 values given for 2 columns'):
        daf.set_irows_icols([0, 1], [0, 1], [100])
    assert daf.lol == _daf3().lol
''',
'test_set_irows_icols_single_row_from_daf': '''def test_set_irows_icols_single_row_from_daf():
    daf = _daf3()
    daf.set_irows_icols(0, None, Daf(lol=[[10, 20, 30]], cols=['a', 'b', 'c']))
    assert daf.lol[0] == [10, 20, 30]
    assert isinstance(daf.lol[0], list)
''',
'test_set_irows_icols_multi_row_from_daf': '''def test_set_irows_icols_multi_row_from_daf():
    daf = _daf3()
    daf.set_irows_icols([0, 1], None, Daf(lol=[[10, 20, 30], [40, 50, 60]], cols=['a', 'b', 'c']))
    assert daf.lol == [[10, 20, 30], [40, 50, 60], [7, 8, 9]]


def test_set_irows_icols_daf_value_wrong_shape_raises():
    daf = _daf3()
    with pytest.raises(ValueError, match='shape'):
        daf.set_irows_icols([0, 1], None, Daf(lol=[[10, 20]], cols=['a', 'b']))
    assert daf.lol == _daf3().lol


def test_set_irows_icols_single_cell_keeps_daf_object():
    # a single cell may hold a Daf as-is.
    daf = _daf3()
    inner = Daf(lol=[[10, 20]], cols=['a', 'b'])
    daf.set_irows_icols(0, 0, inner)
    assert daf.lol[0][0] is inner
''',
'test_set_irows_icols_single_col_from_daf': '''def test_set_irows_icols_single_col_from_daf():
    daf = _daf3()
    daf.set_irows_icols([0, 1], 1, Daf(lol=[[100], [200]], cols=['x']))
    assert daf.lol == [[1, 100, 3], [4, 200, 6], [7, 8, 9]]
''',
'test_setitem_block_from_daf': '''def test_setitem_block_from_daf():
    daf = _daf3()
    daf[0:2, 0:2] = Daf(lol=[[100, 200], [300, 400]], cols=['a', 'b'])
    assert daf.lol == [[100, 200, 3], [300, 400, 6], [7, 8, 9]]


def test_setitem_block_from_larger_daf_raises():
    daf = _daf3()
    with pytest.raises(ValueError, match='shape'):
        daf[0:2, 0:2] = _daf3()
''',
'test_select_icols_slice_uneven_rows_raises_indexerror': '''def test_select_icols_slice_uneven_rows_raises_indexerror():
    daf = Daf(lol=[[1, 2, 3], [4]], cols=['a', 'b', 'c'])
    with pytest.raises(IndexError):
        daf.select_icols(slice(0, 3))
''',
},
'tests/test_daf_coverage_b.py': {
'test_sort_by_colname_unknown_column_raises_keyerror': '''def test_sort_by_colname_unknown_column_raises_keyerror():
    daf = Daf(cols=['a', 'b'], lol=[[2, 1], [1, 2]])
    with pytest.raises(KeyError, match='zz'):
        daf.sort_by_colname('zz')
''',
'test_sort_by_colnames_unknown_column_raises_keyerror': '''def test_sort_by_colnames_unknown_column_raises_keyerror():
    daf = Daf(cols=['a', 'b'], lol=[[2, 1], [1, 2]])
    with pytest.raises(KeyError, match='zz'):
        daf.sort_by_colnames(['a', 'zz'])
''',
'test_apply_formulas_shape_mismatch_raises': '''def test_apply_formulas_shape_mismatch_raises():
    daf = Daf(cols=['a', 'b'], lol=[[1, 2], [3, 4]])
    with pytest.raises(RuntimeError, match='same shape'):
        daf.apply_formulas(Daf(cols=['a'], lol=[['']]))
''',
'test_apply_formulas_formula_error_reraises': '''def test_apply_formulas_formula_error_reraises(capsys):
    daf = Daf(cols=['a', 'b'], lol=[[1, 2], [3, 4]])
    formulas = Daf(cols=['a', 'b'], lol=[['1/0', ''], ['', '']])
    with pytest.raises(ZeroDivisionError):
        daf.apply_formulas(formulas)
    assert "Error in formula for cell [0,0]" in capsys.readouterr().out
''',
'test_apply_in_place_unknown_by_raises': '''def test_apply_in_place_unknown_by_raises():
    daf = Daf(cols=['k', 'v'], lol=[['a', 1]])
    with pytest.raises(NotImplementedError):
        daf.apply_in_place(lambda row: row, by='bogus')
''',
'test_manifest_reduce_sums_all_chunks': '''def test_manifest_reduce_sums_all_chunks():
    manifest = Daf(cols=['name'], lol=[['c1'], ['c2']])
    store = _chunk_store()
    result = manifest.manifest_reduce(Daf.sum_da, load_func=lambda cs: store[cs['name']])
    assert result == {'x': 14, 'y': 26}
''',
'test_reduce_row_func_exception_hits_breakpoint': '''def test_reduce_row_func_exception_propagates():
    daf = Daf(cols=['a', 'b'], lol=[[1, 2], [3, 4]])
    with pytest.raises(ValueError, match='boom'):
        daf.reduce(_failing_reduction)
''',
'test_reduce_sparse_row_func_exception_hits_breakpoint': '''def test_reduce_sparse_row_func_exception_propagates():
    daf = Daf(cols=['id', 'j'], lol=[['x', '{"p": 1}'], ['y', '{"q": 2}']])
    with pytest.raises(ValueError, match='boom'):
        daf.reduce(_failing_reduction, by='sparse_row', indirect_col='j')
''',
'test_sum_da_sparse_unexpected_exception_hits_breakpoint': '''def test_sum_da_sparse_unexpected_exception_propagates():
    with pytest.raises(KeyError):
        Daf.sum_da({'a': _AddRaisesKeyError()}, {'a': 0})
''',
'test_sum_da_astype_uninitialized_accumulator_hits_breakpoint': '''def test_sum_da_astype_uninitialized_accumulator_raises():
    # with astype, reduction_da must be initialized for all cols.
    with pytest.raises(KeyError):
        Daf.sum_da({'a': 1}, {}, cols=['a'], astype=int)
''',
'test_sum_da_cols_unexpected_exception_hits_breakpoint': '''def test_sum_da_cols_unexpected_exception_propagates():
    with pytest.raises(KeyError):
        Daf.sum_da({'a': _AddRaisesKeyError()}, {'a': 0}, cols=['a'])
''',
'test_alter_daf_per_setting_missing_setting_hits_breakpoint': '''def test_alter_daf_per_setting_missing_setting_raises():
    daf = Daf(cols=['x'], lol=[['a']])
    with pytest.raises(KeyError, match="'spec' not found"):
        daf.alter_daf_per_setting({}, 'spec', {})
''',
'test_count_values_da_list_value_does_not_mutate_source_row': '''def test_count_values_da_list_value_does_not_mutate_source_row():
    row1 = {'l': [1, 2]}
    acc = Daf.count_values_da(row1, {}, ['l'])
    Daf.count_values_da({'l': [3]}, acc, ['l'])
    assert row1 == {'l': [1, 2]}


def test_count_values_da_dict_value_does_not_mutate_source_row():
    row1 = {'d': {'x': 1}}
    acc = Daf.count_values_da(row1, {}, ['d'])
    Daf.count_values_da({'d': {'x': 2, 'y': 1}}, acc, ['d'])
    assert row1 == {'d': {'x': 1}}
    assert acc == {'d': {'x': 3, 'y': 1}}
''',
},
'tests/test_daf_sql.py': {
'test_create_index_at_cursor_failure_returns_false': '''@pytest.mark.parametrize("cursor_factory, exc_type", [
    (lambda: _mem_table().cursor(), sqlite3.OperationalError),          # no such column
    (lambda: mock.MagicMock(**{'execute.side_effect': RuntimeError('boom')}), RuntimeError),
])
def test_create_index_at_cursor_failure_raises(cursor_factory, exc_type):
    with pytest.raises(exc_type):
        daf_sql.create_index_at_cursor(cursor_factory(), 'missing_col', 't')
''',
},
'tests/test_daf_utils_coverage.py': {
'test_json_encode_unsupported_type_raises_typeerror': '''def test_json_encode_unsupported_type_raises_typeerror():
    with pytest.raises(TypeError):
        utils.json_encode({'a': object()})
''',
'test_json_encode_nan_falls_back_to_nan_literal': '''def test_json_encode_nan_raises_valueerror():
    with pytest.raises(ValueError):
        utils.json_encode([float('nan'), 1])
''',
'test_test_strbool_unsupported_type_returns_false': '''def test_test_strbool_unsupported_type_raises():
    with pytest.raises(TypeError, match='object'):
        utils.test_strbool(object())
''',
'test_convert_type_value_unsupported_type_no_unbound_local': '''def test_convert_type_value_unsupported_type_raises():
    with pytest.raises(TypeError, match='cannot convert'):
        utils.convert_type_value('x', set)
''',
'test_write_buff_to_fp_local_binary_open_failure_hits_tripwire': '''def test_write_buff_to_fp_local_binary_open_failure_raises(tmp_path):
    fp = str(tmp_path / 'no_such_dir' / 'out.bin')
    with pytest.raises(FileNotFoundError):
        utils.write_buff_to_fp(b'\\x01', fp, rtype='binary')
''',
'test_len_slice_bad_bounds_no_unbound_local': '''def test_len_slice_bad_bounds_raises():
    with pytest.raises(TypeError):
        utils.len_slice(slice('a', 'b'), 5)
''',
'test_slice_to_range_non_int_start_hits_tripwire_returns_none': '''def test_slice_to_range_non_int_start_raises():
    with pytest.raises(TypeError):
        utils.slice_to_range(slice('a', 5), 10)
''',
'test_pandas_dtype_dict_to_python_unknown_dtype_hits_tripwire': '''def test_pandas_dtype_dict_to_python_unknown_dtype_raises():
    import pandas as pd
    with pytest.raises(TypeError, match="Unknown Pandas dtype for column 'cat'"):
        daf_pandas.pandas_dtype_dict_to_python({'cat': pd.CategoricalDtype(['x']), 'f': np.dtype('float64')})
''',
},
}
import ast
for path, repl in NEW.items():
    src = open(path).read(); lines = src.splitlines(keepends=True)
    spans = []
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in repl:
            start = (node.decorator_list[0].lineno if node.decorator_list else node.lineno)
            spans.append((start, node.end_lineno, repl[node.name]))
    assert len(spans) == len(repl), (path, set(repl) - {s[2] for s in spans})
    for start, end, text in sorted(spans, reverse=True):
        lines[start-1:end] = [text]
    open(path, 'w').write(''.join(lines))
print("ok")
