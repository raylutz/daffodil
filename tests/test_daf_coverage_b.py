# test_daf_coverage_b.py
#
# Coverage-driven tests for daf.py, second half of the file (methods from iloc() through
# unpack_indirect()). Focuses on error paths, alternate branches (composite keyfields,
# indirect columns, diagnose=True logging) and rarely used helpers.

import pytest

import daffodil.daf as daf_module
from daffodil.daf import Daf, KeysDisabledError


# =====================================================================
# iloc()
# =====================================================================

def test_iloc_unrecognized_rtype_raises():
    daf = Daf(cols=['a', 'b'], lol=[[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="unrecognized rtype 'bogus'"):
        daf.iloc(0, rtype='bogus')


# =====================================================================
# col_to_la() with indirect_col
# =====================================================================

def _indirect_daf():
    return Daf(cols=['id', 'j'], lol=[['x', '{"p": 1}'], ['y', '{"q": 2}'], ['z', '{"p": 1}']])


def test_col_to_la_indirect_omit_nulls():
    assert _indirect_daf().col_to_la('p', indirect_col='j', omit_nulls=True) == [1, 1]


def test_col_to_la_indirect_uses_default_for_missing():
    assert _indirect_daf().col_to_la('p', indirect_col='j', default='D') == [1, 'D', 1]


def test_col_to_la_indirect_explicit_null_replaced_by_default():
    daf = Daf(cols=['id', 'j'], lol=[['x', '{"p": ""}'], ['y', '{"p": 5}']])
    assert daf.col_to_la('p', indirect_col='j', default='D') == ['D', 5]


def test_col_to_la_indirect_unique():
    assert _indirect_daf().col_to_la('p', indirect_col='j', unique=True) == [1, '']


# =====================================================================
# assign_record() / assign_col()
# =====================================================================

def test_assign_record_without_keyfield_raises():
    daf = Daf(cols=['k', 'v'], lol=[['a', 1]])
    with pytest.raises(KeysDisabledError):
        daf.assign_record({'k': 'a', 'v': 2})


def test_assign_col_on_keyfield_rebuilds_keys():
    daf = Daf(cols=['k', 'v'], lol=[['a', 1], ['b', 2]], keyfield='k')
    daf.assign_col('k', ['c', 'd'])
    assert daf.keys() == ['c', 'd']
    assert daf.select_record('c') == {'k': 'c', 'v': 1}


# =====================================================================
# replace_in_columns() with a composite keyfield
# =====================================================================

def test_replace_in_columns_composite_keyfield_invalidates_kd():
    daf = Daf(cols=['a', 'b', 'c'], lol=[[1, 2, 3], [4, 5, 6]], keyfield=['a', 'b'])
    daf.replace_in_columns(['a'], find_values=[1], replacement=7)
    assert daf.lol == [[7, 2, 3], [4, 5, 6]]
    assert daf.keys() == [(7, 2), (4, 5)]


# =====================================================================
# sort_by_colname() / sort_by_colnames() with an unknown column
# =====================================================================

def test_sort_by_colname_unknown_column_raises_keyerror():
    daf = Daf(cols=['a', 'b'], lol=[[2, 1], [1, 2]])
    with pytest.raises(KeyError, match='zz'):
        daf.sort_by_colname('zz')


def test_sort_by_colnames_unknown_column_raises_keyerror():
    daf = Daf(cols=['a', 'b'], lol=[[2, 1], [1, 2]])
    with pytest.raises(KeyError, match='zz'):
        daf.sort_by_colnames(['a', 'zz'])


def test_sort_by_colnames_single_row_is_noop():
    daf = Daf(cols=['a'], lol=[[1]])
    assert daf.sort_by_colnames(['zz']) is daf      # short-circuits before column lookup
    assert daf.lol == [[1]]


# =====================================================================
# apply_formulas()
# =====================================================================

def test_apply_formulas_empty_daf_returns_none():
    assert Daf().apply_formulas(Daf()) is None


def test_apply_formulas_shape_mismatch_raises():
    daf = Daf(cols=['a', 'b'], lol=[[1, 2], [3, 4]])
    with pytest.raises(RuntimeError, match='same shape'):
        daf.apply_formulas(Daf(cols=['a'], lol=[['']]))


def test_apply_formulas_formula_error_reraises(capsys):
    daf = Daf(cols=['a', 'b'], lol=[[1, 2], [3, 4]])
    formulas = Daf(cols=['a', 'b'], lol=[['1/0', ''], ['', '']])
    with pytest.raises(ZeroDivisionError):
        daf.apply_formulas(formulas)
    assert "Error in formula for cell [0,0]" in capsys.readouterr().out


# =====================================================================
# insert_dif_rows()
# =====================================================================

def test_insert_dif_rows_all_rows():
    daf = Daf(cols=['a', 'b'], lol=[[1, 10], [3, 30], [6, 60]])
    daf.insert_dif_rows()
    assert daf.lol == [[1, 10], [-2, -20], [3, 30], [-3, -30], [6, 60]]


def test_insert_dif_rows_selected_rows():
    daf = Daf(cols=['a', 'b'], lol=[[1, 10], [3, 30], [6, 60]])
    daf.insert_dif_rows([0])
    assert daf.lol == [[1, 10], [-2, -20], [3, 30], [6, 60]]


def test_insert_dif_rows_with_offset_and_cols():
    daf = Daf(cols=['a', 'b'], lol=[[1, 10], [3, 30], [6, 60]])
    daf.insert_dif_rows([1], cols=['a'], offset=1)
    # diff of rows 1 and 2 for col 'a' inserted after row 2; missing col 'b' filled with ''
    assert daf.lol == [[1, 10], [3, 30], [6, 60], [-3, '']]


# =====================================================================
# apply_in_place()
# =====================================================================

def test_apply_in_place_row_func_returning_none_raises():
    daf = Daf(cols=['k', 'v'], lol=[['a', 1]], keyfield='k')
    with pytest.raises(ValueError, match="must return a row"):
        daf.apply_in_place(lambda row: None)


def test_apply_in_place_row_klist_with_rowkeys():
    daf = Daf(cols=['k', 'v'], lol=[['a', 1], ['b', 2], ['c', 3]], keyfield='k')

    def times10(kl):
        kl['v'] = kl['v'] * 10

    daf.apply_in_place(times10, by='row_klist', rowkeys=['b'])
    assert daf.lol == [['a', 1], ['b', 20], ['c', 3]]


def test_apply_in_place_unknown_by_raises():
    daf = Daf(cols=['k', 'v'], lol=[['a', 1]])
    with pytest.raises(NotImplementedError):
        daf.apply_in_place(lambda row: row, by='bogus')


# =====================================================================
# manifest_apply() / manifest_reduce() / manifest_process()
# =====================================================================

def _chunk_store():
    return {
        'c1': Daf(cols=['x', 'y'], lol=[[1, 2], [3, 4]]),
        'c2': Daf(cols=['x', 'y'], lol=[[10, 20]]),
    }


def test_manifest_apply_table():
    manifest = Daf(cols=['name'], lol=[['c1'], ['c2']])
    store = _chunk_store()
    saved = {}

    def func(daf, cols):
        new_daf = Daf(cols=['x'], lol=[[sum(daf.col('x'))]])
        return {'name': f"sum_{daf.lol[0][0]}", 'rows': len(daf)}, new_daf

    result = manifest.manifest_apply(
        func,
        load_func=lambda cs: store[cs['name']],
        save_func=lambda cs, d: saved.__setitem__(cs['name'], d),
        by='table',
    )
    assert result.columns() == ['name', 'rows']
    assert result.lol == [['sum_1', 2], ['sum_10', 1]]
    assert saved['sum_1'].lol == [[4]]
    assert saved['sum_10'].lol == [[10]]


def test_manifest_reduce_requires_load_func():
    manifest = Daf(cols=['name'], lol=[['c1']])
    with pytest.raises(ValueError, match='load_func is required'):
        manifest.manifest_reduce(Daf.sum_da)


def test_manifest_reduce_sums_all_chunks():
    manifest = Daf(cols=['name'], lol=[['c1'], ['c2']])
    store = _chunk_store()
    result = manifest.manifest_reduce(Daf.sum_da, load_func=lambda cs: store[cs['name']])
    assert result == {'x': 14, 'y': 26}


def test_manifest_process():
    manifest = Daf(cols=['name'], lol=[['c1'], ['c2']])
    store = _chunk_store()
    result = manifest.manifest_process(
        lambda cs, mult=1: {'name': cs['name'], 'n': len(store[cs['name']]) * mult}, mult=2)
    assert result.columns() == ['name', 'n']
    assert result.lol == [['c1', 4], ['c2', 2]]


# =====================================================================
# groupby() with single-element colnames
# =====================================================================

def test_groupby_single_colnames_list():
    daf = Daf(cols=['g', 'v'], lol=[['x', 1], ['y', 2], ['x', 3]])
    result = daf.groupby(colnames=['g'])
    assert list(result.keys()) == ['x', 'y']
    assert result['x'].lol == [['x', 1], ['x', 3]]
    assert result['y'].lol == [['y', 2]]


# =====================================================================
# groupby_reduce() / reduce_dodaf_to_daf() / multi_groupby_reduce() with diagnose
# =====================================================================

def _ghv_daf():
    return Daf(cols=['g', 'h', 'v'], lol=[['x', 'p', 1], ['y', 'p', 2], ['x', 'q', 3]])


def test_groupby_reduce_diagnose(capsys):
    result = _ghv_daf().groupby_reduce('g', Daf.sum_da, reduce_cols=['v'], diagnose=True)
    assert result.keyfield == 'g'
    assert result.lol == [['x', '', 4], ['y', '', 2]]
    out = capsys.readouterr().out
    assert "starting groupby 'g' operation" in out
    assert "Grouped into 2 groups." in out
    assert "Post reduction" in out
    assert "result_daf" in out


def test_reduce_dodaf_to_daf_diagnose_limits_group_display(capsys):
    daf = Daf(cols=['g', 'v'], lol=[[str(i), i] for i in range(6)])
    grouped = daf.groupby('g')
    result = Daf.reduce_dodaf_to_daf('g', Daf.sum_da, grouped, reduce_cols=['v'], diagnose=True)
    assert result.lol == [[str(i), i] for i in range(6)]
    out = capsys.readouterr().out
    assert "Grouped into 6 groups." in out
    # the initial summary shows groups 0..2 and the last group only, then each group as processed;
    # so group 3 appears once (processing), group 0 twice (summary + processing).
    assert out.count("## Group 3:") == 1
    assert out.count("## Group 0:") == 2


def test_multi_groupby_reduce_diagnose(capsys):
    result = _ghv_daf().multi_groupby_reduce(['g', 'h'], Daf.sum_da, reduce_cols=['v'], diagnose=True)
    assert result['g'].lol == [['x', '', 4], ['y', '', 2]]
    assert result['h'].lol == [['', 'p', 3], ['', 'q', 3]]
    out = capsys.readouterr().out
    assert "starting multi-groupby" in out


# =====================================================================
# apply_colwise()
# =====================================================================

def test_apply_colwise_new_column_with_default_on_error():
    daf = Daf(cols=['a', 'b'], lol=[[1, 2], [3, 4]])
    daf.apply_colwise('c', lambda r: r['a'] / (r['b'] - 2), default=-1)
    assert daf.columns() == ['a', 'b', 'c']
    assert daf.lol == [[1, 2, -1], [3, 4, 1.5]]


def test_apply_colwise_existing_column():
    daf = Daf(cols=['a', 'b'], lol=[[1, 2], [3, 4]])
    daf.apply_colwise('a', lambda r: r['a'] * 2)
    assert daf.lol == [[2, 2], [6, 4]]


# =====================================================================
# reduce()
# =====================================================================

def test_reduce_row_with_str_cols():
    daf = Daf(cols=['a', 'b'], lol=[[1, 2], [3, 4]])
    assert daf.reduce(Daf.sum_da, cols='a') == {'a': 4, 'b': ''}


def _failing_reduction(row, acc, cols=None, **kwargs):
    raise RuntimeError('boom')


def test_reduce_row_func_exception_propagates():
    daf = Daf(cols=['a', 'b'], lol=[[1, 2], [3, 4]])
    with pytest.raises(RuntimeError, match='boom'):
        daf.reduce(_failing_reduction)


def test_reduce_sparse_row_func_exception_propagates():
    daf = Daf(cols=['id', 'j'], lol=[['x', '{"p": 1}'], ['y', '{"q": 2}']])
    with pytest.raises(RuntimeError, match='boom'):
        daf.reduce(_failing_reduction, by='sparse_row', indirect_col='j')


def test_reduce_row_silent_error_skips_failing_rows():
    def fail_on_two(row, acc, cols=None, **kwargs):
        if row['a'] == 2:
            raise ValueError('bad row')
        return Daf.sum_da(row, acc, cols=cols)
    daf = Daf(cols=['a'], lol=[[1], [2], [3]])
    assert daf.reduce(fail_on_two, silent_error=True) == {'a': 4}


def test_reduce_sparse_row_silent_error_skips_failing_rows():
    daf = Daf(cols=['id', 'j'], lol=[['x', '{"p": 1}'], ['y', '{"q": 2}']])
    assert daf.reduce(_failing_reduction, by='sparse_row', indirect_col='j', silent_error=True) == {}


# =====================================================================
# sum_da()
# =====================================================================

def test_sum_da_sparse_respects_cols_filter():
    result = Daf.sum_da({'a': 1, 'b': 5, 'c': 's'}, {'a': 1}, cols=['a', 'c'], is_sparse=True)
    assert result == {'a': 2}


class _AddRaisesKeyError:
    def __add__(self, other):
        raise KeyError('unexpected')
    __radd__ = __add__


def test_sum_da_sparse_unexpected_exception_propagates():
    with pytest.raises(KeyError):
        Daf.sum_da({'a': _AddRaisesKeyError()}, {'a': 0})


def test_sum_da_astype_float():
    result = Daf.sum_da({'a': '2', 'b': True, 'c': 1.5}, {'a': 0, 'b': 0, 'c': 0},
                        cols=['a', 'b', 'c'], astype=float)
    assert result == {'a': 2.0, 'b': 1.0, 'c': 1.5}


def test_sum_da_astype_str_concatenates():
    result = Daf.sum_da({'a': 2, 'b': 1.5}, {'a': '', 'b': 'x'}, cols=['a', 'b'], astype=str)
    assert result == {'a': '2', 'b': 'x1.5'}


def test_sum_da_astype_int_skips_unconvertible():
    assert Daf.sum_da({'a': 'x'}, {'a': 0}, cols=['a'], astype=int) == {'a': 0}


def test_sum_da_astype_uninitialized_accumulator_raises():
    # with astype, reduction_da must be initialized for all cols.
    with pytest.raises(KeyError):
        Daf.sum_da({'a': 1}, {}, cols=['a'], astype=int)


def test_sum_da_cols_unexpected_exception_propagates():
    with pytest.raises(KeyError):
        Daf.sum_da({'a': _AddRaisesKeyError()}, {'a': 0}, cols=['a'])


# =====================================================================
# daf_valuecount() / multi_groupsum()
# =====================================================================

def test_daf_valuecount():
    result = _ghv_daf().daf_valuecount(cols=['g', 'h'])
    assert result == {'g': {'x': 2, 'y': 1}, 'h': {'p': 2, 'q': 1}, 'v': ''}


def test_multi_groupsum():
    result = _ghv_daf().multi_groupsum(['g', 'h'], reduce_cols=['v'])
    assert result['g'].lol == [['x', '', 4], ['y', '', 2]]
    assert result['h'].lol == [['', 'p', 3], ['', 'q', 3]]


def test_multi_groupsum_requires_colnames():
    with pytest.raises(ValueError, match='colnames is required'):
        _ghv_daf().multi_groupsum()


# =====================================================================
# set_col2_from_col1_using_regex_select()
# =====================================================================

def test_set_col2_from_col1_using_regex_select():
    daf = Daf(cols=['s', 't'], lol=[['ab123cd', ''], ['zz9', '']])
    daf.set_col2_from_col1_using_regex_select('s', 't', regex=r'(\d+)')
    assert daf.lol == [['ab123cd', '123'], ['zz9', '9']]

    # col2 defaults to col1
    daf.set_col2_from_col1_using_regex_select('s', regex=r'([a-z]+)')
    assert daf.lol == [['ab', '123'], ['zz', '9']]


# =====================================================================
# alter_daf_per_setting()
# =====================================================================

def test_alter_daf_per_setting_lod_selects_matching_spec():
    daf = Daf(cols=['bid', 'x'], lol=[['04000_1', 'a'], ['01780_2', 'b']])
    settings = {'spec': [
        {'spec_name': 'A', 'colname': 'bid', 'replace_regex': r'/04000_(\d)/14000_\1/'},
        {'spec_name': 'B', 'colname': 'bid', 'replace_regex': '/0/Z/'},
    ]}
    result = daf.alter_daf_per_setting(settings, 'spec', {'spec_name': 'A'})
    assert result is daf
    assert daf.lol == [['14000_1', 'a'], ['01780_2', 'b']]


def test_alter_daf_per_setting_single_dict():
    daf = Daf(cols=['bid', 'x'], lol=[['1', 'a'], ['2', 'b']])
    settings = {'spec': {'spec_name': 'A', 'colname': 'x', 'replace_regex': '/a/AA/'}}
    daf.alter_daf_per_setting(settings, 'spec', {'spec_name': 'A'})
    assert daf.lol == [['1', 'AA'], ['2', 'b']]


def test_alter_daf_per_setting_empty_setting_is_noop():
    daf = Daf(cols=['x'], lol=[['a']])
    assert daf.alter_daf_per_setting({'spec': []}, 'spec', {}).lol == [['a']]


def test_alter_daf_per_setting_missing_setting_raises():
    daf = Daf(cols=['x'], lol=[['a']])
    with pytest.raises(KeyError, match="'spec'"):
        daf.alter_daf_per_setting({}, 'spec', {})


def test_alter_daf_per_setting_missing_setting_silent_is_noop():
    daf = Daf(cols=['x'], lol=[['a']])
    assert daf.alter_daf_per_setting({}, 'spec', {}, silent_error=True).lol == [['a']]


def test_alter_daf_per_setting_none_setting_is_noop():
    daf = Daf(cols=['x'], lol=[['a']])
    assert daf.alter_daf_per_setting({'spec': None}, 'spec', {}).lol == [['a']]


# =====================================================================
# apply_to_col()
# =====================================================================

def test_apply_to_col_composite_keyfield_invalidates_kd():
    daf = Daf(cols=['a', 'b', 'c'], lol=[[1, 2, 3], [4, 5, 6]], keyfield=['a', 'b'])
    daf.apply_to_col('a', lambda v: v * 10)
    assert daf.lol == [[10, 2, 3], [40, 5, 6]]
    assert daf.keys() == [(10, 2), (40, 5)]


# =====================================================================
# count_values_da() with list values
# =====================================================================

def test_count_values_da_list_value_first_row():
    result = Daf.count_values_da({'l': [1, 2]}, {}, ['l'])
    assert result == {'l': [1, 2]}


def test_count_values_da_list_value_does_not_mutate_source_row():
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


# =====================================================================
# valuecounts_for_colname() / valuecounts_for_colnames_ls_selectedby_colname()
# =====================================================================

def test_valuecounts_for_colname_omit_nulls():
    daf = Daf(cols=['a'], lol=[['x'], [''], ['x']])
    assert daf.valuecounts_for_colname('a', omit_nulls=True) == {'x': 2}


def test_valuecounts_for_colnames_ls_selectedby_colname_defaults_to_all_cols():
    daf = Daf(cols=['a', 'b'], lol=[['x', 1], ['y', 2], ['x', 1]])
    result = daf.valuecounts_for_colnames_ls_selectedby_colname(
        selectedby_colname='a', selectedby_colvalue='x')
    assert result == {'a': {'x': 2}, 'b': {1: 2}}


# =====================================================================
# gen_stats_daf()
# =====================================================================

def test_gen_stats_daf():
    daf = Daf(cols=['n', 's'], lol=[[1, 'a'], [2, 'b'], [2, 'c']])
    info = daf.gen_stats_daf([('n', int, '', 'index'), ('s', str, '', 'attrib')])
    assert set(info.keys()) == {'n', 's'}
    assert info['n']['profile'] == 'index'
    assert info['n']['num_uniques'] == 2
    assert info['n']['num_within_reps'] == 1
    assert info['s']['profile'] == 'attrib'


def test_gen_stats_daf_unknown_profile_raises():
    daf = Daf(cols=['n'], lol=[[1]])
    with pytest.raises(NotImplementedError):
        daf.gen_stats_daf([('n', int, '', 'bogus')])


# =====================================================================
# derive_join_translator_daf()
# =====================================================================

def test_derive_join_translator_daf_tuple_shared_fields_and_other_keyfield():
    t = Daf.derive_join_translator_daf('id', 'oid', ['id', 'name'], ['oid', 'name'],
                                       shared_fields=('id',))
    # 'oid' is appended to shared_fields so it is omitted from the other side;
    # 'name' is common and not shared, so both sides get suffixes.
    assert t.lol == [
        ['id', 'daf1', 'id', True],
        ['name_daf1', 'daf1', 'name', False],
        ['name_daf2', 'daf2', 'name', False],
    ]


def test_derive_join_translator_daf_does_not_mutate_shared_fields():
    shared = ['zz']
    Daf.derive_join_translator_daf('id', 'oid', ['id'], ['oid'], shared_fields=shared)
    assert shared == ['zz']


def test_join_reused_shared_fields_list_is_not_changed():
    # a key left behind in a reused list would make a plain column 'k' look shared in the next join.
    shared = ['x']
    a = Daf(cols=['k', 'v'], lol=[[1, 'a1']], keyfield='k', name='A')
    b = Daf(cols=['k', 'v'], lol=[[1, 'b1']], keyfield='k', name='B')
    a.join(b, shared_fields=shared)
    assert shared == ['x']
    c = Daf(cols=['id', 'k', 'v'], lol=[[1, 'c_k', 'c1']], keyfield='id', name='C')
    d = Daf(cols=['id', 'k', 'v'], lol=[[1, 'd_k', 'd1']], keyfield='id', name='D')
    assert c.join(d, shared_fields=shared).lol == [[1, 'c_k', 'c1', 'd_k', 'd1']]


def test_derive_join_translator_daf_accepts_tuple_shared_fields():
    translator = Daf.derive_join_translator_daf('k', 'k', ['k', 'a'], ['k', 'a'], shared_fields=('a',))
    assert translator.num_rows() == 2


# =====================================================================
# join()
# =====================================================================

def test_join_outer_diagnose(capsys):
    a = Daf(cols=['id', 'name'], lol=[[1, 'A'], [2, 'B']], keyfield='id')
    b = Daf(cols=['id', 'sal'], lol=[[1, 10], [3, 30]], keyfield='id')
    result = a.join(b, how='outer', diagnose=True)
    assert result.lol == [[1, 'A', 10], [2, 'B', None], [3, None, 30]]
    out = capsys.readouterr().out
    assert "Initiating join" in out
    assert "Translator Daf" in out
    assert "Resulting Daf" in out


def test_join_tuple_key_values_raise_keyerror():
    a = Daf(cols=['id', 'name'], lol=[[(1, 2), 'A']], keyfield='id')
    b = Daf(cols=['id', 'sal'], lol=[[(1, 2), 10]], keyfield='id')
    with pytest.raises(KeyError, match='complex keys'):
        a.join(b)


def test_join_composite_keyfield_with_custom_translator_raises_keyerror():
    a = Daf(cols=['a', 'b', 'v'], lol=[[1, 2, 3]], keyfield=['a', 'b'])
    b = Daf(cols=['a', 'b', 'w'], lol=[[1, 2, 4]], keyfield=['a', 'b'])
    translator = Daf(cols=['resolved_colname', 'source_name', 'source_colname', 'is_keyfield'],
                     lol=[['v', 'daf1', 'v', False], ['w', 'daf2', 'w', False]])
    with pytest.raises(KeyError, match='complex keys'):
        a.join(b, custom_translator_daf=translator)


@pytest.mark.xfail(strict=True, reason="BUG: join() with composite keyfields fails with a bare "
                   "AssertionError in derive_join_translator (daf.py:8071) instead of the intended "
                   "KeyError 'join not supported for complex keys' (daf.py:8262)")
def test_join_composite_keyfield_raises_keyerror():
    a = Daf(cols=['a', 'b', 'v'], lol=[[1, 2, 3]], keyfield=['a', 'b'])
    b = Daf(cols=['a', 'b', 'w'], lol=[[1, 2, 4]], keyfield=['a', 'b'])
    with pytest.raises(KeyError):
        a.join(b)


# =====================================================================
# join_records()
# =====================================================================

def _three_source_translator():
    return Daf(cols=['resolved_colname', 'source_name', 'source_colname', 'is_keyfield'],
               lol=[['x', 'd1', 'x', False], ['y', 'd2', 'y', False], ['z', 'd3', 'z', False]])


def test_join_records_requires_names_for_more_than_two_sources():
    with pytest.raises(ValueError, match='join_names_ls must be specified'):
        Daf.join_records([{'x': 1}, {'y': 2}], _three_source_translator())


def test_join_records_skips_sources_not_in_join_names():
    tr = _three_source_translator()
    assert Daf.join_records([{'x': 1}, {'y': 2}], tr, ['d1', 'd2']) == {'x': 1, 'y': 2}
    assert Daf.join_records([{'x': 1}, None], tr, ['d1', 'd2']) == {'x': 1, 'y': None}


# =====================================================================
# to_md() / daf_to_lol_summary() / to_md_cols()
# =====================================================================

def test_to_md_summary_includes_schema():
    daf = Daf(cols=['a'], lol=[[1]], name='nm')
    daf.attrs['schema'] = 'myschema'
    md = daf.to_md(include_summary=True)
    assert "%% daf rows=1; cols=1; keyfield=''; name='nm'; schema='myschema'" in md


def test_daf_to_lol_summary_non_list_disp_cols():
    daf = Daf(cols=['a', 'b'], lol=[[1, 2]])
    assert daf.daf_to_lol_summary(disp_cols=('X', 'Y')) == [['X', 'Y'], [1, 2]]


# =====================================================================
# value_counts_daf()
# =====================================================================

def test_value_counts_daf_basic():
    daf = Daf(cols=['a'], lol=[['x'], [''], ['x'], ['y']])
    result = daf.value_counts_daf('a')
    assert result.columns() == ['a', 'counts']
    assert result.lol == [['x', 2], ['', 1], ['y', 1]]


def test_value_counts_daf_sorted_total_omit_nulls():
    daf = Daf(cols=['a'], lol=[['y'], [''], ['x'], ['x']])
    result = daf.value_counts_daf('a', sort=True, include_total=True, omit_nulls=True)
    assert result.lol == [['x', 2], ['y', 1], [' **Total** ', 3]]


# =====================================================================
# unpack_indirect() (module-level function)
# =====================================================================

def test_unpack_indirect_flattens_indirect_and_defaults():
    daf = Daf(cols=['id', 'j'], lol=[['r1', {'p': 1, 'q': 2}], ['r2', '{"p": 3}']])
    result = daf_module.unpack_indirect(daf, indirect_col='j', cols=['id', 'p', 'q'],
                                        default='D', silent_error=True)
    assert result.columns() == ['id', 'p', 'q']
    assert result.lol == [['r1', 1, 2], ['r2', 3, 'D']]


def test_unpack_indirect_missing_col_raises_keyerror():
    daf = Daf(cols=['id', 'j'], lol=[['r1', {'p': 1}]])
    with pytest.raises(KeyError, match="Column 'q' not found"):
        daf_module.unpack_indirect(daf, indirect_col='j', cols=['id', 'q'], default='D')


def test_unpack_indirect_missing_indirect_col_raises():
    daf = Daf(cols=['id'], lol=[['r1']])
    with pytest.raises(RuntimeError, match='nope not found'):
        daf_module.unpack_indirect(daf, indirect_col='nope', cols=['id'], default='D')
