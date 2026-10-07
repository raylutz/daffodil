# test_daf_profile.py
#
# The profiling mode in daffodil/lib/daf_profile.py.

import gc
import os

import pytest

from daffodil.daf import Daf
from daffodil.lib import daf_profile as P


@pytest.fixture(autouse=True)
def _clean_profile():
    # The suite may run with DAFFODIL_PROFILE=1. Then put that profiling back after each test.
    saved = (P.is_active(), P._stage, P._data_dir, P._report_path, P._print_report)
    P.stop()
    P.reset()
    yield
    P.stop()
    P.reset()
    was_active, P._stage, P._data_dir, P._report_path, P._print_report = saved      # no stray files at exit
    if was_active:
        P.start(stage=P._stage, data_dir=P._data_dir, report_path=P._report_path, print_report=P._print_report)


def _row(daf, **match):
    """ The one row of a Daf whose columns have the given values. """
    rows = [row for row in daf.iter_dict() if all(row[col] == val for col, val in match.items())]
    assert len(rows) == 1, rows
    return rows[0]


# ---- wrapping

def test_start_wraps_and_stop_restores_the_methods():
    original_append = Daf.__dict__['append']
    original_from_lod = Daf.__dict__['from_lod']
    P.start()
    assert P.is_active()
    assert Daf.__dict__['append'] is not original_append
    assert getattr(Daf.append, '__daf_profiled__', False)
    P.stop()
    assert not P.is_active()
    assert Daf.__dict__['append'] is original_append
    assert Daf.__dict__['from_lod'] is original_from_lod


def test_start_twice_wraps_once():
    P.start()
    P.start()
    assert not getattr(Daf.append.__wrapped__, '__daf_profiled__', False)


def test_nothing_is_counted_when_off():
    d = Daf(cols=['a'])
    d.append([1])
    assert P._methods == {}


# ---- counting

def test_outer_calls_are_counted_and_inner_calls_are_not():
    P.start()
    d = Daf(lol=[[i, i % 3] for i in range(30)], cols=['a', 'b'])
    for i in range(5):
        d.append([100 + i, 0])
    d.select_where(lambda row: row['b'] == 0)
    assert P._methods['append'].calls == 5
    assert P._methods['select_where'].calls == 1
    assert P._methods['__init__'].calls == 1
    assert 'clone_empty' not in P._methods          # called by select_where, inside
    assert 'select_irows' not in P._methods


def test_a_table_is_summarized_by_how_and_where_it_was_made():
    P.start()
    d = Daf(cols=['id', 'v'], keyfield='id')
    for i in range(12):
        d.append([i, 'x'])
    sel = d.select_where(lambda row: row['id'] < 4)
    ts = P._live[id(d)]
    assert ts.how == 'Daf()'
    assert ts.site.split(':')[0].endswith('test_daf_profile')
    assert ts.max_rows == 12 and ts.max_cols == 2 and ts.keyed
    assert ts.ops == {'append': 12, 'select_where': 1}
    sel_ts = P._live[id(sel)]
    assert sel_ts.how == 'select_where'
    assert sel_ts.max_rows == 4


def test_a_freed_table_is_added_to_the_totals_of_its_line():
    P.start()
    for _ in range(3):
        d = Daf.from_lod([{'a': 1}, {'a': 2}])
        d.sort_by_colname('a')
    del d
    gc.collect()
    tabs = P.tables()
    row = _row(tabs['tables'], how='from_lod')
    assert row['tables'] == 3 and row['max_rows'] == 2 and row['max_cols'] == 1
    assert row['rows_b0'] == 3 and row['rows_sum'] == 6
    assert _row(tabs['ops'], how='from_lod', method='sort_by_colname')['calls'] == 3


def test_a_table_freed_while_counting_does_not_hang():
    # A table in a reference cycle is freed by the garbage collector, which can run in the
    # middle of a counted call, while the profiler holds its lock.
    P.start()
    gc.set_threshold(1)                             # collect as often as possible
    try:
        for _ in range(200):
            d = Daf(cols=['a'])
            d.attrs['me'] = d                       # a cycle, so only the collector frees it
            d.append([1])
    finally:
        gc.set_threshold(700, 10, 10)
    gc.collect()
    assert _row(P.tables()['tables'], how='Daf()')['tables'] >= 199


def test_rows_at_call_are_counted_in_size_bands():
    P.start()
    d = Daf(lol=[[i] for i in range(500)], cols=['a'])
    d.num_rows()
    row = _row(P.tables()['methods'], method='num_rows')
    assert row[f'rows_b{P.band(500)}'] == 1 and row['calls'] == 1 and row['max_rows'] == 500


def test_a_method_that_raises_leaves_counting_working():
    P.start()
    d = Daf(lol=[[1, 2]], cols=['a', 'b'])
    with pytest.raises(ValueError):
        d.append([1, 2, 3])
    d.append([3, 4])
    assert P._methods['append'].calls == 1          # the call that raised is not counted


def test_an_unusual_table_does_not_break_the_call():
    # select_icols(0, flip=True) makes a Daf whose rows are not lists.
    P.start()
    d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'name'])
    result = d.select_icols(0, flip=True)
    P.stop()
    expected = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'name']).select_icols(0, flip=True)
    assert result.lol == expected.lol


# ---- tables, files and combining

def test_tables_have_the_stage_and_the_documented_columns():
    P.start(stage='tabulate')
    Daf(cols=['a']).append([1])
    tabs = P.tables()
    assert set(tabs) == {'info', 'tables', 'ops', 'methods', 'sites'}
    for kind, daf in tabs.items():
        assert daf.columns() == P.TABLE_COLS[kind]
        assert set(daf.col_to_la('stage')) == {'tabulate'}
    assert _row(tabs['info'], stage='tabulate')['pid'] == os.getpid()


def test_dump_and_load_give_back_the_same_tables(tmp_path):
    P.start(stage='s1')
    d = Daf(cols=['a'])
    for i in range(20):
        d.append([i])
    before = P.tables()
    path = P.dump(str(tmp_path / 'p.md'))
    after = P.load(path)
    for kind in ('tables', 'ops', 'methods', 'sites'):
        assert after[kind].columns() == before[kind].columns()
        assert sorted(map(str, after[kind].lol)) == sorted(map(str, before[kind].lol))
    assert isinstance(_row(after['methods'], method='append')['seconds'], float)


def test_combine_adds_counts_and_keeps_the_largest_sizes():
    P.start(stage='s1')
    a = Daf(cols=['x'])
    for i in range(5):
        a.append([i])
    run1 = P.tables()
    P.reset()
    b = Daf(cols=['x'])
    for i in range(8):
        b.append([i])
    run2 = P.tables()
    P.stop()
    combined = P.combine([run1, run2])
    append = _row(combined['methods'], method='append')
    assert append['calls'] == 13 and append['max_rows'] == 7
    made = [row for row in combined['tables'].iter_dict() if row['how'] == 'Daf()']
    assert sum(row['tables'] for row in made) == 2
    assert max(row['max_rows'] for row in made) == 8
    assert combined['info'].num_rows() == 2


def test_combine_keeps_stages_apart_unless_asked():
    P.start(stage='load')
    Daf(cols=['x']).append([1])
    run1 = P.tables()
    P.reset()
    P._stage = 'tabulate'
    Daf(cols=['x']).append([1])
    run2 = P.tables()
    by_stage = P.combine([run1, run2])
    assert {row['stage'] for row in by_stage['methods'].iter_dict()} == {'load', 'tabulate'}
    overall = P.combine([run1, run2], by_stage=False)
    assert _row(overall['methods'], method='append')['calls'] == 2


def test_report_has_a_section_for_each_stage():
    P.start(stage='load')
    Daf(cols=['x']).append([1])
    run1 = P.tables()
    P.reset()
    P._stage = 'tabulate'
    Daf(cols=['x']).append([1])
    run2 = P.tables()
    text = P.report(P.combine([run1, run2]))
    for heading in ('# Daffodil profile', '## Runs', '# All stages', '# Stage load', '# Stage tabulate',
                    '## Tables by the line that created them', '## Table sizes', '## Methods', '## Busiest call sites'):
        assert heading in text


def test_building_the_report_is_not_counted():
    P.start()
    Daf(cols=['a']).append([1])
    before = {name: ms.calls for name, ms in P._methods.items()}
    P.report()
    P.dump(os.devnull)
    after = {name: ms.calls for name, ms in P._methods.items()}
    assert before == after


def test_report_writes_a_file_with_the_process_id(tmp_path):
    P.start()
    Daf(cols=['a']).append([1])
    P.report(path=str(tmp_path / 'prof_{pid}.md'))
    written = tmp_path / f'prof_{os.getpid()}.md'
    assert written.read_text().startswith('# Daffodil profile')


def test_command_line_combines_a_directory(tmp_path):
    P.start(stage='one')
    Daf(cols=['a']).append([1])
    P.dump(P.default_file_name(str(tmp_path)))
    P.reset()
    P._stage = 'two'
    Daf(cols=['a']).append([1])
    P.dump(P.default_file_name(str(tmp_path)))
    P.stop()
    out = tmp_path / 'report.md'
    assert P.main(['combine', str(tmp_path), '-o', str(out), '--data', str(tmp_path / 'all.md')]) == 0
    text = out.read_text()
    assert '# Stage one' in text and '# Stage two' in text
    assert P.load(str(tmp_path / 'all.md'))['methods'].num_rows() == 4    # append and __init__, two stages


def test_reset_clears_the_totals():
    P.start()
    Daf(cols=['a']).append([1])
    P.reset()
    assert P._methods == {} and P._live == {} and P._groups == {}
    assert P.is_active()


def test_start_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv('DAFFODIL_PROFILE', '0')
    P.start_from_env(Daf)
    assert not P.is_active()
    monkeypatch.setenv('DAFFODIL_PROFILE', '1')
    monkeypatch.setenv('DAFFODIL_PROFILE_STAGE', 'tabulate')
    monkeypatch.setenv('DAFFODIL_PROFILE_DIR', str(tmp_path))
    P.start_from_env(Daf)
    assert P.is_active()
    assert P._stage == 'tabulate' and P._data_dir == str(tmp_path) and P._print_report
    P.stop()                                        # the fixture puts back any profiling from before
