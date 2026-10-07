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
    was_active, report_path, print_report = P.is_active(), P._report_path, P._print_report
    P.stop()
    P.reset()
    yield
    P.stop()
    P.reset()
    P._report_path, P._print_report = report_path, print_report     # no stray report at exit
    if was_active:
        P.start(report_path=report_path, print_report=print_report)


def _groups_by_how():
    """ The totals by how the tables were made, with live tables folded in. """
    groups = {}
    for (site, how), gs in P._groups.items():
        groups[how] = gs
    return groups


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
    assert os.path.basename(ts.site[0]) == 'test_daf_profile.py'
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
    P.report()                                      # adds the freed tables to the totals
    gs = _groups_by_how()['from_lod']
    assert gs.tables == 3
    assert gs.max_rows == 2 and gs.max_cols == 1
    assert gs.ops == {'sort_by_colname': 3}
    assert P._rows_bands[0] == 3                    # all three had 10 rows or fewer


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
    P.report()
    assert _groups_by_how()['Daf()'].tables >= 199


def test_rows_at_call_are_counted_in_size_bands():
    P.start()
    d = Daf(lol=[[i] for i in range(500)], cols=['a'])
    d.num_rows()
    bands = P._methods['num_rows'].bands
    assert bands[P.band(500)] == 1 and sum(bands) == 1
    assert P._methods['num_rows'].max_rows == 500


def test_a_method_that_raises_leaves_counting_working():
    P.start()
    d = Daf(lol=[[1, 2]], cols=['a', 'b'])
    with pytest.raises(ValueError):
        d.append([1, 2, 3])
    d.append([3, 4])
    assert P._methods['append'].calls == 1          # the call that raised is not counted


def test_report_has_each_section_and_names_the_calling_line():
    P.start()
    d = Daf(cols=['a'])
    d.append([1])
    text = P.report()
    for heading in ('# Daffodil profile', '## Tables by the line that created them', '## Table sizes',
                    '## Methods', '## Busiest call sites'):
        assert heading in text
    assert 'test_daf_profile.py:' in text
    assert '| append' in text


def test_building_the_report_is_not_counted():
    P.start()
    Daf(cols=['a']).append([1])
    before = {name: ms.calls for name, ms in P._methods.items()}
    P.report()
    after = {name: ms.calls for name, ms in P._methods.items()}
    assert before == after


def test_report_writes_a_file_with_the_process_id(tmp_path):
    P.start()
    Daf(cols=['a']).append([1])
    P.report(str(tmp_path / 'prof_{pid}.md'))
    written = tmp_path / f'prof_{os.getpid()}.md'
    assert written.read_text().startswith('# Daffodil profile')


def test_reset_clears_the_totals():
    P.start()
    Daf(cols=['a']).append([1])
    P.reset()
    assert P._methods == {} and P._live == {} and P._groups == {}
    assert P.is_active()


def test_start_from_env(monkeypatch):
    monkeypatch.setenv('DAFFODIL_PROFILE', '0')
    P.start_from_env(Daf)
    assert not P.is_active()
    monkeypatch.setenv('DAFFODIL_PROFILE', '1')
    monkeypatch.setenv('DAFFODIL_PROFILE_FILE', 'unused_{pid}.md')
    P.start_from_env(Daf)
    assert P.is_active()
    assert P._report_path == 'unused_{pid}.md' and P._print_report
    P.stop()                                        # the fixture puts back any profiling from before


def test_an_unusual_table_does_not_break_the_call():
    # select_icols(0, flip=True) makes a Daf whose rows are not lists.
    P.start()
    d = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'name'])
    result = d.select_icols(0, flip=True)
    P.stop()
    expected = Daf(lol=[[1, 'a'], [2, 'b']], cols=['id', 'name']).select_icols(0, flip=True)
    assert result.lol == expected.lol
