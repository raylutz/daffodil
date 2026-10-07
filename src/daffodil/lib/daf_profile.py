# daf_profile.py
"""
Profiling mode for Daf: how big the tables get, and what is done with them.

Turn it on with the environment variable `DAFFODIL_PROFILE=1`, or in code:

    from daffodil.lib import daf_profile
    daf_profile.start(stage='tabulate')
    ...
    print(daf_profile.report())

While it is on, every public method of Daf is wrapped. Only the outer call is counted:
when a Daf method calls another one inside, the inner call is not. When it is off,
nothing is wrapped and Daf runs at full speed.

While the program runs, it keeps running totals in plain dicts, which is fast. `tables()`
turns them into Daf tables:

- `info`: one row for the process.
- `tables`: one row for each stage, creation line and way of making a table. How many
  tables, the most rows and columns, and how many tables reached each size band.
- `ops`: the methods called on those tables, with counts.
- `methods`: for each method, the calls, the time, and the rows at each call by size band.
- `sites`: for each call site and method, the calls and the time.

`dump()` writes the tables to one Markdown file. `combine()` adds up the tables of several
runs, such as the stages of a pipeline that run as separate programs. `report()` turns
tables into a readable report.

A call site is the module and line, as in `auditengine.tabulate:212`, so the same line has
the same name in every run. A script run as `__main__` is named by its file name.

With `DAFFODIL_PROFILE=1`, when the program ends the tables are written to
`DAFFODIL_PROFILE_DIR`, the current directory by default, as
`daffodil_profile_<stage>_<host>_<pid>.md`. The report is printed to stderr. Set the stage
name with `DAFFODIL_PROFILE_STAGE`. To combine the files of a run:

    python -m daffodil.lib.daf_profile combine DIR_OR_FILES... -o report.md

Each counted call costs about 3 µs, and each call that makes a new table about 15 µs more.
A call that Daf makes inside another one costs about 0.4 µs. So the times of cheap calls
are inflated. Counts and sizes are exact. A child process that ends with `os._exit()`, as
multiprocessing workers do, writes nothing at exit. Call `dump()` in the worker instead.
"""

import atexit
import bisect
import datetime
import functools
import glob
import os
import platform
import socket
import sys
import threading
import time
import weakref
from typing import List, Dict, Any, Tuple, Callable, Type    # noqa: F401

from daffodil.lib.daf_types import T_ls, T_la, T_da, T_lola


SIZE_BANDS      = (10, 100, 1_000, 10_000, 100_000, 1_000_000)
BAND_NAMES      = ['≤10', '≤100', '≤1k', '≤10k', '≤100k', '≤1M', '>1M']
NUM_BANDS       = len(BAND_NAMES)
ROW_BAND_COLS   = [f'rows_b{i}' for i in range(NUM_BANDS)]
COL_BAND_COLS   = [f'cols_b{i}' for i in range(NUM_BANDS)]
FILE_PREFIX     = 'daffodil_profile_'

# The columns of each table, the columns that identify a row, and the columns that are
# combined by taking the largest value. All other columns are added up.
TABLE_COLS: Dict[str, T_ls] = {
    'info':     ['stage', 'host', 'pid', 'python', 'daffodil', 'started', 'seconds'],
    'tables':   ['stage', 'created_at', 'how', 'tables', 'keyed', 'rows_sum', 'max_rows', 'max_cols']
                + ROW_BAND_COLS + COL_BAND_COLS,
    'ops':      ['stage', 'created_at', 'how', 'method', 'calls'],
    'methods':  ['stage', 'method', 'calls', 'seconds', 'max_rows'] + ROW_BAND_COLS,
    'sites':    ['stage', 'site', 'method', 'calls', 'seconds'],
}
KEY_COLS: Dict[str, T_ls] = {
    'tables':   ['stage', 'created_at', 'how'],
    'ops':      ['stage', 'created_at', 'how', 'method'],
    'methods':  ['stage', 'method'],
    'sites':    ['stage', 'site', 'method'],
}
MAX_COLS = ['max_rows', 'max_cols']
FLOAT_COLS = ['seconds']
TEXT_COLS = ['stage', 'host', 'python', 'daffodil', 'started', 'created_at', 'how', 'method', 'site']


class _TableStats:
    """ The summary of one live table. """
    __slots__ = ('ref', 'site', 'how', 'max_rows', 'max_cols', 'keyed', 'ops')

    def __init__(self, site: str, how: str) -> None:
        self.ref: Any   = None          # a weak reference to the table, whose callback retires it
        self.site       = site
        self.how        = how
        self.max_rows   = 0
        self.max_cols   = 0
        self.keyed      = False
        self.ops: Dict[str, int] = {}


class _GroupStats:
    """ The totals of the freed tables made at one line, in one way. """
    __slots__ = ('tables', 'keyed', 'rows_sum', 'max_rows', 'max_cols', 'row_bands', 'col_bands', 'ops')

    def __init__(self) -> None:
        self.tables     = 0
        self.keyed      = 0
        self.rows_sum   = 0
        self.max_rows   = 0
        self.max_cols   = 0
        self.row_bands  = [0] * NUM_BANDS
        self.col_bands  = [0] * NUM_BANDS
        self.ops: Dict[str, int] = {}


class _MethodStats:
    """ The totals of the outer calls to one method. """
    __slots__ = ('calls', 'seconds', 'bands', 'max_rows')

    def __init__(self) -> None:
        self.calls      = 0
        self.seconds    = 0.0
        self.bands      = [0] * NUM_BANDS
        self.max_rows   = 0


_lock       = threading.Lock()
_local      = threading.local()             # .depth: 1 inside a counted call
_active     = False
_cls: Type | None = None
_originals: Dict[str, Any] = {}
_live: Dict[int, _TableStats] = {}
_freed: List[_TableStats] = []              # freed tables, not yet added to the totals
_groups: Dict[Tuple[str, str], _GroupStats] = {}
_methods: Dict[str, _MethodStats] = {}
_sites: Dict[Tuple[str, str], List[float]] = {}     # (site, method) -> [calls, seconds]
_site_names: Dict[Tuple[Any, int], str] = {}        # (code object, line) -> 'module:line'
_started_at = 0.0
_stage      = ''
_data_dir: str | None = None
_report_path: str | None = None
_print_report = False
_atexit_registered = False


def band(n: int) -> int:
    """ The index of the size band that holds n. """
    return bisect.bisect_left(SIZE_BANDS, n)


def start(
        cls:            Type | None = None,     # the class to profile; Daf if not given
        stage:          str         = '',       # a name for this run, such as a pipeline stage
        data_dir:       str | None  = None,     # at exit, write the tables to a file in this directory
        report_path:    str | None  = None,     # at exit, write the report to this file
        print_report:   bool        = False,    # at exit, print the report to stderr
        ) -> None:
    """
    Start profiling: wrap the public methods of Daf and begin counting.

    Calling it again while profiling is on does nothing. The totals are kept until
    `reset()`.

    Args:
        cls: The class to profile. Daf if not given.
        stage: A name for this run, such as a stage of a pipeline. It fills the `stage`
            column of the tables, so the runs of several stages can be combined.
        data_dir: When the program ends, write the tables to
            `daffodil_profile_<stage>_<host>_<pid>.md` in this directory.
        report_path: When the program ends, write the report to this file. `{pid}` in the
            name is replaced with the process id.
        print_report: When the program ends, print the report to stderr.
    """
    global _active, _cls, _started_at, _stage, _data_dir, _report_path, _print_report, _atexit_registered

    if _active:
        return
    if cls is None:
        from daffodil.daf import Daf
        cls = Daf

    _cls            = cls
    _stage          = stage
    _data_dir       = data_dir
    _report_path    = report_path
    _print_report   = print_report
    if not _started_at:
        _started_at = time.time()

    for name, attr in list(vars(cls).items()):
        if name.startswith('_') and name not in _DUNDERS:
            continue
        if isinstance(attr, staticmethod):
            _originals[name] = attr
            setattr(cls, name, staticmethod(_wrap(attr.__func__, name, 'static')))
        elif isinstance(attr, classmethod):
            _originals[name] = attr
            setattr(cls, name, classmethod(_wrap(attr.__func__, name, 'class')))
        elif callable(attr) and not isinstance(attr, type):
            _originals[name] = attr
            setattr(cls, name, _wrap(attr, name, 'method'))

    _active = True
    if not _atexit_registered:
        atexit.register(_at_exit)
        _atexit_registered = True


def stop() -> None:
    """
    Stop profiling and put the original methods back. The totals are kept.
    """
    global _active
    if not _active or _cls is None:
        return
    for name, attr in _originals.items():
        setattr(_cls, name, attr)
    _originals.clear()
    _active = False


def is_active() -> bool:
    """ True while profiling is on. """
    return _active


def reset() -> None:
    """ Clear all totals. Profiling stays on or off as it was. """
    global _started_at
    with _lock:
        _live.clear()
        _freed.clear()
        _groups.clear()
        _methods.clear()
        _sites.clear()
        _started_at = time.time() if _active else 0.0


_DUNDERS = {'__init__', '__iter__', '__bool__', '__format__', '__eq__', '__str__',
            '__repr__', '__contains__', '__len__', '__getitem__', '__setitem__'}


def _wrap(fn: Callable, name: str, kind: str) -> Callable:
    """ Wrap one method so that its outer calls are counted. """

    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        local = _local
        if getattr(local, 'depth', 0):
            return fn(*args, **kwargs)          # an inner call, or the report being built
        this = args[0] if (kind == 'method' and args) else None
        rows_before = len(this.lol) if (this is not None and name != '__init__') else -1
        local.depth = 1
        start_time = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
        finally:
            seconds = time.perf_counter() - start_time
            local.depth = 0
        try:
            _record(name, this, rows_before, result, seconds, sys._getframe(1))
        except Exception:                       # the profiler must never break the call it watches
            pass
        return result

    wrapped.__daf_profiled__ = True             # type: ignore[attr-defined]
    return wrapped


def _site_name(frame: Any) -> str:
    """ 'module:line' for a frame. A script run as __main__ is named by its file name. """
    key = (frame.f_code, frame.f_lineno)
    name = _site_names.get(key)
    if name is None:
        module = frame.f_globals.get('__name__', '?')
        if module == '__main__':
            module = os.path.basename(frame.f_code.co_filename)
        name = _site_names[key] = f"{module}:{frame.f_lineno}"
    return name


def _record(name: str, this: Any, rows_before: int, result: Any, seconds: float, frame: Any) -> None:
    """ Add one outer call to the totals. """
    site = _site_name(frame)
    with _lock:
        if _freed:
            _fold_freed()
        ms = _methods.get(name)
        if ms is None:
            ms = _methods[name] = _MethodStats()
        ms.calls   += 1
        ms.seconds += seconds

        site_tot = _sites.get((site, name))
        if site_tot is None:
            site_tot = _sites[(site, name)] = [0, 0.0]
        site_tot[0] += 1
        site_tot[1] += seconds

        rows_at_call = rows_before
        if this is not None and isinstance(this, _cls):     # type: ignore[arg-type]
            if name == '__init__':
                ts = _table_stats(this, site, 'Daf()')
                rows_at_call = len(this.lol)
            else:
                ts = _table_stats(this, site, '(made before profiling)')
                ops = ts.ops
                ops[name] = ops.get(name, 0) + 1
            _update_size(ts, this)

        for made in (_dafs_in(result) if result is not this else ()):
            ts = _table_stats(made, site, name)
            _update_size(ts, made)
            if rows_at_call < 0:
                rows_at_call = len(made.lol)

        if rows_at_call >= 0:
            ms.bands[band(rows_at_call)] += 1
            if rows_at_call > ms.max_rows:
                ms.max_rows = rows_at_call


_PLAIN_TYPES = {int, float, str, bool, bytes}


def _dafs_in(result: Any) -> List[Any]:
    """ The Dafs in a result: the result itself, or the items of a list, tuple or dict. """
    if _cls is None or result is None or type(result) in _PLAIN_TYPES:
        return []
    if isinstance(result, _cls):
        return [result]
    if isinstance(result, (list, tuple)):
        return [item for item in result[:1000] if isinstance(item, _cls)]
    if isinstance(result, dict):
        return [item for item in list(result.values())[:1000] if isinstance(item, _cls)]
    return []


def _table_stats(daf: Any, site: str, how: str) -> _TableStats:
    """ The summary of a table, made the first time the table is seen. """
    ts = _live.get(id(daf))
    if ts is None:
        oid = id(daf)
        ts = _live[oid] = _TableStats(site, how)
        ts.ref = weakref.ref(daf, functools.partial(_on_free, oid))
    return ts


def _update_size(ts: _TableStats, daf: Any) -> None:
    num_rows = len(daf.lol)
    if daf.hd:
        num_cols = len(daf.hd)
    elif daf.lol and isinstance(daf.lol[0], list):
        num_cols = len(daf.lol[0])
    else:
        num_cols = 0
    if num_rows > ts.max_rows:
        ts.max_rows = num_rows
    if num_cols > ts.max_cols:
        ts.max_cols = num_cols
    if daf.keyfield:
        ts.keyed = True


def _on_free(oid: int, _ref: Any) -> None:
    """ The weak reference callback: the table with this id was freed. """
    _retire(oid)


def _retire(oid: int) -> None:
    """
    Called when a table is freed. It takes no lock: a table can be freed while the lock is
    held, as when the garbage collector runs inside `_record()`. The summary leaves `_live` at
    once, because a new object can get the same id, and is queued for `_fold_freed()`.
    """
    ts = _live.pop(oid, None)
    if ts is not None:
        _freed.append(ts)


def _fold_freed() -> None:
    """ Add the queued freed tables to the totals. Call with the lock held. """
    while _freed:
        _fold(_freed.pop(), _groups)


def _fold(ts: _TableStats, groups: Dict[Tuple[str, str], _GroupStats]) -> None:
    gs = groups.get((ts.site, ts.how))
    if gs is None:
        gs = groups[(ts.site, ts.how)] = _GroupStats()
    gs.tables   += 1
    gs.keyed    += ts.keyed
    gs.rows_sum += ts.max_rows
    gs.max_rows = max(gs.max_rows, ts.max_rows)
    gs.max_cols = max(gs.max_cols, ts.max_cols)
    gs.row_bands[band(ts.max_rows)] += 1
    gs.col_bands[band(ts.max_cols)] += 1
    gs_ops = gs.ops
    for op, num in ts.ops.items():
        gs_ops[op] = gs_ops.get(op, 0) + num


# ---- the totals as Daf tables

def tables() -> Dict[str, Any]:
    """
    Return the totals so far as Daf tables, keyed by 'info', 'tables', 'ops', 'methods' and
    'sites'. Tables that are still alive are included, as if they were freed now.

    Returns:
        A dict of Daf tables. See the module notes for the columns.
    """
    from daffodil.daf import Daf

    with _lock:
        _fold_freed()
        groups: Dict[Tuple[str, str], _GroupStats] = {}
        for key, gs in _groups.items():
            copy_gs = _GroupStats()
            copy_gs.tables, copy_gs.keyed, copy_gs.rows_sum = gs.tables, gs.keyed, gs.rows_sum
            copy_gs.max_rows, copy_gs.max_cols = gs.max_rows, gs.max_cols
            copy_gs.row_bands, copy_gs.col_bands = list(gs.row_bands), list(gs.col_bands)
            copy_gs.ops = dict(gs.ops)
            groups[key] = copy_gs
        for ts in list(_live.values()):
            _fold(ts, groups)
        methods = {name: (ms.calls, ms.seconds, ms.max_rows, list(ms.bands)) for name, ms in _methods.items()}
        sites   = {key: tuple(val) for key, val in _sites.items()}

    local = _local
    outer_depth = getattr(local, 'depth', 0)
    local.depth = 1                             # building the tables is not counted
    try:
        try:
            from importlib.metadata import version
            daf_version = version('daffodil')
        except Exception:
            daf_version = '?'
        stage = _stage
        info_lol: T_lola = [[stage, socket.gethostname(), os.getpid(), platform.python_version(), daf_version,
                             datetime.datetime.fromtimestamp(_started_at).isoformat(timespec='seconds') if _started_at else '',
                             round(time.time() - _started_at, 3) if _started_at else 0.0]]
        tables_lol: T_lola = []
        ops_lol: T_lola = []
        for (site, how), gs in groups.items():
            tables_lol.append([stage, site, how, gs.tables, gs.keyed, gs.rows_sum, gs.max_rows, gs.max_cols]
                              + gs.row_bands + gs.col_bands)
            for op, num in gs.ops.items():
                ops_lol.append([stage, site, how, op, num])
        methods_lol: T_lola = [[stage, name, calls, seconds, max_rows] + bands
                               for name, (calls, seconds, max_rows, bands) in methods.items()]
        sites_lol: T_lola = [[stage, site, name, int(calls), seconds] for (site, name), (calls, seconds) in sites.items()]

        lols = {'info': info_lol, 'tables': tables_lol, 'ops': ops_lol, 'methods': methods_lol, 'sites': sites_lol}
        return {kind: Daf(lol=lol, cols=TABLE_COLS[kind]) for kind, lol in lols.items()}
    finally:
        local.depth = outer_depth


def dump(path: str | None = None) -> str:
    """
    Write the tables so far to one Markdown file, and return the path.

    Args:
        path: The file. Without it, `daffodil_profile_<stage>_<host>_<pid>.md` in the
            current directory.

    Returns:
        The path written.
    """
    from daffodil.daf import Daf

    if not path:
        path = default_file_name()
    dodaf = tables()
    local = _local
    outer_depth = getattr(local, 'depth', 0)
    local.depth = 1
    try:
        text = Daf.dodaf_to_md(dodaf)
    finally:
        local.depth = outer_depth
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(text)
    return path


def default_file_name(data_dir: str = '.') -> str:
    """ `daffodil_profile_<stage>_<host>_<pid>.md` in data_dir. """
    stage = _stage or 'run'
    return os.path.join(data_dir, f"{FILE_PREFIX}{stage}_{socket.gethostname()}_{os.getpid()}.md")


def load(path: str) -> Dict[str, Any]:
    """
    Read the tables that `dump()` wrote.

    Args:
        path: The file.

    Returns:
        A dict of Daf tables, with numbers converted back from text.
    """
    from daffodil.daf import Daf

    with open(path, encoding='utf-8') as fh:
        dodaf = Daf.dodaf_from_md(fh.read())
    for kind, daf in dodaf.items():
        if kind in TABLE_COLS:
            dtypes = {col: (str if col in TEXT_COLS else float if col in FLOAT_COLS else int) for col in daf.columns()}
            daf.apply_dtypes(dtypes=dtypes)
    return dodaf


def _merge_da(row_da: T_da, reduction_da: T_da, *, cols: Any = None, max_cols: T_ls = MAX_COLS, **kwargs: Any) -> T_da:
    """ A reduction for `groupby_cols_reduce()`: add up each column, but keep the largest of max_cols. """
    for col in cols:
        value = row_da[col]
        if col in max_cols:
            if value > reduction_da[col]:
                reduction_da[col] = value
        else:
            reduction_da[col] += value
    return reduction_da


def combine(runs: List[Dict[str, Any]], by_stage: bool = True) -> Dict[str, Any]:
    """
    Add up the tables of several runs.

    Args:
        runs: The tables of each run, as from `tables()` or `load()`.
        by_stage: If True, keep a row for each stage. If False, add the stages together
            and set `stage` to 'all'.

    Returns:
        A dict of Daf tables of the same form.
    """
    from daffodil.daf import Daf

    combined: Dict[str, Any] = {}
    for kind, cols in TABLE_COLS.items():
        all_daf = Daf(cols=cols)
        for run in runs:
            if kind in run and run[kind]:
                all_daf.append(run[kind])
        if not by_stage and all_daf:
            all_daf[:, 'stage'] = 'all'
        if kind == 'info' or not all_daf:
            combined[kind] = all_daf
            continue
        key_cols = KEY_COLS[kind]
        value_cols = [col for col in cols if col not in key_cols]
        combined[kind] = all_daf.groupby_cols_reduce(key_cols, _merge_da, reduce_cols=value_cols)
    return combined


# ---- the report

def report(dodaf: Dict[str, Any] | None = None, path: str | None = None, top: int = 30) -> str:
    """
    Return a readable report as Markdown, and write it to a file if `path` is given.

    Args:
        dodaf: The tables to report, as from `tables()`, `load()` or `combine()`. Without
            it, the totals of this process so far.
        path: Write the report to this file too. `{pid}` is replaced with the process id.
        top: The most lines to show in each section.

    Returns:
        The report as Markdown text.
    """
    if dodaf is None:
        dodaf = tables()

    local = _local
    outer_depth = getattr(local, 'depth', 0)
    local.depth = 1                         # the report's own Daf calls are not counted
    try:
        text = _build_report(dodaf, top)
    finally:
        local.depth = outer_depth

    if path:
        with open(path.replace('{pid}', str(os.getpid())), 'w', encoding='utf-8') as fh:
            fh.write(text)
    return text


def _band_median(bands: T_la) -> str:
    """ The size band that holds the median, from counts per band. """
    total = sum(bands)
    if not total:
        return ''
    running = 0
    for idx, num in enumerate(bands):
        running += num
        if running * 2 >= total:
            return BAND_NAMES[idx]
    return BAND_NAMES[-1]


def _build_report(dodaf: Dict[str, Any], top: int) -> str:
    from daffodil.daf import Daf

    info = dodaf['info']
    stages = sorted({row['stage'] for row in info.iter_dict()}) if info else []
    parts: T_ls = ["# Daffodil profile\n"]

    if info:
        run_lol = [[row['stage'] or '(none)', row['host'], row['pid'], row['started'], row['seconds'],
                    row['python'], row['daffodil']] for row in info.iter_dict()]
        parts.append("## Runs\n")
        parts.append(Daf(lol=run_lol[:top], cols=['Stage', 'Host', 'Process', 'Started', 'Seconds', 'Python', 'daffodil'])
                     .to_md(just='<<><><<'))
        if len(run_lol) > top:
            parts.append(f"{len(run_lol) - top:,} more runs are not shown.\n")

    if len(stages) > 1:
        overall = combine([dodaf], by_stage=False)
        parts.append("\n# All stages\n")
        parts.extend(_report_sections(overall, top))
        for stage in stages:
            one = {kind: (daf.select_where(lambda row, s=stage: row['stage'] == s) if daf else daf)
                   for kind, daf in dodaf.items()}
            parts.append(f"\n# Stage {stage}\n")
            parts.extend(_report_sections(one, top))
    else:
        parts.extend(_report_sections(dodaf, top))

    return '\n'.join(parts) + '\n'


def _report_sections(dodaf: Dict[str, Any], top: int) -> T_ls:
    from daffodil.daf import Daf

    parts: T_ls = []
    tables_daf, ops_daf, methods_daf, sites_daf = dodaf['tables'], dodaf['ops'], dodaf['methods'], dodaf['sites']

    total_tables = sum(tables_daf.col_to_la('tables')) if tables_daf else 0
    total_calls  = sum(methods_daf.col_to_la('calls')) if methods_daf else 0
    parts.append(f"Tables: {total_tables:,}. Counted calls: {total_calls:,}.\n")

    # tables by creation line
    ops_by_group: Dict[Tuple[str, str], List[Tuple[str, int]]] = {}
    for row in ops_daf.iter_dict() if ops_daf else []:
        ops_by_group.setdefault((row['created_at'], row['how']), []).append((row['method'], row['calls']))
    rows = []
    for row in tables_daf.iter_dict() if tables_daf else []:
        group_ops = sorted(ops_by_group.get((row['created_at'], row['how']), []), key=lambda op: -op[1])
        num_ops = sum(num for _, num in group_ops)
        main_ops = '; '.join(f"{op} {num:,}" for op, num in group_ops[:4])
        row_bands = [row[col] for col in ROW_BAND_COLS]
        mean_rows = round(row['rows_sum'] / row['tables']) if row['tables'] else 0
        rows.append((num_ops, [row['created_at'], row['how'], row['tables'], _band_median(row_bands), mean_rows,
                               row['max_rows'], row['max_cols'], row['keyed'], main_ops]))
    rows.sort(key=lambda r: (-r[0], -r[1][2]))
    parts.append("## Tables by the line that created them\n")
    parts.append("Rows and columns are the most each table had. Keyed is how many had a keyfield.\n")
    parts.append(Daf(lol=[r[1] for r in rows[:top]],
                     cols=['Created at', 'How', 'Tables', 'Rows median', 'Rows mean', 'Rows max', 'Cols max',
                           'Keyed', 'Main operations']).to_md(max_text_len=100, just='<<>>>>>><')
                 if rows else "No tables were seen.\n")

    # size distribution
    row_dist = [sum(tables_daf.col_to_la(col)) for col in ROW_BAND_COLS] if tables_daf else [0] * NUM_BANDS
    col_dist = [sum(tables_daf.col_to_la(col)) for col in COL_BAND_COLS] if tables_daf else [0] * NUM_BANDS
    parts.append("\n## Table sizes\n")
    parts.append("Each table counts once, at the most rows and columns it had.\n")
    parts.append(Daf(lol=[[BAND_NAMES[i], row_dist[i], col_dist[i]] for i in range(NUM_BANDS)],
                     cols=['Size', 'Tables by rows', 'Tables by columns']).to_md(just='<>>'))

    # methods
    method_lol: T_lola = []
    for row in methods_daf.iter_dict() if methods_daf else []:
        calls, seconds = row['calls'], row['seconds']
        method_lol.append([row['method'], calls, round(seconds * 1000, 1), round(seconds / calls * 1e6, 2) if calls else 0]
                    + [row[col] for col in ROW_BAND_COLS] + [row['max_rows']])
    method_lol.sort(key=lambda r: -r[2])
    parts.append("\n## Methods\n")
    parts.append("Outer calls only. Rows are the table's rows at the call, or the rows made by a constructor.\n")
    parts.append(Daf(lol=method_lol[:top], cols=['Method', 'Calls', 'Total ms', 'Mean µs'] + [f"rows {b}" for b in BAND_NAMES]
                     + ['Rows max']).to_md(max_text_len=100, just='<' + '>' * (NUM_BANDS + 4))
                 if method_lol else "No calls were counted.\n")

    # call sites
    site_lol: T_lola = [[row['site'], row['method'], row['calls'], round(row['seconds'] * 1000, 1)]
            for row in (sites_daf.iter_dict() if sites_daf else [])]
    site_lol.sort(key=lambda r: -r[3])
    parts.append("\n## Busiest call sites\n")
    parts.append(Daf(lol=site_lol[:top], cols=['Call site', 'Method', 'Calls', 'Total ms']).to_md(max_text_len=100, just='<<>>')
                 if site_lol else "No calls were counted.\n")
    return parts


def _at_exit() -> None:
    """ Write the tables and the report when the program ends, as set by `start()`. """
    if not (_data_dir or _report_path or _print_report):
        return
    try:
        dodaf = tables()
        if _data_dir:
            os.makedirs(_data_dir, exist_ok=True)
            dump(default_file_name(_data_dir))
        if _report_path or _print_report:
            text = report(dodaf, _report_path)
            if _print_report:
                print(text, file=sys.stderr)
    except Exception as err:                    # a failed report must not hide the program's own exit
        print(f"daffodil profile: writing the profile failed: {err!r}", file=sys.stderr)


def start_from_env(cls: Type) -> None:
    """
    Start profiling if the environment variable DAFFODIL_PROFILE is set and not '0'.

    DAFFODIL_PROFILE_STAGE names the stage. DAFFODIL_PROFILE_DIR is where the tables are
    written at exit, the current directory by default. DAFFODIL_PROFILE_FILE also writes the
    report to a file. The report is printed to stderr.
    """
    flag = os.environ.get('DAFFODIL_PROFILE', '')
    if flag and flag != '0':
        start(cls,
              stage         = os.environ.get('DAFFODIL_PROFILE_STAGE', ''),
              data_dir      = os.environ.get('DAFFODIL_PROFILE_DIR', '.'),
              report_path   = os.environ.get('DAFFODIL_PROFILE_FILE') or None,
              print_report  = True)


def _paths_from_args(args: T_ls) -> T_ls:
    """ The files named, and the profile files in the directories named. """
    paths: T_ls = []
    for arg in args:
        if os.path.isdir(arg):
            paths.extend(sorted(glob.glob(os.path.join(arg, f"{FILE_PREFIX}*.md"))))
        else:
            paths.append(arg)
    return paths


def main(argv: T_ls | None = None) -> int:
    """
    The command line: `python -m daffodil.lib.daf_profile combine DIR_OR_FILES... [-o report.md] [--data combined.md]`.
    """
    import argparse
    parser = argparse.ArgumentParser(prog='python -m daffodil.lib.daf_profile',
                                     description='Combine the profile files of several runs into one report.')
    sub = parser.add_subparsers(dest='command', required=True)
    comb = sub.add_parser('combine', help='combine profile files and write a report')
    comb.add_argument('paths', nargs='+', help='profile files, or directories that hold them')
    comb.add_argument('-o', '--output', help='write the report to this file, instead of printing it')
    comb.add_argument('--data', help='also write the combined tables to this file')
    comb.add_argument('--top', type=int, default=30, help='the most lines in each section')
    args = parser.parse_args(argv)

    from daffodil.daf import Daf

    paths = _paths_from_args(args.paths)
    if not paths:
        print("No profile files were found.", file=sys.stderr)
        return 1
    combined = combine([load(path) for path in paths])
    if args.data:
        with open(args.data, 'w', encoding='utf-8') as fh:
            fh.write(Daf.dodaf_to_md(combined))
    text = report(combined, args.output, top=args.top)
    if not args.output:
        print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
