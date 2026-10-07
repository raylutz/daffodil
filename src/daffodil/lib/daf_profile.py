# daf_profile.py
"""
Profiling mode for Daf: how big the tables get, and what is done with them.

Turn it on with the environment variable `DAFFODIL_PROFILE=1`, or in code:

    from daffodil.lib import daf_profile
    daf_profile.start()
    ...
    print(daf_profile.report())

While it is on, every public method of Daf is wrapped. Only the outer call is counted:
when a Daf method calls another one inside, the inner call is not. When it is off,
nothing is wrapped and Daf runs at full speed.

It keeps running totals, not a log of every call:

- For each table: the line that created it and how, the most rows and columns it had,
  whether it had a keyfield, and a count of each method called on it. When the table is
  freed, this summary is added to the totals for its creation line.
- For each method: calls, total time, and the number of rows at each call, in size bands.
- For each call site: calls and total time, by method.

The report is a set of Markdown tables. With `DAFFODIL_PROFILE=1` it is printed to stderr
when the program ends, and written to a file. `DAFFODIL_PROFILE_FILE` sets the file name.
`{pid}` in the name is replaced with the process id.

Each counted call costs about 3 µs, and each call that makes a new table about 15 µs more.
A call that Daf makes inside another one costs about 0.4 µs. So the times of cheap calls
are inflated. Counts and sizes are exact. Calls in a child process that ends with `os._exit()`, as
multiprocessing workers do, are not reported.
"""

import atexit
import bisect
import datetime
import functools
import os
import platform
import random
import sys
import threading
import time
import weakref
from typing import List, Dict, Any, Tuple, Callable, Type, Optional    # noqa: F401

from daffodil.lib.daf_types import T_ls, T_li, T_lola


SIZE_BANDS      = (10, 100, 1_000, 10_000, 100_000, 1_000_000)
BAND_NAMES      = ['≤10', '≤100', '≤1k', '≤10k', '≤100k', '≤1M', '>1M']
SAMPLE_SIZE     = 10_000        # rows kept per creation line, to estimate the median
DEFAULT_FILE    = 'daffodil_profile_{pid}.md'


class _TableStats:
    """ The summary of one live table. """
    __slots__ = ('site', 'how', 'max_rows', 'max_cols', 'keyed', 'ops', 'ref')

    def __init__(self, site: Tuple[str, int], how: str) -> None:
        self.ref: Any   = None          # a weak reference to the table, whose callback retires it
        self.site       = site
        self.how        = how
        self.max_rows   = 0
        self.max_cols   = 0
        self.keyed      = False
        self.ops: Dict[str, int] = {}


class _GroupStats:
    """ The totals of the freed tables made at one line, in one way. """
    __slots__ = ('tables', 'rows_sample', 'max_rows', 'max_cols', 'keyed', 'ops')

    def __init__(self) -> None:
        self.tables     = 0
        self.rows_sample: T_li = []
        self.max_rows   = 0
        self.max_cols   = 0
        self.keyed      = 0
        self.ops: Dict[str, int] = {}


class _MethodStats:
    """ The totals of the outer calls to one method. """
    __slots__ = ('calls', 'seconds', 'bands', 'max_rows')

    def __init__(self) -> None:
        self.calls      = 0
        self.seconds    = 0.0
        self.bands      = [0] * (len(SIZE_BANDS) + 1)
        self.max_rows   = 0


_lock       = threading.Lock()
_local      = threading.local()             # .depth: 1 inside a counted call
_active     = False
_cls: Optional[Type] = None
_originals: Dict[str, Any] = {}
_live: Dict[int, _TableStats] = {}
_freed: List[_TableStats] = []                      # freed tables, not yet added to the totals
_groups: Dict[Tuple[Tuple[str, int], str], _GroupStats] = {}
_methods: Dict[str, _MethodStats] = {}
_sites: Dict[Tuple[Tuple[str, int], str], List[float]] = {}     # ((file, line), method) -> [calls, seconds]
_rows_bands = [0] * (len(SIZE_BANDS) + 1)           # freed tables by most rows
_cols_bands = [0] * (len(SIZE_BANDS) + 1)           # freed tables by most columns
_started_at = 0.0
_report_path: Optional[str] = None
_print_report = False
_atexit_registered = False


def band(n: int) -> int:
    """ The index of the size band that holds n. """
    return bisect.bisect_left(SIZE_BANDS, n)


def start(
        cls:            Optional[Type]  = None,     # the class to profile; Daf if not given
        report_path:    Optional[str]   = None,     # file for the report at exit; '{pid}' is replaced
        print_report:   bool            = False,    # also print the report to stderr at exit
        ) -> None:
    """
    Start profiling: wrap the public methods of Daf and begin counting.

    Calling it again while profiling is on does nothing. The totals are kept until
    `reset()`.

    Args:
        cls: The class to profile. Daf if not given.
        report_path: Write the report to this file when the program ends. `{pid}` in the
            name is replaced with the process id. None writes no file.
        print_report: Also print the report to stderr when the program ends.
    """
    global _active, _cls, _started_at, _report_path, _print_report, _atexit_registered

    if _active:
        return
    if cls is None:
        from daffodil.daf import Daf
        cls = Daf

    _cls            = cls
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
    Stop profiling and put the original methods back. The totals are kept for `report()`.
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
        _rows_bands[:] = [0] * len(_rows_bands)
        _cols_bands[:] = [0] * len(_cols_bands)
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


_short_paths: Dict[str, str] = {}


def _fmt_site(site: Tuple[str, int]) -> str:
    """ 'path:line', with the path relative to the current directory when inside it. """
    path, line = site
    short = _short_paths.get(path)
    if short is None:
        short = path
        try:
            rel = os.path.relpath(path)
            if not rel.startswith('..'):
                short = rel
        except ValueError:
            pass
        _short_paths[path] = short
    return f"{short}:{line}"


def _record(name: str, this: Any, rows_before: int, result: Any, seconds: float, frame: Any) -> None:
    """ Add one outer call to the totals. """
    site = (frame.f_code.co_filename, frame.f_lineno)     # formatted only in the report
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


def _table_stats(daf: Any, site: Tuple[str, int], how: str) -> _TableStats:
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
        _fold(_freed.pop(), _groups, _rows_bands, _cols_bands)


def _fold(ts: _TableStats, groups: Dict[Tuple[Tuple[str, int], str], _GroupStats], rows_bands: T_li, cols_bands: T_li) -> None:
    gs = groups.get((ts.site, ts.how))
    if gs is None:
        gs = groups[(ts.site, ts.how)] = _GroupStats()
    gs.tables += 1
    if len(gs.rows_sample) < SAMPLE_SIZE:
        gs.rows_sample.append(ts.max_rows)
    else:                                           # keep a fair sample of all the tables
        slot = random.randrange(gs.tables)
        if slot < SAMPLE_SIZE:
            gs.rows_sample[slot] = ts.max_rows
    gs.max_rows = max(gs.max_rows, ts.max_rows)
    gs.max_cols = max(gs.max_cols, ts.max_cols)
    gs.keyed   += ts.keyed
    gs_ops = gs.ops
    for op, num in ts.ops.items():
        gs_ops[op] = gs_ops.get(op, 0) + num
    rows_bands[band(ts.max_rows)] += 1
    cols_bands[band(ts.max_cols)] += 1


def _median(values: T_li) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def report(path: Optional[str] = None, top: int = 30) -> str:
    """
    Return the report as Markdown, and write it to a file if `path` is given.

    Tables that are still alive are included, as if they were freed now.

    Args:
        path: Write the report to this file too. `{pid}` is replaced with the process id.
        top: The most lines to show in each section.

    Returns:
        The report as Markdown text.
    """
    from daffodil.daf import Daf

    local = _local
    outer_depth = getattr(local, 'depth', 0)
    local.depth = 1                         # the report's own Daf calls are not counted
    try:
        text = _build_report(Daf, top)
    finally:
        local.depth = outer_depth

    if path:
        with open(path.replace('{pid}', str(os.getpid())), 'w', encoding='utf-8') as fh:
            fh.write(text)
    return text


def _build_report(Daf: Type, top: int) -> str:

    with _lock:
        _fold_freed()
        groups: Dict[Tuple[Tuple[str, int], str], _GroupStats] = {}
        for key, gs in _groups.items():
            copy_gs = _GroupStats()
            copy_gs.tables, copy_gs.rows_sample = gs.tables, list(gs.rows_sample)
            copy_gs.max_rows, copy_gs.max_cols, copy_gs.keyed = gs.max_rows, gs.max_cols, gs.keyed
            copy_gs.ops = dict(gs.ops)
            groups[key] = copy_gs
        rows_bands = list(_rows_bands)
        cols_bands = list(_cols_bands)
        for ts in list(_live.values()):
            _fold(ts, groups, rows_bands, cols_bands)
        methods = {name: (ms.calls, ms.seconds, list(ms.bands), ms.max_rows) for name, ms in _methods.items()}
        sites   = {key: tuple(val) for key, val in _sites.items()}

    try:
        from importlib.metadata import version
        daf_version = version('daffodil')
    except Exception:
        daf_version = '?'

    total_calls  = sum(m[0] for m in methods.values())
    total_tables = sum(g.tables for g in groups.values())
    elapsed      = time.time() - _started_at if _started_at else 0.0

    parts: T_ls = []
    parts.append("# Daffodil profile\n")
    parts.append(f"- Run: {datetime.datetime.now().isoformat(timespec='seconds')}, process {os.getpid()}, "
                 f"profiled for {elapsed:,.1f} s")
    parts.append(f"- Python {platform.python_version()}, daffodil {daf_version}")
    parts.append(f"- Tables: {total_tables:,}. Counted calls: {total_calls:,}.\n")

    # tables by creation line
    by_ops = sorted(groups.items(), key=lambda kv: (-sum(kv[1].ops.values()), -kv[1].tables))
    lol: T_lola = []
    for (site, how), gs in by_ops[:top]:
        main_ops = '; '.join(f"{op} {n:,}" for op, n in sorted(gs.ops.items(), key=lambda kv: -kv[1])[:4])
        lol.append([_fmt_site(site), how, gs.tables, _median(gs.rows_sample), gs.max_rows, gs.max_cols, gs.keyed, main_ops])
    parts.append("## Tables by the line that created them\n")
    parts.append("Rows and columns are the most each table had. Keyed is how many had a keyfield.\n")
    parts.append(Daf(lol=lol, cols=['Created at', 'How', 'Tables', 'Rows median', 'Rows max',
                                    'Cols max', 'Keyed', 'Main operations'])
                 .to_md(max_text_len=100, just='<<>>>>><') if lol else "No tables were seen.\n")

    # size distribution
    parts.append("\n## Table sizes\n")
    parts.append("Each table counts once, at the most rows and columns it had.\n")
    dist = [[BAND_NAMES[i], rows_bands[i], cols_bands[i]] for i in range(len(BAND_NAMES))]
    parts.append(Daf(lol=dist, cols=['Size', 'Tables by rows', 'Tables by columns']).to_md(just='<>>'))

    # methods
    by_time = sorted(methods.items(), key=lambda kv: -kv[1][1])
    lol = []
    for name, (calls, seconds, bands, max_rows) in by_time[:top]:
        lol.append([name, calls, round(seconds * 1000, 1), round(seconds / calls * 1e6, 2) if calls else 0]
                   + bands + [max_rows])
    parts.append("\n## Methods\n")
    parts.append("Outer calls only. Rows are the table's rows at the call, or the rows made by a constructor.\n")
    parts.append(Daf(lol=lol, cols=['Method', 'Calls', 'Total ms', 'Mean µs'] + [f"rows {b}" for b in BAND_NAMES]
                     + ['Rows max']).to_md(max_text_len=100, just='<' + '>' * (len(BAND_NAMES) + 4))
                 if lol else "No calls were counted.\n")

    # call sites
    by_site_time = sorted(sites.items(), key=lambda kv: -kv[1][1])
    lol = [[_fmt_site(site), name, int(calls), round(seconds * 1000, 1)] for (site, name), (calls, seconds) in by_site_time[:top]]
    parts.append("\n## Busiest call sites\n")
    parts.append(Daf(lol=lol, cols=['Call site', 'Method', 'Calls', 'Total ms']).to_md(max_text_len=100, just='<<>>')
                 if lol else "No calls were counted.\n")

    return '\n'.join(parts) + '\n'


def _at_exit() -> None:
    """ Write and print the report when the program ends, as set by `start()`. """
    if not (_report_path or _print_report):
        return
    try:
        text = report(_report_path)
        if _print_report:
            print(text, file=sys.stderr)
    except Exception as err:                    # a failed report must not hide the program's own exit
        print(f"daffodil profile: the report failed: {err!r}", file=sys.stderr)


def start_from_env(cls: Type) -> None:
    """ Start profiling if the environment variable DAFFODIL_PROFILE is set and not '0'. """
    flag = os.environ.get('DAFFODIL_PROFILE', '')
    if flag and flag != '0':
        start(cls,
              report_path   = os.environ.get('DAFFODIL_PROFILE_FILE', DEFAULT_FILE),
              print_report  = True)
