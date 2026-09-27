# cli.py -- daffy command-line entry point.
#
# v1 (Ray, 2026-09-27: "we can get our feet wet and see what is needed here"): inspect and select.
# v2: show, get, set, add-row, delete-row, diff. export-ods and column ops (add/rename/delete) and
# a dedicated width-metadata command still not done -- see NOT_YET_DONE at the bottom of this file.

from __future__ import annotations

import argparse
import json
import pprint
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from daffodil.daf import Daf

from daffodil.daffy import diff as diff_mod
from daffodil.daffy import profile as profile_mod
from daffodil.daffy import rowlocate
from daffodil.daffy import sniff as sniff_mod
from daffodil.daffy import writeback


def _load_daf(csv_path: str) -> Daf:
    return Daf.from_csv(csv_path)


def _parse_filter(filter_str: Optional[str]) -> Dict[str, str]:
    """ "col=val,col2=val2" -> {'col': 'val', 'col2': 'val2'}.
        Deliberately equality-only for v1 -- maps directly onto Daf.select_by_dict(), a real
        existing daffodil primitive, rather than building a new expression parser. No eval of
        arbitrary text: an unparseable term is a hard error, not silently ignored.
    """
    if not filter_str:
        return {}
    selector: Dict[str, str] = {}
    for term in filter_str.split(','):
        term = term.strip()
        if not term:
            continue
        if '=' not in term:
            raise ValueError(f"--filter term {term!r} is not of the form col=value")
        col, _, val = term.partition('=')
        selector[col.strip()] = val.strip()
    return selector


def _emit(daf: Daf, fmt: str) -> None:
    if fmt == 'md':
        print(daf.to_md())
    elif fmt == 'json':
        print(json.dumps(daf.to_lod(), indent=2))
    elif fmt == 'pyon':
        print(pprint.pformat(daf.to_lod()))
    else:
        raise ValueError(f"Unknown format: {fmt}")


def cmd_inspect(args: argparse.Namespace) -> int:
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"error: {csv_path} does not exist", file=sys.stderr)
        return 2

    csv_profile = sniff_mod.sniff_csv(csv_path)
    prof = profile_mod.load_profile(csv_path, explicit_path=args.profile)
    daf = _load_daf(str(csv_path))

    report: Dict[str, Any] = {
        'path': str(csv_path),
        'columns': daf.columns(),
        'num_rows': len(daf),
        'line_terminator': sniff_mod.line_terminator_label(csv_profile.line_terminator),
        'mixed_line_endings': csv_profile.mixed_line_endings,
        'delimiter': csv_profile.delimiter,
        'quotechar': csv_profile.quotechar,
        'encoding': csv_profile.encoding,
        'profile_path': str(profile_mod.profile_path_for(csv_path)) if prof else None,
        'keyfield': prof.get('keyfield'),
        'dtypes': prof.get('dtypes'),
    }

    if args.format == 'md':
        for key, val in report.items():
            print(f"{key}: {val}")
    elif args.format == 'json':
        print(json.dumps(report, indent=2))
    elif args.format == 'pyon':
        print(pprint.pformat(report))

    return 0


def cmd_select(args: argparse.Namespace) -> int:
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"error: {csv_path} does not exist", file=sys.stderr)
        return 2

    daf = _load_daf(str(csv_path))

    try:
        selector = _parse_filter(args.filter)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if selector:
        missing = [col for col in selector if col not in daf.columns()]
        if missing:
            print(f"error: unknown column(s) in --filter: {', '.join(missing)}", file=sys.stderr)
            return 2
        daf = daf.select_by_dict(selector)

    if args.cols:
        requested_cols: List[str] = [c.strip() for c in args.cols.split(',') if c.strip()]
        missing = [col for col in requested_cols if col not in daf.columns()]
        if missing:
            print(f"error: unknown column(s) in --cols: {', '.join(missing)}", file=sys.stderr)
            return 2
        daf = daf.select_cols(requested_cols)

    if args.limit is not None:
        daf = daf[:args.limit]

    _emit(daf, args.format)
    return 0


def _parse_where(where_str: Optional[str]) -> Optional[Dict[str, str]]:
    if not where_str:
        return None
    return _parse_filter(where_str)


def cmd_show(args: argparse.Namespace) -> int:
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"error: {csv_path} does not exist", file=sys.stderr)
        return 2

    daf = _load_daf(str(csv_path))
    start = args.offset or 0
    end = start + args.limit if args.limit is not None else None
    daf = daf[start:end]

    _emit(daf, args.format)
    return 0


def cmd_get(args: argparse.Namespace) -> int:
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"error: {csv_path} does not exist", file=sys.stderr)
        return 2

    daf = _load_daf(str(csv_path))

    try:
        where = _parse_where(args.where)
        irow, row = rowlocate.locate_row(daf, where, args.pos)
    except rowlocate.RowLocateError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.col:
        if args.col not in daf.columns():
            print(f"error: unknown column: {args.col}", file=sys.stderr)
            return 2
        value = row[args.col]
        if args.format == 'md':
            print(value)
        elif args.format == 'json':
            print(json.dumps(value))
        elif args.format == 'pyon':
            print(pprint.pformat(value))
        return 0

    if args.format == 'md':
        for key, val in row.items():
            print(f"{key}: {val}")
    elif args.format == 'json':
        print(json.dumps(row, indent=2))
    elif args.format == 'pyon':
        print(pprint.pformat(row))
    return 0


def cmd_set(args: argparse.Namespace) -> int:
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"error: {csv_path} does not exist", file=sys.stderr)
        return 2

    csv_profile = sniff_mod.sniff_csv(csv_path)
    daf = _load_daf(str(csv_path))

    try:
        where = _parse_where(args.where)
        irow, row = rowlocate.locate_row(daf, where, args.pos)
    except rowlocate.RowLocateError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.col not in daf.columns():
        print(f"error: unknown column: {args.col}", file=sys.stderr)
        return 2

    old_value = row[args.col]
    if args.expect_old is not None and old_value != args.expect_old:
        print(f"error: expected old value {args.expect_old!r} for column {args.col!r}, "
              f"found {old_value!r} -- refusing to overwrite (row may have changed since read).",
              file=sys.stderr)
        return 2

    daf[irow, args.col] = args.value
    writeback.atomic_write_csv(daf, csv_path, csv_profile)

    print(f"changed: row {irow} ({args.where or f'--pos {args.pos}'}), column {args.col!r}: "
          f"{old_value!r} -> {args.value!r}")
    return 0


def cmd_add_row(args: argparse.Namespace) -> int:
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"error: {csv_path} does not exist", file=sys.stderr)
        return 2

    csv_profile = sniff_mod.sniff_csv(csv_path)
    daf = _load_daf(str(csv_path))

    try:
        values = _parse_filter(args.values)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    unknown = [col for col in values if col not in daf.columns()]
    if unknown:
        print(f"error: unknown column(s) in --values: {', '.join(unknown)}", file=sys.stderr)
        return 2

    new_row = {col: values.get(col, '') for col in daf.columns()}
    daf.append(new_row)
    writeback.atomic_write_csv(daf, csv_path, csv_profile)

    print(f"added row {len(daf) - 1}: {new_row}")
    return 0


def cmd_delete_row(args: argparse.Namespace) -> int:
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"error: {csv_path} does not exist", file=sys.stderr)
        return 2

    csv_profile = sniff_mod.sniff_csv(csv_path)
    daf = _load_daf(str(csv_path))

    try:
        where = _parse_where(args.where)
        irow, row = rowlocate.locate_row(daf, where, args.pos)
    except rowlocate.RowLocateError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    daf = daf.select_irows([irow], invert=True)
    writeback.atomic_write_csv(daf, csv_path, csv_profile)

    print(f"deleted row {irow}: {row}")
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    old_path = Path(args.old_csv_path)
    new_path = Path(args.new_csv_path)
    for p in (old_path, new_path):
        if not p.exists():
            print(f"error: {p} does not exist", file=sys.stderr)
            return 2

    old_daf = _load_daf(str(old_path))
    new_daf = _load_daf(str(new_path))

    try:
        result = diff_mod.diff_daf(old_daf, new_daf, key=args.key)
    except diff_mod.DiffError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.format == 'md':
        print(diff_mod.diff_result_to_md(result))
    elif args.format == 'json':
        print(json.dumps(diff_mod.diff_result_to_dict(result), indent=2))
    elif args.format == 'pyon':
        print(pprint.pformat(diff_mod.diff_result_to_dict(result)))

    if result.duplicate_keys:
        return 2
    return 0 if result.is_equal else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='daffy', description='Inspect, edit, compare, and export CSV tables (built on Daffodil).')
    subparsers = parser.add_subparsers(dest='command', required=True)

    p_inspect = subparsers.add_parser('inspect', help='Report columns, row count, CSV dialect/encoding/line-ending, and sidecar profile.')
    p_inspect.add_argument('csv_path')
    p_inspect.add_argument('--profile', help='Explicit profile sidecar path (overrides discovery of <csv_path>.profile.json).')
    p_inspect.add_argument('--format', choices=['md', 'json', 'pyon'], default='md')
    p_inspect.set_defaults(func=cmd_inspect)

    p_select = subparsers.add_parser('select', help='Choose columns and filter rows by exact-match equality.')
    p_select.add_argument('csv_path')
    p_select.add_argument('--filter', help='Equality filter, e.g. "style_num=103549,is_bmd=0". Comma-separated AND of col=value terms.')
    p_select.add_argument('--cols', help='Comma-separated list of columns to keep. Output keeps the '
                           "source CSV's own column order (Daf.select_cols() is a subset, not a reorder).")
    p_select.add_argument('--limit', type=int, help='Only emit the first N rows.')
    p_select.add_argument('--format', choices=['md', 'json', 'pyon'], default='md')
    p_select.set_defaults(func=cmd_select)

    p_show = subparsers.add_parser('show', help='Display a whole table or a bounded window of it.')
    p_show.add_argument('csv_path')
    p_show.add_argument('--offset', type=int, help='Skip this many rows before showing.')
    p_show.add_argument('--limit', type=int, help='Show at most this many rows.')
    p_show.add_argument('--format', choices=['md', 'json', 'pyon'], default='md')
    p_show.set_defaults(func=cmd_show)

    p_get = subparsers.add_parser('get', help='Retrieve a row or a single cell by --where or --pos.')
    p_get.add_argument('csv_path')
    p_get.add_argument('--where', help='Equality selector, e.g. "id=001". Must match exactly one row.')
    p_get.add_argument('--pos', type=int, help='0-based row position (alternative to --where).')
    p_get.add_argument('--col', help='If given, print just this column\'s value instead of the whole row.')
    p_get.add_argument('--format', choices=['md', 'json', 'pyon'], default='md')
    p_get.set_defaults(func=cmd_get)

    p_set = subparsers.add_parser('set', help='Change a single cell. Atomic write; row selection must be unambiguous.')
    p_set.add_argument('csv_path')
    p_set.add_argument('--where', help='Equality selector, e.g. "id=001". Must match exactly one row.')
    p_set.add_argument('--pos', type=int, help='0-based row position (alternative to --where).')
    p_set.add_argument('--col', required=True, help='Column to change.')
    p_set.add_argument('--value', required=True, help='New value.')
    p_set.add_argument('--expect-old', help='Refuse to write unless the current value equals this '
                        '(guards against overwriting a change made since this was last read).')
    p_set.set_defaults(func=cmd_set)

    p_add_row = subparsers.add_parser('add-row', help='Append a new row. Columns not given default to empty.')
    p_add_row.add_argument('csv_path')
    p_add_row.add_argument('--values', required=True, help='"col1=val1,col2=val2" for the new row.')
    p_add_row.set_defaults(func=cmd_add_row)

    p_delete_row = subparsers.add_parser('delete-row', help='Delete a row. Atomic write; row selection must be unambiguous.')
    p_delete_row.add_argument('csv_path')
    p_delete_row.add_argument('--where', help='Equality selector, e.g. "id=001". Must match exactly one row.')
    p_delete_row.add_argument('--pos', type=int, help='0-based row position (alternative to --where).')
    p_delete_row.set_defaults(func=cmd_delete_row)

    p_diff = subparsers.add_parser('diff', help='Compare two CSV files. Exit 0=equal, 1=different, 2=error.')
    p_diff.add_argument('old_csv_path')
    p_diff.add_argument('new_csv_path')
    p_diff.add_argument('--key', help='Match rows by this column instead of row position. '
                         'Without it, rows are compared by position (stated in the output).')
    p_diff.add_argument('--format', choices=['md', 'json', 'pyon'], default='md')
    p_diff.set_defaults(func=cmd_diff)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())


# NOT_YET_DONE (deferred, not started):
# - export-ods (needs odfdo, already in pyproject.toml's optional [ods] extra, not wired to any code)
# - column operations: add-col (no direct Daf primitive found -- would rebuild via Daf(cols=..., lol=...)),
#   delete-col (Daf.drop_cols() exists, unused so far), rename-col (Daf.rename_cols() exists, unused so far)
# - a dedicated width-metadata command ("daffy width set/show") -- widths currently only reachable
#   by hand-editing the profile.json sidecar directly
