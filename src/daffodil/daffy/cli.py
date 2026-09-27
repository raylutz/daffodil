# cli.py -- daffy command-line entry point.
#
# v1 scope (Ray, 2026-09-27: "we can get our feet wet and see what is needed here"): inspect and
# select only, the two operations actually used ad hoc against real CSV files this session,
# instead of a one-off python3 -c script each time. Other commands (get/set/add-row/delete-row/
# diff/export-ods) come once this shape has been exercised against real data.

from __future__ import annotations

import argparse
import json
import pprint
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from daffodil.daf import Daf

from daffodil.daffy import profile as profile_mod
from daffodil.daffy import sniff as sniff_mod


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='daffy', description='Inspect, edit, compare, and export CSV tables (built on Daffodil).')
    subparsers = parser.add_subparsers(dest='command', required=True)

    p_inspect = subparsers.add_parser('inspect', help='Report columns, row count, CSV dialect/encoding/line-ending, and sidecar profile.')
    p_inspect.add_argument('csv_path')
    p_inspect.add_argument('--profile', help='Explicit profile sidecar path (overrides discovery of <csv_path>,profile.json).')
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

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
