# widths_edit.py -- open a CSV in a real LibreOffice Calc window via the UNO API, apply known
# column widths, and read the (possibly user-adjusted) widths back into the profile sidecar when
# the document is closed.
#
# NOT TESTED END TO END (Ray, 2026-09-27): this remote dev box has no LibreOffice/UNO at all --
# confirmed via `python3 -c "import uno"` (ModuleNotFoundError) and `which soffice` (not found).
# Built against the documented UNO API; needs verification on a real machine with LibreOffice.
#
# Motivating case: arg_specs.csv's description column auto-sizes to 17+ inches in LibreOffice by
# default, unreadable. This command opens the file with sane widths already applied instead.
#
# Real risk (Ray's own words, other thread): UNO integration is fragile -- version mismatches
# between the system Python and LibreOffice's bundled Python are a known trap. If `import uno`
# fails in the venv's own Python, LibreOffice's bundled `python3` (usually
# /usr/lib/libreoffice/program/python3) may need to run this script instead -- not handled here;
# surfaced as a clear error pointing at the likely fix, not a bare ImportError.
#
# Units: LibreOffice's Column.Width property is in 1/100 mm. Stored under 'widths_lo_mm100' in
# the profile, a distinct key from any future markdown-width concept -- per the original design
# doc's own caution ("do not assume a Markdown table width and an ODS width have identical
# units"), never conflate the two.

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from daffodil.daffy import profile as profile_mod

UNO_SOCKET_PORT = 2002
UNO_CONNECT_TIMEOUT_SECS = 15
POLL_INTERVAL_SECS = 1.0


class WidthsEditError(RuntimeError):
    pass


def _import_uno():
    try:
        import uno  # noqa: F401
        return uno
    except ImportError as e:
        raise WidthsEditError(
            "Cannot import the 'uno' module. This must run under the Python that LibreOffice "
            "itself ships with, not necessarily this venv's Python -- try running this command "
            "via LibreOffice's own interpreter, commonly /usr/lib/libreoffice/program/python3 "
            "on Ubuntu (`sudo apt install python3-uno` may also be needed)."
            ) from e


def _connect(uno_mod, timeout_secs: int = UNO_CONNECT_TIMEOUT_SECS):
    """ Connect to a running (or freshly launched) soffice instance listening on UNO_SOCKET_PORT.
        Launches one, visibly (not --headless), if none is already listening -- this command is
        for a person to look at and adjust, not for headless/agent use.
    """
    local_context = uno_mod.getComponentContext()
    resolver = local_context.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", local_context)

    connect_str = f"uno:socket,host=localhost,port={UNO_SOCKET_PORT};urp;StarOffice.ComponentContext"

    deadline = time.time() + timeout_secs
    proc = None
    while time.time() < deadline:
        try:
            ctx = resolver.resolve(connect_str)
            return ctx, proc
        except Exception:
            if proc is None:
                proc = subprocess.Popen([
                    'soffice',
                    f'--accept=socket,host=localhost,port={UNO_SOCKET_PORT};urp;',
                    '--norestore',
                    ])
            time.sleep(0.5)

    raise WidthsEditError(f"Could not connect to LibreOffice via UNO on port {UNO_SOCKET_PORT} "
                           f"within {timeout_secs}s.")


def _text_import_filter_options(num_cols: int) -> str:
    """ Force every column to LibreOffice's CSV-import "Text" column format (code 2), so opening
        a CSV through daffy never lets LibreOffice auto-detect a column as numeric and mangle
        leading zeros / IDs -- the same string-preservation principle as the rest of daffy.
        FilterOptions format: field_sep_code,text_delim_code,charset,lang,
        col1_format/col2_format/... -- '44' = comma, '34' = double-quote, '76' = UTF-8.
    """
    col_formats = '/'.join(f'{i + 1}/2' for i in range(num_cols))
    return f'44,34,76,1,{col_formats},true'


def edit_widths(csv_path: str | Path, explicit_profile_path: Optional[str | Path] = None) -> Dict[str, Any]:
    csv_path = Path(csv_path).resolve()
    uno_mod = _import_uno()
    from com.sun.star.beans import PropertyValue  # type: ignore[import-not-found]

    prof = profile_mod.load_profile(csv_path, explicit_path=explicit_profile_path)
    existing_widths = prof.get('widths_lo_mm100', {})

    # Keyed by the CSV's own column name throughout, not a spreadsheet letter/position -- matches
    # every other daffy command (get/set/select all address columns by name), and survives a
    # later column reorder or insertion the way a positional letter wouldn't.
    with open(csv_path, encoding='utf-8') as f:
        header = f.readline().rstrip('\r\n').split(',')

    ctx, _proc = _connect(uno_mod)
    smgr = ctx.ServiceManager
    desktop = smgr.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)

    def make_prop(name: str, value) -> Any:
        p = PropertyValue()
        p.Name = name
        p.Value = value
        return p

    load_props = (
        make_prop('FilterName', 'Text - txt - csv (StarCalc)'),
        make_prop('FilterOptions', _text_import_filter_options(len(header))),
        )

    doc = desktop.loadComponentFromURL(
        f'file://{csv_path}', '_blank', 0, load_props)

    sheet = doc.Sheets.getByIndex(0)
    columns = sheet.Columns

    for colname, width in existing_widths.items():
        if colname in header:
            columns.getByIndex(header.index(colname)).Width = int(width)

    print(f"Opened {csv_path} in LibreOffice. Adjust column widths, then close the window to save them.")

    while True:
        time.sleep(POLL_INTERVAL_SECS)
        try:
            doc.getURL()
        except Exception:
            break  # document (or its proxy) is gone -- window was closed

    new_widths = {colname: columns.getByIndex(icol).Width for icol, colname in enumerate(header)}

    prof['widths_lo_mm100'] = new_widths
    saved_path = profile_mod.save_profile(csv_path, prof, explicit_path=explicit_profile_path)
    return {'profile_path': str(saved_path), 'widths_lo_mm100': new_widths}
