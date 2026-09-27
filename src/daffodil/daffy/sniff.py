# sniff.py -- detect a CSV file's real line ending and dialect from raw bytes,
# before Daf.from_csv() (which opens in text mode and silently normalizes line endings away).

import csv
from pathlib import Path
from typing import NamedTuple


class CsvProfile(NamedTuple):
    line_terminator: str          # '\r\n', '\n', or '\r' -- what the file actually uses
    delimiter: str
    quotechar: str
    encoding: str                 # currently always 'utf-8' -- matches Daf.from_csv()'s own assumption
    mixed_line_endings: bool      # True if more than one terminator style was found


def sniff_csv(path: str | Path, sample_bytes: int = 65536) -> CsvProfile:
    """ Read a raw sample of the file in binary mode and detect its line ending and dialect.
        Must run before any Daf.from_csv()/open(path, 'r') call -- text-mode reading in Python
        normalizes \\r\\n, \\r, and \\n all down to \\n (universal newlines), so the original
        terminator is unrecoverable once daffodil has already read the file.
    """
    path = Path(path)
    raw = path.read_bytes()[:sample_bytes]

    has_crlf = b'\r\n' in raw
    # count lone \r (not part of \r\n) and lone \n (not preceded by \r)
    has_lone_cr = b'\r' in raw.replace(b'\r\n', b'')
    has_lone_lf = b'\n' in raw.replace(b'\r\n', b'')

    styles_found = sum([has_crlf, has_lone_cr, has_lone_lf])
    mixed = styles_found > 1

    if has_crlf:
        line_terminator = '\r\n'
    elif has_lone_cr and not has_lone_lf:
        line_terminator = '\r'
    else:
        line_terminator = '\n'

    text_sample = raw.decode('utf-8', errors='replace')
    try:
        dialect = csv.Sniffer().sniff(text_sample, delimiters=',;\t|')
        delimiter = dialect.delimiter
        quotechar = dialect.quotechar
    except csv.Error:
        # Sniffer needs at least two rows/columns to work with -- fall back to the common case
        # rather than guessing further.
        delimiter = ','
        quotechar = '"'

    return CsvProfile(
        line_terminator=line_terminator,
        delimiter=delimiter,
        quotechar=quotechar,
        encoding='utf-8',
        mixed_line_endings=mixed,
        )


def line_terminator_label(line_terminator: str) -> str:
    return {'\r\n': 'CRLF', '\n': 'LF', '\r': 'CR'}.get(line_terminator, repr(line_terminator))
