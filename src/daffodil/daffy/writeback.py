# writeback.py -- atomic, line-ending-preserving CSV write for daffy's mutating commands.

import os
from pathlib import Path

from daffodil.daf import Daf

from daffodil.daffy.sniff import CsvProfile


def atomic_write_csv(daf: Daf, csv_path: str | Path, csv_profile: CsvProfile) -> None:
    """ Write daf to csv_path via a temp file in the same directory + os.replace() (atomic on
        POSIX), preserving the source's own line ending. Never partially overwrites csv_path --
        either the write fully succeeds and replaces it, or csv_path is untouched.
    """
    csv_path = Path(csv_path)
    tmp_path = csv_path.with_name(csv_path.name + '.daffy.tmp')
    daf.to_csv_file(str(tmp_path), line_terminator=csv_profile.line_terminator)
    os.replace(tmp_path, csv_path)
