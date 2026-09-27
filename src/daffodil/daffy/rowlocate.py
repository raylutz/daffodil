# rowlocate.py -- unambiguous row addressing shared by get/set/delete-row.
#
# Two ways to name a row, matching the design doc's requirement ("make row selection
# unambiguous"): --where COL=VALUE (one or more columns, via Daf.select_by_dict()) or
# --pos N (explicit 0-based row position). Never both, never neither.

from typing import Dict, Optional, Tuple

from daffodil.daf import Daf


class RowLocateError(ValueError):
    """ Raised for anything that would make a mutation ambiguous or impossible:
        no locator given, both given, zero matches, or more than one match.
    """


def locate_row(daf: Daf, where: Optional[Dict[str, str]], pos: Optional[int]) -> Tuple[int, Dict[str, str]]:
    """ Returns (irow, row_dict_before_change). Raises RowLocateError on anything ambiguous. """
    if where and pos is not None:
        raise RowLocateError("Specify --where or --pos, not both.")
    if not where and pos is None:
        raise RowLocateError("Specify --where COL=VALUE or --pos N to identify the row.")

    if pos is not None:
        if pos < 0 or pos >= len(daf):
            raise RowLocateError(f"--pos {pos} is out of range (table has {len(daf)} rows).")
        return pos, dict(zip(daf.columns(), daf.lol[pos]))

    assert where is not None
    unknown_cols = [col for col in where if col not in daf.columns()]
    if unknown_cols:
        raise RowLocateError(f"unknown column(s) in --where: {', '.join(unknown_cols)}")

    matches = [irow for irow, row in enumerate(daf.lol)
               if all(row[daf.hd[col]] == val for col, val in where.items())]

    if not matches:
        raise RowLocateError(f"--where {where} matched no rows.")
    if len(matches) > 1:
        raise RowLocateError(f"--where {where} matched {len(matches)} rows, not unique: row positions {matches}.")

    irow = matches[0]
    return irow, dict(zip(daf.columns(), daf.lol[irow]))
