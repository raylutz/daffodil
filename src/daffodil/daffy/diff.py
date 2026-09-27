# diff.py -- structural CSV diff: columns added/removed/reordered, rows added/removed/changed.
#
# No reusable diff exists in daffodil itself (Daf.diff_da() is numeric dict subtraction, built
# for comparing vote counts -- unrelated). This is real new logic, built on Daf's own keyfield/
# select primitives, not a wrapper around an existing table-diff facility.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from daffodil.daf import Daf


@dataclass
class DiffResult:
    columns_added:      List[str]
    columns_removed:    List[str]
    columns_reordered:  bool
    old_cols_order:     List[str]
    new_cols_order:     List[str]
    compared_by:        str                        # 'key:<colname>' or 'position'
    rows_added:         List[Dict[str, Any]]        # new rows not in old (full row, new-side columns)
    rows_removed:       List[Dict[str, Any]]        # old rows not in new (full row, old-side columns)
    changed_cells:      List[Dict[str, Any]]        # [{'row': key_or_pos, 'col': c, 'old': x, 'new': y}]
    duplicate_keys:     List[str] = field(default_factory=list)

    @property
    def is_equal(self) -> bool:
        return not (self.columns_added or self.columns_removed or self.rows_added
                    or self.rows_removed or self.changed_cells or self.duplicate_keys)


class DiffError(ValueError):
    """ Raised for a diff that can't proceed safely, e.g. duplicate keys in --key mode. """


def diff_daf(old: Daf, new: Daf, key: Optional[str] = None) -> DiffResult:
    old_cols = old.columns()
    new_cols = new.columns()
    columns_added = [c for c in new_cols if c not in old_cols]
    columns_removed = [c for c in old_cols if c not in new_cols]
    common_cols = [c for c in old_cols if c in new_cols]
    columns_reordered = common_cols != [c for c in new_cols if c in old_cols]

    if key is not None:
        if key not in old_cols or key not in new_cols:
            raise DiffError(f"--key {key!r} is not a column in both files.")
        return _diff_by_key(old, new, key, common_cols, columns_added, columns_removed, columns_reordered)

    return _diff_by_position(old, new, common_cols, columns_added, columns_removed, columns_reordered)


def _find_duplicate_keys(daf: Daf, key: str) -> List[str]:
    icol = daf.hd[key]
    seen: Dict[str, int] = {}
    dups: List[str] = []
    for row in daf.lol:
        val = row[icol]
        seen[val] = seen.get(val, 0) + 1
        if seen[val] == 2:
            dups.append(val)
    return dups


def _diff_by_key(old, new, key, common_cols, columns_added, columns_removed, columns_reordered) -> DiffResult:
    old_dups = _find_duplicate_keys(old, key)
    new_dups = _find_duplicate_keys(new, key)
    duplicate_keys = sorted(set(old_dups) | set(new_dups))
    if duplicate_keys:
        # Detect and report, don't silently pick one -- match an arbitrary row is worse than refusing.
        return DiffResult(
            columns_added=columns_added, columns_removed=columns_removed,
            columns_reordered=columns_reordered, old_cols_order=old.columns(), new_cols_order=new.columns(),
            compared_by=f'key:{key}', rows_added=[], rows_removed=[], changed_cells=[],
            duplicate_keys=duplicate_keys,
            )

    old_by_key = {row[old.hd[key]]: row for row in old.lol}
    new_by_key = {row[new.hd[key]]: row for row in new.lol}

    added_keys = [k for k in new_by_key if k not in old_by_key]
    removed_keys = [k for k in old_by_key if k not in new_by_key]
    common_keys = [k for k in old_by_key if k in new_by_key]

    rows_added = [dict(zip(new.columns(), new_by_key[k])) for k in added_keys]
    rows_removed = [dict(zip(old.columns(), old_by_key[k])) for k in removed_keys]

    changed_cells = []
    for k in common_keys:
        old_row = old_by_key[k]
        new_row = new_by_key[k]
        for col in common_cols:
            old_val = old_row[old.hd[col]]
            new_val = new_row[new.hd[col]]
            if old_val != new_val:
                changed_cells.append({'row': k, 'col': col, 'old': old_val, 'new': new_val})

    return DiffResult(
        columns_added=columns_added, columns_removed=columns_removed,
        columns_reordered=columns_reordered, old_cols_order=old.columns(), new_cols_order=new.columns(),
        compared_by=f'key:{key}', rows_added=rows_added, rows_removed=rows_removed,
        changed_cells=changed_cells,
        )


def _diff_by_position(old, new, common_cols, columns_added, columns_removed, columns_reordered) -> DiffResult:
    num_common = min(len(old), len(new))
    rows_added = [dict(zip(new.columns(), row)) for row in new.lol[num_common:]] if len(new) > num_common else []
    rows_removed = [dict(zip(old.columns(), row)) for row in old.lol[num_common:]] if len(old) > num_common else []

    changed_cells = []
    for pos in range(num_common):
        old_row = old.lol[pos]
        new_row = new.lol[pos]
        for col in common_cols:
            old_val = old_row[old.hd[col]]
            new_val = new_row[new.hd[col]]
            if old_val != new_val:
                changed_cells.append({'row': pos, 'col': col, 'old': old_val, 'new': new_val})

    return DiffResult(
        columns_added=columns_added, columns_removed=columns_removed,
        columns_reordered=columns_reordered, old_cols_order=old.columns(), new_cols_order=new.columns(),
        compared_by='position', rows_added=rows_added, rows_removed=rows_removed,
        changed_cells=changed_cells,
        )


def diff_result_to_dict(result: DiffResult) -> Dict[str, Any]:
    return {
        'compared_by': result.compared_by,
        'columns_added': result.columns_added,
        'columns_removed': result.columns_removed,
        'columns_reordered': result.columns_reordered,
        'old_columns_order': result.old_cols_order,
        'new_columns_order': result.new_cols_order,
        'duplicate_keys': result.duplicate_keys,
        'rows_added': result.rows_added,
        'rows_removed': result.rows_removed,
        'changed_cells': result.changed_cells,
        'equal': result.is_equal,
        }


def diff_result_to_md(result: DiffResult) -> str:
    lines = [f"Compared by: {result.compared_by}"]
    if result.duplicate_keys:
        lines.append(f"\n**ERROR: duplicate key value(s), refusing to diff by key: {result.duplicate_keys}**")
        return '\n'.join(lines)

    if result.columns_added:
        lines.append(f"\nColumns added: {result.columns_added}")
    if result.columns_removed:
        lines.append(f"\nColumns removed: {result.columns_removed}")
    if result.columns_reordered:
        lines.append(f"\nColumns reordered: {result.old_cols_order} -> {result.new_cols_order}")
    if result.rows_added:
        lines.append(f"\nRows added ({len(result.rows_added)}):")
        for row in result.rows_added:
            lines.append(f"  + {row}")
    if result.rows_removed:
        lines.append(f"\nRows removed ({len(result.rows_removed)}):")
        for row in result.rows_removed:
            lines.append(f"  - {row}")
    if result.changed_cells:
        lines.append(f"\nCells changed ({len(result.changed_cells)}):")
        for c in result.changed_cells:
            lines.append(f"  row {c['row']!r}, col {c['col']!r}: {c['old']!r} -> {c['new']!r}")
    if result.is_equal:
        lines.append("\nNo differences.")
    return '\n'.join(lines)
