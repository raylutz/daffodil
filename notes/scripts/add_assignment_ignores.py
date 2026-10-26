"""
Add `# type: ignore[assignment]` to each class-body line mypy reports as incompatible with the base class.

After a move by move_group.py, a method that overrode Daf with another signature had
`# type: ignore[override]` on its def. Its assignment in the class needs this instead.

usage: uv run mypy src/daffodil | python notes/scripts/add_assignment_ignores.py
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

lines_of: dict = defaultdict(set)
for report in sys.stdin:
    m = re.match(r'^(src/daffodil/\w+\.py):(\d+): error: Incompatible types in assignment .*base class .*\[assignment\]$', report.strip())
    if m:
        lines_of[m.group(1)].add(int(m.group(2)))
for rel, nums in lines_of.items():
    path = ROOT / rel
    lines = path.read_text().split('\n')
    for num in nums:
        lines[num - 1] = lines[num - 1].rstrip() + '  # type: ignore[assignment]'
    path.write_text('\n'.join(lines))
    print(rel, sorted(nums))
