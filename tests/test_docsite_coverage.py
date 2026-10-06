# test_docsite_coverage
# copyright (c) 2026 Ray Lutz
#
# The Daf API docs are split over several pages in docsite/api/daf/.
# Each page lists its members by hand, so check that every public member
# of Daf is on exactly one page. A new method fails here until it is added.

import inspect
import re
from collections import Counter
from pathlib import Path

from daffodil.daf import Daf

DAF_PAGES = Path(__file__).resolve().parent.parent / 'docsite' / 'api' / 'daf'


def _documented() -> Counter:
    found: Counter = Counter()
    for page in DAF_PAGES.glob('*.md'):
        text = page.read_text()
        found.update(re.findall(r'^::: daffodil\.daf\.Daf\.(\w+)', text, re.M))
        # Methods defined in helper modules, shown under their Daf name.
        found.update(re.findall(r'^\s+heading: "(\w+)"', text, re.M))
    return found


def _public_members() -> set:
    # Dunders count only when they are methods with a docstring. Python and the copy
    # module add dunder attributes such as __dict__ and __slotnames__ to the class.
    return {name for name, value in vars(Daf).items()
            if not name.startswith('_')
            or (name.startswith('__') and inspect.isfunction(value) and value.__doc__)}


def test_every_daf_member_is_documented():
    missing = _public_members() - set(_documented())
    assert not missing, f"Add these to a page in docsite/api/daf/: {sorted(missing)}"


def test_no_daf_member_is_documented_twice():
    twice = [name for name, n in _documented().items() if n > 1]
    assert not twice, f"Documented on more than one page: {twice}"


def test_documented_names_exist():
    unknown = set(_documented()) - _public_members()
    assert not unknown, f"Not members of Daf: {sorted(unknown)}"
