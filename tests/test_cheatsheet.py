# test_cheatsheet.py
#
# docsite/cheatsheet.md is built from notes/scripts/cheatsheet_content.py by
# notes/scripts/build_cheatsheet.py. Run every snippet, and check that the built file is current.

import os
import sys

import pytest

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'notes', 'scripts')
sys.path.insert(0, SCRIPTS)

import build_cheatsheet                                                  # noqa: E402
from cheatsheet_content import SETUP, SECTIONS, PANDAS_TITLE             # noqa: E402


def _section_code(rows, title):
    """ The code to run for a section, in order: its lines build on each other, as on the sheet. """
    for code, desc, *flags in rows:
        if flags and flags[0] is False:
            continue
        if title == PANDAS_TITLE:
            yield from desc.split('  or  ')
        else:
            yield code


@pytest.mark.parametrize('title, rows', [(title, rows) for title, rows, _notes in SECTIONS])
def test_section_runs(title, rows, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    env: dict = {}
    exec(SETUP, env)
    for code in _section_code(rows, title):
        try:
            exec(code, env)
        except Exception as err:
            pytest.fail(f"[{title}] {code}\n  {type(err).__name__}: {err}")


def test_cheatsheet_md_is_current():
    with open(build_cheatsheet.OUT_PATH, encoding='utf-8') as fh:
        assert fh.read() == build_cheatsheet.build(), \
            "docsite/cheatsheet.md is out of date. Run: uv run python notes/scripts/build_cheatsheet.py"


def test_guide_indexing_reference_is_current():
    with open(build_cheatsheet.GUIDE_PATH, encoding='utf-8') as fh:
        guide = fh.read()
    assert guide == build_cheatsheet.guide_with_reference(guide), \
        "The indexing reference in docsite/guide/selecting.md is out of date. Run: uv run python notes/scripts/build_cheatsheet.py"
