# conftest.py
#
# Library code raises exceptions on error paths rather than calling breakpoint(). Make any
# breakpoint() reached during a test fail loudly instead of dropping into pdb.

import sys

import pytest


@pytest.fixture(autouse=True)
def _fail_on_breakpoint(monkeypatch):
    def _hook(*args, **kwargs):
        raise AssertionError("breakpoint() called during a test")
    monkeypatch.setattr(sys, 'breakpointhook', _hook)
