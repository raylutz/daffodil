# conftest.py
#
# The Guide pages in docsite/guide are Markdown with >>> examples in ```pycon fences, run as doctests:
#   pytest --doctest-glob='*.md' docsite/guide
# For those pages only:
# - The closing ``` of a fence is not part of an example's expected output.
# - Whitespace is normalized, as print(daf.to_md()) ends with a blank line.
# The docstrings keep exact matching.

import doctest
import re

import pytest

_FENCE_AT_END = re.compile(r'(?:^|\n)[ \t]*```[^\n]*\n?\Z')


def pytest_collection_modifyitems(items: list) -> None:
    for item in items:
        if isinstance(item, pytest.DoctestItem) and str(item.fspath).endswith('.md'):
            for example in item.dtest.examples:
                example.want = _strip_fence(example.want)
                if example.exc_msg is not None:         # an expected exception keeps its message apart
                    example.exc_msg = _strip_fence(example.exc_msg)
                example.options[doctest.NORMALIZE_WHITESPACE] = True


def _strip_fence(text: str) -> str:
    """ The text without a closing ``` line at its end. """
    if text.strip() == '```':
        return ''
    return _FENCE_AT_END.sub('\n', text).lstrip('\n')
